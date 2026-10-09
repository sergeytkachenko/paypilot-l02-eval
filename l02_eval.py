#!/usr/bin/env python3
"""L02 eval: three text metrics from DeepEval next to domain correctness.

Faithfulness, answer relevancy and hallucination rate are judged by an LLM.
Domain correctness is not: the expected answer is computed by the stand's own
rules engines (app/engines), so it is free, deterministic and independent of
the agent's tools, which on lesson-02 are themselves defective.

    python l02_eval.py --profiles clean,lesson-02 --runs 3 --baseline-runs 2
    python l02_eval.py --profiles clean,lesson-02 --runs 3 --dry-run
    python l02_eval.py --profiles lesson-02 --only C-03,C-04 --metrics domain

Environment:
    SERVICE_URL        local stand, default http://localhost:8000
    STAND_DIR          paypilot-stand checkout; the engines are imported from it
    ANTHROPIC_API_KEY  key for the judge; the stand's own key will do
    OPENAI_API_KEY     the same for a stand that runs on OpenAI
    GEMINI_API_KEY     a Google AI Studio key for a Gemini judge; when several
                       keys are set, Anthropic wins, then OpenAI, then Gemini
    JUDGE_MODEL        default claude-haiku-5-5 (Anthropic), gpt-4.1-mini (OpenAI)
                       or gemini-3.5-flash-lite (Gemini)
    AGENT_PRICE_IN, AGENT_PRICE_OUT
                       agent price in USD per million tokens, default 0.1 / 0.5,
                       the Haiku 5.5 price for prompts under 100k tokens

Point it at a local stand only: it switches profiles, sets the clock and
resets the data, which on a shared stand changes everyone's session.
"""
import argparse
import json
import os
import re
import statistics
import sys
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
warnings.filterwarnings("ignore", message=".*HallucinationMetric.*")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000").rstrip("/")
CLOCK = "2026-09-15T10:00:00Z"
BASELINE = "clean"
LLM_METRICS = ("faithfulness", "answer_relevancy", "hallucination")
JUDGE_DEFAULTS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-3.5-flash-lite",
}
ANTHROPIC_MODELS_REJECTING_TEMPERATURE = ("claude-haiku-5-5",)
ANTHROPIC_PRICES_MISSING_FROM_DEEPEVAL = {
    "claude-haiku-5-5": (0.10 / 1e6, 0.50 / 1e6),
}
# Judge calls DeepEval 4.2 makes per metric with include_reason=True:
# faithfulness = truths, claims, verdicts, reason; relevancy = statements,
# verdicts, reason; hallucination = verdicts, reason.
JUDGE_CALLS = {"faithfulness": 4, "answer_relevancy": 3, "hallucination": 2}
THRESHOLDS = (0.7, 0.8, 0.9)
GREEN = 0.85  # "faithfulness above 0.85" from step 4 of the lab

# Affirmative-only pattern from the course runner plus two "still" phrasings:
# a bare "eligible" or "within the window" also matches the negated sentence
# and would fail every clean run.
AFFIRMATIVE = (r"\byou\s+can(?:\s+(?:absolutely|certainly|definitely))?(?:\s+still)?\s+(?:dispute|open|file)\b"
               r"|\byou(?:['’]re|\s+are)(?:\s+still)?\s+able\s+to\s+(?:dispute|open|file)\b"
               r"|\bis(?:\s+(?:still|currently|now))?\s+eligible\b"
               r"|\bremains\s+eligible\b"
               r"|(?<!not\s)\ball\s+(?:(?:eligibility|compliance|status|timeline|window|dispute|and)[,\s]+){0,6}"
               r"checks\s+(?:have\s+)?passed\b"
               r"|(?<!not\s)(?<!n['’]t\s)\b(?:passes|passed)\s+all\s+"
               r"(?:(?:eligibility|compliance|status|timeline|window|dispute|and)[,\s]+){0,6}checks\b"
               r"|\byou(?:['’]re|\s+are)\s+still\s+within\b"
               r"|\byou\s+still\s+have\s+(?:time|until|\d+\s+days?)\b")
CONDITIONAL = (r"\b(?:whether|if)\s+(?:(?:a|the|this|that|your|my)\s+)?[\w-]+"
               r"(?:\s+(?:is|remains)(?:\s+(?:still|currently|now))?\s+eligible"
               r"|\s+can(?:\s+still)?\s+(?:dispute|open|file)"
               r"|(?:\s+are|['’]re)(?:\s+still)?\s+able\s+to\s+(?:dispute|open|file))\b")
# Thousands groups only as ",ddd" or " ddd": the course runner's looser
# pattern glues "1,000, 5,000" into one number.
NUM_RE = re.compile(r"\d{1,3}(?:[,\u00a0\u202f ]\d{3}(?!\d))+(?:\.\d+)?|\d+(?:\.\d+)?")


# --- stand -------------------------------------------------------------------

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def stand(method: str, path: str, payload=None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(SERVICE_URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.URLError as e:
        raise SystemExit(f"stand: {method} {path} failed ({e}). "
                         f"Is it running at {SERVICE_URL}?") from e


def spans(tree: dict):
    yield tree
    for child in tree.get("children", []):
        yield from spans(child)


def seen_context(tree: dict) -> list[str]:
    """What the agent had in front of it: tool results and retrieved fragments.
    This is faithfulness' retrieval_context."""
    out = []
    for span in spans(tree):
        attrs = span.get("attributes") or {}
        if not str(span.get("name", "")).startswith("tool."):
            continue
        name, result = attrs.get("tool.name"), attrs.get("tool.result")
        if name == "search_knowledge_base" and isinstance(result, dict):
            out += [f"[{f['id']}] {f['text']}" for f in result.get("fragments", [])]
        else:
            out.append(f"{name}({json.dumps(attrs.get('tool.arguments'))}) "
                       f"returned {json.dumps(result, ensure_ascii=False)}")
    return out or ["(the agent called no tool and retrieved no document)"]


# --- oracle ------------------------------------------------------------------

def load_engines() -> dict:
    for candidate in (os.environ.get("STAND_DIR"), "paypilot-stand", "../paypilot-stand"):
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if (root / "app" / "engines" / "fx.py").exists():
            sys.path.insert(0, str(root))
            from app.engines import disputes, fx, limits, policy
            return {"disputes": disputes, "fx": fx, "limits": limits, "policy": policy}
    raise SystemExit("engines not found: set STAND_DIR to your paypilot-stand checkout")


def oracle(case: dict, engines: dict, offline: bool = False) -> dict | None:
    """The expected facts for a case, computed rather than typed in."""
    spec = case.get("oracle")
    if not spec:
        return None
    if "engine" in spec:
        module, fn = spec["engine"].split(".")
        args = dict(spec["args"])
        for key in ("tx_date", "as_of"):
            if key in args:
                args[key] = date.fromisoformat(args[key])
        if "outgoing_transfers" in args:
            args["outgoing_transfers"] = [
                {"date": date.fromisoformat(t["date"]),
                 "amount_eur": engines["policy"].to_eur(t["amount"], t["currency"])}
                for t in args["outgoing_transfers"]]
        return getattr(engines[module], fn)(**args).as_dict()
    if "db" in spec:
        if offline:
            return {"source": f"stand DB, {spec['db']} {spec['id']}"}
        rows = stand("GET", f"/api/_test/state/{spec['db']}")["rows"]
        return next(r for r in rows if r["id"] == spec["id"])
    return dict(spec["facts"])  # corpus: facts copied from the named document


def numbers(text: str) -> list[float]:
    out = []
    for raw in NUM_RE.findall(text or ""):
        try:
            out.append(float(re.sub(r"[,\u00a0\u202f ]", "", raw)))
        except ValueError:
            continue
    return out


def domain_check(case: dict, facts: dict | None, answer: str):
    """(passed, detail), or None when the case has no domain check."""
    check = case.get("check")
    if not check:
        return None
    kind = check["type"]
    if kind == "number":
        expected = float(facts[check["field"]])
        found = numbers(answer)
        ok = any(abs(n - expected) <= check.get("tolerance", 0.01) for n in found)
        return ok, f"expected {expected:,.2f}; numbers in answer: {found[:6]}"
    if kind == "no_offer":
        if facts[check["field"]] is not False:
            raise ValueError(f"{case['id']}: no_offer needs the engine to say no")
        m = re.search(AFFIRMATIVE, re.sub(CONDITIONAL, " ", answer, flags=re.I), re.I)
        return m is None, "engine: not eligible" + (f"; answer offers it: {m.group(0)!r}" if m else "")
    if kind == "not_regex":
        m = re.search(check["pattern"], answer, re.I)
        return m is None, "" if m is None else f"forbidden: {m.group(0)!r}"
    if kind == "percent":
        pattern = rf"(?<![\d.]){re.escape(format(facts[check['field']], 'g'))}\s*%"
    elif kind == "days":
        pattern = rf"\b{facts[check['field']]}[\s-]*(?:calendar\s+)?days?\b"
    else:
        pattern = check["pattern"]
    ok = re.search(pattern, answer, re.I) is not None
    return ok, "" if ok else f"not found: {pattern}"


# --- judge -------------------------------------------------------------------

def judge_provider() -> str:
    for provider, variable in (
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ):
        if os.environ.get(variable):
            return provider
    raise SystemExit("the judge needs ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY "
                     "(or run with --metrics domain)")


def make_judge(provider: str, model: str):
    from deepeval.models import AnthropicModel, OpenAIModel, GeminiModel

    if provider == "anthropic":
        base = AnthropicModel
        kwargs = {"model": model}
        if model not in ANTHROPIC_MODELS_REJECTING_TEMPERATURE:
            kwargs["temperature"] = 0
        prices = ANTHROPIC_PRICES_MISSING_FROM_DEEPEVAL.get(model)
        if prices:
            kwargs["cost_per_input_token"], kwargs["cost_per_output_token"] = prices

    elif provider == "openai":
        base = OpenAIModel
        kwargs = {
            "model": model,
            "temperature": 0,
        }

    elif provider == "gemini":
        base = GeminiModel
        kwargs = {
            "model": model,
            "api_key": os.environ["GEMINI_API_KEY"],
            "temperature": 0,
        }

    else:
        raise ValueError(f"Unsupported judge provider: {provider}")

    class CountingJudge(base):
        """Counts real judge calls, so the formula can be checked against them."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls, self.cost = 0, 0.0

        def _count(self, out):
            self.calls += 1
            if isinstance(out, tuple) and len(out) == 2:
                self.cost += out[1] or 0.0
            return out

        def generate(self, *args, **kwargs):
            return self._count(super().generate(*args, **kwargs))

        async def a_generate(self, *args, **kwargs):
            return self._count(await super().a_generate(*args, **kwargs))

    return CountingJudge(**kwargs)


def measure(name: str, judge, test_case) -> dict:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
    common = {"model": judge, "async_mode": False}
    metric = {
        # A claim the context does not mention counts as unsupported, as in the
        # longread's definition. DeepEval's default would let it pass.
        "faithfulness": lambda: FaithfulnessMetric(penalize_ambiguous_claims=True, **common),
        "answer_relevancy": lambda: AnswerRelevancyMetric(**common),
        "hallucination": lambda: HallucinationMetric(**common),
    }[name]()
    metric.measure(test_case, _show_indicator=False)
    score = metric.score
    # DeepEval 4 scores hallucination as agreement (1 = good). The longread's
    # hallucination rate is the contradicted share of the curated context.
    if name == "hallucination":
        score = 1 - score
    return {"score": round(score, 3), "reason": metric.reason}


# --- run ---------------------------------------------------------------------

def run_case(case: dict, facts, metrics: set, judge_spec: tuple) -> dict:
    from deepeval.test_case import LLMTestCase

    turn = stand("POST", "/chat", {"message": case["input"]})
    tree = stand("GET", f"/api/_test/traces/{turn['request_id']}")
    answer = turn["answer"] or ""
    rec = {"id": case["id"], "answer": answer, "request_id": turn["request_id"],
           "tools": [s["name"][5:] for s in spans(tree) if s["name"].startswith("tool.")],
           "agent": {**turn["usage"],
                     "llm_calls": sum(1 for s in spans(tree) if s["name"] == "llm.call")},
           "metrics": {}, "judge": {"calls": 0, "cost_usd": 0.0}}
    if "domain" in metrics:
        verdict = domain_check(case, facts, answer)
        rec["domain"] = None if verdict is None else {"passed": verdict[0], "detail": verdict[1]}
    wanted = [m for m in case.get("metrics", []) if m in metrics]
    if not wanted:
        return rec
    judge = make_judge(*judge_spec)
    # Hallucination rate is checked against what we curated, not what the
    # agent retrieved: the rule's document plus the engine's own result.
    curated = None
    if "hallucination" in wanted:
        curated = case["curated_context"] + [f"Rules engine result: {json.dumps(facts)}"]
    test_case = LLMTestCase(input=case["input"], actual_output=answer,
                            retrieval_context=seen_context(tree), context=curated)
    for name in wanted:
        try:
            rec["metrics"][name] = measure(name, judge, test_case)
        except Exception as e:  # one failed judge call must not sink the run
            rec["metrics"][name] = {"score": None, "reason": f"error: {e}"}
    rec["judge"] = {"calls": judge.calls, "cost_usd": round(judge.cost, 5)}
    return rec


def run_profile(profile: str, runs: int, cases: list, facts: dict, metrics: set,
                judge_spec: tuple, workers: int) -> list[dict]:
    stand("PUT", "/api/_test/profile", {"profile": profile})
    stand("POST", "/api/_test/reset")
    stand("POST", "/api/_test/clock", {"now": CLOCK})
    health = stand("GET", "/health")
    version = stand("GET", "/api/_test/prompt")["version"]
    print(f"\n== {profile}: defects {','.join(health['active_defects']) or '-'} · "
          f"prompt {version} · provider {health['provider']} · judge {judge_spec[1] or '-'}")
    if health["provider"] == "mock":
        print("   provider is mock: prompt-layer defects (D04 overlay, D05, D25) will not show")
    records = []
    for run in range(1, runs + 1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch = list(pool.map(
                lambda c: run_case(c, facts[c["id"]], metrics, judge_spec), cases))
        for rec in batch:
            rec.update(profile=profile, run=run)
            dom = rec.get("domain")
            scores = " ".join(f"{k[:5]}={v['score']}" for k, v in rec["metrics"].items())
            mark = "-" if not dom else ("ok" if dom["passed"] else "FAIL")
            print(f"   run {run} {rec['id']:<5} domain={mark:<4} {scores}")
        records += batch
    return records


# --- report ------------------------------------------------------------------

def run_means(records: list, profile: str) -> dict:
    """Per-run values of each metric: mean score, or passed share for domain."""
    out = {}
    for run in sorted({r["run"] for r in records if r["profile"] == profile}):
        rs = [r for r in records if r["profile"] == profile and r["run"] == run]
        for name in LLM_METRICS:
            vals = [r["metrics"][name]["score"] for r in rs
                    if r["metrics"].get(name, {}).get("score") is not None]
            if vals:
                out.setdefault(name, []).append(statistics.mean(vals))
        doms = [r["domain"]["passed"] for r in rs if r.get("domain")]
        if doms:
            out.setdefault("domain", []).append(sum(doms) / len(doms))
    return out


def fmt(vals: list | None) -> str:
    if not vals:
        return "—"
    mean = f"{statistics.mean(vals):.2f}"
    return mean if len(vals) == 1 else f"{mean} ({min(vals):.2f}–{max(vals):.2f})"


LABELS = {"faithfulness": "faithfulness", "answer_relevancy": "answer relevancy",
          "hallucination": "hallucination rate ↓", "domain": "доменна коректність"}


def report(records: list, profiles: list, cases: list, passes: int, metrics: set) -> None:
    means = {p: run_means(records, p) for p in profiles}
    base = profiles[0]
    print("\n" + "Метрика".ljust(23) + "".join(p.ljust(22) for p in profiles)
          + ("Дельта" if len(profiles) > 1 else ""))
    for key, label in LABELS.items():
        row = label.ljust(23) + "".join(fmt(means[p].get(key)).ljust(22) for p in profiles)
        if len(profiles) > 1 and means[base].get(key) and means[profiles[-1]].get(key):
            delta = statistics.mean(means[profiles[-1]][key]) - statistics.mean(means[base][key])
            row += f"{delta:+.2f}"
        print(row)

    others = [r for r in records if r["profile"] != BASELINE]
    wrong = [r for r in others if r.get("domain") and not r["domain"]["passed"]]
    section(f"Зелена метрика на хибному висновку (faithfulness ≥ {GREEN}, домен ✗)", [
        f"{r['profile']} run {r['run']} {r['id']}: faithfulness {score(r, 'faithfulness')} · "
        f"{r['domain']['detail']}\n      «{r['answer'][:220]}»"
        for r in wrong if (score(r, "faithfulness") or 0) >= GREEN])

    lines = []
    for r in records:
        low = {k: score(r, k) for k in ("faithfulness", "answer_relevancy")
               if score(r, k) is not None and score(r, k) < 0.7}
        if low and (not r.get("domain") or r["domain"]["passed"]):
            lines.append(f"{r['profile']} run {r['run']} {r['id']}: {low}\n"
                         f"      «{r['answer'][:220]}»")
    section("Кандидати у false positive (LLM-метрика < 0.7, домен ✓ або без домену) — "
            "перевір очима", lines)

    clean = [r for r in records if r["profile"] == BASELINE]
    lines = []
    for name in ("faithfulness", "answer_relevancy"):
        hit = [r for r in wrong if score(r, name) is not None]
        ok = [r for r in clean if score(r, name) is not None]
        if not hit or not ok:
            continue
        for t in THRESHOLDS:
            caught = sum(score(r, name) < t for r in hit)
            alarms = sum(score(r, name) < t for r in ok)
            lines.append(f"{LABELS[name]:<18} < {t}: ловить {caught}/{len(hit)} · "
                         f"хибні тривоги на clean {alarms}/{len(ok)}")
    section("Поріг: скільки дефектних кейсів ловить і скільки чистих зупиняє без причини",
            lines)

    agent_calls = sum(r["agent"]["llm_calls"] for r in records)
    judge_calls = sum(r["judge"]["calls"] for r in records)
    tok_in = sum(r["agent"]["input_tokens"] for r in records)
    tok_out = sum(r["agent"]["output_tokens"] for r in records)
    price_in = float(os.environ.get("AGENT_PRICE_IN", "0.1"))
    price_out = float(os.environ.get("AGENT_PRICE_OUT", "0.5"))
    agent_usd = (tok_in * price_in + tok_out * price_out) / 1e6
    judge_usd = sum(r["judge"]["cost_usd"] for r in records)
    formula = formula_calls(cases, metrics) * passes
    print(f"\nВартість: формула лонгріда (кейси × (1 + метрики)) = {formula} викликів; "
          f"фактично {agent_calls} llm.call агента + {judge_calls} викликів судді")
    print(f"   агент {tok_in:,} in / {tok_out:,} out токенів ≈ ${agent_usd:.3f} "
          f"(прайс {price_in}/{price_out} за 1M) · суддя ≈ ${judge_usd:.3f} · "
          f"разом ≈ ${agent_usd + judge_usd:.3f}")


def score(rec: dict, name: str):
    return (rec["metrics"].get(name) or {}).get("score")


def section(title: str, lines: list) -> None:
    print(f"\n{title}:")
    for line in lines or ["немає"]:
        print(f"   {line}")


def formula_calls(cases: list, metrics: set) -> int:
    return sum(1 + len([m for m in c.get("metrics", []) if m in metrics]) for c in cases)


def check_judge(records: list) -> None:
    results = [r["metrics"][name] for r in records for name in LLM_METRICS
               if name in r["metrics"]]
    errors = [m["reason"] for m in results if (m["reason"] or "").startswith("error:")]
    if not errors:
        return
    distinct = list(dict.fromkeys(errors))
    message = (f"judge: {len(errors)} of {len(results)} LLM metric results are errors "
               f"({len(distinct)} distinct), first: {distinct[0].removeprefix('error: ')}")
    if len(errors) == len(results):
        raise SystemExit(f"\n{message}\nno LLM metric was scored")
    print(f"\n{message}", file=sys.stderr)


def show(value) -> str:
    return f"{value:,.2f}" if isinstance(value, float) else str(value)


def dry_run(cases: list, facts: dict, plan: dict, metrics: set) -> None:
    print(f"{'id':<6}{'check':<10}{'metrics':<44}expected")
    for c in cases:
        f = facts[c["id"]] or {}
        field = (c.get("check") or {}).get("field")
        expected = (f"{field} = {show(f[field])}" if field in f
                    else ", ".join(f"{k} = {show(v)}" for k, v in f.items()) or "—")
        wanted = [m for m in c.get("metrics", []) if m in metrics]
        kind = (c.get("check") or {}).get("type", "—") if "domain" in metrics else "—"
        print(f"{c['id']:<6}{kind:<10}{','.join(wanted) or '—':<44}{expected[:60]}")
    passes = sum(plan.values())
    judge = sum(JUDGE_CALLS[m] for c in cases for m in c.get("metrics", []) if m in metrics)
    runs = " + ".join(f"{p} ×{n}" for p, n in plan.items())
    print(f"\n{len(cases)} кейсів × ({runs}) = {len(cases) * passes} звернень до агента "
          f"(кожне — зазвичай 2–3 llm.call)")
    print(f"формула лонгріда: {formula_calls(cases, metrics) * passes} викликів")
    print(f"DeepEval насправді: ≈{judge * passes} викликів судді + ≈{2 * len(cases) * passes}–"
          f"{3 * len(cases) * passes} llm.call агента")


def load_cases(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profiles", default="clean,lesson-02",
                    help="comma list; the first one is the baseline")
    ap.add_argument("--runs", type=int, default=1,
                    help="runs per profile; probabilistic defects need 3")
    ap.add_argument("--baseline-runs", type=int, default=1,
                    help="runs of the clean baseline; 2 shows the noise floor")
    ap.add_argument("--only", help="comma list of case ids")
    ap.add_argument("--metrics", default="domain," + ",".join(LLM_METRICS))
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan and the call estimate, call nothing")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    profiles = args.profiles.split(",")
    plan = {p: args.baseline_runs if p == BASELINE else args.runs for p in profiles}
    metrics = set(args.metrics.split(","))
    engines = load_engines()
    facts = {c["id"]: oracle(c, engines, offline=args.dry_run) for c in cases}
    if args.dry_run:
        return dry_run(cases, facts, plan, metrics)
    provider = judge_provider() if metrics & set(LLM_METRICS) else None
    judge_spec = (provider, os.environ.get("JUDGE_MODEL") or JUDGE_DEFAULTS.get(provider))
    records = []
    for profile in profiles:
        records += run_profile(profile, plan[profile], cases, facts, metrics,
                               judge_spec, args.workers)
    report(records, profiles, cases, sum(plan.values()), metrics)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"l02-{'-'.join(profiles)}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"runs": plan, "clock": CLOCK,
                                "judge_provider": judge_spec[0], "judge_model": judge_spec[1],
                                "facts": facts, "records": records},
                               ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nзвіт: {path}")
    check_judge(records)


if __name__ == "__main__":
    main()
