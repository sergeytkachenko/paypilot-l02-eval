# L02 · Скрипт метрик

Три текстові метрики DeepEval (faithfulness, answer relevancy, hallucination
rate) поруч із доменною коректністю, яку рахують рушії стенду PayPilot.
Скрипт до заняття L02 курсу з тестування LLM-агентів.

Python локально не потрібен: скрипт запускається в Docker. Усе для
лабораторної береш звідси одним `git clone`, окремо нічого завантажувати не
треба.

## Що зробити, коротко

Корінь курсу завжди `~/paypilot`: стенд у `~/paypilot/paypilot-stand`, цей репозиторій — у
`~/paypilot/l02` (папка уроку; наступні уроки лягають поруч: `l03`, …).

1. Підняти локальний стенд: у `~/paypilot/paypilot-stand` — `docker compose up -d --build`, потім `docker compose exec stand python scripts/doctor.py`.
2. Склонувати цей репозиторій поруч зі стендом:
   `git clone https://github.com/sergeytkachenko/paypilot-l02-eval.git ~/paypilot/l02`
3. `cd ~/paypilot/l02`, `cp .env.example .env`, вписати в `.env`
   ключ судді (`STAND_DIR` за замовчуванням уже `../paypilot-stand`).
4. `docker compose build` — один раз.
5. `docker compose run --rm eval --runs 3 --baseline-runs 2 --dry-run`, потім
   те саме без `--dry-run`.

Деталі кожного кроку нижче.

Склонував раніше за 09.10.2026 — почни з `git pull`. Суддя за замовчуванням
тепер `claude-haiku-5-5`, а стара версія скрипта надсилає цій моделі
`temperature` і отримує у відповідь 400 з текстом
"`temperature` is deprecated for this model".

## Що в репозиторії

| Файл | Що це |
|---|---|
| `complaints.md` | двадцять скарг C-01…C-20 — вхідний матеріал кроку 1 |
| `triage.md` | дошка кроку 1: бот як новий співробітник підтримки і три купки для скарг — не знайшов, сказав не те, застосував не так |
| `l02_eval.py` | скрипт: шле кейси боту, рахує метрики, друкує звіт |
| `cases.json` | 13 кейсів зі скарг: запит, оракул, перевірка, метрики, контекст |
| `requirements.txt` | залежності, ставляться в образ |
| `Dockerfile`, `docker-compose.yml` | образ і запуск |
| `.env.example` | шаблон `.env`: шлях до стенду і ключ судді |

## Що потрібно

- Docker Desktop (macOS, Windows) або Docker Engine з compose (Linux).
- Піднятий **локальний** стенд `~/paypilot/paypilot-stand` (`docker compose up -d --build`, перевірка — `docker compose exec stand python scripts/doctor.py`)
  на `http://localhost:8000`. Скрипт перемикає профілі, скидає базу й ставить
  годинник, тому на спільному стенді його не запускай.
- Каталог стенду на цьому ж комп'ютері: скрипт імпортує з нього рушії
  (`app/engines`). Клонуй цей репозиторій у `~/paypilot/l02`, поруч зі стендом:
  тоді `STAND_DIR` за замовчуванням (`../paypilot-stand`) уже правильний.

## Запуск

```bash
mkdir -p ~/paypilot
git clone https://github.com/sergeytkachenko/paypilot-l02-eval.git ~/paypilot/l02
cd ~/paypilot/l02
cp .env.example .env        # впиши STAND_DIR і ключ судді
docker compose build        # один раз, близько хвилини

# план і кількість викликів, нічого не викликає
docker compose run --rm eval --runs 3 --baseline-runs 2 --dry-run

# повний прогін: clean двічі, lesson-02 тричі; ≈6 хв, ≈$0.1 на Haiku 5.5
docker compose run --rm eval --runs 3 --baseline-runs 2
```

Команди однакові для macOS, Linux і Windows (PowerShell). Усе після `eval` —
прапорці скрипта. Звіт друкується в термінал, сирі дані лягають у
`reports/*.json` у цьому каталозі. Щоб зберегти й звіт, додай `-T` і `tee`:

```bash
mkdir -p reports
docker compose run --rm -T eval --runs 3 --baseline-runs 2 2>&1 | tee reports/full-run.txt
```

`l02_eval.py` і `cases.json` підмонтовані в контейнер, тож правки в них
діють одразу, без перезбирання. `docker compose build` потрібен лише після
зміни `requirements.txt` або `Dockerfile`.

## `.env`

| Змінна | Що це |
|---|---|
| `STAND_DIR` | шлях до каталогу `paypilot-stand`, за замовчуванням `../paypilot-stand`. Windows: `C:/paypilot/paypilot-stand` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | ключ судді, той самий, що в `.env` стенду; якщо задано кілька ключів, береться Anthropic, потім OpenAI, потім Gemini |
| `GEMINI_API_KEY` | ключ Google AI Studio, якщо суддею буде Gemini. Якщо образ зібраний до появи Gemini, після `git pull` один раз виконай `docker compose build` |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` | для стенду через OpenRouter: `https://openrouter.ai/api` і той самий ключ |
| `JUDGE_MODEL` | суддя, за замовчуванням `claude-haiku-5-5`, `gpt-4.1-mini` або `gemini-3.5-flash-lite`. Порожнє значення залишає дефолт кіту — так і треба |
| `EVAL_STAND_URL` | адреса стенду зсередини контейнера, див. нижче |
| `STAND_PORT`, `STAND_PROFILE` | окремий стенд: порт на `127.0.0.1` (за замовчуванням `8010`) і стартовий профіль (`lesson-02`) |
| `AGENT_PRICE_IN`, `AGENT_PRICE_OUT` | ціна агента, USD за 1M токенів, для рядка вартості. За замовчуванням `0.1` / `0.5` — прайс Haiku 5.5 до 100k токенів промпту, бо курс просить виставити у стенді саме її. Твій стенд на іншій моделі — впиши її прайс, інакше рядок вартості збреше |

## Адреса стенду

Усередині контейнера `localhost` — це сам контейнер, а не твій комп'ютер.
Тому сервіс `eval` ходить на стенд через `http://host.docker.internal:8000`.
Стенд на іншому порту — задай `EVAL_STAND_URL` у `.env`, наприклад
`http://host.docker.internal:8010`.

Linux, стенд опублікований лише на `127.0.0.1`: `host.docker.internal`
туди не дістане. Бери сервіс `eval-host`, він працює в мережі хоста:

```bash
EVAL_STAND_URL=http://127.0.0.1:8010 docker compose run --rm eval-host --runs 3 --baseline-runs 2
```

## Окремий стенд

Якщо `localhost:8000` зайнятий спільним стендом (як на devhub), не чіпай
його: `docker compose up` у `paypilot-stand` перестворить спільний контейнер. Підніми окремий стенд із
цього ж `docker-compose.yml`. Сервіс `stand` збирається з каталогу
`STAND_DIR`, бере його `.env` і слухає лише `127.0.0.1:8010`:

```bash
docker compose up -d --build stand   # профіль lesson-02
curl -s http://127.0.0.1:8010/health
```

У `.env` цього репозиторію додай `EVAL_STAND_URL=http://stand:8000`:
сервіс `eval` дістає стенд за іменем, бо обидва в одному compose. Команди
запуску ті самі, з `eval`. Після прогону прибери стенд:

```bash
docker compose --profile stand down
```

## Корисні прапорці

| Прапорець | Що робить |
|---|---|
| `--profiles clean,lesson-02` | профілі по черзі; перший — baseline |
| `--runs 3` / `--baseline-runs 2` | прогони профілю заняття / прогони `clean` |
| `--only C-03,C-04` | лише ці кейси |
| `--metrics domain` | лише доменна коректність: без судді, безкоштовно і миттєво |
| `--workers 4` | паралельні кейси; при rate limit зменш |
| `--dry-run` | план і кількість викликів, нічого не викликає |
