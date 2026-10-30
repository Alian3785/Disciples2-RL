# Цель проекта

Цель этого проекта это найти наиболее эффективные методы для обучения с подкреплением (Reinforcement learning) в среде повторяющей (за некоторыми исключениями) пошаговую стратегию Disciples 2. На данном этапе ведется поиск алгоритма и оптимальных параметров для Single-agent RL и средой выступают 6 карт из Disciples 2 (3 из игры, 3 кастомные) а целями для агента являются выполнение целей из сценариев в Disciples 2 или кастомных карт (захват городов, победа над отрядом, защита столицы и.т.п.)

## Базовый набор для тестирования новых методов

Ветка `experiment/brute`: эксперимент Brute скрывает все игровые наблюдения
actor/critic и маски допустимости действий. PPO и базовые бюджеты сохранены;
точный контракт и отличие от дерева Brute из Gym Retro — в `docs/TRAINING.md`.
Итоги серии 30.10.2026: `results/brute-six-maps-20261030.txt`;
компактные метрики шести запусков — `results/brute-six-maps-20261030.json`.

Базовый набор для тестирования и сравнения новых методов RL состоит из шести карт:

| Карта | Значение `--map` | Бюджет базовой серии |
|---|---|---:|
| Default Blue Dragon | `default-blue-dragon` | 50 млн |
| Default Capital | `default-capital` | 50 млн |
| Super Last Stand | `super_last_stand` | 15 млн |
| Trade Train | `trade_train` | 60 млн |
| Проклятое дитя | `the_charmed_child` | 60 млн |
| Бегство Бетрезена | `bethrezens_retreat` | 60 млн |

Эти бюджеты уточнены пользователем 29.10.2026. Новый явно заданный бюджет имеет
приоритет. Обучение запускать только по запросу; карты обучать последовательно,
по одной на GPU, и сообщать в чат о старте, промежуточных метриках и завершении.
Проверять бюджеты **до** старта: изменение файла или очереди не меняет уже
работающий Python-процесс. Super Last Stand не запускать автоматически на 50/60 млн.

Результаты сравнения новых методов представлять по всем шести картам, если пользователь
не задал другой набор. Сохранённая таблица запусков от 28.10.2026 —
[results/baseline-six-maps-20261028.txt](results/baseline-six-maps-20261028.txt).
Она фиксирует исторические условия и метрики; совместимость весов с текущей
механикой и наблюдениями проверяется отдельно.

Последняя серия на этом инстансе —
[results/baseline-six-maps-20261029.txt](results/baseline-six-maps-20261029.txt).
Предыдущую таблицу сохранять отдельно; условия и отличия описаны в новом отчёте.
Компактные итоговые данные шести запусков: `results/baseline-six-maps-20261029.json`.

## План исследований и разработки

Источник: [план проекта в GitHub Projects](https://github.com/users/Alian3785/projects/2/views/1).

### Группа 1

1. Настройка состояния и наблюдений.
2. Формирование награды (Reward Shaping).
3. Настройка пространства действий.
4. Оптимизация гиперпараметров.
5. Обучение с постепенным усложнением задач (Curriculum Learning).

### Группа 2

6. Стратегии исследования среды (Exploration Policies).
7. Архитектуры нейронных сетей.
8. DreamerV3 и DreamerV4.
9. Алгоритмы на основе модели среды (Model-Based RL).

### Группа 3

10. Решение с двумя моделями.
11. AlphaZero для одиночной игры.
12. Изучение исследований по RL за прошлый год.
13. Изучение исследований по RL за текущий год.
14. Адаптация PufferLib.

## Графики: W&B или Comet

- Если облако W&B доступно и принимает сохранённый ключ, графики пишутся в W&B; если нет — в Comet (`comet_ml`). Это делает режим по умолчанию `--tracking auto`; причину пропуска сервиса он сохраняет в `tracking.json`.
- После обучения давать ссылку того сервиса, который записал запуск: поле `url` в `tracking.json`, строка `Charts:` в консоли. Если нет ключа Comet, запуск сохраняется офлайн в W&B — сообщить об этом и попросить пользователя сохранить ключ через `login-comet.cmd`.

### Comet на текущем инстансе

- SDK уже установлен: `comet_ml==3.58.8`; workspace — `lentix`.
- Рабочее хранилище ключа — `/workspace/.env`, права `600`.
  Перед запуском SDK загрузить его через `set -a; . /workspace/.env; set +a`.
- Ключ для переноса на другой инстанс сохранён в `secrets/comet.env.gpg`
  (AES-256, пароль передан пользователю отдельно в чате 29.10.2026).
  После клонирования расшифровать файл локально через GPG и добавить строку
  `COMET_API_KEY` в `/workspace/.env`, сохранив права `600`.
  Не хранить пароль рядом с шифротекстом и не включать ключ в логи/артефакты.
- Проект серии 29.10.2026:
  https://www.comet.com/lentix/baseline-six-maps-50m-20261029-093143.
  Слово `50m` в его URL историческое: фактические бюджеты приведены выше.
  Для новой серии можно использовать этот проект с новыми именами экспериментов
  либо создать новый приватный проект по просьбе пользователя.
- Если графики нужны именно в Comet, задавать `--tracking comet`,
  `--comet-workspace lentix`, `--comet-project NAME`. `--tracking auto`
  может выбрать W&B или офлайн-запись и не гарантирует графики в Comet.

### Сохранённые виды графиков и русские подписи

- Русские подписи поддерживаются. Ошибка `400: View name already used`
  была связана с повторным именем шаблона `Disciples II`, а не с кириллицей.
- Перед импортом метрик и каждым обучением экспортировать
  `NUMBERGRID_COMET_VIEW_SCOPE="$COMET_PROJECT_NAME"`. Это добавляет имя
  проекта к имени вида и предотвращает столкновения с другими проектами workspace.
- При поиске scoped-вида использовать
  `api_experiment.get_views(include_workspace_project_views=True)`:
  список только текущего эксперимента может быть пустым при уже существующем
  одноимённом шаблоне. Переиспользовать шаблон своего scope; чужие виды не менять.
- Общий вид и вид `Disciples II — Мизраэль` остаются раздельными:
  общие карты получают 35 campaign-графиков, Capital — также показатели Мизраэля.
  Шаблоны и подписи определены в `numbergrid_comet_dashboard.py`.
- Сначала раскрыт `rollout`: средняя награда слева, длина эпизода справа;
  затем `campaign`. Ось X — `step`, то есть суммарные решения агента.
  Метрики `rollout/*` используют последние 100 завершённых эпизодов,
  `campaign/window/victory_rate` — последний интервал в 1 млн шагов.
- Давать ссылку из `tracking.json → url` с `?viewId=...`, чтобы открывался
  нужный сохранённый вид. Проверять `tracking.json → dashboard.status == saved`.
- После завершения проверять доставку точек через Comet API, например
  `get_metrics('rollout/ep_rew_mean')` и
  `get_metrics('campaign/window/victory_rate')`: должны присутствовать шаги
  1 млн, 2 млн, … до бюджета карты. Один статус online ещё не доказывает доставку.
- Старые локальные `charts.jsonl` можно импортировать без повторного обучения:
  логировать исходный `training_steps` как `step`, использовать
  `compact_comet_metrics` и `configure_comet_dashboard`. Сохранять ключ
  эксперимента и применять `mode='get_or_create'`, чтобы повтор запроса
  не создавал дубликаты. Импортированные серии явно помечать в имени и тегах.

## Окружение и CUDA

**CUDA 13+ не является обязательным требованием задачи.** На новом инстансе
сначала проверять уже установленную CUDA, cuDNN, GPU и драйвер; использовать
совместимую имеющуюся версию, например CUDA 12.8, вместо автоматической загрузки
другой CUDA только ради названия окружения.

JAX поддерживает сборки для CUDA 12 и 13 и варианты `cuda12-local` /
`cuda13-local` для системных библиотек. Сборка JAX должна соответствовать
локальным CUDA/cuDNN, драйверу и архитектуре GPU. Недостающие совместимые
компоненты устанавливать адресно. Подробности и текущие требования:
[официальная установка JAX](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder).

В этом checkout `numbergrid-runtime.txt` пока закрепляет `jax[cuda13]` и
`nvidia-cuda-runtime==13.4.92`, а `runtime-constraints.txt` — CUDA 13 пакеты.
Поэтому существующий `install_runtime.sh` сам по себе **не выбирает системную
CUDA 12**: для неё требуется отдельный согласованный профиль зависимостей,
например JAX `cuda12-local`, с исправленными CUDA-пинами. `run_gpu.sh` также
снимает `LD_LIBRARY_PATH`; для локального профиля учитывать поиск библиотек и
компиляторов. Не смешивать CUDA 12 toolkit с CUDA 13 plugin и не отключать
проверки совместимости JAX ради запуска.

Текущий уже проверенный runtime — `/workspace/.venvs/disciples2-rl`,
`NUMBERGRID_VENV` сохранён в `/workspace/.env`, маркер готовности —
`.numbergrid-ready`. Системная CUDA 12.8 осталась установленной; выполненная
серия использовала pip-библиотеки CUDA 13.4.92 внутри venv и JAX 0.12.0.1.
Эта серия не является проверкой обучения на CUDA 12. Рабочее окружение для
следующего запуска переиспользовать, без повторной установки библиотек.
Драйвер NVIDIA внутри Vast-контейнера не переустанавливать.

## Быстрый запуск следующей серии на шести картах

На этом инстансе репозиторий находится в `/workspace/Disciples2-RL`.
Для новой серии выбрать новый каталог результатов и проверить все бюджеты.
Пример тела foreground-wrapper, который следует запускать через supervisor
по инструкции инстанса, а не оставлять фоновым процессом терминала:

```bash
#!/usr/bin/env bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set -uo pipefail
cd /workspace/Disciples2-RL
export NUMBERGRID_VENV=/workspace/.venvs/disciples2-rl
export COMET_WORKSPACE=lentix
export COMET_PROJECT_NAME=baseline-six-maps-50m-20261029-093143
export NUMBERGRID_COMET_VIEW_SCOPE="$COMET_PROJECT_NAME"
export COMET_DISABLE_AUTO_LOGGING=1
export PYTHONUNBUFFERED=1
batch="results/baseline-six-maps-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$batch"

# До обучения: структурная проверка пяти карт; Bethrezen не проверять
# без отдельного запроса согласно docs/TRAINING.md.
bash scripts/run_gpu.sh scripts/validate_reference_maps.py \
  --map default-blue-dragon --map default-capital --map super_last_stand \
  --map trade_train --map the_charmed_child --output "$batch/preflights.json"

while read -r map steps; do
  bash scripts/run_gpu.sh benchmark_number_grid.py \
    --map "$map" --seed 42 --total-timesteps "$steps" \
    --checkpoint-interval 5000000 --summary-interval 1000000 \
    --no-episode-reports --tracking comet \
    --comet-workspace "$COMET_WORKSPACE" --comet-project "$COMET_PROJECT_NAME" \
    --run-name "$map-$steps-$(basename "$batch")" --output "$batch/$map"
  python3 - "$batch/$map" "$steps" <<'PY'
import json
from pathlib import Path
import sys
output = Path(sys.argv[1])
status = json.loads((output / 'status.json').read_text())
tracking = json.loads((output / 'tracking.json').read_text())
assert status['phase'] == 'complete' and status['training_steps'] == int(sys.argv[2])
assert tracking['service'] == 'comet' and tracking['status'] == 'finished'
PY
done <<'MAPS'
default-blue-dragon 50000000
default-capital 50000000
super_last_stand 15000000
trade_train 60000000
the_charmed_child 60000000
bethrezens_retreat 60000000
MAPS
```

Для supervisor задать `environment=PROC_NAME="%(program_name)s"` и использовать
`stdout_logfile=/dev/stdout`,
`stdout_events_enabled=true`, `stopasgroup=true`, `killasgroup=true` и
`autorestart=false`: конечная серия не должна незаметно запускаться заново.
Обработчик очереди должен дождаться кода выхода 0 и `status.json.phase=complete`
перед переходом к следующей карте; каждый `--output` должен быть новым/пустым.
Не запускать `--evaluate`, `--select-best` или подробные отчёты эпизодов без просьбы.

Локальные средства предыдущей серии — `/workspace/disciples2-jobs/`,
указатель manifest — `/workspace/disciples2-current-batch.json`, служба —
`disciples2-baseline`. Старый manifest содержит завершённые задания и хеш
исходного diff: не использовать его как новую серию и не обходить его проверки.
Состояние очереди, метрики, веса и сводка лежат в каталоге серии под `results/`.
Время на 1 млн брать из `campaign/mean_seconds_per_million_steps`,
скорость последнего интервала — из `performance/steps_per_second`.
Подготовку, компиляцию и установку библиотек учитывать отдельно.

## Реализация и документация

- Приоритет — корректность актуальной версии. Для дальнейших изменений не требуется сохранять старые реализации, карты и совместимость моделей; не усложнять ради этого код. Несовместимые веса не выдавать за обученные актуальной механике.
- Текущие правила: [бои](docs/COMBAT.md), [карты](docs/MAPS.md), [обучение и проверки](docs/TRAINING.md). Полный указатель — в [README.md](README.md#documentation). Обновлять тематические справочники вместе с реализацией; результаты отдельных запусков хранить в `results/`, не создавать отдельные исторические Markdown-отчёты в корне и `docs/`.

## Что сохранять в GitHub

По просьбе пользователя публиковать минимальный набор: итоговую таблицу,
изменения условий и выводы, компактные финальные метрики по каждому запуску,
изменения кода и документации. Графики давать ссылками Comet.
Веса моделей, learner state, архивы состояний, полные CSV/JSONL, копии карт и
конфигураций, служебные файлы очереди и промежуточные чекпойнты оставлять локально.
Не добавлять весь `results/` через `git add -f`. Публикация бинарных артефактов
требует отдельного явного запроса пользователя. Зашифрованный ключ Comet
`secrets/comet.env.gpg` сохраняется для переноса инстанса; пароль отдельно у пользователя.
