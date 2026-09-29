# Training preferences

- Always launch training in `local5` observation mode unless the user explicitly requests a different mode. Set `CAMPAIGN_OBSERVATION_VERSION=local5` (or `observation_version="local5"`) explicitly and use the same mode for evaluation and replays.
- Append completed experiment descriptions, actual steps, results, source version, and Comet links to `/home/barichevavera3785/allexperiments.txt`; `experiment_journal.py` can record a run after its runner finishes.

- Never enable the scripted capital bot for training, evaluation, or replays unless the user explicitly requests it for that run. Pass `--no-scripted-bot` (or `scripted_capital_bot_enabled=False`) explicitly and use the same setting for training and evaluation.

- Never start training, evaluation, or replays with a boss starting roster unless the user explicitly requests that roster for the run.
- For maps with a boss roster by default, explicitly select the ordinary roster with `--no-boss-roster` (or `use_boss_starting_roster=False`). Apply the same roster setting to training and evaluation.
- The ordinary Legions party is two Одержимых, Герцог, and Сектант. Preserve map-specific fixed starting parties.
