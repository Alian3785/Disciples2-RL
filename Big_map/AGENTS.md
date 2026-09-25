# Training preferences

- Never enable the scripted capital bot for training, evaluation, or replays unless the user explicitly requests it for that run. Pass `--no-scripted-bot` (or `scripted_capital_bot_enabled=False`) explicitly and use the same setting for training and evaluation.

- Never start training, evaluation, or replays with a boss starting roster unless the user explicitly requests that roster for the run.
- For maps with a boss roster by default, explicitly select the ordinary roster with `--no-boss-roster` (or `use_boss_starting_roster=False`). Apply the same roster setting to training and evaluation.
- The ordinary Legions party is two Одержимых, Герцог, and Сектант. Preserve map-specific fixed starting parties.
