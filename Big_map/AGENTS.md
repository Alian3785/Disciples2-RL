# Training preferences

- Always launch training in `local5` observation mode unless the user explicitly requests a different mode. Set `CAMPAIGN_OBSERVATION_VERSION=local5` (or `observation_version="local5"`) explicitly and use the same mode for evaluation and replays.
- Append completed experiment descriptions, actual steps, results, source version, and Comet links to `/home/barichevavera3785/allexperiments.txt`; `experiment_journal.py` can record a run after its runner finishes.

- Never enable the scripted capital bot for training, evaluation, or replays unless the user explicitly requests it for that run. Pass `--no-scripted-bot` (or `scripted_capital_bot_enabled=False`) explicitly and use the same setting for training and evaluation.

- Never start training, evaluation, or replays with a boss starting roster unless the user explicitly requests that roster for the run.
- For maps with a boss roster by default, explicitly select the ordinary roster with `--no-boss-roster` (or `use_boss_starting_roster=False`). Apply the same roster setting to training and evaluation.
- The ordinary Legions party is two Одержимых, Герцог, and Сектант. Preserve map-specific fixed starting parties.

# Map conversion: large units

- A large Disciples II unit occupies two formation cells but remains one fighter. Multiple source `POS_*` cells can reference the same `UNIT_*`/`source_unit_id`; create one runtime fighter per unique source unit ID within each group. Never deduplicate by name or unit type: different IDs with identical names are separate legal units.
- Apply this to field armies, settlement/capital garrisons, and ruin guardians. Preserve source IDs and raw cell references, and reserve both cells for each large runtime unit.
- Validate every imported army against unique source IDs before and after battle initialization: composition, fighter count, total HP/XP, and six-cell capacity without overlaps. Include two distinct same-named large units in regression coverage. Run `tests/test_wotan_enemy_import.py` and `tests/test_map_army_integrity.py` when changing map conversion or army construction, and add source comparisons for new maps.
