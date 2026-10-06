So, this is the project to train a Superhuman bot for Disciples 2, popular turn-based strategy game. There are 3 stages:

1. **Single-player scenarios**
   First, an agent is trained to complete individual single-player scenarios. Autoresearch is used to test multiple configurations.
   [Preliminary roadmap](https://github.com/users/Alian3785/projects/2)

2. **Randomly generated and unseen maps**
   Next, a more general agent is trained to play previously unseen maps, face randomly generated enemies, and handle a wider variety of situations.

3. **Self-play training**
   Finally, an AlphaZero-like agent is trained through self-play, with the goal of competing against the best *Disciples II* players and eventually beating the champion—whoever that may be.
   [Why not just use AlphaZero right away?](https://github.com/Alian3785/Disciples2-RL/blob/main/Why%20not%20just%20use%20AlphaZero.md)

The project implements campaign movement, tactical battles, party progression, settlements, economy, magic, and items in Python. It remains an adaptation of the original game: known omissions include rod planters, recruiting additional heroes, thief gameplay, non-interactive map objects, scripted story events, and original visuals. Some spell mechanics, including illusions, are disabled.

You can run agent training for a real Disciples 2 scenario "A Return To Simpler Times" (with the exceptions above) with the following command. By default, a slightly modified Maskable PPO algorithm from [Stable Baselines 3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) is used.

```bash
CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map default --no-scripted-bot --no-boss-roster
```

The agent is being trained for *Disciples II: Rise of the Elves*:
[Disciples II: Rise of the Elves on Steam](https://store.steampowered.com/app/1630/Disciples_II_Rise_of_the_Elves/)

## Maps

All **16 implemented maps** are listed below in a suggested easy-to-hard learning progression, grouped by category. This is an approximate curriculum based on scenario complexity, not a measured difficulty ranking; starting parties, objectives, and training settings can change the order.

- **Training:** focused exercises for individual mechanics.
- **Realistic:** custom or simplified scenarios combining campaign mechanics.
- **Authentic:** adaptations of existing Disciples II scenarios, including Default and Wotan's Vengeance.

These commands target the `obs/navigation-v2` code used for v0.5. Run them from the repository root after installing `Big_map/requirements.txt` in your Python environment. Each command selects the map's default objective, explicitly uses `local5` observations and ordinary starting parties, requests 2,000,000 training steps, and disables Comet logging. Fixed scenario rosters are preserved.

| Map | Category | Training command |
| --- | --- | --- |
| [Orc Duel](Big_map/maps/orc_duel/README.md) (`orc_duel`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map orc_duel --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Builder](Big_map/maps/builder/README.md) (`builder`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map builder --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Hire Training](Big_map/maps/hire_train/README.md) (`hire_train`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map hire_train --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [City Defence Training](Big_map/maps/city_defence_train/README.md) (`city_defence_train`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map city_defence_train --scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Formation Training](Big_map/maps/formation_train/README.md) (`formation_train`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map formation_train --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Magic Training](Big_map/maps/magic_train/README.md) (`magic_train`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map magic_train --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Item Training](Big_map/maps/item_train/README.md) (`item_train`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map item_train --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Trade Training](Big_map/maps/trade_train/README.md) (`trade_train`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map trade_train --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Scroll Training](Big_map/maps/scroll_train/README.md) (`scroll_train`) | Training | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map scroll_train --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Green Dragon Minimal](Big_map/maps/green_dragon_minimal/README.md) (`green_dragon_minimal`) | Realistic | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map green_dragon_minimal --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Siege Training](Big_map/maps/siege_train/README.md) (`siege_train`) | Realistic | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map siege_train --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Small](Big_map/maps/small/README.md) (`small`) | Realistic | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map small --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Mirrow match](Big_map/maps/mirrow_match/README.md) (`mirrow_match`) | Realistic | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map mirrow_match --scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Super Last Stand](Big_map/maps/super_last_stand/README.md) (`super_last_stand`) | Realistic | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map super_last_stand --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Default: A Return to Simpler Times](Big_map/maps/default/README.md) (`default`) | Authentic | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map default --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |
| [Wotan's Vengeance](Big_map/maps/wotans_retribution/README.md) (`wotans_retribution`) | Authentic | `CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py --map wotans_retribution --no-scripted-bot --no-boss-roster --total-steps 2000000 --no-comet` |

- These are Bash commands for Linux/macOS. In PowerShell, first run `$env:CAMPAIGN_OBSERVATION_VERSION = "local5"`, then run the chosen command starting with `python`.
- City Defence Training and Mirrow match explicitly enable the scripted bot because their victory conditions require it. Siege Training and Super Last Stand use separate scheduled attackers, which remain active with `--no-scripted-bot`.
- Wotan's Vengeance uses the existing map ID `wotans_retribution`; its detailed map documentation also calls it Wotan's Retribution. The project spelling `mirrow_match` is intentional.
- Default, Small, and Wotan use city-capture objectives. Green Dragon Minimal already uses its dragon objective, so no additional goal flag is needed.
- Keep each model paired with its own saved normalization statistics and the same map, observation, objective, roster, and bot settings for evaluation and replays. The trainer creates run-specific artifact directories; use `--n-envs` and `--total-steps` to adjust resource use and budget.
- Map links include layouts, army compositions, interaction coordinates, and wave diagrams where applicable. Authentic maps remain adaptations: original visuals and scripted story events are not reproduced.
