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

## JAX prototype

A JAX/GPU prototype is available in **[jax/](jax/README.md)**. It runs NumberGrid's 24 x 24 map, squad battles with five archers and one area-damage mage, and Stoix Anakin PPO entirely on the GPU. The current scope is movement, tactical squad combat, and clearing 24 enemy squads.

The 20-million-step run achieved **84.08% full-map wins (861/1024)** with argmax actions and **82.03% (840/1024)** with sampled actions on its fixed map. The [training report](jax/LOCAL_RESULTS.md), [checkpoint and configuration](jax/results/number_grid-mage-20m-20261006/), and [W&B charts](https://wandb.ai/sergey3784/numbergrid/runs/7cddiev7) are included. Setup, training, replays, and interactive play are documented in the JAX README.

The imported JAX project and vendored Stoa retain their [project license](jax/LICENSE) and [Stoa license](jax/stoa-src/LICENSE).

## Release notes

**[v0.5](https://github.com/Alian3785/Disciples2-RL/releases/tag/v0.5.0)** — navigation observations in `main`, Mirrow match, all 16 map layouts, and combat and campaign fixes. [Detailed release notes](RELEASE_NOTES_v0.5.md).

## Maps

All **16 implemented maps** are listed below in a suggested easy-to-hard learning progression, grouped by category. This is an approximate curriculum based on scenario complexity, not a measured difficulty ranking; starting parties, objectives, and training settings can change the order.

- **Training:** focused exercises for individual mechanics.
- **Realistic:** custom or simplified scenarios combining campaign mechanics.
- **Authentic:** adaptations of existing Disciples II scenarios, including Default and Wotan's Vengeance.

These commands use the v0.5 code in `main`. Install `Big_map/requirements.txt`, then define the shortcut below once in your terminal, from the repository root.

**Bash (Linux/macOS):**

```bash
train_map() {
    CAMPAIGN_OBSERVATION_VERSION=local5 python Big_map/train_campaign.py \
        --no-scripted-bot --no-boss-roster --no-comet --map "$@"
}
```

<details>
<summary>PowerShell (Windows)</summary>

```powershell
function train_map {
    $env:CAMPAIGN_OBSERVATION_VERSION = "local5"
    python Big_map/train_campaign.py --no-scripted-bot --no-boss-roster --no-comet --map @args
}
```

Use this PowerShell definition instead of the Bash one. The commands in the table are the same in both shells.

</details>

Each command uses the map's default objective, `local5` observations, ordinary starting parties, and no Comet logging. The trainer defaults to **2,000,000 steps**. Fixed scenario rosters are preserved. Append options to override the defaults, for example `train_map orc_duel --total-steps 500000 --n-envs 4`.

| Map | Category | Training command |
| --- | --- | --- |
| [Orc Duel](Big_map/maps/orc_duel/README.md) (`orc_duel`) | Training | `train_map orc_duel` |
| [Builder](Big_map/maps/builder/README.md) (`builder`) | Training | `train_map builder` |
| [Hire Training](Big_map/maps/hire_train/README.md) (`hire_train`) | Training | `train_map hire_train` |
| [City Defence Training](Big_map/maps/city_defence_train/README.md) (`city_defence_train`) | Training | `train_map city_defence_train --scripted-bot` |
| [Formation Training](Big_map/maps/formation_train/README.md) (`formation_train`) | Training | `train_map formation_train` |
| [Magic Training](Big_map/maps/magic_train/README.md) (`magic_train`) | Training | `train_map magic_train` |
| [Item Training](Big_map/maps/item_train/README.md) (`item_train`) | Training | `train_map item_train` |
| [Trade Training](Big_map/maps/trade_train/README.md) (`trade_train`) | Training | `train_map trade_train` |
| [Scroll Training](Big_map/maps/scroll_train/README.md) (`scroll_train`) | Training | `train_map scroll_train` |
| [Green Dragon Minimal](Big_map/maps/green_dragon_minimal/README.md) (`green_dragon_minimal`) | Realistic | `train_map green_dragon_minimal` |
| [Siege Training](Big_map/maps/siege_train/README.md) (`siege_train`) | Realistic | `train_map siege_train` |
| [Small](Big_map/maps/small/README.md) (`small`) | Realistic | `train_map small` |
| [Mirrow match](Big_map/maps/mirrow_match/README.md) (`mirrow_match`) | Realistic | `train_map mirrow_match --scripted-bot` |
| [Super Last Stand](Big_map/maps/super_last_stand/README.md) (`super_last_stand`) | Realistic | `train_map super_last_stand` |
| [Default: A Return to Simpler Times](Big_map/maps/default/README.md) (`default`) | Authentic | `train_map default` |
| [Wotan's Vengeance](Big_map/maps/wotans_retribution/README.md) (`wotans_retribution`) | Authentic | `train_map wotans_retribution` |

- City Defence Training and Mirrow match explicitly enable the scripted bot because their victory conditions require it. Siege Training and Super Last Stand use separate scheduled attackers, which remain active with `--no-scripted-bot`.
- Wotan's Vengeance uses the existing map ID `wotans_retribution`; its detailed map documentation also calls it Wotan's Retribution. The project spelling `mirrow_match` is intentional.
- Default, Small, and Wotan use city-capture objectives. Green Dragon Minimal already uses its dragon objective, so no additional goal flag is needed.
- Keep each model paired with its own saved normalization statistics and the same map, observation, objective, roster, and bot settings for evaluation and replays. The trainer creates run-specific artifact directories; use `--n-envs` and `--total-steps` to adjust resource use and budget.
- Map links include layouts, army compositions, interaction coordinates, and wave diagrams where applicable. Authentic maps remain adaptations: original visuals and scripted story events are not reproduced.

## Tests

With `Big_map/requirements.txt` installed, run from `Big_map`:

| Check | Command |
| --- | --- |
| Regression suite (`tests/`) | `python run_tests.py` |
| One affected test file | `python run_tests.py -n 0 tests/test_doppelganger_zero_turn.py` |
| Serial run for debugging | `python run_tests.py -n 0` |
| Slowest checks | `python run_tests.py --durations=20` |

The runner uses two [pytest-xdist workers](https://pytest-xdist.readthedocs.io/en/stable/distribution.html), forwards pytest options, and disables battle dump files. `-n 0` runs without workers; `-n 4` explicitly requests four. Normal `python -m pytest tests` still works.

The permanent-elixir and purchased-spell tests reuse in-memory snapshots of initialized campaigns. Every call receives independent mutable state and random generators. Tests of map construction and reset still perform real initialization. After a small change, run the affected files first; reserve the complete suite for changes spanning several subsystems or final validation.
