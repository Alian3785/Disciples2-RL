# Local observation experiment

Branch: `experiment/local-observation-5x5`, based on original baseline `76a5d3b`.
The default `CampaignEnv` observation on this branch is `local5`, including for
direct constructors, training, evaluation and replay scripts using this checkout.
Explicit compatibility mode: `CAMPAIGN_OBSERVATION_VERSION=baseline`, or
`CampaignEnv(observation_version="baseline", use_boss_starting_roster=False)`.
Models and VecNormalize files must be loaded with their original observation
version. Old baseline models cannot be loaded with the new default.

## Observation contract

The fixed-size vector contains:

- A 5x5 square centered on the hero, with radius two in both axes, including
  diagonals. Tiles are row-major (dy, dx), from (-2, -2) to (+2, +2).
- Each tile has 16 channels: in bounds, walkable, capital, healing, chest, ruin,
  merchant interaction, spell-shop interaction, mercenary interaction, trainer
  interaction, Legions territory, Empire territory, gold mine, mana kind, living
  enemy stack present, and whether that stack is the objective enemy.
- Visible enemy stacks retain baseline per-unit type one-hot and HP ratio, in
  their original formation slot order. Their position is the tile in the patch,
  not a global ID or absolute coordinate. Absent enemies have all-zero features.
  Standard maps have one active stack per tile; overlapping custom-map stacks
  use the first ID, matching encounter ordering.
- The original six-slot party block (alive, HP ratio, wounded, revivable,
  invulnerability potion, strength potion, hero flag, level), original resource
  and equipment block except the remaining-chest count, buildings, learned
  spells/cast limits/scrolls except the nearest-enemy debuff summary.
- Stock and hiring data only for the site where the hero currently stands,
  current trainer services, public next-wave countdown and completed-wave
  progress, mode, turns, gold, and the unchanged current battle observation.

There is no map memory: no visited flags even inside the patch, explored-cell
fraction, stored distant cells, off-screen enemy slots, absolute position,
global object coordinates, remaining chest/defender counts, distant target or
capital pointers, nearest-wave distance, or nearest-enemy spell-effect summary.
Tiles outside map boundaries are zero padded. Local terrain is visible across
the entire square; this is a radius-limited view, not line-of-sight ray tracing.

The local vector is built directly; the full global observation is not built
and cropped. Native grid internals still compute their small legacy vector when
stepping, but it is discarded and never reaches the policy in local mode.

## Scope and reference

This follows the principle of a local world view plus player state used in
[Crafter's original environment](https://github.com/danijar/crafter/blob/main/crafter/env.py).
It is a numerical 5x5 adaptation for the existing MLP, not Crafter's RGB input
or a replication of the paper's complete method.

Rewards, transition rules, battle observations, action space and action masks
are unchanged. In particular, the existing global spell targeting can still
select an off-screen opponent, and mask availability can indirectly reveal
target existence/eligibility. Therefore the **observation vector** is local;
the entire agent interface is not a strict no-information POMDP. In a summoned
battle the current opponent is visible in the battle block even if remote from
the hero. Changing those rules would be a separate experiment.

## Evaluation

`python3 test_local_observation.py` checks hidden-world invariance, radius two
versus radius three, boundary padding, default mode and 600 identical legal
actions per scenario with paired RNG, checking rewards, termination, masks,
positions, gold, battle block equality and immutable previous observations.

`python3 run_local_experiment.py` runs serially with frozen hashes and saves
commands, profiles, models, VecNormalize and per-episode assessments. All train
and evaluation commands explicitly pass `--no-boss-roster`. Fixed map parties
remain unchanged. PPO uses the previous experiment's settings: seed 42, 16
environments, four Torch threads, 20k rollout, batch 500, five epochs.

New training totals 5.3M transitions: orc 500k, four scenarios at 800k, then
800k more for each dragon scenario. Each assessment uses 20 deterministic and
20 stochastic episodes, seeds 10000–10019 and fixed evaluation battle RNG.
Baseline and v2 references come from the completed `../obs_experiment_1`
experiment, with the same budgets and the same two-stage continuation protocol.
They are historical controls, not freshly retrained runs. One training seed
does not establish statistical superiority; speed comparisons also reflect
behavior and machine load at different times.
