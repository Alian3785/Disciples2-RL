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

# Rear-row melee attacks

- Any living ally in the front row blocks melee attacks from the back row, regardless of the ally's combat type. A living large ally also blocks them because it occupies both rows, including when anchored in a rear cell. Exclude the attacker itself; dead allies and small rear-row allies do not block.
- Apply the same reachability to action masks, direct action execution, and enemy AI. Blocked attacks must be absent from the action mask; a forced attack must not deal damage. Keep non-attack actions available under their existing rules and retain ranged, magic, and healing behavior.
- Transformations and Wight level reduction must preserve `position` and `stand`. The formation cell determines the row; never copy `stand` from a unit-form template.

# Map spell targeting

- Spell masks, observations, and casts must share the same nearest-enemy selector: Chebyshev distance `max(abs(dx), abs(dy))` without pathfinding or obstacle checks. Break ties by enemy ID. Preserve living-stack and spell-targetability filters, including summon exceptions; apply this to learned/purchased spells, scrolls, and staves.

# Effect lengths and unit data follow the original tables

- Paralysis, petrification and slow lengths follow each attack's `Gattacks.INFINITE`, not the engine `unit_type`: finite effects cost the target exactly its next turn; infinite paralysis is the long, recoverable one. Mermaid's paralysis is finite and Dark Elf Gast's is infinite although their types share the opposite default (see `FINITE_PARALYSIS_UNIT_IDS`/`INFINITE_PARALYSIS_UNIT_IDS`). Hermit's slow ends when the target's next turn begins.
- Unit reach and immunities follow `Gattacks.REACH` and `Gimmu`/`Gimmuc` (`L_ALWAYS` is an immunity, `L_ONCE` a one-time ward): Dark Elf Lyf is immune to Mind and Water, Beliarh is melee (adjacent), Verdant hits all enemies.

# Orbs and talismans

- Battle items follow their `GItem`/`Gattacks` attack. Fire, Inferno, Lightning, Earth, Stone Rain, Water, Icefall, Nosferat, Vampire and Elder Vampires hit every living enemy (`scope: party`); Thunder hits one. Their strikes carry `accuracy: 80` (POWER): each target rolls the normal hit check and takes damage through the standard pipeline (+0..5, armor, Defend). Vigor and Strength boost every living ally. Life revives with `revive_health_percent: 50` (QTY_HEAL). Drain orbs, like Vampire units, sum the damage dealt to all targets and heal half of it (`drainAttackHeal`/`drainOverflowHeal` = 50); Elder Vampires share the excess with wounded allies. Talismans copy their orb's effect. Effects without `accuracy` keep dealing their exact amount.

# Armor shatter

- Shatter (Theurgist, Sir Allemon secondary attack) cannot miss, as in Disciples II: its own accuracy is ignored, and it lands on every target that the main attack hit and damaged and that is still alive. Immunity and wards to its source still block it.

# Map battles start only by attacking a stack

- As in Disciples II, a battle starts only when the hero steps onto a living enemy stack's tile. That step is an attack: the hero does not enter the tile, fights from its current tile, and stays there after victory, retreat or timeout. Tiles next to stacks are freely passable; there is no 3x3 zone of control.
- An attack costs only the battle-entry movement (half the movement cap), not the enemy tile's terrain cost, and is not a stalled step. Enemy-initiated battles (scheduled waves reaching the hero, the scripted bot) keep their own rules.

# Services requiring a living leader

- Mercenary recruitment, trainer services (including training surviving companions), and buying spells at magic towers require a living leader. Apply this to masks, training previews, and direct execution; rejected actions must not spend gold, change stocks, grant XP, recruit units, or grant spells. Re-evaluate availability immediately after resurrection. Ordinary merchant purchases and resurrection retain their existing rules.

# Doppelganger copying and preparation

- Copy the target form's permanent stats without baking temporary numeric bonuses or penalties into the copy. Keep `damage` and `original_damage` synchronized so a new buff on the copy expires correctly. Preserve the copier's own permanent identity, growth, formation cell, and spent-turn state.
- Before round 1, eligible Doppelgangers on both teams get one preparation action in round 0: transform or skip with Defend. Wait, retreat, items, and damage attacks are unavailable then. Preparation preserves the normal first-round turn; transforming during a normal turn spends that turn. Untransformed Doppelgangers are not copy targets.

# Battle XP distribution

- Award XP once, at battle end, to the winning side: the `XP_KILLED` of units slain in this battle is split equally among winners alive at the end. Who was alive at each kill does not matter. Dead units, units that escaped before the end, thieves, and summoned units get nothing and take no share. A unit revived during the battle shares only kills made after its last revival. Apply the Weapon Master/Tome multiplier to the blue side's shares.
- Every level gained from XP (hero or non-evolving unit) fully restores health on the new maximum, as evolution does. Stopping at `XP_NEXT - 1` for a missing building is not a level and does not heal; dead units are never revived. Growth helpers keep the wound ratio; apply the heal after them in the XP award path.

# Test execution

- For local changes, start with affected test files: `python run_tests.py -n 0 tests/test_name.py`. Use `python run_tests.py` for the complete `tests/` regression suite; it defaults to two workers. Do not repeat the complete suite after passing validation unless further changes or failures justify it.
- The `campaign_factory` fixture may reuse initialized campaign snapshots for tests of later gameplay. Each call must have independent mutable state, callbacks, and random generators. Constructor, reset, and map-generation tests must keep real initialization. Do not cache action results, masks, or expected assertions.
