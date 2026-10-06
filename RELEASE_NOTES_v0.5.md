# Disciples2-RL v0.5

Version 0.5 brings together the changes made after v0.4: the Mirrow match scenario, configurable navigation windows, complete map-layout documentation, and fixes across combat, progression, spell use, settlement defence, and scripted opponents.

The `v0.5.0` tag and standard source archives use **`main`**, whose gameplay and map content were replaced with `obs/navigation-v2`, including the accumulated fixes merged in [PR #54](https://github.com/Alian3785/Disciples2-RL/pull/54). This differs from v0.4, whose archives used the earlier `main` and linked experimental branches separately. The navigation features already described in v0.4 are now included in the main release archive; they are not all new features of v0.5. `local5` and `obs/rosters-v3` remain separate experimental branches.

## New features

- Added **Mirrow match** (`mirrow_match`), a mirrored 48 x 48 scenario with ordinary Legions and Empire starting parties, faction-specific resource sites, 19 neutral armies near each capital, and one optional central Tiamat. The project spelling is preserved.
- Added the `scripted_bot` objective: the hero's own party must defeat the scripted opponent in tactical combat. Either attacking or defending counts; map spells, separate summons, city guards, neutral victories, and killing Tiamat do not substitute for that victory.
- The scripted opponent now pursues the hero when it has no other reachable enemy target. Its existing return-home and recovery behavior remains available.
- Navigation observations support **5 x 5, 7 x 7, and 10 x 10 local windows**, with matching bounds and observation construction. The API mode is still named `local5`; checkpoints must use the same window and observation implementation as training.
- Added runtime-derived layouts for **all 16 maps**, with PNG diagrams, JSON legends, army compositions, resources, items, and interaction cells. City Defence Training, Siege Training, and Super Last Stand also have separate wave sheets. A generator and regression tests keep the diagrams tied to the implemented maps.
- Added an English **Maps** table to the root README: all 16 maps, an approximate difficulty progression, Training / Realistic / Authentic categories, and short `train_map` commands with shared Bash and PowerShell setup. Default and Wotan's Vengeance are classified as Authentic. The shortcuts preserve `local5`, ordinary rosters, the 2,000,000-step default, and disabled Comet logging; City Defence Training and Mirrow match explicitly enable the scripted bot.

## Bug fixes

- Fixed Default and Small terrain loading in source snapshots. Runtime parsing of the original scenario no longer depends on a separate rendering/inspection tool, and missing or malformed terrain data is no longer silently replaced by an empty result.
- Corrected Wotan army imports for field stacks, settlement garrisons, and ruin guardians. Repeated formation-cell references to one large source unit create one fighter; distinct source IDs remain distinct even when their unit names match. Large units still reserve two formation cells.
- Combat XP now promotes the actual recipient rather than another same-named party member. This covers dead or living duplicates, branching promotions, and building requirements.
- Mind immunity and one-use wards are respected by primary and secondary hostile status effects, including transformations and fear. Secondary effects use their own attack source instead of inheriting the primary attack's source.
- Supported scrolls apply their canonical spell effects. Unsupported spells cannot be researched through an action mask or a direct action call.
- Disabled learning, purchasing, and casting illusion spells while preserving their existing action slots. They are not presented as implemented spell effects.
- Purchased level-V spells can be cast by every ruler type, including purchased spells from the ruler's own faction. Researching level-V spells remains restricted to the Archmage ruler.
- Melee action masks now match actual target reachability, avoiding advertised attacks that the battle engine would reject.
- Elven healer progression increases the healing stat actually used in battle. Temporary healer-granted wards preserve native ward provenance and do not erase a creature's original protection when the temporary grant expires.
- Permanent elixir bonuses are rebuilt consistently across equipment changes, level growth, promotions, transformations, copying, and battle transitions. Permanent progression is kept separate from temporary stat layers.
- City guards keep earned XP, level growth, and promotion after a defensive victory. Growth is also preserved for surviving enemies and across temporary battle or campaign buffs.
- Healing, revival, buffs, wards, and related support actions behave consistently for both teams, including their target masks, resource use, and lifecycle cleanup.
- Applied the original attack-characteristic limit after modifiers: **300 normally, or 400 for the ten supported native Heavy Strike hero IDs**. Random damage bonuses, critical effects, and mitigation are applied afterward. Healing and other support amounts are not incorrectly capped as offensive attacks, and copied/transformed attacks use the current form's identity.
- Fortification armor is applied to the defending side only. Attacking a city or capital no longer grants its defensive bonus to the attacker, and the temporary bonus is cleaned up without corrupting permanent armor or other buffs.
- City capture requires a living leader to enter the cleared settlement. A dead leader cannot claim ownership merely because surviving companions reached or cleared the location.
- Damage-over-time reapplication and lifecycle handling no longer lose or duplicate ticks across waiting, defending, cleansing, transformations, and battle cleanup. Slow removal restores initiative without granting an extra activation or skipping an eligible one.
- Hag's Ring fear does not bypass capital-guardian restrictions. In settlement battles where fear cannot cause retreat, it skips one activation instead; voluntary retreat restrictions remain separate.
- Resurrection checks the complete formation footprint, including both cells of a large unit and the form restored on revival. Occupied cells are rejected consistently by action masks and execution, without consuming the item or resource for an invalid revival.
- Restoring transformed units preserves spent activations and the live turn state. Expired temporary buffs are not revived from stale transformation snapshots, including Tiamat and nested form changes.
- XP from corpses and repeated deaths is accounted for without duplicate credit. A surviving Doppelganger returns from a copied form without being rounded down to zero HP and keeps its own permanent growth.
- Transformed summoners retain correct summon ownership and cleanup. Temporary summons are removed at the right battle boundary without transferring ownership or corrupting persistent parties.
- Sequential territory expansion maintains one current owner per cell and applies faction ownership changes consistently.
- Defeated scripted opponents respawn after **two completed turns** on all maps, including occupied-home encounters. City Defence Training therefore uses the shared respawn delay: after a first victory on turn 2, the next assault reaches the city on turn 6 rather than turn 4.
- Final training win-rate accounting includes episodes that reach the evaluation step cap instead of silently omitting those outcomes.

## Other changes

- Promoted the complete `obs/navigation-v2` content to `main`, preserving the previous history and a backup branch. The workflow that automatically closed pull requests based on the triggering account is disabled, so contributors can submit PRs to `main`.

- Rebalanced Siege Training's early patrols and mana access. The map has 32 static armies and a pursuing siege army that appears on turn 27; its scheduled attacker is separate from the optional scripted capital bot.
- Added rolling statistics for unique defeated squads over the last 100 episodes, with deduplication and resets per environment. Auxiliary combat counts remain separate from campaign-objective completion.
- Preserved the nine-variant Wotan observation comparison and city-objective diagnostics. The recorded experiment totals 53,000,000 training steps, 107,726 completed training episodes, and 5,300 evaluation episodes, with **zero campaign victories**. The navigation variant's higher auxiliary reward is not evidence that it learned to complete the scenario.
- The release includes the local observation, city-garrison, recovery-budget, dismissal, and experiment tooling previously documented for the navigation branch in v0.4. This is archive coverage inherited from that branch, not a claim that these features were introduced again.
- Added regression coverage for the listed mechanics and map diagrams. The full local run on the gameplay snapshot published as `e3dcda7` completed with **8,133 passed and 162 subtests passed**, plus the pre-existing `test_local_observation.py::test_locality` failure (37 of 2,335 elements). The suite is therefore not fully green. Documentation changes do not alter that gameplay snapshot.
- CodeRabbit completed the accumulated-fixes review. One minor map-generator font-fallback issue and the docstring-coverage warning remain; neither is described as fixed in this release. No passing GitHub Actions run is claimed for the release.
- Map, objective, observation-window, roster, and scripted-bot settings must match the saved model and normalization statistics. New maps and changed observation sizes require compatible checkpoints or retraining; weights are not interchangeable merely because the API mode is called `local5`.
- Wotan remains a static scenario adaptation. Its original story-event execution, including Uther's appearance and the giant's change of allegiance, is not implemented. The release does not claim complete parity with the original game or successful mastery of the campaigns.

Full tagged history: [v0.4.0 to v0.5.0](https://github.com/Alian3785/Disciples2-RL/compare/v0.4.0...v0.5.0). Because v0.5 promotes the navigation branch into `main`, this comparison also includes the navigation work already documented in v0.4.

Changes since the navigation snapshot linked by v0.4: [8c9fa0b to v0.5.0](https://github.com/Alian3785/Disciples2-RL/compare/8c9fa0bb270ac8ae6fbdfff844d602d298064505...v0.5.0).
