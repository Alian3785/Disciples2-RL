# Changes Log

## [Unreleased]

### Added
- feat: Add automatic changes.md tracking system (cee4997)
- Add new 'walkrandomwalk' file and update action mechanics in 'walkthewalk.py' to include a heal action. Adjust action space and documentation to reflect changes in gameplay dynamics. (064f9eb)
- Add miss check for attacker accuracy in BattleEnv class of lesssimpleenv.py. Enhance combat mechanics by logging missed attacks, improving gameplay interactions. (da888ed)
- Update unit types to 'Death' in lesssimpleenv.py and implement poison application logic in BattleEnv class. Adjust unit attributes for improved gameplay mechanics, including damage and accuracy adjustments for 'Death' units, enhancing combat interactions. (7606752)
- Enhance unit attributes in lesssimpleenv.py by adding 'original_resilience' to track initial resilience values. Update BattleEnv class to initialize and restore these values during combat, improving resilience handling and gameplay mechanics. (3eb2ae0)
- Add 'Runningaway' attribute to units in battle_env.py and lesssimpleenv.py, enhancing gameplay mechanics. Update feature count in BattleEnv class to reflect new attribute and adjust related logic for improved unit behavior. (949641f)
### Changed
- Update unit types from 'Death' to 'Archer' for several units in lesssimpleenv.py, enhancing gameplay dynamics. Adjust accuracy values for specific units to improve combat mechanics and balance. (e32dc33)
### Fixed
### Removed
- Remove obsolete monitor and evaluation files from ppo_grid_combat_autoheal_origin and ppo_grid_combat_heal_action_at_origin directories to clean up the repository and improve organization. (4e5587a)
### Added
- Initial project setup
- Battle environment implementation
- PPO training setup
- Evaluation system

### Changed

### Fixed

### Removed

---

## [v0.1.0] - 2025-09-22

### Added
- Initial commit
- Basic environment files
- Training scripts
- Model checkpoints structure

### Changed

### Fixed

### Removed

---

*Формат: [Major.Minor.Patch] - Date*

*Added: для новых функций*
*Changed: для изменений в существующей функциональности*
*Fixed: для исправления ошибок*
*Removed: для удаленных функций*