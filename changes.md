# Changes Log

## [Unreleased]

### Added
- Refactor gameplay mechanics in walkrandomwalk.py and walkthewalk.py, enhancing combat dynamics with a new GridWorldCombatEnv. Update changes.md to document these improvements and consolidate previous entries for clarity. Streamline the repository by removing obsolete monitor and evaluation files from specific directories. (c7d6077)
- Refactor gameplay mechanics in walkrandomwalk.py and walkthewalk.py, enhancing combat dynamics with a new GridWorldCombatEnv. Update changes.md to document these improvements and consolidate previous entries for clarity. Remove obsolete monitor and evaluation files from ppo_grid_combat_autoheal_origin and ppo_grid_combat_heal_action_at_origin directories to streamline the repository. (a8a5c9f)
- Refactor gameplay mechanics in walkrandomwalk.py and walkthewalk.py. Introduce a new GridWorldCombatEnv for enhanced combat dynamics, including healing actions and improved enemy spawning logic. Update changes.md to document these gameplay enhancements and consolidate previous entries for clarity. (5670413)
- Consolidate updates in changes.md to reflect recent gameplay enhancements, including improved unit attributes and action mechanics. Add new features for automatic tracking and categorization of changes, ensuring clarity and organization in the documentation. (7ba2777)
- feat: Improve automatic categorization of commits in changes.md (45f4104)
- feat: Add automatic changes.md tracking system (cee4997)
- Add new 'walkrandomwalk' file and update action mechanics in 'walkthewalk.py' to include a heal action. Adjust action space and documentation to reflect changes in gameplay dynamics. (064f9eb)
### Changed
- Update .gitignore to exclude 'logs/' and 'changes.md'. Refactor walkrandomwalk.py by moving stable_baselines3 imports to the top and increasing total_timesteps for model training from 1e6 to 5e6, enhancing training duration for improved performance. (1cbb6df)
- Update changes.md to reflect recent enhancements in gameplay mechanics, including improved unit attributes, action mechanics, and automatic tracking system for changes. Consolidate previous entries for clarity and organization. (fcaeabe)
- Update unit types from 'Death' to 'Archer' for several units in lesssimpleenv.py, enhancing gameplay dynamics. Adjust accuracy values for specific units to improve combat mechanics and balance. (e32dc33)
### Removed
- Remove obsolete monitor and evaluation files from ppo_grid_combat_autoheal_origin and ppo_grid_combat_heal_action_at_origin directories to clean up the repository and improve organization. (4e5587a)
### Added
- Refactor gameplay mechanics in walkrandomwalk.py and walkthewalk.py, enhancing combat dynamics with a new GridWorldCombatEnv. Update changes.md to document these improvements and consolidate previous entries for clarity. Remove obsolete monitor and evaluation files from ppo_grid_combat_autoheal_origin and ppo_grid_combat_heal_action_at_origin directories to streamline the repository. (a8a5c9f)
- Refactor gameplay mechanics in walkrandomwalk.py and walkthewalk.py. Introduce a new GridWorldCombatEnv for enhanced combat dynamics, including healing actions and improved enemy spawning logic. Update changes.md to document these gameplay enhancements and consolidate previous entries for clarity. (5670413)
- Consolidate updates in changes.md to reflect recent gameplay enhancements, including improved unit attributes and action mechanics. Add new features for automatic tracking and categorization of changes, ensuring clarity and organization in the documentation. (7ba2777)
- feat: Improve automatic categorization of commits in changes.md (45f4104)
- feat: Add automatic changes.md tracking system (cee4997)
- Add new 'walkrandomwalk' file and update action mechanics in 'walkthewalk.py' to include a heal action. Adjust action space and documentation to reflect changes in gameplay dynamics. (064f9eb)