"""Bridge the mobile scripted opponent to normal hero and spell encounters.

The bot is a dynamic enemy. It has no fixed global-observation slot, so existing
map/checkpoint ABIs are unchanged; local observations use its current tile.
"""
from copy import deepcopy


class CampaignBotEncounterMixin:
    def _is_scripted_bot_enemy(self, enemy_id):
        return enemy_id is not None and int(enemy_id) == self.SCRIPTED_CAPITAL_BOT_INTERNAL_ENEMY_ID

    def _scripted_bot_is_present(self):
        return (bool(getattr(self, "scripted_capital_bot_enabled", False))
                and getattr(self, "scripted_capital_bot_state", "disabled")
                not in {"disabled", "defeated"})

    @staticmethod
    def _convert_scripted_bot_team(team, target_team):
        converted = deepcopy(team)
        offset = -6 if target_team == "red" else 6
        for unit in converted:
            unit["position"] = int(unit["position"]) + offset
            unit["team"] = target_team
            front = (1, 2, 3) if target_team == "red" else (7, 8, 9)
            unit["stand"] = "ahead" if unit["position"] in front else "behind"
        return converted

    def _refresh_scripted_bot_enemy_team(self, *, force=False):
        team = getattr(self, "scripted_capital_bot_team_state", None)
        if team is None or not hasattr(self, "enemy_team_states"):
            return
        if force or getattr(self, "_scripted_bot_enemy_source_team", None) is not team:
            self.enemy_team_states[self.SCRIPTED_CAPITAL_BOT_INTERNAL_ENEMY_ID] = (
                self._convert_scripted_bot_team(team, "red"))
            self._scripted_bot_enemy_source_team = team

    def _export_scripted_bot_enemy_team(self):
        enemy_id = self.SCRIPTED_CAPITAL_BOT_INTERNAL_ENEMY_ID
        team = getattr(self, "enemy_team_states", {}).get(enemy_id)
        if team is not None:
            self.scripted_capital_bot_team_state = self._convert_scripted_bot_team(team, "blue")
            self._scripted_bot_enemy_source_team = self.scripted_capital_bot_team_state

    def _init_scripted_capital_bot_state(self):
        self.scripted_bot_hero_victories = 0
        self.scripted_bot_objective_reward_granted = False
        self._scripted_bot_enemy_source_team = None
        super()._init_scripted_capital_bot_state()
        self._enemy_descriptions = dict(self._enemy_descriptions)
        self._enemy_descriptions[self.SCRIPTED_CAPITAL_BOT_INTERNAL_ENEMY_ID] = (
            f"Скриптовый бот: {self.scripted_capital_bot_faction}")

    def _reset_scripted_capital_bot_state(self):
        self.scripted_bot_hero_victories = 0
        self.scripted_bot_objective_reward_granted = False
        self._scripted_bot_enemy_source_team = None
        super()._reset_scripted_capital_bot_state()

    def _scripted_capital_bot_info(self):
        result = super()._scripted_capital_bot_info()
        result["scripted_bot_hero_victories"] = int(
            getattr(self, "scripted_bot_hero_victories", 0))
        return result

    def _sync_scripted_capital_bot_grid_state(self):
        super()._sync_scripted_capital_bot_grid_state()
        if not hasattr(self, "grid_env"):
            return
        grid = self.grid_env
        enemy_id = self.SCRIPTED_CAPITAL_BOT_INTERNAL_ENEMY_ID
        grid.dynamic_enemy_ids.add(enemy_id)
        present = self._scripted_bot_is_present()
        position = tuple(getattr(self, "scripted_capital_bot_position", ()))
        if present and len(position) == 2:
            grid.enemy_positions[enemy_id] = position
            grid.enemies_alive[enemy_id] = True
            # An enemy tile is an attack destination, not an impassable wall.
            grid.dynamic_blocked_positions.discard(position)
        elif enemy_id in grid.enemy_positions:
            grid.enemies_alive[enemy_id] = False
        else:
            grid.enemies_alive.pop(enemy_id, None)
        self._refresh_scripted_bot_enemy_team(force=True)

    def _get_enemy_team_state(self, enemy_id):
        if self._is_scripted_bot_enemy(enemy_id):
            self._refresh_scripted_bot_enemy_team()
            return self.enemy_team_states.get(int(enemy_id), [])
        return super()._get_enemy_team_state(enemy_id)

    def _advance_scripted_capital_bot_one_turn(self):
        self._refresh_scripted_bot_enemy_team(force=True)
        result = super()._advance_scripted_capital_bot_one_turn()
        self._refresh_scripted_bot_enemy_team(force=True)
        return result

    def _restore_enemy_stack_spell_bases(self, enemy_id):
        super()._restore_enemy_stack_spell_bases(enemy_id)
        if self._is_scripted_bot_enemy(enemy_id):
            self._export_scripted_bot_enemy_team()

    def _cast_from_spell_spec(self, **kwargs):
        self._refresh_scripted_bot_enemy_team(force=True)
        result = super()._cast_from_spell_spec(**kwargs)
        if self._is_scripted_bot_enemy(result.get("target_enemy_id")):
            self._export_scripted_bot_enemy_team()
            if result.get("spell_enemy_defeated"):
                self._mark_scripted_capital_bot_defeated()
                result["objective_reward_eligible"] = False
                result["objective_defeat_source"] = "map_spell"
            self._sync_scripted_capital_bot_grid_state()
        return result

    def _save_enemy_state_from_battle(self, enemy_id):
        super()._save_enemy_state_from_battle(enemy_id)
        if self._is_scripted_bot_enemy(enemy_id):
            self._export_scripted_bot_enemy_team()

    def _init_battle(self, enemy_id, **kwargs):
        if self._is_scripted_bot_enemy(enemy_id):
            self.current_battle_context["scripted_bot"] = True
            self._scripted_bot_pending_hero_encounter = False
        return super()._init_battle(enemy_id, **kwargs)

    def _record_scripted_bot_combat_victory(self, enemy_id, context_kind, reward, info):
        if not self._is_scripted_bot_enemy(enemy_id) or not self._scripted_bot_is_present():
            return float(reward)
        self._mark_scripted_capital_bot_defeated()
        eligible = context_kind == "hero" and bool(self.current_battle_context.get("scripted_bot"))
        info.update(scripted_bot_defeated=True, objective_reward_eligible=eligible,
                    objective_defeat_source="hero_battle" if eligible else context_kind)
        if eligible:
            self.scripted_bot_hero_victories += 1
        if (eligible and self._campaign_objective_is_scripted_bot()
                and not self.scripted_bot_objective_reward_granted):
            bonus = max(0.0, float(self.reward_all_enemies))
            reward += bonus
            self.scripted_bot_objective_reward_granted = True
            info.update(final_objective_reward=bonus, scripted_bot_objective_reward=bonus)
        info["scripted_bot_hero_victories"] = self.scripted_bot_hero_victories
        return float(reward)

    def _terminal_defeat_opponent_info(self, enemy_id):
        info = super()._terminal_defeat_opponent_info(enemy_id)
        if self._is_scripted_bot_enemy(enemy_id):
            info["terminal_defeat_enemy_units"] = self._terminal_defeat_unit_names(
                self.scripted_capital_bot_team_state)
        return info

    def _finalize_summon_spell_battle(self, **kwargs):
        enemy_id = self.current_enemy_id
        result = super()._finalize_summon_spell_battle(**kwargs)
        if self._is_scripted_bot_enemy(enemy_id) and kwargs.get("winner") == "blue":
            self._mark_scripted_capital_bot_defeated()
            obs, reward, done, timeout, info = result
            info = dict(info, objective_reward_eligible=False,
                        objective_defeat_source="summon_battle",
                        **self._scripted_capital_bot_info())
            return self._build_obs(grid_obs=self._get_grid_obs()), reward, done, timeout, info
        return result

    def _step_battle(self, action):
        enemy_id = self.current_enemy_id
        battle = self.battle_env
        is_bot = self._is_scripted_bot_enemy(enemy_id)
        result = super()._step_battle(action)
        if not is_bot or battle is None or self.mode != self.MODE_GRID:
            return result
        obs, reward, terminated, truncated, info = result
        if info.get("battle_result") == "victory":
            if self._scripted_bot_is_present():
                # Summoned armies use their own completion route and cannot win
                # Mirrow match. A city reserve likewise is not the hero army.
                self._mark_scripted_capital_bot_defeated()
        elif self._scripted_bot_is_present():
            # The normal bridge may already have cleared battle_env. Preserve
            # its proven roster/stat cleanup using the completed battle object.
            current = self.battle_env
            try:
                self.battle_env = battle
                self._save_enemy_state_from_battle(enemy_id)
            finally:
                self.battle_env = current
            self.scripted_capital_bot_state = "returning"
        self._sync_scripted_capital_bot_grid_state()
        info = dict(info, **self._scripted_capital_bot_info())
        return self._build_obs(grid_obs=self._get_grid_obs()), reward, terminated, truncated, info

    def _finish_pending_scripted_bot_encounter(self, result):
        obs, reward, terminated, truncated, info = result
        if not getattr(self, "_scripted_bot_pending_hero_encounter", False):
            return result
        if terminated or truncated or not self._scripted_bot_is_present():
            self._scripted_bot_pending_hero_encounter = False
            return result
        if self.mode != self.MODE_GRID:
            return result
        if not self._scripted_bot_can_engage_enemy_from_position(
                self.scripted_capital_bot_position, self.grid_env.agent_pos):
            self._scripted_bot_pending_hero_encounter = False
            return result
        self.current_enemy_id = self.SCRIPTED_CAPITAL_BOT_INTERNAL_ENEMY_ID
        self.battle_origin_pos = tuple(self.grid_env.agent_pos)
        self.current_battle_context = {"kind": "hero", "scripted_bot": True}
        self.mode = self.MODE_BATTLE
        self._init_battle(self.current_enemy_id, attacker_team="red")
        info = dict(info, mode="battle", battle_triggered=True,
                    battle_triggered_by="scripted_bot", enemy_id=self.current_enemy_id)
        return self._build_obs(battle_obs=self.battle_env._obs()), reward, False, False, info
