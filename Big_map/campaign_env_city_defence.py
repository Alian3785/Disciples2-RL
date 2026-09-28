"""Map-specific assault schedule using the normal city-garrison battle rules."""
from maps.city_defence_train import ASSAULT_FACTIONS, ASSAULT_ROSTERS, CITY_NAME


class CampaignCityDefenceMixin:
    def _is_city_defence_scenario(self):
        return self.map_name == "city_defence_train"

    def _init_scripted_capital_bot_state(self):
        self.city_defence_wave = 0
        self.city_defence_wins = 0
        self.city_defence_spawn_turn = 0
        super()._init_scripted_capital_bot_state()

    def _reset_scripted_capital_bot_state(self):
        self.city_defence_wave = 0
        self.city_defence_wins = 0
        self.city_defence_spawn_turn = int(self.turns)
        super()._reset_scripted_capital_bot_state()
        if self._is_city_defence_scenario():
            self.scripted_capital_bot_faction = ASSAULT_FACTIONS[0]
            # Same ownership/frontier as a city captured on this turn, no
            # recruitment, capture reward, or modification of the hero party.
            self.legions_active_settlement_territory_capture_turn_by_name[CITY_NAME] = self.turns
            self.city_garrisons[CITY_NAME] = []
            self._refresh_faction_territories()

    def _create_scripted_capital_bot_team(self):
        if not self._is_city_defence_scenario():
            return super()._create_scripted_capital_bot_team()
        roster = ASSAULT_ROSTERS[self.city_defence_wave]
        units = [self._build_unit_from_data(self._find_unit_data_by_name(name), "blue", pos)
                 for pos, name in roster.items()]
        return self._build_battle_team_with_placeholders("blue", units)

    def _scripted_capital_bot_info(self):
        info = super()._scripted_capital_bot_info()
        if self._is_city_defence_scenario():
            info.update(city_defence_wave=self.city_defence_wave + 1,
                        city_defence_wins=self.city_defence_wins,
                        city_defence_attack_turn=self.city_defence_spawn_turn + 2)
        return info

    def _advance_scripted_capital_bot_one_turn(self):
        if not self._is_city_defence_scenario():
            return super()._advance_scripted_capital_bot_one_turn()
        if (not self.scripted_capital_bot_enabled or self.city_defence_wins == 2
                or not self._garrison_player_owns(CITY_NAME)
                or self.pending_garrison_city
                or self.current_battle_context.get("kind") == "city_garrison"):
            return self._scripted_capital_bot_info()["scripted_capital_bot"]
        target = self.legions_settlement_source_tile_by_name[CITY_NAME]
        path = self._scripted_bot_path_to(target)
        # The empty 20-cell approach takes ten ordinary plain tiles per day.
        moved = self._move_scripted_bot_along_path(path)
        self._sync_scripted_capital_bot_grid_state()
        info = dict(enabled=True, state=self.scripted_capital_bot_state,
                    position=self.scripted_capital_bot_position,
                    events=["city_assault"], **self._scripted_bot_movement_info(moved))
        self.scripted_capital_bot_last_info = info
        return info

    def _finish_pending_city_defence(self, result):
        result = super()._finish_pending_city_defence(result)
        if not self._is_city_defence_scenario():
            return result
        obs, reward, terminated, truncated, info = result
        info = dict(info)
        if not self._garrison_player_owns(CITY_NAME):
            reward += self.reward_loss
            terminated, truncated = True, False
            info.update(campaign_result="defeat", campaign_defeat_reason="defended_city_captured")
        elif (info.get("battle_context_kind") == "city_garrison"
              and info.get("battle_result") == "victory"):
            self.city_defence_wins += 1
            reward += self.reward_defeat_enemy
            if self.city_defence_wins == len(ASSAULT_ROSTERS):
                reward += self.reward_all_enemies
                terminated, truncated = True, False
                info.update(campaign_result="victory", campaign_victory_reason="city_assaults_defeated",
                            final_objective_reward=self.reward_all_enemies)
            else:
                self.city_defence_wave += 1
                self.city_defence_spawn_turn = int(self.turns)
                self.scripted_capital_bot_position = tuple(self.scripted_capital_bot_home)
                self.scripted_capital_bot_faction = ASSAULT_FACTIONS[self.city_defence_wave]
                self.scripted_capital_bot_team_state = self._create_scripted_capital_bot_team()
                self.scripted_capital_bot_state = "hunting"
                self.scripted_capital_bot_respawn_turns_left = 0
                self.scripted_capital_bot_last_info = {"events": ["next_city_assault_spawned"]}
                self._sync_scripted_capital_bot_grid_state()
            obs = self._build_obs(grid_obs=self._get_grid_obs())
        info.update(self._scripted_capital_bot_info(), **self._garrison_info(), reward=float(reward))
        return obs, reward, terminated, truncated, info
