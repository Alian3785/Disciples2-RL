"""City reserves, remote recruitment and agent-controlled city defence.

Action slots are stable: five faction recruits per non-capital settlement.
City levels 1..5 provide 1..5 capacity; large units occupy a whole column.
"""
from copy import deepcopy

import numpy as np
from gymnasium import spaces

from campaign_env_data import BattleEnv, SETTLEMENT_ARMOR_BONUS_BY_LEVEL


class CampaignGarrisonMixin:
    GARRISON_FRONT_POSITIONS = (7, 8, 9)
    GARRISON_BACK_POSITIONS = (10, 11, 12)
    GARRISON_OPTIONS_PER_CITY = 5
    GARRISON_OBS_PER_CITY = 5 + 6 * 3

    def _refresh_dynamic_action_layout(self):
        super()._refresh_dynamic_action_layout()
        self.garrison_city_names = tuple(
            name for name, tile in self.legions_settlement_source_tile_by_name.items()
            if tuple(tile) != tuple(self.CASTLE_POS)
        )
        self.GRID_GARRISON_HIRE_ACTION_START = (
            self.GRID_DISMISS_UNIT_ACTION_START + self.GRID_DISMISS_UNIT_ACTION_COUNT
            if hasattr(self, 'GRID_DISMISS_UNIT_ACTION_START') else
            self.GRID_SWAP_UNIT_ACTION_START + self.GRID_SWAP_UNIT_ACTION_COUNT
        )
        self.GRID_GARRISON_HIRE_ACTION_COUNT = len(self.garrison_city_names) * 5

    def _reset_legions_settlement_territory_state(self):
        self.city_garrisons = {}
        self.bot_city_capture_turns = {}
        self.bot_city_initial_tiles = {}
        self.pending_garrison_city = None
        self._city_capture_event = None
        super()._reset_legions_settlement_territory_state()

    def _garrison_player_owns(self, city):
        return city in self.legions_active_settlement_territory_capture_turn_by_name

    def _garrison_capacity(self, city):
        return min(5, max(1, int(self.legions_settlement_level_by_name.get(city, 1))))

    def _garrison_regeneration_fraction(self, city, unit):
        """DII fort regeneration: unit + warrior lord + city, without terrain.

        Current faction recruits have 5% innate regeneration. Scenario units
        may override it with regeneration_percent (percentage points).
        See D2ModdingToolset unitutils.cpp::getUnitRegen/getFortRegen.
        """
        innate = float(unit.get('regeneration_percent', self.DEFAULT_REST_HEAL_PERCENT * 100)) / 100
        city_bonus = float(self.SETTLEMENT_REST_HEAL_BONUS_BY_LEVEL[self._garrison_capacity(city)]) / 100
        lord_bonus = self._resolve_typeoflord_rest_heal_bonus_percent()
        return min(1.0, max(0.0, innate + city_bonus + lord_bonus))

    def _heal_city_garrisons_for_turn(self):
        """Regenerate living reserves once per campaign day, before bot movement.

        The travelling hero's banners, equipment and terrain bonus do not
        belong to reserves. Do not mutate a roster awaiting/in an active battle.
        """
        active_city = (self.current_battle_context.get('city')
                       if self.current_battle_context.get('kind') == 'city_garrison' else None)
        healed_units, healed_hp = 0, 0.0
        for city, units in self.city_garrisons.items():
            if (not self._garrison_player_owns(city)
                    or city == active_city or city == self.pending_garrison_city):
                continue
            for unit in units:
                if self._is_empty_blue_unit(unit):
                    continue
                hp, max_hp = self._unit_current_hp(unit), self._unit_max_hp(unit)
                if not 0 < hp < max_hp:
                    continue
                restored = min(max_hp - hp, max_hp * self._garrison_regeneration_fraction(city, unit))
                if restored <= 0:
                    continue
                unit['health'] = unit['hp'] = hp + restored
                healed_units += 1
                healed_hp += restored
        if healed_units:
            self._log(f"Городская охрана восстановила {healed_hp:g} HP у {healed_units} юнитов за ход.")
        return healed_units, healed_hp

    def _garrison_occupied_positions(self, city):
        occupied = set()
        for unit in self.city_garrisons.get(city, ()):
            if self._is_empty_blue_unit(unit):
                continue
            pos = int(unit['position'])
            occupied.add(pos)
            if self._unit_data_is_big(unit):
                occupied.add(self._blue_partner_position(pos))
        return occupied

    def _garrison_option_and_city(self, action):
        offset = int(action) - self.GRID_GARRISON_HIRE_ACTION_START
        if not 0 <= offset < self.GRID_GARRISON_HIRE_ACTION_COUNT:
            return None, None
        city_index, option_index = divmod(offset, 5)
        option = self.active_hire_options[option_index] if option_index < len(self.active_hire_options) else None
        return self.garrison_city_names[city_index], option

    def _garrison_hire_positions(self, city, option):
        if not self._garrison_player_owns(city) or not option:
            return ()
        if float(self.gold) < float(option.get('gold', 0)):
            return ()
        if not self._hire_option_required_building_is_built(option):
            return ()
        data = self._find_unit_data_by_name(option['name'])
        if data is None:
            return ()
        big = self._unit_data_is_big(data)
        occupied = self._garrison_occupied_positions(city)
        if len(occupied) + (2 if big else 1) > self._garrison_capacity(city):
            return ()
        if big:
            return tuple(pos for pos in self.GARRISON_FRONT_POSITIONS
                         if pos not in occupied and self._blue_partner_position(pos) not in occupied)
        positions = (self.GARRISON_FRONT_POSITIONS if option.get('role') == 'warrior'
                     else self.GARRISON_BACK_POSITIONS)
        return tuple(pos for pos in positions if pos not in occupied)

    def compute_action_mask(self):
        mask = super().compute_action_mask()
        start = self.GRID_GARRISON_HIRE_ACTION_START
        mask[start:] = False
        if self.mode == self.MODE_GRID:
            for action in range(start, start + self.GRID_GARRISON_HIRE_ACTION_COUNT):
                city, option = self._garrison_option_and_city(action)
                mask[action] = bool(self._garrison_hire_positions(city, option))
        return mask

    def _step_hire_garrison(self, action):
        city, option = self._garrison_option_and_city(action)
        positions = self._garrison_hire_positions(city, option)
        info = {'garrison_hire_action': True, 'garrison_city': city,
                'garrison_hired': False, 'gold_spent': 0.0}
        if positions:
            position = int(self.np_random.choice(positions))
            unit = self._build_unit_from_data(self._find_unit_data_by_name(option['name']), 'blue', position)
            unit['stand'] = 'ahead' if position in self.GARRISON_FRONT_POSITIONS else 'behind'
            self.city_garrisons.setdefault(city, []).append(unit)
            cost = float(option.get('gold', 0))
            self.gold -= cost
            info.update(garrison_hired=True, hired_unit_name=option['name'],
                        hired_position=position, gold_spent=cost)
        info.update(self._garrison_info())
        return self._finalize_grid_step_result(
            grid_obs=self._get_grid_obs(), reward=0.0, terminated=False,
            truncated=False, info=info)

    def _garrison_info(self):
        return {'city_garrisons': deepcopy(self.city_garrisons),
                'city_owners': {city: ('player' if self._garrison_player_owns(city) else
                                     'scripted_bot' if city in self.bot_city_capture_turns else 'neutral')
                                for city in self.garrison_city_names}}

    def _refresh_garrison_observation_space(self):
        # Encoders keep their own base slices. Append a common reserve block.
        if self.observation_space is getattr(self, '_garrison_extended_space', None):
            return
        base_size = self.observation_space.shape[0]
        self.garrison_obs_slice = (base_size, base_size + len(self.garrison_city_names) * self.GARRISON_OBS_PER_CITY)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.garrison_obs_slice[1],), dtype=np.float32)
        self._garrison_extended_space = self.observation_space

    def _build_obs(self, grid_obs=None, battle_obs=None):
        base = super()._build_obs(grid_obs=grid_obs, battle_obs=battle_obs)
        self._refresh_garrison_observation_space()
        block = np.zeros((len(self.garrison_city_names), self.GARRISON_OBS_PER_CITY), dtype=np.float32)
        for index, city in enumerate(self.garrison_city_names):
            owned = self._garrison_player_owns(city)
            # Remote information describes our reserves, not unexplored enemies.
            block[index, :5] = (owned, city in self.bot_city_capture_turns,
                                self._garrison_capacity(city) / 5 if owned else 0,
                                len(self._garrison_occupied_positions(city)) / 5 if owned else 0,
                                self.current_battle_context.get('city') == city)
            if not owned:
                continue
            for unit in self.city_garrisons.get(city, ()):
                pos = int(unit['position']) - 7
                slot = 5 + pos * 3
                name_index = next((i + 1 for i, o in enumerate(self.active_hire_options)
                                   if o['name'] == unit.get('name')), 0)
                block[index, slot:slot + 3] = (name_index / 5, self._normalize_unit_health(unit),
                                               self._unit_data_is_big(unit))
        return np.concatenate((base, block.ravel()))

    def _scripted_bot_alive_enemy_tiles_with_signature(self):
        tiles, signature = super()._scripted_bot_alive_enemy_tiles_with_signature()
        extra = []
        for index, city in enumerate(self.garrison_city_names):
            tile = tuple(self.legions_settlement_source_tile_by_name[city])
            owned = self._garrison_player_owns(city)
            # Synthetic IDs are for pathfinding only, never enemy roster lookup.
            city_id = -1000 - index
            extra.append((city_id, tile, owned, owned))
            if owned:
                tiles[tile] = city_id
        return tiles, signature + tuple(extra)

    def _garrison_city_at(self, tile):
        city = self.legions_settlement_source_name_by_tile.get(tuple(tile))
        return city if city in self.garrison_city_names and self._garrison_player_owns(city) else None

    def _scripted_bot_path_to_engagement_tile(self, path, target_tile):
        if self._garrison_city_at(target_tile):
            return path  # A city attack requires entering its entrance tile.
        return super()._scripted_bot_path_to_engagement_tile(path, target_tile)

    def _move_scripted_bot_along_path(self, path):
        for index, tile in enumerate(path[1:], 1):
            if self._garrison_city_at(tile):
                path = path[:index + 1]
                break
        movement = super()._move_scripted_bot_along_path(path)
        city = self._garrison_city_at(self.scripted_capital_bot_position)
        if city:
            self._queue_city_defence(city)
        return movement

    def _advance_scripted_capital_bot_one_turn(self):
        if self.pending_garrison_city or self.current_battle_context.get('kind') == 'city_garrison':
            return {'enabled': True, 'state': 'attacking_city', 'events': []}
        return super()._advance_scripted_capital_bot_one_turn()

    def _run_scripted_capital_bot_battle(self, enemy_id):
        if self.pending_garrison_city or enemy_id <= -1000:
            return {'city': self.pending_garrison_city, 'winner': 'pending', 'steps': 0}
        return super()._run_scripted_capital_bot_battle(enemy_id)

    def _queue_city_defence(self, city):
        if any(self._unit_current_hp(u) > 0 for u in self.city_garrisons.get(city, ())):
            self.pending_garrison_city = city
        else:
            self._transfer_city_to_bot(city)

    def _transfer_city_to_bot(self, city):
        self.bot_city_initial_tiles[city] = tuple(self.legions_settlement_territory_tiles_by_name.get(city, ()))
        self.bot_city_capture_turns[city] = int(self.turns)
        self.legions_active_settlement_territory_capture_turn_by_name.pop(city, None)
        self.legions_settlement_growth_history_by_name.pop(city, None)
        self.captured_objective_cities.discard(city)
        self.city_garrisons.pop(city, None)
        self.scripted_capital_bot_state = 'returning'
        self._city_capture_event = {'city': city, 'owner': 'scripted_bot',
                                    'faction': self.scripted_capital_bot_faction}
        self._refresh_faction_territories()

    def _activate_legions_settlement_territory_if_cleared(self, enemy_id):
        activated = super()._activate_legions_settlement_territory_if_cleared(enemy_id)
        for city in activated:
            self.bot_city_capture_turns.pop(city, None)
            self.bot_city_initial_tiles.pop(city, None)
        if activated:
            self._refresh_faction_territories()
        return activated

    def _refresh_faction_territories(self):
        super()._refresh_faction_territories()
        bot_tiles = set()
        for city, captured_turn in self.bot_city_capture_turns.items():
            order = self.legions_settlement_territory_orders_by_name.get(city, ())
            count = self._territory_claim_count(len(order), self._settlement_expansion_per_turn(city),
                                               max(0, self.turns - captured_turn))
            bot_tiles.update(self.bot_city_initial_tiles.get(city, ()))
            bot_tiles.update(self._claim_territory_tiles(order, count))
        if not bot_tiles:
            return
        self.legions_territory_tiles = tuple(t for t in self.legions_territory_tiles if t not in bot_tiles)
        self.legions_territory_tile_set = set(self.legions_territory_tiles)
        self.empire_territory_tile_set.update(bot_tiles)
        self.empire_territory_tiles = tuple(sorted(self.empire_territory_tile_set))
        self.legions_captured_gold_mine_tiles = tuple(t for t in self.gold_mine_tiles if tuple(t) in self.legions_territory_tile_set)
        self.legions_captured_gold_mine_count = len(self.legions_captured_gold_mine_tiles)
        self._refresh_legions_captured_mana_source_state()
        if hasattr(self, 'grid_legions_territory_positions'):
            self._refresh_legions_territory_grid_obs_cache()

    def _finish_pending_city_defence(self, result):
        obs, reward, terminated, truncated, info = result
        if self._city_capture_event:
            info = dict(info, city_capture=dict(self._city_capture_event))
            self._city_capture_event = None
        city = self.pending_garrison_city
        if city and self.mode == self.MODE_GRID and not (terminated or truncated):
            self.pending_garrison_city = None
            if self._garrison_player_owns(city):
                self._start_city_defence(city)
                obs = self._build_obs(grid_obs=self._get_grid_obs(), battle_obs=self.battle_env._obs())
                info = dict(info, mode='battle', battle_triggered=True,
                            battle_context_kind='city_garrison', garrison_city=city)
        return obs, reward, terminated, truncated, info

    def _start_city_defence(self, city):
        blue = self._build_battle_team_with_placeholders('blue', self.city_garrisons[city])
        armor = int(SETTLEMENT_ARMOR_BONUS_BY_LEVEL[self._garrison_capacity(city)])
        for unit in blue:
            unit['garrison_base_armor'] = unit.get('armor', 0)
            if not self._is_empty_blue_unit(unit):
                unit['armor'] = min(90, int(unit.get('armor', 0)) + armor)
        red = deepcopy(self.scripted_capital_bot_team_state)
        for unit in red:
            unit['position'] = int(unit['position']) - 6
            unit['team'] = 'red'
        self.battle_env = BattleEnv(reward_win=self.battle_reward_win, reward_loss=self.battle_reward_loss,
                                   reward_step=self.battle_reward_step, log_enabled=self.log_enabled)
        self.battle_env._init_with_custom_teams(self._build_battle_team_with_placeholders('red', red), blue)
        self.current_battle_context = {'kind': 'city_garrison', 'city': city}
        self.current_enemy_id = None
        self.mode = self.MODE_BATTLE

    def _saved_city_battle_team(self, team):
        saved = []
        for raw in self.battle_env.combined:
            if raw.get('team') != team or raw.get('Summoned'):
                continue
            unit = deepcopy(raw)
            self.battle_env._restore_transformed_unit(unit)
            self.battle_env._cleanse_negative_effects(unit)
            if team == 'red':
                unit['position'] = int(unit['position']) + 6
            unit['hp'] = unit['health'] = max(0, float(unit.get('health', 0)))
            unit['damage'] = unit.get('original_damage', unit.get('damage', 0))
            unit['armor'] = unit.pop('garrison_base_armor', unit.get('base_armor', unit.get('armor', 0)))
            unit['base_armor'] = unit['armor']
            saved.append(self._normalize_scripted_bot_saved_unit(unit))
        return self._build_battle_team_with_placeholders('blue', saved)

    def _step_battle(self, action):
        if self.current_battle_context.get('kind') != 'city_garrison':
            return super()._step_battle(action)
        city = self.current_battle_context['city']
        if self.battle_env.winner is None:
            obs, raw_reward, terminated, truncated, info = self.battle_env.step(
                min(int(action), self.battle_env.action_space.n - 1))
        else:
            obs, raw_reward, terminated, truncated, info = self.battle_env._obs(), 0.0, True, False, {}
        reward = float(raw_reward) * self.battle_reward_scale
        info.update(mode='battle', battle_step=True, battle_context_kind='city_garrison', garrison_city=city)
        if not (terminated or truncated):
            return self._build_obs(grid_obs=self._get_grid_obs(), battle_obs=obs), reward, False, False, info
        winner = self.battle_env.winner
        self.city_garrisons[city] = [u for u in self._saved_city_battle_team('blue')
                                     if not self._is_empty_blue_unit(u) and self._unit_current_hp(u) > 0]
        self.scripted_capital_bot_team_state = self._saved_city_battle_team('red')
        if winner == 'red' or (winner is None and not self.city_garrisons[city]):
            self._transfer_city_to_bot(city)
            outcome = 'defeat'
        elif winner == 'blue':
            self.scripted_capital_bot_state = 'defeated'
            self.scripted_capital_bot_respawn_turns_left = 1
            outcome = 'victory'
        else:
            self.scripted_capital_bot_state = 'returning'
            outcome = 'timeout'
        self.mode = self.MODE_GRID
        self.battle_env = None
        self.current_enemy_id = None
        self._clear_current_battle_context()
        self._sync_scripted_capital_bot_grid_state()
        info.update(mode='grid', battle_result=outcome, battle_winner=winner,
                    garrison_city_lost=outcome == 'defeat', **self._garrison_info())
        # Losing a reserve is not losing the travelling hero or the campaign.
        return self._build_obs(grid_obs=self._get_grid_obs()), reward, False, False, info
