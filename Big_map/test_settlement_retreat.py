"""City reserves and heroes on settlement entrances cannot select retreat."""
import os
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'

import unittest
from copy import deepcopy

from battle_env import RUN_AWAY_ACTION_INDEX
from campaign_env import CampaignEnv


class SettlementRetreatTests(unittest.TestCase):
    def make(self, map_name='default'):
        env = CampaignEnv(map_name=map_name, scripted_capital_bot_enabled=False,
                          use_boss_starting_roster=False, log_enabled=False)
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def start(self, env, origin, *, enemy=1, kind='hero', grid_pos=None):
        env.grid_env.agent_pos = tuple(grid_pos or origin)
        env.battle_origin_pos = tuple(origin) if origin is not None else None
        env.current_enemy_id = enemy
        env.current_battle_context = {'kind': kind}
        env.mode = env.MODE_BATTLE
        env._init_battle(enemy)

    def assert_blocked_for_whole_party(self, env):
        battle = env.battle_env
        positions = [u['position'] for u in battle.combined
                     if u['team'] == 'blue' and u['health'] > 0]
        self.assertTrue(positions)
        for position in positions:
            with self.subTest(position=position):
                battle.current_blue_attacker_pos = position
                battle.blue_attacks_left = 1
                self.assertFalse(battle.compute_action_mask()[RUN_AWAY_ACTION_INDEX])
                self.assertFalse(env.compute_action_mask()[RUN_AWAY_ACTION_INDEX])
                before = deepcopy(battle.combined)
                obs, _, terminated, truncated, info = env.step(RUN_AWAY_ACTION_INDEX)
                self.assertTrue(info['battle_retreat_blocked'])
                self.assertFalse(terminated or truncated)
                self.assertEqual(battle.combined, before)
                self.assertEqual(battle.escaped_units, [])
                self.assertTrue(env.observation_space.contains(obs))

    def test_capital_and_every_city_entrance_block_whole_hero_party(self):
        env = self.make()
        self.assertGreater(len(env.castle_heal_tiles), 1)
        action_count = env.action_space.n
        for entrance in env.castle_heal_tiles:
            with self.subTest(entrance=entrance):
                self.start(env, entrance)
                self.assert_blocked_for_whole_party(env)
        self.assertEqual(env.action_space.n, action_count)

    def test_adjacent_owned_land_still_allows_retreat(self):
        env = self.make()
        for entrance in env.castle_heal_tiles:
            adjacent = next((entrance[0] + dx, entrance[1] + dy)
                            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                            if 0 <= entrance[0] + dx < env.grid_size
                            and 0 <= entrance[1] + dy < env.grid_size
                            and (entrance[0] + dx, entrance[1] + dy) not in env.grid_env.obstacle_positions
                            and (entrance[0] + dx, entrance[1] + dy) not in env.castle_heal_tiles)
            env._paint_territory_owner(env.TERRITORY_AGENT, (adjacent,))
            env._refresh_territory_ownership_caches()
            self.start(env, adjacent)
            self.assertTrue(env.battle_env.retreat_enabled)
            self.assertTrue(env.compute_action_mask()[RUN_AWAY_ACTION_INDEX])

    def test_hero_origin_is_used_instead_of_attacked_city_destination(self):
        env = self.make()
        field = tuple(env.grid_env.enemy_positions[1])
        city = env.castle_heal_tiles[1]
        self.assertNotIn(field, env.castle_heal_tiles)
        self.start(env, field, grid_pos=city)
        self.assertTrue(env.battle_env.retreat_enabled)
        self.start(env, city, grid_pos=field)
        self.assertFalse(env.battle_env.retreat_enabled)

    def test_missing_origin_uses_current_hero_position(self):
        env = self.make()
        self.start(env, None, grid_pos=env.CASTLE_POS)
        self.assertFalse(env.battle_env.retreat_enabled)

    def test_actual_recruited_city_guard_cannot_retreat(self):
        env = self.make()
        city = env.garrison_city_names[0]
        env.legions_active_settlement_territory_capture_turn_by_name[city] = 0
        env.gold = 10000.
        action = env.GRID_GARRISON_HIRE_ACTION_START
        self.assertTrue(env.compute_action_mask()[action])
        self.assertTrue(env.step(action)[-1]['garrison_hired'])
        env.scripted_capital_bot_team_state = env._build_battle_team_with_placeholders('blue', [
            env._build_unit_from_data(env._find_unit_data_by_name('Крестьянин'), 'blue', 7)])
        hero = deepcopy(env.blue_team_state)
        env._start_city_defence(city)
        self.assertFalse(env.battle_env.retreat_enabled)
        self.assert_blocked_for_whole_party(env)
        self.assertEqual(env.blue_team_state, hero)

    def test_remote_summon_not_restricted_by_hero_location(self):
        env = self.make()
        self.start(env, env.CASTLE_POS, kind='summon_spell')
        self.assertTrue(env.battle_env.retreat_enabled)
        siege = self.make('siege_train')
        self.start(siege, siege.CASTLE_POS, enemy=100, kind='summon_spell')
        self.assertFalse(siege.battle_env.retreat_enabled)

    def test_actual_queued_encounter_at_capital_blocks_retreat(self):
        env = self.make()
        env._queue_scheduled_enemy_encounter(1)
        info = env.step(8)[-1]
        self.assertTrue(info['battle_triggered'])
        self.assertEqual(env.battle_origin_pos, env.CASTLE_POS)
        self.assert_blocked_for_whole_party(env)


if __name__ == '__main__':
    unittest.main()
