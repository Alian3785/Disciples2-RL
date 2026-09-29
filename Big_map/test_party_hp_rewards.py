"""Campaign HP milestones: strict thresholds, recruitment and promotion."""
import os
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from campaign_env import CampaignEnv


class PartyHpRewardsTests(unittest.TestCase):
    def make(self, **kwargs):
        env = CampaignEnv(map_name='trade_train', use_boss_starting_roster=False,
                          scripted_capital_bot_enabled=False, log_enabled=False,
                          detailed_step_info=False, **kwargs)
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def tick(self, env, *, terminal=False):
        result = (None, 2., terminal, False, {'reward': 2., 'grid_reward_scaled': 2.})
        with patch.object(env, '_step_grid', return_value=result):
            return env.step(8)

    def test_strict_thresholds_wounded_units_and_accounting(self):
        env = self.make(reward_first_unit_hp_200=1.25, reward_first_unit_hp_250=2.5)
        hero = env._resolve_travel_hero()
        for hp, thresholds, bonus in ((200, [], 0.), (201, [200], 1.25),
                                      (250, [], 0.), (251, [250], 2.5),
                                      (500, [], 0.)):
            hero.update(max_health=hp, maxhp=hp, health=1., hp=1.)
            _, reward, _, _, info = self.tick(env)
            self.assertEqual(info.get('party_hp_milestones', []), thresholds)
            self.assertEqual(info.get('party_hp_milestone_reward', 0.), bonus)
            self.assertAlmostEqual(reward, 2. + bonus - info['leadership_step_penalty'])
            self.assertEqual(info['reward'], reward)
            self.assertEqual(info['grid_reward_scaled'], reward)

    def test_both_thresholds_terminal_step_and_no_repeat_after_removal(self):
        env = self.make()
        unit = env._build_unit_from_data(env._find_unit_data_by_name('Молох'), 'blue', 7)
        env.blue_team_state.append(unit)
        result = self.tick(env, terminal=True)
        self.assertTrue(result[2])
        self.assertEqual(result[-1]['party_hp_milestones'], [200, 250])
        self.assertEqual(result[-1]['party_hp_milestone_reward'], 6.)
        env.blue_team_state.remove(unit)
        self.assertNotIn('party_hp_milestone_reward', self.tick(env)[-1])
        env.blue_team_state.append(unit)
        self.assertNotIn('party_hp_milestone_reward', self.tick(env)[-1])
        env.reset(seed=43)
        self.assertEqual(env.party_hp_thresholds_reached, set())
        env.blue_team_state.append(unit)
        self.assertEqual(self.tick(env)[-1]['party_hp_milestone_reward'], 6.)

    def test_starting_thresholds_are_already_consumed(self):
        env = self.make()
        party = deepcopy(env.blue_team_state)
        hero = next(u for u in party if u['name'] == 'Герцог')
        hero.update(max_health=350., maxhp=350., health=350., hp=350.)
        with patch.object(env, '_create_starting_blue_team', return_value=party):
            env.reset(seed=42)
        self.assertEqual(env.party_hp_thresholds_reached, {200, 250})
        self.assertNotIn('party_hp_milestone_reward', self.tick(env)[-1])

    def test_actual_mercenary_hire_and_failed_hire(self):
        env = self.make()
        env.grid_env.agent_pos = env.mercenary_interaction_tiles[0]
        index = next(i for i, option in enumerate(env.active_mercenary_hire_options)
                     if option['unit_name'] == 'Защитник Веры')
        action = env.GRID_MERCENARY_HIRE_ACTION_START + index
        env.gold = 0.
        info = env.step(action)[-1]
        self.assertFalse(info['hired'])
        self.assertEqual(env.party_hp_thresholds_reached, set())
        env.gold = 100000.
        info = env.step(action)[-1]
        self.assertTrue(info['hired'])
        self.assertEqual(info['party_hp_milestones'], [200])
        self.assertEqual(info['party_hp_milestone_reward'], 3.)
        self.assertNotIn('party_hp_milestone_reward', self.tick(env)[-1])

    def test_actual_promotions_through_battle_step(self):
        for source, target, threshold in (('Берсерк', 'Темный паладин', 200),
                                           ('Темный паладин', 'Адский рыцарь', 250)):
            with self.subTest(source=source):
                env = self.make()
                unit = env._build_unit_from_data(env._find_unit_data_by_name(source), 'blue', 7)
                env.blue_team_state = [u for u in env.blue_team_state if u['position'] != 7] + [unit]
                env.party_hp_thresholds_reached = env._party_hp_thresholds()
                for building in env.active_buildings.values():
                    if isinstance(building, dict) and building.get('unit') == target:
                        building['Build'] = 1
                env.battle_env = SimpleNamespace(combined=[deepcopy(unit)], last_levelups=[source])
                env.mode = env.MODE_BATTLE

                def finish_battle(action):
                    self.assertEqual(env._log_turns_into_levelups(), 1)
                    env.mode = env.MODE_GRID
                    return None, 0., False, False, {}

                with patch.object(env, '_step_battle', side_effect=finish_battle):
                    info = env.step(0)[-1]
                self.assertEqual(info['party_hp_milestones'], [threshold])
                self.assertEqual(info['party_hp_milestone_reward'], 3.)
                self.assertNotIn('party_hp_milestone_reward', self.tick(env)[-1])

    def test_battle_only_units_do_not_count(self):
        env = self.make()
        env.mode = env.MODE_BATTLE
        env.battle_env = SimpleNamespace(combined=[{'team': 'blue', 'max_health': 1000.}])
        with patch.object(env, '_step_battle', return_value=(None, 0., False, False, {})):
            self.assertNotIn('party_hp_milestone_reward', env.step(0)[-1])

    def test_zero_and_invalid_rewards(self):
        env = self.make(reward_first_unit_hp_200=0., reward_first_unit_hp_250=0.)
        env._resolve_travel_hero().update(max_health=350.)
        self.assertEqual(self.tick(env)[-1]['party_hp_milestone_reward'], 0.)
        self.assertEqual(env.party_hp_thresholds_reached, {200, 250})
        for name in ('reward_first_unit_hp_200', 'reward_first_unit_hp_250'):
            for value in (-1., float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.make(**{name: value})


if __name__ == '__main__':
    unittest.main()
