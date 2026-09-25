"""Only a hero-army battle victory earns the blue dragon's objective bonus."""
import os
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'

import unittest
from unittest.mock import patch
from campaign_env import CampaignEnv


class BlueDragonObjectiveRewardTests(unittest.TestCase):
    def make(self, objective='blue_dragon'):
        env = CampaignEnv(map_name='default', campaign_objective=objective,
                          use_boss_starting_roster=False, scripted_capital_bot_enabled=False,
                          reward_all_enemies=8., reward_green_dragon_objective_multiplier=2.,
                          reward_defeat_enemy=4., detailed_step_info=False)
        env.reset(seed=42)
        self.addCleanup(env.close)
        return env

    def test_hero_battle_victory_earns_96(self):
        env = self.make()
        env.current_enemy_id = 31
        env.battle_origin_pos = tuple(env.grid_env.agent_pos)
        env.current_battle_context = {'kind': 'hero'}
        env.mode = env.MODE_BATTLE
        env._init_battle(31)
        battle = env.battle_env
        battle.winner = 'blue'
        with patch.object(battle, 'step', return_value=(battle._obs(), 2., True, False, {})):
            _, reward, done, _, info = env.step(0)
        self.assertTrue(done)
        self.assertEqual(info['campaign_result'], 'victory')
        self.assertEqual(info['final_objective_reward'], 96.)
        self.assertEqual(info['enemy_defeat_reward'], 4.)
        self.assertTrue(info['objective_reward_eligible'])
        self.assertEqual(info['objective_defeat_source'], 'hero_battle')
        self.assertEqual(env.hero_defeated_enemy_ids, {31})
        self.assertEqual(info['hero_enemies_defeated_count'], 1)
        self.assertGreaterEqual(reward, 100.)

    def test_map_spell_kill_earns_no_defeat_bonus(self):
        env = self.make()
        target = dict(enemy_id=31, position=env.grid_env.enemy_positions[31], reachable=True)
        with patch.object(env, '_resolve_spell_target_enemy', return_value=target), \
             patch.object(env, '_apply_damage_to_enemy_stack', return_value=(700., 1, 1, 0)), \
             patch.object(env, '_enemy_team_has_living_units', return_value=False):
            result = env._cast_from_spell_spec(source='learned_offensive_spell',
                spell_key='lod_d2_s003', spell_description='test',
                spell_spec={'spell_kind': 'damage', 'damage': 700., 'damage_type': 'Fire'})
        self.assertTrue(result['spell_enemy_defeated'])
        self.assertEqual(result['final_objective_reward'], 0.)
        self.assertEqual(result['enemy_defeat_reward'], 0.)
        self.assertAlmostEqual(result['reward'], env.reward_spell_cast)
        info = env._build_spell_cast_info(spell_key='lod_d2_s003', spell_description='test',
            spell_level=1, spell_kind='damage', spell_spec={}, cast_result=result,
            can_cast=True, mana_costs={})
        self.assertFalse(info['objective_reward_eligible'])
        self.assertEqual(info['final_objective_reward'], 0.)

    def test_summon_victory_ends_episode_without_objective_bonus(self):
        env = self.make()
        env.current_enemy_id = 31
        env.battle_origin_pos = tuple(env.grid_env.agent_pos)
        _, reward, done, _, info = env._finalize_summon_spell_battle(
            obs=None, reward=0., info={}, winner='blue')
        self.assertTrue(done)
        self.assertEqual(info['campaign_result'], 'victory')
        self.assertEqual(info['final_objective_reward'], 0.)
        self.assertEqual(info['enemy_defeat_reward'], 0.)
        self.assertFalse(info['objective_reward_eligible'])
        self.assertEqual(reward, 0.)

    def test_other_objectives_keep_magic_bonus(self):
        env = self.make('dragon')
        for source in ('hero_battle', 'map_spell', 'summon_battle'):
            info = {}
            self.assertEqual(env._apply_green_dragon_objective_reward_if_needed(
                31, 0., info, defeat_source=source), 96.)


if __name__ == '__main__':
    unittest.main()
