"""Regression checks for campaign-wide diminishing action bonuses."""
import os
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'

import unittest
from unittest.mock import patch
from campaign_env import CampaignEnv
from battle_env import DEFEND_ACTION_INDEX


class FirstUseRewardsTests(unittest.TestCase):
    def make(self, map_name='super_last_stand', **kwargs):
        kwargs.setdefault('reward_first_uses', (1.0, 0.5, 0.25))
        env = CampaignEnv(map_name=map_name, use_boss_starting_roster=False,
                          scripted_capital_bot_enabled=False,
                          detailed_step_info=False, **kwargs)
        env.reset(seed=42)
        self.addCleanup(env.close)
        return env

    def test_independent_categories_limit_reset_and_reward_accounting(self):
        env = self.make()
        obs = env._build_obs(grid_obs=env._get_grid_obs())
        events = {
            'spell': {'spell_cast_executed': True},
            'scroll': {'scroll_cast_action': True, 'scroll_item_consumed': True,
                       'spell_cast_executed': True},
            'defend': {'battle_defend_applied': True},
            'merchant_purchase': {'merchant_buy_purchased': True},
        }
        # Interleaving categories must not spend another category's budget.
        for count, expected in enumerate((1., .5, .25, 0.), 1):
            for category, event in events.items():
                result = (obs, 2., False, False, dict(event, reward=2.))
                with patch.object(env, '_step_grid', return_value=result):
                    _, reward, _, _, info = env.step(8)
                self.assertEqual(info['first_use_category'], category)
                self.assertEqual(info['first_use_count'], count)
                self.assertEqual(info['first_use_reward'], expected)
                self.assertAlmostEqual(reward, 2. + expected - info['leadership_step_penalty'])
                self.assertEqual(info['reward'], reward)
        env.reset(seed=43)
        self.assertEqual(env.first_use_counts, {})
        with patch.object(env, '_step_grid', return_value=(obs, 0., True, False, events['spell'])):
            self.assertEqual(env.step(8)[-1]['first_use_reward'], 1.)

    def test_failed_actions_do_not_consume_bonus(self):
        env = self.make()
        for event in ({'spell_cast_executed': False}, {'merchant_buy_purchased': False},
                      {'battle_defend_applied': False},
                      {'scroll_cast_action': True, 'spell_cast_executed': True,
                       'scroll_item_consumed': False}):
            with patch.object(env, '_step_grid', return_value=(None, 0., False, False, event)):
                self.assertNotIn('first_use_reward', env.step(8)[-1])
        self.assertEqual(env.first_use_counts, {})

    def test_real_spells_receive_bonus_without_killing_enemy(self):
        env = self.make()
        for spell in env.active_spells.values():
            if isinstance(spell, dict):
                spell['learned'] = 1
        for attr in env.MANA_ATTR_BY_KIND.values():
            setattr(env, attr, 10000.)
        for expected in (1., .5, .25, 0.):
            mask = env.compute_action_mask()
            actions = [env.grid_legion_damage_spell_action_start + i
                       for i, spec in enumerate(env._current_map_offensive_spell_action_specs())
                       if spec.get('spell_kind', spec.get('kind')) != 'summon_battle'
                       and mask[env.grid_legion_damage_spell_action_start + i]]
            self.assertTrue(actions)
            info = env.step(actions[0])[-1]
            self.assertTrue(info['spell_cast_executed'])
            self.assertEqual(info['spell_cast_reward'], .005)
            self.assertEqual(info['first_use_reward'], expected)

    def test_real_scrolls_minimal_info_and_no_double_reward(self):
        env = self.make('scroll_train')
        for expected in (1., .5, .25, 0.):
            mask = env.compute_action_mask()
            choices = [env.grid_scroll_cast_action_start + i
                       for i, spec in enumerate(env.scroll_cast_slot_entries())
                       if spec['spell_kind'] in ('ward', 'buff')
                       and mask[env.grid_scroll_cast_action_start + i]]
            self.assertTrue(choices)
            info = env.step(choices[0])[-1]
            self.assertTrue(info['scroll_item_consumed'])
            self.assertEqual(info['first_use_category'], 'scroll')
            self.assertEqual(info['first_use_reward'], expected)
        self.assertNotIn('spell', env.first_use_counts)

    def test_real_merchant_purchase_and_failed_purchase(self):
        env = self.make('trade_train')
        # Set up interaction at the actual shop; the purchase handler is unchanged.
        env.grid_env.agent_pos = tuple(env.merchant_interaction_tiles[0])
        env.gold = 100000.
        for expected in (1., .5, .25, 0.):
            mask = env.compute_action_mask()
            actions = [env.GRID_MERCHANT_POTION_BUY_ACTION_START + i
                       for i in range(len(env.scenario_merchant_item_names))
                       if mask[env.GRID_MERCHANT_POTION_BUY_ACTION_START + i]]
            self.assertTrue(actions)
            info = env.step(actions[0])[-1]
            self.assertTrue(info['merchant_buy_purchased'])
            self.assertEqual(info['first_use_reward'], expected)
        env.gold = 0.
        self.assertNotIn('first_use_reward', env.step(actions[-1])[-1])
        self.assertEqual(env.first_use_counts['merchant_purchase'], 4)

    def test_actual_defend_counts_across_battles(self):
        env = self.make()
        for expected in (1., .5, .25, 0.):
            env.current_enemy_id = 1
            env.battle_origin_pos = tuple(env.grid_env.agent_pos)
            env.current_battle_context = {'kind': 'hero'}
            env.mode = env.MODE_BATTLE
            env._init_battle(1)
            self.assertTrue(env.battle_env.compute_action_mask()[DEFEND_ACTION_INDEX])
            info = env.step(DEFEND_ACTION_INDEX)[-1]
            self.assertTrue(info['battle_defend_applied'])
            self.assertEqual(info['first_use_category'], 'defend')
            self.assertEqual(info['first_use_reward'], expected)

    def test_invalid_schedule_rejected(self):
        for schedule in ((1.,), (1., 2., .5), (1., float('nan'), 0.), (1., .5, -1.)):
            with self.assertRaises(ValueError):
                self.make(reward_first_uses=schedule)


if __name__ == '__main__':
    unittest.main()
