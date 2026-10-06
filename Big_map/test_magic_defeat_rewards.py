"""Only hero-army victories earn enemy defeat and ruin bonuses."""
import os
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'
import unittest
from unittest.mock import patch
from campaign_env import CampaignEnv


class MagicDefeatRewardTests(unittest.TestCase):
    def make(self):
        env = CampaignEnv(map_name='super_last_stand', use_boss_starting_roster=False,
                          scripted_capital_bot_enabled=False, reward_defeat_enemy=4.,
                          reward_ruin_clear_bonus=1.)
        env.reset(seed=42)
        self.addCleanup(env.close)
        return env

    def test_hero_reward_unchanged(self):
        env = self.make()
        self.assertEqual(env._compute_enemy_defeat_reward(1), 4.)
        self.assertEqual(env._compute_enemy_defeat_reward(70), 5.)
        self.assertEqual(env._compute_enemy_defeat_reward(1, magic=True), 0.)
        self.assertEqual(env._compute_enemy_defeat_reward(70, magic=True), 0.)

    def test_spell_kill_has_no_defeat_bonus_including_ruin(self):
        for enemy, expected in ((1, 0.), (70, 0.)):
            with self.subTest(enemy=enemy):
                env = self.make()
                target = dict(enemy_id=enemy, position=env.grid_env.enemy_positions[enemy], reachable=True)
                # Force a lethal damage result, exercising the actual reward handler.
                with patch.object(env, '_resolve_spell_target_enemy', return_value=target), \
                     patch.object(env, '_apply_damage_to_enemy_stack', return_value=(100., 1, 1, 0)), \
                     patch.object(env, '_enemy_team_has_living_units', return_value=False):
                    result = env._cast_from_spell_spec(source='learned_offensive_spell',
                        spell_key='lod_d2_s003', spell_description='test',
                        spell_spec={'spell_kind':'damage','damage':100.,'damage_type':'Fire'})
                self.assertTrue(result['spell_enemy_defeated'])
                self.assertEqual(result['enemy_defeat_reward'], expected)
                self.assertEqual(result['enemy_defeat_reward_base'], 0.)
                self.assertAlmostEqual(result['reward'], expected + env.reward_spell_cast)
                info = env._build_spell_cast_info(spell_key='lod_d2_s003', spell_description='test',
                    spell_level=1, spell_kind='damage', spell_spec={}, cast_result=result,
                    can_cast=True, mana_costs={})
                self.assertEqual(info['enemy_defeat_reward_base'], 0.)

    def test_summon_victory_has_no_defeat_bonus_including_ruin(self):
        for enemy, expected in ((1, 0.), (70, 0.)):
            with self.subTest(enemy=enemy):
                env = self.make()
                env.current_enemy_id = enemy
                env.battle_origin_pos = tuple(env.grid_env.agent_pos)
                result = env._finalize_summon_spell_battle(obs=None, reward=0., info={}, winner='blue')
                self.assertAlmostEqual(result[1], expected)
                self.assertEqual(result[-1]['enemy_defeat_reward'], expected)
                self.assertEqual(result[-1]['enemy_defeat_reward_base'], 0.)

    def test_summon_battle_has_no_tactical_win_bonus(self):
        env = self.make()
        env.current_enemy_id = 1
        env.battle_origin_pos = tuple(env.grid_env.agent_pos)
        env.current_battle_context = {'kind': 'summon_spell'}
        env.mode = env.MODE_BATTLE
        env._init_battle(1, blue_team=env._get_blue_state())
        battle = env.battle_env
        self.assertEqual(battle.reward_win, 0.)
        battle.winner = 'blue'
        with patch.object(battle, 'step', return_value=(battle._obs(), battle.reward_win, True, False, {})):
            _, reward, _, _, info = env.step(0)
        self.assertEqual(info['battle_reward_scaled'], 0.)
        self.assertEqual(info['enemy_defeat_reward'], 0.)
        self.assertEqual(reward, -info['leadership_step_penalty'])
        self.assertEqual(env.hero_defeated_enemy_ids, set())

    def test_terminal_bonus_counts_only_hero_victories_and_resets(self):
        env = self.make()
        # Three defeated enemies, but only one was defeated by the hero's army.
        env.grid_env.enemies_alive = {1: True, 2: True, 3: True, 4: True}
        for enemy in (1, 2, 3):
            env.grid_env.mark_enemy_defeated(enemy)
        env.hero_defeated_enemy_ids.add(1)
        info = {}
        bonus = env._episode_enemies_defeated_bonus(info)
        self.assertEqual(info['enemies_defeated_count'], 3)
        self.assertEqual(info['hero_enemies_defeated_count'], 1)
        self.assertAlmostEqual(bonus, env.reward_enemies_defeated_weight / len(env.grid_env.enemies_alive))
        env.reset(seed=43)
        self.assertEqual(env.hero_defeated_enemy_ids, set())
        self.assertEqual(env._episode_enemies_defeated_bonus({}), 0.)


if __name__ == '__main__':
    unittest.main()
