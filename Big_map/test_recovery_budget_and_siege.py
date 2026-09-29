"""Victory-funded HP/kill budgets and the final siege retreat restriction."""
import os
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'

import unittest
from copy import deepcopy
from unittest.mock import patch

from campaign_env import CampaignEnv
from battle_env import RUN_AWAY_ACTION_INDEX


class RecoveryAndSiegeTests(unittest.TestCase):
    def make(self, map_name='siege_train', **kwargs):
        env = CampaignEnv(map_name=map_name, scripted_capital_bot_enabled=False,
                          use_boss_starting_roster=False, log_enabled=False,
                          detailed_step_info=False, **kwargs)
        self.addCleanup(env.close)
        env.reset(seed=42)
        env.gold = 100000.
        return env

    def start_battle(self, env, enemy=1, kind='hero'):
        env.current_enemy_id = enemy
        env.battle_origin_pos = tuple(env.grid_env.agent_pos)
        env.current_battle_context = {'kind': kind}
        env.mode = env.MODE_BATTLE
        env._init_battle(enemy)

    def kill_enemies(self, battle):
        for unit in battle.combined:
            if unit['team'] == 'red' and unit['health'] > 0:
                battle._subtract_health(unit, unit['health'] + 10000)
        return sum(battle.recovery_damage_by_position.values()), len(battle.recovery_killed_positions)

    def win(self, env, enemy=1, kind='hero'):
        self.start_battle(env, enemy, kind)
        self.kill_enemies(env.battle_env)
        env.battle_env.winner = 'blue'
        return env.step(0)

    def test_victory_commits_actual_hp_and_kills_once_and_reset_clears(self):
        env = self.make()
        self.assertEqual(env.healing_reward_hp_balance, 0.)
        self.assertEqual(env.revive_reward_kill_balance, 0)
        self.assertFalse(env.grid_env.enemies_alive[100])
        self.start_battle(env)
        battle = env.battle_env
        damage, kills = self.kill_enemies(battle)
        self.assertGreater(damage, 0.)
        self.assertGreater(kills, 0)
        self.assertEqual(env.healing_reward_hp_balance, 0.)
        battle.winner = 'blue'
        info = env.step(0)[-1]
        self.assertEqual(info['recovery_victory_damage_hp'], damage)
        self.assertEqual(info['healing_reward_hp_balance'], damage)
        self.assertEqual(info['revive_reward_kill_balance'], kills)
        # Duplicate processing cannot grant the same combat result twice.
        env.battle_env = battle
        env._credit_victorious_battle_recovery({})
        self.assertEqual(env.healing_reward_hp_balance, damage)
        _, info = env.reset(seed=43)
        self.assertEqual(info['healing_reward_hp_balance'], 0.)
        self.assertEqual(info['revive_reward_kill_balance'], 0)
        self.assertEqual(info['healing_reward_hp_earned'], 0.)

    def test_heal_and_revive_have_separate_balances_and_partial_payout(self):
        env = self.make()
        env.healing_reward_hp_balance = 30.
        env.revive_reward_kill_balance = 1
        unit = env._blue_unit_at_position(7)
        unit.update(health=60., hp=60.)
        with patch.object(env, '_has_temple_built', return_value=True):
            info = env.step(env.GRID_CASTLE_HEAL_ACTION_START)[-1]
            self.assertEqual(info['healed_amount'], 60.)
            self.assertAlmostEqual(info['castle_heal_reward'], .3)
            self.assertEqual(info['healing_reward_hp_balance'], -30.)
            self.assertEqual(info['revive_reward_kill_balance'], 1)
            unit.update(health=0., hp=0.)
            self.assertTrue(env.compute_action_mask()[env.GRID_CASTLE_REVIVE_ACTION_START])
            info = env.step(env.GRID_CASTLE_REVIVE_ACTION_START)[-1]
            self.assertTrue(info['revived'])
            self.assertEqual(info['castle_revive_reward'], .5)
            self.assertEqual(info['revive_reward_kill_balance'], 0)
            self.assertEqual(info['healing_reward_hp_balance'], -30.)
            unit.update(health=0., hp=0.)
            info = env.step(env.GRID_CASTLE_REVIVE_ACTION_START)[-1]
            self.assertTrue(info['revived'])
            self.assertEqual(info['castle_revive_reward'], 0.)
            self.assertEqual(info['revive_reward_kill_balance'], -1)
            self.assertTrue(env.compute_action_mask()[env.GRID_CASTLE_HEAL_ACTION_START])
            info = env.step(env.GRID_CASTLE_HEAL_ACTION_START)[-1]
            self.assertEqual(info['healed_amount'], 119.)
            self.assertEqual(info['castle_heal_reward'], 0.)
            self.assertEqual(info['healing_reward_hp_balance'], -149.)
            self.assertGreater(info['gold_spent'], 0.)
            self.assertEqual(unit['health'], 120.)
        # Future wins first repay any actual healing/revival performed over budget.
        info = self.win(env)[-1]
        self.assertEqual(env.healing_reward_hp_balance, info['recovery_victory_damage_hp'] - 149.)
        self.assertEqual(env.revive_reward_kill_balance, info['recovery_victory_kills'] - 1)

    def test_failed_recovery_does_not_spend_balances(self):
        env = self.make()
        with patch.object(env, '_has_temple_built', return_value=True):
            for action in (env.GRID_CASTLE_HEAL_ACTION_START, env.GRID_CASTLE_REVIVE_ACTION_START):
                info = env.step(action)[-1]
                self.assertEqual(info['healing_reward_hp_balance'], 0.)
                self.assertEqual(info['revive_reward_kill_balance'], 0)

    def test_loss_retreat_timeout_discard_damage_and_kills(self):
        for result in ('loss', 'retreat', 'timeout'):
            with self.subTest(result=result):
                env = self.make()
                self.start_battle(env)
                battle = env.battle_env
                victim = next(u for u in battle.combined if u['team'] == 'red' and u['health'] > 0)
                battle._subtract_health(victim, 10000)
                self.assertGreater(sum(battle.recovery_damage_by_position.values()), 0.)
                if result == 'retreat':
                    battle.escaped_units = [deepcopy(next(u for u in battle.combined if u['team'] == 'blue' and u['health'] > 0))]
                if result == 'timeout':
                    with patch.object(battle, 'step', return_value=(battle._obs(), 0., False, True, {})):
                        info = env.step(0)[-1]
                    self.assertEqual(info['battle_result'], 'timeout')
                else:
                    battle.winner = 'red'
                    info = env.step(0)[-1]
                    if result == 'retreat':
                        self.assertTrue(info['battle_retreat'])
                self.assertEqual(env.healing_reward_hp_balance, 0.)
                self.assertEqual(env.revive_reward_kill_balance, 0)

    def test_map_magic_kill_does_not_refill_but_summon_battle_win_does(self):
        env = self.make()
        target = dict(enemy_id=1, position=env.grid_env.enemy_positions[1], reachable=True)
        with patch.object(env, '_resolve_spell_target_enemy', return_value=target), \
             patch.object(env, '_apply_damage_to_enemy_stack', return_value=(100., 1, 1, 0)), \
             patch.object(env, '_enemy_team_has_living_units', return_value=False):
            result = env._cast_from_spell_spec(source='learned_offensive_spell',
                spell_key='lod_d2_s003', spell_description='test',
                spell_spec={'spell_kind': 'damage', 'damage': 100., 'damage_type': 'Fire'})
        self.assertTrue(result['spell_enemy_defeated'])
        self.assertEqual(env.healing_reward_hp_balance, 0.)
        self.assertEqual(env.revive_reward_kill_balance, 0)
        info = self.win(env, 2, kind='summon_spell')[-1]
        self.assertGreater(info['recovery_victory_damage_hp'], 0.)
        self.assertGreater(info['recovery_victory_kills'], 0)

    def test_damage_is_capped_per_original_enemy_and_summons_do_not_count(self):
        env = self.make()
        self.start_battle(env)
        battle = env.battle_env
        victim = next(u for u in battle.combined if u['team'] == 'red' and u['health'] > 0)
        pos, hp = victim['position'], victim['health']
        battle._subtract_health(victim, 10)
        victim['health'] = hp  # enemy healing cannot create unlimited allowance
        battle._subtract_health(victim, 10000)
        victim['health'] = hp  # enemy resurrection cannot create another kill
        battle._subtract_health(victim, 10000)
        self.assertEqual(battle.recovery_damage_by_position[pos], hp)
        self.assertEqual(len(battle.recovery_killed_positions), 1)
        summoned = deepcopy(victim)
        summoned.update(health=100., Summoned=2)
        battle._subtract_health(summoned, 10000)
        self.assertEqual(battle.recovery_damage_by_position[pos], hp)
        self.assertEqual(len(battle.recovery_killed_positions), 1)
        own = next(u for u in battle.combined if u['team'] == 'blue' and u['health'] > 0)
        battle._subtract_health(own, 1)
        self.assertEqual(sum(battle.recovery_damage_by_position.values()), hp)

    def test_combat_item_damage_counts_but_transformations_do_not(self):
        env = self.make()
        self.start_battle(env)
        battle = env.battle_env
        victim = next(u for u in battle.combined if u['team'] == 'red' and u['health'] > 0)
        pos, hp = victim['position'], victim['health']
        with patch.object(battle, '_hero_item_type_effect_blocked', return_value=False), \
             patch.object(battle, '_hero_item_effect_amount', return_value=10000.):
            battle._apply_hero_item_damage(source_unit=None, target_unit=victim, effect={})
        self.assertEqual(battle.recovery_damage_by_position[pos], hp)
        self.assertIn(pos, battle.recovery_killed_positions)
        battle._init_with_custom_teams(env._enemy_configs[2], env.blue_team_state)
        self.assertEqual(battle.recovery_damage_by_position, {})
        self.assertEqual(battle.recovery_killed_positions, set())

    def test_city_garrison_victory_refills_before_battle_is_disposed(self):
        env = self.make('default')
        city = env.garrison_city_names[0]
        env.legions_active_settlement_territory_capture_turn_by_name[city] = 0
        env.city_garrisons[city] = [env._build_unit_from_data(env._find_unit_data_by_name('Одержимый'), 'blue', 7)]
        env.scripted_capital_bot_team_state = env._build_battle_team_with_placeholders('blue', [
            env._build_unit_from_data(env._find_unit_data_by_name('Крестьянин'), 'blue', 7)])
        env._start_city_defence(city)
        damage, kills = self.kill_enemies(env.battle_env)
        env.battle_env.winner = 'blue'
        info = env.step(0)[-1]
        self.assertEqual(info['healing_reward_hp_balance'], damage)
        self.assertEqual(info['revive_reward_kill_balance'], kills)
        self.assertGreater(damage, 0.)
        self.assertEqual(kills, 1)
    def test_siege_final_retreat_mask_and_direct_action_blocked(self):
        env = self.make()
        self.start_battle(env, 100)
        battle = env.battle_env
        self.assertFalse(battle.retreat_enabled)
        self.assertFalse(battle.compute_action_mask()[RUN_AWAY_ACTION_INDEX])
        self.assertFalse(env.compute_action_mask()[RUN_AWAY_ACTION_INDEX])
        before = deepcopy(battle.combined)
        obs, reward, terminated, truncated, info = env.step(RUN_AWAY_ACTION_INDEX)
        self.assertTrue(info['battle_retreat_blocked'])
        self.assertFalse(terminated or truncated)
        self.assertLess(reward, 0.)
        self.assertEqual(battle.combined, before)
        self.assertEqual(battle.escaped_units, [])
        self.assertTrue(env.observation_space.contains(obs))
        env.grid_env.agent_pos = tuple(env.grid_env.enemy_positions[1])
        self.start_battle(env, 1)
        self.assertTrue(env.battle_env.retreat_enabled)
        self.assertTrue(env.compute_action_mask()[RUN_AWAY_ACTION_INDEX])
        other = self.make('default')
        other.grid_env.agent_pos = tuple(other.grid_env.enemy_positions[1])
        self.start_battle(other, 1)
        self.assertTrue(other.battle_env.retreat_enabled)

    def test_siege_final_victory_and_defeat_end_campaign(self):
        for winner, result in (('blue', 'victory'), ('red', 'defeat')):
            with self.subTest(winner=winner):
                env = self.make()
                self.start_battle(env, 100)
                env.battle_env.winner = winner
                _, _, terminated, truncated, info = env.step(0)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info['campaign_result'], result)


if __name__ == '__main__':
    unittest.main()
