"""Movement bonuses use actual displacement, per-episode visits and live objects."""
import os
os.environ['CAMPAIGN_OBSERVATION_VERSION'] = 'local5'

import unittest
from campaign_env import CampaignEnv


class MovementRewardTests(unittest.TestCase):
    def make(self):
        env = CampaignEnv(map_name='default', campaign_objective='blue_dragon',
                          use_boss_starting_roster=False, scripted_capital_bot_enabled=False,
                          detailed_step_info=False)
        env.reset(seed=42)
        self.addCleanup(env.close)
        self.prepare_position(env)
        return env

    def prepare_position(self, env):
        env.grid_env.agent_pos = (1, 1)
        env.grid_env.visited_cells = {(1, 1)}
        env.grid_env.obstacle_positions = set()
        env.grid_env.dynamic_blocked_positions = set()
        env.grid_env.enemy_positions = {12: (30, 30)}
        env.grid_env.enemies_alive = {12: True}
        env.moves = 100

    def move(self, env, delta):
        action = list(env.grid_env.ACTION_DELTAS).index(delta)
        return env.step(action)[-1]

    def test_new_empty_cell_then_revisit_and_reset(self):
        env = self.make()
        info = self.move(env, (1, 0))
        self.assertEqual(info['movement_reward'], .01)
        self.assertEqual(info['new_cell_reward'], .08)
        self.assertEqual(info['nonempty_cell_reward'], 0.)
        self.move(env, (-1, 0))
        info = self.move(env, (1, 0))
        self.assertEqual(info['movement_reward'], .01)
        self.assertEqual(info['new_cell_reward'], 0.)
        env.reset(seed=42)
        self.prepare_position(env)
        self.assertEqual(self.move(env, (1, 0))['new_cell_reward'], .08)

    def test_object_bonus_stacks_and_applies_on_revisit(self):
        env = self.make()
        env.grid_env.merchant_positions.add((2, 1))
        info = self.move(env, (1, 0))
        self.assertAlmostEqual(sum(info[k] for k in
            ('movement_reward', 'new_cell_reward', 'nonempty_cell_reward')), .24)
        self.move(env, (-1, 0))
        info = self.move(env, (1, 0))
        self.assertEqual(info['new_cell_reward'], 0.)
        self.assertEqual(info['nonempty_cell_reward'], .05)
        env.reset(seed=42)
        self.prepare_position(env)
        env.grid_env.merchant_positions.add((2, 1))
        self.assertAlmostEqual(self.move(env, (1, 0))['nonempty_cell_reward'], .15)

    def test_rest_boundary_and_obstacle_have_no_movement_bonus(self):
        env = self.make()
        infos = [env.step(8)[-1]]
        env.grid_env.agent_pos = (0, 0)
        infos.append(self.move(env, (-1, 0)))
        env.grid_env.obstacle_positions.add((1, 0))
        infos.append(self.move(env, (1, 0)))
        for info in infos:
            self.assertFalse(info['movement_applied'])
            for key in ('movement_reward', 'new_cell_reward', 'nonempty_cell_reward'):
                self.assertEqual(info[key], 0.)

    def test_chest_bonus_precedes_collection_and_does_not_persist(self):
        env = self.make()
        env.chests[(2, 1)] = ()
        env._sync_grid_chest_positions()
        info = self.move(env, (1, 0))
        self.assertAlmostEqual(info['nonempty_cell_reward'], .15)
        self.assertNotIn((2, 1), env.chests)
        self.move(env, (-1, 0))
        self.assertEqual(self.move(env, (1, 0))['nonempty_cell_reward'], 0.)

    def test_live_enemy_entry_gets_bonus_but_cleared_empty_tile_does_not(self):
        env = self.make()
        env.grid_env.enemy_positions = {12: (2, 1)}
        self.assertTrue(env._movement_tile_is_nonempty((2, 1)))
        info = self.move(env, (1, 0))
        self.assertTrue(info['battle_triggered'])
        self.assertAlmostEqual(info['nonempty_cell_reward'], .15)
        env.grid_env.mark_enemy_defeated(12)
        self.assertFalse(env._movement_tile_is_nonempty((2, 1)))

    def test_rest_penalty_uses_moves_before_refill(self):
        env = self.make()
        rewards = []
        for remaining in (1, 0):
            env.reset(seed=42)
            env.moves = remaining
            turn_before = env.turns
            _, reward, _, _, info = env.step(8)
            rewards.append(reward)
            self.assertEqual(info['rest_moves_before'], remaining)
            self.assertEqual(info['rest_with_moves_penalty'], .20 if remaining else 0.)
            self.assertEqual(env.turns, turn_before + 1)
            self.assertGreater(env.moves, 0)
        self.assertAlmostEqual(rewards[0] - rewards[1], -.20)


if __name__ == '__main__':
    unittest.main()
