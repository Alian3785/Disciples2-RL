import unittest
from unittest.mock import patch

from campaign_env import CampaignEnv
from train_campaign import EvalStepCapWrapper, REWARD_CONFIG


class MagicTrainRewardTests(unittest.TestCase):
    def make_env(self, map_name="magic_train"):
        env = CampaignEnv(
            map_name=map_name,
            observation_version="local5",
            scripted_capital_bot_enabled=False,
            use_boss_starting_roster=False,
            **REWARD_CONFIG,
        )
        env.reset(seed=42)
        self.addCleanup(env.close)
        return env

    def test_magic_map_overrides_training_defaults(self):
        env = self.make_env()
        expected = {
            "reward_spell_cast": 1.0,
            "reward_new_cell": 0.005,
            "reward_turn_penalty": 0.20,
            "reward_rest_with_moves_penalty": 1.0,
            "reward_timeout": -50.0,
            "reward_all_enemies": 200.0,
        }
        for key, value in expected.items():
            self.assertEqual(getattr(env, key), value, key)
        self.assertEqual(env._magic_enemy_defeat_reward_value(), 20.0)

    def test_other_maps_keep_training_defaults(self):
        for map_name in ("default", "super_last_stand"):
            env = self.make_env(map_name)
            for key in (
                "reward_spell_cast", "reward_new_cell", "reward_turn_penalty",
                "reward_rest_with_moves_penalty", "reward_timeout",
            ):
                self.assertEqual(getattr(env, key), REWARD_CONFIG[key], (map_name, key))
            self.assertEqual(env._magic_enemy_defeat_reward_value(), 0.0)
            self.assertEqual(env.reward_all_enemies, REWARD_CONFIG["reward_all_enemies"])

    def kill_with_spell(self, env, enemy_id):
        target = dict(enemy_id=enemy_id,
                      position=env.grid_env.enemy_positions[enemy_id], reachable=False)
        with patch.object(env, "_resolve_spell_target_enemy", return_value=target):
            result = env._cast_from_spell_spec(
                source="learned_offensive_spell", spell_key="lod_d2_s003",
                spell_description="lethal test spell",
                spell_spec={"spell_kind": "damage", "damage": 10000.0, "damage_type": "Fire"},
            )
        self.assertTrue(result["spell_enemy_defeated"])
        self.assertEqual(result["enemy_defeat_reward"], 20.0)
        self.assertEqual(result["enemy_defeat_reward_base"], 20.0)
        return env._finalize_map_spell_step(reward=result["reward"], info={}, cast_result=result)

    def test_spell_kills_and_final_objective_bonus(self):
        env = self.make_env()
        _, first_reward, terminated, _, info = self.kill_with_spell(env, 1)
        self.assertFalse(terminated)
        self.assertEqual(first_reward, 21.0)
        self.assertNotIn("all_enemies_reward", info)
        _, final_reward, terminated, _, info = self.kill_with_spell(env, 2)
        self.assertTrue(terminated)
        self.assertEqual(info["campaign_result"], "victory")
        self.assertEqual(info["all_enemies_reward"], 200.0)
        self.assertEqual(final_reward, 221.0)
        self.assertEqual(env._apply_all_enemies_objective_reward_if_needed(0.0, {}), 0.0)
        env.reset(seed=43)
        self.assertFalse(env.all_enemies_reward_granted)

    def test_summoned_units_also_earn_clear_bonus(self):
        env = self.make_env()
        for enemy_id, expected_reward, expected_done in ((1, 20.0, False), (2, 220.0, True)):
            env.current_enemy_id = enemy_id
            env.battle_origin_pos = tuple(env.grid_env.agent_pos)
            _, reward, terminated, _, info = env._finalize_summon_spell_battle(
                obs=None, reward=0.0, info={}, winner="blue")
            self.assertEqual(reward, expected_reward)
            self.assertEqual(terminated, expected_done)
            self.assertEqual(info["enemy_defeat_reward"], 20.0)
        self.assertEqual(info["all_enemies_reward"], 200.0)

    def test_action_cap_charges_magic_timeout_once(self):
        capped = EvalStepCapWrapper(self.make_env(), max_steps=1)
        uncapped = EvalStepCapWrapper(self.make_env(), max_steps=2)
        _, capped_reward, terminated, truncated, info = capped.step(8)
        _, uncapped_reward, _, _, _ = uncapped.step(8)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(capped_reward - uncapped_reward, -50.0)
        self.assertEqual(info["step_cap_timeout_reward"], -50.0)

    def test_existing_termination_does_not_charge_again(self):
        for terminated, truncated in ((True, False), (False, True)):
            env = self.make_env()
            obs = env.observation_space.sample()
            env.step = lambda action: (obs, -50.0, terminated, truncated, {})
            wrapped = EvalStepCapWrapper(env, max_steps=1)
            _, reward, _, _, info = wrapped.step(8)
            self.assertEqual(reward, -50.0)
            self.assertNotIn("step_cap_timeout_reward", info)


if __name__ == "__main__":
    unittest.main()
