import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from train_campaign import is_better_campaign_eval
from train_campaign import EvalStepCapWrapper
from train_campaign import make_env
from train_campaign import resolve_training_vec_env_config


def test_victory_rate_beats_higher_reward():
    assert is_better_campaign_eval(
        victory_rate=0.3,
        mean_reward=10.0,
        best_victory_rate=0.2,
        best_mean_reward=100.0,
    )


def test_lower_victory_rate_never_beats_best():
    assert not is_better_campaign_eval(
        victory_rate=0.2,
        mean_reward=1000.0,
        best_victory_rate=0.3,
        best_mean_reward=-1000.0,
    )


def test_mean_reward_breaks_tie_when_victory_rate_matches():
    assert is_better_campaign_eval(
        victory_rate=0.3,
        mean_reward=50.0,
        best_victory_rate=0.3,
        best_mean_reward=40.0,
    )
    assert not is_better_campaign_eval(
        victory_rate=0.3,
        mean_reward=40.0,
        best_victory_rate=0.3,
        best_mean_reward=50.0,
    )


def test_resolve_training_vec_env_config_uses_subproc_spawn():
    vec_env_cls, vec_env_kwargs = resolve_training_vec_env_config("subproc")

    assert vec_env_cls is SubprocVecEnv
    assert vec_env_kwargs == {"start_method": "spawn"}


def test_resolve_training_vec_env_config_keeps_dummy_fallback():
    vec_env_cls, vec_env_kwargs = resolve_training_vec_env_config("dummy")

    assert vec_env_cls is DummyVecEnv
    assert vec_env_kwargs == {}


def test_make_env_exposes_native_action_masks_for_subproc():
    env = make_env()
    try:
        base_env = env.unwrapped

        assert callable(base_env.action_masks)
        np.testing.assert_array_equal(
            base_env.action_masks(),
            base_env.compute_action_mask(),
        )
    finally:
        env.close()


def test_eval_wrapped_make_env_exposes_action_masks():
    env = DummyVecEnv([lambda: make_env(eval_max_episode_steps=3)])
    try:
        masks = get_action_masks(env)

        assert masks.shape == (1, env.action_space.n)
        assert masks.dtype == np.bool_
    finally:
        env.close()


class DummyInfiniteEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        info = {"inner_steps": self.steps}
        return np.zeros((1,), dtype=np.float32), 1.0, False, False, info


def test_eval_step_cap_wrapper_truncates_long_episode():
    env = EvalStepCapWrapper(DummyInfiniteEnv(), max_steps=3)
    obs, info = env.reset()
    assert obs.shape == (1,)
    assert info == {}

    for expected_step in (1, 2):
        _, _, terminated, truncated, info = env.step(0)
        assert not terminated
        assert not truncated
        assert info["inner_steps"] == expected_step

    _, _, terminated, truncated, info = env.step(0)
    assert not terminated
    assert truncated
    assert info["campaign_result"] == "eval_timeout"
    assert info["eval_step_cap_hit"] is True
    assert info["eval_step_cap"] == 3
