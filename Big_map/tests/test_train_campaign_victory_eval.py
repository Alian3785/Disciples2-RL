import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np

from train_campaign import is_better_campaign_eval
from train_campaign import EvalStepCapWrapper


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
