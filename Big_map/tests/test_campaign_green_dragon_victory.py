import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import FEATURES_PER_UNIT, UNITS_BLUE
from campaign_env import CampaignEnv


class _DummyBattleEnv:
    def __init__(self):
        self.winner = "blue"
        self.action_space = type("ActionSpace", (), {"n": 15})()
        self.last_battle_exp = 0.0
        self.last_levelups = []
        self.combined = deepcopy(UNITS_BLUE)

    def _obs(self):
        return np.zeros(FEATURES_PER_UNIT * 12, dtype=np.float32)


def _make_reward_isolated_env() -> CampaignEnv:
    return CampaignEnv(
        log_enabled=False,
        persist_blue_hp=False,
        realcapital=2,
        reward_defeat_enemy=0.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
    )


def test_green_dragon_victory_ends_episode_and_gives_x10_final_reward():
    env = _make_reward_isolated_env()
    env.reward_all_enemies = 9.0
    env.reset(seed=123)
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = env.GREEN_DRAGON_ENEMY_ID
    env.battle_env = _DummyBattleEnv()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert info.get("campaign_result") == "victory"
    assert info.get("campaign_victory_reason") == "green_dragon_defeated"
    assert reward == 90.0
    assert info.get("green_dragon_reward") == 90.0
    assert env.grid_env.enemies_alive[env.GREEN_DRAGON_ENEMY_ID] is False


def test_all_enemies_victory_has_no_extra_final_reward():
    env = _make_reward_isolated_env()
    env.reward_all_enemies = 13.0
    env.reset(seed=123)

    for enemy_id in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[enemy_id] = False
    env.grid_env.enemies_alive[1] = True

    env.mode = env.MODE_BATTLE
    env.current_enemy_id = 1
    env.battle_env = _DummyBattleEnv()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert info.get("campaign_result") == "victory"
    assert info.get("campaign_victory_reason") == "all_enemies_defeated"
    assert reward == 0.0
    assert "green_dragon_reward" not in info
