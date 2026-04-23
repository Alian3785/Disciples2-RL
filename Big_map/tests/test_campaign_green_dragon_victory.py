import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

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


class _StepRewardBattleEnv:
    def __init__(self, reward: float, *, terminated: bool = False, winner: str | None = None):
        self.winner = None
        self.action_space = type("ActionSpace", (), {"n": 15})()
        self.last_battle_exp = 0.0
        self.last_levelups = []
        self.combined = deepcopy(UNITS_BLUE)
        self._reward = float(reward)
        self._terminated = bool(terminated)
        self._winner_after_step = winner

    def _obs(self):
        return np.zeros(FEATURES_PER_UNIT * 12, dtype=np.float32)

    def step(self, action):
        del action
        if self._terminated:
            self.winner = self._winner_after_step
        return self._obs(), self._reward, self._terminated, False, {}


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


def test_first_objective_city_capture_gives_reduced_reward_but_does_not_end_episode():
    env = _make_reward_isolated_env()
    env.reward_all_enemies = 8.0
    env.reset(seed=123)
    env.grid_env.enemies_alive[34] = False
    env.grid_env.enemies_alive[68] = True
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = 68
    env.battle_env = _DummyBattleEnv()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is False
    assert "campaign_result" not in info
    assert info.get("captured_objective_cities") == ["Порт Полонис"]
    assert info.get("objective_cities_captured_total") == ["Порт Полонис"]
    assert reward == 8.0
    assert info.get("final_objective_reward") == 8.0
    assert env.grid_env.enemies_alive[68] is False


def test_green_dragon_has_no_special_final_reward():
    env = _make_reward_isolated_env()
    env.reward_all_enemies = 9.0
    env.reset(seed=123)
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = 31
    env.battle_env = _DummyBattleEnv()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is False
    assert info.get("battle_result") == "victory"
    assert "campaign_result" not in info
    assert "final_objective_reward" not in info
    assert reward == 0.0


def test_second_objective_city_capture_ends_episode_and_keeps_full_reward():
    env = _make_reward_isolated_env()
    env.reward_all_enemies = 8.0
    env.reset(seed=123)
    env.captured_objective_cities = {"Порт Полонис"}
    env.grid_env.enemies_alive[35] = False
    env.grid_env.enemies_alive[69] = True

    env.mode = env.MODE_BATTLE
    env.current_enemy_id = 69
    env.battle_env = _DummyBattleEnv()

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert info.get("campaign_result") == "victory"
    assert info.get("campaign_victory_reason") == "objective_cities_cleared"
    assert info.get("captured_objective_cities") == ["Соругирилла"]
    assert info.get("objective_cities_captured_total") == ["Порт Полонис", "Соругирилла"]
    assert reward == 40.0
    assert info.get("final_objective_reward") == 40.0


def test_campaign_truncates_after_max_grid_steps():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=False,
        realcapital=2,
        max_grid_steps=1,
        reward_timeout=-7.0,
    )
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(8)

    assert terminated is False
    assert truncated is True
    assert info.get("campaign_result") == "timeout"
    assert env.max_grid_steps == 1
    assert env.grid_env.max_steps == 1
    assert reward <= -7.0


def test_agent_returns_to_attack_origin_after_battle_victory():
    env = _make_reward_isolated_env()
    env.reset(seed=123)

    move_mask = env.grid_env.compute_action_mask()
    move_action = next(index for index, allowed in enumerate(move_mask[:8]) if allowed)
    attack_origin = tuple(env.grid_env.agent_pos)
    enemy_pos = tuple(env.grid_env._target_pos_for_action(move_action))
    enemy_id = next(iter(env.grid_env.enemy_positions.keys()))

    env.grid_env.enemy_positions[enemy_id] = enemy_pos
    for eid in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[eid] = False
    env.grid_env.enemies_alive[enemy_id] = True

    _, _, terminated, truncated, info = env.step(move_action)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_BATTLE
    assert env.current_enemy_id == enemy_id
    assert info.get("battle_triggered") is True

    env.battle_env = _DummyBattleEnv()
    _, _, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert info.get("battle_result") == "victory"
    assert tuple(env.grid_env.agent_pos) == attack_origin
    assert tuple(info.get("agent_pos")) == attack_origin


def test_campaign_battle_step_uses_scaled_battle_reward():
    env = _make_reward_isolated_env()
    env.reset(seed=123)
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = 1
    env.battle_reward_scale = 0.25
    env.battle_env = _StepRewardBattleEnv(reward=2.4, terminated=False)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.6)
    assert info.get("battle_ongoing") is True
    assert float(info.get("battle_reward_raw", 0.0)) == pytest.approx(2.4)
    assert float(info.get("battle_reward_scaled", 0.0)) == pytest.approx(0.6)


def test_init_battle_passes_campaign_battle_reward_settings_to_battle_env():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=False,
        realcapital=2,
        battle_reward_win=3.5,
        battle_reward_loss=-2.25,
        battle_reward_step=0.125,
    )
    env.reset(seed=123)

    env._init_battle(enemy_id=1)

    assert env.battle_env is not None
    assert env.battle_env.reward_win == pytest.approx(3.5)
    assert env.battle_env.reward_loss == pytest.approx(-2.25)
    assert env.battle_env.reward_step == pytest.approx(0.125)


def test_city_internal_garrison_stays_on_tile_after_outer_stack_is_defeated():
    env = _make_reward_isolated_env()
    env.reset(seed=123)

    outer_enemy_id = 34
    inner_enemy_id = 68
    move_mask = env.grid_env.compute_action_mask()
    move_action = next(index for index, allowed in enumerate(move_mask[:8]) if allowed)
    attack_origin = tuple(env.grid_env.agent_pos)
    target_tile = tuple(env.grid_env._target_pos_for_action(move_action))

    env.grid_env.enemy_positions[outer_enemy_id] = target_tile
    env.grid_env.enemy_positions[inner_enemy_id] = target_tile
    for enemy_id in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[enemy_id] = False
    env.grid_env.enemies_alive[outer_enemy_id] = True
    env.grid_env.enemies_alive[inner_enemy_id] = True

    _, _, terminated, truncated, info = env.step(move_action)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_BATTLE
    assert env.current_enemy_id == outer_enemy_id
    assert info.get("battle_triggered") is True

    env.battle_env = _DummyBattleEnv()
    _, _, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is False
    assert info.get("battle_result") == "victory"
    assert tuple(env.grid_env.agent_pos) == attack_origin
    assert env.grid_env.enemies_alive[outer_enemy_id] is False
    assert env.grid_env.enemies_alive[inner_enemy_id] is True
    assert env.grid_env.get_enemy_at_position(target_tile) == inner_enemy_id

    _, _, terminated, truncated, info = env.step(move_action)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_BATTLE
    assert env.current_enemy_id == inner_enemy_id
    assert info.get("battle_triggered") is True


def test_city_with_only_internal_garrison_starts_battle_against_it_immediately():
    env = _make_reward_isolated_env()
    env.reset(seed=123)

    inner_enemy_id = 69
    move_mask = env.grid_env.compute_action_mask()
    move_action = next(index for index, allowed in enumerate(move_mask[:8]) if allowed)
    target_tile = tuple(env.grid_env._target_pos_for_action(move_action))

    env.grid_env.enemy_positions[35] = target_tile
    env.grid_env.enemy_positions[inner_enemy_id] = target_tile
    for enemy_id in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[enemy_id] = False
    env.grid_env.enemies_alive[inner_enemy_id] = True

    _, _, terminated, truncated, info = env.step(move_action)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_BATTLE
    assert env.current_enemy_id == inner_enemy_id
    assert info.get("battle_triggered") is True
