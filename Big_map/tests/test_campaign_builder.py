import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import FEATURES_PER_UNIT, UNITS_BLUE
from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.builder import BUILDER_BYSTANDER_ENEMY_ID


class _DummyBattleEnv:
    def __init__(self):
        self.winner = "blue"
        self.action_space = type("ActionSpace", (), {"n": 15})()
        self.last_battle_exp = 0.0
        self.last_levelups = []
        self.combined = deepcopy(UNITS_BLUE)

    def _obs(self):
        return np.zeros(FEATURES_PER_UNIT * 12, dtype=np.float32)


def _make_builder_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="builder",
        log_enabled=False,
        persist_blue_hp=False,
        Realcapital=2,
        reward_defeat_enemy=0.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _first_freely_buildable_key(env: CampaignEnv) -> str:
    """Первое здание без prerequisite-требований и без блокировки."""
    for build_key in env.building_keys:
        building = env.active_buildings.get(build_key)
        if not isinstance(building, dict):
            continue
        if int(building.get("built", 0) or 0) == 1:
            continue
        if int(building.get("blocked", 0) or 0) == 1:
            continue
        if env._building_requirement_names(building):
            continue
        return str(build_key)
    raise AssertionError("no freely buildable building found")


def _build_action_for_key(env: CampaignEnv, build_key: str) -> int:
    return int(env.GRID_BUILD_ACTION_START) + env.building_keys.index(build_key)


def test_builder_map_is_registered():
    assert "builder" in available_maps()
    map_config = get_map("builder")
    assert map_config.default_objective == "build_all"
    assert map_config.objective_enemy_id is None
    assert map_config.enemy_positions() == {BUILDER_BYSTANDER_ENEMY_ID: (30, 46)}


def test_builder_env_is_a_clean_field_with_build_all_objective():
    env = _make_builder_env()

    assert env.map_name == "builder"
    assert env.campaign_objective == "build_all"
    assert env._campaign_objective_is_build_all() is True
    assert env._campaign_objective_is_dragon() is False
    assert dict(env.grid_env.enemy_positions) == {BUILDER_BYSTANDER_ENEMY_ID: (30, 46)}
    assert tuple(env.castle_heal_tiles) == ((5, 27),)
    assert env.CASTLE_POS == (5, 27)
    assert env._static_obstacle_tiles == ()
    assert env._static_chests == {}
    assert env._static_mana_sources == {}
    assert env.water_tiles == ()
    assert env.forest_tiles == ()
    assert env.road_tiles == ()
    assert env.FINAL_OBJECTIVE_CITIES == {}
    assert env.RUIN_REWARD_BY_ENEMY_ID == {}
    assert env.BASE_MERCHANT_SITE_DATA == {}
    assert env.BASE_SPELL_SHOP_SITE_DATA == {}
    assert env.BASE_MERCENARY_SITE_DATA == {}
    assert env.BASE_TRAINER_SITE_DATA == {}
    assert env.scripted_capital_bot_config_enabled is False
    assert env.empire_territory_enabled is False

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert dict(env.grid_env.enemies_alive) == {BUILDER_BYSTANDER_ENEMY_ID: True}


def test_building_gives_intermediate_reward_and_does_not_end_episode():
    env = _make_builder_env()
    env.reset(seed=0)

    build_key = _first_freely_buildable_key(env)
    building = env.active_buildings[build_key]
    env.gold = env._get_building_gold_cost(building)

    _, reward, terminated, truncated, info = env.step(_build_action_for_key(env, build_key))

    assert info.get("built") is True
    assert info.get("building_built_reward") == pytest.approx(env.reward_building_built)
    assert reward == pytest.approx(env.reward_building_built)
    assert terminated is False
    assert truncated is False
    assert "campaign_result" not in info
    assert int(info.get("buildable_buildings_remaining", -1)) > 0


def test_build_all_completion_ends_episode_with_dragon_sized_reward():
    env = _make_builder_env()
    env.reward_all_enemies = 9.0
    env.reset(seed=1)

    build_key = _first_freely_buildable_key(env)
    for other_key in env.building_keys:
        if other_key == build_key:
            continue
        env.active_buildings[other_key]["built"] = 1
    env.gold = env._get_building_gold_cost(env.active_buildings[build_key])
    env.active_buildings["alredybuilt"] = 0

    _, reward, terminated, truncated, info = env.step(_build_action_for_key(env, build_key))

    expected_final_reward = (
        9.0
        * (1.0 + env.FINAL_OBJECTIVE_CITY_REWARD_MULTIPLIER)
        * env.reward_green_dragon_objective_multiplier
    )
    assert info.get("built") is True
    assert terminated is True
    assert truncated is False
    assert info.get("campaign_result") == "victory"
    assert info.get("campaign_victory_reason") == "all_buildings_built"
    assert info.get("campaign_objective") == "build_all"
    assert info.get("buildable_buildings_remaining") == 0
    assert info.get("final_objective_reward") == pytest.approx(expected_final_reward)
    assert info.get("build_all_objective_reward") == pytest.approx(expected_final_reward)
    assert reward == pytest.approx(env.reward_building_built + expected_final_reward)


def test_blocked_buildings_do_not_prevent_build_all_victory():
    env = _make_builder_env()
    env.reset(seed=2)

    build_key = _first_freely_buildable_key(env)
    blocked_key = None
    for other_key in env.building_keys:
        if other_key == build_key:
            continue
        if blocked_key is None:
            blocked_key = other_key
            env.active_buildings[other_key]["blocked"] = 1
            continue
        env.active_buildings[other_key]["built"] = 1
    assert blocked_key is not None
    env.gold = env._get_building_gold_cost(env.active_buildings[build_key])

    _, _, terminated, _, info = env.step(_build_action_for_key(env, build_key))

    assert info.get("built") is True
    assert terminated is True
    assert info.get("campaign_result") == "victory"
    assert info.get("campaign_victory_reason") == "all_buildings_built"


def test_defeating_the_bystander_orc_does_not_end_episode():
    env = _make_builder_env()
    env.reset(seed=3)
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = BUILDER_BYSTANDER_ENEMY_ID
    env.battle_env = _DummyBattleEnv()

    _, _, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is False
    assert info.get("campaign_result") is None
    assert info.get("battle_result") == "victory"
    assert dict(env.grid_env.enemies_alive) == {BUILDER_BYSTANDER_ENEMY_ID: False}
    assert env.mode == env.MODE_GRID


def test_invalid_objective_map_combinations_raise():
    with pytest.raises(ValueError):
        CampaignEnv(map_name="builder", campaign_objective="cities", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(map_name="builder", campaign_objective="dragon", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(map_name="builder", campaign_objective="orc", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(campaign_objective="build_all", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(map_name="orc_duel", campaign_objective="build_all", Realcapital=2)


def test_builder_random_masked_rollout_is_stable():
    env = _make_builder_env(freeze_dynamic_action_layout=True)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for step_idx in range(200):
        mask = env.compute_action_mask()
        valid_actions = np.flatnonzero(mask)
        assert valid_actions.size > 0, f"empty action mask at step {step_idx}"
        action = int(rng.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        if terminated or truncated:
            env.reset(seed=step_idx + 1)
