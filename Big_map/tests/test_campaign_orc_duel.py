import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import FEATURES_PER_UNIT, UNITS_BLUE
from campaign_env import CampaignEnv
from campaign_env_data import CampaignConstantsMixin
from maps import available_maps, get_map
from maps.orc_duel import ORC_ENEMY_ID


class _DummyBattleEnv:
    def __init__(self):
        self.winner = "blue"
        self.action_space = type("ActionSpace", (), {"n": 15})()
        self.last_battle_exp = 0.0
        self.last_levelups = []
        self.combined = deepcopy(UNITS_BLUE)

    def _obs(self):
        return np.zeros(FEATURES_PER_UNIT * 12, dtype=np.float32)


def _make_orc_duel_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="orc_duel",
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


def test_orc_duel_map_is_registered():
    assert "orc_duel" in available_maps()
    map_config = get_map("orc_duel")
    assert map_config.default_objective == "orc"
    assert map_config.objective_enemy_id == ORC_ENEMY_ID
    assert map_config.enemy_positions() == {ORC_ENEMY_ID: (30, 46)}


def test_orc_duel_env_is_a_clean_field_with_single_orc():
    env = _make_orc_duel_env()

    assert env.map_name == "orc_duel"
    assert env.campaign_objective == "orc"
    assert env.OBJECTIVE_ENEMY_ID == ORC_ENEMY_ID
    assert dict(env.grid_env.enemy_positions) == {ORC_ENEMY_ID: (30, 46)}
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

    # Единственный отряд врага — один Орк в центре передней линии.
    orc_team = env._enemy_configs[ORC_ENEMY_ID]
    living = [
        unit
        for unit in orc_team
        if str(unit.get("name", "")).strip().lower() != "пусто"
    ]
    assert [unit["name"] for unit in living] == ["Орк"]

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert dict(env.grid_env.enemies_alive) == {ORC_ENEMY_ID: True}


def test_orc_objective_ends_episode_with_dragon_sized_reward():
    env = _make_orc_duel_env()
    env.reward_all_enemies = 9.0
    env.reset(seed=123)
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = ORC_ENEMY_ID
    env.battle_env = _DummyBattleEnv()

    _, reward, terminated, truncated, info = env.step(0)

    expected_reward = (
        9.0
        * (1.0 + env.FINAL_OBJECTIVE_CITY_REWARD_MULTIPLIER)
        * env.reward_green_dragon_objective_multiplier
    )
    assert terminated is True
    assert truncated is False
    assert info.get("campaign_result") == "victory"
    assert info.get("campaign_victory_reason") == "orc_defeated"
    assert info.get("campaign_objective") == "orc"
    assert reward == expected_reward
    assert info.get("final_objective_reward") == expected_reward
    assert info.get("green_dragon_objective_reward") == expected_reward
    assert info.get("green_dragon_objective_base_reward") == pytest.approx(
        9.0 * (1.0 + env.FINAL_OBJECTIVE_CITY_REWARD_MULTIPLIER)
    )
    assert info.get("green_dragon_objective_enemy_id") == ORC_ENEMY_ID


def test_orc_does_not_fully_regenerate_like_the_dragon():
    env = _make_orc_duel_env()
    env.reset(seed=123)
    orc_unit = next(
        unit
        for unit in env.enemy_team_states[ORC_ENEMY_ID]
        if float(unit.get("max_health", 0) or 0) > 0.0
    )
    max_hp = float(orc_unit["max_health"])

    orc_unit["health"] = max_hp - 100.0
    orc_unit["hp"] = max_hp - 100.0
    env._advance_turns(1)

    # Обычная регенерация (доля от max HP), но не мгновенное полное лечение.
    healed_hp = float(orc_unit["health"])
    assert healed_hp > max_hp - 100.0
    assert healed_hp < max_hp


def test_invalid_objective_map_combinations_raise():
    with pytest.raises(ValueError):
        CampaignEnv(map_name="orc_duel", campaign_objective="cities", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(map_name="orc_duel", campaign_objective="dragon", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(map_name="orc_duel", campaign_objective="blue_dragon", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(campaign_objective="orc", Realcapital=2)
    with pytest.raises(ValueError):
        CampaignEnv(map_name="no_such_map", Realcapital=2)


def test_default_map_objective_constants_unchanged():
    env = CampaignEnv(Realcapital=2, log_enabled=False)
    assert env.map_name == "default"
    assert env.campaign_objective == "cities"
    assert env.OBJECTIVE_ENEMY_ID == 31
    assert len(env.FINAL_OBJECTIVE_CITIES) == 2
    assert set(env.RUIN_REWARD_BY_ENEMY_ID) == {70, 71, 72, 73, 74}


def test_default_map_ruin_reward_item_names_match_constants():
    # Имена предметов в maps/default.py должны совпадать с константами
    # CampaignConstantsMixin (защита от дрейфа при правке одного из мест).
    ruin_rewards = get_map("default").ruin_rewards
    expected_by_enemy_id = {
        70: CampaignConstantsMixin.TITAN_ELIXIR_ITEM_NAME,
        71: CampaignConstantsMixin.RUIN_INVULNERABILITY_ELIXIR_ITEM_NAME,
        72: CampaignConstantsMixin.RUIN_LIFE_ELIXIR_ITEM_NAME,
        73: CampaignConstantsMixin.SUPREME_ELIXIR_ITEM_NAME,
        74: CampaignConstantsMixin.RING_OF_AGES_ITEM_NAME,
    }
    for enemy_id, expected_item in expected_by_enemy_id.items():
        assert ruin_rewards[enemy_id]["item"] == expected_item


def test_orc_duel_random_masked_rollout_is_stable():
    env = _make_orc_duel_env(freeze_dynamic_action_layout=True)
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
