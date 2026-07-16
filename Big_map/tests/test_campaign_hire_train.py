import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import FEATURES_PER_UNIT
from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.hire_train import (
    TARGET_ARMY_ENEMY_ID,
    TARGET_ARMY_TILE,
    TRAINING_CITY_GARRISON_ENEMY_ID,
    TRAINING_CITY_NAME,
    TRAINING_CITY_TILE,
)


class _DummyBattleEnv:
    def __init__(self, blue_team):
        self.winner = "blue"
        self.action_space = type("ActionSpace", (), {"n": 15})()
        self.last_battle_exp = 0.0
        self.last_levelups = []
        self.combined = deepcopy(blue_team)

    def _obs(self):
        return np.zeros(FEATURES_PER_UNIT * 12, dtype=np.float32)


def _make_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="hire_train",
        Realcapital=2,
        log_enabled=False,
        persist_blue_hp=False,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_names(env: CampaignEnv, enemy_id: int) -> list[str]:
    return [
        str(unit["name"])
        for unit in env._enemy_configs[enemy_id]
        if not env._is_empty_blue_unit(unit)
    ]


def _finish_battle(env: CampaignEnv, enemy_id: int):
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = enemy_id
    env.battle_env = _DummyBattleEnv(env.blue_team_state)
    return env.step(0)


def test_hire_train_map_is_registered_with_capital_city_and_far_target():
    assert "hire_train" in available_maps()
    map_config = get_map("hire_train")

    assert map_config.play_grid_size == 24
    assert map_config.default_objective == "orc"
    assert map_config.objective_enemy_id == TARGET_ARMY_ENEMY_ID
    assert map_config.enemy_positions() == {
        TARGET_ARMY_ENEMY_ID: TARGET_ARMY_TILE,
        TRAINING_CITY_GARRISON_ENEMY_ID: TRAINING_CITY_TILE,
    }
    assert map_config.starting_roster == {8: "Герцог"}
    assert map_config.starting_gold == 300.0

    env = _make_env()
    env.reset(seed=123)

    assert env.grid_size == 24
    assert env.CASTLE_POS == (2, 13)
    assert env.grid_env.agent_pos == env.CASTLE_POS
    assert tuple(env.castle_heal_tiles) == ((2, 13), (12, 6))
    assert dict(env.grid_env.enemy_positions) == {
        TARGET_ARMY_ENEMY_ID: (23, 23),
        TRAINING_CITY_GARRISON_ENEMY_ID: (12, 6),
    }
    assert env.gold == 300.0

    env.gold = 0.0
    env.reset(seed=124)
    assert env.gold == 300.0


def test_hire_train_starts_with_only_duke_and_all_hire_points_free():
    env = _make_env(use_boss_starting_roster=True, typeoflord=3)
    env.reset(seed=123)

    living_blue = [
        unit for unit in env.blue_team_state if not env._is_empty_blue_unit(unit)
    ]
    assert [(unit["position"], unit["name"]) for unit in living_blue] == [(8, "Герцог")]

    hero = living_blue[0]
    assert hero["hero"] is True
    assert hero["leadership_capacity"] == 3
    assert hero["leadership_points"] == 3
    assert hero["needaunit"] == 1

    env.gold = 999.0
    hire_mask = env._faction_hire_action_mask_values()
    big_idx = next(
        idx
        for idx, option in enumerate(env.active_hire_options)
        if env._is_big_hire_option(option)
    )
    assert any(hire_mask[:big_idx])
    assert hire_mask[big_idx] is True


def test_hire_train_city_has_one_archer_and_is_not_the_final_objective():
    env = _make_env()
    env.reset(seed=123)

    assert _living_names(env, TRAINING_CITY_GARRISON_ENEMY_ID) == ["Гоблин лучник"]
    settlement = env.legions_settlement_territory_source_by_name[TRAINING_CITY_NAME]
    assert settlement["source_tile"] == (12, 6)
    assert settlement["required_enemy_ids"] == (TRAINING_CITY_GARRISON_ENEMY_ID,)

    _, _, terminated, truncated, info = _finish_battle(
        env,
        TRAINING_CITY_GARRISON_ENEMY_ID,
    )

    assert terminated is False
    assert truncated is False
    assert info["battle_result"] == "victory"
    assert info["legions_settlement_territories_activated"] == [TRAINING_CITY_NAME]
    assert env.grid_env.enemies_alive[TARGET_ARMY_ENEMY_ID] is True


def test_hire_train_target_has_reinforced_goblins_and_ends_campaign():
    env = _make_env(
        reward_defeat_enemy=0.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
        reward_needaunit_turn_penalty=0.0,
    )
    env.reset(seed=123)

    target_units = [
        unit
        for unit in env._enemy_configs[TARGET_ARMY_ENEMY_ID]
        if not env._is_empty_blue_unit(unit)
    ]
    assert [(unit["position"], unit["name"]) for unit in target_units] == [
        (1, "Орк"),
        (2, "Гоблин"),
        (3, "Гоблин"),
        (5, "Гоблин лучник"),
    ]

    _, _, terminated, truncated, info = _finish_battle(env, TARGET_ARMY_ENEMY_ID)

    assert terminated is True
    assert truncated is False
    assert info["campaign_result"] == "victory"
    assert info["campaign_victory_reason"] == "orc_defeated"
    assert info["campaign_objective"] == "orc"
