import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import FEATURES_PER_UNIT
from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.siege_train import (
    HERO_START,
    SIEGE_ENEMY_ID,
    SIEGE_MOVES_PER_TURN,
    SIEGE_SPAWN_TILE,
    SIEGE_SPAWN_TURN,
    STARTING_ELIXIR_ITEMS,
    STATIC_ENEMY_STACKS,
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
        map_name="siege_train",
        log_enabled=False,
        persist_blue_hp=False,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_units(units) -> list[tuple[int, str]]:
    return [
        (int(unit["position"]), str(unit["name"]))
        for unit in units
        if str(unit.get("name", "")) != "пусто"
    ]


def _finish_battle(env: CampaignEnv, enemy_id: int):
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = enemy_id
    env.battle_env = _DummyBattleEnv(env.blue_team_state)
    return env.step(0)


def test_siege_train_is_registered_with_twenty_five_static_stacks_and_one_pursuer():
    assert "siege_train" in available_maps()
    map_config = get_map("siege_train")

    assert map_config.grid_size == 48
    assert map_config.hero_start == HERO_START
    assert map_config.default_objective == "target_enemy"
    assert map_config.objective_enemy_id == SIEGE_ENEMY_ID
    assert len(STATIC_ENEMY_STACKS) == 25
    assert len(map_config.enemy_stacks) == 26
    assert map_config.scheduled_enemy_id == SIEGE_ENEMY_ID
    assert map_config.scheduled_enemy_spawn_turn == SIEGE_SPAWN_TURN == 25
    assert map_config.scheduled_enemy_moves_per_turn == SIEGE_MOVES_PER_TURN == 25
    assert map_config.enemy_positions()[SIEGE_ENEMY_ID] == SIEGE_SPAWN_TILE
    assert map_config.obstacle_blocks == ()
    assert map_config.chests == ()
    assert map_config.mana_sources == ()


def test_siege_train_starts_with_standard_legions_warrior_party():
    env = _make_env(
        Realcapital=1,
        typeoflord=3,
        use_boss_starting_roster=True,
    )
    env.reset(seed=123)

    assert env.Realcapital == 2
    assert env.typeoflord == 1
    assert env.grid_env.agent_pos == HERO_START
    assert _living_units(env.blue_team_state) == [
        (7, "Одержимый"),
        (8, "Герцог"),
        (9, "Одержимый"),
        (11, "Сектант"),
    ]
    assert sum(bool(alive) for alive in env.grid_env.enemies_alive.values()) == 25
    assert env.grid_env.enemies_alive[SIEGE_ENEMY_ID] is False
    assert env.scheduled_enemy_spawned is False


def test_siege_train_starts_with_two_of_each_supported_elixir():
    map_config = get_map("siege_train")
    env = _make_env()
    env.reset(seed=123)

    assert len(STARTING_ELIXIR_ITEMS) == 10
    assert Counter(map_config.starting_hero_items) == Counter(
        {item_name: 2 for item_name in STARTING_ELIXIR_ITEMS}
    )

    action_starts = set()
    for item_name in STARTING_ELIXIR_ITEMS:
        canonical_name = env._canonical_potion_item_name(item_name)
        assert canonical_name
        assert canonical_name in env.scenario_potion_item_names
        assert env._count_hero_item(canonical_name) == 2
        assert env._potion_available_count(canonical_name) == 2
        action_starts.add(env._potion_action_start_for_item(canonical_name))

    assert len(action_starts) == len(STARTING_ELIXIR_ITEMS)


def test_five_added_siege_patrols_have_real_formations_with_ranged_support():
    env = _make_env()
    env.reset(seed=123)

    expected_units = {
        21: Counter({"Крестьянин": 2, "Гоблин лучник": 1}),
        22: Counter({"Гоблин": 1, "Бес": 1, "Гоблин лучник": 1}),
        23: Counter({"Разбойник": 2, "Гоблин лучник": 1}),
        24: Counter(
            {
                "Разбойник": 1,
                "Ополченец": 1,
                "Гоблин лучник": 1,
            }
        ),
        25: Counter({"Крестьянин": 1, "Ополченец": 1, "Адепт": 1}),
    }

    for enemy_id, expected in expected_units.items():
        living_units = [
            unit
            for unit in env._enemy_configs[enemy_id]
            if str(unit.get("name", "")) != "пусто"
        ]
        assert Counter(str(unit["name"]) for unit in living_units) == expected
        assert len(living_units) >= 3
        assert any(int(unit["position"]) in (4, 5, 6) for unit in living_units)


def test_static_siege_stacks_get_stronger_as_distance_increases():
    env = _make_env()
    env.reset(seed=123)

    rows = []
    for stack in STATIC_ENEMY_STACKS:
        enemy_id = int(stack["enemy_id"])
        x, y = tuple(stack["position"])
        distance = max(abs(int(x) - HERO_START[0]), abs(int(y) - HERO_START[1]))
        strength = sum(
            int(unit.get("Level", 0) or 0)
            for unit in env._enemy_configs[enemy_id]
            if str(unit.get("name", "")) != "пусто"
        )
        rows.append((distance, strength))

    assert [distance for distance, _ in rows] == sorted(
        distance for distance, _ in rows
    )
    assert [strength for _, strength in rows] == sorted(
        strength for _, strength in rows
    )


def test_turn_25_spawns_pursuer_and_turn_26_contact_starts_standard_battle():
    env = _make_env(
        reward_turn_penalty=0.0,
        reward_needaunit_turn_penalty=0.0,
    )
    env.reset(seed=123)
    env.turns = 24

    _, _, terminated, truncated, spawn_info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_GRID
    assert env.turns == 25
    assert env.scheduled_enemy_spawned is True
    assert env.grid_env.enemies_alive[SIEGE_ENEMY_ID] is True
    assert env.grid_env.enemy_positions[SIEGE_ENEMY_ID] == (22, 27)
    spawn_turn_info = spawn_info["scheduled_enemy_turn_infos"][-1]
    assert spawn_turn_info["spawned_this_turn"] is True
    assert spawn_turn_info["steps_moved"] == 25
    assert tuple(spawn_turn_info["moved_path"][0]) == (46, 46)
    assert tuple(spawn_turn_info["moved_path"][-1]) == (22, 27)
    assert spawn_turn_info["contact_with_agent"] is False

    _, _, terminated, truncated, battle_info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert env.turns == 26
    assert env.mode == env.MODE_BATTLE
    assert env.current_enemy_id == SIEGE_ENEMY_ID
    assert env.grid_env.enemy_positions[SIEGE_ENEMY_ID] == HERO_START
    assert battle_info["battle_triggered"] is True
    assert battle_info["enemy_id"] == SIEGE_ENEMY_ID
    contact_info = battle_info["scheduled_enemy_turn_infos"][-1]
    assert contact_info["steps_moved"] == 17
    assert contact_info["contact_with_agent"] is True

    red_units = [
        (
            int(unit["position"]),
            str(unit["name"]),
            int(unit.get("Level", 0) or 0),
        )
        for unit in env.battle_env.combined
        if unit.get("team") == "red" and str(unit.get("name", "")) != "пусто"
    ]
    assert red_units == [
        (1, "Имперский рыцарь", 3),
        (2, "Рыцарь на пегасе", 3),
        (4, "Рейнджер людей", 2),
        (5, "Белый маг", 4),
    ]


def test_clearing_twenty_five_static_stacks_does_not_finish_siege_objective():
    env = _make_env(
        reward_defeat_enemy=0.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
        reward_needaunit_turn_penalty=0.0,
        battle_reward_scale=0.0,
    )
    env.reset(seed=123)
    static_enemy_ids = [int(stack["enemy_id"]) for stack in STATIC_ENEMY_STACKS]
    for enemy_id in static_enemy_ids[:-1]:
        env.grid_env.enemies_alive[enemy_id] = False

    _, _, terminated, truncated, info = _finish_battle(env, static_enemy_ids[-1])

    assert terminated is False
    assert truncated is False
    assert info["battle_result"] == "victory"
    assert env.grid_env.enemies_alive[SIEGE_ENEMY_ID] is False
    assert env.campaign_objective == "target_enemy"


def test_defeating_siege_army_ends_campaign_with_large_target_reward():
    env = _make_env(
        reward_defeat_enemy=4.0,
        reward_all_enemies=12.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
        reward_needaunit_turn_penalty=0.0,
        battle_reward_scale=0.0,
    )
    env.reset(seed=123)
    env.grid_env.enemies_alive[SIEGE_ENEMY_ID] = True

    _, reward, terminated, truncated, info = _finish_battle(env, SIEGE_ENEMY_ID)

    assert reward == pytest.approx(16.0)
    assert terminated is True
    assert truncated is False
    assert info["enemy_defeat_reward"] == pytest.approx(4.0)
    assert info["target_enemy_objective_reward"] == pytest.approx(12.0)
    assert info["campaign_result"] == "victory"
    assert info["campaign_victory_reason"] == "target_enemy_defeated"
    assert info["campaign_objective"] == "target_enemy"
