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
from maps.super_last_stand import (
    CAPITAL_GOLD_MINE_TILE,
    CAPITAL_INFERNAL_MANA_TILE,
    CENTER_INVULNERABILITY_CHEST,
    DWARF_WAVE_ENEMY_ID,
    ELVEN_RUIN_BANNER_ITEM,
    ELVEN_RUIN_ENEMY_ID,
    ELVEN_RUIN_GOLD,
    ELVEN_RUIN_MARKER_TILE,
    ELVEN_RUIN_ORB_ITEM,
    ELVEN_RUIN_TILE,
    ELF_WAVE_ENEMY_ID,
    GOLD_MINE_TILES,
    HERO_START,
    HUMAN_WAVE_ENEMY_ID,
    LAKE_CHEST_TILE,
    LAKE_ENEMY_ID,
    LEFT_CITY_ENEMY_ID,
    LEFT_CITY_TILE,
    LEFT_DEATH_MANA_TILE,
    LEFT_GOLD_MINE_TILE,
    MERCHANT_ITEMS,
    MERCHANT_NAME,
    OCCULT_MASTER_ENEMY_ID,
    ORC_ENEMY_ID,
    RIGHT_CITY_ENEMY_ID,
    RIGHT_CITY_TILE,
    RIGHT_GOLD_MINE_TILE,
    RIGHT_LIFE_MANA_TILE,
    RUIN_ENEMY_ID,
    SPELL_SHOP_NAME,
    STARTING_GOLD,
    WATER_TILES,
    WAVE_DEFINITIONS,
    WAVE_MOVES_PER_TURN,
    wave_enemy_id,
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
        map_name="super_last_stand",
        log_enabled=False,
        persist_blue_hp=False,
        reward_needaunit_turn_penalty=0.0,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_units(units) -> list[dict]:
    return [
        unit
        for unit in units
        if str(unit.get("name", "")) != "пусто"
    ]


def _finish_battle(env: CampaignEnv, enemy_id: int):
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = int(enemy_id)
    env.battle_env = _DummyBattleEnv(env.blue_team_state)
    return env.step(0)


def _enemy_names(env: CampaignEnv, enemy_id: int) -> Counter:
    return Counter(
        str(unit["name"])
        for unit in env._enemy_configs[int(enemy_id)]
        if str(unit.get("name", "")) != "пусто"
    )


def test_super_last_stand_is_registered_and_starts_with_bethrezen_mage_only():
    assert "super_last_stand" in available_maps()
    map_config = get_map("super_last_stand")

    assert map_config.grid_size == 48
    assert map_config.hero_start == HERO_START
    assert map_config.default_objective == "waves"
    assert len(map_config.scheduled_enemy_waves) == 11
    assert map_config.starting_gold == STARTING_GOLD == 3000.0
    assert map_config.starting_roster == {11: "Утер"}

    env = _make_env(Realcapital=1, typeoflord=1, use_boss_starting_roster=True)
    obs, info = env.reset(seed=123)

    assert env.Realcapital == 2
    assert env.typeoflord == 2
    assert env.campaign_objective == "waves"
    assert env.grid_env.agent_pos == HERO_START
    assert env.gold == pytest.approx(STARTING_GOLD)
    assert obs.shape == env.observation_space.shape
    assert env.grid_wave_obs_size == 5

    living = _living_units(env.blue_team_state)
    assert [
        (int(unit["position"]), str(unit["name"]))
        for unit in living
    ] == [(11, "Бетрезен")]
    hero = living[0]
    assert hero["hero"] is True
    assert hero["unit_type"] == "Mage"
    assert hero["exp_current"] == 0
    assert hero["exp_required"] == 150
    assert hero["next_level_exp"] == 500
    assert hero["dynamic_level_exp_increment"] == 500
    assert env._travel_hero_is_mage() is True
    assert env.scroll_magic_unlocked is True
    assert env._hero_has_sorcery_lore() is True
    assert env._hero_has_artifact_knowledge() is True
    assert env._find_unit_data_by_name("Бетрезен") is not None

    assert info["scheduled_enemy_wave_count"] == 11
    assert info["scheduled_enemy_waves_spawned"] == 0
    assert info["scheduled_enemy_waves_defeated"] == 0
    assert info["scheduled_enemy_stack_count"] == 33
    assert info["scheduled_enemy_stacks_spawned"] == 0
    assert info["scheduled_enemy_stacks_defeated"] == 0
    assert all(
        env.grid_env.enemies_alive[enemy_id] is False
        for wave in map_config.scheduled_enemy_waves
        for enemy_id in wave["enemy_ids"]
    )


def test_super_last_stand_resources_and_water_islands_are_configured():
    map_config = get_map("super_last_stand")
    water_tiles = set(WATER_TILES)

    assert set(map_config.gold_mine_tiles_provider()) == set(GOLD_MINE_TILES)
    assert CAPITAL_GOLD_MINE_TILE in GOLD_MINE_TILES
    assert LEFT_GOLD_MINE_TILE in GOLD_MINE_TILES
    assert RIGHT_GOLD_MINE_TILE in GOLD_MINE_TILES
    assert set(map_config.mana_sources) == {
        ("infernal", CAPITAL_INFERNAL_MANA_TILE),
        ("death", LEFT_DEATH_MANA_TILE),
        ("life", RIGHT_LIFE_MANA_TILE),
    }

    for x in range(1, 9):
        assert (x, 18) in water_tiles
        assert (x, 30) in water_tiles
    for y in range(18, 31):
        assert (1, y) in water_tiles
        assert (8, y) in water_tiles

    for x in range(39, 47):
        assert (x, 18) in water_tiles
        assert (x, 30) in water_tiles
    for y in range(18, 31):
        assert (39, y) in water_tiles
        assert (46, y) in water_tiles

    assert LEFT_CITY_TILE not in water_tiles
    assert RIGHT_CITY_TILE not in water_tiles
    assert LEFT_GOLD_MINE_TILE not in water_tiles
    assert RIGHT_GOLD_MINE_TILE not in water_tiles


def test_capital_territory_cannot_cross_water_but_captured_cities_claim_resources():
    env = _make_env()
    env.reset(seed=123)

    env.turns = 50
    env._refresh_faction_territories()
    assert CAPITAL_GOLD_MINE_TILE in env.legions_territory_tile_set
    assert CAPITAL_INFERNAL_MANA_TILE in env.legions_territory_tile_set
    assert LEFT_GOLD_MINE_TILE not in env.legions_territory_tile_set
    assert LEFT_DEATH_MANA_TILE not in env.legions_territory_tile_set
    assert RIGHT_GOLD_MINE_TILE not in env.legions_territory_tile_set
    assert RIGHT_LIFE_MANA_TILE not in env.legions_territory_tile_set

    env.grid_env.mark_enemy_defeated(LEFT_CITY_ENEMY_ID)
    env.grid_env.mark_enemy_defeated(RIGHT_CITY_ENEMY_ID)
    assert env._activate_legions_settlement_territory_if_cleared(
        LEFT_CITY_ENEMY_ID
    ) == ["Ноктурна"]
    assert env._activate_legions_settlement_territory_if_cleared(
        RIGHT_CITY_ENEMY_ID
    ) == ["Крепость Бурь"]

    env.turns += 10
    env._refresh_faction_territories()
    assert LEFT_GOLD_MINE_TILE in env.legions_territory_tile_set
    assert LEFT_DEATH_MANA_TILE in env.legions_territory_tile_set
    assert RIGHT_GOLD_MINE_TILE in env.legions_territory_tile_set
    assert RIGHT_LIFE_MANA_TILE in env.legions_territory_tile_set


def test_super_last_stand_shops_chests_and_inventory_are_available():
    map_config = get_map("super_last_stand")
    env = _make_env()
    env.reset(seed=123)

    assert tuple(map_config.merchant_buy_items or ()) == MERCHANT_ITEMS
    assert env.merchant_stocks[MERCHANT_NAME] == {
        str(item["name"]): int(item["stock"])
        for item in MERCHANT_ITEMS
    }
    assert env.merchant_stocks[MERCHANT_NAME]["Angel Orb"] == 1
    assert env.merchant_stocks[MERCHANT_NAME]["Zombie Orb"] == 1
    assert env.merchant_stocks[MERCHANT_NAME]["Зелье силы (+30%)"] == 3
    assert env.merchant_stocks[MERCHANT_NAME]["Эликсир Жизни"] == 5
    assert env.merchant_stocks[MERCHANT_NAME]["Эликсир восстановления"] == 5
    assert 'Свиток "Призывание II: Белиарх"' in env.scenario_scroll_item_names
    assert "Angel Orb" in env.scenario_battle_orb_item_names
    assert "Zombie Orb" in env.scenario_battle_orb_item_names
    assert ELVEN_RUIN_ORB_ITEM in env.scenario_battle_orb_item_names
    assert "Elder Vampire Talisman" in env.scenario_battle_magic_item_names
    assert ELVEN_RUIN_BANNER_ITEM in env.BANNER_ITEM_NAMES

    assert env.spell_shop_stocks[SPELL_SHOP_NAME] == {
        "Армагеддон": 1,
        "Вызов Мстителя": 1,
    }
    assert map_config.chests == (
        (CENTER_INVULNERABILITY_CHEST, ("Эликсир неуязвимости",)),
        (LAKE_CHEST_TILE, ("Imperial Crown (Valuable)",)),
    )


def test_static_garrisons_lake_ruins_and_single_stacks_match_the_scenario():
    env = _make_env()
    env.reset(seed=123)

    assert _enemy_names(env, ORC_ENEMY_ID) == Counter({"Орк": 1})
    assert _enemy_names(env, OCCULT_MASTER_ENEMY_ID) == Counter(
        {"Мастер оккультист": 1}
    )
    assert _enemy_names(env, LAKE_ENEMY_ID) == Counter(
        {"Морской змей": 1, "Кракен": 1}
    )
    assert _enemy_names(env, LEFT_CITY_ENEMY_ID) == Counter(
        {
            "Тёмный эльф мясник": 2,
            "Тёмный эльф жнец": 1,
            "Тёмный эльф маг": 1,
            "Тёмный эльф призрак": 1,
        }
    )
    assert _enemy_names(env, RIGHT_CITY_ENEMY_ID) == Counter(
        {"Варвар": 2, "Вождь варваров": 1, "Шаманка": 1}
    )
    assert _enemy_names(env, RUIN_ENEMY_ID) == Counter(
        {"Тролль": 1, "Людоед": 1, "Гоблин старейшина": 1}
    )
    assert _enemy_names(env, ELVEN_RUIN_ENEMY_ID) == Counter(
        {
            "Нейтральный грифон": 1,
            "Нейтральный эльфийский оракул": 1,
            "Нейтральный лорд эльфов": 1,
            "Нейтральный эльф-рейнджер": 1,
        }
    )
    assert env.SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID[LEFT_CITY_ENEMY_ID] == 2
    assert env.SETTLEMENT_DEFENDER_LEVEL_BY_ENEMY_ID[RIGHT_CITY_ENEMY_ID] == 3


def test_ruin_grants_two_artifacts_and_900_gold_once():
    env = _make_env()
    env.reset(seed=123)

    info = env._grant_ruin_reward(RUIN_ENEMY_ID)

    assert info["ruin_reward_applied"] is True
    assert info["ruin_reward_gold"] == pytest.approx(900.0)
    assert info["ruin_reward_items"] == [
        "Runestone (Artifact)",
        "Unholy Chalice (Artifact)",
    ]
    assert env.gold == pytest.approx(STARTING_GOLD + 900.0)
    owned_names = {
        env._hero_item_name(item)
        for item in env.heroitems
    } | {
        str(item_name)
        for item_name in env.equipped_artifact_items
        if item_name is not None
    }
    assert "Runestone (Artifact)" in owned_names
    assert "Unholy Chalice (Artifact)" in owned_names

    repeat = env._grant_ruin_reward(RUIN_ENEMY_ID)
    assert repeat["ruin_reward_applied"] is False
    assert env.gold == pytest.approx(STARTING_GOLD + 900.0)


def test_elven_ruin_grants_banner_witch_orb_and_gold_once():
    env = _make_env()
    env.reset(seed=123)

    assert env.RUIN_REWARD_BY_ENEMY_ID[ELVEN_RUIN_ENEMY_ID] == {
        "title": "Забытые эльфийские руины",
        "ruin_pos": ELVEN_RUIN_TILE,
        "marker_pos": ELVEN_RUIN_MARKER_TILE,
        "items": (
            ELVEN_RUIN_BANNER_ITEM,
            ELVEN_RUIN_ORB_ITEM,
        ),
        "gold": ELVEN_RUIN_GOLD,
    }

    info = env._grant_ruin_reward(ELVEN_RUIN_ENEMY_ID)

    assert info["ruin_reward_applied"] is True
    assert info["ruin_reward_gold"] == pytest.approx(ELVEN_RUIN_GOLD)
    assert info["ruin_reward_items"] == [
        ELVEN_RUIN_BANNER_ITEM,
        ELVEN_RUIN_ORB_ITEM,
    ]
    assert env.gold == pytest.approx(STARTING_GOLD + ELVEN_RUIN_GOLD)
    owned_names = {
        env._hero_item_name(item)
        for item in env.heroitems
    } | {
        str(item_name)
        for item_name in env.equipped_banner_items
        if item_name is not None
    }
    assert ELVEN_RUIN_BANNER_ITEM in owned_names
    assert ELVEN_RUIN_ORB_ITEM in owned_names

    repeat = env._grant_ruin_reward(ELVEN_RUIN_ENEMY_ID)
    assert repeat["ruin_reward_applied"] is False
    assert env.gold == pytest.approx(STARTING_GOLD + ELVEN_RUIN_GOLD)


def test_wave_formations_and_schedule_are_exact():
    map_config = get_map("super_last_stand")
    env = _make_env()
    env.reset(seed=123)

    assert tuple(
        (
            int(wave["wave_index"]),
            tuple(int(enemy_id) for enemy_id in wave["enemy_ids"]),
            int(wave["spawn_turn"]),
            int(wave["spawn_interval"]),
            int(wave["moves_per_turn"]),
        )
        for wave in map_config.scheduled_enemy_waves
    ) == tuple(
        (
            wave_index,
            tuple(wave_enemy_id(wave_index, race_index) for race_index in range(3)),
            (wave_index - 1) * 15 + 5,
            5,
            WAVE_MOVES_PER_TURN,
        )
        for wave_index in range(1, 12)
    )
    assert len(env._scheduled_enemy_wave_configs()) == 33
    assert tuple(
        int(config["spawn_turn"])
        for config in env._scheduled_enemy_wave_configs()
    ) == tuple(range(5, 166, 5))

    for wave_index, (_, _, armies) in enumerate(WAVE_DEFINITIONS, start=1):
        for race_index, (front_names, back_names) in enumerate(armies):
            assert _enemy_names(
                env,
                wave_enemy_id(wave_index, race_index),
            ) == Counter((*front_names, *back_names))

    final_humans = _enemy_names(env, wave_enemy_id(11, 0))
    final_elves = _enemy_names(env, wave_enemy_id(11, 1))
    final_dwarves = _enemy_names(env, wave_enemy_id(11, 2))
    assert "Мизраэль" not in final_humans
    assert "Сэр Аллемон" in final_humans
    assert "Иллюмиэлль" in final_elves
    assert "Видар" not in final_dwarves
    assert "Маг хугин" in final_dwarves

    expected_boss_stats = {
        "Сэр Аллемон": (800, 300),
        "Иллюмиэлль": (900, 250),
        "Маг хугин": (900, 70),
    }
    for boss_name, (expected_health, expected_damage) in expected_boss_stats.items():
        boss = env._find_unit_data_by_name(boss_name)
        assert boss is not None
        assert int(boss["здоровье"]) == expected_health
        assert int(boss["урон"]) == expected_damage


def test_first_wave_reaches_stationary_bethrezen_in_two_turns():
    env = _make_env(
        reward_turn_penalty=0.0,
        reward_needaunit_turn_penalty=0.0,
    )
    env.reset(seed=123)
    env.turns = 4

    _, _, terminated, truncated, spawn_info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_GRID
    human_info = next(
        item
        for item in spawn_info["scheduled_enemy_turn_infos"]
        if int(item["enemy_id"]) == HUMAN_WAVE_ENEMY_ID
    )
    spawned_ids = {
        int(item["enemy_id"])
        for item in spawn_info["scheduled_enemy_turn_infos"]
        if bool(item["spawned_this_turn"])
    }
    assert spawned_ids == {HUMAN_WAVE_ENEMY_ID}
    assert human_info["spawned_this_turn"] is True
    assert human_info["steps_moved"] == 30
    assert human_info["contact_with_agent"] is False
    assert tuple(human_info["target"]) == HERO_START

    _, _, terminated, truncated, contact_info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_BATTLE
    assert env.current_enemy_id == HUMAN_WAVE_ENEMY_ID
    human_contact = next(
        item
        for item in contact_info["scheduled_enemy_turn_infos"]
        if int(item["enemy_id"]) == HUMAN_WAVE_ENEMY_ID
    )
    assert human_contact["steps_moved"] == 12
    assert human_contact["contact_with_agent"] is True
    assert env.scheduled_enemy_pending_encounter_ids == []


def test_wave_fast_path_retargets_current_agent_position_each_turn():
    env = _make_env(reward_turn_penalty=0.0)
    env.reset(seed=123)
    env.grid_env.agent_pos = (10, 40)
    env.turns = 4

    env.step(8)
    first_pos = env.grid_env.enemy_positions[HUMAN_WAVE_ENEMY_ID]
    assert first_pos == (10, 30)

    env.grid_env.agent_pos = (18, 40)
    _, _, _, _, info = env.step(8)
    human_info = next(
        item
        for item in info["scheduled_enemy_turn_infos"]
        if int(item["enemy_id"]) == HUMAN_WAVE_ENEMY_ID
    )
    assert tuple(human_info["target"]) == (18, 40)
    assert human_info["contact_with_agent"] is True
    assert env.current_enemy_id == HUMAN_WAVE_ENEMY_ID


def test_wave_rewards_escalate_after_each_three_army_wave_and_finish_after_wave_11():
    env = _make_env(
        reward_defeat_enemy=4.0,
        reward_all_enemies=12.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
        battle_reward_scale=0.0,
    )
    env.reset(seed=123)

    for wave_index in range(1, 12):
        for race_index in range(3):
            enemy_id = wave_enemy_id(wave_index, race_index)
            is_wave_complete = race_index == 2
            is_campaign_complete = wave_index == 11 and is_wave_complete
            expected_bonus = 4.0 * wave_index if is_wave_complete else 0.0
            expected_reward = 4.0 + expected_bonus
            if is_campaign_complete:
                expected_reward += 12.0

            env.scheduled_enemy_spawned_ids.add(enemy_id)
            env.grid_env.enemies_alive[enemy_id] = True
            _, reward, terminated, truncated, info = _finish_battle(env, enemy_id)

            assert reward == pytest.approx(expected_reward)
            assert terminated is is_campaign_complete
            assert truncated is False
            assert info["wave_index"] == wave_index
            assert info["wave_completed"] is is_wave_complete
            assert info["wave_defeat_bonus"] == pytest.approx(expected_bonus)
            assert info["waves_total"] == 11
            assert info["wave_stacks_total"] == 33

    assert info["waves_final_objective_reward"] == pytest.approx(12.0)
    assert info["campaign_result"] == "victory"
    assert info["campaign_victory_reason"] == "all_waves_defeated"
    assert any(
        env.grid_env.enemies_alive[enemy_id]
        for enemy_id in (
            ORC_ENEMY_ID,
            OCCULT_MASTER_ENEMY_ID,
            LAKE_ENEMY_ID,
            LEFT_CITY_ENEMY_ID,
            RIGHT_CITY_ENEMY_ID,
            RUIN_ENEMY_ID,
            ELVEN_RUIN_ENEMY_ID,
        )
    )
