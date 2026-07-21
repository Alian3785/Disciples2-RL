import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.green_dragon_minimal import (
    BLUE_DRAGON_NAME,
    DEATH_MANA_TILE,
    FOREST_TILES,
    GOLD_MINE_TILE,
    GREEN_DRAGON_ENEMY_ID,
    GREEN_DRAGON_NAME,
    HERO_START,
    KRAKEN_TILE,
    LEVEL_THREE_CITY_ENEMY_ID,
    LEVEL_THREE_CITY_TILE,
    LEVEL_TWO_CITY_ENEMY_ID,
    LEVEL_TWO_CITY_TILE,
    LIFE_MANA_TILE,
    MERCHANT_ITEMS,
    MERCHANT_NAME,
    PLAY_GRID_SIZE,
    RUIN_ENEMY_ID,
    RUIN_MARKER_TILE,
    RUIN_TILE,
    WATER_TILES,
)


def _make_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="green_dragon_minimal",
        log_enabled=False,
        persist_blue_hp=False,
        reward_turn_penalty=0.0,
        reward_needaunit_turn_penalty=0.0,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_names(env: CampaignEnv, enemy_id: int) -> Counter:
    return Counter(
        str(unit["name"])
        for unit in env._enemy_configs[int(enemy_id)]
        if str(unit.get("name", "")) != "пусто"
    )


def _living_state_names(env: CampaignEnv, enemy_id: int) -> Counter:
    return Counter(
        str(unit["name"])
        for unit in env.enemy_team_states[int(enemy_id)]
        if str(unit.get("name", "")) != "пусто"
    )


def _adjacent(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1])) == 1


def test_green_dragon_minimal_is_registered_and_starts_with_requested_legions_party():
    assert "green_dragon_minimal" in available_maps()
    map_config = get_map("green_dragon_minimal")

    assert map_config.grid_size == 48
    assert map_config.play_grid_size == PLAY_GRID_SIZE == 32
    assert map_config.hero_start == HERO_START
    assert map_config.default_objective == "dragon"
    assert map_config.objective_enemy_id == GREEN_DRAGON_ENEMY_ID == 31
    assert len(map_config.enemy_stacks) == 24
    assert map_config.starting_capital_id == 2
    assert map_config.starting_lord_type == 1
    assert map_config.starting_roster == {
        7: "Одержимый",
        8: "Герцог",
        9: "Одержимый",
        11: "Сектант",
    }

    env = _make_env(Realcapital=1, typeoflord=2, use_boss_starting_roster=True)
    obs, _ = env.reset(seed=123)

    assert env.grid_size == 32
    assert env.grid_env.agent_pos == (3, 16)
    assert env.campaign_objective == "dragon"
    assert env.Realcapital == 2
    assert env.typeoflord == 1
    assert obs.shape == env.observation_space.shape
    assert [
        (int(unit["position"]), str(unit["name"]))
        for unit in env.blue_team_state
        if str(unit.get("name", "")) != "пусто"
    ] == [
        (7, "Одержимый"),
        (8, "Герцог"),
        (9, "Одержимый"),
        (11, "Сектант"),
    ]


def test_requested_roaming_armies_have_exact_compositions_and_formations():
    env = _make_env()
    env.reset(seed=123)

    expected = {
        1: {"Орк": 1},
        2: {"Гоблин": 2, "Гоблин лучник": 1},
        3: {"Ополченец": 2, "Лучник": 1},
        4: {"Копейщик": 1, "Служка": 1},
        5: {"Скваер": 3},
        6: {"Демон": 1, "Одержимый": 1},
        7: {"Скелет": 1, "Призрак": 1},
        8: {"Крестьянин": 2, "Инкуб": 1},
        9: {"Одержимый": 2, "Колдунья": 1},
        10: {"Оккультист": 1},
        11: {"Кракен": 1},
        12: {"Рыцарь смерти": 1, "Зомби": 1},
        13: {"Орк": 2},
        14: {"Тёмный эльф мясник": 2, "Призрак": 1},
        15: {"Гном воин": 2, "Алхимик": 1},
        16: {"Берсерк": 2, "Сектант": 1},
        17: {"Оборотень": 1, "Зомби": 1},
        18: {"Кентавр дикарь": 1},
        19: {"Мантикора": 1},
        20: {"Тиамат": 1},
        GREEN_DRAGON_ENEMY_ID: {GREEN_DRAGON_NAME: 1},
    }
    for enemy_id, composition in expected.items():
        assert _living_names(env, enemy_id) == Counter(composition)

    specs = get_map("green_dragon_minimal").enemy_team_specs()
    assert specs[2]["front"] == ["Гоблин", None, "Гоблин"]
    assert specs[2]["back"] == [None, "Гоблин лучник", None]
    assert specs[3]["front"] == ["Ополченец", None, "Ополченец"]
    assert specs[3]["back"] == [None, "Лучник", None]
    assert specs[6]["front"] == [None, "Демон", None]
    assert specs[6]["back"] == ["Одержимый", None, None]


def test_enemy_strength_bands_move_away_from_the_capital_toward_the_dragon():
    env = _make_env()
    env.reset(seed=123)
    positions = env._static_enemy_positions
    hero_start = tuple(env.grid_env.start_position)

    def distance(enemy_id: int) -> int:
        tile = positions[enemy_id]
        return max(abs(tile[0] - hero_start[0]), abs(tile[1] - hero_start[1]))

    weak_band = (2, 3, 5, 4, 7, 10, 1)
    middle_band = (8, 17, 12, 9, 16, 18, 13, 6, 15)
    final_band = (11, 14, 19, 20, GREEN_DRAGON_ENEMY_ID)

    assert max(map(distance, weak_band)) < min(map(distance, middle_band))
    assert max(map(distance, middle_band)) < min(map(distance, final_band))
    assert distance(19) < distance(20) < distance(GREEN_DRAGON_ENEMY_ID)


def test_minimal_terrain_cities_resources_and_ruin_match_the_request():
    map_config = get_map("green_dragon_minimal")

    assert len(map_config.obstacle_blocks) == 5
    assert all(width == height == 1 for _, _, width, height in map_config.obstacle_blocks)
    assert set(map_config.water_tiles_provider()) == set(WATER_TILES)
    assert len(WATER_TILES) == 5
    assert KRAKEN_TILE in WATER_TILES

    assert len(FOREST_TILES) == 9
    assert set(map_config.forest_tiles_provider()) == set(FOREST_TILES)

    assert map_config.village_heal_tiles == (
        LEVEL_TWO_CITY_TILE,
        LEVEL_THREE_CITY_TILE,
    )
    assert dict(map_config.settlement_level_by_heal_tile) == {
        LEVEL_TWO_CITY_TILE: 2,
        LEVEL_THREE_CITY_TILE: 3,
    }
    assert _adjacent(LEVEL_TWO_CITY_TILE, LIFE_MANA_TILE)
    assert _adjacent(LEVEL_THREE_CITY_TILE, DEATH_MANA_TILE)
    assert _adjacent(LEVEL_THREE_CITY_TILE, GOLD_MINE_TILE)
    assert set(map_config.mana_sources) == {
        ("life", LIFE_MANA_TILE),
        ("death", DEATH_MANA_TILE),
    }
    assert map_config.gold_mine_tiles_provider() == (GOLD_MINE_TILE,)

    env = _make_env()
    env.reset(seed=123)
    kraken_play_tile = env._static_enemy_positions[11]
    centaur_play_tile = env._static_enemy_positions[18]
    expected_play_forest = {
        (x, y)
        for x in range(centaur_play_tile[0] - 1, centaur_play_tile[0] + 2)
        for y in range(centaur_play_tile[1] - 1, centaur_play_tile[1] + 2)
    }
    assert len(env.water_tiles) == 5
    assert kraken_play_tile in env.water_tiles
    assert set(env.forest_tiles) == expected_play_forest
    assert set(env._static_obstacle_tiles) == {
        (11, 16),
        (15, 18),
        (21, 13),
        (24, 20),
        (29, 13),
    }
    assert len(set(env._static_enemy_positions.values())) == len(
        env._static_enemy_positions
    )
    assert env.merchant_site_anchors[MERCHANT_NAME] == (9, 27)
    assert set(env.merchant_site_interaction_tiles[MERCHANT_NAME]) == {
        (8, 26),
        (9, 26),
        (10, 26),
        (8, 27),
        (10, 27),
    }
    assert _living_names(env, LEVEL_TWO_CITY_ENEMY_ID) == Counter(
        {"Молох": 1, "Сектант": 1}
    )
    assert _living_names(env, LEVEL_THREE_CITY_ENEMY_ID) == Counter(
        {"Имперский рыцарь": 1, "Патриарх": 1}
    )
    assert _living_names(env, RUIN_ENEMY_ID) == Counter({"Людоед": 1})

    ruin = map_config.ruin_rewards[RUIN_ENEMY_ID]
    assert ruin["ruin_pos"] == RUIN_TILE
    assert ruin["marker_pos"] == RUIN_MARKER_TILE
    assert ruin["item"] == "Emerald (Valuable)"
    assert env._hero_item_gold_value_by_name(str(ruin["item"])) == pytest.approx(750.0)


def test_merchant_and_four_requested_chests_are_available_in_the_environment():
    map_config = get_map("green_dragon_minimal")
    assert tuple(map_config.merchant_buy_items or ()) == MERCHANT_ITEMS
    assert map_config.chests == (
        ((7, 34), ("Эликсир инициативы (+10%)",)),
        ((18, 3), ("Зелье точности (+30%)",)),
        ((28, 27), ("Runestone (Artifact)",)),
        ((40, 18), ("Angel Orb",)),
    )

    env = _make_env()
    env.reset(seed=123)
    assert env.merchant_stocks[MERCHANT_NAME] == {
        "Бутыль лечения (+100 HP)": 7,
        "Целебная мазь": 7,
        "Зелье воскрешения (+1)": 7,
        "Эликсир неуязвимости": 2,
    }
    assert "Angel Orb" in env.scenario_battle_orb_item_names
    assert env.RUNESTONE_ARTIFACT_ITEM_NAME == "Runestone (Artifact)"


def test_dragon_reward_parameters_are_identical_to_default_dragon_mode():
    minimal = _make_env()
    default = CampaignEnv(
        map_name="default",
        campaign_objective="dragon",
        log_enabled=False,
        persist_blue_hp=False,
    )

    for attribute in (
        "reward_green_dragon_reached",
        "reward_green_dragon_objective_multiplier",
        "reward_dragon_damage_fraction",
        "reward_dragon_rounds_record",
        "reward_no_dragon_victory_late_step_penalty",
        "reward_no_dragon_victory_late_step_penalty_cap",
        "reward_no_dragon_victory_late_step_start_fraction",
    ):
        assert getattr(minimal, attribute) == pytest.approx(getattr(default, attribute))


def test_reduced_map_observation_is_measured_against_default():
    minimal = _make_env(campaign_objective="dragon")
    minimal_obs, _ = minimal.reset(seed=123)
    default = CampaignEnv(
        map_name="default",
        campaign_objective="dragon",
        log_enabled=False,
        persist_blue_hp=False,
    )
    default_obs, _ = default.reset(seed=123)

    assert minimal.grid_size == 32
    assert minimal.grid_legions_territory_obs_size == 32**2 == 1024
    assert minimal.GRID_OBS_SIZE == 3395
    assert minimal.BATTLE_OBS_SIZE == default.BATTLE_OBS_SIZE == 1464
    assert minimal_obs.shape == minimal.observation_space.shape == (4862,)
    assert default_obs.shape == default.observation_space.shape == (9864,)


@pytest.mark.parametrize(
    ("objective", "expected_dragon", "replaced_dragon"),
    (
        ("dragon", GREEN_DRAGON_NAME, BLUE_DRAGON_NAME),
        ("blue_dragon", BLUE_DRAGON_NAME, GREEN_DRAGON_NAME),
    ),
)
def test_green_and_blue_dragon_modes_replace_the_boss_and_restore_full_hp_each_turn(
    objective: str,
    expected_dragon: str,
    replaced_dragon: str,
):
    env = _make_env(campaign_objective=objective)
    env.reset(seed=123)

    names = _living_state_names(env, GREEN_DRAGON_ENEMY_ID)
    assert names == Counter({expected_dragon: 1})
    assert replaced_dragon not in names

    dragon = next(
        unit
        for unit in env.enemy_team_states[GREEN_DRAGON_ENEMY_ID]
        if str(unit.get("name", "")) == expected_dragon
    )
    max_hp = float(dragon.get("max_health", dragon.get("maxhp", 0)) or 0.0)
    assert max_hp > 0.0

    for damage in (123.0, 77.0):
        dragon["health"] = max_hp - damage
        dragon["hp"] = max_hp - damage
        env._advance_turns(1)
        assert dragon["health"] == pytest.approx(max_hp)
        assert dragon["hp"] == pytest.approx(max_hp)
