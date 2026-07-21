import pytest

from campaign_env import CampaignEnv
from grid import are_targets_reachable
from maps import get_map
from maps.wotans_retribution import AGENT_CAPITAL_ANCHOR
from maps.wotans_retribution import BOT_CAPITAL_ANCHOR
from maps.wotans_retribution import BOT_CAPITAL_DEFENDER_IDS
from maps.wotans_retribution import BOT_HOME
from maps.wotans_retribution import CHESTS
from maps.wotans_retribution import FINAL_OBJECTIVE_CITIES
from maps.wotans_retribution import FOREST_TILES
from maps.wotans_retribution import HERO_START
from maps.wotans_retribution import MERCHANT_ITEMS_BY_SITE
from maps.wotans_retribution import MOUNTAIN_BLOCKS
from maps.wotans_retribution import PROTECTED_ENEMY_IDS
from maps.wotans_retribution import ROAD_TILES
from maps.wotans_retribution import SOURCE_SNAPSHOT
from maps.wotans_retribution import TARGET_CITY_ENEMY_IDS
from maps.wotans_retribution import TARGET_CITY_SOURCE_IDS
from maps.wotans_retribution import WATER_TILES


def _make_env(*, bot_enabled: bool) -> CampaignEnv:
    env = CampaignEnv(
        map_name="wotans_retribution",
        campaign_objective="cities",
        scripted_capital_bot_enabled=bot_enabled,
        log_enabled=False,
        persist_blue_hp=True,
    )
    env.reset(seed=117)
    return env


def _living_names(team):
    return {
        int(unit["position"]): str(unit["name"])
        for unit in team
        if str(unit.get("name", "")) != "пусто"
    }


def test_map_config_preserves_scenario_geometry_and_static_objects():
    map_config = get_map("wotans_retribution")

    assert map_config.grid_size == 48
    assert map_config.play_grid_size is None
    assert map_config.hero_start == HERO_START == (40, 6)
    assert AGENT_CAPITAL_ANCHOR == (36, 4)
    assert BOT_CAPITAL_ANCHOR == (9, 31)
    assert len(MOUNTAIN_BLOCKS) == 196
    assert len(ROAD_TILES) == 194
    assert len(WATER_TILES) == 181
    assert len(FOREST_TILES) == 441
    assert len(map_config.enemy_stacks) == 71
    assert len(CHESTS) == 24
    assert len(map_config.mana_sources) == 6
    assert len(map_config.gold_mine_tiles_provider()) == 5
    assert len(map_config.ruin_rewards) == 4
    assert len(map_config.merchant_sites) == 2
    assert len(map_config.trainer_sites) == 1
    assert map_config.default_objective == "cities"
    assert map_config.supported_objectives == ("cities",)
    assert map_config.objective_enemy_id is None

    assert set(FINAL_OBJECTIVE_CITIES) == {
        "Kalen's Palace",
        "Frelan",
        "Hasser's Keep",
    }
    assert TARGET_CITY_ENEMY_IDS == {
        "S117FT0004": (6, 14),
        "S117FT0005": (7, 13),
        "S117FT0006": (8,),
    }
    source_targets = {
        row["source_id"]: tuple(row["interaction_tile"])
        for row in SOURCE_SNAPSHOT["settlements"]
        if row["source_id"] in TARGET_CITY_SOURCE_IDS
    }
    assert source_targets == {
        "S117FT0004": (5, 18),
        "S117FT0005": (6, 42),
        "S117FT0006": (7, 4),
    }


@pytest.mark.parametrize("objective", ("dragon", "blue_dragon", "all_enemies"))
def test_map_rejects_every_non_city_objective(objective):
    with pytest.raises(ValueError, match="is not supported"):
        CampaignEnv(
            map_name="wotans_retribution",
            campaign_objective=objective,
            scripted_capital_bot_enabled=False,
            log_enabled=False,
        )


def test_agent_party_is_replaced_and_has_exactly_one_hero():
    env = _make_env(bot_enabled=False)

    assert env.Realcapital == 3
    assert env.typeoflord == 1
    assert env.grid_env.start_position == HERO_START
    assert _living_names(env.blue_team_state) == {
        7: "Гном",
        8: "Королевский страж",
        10: "Метатель топоров",
        11: "Травница",
    }
    heroes = [unit for unit in env.blue_team_state if bool(unit.get("hero", False))]
    assert [(unit["position"], unit["name"]) for unit in heroes] == [
        (8, "Королевский страж")
    ]


def test_empire_bot_roster_home_and_protected_targets():
    env = _make_env(bot_enabled=True)

    assert env.scripted_capital_bot_faction == "Империя"
    assert env.scripted_capital_bot_home == BOT_HOME == (13, 33)
    assert env.scripted_capital_bot_position == BOT_HOME
    assert _living_names(env.scripted_capital_bot_team_state) == {
        7: "Скваер",
        9: "Скваер",
        10: "Рейнджер людей",
        12: "Служка",
    }
    assert set(BOT_CAPITAL_DEFENDER_IDS) == {5, 10}
    assert all(env.grid_env.enemies_alive[enemy_id] for enemy_id in BOT_CAPITAL_DEFENDER_IDS)

    available_targets = set(env._scripted_bot_alive_enemy_tiles().values())
    assert available_targets
    assert available_targets.isdisjoint(PROTECTED_ENEMY_IDS)
    skipped = env._run_scripted_capital_bot_battle(TARGET_CITY_ENEMY_IDS["S117FT0004"][0])
    assert skipped["skipped"] is True
    assert skipped["skip_reason"] == "map_protected_enemy_forbidden"


def test_disabled_bot_has_no_dynamic_party_or_map_marker():
    env = _make_env(bot_enabled=False)

    assert env.scripted_capital_bot_enabled is False
    assert env.scripted_capital_bot_team_state == []
    assert env.scripted_capital_bot_state == "disabled"
    assert env.grid_env.scripted_bot_position is None


def test_each_city_requires_all_defenders_and_victory_is_after_the_third():
    env = _make_env(bot_enabled=False)
    city_names = list(FINAL_OBJECTIVE_CITIES)

    first_ids = FINAL_OBJECTIVE_CITIES[city_names[0]]
    env.grid_env.enemies_alive[first_ids[0]] = False
    assert env._capture_objective_city_if_cleared(first_ids[0]) == []
    assert not env.captured_objective_cities

    for city_index, city_name in enumerate(city_names):
        enemy_ids = FINAL_OBJECTIVE_CITIES[city_name]
        for enemy_id in enemy_ids:
            env.grid_env.enemies_alive[enemy_id] = False
        assert env._capture_objective_city_if_cleared(enemy_ids[-1]) == [city_name]
        assert env._all_objective_cities_captured() is (city_index == 2)


def test_all_required_objects_are_collision_free_and_reachable():
    env = _make_env(bot_enabled=False)
    obstacle_tiles = set(env.grid_env.obstacle_positions)
    target_tiles = set(env.castle_heal_tiles) | set(env.grid_env.enemy_positions.values())
    target_tiles.update(env.grid_env.chest_positions)
    target_tiles.update(
        tuple(source["position"])
        for source in SOURCE_SNAPSHOT["resources"]
    )
    for site in SOURCE_SNAPSHOT["sites"]:
        target_tiles.update(tuple(tile) for tile in site["interaction_tiles"])

    assert obstacle_tiles.isdisjoint(target_tiles)
    assert all(0 <= x < 48 and 0 <= y < 48 for x, y in target_tiles)
    assert are_targets_reachable(
        48,
        HERO_START,
        obstacle_tiles,
        target_tiles,
    )


def test_source_tefas_garrison_ruins_and_exact_shop_stocks_are_preserved():
    source_tefas = next(
        stack for stack in SOURCE_SNAPSHOT["stacks"] if stack["source_id"] == "S117KC0000"
    )
    assert source_tefas["inside"] == "S117FT0000"
    assert source_tefas["group"]["units"][0]["source_name"] == "Tefas"
    assert source_tefas["group"]["front"] == [None, None, "Рыцарь на пегасе"]

    env = _make_env(bot_enabled=False)
    assert env.merchant_stocks == {
        site_name: {
            str(item["name"]): int(item["stock"])
            for item in site_items
        }
        for site_name, site_items in MERCHANT_ITEMS_BY_SITE.items()
    }
    assert len(env.RUIN_REWARD_BY_ENEMY_ID) == 4
    assert all(reward["items"] for reward in env.RUIN_REWARD_BY_ENEMY_ID.values())
