import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _new_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    return env


def _set_hero_level(env: CampaignEnv, level: int) -> dict:
    hero = env._resolve_travel_hero()
    assert hero is not None
    hero["Level"] = int(level)
    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)
    return hero


def _non_empty_blue_units(env: CampaignEnv):
    return [
        unit
        for unit in env.blue_team_state
        if isinstance(unit, dict) and not env._is_empty_blue_unit(unit)
    ]


def _snapshot_combat_stats(env: CampaignEnv) -> dict[int, dict]:
    stat_fields = (
        "damage",
        "damage_secondary",
        "accuracy",
        "accuracy_secondary",
        "initiative_base",
        "initiative",
        "armor",
        "base_armor",
        "hp",
        "health",
        "maxhp",
        "max_health",
    )
    return {
        int(unit.get("position", -1) or -1): {
            field: deepcopy(unit.get(field))
            for field in stat_fields
            if field in unit
        }
        for unit in _non_empty_blue_units(env)
    }


def _merchant_tile(env: CampaignEnv) -> tuple[int, int]:
    site_name = next(iter(env.merchant_site_interaction_tiles))
    return tuple(env.merchant_site_interaction_tiles[site_name][0])


def test_boot_pickup_without_marching_lore_keeps_inventory_without_move_bonus():
    env = _new_env()
    _set_hero_level(env, 7)
    base_moves_per_turn = int(env.moves_per_turn)

    env._add_hero_item(env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME)

    assert env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME in [
        env._hero_item_name(entry) for entry in env.heroitems
    ]
    assert env.equipped_boot_items == [None]
    assert env._active_boot_move_bonus() == 0
    assert env.moves_per_turn == base_moves_per_turn


def test_level_eight_auto_equips_best_boots_by_move_bonus():
    env = _new_env()
    _set_hero_level(env, 8)

    env._add_hero_item(env.BOOTS_OF_SPEED_ITEM_NAME)
    env._add_hero_item(env.BOOTS_OF_TRAVELING_ITEM_NAME)
    env._add_hero_item(env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME)

    assert env.equipped_boot_items == [env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME]
    assert env._active_boot_move_bonus() == 12
    assert env.moves_per_turn == 44
    assert env.moves == 44


def test_marching_lore_ability_token_unlocks_boots_below_level_eight():
    env = _new_env()
    hero = _set_hero_level(env, 6)
    hero["hero_abilities"] = ["marching_lore"]

    env._add_hero_item(env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME)

    assert env.equipped_boot_items == [env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME]
    assert env._active_boot_move_bonus() == 12
    assert env.moves_per_turn == 42
    assert "Походные знания" in " ".join(env.get_travel_hero_visual_info()["abilities"])


def test_guild_lord_can_use_boots_from_level_one():
    env = _new_env()
    env.typeoflord = 3
    env.Typeoflord = 3

    env._add_hero_item(env.BOOTS_OF_TRAVELING_ITEM_NAME)

    assert env.equipped_boot_items == [env.BOOTS_OF_TRAVELING_ITEM_NAME]
    assert env._active_boot_move_bonus() == 8
    assert env.moves_per_turn == 29
    assert "Походные знания" in " ".join(env.get_travel_hero_visual_info()["abilities"])


@pytest.mark.parametrize(
    ("item_name", "move_bonus"),
    [
        (CampaignEnv.ELVEN_BOOTS_ITEM_NAME, 0),
        (CampaignEnv.BOOTS_OF_THE_ELEMENTS_ITEM_NAME, 0),
        (CampaignEnv.BOOTS_OF_SPEED_ITEM_NAME, 4),
        (CampaignEnv.BOOTS_OF_TRAVELING_ITEM_NAME, 8),
        (CampaignEnv.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME, 12),
    ],
)
def test_all_boots_change_only_campaign_moves(item_name: str, move_bonus: int):
    env = _new_env()
    _set_hero_level(env, 8)
    before = _snapshot_combat_stats(env)

    env._add_hero_item(item_name)

    assert env.equipped_boot_items == [item_name]
    assert env._active_boot_move_bonus() == move_bonus
    assert env.moves_per_turn == 32 + move_bonus
    assert _snapshot_combat_stats(env) == before


def test_boots_are_collected_from_chests_and_auto_equipped():
    env = _new_env()
    _set_hero_level(env, 8)
    chest_pos = tuple(env.grid_env.agent_pos)
    env.chests = {chest_pos: (env.BOOTS_OF_TRAVELING_ITEM_NAME,)}
    env._sync_grid_chest_positions()

    collected = env._collect_adjacent_chests()

    assert collected == [
        {
            "pos": chest_pos,
            "items": [env.BOOTS_OF_TRAVELING_ITEM_NAME],
            "granted_items": [],
        }
    ]
    assert env.BOOTS_OF_TRAVELING_ITEM_NAME in [
        env._hero_item_name(entry) for entry in env.heroitems
    ]
    assert env.equipped_boot_items == [env.BOOTS_OF_TRAVELING_ITEM_NAME]
    assert env.moves_per_turn == 40


def test_merchant_autosale_keeps_boots_as_useful_items():
    env = _new_env()
    merchant_tile = _merchant_tile(env)
    env.heroitems = [
        env._make_hero_item_entry(env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME),
        env._make_hero_item_entry("Ruby (Valuable)"),
    ]
    env._invalidate_inventory_cache()

    sale_info = env._sell_sell_only_heroitems_at_merchant(position=merchant_tile)

    assert sale_info["merchant_auto_sale"] is True
    assert [item["name"] for item in sale_info["merchant_sold_items"]] == ["Ruby (Valuable)"]
    assert [env._hero_item_name(entry) for entry in env.heroitems] == [
        env.BOOTS_OF_SEVEN_LEAGUES_ITEM_NAME
    ]


def test_render_and_state_summary_include_equipped_boots():
    env = _new_env()
    _set_hero_level(env, 8)
    env._add_hero_item(env.BOOTS_OF_SPEED_ITEM_NAME)

    rendered = env.render(mode="ansi")
    summary = env.get_state_summary()

    assert rendered is not None
    assert env.BOOTS_OF_SPEED_ITEM_NAME in rendered
    assert summary["equipped_boot_items"] == [env.BOOTS_OF_SPEED_ITEM_NAME]
    assert summary["active_boot_move_bonus"] == 4
