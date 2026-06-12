import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _new_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    return env


def _set_hero_level(env: CampaignEnv, level: int) -> None:
    hero = env._resolve_travel_hero()
    assert hero is not None
    hero["Level"] = int(level)


def _non_empty_blue_units(env: CampaignEnv):
    return [
        unit
        for unit in env.blue_team_state
        if isinstance(unit, dict) and not env._is_empty_blue_unit(unit)
    ]


def _blue_state_unit(env: CampaignEnv, position: int) -> dict:
    return next(
        unit
        for unit in env.blue_team_state
        if int(unit.get("position", -1) or -1) == int(position)
    )


def _battle_blue_unit(env: CampaignEnv, position: int) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(position)
    )


def _snapshot_non_empty_units(env: CampaignEnv) -> dict[int, dict]:
    return {
        int(unit.get("position", -1) or -1): deepcopy(unit)
        for unit in _non_empty_blue_units(env)
    }


def _prime_secondary_stats(env: CampaignEnv) -> None:
    for unit in _non_empty_blue_units(env):
        position = int(unit.get("position", 0) or 0)
        unit["damage_secondary"] = 10 + position
        unit["accuracy"] = 92
        unit["accuracy_secondary"] = 20 + position
        unit["armor"] = 2
        unit["base_armor"] = 2


def test_banner_pickup_before_banner_bearer_keeps_item_without_effects():
    env = _new_env()
    _set_hero_level(env, 6)
    _prime_secondary_stats(env)
    before = _snapshot_non_empty_units(env)

    env._add_hero_item(env.BANNER_OF_WAR_ITEM_NAME)

    assert env.equipped_banner_items == [None]
    assert env.BANNER_OF_WAR_ITEM_NAME in [env._hero_item_name(entry) for entry in env.heroitems]
    for position, before_unit in before.items():
        after_unit = _blue_state_unit(env, position)
        assert after_unit["damage"] == before_unit["damage"]
        assert after_unit["accuracy"] == before_unit["accuracy"]
        assert after_unit["initiative_base"] == before_unit["initiative_base"]
        assert after_unit["armor"] == before_unit["armor"]
        assert after_unit["damage_secondary"] == before_unit["damage_secondary"]
        assert after_unit["accuracy_secondary"] == before_unit["accuracy_secondary"]
        assert "campaign_active_banner" not in after_unit


def test_level_seven_auto_equips_best_banner_by_priority():
    env = _new_env()
    _set_hero_level(env, 7)

    env._add_hero_item(env.BANNER_OF_FORTITUDE_ITEM_NAME)
    env._add_hero_item(env.BANNER_OF_MIGHT_ITEM_NAME)
    env._add_hero_item(env.BANNER_OF_WAR_ITEM_NAME)

    assert env.equipped_banner_items == [env.BANNER_OF_WAR_ITEM_NAME]
    assert all(
        unit.get("campaign_active_banner") == env.BANNER_OF_WAR_ITEM_NAME
        for unit in _non_empty_blue_units(env)
    )


def test_banner_bearer_ability_token_unlocks_banner_below_level_seven():
    env = _new_env()
    _set_hero_level(env, 6)
    hero = env._resolve_travel_hero()
    assert hero is not None
    hero["hero_abilities"] = ["banner_bearer"]
    _prime_secondary_stats(env)
    before = _snapshot_non_empty_units(env)

    env._add_hero_item(env.BANNER_OF_WAR_ITEM_NAME)

    assert env.equipped_banner_items == [env.BANNER_OF_WAR_ITEM_NAME]
    assert "Знаменосец" in " ".join(env.get_travel_hero_visual_info()["abilities"])
    for position, before_unit in before.items():
        after_unit = _blue_state_unit(env, position)
        assert after_unit["campaign_active_banner"] == env.BANNER_OF_WAR_ITEM_NAME
        assert after_unit["damage"] == env._normalize_damage_value(
            float(before_unit["damage"]) * 1.20
        )
        assert after_unit["damage_secondary"] == before_unit["damage_secondary"]
        assert after_unit["accuracy_secondary"] == before_unit["accuracy_secondary"]


@pytest.mark.parametrize(
    ("item_name", "effect"),
    [
        (CampaignEnv.BANNER_OF_PROTECTION_ITEM_NAME, {"armor_bonus": 10}),
        (CampaignEnv.BANNER_OF_RESISTANCE_ITEM_NAME, {"armor_bonus": 15}),
        (CampaignEnv.BANNER_OF_FORTITUDE_ITEM_NAME, {"armor_bonus": 20}),
        (CampaignEnv.BANNER_OF_STRIKING_ITEM_NAME, {"accuracy_bonus": 10}),
        (CampaignEnv.BANNER_OF_BATTLE_ITEM_NAME, {"accuracy_bonus": 15}),
        (CampaignEnv.BANNER_OF_SPEED_ITEM_NAME, {"initiative_multiplier": 1.10}),
        (CampaignEnv.BANNER_OF_CELERITY_ITEM_NAME, {"initiative_multiplier": 1.15}),
        (CampaignEnv.BANNER_OF_STRENGTH_ITEM_NAME, {"damage_multiplier": 1.10}),
        (CampaignEnv.BANNER_OF_MIGHT_ITEM_NAME, {"damage_multiplier": 1.15}),
        (CampaignEnv.BANNER_OF_WAR_ITEM_NAME, {"damage_multiplier": 1.20}),
    ],
)
def test_all_banners_buff_every_non_empty_blue_unit_without_secondary_stats(
    item_name: str,
    effect: dict,
):
    env = _new_env()
    _set_hero_level(env, 7)
    _prime_secondary_stats(env)
    before = _snapshot_non_empty_units(env)

    env._add_hero_item(item_name)

    assert env.equipped_banner_items == [item_name]
    for position, before_unit in before.items():
        after_unit = _blue_state_unit(env, position)
        expected_damage = before_unit["damage"]
        expected_accuracy = before_unit["accuracy"]
        expected_initiative = before_unit["initiative_base"]
        expected_armor = before_unit["armor"]

        if "damage_multiplier" in effect:
            expected_damage = env._normalize_damage_value(
                float(before_unit["damage"]) * float(effect["damage_multiplier"])
            )
        if "accuracy_bonus" in effect:
            expected_accuracy = env._normalize_accuracy_value(
                int(before_unit["accuracy"]) + int(effect["accuracy_bonus"])
            )
        if "initiative_multiplier" in effect:
            expected_initiative = env._normalize_damage_value(
                float(before_unit["initiative_base"]) * float(effect["initiative_multiplier"])
            )
        if "armor_bonus" in effect:
            expected_armor = int(before_unit["armor"]) + int(effect["armor_bonus"])

        assert after_unit["campaign_active_banner"] == item_name
        assert after_unit["damage"] == expected_damage
        assert after_unit["accuracy"] == expected_accuracy
        assert after_unit["initiative_base"] == expected_initiative
        assert after_unit["initiative"] == expected_initiative
        assert after_unit["armor"] == expected_armor
        assert after_unit["damage_secondary"] == before_unit["damage_secondary"]
        assert after_unit["accuracy_secondary"] == before_unit["accuracy_secondary"]


def test_banner_effects_do_not_stack_and_are_saved_once_after_battle():
    env = _new_env()
    _set_hero_level(env, 7)
    _prime_secondary_stats(env)
    before = _snapshot_non_empty_units(env)

    env._add_hero_item(env.BANNER_OF_WAR_ITEM_NAME)
    env._refresh_campaign_equipment_effects(log=False)
    env._refresh_campaign_banner_effects(log=False)

    expected_by_position = {
        position: env._normalize_damage_value(float(unit["damage"]) * 1.20)
        for position, unit in before.items()
    }
    for position, expected_damage in expected_by_position.items():
        assert _blue_state_unit(env, position)["damage"] == expected_damage

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, 7)
    assert battle_unit["campaign_active_banner"] == env.BANNER_OF_WAR_ITEM_NAME
    assert battle_unit["damage"] == expected_by_position[7]

    env._save_blue_state()
    for position, expected_damage in expected_by_position.items():
        after_unit = _blue_state_unit(env, position)
        assert after_unit["campaign_active_banner"] == env.BANNER_OF_WAR_ITEM_NAME
        assert after_unit["campaign_banner_base_damage"] == before[position]["damage"]
        assert after_unit["damage"] == expected_damage
        assert after_unit["damage_secondary"] == before[position]["damage_secondary"]
        assert after_unit["accuracy_secondary"] == before[position]["accuracy_secondary"]


def test_merchant_autosale_keeps_banners_as_useful_items():
    env = _new_env()
    merchant_site_name = next(iter(env.merchant_site_interaction_tiles))
    merchant_tile = tuple(env.merchant_site_interaction_tiles[merchant_site_name][0])
    env.heroitems = [
        env._make_hero_item_entry(env.BANNER_OF_WAR_ITEM_NAME),
        env._make_hero_item_entry("Ruby (Valuable)"),
    ]
    env._invalidate_inventory_cache()

    sale_info = env._sell_sell_only_heroitems_at_merchant(position=merchant_tile)

    assert sale_info["merchant_auto_sale"] is True
    assert [item["name"] for item in sale_info["merchant_sold_items"]] == ["Ruby (Valuable)"]
    assert [env._hero_item_name(entry) for entry in env.heroitems] == [
        env.BANNER_OF_WAR_ITEM_NAME
    ]


def test_render_and_state_summary_include_equipped_banner():
    env = _new_env()
    _set_hero_level(env, 7)
    env._add_hero_item(env.BANNER_OF_SPEED_ITEM_NAME)

    rendered = env.render(mode="ansi")
    summary = env.get_state_summary()

    assert rendered is not None
    assert env.BANNER_OF_SPEED_ITEM_NAME in rendered
    assert summary["equipped_banner_items"] == [env.BANNER_OF_SPEED_ITEM_NAME]
