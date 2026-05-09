import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import (
    FIRST_HERO_ITEM_ACTION_START,
    HERO_ITEM_TARGET_SLOTS,
)
from campaign_env import CampaignEnv


def _blue_state_unit(env: CampaignEnv, position: int) -> dict:
    return next(
        unit for unit in env.blue_team_state if int(unit.get("position", -1)) == int(position)
    )


def _battle_blue_unit(env: CampaignEnv, position: int) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1)) == int(position)
    )


def _equip_action(env: CampaignEnv, item_name: str) -> int:
    return (
        env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
        + env.BATTLE_EQUIPPABLE_ITEM_NAMES.index(item_name)
    )


def _grant_sorcery_lore(env: CampaignEnv) -> dict:
    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    hero["hero_abilities"] = [env.HERO_SORCERY_LORE_ABILITY_KEY]
    assert env._hero_has_sorcery_lore()
    return hero


def _set_battle_hero_turn(env: CampaignEnv, position: int = 8) -> dict:
    assert env.battle_env is not None
    hero = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1)) == int(position)
    )
    hero["initiative"] = max(1, int(hero.get("initiative_base", 1) or 1))
    env.battle_env.current_blue_attacker_pos = int(position)
    env.battle_env.blue_attacks_left = 1
    return hero


def test_counter_backed_battle_items_are_synced_into_heroitems_on_reset():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    assert env._count_hero_item(env.BONUS_SMALL_HEAL_ITEM_NAME) == env._healing_bottles_left()
    assert env._count_hero_item(env.BONUS_LARGE_HEAL_ITEM_NAME) == env._heal_bottles_left()
    assert env._count_hero_item(env.BONUS_REVIVE_ITEM_NAME) == env._revive_bottles_left()


def test_battle_equip_actions_include_magic_items_present_on_current_map():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    assert env.scenario_battle_magic_item_names == (
        "Vampire Orb",
        "Orb of Life",
        "Lich Orb",
        "Zombie Talisman",
    )
    assert env.scenario_battle_orb_item_names == env.scenario_battle_magic_item_names
    assert env.BATTLE_EQUIPPABLE_ITEM_NAMES[: len(env.BASE_BATTLE_EQUIPPABLE_ITEM_NAMES)] == (
        env.BASE_BATTLE_EQUIPPABLE_ITEM_NAMES
    )
    assert env.BATTLE_EQUIPPABLE_ITEM_NAMES[
        len(env.BASE_BATTLE_EQUIPPABLE_ITEM_NAMES) :
    ] == env.scenario_battle_magic_item_names
    assert env.GRID_EQUIP_BOOK_ACTION_START == (
        env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
        + len(env.BATTLE_EQUIPPABLE_ITEM_NAMES)
    )
    assert env.scenario_book_item_names == (env.TOME_OF_ARCANUM_ITEM_NAME,)
    assert env.grid_legion_damage_spell_action_start == (
        env.GRID_EQUIP_BOOK_ACTION_START + len(env.scenario_book_item_names)
    )


def test_orb_equip_action_requires_owned_orb_and_sorcery_lore():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    item_name = "Vampire Orb"
    action = _equip_action(env, item_name)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    env._append_hero_item(item_name)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _grant_sorcery_lore(env)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is True

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_battle_item_equip)
    assert info["equip_battle_item_applied"] is True
    assert info["equip_battle_item_slot"] == 1
    assert info["equip_battle_item_name"] == item_name
    assert env.equipped_hero_items[0] == item_name


def test_merchant_orbs_reserve_equip_actions_but_stay_masked_until_owned():
    class MerchantOrbCampaignEnv(CampaignEnv):
        MERCHANT_BUY_ITEMS = CampaignEnv.MERCHANT_BUY_ITEMS + (
            {"name": "Angel Orb", "price": 800.0, "stock": 1, "grant": "inventory"},
        )

    env = MerchantOrbCampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    assert "Angel Orb" in env.scenario_battle_orb_item_names
    assert "Angel Orb" in env.BATTLE_EQUIPPABLE_ITEM_NAMES
    assert any("Angel Orb" in site_stock for site_stock in env.merchant_stocks.values())

    action = _equip_action(env, "Angel Orb")
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    env._append_hero_item("Angel Orb")
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _grant_sorcery_lore(env)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is True


def test_talisman_equip_action_requires_owned_item_and_sorcery_lore():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    item_name = "Zombie Talisman"
    action = _equip_action(env, item_name)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    env._append_hero_item(item_name)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _grant_sorcery_lore(env)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is True

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_battle_item_equip)
    assert info["equip_battle_item_applied"] is True
    assert info["equip_battle_item_slot"] == 1
    assert info["equip_battle_item_name"] == item_name
    assert info["equip_battle_item_uses_left"] == env.TALISMAN_USES_PER_ITEM
    assert env.equipped_hero_items[0] == item_name
    assert env.equipped_hero_item_uses_left[0] == env.TALISMAN_USES_PER_ITEM


@pytest.mark.parametrize(
    ("item_name", "missing_hp", "used_attr"),
    [
        (CampaignEnv.BONUS_SMALL_HEAL_ITEM_NAME, 35, "healing_bottles_used"),
        (CampaignEnv.BONUS_LARGE_HEAL_ITEM_NAME, 80, "heal_bottles_used"),
        (CampaignEnv.HEALING_OINTMENT_ITEM_NAME, 100, None),
    ],
)
def test_equipped_healing_item_applies_in_battle_and_consumes_source(
    item_name: str,
    missing_hp: int,
    used_attr: str | None,
):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    if item_name == env.HEALING_OINTMENT_ITEM_NAME:
        env._append_hero_item(item_name)
    starting_item_count = env._count_hero_item(item_name)

    _, _, terminated, truncated, equip_info = env.step(_equip_action(env, item_name))
    assert terminated is False
    assert truncated is False
    assert equip_info["equip_battle_item_applied"] is True
    assert equip_info["equip_battle_item_slot"] == 1
    assert equip_info["equip_battle_item_reward"] == pytest.approx(env.reward_battle_item_equip)
    assert equip_info["battle_items_equipped_total"] == 1
    assert env.equipped_hero_items[0] == item_name

    env.current_enemy_id = 1
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env, 8)

    battle_target = _battle_blue_unit(env, 7)
    max_health = int(
        battle_target.get("max_health", 0) or battle_target.get("maxhp", 0) or 0
    )
    battle_target["health"] = max_health - int(missing_hp)
    if "hp" in battle_target:
        battle_target["hp"] = battle_target["health"]

    _, _, terminated, truncated, info = env.step(FIRST_HERO_ITEM_ACTION_START)

    assert int(battle_target.get("health", 0) or 0) == max_health
    assert info["battle_hero_item_action"] is True
    assert info["battle_hero_item_name"] == item_name
    assert info["battle_hero_item_effect_kind"] == "heal"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_item_use_reward"] == pytest.approx(env.reward_battle_item_use)
    assert info["battle_items_used_total"] == 1
    assert env.battle_items_equipped_total == 1
    assert env.battle_items_used_total == 1
    assert env.equipped_hero_items[0] is None
    assert env.battle_env.equipped_hero_items[0] is None
    assert env._count_hero_item(item_name) == starting_item_count - 1
    if used_attr is not None:
        assert int(getattr(env, used_attr)) == 1
    else:
        assert env.healing_bottles_used == 0
        assert env.heal_bottles_used == 0
    assert terminated is False
    assert truncated is False


def test_equipped_revive_item_applies_in_battle_and_consumes_source():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    item_name = env.BONUS_REVIVE_ITEM_NAME
    starting_item_count = env._count_hero_item(item_name)

    _, _, terminated, truncated, equip_info = env.step(_equip_action(env, item_name))
    assert terminated is False
    assert truncated is False
    assert equip_info["equip_battle_item_applied"] is True
    assert equip_info["equip_battle_item_slot"] == 1
    assert equip_info["equip_battle_item_reward"] == pytest.approx(env.reward_battle_item_equip)
    assert env.equipped_hero_items[0] == item_name

    dead_slot = 5
    target_pos = 6 + dead_slot
    env.current_enemy_id = 1
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env, 8)

    battle_target = _battle_blue_unit(env, target_pos)
    battle_target["health"] = 0.0
    if "hp" in battle_target:
        battle_target["hp"] = 0.0

    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(dead_slot)
    _, _, terminated, truncated, info = env.step(action)

    assert int(battle_target.get("health", 0) or 0) == 1
    assert info["battle_hero_item_name"] == item_name
    assert info["battle_hero_item_effect_kind"] == "revive"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_item_use_reward"] == pytest.approx(env.reward_battle_item_use)
    assert env.battle_items_equipped_total == 1
    assert env.battle_items_used_total == 1
    assert env.revive_bottles_used == 1
    assert env._count_hero_item(item_name) == starting_item_count - 1
    assert env.equipped_hero_items[0] is None
    assert env.battle_env.equipped_hero_items[0] is None
    assert terminated is False
    assert truncated is False


def test_equipped_orb_of_life_revives_and_consumes_hero_item_source():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    item_name = "Orb of Life"
    env._append_hero_item(item_name)
    _grant_sorcery_lore(env)
    starting_item_count = env._count_hero_item(item_name)

    _, _, _, _, equip_info = env.step(_equip_action(env, item_name))
    assert equip_info["equip_battle_item_applied"] is True
    assert env.equipped_hero_items[0] == item_name

    dead_slot = 5
    target_pos = 6 + dead_slot
    env.current_enemy_id = 1
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env, 8)

    battle_target = _battle_blue_unit(env, target_pos)
    battle_target["health"] = 0.0
    if "hp" in battle_target:
        battle_target["hp"] = 0.0

    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(dead_slot)
    _, _, terminated, truncated, info = env.step(action)

    assert int(battle_target.get("health", 0) or 0) == 1
    assert info["battle_hero_item_name"] == item_name
    assert info["battle_hero_item_effect_kind"] == "revive"
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_item_use_reward"] == pytest.approx(env.reward_battle_item_use)
    assert env.battle_items_used_total == 1
    assert env._count_hero_item(item_name) == starting_item_count - 1
    assert env.equipped_hero_items[0] is None
    assert env.battle_env.equipped_hero_items[0] is None
    assert terminated is False
    assert truncated is False


def test_equipped_vampire_orb_summons_and_consumes_hero_item_source():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    item_name = "Vampire Orb"
    env._append_hero_item(item_name)
    _grant_sorcery_lore(env)
    starting_item_count = env._count_hero_item(item_name)

    _, _, _, _, equip_info = env.step(_equip_action(env, item_name))
    assert equip_info["equip_battle_item_applied"] is True
    assert env.equipped_hero_items[0] == item_name

    env.current_enemy_id = 1
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env, 8)

    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(4)
    _, _, terminated, truncated, info = env.step(action)

    summoned = _battle_blue_unit(env, 10)
    assert summoned["name"] == "Вампир"
    assert summoned["Summoned"] == 8
    assert info["battle_hero_item_name"] == item_name
    assert info["battle_hero_item_effect_kind"] == "summon"
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_item_use_reward"] == pytest.approx(env.reward_battle_item_use)
    assert env.battle_items_used_total == 1
    assert env._count_hero_item(item_name) == starting_item_count - 1
    assert env.equipped_hero_items[0] is None
    assert env.battle_env.equipped_hero_items[0] is None
    assert terminated is False
    assert truncated is False


def test_equipped_talisman_spends_charge_without_consuming_source():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    item_name = "Zombie Talisman"
    env._append_hero_item(item_name)
    _grant_sorcery_lore(env)
    starting_item_count = env._count_hero_item(item_name)

    _, _, _, _, equip_info = env.step(_equip_action(env, item_name))
    assert equip_info["equip_battle_item_applied"] is True
    assert env.equipped_hero_items[0] == item_name
    assert env.equipped_hero_item_uses_left[0] == env.TALISMAN_USES_PER_ITEM

    env.current_enemy_id = 1
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env, 8)

    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(4)
    _, _, terminated, truncated, info = env.step(action)

    summoned = _battle_blue_unit(env, 10)
    expected_name = env._battle_item_effect_definitions()["Zombie Talisman"][
        "summon_unit_name"
    ]
    assert summoned["name"] == expected_name
    assert info["battle_hero_item_name"] == item_name
    assert info["battle_hero_item_effect_kind"] == "summon"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is False
    assert info["battle_hero_item_charge_spent"] is True
    assert info["battle_hero_item_uses_left"] == env.TALISMAN_USES_PER_ITEM - 1
    assert info["battle_item_use_reward"] == pytest.approx(env.reward_battle_item_use)
    assert env.battle_items_used_total == 1
    assert env._count_hero_item(item_name) == starting_item_count
    assert env.equipped_hero_items[0] == item_name
    assert env.equipped_hero_item_uses_left[0] == env.TALISMAN_USES_PER_ITEM - 1
    assert env.battle_env.equipped_hero_items[0] == item_name
    assert env.battle_env.equipped_hero_item_uses_left[0] == env.TALISMAN_USES_PER_ITEM - 1

    _set_battle_hero_turn(env, 8)
    mask = env.battle_env.compute_action_mask()
    assert bool(mask[action]) is False

    env.current_enemy_id = 1
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env, 8)
    next_battle_mask = env.battle_env.compute_action_mask()
    assert env.battle_env.equipped_hero_item_uses_left[0] == env.TALISMAN_USES_PER_ITEM - 1
    assert bool(next_battle_mask[action]) is True
    assert terminated is False
    assert truncated is False


def test_talisman_final_charge_consumes_source_and_clears_slot():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    item_name = "Zombie Talisman"
    env._append_hero_item(item_name)
    _grant_sorcery_lore(env)
    starting_item_count = env._count_hero_item(item_name)

    _, _, _, _, equip_info = env.step(_equip_action(env, item_name))
    assert equip_info["equip_battle_item_applied"] is True
    env.equipped_hero_item_uses_left[0] = 1
    env.equipped_hero_item_uses_left_item_names[0] = item_name

    env.current_enemy_id = 1
    env._init_battle(enemy_id=1)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    _set_battle_hero_turn(env, 8)

    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(4)
    _, _, terminated, truncated, info = env.step(action)

    assert info["battle_hero_item_name"] == item_name
    assert info["battle_hero_item_effect_kind"] == "summon"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_hero_item_charge_spent"] is True
    assert info["battle_hero_item_uses_left"] == 0
    assert env.battle_items_used_total == 1
    assert env._count_hero_item(item_name) == starting_item_count - 1
    assert env.equipped_hero_items[0] is None
    assert env.equipped_hero_item_uses_left[0] is None
    assert env.battle_env.equipped_hero_items[0] is None
    assert terminated is False
    assert truncated is False


def test_re_equip_is_blocked_until_next_game_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    _, _, _, _, first_info = env.step(_equip_action(env, env.BONUS_SMALL_HEAL_ITEM_NAME))
    _, reward, _, _, second_info = env.step(_equip_action(env, env.BONUS_LARGE_HEAL_ITEM_NAME))

    assert first_info["equip_battle_item_reward"] == pytest.approx(env.reward_battle_item_equip)
    assert first_info["equip_battle_item_slot"] == 1
    assert second_info["equip_battle_item_applied"] is False
    assert second_info["equip_battle_item_turn_locked"] is True
    assert second_info["equip_battle_item_previous"] is None
    assert second_info["equip_battle_item_slot"] == 0
    assert second_info["equip_battle_item_reward"] == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)
    assert env.battle_items_equipped_total == 1
    assert env.equipped_hero_items[0] == env.BONUS_SMALL_HEAL_ITEM_NAME

    env._advance_turns(1)

    _, reward, _, _, third_info = env.step(_equip_action(env, env.BONUS_LARGE_HEAL_ITEM_NAME))

    assert third_info["equip_battle_item_applied"] is True
    assert third_info["equip_battle_item_turn_locked"] is False
    assert third_info["equip_battle_item_previous"] is None
    assert third_info["equip_battle_item_slot"] == 2
    assert third_info["equip_battle_item_reward"] == pytest.approx(env.reward_battle_item_equip)
    assert reward == pytest.approx(env.reward_battle_item_equip)
    assert env.battle_items_equipped_total == 2
    assert env.equipped_hero_items[0] == env.BONUS_SMALL_HEAL_ITEM_NAME
    assert env.equipped_hero_items[1] == env.BONUS_LARGE_HEAL_ITEM_NAME


def test_battle_item_equip_actions_are_masked_when_both_slots_are_full():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    env._append_hero_item(env.HEALING_OINTMENT_ITEM_NAME)

    _, _, _, _, first_info = env.step(_equip_action(env, env.BONUS_SMALL_HEAL_ITEM_NAME))
    env._advance_turns(1)
    _, _, _, _, second_info = env.step(_equip_action(env, env.BONUS_LARGE_HEAL_ITEM_NAME))

    assert first_info["equip_battle_item_slot"] == 1
    assert second_info["equip_battle_item_slot"] == 2
    assert env.equipped_hero_items == [
        env.BONUS_SMALL_HEAL_ITEM_NAME,
        env.BONUS_LARGE_HEAL_ITEM_NAME,
    ]

    mask = env.compute_action_mask()
    equip_start = env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
    equip_end = equip_start + len(env.BATTLE_EQUIPPABLE_ITEM_NAMES)
    assert mask[equip_start:equip_end].tolist() == [False] * len(
        env.BATTLE_EQUIPPABLE_ITEM_NAMES
    )

    _, reward, _, _, blocked_info = env.step(_equip_action(env, env.HEALING_OINTMENT_ITEM_NAME))

    assert blocked_info["equip_battle_item_applied"] is False
    assert blocked_info["equip_battle_item_slot"] == 0
    assert reward == pytest.approx(0.0)
    assert env.equipped_hero_items == [
        env.BONUS_SMALL_HEAL_ITEM_NAME,
        env.BONUS_LARGE_HEAL_ITEM_NAME,
    ]
