import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import (
    FIRST_HERO_ITEM_ACTION_START,
    HERO_ITEM_TARGET_SLOTS,
    SECOND_HERO_ITEM_ACTION_START,
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


def _equip_action(env: CampaignEnv, slot_no: int, item_name: str) -> int:
    action_start = (
        env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
        if int(slot_no) == 1
        else env.GRID_EQUIP_BATTLE_ITEM2_ACTION_START
    )
    return action_start + env.BATTLE_EQUIPPABLE_ITEM_NAMES.index(item_name)


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


@pytest.mark.parametrize(
    ("item_name", "slot_no", "missing_hp", "used_attr"),
    [
        (CampaignEnv.BONUS_SMALL_HEAL_ITEM_NAME, 1, 35, "healing_bottles_used"),
        (CampaignEnv.BONUS_LARGE_HEAL_ITEM_NAME, 1, 80, "heal_bottles_used"),
        (CampaignEnv.HEALING_OINTMENT_ITEM_NAME, 1, 100, None),
    ],
)
def test_equipped_healing_item_applies_in_battle_and_consumes_source(
    item_name: str,
    slot_no: int,
    missing_hp: int,
    used_attr: str | None,
):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    env.heroitems.append(env._make_hero_item_entry(item_name))

    _, _, terminated, truncated, equip_info = env.step(_equip_action(env, slot_no, item_name))
    assert terminated is False
    assert truncated is False
    assert equip_info["equip_battle_item_applied"] is True
    assert equip_info["equip_battle_item_reward"] == pytest.approx(env.reward_battle_item_equip)
    assert equip_info["battle_items_equipped_total"] == 1
    assert env.equipped_hero_items[slot_no - 1] == item_name

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
    assert env.equipped_hero_items[slot_no - 1] is None
    assert env.battle_env.equipped_hero_items[slot_no - 1] is None
    assert env._count_hero_item(item_name) == 0
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
    env.heroitems.append(env._make_hero_item_entry(item_name))

    _, _, terminated, truncated, equip_info = env.step(_equip_action(env, 2, item_name))
    assert terminated is False
    assert truncated is False
    assert equip_info["equip_battle_item_applied"] is True
    assert equip_info["equip_battle_item_reward"] == pytest.approx(env.reward_battle_item_equip)
    assert env.equipped_hero_items[1] == item_name

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

    action = SECOND_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(dead_slot)
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
    assert env._count_hero_item(item_name) == 0
    assert env.equipped_hero_items[1] is None
    assert env.battle_env.equipped_hero_items[1] is None
    assert terminated is False
    assert truncated is False


def test_re_equip_is_blocked_until_next_game_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    env.heroitems.append(env._make_hero_item_entry(env.BONUS_SMALL_HEAL_ITEM_NAME))
    env.heroitems.append(env._make_hero_item_entry(env.BONUS_LARGE_HEAL_ITEM_NAME))

    _, _, _, _, first_info = env.step(_equip_action(env, 1, env.BONUS_SMALL_HEAL_ITEM_NAME))
    _, reward, _, _, second_info = env.step(_equip_action(env, 1, env.BONUS_LARGE_HEAL_ITEM_NAME))

    assert first_info["equip_battle_item_reward"] == pytest.approx(env.reward_battle_item_equip)
    assert second_info["equip_battle_item_applied"] is False
    assert second_info["equip_battle_item_turn_locked"] is True
    assert second_info["equip_battle_item_previous"] is None
    assert second_info["equip_battle_item_reward"] == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)
    assert env.battle_items_equipped_total == 1
    assert env.equipped_hero_items[0] == env.BONUS_SMALL_HEAL_ITEM_NAME

    env._advance_turns(1)

    _, reward, _, _, third_info = env.step(_equip_action(env, 1, env.BONUS_LARGE_HEAL_ITEM_NAME))

    assert third_info["equip_battle_item_applied"] is True
    assert third_info["equip_battle_item_turn_locked"] is False
    assert third_info["equip_battle_item_previous"] == env.BONUS_SMALL_HEAL_ITEM_NAME
    assert third_info["equip_battle_item_reward"] == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)
    assert env.battle_items_equipped_total == 2
    assert env.equipped_hero_items[0] == env.BONUS_LARGE_HEAL_ITEM_NAME
