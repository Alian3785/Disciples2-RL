import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import (
    BattleEnv,
    FIRST_HERO_ITEM_ACTION_START,
    HERO_ITEM_TARGET_SLOTS,
    SECOND_HERO_ITEM_ACTION_START,
)


def _blue_unit(env: BattleEnv, position: int) -> dict:
    return next(
        unit
        for unit in env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1)) == int(position)
    )


def _set_blue_turn(env: BattleEnv, position: int) -> dict:
    unit = _blue_unit(env, position)
    unit["initiative"] = max(1, int(unit.get("initiative_base", 1) or 1))
    env.current_blue_attacker_pos = int(position)
    env.blue_attacks_left = 1
    return unit


def _restore_blue_health(env: BattleEnv) -> None:
    for unit in env.combined:
        if unit.get("team") != "blue":
            continue
        max_health = float(unit.get("max_health", 0) or unit.get("health", 0) or 0)
        if max_health <= 0:
            continue
        unit["health"] = int(round(max_health))
        if "hp" in unit:
            unit["hp"] = int(round(max_health))


def _configure_hero_items(env: BattleEnv) -> None:
    env.equipped_hero_items = ["Test heal", "Test revive"]
    env.hero_item_effects = {
        "Test heal": {"kind": "heal", "amount": 50.0},
        "Test revive": {"kind": "revive", "amount": 1.0},
    }


def test_hero_item_actions_mask_only_on_hero_turn_and_only_for_valid_targets():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    _configure_hero_items(env)

    damaged_ally = _blue_unit(env, 7)
    damaged_ally["health"] = max(1, int(damaged_ally.get("max_health", 1) or 1) - 25)

    dead_ally = _blue_unit(env, 11)
    dead_ally["health"] = 0

    _set_blue_turn(env, 8)
    hero_mask = env.compute_action_mask()

    for idx, slot_no in enumerate(HERO_ITEM_TARGET_SLOTS):
        assert bool(hero_mask[FIRST_HERO_ITEM_ACTION_START + idx]) is (slot_no == 1)
        assert bool(hero_mask[SECOND_HERO_ITEM_ACTION_START + idx]) is (slot_no == 5)

    _set_blue_turn(env, 7)
    non_hero_mask = env.compute_action_mask()
    for idx in range(len(HERO_ITEM_TARGET_SLOTS)):
        assert bool(non_hero_mask[FIRST_HERO_ITEM_ACTION_START + idx]) is False
        assert bool(non_hero_mask[SECOND_HERO_ITEM_ACTION_START + idx]) is False


def test_hero_item_action_heals_target_clears_slot_and_ends_turn():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    env.equipped_hero_items = ["Test heal", None]
    env.hero_item_effects = {"Test heal": {"kind": "heal", "amount": 50.0}}

    hero = _set_blue_turn(env, 8)
    ally = _blue_unit(env, 7)
    max_health = int(ally.get("max_health", 0) or 0)
    ally["health"] = max_health - 35
    before_health = int(ally.get("health", 0) or 0)

    env._advance_until_blue_turn = lambda: None

    _, reward, terminated, truncated, info = env.step(FIRST_HERO_ITEM_ACTION_START)

    assert int(ally.get("health", 0) or 0) == max_health
    assert hero["initiative"] == 0
    assert env.current_blue_attacker_pos is None
    assert env.blue_attacks_left == 0
    assert env.equipped_hero_items == [None, None]
    assert info["battle_hero_item_action"] is True
    assert info["battle_hero_item_name"] == "Test heal"
    assert info["battle_hero_item_effect_kind"] == "heal"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_hero_item_effect_value"] == pytest.approx(max_health - before_health)
    assert info["equipped_hero_items"] == [None, None]
    assert reward == pytest.approx(env.reward_step)
    assert terminated is False
    assert truncated is False


def test_hero_item_action_revives_target_and_clears_second_slot():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    env.equipped_hero_items = [None, "Test revive"]
    env.hero_item_effects = {"Test revive": {"kind": "revive", "amount": 1.0}}

    _set_blue_turn(env, 8)
    ally = _blue_unit(env, 11)
    ally["health"] = 0

    env._advance_until_blue_turn = lambda: None

    action = SECOND_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(5)
    _, _, terminated, truncated, info = env.step(action)

    assert int(ally.get("health", 0) or 0) == 1
    assert env.equipped_hero_items == [None, None]
    assert info["battle_hero_item_name"] == "Test revive"
    assert info["battle_hero_item_effect_kind"] == "revive"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_hero_item_target_pos"] == 11
    assert terminated is False
    assert truncated is False


def test_hero_item_actions_follow_explicit_hero_flag_not_blue_pos8():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    env.equipped_hero_items = ["Test heal", None]
    env.hero_item_effects = {"Test heal": {"kind": "heal", "amount": 50.0}}

    promoted_ally = _blue_unit(env, 7)
    promoted_ally["hero"] = True
    promoted_ally["next_level_exp"] = 0
    promoted_ally["health"] = max(1, int(promoted_ally.get("max_health", 1) or 1) - 10)

    duke = _blue_unit(env, 8)
    duke["hero"] = False
    duke["next_level_exp"] = 500

    _set_blue_turn(env, 7)
    promoted_mask = env.compute_action_mask()
    assert bool(promoted_mask[FIRST_HERO_ITEM_ACTION_START]) is True

    _set_blue_turn(env, 8)
    duke_mask = env.compute_action_mask()
    for idx in range(len(HERO_ITEM_TARGET_SLOTS)):
        assert bool(duke_mask[FIRST_HERO_ITEM_ACTION_START + idx]) is False
        assert bool(duke_mask[SECOND_HERO_ITEM_ACTION_START + idx]) is False
