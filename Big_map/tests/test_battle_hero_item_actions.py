import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import (
    BattleEnv,
    FIRST_HERO_ITEM_ACTION_START,
    FIRST_HERO_ITEM_ENEMY_ACTION_START,
    HERO_ITEM_TARGET_SLOTS,
    SECOND_HERO_ITEM_ACTION_START,
    SECOND_HERO_ITEM_ENEMY_ACTION_START,
    TOTAL_AGENT_ACTIONS,
)
from campaign_env import CampaignEnv


def _blue_unit(env: BattleEnv, position: int) -> dict:
    return next(
        unit
        for unit in env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1)) == int(position)
    )


def _red_unit(env: BattleEnv, position: int) -> dict:
    return next(
        unit
        for unit in env.combined
        if unit.get("team") == "red" and int(unit.get("position", -1)) == int(position)
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


def _configure_campaign_orbs(env: BattleEnv) -> None:
    env.hero_item_effects = dict(CampaignEnv._battle_item_effect_definitions())


def _use_first_item(env: BattleEnv, item_name: str, action: int):
    _configure_campaign_orbs(env)
    env.equipped_hero_items = [item_name, None]
    _set_blue_turn(env, 8)
    env._advance_until_blue_turn = lambda: None
    return env.step(action)


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
        assert bool(hero_mask[FIRST_HERO_ITEM_ACTION_START + idx]) is (
            slot_no in (1, 2, 3)
        )
        assert bool(hero_mask[SECOND_HERO_ITEM_ACTION_START + idx]) is (slot_no == 5)

    _set_blue_turn(env, 7)
    non_hero_mask = env.compute_action_mask()
    for idx in range(len(HERO_ITEM_TARGET_SLOTS)):
        assert bool(non_hero_mask[FIRST_HERO_ITEM_ACTION_START + idx]) is False
        assert bool(non_hero_mask[SECOND_HERO_ITEM_ACTION_START + idx]) is False


def test_hero_item_enemy_actions_mask_for_red_targets_only():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    env.equipped_hero_items = ["Test damage", "Test damage 2"]
    env.hero_item_effects = {
        "Test damage": {"kind": "damage", "amount": 30.0},
        "Test damage 2": {"kind": "damage", "amount": 30.0},
    }

    for unit in env.combined:
        if unit.get("team") == "red":
            unit["health"] = 0
    red_target = _red_unit(env, 1)
    red_target["health"] = max(1, int(red_target.get("max_health", 1) or 1))

    _set_blue_turn(env, 8)
    mask = env.compute_action_mask()

    assert env.action_space.n == TOTAL_AGENT_ACTIONS == 39
    for idx, slot_no in enumerate(HERO_ITEM_TARGET_SLOTS):
        assert bool(mask[FIRST_HERO_ITEM_ACTION_START + idx]) is False
        assert bool(mask[SECOND_HERO_ITEM_ACTION_START + idx]) is False
        assert bool(mask[FIRST_HERO_ITEM_ENEMY_ACTION_START + idx]) is (slot_no == 1)
        assert bool(mask[SECOND_HERO_ITEM_ENEMY_ACTION_START + idx]) is (slot_no == 1)


def test_hero_item_enemy_action_damages_target_clears_slot_and_ends_turn():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    env.equipped_hero_items = ["Test damage", None]
    env.hero_item_effects = {"Test damage": {"kind": "damage", "amount": 25.0}}

    hero = _set_blue_turn(env, 8)
    target = _red_unit(env, 1)
    target["health"] = max(50, int(target.get("max_health", 1) or 1))
    before_health = int(target.get("health", 0) or 0)

    env._advance_until_blue_turn = lambda: None

    _, reward, terminated, truncated, info = env.step(FIRST_HERO_ITEM_ENEMY_ACTION_START)

    assert int(target.get("health", 0) or 0) == before_health - 25
    assert hero["initiative"] == 0
    assert env.current_blue_attacker_pos is None
    assert env.blue_attacks_left == 0
    assert env.equipped_hero_items == [None, None]
    assert info["battle_hero_item_action"] is True
    assert info["battle_hero_item_slot"] == 1
    assert info["battle_hero_item_target_pos"] == 1
    assert info["battle_hero_item_target_team"] == "red"
    assert info["battle_hero_item_name"] == "Test damage"
    assert info["battle_hero_item_effect_kind"] == "damage"
    assert info["battle_hero_item_effect_value"] == pytest.approx(25.0)
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert reward == pytest.approx(env.reward_step)
    assert terminated is False
    assert truncated is False


def test_second_slot_hero_item_enemy_action_targets_red_position():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    env.equipped_hero_items = [None, "Test damage"]
    env.hero_item_effects = {"Test damage": {"kind": "damage", "amount": 10.0}}

    _set_blue_turn(env, 8)
    target = _red_unit(env, 2)
    target["health"] = max(50, int(target.get("max_health", 1) or 1))
    before_health = int(target.get("health", 0) or 0)

    env._advance_until_blue_turn = lambda: None

    action = SECOND_HERO_ITEM_ENEMY_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(2)
    _, _, terminated, truncated, info = env.step(action)

    assert int(target.get("health", 0) or 0) == before_health - 10
    assert env.equipped_hero_items == [None, None]
    assert info["battle_hero_item_slot"] == 2
    assert info["battle_hero_item_target_pos"] == 2
    assert info["battle_hero_item_target_team"] == "red"
    assert info["battle_hero_item_effect_kind"] == "damage"
    assert info["battle_hero_item_consumed"] is True
    assert terminated is False
    assert truncated is False


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


def test_non_hero_item_action_is_safe_noop_if_mask_is_bypassed():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    env.equipped_hero_items = ["Test heal", None]
    env.hero_item_effects = {"Test heal": {"kind": "heal", "amount": 50.0}}

    attacker = _set_blue_turn(env, 7)
    attacker["hero"] = False
    env._advance_until_blue_turn = lambda: None

    _, reward, terminated, truncated, info = env.step(FIRST_HERO_ITEM_ACTION_START)

    assert attacker["initiative"] == 0
    assert info["battle_hero_item_action"] is True
    assert info["battle_hero_item_name"] == "Test heal"
    assert info["battle_hero_item_applied"] is False
    assert info["battle_hero_item_consumed"] is False
    assert env.equipped_hero_items == ["Test heal", None]
    assert reward == pytest.approx(env.reward_step)
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


def test_orb_masks_are_split_by_slot_team_and_target_category():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    _configure_campaign_orbs(env)
    env.equipped_hero_items = ["Vampire Orb", "Orb of Fire"]

    dead_ally = _blue_unit(env, 11)
    dead_ally["health"] = 0
    for unit in env.combined:
        if unit.get("team") == "red" and int(unit.get("position", 0)) != 1:
            unit["health"] = 0

    _set_blue_turn(env, 8)
    mask = env.compute_action_mask()

    for idx, slot_no in enumerate(HERO_ITEM_TARGET_SLOTS):
        assert bool(mask[FIRST_HERO_ITEM_ACTION_START + idx]) is (
            slot_no in (4, 5, 6)
        )
        assert bool(mask[FIRST_HERO_ITEM_ENEMY_ACTION_START + idx]) is False
        assert bool(mask[SECOND_HERO_ITEM_ACTION_START + idx]) is False
        assert bool(mask[SECOND_HERO_ITEM_ENEMY_ACTION_START + idx]) is (slot_no == 1)


def test_party_heal_mask_allows_any_living_ally_even_at_full_hp():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    _configure_campaign_orbs(env)
    env.equipped_hero_items = ["Orb of Restoration", None]
    _set_blue_turn(env, 8)

    full_mask = env.compute_action_mask()
    for idx, slot_no in enumerate(HERO_ITEM_TARGET_SLOTS):
        assert bool(full_mask[FIRST_HERO_ITEM_ACTION_START + idx]) is (
            slot_no in (1, 2, 3, 5)
        )

    damaged_ally = _blue_unit(env, 7)
    damaged_ally["health"] = int(damaged_ally["max_health"]) - 10

    damaged_mask = env.compute_action_mask()
    for idx, slot_no in enumerate(HERO_ITEM_TARGET_SLOTS):
        assert bool(damaged_mask[FIRST_HERO_ITEM_ACTION_START + idx]) is (
            slot_no in (1, 2, 3, 5)
        )
        assert bool(damaged_mask[FIRST_HERO_ITEM_ENEMY_ACTION_START + idx]) is False


@pytest.mark.parametrize(
    ("item_name", "expected_name"),
    [
        ("Vampire Orb", "Вампир"),
        ("Lich Orb", "Лич"),
    ],
)
def test_summon_orbs_create_temporary_blue_units(item_name: str, expected_name: str):
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)

    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(4)
    _, _, terminated, truncated, info = _use_first_item(env, item_name, action)

    summoned = _blue_unit(env, 10)
    assert summoned["name"] == expected_name
    assert summoned["Summoned"] == 8
    assert int(summoned["health"]) == int(summoned["max_health"])
    assert info["battle_hero_item_effect_kind"] == "summon"
    assert info["battle_hero_item_consumed"] is True
    assert env.equipped_hero_items == [None, None]
    assert terminated is False
    assert truncated is False


def test_talisman_spends_charge_once_per_battle_and_keeps_slot():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    _configure_campaign_orbs(env)
    env.equipped_hero_items = ["Zombie Talisman", None]
    env.equipped_hero_item_uses_left = [5, None]

    _set_blue_turn(env, 8)
    env._advance_until_blue_turn = lambda: None
    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(4)
    _, _, terminated, truncated, info = env.step(action)

    summoned = _blue_unit(env, 10)
    expected_name = CampaignEnv._battle_item_effect_definitions()["Zombie Talisman"][
        "summon_unit_name"
    ]
    assert summoned["name"] == expected_name
    assert info["battle_hero_item_effect_kind"] == "summon"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is False
    assert info["battle_hero_item_charge_spent"] is True
    assert info["battle_hero_item_uses_left"] == 4
    assert info["equipped_hero_items"] == ["Zombie Talisman", None]
    assert info["equipped_hero_item_uses_left"] == [4, None]
    assert env.equipped_hero_items == ["Zombie Talisman", None]
    assert env.equipped_hero_item_uses_left == [4, None]

    _set_blue_turn(env, 8)
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False
    assert terminated is False
    assert truncated is False


def test_talisman_final_charge_clears_slot():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    _configure_campaign_orbs(env)
    env.equipped_hero_items = ["Zombie Talisman", None]
    env.equipped_hero_item_uses_left = [1, None]

    _set_blue_turn(env, 8)
    env._advance_until_blue_turn = lambda: None
    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(4)
    _, _, _, _, info = env.step(action)

    assert info["battle_hero_item_effect_kind"] == "summon"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert info["battle_hero_item_charge_spent"] is True
    assert info["battle_hero_item_uses_left"] == 0
    assert env.equipped_hero_items == [None, None]
    assert env.equipped_hero_item_uses_left == [None, None]


def test_orb_of_life_revives_dead_blue_unit():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    ally = _blue_unit(env, 11)
    ally["health"] = 0

    action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(5)
    _, _, _, _, info = _use_first_item(env, "Orb of Life", action)

    assert int(ally["health"]) == 1
    assert info["battle_hero_item_effect_kind"] == "revive"
    assert info["battle_hero_item_consumed"] is True


def test_party_heal_orb_heals_all_wounded_blue_units():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    ally_a = _blue_unit(env, 7)
    ally_b = _blue_unit(env, 11)
    ally_a["health"] = int(ally_a["max_health"]) - 20
    ally_b["health"] = int(ally_b["max_health"]) - 40

    _, _, _, _, info = _use_first_item(env, "Orb of Restoration", FIRST_HERO_ITEM_ACTION_START)

    assert int(ally_a["health"]) == int(ally_a["max_health"])
    assert int(ally_b["health"]) == int(ally_b["max_health"]) - 10
    assert info["battle_hero_item_effect_kind"] == "heal"
    assert info["battle_hero_item_effect_value"] == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("blocked_field", "blocked_values"),
    [
        ("immunity", ["Fire"]),
        ("resistance", ["Fire"]),
    ],
)
def test_typed_damage_blocked_by_immunity_or_protection_still_consumes_orb(
    blocked_field: str,
    blocked_values: list[str],
):
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    target = _red_unit(env, 1)
    target["health"] = 100
    target[blocked_field] = list(blocked_values)

    _, _, _, _, info = _use_first_item(env, "Orb of Fire", FIRST_HERO_ITEM_ENEMY_ACTION_START)

    assert int(target["health"]) == 100
    assert info["battle_hero_item_effect_kind"] == "damage"
    assert info["battle_hero_item_effect_value"] == pytest.approx(0.0)
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert env.equipped_hero_items == [None, None]


def test_heal_orb_on_full_hp_ally_still_consumes_orb():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    ally = _blue_unit(env, 7)

    _, _, _, _, info = _use_first_item(env, "Orb of Healing", FIRST_HERO_ITEM_ACTION_START)

    assert int(ally["health"]) == int(ally["max_health"])
    assert info["battle_hero_item_effect_kind"] == "heal"
    assert info["battle_hero_item_effect_value"] == pytest.approx(0.0)
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_consumed"] is True
    assert env.equipped_hero_items == [None, None]


def test_damage_orb_wrong_team_window_is_safe_noop():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    before_items = ["Orb of Fire", None]
    _configure_campaign_orbs(env)
    env.equipped_hero_items = list(before_items)
    _set_blue_turn(env, 8)
    env._advance_until_blue_turn = lambda: None

    _, _, _, _, info = env.step(FIRST_HERO_ITEM_ACTION_START)

    assert info["battle_hero_item_applied"] is False
    assert info["battle_hero_item_consumed"] is False
    assert env.equipped_hero_items == before_items


def test_drain_orb_heals_hero_and_elder_overflow_heals_allies():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _restore_blue_health(env)
    hero = _blue_unit(env, 8)
    ally = _blue_unit(env, 7)
    target = _red_unit(env, 1)
    hero["health"] = int(hero["max_health"]) - 10
    ally["health"] = int(ally["max_health"]) - 50
    target["health"] = 100
    target["immunity"] = []

    _, _, _, _, info = _use_first_item(
        env,
        "Orb of Elder Vampires",
        FIRST_HERO_ITEM_ENEMY_ACTION_START,
    )

    assert int(target["health"]) == 25
    assert int(hero["health"]) == int(hero["max_health"])
    assert int(ally["health"]) == int(ally["max_health"])
    assert info["battle_hero_item_effect_kind"] == "drain"
    assert info["battle_hero_item_effect_value"] == pytest.approx(75.0)


@pytest.mark.parametrize(
    ("item_name", "turns_key", "damage_key", "damage"),
    [
        ("Orb of Thanatos", "poison_turns_left", "poison_damage_per_tick", 30),
        ("Orb of Freezing", "uran_turns_left", "uran_damage_per_tick", 30),
        ("Orb of Flame", "burn_turns_left", "burn_damage_per_tick", 15),
    ],
)
def test_group_dot_orbs_apply_to_all_living_red_units(
    item_name: str,
    turns_key: str,
    damage_key: str,
    damage: int,
):
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)

    _, _, _, _, info = _use_first_item(env, item_name, FIRST_HERO_ITEM_ENEMY_ACTION_START)

    living_red = [
        unit for unit in env.combined if unit.get("team") == "red" and env._alive(unit)
    ]
    assert living_red
    assert all(int(unit.get(turns_key, 0) or 0) >= 1 for unit in living_red)
    assert all(int(unit.get(damage_key, 0) or 0) == damage for unit in living_red)
    assert info["battle_hero_item_effect_kind"] == "dot"
    assert info["battle_hero_item_effect_value"] == pytest.approx(float(damage * len(living_red)))


def test_status_transform_weakening_and_rage_orbs_apply():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    target = _red_unit(env, 1)
    _, _, _, _, fear_info = _use_first_item(
        env,
        "Orb of Fear",
        FIRST_HERO_ITEM_ENEMY_ACTION_START,
    )
    assert target["paralyzed"] == 1
    assert fear_info["battle_hero_item_effect_kind"] == "paralysis"

    env.reset(seed=123)
    target = _red_unit(env, 1)
    _, _, _, _, witch_info = _use_first_item(
        env,
        "Orb of Witches",
        FIRST_HERO_ITEM_ENEMY_ACTION_START,
    )
    assert target["transformed"] == 1
    assert target["unit_type"] == "Warrior"
    assert witch_info["battle_hero_item_effect_kind"] == "transform"

    env.reset(seed=123)
    target = _red_unit(env, 1)
    _, _, _, _, lycan_info = _use_first_item(
        env,
        "Orb of Lycanthropy",
        FIRST_HERO_ITEM_ENEMY_ACTION_START,
    )
    assert target["name"] == "Оборотень"
    assert target["damage"] == 40
    assert target["initiative_base"] == 50
    assert lycan_info["battle_hero_item_effect_kind"] == "lycanthropy"

    env.reset(seed=123)
    target = _red_unit(env, 1)
    target["damage"] = 90
    _, _, _, _, weak_info = _use_first_item(
        env,
        "Orb of Weakening",
        FIRST_HERO_ITEM_ENEMY_ACTION_START,
    )
    assert target["damage"] == 60
    assert weak_info["battle_hero_item_effect_kind"] == "debuff_damage"

    env.reset(seed=123)
    ally = _blue_unit(env, 7)
    ally["initiative"] = 0
    ally["bonusturn"] = 0
    _, _, _, _, rage_info = _use_first_item(env, "Orb of Rage", FIRST_HERO_ITEM_ACTION_START)
    assert ally["initiative"] == ally["initiative_base"]
    assert ally["bonusturn"] == 1
    assert rage_info["battle_hero_item_effect_kind"] == "extra_turn"


def test_talisman_of_nightmare_fears_single_living_red_target():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _configure_campaign_orbs(env)
    env.equipped_hero_items = ["Talisman of Nightmare", None]
    env.equipped_hero_item_uses_left = [5, None]

    for unit in env.combined:
        if unit.get("team") == "red" and int(unit.get("position", 0)) != 1:
            unit["health"] = 0
    target = _red_unit(env, 1)
    target["health"] = max(1, int(target.get("max_health", 1) or 1))

    _set_blue_turn(env, 8)
    mask = env.compute_action_mask()
    assert bool(mask[FIRST_HERO_ITEM_ACTION_START]) is False
    assert bool(mask[FIRST_HERO_ITEM_ENEMY_ACTION_START]) is True

    env._advance_until_blue_turn = lambda: None
    _, _, _, _, info = env.step(FIRST_HERO_ITEM_ENEMY_ACTION_START)

    assert target["running_away"] == 1
    assert info["battle_hero_item_effect_kind"] == "fear"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_effect_value"] == pytest.approx(1.0)
    assert info["battle_hero_item_consumed"] is False
    assert info["battle_hero_item_charge_spent"] is True
    assert info["battle_hero_item_uses_left"] == 4


def test_talisman_of_nightmare_mind_resistance_blocks_fear_but_spends_charge():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    _configure_campaign_orbs(env)
    env.equipped_hero_items = ["Talisman of Nightmare", None]
    env.equipped_hero_item_uses_left = [5, None]
    target = _red_unit(env, 1)
    target["health"] = max(1, int(target.get("max_health", 1) or 1))
    target["resistance"] = ["Mind"]

    _set_blue_turn(env, 8)
    env._advance_until_blue_turn = lambda: None
    _, _, _, _, info = env.step(FIRST_HERO_ITEM_ENEMY_ACTION_START)

    assert target["running_away"] == 0
    assert info["battle_hero_item_effect_kind"] == "fear"
    assert info["battle_hero_item_applied"] is True
    assert info["battle_hero_item_effect_value"] == pytest.approx(0.0)
    assert info["battle_hero_item_consumed"] is False
    assert info["battle_hero_item_charge_spent"] is True
    assert info["battle_hero_item_uses_left"] == 4
    assert env.equipped_hero_items == ["Talisman of Nightmare", None]
