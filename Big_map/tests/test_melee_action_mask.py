"""B14: an empty melee target set must never advertise unreachable attacks."""

from copy import deepcopy

import numpy as np
import pytest

from battle_env import (
    BLUE_POSITIONS,
    DEFEND_ACTION_INDEX,
    DOUBLE_STRIKE_TYPES,
    FIRST_HERO_ITEM_ACTION_START,
    MELEE_TYPES,
    RED_POSITIONS,
    RUN_AWAY_ACTION_INDEX,
    TARGET_POSITIONS,
    TOTAL_AGENT_ACTIONS,
    WAIT_ACTION_INDEX,
    BattleEnv,
)
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

UNITS = {unit["кто"]: unit for unit in DATA}
MELEE_NAMES = sorted(name for name, unit in UNITS.items() if unit["тип"] in MELEE_TYPES)
SMALL_MELEE_NAMES = [name for name in MELEE_NAMES if not UNITS[name]["размер"]]
BIG_NAMES = sorted(name for name, unit in UNITS.items() if unit["размер"])


def unit(name, team, position, *, initiative=50):
    fighter = map_unit_to_battle(UNITS[name], team, position)
    fighter.update(
        health=10000, max_health=10000, accuracy=100, accuracy_secondary=100,
        initiative_base=initiative, immunity=[], resistance=[], armor=0,
    )
    return fighter


def make_battle(blue, red=None, *, retreat_enabled=True):
    if red is None:
        red = [unit("Рыцарь", "red", pos) for pos in RED_POSITIONS]
    occupied = {fighter["position"] for fighter in blue + red}
    battle = BattleEnv(log_enabled=True, retreat_enabled=retreat_enabled)
    battle.rng.seed(42)
    battle._init_with_custom_teams(
        red + [placeholder_unit("red", p) for p in RED_POSITIONS if p not in occupied],
        blue + [placeholder_unit("blue", p) for p in BLUE_POSITIONS if p not in occupied],
    )
    return battle


def assert_advertised_attacks_execute(battle):
    """Run real step() separately for every advertised attack, without mocks."""
    mask = battle.compute_action_mask()
    attacker = battle._unit_by_position(battle.current_blue_attacker_pos)
    expected = set(battle._warrior_allowed_targets(attacker))
    actual = {pos for i, pos in enumerate(TARGET_POSITIONS) if mask[i]}
    assert actual == expected
    assert mask.dtype == np.bool_
    assert mask.shape == (TOTAL_AGENT_ACTIONS,) == (battle.action_space.n,)
    for action in np.flatnonzero(mask[:len(TARGET_POSITIONS)]):
        trial = deepcopy(battle)
        trial.pop_pretty_events()
        _, reward, terminated, truncated, _ = trial.step(int(action))
        assert not terminated and not truncated
        assert reward == trial.reward_step
        assert not any("не может достать" in event for event in trial.pop_pretty_events())


@pytest.mark.parametrize("back_pos", [10, 11, 12])
@pytest.mark.parametrize("front_pos", [7, 8, 9])
@pytest.mark.parametrize("blocker_state", ["alive", "dead", "empty"])
@pytest.mark.parametrize("front_name", ["Рыцарь", "Лучник", "Ученик", "Служка", "Сущий"])
def test_rear_knight_live_dead_or_empty_front(back_pos, front_pos, blocker_state, front_name):
    attacker = unit("Рыцарь", "blue", back_pos, initiative=100)
    blue = [attacker]
    if blocker_state != "empty":
        blocker = unit(front_name, "blue", front_pos, initiative=90)
        if blocker_state == "dead":
            blocker["health"] = 0
        blue.append(blocker)
    battle = make_battle(blue)
    assert battle.current_blue_attacker_pos == back_pos
    mask = battle.compute_action_mask()
    assert bool(mask[:len(TARGET_POSITIONS)].any()) == (blocker_state != "alive")
    assert mask[DEFEND_ACTION_INDEX] and mask[WAIT_ACTION_INDEX]
    assert mask[RUN_AWAY_ACTION_INDEX]
    assert_advertised_attacks_execute(battle)


@pytest.mark.parametrize("name", SMALL_MELEE_NAMES)
def test_every_small_melee_unit_is_masked_when_blocked(name):
    # The complete data catalog includes all five factions and neutral units.
    battle = make_battle([
        unit(name, "blue", 10, initiative=100),
        unit("Сущий", "blue", 7, initiative=90),
    ])
    assert not battle.compute_action_mask()[:len(TARGET_POSITIONS)].any()
    assert_advertised_attacks_execute(battle)


@pytest.mark.parametrize("name", MELEE_NAMES)
def test_every_melee_unit_can_still_attack_from_front(name):
    battle = make_battle([unit(name, "blue", 7, initiative=100)])
    assert_advertised_attacks_execute(battle)


@pytest.mark.parametrize("attacker_pos", BLUE_POSITIONS)
@pytest.mark.parametrize("enemy_bits", range(1, 64))
def test_every_enemy_formation_matches_real_step(attacker_pos, enemy_bits):
    enemies = [unit("Рыцарь", "red", pos) for pos in RED_POSITIONS
               if enemy_bits & (1 << (pos - 1))]
    battle = make_battle([unit("Рыцарь", "blue", attacker_pos, initiative=100)], enemies)
    assert_advertised_attacks_execute(battle)


@pytest.mark.parametrize("attacker_pos,expected", [
    (7, {1, 2}), (8, {1, 2, 3}), (9, {2, 3}),
    (10, {1, 2}), (11, {1, 2, 3}), (12, {2, 3}),
])
def test_live_enemy_front_blocks_enemy_back(attacker_pos, expected):
    battle = make_battle([unit("Рыцарь", "blue", attacker_pos, initiative=100)])
    mask = battle.compute_action_mask()
    assert {pos for i, pos in enumerate(TARGET_POSITIONS) if mask[i]} == expected
    for pos in (1, 2, 3):
        battle._unit_by_position(pos)["health"] = 0
    mask = battle.compute_action_mask()
    assert {pos for i, pos in enumerate(TARGET_POSITIONS) if mask[i]} == {p + 3 for p in expected}
    assert_advertised_attacks_execute(battle)


@pytest.mark.parametrize("name", BIG_NAMES)
@pytest.mark.parametrize("anchor", [8, 11])
def test_large_ally_blocks_rear_without_overlapping_cells(name, anchor):
    # Both anchors occupy pos8 and pos11; the rear attacker is at pos10.
    battle = make_battle([
        unit("Рыцарь", "blue", 10, initiative=100),
        unit(name, "blue", anchor, initiative=90),
    ])
    assert not battle.compute_action_mask()[:len(TARGET_POSITIONS)].any()
    assert battle._unit_by_position(19 - anchor)["health"] == 0
    battle._unit_by_position(anchor)["health"] = 0
    assert battle.compute_action_mask()[:len(TARGET_POSITIONS)].any()
    assert_advertised_attacks_execute(battle)


def test_large_enemy_uses_one_target_and_reserves_rear_cell():
    battle = make_battle([unit("Рыцарь", "blue", 7, initiative=100)], [
        unit("Астерот", "red", 1), unit("Рыцарь", "red", 5),
    ])
    mask = battle.compute_action_mask()
    assert list(np.flatnonzero(mask[:len(TARGET_POSITIONS)])) == [0]
    assert_advertised_attacks_execute(battle)


@pytest.mark.parametrize("retreat_enabled", [False, True])
@pytest.mark.parametrize("waited", [0, 1])
def test_blocked_unit_preserves_other_action_rules(retreat_enabled, waited):
    battle = make_battle([
        unit("Рыцарь", "blue", 10, initiative=100),
        unit("Рыцарь", "blue", 7, initiative=90),
    ], retreat_enabled=retreat_enabled)
    battle._unit_by_position(10)["waited"] = waited
    mask = battle.compute_action_mask()
    assert not mask[:len(TARGET_POSITIONS)].any()
    assert mask[DEFEND_ACTION_INDEX]
    assert bool(mask[WAIT_ACTION_INDEX]) == (waited == 0)
    assert bool(mask[RUN_AWAY_ACTION_INDEX]) == retreat_enabled
    assert not mask[FIRST_HERO_ITEM_ACTION_START:].any()
    assert mask.any()
    _, reward, _, _, info = battle.step(DEFEND_ACTION_INDEX)
    assert info["battle_defend_applied"]
    assert reward == battle.reward_step


@pytest.mark.parametrize("front_name", ["Рыцарь", "Сущий", "Тиамат", "Драколич"])
def test_forced_unreachable_action_keeps_existing_execution_penalty(front_name):
    battle = make_battle([
        unit("Имперский рыцарь", "blue", 10, initiative=100),
        unit(front_name, "blue", 8, initiative=90),
    ], [unit("Астерот", "red", 1)])
    hp_before = battle._unit_by_position(1)["health"]
    assert not battle.compute_action_mask()[:len(TARGET_POSITIONS)].any()
    _, reward, _, _, _ = battle.step(0)
    assert reward == battle.reward_step + battle.penalty_unreachable_warrior
    assert battle._unit_by_position(1)["health"] == hp_before
    assert battle._unit_by_position(10)["initiative"] == 0
    assert battle.current_blue_attacker_pos == 8


@pytest.mark.parametrize("route", ["wight", "lycanthropy"])
@pytest.mark.parametrize("front_pos", [7, 8, 9])
@pytest.mark.parametrize("back_pos", [10, 11, 12])
def test_transformed_rear_knight_stays_masked_behind_living_squire(route, front_pos, back_pos):
    battle = make_battle([
        unit("Рыцарь", "blue", back_pos, initiative=100),
        unit("Скваер", "blue", front_pos, initiative=90),
    ], [unit("Астерот", "red", 2), unit("Сущий", "red", 1)])
    target = battle._unit_by_position(back_pos)
    caster = battle._unit_by_position(1)
    if route == "wight":
        assert battle._apply_wight_decay_effect(caster, target)
    else:
        assert battle._apply_hero_item_lycanthropy(
            source_unit=caster, target_unit=target,
            effect={"transform_unit_name": "Оборотень"},
        ) == 1
    assert target["position"] == back_pos
    assert target["stand"] == "behind"
    assert battle.current_blue_attacker_pos == back_pos
    mask = battle.compute_action_mask()
    assert not mask[:len(TARGET_POSITIONS)].any()
    assert mask[DEFEND_ACTION_INDEX] and mask[WAIT_ACTION_INDEX]
    trial = deepcopy(battle)
    hp_before = trial._unit_by_position(2)["health"]
    _, reward, _, _, _ = trial.step(TARGET_POSITIONS.index(2))
    assert reward == trial.reward_step + trial.penalty_unreachable_warrior
    assert trial._unit_by_position(2)["health"] == hp_before
    assert any("не может достать" in event for event in trial.pop_pretty_events())
    battle._unit_by_position(front_pos)["health"] = 0
    assert battle.compute_action_mask()[TARGET_POSITIONS.index(2)]


@pytest.mark.parametrize("name", ["Лучник", "Ученик", "Служка"])
def test_ranged_magic_and_healing_are_not_blocked_by_front_melee(name):
    battle = make_battle([
        unit(name, "blue", 10, initiative=100),
        unit("Рыцарь", "blue", 7, initiative=90),
    ])
    ally = battle._unit_by_position(7)
    ally["health"] -= 500
    mask = battle.compute_action_mask()
    expected = {0, 3} if name == "Служка" else set(range(6))
    assert set(np.flatnonzero(mask[:len(TARGET_POSITIONS)])) == expected
    hp_before = ally["health"]
    enemy_hp_before = battle._unit_by_position(1)["health"]
    _, reward, _, _, _ = battle.step(0)
    assert reward == battle.reward_step
    if name == "Служка":
        assert ally["health"] > hp_before
    else:
        assert battle._unit_by_position(1)["health"] < enemy_hp_before


@pytest.mark.parametrize("name", ["Мастер клинка", "Астерот", "Бандит"])
def test_double_strike_recomputes_targets_and_preserves_second_strike_rules(name):
    battle = make_battle([unit(name, "blue", 7, initiative=100)], [
        unit("Рыцарь", "red", 1), unit("Рыцарь", "red", 6),
    ])
    attacker = battle._unit_by_position(7)
    assert attacker["unit_type"] in DOUBLE_STRIKE_TYPES
    battle._unit_by_position(1)["health"] = 1
    assert battle.compute_action_mask()[0]
    _, reward, terminated, _, _ = battle.step(0)
    assert not terminated and reward == battle.reward_step
    assert battle.blue_attacks_left == 1
    mask = battle.compute_action_mask()
    assert mask[5] and not mask[0]
    assert not mask[DEFEND_ACTION_INDEX:RUN_AWAY_ACTION_INDEX + 1].any()
    assert mask.any()
    _, reward, terminated, _, _ = battle.step(5)
    assert not terminated and reward == battle.reward_step


@pytest.mark.parametrize("name", ["Рыцарь", "Мастер клинка", "Скелет призрак"])
@pytest.mark.parametrize("front_name", ["Рыцарь", "Сущий", "Тиамат"])
def test_red_auto_melee_also_respects_blocked_rear(name, front_name):
    battle = make_battle([unit("Рыцарь", "blue", 7, initiative=200)], [
        unit(name, "red", 4, initiative=150),
        unit(front_name, "red", 2, initiative=100),
    ])
    rear = battle._unit_by_position(4)
    assert battle._warrior_allowed_targets(rear) == []
    battle.pop_pretty_events()
    battle.step(DEFEND_ACTION_INDEX)
    events = battle.pop_pretty_events()
    assert any(f"{name}#4" in event and "не может атаковать" in event for event in events)
    assert not any(f"{name}#4" in event and "> цель" in event for event in events)


@pytest.mark.parametrize("front_name", sorted(UNITS))
def test_every_living_front_unit_masks_rear_melee_attacks(front_name):
    battle = make_battle([
        unit("Имперский рыцарь", "blue", 10, initiative=100),
        unit(front_name, "blue", 8, initiative=90),
    ])
    if battle.round_no == 0:
        # This checks the knight's normal turn, after Doppelganger preparation.
        battle.step(DEFEND_ACTION_INDEX)
    assert battle.current_blue_attacker_pos == 10
    mask = battle.compute_action_mask()
    assert not mask[:len(TARGET_POSITIONS)].any()
    assert mask[DEFEND_ACTION_INDEX] and mask[WAIT_ACTION_INDEX]
    battle._unit_by_position(8)["health"] = 0
    assert battle.compute_action_mask()[:len(TARGET_POSITIONS)].any()
    assert_advertised_attacks_execute(battle)


@pytest.mark.parametrize("rear_name", ["Рыцарь", "Сущий"])
def test_small_rear_ally_does_not_block_rear_melee(rear_name):
    battle = make_battle([
        unit("Имперский рыцарь", "blue", 10, initiative=100),
        unit(rear_name, "blue", 11, initiative=90),
    ])
    assert battle.compute_action_mask()[:len(TARGET_POSITIONS)].any()
    assert_advertised_attacks_execute(battle)


def test_large_rear_anchored_attacker_does_not_block_itself():
    battle = make_battle([unit("Астерот", "blue", 11, initiative=100)])
    assert battle.compute_action_mask()[:len(TARGET_POSITIONS)].any()
    assert_advertised_attacks_execute(battle)


def test_blocked_melee_hero_can_still_use_an_available_item():
    battle = make_battle([
        unit("Герцог", "blue", 10, initiative=100),
        unit("Рыцарь", "blue", 7, initiative=90),
    ])
    battle.equipped_hero_items = ["B14 healing fixture", None]
    battle.hero_item_effects = {
        "B14 healing fixture": {"kind": "heal", "amount": 50, "target_team": "blue"},
    }
    hero = battle._unit_by_position(10)
    hero["health"] -= 500
    mask = battle.compute_action_mask()
    assert not mask[:len(TARGET_POSITIONS)].any()
    item_action = FIRST_HERO_ITEM_ACTION_START + 3
    assert mask[item_action]
    hp_before = hero["health"]
    _, reward, _, _, info = battle.step(item_action)
    assert reward == battle.reward_step
    assert info["battle_hero_item_applied"]
    assert hero["health"] == hp_before + 50


def test_no_living_enemy_does_not_reenable_melee_targets():
    battle = make_battle([unit("Рыцарь", "blue", 7, initiative=100)])
    for fighter in battle.combined:
        if fighter["team"] == "red":
            fighter["health"] = 0
    mask = battle.compute_action_mask()
    assert not mask[:len(TARGET_POSITIONS)].any()
    assert mask[DEFEND_ACTION_INDEX]


@pytest.mark.parametrize("action", [DEFEND_ACTION_INDEX, WAIT_ACTION_INDEX, RUN_AWAY_ACTION_INDEX])
def test_blocked_double_striker_alternative_never_leaves_empty_second_strike(action):
    battle = make_battle([
        unit("Мастер клинка", "blue", 10, initiative=100),
        unit("Рыцарь", "blue", 7, initiative=90),
    ])
    assert battle.blue_attacks_left == 2
    mask = battle.compute_action_mask()
    assert not mask[:len(TARGET_POSITIONS)].any()
    assert mask[action]
    _, reward, terminated, truncated, _ = battle.step(action)
    assert not terminated and not truncated
    assert reward == battle.reward_step
    assert battle.current_blue_attacker_pos == 7
    assert battle.compute_action_mask().any()
