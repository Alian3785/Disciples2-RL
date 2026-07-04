import sys
from pathlib import Path
from copy import deepcopy

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _unit_by_position(env: CampaignEnv, position: int) -> dict:
    for unit in env._get_blue_state():
        if int(unit.get("position", -1) or -1) == int(position):
            return unit
    raise AssertionError(f"No BLUE unit at position {position}")


def _assert_unit_swap_penalty(env: CampaignEnv, reward: float, info: dict) -> None:
    expected_reward = -float(env.reward_unit_swap_penalty)
    assert reward == pytest.approx(expected_reward)
    assert info["unit_swap_penalty"] == pytest.approx(float(env.reward_unit_swap_penalty))
    assert info["unit_swap_reward"] == pytest.approx(expected_reward)


def _put_real_unit_at_position(env: CampaignEnv, position: int, source_position: int = 7) -> dict:
    recruited = deepcopy(_unit_by_position(env, source_position))
    recruited["name"] = f"Hired slot {int(position)}"
    recruited["position"] = int(position)
    recruited["stand"] = env._blue_stand_for_position(int(position))
    for idx, unit in enumerate(env.blue_team_state):
        if int(unit.get("position", -1) or -1) == int(position):
            env.blue_team_state[idx] = recruited
            return recruited
    env.blue_team_state.append(recruited)
    env.blue_team_state.sort(key=lambda unit: int(unit.get("position", 999) or 999))
    return recruited


def _put_empty_unit_at_position(env: CampaignEnv, position: int) -> dict:
    empty = {
        "name": "пусто",
        "initiative": 0,
        "initiative_base": 0,
        "team": "blue",
        "position": int(position),
        "stand": env._blue_stand_for_position(int(position)),
        "unit_type": "Archer",
        "damage": 0,
        "damage_secondary": 0,
        "health": 0,
        "max_health": 0,
        "armor": 0,
        "accuracy": 0,
        "accuracy_secondary": 0,
        "immunity": [],
        "resistance": [],
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "transformed": 0,
        "hero": False,
        "basestats": [],
        "Level": 0,
        "needaunit": 0,
    }
    for idx, unit in enumerate(env.blue_team_state):
        if int(unit.get("position", -1) or -1) == int(position):
            env.blue_team_state[idx] = empty
            return empty
    env.blue_team_state.append(empty)
    env.blue_team_state.sort(key=lambda unit: int(unit.get("position", 999) or 999))
    return empty


def _make_big_unit_at_position(env: CampaignEnv, position: int) -> dict:
    big = _unit_by_position(env, position)
    big["name"] = f"Big slot {int(position)}"
    big["big"] = True
    big["health"] = max(float(big.get("health", 0) or 0), 300.0)
    big["max_health"] = max(float(big.get("max_health", 0) or 0), 300.0)
    big["hp"] = big["health"]
    big["maxhp"] = big["max_health"]
    big["position"] = int(position)
    big["stand"] = env._blue_stand_for_position(int(position))
    _put_empty_unit_at_position(env, env._blue_partner_position(int(position)))
    return big


def test_grid_unit_swap_actions_cover_initial_occupied_positions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env.GRID_SWAP_UNIT_PAIRS == (
        (7, 8),
        (7, 9),
        (7, 11),
        (8, 9),
        (8, 11),
        (9, 11),
        (7, 10),
        (8, 10),
        (9, 10),
        (10, 11),
        (7, 12),
        (8, 12),
        (9, 12),
        (10, 12),
        (11, 12),
    )
    assert env.GRID_SWAP_UNIT_ACTION_COUNT == 15
    assert env.action_space.n == (
        env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_ACTION_COUNT
    )

    mask = env.compute_action_mask()
    swap_slice = mask[
        env.GRID_SWAP_UNIT_ACTION_START : env.GRID_SWAP_UNIT_ACTION_START
        + env.GRID_SWAP_UNIT_ACTION_COUNT
    ]
    assert swap_slice.tolist() == [True] * 13 + [False, True]


def test_grid_unit_swap_moves_dead_units_and_is_once_per_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    action = env.GRID_SWAP_UNIT_ACTION_START
    pos_a, pos_b = env.GRID_SWAP_UNIT_PAIRS[0]
    unit_a = _unit_by_position(env, pos_a)
    unit_b = _unit_by_position(env, pos_b)
    name_a = unit_a["name"]
    name_b = unit_b["name"]
    max_hp_a = unit_a.get("maxhp", unit_a.get("max_health"))
    hp_b = unit_b.get("hp", unit_b.get("health"))

    unit_a["hp"] = 0.0
    unit_a["health"] = 0.0
    env.active_strength_potion_positions.add(int(pos_a))
    env.active_haste_elixir_positions.add(int(pos_b))
    moves_before = env.moves
    turns_before = env.turns

    _obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["unit_swap_action"] is True
    assert info["unit_swap_applied"] is True
    _assert_unit_swap_penalty(env, reward, info)
    assert info["unit_swap_positions"] == (pos_a, pos_b)
    assert env.moves == moves_before
    assert env.turns == turns_before
    assert env.grid_unit_swaps_this_turn == 1
    assert env.grid_unit_swaps_total == 1

    new_a = _unit_by_position(env, pos_a)
    new_b = _unit_by_position(env, pos_b)
    assert new_a["name"] == name_b
    assert new_a.get("hp", new_a.get("health")) == hp_b
    assert new_a["stand"] == "ahead"
    assert new_b["name"] == name_a
    assert new_b.get("hp", new_b.get("health")) == 0.0
    assert new_b.get("maxhp", new_b.get("max_health")) == max_hp_a
    assert new_b["stand"] == "ahead"
    assert int(pos_b) in env.active_strength_potion_positions
    assert int(pos_a) not in env.active_strength_potion_positions
    assert int(pos_a) in env.active_haste_elixir_positions
    assert int(pos_b) not in env.active_haste_elixir_positions

    assert bool(env.compute_action_mask()[action]) is False

    env._advance_turns(1)
    assert env.grid_unit_swaps_this_turn == 0
    assert bool(env.compute_action_mask()[action]) is True


def test_grid_unit_swap_actions_mask_after_three_swaps_per_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    swap_start = env.GRID_SWAP_UNIT_ACTION_START
    swap_stop = swap_start + env.GRID_SWAP_UNIT_ACTION_COUNT

    for expected_count in range(1, 4):
        mask = env.compute_action_mask()
        available_offsets = [
            offset
            for offset, available in enumerate(mask[swap_start:swap_stop])
            if bool(available)
        ]
        assert available_offsets

        action = swap_start + available_offsets[0]
        _obs, reward, terminated, truncated, info = env.step(action)

        assert terminated is False
        assert truncated is False
        assert info["unit_swap_action"] is True
        assert info["unit_swap_applied"] is True
        _assert_unit_swap_penalty(env, reward, info)
        assert env.grid_unit_swaps_this_turn == expected_count
        assert env.grid_unit_swaps_total == expected_count

    mask = env.compute_action_mask()
    assert not any(bool(value) for value in mask[swap_start:swap_stop])

    _obs, reward, terminated, truncated, info = env.step(swap_start)
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info["unit_swap_action"] is True
    assert info["unit_swap_applied"] is False
    assert info["unit_swap_invalid_reason"] == "masked_or_missing_unit"
    assert env.grid_unit_swaps_this_turn == 3
    assert env.grid_unit_swaps_total == 3

    env._advance_turns(1)
    assert env.grid_unit_swaps_this_turn == 0
    mask = env.compute_action_mask()
    assert any(bool(value) for value in mask[swap_start:swap_stop])


def test_grid_unit_swap_moves_normal_unit_to_empty_position_10():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    extra_pairs = env.GRID_SWAP_UNIT_PAIRS[6:10]
    assert extra_pairs == ((7, 10), (8, 10), (9, 10), (10, 11))
    extra_actions = [
        env.GRID_SWAP_UNIT_ACTION_START + 6 + idx for idx in range(len(extra_pairs))
    ]
    mask = env.compute_action_mask()
    assert [bool(mask[action]) for action in extra_actions] == [True] * 4

    action = extra_actions[0]
    original_pos7 = _unit_by_position(env, 7)
    original_pos7_name = original_pos7["name"]
    env.active_strength_potion_positions.add(7)
    _obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["unit_swap_action"] is True
    assert info["unit_swap_applied"] is True
    assert info["unit_swap_move_to_empty"] is True
    _assert_unit_swap_penalty(env, reward, info)
    assert info["unit_swap_positions"] == (7, 10)
    assert env._is_empty_blue_unit(_unit_by_position(env, 7))
    moved_unit = _unit_by_position(env, 10)
    assert moved_unit["name"] == original_pos7_name
    assert moved_unit["stand"] == "behind"
    assert int(10) in env.active_strength_potion_positions
    assert int(7) not in env.active_strength_potion_positions
    assert bool(env.compute_action_mask()[action]) is False


def test_grid_unit_swap_moves_hero_to_empty_back_right_position_12():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    slot_12_pairs = env.GRID_SWAP_UNIT_PAIRS[10:]
    assert slot_12_pairs == ((7, 12), (8, 12), (9, 12), (10, 12), (11, 12))
    slot_12_actions = [
        env.GRID_SWAP_UNIT_ACTION_START + 10 + idx for idx in range(len(slot_12_pairs))
    ]
    mask = env.compute_action_mask()
    assert [bool(mask[action]) for action in slot_12_actions] == [
        True,
        True,
        True,
        False,
        True,
    ]

    action = slot_12_actions[1]
    hero = _unit_by_position(env, 8)
    hero_name = hero["name"]
    assert bool(hero.get("hero", False)) is True
    _obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["unit_swap_action"] is True
    assert info["unit_swap_applied"] is True
    assert info["unit_swap_move_to_empty"] is True
    _assert_unit_swap_penalty(env, reward, info)
    assert info["unit_swap_positions"] == (8, 12)
    assert env._is_empty_blue_unit(_unit_by_position(env, 8))
    moved_hero = _unit_by_position(env, 12)
    assert moved_hero["name"] == hero_name
    assert bool(moved_hero.get("hero", False)) is True
    assert moved_hero["stand"] == "behind"
    assert bool(env.compute_action_mask()[action]) is False


def test_big_unit_swap_exchanges_whole_columns_with_two_normal_units():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    big = _make_big_unit_at_position(env, 9)
    front_left = _unit_by_position(env, 7)
    back_left = _put_real_unit_at_position(env, 10)
    front_left_name = front_left["name"]
    back_left_name = back_left["name"]
    env.active_strength_potion_positions.add(9)
    env.active_haste_elixir_positions.add(7)
    env.active_fire_ward_positions.add(10)

    action = env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_PAIRS.index((7, 9))
    assert bool(env.compute_action_mask()[action]) is True

    _obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["unit_swap_action"] is True
    assert info["unit_swap_applied"] is True
    _assert_unit_swap_penalty(env, reward, info)
    assert info["unit_swap_big_area"] is True
    assert info["unit_swap_affected_positions"] == (7, 9, 10, 12)

    assert _unit_by_position(env, 7)["name"] == big["name"]
    assert _unit_by_position(env, 7)["big"] is True
    assert _unit_by_position(env, 7)["stand"] == "ahead"
    assert env._is_empty_blue_unit(_unit_by_position(env, 10))
    assert _unit_by_position(env, 9)["name"] == front_left_name
    assert _unit_by_position(env, 9)["stand"] == "ahead"
    assert _unit_by_position(env, 12)["name"] == back_left_name
    assert _unit_by_position(env, 12)["stand"] == "behind"
    assert env.active_strength_potion_positions == {7}
    assert env.active_haste_elixir_positions == {9}
    assert env.active_fire_ward_positions == {12}


def test_big_unit_swap_can_move_to_empty_back_slot_without_extra_action():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    big = _make_big_unit_at_position(env, 9)
    _put_empty_unit_at_position(env, 7)
    _put_empty_unit_at_position(env, 10)

    action = env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_PAIRS.index((9, 10))
    assert bool(env.compute_action_mask()[action]) is True

    _obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["unit_swap_action"] is True
    assert info["unit_swap_applied"] is True
    _assert_unit_swap_penalty(env, reward, info)
    assert info["unit_swap_big_area"] is True
    assert _unit_by_position(env, 7)["name"] == big["name"]
    assert _unit_by_position(env, 7)["big"] is True
    assert _unit_by_position(env, 7)["stand"] == "ahead"
    assert env._is_empty_blue_unit(_unit_by_position(env, 10))
    assert env._is_empty_blue_unit(_unit_by_position(env, 9))
    assert env._is_empty_blue_unit(_unit_by_position(env, 12))
    assert env.GRID_SWAP_UNIT_ACTION_COUNT == 15


def test_big_unit_swap_exchanges_with_another_big_unit_column():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    left_big = _make_big_unit_at_position(env, 7)
    right_big = _make_big_unit_at_position(env, 9)
    left_name = left_big["name"]
    right_name = right_big["name"]

    action = env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_PAIRS.index((7, 9))
    assert bool(env.compute_action_mask()[action]) is True

    _obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["unit_swap_action"] is True
    assert info["unit_swap_applied"] is True
    _assert_unit_swap_penalty(env, reward, info)
    assert info["unit_swap_big_area"] is True
    assert _unit_by_position(env, 7)["name"] == right_name
    assert _unit_by_position(env, 7)["big"] is True
    assert env._is_empty_blue_unit(_unit_by_position(env, 10))
    assert _unit_by_position(env, 9)["name"] == left_name
    assert _unit_by_position(env, 9)["big"] is True
    assert env._is_empty_blue_unit(_unit_by_position(env, 12))
