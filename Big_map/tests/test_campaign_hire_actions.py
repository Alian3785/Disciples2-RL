import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv as _CampaignEnv
from hire import empire_hire_d2, legions_hire_d2


class CampaignEnv(_CampaignEnv):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_boss_starting_roster", False)
        super().__init__(*args, **kwargs)


def _hire_action_index(env: CampaignEnv, idx: int) -> int:
    return env.GRID_HIRE_ACTION_START + idx


def _big_hire_action_and_option(env: CampaignEnv) -> tuple[int, dict]:
    for idx, option in enumerate(env.active_hire_options):
        if env._is_big_hire_option(option):
            return _hire_action_index(env, idx), option
    raise AssertionError("No faction big hire option found")


def _blue_unit(env: CampaignEnv, position: int) -> dict:
    return next(
        u for u in env.blue_team_state if int(u.get("position", -1)) == int(position)
    )


def _enable_hero_hire(env: CampaignEnv, gold: float) -> dict:
    env.reset(seed=123)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]
    hero = _blue_unit(env, 8)
    hero["Level"] = 3
    env._sync_hero_progression_flags(env.blue_team_state)
    env.gold = float(gold)
    return hero


def _clear_slot(env: CampaignEnv, position: int) -> None:
    unit = _blue_unit(env, position)
    unit["name"] = "пусто"
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["maxhp"] = 0.0
    unit["max_health"] = 0.0
    unit["exp_required"] = 0
    unit["exp_current"] = 0
    unit["next_level_exp"] = 0


def _clear_all_companions(env: CampaignEnv) -> None:
    for unit in list(env.blue_team_state):
        if env._is_hero_unit(unit) or env._is_empty_blue_unit(unit):
            continue
        _clear_slot(env, int(unit["position"]))
    env._sync_hero_progression_flags(env.blue_team_state)


def _swap_slot_positions(env: CampaignEnv, position_a: int, position_b: int) -> None:
    unit_a = _blue_unit(env, position_a)
    unit_b = _blue_unit(env, position_b)
    unit_a["position"], unit_b["position"] = position_b, position_a
    env.blue_team_state.sort(key=lambda unit: int(unit.get("position", 999) or 999))


def _hire_reward_after_penalty(env: CampaignEnv, leadership_points: int) -> float:
    return 3.0 * float(env.reward_defeat_enemy) - (
        float(env.reward_needaunit_turn_penalty) * float(leadership_points)
    )


def test_legions_hire_action_count_matches_hire_py():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    _enable_hero_hire(env, gold=999.0)

    assert len(env.active_hire_options) == len(legions_hire_d2) + 1
    assert (
        env.GRID_MERCENARY_HIRE_ACTION_START - env.GRID_HIRE_ACTION_START
        == len(legions_hire_d2) + 1
    )
    assert env.GRID_TRAINER_ACTION_START - env.GRID_MERCENARY_HIRE_ACTION_START == len(
        env.active_mercenary_hire_options
    )
    assert env.GRID_MERCHANT_BUY_ACTION_START - env.GRID_TRAINER_ACTION_START == len(
        env.TRAINER_POSITIONS
    )


def test_legions_mage_hire_uses_name_from_hire_py_and_places_unit_on_backline():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_hero_hire(env, gold=60.0)

    hire_idx, hire_option = next(
        (idx, option)
        for idx, option in enumerate(env.active_hire_options)
        if str(option.get("role", "")).strip().lower() == "mage"
    )
    action = _hire_action_index(env, hire_idx)
    mask = env.compute_action_mask()

    assert bool(mask[action]) is True

    _, reward, terminated, truncated, info = env.step(action)

    expected = env._build_unit_from_data(
        env._find_unit_data_by_name(str(hire_option["name"])),
        "blue",
        10,
    )
    recruited = _blue_unit(env, 10)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(_hire_reward_after_penalty(env, 0))
    assert recruited["name"] == expected["name"]
    assert recruited["health"] == expected["health"]
    assert recruited["max_health"] == expected["max_health"]
    assert recruited["damage"] == expected["damage"]
    assert recruited["initiative"] == expected["initiative"]
    assert recruited["unit_type"] == expected["unit_type"]
    assert hero["needaunit"] == 0
    assert env.gold == pytest.approx(0.0)
    assert info.get("hired") is True
    assert info.get("hired_units_count") == 1
    assert info.get("hired_unit_name") == hire_option["name"]
    assert info.get("hired_position") == 10
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(
        float(hire_option["gold"])
    )
    assert float(info.get("hire_reward", 0.0)) == pytest.approx(
        3.0 * float(env.reward_defeat_enemy)
    )
    assert info.get("hire_role") == "mage"


def test_legions_warrior_hire_places_unit_on_frontline():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_hero_hire(env, gold=50.0)
    _clear_slot(env, 7)

    hire_idx, hire_option = next(
        (idx, option)
        for idx, option in enumerate(env.active_hire_options)
        if str(option.get("role", "")).strip().lower() == "warrior"
    )
    action = _hire_action_index(env, hire_idx)
    mask = env.compute_action_mask()

    assert bool(mask[action]) is True

    _, reward, _, _, info = env.step(action)

    expected = env._build_unit_from_data(
        env._find_unit_data_by_name(str(hire_option["name"])),
        "blue",
        7,
    )
    recruited = _blue_unit(env, 7)

    assert recruited["name"] == expected["name"]
    assert recruited["health"] == expected["health"]
    assert recruited["max_health"] == expected["max_health"]
    assert recruited["damage"] == expected["damage"]
    assert recruited["initiative"] == expected["initiative"]
    assert recruited["unit_type"] == expected["unit_type"]
    assert reward == pytest.approx(_hire_reward_after_penalty(env, 1))
    assert hero["needaunit"] == 1
    assert info["leadership_points"] == 1
    assert info["leadership_step_penalty"] == pytest.approx(
        float(env.reward_needaunit_turn_penalty)
    )
    assert env.gold == pytest.approx(0.0)
    assert info.get("hired") is True
    assert info.get("hired_units_count") == 1
    assert info.get("hired_unit_name") == hire_option["name"]
    assert info.get("hired_position") == 7
    assert float(info.get("hire_reward", 0.0)) == pytest.approx(
        3.0 * float(env.reward_defeat_enemy)
    )
    assert info.get("hire_role") == "warrior"


def test_hire_action_is_blocked_when_gold_is_insufficient():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    _enable_hero_hire(env, gold=59.0)

    hire_idx, hire_option = next(
        (idx, option)
        for idx, option in enumerate(env.active_hire_options)
        if float(option.get("gold", 0) or 0) > 59.0
    )
    action = _hire_action_index(env, hire_idx)
    mask = env.compute_action_mask()

    assert bool(mask[action]) is False

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(-float(env.reward_needaunit_turn_penalty))
    assert info.get("hired") is False
    assert info.get("hire_unit_name") == hire_option["name"]
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(0.0)
    assert env.gold == pytest.approx(59.0)


def test_hire_actions_require_castle_tile_and_follow_race_specific_hire_array_length():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    _enable_hero_hire(env, gold=100.0)
    env.grid_env.agent_pos = (0, 0)

    mask = env.compute_action_mask()
    for idx in range(len(env.active_hire_options)):
        assert bool(mask[_hire_action_index(env, idx)]) is False

    other_race_env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=1)
    _enable_hero_hire(other_race_env, gold=100.0)

    assert len(other_race_env.active_hire_options) == len(empire_hire_d2) + 1
    assert (
        other_race_env.GRID_MERCENARY_HIRE_ACTION_START
        - other_race_env.GRID_HIRE_ACTION_START
        == len(empire_hire_d2) + 1
    )
    assert (
        other_race_env.GRID_TRAINER_ACTION_START
        - other_race_env.GRID_MERCENARY_HIRE_ACTION_START
        == len(other_race_env.active_mercenary_hire_options)
    )
    assert (
        other_race_env.GRID_MERCHANT_BUY_ACTION_START
        - other_race_env.GRID_TRAINER_ACTION_START
        == len(other_race_env.TRAINER_POSITIONS)
    )


def test_base_party_uses_all_three_points_until_level_three_adds_one():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]
    hero = _blue_unit(env, 8)
    hero["Level"] = env.HERO_LEADERSHIP_LEVEL - 1
    hero["needaunit"] = 1
    env.gold = 999.0

    assert env._sync_hero_needaunit_from_party() == 0
    assert hero["needaunit"] == 0
    assert not any(env._faction_hire_action_mask_values())

    hero["Level"] = env.HERO_LEADERSHIP_LEVEL
    assert env._sync_hero_needaunit_from_party() == 1
    assert hero["needaunit"] == 1
    assert any(env._faction_hire_action_mask_values())


def test_hero_alone_has_three_points_and_is_penalized_on_every_campaign_step():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]
    hero = _blue_unit(env, 8)
    hero["Level"] = 1
    _clear_all_companions(env)
    env.gold = 0.0

    composition = env._party_hire_composition()
    assert composition == {
        "small_units": 0,
        "big_units": 0,
        "leadership_capacity": 3,
        "occupied_capacity": 0,
        "leadership_points": 3,
    }
    assert hero["needaunit"] == 1

    action = _hire_action_index(env, 0)
    _, reward, _, _, info = env.step(action)

    expected_penalty = 3.0 * float(env.reward_needaunit_turn_penalty)
    assert reward == pytest.approx(-expected_penalty)
    assert info["leadership_points"] == 3
    assert info["leadership_step_penalty"] == pytest.approx(expected_penalty)
    assert env._advance_turns(1) == pytest.approx(-float(env.reward_turn_penalty))


def test_three_small_hires_consume_base_points_one_by_one():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]
    hero = _blue_unit(env, 8)
    hero["Level"] = 1
    _clear_all_companions(env)
    env.gold = 200.0

    warrior_idx = next(
        idx
        for idx, option in enumerate(env.active_hire_options)
        if str(option.get("role", "")).strip().lower() == "warrior"
    )
    mage_idx = next(
        idx
        for idx, option in enumerate(env.active_hire_options)
        if str(option.get("role", "")).strip().lower() == "mage"
    )
    actions = (
        _hire_action_index(env, warrior_idx),
        _hire_action_index(env, mage_idx),
        _hire_action_index(env, warrior_idx),
    )

    assert env._sync_hero_needaunit_from_party() == 3
    for action, expected_points in zip(actions, (2, 1, 0), strict=True):
        assert bool(env.compute_action_mask()[action]) is True
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(
            _hire_reward_after_penalty(env, expected_points)
        )
        assert info["hired"] is True
        assert env._available_leadership_points() == expected_points
        assert info["leadership_points"] == expected_points
        assert hero["needaunit"] == int(expected_points > 0)

    assert env._advance_turns(1) == pytest.approx(-float(env.reward_turn_penalty))


def test_big_hire_option_unlocks_on_second_leadership_if_first_hire_was_skipped():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_hero_hire(env, gold=100.0)
    big_action, big_option = _big_hire_action_and_option(env)
    _swap_slot_positions(env, 9, 10)

    mask = env.compute_action_mask()
    assert bool(mask[big_action]) is False

    hero["Level"] = env.HERO_SECOND_LEADERSHIP_LEVEL
    mask = env.compute_action_mask()

    assert bool(mask[big_action]) is True
    assert env._available_leadership_points() == 2
    assert big_option["name"] == "\u0427\u0451\u0440\u0442"
    assert env._unit_data_is_big(env._find_unit_data_by_name(str(big_option["name"])))


def test_big_hire_option_requires_a_fully_empty_column():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_hero_hire(env, gold=100.0)
    hero["Level"] = env.HERO_SECOND_LEADERSHIP_LEVEL
    big_action, _ = _big_hire_action_and_option(env)

    mask = env.compute_action_mask()
    assert bool(mask[big_action]) is False

    _swap_slot_positions(env, 9, 10)
    mask = env.compute_action_mask()
    assert bool(mask[big_action]) is True


def test_big_hire_places_large_unit_on_front_and_clears_needaunit():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_hero_hire(env, gold=100.0)
    hero["Level"] = env.HERO_SECOND_LEADERSHIP_LEVEL
    _swap_slot_positions(env, 9, 10)
    big_action, big_option = _big_hire_action_and_option(env)

    _, reward, terminated, truncated, info = env.step(big_action)
    recruited = _blue_unit(env, 9)
    partner = _blue_unit(env, 12)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(3.0 * float(env.reward_defeat_enemy))
    assert recruited["name"] == big_option["name"]
    assert recruited["big"] is True
    assert recruited["stand"] == "ahead"
    assert env._is_empty_blue_unit(partner)
    assert hero["needaunit"] == 0
    assert env.gold == pytest.approx(0.0)
    assert info.get("hire_big_unit") is True
    assert info.get("hire_column_positions") == (9, 12)
    assert info.get("hired") is True
    assert info.get("hired_unit_name") == big_option["name"]
    assert info.get("hired_position") == 9
    assert env._party_hire_composition() == {
        "small_units": 3,
        "big_units": 1,
        "leadership_capacity": 5,
        "occupied_capacity": 5,
        "leadership_points": 0,
    }


def test_big_hire_is_available_to_level_one_hero_with_two_free_points():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_hero_hire(env, gold=100.0)
    _clear_all_companions(env)
    hero["Level"] = 1
    big_action, _ = _big_hire_action_and_option(env)

    assert bool(env.compute_action_mask()[big_action]) is True
    _, reward, _, _, info = env.step(big_action)

    assert reward == pytest.approx(_hire_reward_after_penalty(env, 1))
    assert info["hired"] is True
    assert env._party_hire_composition() == {
        "small_units": 0,
        "big_units": 1,
        "leadership_capacity": 3,
        "occupied_capacity": 2,
        "leadership_points": 1,
    }
    assert hero["needaunit"] == 1
    assert info["leadership_step_penalty"] == pytest.approx(
        float(env.reward_needaunit_turn_penalty)
    )


def test_level_six_allows_big_only_if_level_three_point_was_not_filled():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_hero_hire(env, gold=999.0)
    mage_idx = next(
        idx
        for idx, option in enumerate(env.active_hire_options)
        if str(option.get("role", "")).strip().lower() == "mage"
    )
    mage_action = _hire_action_index(env, mage_idx)
    _, reward, _, _, info = env.step(mage_action)

    assert reward == pytest.approx(_hire_reward_after_penalty(env, 0))
    assert info["leadership_points"] == 0
    assert hero["needaunit"] == 0

    hero["Level"] = env.HERO_SECOND_LEADERSHIP_LEVEL
    big_action, _ = _big_hire_action_and_option(env)

    assert env._sync_hero_needaunit_from_party() == 1
    assert hero["needaunit"] == 1
    assert env._hero_can_use_big_leadership_hire(hero) is False
    mask = env.compute_action_mask()
    assert bool(mask[big_action]) is False
    assert bool(mask[mage_action]) is True

    _, reward, _, _, info = env.step(mage_action)
    assert reward == pytest.approx(_hire_reward_after_penalty(env, 0))
    assert info["leadership_points"] == 0
    assert hero["needaunit"] == 0
