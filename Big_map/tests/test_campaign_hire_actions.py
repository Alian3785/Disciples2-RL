import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from hire import empire_hire_d2, legions_hire_d2


def _hire_action_index(env: CampaignEnv, idx: int) -> int:
    return env.GRID_HIRE_ACTION_START + idx


def _blue_unit(env: CampaignEnv, position: int) -> dict:
    return next(u for u in env.blue_team_state if int(u.get("position", -1)) == int(position))


def _enable_hero_hire(env: CampaignEnv, gold: float) -> dict:
    env.reset(seed=123)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]
    hero = _blue_unit(env, 8)
    hero["Level"] = 3
    hero["needaunit"] = 1
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


def test_legions_hire_action_count_matches_hire_py():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _enable_hero_hire(env, gold=999.0)

    assert len(env.active_hire_options) == len(legions_hire_d2)
    assert (
        env.GRID_MERCENARY_HIRE_ACTION_START - env.GRID_HIRE_ACTION_START
        == len(legions_hire_d2)
    )
    assert (
        env.GRID_TRAINER_ACTION_START - env.GRID_MERCENARY_HIRE_ACTION_START
        == len(env.active_mercenary_hire_options)
    )
    assert (
        env.GRID_MERCHANT_BUY_ACTION_START - env.GRID_TRAINER_ACTION_START
        == len(env.TRAINER_POSITIONS)
    )


def test_legions_mage_hire_uses_name_from_hire_py_and_places_unit_on_backline():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
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
    assert reward == pytest.approx(3.0 * float(env.reward_defeat_enemy))
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
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(float(hire_option["gold"]))
    assert float(info.get("hire_reward", 0.0)) == pytest.approx(3.0 * float(env.reward_defeat_enemy))
    assert info.get("hire_role") == "mage"


def test_legions_warrior_hire_places_unit_on_frontline():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
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
    assert reward == pytest.approx(3.0 * float(env.reward_defeat_enemy))
    assert hero["needaunit"] == 0
    assert env.gold == pytest.approx(0.0)
    assert info.get("hired") is True
    assert info.get("hired_units_count") == 1
    assert info.get("hired_unit_name") == hire_option["name"]
    assert info.get("hired_position") == 7
    assert float(info.get("hire_reward", 0.0)) == pytest.approx(3.0 * float(env.reward_defeat_enemy))
    assert info.get("hire_role") == "warrior"


def test_hire_action_is_blocked_when_gold_is_insufficient():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
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
    assert reward == pytest.approx(0.0)
    assert info.get("hired") is False
    assert info.get("hire_unit_name") == hire_option["name"]
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(0.0)
    assert env.gold == pytest.approx(59.0)


def test_hire_actions_require_castle_tile_and_follow_race_specific_hire_array_length():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _enable_hero_hire(env, gold=100.0)
    env.grid_env.agent_pos = (0, 0)

    mask = env.compute_action_mask()
    for idx in range(len(env.active_hire_options)):
        assert bool(mask[_hire_action_index(env, idx)]) is False

    other_race_env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=1)
    _enable_hero_hire(other_race_env, gold=100.0)

    assert len(other_race_env.active_hire_options) == len(empire_hire_d2)
    assert (
        other_race_env.GRID_MERCENARY_HIRE_ACTION_START - other_race_env.GRID_HIRE_ACTION_START
        == len(empire_hire_d2)
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


def test_pending_hire_flag_adds_extra_turn_penalty_until_recruitment_happens():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    hero = _enable_hero_hire(env, gold=100.0)

    turn_penalty = env._advance_turns(1)
    assert turn_penalty == pytest.approx(
        -float(env.reward_turn_penalty) - float(env.reward_needaunit_turn_penalty)
    )

    hero["needaunit"] = 0
    turn_penalty_after_hire = env._advance_turns(1)
    assert turn_penalty_after_hire == pytest.approx(-float(env.reward_turn_penalty))
