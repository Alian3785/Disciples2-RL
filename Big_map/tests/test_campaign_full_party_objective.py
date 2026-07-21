from copy import deepcopy

import pytest

from campaign_env import CampaignEnv


def _env(**kwargs) -> CampaignEnv:
    return CampaignEnv(
        campaign_objective="full_party",
        use_boss_starting_roster=False,
        log_enabled=False,
        persist_blue_hp=True,
        Realcapital=1,
        **kwargs,
    )


def _unit_at(env: CampaignEnv, position: int) -> dict:
    return next(
        unit
        for unit in env.blue_team_state
        if int(unit.get("position", -1)) == int(position)
    )


def _empty_slot(env: CampaignEnv, position: int) -> None:
    unit = _unit_at(env, position)
    unit.update(
        {
            "name": "empty",
            "hero": False,
            "big": False,
            "hp": 0.0,
            "health": 0.0,
            "maxhp": 0.0,
            "max_health": 0.0,
            "exp_required": 0,
            "exp_current": 0,
            "next_level_exp": 0,
        }
    )


def _copy_small_unit(env: CampaignEnv, source: int, target: int) -> None:
    copied = deepcopy(_unit_at(env, source))
    copied["position"] = int(target)
    copied["hero"] = False
    copied["big"] = False
    env.blue_team_state[int(target) - 7] = copied


@pytest.mark.parametrize(
    ("big_hero", "big_companions", "small_companions", "hero_level"),
    [
        (False, (), (7, 9, 10, 11, 12), 6),  # six small units
        (False, (7,), (9, 11, 12), 6),  # four small units and one big unit
        (True, (7, 9), (), 3),  # three big units
    ],
)
def test_full_party_accepts_every_six_slot_composition(
    big_hero,
    big_companions,
    small_companions,
    hero_level,
):
    env = _env()
    env.reset(seed=123)
    source_small = 9
    hero = _unit_at(env, 8)
    hero["Level"] = hero_level
    hero["big"] = bool(big_hero)

    occupied_companions = set(big_companions) | set(small_companions)
    for position in (7, 9, 10, 11, 12):
        if position not in occupied_companions:
            _empty_slot(env, position)
        elif env._is_empty_blue_unit(_unit_at(env, position)):
            _copy_small_unit(env, source_small, position)
        _unit_at(env, position)["big"] = position in set(big_companions)

    env._sync_hero_progression_flags(env.blue_team_state)
    env._reset_full_party_objective_tracking()
    _, reward, terminated, truncated, info = env._apply_leadership_step_penalty(
        (None, 0.0, False, False, {})
    )

    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(env.reward_full_party_complete)
    assert info["full_party_occupied_slots"] == 6
    assert info["full_party_complete"] is True
    assert info["campaign_result"] == "victory"
    assert info["campaign_victory_reason"] == "full_party_assembled"


def test_full_party_hiring_rewards_new_records_and_finishes_on_sixth_slot():
    env = _env(reward_needaunit_turn_penalty=0.0)
    env.reset(seed=123)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]
    hero = _unit_at(env, 8)
    hero["Level"] = 6
    env._sync_hero_progression_flags(env.blue_team_state)
    env.gold = 999.0

    mage_action = next(
        env.GRID_HIRE_ACTION_START + index
        for index, option in enumerate(env.active_hire_options)
        if str(option.get("role", "")).strip().lower() == "mage"
    )

    _, first_reward, first_terminated, _, first_info = env.step(mage_action)

    assert first_terminated is False
    assert first_info["hired"] is True
    assert first_info["hire_reward"] == pytest.approx(
        3.0
        * env.reward_defeat_enemy
        * env.reward_full_party_hire_multiplier
    )
    assert first_info["full_party_new_slots"] == 1
    assert first_info["full_party_unlocked_capacity"] == 2
    assert first_info["full_party_near_complete_reward"] == pytest.approx(
        env.reward_full_party_near_complete
    )
    assert first_reward > first_info["hire_reward"]

    _, second_reward, second_terminated, second_truncated, second_info = env.step(
        mage_action
    )

    assert second_terminated is True
    assert second_truncated is False
    assert second_info["hired"] is True
    assert second_info["full_party_occupied_slots"] == 6
    assert second_info["full_party_new_slots"] == 1
    assert second_info["full_party_completion_reward"] == pytest.approx(
        env.reward_full_party_complete
    )
    assert second_info["campaign_victory_reason"] == "full_party_assembled"
    assert second_reward > second_info["hire_reward"]


def test_full_party_progress_cannot_be_farmed_after_losing_and_refilling_a_slot():
    env = _env(reward_needaunit_turn_penalty=0.0)
    env.reset(seed=123)
    hero = _unit_at(env, 8)
    hero["Level"] = 6
    env._sync_hero_progression_flags(env.blue_team_state)

    _copy_small_unit(env, 9, 10)
    _, _, _, _, first_info = env._apply_leadership_step_penalty(
        (None, 0.0, False, False, {})
    )
    assert first_info["full_party_new_slots"] == 1

    saved_unit = deepcopy(_unit_at(env, 10))
    _empty_slot(env, 10)
    _, loss_reward, _, _, loss_info = env._apply_leadership_step_penalty(
        (None, 0.0, False, False, {})
    )
    assert loss_info["full_party_lost_slots"] == 1
    assert loss_reward == pytest.approx(-env.reward_full_party_unit_lost_penalty)

    env.blue_team_state[3] = saved_unit
    _, refill_reward, _, _, refill_info = env._apply_leadership_step_penalty(
        (None, 0.0, False, False, {})
    )
    assert refill_info["full_party_new_slots"] == 0
    assert refill_info["full_party_new_slot_reward"] == pytest.approx(0.0)
    assert refill_reward == pytest.approx(0.0)


@pytest.mark.parametrize("target_level", [5, 6])
def test_full_party_extends_and_increases_hero_levelup_reward(target_level):
    env = _env(
        reward_unit_upgrade=6.0,
        reward_unit_tier_bonus=0.5,
        reward_full_party_hero_levelup_multiplier=1.5,
    )
    env.reset(seed=123)
    hero = deepcopy(_unit_at(env, 8))
    hero["Level"] = target_level
    hero["hero"] = True
    hero["next_level_exp"] = 500
    env.battle_env = type(
        "DummyBattle",
        (),
        {"combined": [hero], "last_levelups": [hero["name"]]},
    )()

    assert env._log_turns_into_levelups() == 0
    expected = env._unit_upgrade_reward_for_tier(target_level) * 1.5
    assert env._last_hero_levelup_reward == pytest.approx(expected)
    assert env._last_upgrade_reward == pytest.approx(expected)


def test_full_party_objective_is_limited_to_default_map():
    with pytest.raises(ValueError, match="full_party.*small"):
        CampaignEnv(
            map_name="small",
            campaign_objective="full_party",
            log_enabled=False,
        )
