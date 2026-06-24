import pytest

from battle_env import UNITS_BLUE
from campaign_env import CampaignEnv


def _reward_test_env() -> CampaignEnv:
    return CampaignEnv(
        reward_unit_upgrade=6.0,
        reward_unit_tier_bonus=0.5,
        reward_unit_tier3_multiplier=2.0,
        log_enabled=False,
        persist_blue_hp=True,
        Realcapital=2,
    )


def test_tier3_unit_upgrade_reward_is_doubled_without_changing_other_tiers():
    env = _reward_test_env()

    assert env._unit_upgrade_reward_for_tier(2) == pytest.approx(7.0)
    assert env._unit_upgrade_reward_for_tier(3) == pytest.approx(15.0)
    assert env._unit_upgrade_reward_for_tier(4) == pytest.approx(8.0)


@pytest.mark.parametrize(
    ("target_level", "expected_reward"),
    [
        (2, 7.0),
        (3, 15.0),
        (4, 8.0),
    ],
)
def test_hero_levelup_uses_unit_upgrade_reward_through_level_four(
    target_level,
    expected_reward,
):
    env = _reward_test_env()
    hero = dict(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    hero["Level"] = target_level
    hero["hero"] = True
    hero["next_level_exp"] = 500

    env.battle_env = type(
        "DummyBattle",
        (),
        {"combined": [hero], "last_levelups": [hero["name"]]},
    )()

    assert env._log_turns_into_levelups() == 0
    assert env._last_hero_levelup_count == 1
    assert env._last_hero_levelup_max_level == target_level
    assert env._last_hero_levelup_reward == pytest.approx(expected_reward)
    assert env._last_upgrade_reward == pytest.approx(expected_reward)
    assert env._last_upgrade_max_tier == 0


@pytest.mark.parametrize("target_level", [1, 5])
def test_hero_levelup_reward_only_applies_to_levels_two_through_four(target_level):
    env = _reward_test_env()
    hero = dict(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    hero["Level"] = target_level
    hero["hero"] = True
    hero["next_level_exp"] = 500

    env.battle_env = type(
        "DummyBattle",
        (),
        {"combined": [hero], "last_levelups": [hero["name"]]},
    )()

    assert env._log_turns_into_levelups() == 0
    assert env._last_hero_levelup_count == 0
    assert env._last_hero_levelup_max_level == 0
    assert env._last_hero_levelup_reward == pytest.approx(0.0)
    assert env._last_upgrade_reward == pytest.approx(0.0)
