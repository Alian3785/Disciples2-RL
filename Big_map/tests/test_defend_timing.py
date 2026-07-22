import pytest

from battle_env import BattleEnv


def _pick_safe_blue_unit(env: BattleEnv) -> dict:
    """Pick a BLUE unit with minimal side effects at start-of-turn for stable testing."""
    excluded_types = {"Doppelganger", "Sundancer", "Sylfid", "Deva roshi"}
    for unit in env.combined:
        if unit.get("team") != "blue":
            continue
        if not env._alive(unit):
            continue
        if unit.get("unit_type") in excluded_types:
            continue
        return unit
    pytest.fail("No suitable BLUE unit found for DEFEND timing test")


def test_defend_status_expires_only_on_owner_next_turn_start():
    env = BattleEnv(log_enabled=False)
    env.seed(123)
    env.reset()

    unit = _pick_safe_blue_unit(env)

    base_armor = int(unit.get("armor", 0) or 0)
    unit["defense"] = 1

    # End of round must not remove DEFEND; armor itself never changes.
    env._end_round_restore()
    assert unit["defense"] == 1
    assert unit["armor"] == base_armor

    # DEFEND expires exactly when this unit starts its next turn.
    alive_after_start = env._apply_start_of_turn_effects(unit)
    assert alive_after_start is True
    assert unit["defense"] == 0
    assert unit["armor"] == base_armor


def test_defend_halves_damage_after_armor(monkeypatch):
    env = BattleEnv(log_enabled=False)
    env.seed(123)
    env.reset()
    monkeypatch.setattr(env.rng, "randint", lambda _low, _high: 0)

    attacker = {"unit_type": "Warrior"}
    victim = {"armor": 40, "defense": 0}

    assert env._apply_damage_with_armor(attacker, 100, victim) == 60

    victim["defense"] = 1
    assert env._apply_damage_with_armor(attacker, 100, victim) == 30
    assert victim["armor"] == 40
