import pytest

from battle_env import BattleEnv


def _unit(env: BattleEnv, team: str, position: int) -> dict:
    return next(
        unit
        for unit in env.combined
        if unit.get("team") == team and int(unit.get("position", -1)) == position
    )


def test_direct_overkill_clamps_health_to_zero():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=1)
    attacker = _unit(env, "blue", 7)
    victim = _unit(env, "red", 1)
    attacker.update(
        {
            "unit_type": "Warrior",
            "damage": 1000,
            "accuracy": 100,
            "attack_type_primary": "Weapon",
        }
    )
    victim.update({"health": 5, "hp": 5, "armor": 0, "immunity": [], "resistance": []})
    env._roll_hit = lambda _: True

    env._attack(attacker, victim["position"])

    assert victim["health"] == 0
    assert victim["hp"] == 0


def test_aoe_overkill_clamps_health_to_zero():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=2)
    attacker = _unit(env, "blue", 7)
    victim = _unit(env, "red", 1)
    attacker.update(
        {
            "unit_type": "Mage",
            "damage": 1000,
            "accuracy": 100,
            "attack_type_primary": "Air",
        }
    )
    victim.update({"health": 5, "hp": 5, "armor": 0, "immunity": [], "resistance": []})
    env._roll_hit = lambda _: True

    env._attack(attacker, victim["position"])

    assert victim["health"] == 0
    assert victim["hp"] == 0


@pytest.mark.parametrize(
    ("turns_field", "damage_field"),
    [
        ("poison_turns_left", "poison_damage_per_tick"),
        ("burn_turns_left", "burn_damage_per_tick"),
        ("uran_turns_left", "uran_damage_per_tick"),
    ],
)
def test_dot_overkill_clamps_health_to_zero(turns_field: str, damage_field: str):
    env = BattleEnv(log_enabled=False)
    env.reset(seed=3)
    victim = _unit(env, "blue", 7)
    victim.update(
        {
            "health": 3,
            "hp": 3,
            "initiative": 10,
            "immunity": [],
            "round_effects_done": 0,
            turns_field: 1,
            damage_field: 30,
        }
    )

    env._apply_start_of_turn_effects(victim)

    assert victim["health"] == 0
    assert victim["hp"] == 0


def test_centaur_critical_overkill_clamps_health_to_zero(monkeypatch):
    env = BattleEnv(log_enabled=False)
    env.reset(seed=4)
    attacker = _unit(env, "blue", 7)
    victim = _unit(env, "red", 1)
    attacker.update(
        {
            "unit_type": "Centaur Savage",
            "damage": 100,
            "accuracy": 100,
            "attack_type_primary": "Weapon",
        }
    )
    victim.update({"health": 3, "hp": 3, "armor": 0, "immunity": [], "resistance": []})
    env._roll_hit = lambda _: True
    monkeypatch.setattr(env, "_apply_damage_with_armor", lambda *_: 1)

    env._attack(attacker, victim["position"])

    assert victim["health"] == 0
    assert victim["hp"] == 0
