import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv


def _unit(env: BattleEnv, team: str, position: int) -> dict:
    return next(
        unit
        for unit in env.combined
        if unit.get("team") == team and int(unit.get("position", -1)) == position
    )


def test_lord_burn_uses_shared_dot_cache_without_source_field():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=7)

    attacker = _unit(env, "blue", 7)
    victim = _unit(env, "red", 1)
    attacker.update(
        {
            "name": "Владыка",
            "unit_type": "Lord",
            "damage": 0,
            "accuracy": 100,
            "accuracy_secondary": 100,
            "damage_secondary": 9,
            "attack_type_primary": "Weapon",
            "attack_type_secondary": "Fire",
        }
    )
    victim.update(
        {
            "health": 100,
            "max_health": 100,
            "armor": 0,
            "immunity": [],
            "resistance": [],
            "resilience_used_types": [],
            "burn_turns_left": 0,
            "burn_damage_per_tick": 0,
        }
    )

    applied, reason = env._attack(attacker, victim["position"])

    assert (applied, reason) == (True, "ok")
    assert victim["burn_turns_left"] > 0
    assert victim["burn_damage_per_tick"] == 9
    assert env._lord_applied_burn == {attacker["position"]: True}
    assert "burn_source_lord_pos" not in victim

    victim["initiative"] = 20
    victim["burn_turns_left"] = 1
    victim["burn_damage_per_tick"] = 0

    assert env._apply_start_of_turn_effects(victim) is True
    assert victim["burn_turns_left"] == 0
    assert victim["burn_damage_per_tick"] == 0
    assert env._lord_applied_burn == {}
