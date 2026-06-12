import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv


def _battle_unit(
    *,
    name: str,
    team: str,
    position: int,
    unit_type: str,
    immunity: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "team": team,
        "position": position,
        "stand": "ahead",
        "unit_type": unit_type,
        "health": 100,
        "max_health": 100,
        "damage": 0,
        "damage_secondary": 0,
        "initiative_base": 50,
        "accuracy": 100,
        "accuracy_secondary": 100,
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "Mind",
        "immunity": list(immunity or []),
        "resistance": [],
        "armor": 0,
        "big": False,
    }


def test_succub_transform_respects_status_immunity():
    env = BattleEnv(log_enabled=False)
    env._init_with_custom_teams(
        [
            _battle_unit(
                name="Succub",
                team="red",
                position=1,
                unit_type="Succub",
            )
        ],
        [
            _battle_unit(
                name="Immune Target",
                team="blue",
                position=7,
                unit_type="Warrior",
                immunity=["Mind"],
            )
        ],
    )

    attacker = next(unit for unit in env.combined if unit["team"] == "red")
    victim = next(unit for unit in env.combined if unit["team"] == "blue")

    assert env._is_immune_status(attacker, victim) is True

    applied, reason = env._attack(attacker, None)

    assert (applied, reason) == (True, "aoe")
    assert victim["transformed"] == 0
    assert victim["unit_type"] == "Warrior"
