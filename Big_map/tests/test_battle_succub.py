import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv


def _battle_unit(
    *,
    name: str,
    team: str,
    position: int,
    unit_type: str,
    immunity: list[str] | None = None,
    armor: int = 0,
    big: bool = False,
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
        "armor": armor,
        "big": big,
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


@pytest.mark.parametrize(
    ("attacker_name", "unit_type"),
    [
        ("Ведьма", "Witch"),
        ("Колдунья", "Witch"),
        ("Суккуб", "Succub"),
    ],
)
@pytest.mark.parametrize(
    ("is_big", "expected_accuracy", "expected_damage", "expected_initiative"),
    [
        (False, 80, 20, 30),
        (True, 70, 30, 50),
    ],
)
def test_witch_family_transform_uses_imp_armor_immunity_and_accuracy(
    attacker_name: str,
    unit_type: str,
    is_big: bool,
    expected_accuracy: int,
    expected_damage: int,
    expected_initiative: int,
):
    env = BattleEnv(log_enabled=False)
    env._init_with_custom_teams(
        [
            _battle_unit(
                name=attacker_name,
                team="red",
                position=1,
                unit_type=unit_type,
            )
        ],
        [
            _battle_unit(
                name="Transform Target",
                team="blue",
                position=7,
                unit_type="Mage",
                immunity=["Fire", "Death"],
                armor=35,
                big=is_big,
            )
        ],
    )
    attacker = next(unit for unit in env.combined if unit["team"] == "red")
    victim = next(unit for unit in env.combined if unit["team"] == "blue")
    attacker["attack_type_primary"] = "Mind"
    attacker["attack_type_secondary"] = ""
    env._roll_hit = lambda _: True

    target_pos = victim["position"] if unit_type == "Witch" else None
    applied, _ = env._attack(attacker, target_pos)

    assert applied is True
    assert victim["transformed"] == 1
    assert victim["unit_type"] == "Warrior"
    assert victim["accuracy"] == expected_accuracy
    assert victim["damage"] == expected_damage
    assert victim["initiative_base"] == expected_initiative
    assert victim["armor"] == 0
    assert victim["immunity"] == []

    assert env._restore_transformed_unit(victim) is True
    assert victim["armor"] == 35
    assert victim["immunity"] == ["Fire", "Death"]
