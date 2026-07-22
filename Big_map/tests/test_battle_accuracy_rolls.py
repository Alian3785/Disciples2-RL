import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv


class _SequenceRng:
    def __init__(self, values):
        self._values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self._values)


@pytest.mark.parametrize(
    ("values", "accuracy", "expected"),
    [
        ((0.79, 0.80), 80, True),   # average of integer rolls 79 and 80 is 79.5
        ((0.80, 0.80), 80, False),  # strict comparison: 80 is not below 80
        ((0.00, 0.00), 0, False),
        ((0.99, 0.99), 100, True),
    ],
)
def test_accuracy_uses_two_integer_results_and_strict_comparison(
    values,
    accuracy,
    expected,
):
    env = BattleEnv(log_enabled=False)
    env.rng = _SequenceRng(values)

    assert env._roll_accuracy(accuracy) is expected
    assert env.rng.calls == 2


def test_displayed_eighty_percent_has_original_92_2_percent_hit_rate():
    # Enumerate every equally likely pair from the original 0..99 roll domain.
    hits = sum(
        ((first + second) / 2.0) < 80
        for first in range(100)
        for second in range(100)
    )

    assert hits == 9220
    assert hits / 10000 == pytest.approx(0.922)


def test_primary_and_secondary_accuracy_share_the_same_two_result_roll():
    env = BattleEnv(log_enabled=False)

    env.rng = _SequenceRng((0.79, 0.80))
    assert env._roll_hit({"accuracy": 80}) is True
    assert env.rng.calls == 2

    env.rng = _SequenceRng((0.80, 0.80))
    assert env._roll_status(80) is False
    assert env.rng.calls == 2


@pytest.mark.parametrize(
    "hero_name",
    (
        "Архимаг",
        "Архидьявол",
        "Хранитель знаний",
        "Королева лич",
        "Дриада",
    ),
)
def test_every_mage_hero_loses_ten_accuracy_per_subsequent_aoe_target(hero_name):
    env = BattleEnv(log_enabled=False)
    attacker = {
        "name": hero_name,
        "hero": True,
        "unit_type": "Mage",
        "accuracy": 80,
    }

    assert [
        env._hero_aoe_accuracy_for_target(attacker, target_index)
        for target_index in range(6)
    ] == [80.0, 70.0, 60.0, 50.0, 40.0, 30.0]


def test_nosferatu_uses_mage_hero_aoe_accuracy_falloff():
    env = BattleEnv(log_enabled=False)
    nosferatu = {
        "name": "Носферату",
        "hero": True,
        "unit_type": "Vampire",
        "accuracy": 80,
    }

    assert [
        env._hero_aoe_accuracy_for_target(nosferatu, target_index)
        for target_index in range(6)
    ] == [80.0, 70.0, 60.0, 50.0, 40.0, 30.0]


@pytest.mark.parametrize(
    "attacker",
    (
        {"name": "Маг", "hero": False, "unit_type": "Mage", "accuracy": 80},
        {"name": "Вампир", "hero": False, "unit_type": "Vampire", "accuracy": 80},
        {"name": "Носферату", "hero": False, "unit_type": "Vampire", "accuracy": 80},
    ),
)
def test_ordinary_mages_and_vampires_do_not_get_hero_accuracy_falloff(attacker):
    env = BattleEnv(log_enabled=False)

    assert [
        env._hero_aoe_accuracy_for_target(attacker, target_index)
        for target_index in range(6)
    ] == [80.0] * 6


def test_hero_aoe_accuracy_falloff_is_applied_in_living_position_order():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=123)
    attacker = next(
        unit
        for unit in env.combined
        if unit.get("team") == "blue" and int(unit.get("position", 0)) == 8
    )
    attacker.update(
        {
            "name": "Архимаг",
            "hero": True,
            "unit_type": "Mage",
            "accuracy": 80,
            "damage": 0,
            "attack_type_primary": "Air",
        }
    )
    red_units = sorted(
        (unit for unit in env.combined if unit.get("team") == "red"),
        key=lambda unit: int(unit.get("position", 999)),
    )
    for unit in red_units:
        position = int(unit.get("position", 0))
        unit.update(
            {
                "name": f"Enemy {position}",
                "health": 100,
                "max_health": 100,
                "armor": 0,
                "immunity": [],
                "resistance": [],
            }
        )
    # A dead/empty slot does not consume one of the successive-target penalties.
    red_units[1]["health"] = 0

    rolled_accuracies = []
    env._roll_accuracy = lambda accuracy: rolled_accuracies.append(accuracy) or False

    applied, reason = env._attack(attacker, None)

    assert (applied, reason) == (True, "aoe")
    assert rolled_accuracies == [80.0, 70.0, 60.0, 50.0, 40.0]
