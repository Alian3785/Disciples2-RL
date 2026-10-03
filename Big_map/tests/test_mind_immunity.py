"""Mind immunity must block primary and secondary hostile status effects."""

from copy import deepcopy

import pytest

from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle

UNITS = {unit["кто"]: unit for unit in DATA}
MIND_IMMUNE_NAMES = [
    name for name, unit in UNITS.items() if "mind" in unit["иммунитет"]
]
MIND_ATTACKER_NAMES = [
    name for name, unit in UNITS.items()
    if "mind" in (unit["тип атаки1"], unit["тип атаки2"])
    and unit["тип"] != "Doppelganger"  # Copies allies, not a hostile effect.
]
STATUS_FLAGS = (
    "transformed", "paralyzed", "long_paralyzed", "running_away", "feared",
    "teamated",
)


def make_battle(attacker_name, victim_name, attacker_team="red"):
    battle = BattleEnv(log_enabled=True)
    victim_team = "blue" if attacker_team == "red" else "red"
    attacker = map_unit_to_battle(
        UNITS[attacker_name], attacker_team, 1 if attacker_team == "red" else 7,
    )
    victim = map_unit_to_battle(
        UNITS[victim_name], victim_team, 1 if victim_team == "red" else 7,
    )
    # Guarantee hits and enough HP for secondary effects to reach a living target.
    attacker.update(accuracy=100, accuracy_secondary=100)
    victim.update(health=10000, max_health=10000)
    battle.combined = [attacker, victim]
    battle.rng.seed(42)
    return battle, attacker, victim


@pytest.mark.parametrize("attacker_team", ["red", "blue"])
@pytest.mark.parametrize("victim_name", MIND_IMMUNE_NAMES)
@pytest.mark.parametrize("attacker_name", MIND_ATTACKER_NAMES)
def test_all_mind_attacks_respect_every_immune_unit(
    attacker_name, victim_name, attacker_team,
):
    battle, attacker, victim = make_battle(attacker_name, victim_name, attacker_team)
    before = deepcopy(victim)
    # A primary ward can absorb the first hit before its secondary effect runs.
    for _ in range(2):
        assert battle._attack(attacker, victim["position"])[0]
        assert victim["health"] > 0
        assert not any(victim.get(flag, 0) for flag in STATUS_FLAGS)
        for key in ("unit_type", "damage", "immunity", "resistance", "basestats"):
            assert victim[key] == before[key]
    if not attacker["attack_type_secondary"]:
        assert victim["health"] == before["health"]


@pytest.mark.parametrize("attacker_name,flag", [
    ("Суккуб", "transformed"), ("Шаманка", "feared"),
])
def test_mind_ward_blocks_once_then_effect_applies(attacker_name, flag):
    battle, attacker, victim = make_battle(attacker_name, "Скваер")
    victim["resistance"] = ["Mind"]
    battle._attack(attacker, victim["position"])
    assert not victim.get(flag, 0)
    assert victim["resilience_used_types"] == ["Mind"]
    battle._attack(attacker, victim["position"])
    assert victim[flag] == 1


@pytest.mark.parametrize("attacker_name,flag", [
    ("Суккуб", "transformed"), ("Шаманка", "feared"),
])
@pytest.mark.parametrize("immunity", ["Mind", "mind", "MIND"])
def test_immunity_does_not_consume_mind_ward(attacker_name, flag, immunity):
    battle, attacker, victim = make_battle(attacker_name, "Скваер")
    victim.update(immunity=[immunity], resistance=["Mind"])
    battle._attack(attacker, victim["position"])
    assert not victim.get(flag, 0)
    assert not victim.get("resilience_used_types", [])
    assert any("Mind" in event for event in battle._pretty_events)


def test_succub_mixed_targets_keep_immunity_and_normal_transformations():
    battle, attacker, protected = make_battle("Суккуб", "Охотник на ведьм")
    protected_before = deepcopy(protected)
    small = map_unit_to_battle(UNITS["Скваер"], "blue", 8)
    big = map_unit_to_battle(UNITS["Демон"], "blue", 9)
    battle.combined.extend([small, big])
    assert battle._attack(attacker, protected["position"]) == (True, "aoe")
    assert protected == protected_before
    for victim, form in ((small, "Бес"), (big, "Толстый бес")):
        assert victim["transformed"] == 1
        assert victim["damage"] == UNITS[form]["урон"]


@pytest.mark.parametrize("guard_name", [
    "Ашган", "Ашкаэль", "Видар", "Мизраэль", "Иллюмиэлль",
])
def test_succub_still_cannot_transform_capital_guards(guard_name):
    battle, attacker, guard = make_battle("Суккуб", guard_name)
    before = deepcopy(guard)
    battle._attack(attacker, guard["position"])
    assert guard == before


def test_shamanka_secondary_fear_uses_mind_not_primary_earth():
    battle, attacker, victim = make_battle("Шаманка", "Скваер")
    victim["resistance"] = ["Earth", "Mind"]
    # Earth absorbs the primary attack, then Mind absorbs the secondary fear.
    battle._attack(attacker, victim["position"])
    assert victim["resilience_used_types"] == ["Earth"]
    assert not victim.get("feared", 0)
    battle._attack(attacker, victim["position"])
    assert victim["resilience_used_types"] == ["Earth", "Mind"]
    assert not victim.get("feared", 0)
    battle._attack(attacker, victim["position"])
    assert victim["feared"] == 1


def test_primary_immunity_does_not_block_unrelated_secondary_status():
    battle, attacker, victim = make_battle("Шаманка", "Скваер")
    victim["immunity"] = ["Earth"]
    assert battle._apply_fear_effect(attacker, victim)
    assert victim["feared"] == 1
