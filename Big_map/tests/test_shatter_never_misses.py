"""Shatter ignores its own accuracy, as in Disciples II, once the main attack hits."""

import pytest

from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


UNITS = {unit["кто"]: unit for unit in DATA}


def setup(attacker_name, attacker_pos, victim_positions, accuracy):
    attacker = map_unit_to_battle(UNITS[attacker_name], "blue", attacker_pos)
    attacker.update(initiative_base=1000, initiative=1000)
    victims = [map_unit_to_battle(UNITS["Рыцарь"], "red", pos) for pos in victim_positions]
    for victim in victims:
        victim.update(initiative_base=0, initiative=0, immunity=[], resistance=[])
    red = victims + [placeholder_unit("red", pos) for pos in range(1, 7)
                     if pos not in victim_positions]
    blue = [attacker] + [placeholder_unit("blue", pos) for pos in range(7, 13)
                         if pos != attacker_pos]
    battle = BattleEnv(log_enabled=False)
    battle.rng.seed(3)
    battle._init_with_custom_teams(red, blue)
    attacker = battle._unit_by_position(attacker_pos)
    # Shatter's own POWER is zero: only a never-missing shatter can land.
    attacker.update(accuracy=accuracy, accuracy_secondary=0)
    victims = [battle._unit_by_position(pos) for pos in victim_positions]
    for victim in victims:
        victim.update(armor=50, base_armor=50, health=5000, hp=5000, max_health=5000)
    return battle, attacker, victims


@pytest.mark.parametrize("name,attacker_pos,victim_positions", [
    ("Теург", 10, (1, 2, 4)),
    ("Сэр Аллемон", 7, (1,)),
])
def test_shatter_lands_on_every_hit_target(name, attacker_pos, victim_positions):
    battle, attacker, victims = setup(name, attacker_pos, victim_positions, accuracy=100)
    battle._attack(attacker, victim_positions[0])
    for victim in victims:
        assert victim["health"] < 5000
        assert victim["armor"] == 35
        assert victim["shattered_armor"] == 15


@pytest.mark.parametrize("name,attacker_pos,victim_positions", [
    ("Теург", 10, (1, 2, 4)),
    ("Сэр Аллемон", 7, (1,)),
])
def test_missed_main_attack_does_not_shatter(name, attacker_pos, victim_positions):
    battle, attacker, victims = setup(name, attacker_pos, victim_positions, accuracy=0)
    battle._attack(attacker, victim_positions[0])
    for victim in victims:
        assert victim["health"] == 5000
        assert victim["armor"] == 50
        assert not victim.get("shattered_armor")
