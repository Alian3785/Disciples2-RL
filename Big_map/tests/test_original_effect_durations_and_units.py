"""Effect lengths follow Gattacks.INFINITE; unit reach/immunities follow Gunits."""

import pytest

from battle_env import AOE_TYPES, MELEE_TYPES, BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


UNITS = {unit["кто"]: unit for unit in DATA}


def setup(attacker_name, attacker_pos, victim_positions=(1,)):
    attacker = map_unit_to_battle(UNITS[attacker_name], "blue", attacker_pos)
    attacker.update(initiative_base=1000, initiative=1000)
    victims = [map_unit_to_battle(UNITS["Рыцарь"], "red", pos) for pos in victim_positions]
    for victim in victims:
        victim.update(initiative_base=0, initiative=0, immunity=[], resistance=[])
    red = victims + [placeholder_unit("red", pos) for pos in range(1, 7)
                     if pos not in victim_positions]
    occupied = {attacker_pos, attacker_pos + 3} if attacker.get("big") else {attacker_pos}
    blue = [attacker] + [placeholder_unit("blue", pos) for pos in range(7, 13)
                         if pos not in occupied]
    battle = BattleEnv(log_enabled=False)
    battle.rng.seed(5)
    battle._init_with_custom_teams(red, blue)
    attacker = battle._unit_by_position(attacker_pos)
    attacker.update(accuracy=100, accuracy_secondary=100)
    victims = [battle._unit_by_position(pos) for pos in victim_positions]
    for victim in victims:
        victim.update(health=5000, hp=5000, max_health=5000)
    return battle, attacker, victims


@pytest.mark.parametrize("name,long_paralysis", [
    ("Русалка", False),      # Paralyze INFINITE=F
    ("Утер демон", True),    # same engine type, INFINITE=T
])
def test_mass_paralysis_length_follows_infinite_flag(name, long_paralysis):
    battle, caster, (victim,) = setup(name, 10)
    battle._attack(caster, victim["position"])
    assert bool(victim.get("long_paralyzed")) is long_paralysis
    assert bool(victim.get("paralyzed")) is not long_paralysis


@pytest.mark.parametrize("name,long_paralysis", [
    ("Тёмный эльф призрак", True),   # Paralyze INFINITE=T
    ("Призрак", False),              # same engine type, finite
])
def test_ghost_paralysis_length_follows_infinite_flag(name, long_paralysis):
    battle, ghost, (victim,) = setup(name, 10)
    battle._attack(ghost, victim["position"])
    assert bool(victim.get("long_paralyzed")) is long_paralysis
    assert bool(victim.get("paralyzed")) is not long_paralysis


@pytest.mark.parametrize("seed", range(8))
def test_hermit_slow_ends_when_the_next_turn_begins(seed):
    battle, hermit, (victim,) = setup("Отшельник", 10)
    battle.rng.seed(seed)
    base = victim["initiative_base"]
    assert battle._apply_hermit_initiative_slow(hermit, victim)
    assert victim["initiative_base"] == round(base * 0.5)
    victim["round_effects_done"] = 0
    battle._apply_start_of_turn_effects(victim)
    assert not victim.get("hermited")
    assert victim["initiative_base"] == base


def test_lyf_is_always_immune_to_mind_and_water():
    lyf = map_unit_to_battle(UNITS["Тёмный эльф Лиф"], "red", 1)
    assert {str(i).lower() for i in lyf["immunity"]} == {"mind", "water"}
    assert not lyf.get("resistance")
    battle = BattleEnv(log_enabled=False)
    for source in ("Water", "Mind", "Water"):
        assert battle._is_immune_damage({"attack_type_primary": source}, lyf)


def test_beliarh_is_melee_and_verdant_hits_everyone():
    beliarh = map_unit_to_battle(UNITS["Белиарх"], "blue", 7)
    assert beliarh["unit_type"] in MELEE_TYPES
    verdant = map_unit_to_battle(UNITS["Буйный энт"], "blue", 7)
    assert verdant["unit_type"] in AOE_TYPES

    battle, verdant, victims = setup("Буйный энт", 7, (1, 2, 4, 6))
    battle._attack(verdant, 1)
    assert all(victim["health"] < 5000 for victim in victims)
