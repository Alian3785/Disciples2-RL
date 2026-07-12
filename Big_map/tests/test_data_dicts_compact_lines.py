import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_dicts_compact_lines import (  # noqa: E402
    DATA,
    map_unit_to_battle,
)


def _unit_entry(name: str) -> dict:
    return next(entry for entry in DATA if entry.get("кто") == name)


def test_map_unit_to_battle_uses_position_for_stand():
    archer = map_unit_to_battle(_unit_entry("Стрелок"), team="blue", position=10)
    warrior = map_unit_to_battle(_unit_entry("Скваер"), team="blue", position=7)

    assert archer["unit_type"] == "Archer"
    assert archer["attack_type_primary"] == "Weapon"
    assert archer["stand"] == "behind"
    assert warrior["stand"] == "ahead"


def test_paladin_and_mage_upgrade_lines_match_original_unit_stats():
    expected = {
        "Паладин": {
            "уровень": 4,
            "здоровье": 175,
            "броня": 0.3,
            "урон": 100,
            "инит": 50,
            "опыт убийства": 200,
            "нужный опыт": 1600,
            "происходитиз": "Имперский рыцарь",
            "превращается в": ["Защитник Веры", "Мастер клинка"],
        },
        "Защитник Веры": {
            "уровень": 5,
            "здоровье": 225,
            "броня": 0.3,
            "урон": 125,
            "инит": 70,
            "опыт убийства": 320,
            "нужный опыт": 2100,
            "происходитиз": "Паладин",
            "превращается в": [],
        },
        "Маг": {
            "уровень": 2,
            "здоровье": 65,
            "урон": 30,
            "инит": 40,
            "опыт убийства": 55,
            "нужный опыт": 550,
            "происходитиз": "Ученик",
            "превращается в": ["Волшебник", "Элементалист"],
        },
        "Волшебник": {
            "уровень": 3,
            "здоровье": 95,
            "урон": 45,
            "инит": 40,
            "опыт убийства": 120,
            "нужный опыт": 1200,
            "происходитиз": "Маг",
            "превращается в": ["Белый маг"],
        },
    }

    for unit_name, expected_fields in expected.items():
        unit = _unit_entry(unit_name)
        for field, expected_value in expected_fields.items():
            assert unit[field] == expected_value


def test_paralyzing_secondary_attacks_have_mind_source():
    for unit_name in ("Упырь", "Гигантский паук"):
        source = _unit_entry(unit_name)
        battle_unit = map_unit_to_battle(source, team="red", position=1)

        assert source["тип атаки2"] == "mind"
        assert battle_unit["attack_type_secondary"] == "Mind"


def test_thug_poison_source_and_experience_match_original_unit():
    source = _unit_entry("Головорез")
    battle_unit = map_unit_to_battle(source, team="red", position=1)

    assert source["тип атаки2"] == "death"
    assert battle_unit["attack_type_secondary"] == "Death"
    assert source["опыт убийства"] == 30
    assert source["нужный опыт"] == 250


def test_sir_allemon_armor_matches_original_unit():
    source = _unit_entry("Сэр Аллемон")
    battle_unit = map_unit_to_battle(source, team="red", position=1)

    assert source["броня"] == 0.2
    assert battle_unit["armor"] == 20


def test_sentry_and_watcher_attacks_match_original_units():
    sentry = _unit_entry("Дозорный")
    sentry_battle = map_unit_to_battle(sentry, team="blue", position=1)
    watcher = _unit_entry("Смотритель")
    watcher_battle = map_unit_to_battle(watcher, team="blue", position=1)

    assert sentry["точн"] == 0.85
    assert sentry_battle["accuracy"] == 85
    assert watcher["урон"] == 60
    assert watcher_battle["damage"] == 60
    assert watcher["урон2"] == 15
    assert watcher_battle["damage_secondary"] == 15
