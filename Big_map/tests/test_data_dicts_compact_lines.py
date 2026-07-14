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


def test_elf_queen_accuracy_matches_original_unit():
    source = _unit_entry("Королева эльфов")
    battle_unit = map_unit_to_battle(source, team="red", position=1)

    assert source["точн"] == 0.95
    assert battle_unit["accuracy"] == 95


def test_corrected_original_unit_levels_are_one():
    unit_names = (
        "Оживший доспех",
        "Голем",
        "Рыцарь на пегасе",
        "Рейнджер людей",
        "Архимаг",
        "Архангел",
        "Вор империи",
        "Ашкаэль",
        "Вор легионов",
        "Рух",
        "Каменный предок",
        "Видар",
        "Вор гномов",
        "Скелет",
        "Темный энт",
        "Ашган",
        "Вор нежити",
        "Малый энт",
        "Энт",
        "Великий энт",
        "Буйный энт",
        "Вор эльфов",
        "Титан",
        "Сатир",
        "Йети",
        "Оборотень",
    )

    for unit_name in unit_names:
        assert _unit_entry(unit_name)["уровень"] == 1


def test_corrected_special_unit_levels_sizes_and_experience():
    expected = {
        "Адская гончая": {"уровень": 1, "размер": 1},
        "Белиарх": {"уровень": 1, "размер": 1},
        "Мститель": {"уровень": 1, "размер": 1},
        "Валькирия": {"уровень": 1, "размер": 0},
        "Кошмар": {"уровень": 1, "размер": 0},
        "Танатос": {"уровень": 1, "иммунитет": ["poison", "death"]},
        "Грифон": {"уровень": 1, "опыт убийства": 105},
        "Повелитель небес": {"уровень": 2, "опыт убийства": 240},
    }

    for unit_name, expected_fields in expected.items():
        source = _unit_entry(unit_name)
        for field, expected_value in expected_fields.items():
            assert source[field] == expected_value

    for unit_name in ("Адская гончая", "Белиарх", "Мститель"):
        assert map_unit_to_battle(_unit_entry(unit_name), team="red", position=1)["big"] is True
    for unit_name in ("Валькирия", "Кошмар"):
        assert map_unit_to_battle(_unit_entry(unit_name), team="red", position=1)["big"] is False


def test_corrected_experience_hp_defenses_and_initiative():
    expected = {
        "Ученик": {"нужный опыт": 75},
        "Берсерк": {"нужный опыт": 550},
        "Сектант": {"нужный опыт": 75},
        "Священник": {"опыт убийства": 160},
        "Король гномов": {"опыт убийства": 300, "нужный опыт": 2500},
        "Воин": {"опыт убийства": 25, "нужный опыт": 95},
        "Адепт": {"опыт убийства": 20},
        "Вампир": {"нужный опыт": 2300},
        "Лич": {"нужный опыт": 2425},
        "Архилич": {"нужный опыт": 3500},
        "Разбойник": {"опыт убийства": 15, "нужный опыт": 50},
        "Морской змей": {"нужный опыт": 2000},
        "Элементаль Воздуха": {"нужный опыт": 400},
        "Толстый бес": {"нужный опыт": 170},
        "Дикий гигант": {"нужный опыт": 1250},
        "Белый маг": {"здоровье": 125},
        "Дух": {"здоровье": 75},
        "Элементалист": {"защита": ["air"]},
        "Модеус": {"защита": ["fire"]},
        "Ледяной гигант": {"иммунитет": ["water"]},
        "Горец": {"инит": 40},
        "Отшельник": {"инит": 40},
        "Хьюберт де Лейли": {"инит": 60},
    }

    for unit_name, expected_fields in expected.items():
        source = _unit_entry(unit_name)
        for field, expected_value in expected_fields.items():
            assert source[field] == expected_value

    assert map_unit_to_battle(_unit_entry("Белый маг"), team="blue", position=1)["health"] == 125
    assert map_unit_to_battle(_unit_entry("Дух"), team="blue", position=1)["health"] == 75
    assert map_unit_to_battle(_unit_entry("Элементалист"), team="blue", position=1)["resistance"] == ["Air"]
    assert map_unit_to_battle(_unit_entry("Модеус"), team="blue", position=1)["resistance"] == ["Fire"]
    assert map_unit_to_battle(_unit_entry("Ледяной гигант"), team="blue", position=1)["immunity"] == ["Water"]


def test_undead_fighter_upgrade_branches_include_templar():
    fighter = _unit_entry("Воин")
    zombie = _unit_entry("Зомби")
    templar = _unit_entry("Тамплиер")

    assert fighter["превращается в"] == ["Зомби", "Тамплиер"]
    assert zombie["происходитиз"] == "Воин"
    assert templar["происходитиз"] == "Воин"


def test_neutral_elf_variants_are_distinct_game_units_with_original_stats():
    expected = {
        "Нейтральный лорд эльфов": {
            "unit_id": "g000uu5008",
            "health": 195,
            "damage": 50,
            "initiative": 50,
            "unit_type": "Mage",
            "attack_type_primary": "Air",
            "resistance": ["Air"],
            "big": False,
            "exp_kill": 290,
            "exp_required": 2850,
        },
        "Нейтральный эльф-рейнджер": {
            "unit_id": "g000uu5009",
            "health": 45,
            "damage": 25,
            "initiative": 65,
            "unit_type": "Archer",
            "attack_type_primary": "Weapon",
            "resistance": [],
            "big": False,
            "exp_kill": 20,
            "exp_required": 75,
        },
        "Нейтральный грифон": {
            "unit_id": "g000uu5004",
            "health": 200,
            "damage": 95,
            "initiative": 50,
            "unit_type": "Warrior",
            "attack_type_primary": "Weapon",
            "resistance": [],
            "big": True,
            "exp_kill": 150,
            "exp_required": 1200,
        },
        "Нейтральный повелитель небес": {
            "unit_id": "g000uu5010",
            "health": 420,
            "damage": 140,
            "initiative": 50,
            "unit_type": "Warrior",
            "attack_type_primary": "Weapon",
            "resistance": [],
            "big": True,
            "exp_kill": 470,
            "exp_required": 2350,
        },
        "Нейтральный эльфийский оракул": {
            "unit_id": "g000uu5006",
            "health": 125,
            "damage": 60,
            "initiative": 10,
            "unit_type": "Profit",
            "attack_type_primary": "Life",
            "resistance": ["Air"],
            "big": False,
            "exp_kill": 300,
            "exp_required": 2200,
        },
        "Нейтральный кентавр-копейщик": {
            "unit_id": "g000uu5011",
            "health": 165,
            "damage": 65,
            "initiative": 55,
            "unit_type": "Warrior",
            "attack_type_primary": "Weapon",
            "resistance": [],
            "big": False,
            "exp_kill": 80,
            "exp_required": 625,
        },
    }

    assert len({entry["unit_id"] for entry in expected.values()}) == len(expected)

    for unit_name, expected_fields in expected.items():
        source = _unit_entry(unit_name)
        battle_unit = map_unit_to_battle(source, team="red", position=1)

        assert source["уровень"] == 1
        assert source["столица"] == 0
        assert battle_unit["Level"] == 1
        assert battle_unit["capital"] == 0
        assert battle_unit["is_neutral_unit"] is True
        assert battle_unit["hero"] is False
        for field, expected_value in expected_fields.items():
            assert battle_unit[field] == expected_value
