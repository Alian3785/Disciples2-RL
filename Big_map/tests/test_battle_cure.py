import pytest

from battle_env import BattleEnv


def _unit(
    *,
    name: str,
    team: str,
    position: int,
    unit_type: str = "Warrior",
    damage: int = 100,
    armor: int = 40,
) -> dict:
    return {
        "name": name,
        "team": team,
        "position": position,
        "stand": "ahead",
        "unit_type": unit_type,
        "health": 100,
        "max_health": 100,
        "damage": damage,
        "original_damage": damage,
        "damage_secondary": 0,
        "initiative_base": 50,
        "initiative": 50,
        "accuracy": 100,
        "accuracy_secondary": 100,
        "attack_type_primary": "Life" if unit_type == "Profit" else "Weapon",
        "attack_type_secondary": "",
        "immunity": [],
        "resistance": [],
        "resilience_used_types": [],
        "armor": armor,
        "base_armor": armor,
        "big": False,
        "paralyzed": 0,
        "long_paralyzed": 0,
        "running_away": 0,
        "feared": 0,
        "transformed": 0,
        "basestats": {},
        "hero": False,
    }


@pytest.mark.parametrize(
    ("healer_name", "should_cure"),
    [
        ("Клирик", False),
        ("Нейтральный эльфийский оракул", False),
        ("Аббатиса", True),
        ("Прорицательница", True),
        ("Matriarch", True),
        ("Prophetess", True),
    ],
)
def test_only_original_profit_units_have_mass_cure(
    healer_name: str, should_cure: bool
):
    env = BattleEnv(log_enabled=False)
    healer = _unit(
        name=healer_name,
        team="red",
        position=1,
        unit_type="Profit",
        damage=40,
        armor=0,
    )
    ally = _unit(name="Ally", team="red", position=2)
    ally["paralyzed"] = 1
    ally["poison_turns_left"] = 2
    ally["poison_damage_per_tick"] = 10
    enemy = _unit(name="Enemy", team="blue", position=7)
    env.combined = [healer, ally, enemy]

    assert env._attack(healer, None) == (True, "aoe")
    assert (ally["paralyzed"] == 0) is should_cure
    assert (ally["poison_turns_left"] == 0) is should_cure


def test_cure_removes_every_supported_original_battle_debuff():
    env = BattleEnv(log_enabled=False)
    unit = _unit(name="Target", team="blue", position=7)
    unit.update(
        {
            "poison_turns_left": 2,
            "poison_damage_per_tick": 10,
            "burn_turns_left": 3,
            "burn_damage_per_tick": 12,
            "uran_turns_left": 1,
            "uran_damage_per_tick": 15,
            "paralyzed": 1,
            "long_paralyzed": 1,
            "running_away": 1,
            "feared": 1,
            "hermited": 1,
            "initiative_base": 25,
            "initiative": 20,
            "teamated": 1,
            "damage": 68,
            "lower_damage_original_damage": 100,
            "lower_damage_original_unit_type": "Warrior",
            "armor": 10,
            "shattered_armor": 30,
            "shatter_original_armor": 40,
            "shatter_original_unit_type": "Warrior",
        }
    )

    assert env._cleanse_negative_effects(unit) is True

    assert unit["poison_turns_left"] == 0
    assert unit["poison_damage_per_tick"] == 0
    assert unit["burn_turns_left"] == 0
    assert unit["burn_damage_per_tick"] == 0
    assert unit["uran_turns_left"] == 0
    assert unit["uran_damage_per_tick"] == 0
    assert unit["paralyzed"] == 0
    assert unit["long_paralyzed"] == 0
    assert unit["running_away"] == 0
    assert unit["feared"] == 0
    assert unit["hermited"] == 0
    assert unit["initiative_base"] == 50
    assert unit["initiative"] == 50
    assert unit["teamated"] == 0
    assert unit["damage"] == 100
    assert unit["shattered_armor"] == 0
    assert unit["armor"] == 40


def test_cure_does_not_cancel_voluntary_retreat():
    env = BattleEnv(log_enabled=False)
    unit = _unit(name="Retreating", team="blue", position=7)
    unit["running_away"] = 1
    unit["feared"] = 0
    unit["poison_turns_left"] = 1
    unit["poison_damage_per_tick"] = 5

    assert env._cleanse_negative_effects(unit) is True
    assert unit["running_away"] == 1
    assert unit["poison_turns_left"] == 0


def test_fear_is_marked_separately_and_cure_cancels_it():
    env = BattleEnv(log_enabled=False)
    attacker = _unit(name="Baroness", team="red", position=1)
    attacker["attack_type_primary"] = "Mind"
    victim = _unit(name="Target", team="blue", position=7)

    assert env._apply_fear_effect(attacker, victim) is True
    assert victim["running_away"] == 1
    assert victim["feared"] == 1

    assert env._cleanse_negative_effects(victim) is True
    assert victim["running_away"] == 0
    assert victim["feared"] == 0


def test_cure_restores_tiamat_lowered_damage():
    env = BattleEnv(log_enabled=False)
    attacker = _unit(name="Tiamat", team="red", position=1)
    attacker["attack_type_secondary"] = "Mind"
    victim = _unit(name="Target", team="blue", position=7, damage=100)

    assert env._apply_tiamat_damage_debuff(attacker, victim) is True
    assert victim["damage"] == 68
    assert victim["teamated"] == 1

    assert env._cleanse_negative_effects(victim) is True
    assert victim["damage"] == 100
    assert victim["teamated"] == 0


def test_cure_restores_all_armor_removed_by_repeated_shatter():
    env = BattleEnv(log_enabled=False)
    attacker = _unit(name="Theurgist", team="red", position=1)
    attacker["attack_type_secondary"] = "Earth"
    victim = _unit(name="Target", team="blue", position=7, armor=40)

    assert env._apply_teurg_armor_shred(attacker, victim) is True
    assert env._apply_teurg_armor_shred(attacker, victim) is True
    assert victim["armor"] == 10
    assert victim["shattered_armor"] == 30

    assert env._cleanse_negative_effects(victim) is True
    assert victim["armor"] == 40
    assert victim["shattered_armor"] == 0
