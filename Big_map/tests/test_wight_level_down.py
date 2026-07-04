from types import SimpleNamespace

from battle_env import BattleEnv, WIGHT_LEVEL_DOWN_RECOVERY_CHANCE
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _unit(name: str, team: str, position: int) -> dict:
    unit_data = next(entry for entry in DATA if entry["кто"] == name)
    unit = map_unit_to_battle(unit_data, team, position)
    unit.setdefault("original_damage", unit.get("damage", 0))
    return unit


def test_wight_level_down_uses_proishoditiz_chain_and_preserves_hp_ratio():
    env = BattleEnv(log_enabled=False)
    attacker = _unit("Сущий", "red", 1)
    victim = _unit("Ангел", "blue", 7)
    victim["health"] = 90

    assert env._apply_wight_decay_effect(attacker, victim) is True

    imperial_knight = _unit("Имперский рыцарь", "blue", 7)
    assert victim["name"] == "Ангел"
    assert victim["wight_form_name"] == "Имперский рыцарь"
    assert victim["damage"] == imperial_knight["damage"]
    assert victim["unit_type"] == imperial_knight["unit_type"]
    assert victim["max_health"] == imperial_knight["max_health"]
    assert victim["health"] == 80
    assert victim["transformed"] == 1
    assert victim["transform_recover_chance"] == WIGHT_LEVEL_DOWN_RECOVERY_CHANCE

    assert env._apply_wight_decay_effect(attacker, victim) is True

    knight = _unit("Рыцарь", "blue", 7)
    assert victim["wight_form_name"] == "Рыцарь"
    assert victim["damage"] == knight["damage"]
    assert victim["max_health"] == knight["max_health"]
    assert victim["health"] == 60

    assert env._cleanse_negative_effects(victim) is True
    original = _unit("Ангел", "blue", 7)
    assert victim.get("wight_form_name") is None
    assert victim["transformed"] == 0
    assert victim["damage"] == original["damage"]
    assert victim["unit_type"] == original["unit_type"]
    assert victim["max_health"] == original["max_health"]
    assert victim["health"] == 90


def test_wight_level_down_does_not_affect_neutral_units():
    env = BattleEnv(log_enabled=False)
    attacker = _unit("Сущий", "red", 1)
    victim = _unit("Орк", "blue", 7)
    victim["health"] = 160
    before = {
        key: victim.get(key)
        for key in (
            "accuracy",
            "damage",
            "unit_type",
            "max_health",
            "health",
            "armor",
            "Level",
        )
    }

    assert victim["is_neutral_unit"] is True
    assert env._apply_wight_decay_effect(attacker, victim) is False
    assert victim.get("wight_form_name") is None
    assert victim.get("transformed", 0) == 0
    assert {
        key: victim.get(key)
        for key in (
            "accuracy",
            "damage",
            "unit_type",
            "max_health",
            "health",
            "armor",
            "Level",
        )
    } == before


def test_wight_attacks_neutral_like_archer_without_secondary_roll():
    env = BattleEnv(log_enabled=False)
    attacker = _unit("Сущий", "red", 1)
    victim = _unit("Орк", "blue", 7)
    env.combined = [attacker, victim]
    env._roll_hit = lambda _attacker: True
    env._apply_damage_with_armor = lambda _attacker, base_dmg, _victim: int(base_dmg)

    def fail_status_roll(_chance):
        raise AssertionError("neutral Wight attack must not roll a secondary status")

    env._roll_status = fail_status_roll

    hit, reason = env._attack(attacker, victim["position"])

    assert (hit, reason) == (True, "ok")
    assert victim["health"] == victim["max_health"] - attacker["damage"]
    assert victim.get("wight_form_name") is None
    assert victim.get("transformed", 0) == 0


def test_wight_level_down_has_one_percent_start_turn_recovery():
    env = BattleEnv(log_enabled=False)
    attacker = _unit("Сущий", "red", 1)
    victim = _unit("Ангел", "blue", 7)
    victim["health"] = 90

    assert env._apply_wight_decay_effect(attacker, victim) is True

    env.rng = SimpleNamespace(random=lambda: 0.02)
    env._apply_start_of_turn_effects(victim)
    assert victim["transformed"] == 1
    assert victim["wight_form_name"] == "Имперский рыцарь"

    env.rng = SimpleNamespace(random=lambda: 0.005)
    # Имитация новой активации в следующем раунде (флаг сбрасывает _end_round_restore).
    victim["round_effects_done"] = 0
    env._apply_start_of_turn_effects(victim)
    assert victim["transformed"] == 0
    assert victim.get("wight_form_name") is None
    assert victim["health"] == 90
    assert victim["max_health"] == 225
