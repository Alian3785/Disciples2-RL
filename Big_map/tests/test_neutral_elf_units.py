from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _unit(name: str, team: str, position: int) -> dict:
    source = next(entry for entry in DATA if entry.get("кто") == name)
    return map_unit_to_battle(source, team=team, position=position)


def test_neutral_elf_lord_uses_original_mass_air_attack():
    env = BattleEnv(log_enabled=False)
    attacker = _unit("Нейтральный лорд эльфов", "blue", 7)
    target_a = _unit("Скваер", "red", 1)
    target_b = _unit("Скваер", "red", 2)
    attacker["accuracy"] = 100
    env.combined = [attacker, target_a, target_b]

    hit, reason = env._attack(attacker, target_pos=None)

    assert (hit, reason) == (True, "aoe")
    assert target_a["health"] < target_a["max_health"]
    assert target_b["health"] < target_b["max_health"]


def test_neutral_oracle_uses_original_mass_heal():
    env = BattleEnv(log_enabled=False)
    oracle = _unit("Нейтральный эльфийский оракул", "blue", 10)
    ally_a = _unit("Скваер", "blue", 7)
    ally_b = _unit("Скваер", "blue", 8)
    enemy = _unit("Скваер", "red", 1)
    ally_a["health"] = 10
    ally_b["health"] = 20
    env.combined = [oracle, ally_a, ally_b, enemy]

    hit, reason = env._attack(oracle, target_pos=None)

    assert (hit, reason) == (True, "aoe")
    assert ally_a["health"] == 70
    assert ally_b["health"] == 80
    assert enemy["health"] == 100
