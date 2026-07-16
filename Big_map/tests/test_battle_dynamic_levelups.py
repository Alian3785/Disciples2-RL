import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _unit(name: str, position: int = 7) -> dict:
    source = next(entry for entry in DATA if entry.get("кто") == name)
    return map_unit_to_battle(source, team="blue", position=position)


def _enemy(exp_kill: int = 1) -> dict:
    return {
        "name": "Test enemy",
        "team": "red",
        "position": 1,
        "health": 0,
        "max_health": 1,
        "running_away": 0,
        "exp_kill": int(exp_kill),
    }


def _reach_level(env: BattleEnv, unit: dict) -> None:
    unit["exp_current"] = int(unit["exp_required"]) - 1
    env.combined = [unit, _enemy()]
    env._apply_battle_exp("red")


def test_neutral_unit_gains_level_and_keeps_same_exp_requirement():
    env = BattleEnv(log_enabled=False)
    goblin = _unit("Гоблин")
    goblin["health"] = 25
    goblin["max_health"] = 50

    _reach_level(env, goblin)

    assert goblin["Level"] == 2
    assert goblin["exp_current"] == 0
    assert goblin["exp_required"] == 50
    assert goblin["health"] == 28
    assert goblin["max_health"] == 55
    assert goblin["damage"] == 17
    assert goblin["original_damage"] == 17
    assert goblin["accuracy"] == 81
    assert env.last_levelups == ["Гоблин"]


def test_terminal_faction_healer_gains_level_without_transforming():
    env = BattleEnv(log_enabled=False)
    patriarch = _unit("Патриарх")

    _reach_level(env, patriarch)

    assert patriarch["Level"] == 5
    assert patriarch["exp_current"] == 0
    assert patriarch["exp_required"] == 2000
    assert patriarch["health"] == 138
    assert patriarch["max_health"] == 138
    assert patriarch["damage"] == 132
    assert patriarch["accuracy"] == 100
    assert patriarch["turns_into"] == []


def test_intermediate_faction_unit_still_waits_for_campaign_evolution():
    env = BattleEnv(log_enabled=False)
    priest = _unit("Жрец")
    initial = deepcopy(priest)

    _reach_level(env, priest)

    assert priest["Level"] == initial["Level"]
    assert priest["exp_current"] == priest["exp_required"] - 1
    assert priest["exp_required"] == initial["exp_required"]
    assert priest["health"] == initial["health"]
    assert priest["damage"] == initial["damage"]
    assert priest["accuracy"] == initial["accuracy"]
    assert priest["turns_into"] == ["Священник"]
    assert env.last_levelups == ["Жрец"]


def test_explicit_boss_flag_uses_dynamic_level_even_with_evolution_targets():
    env = BattleEnv(log_enabled=False)
    boss = _unit("Жрец")
    boss["boss"] = True

    _reach_level(env, boss)

    assert boss["Level"] == 3
    assert boss["exp_current"] == 0
    assert boss["exp_required"] == 475
    assert boss["health"] == 83
    assert boss["damage"] == 44
    assert boss["accuracy"] == 100


def test_super_last_stand_bethrezen_uses_neutral_dynamic_leveling():
    campaign = CampaignEnv(map_name="super_last_stand", log_enabled=False)
    campaign.reset(seed=123)
    bethrezen = deepcopy(campaign._resolve_travel_hero())
    battle = BattleEnv(log_enabled=False)

    _reach_level(battle, bethrezen)

    assert bethrezen["name"] == "Бетрезен"
    assert bethrezen["Level"] == 2
    assert bethrezen["exp_current"] == 0
    assert bethrezen["exp_required"] == 650
    assert bethrezen["health"] == 330
    assert bethrezen["max_health"] == 330
    assert bethrezen["damage"] == 110
    assert bethrezen["accuracy"] == 91
    assert _unit("Утер")["exp_required"] == 5750


def test_super_last_stand_bethrezen_level_persists_back_to_campaign():
    campaign = CampaignEnv(
        map_name="super_last_stand",
        log_enabled=False,
        persist_blue_hp=True,
    )
    campaign.reset(seed=123)
    battle = BattleEnv(log_enabled=False)
    battle_blue = deepcopy(campaign.blue_team_state)
    bethrezen = next(unit for unit in battle_blue if unit.get("name") == "Бетрезен")
    bethrezen["exp_current"] = bethrezen["exp_required"] - 1
    battle.combined = [*battle_blue, _enemy()]
    battle._apply_battle_exp("red")
    campaign.battle_env = battle

    campaign._save_blue_state()

    saved = campaign._resolve_travel_hero()
    assert saved["Level"] == 2
    assert saved["exp_current"] == 0
    assert saved["exp_required"] == 650
    assert saved["max_health"] == 330
    assert saved["damage"] == 110
    assert saved["accuracy"] == 91
    assert campaign.moves_per_turn == 25


def test_super_last_stand_bethrezen_uses_archdevil_xp_curve_across_levels():
    campaign = CampaignEnv(map_name="super_last_stand", log_enabled=False)
    campaign.reset(seed=123)
    bethrezen = deepcopy(campaign._resolve_travel_hero())
    battle = BattleEnv(log_enabled=False)

    _reach_level(battle, bethrezen)
    assert bethrezen["Level"] == 2
    assert bethrezen["exp_required"] == 650

    _reach_level(battle, bethrezen)
    assert bethrezen["Level"] == 3
    assert bethrezen["exp_current"] == 0
    assert bethrezen["exp_required"] == 1150
