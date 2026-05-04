import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _unit_at(units: list[dict], position: int) -> dict:
    return next(unit for unit in units if int(unit.get("position", -1)) == int(position))


def _summoned_blue_unit(position: int, summoner_position: int = 8) -> dict:
    return {
        "name": "Summoned Zombie",
        "team": "blue",
        "position": int(position),
        "unit_type": "Warrior",
        "initiative": 50,
        "initiative_base": 50,
        "damage": 30,
        "damage_secondary": 0,
        "health": 100.0,
        "hp": 100.0,
        "max_health": 100.0,
        "maxhp": 100.0,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "immunity": [],
        "resistance": [],
        "big": False,
        "Summoned": int(summoner_position),
    }


def test_save_blue_state_restores_empty_start_slot_after_summon():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    position = 10
    assert env._is_empty_blue_unit(_unit_at(env.blue_team_state, position))

    battle_units = deepcopy(env.blue_team_state)
    slot = _unit_at(battle_units, position)
    slot.clear()
    slot.update(_summoned_blue_unit(position))

    env.battle_env = type("DummyBattle", (), {"combined": battle_units})()
    env._save_blue_state()

    saved_slot = _unit_at(env.blue_team_state, position)
    assert env._is_empty_blue_unit(saved_slot)
    assert "Summoned" not in saved_slot


def test_save_blue_state_removes_summoned_unit_from_occupied_start_slot():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    position = 7
    original_unit = deepcopy(_unit_at(env.blue_team_state, position))
    assert not env._is_empty_blue_unit(original_unit)

    battle_units = deepcopy(env.blue_team_state)
    slot = _unit_at(battle_units, position)
    slot.clear()
    slot.update(_summoned_blue_unit(position))

    env.battle_env = type("DummyBattle", (), {"combined": battle_units})()
    env._save_blue_state()

    saved_slot = _unit_at(env.blue_team_state, position)
    assert saved_slot["name"] == original_unit["name"]
    assert float(saved_slot.get("health", -1)) == 0.0
    assert float(saved_slot.get("hp", -1)) == 0.0
    assert "Summoned" not in saved_slot
