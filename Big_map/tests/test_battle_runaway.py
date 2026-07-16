import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, FEATURES_PER_UNIT, UNITS_BLUE, UNITS_RED
from campaign_env import CampaignEnv
from data_dicts_compact_lines import placeholder_unit


def _unit_at(units: list[dict], position: int) -> dict:
    return next(unit for unit in units if int(unit.get("position", -1)) == int(position))


class _EndedBattle:
    def __init__(
        self,
        *,
        winner: str,
        combined: list[dict],
        escaped_units: list[dict] | None = None,
    ) -> None:
        self.winner = winner
        self.combined = combined
        self.escaped_units = escaped_units or []
        self.last_battle_exp = 0.0
        self.last_levelups = []

    def _obs(self):
        return np.zeros(FEATURES_PER_UNIT * 12, dtype=np.float32)


def test_runaway_unit_becomes_empty_slot_and_is_snapshotted():
    env = BattleEnv(log_enabled=False)
    runner = deepcopy(_unit_at(UNITS_BLUE, 7))
    ally = deepcopy(_unit_at(UNITS_BLUE, 8))
    red = deepcopy(_unit_at(UNITS_RED, 1))
    runner["health"] = 37
    runner["hp"] = 37
    runner["running_away"] = 1
    runner["exp_current"] = 11
    runner["exp_required"] = 999
    ally["exp_current"] = 0
    ally["exp_required"] = 999
    red["exp_kill"] = 90
    env.combined = [red, runner, ally]

    assert env._apply_start_of_turn_effects(runner) is False

    assert len(env.escaped_units) == 1
    escaped = env.escaped_units[0]
    assert escaped["name"] != "пусто"
    assert escaped["health"] == 37
    assert escaped["exp_current"] == 11
    assert escaped["running_away"] == 0

    empty = placeholder_unit("blue", 7)
    assert runner["name"] == empty["name"]
    assert runner["health"] == 0
    assert runner["max_health"] == 0
    assert runner["damage"] == 0
    assert runner["immunity"] == []
    assert runner["resistance"] == []

    red["health"] = 0
    env._apply_battle_exp("red")

    assert escaped["exp_current"] == 11
    assert ally["exp_current"] == pytest.approx(90)


def test_dead_running_away_unit_is_not_escaped_and_counts_for_exp_on_battle_end():
    env = BattleEnv(log_enabled=False)
    winner = deepcopy(_unit_at(UNITS_BLUE, 8))
    loser = deepcopy(_unit_at(UNITS_RED, 1))
    winner["health"] = 100
    winner["hp"] = 100
    winner["exp_current"] = 0
    winner["exp_required"] = 999
    loser["health"] = 0
    loser["hp"] = 0
    loser["running_away"] = 1
    loser["exp_kill"] = 40
    env.combined = [loser, winner]

    env._check_victory_after_hit()

    assert env.winner == "blue"
    assert loser["running_away"] == 0
    assert env.escaped_units == []
    assert winner["exp_current"] == pytest.approx(40)


def test_save_blue_state_restores_escaped_units_after_blue_victory():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    position = 7
    original = deepcopy(_unit_at(env.blue_team_state, position))
    escaped = deepcopy(original)
    escaped["health"] = 23.0
    escaped["hp"] = 23.0
    escaped["exp_current"] = 17.0
    escaped["exp_required"] = 999
    escaped["running_away"] = 0

    battle_units = deepcopy(env.blue_team_state)
    slot = _unit_at(battle_units, position)
    slot.clear()
    slot.update(placeholder_unit("blue", position))

    env.battle_env = type(
        "DummyBattle",
        (),
        {
            "combined": battle_units,
            "escaped_units": [escaped],
            "winner": "blue",
        },
    )()

    env._save_blue_state()

    restored = _unit_at(env.blue_team_state, position)
    assert restored["name"] == original["name"]
    assert restored["running_away"] == 0
    assert restored["health"] == pytest.approx(23.0)
    assert restored["hp"] == pytest.approx(23.0)
    assert restored["exp_current"] == pytest.approx(17.0)


def test_save_blue_state_prefers_zero_health_over_stale_hp():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    position = 7
    battle_units = deepcopy(env.blue_team_state)
    stale_hp_unit = _unit_at(battle_units, position)
    stale_hp_unit["hp"] = float(stale_hp_unit.get("health", 0) or 0)
    stale_hp_unit["health"] = 0.0

    env.battle_env = _EndedBattle(winner="red", combined=battle_units)

    env._save_blue_state()

    saved = _unit_at(env.blue_team_state, position)
    assert saved["health"] == pytest.approx(0.0)
    assert saved["hp"] == pytest.approx(0.0)


def test_save_enemy_state_prefers_zero_health_over_stale_hp():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    enemy_id = 7778
    original = deepcopy(_unit_at(UNITS_RED, 1))
    original["health"] = 80.0
    original["hp"] = 80.0
    env.enemy_team_states[enemy_id] = env._build_battle_team_with_placeholders(
        "red",
        [original],
    )

    battle_unit = deepcopy(original)
    battle_unit["health"] = 0.0
    battle_unit["hp"] = 80.0
    env.battle_env = _EndedBattle(
        winner="blue",
        combined=[battle_unit] + deepcopy(env.blue_team_state),
    )

    env._save_enemy_state_from_battle(enemy_id)

    saved = _unit_at(env.enemy_team_states[enemy_id], 1)
    assert saved["health"] == pytest.approx(0.0)
    assert saved["hp"] == pytest.approx(0.0)


def test_save_blue_state_strips_non_escaped_battle_modifiers():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    position = 7
    battle_units = deepcopy(env.blue_team_state)
    unit = _unit_at(battle_units, position)
    base_damage = int(unit.get("damage", 0) or 0)
    base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
    unit["original_damage"] = base_damage
    unit["damage"] = base_damage * 2
    unit["powerup"] = 1
    unit["teamated"] = 1
    unit["initiative_base"] = max(1, base_initiative // 2)
    unit["initiative"] = unit["initiative_base"]
    unit["hermited"] = 1
    unit["uran_turns_left"] = 3
    unit["uran_damage_per_tick"] = 11
    unit["burn_source_lord_pos"] = 4
    unit["resilience_used_types"] = ["fire"]

    env.battle_env = _EndedBattle(winner="blue", combined=battle_units)

    env._save_blue_state()

    saved = _unit_at(env.blue_team_state, position)
    assert saved["damage"] == base_damage
    assert saved["initiative_base"] == base_initiative
    assert saved["initiative"] == base_initiative
    assert saved["powerup"] == 0
    assert "teamated" not in saved
    assert "hermited" not in saved
    assert saved["uran_turns_left"] == 0
    assert saved["uran_damage_per_tick"] == 0
    assert saved["burn_source_lord_pos"] is None
    assert saved["resilience_used_types"] == []


def test_campaign_defeat_with_escaped_units_retreats_and_saves_states():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    enemy_id = 7777
    red = deepcopy(_unit_at(UNITS_RED, 1))
    red["health"] = 80.0
    red["hp"] = 80.0
    red["max_health"] = 100.0
    red["maxhp"] = 100.0
    env.enemy_team_states[enemy_id] = env._build_battle_team_with_placeholders(
        "red",
        [red],
    )

    escaped = deepcopy(_unit_at(env.blue_team_state, 7))
    original_damage = int(escaped.get("damage", 0) or 0)
    escaped["health"] = 31.0
    escaped["hp"] = 31.0
    escaped["running_away"] = 0
    escaped["paralyzed"] = 1
    escaped["powerup"] = 1
    escaped["teamated"] = 1
    escaped["original_damage"] = original_damage
    escaped["damage"] = original_damage + 25

    battle_blue = deepcopy(env.blue_team_state)
    slot7 = _unit_at(battle_blue, 7)
    slot7.clear()
    slot7.update(placeholder_unit("blue", 7))
    dead_slot8 = _unit_at(battle_blue, 8)
    dead_slot8["health"] = 0.0
    dead_slot8["hp"] = 0.0

    red_after = deepcopy(red)
    red_after["health"] = 12.0
    red_after["hp"] = 12.0

    env.current_enemy_id = enemy_id
    env.mode = env.MODE_BATTLE
    env.battle_origin_pos = (3, 4)
    env.grid_env.agent_pos = (3, 5)
    env.battle_env = _EndedBattle(
        winner="red",
        combined=[red_after] + battle_blue,
        escaped_units=[escaped],
    )

    _obs, _reward, terminated, truncated, info = env._step_battle(0)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_GRID
    assert env.battle_env is None
    assert tuple(env.grid_env.agent_pos) == (3, 4)
    assert info["battle_result"] == "defeat"
    assert info["battle_retreat"] is True
    assert "campaign_result" not in info

    saved_enemy = _unit_at(env.enemy_team_states[enemy_id], 1)
    assert saved_enemy["health"] == pytest.approx(12.0)
    assert saved_enemy["hp"] == pytest.approx(12.0)

    restored_escape = _unit_at(env.blue_team_state, 7)
    assert restored_escape["name"] == escaped["name"]
    assert restored_escape["health"] == pytest.approx(31.0)
    assert restored_escape["hp"] == pytest.approx(31.0)
    assert restored_escape["running_away"] == 0
    assert restored_escape["paralyzed"] == 0
    assert restored_escape["powerup"] == 0
    assert "teamated" not in restored_escape
    assert restored_escape["damage"] == original_damage

    dead = _unit_at(env.blue_team_state, 8)
    assert dead["health"] == pytest.approx(0.0)
    assert dead["hp"] == pytest.approx(0.0)


def test_campaign_defeat_with_escaped_units_ends_when_hp_persistence_disabled():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=False, Realcapital=2)
    env.reset(seed=123)

    escaped = deepcopy(_unit_at(UNITS_BLUE, 7))
    escaped["health"] = 31.0
    escaped["hp"] = 31.0

    battle_blue = deepcopy(UNITS_BLUE)
    slot7 = _unit_at(battle_blue, 7)
    slot7.clear()
    slot7.update(placeholder_unit("blue", 7))
    red = deepcopy(_unit_at(UNITS_RED, 1))
    red["health"] = 10.0
    red["hp"] = 10.0

    env.current_enemy_id = 1
    env.mode = env.MODE_BATTLE
    env.battle_origin_pos = (3, 4)
    env.battle_env = _EndedBattle(
        winner="red",
        combined=[red] + battle_blue,
        escaped_units=[escaped],
    )

    _obs, _reward, terminated, truncated, info = env._step_battle(0)

    assert terminated is True
    assert truncated is False
    assert info["battle_result"] == "defeat"
    assert info["campaign_result"] == "defeat"
    assert info["terminal_defeat_enemy_id"] == 1
    assert info["terminal_defeat_enemy_key"] == "enemy_1"
    assert info["terminal_defeat_enemy_description"] == env._enemy_descriptions[1]
    assert info["terminal_defeat_enemy_units"]
    assert info["terminal_defeat_enemy_survivors"] == [red["name"]]
    assert not info.get("battle_retreat", False)
    assert "battle_defeat_with_escaped_units" not in info


def test_campaign_defeat_without_escaped_units_still_ends_episode():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    battle_blue = deepcopy(env.blue_team_state)
    for unit in battle_blue:
        unit["health"] = 0.0
        unit["hp"] = 0.0
    red = deepcopy(_unit_at(UNITS_RED, 1))
    red["health"] = 10.0
    red["hp"] = 10.0

    env.current_enemy_id = 1
    env.mode = env.MODE_BATTLE
    env.battle_origin_pos = (3, 4)
    env.battle_env = _EndedBattle(
        winner="red",
        combined=[red] + battle_blue,
        escaped_units=[],
    )

    _obs, _reward, terminated, truncated, info = env._step_battle(0)

    assert terminated is True
    assert truncated is False
    assert info["battle_result"] == "defeat"
    assert info["campaign_result"] == "defeat"
    assert info["terminal_defeat_enemy_id"] == 1
    assert info["terminal_defeat_enemy_key"] == "enemy_1"
    assert info["terminal_defeat_enemy_description"] == env._enemy_descriptions[1]
    assert info["terminal_defeat_enemy_units"]
    assert info["terminal_defeat_enemy_survivors"] == [red["name"]]
    assert not info.get("battle_retreat", False)
