import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import BattleEnv, UNITS_BLUE
from campaign_env import CampaignEnv


def _make_enemy(exp_kill: int) -> dict:
    return {
        "name": "Test enemy",
        "team": "red",
        "position": 1,
        "health": 1,
        "max_health": 1,
        "running_away": 0,
        "exp_kill": exp_kill,
    }


def test_hero_levelup_raises_level_resets_exp_and_increases_required_exp():
    env = BattleEnv(log_enabled=False)
    duke = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    duke["exp_current"] = 149
    duke["exp_required"] = 150
    duke["Level"] = 0
    duke["next_level_exp"] = 500

    env.combined = [duke, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert duke["Level"] == 1
    assert duke["exp_current"] == 0
    assert duke["exp_required"] == 650
    assert env.last_levelups == [duke["name"]]


def test_regular_unit_without_next_level_exp_keeps_transform_upgrade_flow():
    env = BattleEnv(log_enabled=False)
    fighter = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 7))
    fighter["exp_current"] = 94
    fighter["exp_required"] = 95
    fighter["next_level_exp"] = 0
    initial_level = fighter.get("Level", 0)

    env.combined = [fighter, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert fighter["Level"] == initial_level
    assert fighter["exp_current"] == 94
    assert fighter["exp_required"] == 95
    assert env.last_levelups == [fighter["name"]]


def test_campaign_save_preserves_hero_level_and_exp_threshold_after_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    battle_units = deepcopy(UNITS_BLUE)
    duke = next(u for u in battle_units if int(u.get("position", -1)) == 8)
    duke["Level"] = 1
    duke["exp_required"] = 650
    duke["exp_current"] = 0
    duke["next_level_exp"] = 500

    env.battle_env = type("DummyBattle", (), {"combined": battle_units})()
    env._save_blue_state()

    saved_duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    assert saved_duke["Level"] == 1
    assert saved_duke["exp_required"] == 650
    assert saved_duke["exp_current"] == 0
    assert saved_duke["next_level_exp"] == 500
