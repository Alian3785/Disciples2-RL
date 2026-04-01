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


def test_hero_levelup_raises_level_resets_exp_and_buffs_hero_stats():
    env = BattleEnv(log_enabled=False)
    duke = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    duke["exp_current"] = 149
    duke["exp_required"] = 150
    duke["Level"] = 0
    duke["next_level_exp"] = 500
    duke["damage"] = 50
    duke["original_damage"] = 50
    duke["health"] = 150
    duke["max_health"] = 150
    duke["exp_kill"] = 60
    duke["accuracy"] = 80
    duke["needaunit"] = 0

    env.combined = [duke, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert duke["Level"] == 1
    assert duke["exp_current"] == 0
    assert duke["exp_required"] == 650
    assert duke["damage"] == 55
    assert duke["original_damage"] == 55
    assert duke["health"] == 165
    assert duke["max_health"] == 165
    assert duke["exp_kill"] == 66
    assert duke["accuracy"] == 81
    assert duke["needaunit"] == 0
    assert env.last_levelups == [duke["name"]]


def test_hero_levelup_to_third_level_sets_needaunit_flag():
    env = BattleEnv(log_enabled=False)
    duke = deepcopy(next(u for u in UNITS_BLUE if int(u.get("position", -1)) == 8))
    duke["exp_current"] = 649
    duke["exp_required"] = 650
    duke["Level"] = 2
    duke["next_level_exp"] = 500
    duke["needaunit"] = 0

    env.combined = [duke, _make_enemy(exp_kill=1)]
    env._apply_battle_exp("red")

    assert duke["Level"] == 3
    assert duke["needaunit"] == 1


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
    assert env.moves_per_turn == 21
    assert env.moves == 21


def test_campaign_moves_per_turn_scales_with_travel_hero_level():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    assert duke["Level"] == 1
    assert duke["needaunit"] == 0
    assert env.moves_per_turn == 21
    env.moves = 5
    duke["Level"] = 2

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    assert env.moves_per_turn == 26
    assert env.moves == 10
    assert env.get_travel_hero_visual_info() == {
        "name": duke["name"],
        "level": 2,
        "abilities": ["Нахождение пути (+20% шагов)"],
    }

    env._advance_turns(1)
    assert env.moves == 26


def test_campaign_travel_hero_visual_info_shows_leadership_on_level_three():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    duke = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 8)
    duke["Level"] = 3
    duke["needaunit"] = 0

    env._sync_moves_per_turn_with_hero(units=env.blue_team_state, grant_delta=True)

    assert env.get_travel_hero_visual_info() == {
        "name": duke["name"],
        "level": 3,
        "abilities": [
            "Нахождение пути (+20% шагов)",
            "Лидерство",
        ],
    }
