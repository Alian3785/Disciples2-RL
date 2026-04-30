import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import campaign_env_scripted_bot
from battle_env import BattleEnv
from campaign_env import CampaignEnv


def _make_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    return env


def test_scripted_capital_bot_starts_at_empire_capital_and_disables_static_stack():
    env = _make_env()

    assert env.scripted_capital_bot_position == env.empire_territory_source_tile
    assert env.scripted_capital_bot_state == "hunting"
    assert env.grid_env.enemies_alive[env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID] is False
    assert env.grid_env.scripted_bot_position == env.empire_territory_source_tile

    names = {unit["name"] for unit in env.scripted_capital_bot_team_state}
    assert {"Рыцарь", "Рыцарь на пегасе", "Стрелок"}.issubset(names)


def test_scripted_capital_bot_moves_up_to_twenty_tiles_without_crossing_obstacles():
    env = _make_env()
    env.grid_env.enemy_positions = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: env.empire_territory_source_tile,
        1: (1, 1),
    }
    env.grid_env.enemies_alive = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: False,
        1: True,
    }
    env.grid_env.obstacle_positions = {(25, 10), (24, 9)}
    env.scripted_capital_bot_position = env.empire_territory_source_tile
    env._sync_scripted_capital_bot_grid_state()

    info = env._advance_scripted_capital_bot_one_turn()
    path = info["path"]

    assert 0 < len(path) <= env.SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN
    assert all(tuple(tile) not in env.grid_env.obstacle_positions for tile in path)
    assert env.scripted_capital_bot_position == tuple(path[-1])


def test_scripted_capital_bot_victory_marks_enemy_defeated_and_returns(monkeypatch):
    env = _make_env()
    enemy_id = 1
    env.grid_env.enemies_alive[enemy_id] = True
    first_unit = next(unit for unit in env.scripted_capital_bot_team_state if unit["name"] != "пусто")
    first_unit["health"] = first_unit["hp"] = 10

    def fake_init(self, red_team, blue_team):
        self.combined = [
            dict(unit, team="red") for unit in red_team
        ] + [
            dict(unit, team="blue") for unit in blue_team
        ]
        self.winner = "blue"
        self.last_battle_exp = 120.0
        self.last_levelups = []

    monkeypatch.setattr(BattleEnv, "_init_with_custom_teams", fake_init)

    result = env._run_scripted_capital_bot_battle(enemy_id)

    assert result["winner"] == "blue"
    assert env.grid_env.enemies_alive[enemy_id] is False
    assert env.scripted_capital_bot_state == "returning"
    saved_unit = next(unit for unit in env.scripted_capital_bot_team_state if unit["name"] == first_unit["name"])
    assert saved_unit["hp"] == 10


def test_scripted_capital_bot_defeat_respawns_after_one_turn(monkeypatch):
    env = _make_env()

    def fake_init(self, red_team, blue_team):
        self.combined = [
            dict(unit, team="red") for unit in red_team
        ] + [
            dict(unit, team="blue") for unit in blue_team
        ]
        self.winner = "red"
        self.last_battle_exp = 0.0
        self.last_levelups = []

    monkeypatch.setattr(BattleEnv, "_init_with_custom_teams", fake_init)

    result = env._run_scripted_capital_bot_battle(1)
    assert result["winner"] == "red"
    assert env.scripted_capital_bot_state == "defeated"

    info = env._advance_scripted_capital_bot_one_turn()

    assert "respawned" in info["events"]
    assert env.scripted_capital_bot_state == "hunting"
    assert env.scripted_capital_bot_position == env.empire_territory_source_tile


def test_scripted_capital_bot_rest_fully_heals_and_revives():
    env = _make_env()
    damaged = 0
    killed = 0
    for unit in env.scripted_capital_bot_team_state:
        if unit["name"] == "пусто":
            continue
        if damaged == 0:
            unit["hp"] = unit["health"] = 1
            damaged += 1
        elif killed == 0:
            unit["hp"] = unit["health"] = 0
            killed += 1
    env.scripted_capital_bot_state = "resting"
    env.scripted_capital_bot_rest_turns_left = 1

    info = env._advance_scripted_capital_bot_one_turn()

    assert "fully_restored" in info["events"]
    assert env.scripted_capital_bot_state == "hunting"
    for unit in env.scripted_capital_bot_team_state:
        if unit["name"] != "пусто":
            assert unit["hp"] == unit["maxhp"]
            assert unit["health"] == unit["max_health"]


def test_scripted_blue_action_uses_battle_mask(monkeypatch):
    env = _make_env()
    battle = BattleEnv(log_enabled=False)
    red_team = env._build_battle_team_with_placeholders("red", env._get_enemy_team_state(1))
    blue_team = env.scripted_capital_bot_team_state
    battle._init_with_custom_teams(red_team, blue_team)

    action = battle.scripted_action_for_current_blue()
    mask = battle.compute_action_mask()

    assert bool(mask[action])


def test_grid_render_marks_scripted_bot_with_letter():
    env = _make_env()

    rendered = env.grid_env.render(mode="ansi")

    assert "B " in rendered
