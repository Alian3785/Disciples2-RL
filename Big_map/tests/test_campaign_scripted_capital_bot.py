import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import campaign_env_scripted_bot
from battle_env import BattleEnv
from campaign_env import CampaignEnv


def _make_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    return env


def _enable_scripted_bot(env: CampaignEnv) -> None:
    env.scripted_capital_bot_enabled = True
    env.scripted_capital_bot_state = "hunting"
    env.scripted_capital_bot_position = env.empire_territory_source_tile
    env._sync_scripted_capital_bot_grid_state()


def test_minimal_grid_step_info_reports_scripted_bot_defeat_count():
    env = CampaignEnv(
        log_enabled=False,
        detailed_step_info=False,
        persist_blue_hp=True,
        Realcapital=2,
    )
    env.reset(seed=123)
    env.scripted_capital_bot_enemies_defeated = 2

    _, _, _, _, info = env._finalize_grid_step_result(
        grid_obs=env._get_grid_obs(),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"agent_pos": env.grid_env.agent_pos},
    )

    assert info["scripted_capital_bot_enemies_defeated"] == 2


def test_scripted_capital_bot_can_be_disabled_at_construction():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        Realcapital=2,
        scripted_capital_bot_enabled=False,
    )
    env.reset(seed=123)

    assert env.scripted_capital_bot_enabled is False
    assert env.scripted_capital_bot_state == "disabled"
    assert env.grid_env.scripted_bot_position is None
    assert env.grid_env.dynamic_blocked_positions == set()

    info = env._advance_scripted_capital_bot_one_turn()
    assert info["enabled"] is False
    assert env.grid_env.scripted_bot_position is None
    assert env.grid_env.dynamic_blocked_positions == set()


def test_scripted_capital_bot_enemy_search_reuses_cache_for_same_state():
    env = _make_env()

    first = env._scripted_bot_explore_for_enemies()
    second = env._scripted_bot_explore_for_enemies()

    assert second is first

    env.scripted_capital_bot_position = (
        env.scripted_capital_bot_position[0] + 1,
        env.scripted_capital_bot_position[1],
    )
    third = env._scripted_bot_explore_for_enemies()

    assert third is not first


def test_scripted_capital_bot_starts_at_empire_capital_without_targeting_mizrael():
    env = _make_env()

    assert env.scripted_capital_bot_position == env.empire_territory_source_tile
    assert env.scripted_capital_bot_enabled is True
    assert env.scripted_capital_bot_state == "hunting"
    assert env.grid_env.enemies_alive[env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID] is True
    assert env.grid_env.scripted_bot_position == env.empire_territory_source_tile
    assert env.empire_territory_source_tile not in env._scripted_bot_alive_enemy_tiles()

    names = {unit["name"] for unit in env.scripted_capital_bot_team_state}
    assert {"Рыцарь", "Рыцарь на пегасе", "Стрелок"}.issubset(names)
    assert "Мизраэль" not in names


def test_scripted_capital_bot_targets_other_enemy_instead_of_mizrael(monkeypatch):
    env = _make_env()
    env.grid_env.enemy_positions = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: env.empire_territory_source_tile,
        1: (25, 10),
    }
    env.grid_env.enemies_alive = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: True,
        1: True,
    }
    env.grid_env.obstacle_positions = set()
    env.grid_env.dynamic_blocked_positions = set()
    battles = []

    def fake_battle(target_enemy_id: int):
        battles.append(int(target_enemy_id))
        env.scripted_capital_bot_state = "returning"
        return {"enemy_id": int(target_enemy_id), "winner": "blue"}

    monkeypatch.setattr(env, "_run_scripted_capital_bot_battle", fake_battle)

    info = env._advance_scripted_capital_bot_one_turn()

    assert battles == [1]
    assert info["target"]["enemy_id"] == 1
    assert info["target"]["position"] == (25, 10)
    assert "battle" in info["events"]


def test_scripted_capital_bot_direct_mizrael_battle_call_is_skipped():
    env = _make_env()

    result = env._run_scripted_capital_bot_battle(env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID)

    assert result["winner"] == "skipped"
    assert result["skipped"] is True
    assert result["skip_reason"] == "empire_capital_enemy_forbidden"
    assert env.grid_env.enemies_alive[env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID] is True


def test_scripted_capital_bot_moves_up_to_twenty_tiles_without_crossing_obstacles():
    env = _make_env()
    _enable_scripted_bot(env)
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

    assert 0 < len(path) <= env._scripted_bot_max_grid_moves_per_turn()
    assert info["move_cost"] >= 1
    assert 0 < info["move_points_spent"] <= info["move_points_budget"]
    assert all(tuple(tile) not in env.grid_env.obstacle_positions for tile in path)
    assert env.scripted_capital_bot_position == tuple(path[-1])


def test_scripted_capital_bot_uses_two_move_points_per_grid_step():
    env = _make_env()
    _enable_scripted_bot(env)
    env.grid_env.enemy_positions = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: env.empire_territory_source_tile,
        1: (1, 1),
    }
    env.grid_env.enemies_alive = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: False,
        1: True,
    }
    env.grid_env.obstacle_positions = set()
    env.scripted_capital_bot_position = env.empire_territory_source_tile
    env._sync_scripted_capital_bot_grid_state()

    info = env._advance_scripted_capital_bot_one_turn()
    path = info["path"]

    assert 0 < len(path) <= env._scripted_bot_max_grid_moves_per_turn()
    assert info["move_points_budget"] == env.SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN
    assert info["move_points_spent"] == env.SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN


def test_scripted_capital_bot_battle_starts_from_adjacent_tile(monkeypatch):
    env = _make_env()
    _enable_scripted_bot(env)
    enemy_id = 1
    env.scripted_capital_bot_position = (5, 5)
    env.grid_env.enemy_positions = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: env.empire_territory_source_tile,
        enemy_id: (7, 5),
    }
    env.grid_env.enemies_alive = {
        env.EMPIRE_TERRITORY_SOURCE_ENEMY_ID: False,
        enemy_id: True,
    }
    env.grid_env.obstacle_positions = set()
    env.grid_env.dynamic_blocked_positions = set()

    battles = []

    def fake_battle(target_enemy_id: int):
        battles.append(int(target_enemy_id))
        env.grid_env.mark_enemy_defeated(int(target_enemy_id))
        env.scripted_capital_bot_state = "returning"
        return {"enemy_id": int(target_enemy_id), "winner": "blue"}

    monkeypatch.setattr(env, "_run_scripted_capital_bot_battle", fake_battle)

    info = env._advance_scripted_capital_bot_one_turn()

    assert battles == [enemy_id]
    assert "battle" in info["events"]
    assert max(
        abs(env.scripted_capital_bot_position[0] - env.grid_env.enemy_positions[enemy_id][0]),
        abs(env.scripted_capital_bot_position[1] - env.grid_env.enemy_positions[enemy_id][1]),
    ) == 1
    assert env.scripted_capital_bot_position != env.grid_env.enemy_positions[enemy_id]
    assert len(info["path"]) == 1
    assert info["move_points_spent"] == 2


def test_scripted_capital_bot_victory_marks_enemy_defeated_and_returns(monkeypatch):
    env = _make_env()
    _enable_scripted_bot(env)
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
    assert env.scripted_capital_bot_enemies_defeated == 1
    assert env._scripted_capital_bot_info()["scripted_capital_bot_enemies_defeated"] == 1
    saved_unit = next(unit for unit in env.scripted_capital_bot_team_state if unit["name"] == first_unit["name"])
    assert saved_unit["hp"] == 10


def test_scripted_capital_bot_defeat_respawns_after_one_turn(monkeypatch):
    env = _make_env()
    _enable_scripted_bot(env)

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
    _enable_scripted_bot(env)
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
    _enable_scripted_bot(env)
    battle = BattleEnv(log_enabled=False)
    red_team = env._build_battle_team_with_placeholders("red", env._get_enemy_team_state(1))
    blue_team = env.scripted_capital_bot_team_state
    battle._init_with_custom_teams(red_team, blue_team)

    action = battle.scripted_action_for_current_blue()
    mask = battle.compute_action_mask()

    assert bool(mask[action])


def test_grid_env_tracks_scripted_bot_with_letter():
    env = _make_env()
    _enable_scripted_bot(env)

    assert env.grid_env.scripted_bot_position == env.empire_territory_source_tile
    assert env.grid_env.scripted_bot_symbol == "B"
