"""Global bot lifecycle and no-neutral-target pursuit regressions.

These are deterministic unit/integration tests, not model training or evaluation.
The scripted bot is explicitly enabled, with local5 and the ordinary hero roster.
"""
from copy import deepcopy

import pytest

from campaign_env import CampaignEnv
from maps import available_maps, get_map


BOT_MAPS = tuple(name for name in available_maps() if get_map(name).scripted_capital_bot_supported)
NEUTRAL_BOT_MAPS = tuple(name for name in BOT_MAPS if name != "city_defence_train")


def make_env(map_name="default"):
    env = CampaignEnv(map_name=map_name, Realcapital=2,
                      observation_version="local5", scripted_capital_bot_enabled=True,
                      use_boss_starting_roster=False, log_enabled=False)
    env.reset(seed=42)
    return env


@pytest.fixture
def env():
    e = make_env()
    yield e
    e.close()


def living_names(team):
    return sorted(u["name"] for u in team if u.get("health", 0) > 0)


def clear_targets(e):
    for enemy_id in e.grid_env.enemies_alive:
        if enemy_id != e.scripted_capital_bot_enemy_id:
            e.grid_env.enemies_alive[enemy_id] = False
    e.legions_active_settlement_territory_capture_turn_by_name.clear()
    e.grid_env.obstacle_positions = set()
    e.scripted_capital_bot_position = (10, 10)
    e.grid_env.agent_pos = (25, 10)
    e._sync_scripted_capital_bot_grid_state()


@pytest.mark.parametrize("map_name", BOT_MAPS)
def test_defeat_between_turns_respawns_only_after_two_completed_turns(map_name):
    e = make_env(map_name)
    try:
        expected_roster = deepcopy(e._create_scripted_capital_bot_team())
        e.scripted_capital_bot_position = (10, 10)
        e.scripted_capital_bot_rest_turns_left = 7
        e._scripted_bot_pending_hero_encounter = True
        for unit in e.scripted_capital_bot_team_state:
            unit["health"] = unit["hp"] = 0
        e._refresh_scripted_bot_enemy_team(force=True)
        e._mark_scripted_capital_bot_defeated()
        assert e.turns == 0
        assert e.scripted_capital_bot_respawn_turns_left == 2
        assert e.scripted_capital_bot_rest_turns_left == 0
        assert e.grid_env.scripted_bot_position is None
        assert not e._scripted_bot_pending_hero_encounter
        assert (10, 10) not in e.grid_env.dynamic_blocked_positions
        e._advance_turns(1)
        assert e.turns == 1
        assert e.scripted_capital_bot_state == "defeated"
        assert e.scripted_capital_bot_respawn_turns_left == 1
        assert e.grid_env.scripted_bot_position is None
        assert not living_names(e.scripted_capital_bot_team_state)
        # Multiple outcome hooks may report the same defeat; do not delay again.
        e._mark_scripted_capital_bot_defeated()
        assert e.scripted_capital_bot_respawn_turns_left == 1
        e._advance_turns(1)
        assert e.turns == 2
        assert e.scripted_capital_bot_state == "hunting"
        assert e.scripted_capital_bot_respawn_turns_left == 0
        assert e.scripted_capital_bot_position == e.scripted_capital_bot_home
        assert e.grid_env.scripted_bot_position == e.scripted_capital_bot_home
        assert e.scripted_capital_bot_team_state == expected_roster
        assert "respawned" in e.scripted_capital_bot_turn_infos[-1]["events"]
        assert not e.scripted_capital_bot_turn_infos[-1].get("path")
    finally:
        e.close()


@pytest.mark.parametrize("map_name", NEUTRAL_BOT_MAPS)
def test_neutral_defeat_on_turn_boundary_does_not_consume_respawn_turn(map_name, monkeypatch):
    e = make_env(map_name)
    try:
        targets = e._scripted_bot_alive_enemy_tiles()
        tile, enemy_id = next((tile, enemy_id) for tile, enemy_id in targets.items() if enemy_id >= 0)
        e.scripted_capital_bot_position = (tile[0] - 1, tile[1])
        target = dict(position=tile, enemy_id=enemy_id)
        monkeypatch.setattr(e, "_pick_scripted_bot_nearest_enemy", lambda *_: target)
        monkeypatch.setattr(e, "_scripted_bot_path_to_engagement_tile", lambda *_: [])
        monkeypatch.setattr(e, "SCRIPTED_CAPITAL_BOT_BATTLE_STEP_LIMIT", 0)
        e._advance_turns(1)
        assert e.turns == 1
        assert e.scripted_capital_bot_state == "defeated"
        assert e.scripted_capital_bot_respawn_turns_left == 2
        assert e.scripted_capital_bot_turn_infos[-1]["battle"]["winner"] == "timeout"
        e._advance_turns(1)
        assert e.turns == 2 and e.scripted_capital_bot_state == "defeated"
        assert e.scripted_capital_bot_respawn_turns_left == 1
        e._advance_turns(1)
        assert e.turns == 3 and e.scripted_capital_bot_state == "hunting"
        assert e.scripted_capital_bot_position == e.scripted_capital_bot_home
    finally:
        e.close()


@pytest.mark.parametrize("offset", [(1, 0), (0, 1), (1, 1), (-1, -1)])
def test_no_targets_adjacent_bot_queues_real_hero_battle(env, monkeypatch, offset):
    clear_targets(env)
    env.grid_env.agent_pos = (10 + offset[0], 10 + offset[1])
    hero_before = deepcopy(env.blue_team_state)
    monkeypatch.setattr(env, "_run_scripted_capital_bot_battle",
                        lambda *_: pytest.fail("The player party must never be autoresolved"))
    info = env._advance_scripted_capital_bot_one_turn()
    assert info["target"] == {"kind": "agent", "position": env.grid_env.agent_pos}
    assert "agent_battle_pending" in info["events"]
    assert env._scripted_bot_pending_hero_encounter
    assert env.scripted_capital_bot_position == (10, 10)
    assert env.scripted_capital_bot_position != env.grid_env.agent_pos
    assert env.blue_team_state == hero_before
    assert env.battle_env is None


def test_no_targets_bot_moves_toward_agent_with_ordinary_budget(env, monkeypatch):
    clear_targets(env)
    monkeypatch.setattr(env, "SCRIPTED_CAPITAL_BOT_MAX_STEPS_PER_TURN", 4)
    monkeypatch.setattr(env, "_campaign_tile_move_cost", lambda *_, **__: 2)
    info = env._advance_scripted_capital_bot_one_turn()
    assert "pursuing_agent" in info["events"]
    assert info["move_points_budget"] == info["move_points_spent"] == 4
    assert len(info["path"]) == 2
    assert info["move_costs"] == [2, 2]
    assert max(abs(env.scripted_capital_bot_position[i] - env.grid_env.agent_pos[i])
               for i in range(2)) < 15
    assert not env._scripted_bot_pending_hero_encounter


def test_pursuit_stops_adjacent_after_longer_move(env, monkeypatch):
    clear_targets(env)
    env.grid_env.agent_pos = (14, 10)
    monkeypatch.setattr(env, "_run_scripted_capital_bot_battle",
                        lambda *_: pytest.fail("The player party must never be autoresolved"))
    info = env._advance_scripted_capital_bot_one_turn()
    assert info["path"]
    assert env.grid_env.agent_pos not in info["path"]
    assert env._scripted_bot_can_engage_enemy_from_position(
        env.scripted_capital_bot_position, env.grid_env.agent_pos)
    assert env._scripted_bot_pending_hero_encounter


def test_unreachable_agent_is_a_wait_without_fake_contact_or_wall_crossing(env):
    clear_targets(env)
    ax, ay = env.grid_env.agent_pos
    env.grid_env.obstacle_positions = {
        (ax + dx, ay + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy}
    for _ in range(2):
        info = env._advance_scripted_capital_bot_one_turn()
        assert "agent_unreachable" in info["events"]
        assert info["path"] == []
        assert info["move_points_spent"] == 0
        assert env.scripted_capital_bot_position == (10, 10)
        assert not env._scripted_bot_pending_hero_encounter


def test_eligible_neutral_target_retains_priority_even_when_unreachable(env, monkeypatch):
    clear_targets(env)
    enemy_id = next(enemy_id for enemy_id in env.grid_env.enemy_positions
                    if enemy_id >= 0 and enemy_id not in env._scripted_bot_forbidden_enemy_ids())
    tile = (30, 30)
    env.grid_env.enemy_positions[enemy_id] = tile
    env.grid_env.enemies_alive[enemy_id] = True
    env.grid_env.agent_pos = (11, 10)
    env.grid_env.obstacle_positions = {
        (tile[0] + dx, tile[1] + dy)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy}
    monkeypatch.setattr(env, "_pursue_agent_with_scripted_bot",
                        lambda *_: pytest.fail("Living neutral targets must keep their priority"))
    info = env._advance_scripted_capital_bot_one_turn()
    assert info["target_enemy_id"] == enemy_id
    assert "hunting" in info["events"]
    assert not env._scripted_bot_pending_hero_encounter


def test_returning_and_resting_bot_does_not_switch_to_pursuit(env, monkeypatch):
    clear_targets(env)
    env.scripted_capital_bot_position = env.scripted_capital_bot_home
    env.scripted_capital_bot_state = "returning"
    monkeypatch.setattr(env, "_pursue_agent_with_scripted_bot",
                        lambda *_: pytest.fail("Return/rest must complete before pursuing"))
    info = env._advance_scripted_capital_bot_one_turn()
    assert env.scripted_capital_bot_state == "resting"
    assert "arrived_home" in info["events"]
    unit = next(u for u in env.scripted_capital_bot_team_state if u.get("health", 0) > 0)
    unit["health"] = unit["hp"] = 0
    env._refresh_scripted_bot_enemy_team(force=True)
    info = env._advance_scripted_capital_bot_one_turn()
    assert env.scripted_capital_bot_state == "hunting"
    assert "fully_restored" in info["events"]
    assert info["revived_units"] == 1
    restored = next(u for u in env.scripted_capital_bot_team_state if u["position"] == unit["position"])
    assert restored["health"] == restored["max_health"]
    assert not env._scripted_bot_pending_hero_encounter


@pytest.mark.parametrize("map_name", BOT_MAPS)
def test_reset_restores_bot_without_pending_encounter_or_boss_roster(map_name):
    e = make_env(map_name)
    try:
        hero_before = deepcopy(e.blue_team_state)
        bot_before = deepcopy(e.scripted_capital_bot_team_state)
        e._mark_scripted_capital_bot_defeated()
        e._scripted_bot_pending_hero_encounter = True
        e.scripted_capital_bot_enemies_defeated = 9
        obs, _ = e.reset(seed=42)
        assert not e.use_boss_starting_roster
        assert e.blue_team_state == hero_before
        assert e.scripted_capital_bot_team_state == bot_before
        assert e.scripted_capital_bot_state == "hunting"
        assert e.scripted_capital_bot_respawn_turns_left == 0
        assert e.scripted_capital_bot_enemies_defeated == 0
        assert not e._scripted_bot_pending_hero_encounter
        assert e.scripted_capital_bot_position == e.scripted_capital_bot_home
        assert e.observation_space.contains(obs)
        if e._map.starting_roster is None:
            assert living_names(e.blue_team_state) == ["Герцог", "Одержимый", "Одержимый", "Сектант"]
    finally:
        e.close()


def test_dead_bot_unit_is_not_revived_by_stale_hp_alias(env):
    unit = deepcopy(next(u for u in env.scripted_capital_bot_team_state if u.get("health", 0) > 0))
    unit["health"] = 0
    unit["hp"] = unit["max_health"]
    saved = env._normalize_scripted_bot_saved_unit(unit)
    assert saved["health"] == saved["hp"] == 0
    assert saved["initiative"] == 0


@pytest.mark.parametrize("map_name", ["default", "small", "wotans_retribution", "mirrow_match"])
def test_end_turn_pursuit_opens_agent_controlled_battle(map_name, monkeypatch):
    e = make_env(map_name)
    try:
        clear_targets(e)
        e.grid_env.agent_pos = (11, 10)
        e._sync_scripted_capital_bot_grid_state()
        expected_bot = living_names(e.scripted_capital_bot_team_state)
        monkeypatch.setattr(e, "_run_scripted_capital_bot_battle",
                            lambda *_: pytest.fail("Hero fight must be agent-controlled"))
        obs, _, terminated, truncated, info = e.step(8)
        assert not (terminated or truncated)
        assert e.turns == 1 and e.mode == e.MODE_BATTLE
        assert e.battle_env is not None
        assert e.current_enemy_id == e.scripted_capital_bot_enemy_id
        assert e.current_battle_context["kind"] == "hero"
        assert e.current_battle_context["attacker_team"] == "red"
        assert e.current_battle_context["defender_team"] == "blue"
        assert info["battle_triggered"]
        assert info["battle_triggered_by"] == "scripted_bot"
        assert not e._scripted_bot_pending_hero_encounter
        assert e.observation_space.contains(obs)
        assert living_names([u for u in e.battle_env.combined if u["team"] == "red"]) == expected_bot
        # The next public environment step goes through BattleEnv for BLUE.
        obs, _, _, _, battle_info = e.step(e.battle_env.scripted_action_for_current_blue())
        assert battle_info["battle_step"]
        assert e.observation_space.contains(obs)
    finally:
        e.close()


def test_protected_capital_guard_does_not_prevent_agent_pursuit(env):
    clear_targets(env)
    protected_id = next(enemy_id for enemy_id in env._scripted_bot_forbidden_enemy_ids()
                        if enemy_id in env.grid_env.enemy_positions)
    env.grid_env.enemies_alive[protected_id] = True
    env.grid_env.agent_pos = (11, 10)
    info = env._advance_scripted_capital_bot_one_turn()
    assert info["target"]["kind"] == "agent"
    assert env._scripted_bot_pending_hero_encounter


def test_actual_neutral_combat_loss_enters_shared_two_turn_respawn(env):
    enemy_id = next(enemy_id for enemy_id in env._scripted_bot_alive_enemy_tiles().values()
                    if enemy_id >= 0)
    for unit in env.scripted_capital_bot_team_state:
        if unit.get("health", 0) > 0:
            unit.update(health=1, hp=1, initiative=1, initiative_base=1,
                        damage=1, original_damage=1)
    for unit in env._get_enemy_team_state(enemy_id):
        if unit.get("health", 0) > 0:
            unit.update(health=5000, hp=5000, max_health=5000, maxhp=5000,
                        initiative=100, initiative_base=100, damage=999, original_damage=999)
    result = env._run_scripted_capital_bot_battle(enemy_id)
    assert result["winner"] == "red"
    assert result["steps"] < env.SCRIPTED_CAPITAL_BOT_BATTLE_STEP_LIMIT
    assert env.scripted_capital_bot_state == "defeated"
    assert env.scripted_capital_bot_respawn_turns_left == 2
    assert env.grid_env.scripted_bot_position is None
    assert env.grid_env.enemies_alive[enemy_id]
    env._advance_turns(1)
    assert env.scripted_capital_bot_state == "defeated"
    env._advance_turns(1)
    assert env.scripted_capital_bot_state == "hunting"
    assert env.scripted_capital_bot_position == env.scripted_capital_bot_home


@pytest.mark.parametrize("distance", [1, 6, 20])
def test_return_home_blocked_by_hero_stops_adjacent_and_attacks(env, monkeypatch, distance):
    clear_targets(env)
    env.scripted_capital_bot_home = (30, 10)
    env.grid_env.agent_pos = env.scripted_capital_bot_home
    env.scripted_capital_bot_position = (30 - distance, 10)
    env.scripted_capital_bot_state = "returning"
    monkeypatch.setattr(env, "_campaign_tile_move_cost", lambda *_, **__: 2)
    monkeypatch.setattr(env, "_run_scripted_capital_bot_battle",
                        lambda *_: pytest.fail("A returning bot must not autoresolve the hero"))
    unit = next(u for u in env.scripted_capital_bot_team_state if u.get("health", 0) > 0)
    unit["health"] = unit["hp"] = 1
    info = env._advance_scripted_capital_bot_one_turn()
    assert env.scripted_capital_bot_position != env.grid_env.agent_pos
    assert env.grid_env.agent_pos not in info["path"]
    assert env.scripted_capital_bot_state == "returning"
    assert env.scripted_capital_bot_rest_turns_left == 0
    assert "arrived_home" not in info["events"]
    assert "home_blocked_by_agent" in info["events"]
    assert unit["health"] == 1
    if distance > 11:
        assert not env._scripted_bot_pending_hero_encounter
        info = env._advance_scripted_capital_bot_one_turn()
    assert env._scripted_bot_pending_hero_encounter
    result = (env._build_obs(), 0.0, False, False, {})
    obs, _, terminated, truncated, info = env._finish_pending_scripted_bot_encounter(result)
    assert env.mode == env.MODE_BATTLE
    assert env.current_battle_context["attacker_team"] == "red"
    assert info["battle_triggered_by"] == "scripted_bot"
    assert not (terminated or truncated)
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("map_name", BOT_MAPS)
def test_respawn_on_hero_occupied_home_immediately_opens_real_battle(map_name, monkeypatch):
    e = make_env(map_name)
    try:
        # Existing neutral objectives and city waves do not prevent contact at home.
        e._mark_scripted_capital_bot_defeated()
        e.grid_env.agent_pos = tuple(e.scripted_capital_bot_home)
        monkeypatch.setattr(e, "_run_scripted_capital_bot_battle",
                            lambda *_: pytest.fail("Respawn contact must never autoresolve hero"))
        obs, _, terminated, truncated, _ = e.step(8)
        assert not (terminated or truncated)
        assert e.turns == 1 and e.mode == e.MODE_GRID
        assert e.scripted_capital_bot_state == "defeated"
        assert e.grid_env.scripted_bot_position is None
        obs, _, terminated, truncated, info = e.step(8)
        assert not (terminated or truncated)
        assert e.turns == 2 and e.mode == e.MODE_BATTLE
        assert e.scripted_capital_bot_respawn_turns_left == 0
        assert e.scripted_capital_bot_position == e.scripted_capital_bot_home
        assert e.current_enemy_id == e.scripted_capital_bot_enemy_id
        assert e.current_battle_context["attacker_team"] == "red"
        assert info["battle_triggered_by"] == "scripted_bot"
        assert "respawned" in e.scripted_capital_bot_turn_infos[-1]["events"]
        assert "home_blocked_by_agent" in e.scripted_capital_bot_turn_infos[-1]["events"]
        assert not e.scripted_capital_bot_turn_infos[-1].get("path")
        assert e.observation_space.contains(obs)
    finally:
        e.close()
