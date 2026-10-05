"""Fear uses runtime defended entrances, independently of armor or retreat masks.

Fixtures position armies near sites; public CampaignEnv.step drives battle entry
and combat. No training, scripted-bot enablement or synthetic battle winner.
"""
from copy import deepcopy

import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX
from campaign_env import CampaignEnv
from maps import available_maps


@pytest.fixture
def make_env():
    envs = []

    def make(map_name="default", **kwargs):
        env = CampaignEnv(
            map_name=map_name, observation_version="local5",
            scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
            log_enabled=False, **kwargs,
        )
        env.reset(seed=42)
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


def unit(env, name, team, position):
    return env._build_unit_from_data(env._find_unit_data_by_name(name), team, position)


def site_tiles(env):
    return {
        tuple(env.CASTLE_POS), tuple(env.empire_territory_source_tile),
        *env.legions_settlement_source_name_by_tile,
        *(env._static_enemy_positions[eid] for eid in env.RUIN_REWARD_BY_ENEMY_ID),
    }


def field_tile(env):
    return next((x, y) for x in range(env.grid_size) for y in range(env.grid_size)
                if (x, y) not in site_tiles(env)
                and (x, y) not in env.grid_env.obstacle_positions)


@pytest.mark.parametrize("map_name", available_maps())
@pytest.mark.parametrize("attacker", ["blue", "red"])
def test_exact_entrances_on_all_maps(make_env, map_name, attacker):
    env = make_env(map_name)
    eid = next(iter(env._static_enemy_positions), 1)
    for tile in site_tiles(env):
        env.battle_origin_pos = tile
        env.grid_env.enemy_positions[eid] = tile
        assert env._fear_paralysis_teams_for_battle(eid, attacker_team=attacker) == (
            "red" if attacker == "blue" else "blue",
        )
    tile = field_tile(env)
    env.battle_origin_pos = tile
    env.grid_env.enemy_positions[eid] = tile
    assert env._fear_paralysis_teams_for_battle(eid, attacker_team=attacker) == ()


@pytest.mark.parametrize("map_name", ["default", "small", "green_dragon_minimal", "wotans_retribution"])
@pytest.mark.parametrize("grid_size", [24, 48, 64])
def test_ruin_guard_uses_scaled_entrance_and_loses_protection_outside(make_env, map_name, grid_size):
    env = make_env(map_name, grid_size=grid_size)
    for eid in env.RUIN_REWARD_BY_ENEMY_ID:
        assert env._fear_paralysis_teams_for_battle(eid) == ("red",)
        env.grid_env.enemy_positions[eid] = field_tile(env)
        assert env._fear_paralysis_teams_for_battle(eid) == ()


@pytest.mark.parametrize("map_name", ["default", "wotans_retribution"])
def test_neutral_city_defenders_and_nearby_tiles(make_env, map_name):
    env = make_env(map_name)
    for city, source in env.legions_settlement_territory_source_by_name.items():
        tile = env.legions_settlement_source_tile_by_name[city]
        for eid in source["required_enemy_ids"]:
            assert env._fear_paralysis_teams_for_battle(eid) == ("red",)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                adjacent = (tile[0] + dx, tile[1] + dy)
                if adjacent not in site_tiles(env):
                    env.grid_env.enemy_positions[eid] = adjacent
                    assert env._fear_paralysis_teams_for_battle(eid) == ()
            env.grid_env.enemy_positions[eid] = tile


def test_attack_departing_capital_and_remote_summon_do_not_protect_blue(make_env):
    env = make_env("siege_train")
    env.battle_origin_pos = env.CASTLE_POS
    env.grid_env.enemy_positions[1] = field_tile(env)
    assert env._fear_paralysis_teams_for_battle(1) == ()
    env.current_battle_context = {"kind": "summon_spell"}
    assert env._fear_paralysis_teams_for_battle(1, attacker_team="red") == ()
    env.grid_env.enemy_positions[1] = env.CASTLE_POS
    assert env._fear_paralysis_teams_for_battle(1) == ("red",)


def fast_baroness(env, team, position):
    caster = unit(env, "Баронесса", team, position)
    caster.update(initiative=10000, initiative_base=10000, accuracy=100,
                  health=5000, hp=5000, max_health=5000)
    return caster


def attack_site(env, eid, red=None):
    """Enter from adjacent via normal movement; retain actual red guardian."""
    target = tuple(env.grid_env.enemy_positions[eid])
    for other in env.grid_env.enemies_alive:
        env.grid_env.enemies_alive[other] = other == eid
    env.grid_env.agent_pos = (target[0] + 1, target[1])
    env.grid_env.obstacle_positions.discard(env.grid_env.agent_pos)
    env.blue_team_state = env._build_battle_team_with_placeholders(
        "blue", [fast_baroness(env, "blue", 7)])
    if red is not None:
        env.enemy_team_states[eid] = env._build_battle_team_with_placeholders("red", red)
    env.moves = 100
    result = env.step(2)
    assert env.mode == env.MODE_BATTLE, result[4]
    assert env.current_enemy_id == eid
    assert env.battle_env.fear_paralysis_teams == frozenset({"red"})
    return env.battle_env


@pytest.mark.parametrize("location", ["city", "capital", "ruin"])
def test_public_entry_fear_cannot_defeat_or_despawn_guardian(make_env, location):
    env = make_env(Realcapital=2)
    eid = {"city": 22, "capital": 75, "ruin": 70}[location]
    red = None if location == "capital" else [unit(env, "Скваер", "red", 1)]
    battle = attack_site(env, eid, red)
    caster = battle._unit_by_position(7)
    victims = [u for u in battle.combined if u["team"] == "red" and battle._alive(u)]
    assert victims
    if location == "capital":
        assert [(u["name"], u["health"]) for u in victims] == [("Мизраэль", 900)]
    original = [(id(u), u["name"], u["health"], u["exp_current"]) for u in victims]
    for victim in victims:
        assert battle._attack(caster, victim["position"])[0]
        assert victim["paralyzed"] == 1
        assert victim["running_away"] == victim["feared"] == 0
    caster["accuracy"] = 0  # The next turn should not reapply fear.
    result = env.step(DEFEND_ACTION_INDEX)
    assert not any(result[2:4])
    assert env.mode == env.MODE_BATTLE
    assert battle.winner is None
    assert env.grid_env.enemies_alive[eid]
    assert battle.escaped_units == []
    assert battle.last_battle_exp == 0
    assert [(id(u), u["name"], u["health"], u["exp_current"]) for u in victims] == original
    assert all(u["paralyzed"] == 0 for u in victims)
    health_before = caster["health"]
    for victim in victims:
        victim["accuracy"] = 100
    env.step(DEFEND_ACTION_INDEX)
    assert caster["health"] < health_before  # Defenders act on the next activation.
    assert battle.escaped_units == []


@pytest.mark.parametrize("location", ["capital", "city", "ruin"])
def test_defending_hero_keeps_frightened_party_at_site(make_env, location):
    env = make_env()
    tile = {"capital": env.CASTLE_POS,
            "city": next(iter(env.legions_settlement_source_name_by_tile)),
            "ruin": env._static_enemy_positions[next(iter(env.RUIN_REWARD_BY_ENEMY_ID))]}[location]
    env.battle_origin_pos = env.grid_env.agent_pos = tile
    env.current_battle_context = {"kind": "hero"}
    env.current_enemy_id = 1
    env.mode = env.MODE_BATTLE
    env.blue_team_state = env._build_battle_team_with_placeholders(
        "blue", [unit(env, "Одержимый", "blue", 7), unit(env, "Одержимый", "blue", 8)])
    env.enemy_team_states[1] = env._build_battle_team_with_placeholders(
        "red", [fast_baroness(env, "red", 1)])
    # Opening automatic fear is deterministic; stop before it to fear both units.
    original_advance = BattleEnv._advance_until_blue_turn
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(BattleEnv, "_advance_until_blue_turn", lambda self: True)
        env._init_battle(1, attacker_team="red")
    battle = env.battle_env
    caster = battle._unit_by_position(1)
    victims = [u for u in battle.combined if u["team"] == "blue" and battle._alive(u)]
    before = [(id(u), u["name"], u["health"]) for u in victims]
    for u in victims:
        assert battle._attack(caster, u["position"])[0]
    caster["accuracy"] = 0
    result = env.step(DEFEND_ACTION_INDEX)
    assert not any(result[2:4])
    assert env.mode == env.MODE_BATTLE and battle.winner is None
    assert battle.escaped_units == []
    assert [(id(u), u["name"], u["health"]) for u in victims] == before
    assert all(u["paralyzed"] == u["running_away"] == u["feared"] == 0 for u in victims)
    assert battle.current_blue_attacker_pos in [7, 8]
    assert BattleEnv._advance_until_blue_turn is original_advance


def test_garrison_fear_does_not_remove_guard_or_capture_city(make_env):
    env = make_env()
    city = env.garrison_city_names[0]
    for eid in env.legions_settlement_territory_source_by_name[city]["required_enemy_ids"]:
        env.grid_env.mark_enemy_defeated(eid)
    env.grid_env.agent_pos = env.legions_settlement_source_tile_by_name[city]
    env._capture_cities_on_hero_entry(0, {})
    hero_before = deepcopy(env.blue_team_state)
    env.city_garrisons[city] = [unit(env, "Одержимый", "blue", 7)]
    env.scripted_capital_bot_team_state = [fast_baroness(env, "blue", 7)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(BattleEnv, "_advance_until_blue_turn", lambda self: True)
        env._start_city_defence(city)
    battle = env.battle_env
    guard = battle._unit_by_position(7)
    caster = battle._unit_by_position(1)
    health_before = guard["health"]
    assert battle._attack(caster, 7)[0]
    caster["accuracy"] = 0
    env.step(DEFEND_ACTION_INDEX)
    assert battle.winner is None and battle.escaped_units == []
    assert guard["health"] == health_before and guard["paralyzed"] == 0
    assert env._garrison_player_owns(city)
    assert env.blue_team_state == hero_before
    assert len(env.city_garrisons[city]) == 1
    # A normal win saves the frightened guard instead of dropping it from reserve.
    caster["health"] = caster["hp"] = 1
    guard["accuracy"] = 100
    result = env.step(0)  # Guard's native attack against RED position 1.
    assert not any(result[2:4])
    assert battle.winner == "blue" and env.mode == env.MODE_GRID
    assert env._garrison_player_owns(city)
    saved = [u for u in env.city_garrisons[city] if u.get("health", 0) > 0]
    assert len(saved) == 1 and saved[0]["name"] == "Одержимый"
    assert saved[0]["health"] == health_before
    assert env.blue_team_state == hero_before


@pytest.mark.parametrize("map_name", ["siege_train", "super_last_stand"])
def test_natural_incoming_wave_protects_only_capital_defenders(make_env, map_name):
    env = make_env(map_name)
    for u in env.blue_team_state:
        if u.get("health", 0) > 0:
            u.update(initiative=10000, initiative_base=10000)
    for _ in range(100):
        result = env.step(8)
        assert not any(result[2:4])
        if env.mode == env.MODE_BATTLE:
            break
    else:
        pytest.fail("No incoming wave reached the capital")
    assert env.current_battle_context["attacker_team"] == "red"
    assert env.battle_env.fear_paralysis_teams == frozenset({"blue"})


def test_public_attack_out_of_capital_is_field_fear(make_env):
    env = make_env("siege_train")
    env.grid_env.enemy_positions[1] = (env.CASTLE_POS[0] + 1, env.CASTLE_POS[1])
    env.blue_team_state = env._build_battle_team_with_placeholders(
        "blue", [fast_baroness(env, "blue", 7)])
    result = env.step(3)
    assert env.mode == env.MODE_BATTLE, result[4]
    assert env.battle_origin_pos == env.CASTLE_POS
    battle = env.battle_env
    assert not battle.fear_paralysis_teams
    target = next(u for u in battle.combined if u["team"] == "red" and battle._alive(u))
    assert battle._attack(battle._unit_by_position(7), target["position"])[0]
    assert target["running_away"] == target["feared"] == 1
    assert target["paralyzed"] == 0
