"""B17 edge coverage: explicit roles, ownership, stat layers and lifecycle.

Integration entry tests live in test_defender_fortification_integration.py.
These fixtures stop automatic opening attacks to inspect initialization exactly;
they never train a model or enable the scripted capital bot.
"""
from copy import deepcopy

import pytest

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from maps import available_maps
from permanent_unit_stats import add_permanent_effect

# city_defence_train has no field enemy rosters; its separate reserve battle
# initializer is covered in the integration module.
MAPS = tuple(name for name in available_maps() if name != "city_defence_train")


@pytest.fixture
def make_env(monkeypatch):
    monkeypatch.setattr(BattleEnv, "_advance_until_blue_turn", lambda self: True)
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


def fighters(units, team=None):
    return [u for u in units
            if u.get("max_health", 0) > 0 and (team is None or u["team"] == team)]


def armors(units):
    return {u["position"]: u["armor"] for u in fighters(units)}


def assert_bonus(env, team, originals, expected):
    units = fighters(env.battle_env.combined, team)
    assert units
    for unit in units:
        assert unit.get("settlement_armor_bonus", 0) == expected
        assert unit["armor"] == originals[unit["position"]] + expected


def start(env, enemy_id, position, attacker):
    env.battle_origin_pos = tuple(position)
    env.grid_env.agent_pos = tuple(position)
    env.current_enemy_id = enemy_id
    env.current_battle_context = {"kind": "hero"}
    env.mode = env.MODE_BATTLE
    env._init_battle(enemy_id, attacker_team=attacker)


def field_tile(env):
    excluded = set(env.castle_heal_tiles) | {tuple(env.empire_territory_source_tile)}
    return next((x, y) for x in range(env.grid_size) for y in range(env.grid_size)
                if (x, y) not in excluded
                and (x, y) not in env.grid_env.obstacle_positions)


@pytest.mark.parametrize("map_name", MAPS)
@pytest.mark.parametrize("attacker", ["blue", "red"])
def test_role_and_ownership_matrix_on_every_map(make_env, map_name, attacker):
    env = make_env(map_name)
    enemy_id = next(iter(env._enemy_configs))
    env.grid_env.enemy_positions[enemy_id] = tuple(env.CASTLE_POS)
    blue, red = armors(env.blue_team_state), armors(env.enemy_team_states[enemy_id])
    start(env, enemy_id, env.CASTLE_POS, attacker)
    assert env.current_battle_context["attacker_team"] == attacker
    assert env.current_battle_context["defender_team"] == (
        "red" if attacker == "blue" else "blue"
    )
    assert_bonus(env, "blue", blue, 40 if attacker == "red" else 0)
    # Occupying another side's capital never grants the invader ownership.
    assert_bonus(env, "red", red, 0)


@pytest.mark.parametrize("map_name", MAPS)
def test_field_battles_never_inherit_fortification_on_any_map(make_env, map_name):
    env = make_env(map_name)
    enemy_id = next(iter(env._enemy_configs))
    tile = field_tile(env)
    env.grid_env.enemy_positions[enemy_id] = tile
    blue, red = armors(env.blue_team_state), armors(env.enemy_team_states[enemy_id])
    for attacker in ("blue", "red"):
        start(env, enemy_id, tile, attacker)
        assert_bonus(env, "blue", blue, 0)
        assert_bonus(env, "red", red, 0)


@pytest.mark.parametrize("map_name", ["siege_train", "super_last_stand"])
def test_queued_wave_is_red_attacker(make_env, map_name):
    env = make_env(map_name)
    enemy_id = env._scheduled_enemy_wave_configs()[0]["enemy_id"]
    env.grid_env.enemies_alive[enemy_id] = True
    env.grid_env.enemy_positions[enemy_id] = tuple(env.CASTLE_POS)
    env._queue_scheduled_enemy_encounter(enemy_id)
    blue, red = armors(env.blue_team_state), armors(env.enemy_team_states[enemy_id])
    _, _, terminated, truncated, info = env.step(8)
    assert info["battle_triggered_by"] == "scheduled_wave_queue"
    assert not (terminated or truncated)
    assert env.current_battle_context["attacker_team"] == "red"
    assert_bonus(env, "blue", blue, 40)
    assert_bonus(env, "red", red, 0)


@pytest.mark.parametrize("attacker", ["blue", "red"])
@pytest.mark.parametrize("base_armor", [0, 20, 85, 99])
def test_fortification_adds_to_armor_and_preserves_damage_cap(
    make_env, attacker, base_armor, monkeypatch
):
    env = make_env()
    enemy_id = next(iter(env._enemy_configs))
    defender = "red" if attacker == "blue" else "blue"
    tile = env.empire_territory_source_tile if defender == "red" else env.CASTLE_POS
    env.grid_env.enemy_positions[enemy_id] = tuple(tile)
    units = env.enemy_team_states[enemy_id] if defender == "red" else env.blue_team_state
    for unit in fighters(units):
        unit["armor"] = unit["base_armor"] = base_armor
    start(env, enemy_id, tile, attacker)
    originals = {u["position"]: base_armor for u in fighters(units)}
    assert_bonus(env, defender, originals, 40)
    if base_armor >= 85:
        monkeypatch.setattr(env.battle_env.rng, "randint", lambda low, high: 0)
        victim = fighters(env.battle_env.combined, defender)[0]
        assert env.battle_env._apply_damage_with_armor(
            {"unit_type": "Warrior"}, 100, victim
        ) == 10  # Existing 90% combat cap still applies.


@pytest.mark.parametrize("attacker", ["blue", "red"])
def test_timeout_strips_bonus_before_reentry_and_reset(make_env, attacker, monkeypatch):
    env = make_env()
    enemy_id = next(iter(env._enemy_configs))
    tile = env.empire_territory_source_tile if attacker == "blue" else env.CASTLE_POS
    env.grid_env.enemy_positions[enemy_id] = tuple(tile)
    blue_before, red_before = armors(env.blue_team_state), armors(env.enemy_team_states[enemy_id])
    for _ in range(2):
        start(env, enemy_id, tile, attacker)
        assert_bonus(env, "blue", blue_before, 40 if attacker == "red" else 0)
        assert_bonus(env, "red", red_before, 40 if attacker == "blue" else 0)
        battle = env.battle_env
        monkeypatch.setattr(battle, "step", lambda action: (battle._obs(), 0, False, True, {}))
        _, _, terminated, truncated, info = env.step(0)
        assert info["battle_timeout_no_winner"]
        assert not (terminated or truncated)
        assert env.mode == env.MODE_GRID
        assert env.current_battle_context == {}
        assert env.battle_origin_pos is None
        assert armors(env.blue_team_state) == blue_before
        assert armors(env.enemy_team_states[enemy_id]) == red_before
        for unit in env.blue_team_state + env.enemy_team_states[enemy_id]:
            assert "settlement_armor_bonus" not in unit
            assert "campaign_heal_tile_armor_bonus" not in unit
    env.reset(seed=42)
    assert env.current_battle_context == {}
    assert armors(env.blue_team_state) == blue_before


@pytest.mark.parametrize("attacker", ["blue", "red"])
def test_dead_fighters_stay_dead_and_placeholders_stay_empty(make_env, attacker):
    env = make_env()
    enemy_id = next(iter(env._enemy_configs))
    for team in (env.blue_team_state, env.enemy_team_states[enemy_id]):
        unit = fighters(team)[-1]
        unit["health"] = unit["hp"] = 0
    tile = env.empire_territory_source_tile if attacker == "blue" else env.CASTLE_POS
    env.grid_env.enemy_positions[enemy_id] = tuple(tile)
    original = deepcopy(env.blue_team_state + env.enemy_team_states[enemy_id])
    start(env, enemy_id, tile, attacker)
    for before in original:
        after = next(u for u in env.battle_env.combined
                     if u["team"] == before["team"] and u["position"] == before["position"])
        if before["health"] == 0:
            assert after["health"] == 0
        if before["max_health"] == 0:
            assert not after.get("settlement_armor_bonus", 0)
            assert after["armor"] == 0


def test_permanent_elixir_armor_survives_defence_cleanup(make_env):
    env = make_env()
    enemy_id = next(iter(env._enemy_configs))
    hero = env._resolve_travel_hero()
    add_permanent_effect(hero, {"kind": "armor", "armor_bonus": 7})
    base = armors(env.blue_team_state)
    for _ in range(2):
        start(env, enemy_id, env.CASTLE_POS, "red")
        assert_bonus(env, "blue", base, 40)
        env._save_blue_state()
        assert armors(env.blue_team_state) == base
    start(env, enemy_id, field_tile(env), "blue")
    assert_bonus(env, "blue", base, 0)


@pytest.mark.parametrize("enemy_id", [5, 10])
def test_summon_cast_path_attacks_enemy_capital_without_hero_tile_armor(make_env, enemy_id):
    env = make_env("wotans_retribution")
    summoned, _ = env._build_summoned_blue_team("Адская гончая")
    blue, red = armors(summoned), armors(env.enemy_team_states[enemy_id])
    result = env._cast_from_spell_spec(
        source="scroll", spell_key="lod_d2_s001", spell_description="summon fixture",
        spell_spec=env._map_offensive_spell_spec("lod_d2_s001"),
        nearest_any_enemy={"enemy_id": enemy_id, "position": env.empire_territory_source_tile},
    )
    assert result["battle_triggered"]
    assert env.current_battle_context["kind"] == "summon_spell"
    assert env.current_battle_context["defender_team"] == "red"
    assert_bonus(env, "blue", blue, 0)
    assert_bonus(env, "red", red, 40)


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("attacker", ["blue", "red"])
def test_default_and_persistent_rosters_follow_same_role_rule(make_env, persist, attacker):
    env = make_env(persist_blue_hp=persist)
    enemy_id = next(iter(env._enemy_configs))
    env.grid_env.enemy_positions[enemy_id] = tuple(env.empire_territory_source_tile)
    start(env, enemy_id, env.CASTLE_POS, attacker)
    for team in ("blue", "red"):
        expected = 40 if team != attacker else 0
        assert all(u.get("settlement_armor_bonus", 0) == expected
                   for u in fighters(env.battle_env.combined, team))


def test_invalid_attacker_fails_before_battle_is_created(make_env):
    env = make_env()
    with pytest.raises(ValueError, match="Unknown attacking team"):
        env._init_battle(1, attacker_team="unknown")
    assert env.battle_env is None
