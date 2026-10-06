"""W7: exclusive last-paint ownership with agent-then-bot daily growth.

Public REST regressions use ordinary parties, local5, and no scripted army.
Other tests isolate source reach, city ownership, and publication of caches.
"""
from copy import deepcopy

import numpy as np
import pytest

from campaign_env import CampaignEnv
from maps import available_maps, get_map


@pytest.fixture
def make_env():
    environments = []

    def create(map_name="small", **kwargs):
        options = dict(map_name=map_name, Realcapital=2, observation_version="local5",
                       scripted_capital_bot_enabled=False,
                       use_boss_starting_roster=False, log_enabled=False)
        options.update(kwargs)
        env = CampaignEnv(**options)
        environments.append(env)
        env.reset(seed=42)
        return env

    yield create
    for env in environments:
        env.close()


def owned(grid, owner):
    ys, xs = np.nonzero(grid == owner)
    return frozenset((int(x), int(y)) for y, x in zip(ys, xs))


def paint_expected(grid, owner, tiles):
    for x, y in tiles:
        grid[y, x] = owner


def assert_ownership_consistent(env):
    grid = env.territory_owner_grid
    assert grid.shape == (env.grid_size, env.grid_size)
    assert grid.dtype == np.uint8
    assert (env.TERRITORY_NEUTRAL, env.TERRITORY_AGENT, env.TERRITORY_BOT) == (0, 1, 2)
    assert set(np.unique(grid)) <= {0, 1, 2}
    assert isinstance(env.legions_territory_tile_set, frozenset)
    assert isinstance(env.empire_territory_tile_set, frozenset)
    assert env.legions_territory_tile_set == owned(grid, env.TERRITORY_AGENT)
    assert env.empire_territory_tile_set == owned(grid, env.TERRITORY_BOT)
    assert not env.legions_territory_tile_set & env.empire_territory_tile_set
    assert len(env.legions_territory_tiles) == len(env.legions_territory_tile_set)
    assert len(env.empire_territory_tiles) == len(env.empire_territory_tile_set)
    assert frozenset(env.legions_territory_tiles) == env.legions_territory_tile_set
    assert frozenset(env.empire_territory_tiles) == env.empire_territory_tile_set
    mines = tuple(t for t in env.gold_mine_tiles if grid[t[1], t[0]] == env.TERRITORY_AGENT)
    assert env.legions_captured_gold_mine_tiles == mines
    assert env.legions_captured_gold_mine_count == len(mines)
    for kind in env.MANA_KIND_ORDER:
        sources = tuple(sorted(t for t, meta in env.mana_sources.items()
                               if meta["kind"] == kind and grid[t[1], t[0]] == env.TERRITORY_AGENT))
        assert env.legions_captured_mana_source_tiles_by_kind[kind] == sources
        assert env.legions_captured_mana_source_counts_by_kind[kind] == len(sources)


def capture_city(env, city=None):
    city = city or env.garrison_city_names[0]
    source = env.legions_settlement_territory_source_by_name[city]
    for enemy_id in source["required_enemy_ids"]:
        env.grid_env.mark_enemy_defeated(enemy_id)
    env.grid_env.agent_pos = tuple(env.legions_settlement_source_tile_by_name[city])
    env._capture_cities_on_hero_entry(0, {})
    assert env._garrison_player_owns(city)
    return city


def capital_reach(env):
    # Independent of the new combined-source helper: preserve original BFS
    # ordering, rates, and cap rather than accidentally testing its own answer.
    agent_count = min(len(env.legions_territory_order),
                      1 + env.turns * env.LEGIONS_TERRITORY_EXPANSION_PER_TURN)
    bot_count = min(len(env.empire_territory_order),
                    1 + env.turns * env.EMPIRE_TERRITORY_EXPANSION_PER_TURN)
    return (set(env.legions_territory_order[:agent_count]),
            set(env.empire_territory_order[:bot_count]) if env.empire_territory_enabled else set())


def test_owner_grid_and_compatibility_sets_are_safe_snapshots(make_env):
    env = make_env()
    assert_ownership_consistent(env)
    snapshot = env.territory_owner_grid
    before = env._territory_owner_grid.copy()
    assert not snapshot.flags.writeable
    with pytest.raises(ValueError):
        snapshot[0, 0] = env.TERRITORY_BOT
    with pytest.raises(AttributeError):
        env.legions_territory_tile_set.add((0, 0))
    with pytest.raises(AttributeError):
        env.empire_territory_tile_set.discard((0, 0))
    # Even an explicit NumPy writeability override cannot mutate the live grid.
    snapshot.setflags(write=True)
    snapshot[:] = env.TERRITORY_NEUTRAL
    np.testing.assert_array_equal(env._territory_owner_grid, before)


def test_neutral_agent_bot_repainting_uses_y_x_and_one_final_owner(make_env):
    env = make_env()
    tile = (3, 7)
    for owner in (env.TERRITORY_AGENT, env.TERRITORY_BOT, env.TERRITORY_AGENT,
                  env.TERRITORY_NEUTRAL):
        env._paint_territory_owner(owner, (tile, tile))
        env._refresh_territory_ownership_caches()
        assert env.territory_owner_grid[7, 3] == owner
        assert_ownership_consistent(env)


@pytest.mark.parametrize("owner,tiles", [(3, [(2, 3)]), (1, [(2, 3), (-1, 0)]),
                                         (2, [(2, 3), (24, 0)])])
def test_invalid_paint_does_not_partially_mutate_ownership(make_env, owner, tiles):
    env = make_env()
    before = env.territory_owner_grid
    with pytest.raises(ValueError):
        env._paint_territory_owner(owner, tiles)
    np.testing.assert_array_equal(env.territory_owner_grid, before)


@pytest.mark.parametrize("start_day,overlap", [(0, 0), (7, 13)])
def test_completed_turn_paints_full_agent_area_then_full_bot_area(make_env, monkeypatch,
                                                                 start_day, overlap):
    env = make_env()
    env._advance_turns(start_day)
    before = env.territory_owner_grid.copy()
    phases = []
    original = env._apply_territory_growth_phase

    def observe_phase(owner, tiles):
        tiles = tuple(tiles)
        original(owner, tiles)
        phases.append((owner, set(tiles), env.territory_owner_grid.copy()))

    monkeypatch.setattr(env, "_apply_territory_growth_phase", observe_phase)
    env._advance_turns(1)
    agent, bot = capital_reach(env)
    assert len(agent & bot) == overlap
    assert [p[0] for p in phases] == [env.TERRITORY_AGENT, env.TERRITORY_BOT]
    assert phases[0][1] == agent and phases[1][1] == bot
    paint_expected(before, env.TERRITORY_AGENT, agent)
    np.testing.assert_array_equal(phases[0][2], before)
    assert all(phases[0][2][y, x] == env.TERRITORY_AGENT for x, y in agent & bot)
    paint_expected(before, env.TERRITORY_BOT, bot)
    np.testing.assert_array_equal(phases[1][2], before)
    np.testing.assert_array_equal(env.territory_owner_grid, before)
    assert_ownership_consistent(env)


def test_old_overlap_is_repainted_every_turn_even_after_saturation(make_env, monkeypatch):
    env = make_env()
    env._advance_turns(15)
    agent, bot = capital_reach(env)
    assert agent == bot and len(agent) == 149
    phases = []
    original = env._apply_territory_growth_phase

    def observe_phase(owner, tiles):
        tiles = tuple(tiles)
        original(owner, tiles)
        phases.append((owner, frozenset(tiles), env.territory_owner_grid.copy()))

    monkeypatch.setattr(env, "_apply_territory_growth_phase", observe_phase)
    env._advance_turns(2)
    assert [p[0] for p in phases] == [env.TERRITORY_AGENT, env.TERRITORY_BOT] * 2
    for owner, tiles, grid in phases:
        assert tiles == agent
        assert all(grid[y, x] == owner for x, y in agent)
    assert not env.legions_territory_tile_set
    assert env.empire_territory_tile_set == bot
    assert_ownership_consistent(env)


def test_public_rest_small_day_8_14_15_income_and_final_owner(make_env):
    env = make_env()
    assert not env.scripted_capital_bot_config_enabled
    assert env.empire_territory_enabled
    assert not env.use_boss_starting_roster
    for day in range(1, 17):
        old_gold, old_mana = env.gold, env._current_mana_totals()
        observation, _, terminated, truncated, info = env.step(8)
        assert env.turns == day
        assert not terminated and not truncated
        assert env.observation_space.contains(observation)
        assert_ownership_consistent(env)
        agent, bot = capital_reach(env)
        assert env.legions_territory_tile_set == agent - bot
        assert env.empire_territory_tile_set == bot
        assert not set(info["legions_territory_tiles"]) & set(info["empire_territory_tiles"])
        assert env.gold - old_gold == env._legions_gold_income_per_turn()
        for kind, income in env._compute_mana_income_per_turn().items():
            assert env._current_mana_totals()[kind] - old_mana[kind] == income
        if day == 8:
            assert len(agent & bot) == 13
            assert (0, 2) in agent & bot
            assert env.territory_owner_grid[2, 0] == env.TERRITORY_BOT
            assert env.mana_income_per_turn["runes"] == 0
            assert env._current_mana_totals()["runes"] - old_mana["runes"] == 0
        if day == 14:
            assert len(agent & bot) == 133
            assert (1, 15) in agent & bot
            assert env.territory_owner_grid[15, 1] == env.TERRITORY_BOT
            assert env.gold - old_gold == env.GOLD_PER_TURN == 100
        if day >= 15:
            assert len(agent & bot) == 149
            assert not env.legions_territory_tile_set
            assert env.gold - old_gold == 100


def test_income_is_evaluated_only_after_both_growth_phases(make_env, monkeypatch):
    env = make_env()
    env._advance_turns(13)
    events = []
    original_phase = env._apply_territory_growth_phase
    original_gold = env._legions_gold_income_per_turn
    original_mana = env._apply_mana_income_for_turn

    def phase(owner, tiles):
        original_phase(owner, tiles)
        events.append(owner)

    def gold():
        assert events[:2] == [env.TERRITORY_AGENT, env.TERRITORY_BOT]
        assert env.territory_owner_grid[15, 1] == env.TERRITORY_BOT
        events.append("gold")
        return original_gold()

    def mana():
        assert events[:2] == [env.TERRITORY_AGENT, env.TERRITORY_BOT]
        assert env.territory_owner_grid[2, 0] == env.TERRITORY_BOT
        events.append("mana")
        return original_mana()

    monkeypatch.setattr(env, "_apply_territory_growth_phase", phase)
    monkeypatch.setattr(env, "_legions_gold_income_per_turn", gold)
    monkeypatch.setattr(env, "_apply_mana_income_for_turn", mana)
    env._advance_turns(1)
    assert events.count("gold") == events.count("mana") == 1


@pytest.mark.parametrize("owner", [0, 1, 2])
def test_healing_local_and_global_observations_use_final_owner(make_env, owner):
    env = make_env()
    env._advance_turns(8)
    tile = (0, 2)  # Reached by both sources, away from a settlement/capital.
    env._paint_territory_owner(owner, (tile,))
    env._refresh_territory_ownership_caches()
    env.grid_env.agent_pos = tile
    player = owner == env.TERRITORY_AGENT
    bot = owner == env.TERRITORY_BOT
    assert env._is_player_controlled_territory_tile(tile) is player
    expected_heal = (env.PLAYER_TERRITORY_REST_HEAL_PERCENT if player
                     else env.DEFAULT_REST_HEAL_PERCENT)
    assert env._resolve_rest_heal_percent(tile) == expected_heal
    local = env._local_observation
    center = local.tiles().reshape(local.tile_count, local.tile_width)[local.offsets.index((0, 0))]
    assert center[local.TILE_FEATURES.index("legions_territory")] == float(player)
    assert center[local.TILE_FEATURES.index("empire_territory")] == float(bot)
    index = env.grid_legions_territory_index_by_pos[tile]
    assert env._build_legions_territory_grid_obs()[index] == float(player)
    assert env._build_empire_territory_grid_obs()[index] == float(bot)
    assert env.observation_space.contains(env._build_obs())


@pytest.mark.parametrize("owner,fraction", [(0, .15), (1, .05), (2, .15)])
def test_actual_enemy_regeneration_uses_final_owner_without_repainting(make_env, monkeypatch,
                                                                     owner, fraction):
    env = make_env()
    tile = (0, 2)
    assert tile not in env.legions_settlement_source_name_by_tile
    assert tile != tuple(env.empire_territory_source_tile)
    excluded = set(env.legions_settlement_territory_source_name_by_enemy_id)
    excluded.add(env.GREEN_DRAGON_OBJECTIVE_ENEMY_ID)
    enemy_id = next(eid for eid in env.grid_env.enemies_alive if eid not in excluded)
    ordinary_unit = next(u for u in env._get_enemy_team_state(enemy_id)
                         if not env._is_empty_enemy_unit(u))
    ordinary_unit.update(health=10, hp=10, max_health=200, maxhp=200)
    env.enemy_team_states[enemy_id] = [ordinary_unit]
    for eid in env.grid_env.enemies_alive:
        env.grid_env.enemies_alive[eid] = eid == enemy_id
    env.grid_env.enemy_positions[enemy_id] = tile
    env._paint_territory_owner(owner, (tile,))
    env._refresh_territory_ownership_caches()
    before = env.territory_owner_grid

    def forbidden_growth(*args, **kwargs):
        pytest.fail("Enemy regeneration replayed territory growth")

    monkeypatch.setattr(env, "_apply_territory_growth_phase", forbidden_growth)
    healed_count, healed_hp = env._heal_wounded_enemy_teams_for_turn()
    assert healed_count == 1
    assert healed_hp == pytest.approx(200 * fraction)
    assert ordinary_unit["health"] == ordinary_unit["hp"] == pytest.approx(10 + 200 * fraction)
    np.testing.assert_array_equal(env.territory_owner_grid, before)
    assert_ownership_consistent(env)


def test_cache_and_read_queries_never_replay_growth(make_env, monkeypatch):
    env = make_env()
    env._advance_turns(8)
    tile = (0, 2)
    env._paint_territory_owner(env.TERRITORY_AGENT, (tile,))
    before = env.territory_owner_grid
    history = deepcopy(env.legions_settlement_growth_history_by_name)

    def forbidden_growth(*args, **kwargs):
        pytest.fail("A read/cache operation replayed a territory growth phase")

    monkeypatch.setattr(env, "_apply_territory_growth_phase", forbidden_growth)
    for _ in range(2):
        env._refresh_territory_ownership_caches()
        env._refresh_legions_territory_tiles()
        env._refresh_legions_territory_grid_obs_cache()
        env._refresh_legions_captured_mana_source_state()
        env._territory_reached_tiles()
        env._legions_settlement_territory_info()
        env._is_player_controlled_territory_tile(tile)
        env._resolve_rest_heal_percent(tile)
        env._legions_gold_income_per_turn()
        env._compute_mana_income_per_turn()
        env._build_legions_territory_grid_obs()
        env._build_empire_territory_grid_obs()
        env._build_obs()
        env.compute_action_mask()
        np.testing.assert_array_equal(env.territory_owner_grid, before)
        assert env.legions_settlement_growth_history_by_name == history
        assert_ownership_consistent(env)


def test_default_capital_geometry_and_growth_rates_are_unchanged(make_env):
    env = make_env("default")
    assert len(env.legions_territory_order) == 309
    assert len(env.empire_territory_order) == 161
    assert not set(env.legions_territory_order) & set(env.empire_territory_order)
    assert env.LEGIONS_TERRITORY_EXPANSION_PER_TURN == 10
    assert env.EMPIRE_TERRITORY_EXPANSION_PER_TURN == 10
    for day in (0, 1, 8, 14, 31, 32):
        env._advance_turns(day - env.turns)
        agent, bot = capital_reach(env)
        assert env.legions_territory_tile_set == agent
        assert env.empire_territory_tile_set == bot
        assert_ownership_consistent(env)


@pytest.mark.parametrize("empire,scripted", [(False, False), (False, True),
                                           (True, False), (True, True)])
def test_capital_territory_flag_is_independent_of_scripted_army(make_env, monkeypatch,
                                                             empire, scripted):
    env = make_env(empire_territory_enabled=empire, scripted_capital_bot_enabled=scripted)
    assert env.empire_territory_enabled is empire
    assert env.scripted_capital_bot_config_enabled is scripted
    # Exercise territory wiring without starting a scripted army encounter.
    monkeypatch.setattr(env, "_advance_scripted_capital_bot_one_turn", lambda: {})
    env._advance_turns(8)
    agent, bot = capital_reach(env)
    assert env.legions_territory_tile_set == agent - bot
    assert env.empire_territory_tile_set == bot
    assert bool(bot) is empire
    assert_ownership_consistent(env)


def test_city_capture_transfers_reached_area_without_replaying_unrelated_sources(make_env,
                                                                               monkeypatch):
    env = make_env("green_dragon_minimal")
    city = capture_city(env)
    env._advance_turns(3)
    reached = set(env.legions_settlement_territory_tiles_by_name[city])
    assert len(reached) > 1
    unrelated = next(t for t in env.legions_territory_tile_set if t not in reached)
    env._paint_territory_owner(env.TERRITORY_BOT, (unrelated,))
    env._refresh_territory_ownership_caches()
    before = env.territory_owner_grid.copy()

    def forbidden_growth(*args, **kwargs):
        pytest.fail("City capture replayed a daily growth phase")

    monkeypatch.setattr(env, "_apply_territory_growth_phase", forbidden_growth)
    env._transfer_city_to_bot(city)
    paint_expected(before, env.TERRITORY_BOT, reached)
    np.testing.assert_array_equal(env.territory_owner_grid, before)
    assert set(env.bot_city_initial_tiles[city]) == reached
    assert city not in env.legions_active_settlement_territory_capture_turn_by_name
    assert city not in env.legions_settlement_growth_history_by_name
    assert env.territory_owner_grid[unrelated[1], unrelated[0]] == env.TERRITORY_BOT
    assert_ownership_consistent(env)


def test_recapture_changes_source_atomically_and_retains_other_land_until_repaint(make_env,
                                                                               monkeypatch):
    env = make_env("green_dragon_minimal")
    city = capture_city(env)
    env._advance_turns(3)
    env._transfer_city_to_bot(city)
    before = env.territory_owner_grid.copy()
    source = env.legions_settlement_source_tile_by_name[city]
    original_phase = env._apply_territory_growth_phase

    def forbidden_growth(*args, **kwargs):
        pytest.fail("Agent recapture replayed daily growth")

    monkeypatch.setattr(env, "_apply_territory_growth_phase", forbidden_growth)
    capture_city(env, city)
    paint_expected(before, env.TERRITORY_AGENT, (source,))
    np.testing.assert_array_equal(env.territory_owner_grid, before)
    assert city not in env.bot_city_capture_turns
    assert city not in env.bot_city_initial_tiles
    assert env.legions_active_settlement_territory_capture_turn_by_name[city] == env.turns
    assert env.legions_settlement_territory_tiles_by_name[city] == (source,)
    assert_ownership_consistent(env)
    monkeypatch.setattr(env, "_apply_territory_growth_phase", original_phase)
    env._advance_turns(1)
    agent, bot = env._territory_reached_tiles()
    paint_expected(before, env.TERRITORY_AGENT, agent)
    paint_expected(before, env.TERRITORY_BOT, bot)
    np.testing.assert_array_equal(env.territory_owner_grid, before)
    assert_ownership_consistent(env)


def test_losing_cells_does_not_rewind_city_reach_or_upgrade_history(make_env):
    env = make_env("green_dragon_minimal")
    city = capture_city(env)
    original_rate = env._settlement_expansion_per_turn(city)
    env._advance_turns(2)
    reached = tuple(env.legions_settlement_territory_tiles_by_name[city])
    assert len(reached) == 1 + 2 * original_rate
    history = deepcopy(env.legions_settlement_growth_history_by_name)
    env._paint_territory_owner(env.TERRITORY_BOT, reached)
    env._refresh_territory_ownership_caches()
    assert env.legions_settlement_territory_tiles_by_name[city] == reached
    assert env.legions_settlement_growth_history_by_name == history
    assert env._garrison_player_owns(city)
    before_upgrade = env.territory_owner_grid
    env._set_settlement_level_with_territory_growth(city, 5)
    np.testing.assert_array_equal(env.territory_owner_grid, before_upgrade)
    env._advance_turns(1)
    expected_count = 1 + 2 * original_rate + env._settlement_expansion_per_turn(city)
    expected = env.legions_settlement_territory_orders_by_name[city][:expected_count]
    assert env.legions_settlement_territory_tiles_by_name[city] == expected
    assert all(env.territory_owner_grid[y, x] == env.TERRITORY_AGENT for x, y in expected)
    assert_ownership_consistent(env)


def test_bot_city_growth_and_observation_work_without_empire_capital_or_scripted_army(make_env):
    env = make_env("green_dragon_minimal", empire_territory_enabled=False)
    city = capture_city(env)
    env._transfer_city_to_bot(city)
    assert not env.empire_territory_enabled and not env.scripted_capital_bot_config_enabled
    assert not env.empire_territory_order
    env._advance_turns(2)
    count = 1 + 2 * env._settlement_expansion_per_turn(city)
    expected = set(env.legions_settlement_territory_orders_by_name[city][:count])
    assert expected <= env.empire_territory_tile_set
    full_obs = env._build_empire_territory_grid_obs()
    for tile in expected:
        assert full_obs[env.grid_legions_territory_index_by_pos[tile]] == 1
    assert_ownership_consistent(env)


@pytest.mark.parametrize("map_name", available_maps())
def test_registered_map_reset_and_one_turn_keep_exclusive_ownership(make_env, map_name):
    env = make_env(map_name)
    shape, action_count = env.observation_space.shape, env.action_space.n
    initial = env.territory_owner_grid
    assert_ownership_consistent(env)
    if get_map(map_name).empire_territory_source_tile is None:
        assert not env.empire_territory_enabled
        assert not env.empire_territory_tile_set
    env._advance_turns(1)
    assert_ownership_consistent(env)
    observation, _ = env.reset(seed=42)
    np.testing.assert_array_equal(env.territory_owner_grid, initial)
    assert env.observation_space.shape == shape
    assert env.action_space.n == action_count
    assert env.observation_space.contains(observation)
    assert env.compute_action_mask().shape == (action_count,)
    assert_ownership_consistent(env)


def test_reset_clears_far_land_and_city_sources_without_affecting_another_env(make_env):
    env = make_env("green_dragon_minimal")
    other = make_env("green_dragon_minimal")
    initial = env.territory_owner_grid
    other_initial = other.territory_owner_grid
    city = capture_city(env)
    env._advance_turns(4)
    env._transfer_city_to_bot(city)
    env._paint_territory_owner(env.TERRITORY_AGENT, ((31, 31),))
    env._refresh_territory_ownership_caches()
    observation, _ = env.reset(seed=42)
    np.testing.assert_array_equal(env.territory_owner_grid, initial)
    np.testing.assert_array_equal(other.territory_owner_grid, other_initial)
    assert not env.bot_city_capture_turns and not env.bot_city_initial_tiles
    assert not env.legions_active_settlement_territory_capture_turn_by_name
    assert not env.legions_settlement_growth_history_by_name
    assert env.observation_space.contains(observation)
    assert_ownership_consistent(env)


def test_deepcopy_owns_an_independent_grid_and_preserves_growth(make_env):
    env = make_env()
    env._advance_turns(8)
    clone = deepcopy(env)
    try:
        assert not np.shares_memory(env._territory_owner_grid, clone._territory_owner_grid)
        np.testing.assert_array_equal(clone.territory_owner_grid, env.territory_owner_grid)
        before = env.territory_owner_grid
        clone._paint_territory_owner(clone.TERRITORY_AGENT, ((0, 2),))
        clone._refresh_territory_ownership_caches()
        np.testing.assert_array_equal(env.territory_owner_grid, before)
        assert clone.territory_owner_grid[2, 0] == clone.TERRITORY_AGENT
        clone._advance_turns(1)
        env._advance_turns(1)
        np.testing.assert_array_equal(clone.territory_owner_grid, env.territory_owner_grid)
        assert_ownership_consistent(clone)
    finally:
        clone.close()
