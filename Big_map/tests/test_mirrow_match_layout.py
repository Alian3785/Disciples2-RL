"""Mirrow match layout, resources and one-fighter/two-cell army invariants."""

from collections import Counter, deque
from copy import deepcopy

import numpy as np
import pytest

from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA
from enemy_configs import build_enemy_configs
from maps import get_map
from maps.base import footprint_tiles
from maps.mirrow_match import (
    ADDITIONAL_FORMATIONS,
    AGENT_CAPITAL_BLOCK,
    AGENT_GOLD_MINE_TILE,
    AGENT_MANA_TILE,
    BASE_FORMATIONS,
    BOT_CAPITAL_BLOCK,
    BOT_GOLD_MINE_TILE,
    BOT_HOME,
    BOT_MANA_TILE,
    FOREST_TILES,
    FORMATIONS,
    GOLD_MINE_TILES,
    GRID_SIZE,
    HERO_START,
    ROAD_TILES,
    TIAMAT_ENEMY_ID,
    TIAMAT_TILE,
    WATER_TILES,
    mirror_tile,
)


def _fighters(team):
    return [u for u in team if u.get("name") != "пусто"]


def _assert_tactical_cells(units):
    occupied = set()
    for unit in units:
        position = unit["position"]
        cells = {position}
        assert 1 <= position <= 6
        if unit["big"]:
            assert position <= 3
            cells.add(position + 3)
        assert not cells & occupied
        occupied.update(cells)
    assert len(occupied) <= 6
    assert len({id(unit) for unit in units}) == len(units)


def _reachable(start, blocked):
    """Use the stricter cardinal graph, without diagonal corner cutting."""
    seen = {start}
    pending = deque([start])
    while pending:
        x, y = pending.popleft()
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            tile = (x + dx, y + dy)
            if (0 <= tile[0] < GRID_SIZE and 0 <= tile[1] < GRID_SIZE
                    and tile not in seen and tile not in blocked):
                seen.add(tile)
                pending.append(tile)
    return seen


@pytest.fixture
def env():
    # The requested map objective requires its ordinary scripted opponent.
    environment = CampaignEnv(
        map_name="mirrow_match", observation_version="local5",
        scripted_capital_bot_enabled=True, use_boss_starting_roster=False,
        log_enabled=False,
    )
    environment.reset(seed=42)
    yield environment
    environment.close()


def test_map_has_full_mirrored_roster_and_one_central_tiamat():
    config = get_map("mirrow_match")
    assert config.grid_size == GRID_SIZE == 48
    assert config.hero_start == HERO_START == (5, 5)
    assert config.scripted_capital_bot_home_tile == BOT_HOME == (42, 42)
    assert mirror_tile(HERO_START) == BOT_HOME
    assert len(BASE_FORMATIONS) == 14
    assert len(ADDITIONAL_FORMATIONS) == 5
    assert len(FORMATIONS) == 19
    assert len(config.enemy_stacks) == 39
    by_region = {
        region: {row["formation_key"]: row for row in config.enemy_stacks
                 if row["region"] == region}
        for region in ("agent", "bot", "center")
    }
    assert len(by_region["agent"]) == len(by_region["bot"]) == 19
    assert len(by_region["center"]) == 1
    assert by_region["agent"].keys() == by_region["bot"].keys()
    for key, agent in by_region["agent"].items():
        bot = by_region["bot"][key]
        assert mirror_tile(agent["position"]) == bot["position"]
        assert (agent["front"], agent["back"]) == (bot["front"], bot["back"])
        assert agent["enemy_id"] != bot["enemy_id"]
        assert agent["front"] is not bot["front"]
        assert agent["back"] is not bot["back"]
    central = by_region["center"]["tiamat"]
    assert central["enemy_id"] == TIAMAT_ENEMY_ID
    assert central["position"] == TIAMAT_TILE == (24, 24)
    names = Counter(name for row in config.enemy_stacks
                    for name in (*row["front"], *row["back"]) if name)
    assert names["Тиамат"] == 1
    assert len(config.enemy_positions()) == len(config.enemy_stacks)
    assert len(set(config.enemy_positions().values())) == len(config.enemy_stacks)
    assert BOT_HOME not in config.enemy_positions().values()
    assert HERO_START not in config.enemy_positions().values()


def test_all_five_difficulty_tiers_move_outward_from_each_capital():
    config = get_map("mirrow_match")
    for region, home in (("agent", HERO_START), ("bot", BOT_HOME)):
        tiers = {}
        for row in config.enemy_stacks:
            if row["region"] != region:
                continue
            tile = row["position"]
            distance = max(abs(tile[0] - home[0]), abs(tile[1] - home[1]))
            tiers.setdefault(row["difficulty_tier"], []).append(distance)
            other_home = BOT_HOME if region == "agent" else HERO_START
            assert distance < max(abs(tile[0] - other_home[0]), abs(tile[1] - other_home[1]))
        assert set(tiers) == {1, 2, 3, 4, 5}
        for tier in range(1, 5):
            assert max(tiers[tier]) < min(tiers[tier + 1])


def test_capital_footprints_and_all_environment_layers_are_symmetric(env):
    agent_walls = footprint_tiles(AGENT_CAPITAL_BLOCK) - {HERO_START}
    bot_walls = footprint_tiles(BOT_CAPITAL_BLOCK) - {BOT_HOME}
    assert len(agent_walls) == len(bot_walls) == 24
    assert {mirror_tile(tile) for tile in agent_walls} == bot_walls
    assert set(env._static_obstacle_tiles) == agent_walls | bot_walls
    assert not {HERO_START, BOT_HOME} & set(env._static_obstacle_tiles)
    for tiles in (WATER_TILES, FOREST_TILES, ROAD_TILES, GOLD_MINE_TILES):
        assert {mirror_tile(tile) for tile in tiles} == set(tiles)
        assert all(0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE for x, y in tiles)
    assert set(env.water_tiles) == set(WATER_TILES)
    assert set(env.forest_tiles) == set(FOREST_TILES)
    assert set(env.road_tiles) == set(ROAD_TILES)
    assert not set(WATER_TILES) & (set(FOREST_TILES) | set(ROAD_TILES))
    assert not (set(WATER_TILES) | set(FOREST_TILES) | set(ROAD_TILES)) & set(env._static_obstacle_tiles)


def test_resources_are_near_each_capital_and_stay_at_configured_tiles(env):
    assert set(env.gold_mine_tiles) == {AGENT_GOLD_MINE_TILE, BOT_GOLD_MINE_TILE}
    assert set(env.mana_sources) == {AGENT_MANA_TILE, BOT_MANA_TILE}
    assert env.mana_sources[AGENT_MANA_TILE]["kind"] == "infernal"
    assert env.mana_sources[BOT_MANA_TILE]["kind"] == "life"
    for home, gold, mana in ((HERO_START, AGENT_GOLD_MINE_TILE, AGENT_MANA_TILE),
                             (BOT_HOME, BOT_GOLD_MINE_TILE, BOT_MANA_TILE)):
        for tile in (gold, mana):
            assert max(abs(tile[0] - home[0]), abs(tile[1] - home[1])) <= 3
            assert tile not in env._static_obstacle_tiles
            assert tile not in env.water_tiles
            assert tile not in env._map.enemy_positions().values()


def test_all_armies_resources_and_other_capital_are_reachable_without_mandatory_fights(env):
    enemies = set(env._map.enemy_positions().values())
    blocked = set(env._static_obstacle_tiles) | set(env.water_tiles)
    resources = set(env.gold_mine_tiles) | set(env.mana_sources)
    for home in (HERO_START, BOT_HOME):
        seen = _reachable(home, blocked)
        assert enemies | resources | {HERO_START, BOT_HOME} <= seen
        # Treat every neutral as an unpassable tile: both homes/resources and
        # a neighboring attack tile for every encounter must still be reachable.
        peaceful = _reachable(home, blocked | enemies)
        assert resources | {HERO_START, BOT_HOME} <= peaceful
        for x, y in enemies:
            assert {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)} & peaceful


def test_runtime_armies_preserve_exact_fighters_hp_xp_and_large_cells(env):
    hero = deepcopy(env._resolve_travel_hero())
    hero.update(initiative=10000, initiative_base=10000)
    unit_data = {row["кто"]: row for row in DATA if row.get("кто")}
    for row in env._map.enemy_stacks:
        enemy_id = row["enemy_id"]
        expected_names = Counter(name for name in (*row["front"], *row["back"]) if name)
        expected_hp = sum(unit_data[name]["здоровье"] * count for name, count in expected_names.items())
        expected_xp = sum(unit_data[name]["опыт убийства"] * count for name, count in expected_names.items())
        for team in (env._enemy_configs[enemy_id], env.enemy_team_states[enemy_id]):
            units = _fighters(team)
            assert Counter(u["name"] for u in units) == expected_names, enemy_id
            assert sum(u["max_health"] for u in units) == expected_hp, enemy_id
            assert sum(u["exp_kill"] for u in units) == expected_xp, enemy_id
            _assert_tactical_cells(units)
        env._init_battle(enemy_id, blue_team=[hero])
        units = [u for u in _fighters(env.battle_env.combined) if u["team"] == "red"]
        assert Counter(u["name"] for u in units) == expected_names, enemy_id
        assert sum(u["max_health"] for u in units) == expected_hp, enemy_id
        assert sum(u["exp_kill"] for u in units) == expected_xp, enemy_id
        _assert_tactical_cells(units)


def test_large_creatures_are_single_fighters_and_support_uses_a_free_rear_cell(env):
    for row in env._map.enemy_stacks:
        units = _fighters(env.enemy_team_states[row["enemy_id"]])
        if row["formation_key"] in ("kraken", "manticore", "tiamat", "yeti"):
            assert len(units) == 1 and units[0]["big"]
        if row["formation_key"] == "giant_alchemist":
            assert len(units) == 2
            giant = next(u for u in units if u["big"])
            alchemist = next(u for u in units if u["name"] == "Алхимик")
            assert giant["position"] in (1, 2, 3)
            assert alchemist["position"] in (4, 5, 6)
            assert alchemist["position"] != giant["position"] + 3
        if row["formation_key"] == "angels":
            assert len(units) == 2
            assert units[0] is not units[1]


def test_two_same_named_large_fighters_are_never_deduplicated(env):
    """Reserve separate columns, preserving two distinct identical creatures."""
    spec = {999: {"description": "two distinct Manticores", "front": ["Мантикора", None, "Мантикора"],
                  "back": [None, None, None]}}
    team = build_enemy_configs(spec)[999]
    units = _fighters(team)
    assert len(units) == 2
    assert units[0] is not units[1]
    _assert_tactical_cells(units)
    expected = Counter((u["name"], u["max_health"], u["exp_kill"]) for u in units)
    enemy_id = TIAMAT_ENEMY_ID
    env.enemy_team_states[enemy_id] = deepcopy(team)
    hero = deepcopy(env._resolve_travel_hero())
    hero.update(initiative=10000, initiative_base=10000)
    env._init_battle(enemy_id, blue_team=[hero])
    actual = [u for u in _fighters(env.battle_env.combined) if u["team"] == "red"]
    assert Counter((u["name"], u["max_health"], u["exp_kill"]) for u in actual) == expected
    _assert_tactical_cells(actual)


def test_both_capitals_spread_territory_without_an_extra_home_guardian(env, monkeypatch):
    assert env._map.empire_territory_source_enemy_id is None
    assert env.empire_territory_enabled
    assert env.legions_territory_source_tile == HERO_START
    assert env.empire_territory_source_tile == BOT_HOME
    assert set(env._static_enemy_positions) == set(env._map.enemy_positions())
    phases = []
    original = env._apply_territory_growth_phase

    def observe(owner, tiles):
        tiles = tuple(tiles)
        original(owner, tiles)
        phases.append((owner, set(tiles), env.territory_owner_grid.copy()))

    monkeypatch.setattr(env, "_apply_territory_growth_phase", observe)
    monkeypatch.setattr(env, "_advance_scripted_capital_bot_one_turn", lambda: {})
    env._advance_turns(1)
    assert [phase[0] for phase in phases] == [env.TERRITORY_AGENT, env.TERRITORY_BOT]
    assert len(env.legions_territory_tile_set) > 1
    assert len(env.empire_territory_tile_set) > 1
    assert AGENT_GOLD_MINE_TILE in env.legions_territory_tile_set
    assert BOT_GOLD_MINE_TILE in env.empire_territory_tile_set
    assert AGENT_MANA_TILE in env.legions_territory_tile_set
    assert BOT_MANA_TILE in env.empire_territory_tile_set
    assert not env.legions_territory_tile_set & env.empire_territory_tile_set
    np.testing.assert_array_equal(phases[-1][2], env.territory_owner_grid)


def test_w7_repaints_the_whole_shared_reach_agent_then_bot_each_late_turn(env, monkeypatch):
    monkeypatch.setattr(env, "_advance_scripted_capital_bot_one_turn", lambda: {})
    common_reach = set(env.legions_territory_order)
    assert common_reach == set(env.empire_territory_order)
    # Advance the day counter to saturated reach without replaying hundreds
    # of unrelated economy ticks; the next two normal ticks must repaint all.
    env.turns = max(len(env.legions_territory_order), len(env.empire_territory_order))
    phases = []
    original = env._apply_territory_growth_phase

    def observe(owner, tiles):
        tiles = tuple(tiles)
        original(owner, tiles)
        phases.append((owner, set(tiles), env.territory_owner_grid.copy()))

    monkeypatch.setattr(env, "_apply_territory_growth_phase", observe)
    env._advance_turns(2)
    assert [p[0] for p in phases] == [env.TERRITORY_AGENT, env.TERRITORY_BOT] * 2
    for owner, painted, grid in phases:
        assert painted == common_reach
        assert all(grid[y, x] == owner for x, y in common_reach)
    assert not env.legions_territory_tile_set
    assert env.empire_territory_tile_set == common_reach


def test_default_parties_are_ordinary_legions_against_ordinary_empire(env):
    assert env.Realcapital == 2
    assert not env.use_boss_starting_roster
    assert env._map.starting_capital_id is None
    assert env._map.starting_roster is None
    assert {u["position"]: u["name"] for u in _fighters(env.blue_team_state)} == {
        7: "Одержимый", 8: "Герцог", 9: "Одержимый", 11: "Сектант",
    }
    assert env.scripted_capital_bot_enabled
    assert env.scripted_capital_bot_faction == "Империя"
    assert {u["position"]: u["name"] for u in _fighters(env.scripted_capital_bot_team_state)} == {
        7: "Скваер", 8: "Рыцарь на пегасе", 9: "Скваер", 11: "Служка",
    }
