"""Map armies must preserve fighters, not turn large-unit cells into fighters."""

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

from campaign_env import CampaignEnv
from maps import available_maps, get_map
from scenario_map_data import _int_field, _object_chunks


def fighters(team):
    return [unit for unit in team
            if unit["team"] == "red" and unit["name"] != "пусто"]


def assert_valid_cells(units, context):
    occupied = set()
    for unit in units:
        position = unit["position"]
        assert position in range(1, 7), (context, unit)
        cells = {position}
        if unit["big"]:
            assert position in (1, 2, 3), (context, unit)
            cells.add(position + 3)
        assert not (occupied & cells), (context, unit)
        occupied.update(cells)
    assert len(occupied) <= 6, context


@pytest.mark.parametrize("map_name", available_maps())
def test_all_map_armies_preserve_fighters_on_battle_initialization(map_name):
    env = CampaignEnv(
        map_name=map_name, observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    try:
        env.reset(seed=42)
        hero = deepcopy(env._resolve_travel_hero())
        # Stop before enemy attacks, transformations or summons can change teams.
        hero.update(initiative=10000, initiative_base=10000)
        assert set(env._enemy_configs) == set(env._map.enemy_team_specs())
        for enemy_id, configured in env._enemy_configs.items():
            context = (map_name, enemy_id)
            expected = fighters(configured)
            expected_names = Counter(unit["name"] for unit in expected)
            assert_valid_cells(expected, context)
            state = fighters(env.enemy_team_states[enemy_id])
            assert Counter(unit["name"] for unit in state) == expected_names, context
            assert_valid_cells(state, context)
            # Per-map level overrides can legitimately change HP/XP at reset.
            expected_stats = Counter(
                (u["name"], u["max_health"], u["exp_kill"]) for u in state
            )
            env._init_battle(enemy_id, blue_team=[hero])
            assert env.battle_env.current_blue_attacker_pos == hero["position"], context
            actual = fighters(env.battle_env.combined)
            assert Counter((u["name"], u["max_health"], u["exp_kill"])
                           for u in actual) == expected_stats, context
            assert_valid_cells(actual, context)
    finally:
        env.close()


def source_string(chunk, field):
    start = chunk.index(field) + len(field) + 4
    return chunk[start:start + _int_field(chunk, field)].rstrip(b"\0").decode("ascii")


@pytest.fixture(scope="module")
def default_source_groups():
    """Read unique unit IDs independently from the original .sg formation cells."""
    data = (Path(__file__).resolve().parents[1] / "A Return To Simpler Times.sg").read_bytes()
    groups = defaultdict(list)
    settlements = {}
    classes = (
        (b".?AVCCapital@@", (4, 2), "capital"),
        (b".?AVCMidVillage@@", (3, 3), "garrison"),
        (b".?AVCMidRuin@@", (2, 2), "ruin"),
        (b".?AVCMidStack@@", (0, 0), "stack"),
    )
    for class_name, offset, kind in classes:
        for chunk in _object_chunks(data, class_name):
            source_id = source_string(chunk, b"GROUP_ID")
            position = (_int_field(chunk, b"POS_X") + offset[0],
                        _int_field(chunk, b"POS_Y") + offset[1])
            if kind in ("capital", "garrison"):
                settlements[source_id] = position
            elif kind == "stack":
                position = settlements.get(source_string(chunk, b"INSIDE"), position)
            ids = [source_string(chunk, f"UNIT_{i}".encode()) for i in range(6)]
            cells = Counter()
            for i in range(6):
                index = _int_field(chunk, f"POS_{i}".encode())
                if index == 0xFFFFFFFF:
                    continue
                assert 0 <= index < 6, (source_id, index)
                assert ids[index] != "G000000000", source_id
                cells[ids[index]] += 1
            assert all(count in (1, 2) for count in cells.values()), source_id
            groups[(position, kind)].append((source_id, cells))
    return groups


@pytest.mark.parametrize("map_name", ["default", "small"])
def test_default_and_small_match_unique_source_fighter_counts(map_name, default_source_groups):
    from enemy_configs import build_enemy_configs

    config = get_map(map_name)
    teams = build_enemy_configs(config.enemy_team_specs())
    for enemy_id, position in config.enemy_positions().items():
        if enemy_id == 75:
            # These maps explicitly retain only Mizrael from the Empire capital.
            assert [u["name"] for u in fighters(teams[enemy_id])] == ["Мизраэль"]
            continue
        kind = ("garrison" if enemy_id in (67, 68, 69)
                else "ruin" if enemy_id in range(70, 75) else "stack")
        matches = default_source_groups[(position, kind)]
        assert len(matches) == 1, (map_name, enemy_id, matches)
        source_id, cells = matches[0]
        units = fighters(teams[enemy_id])
        assert len(units) == len(cells), (map_name, enemy_id, source_id)
        assert sum(bool(u["big"]) for u in units) == sum(n == 2 for n in cells.values()), (
            map_name, enemy_id, source_id,
        )
