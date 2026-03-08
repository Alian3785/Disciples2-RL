import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _castle_heal_action_slice(env: CampaignEnv):
    start = env.GRID_CASTLE_HEAL_ACTION_START
    stop = start + len(env.CASTLE_HEAL_POSITIONS)
    return slice(start, stop)


def _castle_revive_action_slice(env: CampaignEnv):
    start = env.GRID_CASTLE_REVIVE_ACTION_START
    stop = start + len(env.CASTLE_REVIVE_POSITIONS)
    return slice(start, stop)


def _pairwise_manhattan_distances(tiles):
    distances = []
    for index, (x1, y1) in enumerate(tiles):
        for x2, y2 in tiles[index + 1:]:
            distances.append(abs(x1 - x2) + abs(y1 - y2))
    return distances


def test_hero_starts_at_1_1_and_returns_there_on_reset():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    assert env.grid_env.agent_pos == (1, 1)

    _, info = env.reset(seed=123)
    assert env.grid_env.agent_pos == (1, 1)
    assert tuple(info["agent_pos"]) == (1, 1)


def test_heal_tiles_are_far_apart_and_do_not_overlap_enemies():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    heal_tiles = tuple(env.castle_heal_tiles)
    assert len(heal_tiles) == 3
    assert set(heal_tiles) == {(1, 1), (38, 1), (1, 38)}

    pairwise_distances = _pairwise_manhattan_distances(heal_tiles)
    assert min(pairwise_distances) >= env.grid_size - 3

    enemy_tiles = set(env.grid_env.enemy_positions.values())
    for tile in heal_tiles:
        assert tile not in enemy_tiles


def test_castle_heal_actions_available_on_all_heal_tiles():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    env.gold = 10.0

    # Wound one unit so castle-heal actions become valid in the mask.
    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 7)
    current_hp = float(unit.get("health", unit.get("hp", 0)))
    unit["hp"] = max(1.0, current_hp - 1.0)
    unit["health"] = unit["hp"]

    castle_slice = _castle_heal_action_slice(env)

    for tile in env.castle_heal_tiles:
        env.grid_env.agent_pos = tile
        mask = env.compute_action_mask()
        assert mask[castle_slice].any()


def test_castle_revive_actions_available_on_all_heal_tiles_when_gold_is_enough():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    env.gold = 50.0

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == 7)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = 1

    castle_revive_slice = _castle_revive_action_slice(env)

    for tile in env.castle_heal_tiles:
        env.grid_env.agent_pos = tile
        mask = env.compute_action_mask()
        assert mask[castle_revive_slice].any()


def test_castle_heal_spends_only_required_gold_when_enough():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 12.0
    unit["health"] = unit["hp"]
    unit["Level"] = 1

    env.gold = 50.0
    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0) == pytest.approx(max_hp)
    assert env.gold == pytest.approx(38.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(12.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(12.0)


def test_castle_heal_level_2_or_3_costs_two_gold_per_hp():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 20.0
    unit["health"] = unit["hp"]
    unit["Level"] = 3

    env.gold = 50.0
    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(max_hp)
    assert float(unit.get("health", 0) or 0) == pytest.approx(max_hp)
    assert env.gold == pytest.approx(10.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(20.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(40.0)


def test_castle_heal_level_4_or_5_costs_three_gold_per_hp_and_can_be_partial():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_HEAL_ACTION_START + env.CASTLE_HEAL_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))
    unit["hp"] = max_hp - 20.0
    unit["health"] = unit["hp"]
    unit["Level"] = 5

    env.gold = 5.5
    _, _, _, _, info = env.step(action)

    expected_healed = 5.5 / 3.0
    assert float(unit.get("hp", 0) or 0) == pytest.approx((max_hp - 20.0) + expected_healed)
    assert float(unit.get("health", 0) or 0) == pytest.approx((max_hp - 20.0) + expected_healed)
    assert env.gold == pytest.approx(0.0)
    assert float(info.get("healed_amount", 0.0)) == pytest.approx(expected_healed)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(5.5)


@pytest.mark.parametrize(
    ("level", "revive_cost"),
    [
        (1, 50.0),
        (2, 200.0),
        (3, 400.0),
        (4, 600.0),
        (5, 800.0),
    ],
)
def test_castle_revive_spends_gold_by_unit_level(level, revive_cost):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_REVIVE_ACTION_START + env.CASTLE_REVIVE_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = level

    env.gold = revive_cost + 25.0
    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(1.0)
    assert float(unit.get("health", 0) or 0) == pytest.approx(1.0)
    assert env.gold == pytest.approx(25.0)
    assert info.get("revived") is True
    assert float(info.get("revive_cost", 0.0)) == pytest.approx(revive_cost)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(revive_cost)


def test_castle_revive_action_is_blocked_when_gold_is_below_required_cost():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = env.GRID_CASTLE_REVIVE_ACTION_START + env.CASTLE_REVIVE_POSITIONS.index(target_pos)
    env.grid_env.agent_pos = env.castle_heal_tiles[0]

    unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == target_pos)
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["Level"] = 4

    env.gold = 599.0
    mask = env.compute_action_mask()
    assert bool(mask[action]) is False

    _, _, _, _, info = env.step(action)

    assert float(unit.get("hp", 0) or 0) == pytest.approx(0.0)
    assert float(unit.get("health", 0) or 0) == pytest.approx(0.0)
    assert env.gold == pytest.approx(599.0)
    assert info.get("revived") is False
    assert float(info.get("revive_cost", 0.0)) == pytest.approx(600.0)
    assert float(info.get("gold_spent", 0.0)) == pytest.approx(0.0)
