import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS


def test_campaign_grid_obs_includes_campaign_state_features():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    obs, _ = env.reset(seed=123)

    assert env.grid_campaign_obs_size > 0
    assert (
        env.GRID_OBS_SIZE
        == env.grid_obs_base_size + env.grid_enemy_obs_size + env.grid_campaign_obs_size
    )

    grid_obs = obs[1 : 1 + env.GRID_OBS_SIZE]
    campaign_obs = grid_obs[env.grid_obs_base_size + env.grid_enemy_obs_size :]
    assert campaign_obs.shape == (env.grid_campaign_obs_size,)

    baseline = campaign_obs.copy()

    unit = next(
        entry for entry in env.blue_team_state if int(entry.get("position", -1)) == 7
    )
    current_hp = float(unit.get("hp", unit.get("health", 0)) or 0.0)
    unit["hp"] = max(1.0, current_hp - 10.0)
    unit["health"] = unit["hp"]

    env.moves = 0
    env.heroitems.append(env.INVULNERABILITY_POTION_ITEM_NAME)
    env.active_invulnerability_potion_positions.add(7)

    build_key = env.building_keys[0]
    env.active_buildings[build_key]["built"] = 1

    chest_pos = next(iter(sorted(env.chests)))
    env.chests.pop(chest_pos)

    objective_name = env.grid_objective_city_names[0]
    env.captured_objective_cities.add(objective_name)

    updated = env._build_campaign_grid_obs()
    assert np.array_equal(updated, baseline) is False


def test_campaign_enemy_grid_obs_uses_one_hot_type_encoding():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    enemy_obs = env._build_enemy_grid_obs()
    slot_size = env.grid_enemy_unit_feature_size
    enemy_block_size = 2 + env.grid_enemy_unit_slots * slot_size

    assert env.grid_enemy_unit_type_feature_size > 1

    for enemy_index, enemy_id in enumerate(sorted(env.grid_env.enemy_positions.keys())):
        team = sorted(
            ENEMY_CONFIGS.get(enemy_id, []),
            key=lambda unit: int(unit.get("position", 0) or 0),
        )
        for unit_slot, unit in enumerate(team[: env.grid_enemy_unit_slots]):
            if env._is_empty_enemy_unit(unit):
                continue

            offset = enemy_index * enemy_block_size + 2 + unit_slot * slot_size
            unit_slice = enemy_obs[offset : offset + slot_size]
            type_slice = unit_slice[: env.grid_enemy_unit_type_feature_size]

            assert type_slice.sum() == pytest.approx(1.0)
            assert set(np.unique(type_slice)).issubset({0.0, 1.0})
            assert 0.0 <= float(unit_slice[-1]) <= 1.0
            return

    pytest.fail("expected at least one non-empty enemy unit in ENEMY_CONFIGS")


def test_campaign_economy_uses_higher_turn_income_and_scaled_building_costs():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _, info = env.reset(seed=123)

    assert info["gold"] == pytest.approx(0.0)
    assert env.GOLD_PER_TURN == pytest.approx(100.0)
    assert env.gold_norm_k == pytest.approx(env.turn_norm_k * env.GOLD_PER_TURN)

    turn_penalty = env._advance_turns(1)
    assert turn_penalty == pytest.approx(-env.reward_turn_penalty)
    assert env.gold == pytest.approx(100.0)

    assert env.active_buildings["lod_d2_b001"]["gold"] == pytest.approx(200.0)
    assert env.active_buildings["lod_d2_b004"]["gold"] == pytest.approx(1500.0)
    assert env.active_buildings["lod_d2_b020"]["gold"] == pytest.approx(4500.0)
    assert env.grid_max_build_price == pytest.approx(4500.0)


def test_legions_captured_gold_mines_add_bonus_gold_each_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _, info = env.reset(seed=123)

    assert info["legions_captured_gold_mine_count"] == 0
    assert info["legions_gold_mine_income_per_turn"] == pytest.approx(0.0)

    env._advance_turns(3)
    assert env.gold == pytest.approx(300.0)
    assert env.legions_captured_gold_mine_count == 0
    assert (2, 31) not in env.legions_territory_tile_set

    env._advance_turns(1)
    assert (2, 31) in env.legions_territory_tile_set
    assert env.legions_captured_gold_mine_tiles == ((2, 31),)
    assert env.legions_captured_gold_mine_count == 1
    assert env._legions_gold_income_per_turn() == pytest.approx(150.0)
    assert env.gold == pytest.approx(450.0)

    env._advance_turns(1)
    assert env.gold == pytest.approx(600.0)


def test_merchant_sell_values_remain_unchanged():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    assert env._hero_item_gold_value_by_name("Ruby (Valuable)") == 1250
    assert env._hero_item_gold_value_by_name("Banner of War") == 5000
    assert env._hero_item_gold_value_by_name("Bronze Ring (Valuable)") == 250


def test_legions_territory_expands_by_ten_tiles_per_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _, info = env.reset(seed=123)

    assert tuple(env.legions_territory_tiles) == (tuple(env.grid_env.start_position),)
    assert info["legions_territory_count"] == 1
    assert tuple(env.grid_env.start_position) in set(info["legions_territory_tiles"])
    assert tuple(env.grid_env.start_position) not in env.legions_territory_forbidden_tile_set

    env._advance_turns(1)
    assert len(env.legions_territory_tiles) == 1 + env.LEGIONS_TERRITORY_EXPANSION_PER_TURN
    assert tuple(env.grid_env.start_position) in env.legions_territory_tile_set

    env._advance_turns(2)
    assert len(env.legions_territory_tiles) == 1 + 3 * env.LEGIONS_TERRITORY_EXPANSION_PER_TURN
    assert env.legions_territory_tile_set.isdisjoint(env.legions_territory_forbidden_tile_set)


def test_legions_territory_grid_obs_marks_claimed_tiles_and_updates_with_turns():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    territory_obs = env._build_legions_territory_grid_obs()
    assert territory_obs.shape == (env.grid_legions_territory_obs_size,)
    assert env.grid_legions_territory_obs_size == env.grid_size**2
    assert float(territory_obs.sum()) == pytest.approx(float(len(env.legions_territory_tiles)))

    start_index = env.grid_legions_territory_positions.index(tuple(env.grid_env.start_position))
    assert territory_obs[start_index] == pytest.approx(1.0)

    baseline = territory_obs.copy()
    env._advance_turns(1)
    updated = env._build_legions_territory_grid_obs()

    assert np.array_equal(updated, baseline) is False
    assert float(updated.sum()) == pytest.approx(float(len(env.legions_territory_tiles)))


def test_empire_territory_expands_by_ten_tiles_per_turn_after_legions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _, info = env.reset(seed=123)

    assert tuple(env.empire_territory_tiles) == (tuple(env.empire_territory_source_tile),)
    assert info["empire_territory_count"] == 1
    assert tuple(env.empire_territory_source_tile) in set(info["empire_territory_tiles"])
    assert tuple(env.empire_territory_source_tile) not in env.empire_territory_forbidden_tile_set

    env._advance_turns(1)
    assert len(env.empire_territory_tiles) == 1 + env.EMPIRE_TERRITORY_EXPANSION_PER_TURN
    assert tuple(env.empire_territory_source_tile) in env.empire_territory_tile_set

    env._advance_turns(2)
    assert len(env.empire_territory_tiles) == 1 + 3 * env.EMPIRE_TERRITORY_EXPANSION_PER_TURN
    assert env.empire_territory_tile_set.isdisjoint(env.empire_territory_forbidden_tile_set)


def test_empire_territory_grid_obs_marks_claimed_tiles_and_updates_with_turns():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    territory_obs = env._build_empire_territory_grid_obs()
    assert territory_obs.shape == (env.grid_empire_territory_obs_size,)
    assert env.grid_empire_territory_obs_size == env.grid_size**2
    assert float(territory_obs.sum()) == pytest.approx(float(len(env.empire_territory_tiles)))

    source_index = env.grid_empire_territory_positions.index(tuple(env.empire_territory_source_tile))
    assert territory_obs[source_index] == pytest.approx(1.0)

    baseline = territory_obs.copy()
    env._advance_turns(1)
    updated = env._build_empire_territory_grid_obs()

    assert np.array_equal(updated, baseline) is False
    assert float(updated.sum()) == pytest.approx(float(len(env.empire_territory_tiles)))


def test_empire_territory_is_excluded_from_campaign_grid_obs():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    baseline = env._build_campaign_grid_obs().copy()

    env.empire_territory_tiles = ((0, 0), (1, 0), (2, 0))
    env.empire_territory_tile_set = set(env.empire_territory_tiles)

    updated = env._build_campaign_grid_obs()

    assert np.array_equal(updated, baseline)


def test_legions_and_empire_territories_only_claim_reachable_regions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    env._advance_turns(999)

    assert tuple(env.grid_env.start_position) in env.legions_territory_tile_set
    assert tuple(env.empire_territory_source_tile) in env.empire_territory_tile_set
    assert tuple(env.grid_env.start_position) not in env.empire_territory_tile_set
    assert tuple(env.empire_territory_source_tile) not in env.legions_territory_tile_set
    assert tuple(env.grid_env.start_position) not in env.empire_territory_path_distances
    assert tuple(env.empire_territory_source_tile) not in env.legions_territory_path_distances
    assert not (env.legions_territory_tile_set & env.empire_territory_tile_set)


def test_legions_territory_skips_capital_footprint_water_and_other_objects():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    # Nearest capital footprint tiles around the start stay excluded.
    assert (4, 27) in env.legions_territory_forbidden_tile_set
    assert (4, 27) not in env.legions_territory_tile_set

    # Water/object tiles from the scenario are never claimed.
    assert (0, 32) in env.legions_territory_forbidden_tile_set

    env._advance_turns(20)

    assert (4, 27) not in env.legions_territory_tile_set
    assert (0, 32) not in env.legions_territory_tile_set
    assert env.legions_territory_tile_set.isdisjoint(env.legions_territory_forbidden_tile_set)


def test_legions_territory_keeps_enemy_cells_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    enemy_tiles = {tuple(pos) for pos in env.grid_env.enemy_positions.values()}
    assert env.legions_territory_forbidden_tile_set.isdisjoint(enemy_tiles)


def test_legions_territory_keeps_gold_mines_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    gold_mine_tiles = {(2, 31), (24, 4), (29, 46), (41, 46)}
    assert env.legions_territory_forbidden_tile_set.isdisjoint(gold_mine_tiles)

    env._advance_turns(999)
    reachable_gold_mines = {
        tile for tile in gold_mine_tiles if tile in env.legions_territory_path_distances
    }
    unreachable_gold_mines = gold_mine_tiles - reachable_gold_mines
    assert reachable_gold_mines.issubset(env.legions_territory_tile_set)
    assert env.legions_territory_tile_set.isdisjoint(unreachable_gold_mines)


def test_legions_territory_keeps_mana_crystals_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    mana_crystal_tiles = {
        (10, 29),  # red mana
        (7, 41),   # orange mana
        (6, 16),   # red mana
        (6, 11),   # yellow mana
    }
    assert env.legions_territory_forbidden_tile_set.isdisjoint(mana_crystal_tiles)

    env._advance_turns(999)
    reachable_mana_crystals = {
        tile for tile in mana_crystal_tiles if tile in env.legions_territory_path_distances
    }
    unreachable_mana_crystals = mana_crystal_tiles - reachable_mana_crystals
    assert reachable_mana_crystals.issubset(env.legions_territory_tile_set)
    assert env.legions_territory_tile_set.isdisjoint(unreachable_mana_crystals)


def test_legions_territory_keeps_chest_tiles_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    chest_tiles = set(env.chests.keys())
    assert env.legions_territory_forbidden_tile_set.isdisjoint(chest_tiles)

    env._advance_turns(999)
    reachable_chests = {
        tile for tile in chest_tiles if tile in env.legions_territory_path_distances
    }
    unreachable_chests = chest_tiles - reachable_chests
    assert reachable_chests.issubset(env.legions_territory_tile_set)
    assert env.legions_territory_tile_set.isdisjoint(unreachable_chests)


def test_legions_territory_keeps_landmarks_and_locations_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    passable_object_tiles = {
        (4, 23),  # landmark near the capital
        (8, 22),  # landmark near the capital
        (8, 23),  # location near the capital
        (1, 30),  # landmark south-west of the capital
        (7, 31),  # landmark south-east of the capital
    }
    assert env.legions_territory_forbidden_tile_set.isdisjoint(passable_object_tiles)


def test_territories_keep_road_tiles_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    road_tiles = {
        (27, 13),
        (27, 14),
        (24, 15),
        (25, 15),
        (26, 15),
        (27, 15),
        (24, 16),
    }
    assert env.legions_territory_forbidden_tile_set.isdisjoint(road_tiles)
    assert env.empire_territory_forbidden_tile_set.isdisjoint(road_tiles)

    env._advance_turns(999)

    assert road_tiles.issubset(env.empire_territory_tile_set)
