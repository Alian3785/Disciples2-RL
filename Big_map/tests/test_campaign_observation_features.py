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
