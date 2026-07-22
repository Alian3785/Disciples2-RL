import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS


def _refresh_territories_at_turn(env: CampaignEnv, turns: int) -> None:
    """Jump to a turn and refresh only the territory state under test."""
    env.turns = int(turns)
    env._refresh_faction_territories()


def test_campaign_grid_obs_includes_campaign_state_features():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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


def test_campaign_obs_includes_mercenary_site_positions_and_stock_state():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    site_obs = env._build_mercenary_site_grid_obs()
    grid_scale = max(1, env.grid_env.grid_size - 1)
    feature_count = env.grid_mercenary_site_features_per_entry

    assert site_obs.shape == (env.grid_mercenary_site_obs_size,)
    assert env.grid_mercenary_site_obs_size == len(env.grid_mercenary_site_names) * feature_count

    barbarian_site_index = env.grid_mercenary_site_names.index("Лагерь северных варваров")
    offset = barbarian_site_index * feature_count
    barbarian_anchor = env.mercenary_site_anchors["Лагерь северных варваров"]

    assert site_obs[offset] == pytest.approx(barbarian_anchor[0] / grid_scale)
    assert site_obs[offset + 1] == pytest.approx(barbarian_anchor[1] / grid_scale)
    assert site_obs[offset + 2] == pytest.approx(0.0)
    assert site_obs[offset + 3] == pytest.approx(1.0)
    assert site_obs[offset + 4] == pytest.approx(1.0)

    env.mercenary_site_rosters["Лагерь северных варваров"][0]["stock"] = 0
    depleted_site_obs = env._build_mercenary_site_grid_obs()
    assert depleted_site_obs[offset + 3] == pytest.approx(0.0)
    assert depleted_site_obs[offset + 4] == pytest.approx(0.0)


def test_campaign_obs_includes_current_mercenary_roster_and_hireability():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.grid_env.agent_pos = (18, 2)

    hero = next(unit for unit in env.blue_team_state if int(unit.get("position", -1)) == 8)
    hero["Level"] = 3
    env.gold = 999.0

    for position in (7, 10):
        unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == position)
        unit["name"] = "пусто"
        unit["hp"] = 0.0
        unit["health"] = 0.0
        unit["maxhp"] = 0.0
        unit["max_health"] = 0.0

    roster_obs = env._build_current_mercenary_roster_grid_obs()
    feature_count = env.grid_mercenary_slot_features_per_entry

    assert roster_obs.shape == (env.grid_mercenary_roster_obs_size,)
    assert env.grid_mercenary_slot_count == 2

    assert roster_obs[0] == pytest.approx(1.0)
    assert roster_obs[1] == pytest.approx(1.0)
    assert roster_obs[2] == pytest.approx(1.0)
    assert roster_obs[3] == pytest.approx(0.0)
    assert roster_obs[7] == pytest.approx(1.0)

    second_offset = feature_count
    assert roster_obs[second_offset] == pytest.approx(1.0)
    assert roster_obs[second_offset + 1] == pytest.approx(1.0)
    assert roster_obs[second_offset + 2] == pytest.approx(0.0)
    assert roster_obs[second_offset + 3] == pytest.approx(1.0)
    assert roster_obs[second_offset + 7] == pytest.approx(1.0)

    env.grid_env.agent_pos = (0, 0)
    assert np.allclose(env._build_current_mercenary_roster_grid_obs(), 0.0)


def test_campaign_obs_includes_merchant_site_positions_and_stock_state():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    site_obs = env._build_merchant_site_grid_obs()
    grid_scale = max(1, env.grid_env.grid_size - 1)
    feature_count = env.grid_merchant_site_features_per_entry

    assert site_obs.shape == (env.grid_merchant_site_obs_size,)
    assert env.grid_merchant_site_obs_size == len(env.grid_merchant_site_names) * feature_count

    stocked_site_name = env.grid_merchant_site_names[0]
    empty_site_name = env.grid_merchant_site_names[1]

    stocked_anchor = env.merchant_site_anchors[stocked_site_name]
    empty_anchor = env.merchant_site_anchors[empty_site_name]

    assert site_obs[0] == pytest.approx(stocked_anchor[0] / grid_scale)
    assert site_obs[1] == pytest.approx(stocked_anchor[1] / grid_scale)
    assert site_obs[2] == pytest.approx(0.0)
    assert site_obs[3] == pytest.approx(1.0)
    assert site_obs[4] == pytest.approx(1.0)

    second_offset = feature_count
    assert site_obs[second_offset] == pytest.approx(empty_anchor[0] / grid_scale)
    assert site_obs[second_offset + 1] == pytest.approx(empty_anchor[1] / grid_scale)
    assert site_obs[second_offset + 2] == pytest.approx(0.0)
    assert site_obs[second_offset + 3] == pytest.approx(0.0)
    assert site_obs[second_offset + 4] == pytest.approx(0.0)

    for item_name in env.grid_merchant_item_names:
        env.merchant_stocks[stocked_site_name][item_name] = 0
    depleted_site_obs = env._build_merchant_site_grid_obs()
    assert depleted_site_obs[3] == pytest.approx(0.0)
    assert depleted_site_obs[4] == pytest.approx(0.0)


def test_campaign_obs_includes_current_merchant_stock_ratios():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    stocked_site_name = env.grid_merchant_site_names[0]
    empty_site_name = env.grid_merchant_site_names[1]
    stocked_tile = tuple(env.merchant_site_interaction_tiles[stocked_site_name][0])
    empty_tile = tuple(env.merchant_site_interaction_tiles[empty_site_name][0])

    env.grid_env.agent_pos = stocked_tile
    stock_obs = env._build_current_merchant_stock_grid_obs()
    assert stock_obs.shape == (env.grid_current_merchant_stock_obs_size,)
    assert env.grid_current_merchant_stock_obs_size == len(env.grid_merchant_item_names)
    assert np.allclose(stock_obs, 1.0)

    first_item_name = env.grid_merchant_item_names[0]
    env.merchant_stocks[stocked_site_name][first_item_name] = 5
    updated_stock_obs = env._build_current_merchant_stock_grid_obs()
    assert updated_stock_obs[0] == pytest.approx(0.5)

    env.grid_env.agent_pos = empty_tile
    empty_stock_obs = env._build_current_merchant_stock_grid_obs()
    assert np.allclose(empty_stock_obs, 0.0)


def test_campaign_obs_includes_compact_trainer_features():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    env.gold = 100.0

    for position in env.TRAINER_POSITIONS:
        unit = next(u for u in env.blue_team_state if int(u.get("position", -1)) == int(position))
        if int(position) == 7:
            unit["hero"] = 0
            unit["capital"] = env.Realcapital
            unit["is_neutral_unit"] = False
            unit["Level"] = 1
            unit["exp_current"] = 10
            unit["exp_required"] = 20
            unit["next_level_exp"] = 0
            unit["hp"] = max(1.0, float(unit.get("hp", unit.get("health", 1.0)) or 1.0))
            unit["health"] = unit["hp"]
            unit["maxhp"] = max(
                float(unit.get("maxhp", unit.get("max_health", unit["hp"])) or unit["hp"]),
                unit["hp"],
            )
            unit["max_health"] = unit["maxhp"]
        else:
            unit["hp"] = 0.0
            unit["health"] = 0.0

    trainer_obs = env._build_trainer_grid_obs()

    assert trainer_obs.shape == (env.grid_trainer_obs_size,)
    assert env.grid_trainer_obs_size == 5

    current_pos = tuple(int(coord) for coord in tuple(env.grid_env.agent_pos)[:2])
    nearest_tile = min(
        env.trainer_interaction_tiles,
        key=lambda tile: (
            abs(int(tile[0]) - int(current_pos[0])) + abs(int(tile[1]) - int(current_pos[1])),
            int(tile[1]),
            int(tile[0]),
        ),
    )
    grid_scale = max(1, env.grid_env.grid_size - 1)
    expected_dx = np.clip(
        ((int(nearest_tile[0]) - int(current_pos[0])) / grid_scale + 1.0) * 0.5,
        0.0,
        1.0,
    )
    expected_dy = np.clip(
        ((int(nearest_tile[1]) - int(current_pos[1])) / grid_scale + 1.0) * 0.5,
        0.0,
        1.0,
    )

    assert trainer_obs[0] == pytest.approx(expected_dx)
    assert trainer_obs[1] == pytest.approx(expected_dy)
    assert trainer_obs[2] == pytest.approx(0.0)
    assert trainer_obs[3] == pytest.approx(1.0 / len(env.TRAINER_POSITIONS))
    assert trainer_obs[4] == pytest.approx(
        env._normalize_saturating_count(
            9,
            half_point=float(env.grid_trainer_total_affordable_xp_half_point),
        )
    )

    env.grid_env.agent_pos = (41, 31)
    onsite_obs = env._build_trainer_grid_obs()
    assert onsite_obs[2] == pytest.approx(1.0)

    campaign_obs = env._build_campaign_grid_obs()
    assert np.allclose(campaign_obs[-env.grid_trainer_obs_size :], onsite_obs)


def test_campaign_enemy_grid_obs_uses_one_hot_type_encoding():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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


@pytest.mark.parametrize("Realcapital", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    ("typeoflord", "expected_built"),
    [
        (1, set()),
        (2, {"Башня магии"}),
        (3, {"Гильдия"}),
    ],
)
def test_lord_type_applies_original_starting_building(Realcapital, typeoflord, expected_built):
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        Realcapital=Realcapital,
        typeoflord=typeoflord,
    )

    env.reset(seed=123)

    assert set(env.get_built_building_names()) == expected_built
    assert int(env.active_buildings.get("alredybuilt", 0) or 0) == 0


def test_reset_options_reapply_lord_starting_building_for_new_capital():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2, typeoflord=1)

    env.reset(seed=123, options={"Realcapital": 4, "typeoflord": 2})
    assert set(env.get_built_building_names()) == {"Башня магии"}

    env.reset(seed=456, options={"Realcapital": 5, "typeoflord": 3})
    assert set(env.get_built_building_names()) == {"Гильдия"}


def test_building_state_is_isolated_between_campaign_env_instances():
    env1 = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=1)
    env2 = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=1)
    env1.reset(seed=123)
    env2.reset(seed=456)

    build_key = env1.building_keys[0]
    original_cost = float(env1.active_buildings[build_key]["gold"])

    assert env1.active_buildings is not env2.active_buildings
    assert env1.active_buildings[build_key] is not env2.active_buildings[build_key]

    build_action = env1.GRID_BUILD_ACTION_START + env1.building_keys.index(build_key)
    env1.gold = env1._get_building_gold_cost(env1.active_buildings[build_key])
    _, _, terminated, truncated, info = env1.step(build_action)

    assert terminated is False
    assert truncated is False
    assert info["built"] is True
    assert int(env1.active_buildings[build_key]["built"]) == 1
    assert int(env2.active_buildings[build_key]["built"]) == 0

    env2.reset(seed=789)
    assert int(env1.active_buildings[build_key]["built"]) == 1
    assert env1.active_buildings[build_key]["gold"] == pytest.approx(original_cost)

    env1.reset(seed=321)
    assert env1.active_buildings[build_key]["gold"] == pytest.approx(original_cost)


def test_building_requires_block_mask_and_direct_step_until_prerequisite_is_built():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=1)
    env.reset(seed=123)

    dependent_key = next(
        key
        for key in env.building_keys
        if env._building_requirement_names(env.active_buildings.get(key))
    )
    dependent_building = env.active_buildings[dependent_key]
    prerequisite_name = env._building_requirement_names(dependent_building)[0]
    prerequisite_key = next(
        key
        for key in env.building_keys
        if str(env.active_buildings[key].get("name", "") or "").strip()
        == prerequisite_name
    )
    dependent_action = env.GRID_BUILD_ACTION_START + env.building_keys.index(dependent_key)

    env.gold = 100_000.0
    gold_before = float(env.gold)
    assert bool(env.compute_action_mask()[dependent_action]) is False

    _, _, terminated, truncated, info = env.step(dependent_action)

    assert terminated is False
    assert truncated is False
    assert info["build_key"] == dependent_key
    assert info["built"] is False
    assert info["build_requires"] == [prerequisite_name]
    assert info["build_missing_requirements"] == [prerequisite_name]
    assert info["build_requirements_met"] is False
    assert env.gold == pytest.approx(gold_before)
    assert int(dependent_building.get("built", 0) or 0) == 0

    env.active_buildings[prerequisite_key]["built"] = 1
    env.active_buildings["alredybuilt"] = 0
    assert bool(env.compute_action_mask()[dependent_action]) is True

    _, _, terminated, truncated, info = env.step(dependent_action)

    assert terminated is False
    assert truncated is False
    assert info["built"] is True
    assert info["build_missing_requirements"] == []
    assert info["build_requirements_met"] is True
    assert int(dependent_building.get("built", 0) or 0) == 1


@pytest.mark.parametrize("Realcapital", [1, 2, 3, 4, 5])
def test_typeoflord_three_uses_normal_building_costs_for_all_capitals(Realcapital: int):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=Realcapital)
    env.reset(seed=123)
    env.typeoflord = 3

    effective_costs = {}
    for build_key in env.building_keys:
        building = env.active_buildings.get(build_key)
        assert isinstance(building, dict)
        raw_cost = float(building.get("gold", 0.0) or 0.0)
        effective_costs[build_key] = env._get_building_gold_cost(building)
        assert effective_costs[build_key] == pytest.approx(raw_cost)

    first_build_key = env.building_keys[0]
    first_action = env.GRID_BUILD_ACTION_START + env.building_keys.index(first_build_key)
    first_cost = effective_costs[first_build_key]

    env.gold = max(0.0, float(first_cost) - 1.0)
    assert bool(env.compute_action_mask()[first_action]) is False

    env.gold = float(first_cost)
    assert bool(env.compute_action_mask()[first_action]) is True

    _, reward, terminated, truncated, info = env.step(first_action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info["build_action"] is True
    assert info["build_key"] == first_build_key
    assert info["built"] is True
    assert info["build_cost"] == pytest.approx(first_cost)
    assert env.gold == pytest.approx(0.0)

    building_obs = env._build_building_grid_obs()
    expected_max_cost = max(effective_costs.values())
    assert building_obs[2] == pytest.approx(first_cost / expected_max_cost)


def test_legions_captured_gold_mines_add_bonus_gold_each_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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


@pytest.mark.parametrize(
    ("Realcapital", "base_mana_kind", "base_attr"),
    [
        (1, "life", "life_mana"),
        (2, "infernal", "infernal_mana"),
        (3, "runes", "runes_mana"),
        (4, "death", "death_mana"),
        (5, "elves", "elven_mana"),
    ],
)
def test_mana_pools_start_at_zero_and_base_income_matches_capital(
    Realcapital: int,
    base_mana_kind: str,
    base_attr: str,
):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=Realcapital)
    _, info = env.reset(seed=123)

    assert env.infernal_mana == pytest.approx(0.0)
    assert env.life_mana == pytest.approx(0.0)
    assert env.death_mana == pytest.approx(0.0)
    assert env.runes_mana == pytest.approx(0.0)
    assert env.elven_mana == pytest.approx(0.0)
    assert info["mana_totals"]["infernal"] == pytest.approx(0.0)
    for mana_kind in env.MANA_KIND_ORDER:
        expected_income = 25.0 if mana_kind == base_mana_kind else 0.0
        assert info["mana_income_per_turn"][mana_kind] == pytest.approx(expected_income)

    env._advance_turns(1)

    for mana_kind in env.MANA_KIND_ORDER:
        attr_name = env.MANA_ATTR_BY_KIND[mana_kind]
        expected_total = 25.0 if attr_name == base_attr else 0.0
        assert getattr(env, attr_name) == pytest.approx(expected_total)


def test_captured_mana_sources_add_income_by_type_each_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    captured_tiles = {
        (6, 16),   # infernal
        (10, 29),  # infernal
        (25, 16),  # life
        (7, 41),   # death
        (1, 4),    # runes
    }
    env.legions_territory_tiles = tuple(sorted(set(env.legions_territory_tiles) | captured_tiles))
    env.legions_territory_tile_set = set(env.legions_territory_tiles)
    env._refresh_legions_captured_mana_source_state()

    assert env.legions_captured_mana_source_counts_by_kind["infernal"] == 2
    assert env.legions_captured_mana_source_counts_by_kind["life"] == 1
    assert env.legions_captured_mana_source_counts_by_kind["death"] == 1
    assert env.legions_captured_mana_source_counts_by_kind["runes"] == 1
    assert env.legions_captured_mana_source_counts_by_kind["elves"] == 0
    assert env.mana_income_per_turn["infernal"] == pytest.approx(125.0)
    assert env.mana_income_per_turn["life"] == pytest.approx(50.0)
    assert env.mana_income_per_turn["death"] == pytest.approx(50.0)
    assert env.mana_income_per_turn["runes"] == pytest.approx(50.0)
    assert env.mana_income_per_turn["elves"] == pytest.approx(0.0)

    env._apply_mana_income_for_turn()

    assert env.infernal_mana == pytest.approx(125.0)
    assert env.life_mana == pytest.approx(50.0)
    assert env.death_mana == pytest.approx(50.0)
    assert env.runes_mana == pytest.approx(50.0)
    assert env.elven_mana == pytest.approx(0.0)


def test_campaign_obs_includes_compact_mana_source_positions_types_and_totals():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env.grid_mana_source_obs_size == len(env.mana_sources) * 3
    assert env.grid_resource_obs_size == env.grid_resource_core_obs_size + len(env.MANA_KIND_ORDER)

    mana_source_obs = env._build_mana_source_grid_obs()
    assert mana_source_obs.shape == (env.grid_mana_source_obs_size,)

    grid_scale = max(1, env.grid_env.grid_size - 1)
    for index, entry in enumerate(env.grid_mana_source_entries):
        offset = index * env.grid_mana_source_features_per_entry
        expected_x = int(entry["pos"][0]) / grid_scale
        expected_y = int(entry["pos"][1]) / grid_scale
        expected_type = env.grid_mana_kind_to_norm[str(entry["kind"])]
        assert mana_source_obs[offset] == pytest.approx(expected_x)
        assert mana_source_obs[offset + 1] == pytest.approx(expected_y)
        assert mana_source_obs[offset + 2] == pytest.approx(expected_type)

    resource_obs = env._build_resource_grid_obs()
    mana_totals_slice = resource_obs[-len(env.MANA_KIND_ORDER) :]
    assert np.allclose(mana_totals_slice, 0.0)

    env._advance_turns(1)
    updated_resource_obs = env._build_resource_grid_obs()
    infernal_idx = env.grid_mana_total_kinds.index("infernal")
    infernal_feature = updated_resource_obs[env.grid_resource_core_obs_size + infernal_idx]
    assert infernal_feature > 0.0
    assert updated_resource_obs[env.grid_resource_core_obs_size + env.grid_mana_total_kinds.index("life")] == pytest.approx(0.0)
    assert updated_resource_obs[env.grid_resource_core_obs_size + env.grid_mana_total_kinds.index("death")] == pytest.approx(0.0)
    assert updated_resource_obs[env.grid_resource_core_obs_size + env.grid_mana_total_kinds.index("runes")] == pytest.approx(0.0)
    assert updated_resource_obs[env.grid_resource_core_obs_size + env.grid_mana_total_kinds.index("elves")] == pytest.approx(0.0)


def test_campaign_obs_includes_spell_learning_features():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env.grid_spell_obs_size == (
        env.grid_spell_core_obs_size
        + len(env.spell_keys)
        + env.grid_legion_spell_used_obs_size
        + env.grid_nearest_enemy_debuff_obs_size
        + env.grid_scroll_obs_size
    )

    baseline = env._build_spell_grid_obs()
    assert baseline.shape == (env.grid_spell_obs_size,)
    assert baseline[0] == pytest.approx(0.0)  # magic tower not built
    assert baseline[1] == pytest.approx(0.0)  # turn lock off
    assert baseline[2] == pytest.approx(0.0)  # nothing learned
    assert np.allclose(
        baseline[
            env.grid_spell_core_obs_size : env.grid_spell_core_obs_size + len(env.spell_keys)
        ],
        0.0,
    )
    assert np.allclose(
        baseline[
            env.grid_spell_core_obs_size + len(env.spell_keys) :
        ],
        0.0,
    )

    for build_key in env.building_keys:
        building = env.active_buildings.get(build_key)
        if not isinstance(building, dict):
            continue
        if str(building.get("name", "") or "").strip() == env.MAGIC_TOWER_BUILDING_NAME:
            building["built"] = 1
            break
    first_spell_key = env.spell_keys[0]
    env.active_spells[first_spell_key]["learned"] = 1
    env.spell_learning_locked = True

    updated = env._build_spell_grid_obs()
    assert updated[0] == pytest.approx(1.0)
    assert updated[1] == pytest.approx(1.0)
    assert updated[2] > 0.0
    assert updated[env.grid_spell_core_obs_size + env.spell_keys.index(first_spell_key)] == pytest.approx(1.0)

    campaign_obs = env._build_campaign_grid_obs()
    spell_start = (
        env.grid_blue_obs_size
        + env.grid_resource_obs_size
        + env.grid_mana_source_obs_size
        + env.grid_building_obs_size
    )
    spell_end = spell_start + env.grid_spell_obs_size
    assert np.allclose(campaign_obs[spell_start:spell_end], updated)


def test_campaign_obs_includes_legion_spell_used_this_turn_and_nearest_enemy_debuff_flags():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (8, 8),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()

    for spell_key in ("lod_d2_s005", "lod_d2_s015"):
        env.active_spells[spell_key]["learned"] = 1
        spell = env.active_spells[spell_key]
        for mana_kind, amount in env._get_spell_use_costs(spell).items():
            attr_name = env.MANA_ATTR_BY_KIND[mana_kind]
            setattr(env, attr_name, float(getattr(env, attr_name, 0.0) or 0.0) + float(amount))

    env.enemy_team_states[10] = [
        {
            "name": "Nearest",
            "team": "red",
            "position": 1,
            "stand": "ahead",
            "health": 100.0,
            "hp": 100.0,
            "max_health": 100.0,
            "maxhp": 100.0,
            "unit_type": "Warrior",
            "armor": 40,
            "damage": 60,
            "damage_secondary": 0,
            "initiative": 50,
            "initiative_base": 50,
            "accuracy": 80,
            "accuracy_secondary": 0,
            "immunity": [],
            "resistance": [],
        }
    ]

    used_start = env.grid_spell_core_obs_size + len(env.spell_keys)
    debuff_start = used_start + env.grid_legion_spell_used_obs_size
    legion_spell_index = {
        str(spec.get("id", "") or ""): idx
        for idx, spec in enumerate(env.LEGION_DAMAGE_SPELL_ACTION_SPECS)
    }

    env.step(env.grid_legion_damage_spell_action_start + legion_spell_index["lod_d2_s005"])
    env.step(env.grid_legion_damage_spell_action_start + legion_spell_index["lod_d2_s015"])

    spell_obs = env._build_spell_grid_obs()

    assert spell_obs[used_start + legion_spell_index["lod_d2_s005"]] == pytest.approx(1.0)
    assert spell_obs[used_start + legion_spell_index["lod_d2_s015"]] == pytest.approx(1.0)
    assert spell_obs[used_start + legion_spell_index["lod_d2_s003"]] == pytest.approx(0.0)

    nearest_debuff_flags = spell_obs[debuff_start : debuff_start + env.grid_nearest_enemy_debuff_obs_size]
    assert np.allclose(nearest_debuff_flags, np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_spell_obs_uses_mask_nearest_enemy_candidate_without_bfs():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env.enemy_team_states[20] = [
        {
            "name": "Observed target",
            "team": "red",
            "position": 1,
            "stand": "ahead",
            "health": 100.0,
            "hp": 100.0,
            "max_health": 100.0,
            "maxhp": 100.0,
            "unit_type": "Warrior",
            "armor": 40,
            "damage": 60,
            "damage_secondary": 0,
            "initiative": 50,
            "initiative_base": 50,
            "accuracy": 80,
            "accuracy_secondary": 0,
            "immunity": [],
            "resistance": [],
        }
    ]
    env.enemy_map_spell_effects[20] = {"lod_d2_s005"}

    def _unexpected_exact_nearest(*args, **kwargs):
        raise AssertionError("spell observation must not call exact nearest-enemy BFS")

    env.get_nearest_enemy_stack = _unexpected_exact_nearest
    env._get_nearest_enemy_stack_for_mask = lambda *args, **kwargs: {
        "enemy_id": 20,
        "position": (8, 8),
        "path_distance": None,
        "fallback_distance": 5,
        "reachable": False,
        "is_alive": True,
        "description": "Observed target",
    }

    used_start = env.grid_spell_core_obs_size + len(env.spell_keys)
    debuff_start = used_start + env.grid_legion_spell_used_obs_size

    spell_obs = env._build_spell_grid_obs()

    nearest_debuff_flags = spell_obs[debuff_start : debuff_start + env.grid_nearest_enemy_debuff_obs_size]
    assert np.allclose(nearest_debuff_flags, np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_undead_spell_obs_includes_used_this_turn_and_nearest_enemy_debuff_flags():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=4)
    env.reset(seed=123)

    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (8, 8),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()

    for spell_key in ("und_d2_s005", "und_d2_s013"):
        env.active_spells[spell_key]["learned"] = 1
        spell = env.active_spells[spell_key]
        for mana_kind, amount in env._get_spell_use_costs(spell).items():
            attr_name = env.MANA_ATTR_BY_KIND[mana_kind]
            setattr(env, attr_name, float(getattr(env, attr_name, 0.0) or 0.0) + float(amount))

    env.enemy_team_states[10] = [
        {
            "name": "Nearest",
            "team": "red",
            "position": 1,
            "stand": "ahead",
            "health": 100.0,
            "hp": 100.0,
            "max_health": 100.0,
            "maxhp": 100.0,
            "unit_type": "Warrior",
            "armor": 40,
            "damage": 60,
            "damage_secondary": 0,
            "initiative": 50,
            "initiative_base": 50,
            "accuracy": 80,
            "accuracy_secondary": 0,
            "immunity": [],
            "resistance": [],
        }
    ]

    used_start = env.grid_spell_core_obs_size + len(env.spell_keys)
    debuff_start = used_start + env.grid_legion_spell_used_obs_size
    spell_index = {
        str(spec.get("id", "") or ""): idx
        for idx, spec in enumerate(env._current_map_offensive_spell_action_specs())
    }

    env.step(env.grid_legion_damage_spell_action_start + spell_index["und_d2_s005"])
    env.step(env.grid_legion_damage_spell_action_start + spell_index["und_d2_s013"])

    spell_obs = env._build_spell_grid_obs()

    assert spell_obs[used_start + spell_index["und_d2_s005"]] == pytest.approx(1.0)
    assert spell_obs[used_start + spell_index["und_d2_s013"]] == pytest.approx(1.0)
    assert spell_obs[used_start + spell_index["und_d2_s002"]] == pytest.approx(0.0)

    nearest_debuff_flags = spell_obs[debuff_start : debuff_start + env.grid_nearest_enemy_debuff_obs_size]
    assert np.allclose(nearest_debuff_flags, np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_merchant_sell_values_remain_unchanged():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env._hero_item_gold_value_by_name("Ruby (Valuable)") == 1250
    assert env._hero_item_gold_value_by_name("Banner of War") == 5000
    assert env._hero_item_gold_value_by_name("Bronze Ring (Valuable)") == 250


def test_legions_territory_expands_by_ten_tiles_per_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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


def test_legions_settlement_territory_requires_full_city_clear_and_then_grows_from_source():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env._activate_legions_settlement_territory_if_cleared(32) == []

    env.grid_env.mark_enemy_defeated(32)
    assert env._activate_legions_settlement_territory_if_cleared(32) == []
    assert "Стагириас" not in env.legions_active_settlement_territory_capture_turn_by_name

    env.grid_env.mark_enemy_defeated(67)
    assert env._activate_legions_settlement_territory_if_cleared(67) == ["Стагириас"]
    assert env.legions_active_settlement_territory_capture_turn_by_name["Стагириас"] == 0
    assert env.legions_settlement_territory_tiles_by_name["Стагириас"] == ((6, 4),)
    assert (6, 4) in env.legions_territory_tile_set

    env._advance_turns(1)
    assert len(env.legions_settlement_territory_tiles_by_name["Стагириас"]) == 16


def test_legions_settlement_territory_levels_map_to_expected_growth_rules():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    sources = env.legions_settlement_territory_source_by_name
    assert sources["Стагириас"]["territory_level"] == 1
    assert sources["Порт Бирмингем"]["territory_level"] == 1
    assert sources["Кордилия"]["territory_level"] == 2
    assert sources["Порт Полонис"]["territory_level"] == 2
    assert sources["Соругирилла"]["territory_level"] == 2

    assert sources["Стагириас"]["expansion_per_turn"] == 15
    assert sources["Порт Бирмингем"]["expansion_per_turn"] == 15
    assert sources["Кордилия"]["expansion_per_turn"] == 15
    assert sources["Порт Полонис"]["expansion_per_turn"] == 15
    assert sources["Соругирилла"]["expansion_per_turn"] == 15


def test_legions_settlement_territories_use_configured_growth_rates():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env.grid_env.mark_enemy_defeated(22)
    assert env._activate_legions_settlement_territory_if_cleared(22) == ["Порт Бирмингем"]

    env.grid_env.mark_enemy_defeated(33)
    assert env._activate_legions_settlement_territory_if_cleared(33) == ["Кордилия"]

    env.grid_env.mark_enemy_defeated(35)
    env.grid_env.mark_enemy_defeated(69)
    assert env._activate_legions_settlement_territory_if_cleared(69) == ["Соругирилла"]

    env._advance_turns(1)

    assert len(env.legions_settlement_territory_tiles_by_name["Порт Бирмингем"]) == 16
    assert len(env.legions_settlement_territory_tiles_by_name["Кордилия"]) == 16
    assert len(env.legions_settlement_territory_tiles_by_name["Соругирилла"]) == 16


def test_legions_territory_grid_obs_marks_claimed_tiles_and_updates_with_turns():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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


def test_empire_territory_can_be_disabled_while_legions_still_expand():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        Realcapital=2,
        empire_territory_enabled=False,
    )
    _, info = env.reset(seed=123)

    assert env.empire_territory_tiles == ()
    assert env.empire_territory_tile_set == set()
    assert env.empire_territory_order == ()
    assert env.empire_territory_path_distances == {}
    assert info["empire_territory_count"] == 0
    assert tuple(env.empire_territory_source_tile) == (26, 10)

    env._advance_turns(3)

    assert env.empire_territory_tiles == ()
    assert env.empire_territory_tile_set == set()
    assert len(env.legions_territory_tiles) == 1 + 3 * env.LEGIONS_TERRITORY_EXPANSION_PER_TURN
    assert float(env._build_empire_territory_grid_obs().sum()) == pytest.approx(0.0)


def test_empire_territory_grid_obs_marks_claimed_tiles_and_updates_with_turns():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    baseline = env._build_campaign_grid_obs().copy()

    env.empire_territory_tiles = ((0, 0), (1, 0), (2, 0))
    env.empire_territory_tile_set = set(env.empire_territory_tiles)

    updated = env._build_campaign_grid_obs()

    assert np.array_equal(updated, baseline)


def test_legions_and_empire_territories_only_claim_reachable_regions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    _refresh_territories_at_turn(env, 999)

    assert tuple(env.grid_env.start_position) in env.legions_territory_tile_set
    assert tuple(env.empire_territory_source_tile) in env.empire_territory_tile_set
    assert tuple(env.grid_env.start_position) not in env.empire_territory_tile_set
    assert tuple(env.empire_territory_source_tile) not in env.legions_territory_tile_set
    assert tuple(env.grid_env.start_position) not in env.empire_territory_path_distances
    assert tuple(env.empire_territory_source_tile) not in env.legions_territory_path_distances
    assert not (env.legions_territory_tile_set & env.empire_territory_tile_set)


def test_legions_territory_skips_capital_footprint_water_and_other_objects():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    enemy_tiles = {tuple(pos) for pos in env.grid_env.enemy_positions.values()}
    assert env.legions_territory_forbidden_tile_set.isdisjoint(enemy_tiles)


def test_legions_territory_keeps_gold_mines_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    gold_mine_tiles = {(2, 31), (24, 4), (29, 46), (41, 46)}
    assert env.legions_territory_forbidden_tile_set.isdisjoint(gold_mine_tiles)

    _refresh_territories_at_turn(env, 999)
    reachable_gold_mines = {
        tile for tile in gold_mine_tiles if tile in env.legions_territory_path_distances
    }
    unreachable_gold_mines = gold_mine_tiles - reachable_gold_mines
    assert reachable_gold_mines.issubset(env.legions_territory_tile_set)
    assert env.legions_territory_tile_set.isdisjoint(unreachable_gold_mines)


def test_legions_territory_keeps_mana_crystals_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    mana_crystal_tiles = {
        (10, 29),  # red mana
        (7, 41),   # orange mana
        (6, 16),   # red mana
        (6, 11),   # yellow mana
    }
    assert env.legions_territory_forbidden_tile_set.isdisjoint(mana_crystal_tiles)

    _refresh_territories_at_turn(env, 999)
    reachable_mana_crystals = {
        tile for tile in mana_crystal_tiles if tile in env.legions_territory_path_distances
    }
    unreachable_mana_crystals = mana_crystal_tiles - reachable_mana_crystals
    assert reachable_mana_crystals.issubset(env.legions_territory_tile_set)
    assert env.legions_territory_tile_set.isdisjoint(unreachable_mana_crystals)


def test_legions_territory_keeps_chest_tiles_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    chest_tiles = set(env.chests.keys())
    assert env.legions_territory_forbidden_tile_set.isdisjoint(chest_tiles)

    _refresh_territories_at_turn(env, 999)
    reachable_chests = {
        tile for tile in chest_tiles if tile in env.legions_territory_path_distances
    }
    unreachable_chests = chest_tiles - reachable_chests
    assert reachable_chests.issubset(env.legions_territory_tile_set)
    assert env.legions_territory_tile_set.isdisjoint(unreachable_chests)


def test_legions_territory_keeps_landmarks_and_locations_claimable():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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

    _refresh_territories_at_turn(env, 999)

    assert road_tiles.issubset(env.empire_territory_tile_set)


def test_campaign_exposes_mana_sources_with_expected_positions_and_labels():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    expected = {
        (6, 16): ("Мана преисподней", "П"),
        (44, 21): ("Мана преисподней", "П"),
        (10, 29): ("Мана преисподней", "П"),
        (6, 11): ("Мана жизни", "Ж"),
        (25, 16): ("Мана жизни", "Ж"),
        (29, 40): ("Мана жизни", "Ж"),
        (7, 41): ("Мана смерти", "С"),
        (1, 4): ("Мана рун", "Р"),
    }

    assert set(env.mana_sources.keys()) == set(expected.keys())
    assert set(env.grid_env.mana_sources.keys()) == set(expected.keys())

    for tile, (name, letter) in expected.items():
        env_meta = env.mana_sources[tile]
        grid_meta = env.grid_env.mana_sources[tile]
        assert env_meta["name"] == name
        assert env_meta["letter"] == letter
        assert grid_meta["name"] == name
        assert grid_meta["letter"] == letter


def test_grid_env_exposes_mana_source_labels_and_colors():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    mana_letters = {meta["letter"] for meta in env.grid_env.mana_sources.values()}
    mana_colors = {meta["ansi_color"] for meta in env.grid_env.mana_sources.values()}
    mana_names = {meta["name"] for meta in env.grid_env.mana_sources.values()}

    assert {"П", "Ж", "С", "Р"}.issubset(mana_letters)
    assert {"\x1b[31m", "\x1b[34m", "\x1b[90m", "\x1b[96m"}.issubset(mana_colors)
    assert "Мана преисподней" in mana_names
    assert "Мана жизни" in mana_names
    assert "Мана смерти" in mana_names
    assert "Мана рун" in mana_names


def test_campaign_visualizer_draws_mana_source_letters():
    import matplotlib.pyplot as plt

    from visualize_campaign import CampaignVisualizer

    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    viz = CampaignVisualizer(
        grid_size=env.grid_size,
        background_map_path=None,
        scenario_path=None,
    )
    try:
        viz.draw_grid(
            agent_pos=env.grid_env.agent_pos,
            enemy_positions=env.grid_env.enemy_positions,
            enemies_alive=env.grid_env.enemies_alive,
            visited_cells=env.grid_env.visited_cells,
            castle_heal_tiles=env.castle_heal_tiles,
            legions_territory_tiles=env.legions_territory_tiles,
            empire_territory_tiles=env.empire_territory_tiles,
            obstacle_tiles=env.grid_env.obstacle_positions,
            merchant_positions=env.grid_env.merchant_positions,
            merchant_site_anchors=env.merchant_site_anchors,
            chest_positions=env.chests,
            mana_sources=env.grid_env.mana_sources,
            mana_totals=env._current_mana_totals(),
            mana_income_per_turn=env.mana_income_per_turn,
            turns=env.turns,
            gold=env.gold,
            steps=env.moves,
        )
        text_values = [
            text.get_text()
            for axis in (viz.ax, viz.info_ax)
            for text in axis.texts
        ]
        assert "П" in text_values
        assert "Ж" in text_values
        assert "С" in text_values
        assert "Р" in text_values
        assert any(
            "\u041c\u0430\u043d\u0430 \u043f\u0440\u0435\u0438\u0441\u043f\u043e\u0434\u043d\u0435\u0439: 0 (+25/\u0445\u043e\u0434)"
            in text
            for text in text_values
        )
        assert any(
            "\u041c\u0430\u043d\u0430 \u044d\u043b\u044c\u0444\u043e\u0432: 0 (+0/\u0445\u043e\u0434)"
            in text
            for text in text_values
        )
    finally:
        plt.close(viz.fig)


def test_visualize_campaign_formats_spell_action_label():
    from visualize_campaign import _format_grid_action_label

    label = _format_grid_action_label(
        42,
        {
            "spell_cast_action": True,
            "spell_description": "Fireball",
            "target_enemy_id": 17,
        },
    )

    assert label == "Spell: Fireball -> E17"


def test_visualize_campaign_formats_mercenary_hire_action_label():
    from visualize_campaign import _format_grid_action_label

    label = _format_grid_action_label(
        47,
        {
            "mercenary_hire_action": True,
            "mercenary_hire_unit_name": "Варвар",
            "mercenary_hire_site": "Лагерь северных варваров",
        },
    )

    assert label == "Merc: Варвар (Лагерь северных варваров)"
