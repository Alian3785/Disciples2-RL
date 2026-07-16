import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.magic_train import (
    ENEMY_ENCLOSURE_BLOCKS,
    FIRST_ENEMY_ID,
    FIRST_ENEMY_TILE,
    HERO_START,
    MANA_SOURCE_TILES,
    SECOND_ENEMY_ID,
    SECOND_ENEMY_TILE,
)

POWERFUL_DAMAGE_SPELL_ID = "lod_d2_s023"
MAGIC_TOWER_BUILDING_ID = "lod_d2_b024"


def _make_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="magic_train",
        log_enabled=False,
        persist_blue_hp=False,
        reward_turn_penalty=0.0,
        reward_needaunit_turn_penalty=0.0,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_units(units) -> list[tuple[int, str]]:
    return [
        (int(unit["position"]), str(unit["name"]))
        for unit in units
        if str(unit.get("name", "")) != "пусто"
    ]


def _spell_learning_action(env: CampaignEnv, spell_key: str) -> int:
    return int(env.grid_spell_action_start) + env.spell_keys.index(spell_key)


def _spell_cast_action(env: CampaignEnv, spell_key: str) -> int:
    spell_idx = next(
        idx
        for idx, spec in enumerate(env._current_map_offensive_spell_action_specs())
        if str(spec.get("id", "") or "") == spell_key
    )
    return int(env.grid_legion_damage_spell_action_start) + int(spell_idx)


def _set_mana_for_costs(
    env: CampaignEnv,
    costs: dict[str, float],
    *,
    multiplier: float = 1.0,
) -> None:
    for mana_kind in env.MANA_KIND_ORDER:
        setattr(env, env.MANA_ATTR_BY_KIND[mana_kind], 0.0)
    for mana_kind, amount in costs.items():
        setattr(
            env,
            env.MANA_ATTR_BY_KIND[mana_kind],
            float(amount) * float(multiplier),
        )


def test_magic_train_is_registered_and_contains_only_requested_map_objects():
    assert "magic_train" in available_maps()
    map_config = get_map("magic_train")

    assert map_config.grid_size == 48
    assert map_config.hero_start == HERO_START
    assert map_config.default_objective == "all_enemies"
    assert map_config.enforce_enemy_reachability is False
    assert map_config.enemy_positions() == {
        FIRST_ENEMY_ID: FIRST_ENEMY_TILE,
        SECOND_ENEMY_ID: SECOND_ENEMY_TILE,
    }
    assert dict(map_config.mana_sources) == MANA_SOURCE_TILES
    assert map_config.obstacle_blocks == ENEMY_ENCLOSURE_BLOCKS
    assert map_config.chests == ()
    assert map_config.village_heal_tiles == ()
    assert map_config.merchant_sites == {}
    assert map_config.spell_shop_sites == {}
    assert map_config.mercenary_sites == {}
    assert map_config.trainer_sites == {}
    assert map_config.ruin_rewards == {}


def test_magic_train_starts_with_basic_legions_mage_party_and_tower_gold():
    env = _make_env(Realcapital=1, typeoflord=1, use_boss_starting_roster=True)
    env.reset(seed=123)

    assert env.Realcapital == 2
    assert env.typeoflord == 2
    assert env.grid_env.agent_pos == HERO_START
    assert env.gold == pytest.approx(200.0)
    assert _living_units(env.blue_team_state) == [
        (7, "Одержимый"),
        (9, "Одержимый"),
        (10, "Архидьявол"),
        (12, "Сектант"),
    ]

    tower_action = (
        int(env.GRID_BUILD_ACTION_START)
        + env.building_keys.index(MAGIC_TOWER_BUILDING_ID)
    )
    assert env._get_building_gold_cost(
        env.active_buildings[MAGIC_TOWER_BUILDING_ID]
    ) == pytest.approx(150.0)
    assert bool(env.compute_action_mask()[tower_action]) is True


def test_both_enemy_stacks_are_fully_enclosed_and_unreachable_by_movement():
    env = _make_env()
    env.reset(seed=123)

    obstacle_tiles = set(env.grid_env.obstacle_positions)
    assert len(obstacle_tiles) == 16

    for enemy_tile in (FIRST_ENEMY_TILE, SECOND_ENEMY_TILE):
        x, y = enemy_tile
        adjacent_tiles = {
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        }
        assert adjacent_tiles <= obstacle_tiles

    path_distances = env._build_territory_path_distances(
        source_tile=env.CASTLE_POS,
        blocked_tile_set=obstacle_tiles,
    )
    assert FIRST_ENEMY_TILE not in path_distances
    assert SECOND_ENEMY_TILE not in path_distances

    nearest_enemy = env.get_nearest_enemy_stack()
    assert nearest_enemy is not None
    assert nearest_enemy["enemy_id"] == FIRST_ENEMY_ID
    assert nearest_enemy["reachable"] is False
    assert nearest_enemy["path_distance"] is None


def test_four_adjacent_mana_sources_are_captured_after_first_turn():
    env = _make_env()
    env.reset(seed=123)

    assert {
        position: str(meta["kind"])
        for position, meta in env.mana_sources.items()
    } == {
        position: mana_kind
        for mana_kind, position in MANA_SOURCE_TILES.items()
    }

    env._advance_turns(1)

    assert env.legions_captured_mana_source_counts_by_kind == {
        "infernal": 1,
        "life": 1,
        "death": 1,
        "runes": 1,
        "elves": 0,
    }
    assert env.mana_income_per_turn == {
        "infernal": pytest.approx(75.0),
        "life": pytest.approx(50.0),
        "death": pytest.approx(50.0),
        "runes": pytest.approx(50.0),
        "elves": pytest.approx(0.0),
    }


def test_learned_magic_can_hit_unreachable_stack_without_starting_battle():
    env = _make_env()
    env.reset(seed=123)

    tower_action = (
        int(env.GRID_BUILD_ACTION_START)
        + env.building_keys.index(MAGIC_TOWER_BUILDING_ID)
    )
    env.step(tower_action)
    assert env._has_magic_tower_built() is True

    spell = env.active_spells[POWERFUL_DAMAGE_SPELL_ID]
    learning_costs = env._get_spell_learning_costs(spell)
    _set_mana_for_costs(env, learning_costs)
    learn_action = _spell_learning_action(env, POWERFUL_DAMAGE_SPELL_ID)
    assert bool(env.compute_action_mask()[learn_action]) is True

    _, learn_reward, terminated, truncated, learn_info = env.step(learn_action)

    assert terminated is False
    assert truncated is False
    assert learn_reward == pytest.approx(env.reward_spell_learn)
    assert learn_info["spell_learned"] is True

    use_costs = env._get_spell_use_costs(spell)
    _set_mana_for_costs(env, use_costs)
    cast_action = _spell_cast_action(env, POWERFUL_DAMAGE_SPELL_ID)
    assert bool(env.compute_action_mask()[cast_action]) is True

    _, _, terminated, truncated, cast_info = env.step(cast_action)

    assert terminated is False
    assert truncated is False
    assert env.mode == env.MODE_GRID
    assert cast_info["battle_triggered"] is False
    assert cast_info["spell_cast_executed"] is True
    assert cast_info["spell_cast_applied"] is True
    assert cast_info["target_enemy_id"] == FIRST_ENEMY_ID
    assert cast_info["target_enemy_reachable"] is False
    assert cast_info["spell_damage_total"] > 0.0


def test_both_enclosed_stacks_can_be_cleared_with_learned_magic():
    env = _make_env()
    env.reset(seed=123)
    env.active_spells[POWERFUL_DAMAGE_SPELL_ID]["learned"] = 1

    spell = env.active_spells[POWERFUL_DAMAGE_SPELL_ID]
    use_costs = env._get_spell_use_costs(spell)
    _set_mana_for_costs(env, use_costs, multiplier=3.0)
    cast_action = _spell_cast_action(env, POWERFUL_DAMAGE_SPELL_ID)

    _, _, terminated, truncated, first_info = env.step(cast_action)
    assert terminated is False
    assert truncated is False
    assert first_info["target_enemy_id"] == FIRST_ENEMY_ID
    assert env.grid_env.enemies_alive[FIRST_ENEMY_ID] is True

    _, _, terminated, truncated, second_info = env.step(cast_action)
    assert terminated is False
    assert truncated is False
    assert second_info["spell_enemy_defeated"] is True
    assert env.grid_env.enemies_alive[FIRST_ENEMY_ID] is False

    env._advance_turns(1)
    _, _, terminated, truncated, final_info = env.step(cast_action)

    assert terminated is True
    assert truncated is False
    assert final_info["target_enemy_id"] == SECOND_ENEMY_ID
    assert final_info["target_enemy_reachable"] is False
    assert final_info["spell_enemy_defeated"] is True
    assert final_info["campaign_result"] == "victory"
    assert final_info["campaign_victory_reason"] == "all_enemies_defeated"
