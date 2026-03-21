import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _blue_state_unit(env: CampaignEnv, position: int) -> dict:
    return next(
        unit for unit in env.blue_team_state if int(unit.get("position", -1)) == int(position)
    )


def _battle_blue_unit(env: CampaignEnv, position: int) -> dict:
    assert env.battle_env is not None
    return next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1)) == int(position)
    )


def _invulnerability_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_INVULNERABILITY_ACTION_START + env.INVULNERABILITY_POTION_POSITIONS.index(
        target_pos
    )


def _strength_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_STRENGTH_ACTION_START + env.STRENGTH_POTION_POSITIONS.index(target_pos)


def test_collecting_combat_potions_adds_inventory_and_pickup_reward():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    start_x, start_y = env.grid_env.agent_pos
    chest_tile = (start_x + 2, start_y)
    move_target = (start_x + 1, start_y)
    env.grid_env.obstacle_positions.discard(move_target)
    env.grid_env.obstacle_positions.discard(chest_tile)
    env.chests = {
        chest_tile: (
            env.INVULNERABILITY_POTION_ITEM_NAME,
            env.STRENGTH_POTION_ITEM_NAME,
        )
    }
    env.heroitems = []
    env._sync_grid_chest_positions()

    _, reward, terminated, truncated, info = env.step(env.grid_env.ACTION_RIGHT)

    expected_pickup_reward = 2.0 * float(env.reward_defeat_enemy)
    expected_reward = (
        0.2 * float(info.get("grid_reward_raw", 0.0))
        + float(info.get("stagnation_penalty", 0.0))
        + float(info.get("battle_engage_bonus", 0.0))
        + expected_pickup_reward
    )

    assert terminated is False
    assert truncated is False
    assert env.grid_env.agent_pos == move_target
    assert float(info.get("combat_potion_pickup_reward", 0.0)) == pytest.approx(expected_pickup_reward)
    assert int(info.get("collected_combat_potions", 0)) == 2
    assert reward == pytest.approx(expected_reward)
    assert env.heroitems == [
        env.INVULNERABILITY_POTION_ITEM_NAME,
        env.STRENGTH_POTION_ITEM_NAME,
    ]


def test_combat_potion_action_mask_requires_item_alive_target_and_no_duplicate_effect():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    inv_action = _invulnerability_action_index(env, target_pos)
    str_action = _strength_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))

    mask = env.compute_action_mask()
    assert bool(mask[inv_action]) is False
    assert bool(mask[str_action]) is False

    env.heroitems = [
        env.INVULNERABILITY_POTION_ITEM_NAME,
        env.STRENGTH_POTION_ITEM_NAME,
    ]
    mask = env.compute_action_mask()
    assert bool(mask[inv_action]) is True
    assert bool(mask[str_action]) is True

    env.active_invulnerability_potion_positions.add(target_pos)
    env.active_strength_potion_positions.add(target_pos)
    mask = env.compute_action_mask()
    assert bool(mask[inv_action]) is False
    assert bool(mask[str_action]) is False

    env.active_invulnerability_potion_positions.clear()
    env.active_strength_potion_positions.clear()
    unit["hp"] = 0.0
    unit["health"] = 0.0
    mask = env.compute_action_mask()
    assert bool(mask[inv_action]) is False
    assert bool(mask[str_action]) is False

    unit["hp"] = max_hp
    unit["health"] = max_hp


def test_invulnerability_potion_use_consumes_item_rewards_and_buffs_next_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = _invulnerability_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    base_armor = int(unit.get("armor", 0) or 0)
    env.heroitems = [env.INVULNERABILITY_POTION_ITEM_NAME]

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(float(env.reward_defeat_enemy))
    assert info.get("combat_potion_applied") is True
    assert info.get("combat_potion_consumed") is True
    assert info.get("combat_potion_item") == env.INVULNERABILITY_POTION_ITEM_NAME
    assert info.get("combat_potion_target_pos") == target_pos
    assert env.heroitems == []
    assert target_pos in env.active_invulnerability_potion_positions
    assert int(unit.get("armor", 0) or 0) == base_armor

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("armor", 0) or 0) == base_armor + env.INVULNERABILITY_POTION_ARMOR_BONUS


def test_strength_potion_use_consumes_item_rewards_and_buffs_next_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = _strength_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    base_damage = int(unit.get("damage", 0) or 0)
    env.heroitems = [env.STRENGTH_POTION_ITEM_NAME]

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(float(env.reward_defeat_enemy))
    assert info.get("combat_potion_applied") is True
    assert info.get("combat_potion_consumed") is True
    assert info.get("combat_potion_item") == env.STRENGTH_POTION_ITEM_NAME
    assert info.get("combat_potion_target_pos") == target_pos
    assert env.heroitems == []
    assert target_pos in env.active_strength_potion_positions
    assert int(unit.get("damage", 0) or 0) == base_damage

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("damage", 0) or 0) == int(
        round(base_damage * env.STRENGTH_POTION_DAMAGE_MULTIPLIER)
    )


def test_battle_save_strips_combat_potion_bonuses_from_persistent_blue_state():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    unit = _blue_state_unit(env, target_pos)
    base_armor = int(unit.get("armor", 0) or 0)
    base_damage = int(unit.get("damage", 0) or 0)
    env.heroitems = [
        env.INVULNERABILITY_POTION_ITEM_NAME,
        env.STRENGTH_POTION_ITEM_NAME,
    ]

    env.step(_invulnerability_action_index(env, target_pos))
    env.step(_strength_action_index(env, target_pos))
    env._init_battle(enemy_id=1)

    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("armor", 0) or 0) == base_armor + env.INVULNERABILITY_POTION_ARMOR_BONUS
    assert int(battle_unit.get("damage", 0) or 0) == int(
        round(base_damage * env.STRENGTH_POTION_DAMAGE_MULTIPLIER)
    )

    env._save_blue_state()

    saved_unit = _blue_state_unit(env, target_pos)
    assert int(saved_unit.get("armor", 0) or 0) == base_armor
    assert int(saved_unit.get("damage", 0) or 0) == base_damage


def test_combat_potion_effects_expire_after_turn_advance():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    unit = _blue_state_unit(env, target_pos)
    base_armor = int(unit.get("armor", 0) or 0)
    base_damage = int(unit.get("damage", 0) or 0)
    env.heroitems = [
        env.INVULNERABILITY_POTION_ITEM_NAME,
        env.STRENGTH_POTION_ITEM_NAME,
    ]

    env.step(_invulnerability_action_index(env, target_pos))
    env.step(_strength_action_index(env, target_pos))
    assert target_pos in env.active_invulnerability_potion_positions
    assert target_pos in env.active_strength_potion_positions

    env._advance_turns(1)

    assert env.active_invulnerability_potion_positions == set()
    assert env.active_strength_potion_positions == set()

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("armor", 0) or 0) == base_armor
    assert int(battle_unit.get("damage", 0) or 0) == base_damage
