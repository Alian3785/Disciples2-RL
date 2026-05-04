import sys
from pathlib import Path
from types import SimpleNamespace

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


def _hero_item_names(env: CampaignEnv) -> list[str]:
    return [env._hero_item_name(entry) for entry in env.heroitems]


def _invulnerability_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_INVULNERABILITY_ACTION_START + env.INVULNERABILITY_POTION_POSITIONS.index(
        target_pos
    )


def _strength_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_STRENGTH_ACTION_START + env.STRENGTH_POTION_POSITIONS.index(target_pos)


def _energy_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_ENERGY_ACTION_START + env.ENERGY_ELIXIR_POSITIONS.index(target_pos)


def _haste_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_HASTE_ACTION_START + env.HASTE_ELIXIR_POSITIONS.index(target_pos)


def _fire_ward_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_FIRE_WARD_ACTION_START + env.FIRE_WARD_POSITIONS.index(target_pos)


def _earth_ward_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_EARTH_WARD_ACTION_START + env.EARTH_WARD_POSITIONS.index(target_pos)


def _water_ward_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_WATER_WARD_ACTION_START + env.WATER_WARD_POSITIONS.index(target_pos)


def _air_ward_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_AIR_WARD_ACTION_START + env.AIR_WARD_POSITIONS.index(target_pos)


def _titan_elixir_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_TITAN_ELIXIR_ACTION_START + env.TITAN_ELIXIR_POSITIONS.index(target_pos)


def _supreme_elixir_action_index(env: CampaignEnv, target_pos: int) -> int:
    return env.GRID_SUPREME_ELIXIR_ACTION_START + env.SUPREME_ELIXIR_POSITIONS.index(target_pos)


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


def test_support_elixir_action_mask_requires_item_alive_target_and_blocks_same_effect_duplicates():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    energy_action = _energy_action_index(env, target_pos)
    haste_action = _haste_action_index(env, target_pos)
    fire_action = _fire_ward_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0))

    env.heroitems = [
        env.ENERGY_ELIXIR_ITEM_NAME,
        env.HASTE_ELIXIR_ITEM_NAME,
        env.FIRE_WARD_ITEM_NAME,
    ]
    mask = env.compute_action_mask()
    assert bool(mask[energy_action]) is True
    assert bool(mask[haste_action]) is True
    assert bool(mask[fire_action]) is True

    env.active_energy_elixir_positions.add(target_pos)
    env.active_fire_ward_positions.add(target_pos)
    mask = env.compute_action_mask()
    assert bool(mask[energy_action]) is False
    assert bool(mask[haste_action]) is True
    assert bool(mask[fire_action]) is False

    env.active_haste_elixir_positions.add(target_pos)
    mask = env.compute_action_mask()
    assert bool(mask[energy_action]) is False
    assert bool(mask[haste_action]) is False
    assert bool(mask[fire_action]) is False

    env.active_energy_elixir_positions.clear()
    env.active_haste_elixir_positions.clear()
    env.active_fire_ward_positions.clear()
    unit["hp"] = 0.0
    unit["health"] = 0.0
    mask = env.compute_action_mask()
    assert bool(mask[energy_action]) is False
    assert bool(mask[haste_action]) is False
    assert bool(mask[fire_action]) is False

    unit["hp"] = max_hp
    unit["health"] = max_hp


def test_permanent_elixir_action_mask_requires_item_and_living_target():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    titan_action = _titan_elixir_action_index(env, target_pos)
    supreme_action = _supreme_elixir_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    max_hp = float(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)

    mask = env.compute_action_mask()
    assert bool(mask[titan_action]) is False
    assert bool(mask[supreme_action]) is False

    env.heroitems = [
        env.TITAN_ELIXIR_ITEM_NAME,
        env.SUPREME_ELIXIR_ITEM_NAME,
    ]
    mask = env.compute_action_mask()
    assert bool(mask[titan_action]) is True
    assert bool(mask[supreme_action]) is True

    unit["hp"] = 0.0
    unit["health"] = 0.0
    mask = env.compute_action_mask()
    assert bool(mask[titan_action]) is False
    assert bool(mask[supreme_action]) is False

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
    assert env.INVULNERABILITY_POTION_ITEM_NAME not in _hero_item_names(env)
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
    assert env.STRENGTH_POTION_ITEM_NAME not in _hero_item_names(env)
    assert target_pos in env.active_strength_potion_positions
    assert int(unit.get("damage", 0) or 0) == base_damage

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("damage", 0) or 0) == int(
        round(base_damage * env.STRENGTH_POTION_DAMAGE_MULTIPLIER)
    )


def test_energy_elixir_use_consumes_item_rewards_and_buffs_next_battle_damage():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 8
    action = _energy_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    base_damage = int(unit.get("damage", 0) or 0)
    base_damage_secondary = int(unit.get("damage_secondary", 0) or 0)
    env.heroitems = [env.ENERGY_ELIXIR_ITEM_NAME]

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(float(env.reward_defeat_enemy))
    assert info.get("combat_potion_applied") is True
    assert info.get("combat_potion_consumed") is True
    assert info.get("combat_potion_item") == env.ENERGY_ELIXIR_ITEM_NAME
    assert info.get("combat_potion_target_pos") == target_pos
    assert env.ENERGY_ELIXIR_ITEM_NAME not in _hero_item_names(env)
    assert target_pos in env.active_energy_elixir_positions
    assert int(unit.get("damage", 0) or 0) == base_damage
    assert int(unit.get("damage_secondary", 0) or 0) == base_damage_secondary

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("damage", 0) or 0) == int(
        round(base_damage * env.ENERGY_ELIXIR_DAMAGE_MULTIPLIER)
    )
    assert int(battle_unit.get("damage_secondary", 0) or 0) == int(
        round(base_damage_secondary * env.ENERGY_ELIXIR_DAMAGE_MULTIPLIER)
    )

    env._save_blue_state()
    saved_unit = _blue_state_unit(env, target_pos)
    assert int(saved_unit.get("damage", 0) or 0) == base_damage
    assert int(saved_unit.get("damage_secondary", 0) or 0) == base_damage_secondary


def test_titan_elixir_permanently_buffs_damage_through_turn_and_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    action = _titan_elixir_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    base_damage = int(unit.get("damage", 0) or 0)
    base_damage_secondary = int(unit.get("damage_secondary", 0) or 0)
    env.heroitems = [env.TITAN_ELIXIR_ITEM_NAME]

    _, reward, terminated, truncated, info = env.step(action)

    expected_damage = int(round(base_damage * env.TITAN_ELIXIR_DAMAGE_MULTIPLIER))
    expected_damage_secondary = int(
        round(base_damage_secondary * env.TITAN_ELIXIR_DAMAGE_MULTIPLIER)
    )

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info.get("persistent_elixir_action") is True
    assert info.get("persistent_elixir_item") == env.TITAN_ELIXIR_ITEM_NAME
    assert info.get("persistent_elixir_target_pos") == target_pos
    assert info.get("persistent_elixir_applied") is True
    assert info.get("persistent_elixir_consumed") is True
    assert env.TITAN_ELIXIR_ITEM_NAME not in _hero_item_names(env)
    assert int(unit.get("damage", 0) or 0) == expected_damage
    assert int(unit.get("damage_secondary", 0) or 0) == expected_damage_secondary

    env._advance_turns(1)
    unit_after_turn = _blue_state_unit(env, target_pos)
    assert int(unit_after_turn.get("damage", 0) or 0) == expected_damage
    assert int(unit_after_turn.get("damage_secondary", 0) or 0) == expected_damage_secondary

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("damage", 0) or 0) == expected_damage
    assert int(battle_unit.get("damage_secondary", 0) or 0) == expected_damage_secondary

    env._save_blue_state()
    saved_unit = _blue_state_unit(env, target_pos)
    assert int(saved_unit.get("damage", 0) or 0) == expected_damage
    assert int(saved_unit.get("damage_secondary", 0) or 0) == expected_damage_secondary


def test_supreme_elixir_permanently_buffs_health_through_turn_and_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 8
    action = _supreme_elixir_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    base_max_hp = int(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0)
    base_hp = max(1, base_max_hp - 25)
    unit["hp"] = base_hp
    unit["health"] = base_hp
    env.heroitems = [env.SUPREME_ELIXIR_ITEM_NAME]

    _, reward, terminated, truncated, info = env.step(action)

    expected_max_hp = int(round(base_max_hp * env.SUPREME_ELIXIR_HEALTH_MULTIPLIER))
    if expected_max_hp <= base_max_hp:
        expected_max_hp = base_max_hp + 1
    expected_hp = min(
        expected_max_hp,
        int(round(base_hp * env.SUPREME_ELIXIR_HEALTH_MULTIPLIER)),
    )

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info.get("persistent_elixir_action") is True
    assert info.get("persistent_elixir_item") == env.SUPREME_ELIXIR_ITEM_NAME
    assert info.get("persistent_elixir_target_pos") == target_pos
    assert info.get("persistent_elixir_applied") is True
    assert info.get("persistent_elixir_consumed") is True
    assert env.SUPREME_ELIXIR_ITEM_NAME not in _hero_item_names(env)
    assert int(unit.get("maxhp", 0) or unit.get("max_health", 0) or 0) == expected_max_hp
    assert int(unit.get("hp", 0) or unit.get("health", 0) or 0) == expected_hp

    env._advance_turns(1)
    unit_after_turn = _blue_state_unit(env, target_pos)
    assert int(unit_after_turn.get("maxhp", 0) or unit_after_turn.get("max_health", 0) or 0) == expected_max_hp
    assert int(unit_after_turn.get("hp", 0) or unit_after_turn.get("health", 0) or 0) == expected_hp

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("max_health", 0) or 0) == expected_max_hp
    assert int(battle_unit.get("health", 0) or 0) == expected_hp

    env._save_blue_state()
    saved_unit = _blue_state_unit(env, target_pos)
    assert int(saved_unit.get("maxhp", 0) or saved_unit.get("max_health", 0) or 0) == expected_max_hp
    assert int(saved_unit.get("hp", 0) or saved_unit.get("health", 0) or 0) == expected_hp


def test_persistent_elixir_bonuses_transfer_to_promoted_unit(monkeypatch):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    unit = _blue_state_unit(env, target_pos)
    unit["name"] = "Test Source"
    unit["team"] = "blue"
    unit["turns_into"] = ["Test Promoted"]
    unit["damage"] = 110
    unit["damage_secondary"] = 20
    unit["original_damage"] = 110
    unit["max_health"] = 230
    unit["maxhp"] = 230
    unit["health"] = 180
    unit["hp"] = 180
    unit["campaign_titan_elixir_uses"] = 1
    unit["campaign_supreme_elixir_uses"] = 1

    battle_unit = dict(unit)
    env.battle_env = SimpleNamespace(
        combined=[battle_unit],
        last_levelups=["Test Source"],
    )

    monkeypatch.setattr(
        env,
        "_find_unit_data_by_name",
        lambda name: {"capital": 2, "name": str(name)},
    )
    monkeypatch.setattr(
        env,
        "_get_buildings_for_capital",
        lambda capital: {"promo": {"unit": "Test Promoted", "Build": 1}},
    )
    monkeypatch.setattr(
        env,
        "_build_unit_from_data",
        lambda entry, team, position: {
            "name": "Test Promoted",
            "team": str(team),
            "position": int(position),
            "turns_into": [],
            "damage": 200,
            "damage_secondary": 40,
            "original_damage": 200,
            "health": 300,
            "hp": 300,
            "max_health": 300,
            "maxhp": 300,
        },
    )

    upgraded_count = env._log_turns_into_levelups()

    expected_damage = int(round(200 * env.TITAN_ELIXIR_DAMAGE_MULTIPLIER))
    expected_damage_secondary = int(round(40 * env.TITAN_ELIXIR_DAMAGE_MULTIPLIER))
    expected_max_hp = int(round(300 * env.SUPREME_ELIXIR_HEALTH_MULTIPLIER))
    if expected_max_hp <= 300:
        expected_max_hp = 301

    assert upgraded_count == 1
    promoted_battle_unit = env.battle_env.combined[0]
    promoted_grid_unit = _blue_state_unit(env, target_pos)

    assert promoted_battle_unit["name"] == "Test Promoted"
    assert promoted_grid_unit["name"] == "Test Promoted"
    assert int(promoted_battle_unit.get("damage", 0) or 0) == expected_damage
    assert int(promoted_grid_unit.get("damage", 0) or 0) == expected_damage
    assert int(promoted_battle_unit.get("damage_secondary", 0) or 0) == expected_damage_secondary
    assert int(promoted_grid_unit.get("damage_secondary", 0) or 0) == expected_damage_secondary
    assert int(promoted_battle_unit.get("original_damage", 0) or 0) == expected_damage
    assert int(promoted_grid_unit.get("original_damage", 0) or 0) == expected_damage
    assert int(promoted_battle_unit.get("max_health", 0) or 0) == expected_max_hp
    assert int(promoted_grid_unit.get("max_health", 0) or 0) == expected_max_hp
    assert int(promoted_battle_unit.get("maxhp", 0) or 0) == expected_max_hp
    assert int(promoted_grid_unit.get("maxhp", 0) or 0) == expected_max_hp
    assert int(promoted_battle_unit.get("health", 0) or 0) == expected_max_hp
    assert int(promoted_grid_unit.get("health", 0) or 0) == expected_max_hp
    assert int(promoted_battle_unit.get("hp", 0) or 0) == expected_max_hp
    assert int(promoted_grid_unit.get("hp", 0) or 0) == expected_max_hp
    assert int(promoted_battle_unit.get("campaign_titan_elixir_uses", 0) or 0) == 1
    assert int(promoted_grid_unit.get("campaign_titan_elixir_uses", 0) or 0) == 1
    assert int(promoted_battle_unit.get("campaign_supreme_elixir_uses", 0) or 0) == 1
    assert int(promoted_grid_unit.get("campaign_supreme_elixir_uses", 0) or 0) == 1


def test_haste_elixir_buffs_next_battle_and_restore_after_save():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 8
    action = _haste_action_index(env, target_pos)
    unit = _blue_state_unit(env, target_pos)
    base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
    env.heroitems = [env.HASTE_ELIXIR_ITEM_NAME]

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(float(env.reward_defeat_enemy))
    assert info.get("combat_potion_applied") is True
    assert info.get("combat_potion_consumed") is True
    assert info.get("combat_potion_target_pos") == target_pos
    assert info.get("combat_potion_item") == env.HASTE_ELIXIR_ITEM_NAME
    assert env.HASTE_ELIXIR_ITEM_NAME not in _hero_item_names(env)

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("initiative_base", 0) or 0) == int(
        round(base_initiative * env.HASTE_ELIXIR_INITIATIVE_MULTIPLIER)
    )

    env._save_blue_state()
    saved_unit = _blue_state_unit(env, target_pos)
    assert int(saved_unit.get("initiative_base", 0) or 0) == base_initiative


def test_energy_elixir_and_haste_can_stack_on_same_unit_for_next_battle():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    unit = _blue_state_unit(env, target_pos)
    base_damage = int(unit.get("damage", 0) or 0)
    base_damage_secondary = int(unit.get("damage_secondary", 0) or 0)
    base_initiative = int(unit.get("initiative_base", unit.get("initiative", 0)) or 0)
    env.heroitems = [
        env.ENERGY_ELIXIR_ITEM_NAME,
        env.HASTE_ELIXIR_ITEM_NAME,
    ]

    env.step(_energy_action_index(env, target_pos))
    env.step(_haste_action_index(env, target_pos))

    assert target_pos in env.active_energy_elixir_positions
    assert target_pos in env.active_haste_elixir_positions

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    assert int(battle_unit.get("damage", 0) or 0) == int(
        round(base_damage * env.ENERGY_ELIXIR_DAMAGE_MULTIPLIER)
    )
    assert int(battle_unit.get("damage_secondary", 0) or 0) == int(
        round(base_damage_secondary * env.ENERGY_ELIXIR_DAMAGE_MULTIPLIER)
    )
    assert int(battle_unit.get("initiative_base", 0) or 0) == int(
        round(base_initiative * env.HASTE_ELIXIR_INITIATIVE_MULTIPLIER)
    )

    env._save_blue_state()
    saved_unit = _blue_state_unit(env, target_pos)
    assert int(saved_unit.get("damage", 0) or 0) == base_damage
    assert int(saved_unit.get("damage_secondary", 0) or 0) == base_damage_secondary
    assert int(saved_unit.get("initiative_base", 0) or 0) == base_initiative


def test_elemental_wards_apply_resistance_to_next_battle_and_expire():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    target_pos = 7
    unit = _blue_state_unit(env, target_pos)
    base_resistance = list(unit.get("resistance", []))
    env.heroitems = [
        env.FIRE_WARD_ITEM_NAME,
        env.EARTH_WARD_ITEM_NAME,
        env.WATER_WARD_ITEM_NAME,
        env.AIR_WARD_ITEM_NAME,
    ]

    env.step(_fire_ward_action_index(env, target_pos))
    env.step(_earth_ward_action_index(env, target_pos))
    env.step(_water_ward_action_index(env, target_pos))
    env.step(_air_ward_action_index(env, target_pos))

    env._init_battle(enemy_id=1)
    battle_unit = _battle_blue_unit(env, target_pos)
    battle_resistance = set(str(res) for res in (battle_unit.get("resistance") or []))
    assert {"Fire", "Earth", "Water", "Air"}.issubset(battle_resistance)

    env._save_blue_state()
    saved_unit = _blue_state_unit(env, target_pos)
    assert list(saved_unit.get("resistance", [])) == base_resistance

    env.active_fire_ward_positions.add(target_pos)
    env.active_earth_ward_positions.add(target_pos)
    env.active_water_ward_positions.add(target_pos)
    env.active_air_ward_positions.add(target_pos)
    env._advance_turns(1)
    assert env.active_fire_ward_positions == set()
    assert env.active_earth_ward_positions == set()
    assert env.active_water_ward_positions == set()
    assert env.active_air_ward_positions == set()


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


def test_first_battle_after_combat_potion_use_gets_small_participation_bonus():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        realcapital=2,
        reward_engage_battle=0.0,
        reward_defeat_enemy=0.0,
        reward_combat_potion_battle_participation=0.07,
    )
    env.reset(seed=123)

    env.heroitems = [env.STRENGTH_POTION_ITEM_NAME]
    _, _, _, _, potion_info = env.step(_strength_action_index(env, 7))

    assert potion_info.get("combat_potion_applied") is True
    assert env.combat_potion_battle_bonus_pending is True

    move_mask = env.grid_env.compute_action_mask()
    move_action = next(index for index, allowed in enumerate(move_mask[:8]) if allowed)
    target_tile = tuple(env.grid_env._target_pos_for_action(move_action))
    enemy_id = next(iter(env.grid_env.enemy_positions.keys()))

    env.grid_env.enemy_positions[enemy_id] = target_tile
    for eid in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[eid] = False
    env.grid_env.enemies_alive[enemy_id] = True

    _, reward, terminated, truncated, info = env.step(move_action)

    expected_reward = (
        0.2 * float(info.get("grid_reward_raw", 0.0))
        + float(info.get("stagnation_penalty", 0.0))
        + float(info.get("combat_potion_battle_bonus", 0.0))
    )

    assert terminated is False
    assert truncated is False
    assert info.get("battle_triggered") is True
    assert float(info.get("combat_potion_battle_bonus", 0.0)) == pytest.approx(0.07)
    assert reward == pytest.approx(expected_reward)
    assert env.combat_potion_battle_bonus_pending is False


def test_combat_potion_battle_participation_bonus_expires_after_turn_end():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        realcapital=2,
        reward_engage_battle=0.0,
        reward_defeat_enemy=0.0,
        reward_combat_potion_battle_participation=0.07,
    )
    env.reset(seed=123)

    env.heroitems = [env.INVULNERABILITY_POTION_ITEM_NAME]
    env.step(_invulnerability_action_index(env, 7))
    assert env.combat_potion_battle_bonus_pending is True

    env._advance_turns(1)
    assert env.combat_potion_battle_bonus_pending is False

    move_mask = env.grid_env.compute_action_mask()
    move_action = next(index for index, allowed in enumerate(move_mask[:8]) if allowed)
    target_tile = tuple(env.grid_env._target_pos_for_action(move_action))
    enemy_id = next(iter(env.grid_env.enemy_positions.keys()))

    env.grid_env.enemy_positions[enemy_id] = target_tile
    for eid in list(env.grid_env.enemies_alive.keys()):
        env.grid_env.enemies_alive[eid] = False
    env.grid_env.enemies_alive[enemy_id] = True

    _, _, terminated, truncated, info = env.step(move_action)

    assert terminated is False
    assert truncated is False
    assert info.get("battle_triggered") is True
    assert float(info.get("combat_potion_battle_bonus", 0.0)) == pytest.approx(0.0)
