"""Strategic spell protection and ruin entry follow sites on the active map."""
from copy import deepcopy

import pytest

from maps.wotans_retribution import SOURCE_SNAPSHOT


def hp(env, enemy_id):
    return sum(float(unit.get("health", 0)) for unit in env._get_enemy_team_state(enemy_id))


def damage_action(env):
    key = "mcl_d2_s004"
    if key in env.active_spells:
        env.active_spells[key]["learned"] = 1
        entries = env._current_map_offensive_spell_action_specs()
        action = env.grid_legion_damage_spell_action_start + next(
            index for index, entry in enumerate(entries) if entry["id"] == key)
    else:
        env.spell_shop_purchased_spell_ids.add(key)
        action = env.grid_spell_shop_cast_action_start + next(
            index for index, entry in enumerate(env._spell_shop_cast_spell_entries())
            if entry["spell_id"] == key)
    for attr in env.MANA_ATTR_BY_KIND.values():
        setattr(env, attr, 10000.0)
    return action, key


def stand_beside(env, tile):
    for action, (dx, dy) in enumerate(env.grid_env.ACTION_DELTAS):
        start = (tile[0] - dx, tile[1] - dy)
        if (env.grid_env._is_within_grid(start)
                and start not in env.grid_env.obstacle_positions
                and start not in env.grid_env.dynamic_blocked_positions
                and env.grid_env.get_enemy_at_position(start) is None):
            env.grid_env.agent_pos = start
            env.moves = env.moves_per_turn
            return action
    pytest.fail(f"No entrance approach at {tile}")


def test_wotan_protection_matches_every_original_source_object(campaign_factory):
    env = campaign_factory(map_name="wotans_retribution")
    protected = set()
    for row in SOURCE_SNAPSHOT["enemies"]:
        inside = str(row.get("inside", ""))
        expected = row["kind"] in {"ruin_guardian", "settlement_garrison"} or inside.startswith(("S117RU", "S117FT"))
        enemy_id = int(row["enemy_id"])
        assert env._is_enemy_stack_spell_targetable(enemy_id) is not expected, row
        if expected:
            protected.add(enemy_id)
    assert env._spell_untargetable_enemy_ids() == protected


@pytest.mark.parametrize("map_name", ["default", "small", "wotans_retribution", "green_dragon_minimal"])
def test_protection_follows_current_stack_location_not_its_numeric_id(campaign_factory, map_name):
    env = campaign_factory(map_name=map_name)
    ruin_id = next(iter(env.RUIN_REWARD_BY_ENEMY_ID))
    ruin_tile = env.grid_env.enemy_positions[ruin_id]
    field_id = next(eid for eid in env.grid_env.enemy_positions
                    if env._is_enemy_stack_spell_targetable(eid))
    field_tile = env.grid_env.enemy_positions[field_id]
    assert not env._is_enemy_stack_spell_targetable(ruin_id)
    env.grid_env.enemy_positions[ruin_id] = field_tile
    env.grid_env.enemy_positions[field_id] = ruin_tile
    assert env._is_enemy_stack_spell_targetable(ruin_id)
    assert not env._is_enemy_stack_spell_targetable(field_id)
    env.grid_env.enemy_positions[ruin_id] = ruin_tile
    assert not env._is_enemy_stack_spell_targetable(ruin_id)


@pytest.mark.parametrize("enemy_id,allowed", [(1, False), (5, False), (14, False), (22, True)])
def test_wotan_public_cast_respects_ruins_cities_capital_and_field(campaign_factory, enemy_id, allowed):
    env = campaign_factory(map_name="wotans_retribution", detailed_step_info=True)
    env.grid_env.enemies_alive = {eid: eid == enemy_id for eid in env.grid_env.enemy_positions}
    stand_beside(env, env.grid_env.enemy_positions[enemy_id])
    action, key = damage_action(env)
    assert bool(env.compute_action_mask()[action]) is allowed
    for for_mask in (False, True):
        target = env._get_map_offensive_spell_target(key, for_mask=for_mask)
        assert (target["enemy_id"] if target else None) == (enemy_id if allowed else None)
    before_hp, before_mana = hp(env, enemy_id), env._current_mana_totals()
    info = env.step(action)[-1]  # Bypass the mask for protected sites.
    assert bool(info.get("spell_cast_executed")) is allowed
    if allowed:
        assert hp(env, enemy_id) < before_hp
    else:
        assert hp(env, enemy_id) == before_hp
        assert env._current_mana_totals() == before_mana


def test_original_wotan_cast_skips_ruin_and_hits_nearest_field_stack(campaign_factory):
    env = campaign_factory(map_name="wotans_retribution", detailed_step_info=True)
    env.grid_env.agent_pos = (34, 5)
    action, key = damage_action(env)
    fields = [int(row["enemy_id"]) for row in SOURCE_SNAPSHOT["enemies"]
              if row["kind"] == "stack" and row.get("inside") == "G000000000"]
    expected = min(fields, key=lambda eid: (
        max(abs(env.grid_env.enemy_positions[eid][0] - 34),
            abs(env.grid_env.enemy_positions[eid][1] - 5)), eid))
    assert env._get_map_offensive_spell_target(key, for_mask=True)["enemy_id"] == expected
    assert env.compute_action_mask()[action]
    before = {eid: hp(env, eid) for eid in env.grid_env.enemy_positions}
    assert before[1] == 500
    info = env.step(action)[-1]
    assert info["spell_cast_executed"] and info["target_enemy_id"] == expected
    for eid, previous_hp in before.items():
        if eid == expected:
            assert hp(env, eid) < previous_hp
        else:
            assert hp(env, eid) == previous_hp


@pytest.mark.parametrize("map_name", ["default", "small", "wotans_retribution", "green_dragon_minimal"])
@pytest.mark.parametrize("leader", ["dead", "missing", "stale_hp"])
def test_no_living_leader_blocks_ruin_mask_and_forced_entry(campaign_factory, map_name, leader):
    env = campaign_factory(map_name=map_name, detailed_step_info=True)
    ruin_id = next(iter(env.RUIN_REWARD_BY_ENEMY_ID))
    tile = env.grid_env.enemy_positions[ruin_id]
    action = stand_beside(env, tile)
    assert env.compute_action_mask()[action]
    hero = env._resolve_travel_hero()
    assert hero is not None
    if leader == "missing":
        env.blue_team_state = [unit for unit in env.blue_team_state if not env._is_hero_unit(unit)]
    else:
        hero.update(health=0.0, hp=100.0 if leader == "stale_hp" else 0.0)
    assert env._resolve_travel_hero(alive_only=True) is None
    assert any(unit.get("health", 0) > 0 for unit in env.blue_team_state)
    before = deepcopy((env.grid_env.agent_pos, env.moves, env.gold, env.turns,
                       env.blue_team_state, env.grid_env.enemies_alive, env.chests))
    assert not env.compute_action_mask()[action]
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert info["blocked_by_dead_leader"]
    assert not info["battle_triggered"]
    assert env.mode == env.MODE_GRID and env.battle_env is None
    assert (env.grid_env.agent_pos, env.moves, env.gold, env.turns,
            env.blue_team_state, env.grid_env.enemies_alive, env.chests) == before


@pytest.mark.parametrize("map_name", ["default", "wotans_retribution"])
def test_resurrected_leader_can_attack_ruin_without_reset(campaign_factory, map_name):
    env = campaign_factory(map_name=map_name, detailed_step_info=True)
    ruin_id = next(iter(env.RUIN_REWARD_BY_ENEMY_ID))
    action = stand_beside(env, env.grid_env.enemy_positions[ruin_id])
    hero = env._resolve_travel_hero()
    hero.update(health=0.0, hp=0.0)
    assert not env.compute_action_mask()[action]
    env.extra_revive_bottles += 1
    revive_action = env.GRID_REVIVE_ACTION_START + list(env.GRID_BOTTLE_POSITIONS).index(hero["position"])
    assert env.compute_action_mask()[revive_action]
    assert env.step(revive_action)[-1]["revived"]
    assert env.compute_action_mask()[action]
    info = env.step(action)[-1]
    assert info["battle_triggered"]
    assert env.mode == env.MODE_BATTLE and env.current_enemy_id == ruin_id


def test_dead_leader_can_still_attack_a_field_stack(campaign_factory):
    env = campaign_factory(map_name="wotans_retribution", detailed_step_info=True)
    # ID 70 is a ruin only on default; here it is an ordinary field stack.
    action = stand_beside(env, env.grid_env.enemy_positions[70])
    env._resolve_travel_hero().update(health=0.0, hp=0.0)
    assert env.compute_action_mask()[action]
    assert env.step(action)[-1]["battle_triggered"]
    assert env.current_enemy_id == 70
