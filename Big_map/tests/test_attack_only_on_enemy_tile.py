"""Only a step onto a stack starts a battle; passing next to it is free."""

import pytest

from battle_env import RUN_AWAY_ACTION_INDEX
from campaign_env import CampaignEnv


@pytest.fixture
def env():
    value = CampaignEnv(
        map_name="small", observation_version="local5", Realcapital=2,
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    value.reset(seed=1)
    yield value
    value.close()


def action_towards(env, source, target):
    delta = (target[0] - source[0], target[1] - source[1])
    return list(env.grid_env.ACTION_DELTAS).index(delta)


def free_tile(env, tile):
    grid = env.grid_env
    return (grid._is_within_grid(tile)
            and tile not in grid.obstacle_positions
            and tile not in grid.dynamic_blocked_positions
            and grid.get_enemy_at_position(tile) is None)


def flank(env):
    """An enemy and two neighbouring tiles A, B that are also adjacent to it."""
    grid = env.grid_env
    for enemy_id, enemy_pos in sorted(grid.enemy_positions.items()):
        if not grid.enemies_alive.get(enemy_id):
            continue
        ex, ey = enemy_pos
        ring = [(ex + dx, ey + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        for a in ring:
            for b in ring:
                if a != b and max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 \
                        and free_tile(env, a) and free_tile(env, b):
                    return enemy_id, tuple(enemy_pos), a, b
    raise AssertionError("no enemy with two free adjacent tiles")


def stand(env, tile):
    env.grid_env.agent_pos = tile
    env.moves = env.moves_per_turn


def test_passing_next_to_a_stack_never_starts_a_battle(env):
    enemy_id, _, a, b = flank(env)
    stand(env, a)
    info = env.step(action_towards(env, a, b))[-1]
    assert not info["battle_triggered"]
    assert env.mode == env.MODE_GRID
    assert tuple(env.grid_env.agent_pos) == b
    assert env.grid_env.enemies_alive[enemy_id]


def test_step_onto_stack_attacks_from_current_tile(env):
    enemy_id, enemy_pos, a, _ = flank(env)
    stand(env, a)
    moves_before = env.moves
    battle_cost = env._battle_grid_move_cost()
    info = env.step(action_towards(env, a, enemy_pos))[-1]
    assert info["battle_triggered"] and info["enemy_id"] == enemy_id
    assert env.mode == env.MODE_BATTLE and env.current_enemy_id == enemy_id
    assert tuple(env.grid_env.agent_pos) == a
    assert env.battle_origin_pos == a
    assert not info["movement_applied"] and info["stagnation_penalty"] == 0
    assert moves_before - env.moves == min(moves_before, battle_cost)


def test_after_victory_hero_stays_and_can_enter_the_cleared_tile(env):
    enemy_id, enemy_pos, a, _ = flank(env)
    stand(env, a)
    env.step(action_towards(env, a, enemy_pos))
    battle = env.battle_env
    for unit in battle.combined:
        if unit.get("team") == "red" and unit.get("health", 0) > 0:
            battle._subtract_health(unit, unit["health"])
    battle._finalize_victory("blue")
    env.step(0)
    assert env.mode == env.MODE_GRID
    assert not env.grid_env.enemies_alive[enemy_id]
    assert tuple(env.grid_env.agent_pos) == a
    stand(env, a)
    info = env.step(action_towards(env, a, enemy_pos))[-1]
    assert not info["battle_triggered"]
    assert tuple(env.grid_env.agent_pos) == enemy_pos


def test_after_retreat_hero_can_walk_along_the_stack(env):
    enemy_id, enemy_pos, a, b = flank(env)
    stand(env, a)
    env.step(action_towards(env, a, enemy_pos))
    env.battle_env.rng.seed(42)
    for _ in range(12):
        info = env.step(RUN_AWAY_ACTION_INDEX)[-1]
        if env.mode == env.MODE_GRID:
            break
    assert env.mode == env.MODE_GRID, info
    assert info.get("battle_retreat")
    assert tuple(env.grid_env.agent_pos) == a
    assert env.grid_env.enemies_alive[enemy_id]
    stand(env, a)
    info = env.step(action_towards(env, a, b))[-1]
    assert not info["battle_triggered"]
    assert tuple(env.grid_env.agent_pos) == b
