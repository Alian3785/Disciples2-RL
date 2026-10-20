"""CUDA preflight of the armies added to the current map (2026-10-17); no learning.

For every added army: a legal approach must start a real battle, and random legal
actions must finish it with finite rewards and observations and HP within bounds.
Enemy copies and summons are recorded. Required/optional goals follow the map.
Planner and random actions are never exposed to the learner.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid import NumberGrid, load_map

ADDED = range(46, 51)


def approach(env, game_map, state, enemy):
    """Hero position next to the army and the legal direction into it."""
    target = game_map['opponent_positions'][enemy]
    blocked = np.asarray(env.territory.blocked_grid).reshape(env.size, env.size)
    for direction, (dr, dc) in enumerate(np.asarray(env.directions).tolist()):
        r, c = target[0]-dr, target[1]-dc
        if 0 < r < env.size-1 and 0 < c < env.size-1 and not blocked[r, c]:
            near = state.replace(position=jnp.array([r, c], jnp.int32), movement_points=jnp.int32(100))
            if bool(env.action_mask(near)[direction]):
                return near, direction
    raise AssertionError(f'No legal approach to army {enemy}')


def battle(env, step, mask, state, key, limit=2000):
    """Play legal random actions until the battle ends; check invariants every step."""
    seen = dict(enemy_summons=0, enemy_copies=0, steps=0)
    while bool(state.in_battle) and not bool(state.done):
        legal = np.flatnonzero(np.asarray(mask(state)))
        assert legal.size, 'Battle state without a legal action'
        key, pick = jax.random.split(key)
        action = int(legal[int(jax.random.randint(pick, (), 0, legal.size))])
        state, ts = step(state, jnp.int32(action))
        assert bool(jnp.isfinite(ts.reward)), 'Non-finite reward'
        assert bool(jnp.all(jnp.isfinite(ts.observation))), 'Non-finite observation'
        assert bool(jnp.all(state.hp >= 0)), 'Negative HP'
        seen['enemy_summons'] = max(seen['enemy_summons'], int(jnp.sum(state.summon_owner[6:] >= 0)))
        seen['enemy_copies'] = max(seen['enemy_copies'], int(jnp.sum(state.copied[6:])))
        seen['steps'] += 1
        assert seen['steps'] < limit, 'Battle did not finish within the decision limit'
    return state, seen


def validate(episodes):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Map preflight requires CUDA')
    game_map = load_map('current')
    env = NumberGrid(map_config=game_map)
    reset, step, mask = jax.jit(env.reset), jax.jit(env.step), jax.jit(env.action_mask)
    initial, _ = reset(jax.random.PRNGKey(42))
    assert env.construction.lord['id'] == 'mage' and env.construction.faction == 'legions'
    optional = game_map['optional_enemies']
    nobody = jnp.zeros_like(initial.alive)
    # An empty map is cleared; optional armies alone cannot block completion.
    optional_alive = nobody.at[jnp.array(optional, jnp.int32)].set(True)
    assert not bool(env._goals_left(optional_alive, initial.scenario_spawned))
    for enemy in [enemy for enemy in ADDED if enemy not in optional]:
        assert bool(env._goals_left(nobody.at[enemy].set(True), initial.scenario_spawned))
    armies = []
    for enemy in ADDED:
        key = game_map['enemy_rosters'][enemy][4]
        totals = dict(army=enemy, unit=key, battles=0, won=0, lost=0, retreated=0, enemy_summons=0,
                      enemy_copies=0, max_steps=0)
        for episode in range(episodes):
            state, direction = approach(env, game_map, initial, enemy)
            state, ts = step(state, jnp.int32(direction))
            assert bool(state.in_battle) and int(state.enemy) == enemy, f'No battle with army {enemy}'
            alive_before = bool(state.alive[enemy])
            state, seen = battle(env, step, mask, state, jax.random.PRNGKey(1000*enemy+episode))
            totals['battles'] += 1
            totals['won'] += int(alive_before and not bool(state.alive[enemy]))
            totals['lost'] += int(bool(state.lost))
            totals['retreated'] += int(bool(state.alive[enemy]) and not bool(state.lost))
            totals['enemy_summons'] = max(totals['enemy_summons'], seen['enemy_summons'])
            totals['enemy_copies'] = max(totals['enemy_copies'], seen['enemy_copies'])
            totals['max_steps'] = max(totals['max_steps'], seen['steps'])
        armies.append(totals)
        print(json.dumps(totals, ensure_ascii=False), flush=True)
    return dict(map=game_map['name'], gpu=str(jax.devices()[0]), armies=env.num_opponents,
                observation_size=env.observation_size, actions=env.num_actions, optional_enemies=optional,
                ruler=[env.construction.faction, env.construction.lord['id']], added=armies,
                preflight='passed', training_transitions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
