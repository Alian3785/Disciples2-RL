"""CUDA map preflight before requested training; isolated legal walkthrough.

No learning, throughput timing, state replacement or demonstrations for PPO.
Every transition uses the same mask/step as human play and training.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from stoix.envs.number_grid import CONTINUE, DEFEND, REST, SHOOT, NumberGrid, action_names, load_map


def route(env, state, blocked):
    start, goal = tuple(state.position), tuple(env.map_config['objective_position'])
    occupied = {tuple(p) for p, alive in zip(state.enemy_positions, state.alive) if alive}
    previous, queue = {start: None}, deque([start])
    directions = env.directions.tolist()
    while queue:
        current = queue.popleft()
        if max(abs(a-b) for a, b in zip(current, goal)) <= env.objective_radius:
            result = []
            while previous[current] is not None:
                current, action = previous[current]
                result.append(action)
            return list(reversed(result))
        for action, delta in enumerate(directions):
            point = (current[0]+delta[0], current[1]+delta[1])
            if (point not in previous and point not in occupied
                    and all(0 < n < env.size-1 for n in point) and not blocked[point]):
                previous[point] = (current, action)
                queue.append(point)
    raise AssertionError('No unoccupied route to the objective')


def validate(name, seed):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('CUDA is required; run through scripts/run_gpu.sh')
    game_map = load_map(name)
    env = NumberGrid(map_config=game_map)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    view = jax.jit(lambda s: (env.action_mask(s), jnp.all(jnp.isfinite(env.observation(s)))))
    print('Preparing '+name, flush=True)
    state, timestep = reset(jax.random.PRNGKey(seed))
    host = jax.device_get(state)
    assert env.observation_size == timestep.observation.shape[0]
    assert env.num_actions == len(action_names(game_map))
    assert int(host.alive.sum()) == 105 and not host.done
    assert int(host.day) == 1 and int(host.gold) == 300
    assert int(env.recruitment.composition(state)[0]) == game_map['hero_leadership_capacity']
    assert env.progression.max_hero_level == game_map['max_hero_level']
    assert env.spell_research.scenario_max_level == game_map['max_spell_level']
    initial_position = host.position.tolist()
    blocked = jax.device_get(env.territory.blocked_grid).reshape(env.size, env.size)
    actions, reward, fights = [], 0., 0
    while not host.done:
        mask, finite = jax.device_get(view(state))
        assert finite
        if host.in_battle:
            targets = [a for a in range(SHOOT, DEFEND) if mask[a]]
            action = targets[0] if targets else DEFEND if mask[DEFEND] else CONTINUE
        elif mask[CONTINUE] and (host.pending_encounters & host.alive).any():
            action = CONTINUE
        else:
            path = route(env, host, blocked)
            if not path:
                event = env.script_events.objective_event
                assert (env.script_events.enabled and event >= 0 and host.script_events_fired[event]
                        and not host.script_objective_ready), 'Objective tile did not finish the campaign'
            action = path[0] if path and mask[path[0]] else REST
            # Enter the mandatory ambush with a full movement allowance. These
            # source-aware planning choices never enter PPO's training data.
            if env.script_events.enabled and action < 8:
                destination = host.position+env.directions[action].tolist()
                for event, fired in zip(env.script_events.events, host.script_events_fired):
                    if not fired and any(e['kind'] == 'CREATE_STACK' for e in event['effects']):
                        for condition in event['conditions']:
                            if (condition['kind'] == 'LEADER_TO_ZONE'
                                    and max(abs(destination-condition['position'])) <= condition['radius']
                                    and int(host.movement_points) < int(env.movement_cap(state))):
                                action = REST
        assert mask[action], f'Planner action {action} disagrees with mask'
        state, timestep = step(state, jnp.int32(action))
        gain = float(timestep.reward)
        assert bool(jnp.isfinite(timestep.reward))
        next_host = jax.device_get(state)
        fights += int(not host.in_battle and next_host.in_battle)
        actions.append(action)
        reward += gain
        host = next_host
        assert not host.lost, (f'Walkthrough lost on decision {len(actions)}, day {host.day}, '
                               f'position {host.position}, fired {host.script_events_fired}')
        assert len(actions) <= game_map['max_steps']
    assert host.won and max(abs(host.position-env.objective_position)) <= env.objective_radius
    assert any(env.progression.rows[int(uid)].get('key') == 'uter_betrezen' and hp > 0
               for uid, hp in zip(host.unit_ids[:6], host.hp[:6]))
    result = dict(map=name, seed=seed, gpu=str(jax.devices()[0]),
        scenario_events_enabled=env.script_events.enabled, start=initial_position,
        objective=host.position.tolist(), decisions=len(actions), day=int(host.day),
        battles=fights, won=True, reward=reward, actions=actions, training_transitions=0,
        observation_size=env.observation_size, num_actions=env.num_actions,
        fired=[event['id'] for event, fired in zip(env.script_events.events, host.script_events_fired) if fired],
        map_sha256=hashlib.sha256(json.dumps(game_map, sort_keys=True).encode()).hexdigest(),
        restrictions={k: game_map[k] for k in ('hero_leadership_capacity', 'max_hero_level',
            'max_building_level', 'max_spell_level', 'objective_radius')})
    print(json.dumps({k: v for k, v in result.items() if k != 'actions'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.map, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
