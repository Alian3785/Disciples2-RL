"""Export actual restored PPO rollouts and a standalone offline HTML viewer."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf
from stoix.envs.number_grid import MAP, NumberGrid, wrap_wall_action_mask
from stoix.networks.base import FeedForwardActor
from numbergrid_config import ROOT
from numbergrid_reference import shortest_path


def plain(state):
    return {f.name: np.asarray(getattr(state, f.name)).tolist() for f in dataclasses.fields(state)}


def build(output):
    result = json.loads((output / 'results.json').read_text())
    game_map = json.loads((output / 'map.json').read_text())
    if game_map.get('battle_mode'):
        from export_battle_viewer import build as build_battle
        return build_battle(output)
    config = OmegaConf.load(output / 'config.json')
    checkpoint = (output / 'params.msgpack').read_bytes()
    assert hashlib.sha256(checkpoint).hexdigest() == result['checkpoint_sha256']
    actor_params = jax.tree.map(jnp.asarray, serialization.msgpack_restore(checkpoint)['actor_params'])
    actor = FeedForwardActor(
        torso=hydra.utils.instantiate(config.network.actor_network.pre_torso),
        action_head=hydra.utils.instantiate(config.network.actor_network.action_head, action_dim=8))
    env = NumberGrid(map_config=game_map)
    policy_env = wrap_wall_action_mask(env)
    advance = jax.jit(env.step)
    @jax.jit
    def rollout(seed, greedy_policy):
        state, ts = policy_env.reset(jax.random.PRNGKey(seed))
        initial = state
        def step(carry, unused):
            state, ts, key = carry
            key, subkey = jax.random.split(key)
            dist = actor.apply(actor_params, jax.tree.map(lambda x: x[None], ts.observation))
            action = jnp.where(greedy_policy, dist.mode()[0], dist.sample(seed=subkey)[0])
            state, ts = policy_env.step(state, action)
            return (state, ts, key), (state, action, ts.reward)
        (final, _, _), frames = jax.lax.scan(
            step, (state, ts, jax.random.PRNGKey(seed)), None, length=env.max_steps)
        # Transfer the entire episode once; raw terminal states are absorbing.
        return initial, final, frames
    records = []
    for i in range(9):
        seed = 42 + i
        initial, state, (states, actions, rewards) = jax.device_get(rollout(jnp.int32(seed), i == 0))
        frames = [dict(state=plain(initial), action=None, reward=0.0, total_reward=0.0)]
        total = 0.0
        for t in range(int(state.step_count)):
            reward = float(rewards[t])
            total += reward
            frame_state = {f.name: np.asarray(getattr(states, f.name)[t]).tolist()
                           for f in dataclasses.fields(states)}
            frames.append(dict(state=frame_state, action=int(actions[t]), reward=reward, total_reward=total))
        ending = 'победа' if bool(state.won) else 'поражение' if bool(state.lost) else 'лимит'
        label = f'{"Argmax" if i == 0 else "Выборка " + str(seed)} · {ending} · {int(state.step_count)} ходов'
        records.append(dict(label=label, seed=seed, policy='argmax' if i == 0 else 'sample', frames=frames))
    # Cross-language cases are computed by the actual JAX environment, including
    # edge states unlikely to be visited by a trained policy.
    cases = []
    rng = np.random.default_rng(874)
    state, _ = env.reset(jax.random.PRNGKey(0))
    for _ in range(512):
        action = int(rng.integers(-1, 9))
        nxt, ts = advance(state, jnp.int32(action))
        cases.append(dict(map=game_map, before=plain(state), action=action, after=plain(nxt), reward=float(ts.reward)))
        state = nxt if not bool(nxt.done) else env.reset(jax.random.PRNGKey(0))[0]
    base, _ = env.reset(jax.random.PRNGKey(0))
    last_enemy = game_map['opponent_positions'][-1]
    only_last_alive = jnp.arange(env.num_opponents) == env.num_opponents - 1
    contact_position = jnp.asarray([last_enemy[0] - 2, last_enemy[1]])
    for special, action in [
        (base.replace(step_count=jnp.int32(env.max_steps - 1)), 0),
        (base.replace(position=contact_position, number=jnp.int32(game_map['opponent_numbers'][-1] + 1), alive=only_last_alive), 4),
        (base.replace(position=contact_position, number=jnp.int32(game_map['opponent_numbers'][-1]), alive=only_last_alive), 4),
        (base.replace(position=jnp.asarray([1,1])), 7),
    ]:
        nxt, ts = advance(special, jnp.int32(action))
        cases.append(dict(map=game_map, before=plain(special), action=action, after=plain(nxt), reward=float(ts.reward)))
        again, terminal_ts = advance(nxt, jnp.int32(action))
        cases.append(dict(map=game_map, before=plain(nxt), action=action, after=plain(again), reward=float(terminal_ts.reward)))
    close_map = {**game_map,
                 'opponent_positions': [[5,5], [5,7]] + game_map['opponent_positions'][2:],
                 'opponent_numbers': [1,2] + game_map['opponent_numbers'][2:]}
    close_env = NumberGrid(map_config=close_map)
    before = base.replace(position=jnp.asarray([3,6]), number=jnp.int32(2))
    after, ts = close_env.step(before, jnp.int32(4))
    cases.append(dict(map=close_map, before=plain(before), action=4, after=plain(after), reward=float(ts.reward)))
    actions = shortest_path(game_map) if env.num_opponents <= 4 else None
    payload = dict(map=game_map, result=result, shortest_path_steps=len(actions) if actions else None,
                   shortest_path_actions=actions, records=records, validation_cases=cases)
    if env.mask_walls:
        positions = jnp.asarray([(r,c) for r in range(1,env.size-1) for c in range(1,env.size-1)])
        masks = jax.jit(jax.vmap(lambda p: env.action_mask(base.replace(position=p))))(positions)
        payload['wall_mask_cases'] = [dict(position=p, mask=m) for p,m in
            zip(np.asarray(positions).tolist(), np.asarray(masks).tolist())]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    (output / 'trajectories.json').write_text(encoded + '\n', encoding='utf-8')
    template = (ROOT / 'web/viewer.template.html').read_text(encoding='utf-8')
    html = template.replace('__DATA__', encoded)
    html = html.replace('/*__CSS__*/', (ROOT / 'web/viewer.css').read_text(encoding='utf-8'))
    html = html.replace('/*__ENGINE__*/', (ROOT / 'web/number_grid_engine.js').read_text(encoding='utf-8'))
    html = html.replace('/*__APP__*/', (ROOT / 'web/viewer_app.js').read_text(encoding='utf-8'))
    (ROOT / 'viewer.html').write_text(html, encoding='utf-8')
    print(json.dumps(dict(viewer=str(ROOT / 'viewer.html'), records=len(records),
                         transitions=sum(len(r['frames'])-1 for r in records),
                         validation_cases=len(cases), greedy_record=records[0]['label']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results' / (f"number_grid-{MAP['size']}x{MAP['size']}" + ('-step-cost' if MAP.get('step_cost', 0) else '') + ('-exploration' if MAP.get('exploration_bonus', 0) else '') + ('-wall-mask' if MAP.get('mask_walls', False) else '')))
    build(parser.parse_args().results)
