"""Run a saved policy on JAX/GPU and write detailed episode reports, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from flax import serialization
from omegaconf import OmegaConf

from numbergrid_episode_reports import EpisodeReports, read_episode_batch
from stoix.envs.number_grid import NumberGrid, wrap_wall_action_mask
from stoix.envs.number_grid_episode_reports import compact_episode_events
from stoix.networks.base import FeedForwardActor


def run(args):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Episode diagnostics require CUDA: use scripts/run_gpu.sh')
    if args.episodes <= 0 or not 0 < args.batch_size <= 64:
        raise ValueError('episodes must be positive; batch-size must be in 1..64')
    source, output = Path(args.results), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty diagnostic output directory: {output}')
    game_map = json.loads((source/'map.json').read_text(encoding='utf-8'))
    result = json.loads((source/'results.json').read_text(encoding='utf-8'))
    checkpoint = (source/'params.msgpack').read_bytes()
    if hashlib.sha256(checkpoint).hexdigest() != result['checkpoint_sha256']:
        raise ValueError('Checkpoint hash does not match results.json')
    raw = NumberGrid(map_config=game_map, episode_reports=True)
    if not raw.episode_reports:
        raise ValueError('Detailed reports require a battle map with unit progression')
    env = wrap_wall_action_mask(raw)
    config = OmegaConf.load(source/'config.json')
    actor = FeedForwardActor(torso=hydra.utils.instantiate(config.network.actor_network.pre_torso),
        action_head=hydra.utils.instantiate(config.network.actor_network.action_head,
                                          action_dim=raw.action_space().num_values))
    params = jax.tree.map(jnp.asarray, serialization.msgpack_restore(checkpoint)['actor_params'])
    reports = EpisodeReports(output, game_map, raw.combat_info, phase='diagnostic_'+args.policy)
    (output/'diagnostic_config.json').write_text(json.dumps(dict(
        checkpoint=str(source.resolve()), checkpoint_sha256=result['checkpoint_sha256'],
        training_steps=result['training_steps'], episodes=args.episodes, seed=args.seed,
        batch_size=args.batch_size, policy=args.policy, backend='gpu',
        note='Fresh episodes of the saved policy, not historical training trajectories'), indent=2)+'\n', encoding='utf-8')

    @jax.jit
    def evaluate(parameters, episode_ids):
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(args.seed), i))(episode_ids)
        states, ts = jax.vmap(env.reset)(keys)
        policy_keys = jax.vmap(lambda key: jax.random.fold_in(key, 1))(keys)
        def advance(carry, _):
            state, timestep, returns, random_keys = carry
            split = jax.vmap(jax.random.split)(random_keys)
            random_keys, draw_keys = split[:, 0], split[:, 1]
            if args.policy == 'argmax':
                actions = actor.apply(parameters, timestep.observation).mode()
            else:
                actions = jax.vmap(lambda obs, key: actor.apply(parameters,
                    jax.tree.map(lambda x: x[None], obs)).sample(seed=key)[0])(
                        timestep.observation, draw_keys)
            state, timestep = jax.vmap(env.step)(state, actions)
            returns = returns + timestep.reward
            records = dict(episode_event=timestep.extras['episode_event'], episode_return=returns)
            return (state, timestep, returns, random_keys), records
        _, records = jax.lax.scan(advance, (states, ts, jnp.zeros(episode_ids.size), policy_keys),
                                  None, length=raw.max_steps)
        return compact_episode_events(records)

    for first in range(0, args.episodes, args.batch_size):
        ids = jnp.arange(first, min(first+args.batch_size, args.episodes), dtype=jnp.uint32)
        count, packets = evaluate(params, ids)
        count = int(jax.device_get(count))
        packets = {**packets, 'stream': packets['stream'] + first}
        reports.consume(read_episode_batch(packets, count))
        reports.write_summary(expected_completed=min(first+args.batch_size, args.episodes))
        print(f'Episode reports: {reports.completed}/{args.episodes}', flush=True)
    return reports.write_summary(expected_completed=args.episodes, final=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='Saved training run with params.msgpack')
    parser.add_argument('--output', required=True, help='New directory for diagnostic reports')
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1042)
    parser.add_argument('--policy', choices=('argmax', 'sample'), default='argmax')
    run(parser.parse_args())
