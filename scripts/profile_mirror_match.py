"""Fixed-input CUDA timings for Mirror Match; no learner updates or rule ablations.

Prepare a corpus once with a saved policy, then reuse it for both implementations.
Each measured call consumes the same states, actions and PRNG. Compilation and
warmup are excluded. Component timings overlap and must not be added together.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from stoix.envs.number_grid import NumberGrid, REST, load_map
from stoix.networks.base import FeedForwardActor


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def load_bot(env, path):
    if path is not None:
        spec = importlib.util.spec_from_file_location('mirror_profile_bot', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env.bot = module.ScriptedBotRules(env.map_config, env.territory, env.progression)


def fixed_input_program(operation):
    def run(states, actions, resting, targets):
        def body(total, inputs):
            output = operation(*inputs)
            checksum = sum(jnp.sum(jnp.asarray(x, jnp.float32)) for x in jax.tree.leaves(output))
            return total + checksum, None
        return jax.lax.scan(body, jnp.float32(0), (states, actions, resting, targets))[0]
    return jax.jit(run)


def prepare(args, env):
    """Record spaced batches from an existing policy, including unfinished episodes."""
    config = OmegaConf.load(args.run / 'config.json')
    actor = FeedForwardActor(
        torso=hydra.utils.instantiate(config.network.actor_network.pre_torso),
        action_head=hydra.utils.instantiate(config.network.actor_network.action_head,
                                          action_dim=env.num_actions))
    checkpoint = args.run / args.checkpoint
    params = jax.tree.map(jnp.asarray, serialization.msgpack_restore(
        checkpoint.read_bytes())['actor_params'])
    fresh, _ = jax.jit(jax.vmap(env.reset))(
        jax.random.split(jax.random.PRNGKey(args.seed), args.batch_size))

    def policy(state, key):
        observation = {'observation': jax.vmap(env.observation)(state),
                       'action_mask': jax.vmap(env.action_mask)(state)}
        return actor.apply(params, observation).sample(seed=key)

    def one(carry, _):
        state, key = carry
        key, draw = jax.random.split(key)
        actions = policy(state, draw)
        after, _ = jax.vmap(env.step)(state, actions)
        after = jax.tree.map(lambda f, x: jnp.where(
            after.done.reshape(after.done.shape + (1,) * (x.ndim - 1)), f, x), fresh, after)
        return (after, key), None

    def sample(carry, _):
        state, key = carry
        actions = policy(state, jax.random.fold_in(key, 991))
        following, _ = jax.lax.scan(one, carry, None, length=args.spacing)
        return following, (state, actions)

    program = jax.jit(lambda s, k: jax.lax.scan(
        sample, (s, k), None, length=args.samples))
    started = time.perf_counter()
    compiled = program.lower(fresh, jax.random.PRNGKey(args.seed + 1)).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    _, (states, actions) = jax.block_until_ready(compiled(
        fresh, jax.random.PRNGKey(args.seed + 1)))
    seconds = time.perf_counter() - started
    metadata = dict(batch_size=args.batch_size, samples=args.samples, spacing=args.spacing,
                    seed=args.seed, checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    map=env.map_config, compile_seconds=compile_seconds,
                    collection_seconds=seconds)
    args.corpus.parent.mkdir(parents=True, exist_ok=True)
    with args.corpus.open('xb') as stream:
        pickle.dump(dict(states=jax.device_get(states), actions=np.asarray(actions),
                         metadata=metadata), stream, protocol=5)
    resting = np.asarray((actions == REST) & ~states.in_battle & ~states.done)
    metadata = {k: v for k, v in metadata.items() if k != 'map'}
    metadata.update(resting_per_sample=resting.sum(axis=1).tolist(),
                    in_battle_per_sample=np.asarray(states.in_battle).sum(axis=1).tolist(),
                    bot_modes={str(i): int(np.sum(np.asarray(states.bot_mode) == i)) for i in range(4)})
    save_json(args.corpus.with_suffix('.json'), metadata)
    print(json.dumps(dict(phase='corpus_saved', **metadata)), flush=True)


def measure(args, env, corpus):
    states = jax.tree.map(jnp.asarray, corpus['states'])
    actions = jnp.asarray(corpus['actions'])
    resting = (actions == REST) & ~states.in_battle & ~states.done
    sample_count, batch = actions.shape
    targets = jnp.maximum(states.bot_target, 0)
    if args.case == 'autobattle':
        navigate = jax.jit(jax.vmap(jax.vmap(
            lambda s, active: env.bot.navigate(env, s, active))))
        states, targets, fights = jax.block_until_ready(navigate(
            states, resting & (states.bot_mode <= 1)))
        resting = resting & fights & (states.bot_mode == 0)
        print(json.dumps(dict(phase='autobattle_inputs', fights_per_sample=np.asarray(
            resting).sum(axis=1).tolist())), flush=True)

    def navigation(state, active):
        return env.bot.navigate(env, state, active)

    cases = {
        'search': lambda s, a, r, t: jax.vmap(env.bot._search)(s, jnp.zeros(batch, bool), r & (s.bot_mode <= 1)),
        'navigation': lambda s, a, r, t: jax.vmap(navigation)(s, r & (s.bot_mode <= 1)),
        'bot_turn': lambda s, a, r, t: jax.vmap(lambda x, active: env.bot.advance(env, x, active))(s, r),
        'autobattle': lambda s, a, r, t: jax.vmap(
            lambda x, target, active: env.bot._autobattle(env, x, target, active))(s, t, r),
        'step': lambda s, a, r, t: jax.vmap(env.step)(s, a),
    }
    selected = ['search', 'navigation', 'bot_turn', 'step'] if args.case == 'all' else [args.case]
    results = dict(device=jax.devices()[0].device_kind, jax=jax.__version__,
                   batch_size=batch, samples=sample_count, corpus=str(args.corpus),
                   corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                   bot_source=str(args.bot_source), timings={})
    for name in selected:
        # Every input sample affects a checksum: prevent hoisting/dead-code removal
        # while avoiding host dispatch between steps of the measured workload.
        program = fixed_input_program(cases[name])
        started = time.perf_counter()
        compiled = program.lower(states, actions, resting, targets).compile()
        compilation = time.perf_counter() - started
        jax.block_until_ready(compiled(states, actions, resting, targets))
        times = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            checksum = jax.block_until_ready(compiled(states, actions, resting, targets))
            times.append(time.perf_counter() - started)
        row = dict(compile_seconds=compilation, seconds=times,
                   median_ms_per_batch=float(np.median(times) * 1000 / sample_count),
                   checksum=float(checksum))
        results['timings'][name] = row
        print(json.dumps(dict(phase='measured', case=name, **row)), flush=True)
        save_json(args.output, results)
        del compiled, program
        jax.clear_caches()


def verify(args, corpus):
    """Require bit-identical complete outputs, including private state and PRNG."""
    states = jax.tree.map(jnp.asarray, corpus['states'])
    actions = jnp.asarray(corpus['actions'])
    rows = {}
    for name in ('navigation', 'step'):
        digests = {}
        for label, source in (('reference', args.verify_against), ('candidate', args.bot_source)):
            env = NumberGrid(map_config=corpus['metadata']['map'])
            load_bot(env, source)
            if name == 'navigation':
                def operation(s, a, game=env):
                    active = (a == REST) & ~s.in_battle & ~s.done & (s.bot_mode <= 1)
                    return jax.vmap(lambda x, enabled: game.bot.navigate(game, x, enabled))(s, active)
            else:
                def operation(s, a, game=env):
                    return jax.vmap(game.step)(s, a)
            first = jax.tree.map(lambda x: x[0], states)
            compiled = jax.jit(operation).lower(first, actions[0]).compile()
            digest = hashlib.sha256()
            values = 0
            for i in range(actions.shape[0]):
                inputs = jax.tree.map(lambda x, index=i: x[index], states)
                output = jax.device_get(compiled(inputs, actions[i]))
                for path, leaf in jax.tree_util.tree_flatten_with_path(output)[0]:
                    leaf = np.asarray(leaf)
                    digest.update(str(path).encode())
                    digest.update(str((leaf.dtype.str, leaf.shape)).encode())
                    digest.update(leaf.tobytes())
                    values += leaf.size
            digests[label] = digest.hexdigest()
            del compiled, operation, env, output
            jax.clear_caches()
            gc.collect()
        row = dict(**digests, equal=digests['reference'] == digests['candidate'],
                   compared_values=values, transitions=int(actions.size))
        rows[name] = row
        save_json(args.output, rows)
        print(json.dumps(dict(phase='verified', case=name, **row)), flush=True)
        if not row['equal']:
            raise AssertionError(f'{name}: full-output digests differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', type=Path, default=Path('results/campaign-five-20261031/mirror_match'))
    parser.add_argument('--checkpoint', default='params.msgpack')
    parser.add_argument('--bot-source', type=Path)
    parser.add_argument('--verify-against', type=Path,
                        help='Compare complete navigation and step outputs against a saved bot source')
    parser.add_argument('--batch-size', type=int, default=250)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--spacing', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--case', choices=('all', 'search', 'navigation', 'bot_turn', 'autobattle', 'step'), default='all')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if jax.default_backend() != 'gpu':
        parser.error('CUDA is required')
    if args.prepare:
        env = NumberGrid(map_config=load_map('mirror_match'))
        load_bot(env, args.bot_source)
        prepare(args, env)
    else:
        if args.output is None:
            parser.error('--output is required for measurements')
        # Only read the locally generated performance corpus, never external pickle files.
        with args.corpus.open('rb') as stream:
            corpus = pickle.load(stream)
        if args.verify_against:
            verify(args, corpus)
        else:
            env = NumberGrid(map_config=corpus['metadata']['map'])
            load_bot(env, args.bot_source)
            measure(args, env, corpus)


if __name__ == '__main__':
    main()
