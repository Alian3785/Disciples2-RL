"""Export a real episode from final baseline weights, including recorded zone commands."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch', type=Path, required=True)
parser.add_argument('--map', required=True)
parser.add_argument('--destination', type=Path, required=True)
parser.add_argument('--attempts', type=int, default=32)
args = parser.parse_args()
if not 1 <= args.attempts <= 128:
    parser.error('attempts must be between 1 and 128')
batch = args.batch.resolve()
source = batch / 'source'
sys.path.insert(0, str(source))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf
from export_battle_viewer import pack
from scripts.export_focused_replay import FIELDS
from stoix.envs.number_grid import NumberGrid, action_names, wrap_wall_action_mask
from stoix.networks.base import FeedForwardActor

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def export():
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Replay export requires CUDA')
    series = read(batch.parent / 'baseline-six-maps-20261103.json')
    training = next(row for row in series['maps'] if row['map'] == args.map)
    run = batch / ('trade_train-continuation' if args.map == 'trade_train' else args.map)
    result, status = read(run / 'results.json'), read(run / 'status.json')
    assert status['phase'] == 'complete' and result['training_steps'] == training['training_steps']
    manifest = read(batch / 'manifest.json')
    for name, expected in manifest['source_hashes'].items():
        assert digest(source / name) == expected, 'Frozen source changed: ' + name
    hashes = read(run / 'source_hashes.json')
    for name, expected in hashes.items():
        assert digest(source / name) == expected, 'Training source mismatch: ' + name
    blob = (run / 'params.msgpack').read_bytes()
    assert hashlib.sha256(blob).hexdigest() == result['checkpoint_sha256']
    game_map, config = read(run / 'map.json'), OmegaConf.load(run / 'config.json')
    raw = NumberGrid(map_config=game_map)
    assert raw.observation_size == result['observation_size']
    env = wrap_wall_action_mask(raw)
    actor = FeedForwardActor(torso=hydra.utils.instantiate(config.network.actor_network.pre_torso),
        action_head=hydra.utils.instantiate(config.network.actor_network.action_head, action_dim=raw.num_actions))
    params = jax.tree.map(jnp.asarray, serialization.msgpack_restore(blob)['actor_params'])
    objective = raw.objective_enemy

    def project(state, ts):
        stats = raw.unit_stats(state)
        destinations = state.position[None] + raw.directions
        if raw.territory_enabled:
            direct = raw.territory.enemy_at(state, destinations) >= 0
        else:
            direct = jnp.any(state.alive[None] & jnp.all(
                destinations[:, None] == state.enemy_positions[None], axis=-1), axis=1)
        enters = jax.vmap(lambda d: raw.zones.resolve(state, d, jnp.bool_(True))[0])(destinations) & ~direct
        damage = jnp.float32(0)
        reached = jnp.bool_(False)
        if objective is not None:
            reached = state.in_battle & (state.enemy == objective)
            maximum = jnp.sum(stats[6:, 0])
            damage = jnp.where(reached, jnp.clip(1 - jnp.sum(state.hp[6:]) / jnp.maximum(maximum, 1), 0, 1), 0)
            damage = jnp.where(state.scenario_spawned[objective] & ~state.alive[objective], 1., damage)
        return {**{name: getattr(state, name) for name in FIELDS}, 'stats': stats,
            'experience': raw.unit_experience(state), 'map_commands': raw.map_commands(state),
            'move_enters': enters, 'move_direct': direct,
            'move_legal': ts.observation['action_mask'][:8] & ~state.in_battle & ~state.done,
            'zone_holders': raw.zones.holders(state), 'objective_damage': damage, 'objective_reached': reached}

    @jax.jit
    def rollout(seed, greedy):
        state, ts = env.reset(jax.random.PRNGKey(seed))
        initial = project(state, ts)
        def advance(carry, unused):
            state, ts, key, total = carry
            key, subkey = jax.random.split(key)
            def active(_):
                dist = actor.apply(params, jax.tree.map(lambda x: x[None], ts.observation))
                action = jnp.where(greedy, dist.mode()[0], dist.sample(seed=subkey)[0])
                legal = ts.observation['action_mask'][action]
                next_state, next_ts = env.step(state, action)
                return next_state, next_ts, action, next_ts.reward, legal
            next_state, next_ts, action, reward, legal = jax.lax.cond(state.done,
                lambda _: (state, ts, jnp.int32(-1), jnp.float32(0), jnp.bool_(True)), active, None)
            total += reward
            return (next_state, next_ts, key, total), (project(next_state, next_ts), action, reward, total, legal)
        (final, _, _, total), frames = jax.lax.scan(advance,
            (state, ts, jax.random.PRNGKey(seed + 100), jnp.float32(0)), None, length=raw.max_steps)
        return initial, final, frames, total

    started = time.monotonic()
    candidates, best = [], None
    print(json.dumps(dict(phase='compiling_replay', map=args.map, budget=result['training_steps'])), flush=True)
    for attempt in range(args.attempts):
        seed = 42 + attempt
        initial, final, (states, actions, rewards, totals, legal), total = jax.device_get(
            rollout(jnp.int32(seed), jnp.bool_(attempt == 0)))
        steps = int(final.step_count)
        assert bool(final.done) and np.all(legal[:steps]), 'Incomplete or illegal episode'
        defeated = int(np.sum(~np.asarray(final.alive) & np.asarray(final.scenario_spawned)))
        goal_cities = np.array([city.get('objective', False) for city in game_map.get('cities', [])], bool)
        cities = int(np.sum(np.asarray(final.city_owned)[goal_cities])) if goal_cities.any() else 0
        reached = bool(np.any(states['objective_reached'][:steps]))
        damage = float(np.max(states['objective_damage'][:steps], initial=0))
        outcome = 'victory' if bool(final.won) else 'defeat' if bool(final.lost) else 'timeout'
        row = dict(seed=seed, policy='argmax' if attempt == 0 else 'sample', steps=steps,
            day=int(final.day), reward=float(total), defeated=defeated, goal_progress=cities,
            objective_reached=reached, objective_damage_fraction=damage, outcome=outcome)
        candidates.append(row)
        rank = (outcome == 'victory', cities, damage, reached, defeated, float(total), -steps)
        if best is None or rank > best[0]:
            best = rank, row, initial, states, actions, rewards, totals
        print(json.dumps(dict(phase='candidate', map=args.map, **row)), flush=True)
        if outcome == 'victory':
            break
    _, selected, initial, states, actions, rewards, totals = best
    def frame_state(tree, index=None):
        return {name: np.asarray(value if index is None else value[index]).tolist() for name, value in tree.items()}
    frames = [dict(state=frame_state(initial), action=None, reward=0., total_reward=0.)]
    frames += [dict(state=frame_state(states, i), action=int(actions[i]), reward=float(rewards[i]),
                    total_reward=float(totals[i])) for i in range(selected['steps'])]
    interactions = {'direct': 0, 'zone_enter': 0, 'zone_held': 0}
    for previous, current in zip(frames, frames[1:]):
        before, after, action = previous['state'], current['state'], current['action']
        if before['in_battle'] or not 0 <= action < 8:
            continue
        target, cost = before['map_commands'][action]
        if target < 0:
            continue
        destination = [a + b for a, b in zip(before['position'], raw.directions.tolist()[action])]
        direct = before['move_direct'][action]
        enters = before['move_enters'][action]
        kind = 'direct' if direct else 'zone_enter' if enters else 'zone_held'
        assert after['position'] == (destination if enters else before['position']), 'Zone movement mismatch'
        assert before['chest_alive'] == after['chest_alive'], 'Chest opened during battle entry'
        current['engagement'] = dict(kind=kind, enemy=target, destination=destination, cost=cost)
        interactions[kind] += 1
    summary = {key: value for key, value in training.items() if key not in ('run',)}
    metadata = dict(id=args.map, title=training['title'], training_steps=result['training_steps'],
        checkpoint_steps=result['training_steps'], checkpoint_sha256=result['checkpoint_sha256'],
        source_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        zone_source_sha256=digest(source / 'stoix/envs/number_grid_zones.py'), episode=selected,
        selection_attempts=candidates, selection_rule='First victory; otherwise objective cities, target damage/reached, armies defeated, return',
        checkpoint_selection='Final weights from the completed 2026-11-03 baseline series',
        evaluation=None, sampled_evaluation=None, training_summary=summary, engagement_counts=interactions,
        charts_url=training['charts_url'], tracking_service='comet', observation_has_day=True,
        export_seconds=round(time.monotonic()-started, 2))
    payload = dict(map=game_map, result=metadata, combat=raw.combat_info, action_names=action_names(game_map),
        construction=raw.construction.metadata(), potions=raw.potion_rules.metadata(),
        equipment=raw.item_rules.metadata, territory=raw.territory.metadata,
        sites=raw.site_rules.metadata if raw.sites_enabled else None,
        ruins=raw.ruins.metadata if raw.ruins_enabled else None,
        spell_casting=raw.casting.metadata, spell_research=raw.spell_research.metadata,
        battle_items=raw.battle_item_rules.metadata if raw.battle_items_enabled else None,
        zones={**raw.zones.metadata, 'inside': raw.zones.inside.tolist(), 'forts': raw.zones.forts.tolist(),
               'directions': raw.directions.tolist(), 'source_sha256': metadata['zone_source_sha256']},
        records=[dict(label=args.map, seed=selected['seed'], policy=selected['policy'], frames=frames)])
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / (args.map + '.json')).write_text(
        json.dumps(pack(payload), ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    (args.destination / (args.map + '-summary.json')).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(phase='exported', map=args.map, episode=selected, engagements=interactions)), flush=True)

export()
