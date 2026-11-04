"""Select and export a real GPU replay of a trained focused-map PPO checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from export_battle_viewer import pack
from stoix.envs.number_grid import NumberGrid, action_names, wrap_wall_action_mask
from stoix.networks.base import FeedForwardActor

FIELDS = ('position', 'day', 'gold', 'movement_points', 'hp', 'unit_ids', 'unit_levels', 'unit_xp',
          'enemy_positions', 'alive', 'in_battle', 'enemy', 'actor', 'round', 'done', 'won', 'lost',
          'step_count', 'city_owned', 'city_levels', 'chest_alive', 'ruin_looted', 'scenario_spawned',
          'script_events_fired', 'potions', 'buildings', 'mana', 'last_event', 'last_actor',
          'last_target', 'last_damage', 'last_loot', 'last_item_loot', 'equipped', 'item_inventory',
          'paralyzed', 'long_paralyzed', 'defended', 'escaped', 'poison_mode', 'burn_mode', 'water_mode',
          'imp', 'copied', 'summon_owner', 'fenrir', 'decay_form', 'summon_battle', 'last_cast_spell',
          'last_cast_item', 'last_researched_spell', 'spell_effects', 'enemy_spell_effects',
          'battle_items', 'battle_item_charges', 'battle_item_used', 'item_charges_spent', 'last_battle_item',
          'bot_ids', 'bot_mode', 'bot_timer', 'bot_hero_victories', 'bot_defeated_enemies')


def export(run, map_id, destination, attempts=32):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Replay export requires CUDA')
    status = json.loads((run/'status.json').read_text())
    if status['phase'] != 'complete' or status['training_steps'] != 50_000_000:
        raise ValueError('Focused run must finish exactly 50 million decisions before export')
    source = run/'best' if (run/'best'/'params.msgpack').is_file() else run
    result = json.loads((source/'results.json').read_text())
    hashes = json.loads((source/'source_hashes.json').read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Replay code differs from training: '+name)
    blob = (source/'params.msgpack').read_bytes()
    if hashlib.sha256(blob).hexdigest() != result['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    game_map = json.loads((source/'map.json').read_text())
    config = OmegaConf.load(source/'config.json')
    raw = NumberGrid(map_config=game_map)
    if raw.observation_size != result['observation_size']:
        raise ValueError('Checkpoint observation size does not match the recorded environment')
    env = wrap_wall_action_mask(raw)
    params = jax.tree.map(jnp.asarray, serialization.msgpack_restore(blob)['actor_params'])
    actor = FeedForwardActor(torso=hydra.utils.instantiate(config.network.actor_network.pre_torso),
                            action_head=hydra.utils.instantiate(config.network.actor_network.action_head,
                                                               action_dim=raw.num_actions))

    def project(state):
        return {**{name: getattr(state, name) for name in FIELDS},
                'stats': raw.unit_stats(state), 'experience': raw.unit_experience(state)}

    @jax.jit
    def rollout(seed, greedy):
        state, ts = env.reset(jax.random.PRNGKey(seed))
        initial = project(state)
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
            return (next_state, next_ts, key, total), (project(next_state), action, reward, total, legal)
        (final, _, _, total), frames = jax.lax.scan(advance,
            (state, ts, jax.random.PRNGKey(seed+100), jnp.float32(0)), None, length=raw.max_steps)
        return initial, final, frames, total

    started = time.monotonic()
    candidates, best = [], None
    for i in range(attempts):
        seed = 42+i
        initial, final, (states, actions, rewards, totals, legal), total = jax.device_get(
            rollout(jnp.int32(seed), jnp.bool_(i == 0)))
        steps = int(final.step_count)
        if not bool(final.done) or not np.all(legal[:steps]):
            raise ValueError('Recorded episode is incomplete or contains an illegal action')
        cleared = int(np.sum(~np.asarray(final.alive) & np.asarray(final.scenario_spawned)))
        bot_defeated = int(final.bot_defeated_enemies) if raw.bot.enabled else 0
        defeated = max(0, cleared-bot_defeated)
        city_goals = np.array([c.get('objective', False) for c in game_map.get('cities', [])])
        goal_progress = int(np.sum(np.asarray(final.city_owned)[city_goals])) if city_goals.any() else 0
        outcome = 'victory' if bool(final.won) else 'defeat' if bool(final.lost) else 'timeout'
        row = dict(seed=seed, policy='argmax' if i == 0 else 'sample', steps=steps, day=int(final.day),
                   reward=float(total), defeated=defeated, cleared=cleared, bot_defeated=bot_defeated,
                   goal_progress=goal_progress, outcome=outcome)
        candidates.append(row)
        print(json.dumps(dict(phase='replay', map=map_id, **row)), flush=True)
        rank = (outcome == 'victory', goal_progress, defeated, float(total), -steps)
        if best is None or rank > best[0]:
            best = rank, row, initial, states, actions, rewards, totals
        if outcome == 'victory':
            break
    _, selected, initial, states, actions, rewards, totals = best
    def frame_state(tree, index=None):
        return {name: np.asarray(value if index is None else value[index]).tolist() for name, value in tree.items()}
    frames = [dict(state=frame_state(initial), action=None, reward=0., total_reward=0.)]
    frames += [dict(state=frame_state(states, i), action=int(actions[i]), reward=float(rewards[i]),
                    total_reward=float(totals[i])) for i in range(selected['steps'])]
    tracking = json.loads((run/'tracking.json').read_text())
    metadata = dict(id=map_id, title=game_map.get('display_name', map_id), training_steps=50_000_000,
        checkpoint_steps=result['training_steps'], checkpoint_sha256=result['checkpoint_sha256'],
        source_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        episode=selected, selection_attempts=candidates,
        selection_rule='First victory; otherwise most objective cities captured, then armies defeated by the agent, then highest return',
        checkpoint_selection='128 fixed evaluation episodes per checkpoint; wins, then mean return',
        evaluation=result.get('final_evaluation'), sampled_evaluation=result.get('sampled_evaluation'),
        charts_url=tracking.get('url'), tracking_service=tracking.get('service'), observation_has_day=True,
        export_seconds=round(time.monotonic()-started, 2))
    payload = dict(map=game_map, result=metadata, combat=raw.combat_info, action_names=action_names(game_map),
        construction=raw.construction.metadata(), potions=raw.potion_rules.metadata(),
        equipment=raw.item_rules.metadata, territory=raw.territory.metadata,
        sites=raw.site_rules.metadata if raw.sites_enabled else None,
        ruins=raw.ruins.metadata if raw.ruins_enabled else None,
        spell_casting=raw.casting.metadata, spell_research=raw.spell_research.metadata,
        battle_items=raw.battle_item_rules.metadata if raw.battle_items_enabled else None,
        records=[dict(label=map_id, seed=selected['seed'], policy=selected['policy'], frames=frames)])
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/(map_id+'.json')
    target.write_text(json.dumps(pack(payload), ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    (run/'replay-summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=32)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error('--attempts must be positive')
    export(args.run, args.map, args.destination, args.attempts)
