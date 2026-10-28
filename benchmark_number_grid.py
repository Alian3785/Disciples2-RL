"""Measured local run of the unchanged Stoix Anakin feedforward PPO learner."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import time

os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf
from stoix.systems.ppo.anakin.ff_ppo import learner_setup
from stoix.utils.make_env import make
from stoix.envs.number_grid import MAP, NumberGrid, load_map, wrap_wall_action_mask
from numbergrid_config import ROOT, make_config
from numbergrid_tracking import NumberGridTracking, add_tracking_arguments
from stoix.envs.number_grid_metrics import summarize_campaign
from stoix.envs.number_grid_episode_reports import compact_episode_events
from numbergrid_episode_reports import read_episode_batch
from numbergrid_async_episode_reports import AsyncEpisodeReports
from numbergrid_warm_start import prepare_warm_start


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def unreplicate(tree):
    return jax.tree.map(lambda x: x[0, 0], tree)


def make_evaluator(actor, episodes=1024, seed=1042, greedy=True, map_config=None):
    """Compile once and evaluate multiple checkpoints on the same episodes."""
    base_env = NumberGrid(map_config=map_config)
    env = wrap_wall_action_mask(base_env)
    target = base_env.objective_enemy
    track_damage = target is not None and base_env.spell_casting_enabled
    @jax.jit
    def run(p):
        states, ts = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(seed), episodes))
        target_hp = (jnp.sum(jax.vmap(base_env.casting.enemy_max_hp)(states)[:, target], axis=-1)
                     if track_damage else jnp.ones(episodes))
        def condition(carry):
            return ~jnp.all(carry[0].done)
        def advance(carry):
            states, ts, returns, key, reached, damage = carry
            key, subkey = jax.random.split(key)
            dist = actor.apply(p, ts.observation)
            actions = dist.mode() if greedy else dist.sample(seed=subkey)
            states, ts = jax.vmap(env.step)(states, actions)
            if target is not None:
                reached |= states.enemy == target
            if track_damage:
                wounds = jnp.sum(states.enemy_wounds[:, target], axis=-1)
                in_target_battle = states.in_battle & (states.enemy == target)
                remaining = jnp.where(in_target_battle, jnp.sum(states.hp[:, 6:], axis=-1), target_hp-wounds)
                fraction = jnp.clip(1.-remaining/jnp.maximum(1, target_hp), 0., 1.)
                fraction = jnp.where(states.alive[:, target], fraction, 1.)
                damage = jnp.maximum(damage, fraction)
            return states, ts, returns + ts.reward, key, reached, damage
        states, ts, returns, _, reached, damage = jax.lax.while_loop(
            condition, advance, (states, ts, jnp.zeros(episodes), jax.random.PRNGKey(seed + 1),
                                 jnp.zeros(episodes, bool), jnp.zeros(episodes)))
        return (states.won, states.step_count, returns, reached, damage,
                jnp.sum(~states.alive & states.scenario_spawned, axis=-1), states.lost)
    def evaluate_params(params):
        won, lengths, returns, reached, damage, defeated, lost = jax.device_get(run(params))
        return dict(episodes=episodes, successes=int(won.sum()), success_rate=float(won.mean()),
                    mean_return=float(returns.mean()), mean_length=float(lengths.mean()),
                    min_length=int(lengths.min()), max_length=int(lengths.max()),
                    defeats=int(lost.sum()), timeouts=int((~won & ~lost).sum()),
                    target_reached_episodes=int(reached.sum()),
                    mean_target_damage_fraction=float(damage.mean()),
                    maximum_target_damage_fraction=float(damage.max()),
                    mean_defeated_armies=float(defeated.mean()), max_defeated_armies=int(defeated.max()),
                    action_selection='argmax' if greedy else 'sample', seed=seed)
    return evaluate_params


def save_checkpoint(output, params, steps, state=None):
    directory = output / 'checkpoints'
    directory.mkdir(exist_ok=True)
    checkpoint = serialization.to_bytes(params)
    path = directory / f'{steps:09d}.msgpack'
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(checkpoint)
    temporary.replace(path)
    record = dict(training_steps=steps, path=str(path.relative_to(output)),
                  sha256=hashlib.sha256(checkpoint).hexdigest())
    if state is not None:
        learner = serialization.to_bytes(state)
        learner_path = directory / f'{steps:09d}.learner.msgpack'
        temporary = learner_path.with_suffix('.tmp')
        temporary.write_bytes(learner)
        temporary.replace(learner_path)
        record.update(learner_state_path=str(learner_path.relative_to(output)),
                      learner_state_sha256=hashlib.sha256(learner).hexdigest())
    return record


def checkpoint_score(candidate):
    """Actual map wins first; return breaks ties, then prefer an earlier checkpoint."""
    evaluation = candidate['evaluation']
    return (evaluation['success_rate'], evaluation['mean_return'], -candidate['training_steps'])


def select_best_checkpoint(output, candidates, actor, params, game_map, result,
                           final_evaluator, sampled_evaluator, tracking, status):
    started = time.perf_counter()
    evaluate_candidate = make_evaluator(actor, episodes=128, seed=2042, map_config=game_map)
    records = []
    for candidate in candidates:
        checkpoint = (output / candidate['path']).read_bytes()
        if hashlib.sha256(checkpoint).hexdigest() != candidate['sha256']:
            raise ValueError('Intermediate checkpoint hash mismatch')
        restored = serialization.from_bytes(params, checkpoint)
        evaluation = evaluate_candidate(restored.actor_params)
        record = {**candidate, 'evaluation': evaluation}
        records.append(record)
        save_json(output / 'checkpoint_selection.json', records)
        status('selecting', checkpoint_steps=candidate['training_steps'],
               checked=len(records), candidates=len(candidates), evaluation=evaluation)
        tracking.log({'training_steps': candidate['training_steps'],
                      'selection/win_rate': evaluation['success_rate'],
                      'selection/return_mean': evaluation['mean_return']})
    best = max(records, key=checkpoint_score)
    checkpoint = (output / best['path']).read_bytes()
    restored = serialization.from_bytes(params, checkpoint)
    if best['training_steps'] == result['training_steps']:
        final, sampled = result['final_evaluation'], result['sampled_evaluation']
    else:
        final = final_evaluator(restored.actor_params)
        sampled = sampled_evaluator(restored.actor_params)
    destination = output / 'best'
    destination.mkdir()
    (destination / 'params.msgpack').write_bytes(checkpoint)
    for name in ('config.json', 'map.json', 'source_hashes.json', 'packages.txt'):
        shutil.copyfile(output / name, destination / name)
    best_result = {key: result[key] for key in (
        'environment', 'algorithm', 'recurrent', 'device', 'backend', 'jax', 'cuda_runtime',
        'python', 'seed', 'mean_steps_per_second', 'parameter_count', 'num_envs',
        'rollout_length', 'observation_size', 'learner_source_sha256', 'source_hashes', 'tracking')}
    best_result.update(training_steps=best['training_steps'], checkpoint_sha256=best['sha256'],
                       final_evaluation=final, sampled_evaluation=sampled,
                       selection_evaluation=best['evaluation'], source_run=str(output),
                       source_run_training_steps=result['training_steps'],
                       timing='Learner speed belongs to the full source run; selected weights are from training_steps.')
    save_json(destination / 'results.json', best_result)
    tracking.log({'training_steps': result['training_steps'],
                  'best/checkpoint_steps': best['training_steps'],
                  'best/argmax_win_rate': final['success_rate'],
                  'best/sample_win_rate': sampled['success_rate']})
    return dict(path='best', training_steps=best['training_steps'],
                selection_episodes=128, selection_seed=2042, candidates=len(records),
                final_evaluation=final, sampled_evaluation=sampled,
                selection_seconds=time.perf_counter()-started)


def train(args):
    training_budget(args)  # Reject oversized test runs before tracking or CUDA setup.
    started = time.perf_counter()
    with NumberGridTracking(args) as tracking:
        result = run_training(args, tracking)
    result['tracking'] = tracking.metadata.copy()
    if result.get('best_checkpoint'):
        best_path = tracking.output / result['best_checkpoint']['path'] / 'results.json'
        best_result = json.loads(best_path.read_text())
        best_result['tracking'] = result['tracking']
        save_json(best_path, best_result)
    result['total_runner_seconds'] = time.perf_counter() - started
    save_json(tracking.output / 'results.json', result)
    save_json(tracking.output / 'status.json', dict(phase='complete', **result))
    if result['tracking']['url']:
        print('Charts: ' + result['tracking']['url'], flush=True)
    return result


def training_budget(args):
    """The budget counts transitions across all parallel environments together."""
    testing = (getattr(args, 'test_run', False)
               or os.environ.get('NUMBERGRID_TEST_RUN') == '1'
               or 'PYTEST_CURRENT_TEST' in os.environ)
    total = 250_000 if args.smoke else args.total_timesteps
    if total is None:
        total = 1_000_000 if testing else 5_000_000
    if testing and total > 1_000_000:
        raise ValueError('Test training is limited to 1,000,000 total transitions per run.')
    return total


def run_training(args, tracking):
    started = time.perf_counter()
    total = training_budget(args)
    seed = secrets.randbits(31) if args.seed is None else args.seed
    print(f'Training seed: {seed}', flush=True)
    if os.environ.get('NUMBERGRID_ALLOW_CPU') != '1' and (len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu'):
        raise RuntimeError('This measured profile requires exactly one CUDA GPU.')
    game_map = load_map(args.map) if args.map else MAP
    if args.no_unit_swaps:
        # Training without party swaps: the map then has no swap actions at all (docs/UNITS.md).
        game_map = {**game_map, 'unit_swaps': False}
    config = make_config(total, seed, map_config=game_map)
    initial_steps, warm_start, warm_bytes = 0, None, None
    if getattr(args, 'reward_transfer', None) and not getattr(args, 'warm_start', None):
        raise ValueError('--reward-transfer requires --warm-start')
    if getattr(args, 'warm_start', None):
        warm_bytes, warm_start = prepare_warm_start(
            args.warm_start, total, game_map, OmegaConf.to_container(config, resolve=True), ROOT,
            reward_transfer=getattr(args, 'reward_transfer', None))
        initial_steps = warm_start['initial_training_steps']
        config = make_config(total-initial_steps, seed, map_config=game_map)
        for name, rate in warm_start['learning_rates'].items():
            config.system[name] = rate
        config.warm_start = warm_start
    session_total = total-initial_steps
    learner_block = config.arch.num_updates_per_eval * config.arch.num_envs * config.system.rollout_length
    if tracking.summary_interval % learner_block:
        raise ValueError(f'summary_interval must be a multiple of the learner block ({learner_block})')
    save_checkpoints = getattr(args, 'save_checkpoints', True) or getattr(args, 'select_best', False)
    checkpoint_interval = getattr(args, 'checkpoint_interval', 5_000_000)
    if checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be positive')
    if save_checkpoints and checkpoint_interval % learner_block:
        raise ValueError(f'checkpoint_interval must be a multiple of the learner block ({learner_block})')
    config.checkpointing = dict(enabled=save_checkpoints,
                               interval=checkpoint_interval if save_checkpoints else None)
    config.env.kwargs.episode_reports = bool(game_map.get('battle_mode')
        and game_map.get('unit_progression') and getattr(args, 'episode_reports', False)
        and not getattr(args, 'no_episode_reports', False))
    run_name = f"number_grid-{game_map['size']}x{game_map['size']}" + ('-step-cost' if game_map.get('step_cost', 0) else '') + ('-smoke' if args.smoke else '')
    if game_map.get('objective_enemy') is not None:
        run_name += '-'+game_map.get('campaign_objective', 'target-enemy')
    if game_map.get('exploration_bonus', 0):
        run_name += '-exploration'
    if game_map.get('mask_walls', False):
        run_name += '-wall-mask'
    if game_map.get('battle_mode', False):
        run_name += '-archer-battles'
    if args.no_unit_swaps:
        run_name += '-no-unit-swaps'
    output = Path(args.output) if args.output else ROOT / 'results' / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already contains files; choose another --output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    measured_sources = ['benchmark_number_grid.py', 'numbergrid_config.py', 'numbergrid_tracking.py',
                        'numbergrid_episode_reports.py', 'numbergrid_async_episode_reports.py',
                        'stoix/envs/number_grid_episode_schema.py',
                        'stoix/envs/number_grid_episode_reports.py',
                        'stoix/envs/number_grid_scenarios.py', 'stoix/envs/number_grid_reference_rewards.py',
                        'stoix/envs/number_grid_script_events.py',
                        'stoix/envs/number_grid_hero_skills.py',
                        'stoix/envs/number_grid_capital_training.py',
                        'stoix/envs/number_grid_target_rewards.py',
                        'numbergrid_campaign.py', 'numbergrid_comet_dashboard.py', 'numbergrid_wandb_dashboard.py',
                        'stoix/envs/number_grid_metrics.py',
                        'stoix/envs/number_grid_buildings.py', 'stoix/envs/data/buildings.json',
                        'stoix/envs/number_grid_spells.py', 'stoix/envs/number_grid_casting.py', 'stoix/envs/number_grid_unit_swaps.py', 'stoix/envs/number_grid_lords.py', 'stoix/envs/data/spells.json',
                        'stoix/envs/data/units.json', 'stoix/envs/number_grid_effects.py', 'stoix/envs/number_grid_wards.py', 'stoix/envs/number_grid_summoning.py',
                        'stoix/envs/number_grid.py', 'stoix/envs/number_grid_combat.py', 'stoix/envs/number_grid_progression.py', 'stoix/envs/number_grid_capital.py', 'stoix/envs/number_grid_potions.py', 'stoix/envs/number_grid_chests.py', 'stoix/envs/number_grid_items.py', 'stoix/envs/number_grid_item_combat.py', 'stoix/envs/number_grid_sites.py', 'stoix/envs/number_grid_recruitment.py', 'stoix/envs/number_grid_territory.py', 'stoix/envs/number_grid_ruins.py', 'stoix/envs/number_grid_terrain.py', 'stoix/envs/data/training_prices.json', 'stoix/envs/number_grid_legacy.py',
                        'stoix/utils/make_env.py', 'stoix/wrappers/number_grid_metrics.py',
                        'stoix/wrappers/number_grid_reset.py', 'stoix/systems/ppo/anakin/ff_ppo.py']
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in measured_sources if (ROOT / name).exists()}
    source_hashes['numbergrid_warm_start.py'] = hashlib.sha256((ROOT / 'numbergrid_warm_start.py').read_bytes()).hexdigest()
    save_json(output / 'source_hashes.json', source_hashes)
    save_json(output / 'config.json', OmegaConf.to_container(config, resolve=True))
    save_json(output / 'map.json', game_map)
    if warm_start is not None:
        save_json(output / 'warm_start.json', warm_start)
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (output / 'packages.txt').write_text('\n'.join(packages) + '\n')
    tracking.start(output, OmegaConf.to_container(config, resolve=True), game_map, source_hashes)
    def status(phase, **kwargs):
        record = dict(phase=phase, **kwargs)
        save_json(output / 'status.json', record)
        print(json.dumps(record), flush=True)
    status('initializing', device=str(jax.devices()[0]), total_timesteps=total)
    env, _ = make(config)
    keys = tuple(jax.random.split(jax.random.PRNGKey(seed), 3))
    learner, actor, state = learner_setup(env, keys, config)
    if warm_bytes is not None:
        template = unreplicate(state.params)
        restored = serialization.from_bytes(template, warm_bytes)
        for expected, actual in zip(jax.tree.leaves(template), jax.tree.leaves(restored)):
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError('Warm-start parameter shape or dtype mismatch')
            if not np.isfinite(np.asarray(actual)).all():
                raise FloatingPointError('Non-finite warm-start parameter')
        params = jax.tree.map(lambda reference, value: jnp.broadcast_to(
            jnp.asarray(value), reference.shape), state.params, restored)
        state = state._replace(params=params)
        print(f'Loaded checkpoint at {initial_steps} steps; optimizer and episodes reset', flush=True)
    initial_params = jax.device_get(unreplicate(state.params))
    status('compiling')
    def chunk(s):
        result = learner(s)
        losses = {k: jnp.mean(v) for k, v in result.train_metrics.items()}
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in result.train_metrics.values()]))
        episodes = result.episode_metrics
        mask = episodes['is_terminal_step']
        count = jnp.sum(mask)
        wins = jnp.sum(episodes['episode_success'] & mask)
        losses.update(
            all_finite=finite, episodes=count, episode_wins=wins,
            mean_episode_return=jnp.sum(episodes['episode_return'] * mask) / jnp.maximum(count, 1),
            mean_episode_length=jnp.sum(episodes['episode_length'] * mask) / jnp.maximum(count, 1),
            # Exploration can increase return without a win: use the actual flag.
            episode_success_rate=wins / jnp.maximum(count, 1),
        )
        for name in ('battle_transition', 'player_battle_transition', 'enemy_battle_transition', 'automatic_battle_transition', 'battle_victory', 'building_constructed', 'turn_ended', 'rest_penalty'):
            if name in episodes:
                losses[name] = jnp.sum(episodes[name])
        if 'campaign_events' in episodes:
            losses['campaign'] = summarize_campaign(episodes)
        packets = None
        if 'episode_event' in episodes:
            losses['episode_report_events'], packets = compact_episode_events(episodes)
        return result.learner_state, losses, packets
    jax.block_until_ready(state)
    begin_compile = time.perf_counter()
    compiled = jax.jit(chunk).lower(state).compile()
    compile_seconds = time.perf_counter() - begin_compile
    step_size = config.arch.num_updates_per_eval * config.arch.num_envs * config.system.rollout_length
    rows = []
    training_seconds = 0.0
    checkpoints = []
    intermediate_checkpoint_seconds = 0.0
    episode_report_seconds = 0.0
    reported_expected = 0
    report_summary = None
    status('training', training_steps=initial_steps, total_timesteps=total, session_training_steps=0)
    report_worker = (AsyncEpisodeReports(output, game_map, env.combat_info)
                     if config.env.kwargs.episode_reports else None)
    if report_worker is not None:
        episode_report_seconds += report_worker.startup_seconds
    with report_worker if report_worker is not None else nullcontext() as reports:
        loop_started = time.perf_counter()
        for i in range(session_total // step_size):
            begin = time.perf_counter()
            state, metrics, packets = compiled(state)
            jax.block_until_ready((state, metrics, packets))
            elapsed = time.perf_counter() - begin
            training_seconds += elapsed
            host_metrics = jax.device_get(metrics)
            if reports is not None:
                report_started = time.perf_counter()
                count = int(host_metrics.pop('episode_report_events'))
                # One count and one populated batch per learner block, never per step.
                batch = read_episode_batch(packets, count)
                reported_expected += int(host_metrics['episodes'])
                reports.submit(batch, transition_base=initial_steps + i * step_size,
                               expected_completed=reported_expected)
                episode_report_seconds += time.perf_counter() - report_started
            campaign = host_metrics.pop('campaign', None)
            if campaign is not None:
                tracking.campaign.add(campaign)
            row = {k: float(v) for k, v in host_metrics.items()}
            if not row['all_finite']:
                raise FloatingPointError('Non-finite learner metric')
            row.update(training_steps=initial_steps + (i + 1) * step_size,
                       session_training_steps=(i + 1) * step_size, training_seconds=training_seconds)
            rows.append(row)
            summary = tracking.log_training(row, step_size, elapsed,
                                            time.perf_counter() - loop_started,
                                            final=(i + 1) * step_size == session_total)
            if summary is not None:
                if reports is not None:
                    report_started = time.perf_counter()
                    reports.write_summary(expected_completed=reported_expected)
                    episode_report_seconds += time.perf_counter() - report_started
                status('training', training_steps=summary['training_steps'], total_timesteps=total,
                       session_training_steps=(i + 1) * step_size,
                       mean_steps_per_second=(i + 1) * step_size/training_seconds,
                       mean_episode_return=summary['mean_episode_return'],
                       episode_success_rate=summary['episode_success_rate'],
                       mean_episode_length=summary['mean_episode_length'],
                       episodes=int(summary['episodes']), episode_wins=int(summary['episode_wins']))
            completed_steps = initial_steps + (i + 1) * step_size
            if save_checkpoints and (completed_steps % checkpoint_interval == 0 or completed_steps == total):
                checkpoint_started = time.perf_counter()
                checkpoints.append(save_checkpoint(output, unreplicate(state.params),
                                                   completed_steps, state=state))
                save_json(output / 'checkpoints.json', checkpoints)
                intermediate_checkpoint_seconds += time.perf_counter() - checkpoint_started
        if reports is not None:
            report_started = time.perf_counter()
            report_summary = reports.finish()
            episode_report_seconds += time.perf_counter() - report_started
        training_loop_seconds = time.perf_counter() - loop_started
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status('saving', training_steps=total, training_seconds=training_seconds)
    params = unreplicate(state.params)
    for leaf in jax.tree.leaves(params):
        if not np.isfinite(np.asarray(leaf)).all():
            raise FloatingPointError('Non-finite parameter')
    changed = any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial_params), jax.tree.leaves(params)))
    if not changed:
        raise AssertionError('PPO did not update weights')
    expected_updates = config.arch.num_updates * config.system.epochs * config.system.num_minibatches
    actual_actor_updates = int(np.asarray(state.opt_states.actor_opt_state[1][0].count).reshape(-1)[0])
    actual_critic_updates = int(np.asarray(state.opt_states.critic_opt_state[1][0].count).reshape(-1)[0])
    if actual_actor_updates != expected_updates or actual_critic_updates != expected_updates:
        raise AssertionError('Unexpected optimizer update count')
    checkpoint_started = time.perf_counter()
    checkpoint = serialization.to_bytes(params)
    (output / 'params.msgpack').write_bytes(checkpoint)
    (output / 'learner_state.msgpack').write_bytes(serialization.to_bytes(state))
    restored = serialization.from_bytes(params, (output / 'params.msgpack').read_bytes())
    if not all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored))):
        raise AssertionError('Checkpoint roundtrip failed')
    checkpoint_seconds = time.perf_counter()-checkpoint_started
    evaluate = getattr(args, 'evaluate', False) or getattr(args, 'select_best', False)
    final = sampled = None
    final_evaluation_seconds = sampled_evaluation_seconds = 0.0
    if evaluate:
        status('evaluating', training_steps=total, training_seconds=training_seconds)
        final_eval_started = time.perf_counter()
        final_evaluator = make_evaluator(actor, map_config=game_map)
        final = final_evaluator(restored.actor_params)
        final_evaluation_seconds = time.perf_counter()-final_eval_started
        sample_eval_started = time.perf_counter()
        sampled_evaluator = make_evaluator(actor, greedy=False, map_config=game_map)
        sampled = sampled_evaluator(restored.actor_params)
        sampled_evaluation_seconds = time.perf_counter()-sample_eval_started
        tracking.evaluation(final, total)
        tracking.evaluation(sampled, total)
    source = ROOT / 'stoix/systems/ppo/anakin/ff_ppo.py'
    result = dict(
        environment=game_map['name'], algorithm='Stoix Anakin feedforward PPO', recurrent=False,
        device=jax.devices()[0].device_kind, backend=jax.default_backend(), jax=jax.__version__,
        cuda_runtime=(importlib.metadata.version('nvidia-cuda-runtime') if jax.default_backend() == 'gpu' else None), python=platform.python_version(),
        training_steps=total, initial_training_steps=initial_steps,
        session_training_steps=session_total, warm_start=warm_start,
        ppo_updates=config.arch.num_updates,
        actor_optimizer_updates=actual_actor_updates, critic_optimizer_updates=actual_critic_updates,
        seed=seed, training_seconds=training_seconds,
        mean_steps_per_second=session_total/training_seconds, ppo_compile_seconds=compile_seconds,
        training_loop_seconds=training_loop_seconds,
        training_loop_steps_per_second=session_total/training_loop_seconds,
        tracking_logging_seconds=tracking.logging_seconds,
        # Parent blocking time and worker work overlap; never add worker time to wall time.
        episode_report_seconds=episode_report_seconds,
        episode_report_mode='async_process' if report_worker is not None else 'disabled',
        episode_report_worker_seconds=report_worker.worker_seconds if report_worker else 0.0,
        episode_report_queue_wait_seconds=report_worker.queue_wait_seconds if report_worker else 0.0,
        episode_report_drain_seconds=report_worker.drain_seconds if report_worker else 0.0,
        episode_report_startup_seconds=report_worker.startup_seconds if report_worker else 0.0,
        episode_reports=(None if report_summary is None else {
            **{key: report_summary[key] for key in ('schema_version', 'completed_episodes',
                'pending_episodes', 'events', 'outcomes', 'mean_awarded_xp')},
            'episodes_file': report_summary['episodes_file'],
            'episodes_file_bytes': report_summary['episodes_file_bytes'],
            'episodes_uncompressed_bytes': report_summary['episodes_uncompressed_bytes'],
            'summary_file': 'episode_reports.summary.json',
            'readable_summary_file': 'episode_reports.md'}),
        tracking_init_seconds=tracking.metadata['init_seconds'],
        completed_training_episodes=tracking.episodes, won_training_episodes=tracking.wins,
        tracking=tracking.metadata.copy(),
        evaluation_status='completed' if evaluate else 'not_requested',
        final_evaluation=final, sampled_evaluation=sampled,
        final_evaluation_seconds=final_evaluation_seconds,
        sampled_evaluation_seconds=sampled_evaluation_seconds,
        checkpoint_seconds=checkpoint_seconds,
        intermediate_checkpoint_seconds=intermediate_checkpoint_seconds,
        summary_interval=tracking.summary_interval,
        checkpoint_interval=(checkpoint_interval if save_checkpoints else None),
        parameter_count=sum(x.size for x in jax.tree.leaves(params)),
        num_envs=config.arch.num_envs, rollout_length=config.system.rollout_length,
        observation_size=int((env.observation_space().spaces['observation']
                              if hasattr(env.observation_space(), 'spaces')
                              else env.observation_space()).shape[0]),
        weights_changed=changed, checkpoint_roundtrip_verified=True,
        initial_checkpoint_sha256=hashlib.sha256(serialization.to_bytes(initial_params)).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        learner_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        total_runner_seconds=time.perf_counter()-started,
        timing='Sum of synchronized learner calls, including rollout and PPO updates; excludes compilation, evaluation, host logging and saving.',
        reconstruction='Fixed map and observation encoding recreated from README; remote weights and results were not recovered.',
    )
    if game_map.get('battle_mode'):
        result['combat_counts'] = {name: int(sum(row[name] for row in rows)) for name in (
            'battle_transition', 'player_battle_transition', 'enemy_battle_transition', 'automatic_battle_transition', 'battle_victory')}
        result['construction_count'] = int(sum(row.get('building_constructed', 0) for row in rows))
        result['rest_count'] = int(sum(row.get('turn_ended', 0) for row in rows))
        result['rest_penalty_total'] = float(sum(row.get('rest_penalty', 0) for row in rows))
        result['transition_definition'] = 'One agent decision, including all automatic enemy turns, skips and completed retreats up to the next decision. Internal activations spend no additional step budget or per-step penalties.'
    result['source_hashes'] = source_hashes
    result['campaign_metrics'] = tracking.last_campaign_metrics
    save_json(output / 'results.json', result)
    if getattr(args, 'select_best', False):
        result['best_checkpoint'] = select_best_checkpoint(
            output, checkpoints, actor, params, game_map, result,
            final_evaluator, sampled_evaluator, tracking, status)
        result['total_runner_seconds'] = time.perf_counter() - started
    save_json(output / 'results.json', result)
    tracking.summary(result)
    status('complete', **result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='250,000 transitions, separate output')
    p.add_argument('--test-run', action='store_true', help='Test/benchmark mode: at most 1,000,000 total transitions')
    p.add_argument('--total-timesteps', type=int, help='Default: 1,000,000 in test mode, otherwise 5,000,000')
    p.add_argument('--seed', type=int, help='Training seed; omitted: generate a fresh random seed for each run')
    p.add_argument('--output')
    p.add_argument('--warm-start', help='Intermediate checkpoint to continue from; total-timesteps is cumulative. Restores actor/critic weights, resets optimizer and episodes.')
    p.add_argument('--reward-transfer', help='Reviewed JSON manifest for a rewards/goals-only warm-start migration; all mechanics and network checks remain enforced')
    p.add_argument('--evaluate', action='store_true',
                   help='Opt in to final argmax and sampled evaluation after training; disabled by default')
    p.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True,
                   help='Save weights and full learner state periodically and at the final step; enabled by default, without evaluation')
    p.add_argument('--checkpoint-interval', type=int, default=5_000_000,
                   help='Checkpoint interval in total transitions (default: 5,000,000); independent of summary-interval')
    p.add_argument('--select-best', action='store_true',
                   help='Opt in to evaluation and checkpoint selection: save at checkpoint boundaries, select on 128 fixed episodes, evaluate final and best on 1024 other episodes')
    p.add_argument('--map', help='current, default, default-green-dragon, default-blue-dragon, default-capital, small, small-green-dragon, small-blue-dragon, super_last_stand, siege_train, the_charmed_child, trade_train, mirror_match (mirrow_match), bethrezens_retreat, or a saved map JSON')
    p.add_argument('--no-unit-swaps', action='store_true',
                   help='Remove the party swap actions and features from the map (unit_swaps: false)')
    reports = p.add_mutually_exclusive_group()
    reports.add_argument('--episode-reports', action='store_true',
                         help='Opt in to detailed per-episode collection, background processing and files; disabled by default')
    reports.add_argument('--no-episode-reports', action='store_true',
                         help='Keep detailed per-episode reports disabled (the default); aggregate campaign metrics remain')
    add_tracking_arguments(p)
    train(p.parse_args())
