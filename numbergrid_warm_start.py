"""Validated weights-only continuation; optimizer and episodes start afresh."""
import hashlib
import json
from pathlib import Path


def _reward_transfer(manifest_path, checkpoint_hash, old_map, game_map, hashes, changed, root):
    """Accept only a recorded, reviewed reward/objective migration, never a silent bypass."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('kind') != 'reward-objective-transfer' or manifest.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Reward transfer manifest does not identify this checkpoint')
    map_changes = {key: dict(before=old_map.get(key), after=game_map.get(key))
                   for key in old_map.keys() | game_map.keys() if old_map.get(key) != game_map.get(key)}
    if (set(map_changes) - {'name', 'optional_enemies', 'target_rewards'}
            or map_changes != manifest.get('map_changes') or not game_map.get('target_rewards')):
        raise ValueError('Reward transfer changes something other than recorded rewards/goals')
    allowed = {'stoix/envs/number_grid.py', 'stoix/envs/number_grid_casting.py', 'numbergrid_config.py',
               'stoix/envs/number_grid_target_rewards.py'}
    if set(changed) - allowed:
        raise ValueError(f'Unrelated source changes in reward transfer: {changed}')
    actual = {name: dict(before=hashes[name], after=hashlib.sha256((root / name).read_bytes()).hexdigest())
              for name in changed}
    if actual != manifest.get('source_changes'):
        raise ValueError('Reward transfer source hashes differ from the reviewed manifest')
    new_source = 'stoix/envs/number_grid_target_rewards.py'
    new_hash = hashlib.sha256((root / new_source).read_bytes()).hexdigest()
    if manifest.get('target_reward_source_sha256') != new_hash:
        raise ValueError('Target reward implementation differs from the reviewed manifest')
    return dict(manifest=str(Path(manifest_path).resolve()), **manifest)


def prepare_warm_start(checkpoint, total, game_map, config, root, reward_transfer=None):
    """Validate a checkpoint and continue its LR schedule without restoring optimizer state."""
    path = Path(checkpoint).resolve()
    if path.name == 'params.msgpack' and (path.parent / 'results.json').exists():
        # Final weights of a finished run (user request 2026-10-17); results.json
        # records their steps and hash.
        parent = path.parent
        result = json.loads((parent / 'results.json').read_text())
        record = dict(training_steps=result['training_steps'], sha256=result['checkpoint_sha256'])
    else:
        parent = path.parent.parent
        records = json.loads((parent / 'checkpoints.json').read_text())
        record = next((r for r in records if (parent / r['path']).resolve() == path), None)
        if record is None:
            raise ValueError('Warm-start checkpoint is absent from its saved manifest')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Warm-start checkpoint hash mismatch')
    initial = record['training_steps']
    if not 0 < initial < total:
        raise ValueError('Total budget must exceed the checkpoint training steps')
    old_map = json.loads((parent / 'map.json').read_text())
    if old_map != game_map and reward_transfer is None:
        raise ValueError('Warm-start map differs from the saved training map')
    hashes = json.loads((parent / 'source_hashes.json').read_text())
    # Only orchestration/logging changed to support this continuation. All
    # recorded environment, learner and configuration sources must still match.
    # Chart layouts of Comet and W&B are logging only.
    orchestration = {'benchmark_number_grid.py', 'numbergrid_tracking.py', 'numbergrid_warm_start.py',
                     'numbergrid_comet_dashboard.py', 'numbergrid_wandb_dashboard.py'}
    changed = [name for name, digest in hashes.items()
               if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    transfer = None
    incompatible = [name for name in changed if name not in orchestration]
    if reward_transfer is not None:
        transfer = _reward_transfer(reward_transfer, record['sha256'], old_map, game_map,
                                    hashes, incompatible, root)
    elif incompatible:
        raise ValueError(f'Warm-start source incompatibility: {changed}')
    previous = json.loads((parent / 'config.json').read_text())
    rate_names = ('actor_lr', 'critic_lr')
    parent_warm_start = previous.get('warm_start') or {}
    base_rates = parent_warm_start.get('base_learning_rates',
        {name: previous['system'][name] for name in rate_names})
    if {name: config['system'][name] for name in rate_names} != base_rates:
        raise ValueError('Warm-start base learning rates differ')
    # A child run stores decayed session rates in system. Compare the original
    # rates separately so another continuation still validates its configuration.
    system = {key: value for key, value in config['system'].items() if key not in rate_names}
    previous_system = {key: value for key, value in previous['system'].items() if key not in rate_names}
    if system != previous_system:
        raise ValueError('Warm-start system configuration differs')
    if config['network'] != previous['network']:
        raise ValueError('Warm-start network configuration differs')
    for name in ('num_envs', 'update_batch_size', 'num_updates_per_eval'):
        if config['arch'][name] != previous['arch'][name]:
            raise ValueError(f'Warm-start configuration differs: {name}')
    parent_initial = parent_warm_start.get('initial_training_steps', 0)
    decay = previous['system']['decay_learning_rates']
    fraction = (initial - parent_initial) / previous['arch']['total_timesteps']
    if fraction == 1:
        # A finished schedule has decayed to zero. Continue at the rate of one linear
        # schedule over the whole cumulative budget, decaying over the remainder.
        rates = {name: base_rates[name] * (1 - initial / total if decay else 1) for name in rate_names}
        schedule = 'Finished parent: cumulative linear-schedule rate, decay over remaining budget'
    elif 0 <= fraction < 1:
        rates = {name: previous['system'][name] * (1 - fraction if decay else 1) for name in rate_names}
        schedule = 'Continue from parent next-update rate, decay over remaining budget'
    else:
        raise ValueError('Warm-start steps exceed the parent schedule')
    return data, dict(
        checkpoint=str(path), parent_results=str(parent), checkpoint_sha256=record['sha256'],
        initial_training_steps=initial, target_training_steps=total,
        session_training_steps=total-initial, optimizer_restored=False,
        environment_state_restored=False, source_changes=changed, reward_transfer=transfer,
        base_learning_rates=base_rates, learning_rates=rates, learning_rate_schedule=schedule)
