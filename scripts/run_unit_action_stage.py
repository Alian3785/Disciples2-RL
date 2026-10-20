"""Sequential CUDA PPO checks, limited to 1M transitions per fresh run.

This host orchestrator does no simulation. Child learners use run_gpu.sh; no
checkpoint is accepted, so every requested stage starts from random weights.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess


def map_changes(before, after, path=()):
    """List exact map edits, including added/removed keys and roster slots."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            if key in before and key in after:
                changes.extend(map_changes(before[key], after[key], (*path, key)))
            else:
                change = {'path': list((*path, key))}
                if key in before:
                    change['before'] = before[key]
                if key in after:
                    change['after'] = after[key]
                changes.append(change)
        return changes
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [change for i, (old, new) in enumerate(zip(before, after))
                for change in map_changes(old, new, (*path, i))]
    return [] if before == after else [{'path': list(path), 'before': before, 'after': after}]


def normalized_packages(path):
    """Compare pinned distributions independently of order, whitespace and name spelling."""
    packages = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        name, separator, version = line.strip().partition('==')
        name = re.sub(r'[-_.]+', '-', name.strip()).lower()
        if not separator or not name or not version.strip() or name in packages:
            raise ValueError('Invalid or duplicate package entry: '+str(path))
        packages[name] = version.strip()
    if not packages:
        raise ValueError('Empty package manifest: '+str(path))
    return packages


def validate_comparison(before_dir, after_dir, declared_map_changes=()):
    """Reject incompatible runs before interpreting their speed difference."""
    def read(directory, name):
        return json.loads((directory/name).read_text(encoding='utf-8'))

    before, after = (read(directory, 'results.json') for directory in (before_dir, after_dir))
    for key in ('seed', 'training_steps', 'device', 'backend', 'jax', 'python',
                'cuda_runtime', 'num_envs', 'rollout_length'):
        if before[key] != after[key]:
            raise ValueError('Non-comparable baseline: '+key)
    if before['backend'] != 'gpu':
        raise ValueError('CUDA baseline required')
    configs, maps = [], []
    for directory in (before_dir, after_dir):
        config, game_map = read(directory, 'config.json'), read(directory, 'map.json')
        if config['env']['kwargs'].pop('map_config') != game_map:
            raise ValueError('config.json and map.json disagree: '+str(directory))
        configs.append(config)
        maps.append(game_map)
    if configs[0] != configs[1]:
        raise ValueError('Non-comparable baseline: config.json (seed, budget, PPO/network settings)')
    if normalized_packages(before_dir/'packages.txt') != normalized_packages(after_dir/'packages.txt'):
        raise ValueError('Non-comparable baseline: packages.txt')
    actual = map_changes(*maps)
    canonical = lambda changes: sorted(json.dumps(c, sort_keys=True) for c in changes)
    if canonical(actual) != canonical(declared_map_changes):
        raise ValueError('Undeclared or mismatched map changes: '+json.dumps(actual, ensure_ascii=False))
    return before, after


def main():
    """Run fresh CUDA stages and compare only explicitly compatible measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--root', default='results/unit-actions-20261009')
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument('--baseline', help='Prefix of paired 1M comparison runs')
    baseline.add_argument('--baseline-run', help='Previous stage fresh seed-42 1M run')
    parser.add_argument('--tracking', choices=('auto', 'wandb', 'wandb-offline', 'comet'), default='wandb-offline')
    parser.add_argument('--map-changes', type=Path, help='JSON list of exact allowed map edits: path, before, after')
    args = parser.parse_args()
    declared_map_changes = (json.loads(args.map_changes.read_text(encoding='utf-8'))
                            if args.map_changes else [])
    if not isinstance(declared_map_changes, list):
        parser.error('--map-changes must contain a JSON list')
    workspace = Path(__file__).resolve().parents[1]
    output = (workspace / args.root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def read(name):
        return json.loads((output / name / 'results.json').read_text())

    def run(name, transitions, seed, mode):
        command = ['bash', 'scripts/run_gpu.sh', 'benchmark_number_grid.py',
                   '--test-run', '--total-timesteps', str(transitions), '--seed', str(seed),
                   '--output', str(output / name), '--tracking', mode,
                   '--wandb-project', 'numbergrid', '--wandb-entity', 'sergey3784',
                   '--run-name', name+'-20261009']
        (output / (args.stage+'-status.json')).write_text(json.dumps(dict(phase=name)))
        with (output / (name+'.log')).open('w') as stdout, (output / (name+'.errors.log')).open('w') as stderr:
            result = subprocess.run(command, cwd=workspace, stdout=stdout, stderr=stderr)
        return result.returncode

    before, after = [], []
    for seed in ((42, 43) if args.baseline else ()):
        name = f'{args.stage}-bench-{seed}'
        if run(name, 1_000_000, seed, 'wandb-offline'):
            raise SystemExit('Comparison run failed: '+name)
        previous, measured = validate_comparison(
            output/f'{args.baseline}-{seed}', output/name, declared_map_changes)
        before.append(previous['mean_steps_per_second'])
        after.append(measured['mean_steps_per_second'])
    comparison = None
    if before:
        regression = 1-sum(after)/sum(before)
        comparison = dict(before=before, after=after, regression_fraction=regression)
        (output / (args.stage+'-comparison.json')).write_text(json.dumps(comparison, indent=2))
        if regression >= .07:
            raise SystemExit('Repeated PPO regression >=7%; optimize and review before proceeding.')

    name = args.stage+'-1m'
    result = run(name, 1_000_000, 42, args.tracking)
    if result and args.tracking == 'wandb':
        error = (output / (name+'.errors.log')).read_text()
        # Only a confirmed service permission error before learner setup permits
        # an offline retry; training failures must never be silently retried.
        log = (output / (name+'.log')).read_text()
        if ('403' in error or 'PERMISSION_ERROR' in error) and '"phase": "initializing"' not in log:
            name += '-offline'
            result = run(name, 1_000_000, 42, 'wandb-offline')
    if result:
        raise SystemExit('1M test training failed: '+name)
    trained = read(name)
    if (trained['training_steps'] != 1_000_000 or not trained['weights_changed']
            or not trained['checkpoint_roundtrip_verified'] or trained['backend'] != 'gpu'):
        raise SystemExit('1M test checkpoint verification failed')
    if args.baseline_run:
        previous, trained = validate_comparison(
            output/args.baseline_run, output/name, declared_map_changes)
        regression = 1-trained['mean_steps_per_second']/previous['mean_steps_per_second']
        comparison = dict(before=[previous['mean_steps_per_second']],
                          after=[trained['mean_steps_per_second']], regression_fraction=regression)
        (output / (args.stage+'-comparison.json')).write_text(json.dumps(comparison, indent=2))
        if regression >= .07:
            raise SystemExit('PPO regression >=7%; repeat matched measurements and optimize before accepting.')
    (output / (args.stage+'-status.json')).write_text(json.dumps(dict(
        phase='complete', results=name, comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
