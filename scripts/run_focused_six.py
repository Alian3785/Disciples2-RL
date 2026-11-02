"""Prepare and supervise the authorized local six-map, 50M-per-map series.

Run --prepare after both focused tests and the mandatory CUDA suite have passed.
The queue trains from an immutable source copy and stops on any failed job.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAPS = ('scroll_train', 'magic_train', 'item_train', 'hire_train', 'orc_duel', 'formation_train')
SERIES = {'focused-six': MAPS,
          'campaign-five': ('mirror_match', 'siege_train', 'default', 'default-green-dragon', 'current')}
STEPS = 50_000_000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def prepare(batch, project, workspace, series='focused-six'):
    manifest = batch/'manifest.json'
    if manifest.exists():
        raise FileExistsError('This batch is already prepared; use a new batch for a new series')
    core = json.loads((batch/'core-gpu'/'results.json').read_text())
    if not core['passed'] or core['backend'] != 'cuda':
        raise ValueError('Mandatory CUDA regression has not passed')
    evidence = ['core-gpu/results.json']
    if series == 'campaign-five':
        evidence += ['campaign-preflight.json', 'original-game-audit.json', 'tracking-preflight.json']
        for name in evidence[1:]:
            if not json.loads((batch/name).read_text())['passed']:
                raise ValueError('Campaign preflight did not pass: '+name)
        preflight = json.loads((batch/'campaign-preflight.json').read_text())
        if preflight['backend'] != 'cuda' or tuple(row['map'] for row in preflight['maps']) != SERIES[series]:
            raise ValueError('Campaign CUDA preflight must cover all five maps')
    else:
        evidence += ['focused-tests.log', 'magic-win.log']
        for name, count in (('focused-tests.log', '8 passed'), ('magic-win.log', '1 passed')):
            if count not in (batch/name).read_text():
                raise ValueError('Focused preflight did not pass: '+name)
    snapshot = batch/'source'
    snapshot.mkdir(exist_ok=False)
    paths = set(ROOT.glob('numbergrid*.py'))
    paths.update(ROOT.glob('*.json'))
    paths.update(ROOT/name for name in ('benchmark_number_grid.py', 'export_battle_viewer.py', 'pyproject.toml'))
    for directory in ('stoix', 'stoa-src', 'maps', 'scripts'):
        paths.update(p for p in (ROOT/directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.yml', '.toml', '.sh', '.cjs')
                     and '__pycache__' not in p.parts)
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = digest(target)
    manifest_data = dict(created_at=datetime.now(timezone.utc).isoformat(),
        series=series, maps=[dict(map=name, total_timesteps=STEPS) for name in SERIES[series]],
        seed=42, checkpoint_interval=5_000_000, summary_interval=1_000_000,
        source=str(snapshot), source_hashes=hashes,
        tests={name: digest(batch/name) for name in evidence},
        tracking='comet' if series == 'campaign-five' else 'auto', comet_project=project, comet_workspace=workspace,
        site_data=str(batch/'replays' if series == 'campaign-five' else ROOT/'sites/disciples-focused-replays/dist/data'))
    save(manifest, manifest_data)
    print(json.dumps(dict(phase='prepared', batch=str(batch), maps=manifest_data['maps'])), flush=True)


def run(batch):
    manifest = json.loads((batch/'manifest.json').read_text())
    source = Path(manifest['source'])
    maps = SERIES[manifest.get('series', 'focused-six')]
    if manifest['maps'] != [dict(map=name, total_timesteps=STEPS) for name in maps]:
        raise ValueError('Unexpected map order or budget')
    for name, expected in manifest['source_hashes'].items():
        if digest(source/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    for name, expected in manifest['tests'].items():
        if digest(batch/name) != expected:
            raise ValueError('Preflight evidence changed: '+name)
    environment = dict(os.environ, PYTHONUNBUFFERED='1', COMET_DISABLE_AUTO_LOGGING='1',
        COMET_PROJECT_NAME=manifest['comet_project'], COMET_WORKSPACE=manifest['comet_workspace'],
        NUMBERGRID_COMET_VIEW_SCOPE=manifest['comet_project'])
    completed = []
    current = None

    def status(phase, **extra):
        row = dict(phase=phase, map=current, completed=completed,
                   updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        save(batch/'queue-status.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def launch(arguments, log):
        with log.open('a', encoding='utf-8') as stream:
            subprocess.run(['bash', str(source/'scripts/run_gpu.sh'), *arguments], cwd=source,
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        for current in maps:
            output = batch/current
            if shutil.disk_usage(batch).free < 1024**3:
                raise RuntimeError('Less than 1 GiB free; stopping before the next job')
            if output.exists() and any(output.iterdir()):
                prior = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
                if prior.get('phase') != 'complete' or prior.get('training_steps') != STEPS:
                    raise RuntimeError('Refusing to overwrite or restart an incomplete run: '+str(output))
                for name, expected in json.loads((output/'source_hashes.json').read_text()).items():
                    if digest(source/name) != expected:
                        raise RuntimeError('Completed run used different source: '+name)
            else:
                status('training', total_timesteps=STEPS)
                launch(['benchmark_number_grid.py', '--map', current, '--seed', str(manifest['seed']),
                    '--total-timesteps', str(STEPS), '--checkpoint-interval', str(manifest['checkpoint_interval']),
                    '--summary-interval', str(manifest['summary_interval']), '--no-episode-reports',
                    '--tracking', manifest['tracking'], '--comet-workspace', manifest['comet_workspace'],
                    '--comet-project', manifest['comet_project'], '--run-name', current+'-50m-'+batch.name,
                    '--select-best', '--output', str(output)], batch/(current+'.log'))
            finished = json.loads((output/'status.json').read_text())
            tracking = json.loads((output/'tracking.json').read_text())
            if finished['phase'] != 'complete' or finished['training_steps'] != STEPS:
                raise RuntimeError('Run did not finish the exact requested budget')
            if tracking['status'] != 'finished':
                raise RuntimeError('Tracking did not finish')
            if not (output/'best/params.msgpack').is_file():
                raise RuntimeError('Best-checkpoint selection did not finish')
            launch(['scripts/verify_focused_tracking.py', '--run', str(output)],
                   batch/(current+'-tracking.log'))
            verification = json.loads((output/'tracking-verification.json').read_text())
            if verification['status'] != 'verified':
                raise RuntimeError('Chart delivery needs recovery without retraining: '+current)
            status('exporting')
            replay = Path(manifest['site_data'])/(current+'.json')
            if replay.exists() and (output/'replay-summary.json').exists():
                expected = digest(output/'best/params.msgpack')
                recorded = json.loads(replay.read_text())['metadata']['result']
                if recorded['checkpoint_sha256'] != expected:
                    raise RuntimeError('Existing replay differs from the selected checkpoint')
            else:
                launch(['scripts/export_focused_replay.py', '--run', str(output), '--map', current,
                        '--destination', manifest['site_data']], batch/(current+'-replay.log'))
            launch(['scripts/validate_focused_replays.py', '--directory', manifest['site_data'],
                    '--map', current, '--output', str(output/'replay-validation.json')],
                   batch/(current+'-validation.log'))
            completed.append(current)
            summaries = [json.loads((batch/name/'replay-summary.json').read_text()) for name in completed]
            save(batch/'summary.json', summaries)
            save(Path(manifest['site_data'])/'collection.json', summaries)
            status('map_complete', summary=summaries[-1])
        status('complete', total_training_steps=len(maps)*STEPS)
    except BaseException as exc:
        status('failed', error=type(exc).__name__+': '+str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--series', choices=SERIES, default='focused-six')
    parser.add_argument('--comet-project', default='baseline-six-maps-50m-20261029-093143')
    parser.add_argument('--comet-workspace', default='lentix')
    args = parser.parse_args()
    batch = args.batch.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    with (batch/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare(batch, args.comet_project, args.comet_workspace, args.series)
        else:
            run(batch)
