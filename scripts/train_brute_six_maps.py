"""Run the authorized Brute experiment serially, with verified Comet charts."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def check_sources(manifest):
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'experiment/brute'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == manifest['source_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True), 'Uncommitted source changes'


def verify(job, manifest):
    output = Path(job['output'])
    status, result, tracking = [read(output / name) for name in ('status.json', 'results.json', 'tracking.json')]
    assert status['phase'] == 'complete'
    assert status['training_steps'] == result['training_steps'] == job['steps']
    assert result['seed'] == manifest['seed'] and result['initial_training_steps'] == 0
    assert result['experiment'] == 'Brute' and result['observation_mode'] == 'brute_constant_zero'
    assert result['action_mask_mode'] == 'all_actions' and result['backend'] == 'gpu'
    assert result['weights_changed'] and result['checkpoint_roundtrip_verified']
    assert result['evaluation_status'] == 'not_requested'
    assert read(output / 'map.json')['equipment_requires_skills'] is True
    assert hashlib.sha256((output / 'params.msgpack').read_bytes()).hexdigest() == result['checkpoint_sha256']
    for name, value in result['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, name
    assert tracking['service'] == 'comet' and tracking['mode'] == 'online'
    assert tracking['status'] == 'finished' and tracking['project'] == manifest['project']
    assert tracking['dashboard']['status'] == 'saved'
    import comet_ml
    expected = set(range(1_000_000, job['steps']+1, 1_000_000))
    delivery = {}
    for attempt in range(3):
        experiment = comet_ml.API().get_experiment_by_key(tracking['id'])
        for metric in ('rollout/ep_rew_mean', 'campaign/window/victory_rate',
                       'campaign/mean_seconds_per_million_steps'):
            steps = {int(float(point['step'])) for point in experiment.get_metrics(metric) or []
                     if point.get('step') is not None}
            delivery[metric] = {'points': len(steps), 'missing': sorted(expected-steps)}
        if all(not item['missing'] for item in delivery.values()):
            break
        if attempt < 2:
            time.sleep(5)
    save(output / 'comet-verified.json', {'checked_at': now(), 'metrics': delivery, 'url': tracking['url']})
    assert all(not item['missing'] for item in delivery.values()), delivery
    chart = json.loads((output / 'charts.jsonl').read_text().splitlines()[-1])
    return {'map': job['map'], 'name': job['name'], 'training_steps': result['training_steps'],
            'training_seconds': result['training_seconds'], 'total_runner_seconds': result['total_runner_seconds'],
            'seconds_per_million': chart['campaign/mean_seconds_per_million_steps'],
            'final_interval_victory_rate': chart['campaign/window/victory_rate'],
            'best_interval_victory_rate': chart['campaign/best_recent_victory_rate'],
            'reward_last_100': chart['rollout/ep_rew_mean'],
            'completed_episodes': result['completed_training_episodes'],
            'won_episodes': result['won_training_episodes'], 'charts_url': tracking['url'],
            'output': str(output)}


def main():
    manifest = read(sys.argv[1])
    batch = Path(manifest['batch'])
    lock = (batch / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = batch / 'queue-status.json'
    state = read(state_path) if state_path.exists() else {'started_at': now(), 'completed': []}

    def report(**updates):
        state.update(updates, updated_at=now())
        save(state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def summarize():
        save(batch / 'summary.json', {'experiment': 'Brute', 'branch': 'experiment/brute',
                                     'source_commit': manifest['source_commit'], 'seed': manifest['seed'],
                                     'phase': state['phase'], 'runs': state['completed']})
        lines = ['Brute: PPO with constant zero observations and no legal-action information.',
                 'Equipment skills required on all six maps; training episode metrics only.', '']
        for row in state['completed']:
            lines += [f"{row['name']}: {row['training_steps']//1_000_000}M steps, "
                      f"{row['seconds_per_million']:.3f}s/M, "
                      f"last-interval wins {row['final_interval_victory_rate']:.2%}, "
                      f"reward/last 100 {row['reward_last_100']:.3f}.", row['charts_url'], '']
        (batch / 'summary.txt').write_text('\n'.join(lines) + '\n')

    try:
        check_sources(manifest)
        import comet_ml
        api = comet_ml.API()
        if api.get_project(manifest['workspace'], manifest['project']) is None:
            api.create_project(manifest['workspace'], manifest['project'],
                               project_description='Brute observation ablation: six Disciples II maps, PPO, no game features or masks.',
                               public=False)
        assert api.get_project(manifest['workspace'], manifest['project']) is not None
        save(batch / 'comet-project.json', {'verified': True, 'url': manifest['project_url']})
        for job in manifest['jobs']:
            if any(row['map'] == job['map'] for row in state['completed']):
                verify(job, manifest)
                continue
            check_sources(manifest)
            output = Path(job['output'])
            assert not output.exists() or not any(output.iterdir()), f'Refusing to restart nonempty output: {output}'
            report(phase='preflight', current_map=job['map'], current_index=job['index'],
                   current_output=job['output'], total_maps=len(manifest['jobs']), child_pid=None)
            if job['map'] != 'bethrezens_retreat':
                subprocess.run(['bash', 'scripts/run_gpu.sh', 'scripts/validate_reference_maps.py',
                                '--map', job['map'], '--output', str(batch / f"{job['index']:02d}-preflight.json")],
                               cwd=ROOT, check=True)
            command = ['bash', 'scripts/run_gpu.sh', 'benchmark_number_grid.py', '--map', job['map'],
                       '--seed', str(manifest['seed']), '--total-timesteps', str(job['steps']),
                       '--checkpoint-interval', '5000000', '--summary-interval', '1000000',
                       '--no-episode-reports', '--tracking', 'comet',
                       '--comet-workspace', manifest['workspace'], '--comet-project', manifest['project'],
                       '--run-name', f"Brute {job['index']:02d} {job['name']} {job['steps']//1_000_000}M seed {manifest['seed']}",
                       '--output', job['output']]
            save(batch / f"{job['index']:02d}-command.json", command)
            report(phase='training', map_started_at=now())
            with (batch / f"{job['index']:02d}-{job['map']}.log").open('a', buffering=1) as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, errors='replace', bufsize=1)
                report(child_pid=process.pid)
                for line in process.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                code = process.wait()
            report(last_exit_code=code, child_pid=None)
            assert code == 0, f"{job['map']} exited with code {code}"
            state['completed'].append(verify(job, manifest))
            report(phase='map_complete')
            summarize()
        report(phase='complete', finished_at=now(), current_map=None, child_pid=None)
        summarize()
    except BaseException as exc:
        report(phase='failed', error=f'{type(exc).__name__}: {exc}')
        summarize()
        raise


if __name__ == '__main__':
    main()
