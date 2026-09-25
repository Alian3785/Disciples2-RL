"""Serial, resumable local-observation experiment with frozen source checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/local_observation_experiment'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify(plan):
    for name, expected in plan['frozen_sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def process(command, log_path, state, phase):
    env = dict(os.environ, CAMPAIGN_OBSERVATION_VERSION='local5', MPLBACKEND='Agg',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with log_path.open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        state.update(phase=phase, child_pid=child.pid, command=command, phase_started_at=now())
        save(OUTPUT / 'status.json', state)
        print(f'START {state["run"]} {phase} pid={child.pid}', flush=True)
        while child.poll() is None:
            time.sleep(5)
            state['heartbeat_at'] = now()
            save(OUTPUT / 'status.json', state)
        assert child.returncode == 0, f'{phase} exit {child.returncode}: {log_path}'
    print(f'DONE {state["run"]} {phase}', flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / 'experiment_plan.json').read_text())
    state = {'status': 'running', 'started_at': now(), 'runner_pid': os.getpid(), 'completed_runs': []}
    save(OUTPUT / 'status.json', state)
    try:
        for spec in plan['runs']:
            verify(plan)
            run = OUTPUT / spec['name']
            run.mkdir(exist_ok=True)
            state['run'] = spec['name']
            if (run / 'completed.json').exists():
                state['completed_runs'].append(spec['name'])
                continue
            resume = spec.get('resume_from')
            cmd = [sys.executable, '-u', str(ROOT / ('experiment_continue.py' if resume else 'experiment_train.py')),
                   str(run / 'profile.json'), '--map', spec['map'], '--total-steps', str(spec['steps']),
                   '--n-envs', '16', '--vec-env', 'subproc', '--vec-start-method', 'spawn',
                   '--torch-num-threads', '4', '--ppo-n-steps', '1250', '--ppo-batch-size', '500',
                   '--ppo-n-epochs', '5', '--seed', '42', '--no-boss-roster', '--no-comet', '--eval-freq', '0',
                   '--checkpoint-freq', str(spec['milestones'][0] if not resume else 800000),
                   '--run-id', spec['name'], '--output-root', str(run), '--comet-log-freq', '20000',
                   '--behavior-report-path', str(run / 'behavior.jsonl')]
            if spec['objective']:
                cmd.append('--blue-dragon' if spec['objective'] == 'blue_dragon' else '--dragon')
            if resume:
                source = OUTPUT / resume
                assert (source / 'completed.json').exists()
                cmd += ['--resume-model', str(source / 'models/final_model.zip'),
                        '--resume-vecnormalize', str(source / 'models/vecnormalize.pkl'), '--resume-total-steps-is-target']
            save(run / 'config.json', {**spec, 'command': cmd, 'observation_version': 'local5',
                                      'use_boss_starting_roster': False})
            if not (run / 'profile.json').exists() or not (run / 'models/final_model.zip').exists():
                process(cmd, run / 'train.log', state, 'training')
            profile = json.loads((run / 'profile.json').read_text())
            assert profile['actual_steps'] == spec['steps']
            if resume:
                assert profile['additional_steps'] == 800000
            with zipfile.ZipFile(run / 'models/final_model.zip') as archive:
                assert archive.testzip() is None
                assert json.loads(archive.read('data'))['num_timesteps'] == spec['steps']
            for milestone in spec['milestones']:
                verify(plan)
                dest = run / 'assessments' / str(milestone)
                dest.mkdir(parents=True, exist_ok=True)
                evaluation_file = dest / 'evaluation.json'
                if evaluation_file.exists() and 'evaluation_seconds' in json.loads(evaluation_file.read_text()):
                    continue
                if milestone == spec['steps']:
                    model, vec = run / 'models/final_model.zip', run / 'models/vecnormalize.pkl'
                else:
                    model = run / f'checkpoints/campaign_ppo_step_{milestone}.zip'
                    vec = run / f'checkpoints/vecnormalize_step_{milestone}.pkl'
                cmd = [sys.executable, '-u', str(ROOT / 'benchmark_training.py'), 'eval', '--run-dir', str(run),
                       '--model-path', str(model), '--vecnormalize-path', str(vec), '--eval-output', str(dest),
                       '--map', spec['map'], '--no-boss-roster', '--episodes', '20', '--seed', '10000']
                if spec['objective']:
                    cmd += ['--objective', spec['objective']]
                process(cmd, dest / 'eval.log', state, f'evaluation_{milestone}')
                ev = json.loads(evaluation_file.read_text())
                assert ev['model_steps'] == milestone and ev['use_boss_starting_roster'] is False
                assert all(len(ev[m]['episodes']) == 20 for m in ('deterministic', 'stochastic'))
            ev = json.loads((run / 'assessments' / str(spec['steps']) / 'evaluation.json').read_text())
            save(run / 'completed.json', {'finished_at': now(), 'profile': profile})
            state['completed_runs'].append(spec['name'])
            save(OUTPUT / 'status.json', state)
            print(f'COMPLETE {spec["name"]}: {profile["fps"]:.1f} fps; '
                  f'wins {ev["deterministic"]["wins"]}/20 + {ev["stochastic"]["wins"]}/20', flush=True)
        verify(plan)
        state.update(status='completed', finished_at=now())
        save(OUTPUT / 'status.json', state)
        print('EXPERIMENT COMPLETED', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save(OUTPUT / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
