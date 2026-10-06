"""Run local5: default blue dragon 5M, then corrected last stand 5M."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/local5_blue_laststand_5m_20260924'
SPECS = [
    dict(name='default_blue_local5_5m_s42', map='default',
         objective='blue_dragon', steps=5000000),
    dict(name='last_stand_laclaan_local5_5m_s42', map='super_last_stand',
         objective='waves', steps=5000000),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def verify(plan):
    for name, expected in plan['frozen_sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def model_steps(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path
        return json.loads(archive.read('data'))['num_timesteps']


def process(command, log, state, phase):
    env = dict(os.environ, CAMPAIGN_OBSERVATION_VERSION='local5', MPLBACKEND='Agg',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4')
    with log.open('w') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        state.update(phase=phase, child_pid=child.pid, command=command,
                     phase_started_at=now(), log=str(log))
        save(OUTPUT / 'status.json', state)
        print(f'START {state["run"]} {phase} pid={child.pid}', flush=True)
        while child.poll() is None:
            time.sleep(5)
            state['heartbeat_at'] = now()
            save(OUTPUT / 'status.json', state)
        assert child.returncode == 0, f'{phase}: exit {child.returncode}; {log}'
    print(f'DONE {state["run"]} {phase}', flush=True)


def summarize(plan):
    rows = []
    for spec in SPECS:
        run = OUTPUT / spec['name']
        assert model_steps(run / 'models/final_model.zip') == spec['steps']
        assert (run / 'models/vecnormalize.pkl').is_file()
        ev = read(run / 'assessment/evaluation.json')
        assert ev['model_steps'] == spec['steps']
        assert ev['use_boss_starting_roster'] is False
        assert ev['map'] == spec['map'] and ev['objective'] == spec['objective']
        for mode in ('deterministic', 'stochastic'):
            data = ev[mode]
            assert [e['seed'] for e in data['episodes']] == list(range(10000, 10020))
            assert data['wins'] == sum(e['result'] == 'victory' for e in data['episodes'])
            rows.append(dict(run=spec['name'], map=spec['map'], steps=spec['steps'],
                             mode=mode, **{k: data[k] for k in (
                                 'wins', 'timeouts', 'mean_reward', 'mean_length',
                                 'mean_waves', 'max_waves_defeated', 'dragon_encounters',
                                 'dragon_victories', 'mean_enemies')}))
    verify(plan)
    save(OUTPUT / 'summary.json', dict(status='verified', completed_at=now(),
         training_steps=10000000, evaluation_episodes=80, rows=rows))
    lines = ['# Local5: default / синий дракон и last stand / Лаклаан — по 5 млн', '',
             'Оба запуска начаты с нуля, seed 42. Наблюдение local5 (5×5) сохранено; '
             'в last stand Иллюмиэлль заменена на обычного Лаклаана-мага. '
             'PPO: 16 сред, 4 потока, rollout 20 000, '
             'batch 500, 5 эпох; нормируются только награды. В обучении и оценке '
             '`--no-boss-roster`; фиксированный состав last stand сохранён.', '',
             'Каждая финальная модель проверена на seeds 10000–10019 в двух режимах, '
             'с фиксированным RNG боёв и лимитом 5000 действий на эпизод. '
             'Один seed обучения; результаты не оценивают разброс между обучениями.', '',
             '| Карта | Шаги | Режим | Победы / 20 | Таймауты / 20 | Награда | Среднее волн | Встречи с драконом / 20 |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["map"]} | {r["steps"]:,} | {r["mode"]} | {r["wins"]} | '
                     f'{r["timeouts"]} | {r["mean_reward"]:.2f} | {r["mean_waves"]:.2f} | '
                     f'{r["dragon_encounters"]} |')
    lines += ['', 'Локален вектор наблюдения; прежние маски действий и глобальная магия сохранены.', '',
              '[План и хеши исходников](plan.json), [проверенные результаты](summary.json).', '']
    for spec in SPECS:
        name = spec['name']
        lines += [f'- {name}: [модель]({name}/models/final_model.zip), '
                  f'[нормализация]({name}/models/vecnormalize.pkl), '
                  f'[оценка]({name}/assessment/evaluation.json), '
                  f'[графики обучения]({name}/plots/).']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_plan = read(ROOT / 'experiment_plan.json')
    actual_sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in source_plan['frozen_sources']}
    changes = {name for name, value in actual_sources.items()
               if value != source_plan['frozen_sources'][name]}
    assert changes == {'maps/super_last_stand.py'}, changes
    for name in ('run_long_training.py', 'finalize_long_training.py'):
        actual_sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    plan_path = OUTPUT / 'plan.json'
    if not plan_path.exists():
        save(plan_path, dict(created_at=now(), observation_version='local5',
             start_from_scratch=True, use_boss_starting_roster=False,
             seed=42, checkpoint_frequency=500000, runs=SPECS,
             frozen_sources=actual_sources,
             last_stand_wave_11='Лаклаан (Mage), не Тёмный Лаклаан',
             parent_map_sha256=source_plan['frozen_sources']['maps/super_last_stand.py']))
    plan = read(plan_path)
    assert plan['runs'] == SPECS
    verify(plan)
    old = read(OUTPUT / 'status.json') if (OUTPUT / 'status.json').exists() else {}
    state = dict(status='running', started_at=old.get('started_at', now()),
                 runner_started_at=now(), runner_pid=os.getpid(), completed_runs=[])
    save(OUTPUT / 'status.json', state)
    try:
        for spec in SPECS:
            verify(plan)
            run = OUTPUT / spec['name']
            run.mkdir(exist_ok=True)
            state['run'] = spec['name']
            if (run / 'completed.json').exists():
                state['completed_runs'].append(spec['name'])
                continue
            cmd = [sys.executable, '-u', str(ROOT / 'experiment_train.py'), str(run / 'profile.json'),
                   '--map', spec['map'], '--total-steps', str(spec['steps']),
                   '--n-envs', '16', '--vec-env', 'subproc', '--vec-start-method', 'spawn',
                   '--torch-num-threads', '4', '--ppo-n-steps', '1250', '--ppo-batch-size', '500',
                   '--ppo-n-epochs', '5', '--seed', '42', '--no-boss-roster', '--no-comet',
                   '--eval-freq', '0', '--checkpoint-freq', '500000', '--run-id', spec['name'],
                   '--output-root', str(run), '--comet-log-freq', '20000',
                   '--behavior-report-path', str(run / 'behavior.jsonl')]
            if spec['objective'] in ('dragon', 'blue_dragon'):
                cmd.append('--blue-dragon' if spec['objective'] == 'blue_dragon' else '--dragon')
            save(run / 'config.json', dict(**spec, observation_version='local5',
                 start_from_scratch=True, use_boss_starting_roster=False, command=cmd))
            if not (run / 'models/final_model.zip').exists():
                # Never silently restart an interrupted multi-million-step run.
                assert not (run / 'train.log').exists(), f'Interrupted run requires explicit checkpoint recovery: {run}'
                process(cmd, run / 'train.log', state, 'training')
            assert model_steps(run / 'models/final_model.zip') == spec['steps']
            assert read(run / 'profile.json')['actual_steps'] == spec['steps']
            dest = run / 'assessment'
            dest.mkdir(exist_ok=True)
            ev_path = dest / 'evaluation.json'
            if not ev_path.exists() or 'evaluation_seconds' not in read(ev_path):
                verify(plan)
                cmd = [sys.executable, '-u', str(ROOT / 'benchmark_training.py'), 'eval',
                       '--run-dir', str(run), '--eval-output', str(dest), '--map', spec['map'],
                       '--no-boss-roster', '--episodes', '20', '--seed', '10000']
                if spec['objective'] in ('dragon', 'blue_dragon'):
                    cmd += ['--objective', spec['objective']]
                process(cmd, dest / 'eval.log', state, 'evaluation')
            save(run / 'completed.json', dict(completed_at=now(), steps=spec['steps']))
            state['completed_runs'].append(spec['name'])
            save(OUTPUT / 'status.json', state)
        summarize(plan)
        state.update(status='completed', finished_at=now())
        save(OUTPUT / 'status.json', state)
        subprocess.run([sys.executable, str(ROOT / 'finalize_long_training.py')], cwd=ROOT, check=True)
        print('QUEUE COMPLETED: 10M training steps; final evaluations and audit passed', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), failed_at=now())
        save(OUTPUT / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
