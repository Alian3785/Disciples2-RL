"""Check frozen implementation, model hyperparameters and every evaluation."""
import json
import pickle
from pathlib import Path
import re
import subprocess
import zipfile
from datetime import datetime, timezone
import numpy as np

from run_local_experiment import verify

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'runs/local_observation_experiment'
DIMS = {'orc_duel': 2388, 'small': 3754, 'super_last_stand': 5854, 'magic_train': 2688}


def main():
    plan = json.loads((ROOT / 'experiment_plan.json').read_text())
    verify(plan)
    assert json.loads((BASE / 'status.json').read_text())['status'] == 'completed'
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    assert branch == plan['branch']
    changes = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=MD', '76a5d3b'], cwd=ROOT, text=True).splitlines()
    assert changes == ['campaign_env.py', 'campaign_env_observation.py'], changes
    assert 'ALL CHECKS PASSED' in (ROOT / 'observation_checks.log').read_text()
    rows = []
    total_new_steps = 0
    evaluations = 0
    for spec in plan['runs']:
        path = BASE / spec['name']
        config = json.loads((path / 'config.json').read_text())
        profile = json.loads((path / 'profile.json').read_text())
        assert config['observation_version'] == 'local5'
        assert config['use_boss_starting_roster'] is False and '--no-boss-roster' in config['command']
        size = DIMS[spec['map']]
        added = 800000 if spec.get('resume_from') else spec['steps']
        total_new_steps += added
        assert profile['status'] == 'completed' and profile['variant'] == 'local5'
        assert profile['observation_size'] == size and profile['actual_steps'] == spec['steps']
        assert abs(profile['fps'] - added / profile['learn_seconds']) < 1e-6
        if spec.get('resume_from'):
            assert profile['start_steps'] == 800000 and profile['additional_steps'] == added
        with zipfile.ZipFile(path / 'models/final_model.zip') as archive:
            assert archive.testzip() is None
            model = json.loads(archive.read('data'))
        for key, expected in {'num_timesteps': spec['steps'], 'n_envs': 16, 'n_steps': 1250,
                              'batch_size': 500, 'n_epochs': 5, 'learning_rate': 0.0003,
                              'gamma': 0.995, 'gae_lambda': 0.95, 'ent_coef': 0.01, 'seed': 42}.items():
            assert model[key] == expected, (spec['name'], key, model[key])
        with (path / 'models/vecnormalize.pkl').open('rb') as stream:
            state = vars(pickle.load(stream))
        assert state['norm_obs'] is False and state['norm_reward'] is True
        assert state['observation_space'].shape == (size,) and state['old_obs'].shape == (16, size)
        assert np.isfinite(state['old_obs']).all()
        assert np.isfinite(state['ret_rms'].mean).all() and np.isfinite(state['ret_rms'].var).all()
        wins = []
        for step in spec['milestones']:
            ev = json.loads((path / 'assessments' / str(step) / 'evaluation.json').read_text())
            assert ev['model_steps'] == step and ev['use_boss_starting_roster'] is False
            assert ev['map'] == spec['map']
            expected_objective = spec['objective'] or {'orc_duel': 'orc', 'super_last_stand': 'waves', 'magic_train': 'all_enemies'}[spec['map']]
            assert ev['objective'] == expected_objective
            assert ev['evaluation_seconds'] > 0
            for mode in ('deterministic', 'stochastic'):
                data = ev[mode]
                episodes = data['episodes']
                assert [ep['seed'] for ep in episodes] == list(range(10000, 10020))
                assert data['wins'] == sum(ep['result'] == 'victory' for ep in episodes)
                assert data['timeouts'] == sum(ep['result'] in ('timeout', 'eval_timeout') for ep in episodes)
                assert abs(data['mean_reward'] - np.mean([ep['reward'] for ep in episodes])) < 1e-8
                assert abs(data['mean_waves'] - np.mean([ep['waves'] for ep in episodes])) < 1e-8
                assert all(np.isfinite(ep['reward']) and 0 < ep['length'] <= 5000 for ep in episodes)
            wins.append({'steps': step, 'wins': [ev[m]['wins'] for m in ('deterministic', 'stochastic')]})
            evaluations += 1
        rows.append({'run': spec['name'], 'new_steps': added, 'observation_size': size, 'evaluations': wins})
    assert total_new_steps == 5300000 and evaluations == 11 and len(rows) == 7
    report = (ROOT / 'report_2.md').read_text()
    for target in re.findall(r'\]\(([^)]+)\)', report):
        if '://' not in target and target != 'completion_audit.json':
            assert (ROOT / target).exists(), target
    audit = dict(status='passed', audited_at=datetime.now(timezone.utc).isoformat(), branch=branch,
                 frozen_source_files_verified=len(plan['frozen_sources']), training_steps=total_new_steps,
                 evaluation_points=evaluations, evaluation_episodes=evaluations * 40, runs=rows,
                 original_trainer_rewards_actions_masks_unchanged=True,
                 hidden_world_observation_invariance=True, report_links_verified=True,
                 controls='historical baseline and v2, not retrained')
    (ROOT / 'completion_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print('AUDIT PASSED: 7 runs, 5.3M new transitions, 11 evaluations, 440 episodes')


if __name__ == '__main__':
    main()
