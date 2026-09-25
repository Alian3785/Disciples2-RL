"""Audit the completed blue-dragon / corrected-last-stand queue and plot training."""
import json
import pickle
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_long_training import OUTPUT, SPECS, read, save, verify


def main():
    assert read(OUTPUT / 'status.json')['status'] == 'completed'
    plan = read(OUTPUT / 'plan.json')
    verify(plan)
    dimensions = {'super_last_stand': 5854, 'default': 3916}
    training = []
    assessments = []
    curves = {}
    checkpoint_pairs = 0
    for spec in SPECS:
        run = OUTPUT / spec['name']
        config = read(run / 'config.json')
        assert config['observation_version'] == 'local5'
        assert config['start_from_scratch'] and config['use_boss_starting_roster'] is False
        assert '--no-boss-roster' in config['command']
        assert '--resume-model' not in config['command']
        profile = read(run / 'profile.json')
        assert profile['actual_steps'] == spec['steps'] and profile['variant'] == 'local5'
        assert profile['observation_size'] == dimensions[spec['map']]
        with zipfile.ZipFile(run / 'models/final_model.zip') as archive:
            assert archive.testzip() is None
            model = json.loads(archive.read('data'))
        for key, expected in dict(num_timesteps=spec['steps'], n_envs=16, n_steps=1250,
                                  batch_size=500, n_epochs=5, seed=42).items():
            assert model[key] == expected, (spec['name'], key)
        with (run / 'models/vecnormalize.pkl').open('rb') as stream:
            norm = pickle.load(stream)
        assert not norm.norm_obs and norm.norm_reward
        assert norm.observation_space.shape == (dimensions[spec['map']],)
        assert np.isfinite(norm.old_obs).all()
        for step in range(500000, spec['steps'] + 1, 500000):
            assert (run / f'checkpoints/campaign_ppo_step_{step}.zip').is_file()
            assert (run / f'checkpoints/vecnormalize_step_{step}.pkl').is_file()
            checkpoint_pairs += 1
        curves[spec['name']] = [json.loads(line) for line in (run / 'behavior.jsonl').read_text().splitlines() if line.strip()]
        last = curves[spec['name']][-1]
        assert last['step'] == spec['steps']
        training.append(dict(run=spec['name'], steps=spec['steps'],
                             results=last['results_total'], profile=profile))
        files = [run / 'assessment/evaluation.json', *sorted((run / 'assessments').glob('*/evaluation.json'))]
        for path in files:
            ev = read(path)
            assert ev['map'] == spec['map'] and ev['objective'] == spec['objective']
            assert ev['use_boss_starting_roster'] is False and ev['evaluation_seconds'] > 0
            for mode in ('deterministic', 'stochastic'):
                data = ev[mode]
                episodes = data['episodes']
                assert [e['seed'] for e in episodes] == list(range(10000, 10020))
                assert data['wins'] == sum(e['result'] == 'victory' for e in episodes)
                assert data['timeouts'] == sum(e['result'] in ('timeout', 'eval_timeout') for e in episodes)
                assert np.isclose(data['mean_reward'], np.mean([e['reward'] for e in episodes]))
                assert np.isclose(data['mean_waves'], np.mean([e['waves'] for e in episodes]))
                assert all(0 < e['length'] <= 5000 and np.isfinite(e['reward']) for e in episodes)
            assessments.append(dict(path=str(path.relative_to(OUTPUT)), steps=ev['model_steps'], episodes=40))
    assert sum(t['steps'] for t in training) == 10000000
    assert sum(a['episodes'] for a in assessments) >= 80
    assert checkpoint_pairs == 20

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for index, spec in enumerate(SPECS):
        records = curves[spec['name']]
        steps = [r['step'] / 1e6 for r in records]
        rewards = [r['reward_mean_window'] if r.get('episodes_window', 0) else None for r in records]
        victories = [r['metrics']['campaign/window/victory_rate'] if r.get('episodes_window', 0) else None for r in records]
        for ax, values, label in zip(axes[index], [rewards, victories], ['Награда в окне обучения', 'Доля побед в окне обучения']):
            ax.plot(steps, values, color='#087f8c')
            ax.set(title=('default / синий дракон' if spec['map'] == 'default' else 'last stand / Лаклаан'), xlabel='Шагов, млн', ylabel=label)
            ax.grid(alpha=.25)
    fig.suptitle('Local5: синий дракон и last stand с Лаклааном; seed 42')
    fig.savefig(OUTPUT / 'training_curves.png', dpi=150)
    plt.close(fig)

    episode_count = sum(a['episodes'] for a in assessments)
    intermediate_count = episode_count - 80
    audit = dict(status='passed', audited_at=datetime.now(timezone.utc).isoformat(),
                 training_steps=10000000, final_evaluation_episodes=80,
                 intermediate_evaluation_episodes=intermediate_count, evaluation_episodes=episode_count,
                 checkpoint_pairs=checkpoint_pairs, frozen_sources_verified=len(plan['frozen_sources']),
                 runs=training, assessments=assessments)
    save(OUTPUT / 'completion_audit.json', audit)
    summary = read(OUTPUT / 'summary.json')
    summary.update(evaluation_episodes=episode_count, final_evaluation_episodes=80,
                   intermediate_evaluation_episodes=intermediate_count,
                   training_results=training,
                   last_stand_map_revision='Wave 11: Лаклаан, Mage, 800 HP, 125 life damage, no armor')
    save(OUTPUT / 'summary.json', summary)
    report_path = OUTPUT / 'report.md'
    report = report_path.read_text().split('\n## Дополнительные результаты')[0].rstrip()
    lines = ['', '', '## Дополнительные результаты', '',
             f'Всего выполнено 10 млн шагов обучения и {episode_count} тестовых эпизодов. '
             f'Из них 80 — финальные оценки, {intermediate_count} — промежуточные.', '',
             '| Запуск | Победы в ходе обучения | Поражения в ходе обучения |',
             '|---|---:|---:|']
    for row in training:
        lines.append(f'| {row["run"]} | {row["results"].get("victory", 0)} | {row["results"].get("defeat", 0)} |')
    lines += ['', 'Обучающие эпизоды относятся к меняющейся политике; их суммарные победы '
              'не являются оценкой качества финальной модели.', '',
              '![Награда и доля побед в обучении](training_curves.png)', '',
              'Окна без завершённых эпизодов показаны пропусками.', '',
              '## Ревизия last stand', '',
              'В этом запуске эльфийский отряд №132 последней волны содержит обычного '
              'Лаклаана-мага вместо Иллюмиэлль. Изменён только этот юнит; наблюдение local5, '
              'прочие отряды, расписание волн и фиксированный стартовый герой сохранены. '
              'Все оценки выполнялись на той же ревизии карты, что и обучение.', '',
              'Результаты прежней карты с Иллюмиэлль относятся к другому сценарию; '
              'прямое сравнение побед не является сравнением одинаковых условий.', '',
              '[Итоговая проверка моделей, нормализаций, checkpoints и оценок](completion_audit.json).', '']
    report_path.write_text(report + '\n'.join(lines))
    print(f'AUDIT PASSED: 10M steps, {episode_count} test episodes, 20 checkpoint pairs; report and curves saved.')


if __name__ == '__main__':
    main()
