"""Generate comparison tables and figures from saved episode data only."""
import csv
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'runs/local_observation_experiment'
REFERENCE = ROOT.parent / 'obs_experiment_1'
TASKS = {'small_green': 'Зелёный дракон', 'small_blue': 'Синий дракон',
         'last_stand': 'super_last_stand', 'magic': 'magic_train', 'orc': 'orc_duel'}
MODES = ('deterministic', 'stochastic')


def read(path):
    return json.loads(path.read_text())


def collect():
    rows, profiles = [], {}
    def add(task, variant, path, milestones, historical, parent=None):
        if not (path / 'profile.json').exists():
            return
        profile = read(path / 'profile.json')
        seconds = profile['learn_seconds']
        if parent:
            seconds += read(parent / 'profile.json')['learn_seconds']
        profile = {**profile, 'cumulative_fps': profile['actual_steps'] / seconds,
                   'cumulative_seconds': seconds, 'historical': historical}
        profiles[f'{task}/{variant}/{profile["actual_steps"]}'] = profile
        for step, relative in milestones:
            p = path / relative / 'evaluation.json'
            if not p.exists():
                continue
            ev = read(p)
            for mode in MODES:
                data = ev[mode]
                rows.append(dict(task=task, variant=variant, steps=step, mode=mode,
                                 historical=historical, source=str(p.relative_to(ROOT.parent)),
                                 episodes=len(data['episodes']), wins=data['wins'],
                                 timeouts=data['timeouts'], reward=data['mean_reward'],
                                 waves=data['mean_waves'], enemies=data['mean_enemies'],
                                 dragon_encounters=data['dragon_encounters'], length=data['mean_length'],
                                 unique_positions=data['mean_unique_positions'],
                                 activity=data['mean_activity']))
    old = REFERENCE / 'runs/observation_experiment_1'
    for task in TASKS:
        for variant in ('baseline', 'v2'):
            if task == 'orc':
                if variant == 'baseline':
                    continue
                name, steps = 'orc_gate_v2_s42', 500000
            else:
                name, steps = f'{task}_{variant}_s42', 800000
            main = old / name
            milestones = [(steps, f'assessments/{steps}')] if task == 'orc' else [(s, f'assessments/{s}') for s in (400000, 800000)]
            add(task, variant, main, milestones, True)
            if task.startswith('small'):
                add(task, variant, old / 'extensions' / name, [(1600000, 'assessment')], True, main)
        name = task + '_local5_s42'
        steps = 500000 if task == 'orc' else 800000
        milestones = [(steps, f'assessments/{steps}')] if task == 'orc' else [(s, f'assessments/{s}') for s in (400000, 800000)]
        main = BASE / name
        add(task, 'local5', main, milestones, False)
        if task.startswith('small'):
            add(task, 'local5', BASE / (name + '_continue'), [(1600000, 'assessments/1600000')], False, main)
    return rows, profiles


def main():
    state = read(BASE / 'status.json')
    assert state['status'] == 'completed', 'Generate final report only after all runs finish'
    rows, profiles = collect()
    local = [r for r in rows if r['variant'] == 'local5']
    assert len(local) == 22 and sum(r['episodes'] for r in local) == 440
    (ROOT / 'comparison_summary.json').write_text(json.dumps({'rows': rows, 'profiles': profiles}, ensure_ascii=False, indent=2) + '\n')
    fields = [k for k in rows[0] if k != 'activity']
    with (ROOT / 'comparison_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'baseline': '#606a7a', 'v2': '#e19132', 'local5': '#087f8c'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, task in zip(axes.flat, list(TASKS)[:4]):
        for variant in colors:
            for mode, style in zip(MODES, ['-', '--']):
                subset = sorted([r for r in rows if r['task'] == task and r['variant'] == variant and r['mode'] == mode], key=lambda r: r['steps'])
                ax.plot([r['steps']/1e6 for r in subset], [r['reward'] for r in subset], style,
                        marker='o', color=colors[variant], label=f'{variant} {mode}')
        ax.set(title=TASKS[task], xlabel='Шагов обучения, млн', ylabel='Средняя тестовая награда')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Локальное окно 5×5: один seed обучения; baseline и v2 — предыдущие запуски')
    fig.savefig(ROOT / 'comparison_learning.png', dpi=160)
    plt.close(fig)
    # Training windows supplement the held-out checkpoint assessments. They
    # describe the sampled training policy, not deterministic test performance.
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    training_rows = []
    for index, task in enumerate(list(TASKS)[:4]):
        for variant in colors:
            if variant == 'local5':
                path = BASE / f'{task}_local5_s42'
                paths = [path, BASE / (path.name + '_continue')] if task.startswith('small') else [path]
            else:
                path = REFERENCE / 'runs/observation_experiment_1' / f'{task}_{variant}_s42'
                paths = [path, path.parent / 'extensions' / path.name] if task.startswith('small') else [path]
            records = []
            for path in paths:
                behavior = path / 'behavior.jsonl'
                if behavior.exists():
                    records.extend(json.loads(line) for line in behavior.read_text().splitlines() if line.strip())
            for record in records:
                counts = record.get('results_window', {})
                training_rows.append(dict(task=task, variant=variant, step=record['step'],
                                          reward=record['reward_mean_window'],
                                          victory_rate=counts.get('victory', 0) / max(1, sum(counts.values()))))
            data = [r for r in training_rows if r['task'] == task and r['variant'] == variant]
            for ax, key in zip(axes[index], ['reward', 'victory_rate']):
                ax.plot([r['step']/1e6 for r in data], [r[key] for r in data], color=colors[variant], label=variant)
                ax.grid(alpha=.2)
                ax.set(xlabel='Шагов обучения, млн', title=TASKS[task])
        axes[index, 0].set_ylabel('Награда в окне обучения')
        axes[index, 1].set_ylabel('Доля побед в окне обучения')
    axes[0, 0].legend()
    fig.suptitle('Динамика обучения: окна штатного журнала, один seed на вариант')
    fig.savefig(ROOT / 'comparison_training.png', dpi=150)
    plt.close(fig)
    with (ROOT / 'training_curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training_rows[0]))
        writer.writeheader()
        writer.writerows(training_rows)
    def row(task, variant, steps, mode):
        return next(r for r in rows if (r['task'], r['variant'], r['steps'], r['mode']) == (task, variant, steps, mode))
    elapsed = (datetime.fromisoformat(state['finished_at']) - datetime.fromisoformat(state['started_at'])).total_seconds()/60
    lines = ['# Эксперимент 2 — локальное наблюдение 5×5', '',
             f'Обучение и оценки завершены за **{elapsed:.1f} минуты**: 7 обучений/продолжений, **5,3 млн новых переходов**, 11 оценок, **440 новых тестовых эпизодов**.', '',
             'Ветка `experiment/local-observation-5x5` от baseline `76a5d3b`. Новое наблюдение включено по умолчанию в этой ветке. '
             'Предыдущая ветка observation-v2 и исходный рабочий каталог сохранены.', '',
             '## Что изменено', '',
             'Глобальная карта заменена числовым окном 5×5 вокруг героя (радиус две клетки, включая диагонали). '
             'Видны локальная проходимость и объекты, а у видимых врагов — прежние послотовые типы и HP. '
             'Сохранены прежние признаки отряда, ресурсов, экипировки, построек, магии, текущего магазина и текущего боя. '
             'Удалены абсолютные координаты, списки далёких врагов и объектов, расстояния до них, вся посещённость, '
             'число оставшихся сундуков, сведения о защитниках далёких городов и эффектах ближайшего далёкого врага. '
             'Новый вектор строится напрямую, без построения полного старого наблюдения.', '',
             'Принцип локального обзора и состояния игрока взят из [Crafter](https://github.com/danijar/crafter/blob/main/crafter/env.py); '
             'это адаптация для прежнего MLP, а не повторение RGB-эксперимента Crafter. '
             'Полная спецификация: [LOCAL_OBSERVATION.md](LOCAL_OBSERVATION.md).', '',
             '## Условия сравнения', '',
             'Baseline и v2 взяты из завершённого предыдущего эксперимента, повторно не обучались. '
             'Бюджеты, PPO и тестовые seeds совпадают: orc — 500 тыс.; основные задачи — 800 тыс.; '
             'small дополнительно продолжен с 800 тыс. до 1,6 млн с перезапуском сред, как в предыдущем эксперименте. '
             'Seed обучения 42; 16 сред, четыре потока PyTorch, rollout 20 000, batch 500, пять эпох, '
             'learning rate 0.0003, gamma 0.995, GAE 0.95, entropy 0.01. '
             'Нормируются только награды, `norm_obs=False`. Во всех обучениях и оценках `--no-boss-roster`; фиксированные составы карт сохранены.', '',
             'Каждая точка проверена в 20 deterministic и 20 stochastic эпизодах, seeds 10000–10019, '
             'с фиксированным боевым RNG и лимитом 5000 действий. Intermediate checkpoints сняты перед обновлением PPO соответствующего rollout; '
             'финальные модели — после последнего обновления.', '',
             '## Финальные результаты', '',
             'В каждой тройке порядок: **baseline / v2 / local5**. Для small — 1,6 млн шагов, для остальных — 800 тыс.', '',
             '| Задача | Режим | Победы из 20 | Награда | Таймауты из 20 | Встречи с драконом из 20 / среднее волн |',
             '|---|---|---|---|---|---|']
    for task in list(TASKS)[:4]:
        steps = 1600000 if task.startswith('small') else 800000
        for mode in MODES:
            group = [row(task, v, steps, mode) for v in colors]
            fmt = lambda key, spec='': ' / '.join(format(r[key], spec) for r in group)
            progress = fmt('dragon_encounters') if task.startswith('small') else fmt('waves', '.2f')
            lines.append(f'| {TASKS[task]} | {mode} | {fmt("wins")} | {fmt("reward", ".2f")} | {fmt("timeouts")} | {progress} |')
    orc = [row('orc', 'local5', 500000, mode)['wins'] for mode in MODES]
    lines += ['', f'Проверка `orc_duel`: local5 **{orc[0]}/20 deterministic и {orc[1]}/20 stochastic**; у прежнего v2 — 20/20 и 20/20. '
              'Парного baseline на orc в этом протоколе нет.', '', '## Скорость и размер', '',
              'Скорость включает rollout и оптимизацию PPO; оценки, запуск процесса и финальные графики не включены. '
              'Для small суммируется время обеих стадий. Это историческое сравнение: различия нагрузки машины и поведения агента влияют на FPS.', '',
              '| Задача | Размер obs baseline → local5 | Параметры baseline → local5 | Шагов/с baseline → local5 | Изменение |',
              '|---|---:|---:|---:|---:|']
    for task in list(TASKS)[:4]:
        steps = 1600000 if task.startswith('small') else 800000
        a, b = [profiles[f'{task}/{v}/{steps}'] for v in ('baseline', 'local5')]
        lines.append(f'| {TASKS[task]} | {a["observation_size"]} → {b["observation_size"]} | '
                     f'{a["policy_parameters"]:,} → {b["policy_parameters"]:,} | '
                     f'{a["cumulative_fps"]:.1f} → {b["cumulative_fps"]:.1f} | {(b["cumulative_fps"]/a["cumulative_fps"]-1)*100:+.1f}% |')
    lines += ['', '## Все контрольные точки local5', '',
              '| Задача | Шаги | Режим | Победы | Таймауты | Награда | Побеждённых врагов | Уникальных клеток |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in local:
        lines.append(f'| {TASKS[r["task"]]} | {r["steps"]:,} | {r["mode"]} | {r["wins"]}/20 | {r["timeouts"]}/20 | '
                     f'{r["reward"]:.2f} | {r["enemies"]:.2f} | {r["unique_positions"]:.2f} |')
    lines += ['', '![Кривые тестовой награды](comparison_learning.png)', '',
              '## Динамика обучения', '',
              'Ниже — награда и доля побед в окнах штатного training-журнала. Они относятся к '
              'выборке действий обучающейся политики и не заменяют отдельную тестовую оценку. '
              'Продолжения small показаны на общей шкале шагов; на 800 тыс. среды перезапускались.', '',
              '![Динамика обучения](comparison_training.png)', '',
              '[Численные данные кривых](training_curves.csv).', '', '## Ограничения', '',
              'Один seed обучения и исторические контрольные запуски дают предварительное сравнение. '
              'Боевой RNG при обучении остаётся исходным нефиксированным; одинаковый PPO seed не делает траектории одинаковыми. '
              'Тестовые эпизоды оценивают одну обученную модель, а не вариативность обучения. '
              'Различие 0/20 и 1/20 побед само по себе неубедительно. Награда с shaping не заменяет успех всей задачи.', '',
              'Награды, переходы, action space и маски не менялись. Доступность глобальных заклинаний может косвенно '
              'сообщать о наличии/пригодности цели вне окна. В бою, в том числе вызванном дальним призывом, текущий противник виден. '
              'Поэтому локален именно вектор наблюдения; весь интерфейс агента не обеспечивает строгого отсутствия любых косвенных сведений о далёких врагах.', '',
              'MLP не имеет рекуррентной памяти. Удаление посещённости и глобальной навигации делает исследование частично наблюдаемой задачей; '
              'сравнение проверяет существующий PPO с новой видимостью, без добавления памяти или изменения наград.', '',
              '## Артефакты и проверки', '',
              '- [План и замороженные хеши](experiment_plan.json).',
              '- [Проверки локальности и совпадения переходов](observation_checks.log).',
              '- [Результаты JSON](comparison_summary.json), [CSV](comparison_summary.csv).',
              '- [Логи, модели и нормализации](runs/local_observation_experiment/).',
              '- [Итоговый аудит](completion_audit.json).', '']
    (ROOT / 'report_2.md').write_text('\n'.join(lines))
    print('Comparison JSON, CSV, figure and report generated.')


if __name__ == '__main__':
    main()
