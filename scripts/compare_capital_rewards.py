"""Prepare and report the user-requested paired 20M Capital reward experiment.

Training uses benchmark_number_grid.py, unchanged PPO and CUDA. Source reward
constants are read as data; the Python reference environment is never run.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def prepare(root, reference):
    from stoix.envs.number_grid import load_map
    root.mkdir(parents=True, exist_ok=False)
    current = load_map('default-capital')
    python = deepcopy(current)
    python.update(reward_profile='python_reference', reference_preserve_target_reward=True, step_cost=0.)
    save(root/'current-map.json', current)
    save(root/'python-map.json', python)
    sources = ('train_campaign.py', 'campaign_env.py', 'campaign_env_territory.py',
               'campaign_env_battle.py', 'campaign_env_economy.py', 'campaign_env_inventory.py')
    hashes = {name: hashlib.sha256((reference/name).read_bytes()).hexdigest() for name in sources}
    tree = ast.parse((reference/'train_campaign.py').read_text(encoding='utf-8-sig'))
    rewards = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'REWARD_CONFIG' for t in node.targets))
    save(root/'protocol.json', dict(training_steps_each=20_000_000, seed=42, from_scratch=True,
        common_map='default-capital', objective='Mizrael, army 75',
        fixed_target_reward='1 battle + 3 objective + 5 * sum(max(entry_level-1, 0))',
        python_target_adaptation='Original target battle rewards retained; no reference +8/+96 goal or +6 first contact. Other reference rewards, including episode-wide defeated-army bonus, remain.',
        implementation='Same JAX mechanics, observations, actions and PPO; imported Python reward profile.',
        evaluation='Final checkpoints only: 1024 argmax and 1024 sampled episodes each, seed 1042. No best-checkpoint selection.',
        timing='Whole synchronized learner; compile, evaluation, logging and checkpoint timings separate.',
        limitations='One training seed. Shared JAX mechanics and one decision per step; this is not a run of the original Python engine. Reward-history state follows the existing JAX reference-profile implementation.',
        source_hashes=hashes, reference_reward_config=rewards))
    print(json.dumps({'prepared': str(root)}, ensure_ascii=False), flush=True)


def report(root):
    names = ('current', 'python')
    results = [read(root/name/'results.json') for name in names]
    configs = [read(root/name/'config.json') for name in names]
    maps = [read(root/name/'map.json') for name in names]
    allowed = {'reward_profile', 'reference_preserve_target_reward', 'step_cost'}
    differences = {k for k in maps[0].keys() | maps[1].keys() if maps[0].get(k) != maps[1].get(k)}
    if differences != allowed:
        raise ValueError(f'Unexpected map differences: {differences}')
    normalized = deepcopy(configs)
    for config in normalized:
        del config['env']['kwargs']['map_config']
    if normalized[0] != normalized[1]:
        raise ValueError('PPO/network configurations differ')
    common = ('seed', 'device', 'jax', 'cuda_runtime', 'python', 'parameter_count', 'num_envs',
              'rollout_length', 'observation_size', 'ppo_updates', 'source_hashes', 'initial_checkpoint_sha256')
    for key in common:
        if results[0][key] != results[1][key]:
            raise ValueError('Run settings differ: '+key)
    if (root/'current/packages.txt').read_bytes() != (root/'python/packages.txt').read_bytes():
        raise ValueError('Installed packages differ')
    for result in results:
        if (result['training_steps'] != 20_000_000 or result['initial_training_steps'] != 0
                or result['seed'] != 42 or result['backend'] != 'gpu'
                or result['evaluation_status'] != 'completed' or not result['weights_changed']
                or not result['checkpoint_roundtrip_verified']):
            raise ValueError('Run budget/backend/evaluation/checkpoint mismatch')
    timing = ('training_seconds', 'mean_steps_per_second', 'ppo_compile_seconds',
              'training_loop_seconds', 'tracking_logging_seconds', 'episode_report_seconds',
              'tracking_init_seconds', 'checkpoint_seconds', 'total_runner_seconds',
              'final_evaluation_seconds', 'sampled_evaluation_seconds')
    change = 100*(results[1]['mean_steps_per_second']/results[0]['mean_steps_per_second']-1)
    summary = dict(protocol=read(root/'protocol.json'), matched_configs=True, matched_initial_weights=True,
        matched_sources_and_packages=True, map_differences=sorted(differences),
        speed_change_percent=change, slowdown_at_least_seven_percent=change <= -7,
        runs={name: {**{k: result[k] for k in timing},
            'training_episodes': result['completed_training_episodes'],
            'training_wins': result['won_training_episodes'], 'argmax': result['final_evaluation'],
            'sampled': result['sampled_evaluation'], 'charts': result['tracking']['url'],
            'tracking_service': result['tracking']['service'],
            'episode_summary': read(root/name/'episode_reports.summary.json')}
              for name, result in zip(names, results)})
    save(root/'comparison.json', summary)
    lines = ['# Default Capital: сравнение наград', '',
        'Два независимых запуска по 20 млн переходов, seed 42, одна GPU: '
        f"{results[0]['device']}. Сеть, начальные веса, PPO, карта и награды целевого боя совпадают. "
        'Сравнивается Python-схема наград в общей JAX-среде; исходный Python-движок не запускается.', '',
        '## Финальные модели', '',
        'Оценка только финальных весов, по 1024 эпизода каждого способа выбора действий, seed 1042.', '',
        '| Награды | Выбор действия | Победы | Вступили в бой с Мизраэлем | Максимальный урон цели | Отрядов побеждено, среднее | Поражения / таймауты |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in names:
        for mode in ('argmax', 'sampled'):
            row = summary['runs'][name][mode]
            lines.append(f"| {name} | {mode} | {row['successes']}/1024 | "
                f"{row['target_reached_episodes']}/1024 | {100*row['maximum_target_damage_fraction']:.2f}% | "
                f"{row['mean_defeated_armies']:.2f} | {row['defeats']} / {row['timeouts']} |")
    lines += ['', '## Скорость и длительности', '',
        '| Метрика | Текущая | Python |', '|---|---:|---:|']
    for label, key in (
            ('Learner, переходов/с', 'mean_steps_per_second'),
            ('Learner, с', 'training_seconds'), ('Компиляция PPO, с', 'ppo_compile_seconds'),
            ('Цикл обучения вместе с отчётами, с', 'training_loop_seconds'),
            ('Обработка отчётов эпизодов на CPU, с', 'episode_report_seconds'),
            ('Вызовы логирования, с', 'tracking_logging_seconds'),
            ('Инициализация трекера, с', 'tracking_init_seconds'),
            ('Сохранение весов, с', 'checkpoint_seconds'),
            ('Оценка argmax, включая её компиляцию, с', 'final_evaluation_seconds'),
            ('Оценка sampled, включая её компиляцию, с', 'sampled_evaluation_seconds'),
            ('Весь запуск, с', 'total_runner_seconds')):
        a, b = (summary['runs'][name][key] for name in names)
        lines.append(f'| {label} | {a:.2f} | {b:.2f} |')
    lines += ['', f'Изменение скорости Python-профиля: **{change:+.2f}%**. '
        'Замер learner включает rollout и PPO с ожиданием GPU. Времена в таблице частично вложены; '
        'их нельзя складывать. Поведение политик различается, поэтому это не изолированный замер формулы награды.', '',
        '## Артефакты и ограничения', '',
        'Совпадение конфигураций, начальных весов, исходников и пакетов проверено автоматически. '
        'Проверка карты на CUDA пройдена перед обучением; она подтверждает структурную достижимость цели, '
        'а не наличие победного маршрута. Регрессионные тесты не запускались.', '']
    for name in names:
        row = summary['runs'][name]
        lines.append(f"- {name}: [графики {row['tracking_service']}]({row['charts']}), "
                     f"побед в обучении {row['training_wins']} / {row['training_episodes']} эпизодов, "
                     f"веса `{name}/params.msgpack`.")
    lines += ['', 'Полные данные: `comparison.json`; условия и исходные Python-константы: `protocol.json`. '
        'Суммы наград разных схем не сравниваются как качество политики. '
        'Один seed не доказывает устойчивое преимущество.', '']
    (root/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('protocol', 'runs')}), flush=True)
    for name in names:
        print(json.dumps({name: {k: v for k, v in summary['runs'][name].items()
                                if k != 'episode_summary'}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.reference is None:
            parser.error('--reference is required for prepare')
        prepare(args.output, args.reference)
    else:
        report(args.output)
