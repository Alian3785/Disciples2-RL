"""Summarize the user-requested paired 10M trainings without running JAX.

Requires completed runs with identical sources, packages, seed, network and PPO.
The only map differences allowed are names and the event enable switch.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def process_seconds(path):
    text = path.read_text(encoding='utf-8')
    match = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', text)
    if match is None:
        raise ValueError('Missing process wall time: '+str(path))
    seconds = 0.
    for part in match[1].split(':'):
        seconds = 60*seconds+float(part)
    return seconds


def compare(root):
    names = ('no-events', 'events')
    results = [read(root/name/'results.json') for name in names]
    configs = [read(root/name/'config.json') for name in names]
    maps = [read(root/name/'map.json') for name in names]
    normalized = deepcopy(configs)
    for config in normalized:
        del config['env']['kwargs']['map_config']
    if normalized[0] != normalized[1]:
        raise ValueError('PPO, network or other training settings differ')
    differences = {key for key in maps[0].keys() | maps[1].keys() if maps[0].get(key) != maps[1].get(key)}
    if differences != {'name', 'display_name', 'scenario_events_enabled'}:
        raise ValueError('Unexpected map differences: '+str(differences))
    if maps[0]['scenario_events_enabled'] or not maps[1]['scenario_events_enabled']:
        raise ValueError('Reversed or missing event flags')
    if results[0]['source_hashes'] != results[1]['source_hashes']:
        raise ValueError('Measured source files differ')
    if (root/names[0]/'packages.txt').read_bytes() != (root/names[1]/'packages.txt').read_bytes():
        raise ValueError('Installed packages differ')
    for result in results:
        if (result['training_steps'] != 10_000_000 or result['seed'] != 42
                or result['backend'] != 'gpu' or result['evaluation_status'] != 'not_requested'
                or not result['weights_changed'] or not result['checkpoint_roundtrip_verified']):
            raise ValueError('Run budget, backend, evaluation or checkpoint validation mismatch')
    for key in ('device', 'jax', 'cuda_runtime', 'python', 'parameter_count', 'num_envs',
                'rollout_length', 'observation_size', 'ppo_updates'):
        if results[0][key] != results[1][key]:
            raise ValueError('Run settings differ: '+key)
    before, after = [r['mean_steps_per_second'] for r in results]
    change = 100*(after/before-1)
    timing_keys = ('training_seconds', 'mean_steps_per_second', 'ppo_compile_seconds',
                   'training_loop_seconds', 'tracking_logging_seconds', 'episode_report_seconds',
                   'tracking_init_seconds', 'checkpoint_seconds', 'total_runner_seconds')
    summary = dict(
        training_steps_each=10_000_000, training_steps_total=20_000_000, seed=42,
        device=results[0]['device'], jax=results[0]['jax'], cuda_runtime=results[0]['cuda_runtime'],
        python=results[0]['python'], observation_size=results[0]['observation_size'],
        num_envs=results[0]['num_envs'], rollout_length=results[0]['rollout_length'],
        ppo_updates=results[0]['ppo_updates'], parameter_count=results[0]['parameter_count'],
        speed_change_percent=change, slowdown_at_least_seven_percent=change <= -7,
        equivalent_configs=True, identical_source_hashes=True, identical_packages=True,
        map_differences=sorted(differences),
        timing='Whole synchronized learner: rollout, automatic turns, PPO and device metrics. Compilation, host logging, reports and saving are separate.',
        interpretation='Paired end-to-end training at one seed. Events change trajectories and combat workload; this does not isolate event-kernel overhead or establish a statistically stable speed difference.',
        runs={name: {**{key: result[key] for key in timing_keys},
                     'process_wall_seconds': process_seconds(root/(name+'-time.txt')),
                     'seconds_per_million': result['training_seconds']/10,
                     'charts': result['tracking']['url'], 'tracking_service': result['tracking']['service'],
                     'episodes': result['completed_training_episodes'], 'wins': result['won_training_episodes'],
                     'combat_counts': result['combat_counts'], 'rest_count': result['rest_count'],
                     'weights': str(root/name/'params.msgpack'),
                     'checkpoint_sha256': result['checkpoint_sha256']}
              for name, result in zip(names, results)})
    (root/'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = [
        '# Бегство Бетрезена: события и скорость обучения', '',
        'Два последовательных запуска по 10 000 000 решений, seed 42. '
        'Обычный ИИ неподвижен; «Сила огня» использует принятую механику JAX.', '',
        '| Режим | Решений/с | Секунд на 1 млн | Learner, с | Компиляция PPO, с |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, title in zip(names, ('Без событий', 'С событиями')):
        row = summary['runs'][name]
        lines.append(f"| {title} | {row['mean_steps_per_second']:,.0f} | {row['seconds_per_million']:.3f} | "
                     f"{row['training_seconds']:.3f} | {row['ppo_compile_seconds']:.3f} |")
    lines += ['', f'Изменение скорости learner: **{change:+.2f}%**.', '',
        f"GPU: {summary['device']}; JAX {summary['jax']}; CUDA runtime {summary['cuda_runtime']}; "
        f"Python {summary['python']}. 250 сред, rollout 100, 400 обновлений PPO, "
        '4 эпохи, 10 minibatches, MLP [128,128], tanh. Наблюдение: 3899; действий: 277.', '',
        'Конфигурации PPO/сети, исходники и пакеты сверены: совпадают. Карты отличаются '
        'только именами и переключателем событий. В обоих режимах зарезервированы '
        '113 слотов армий, поэтому размеры сети одинаковы. Другие GPU-задачи не запускались.', '',
        'Замер включает весь learner с синхронизацией GPU. События меняют траектории '
        'и число автоматических активаций в боях: это сравнение двух полных обучений '
        'на одном seed, не изолированная стоимость обработчика событий.', '',
        '| Отдельные затраты, с | Без событий | С событиями |', '|---|---:|---:|',
    ]
    for key, label in (('tracking_init_seconds', 'Подключение графиков'),
                       ('tracking_logging_seconds', 'Отправка/запись графиков'),
                       ('episode_report_seconds', 'Отчёты эпизодов'),
                       ('checkpoint_seconds', 'Сохранение весов'),
                       ('training_loop_seconds', 'Цикл обучения вместе с отчётностью'),
                       ('total_runner_seconds', 'Весь runner, включая завершение облачного логирования')):
        lines.append(f'| {label} | {results[0][key]:.3f} | {results[1][key]:.3f} |')
    lines.append('| Весь процесс, включая запуск и завершение облачного логирования | '
                 + ' | '.join(f"{summary['runs'][name]['process_wall_seconds']:.3f}" for name in names)+' |')
    lines += ['', 'Оценка политики и выбор промежуточной модели не запускались. '
        'Проверки конечности метрик, изменения весов и восстановления checkpoint прошли в штатном runner. '
        'Регрессионные тесты не запускались; до обучения обе карты пройдены проверочным планировщиком на CUDA. '
        'Его действия не передавались обучению.', '']
    for name, title in zip(names, ('Без событий', 'С событиями')):
        row = summary['runs'][name]
        lines.append(f"* {title}: [графики {row['tracking_service']}]({row['charts']}), "
                     f"эпизодов в обучении {row['episodes']}, побед {row['wins']}.")
    lines += ['', 'Все настройки, версии, хеши, логи, итоговые веса и подробные результаты '
        'сохранены в соседних каталогах `no-events/` и `events/`. '
        'Их статистика побед относится к обучению, не к отдельной оценке политики.', '']
    (root/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('speed_change_percent', 'slowdown_at_least_seven_percent', 'runs')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    compare(parser.parse_args().root)
