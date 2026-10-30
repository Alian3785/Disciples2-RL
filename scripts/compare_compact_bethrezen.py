"""Verify completed observation experiments and summarize their measured speed.

No training, evaluation, map planner or game-mechanics tests are run here.
Usage: scripts/run_gpu.sh scripts/compare_compact_bethrezen.py RESULTS_DIRECTORY
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def compare(root):
    names = ('main-speed-1m', 'compact-speed-1m', 'compact-60m')
    runs = [read(root/name/'results.json') for name in names]
    configs = [read(root/name/'config.json') for name in names]
    maps = [read(root/name/'map.json') for name in names]
    for name, result, budget, size in zip(names, runs, (1_000_000, 1_000_000, 60_000_000),
                                         (3914, 2239, 2239)):
        assert read(root/name/'status.json')['phase'] == 'complete', name
        assert result['training_steps'] == budget and result['initial_training_steps'] == 0, name
        assert result['observation_size'] == size and result['seed'] == 42, name
        assert result['backend'] == 'gpu' and result['evaluation_status'] == 'not_requested', name
        assert result['weights_changed'] and result['checkpoint_roundtrip_verified'], name
        assert hashlib.sha256((root/name/'params.msgpack').read_bytes()).hexdigest() == result['checkpoint_sha256']
    for other in maps[1:]:
        assert {**maps[0], 'compact_observations': True} == other
    # The paired 1M measurements share the same LR schedule and report/checkpoint settings.
    configs[0]['env']['kwargs']['map_config']['compact_observations'] = True
    assert configs[0] == configs[1], 'Paired training configurations differ'
    assert (root/names[0]/'packages.txt').read_bytes() == (root/names[1]/'packages.txt').read_bytes()
    allowed = {'stoix/envs/'+name+'.py' for name in ('number_grid', 'number_grid_chests',
               'number_grid_items', 'number_grid_ruins', 'number_grid_territory')}
    changed = {name for name, digest in runs[0]['source_hashes'].items()
               if digest != runs[1]['source_hashes'][name]}
    assert changed == allowed, changed
    assert runs[1]['source_hashes'] == runs[2]['source_hashes']
    for key in ('device', 'jax', 'cuda_runtime', 'python', 'num_envs', 'rollout_length'):
        assert len({r[key] for r in runs}) == 1, key

    tracking = read(root/names[2]/'tracking.json')
    assert tracking['service'] == 'comet' and tracking['mode'] == 'online'
    assert tracking['status'] == 'finished' and tracking['dashboard']['status'] == 'saved'
    charts = [json.loads(line) for line in (root/names[2]/'charts.jsonl').read_text().splitlines()]
    import comet_ml
    experiment = comet_ml.API().get_experiment_by_key(tracking['id'])
    delivery = {}
    for metric in ('rollout/ep_rew_mean', 'campaign/window/victory_rate',
                   'campaign/mean_seconds_per_million_steps'):
        expected = {int(row['training_steps']) for row in charts if metric in row}
        actual = {int(float(point['step'])) for point in experiment.get_metrics(metric) or []
                  if point.get('step') is not None}
        assert expected == set(range(1_000_000, 60_000_001, 1_000_000)), (metric, expected)
        assert expected <= actual, (metric, sorted(expected-actual))
        delivery[metric] = {'expected_points': len(expected), 'delivered_points': len(actual)}

    historical = next(r for r in read(root.parent/'baseline-six-maps-20261029.json')['runs']
                      if r['map'] == 'bethrezens_retreat')
    summary = dict(map='bethrezens_retreat', branch='experiment/compact-observations-bethrezen',
        base_commit=(root/'base-commit.txt').read_text().strip(), seed=42,
        observation_before=3914, observation_after=2239, removed=1675,
        reduction_percent=100*1675/3914, removed_chests_and_ruins=1599,
        removed_city_and_mine_constants=76,
        paired_speed_change_percent=100*(runs[1]['mean_steps_per_second']/runs[0]['mean_steps_per_second']-1),
        historical_60m_speed_change_percent=100*(historical['seconds_per_million']/(runs[2]['training_seconds']/60)-1),
        device=runs[2]['device'], jax=runs[2]['jax'], cuda_runtime=runs[2]['cuda_runtime'],
        source_files_changed=sorted(changed), comet_delivery=delivery, charts_url=tracking['url'],
        historical_60m=historical,
        caveat='One paired 1M training measurement per mode. Removing constant inputs changes network initialization and trajectories; speed differences include combat workload. The historical 60M run also preceded equipment-skill requirements.',
        timing='Synchronized learner calls: rollout and PPO. Compilation, logging and saving are separate.',
        runs={})
    for name, result in zip(names, runs):
        row = {key: result[key] for key in ('training_steps', 'observation_size', 'parameter_count',
            'training_seconds', 'mean_steps_per_second', 'ppo_compile_seconds', 'total_runner_seconds',
            'completed_training_episodes', 'won_training_episodes', 'combat_counts', 'checkpoint_sha256')}
        row['seconds_per_million'] = result['training_seconds']*1_000_000/result['training_steps']
        summary['runs'][name] = row
    final = charts[-1]
    summary['final_interval_victory_rate'] = final['campaign/window/victory_rate']
    summary['best_interval_victory_rate'] = final['campaign/best_recent_victory_rate']
    summary['reward_last_100'] = final['rollout/ep_rew_mean']
    save(root/'comparison.json', summary)
    lines = ['Бегство Бетрезена: компактное наблюдение', '',
        'Ветка: '+summary['branch']+'. Основана на main '+summary['base_commit']+'.',
        'Вход: 3914 → 2239 признаков (−42,79%). Удалены константы фиксированной карты:',
        '1599 признаков сундуков/руин и 76 координат/типов/доходов городов и шахт.',
        'Сундуки представлены 35 флагами наличия, руины — 8 флагами зачистки.',
        'Индекс соответствует одному объекту карты; координаты и содержимое напрямую агенту не передаются.',
        'Полученная добыча видна в текущем инвентаре.',
        'Динамические признаки, правила, награды, действия и настройки PPO сохранены.', '',
        '| Запуск | Шагов | Решений/с | Секунд/млн | Компиляция PPO, с |',
        '|---|---:|---:|---:|---:|']
    for name in names:
        row = summary['runs'][name]
        lines.append(f"| {name} | {row['training_steps']} | {row['mean_steps_per_second']:.0f} | "
                     f"{row['seconds_per_million']:.3f} | {row['ppo_compile_seconds']:.3f} |")
    lines += ['', f"Изменение скорости в парном замере 1 млн: {summary['paired_speed_change_percent']:+.2f}%.",
        f"Исторический запуск 60 млн: {historical['seconds_per_million']:.3f} с/млн; "
        f"изменение скорости {summary['historical_60m_speed_change_percent']:+.2f}%.",
        'Один seed и один замер на режим; изменение входа меняет инициализацию и траектории.',
        'Историческая серия также предшествует обязательным навыкам экипировки.',
        'Это сравнение полных обучений; изолированная стоимость сети отдельно не измерена.', '',
        f"GPU: {summary['device']}; JAX {summary['jax']}; CUDA runtime {summary['cuda_runtime']}.",
        'Seed 42, 250 сред, rollout 100, MLP [128,128], FP32, PPO без изменений.',
        'Время learner включает сбор траекторий и PPO; компиляция, логирование и сохранение отдельно.',
        'Проверены бюджеты, конфигурации пары 1 млн, версии пакетов, хеши исходников и весов,',
        'изменение весов, восстановление checkpoint и доставка всех 60 точек каждой из трёх метрик в Comet.',
        'Планировщик карты, отдельная оценка политики и тесты механик не запускались.', '',
        f"Последний интервал побед: {summary['final_interval_victory_rate']:.2%}; "
        f"лучший: {summary['best_interval_victory_rate']:.2%}; reward последних 100: {summary['reward_last_100']:.3f}.",
        'Это метрики эпизодов обучения, отдельной оценки нет.',
        'Comet: '+tracking['url'], '',
        'Полные метрики, конфигурации, промежуточные checkpoints и веса остаются локально.', '']
    (root/'comparison.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    compare(parser.parse_args().results)
