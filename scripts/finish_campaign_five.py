"""Validate the completed five-map batch, summarize it, and merge its real replays.

Run after opening the existing Site with the Sites source helper. No training,
GPU evaluation, publishing, credentials or model weights are handled here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from validate_focused_replays import validate, MAPS as PREVIOUS_MAPS

NAMES = ('mirror_match', 'siege_train', 'default', 'default-green-dragon', 'current')
read = lambda path: json.loads(path.read_text(encoding='utf-8'))


def finish(batch, site):
    assert read(batch/'queue-status.json')['phase'] == 'complete'
    validation = validate(batch/'replays', NAMES)
    rows = []
    for name in NAMES:
        run = batch/name
        status, result = read(run/'status.json'), read(run/'results.json')
        tracking, verification = read(run/'tracking.json'), read(run/'tracking-verification.json')
        replay, best = read(run/'replay-summary.json'), read(run/'best/results.json')
        assert status['phase'] == 'complete' and status['training_steps'] == 50_000_000
        assert tracking['service'] == 'comet' and tracking['status'] == 'finished'
        assert verification['status'] == 'verified'
        assert hashlib.sha256((run/'best/params.msgpack').read_bytes()).hexdigest() == best['checkpoint_sha256']
        assert replay['checkpoint_sha256'] == best['checkpoint_sha256']
        rows.append(dict(map=name, training_steps=50_000_000, selected_checkpoint_steps=best['training_steps'],
            checkpoint_sha256=best['checkpoint_sha256'], evaluation=best['final_evaluation'],
            sampled_evaluation=best['sampled_evaluation'], replay=replay['episode'],
            training_seconds=result['training_seconds'], mean_steps_per_second=result['mean_steps_per_second'],
            charts_url=tracking['url'], metric_delivery=verification['status']))
    report = dict(hardware='Local GTX 1660 Ti 6 GB / WSL2', algorithm='PPO', seed=42,
        total_training_steps=250_000_000, maps=rows, replay_validation=validation,
        source_manifest_sha256=hashlib.sha256((batch/'manifest.json').read_bytes()).hexdigest(),
        original_data_audit=read(batch/'original-game-audit.json'),
        site_url='https://disciples-focused-six-20261031.alian3785.chatgpt.site')
    target = batch.parent/'campaign-five-maps-20261031'
    target.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['Пять JAX-карт / локальная GTX 1660 Ti / по 50 млн решений, 250 млн всего.',
             'Карта | Best, млн | Победы argmax/1024 | Победы sample/1024 | Реплей']
    for r in rows:
        lines.append(f"{r['map']} | {r['selected_checkpoint_steps']/1e6:g} | {r['evaluation']['successes']} | "
                     f"{r['sampled_evaluation']['successes']} | {r['replay']['outcome']}, {r['replay']['steps']} решений")
    lines += ['', 'Графики:']+[f"{r['map']}: {r['charts_url']}" for r in rows]
    lines += ['', 'Условия: '+str(batch/'conditions.txt'),
              'Реплей отобран из максимум 32 эпизодов; он не заменяет независимую оценку модели.',
              'Сайт: '+report['site_url']]
    target.with_suffix('.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    directory = site/'dist/data'
    collection = {r['id']: r for r in read(directory/'collection.json')}
    assert all(name in collection and (directory/(name+'.json')).exists() for name in PREVIOUS_MAPS)
    for name in NAMES:
        shutil.copyfile(batch/'replays'/(name+'.json'), directory/(name+'.json'))
        collection[name] = read(batch/name/'replay-summary.json')
    (directory/'collection.json').write_text(json.dumps(list(collection.values()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    validate(directory, PREVIOUS_MAPS+NAMES)
    print(json.dumps(dict(passed=True, records=len(collection), report=str(target.with_suffix('.json')))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--site',type=Path,required=True)
    args=parser.parse_args()
    finish(args.batch.resolve(),args.site.resolve())
