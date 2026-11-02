"""Verify actual Comet metric delivery after a focused-map run, without logging secrets."""
import argparse
import json
import math
from pathlib import Path
import time

METRICS = ('rollout/ep_rew_mean', 'campaign/window/victory_rate')


def expected_samples(run):
    """Distinguish undelivered samples from statistics undefined for empty windows."""
    required = set(range(1_000_000, 50_000_001, 1_000_000))
    rows = {int(row['training_steps']): row
            for row in (json.loads(line) for line in (run/'charts.jsonl').read_text().splitlines())
            if 'episodes/completed' in row}
    if set(rows) != required:
        raise ValueError('Local training summaries do not cover every million decisions')
    expected, undefined = {}, {}
    for metric in METRICS:
        expected[metric], undefined[metric] = {}, []
        for step, row in sorted(rows.items()):
            if metric in row:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError('Local metric is non-finite')
                expected[metric][step] = value
            else:
                count = (row['episodes/completed_total'] if metric == 'rollout/ep_rew_mean'
                         else row['campaign/window/episodes'])
                if count != 0:
                    raise ValueError('Metric is absent despite available episode samples')
                undefined[metric].append(step)
    return expected, undefined


def verify(run):
    tracking = json.loads((run/'tracking.json').read_text())
    report = dict(service=tracking['service'], url=tracking.get('url'), status='not_comet', metrics={})
    if tracking['service'] == 'comet':
        import comet_ml
        experiment = comet_ml.API().get_experiment_by_key(tracking['id'])
        expected, undefined = expected_samples(run)
        for attempt in range(3):
            for name in METRICS:
                points = experiment.get_metrics(name)
                received = {int(float(row['step'])): float(row['metricValue']) for row in points
                            if row.get('step') is not None}
                mismatches = [step for step, value in expected[name].items() if step in received
                              and not math.isclose(value, received[step], rel_tol=1e-6, abs_tol=1e-6)]
                report['metrics'][name] = dict(points=len(points), expected_points=len(expected[name]),
                    missing_steps=sorted(set(expected[name])-set(received)),
                    value_mismatches=mismatches, undefined_steps=undefined[name])
            if all(not row['missing_steps'] and not row['value_mismatches'] for row in report['metrics'].values()):
                break
            if attempt < 2:
                time.sleep(10)
        report['dashboard'] = tracking.get('dashboard', {}).get('status')
        report['undefined_interval_reason'] = ('A mean before the first completed episode, or a victory rate '
                                              'in a window with zero completed episodes, is undefined; '
                                              'no artificial zero samples are uploaded.')
        report['status'] = ('verified' if report['dashboard'] == 'saved' and
                            all(not row['missing_steps'] and not row['value_mismatches']
                                for row in report['metrics'].values()) else 'incomplete')
    (run/'tracking-verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.run)
    except Exception as exc:
        report = dict(status='error', error=type(exc).__name__)
        (args.run/'tracking-verification.json').write_text(json.dumps(report)+'\n')
        print(json.dumps(report), flush=True)
