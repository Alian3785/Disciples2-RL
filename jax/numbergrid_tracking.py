"""Host-side W&B logging of already aggregated NumberGrid metrics."""
import json
import os
from pathlib import Path
import time

SUMMARY_INTERVAL = 1_000_000


def add_tracking_arguments(parser):
    parser.add_argument('--wandb-mode', choices=('auto', 'online', 'offline', 'disabled'),
                        default=os.environ.get('WANDB_MODE', 'auto'),
                        help='Auto: online after wandb login, otherwise save offline')
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'numbergrid'))
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--wandb-name', help='Optional name displayed in W&B')


class NumberGridTracking:
    def __init__(self, args):
        self.args = args
        self.run = None
        self.output = None
        self.episodes = 0
        self.wins = 0
        self.battle_wins = 0
        self.logging_seconds = 0.0
        self.metadata = {}
        self.pending = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            begin = time.perf_counter()
            self.run.finish(exit_code=0 if exc_type is None else 1)
            self.metadata['finish_seconds'] = time.perf_counter() - begin
        if self.output is not None:
            self.metadata.update(status='finished' if exc_type is None else 'failed',
                                 logging_seconds=self.logging_seconds)
            self._save_metadata()

    def _save_metadata(self):
        (self.output / 'wandb.json').write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def start(self, output, config, game_map, source_hashes):
        self.output = Path(output)
        requested = self.args.wandb_mode
        self.metadata = {'requested_mode': requested, 'mode': requested, 'status': 'running',
                         'summary_interval': SUMMARY_INTERVAL}
        begin = time.perf_counter()
        if requested != 'disabled':
            import wandb

            mode = requested
            if mode == 'auto':
                mode = 'online' if wandb.login(prompt=False, verify=False) else 'offline'
            elif mode == 'online' and not wandb.login(prompt=False, verify=False):
                raise RuntimeError('W&B online requires authentication: run python -m wandb login')
            self.run = wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.wandb_name or self.output.name,
                dir=str(self.output.resolve()), mode=mode,
                config={'training': config, 'environment': game_map, 'source_hashes': source_hashes,
                        'logging': {'summary_interval': SUMMARY_INTERVAL}},
                job_type='train', tags=['numbergrid', 'ppo', 'jax'],
                settings=wandb.Settings(disable_git=True, save_code=False, console='off',
                                        x_disable_stats=True, init_timeout=30),
            )
            self.run.define_metric('training_steps', hidden=True)
            for group in ('episodes', 'battles', 'ppo', 'performance', 'eval_argmax', 'eval_sample'):
                self.run.define_metric(f'{group}/*', step_metric='training_steps')
            self.metadata.update(mode=mode, id=self.run.id, project=self.run.project,
                                 directory=str(Path(self.run.dir).parent),
                                 url=self.run.url if mode == 'online' else None,
                                 wandb_version=wandb.__version__)
            if mode == 'offline':
                print('W&B: offline recording. Log in with python -m wandb login for live charts; '
                      'sync this run later with python -m wandb sync ' + self.metadata['directory'],
                      flush=True)
            else:
                print('W&B charts: ' + self.metadata['url'], flush=True)
        self.metadata['init_seconds'] = time.perf_counter() - begin
        self._save_metadata()

    def log_training(self, row, step_size, elapsed, loop_seconds, final=False):
        """Combine existing host summaries; leave learner chunk sizes unchanged."""
        self.pending.append((row, step_size, elapsed))
        steps = sum(item[1] for item in self.pending)
        if steps < SUMMARY_INTERVAL and not final:
            return None
        rows = [item[0] for item in self.pending]
        counts = ('episodes', 'episode_wins', 'battle_transition', 'player_battle_transition',
                  'enemy_battle_transition', 'battle_victory')
        means = ('mean_episode_length', 'mean_episode_return')
        summary = row.copy()
        for key in counts:
            if key in row:
                summary[key] = sum(item[key] for item in rows)
        episodes = summary['episodes']
        for key in means:
            summary[key] = sum(item[key] * item['episodes'] for item in rows
                               if item['episodes']) / max(episodes, 1)
        summary['episode_success_rate'] = summary['episode_wins'] / max(episodes, 1)
        summary['all_finite'] = all(item['all_finite'] for item in rows)
        excluded = {*counts, *means, 'episode_success_rate', 'all_finite',
                    'training_steps', 'training_seconds'}
        for key in row.keys() - excluded:
            summary[key] = sum(item[key] * count for item, count, _ in self.pending) / steps
        seconds = sum(item[2] for item in self.pending)
        self.log(self.training_metrics(summary, steps, seconds, loop_seconds))
        self.pending.clear()
        return summary

    def training_metrics(self, row, step_size, elapsed, loop_seconds):
        """Count completed episodes, keeping whole-map wins separate from battles."""
        episodes = int(row['episodes'])
        wins = int(row['episode_wins'])
        self.episodes += episodes
        self.wins += wins
        self.battle_wins += int(row.get('battle_victory', 0))
        data = {
            'training_steps': int(row['training_steps']),
            'episodes/completed': episodes,
            'episodes/completed_total': self.episodes,
            'episodes/wins': wins,
            'episodes/wins_total': self.wins,
            'performance/steps_per_second': step_size / elapsed,
            'performance/mean_steps_per_second': row['training_steps'] / row['training_seconds'],
            'performance/training_seconds': row['training_seconds'],
            'performance/loop_steps_per_second': row['training_steps'] / loop_seconds,
        }
        if episodes:
            data.update({
                'episodes/win_rate': wins / episodes,
                'episodes/length_mean': row['mean_episode_length'],
                'episodes/return_mean': row['mean_episode_return'],
            })
        if self.episodes:
            data['episodes/win_rate_total'] = self.wins / self.episodes
        if 'battle_victory' in row:
            data['battles/wins'] = int(row['battle_victory'])
            data['battles/wins_total'] = self.battle_wins
            data['battles/transition_fraction'] = row['battle_transition'] / step_size
        excluded = {'all_finite', 'episodes', 'episode_wins', 'episode_success_rate',
                    'mean_episode_length', 'mean_episode_return', 'training_steps', 'training_seconds',
                    'battle_transition', 'player_battle_transition', 'enemy_battle_transition',
                    'battle_victory'}
        data.update({f'ppo/{key}': value for key, value in row.items() if key not in excluded})
        return data

    def log(self, data):
        if self.run is not None:
            begin = time.perf_counter()
            self.run.log(data)
            self.logging_seconds += time.perf_counter() - begin

    def evaluation(self, evaluation, steps):
        group = 'eval_argmax' if evaluation['action_selection'] == 'argmax' else 'eval_sample'
        self.log({'training_steps': steps, **{
            f'{group}/{name}': evaluation[key] for name, key in (
                ('wins', 'successes'), ('episodes', 'episodes'), ('win_rate', 'success_rate'),
                ('length_mean', 'mean_length'), ('return_mean', 'mean_return'))}})

    def summary(self, result):
        if self.run is not None:
            self.run.summary.update(result)
