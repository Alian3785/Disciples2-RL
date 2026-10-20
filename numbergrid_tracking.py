"""Host-side W&B or Comet logging of already aggregated NumberGrid metrics."""
import copy
import json
import os
from pathlib import Path
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from numbergrid_campaign import CampaignMetrics
from numbergrid_comet_dashboard import compact_comet_metrics, configure_comet_dashboard
from numbergrid_wandb_dashboard import configure_wandb_dashboard

SUMMARY_INTERVAL = 1_000_000
# Every W&B series uses training_steps as its X axis, like Comet's step.
CHART_GROUPS = ('episodes', 'battles', 'construction', 'turns', 'ppo', 'performance', 'eval_argmax', 'eval_sample',
                'campaign', 'eval', 'rollout', 'train', 'selection', 'best')
TAGS = ['numbergrid', 'ppo', 'jax']


def add_tracking_arguments(parser):
    parser.add_argument('--summary-interval', type=int, default=SUMMARY_INTERVAL,
                        help='Transitions per console/cloud summary and optional checkpoint (default: 1000000)')
    parser.add_argument('--tracking', choices=('auto', 'wandb', 'wandb-offline', 'comet', 'disabled'),
                        default=os.environ.get('NUMBERGRID_TRACKING', 'auto'),
                        help='Auto: W&B cloud when it accepts the saved login, otherwise Comet with a saved '
                             'API key, otherwise offline W&B')
    parser.add_argument('--run-name', help='Optional run name displayed in W&B or Comet')
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'numbergrid'))
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--comet-project', default=os.environ.get('COMET_PROJECT_NAME', 'numbergrid'))
    parser.add_argument('--comet-workspace', default=os.environ.get('COMET_WORKSPACE'))


def describe(exc):
    lines = str(exc).strip().splitlines()
    return type(exc).__name__ + (': ' + lines[0][:200] if lines else '')


def wandb_cloud_reason(wandb):
    """None when the W&B cloud accepts the saved key; otherwise why it cannot be used."""
    try:  # wandb verifies with a 10 s HTTP timeout
        return None if wandb.login(prompt=False, verify=True) else 'not logged in'
    except Exception as exc:  # blocked (HTTP 403), unreachable or rejected key
        return describe(exc)


class CometRun:
    """The W&B calls used below, sent to a Comet experiment."""

    def __init__(self, experiment):
        self.experiment = experiment

    def log(self, data):
        data = compact_comet_metrics(data)
        step = data.pop('training_steps')
        self.experiment.log_metrics(data, step=step)

    def finish(self, exit_code=0):
        self.experiment.log_other('exit_code', exit_code)
        self.experiment.end()

    def update_summary(self, result):
        self.experiment.log_others({key: value for key, value in result.items()
                                    if isinstance(value, (bool, int, float, str))})
        self.experiment.log_asset_data(json.dumps(result, indent=2, ensure_ascii=False), name='results.json')


class NumberGridTracking:
    def __init__(self, args):
        self.args = args
        self.summary_interval = getattr(args, 'summary_interval', SUMMARY_INTERVAL)
        if self.summary_interval <= 0:
            raise ValueError('summary_interval must be positive')
        self.run = None
        self.output = None
        self.episodes = 0
        self.wins = 0
        self.battle_wins = 0
        self.buildings_built = 0
        self.turns_ended = 0
        self.rest_penalty_total = 0.
        self.logging_seconds = 0.0
        self.metadata = {}
        self.pending = []
        self.campaign = CampaignMetrics()
        self.last_campaign_metrics = {}

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
        (self.output / 'tracking.json').write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def start(self, output, config, game_map, source_hashes):
        """Auto: W&B cloud, then Comet, then offline W&B; explicit choices never fall back."""
        self.output = Path(output)
        requested = self.args.tracking
        self.metadata = {'requested': requested, 'service': None, 'mode': 'disabled', 'url': None,
                         'skipped': [], 'status': 'running', 'summary_interval': self.summary_interval}
        self.metadata['campaign_metrics'] = {
            'source': 'Python reference Big_map/train_campaign.py::CampaignMetricsCallback',
            'episode_window': 100,
            'window_rates': ('victory/defeat/timeout/battle win: whole logging interval; '
                             'dragon reached rate and recent means: last 100 completed episodes'),
            'potions_used': 'successful doses on the map and in hero battle slots; completed-episode mean',
            'mercenaries_hired': 'successful hires from mercenary camps only; completed-episode mean',
            'trainer_uses': 'successful XP purchases for individual fighters; completed-episode mean',
            'merchant_purchases': 'successful merchant purchases of potions or items (one stock unit each); completed-episode mean',
            'dragon_reached': 'hero army entered the objective dragon battle at least once in an episode',
            'capital_defense': ('new defensive bottles acquired; peak simultaneous living recipients per potion kind '
                'during the guardian battle, including permanent HP; peaks and new rounds are counted once per episode'),
            'capital_entry_strength': ('sum of max(level-1, 0) over living permanent fighters at the first hero '
                'encounter with the guardian; zero-strength contacts count, repeats do not'),
            'unavailable': {'scripted_bot': 'Scripted capital bot is not implemented in the JAX environment'},
            'local_series': 'charts.jsonl',
        }
        name = self.args.run_name or self.output.name
        settings = {'training': config, 'environment': game_map, 'source_hashes': source_hashes,
                    'logging': {'summary_interval': self.summary_interval,
                                'campaign_metrics': self.metadata['campaign_metrics']}}
        begin = time.perf_counter()
        if requested in ('auto', 'wandb'):
            import wandb
            reason = wandb_cloud_reason(wandb)
            if reason is None:
                try:
                    self._start_wandb(wandb, name, settings, 'online')
                except Exception as exc:
                    if requested == 'wandb':
                        raise
                    reason = describe(exc)
            if reason is not None:
                if requested == 'wandb':
                    raise RuntimeError('W&B cloud is unavailable (' + reason + '); check python -m wandb login')
                self.metadata['skipped'].append('W&B cloud: ' + reason)
        if self.run is None and requested in ('auto', 'comet'):
            try:
                self._start_comet(name, settings)
            except Exception as exc:
                if requested == 'comet':
                    raise
                self.metadata['skipped'].append('Comet: ' + describe(exc))
        if self.run is None and requested in ('auto', 'wandb-offline'):
            import wandb
            self._start_wandb(wandb, name, settings, 'offline')
        for reason in self.metadata['skipped']:
            print('Charts: skipped ' + reason, flush=True)
        if self.metadata['service'] == 'comet':
            print('Comet charts: ' + self.metadata['url'], flush=True)
        elif self.metadata['mode'] == 'online':
            print('W&B charts: ' + self.metadata['url'], flush=True)
        elif self.metadata['mode'] == 'offline':
            print('W&B: offline recording; sync later with python -m wandb sync ' + self.metadata['directory'],
                  flush=True)
        self.metadata['init_seconds'] = time.perf_counter() - begin
        self._save_metadata()

    def _start_wandb(self, wandb, name, settings, mode):
        run = wandb.init(
            project=self.args.wandb_project, entity=self.args.wandb_entity, name=name,
            dir=str(self.output.resolve()), mode=mode, config=settings,
            job_type='train', tags=TAGS,
            settings=wandb.Settings(disable_git=True, save_code=False, console='off',
                                    x_disable_stats=True, init_timeout=30),
        )
        run.define_metric('training_steps', hidden=True)
        for group in CHART_GROUPS:
            run.define_metric(f'{group}/*', step_metric='training_steps')
        self.run = run
        self.metadata.update(service='wandb', mode=mode, id=run.id, project=run.project,
                             directory=str(Path(run.dir).parent),
                             url=run.url if mode == 'online' else None,
                             version=wandb.__version__)
        if mode != 'online':
            command = 'python numbergrid_wandb_dashboard.py --project ' + self.args.wandb_project
            if self.args.wandb_entity:
                command += ' --entity ' + self.args.wandb_entity
            self.metadata['dashboard'] = {'status': 'not_created', 'reason': 'offline recording',
                                          'after_sync': command}
            return
        # The user reads charts only in the shared view (2026-10-17); the run page
        # link is not recorded or printed and remains in `url` only as a fallback.
        try:
            self.metadata['dashboard'] = configure_wandb_dashboard(run.entity, run.project)
            self.metadata['url'] = self.metadata['dashboard']['url']
        except Exception as exc:
            # As with Comet, a view error must not discard the run or interrupt
            # training; the run page and its default charts stay available.
            self.metadata['dashboard'] = {'status': 'failed', 'reason': describe(exc)}
            print('W&B dashboard: ' + describe(exc), flush=True)

    def _start_comet(self, name, settings):
        os.environ.setdefault('COMET_DISABLE_AUTO_LOGGING', '1')  # no framework hooks
        import comet_ml

        if not comet_ml.get_config().get_string(None, 'comet.api_key'):
            raise RuntimeError('API key is not configured; run login-comet.cmd')
        experiment = comet_ml.start(
            workspace=self.args.comet_workspace, project_name=self.args.comet_project,
            mode='create', online=True,
            experiment_config=comet_ml.ExperimentConfig(
                name=name, tags=TAGS, display_summary_level=0, parse_args=False,
                log_code=False, log_graph=False, log_git_metadata=False, log_git_patch=False,
                log_env_details=False, log_env_gpu=False, log_env_cpu=False, log_env_host=False,
                log_env_network=False, log_env_disk=False, auto_output_logging=False,
                auto_param_logging=False, auto_metric_logging=False, auto_log_co2=False))
        if not experiment.alive:
            experiment.end()
            raise RuntimeError('experiment was not created on the Comet server')
        self.run = CometRun(experiment)
        training = copy.deepcopy(settings['training'])
        training.get('env', {}).get('kwargs', {}).pop('map_config', None)  # logged as map.json
        experiment.log_parameters({'training': training, 'logging': settings['logging']})
        for asset, value in (('map.json', settings['environment']),
                             ('source_hashes.json', settings['source_hashes'])):
            experiment.log_asset_data(json.dumps(value, indent=2, ensure_ascii=False), name=asset)
        self.metadata.update(service='comet', mode='online', id=experiment.get_key(),
                             project=experiment.project_name, url=experiment.url,
                             version=comet_ml.__version__)
        try:
            api_experiment = comet_ml.API().get_experiment_by_key(experiment.get_key())
            self.metadata['dashboard'] = configure_comet_dashboard(api_experiment)
            parts = urlsplit(experiment.url)
            query = dict(parse_qsl(parts.query))
            query['viewId'] = self.metadata['dashboard']['view_id']
            self.metadata['experiment_url'] = experiment.url
            self.metadata['url'] = urlunsplit(parts._replace(query=urlencode(query)))
        except Exception as exc:
            # Dashboard permissions or API availability must not discard a
            # successfully created experiment or interrupt training.
            self.metadata['dashboard'] = {'status': 'failed', 'reason': describe(exc)}
            print('Comet dashboard: ' + describe(exc), flush=True)

    def log_training(self, row, step_size, elapsed, loop_seconds, final=False):
        """Combine existing host summaries; leave learner chunk sizes unchanged."""
        self.pending.append((row, step_size, elapsed))
        steps = sum(item[1] for item in self.pending)
        if steps < self.summary_interval and not final:
            return None
        rows = [item[0] for item in self.pending]
        counts = ('episodes', 'episode_wins', 'battle_transition', 'player_battle_transition',
                  'enemy_battle_transition', 'automatic_battle_transition', 'battle_victory', 'building_constructed', 'turn_ended', 'rest_penalty')
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
                    'training_steps', 'session_training_steps', 'training_seconds'}
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
        self.buildings_built += int(row.get('building_constructed', 0))
        data = {
            'training_steps': int(row['training_steps']),
            'episodes/completed': episodes,
            'episodes/completed_total': self.episodes,
            'episodes/wins': wins,
            'episodes/wins_total': self.wins,
            'performance/steps_per_second': step_size / elapsed,
            'performance/mean_steps_per_second': row.get('session_training_steps', row['training_steps']) / row['training_seconds'],
            'performance/training_seconds': row['training_seconds'],
            'performance/loop_steps_per_second': row.get('session_training_steps', row['training_steps']) / loop_seconds,
        }
        if episodes:
            data.update({
                'episodes/win_rate': wins / episodes,
                'episodes/length_mean': row['mean_episode_length'],
                'episodes/return_mean': row['mean_episode_return'],
            })
        if self.episodes:
            data['episodes/win_rate_total'] = self.wins / self.episodes
        if episodes:
            data['rollout/ep_rew_mean'] = row['mean_episode_return']
            data['rollout/ep_len_mean'] = row['mean_episode_length']
        if 'battle_victory' in row:
            data['battles/wins'] = int(row['battle_victory'])
            data['battles/wins_total'] = self.battle_wins
            data['battles/unit_activations_per_decision'] = row['battle_transition'] / step_size
            data['battles/automatic_activations'] = int(row.get('automatic_battle_transition', 0))
        if 'building_constructed' in row:
            data['construction/built'] = int(row['building_constructed'])
            data['construction/built_total'] = self.buildings_built
        if 'turn_ended' in row:
            self.turns_ended += int(row['turn_ended'])
            self.rest_penalty_total += float(row['rest_penalty'])
            data['turns/rests'] = int(row['turn_ended'])
            data['turns/rests_total'] = self.turns_ended
            data['turns/rest_penalty'] = float(row['rest_penalty'])
            data['turns/rest_penalty_total'] = self.rest_penalty_total
        excluded = {'all_finite', 'episodes', 'episode_wins', 'episode_success_rate',
                    'mean_episode_length', 'mean_episode_return', 'training_steps', 'session_training_steps', 'training_seconds',
                    'battle_transition', 'player_battle_transition', 'enemy_battle_transition', 'automatic_battle_transition',
                    'battle_victory', 'building_constructed', 'turn_ended', 'rest_penalty'}
        data.update({f'ppo/{key}': value for key, value in row.items() if key not in excluded})
        for source, target in (('actor_loss', 'policy_gradient_loss'), ('value_loss', 'value_loss'),
                               ('approx_kl', 'approx_kl'), ('clip_fraction', 'clip_fraction'),
                               ('explained_variance', 'explained_variance')):
            if source in row:
                data[f'train/{target}'] = row[source]
        if 'entropy' in row:
            data['train/entropy_loss'] = -row['entropy']
        self.last_campaign_metrics = self.campaign.metrics(reset_window=True)
        data.update(self.last_campaign_metrics)
        for reference, source in (('ep_rew_mean', 'episode_return_mean'), ('ep_len_mean', 'episode_length_mean')):
            key = 'campaign/window/' + source
            if key in data:
                data['rollout/' + reference] = data[key]
        return data

    def log(self, data):
        begin = time.perf_counter()
        if self.output is not None:
            with (self.output / 'charts.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
        if self.run is not None:
            self.run.log(data)
        self.logging_seconds += time.perf_counter() - begin

    def evaluation(self, evaluation, steps):
        group = 'eval_argmax' if evaluation['action_selection'] == 'argmax' else 'eval_sample'
        mode = 'deterministic' if evaluation['action_selection'] == 'argmax' else 'stochastic'
        data = {'training_steps': steps, **{
            f'{group}/{name}': evaluation[key] for name, key in (
                ('wins', 'successes'), ('episodes', 'episodes'), ('win_rate', 'success_rate'),
                ('length_mean', 'mean_length'), ('return_mean', 'mean_return'))}, **{
            f'eval/{mode}_{name}': evaluation[key] for name, key in (
                ('victory_rate', 'success_rate'), ('success_rate', 'success_rate'),
                ('mean_ep_length', 'mean_length'), ('mean_reward', 'mean_return'))}}
        if mode == 'deterministic':
            data.update({f'eval/{name}': evaluation[key] for name, key in (
                ('victory_rate', 'success_rate'), ('success_rate', 'success_rate'),
                ('mean_ep_length', 'mean_length'), ('mean_reward', 'mean_return'))})
        self.log(data)

    def summary(self, result):
        if isinstance(self.run, CometRun):
            self.run.update_summary(result)
        elif self.run is not None:
            self.run.summary.update(result)
