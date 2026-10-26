"""Reporting regressions included in the project's GPU-only test session."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from numbergrid_tracking import NumberGridTracking, add_tracking_arguments


pytestmark = pytest.mark.core


def tracker():
    return NumberGridTracking(tracking_args('--tracking', 'disabled'))


def tracking_args(*argv):
    parser = ArgumentParser()
    add_tracking_arguments(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def services(monkeypatch):
    """Fake W&B and Comet modules: no network, no saved keys touched."""
    monkeypatch.setenv('COMET_DISABLE_AUTO_LOGGING', '1')
    wandb = SimpleNamespace(__version__='test', login=Mock(return_value=True), Settings=dict, init=Mock())
    wandb.init.side_effect = lambda **kwargs: Mock(
        id='w1', project=kwargs['project'], dir=kwargs['dir'] + '/wandb/run/files',
        url='https://wandb.test/run' if kwargs['mode'] == 'online' else None)
    experiment = Mock(alive=True, url='https://comet.test/run', project_name='numbergrid')
    experiment.get_key.return_value = 'c1'
    comet = SimpleNamespace(__version__='test', get_config=Mock(), ExperimentConfig=dict,
                            start=Mock(return_value=experiment))
    comet.get_config.return_value.get_string.return_value = 'saved-key'
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    monkeypatch.setitem(sys.modules, 'comet_ml', comet)
    monkeypatch.setattr('numbergrid_tracking.configure_wandb_dashboard', Mock(
        return_value={'status': 'created', 'url': 'https://wandb.test/view'}))
    return wandb, comet, experiment


def start(tmp_path, *argv, game_map=None):
    tracking = NumberGridTracking(tracking_args(*argv))
    game_map = game_map or {'size': 48}
    tracking.start(tmp_path, {'env': {'kwargs': {'map_config': game_map}}}, game_map, {})
    return tracking


@pytest.mark.parametrize('view_error', [None, RuntimeError('403 Forbidden')])
def test_auto_tracking_prefers_reachable_wandb_cloud(tmp_path, services, view_error):
    import numbergrid_tracking
    wandb, comet, _ = services
    numbergrid_tracking.configure_wandb_dashboard.side_effect = view_error
    tracking = start(tmp_path)
    assert tracking.metadata['service'] == 'wandb' and tracking.metadata['mode'] == 'online'
    assert 'run_url' not in tracking.metadata  # only the shared view is reported
    if view_error is None:  # the link opens the saved "Disciples II" view
        assert tracking.metadata['url'] == 'https://wandb.test/view'
    else:  # a view error keeps the run and its default charts
        assert tracking.metadata['url'] == 'https://wandb.test/run'
        assert tracking.metadata['dashboard'] == {'status': 'failed', 'reason': 'RuntimeError: 403 Forbidden'}
    assert wandb.login.call_args.kwargs['verify'] is True
    comet.start.assert_not_called()


def test_auto_tracking_uses_comet_when_wandb_cloud_is_blocked(tmp_path, services, capsys):
    wandb, comet, experiment = services
    wandb.login.side_effect = RuntimeError('Failed to authenticate: 403 Forbidden')
    with start(tmp_path) as tracking:
        tracking.log({'training_steps': 1_000_000, 'episodes/wins': 3})
        tracking.summary({'training_steps': 1_000_000, 'tracking': {'nested': True}})
    wandb.init.assert_not_called()
    metadata = json.loads((tmp_path / 'tracking.json').read_text())
    assert metadata['service'] == 'comet' and metadata['url'] == 'https://comet.test/run'
    assert metadata['skipped'] == ['W&B cloud: RuntimeError: Failed to authenticate: 403 Forbidden']
    assert metadata['status'] == 'finished'
    assert 'Comet charts: https://comet.test/run' in capsys.readouterr().out
    experiment.log_metrics.assert_called_once_with({'episodes/wins': 3}, step=1_000_000)
    experiment.log_others.assert_called_once_with({'training_steps': 1_000_000})
    parameters = experiment.log_parameters.call_args.args[0]
    assert 'map_config' not in parameters['training']['env']['kwargs']
    experiment.end.assert_called_once()


def test_auto_tracking_records_offline_wandb_without_comet_key(tmp_path, services):
    wandb, comet, _ = services
    wandb.login.return_value = False
    comet.get_config.return_value.get_string.return_value = None
    tracking = start(tmp_path)
    comet.start.assert_not_called()
    assert tracking.metadata['service'] == 'wandb' and tracking.metadata['mode'] == 'offline'
    assert tracking.metadata['url'] is None
    assert tracking.metadata['dashboard']['status'] == 'not_created'
    assert tracking.metadata['dashboard']['after_sync'].startswith('python numbergrid_wandb_dashboard.py --project ')
    assert tracking.metadata['skipped'] == ['W&B cloud: not logged in',
                                            'Comet: RuntimeError: API key is not configured; run login-comet.cmd']


@pytest.mark.parametrize('choice', ['wandb', 'comet'])
def test_explicit_tracking_service_never_falls_back(tmp_path, services, choice):
    wandb, comet, experiment = services
    wandb.login.return_value = False
    experiment.alive = False
    with pytest.raises(RuntimeError):
        start(tmp_path, '--tracking', choice)
    wandb.init.assert_not_called()
    assert comet.start.called == (choice == 'comet')


@pytest.mark.parametrize('source', ['flag', 'environment', 'pytest'])
def test_test_training_budget_rejects_before_cuda_or_tracking(monkeypatch, source):
    import benchmark_number_grid as benchmark
    monkeypatch.delenv('NUMBERGRID_TEST_RUN', raising=False)
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    if source == 'environment':
        monkeypatch.setenv('NUMBERGRID_TEST_RUN', '1')
    elif source == 'pytest':
        monkeypatch.setenv('PYTEST_CURRENT_TEST', 'budget-regression')
    args = Namespace(test_run=source == 'flag', smoke=False, total_timesteps=1_250_000)
    cuda = Mock(side_effect=AssertionError('CUDA must not start for a rejected budget'))
    tracking = Mock(side_effect=AssertionError('Tracking must not start for a rejected budget'))
    monkeypatch.setattr(benchmark.jax, 'devices', cuda)
    monkeypatch.setattr(benchmark, 'NumberGridTracking', tracking)
    with pytest.raises(ValueError, match='1,000,000'):
        benchmark.train(args)
    with pytest.raises(ValueError, match='1,000,000'):
        benchmark.run_training(args, None)
    cuda.assert_not_called()
    tracking.assert_not_called()
    args.total_timesteps = 1_000_000
    assert benchmark.training_budget(args) == 1_000_000
    args.total_timesteps = None
    assert benchmark.training_budget(args) == 1_000_000


def test_test_budget_preserves_smoke_and_explicit_non_test_training(monkeypatch):
    from benchmark_number_grid import training_budget
    assert training_budget(Namespace(test_run=True, smoke=True, total_timesteps=None)) == 250_000
    monkeypatch.delenv('NUMBERGRID_TEST_RUN', raising=False)
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    assert training_budget(Namespace(test_run=False, smoke=False, total_timesteps=2_000_000)) == 2_000_000


def row(episodes, wins, steps=250_000, **extra):
    return {
        'episodes': float(episodes), 'episode_wins': float(wins),
        'episode_success_rate': wins / max(episodes, 1),
        'training_steps': steps, 'training_seconds': 1.0,
        'mean_episode_length': 123.0, 'mean_episode_return': 4.0,
        'all_finite': True, 'actor_loss': -0.1, **extra,
    }


def test_win_rate_counts_episodes_instead_of_averaging_windows():
    tracking = tracker()
    tracking.training_metrics(row(2, 2), 250_000, 1.0, 1.1)
    metrics = tracking.training_metrics(row(8, 0, 500_000), 250_000, 1.0, 2.2)
    assert metrics['episodes/completed_total'] == 10
    assert metrics['episodes/wins_total'] == 2
    assert metrics['episodes/win_rate'] == 0
    assert metrics['episodes/win_rate_total'] == pytest.approx(0.2)
    assert metrics['episodes/length_mean'] == 123


def test_empty_window_does_not_draw_false_zero_episode_means():
    tracking = tracker()
    metrics = tracking.training_metrics(row(0, 0), 250_000, 1.0, 1.1)
    assert metrics['episodes/completed'] == 0
    assert 'episodes/length_mean' not in metrics
    assert 'episodes/return_mean' not in metrics
    assert 'episodes/win_rate' not in metrics
    assert 'episodes/win_rate_total' not in metrics


def test_continuation_steps_affect_speed_but_do_not_appear_as_ppo_metrics():
    """A warm-start summary uses session work for speed and cumulative steps for its axis."""
    tracking = tracker()
    tracking.run = Mock()
    tracking.log_training(row(2, 1, 20_250_000, session_training_steps=250_000),
                          250_000, .5, 2., final=True)
    metrics = tracking.run.log.call_args.args[0]
    assert metrics['training_steps'] == 20_250_000
    assert metrics['performance/mean_steps_per_second'] == 250_000
    assert metrics['campaign/mean_seconds_per_million_steps'] == 4.0
    assert metrics['performance/loop_steps_per_second'] == 125_000
    assert 'ppo/session_training_steps' not in metrics
    assert metrics['ppo/actor_loss'] == -.1


def test_battle_wins_are_not_counted_as_whole_map_wins():
    tracking = tracker()
    metrics = tracking.training_metrics(
        row(10, 1, battle_victory=42, battle_transition=125_000), 250_000, 1.0, 1.1)
    assert metrics['episodes/wins'] == 1
    assert metrics['battles/wins'] == 42
    assert metrics['battles/unit_activations_per_decision'] == 0.5
    assert metrics['ppo/actor_loss'] == -0.1
    assert 'ppo/battle_victory' not in metrics


def test_failed_training_finishes_wandb_as_failed():
    tracking = tracker()
    run = Mock()
    tracking.run = run
    with pytest.raises(ValueError, match='training failed'):
        with tracking:
            raise ValueError('training failed')
    run.finish.assert_called_once_with(exit_code=1)


def test_disabled_tracking_preserves_metadata_without_creating_a_run(tmp_path, services):
    wandb, comet, _ = services
    with tracker() as tracking:
        tracking.start(tmp_path, {}, {}, {})
        tracking.log({'training_steps': 0})
        assert tracking.run is None
    metadata = json.loads((tmp_path / 'tracking.json').read_text())
    assert metadata['mode'] == 'disabled' and metadata['service'] is None
    assert metadata['status'] == 'finished'
    wandb.login.assert_not_called()
    comet.start.assert_not_called()


def test_million_step_summary_includes_all_episodes_with_weighted_means():
    tracking = tracker()
    tracking.run = Mock()
    for index, (episodes, wins, length) in enumerate(((1, 1, 10), (3, 0, 30), (0, 0, 0), (6, 2, 50))):
        summary = tracking.log_training(
            row(episodes, wins, (index + 1) * 250_000, mean_episode_length=length,
                mean_episode_return=length / 10, battle_victory=10, battle_transition=400_000,
                automatic_battle_transition=300_000, building_constructed=index+1,
                turn_ended=2*(index+1), rest_penalty=.01*(index+1)),
            250_000, 0.5, (index + 1) * 0.6)
        if index < 3:
            assert summary is None
            tracking.run.log.assert_not_called()
    tracking.run.log.assert_called_once()
    logged = tracking.run.log.call_args.args[0]
    assert logged['training_steps'] == 1_000_000
    assert logged['episodes/completed'] == 10
    assert logged['episodes/wins'] == 3
    assert logged['episodes/win_rate'] == pytest.approx(0.3)
    assert logged['episodes/length_mean'] == pytest.approx(40)
    assert logged['episodes/return_mean'] == pytest.approx(4)
    assert logged['battles/wins'] == 40
    assert logged['construction/built'] == logged['construction/built_total'] == 10
    assert logged['turns/rests'] == logged['turns/rests_total'] == 20
    assert logged['turns/rest_penalty'] == logged['turns/rest_penalty_total'] == pytest.approx(.1)
    assert 'ppo/turn_ended' not in logged and 'ppo/rest_penalty' not in logged
    assert 'ppo/building_constructed' not in logged
    assert logged['battles/unit_activations_per_decision'] == pytest.approx(1.6)
    assert logged['battles/automatic_activations'] == 1_200_000
    assert 'ppo/automatic_battle_transition' not in logged
    assert logged['performance/steps_per_second'] == 500_000


def test_final_partial_window_is_logged_once_and_counts_are_preserved():
    tracking = tracker()
    tracking.run = Mock()
    for index in range(5):
        tracking.log_training(row(2, 1, (index + 1) * 250_000, turn_ended=3, rest_penalty=.015), 250_000, 0.5,
                              (index + 1) * 0.6, final=index == 4)
    calls = [call.args[0] for call in tracking.run.log.call_args_list]
    assert [call['training_steps'] for call in calls] == [1_000_000, 1_250_000]
    assert calls[0]['episodes/completed'] == 8
    assert calls[1]['episodes/completed'] == 2
    assert calls[1]['episodes/completed_total'] == 10
    assert calls[1]['episodes/wins_total'] == 5
    assert calls[1]['turns/rests'] == 3 and calls[1]['turns/rests_total'] == 15
    assert calls[1]['turns/rest_penalty'] == pytest.approx(.015)
    assert calls[1]['turns/rest_penalty_total'] == pytest.approx(.075)
    assert not tracking.pending


def test_smoke_run_shorter_than_interval_still_has_training_chart_point():
    tracking = tracker()
    tracking.run = Mock()
    tracking.log_training(row(4, 1), 250_000, 0.5, 0.6, final=True)
    tracking.run.log.assert_called_once()
    assert tracking.run.log.call_args.args[0]['episodes/wins'] == 1


@pytest.mark.parametrize('backend', ['comet', 'wandb'])
@pytest.mark.parametrize('objective', ['waves', 'capital_guardian'])
def test_reference_campaign_charts_reach_both_backends_and_local_series(tmp_path, services, backend, objective):
    from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS
    from numbergrid_comet_dashboard import CAMPAIGN_CHARTS
    _, _, experiment = services
    tracking = start(tmp_path, '--tracking', backend, game_map={'size': 48, 'campaign_objective': objective})
    values = dict.fromkeys(CAMPAIGN_FIELDS, 0.)
    values.update(victories=1, timeouts=1, battle_victories=4, battle_retreats=1,
                  hired_units=3, mercenaries_hired=2, spell_casts=5, enemy_squads_defeated=4,
                  potions_used=6, dragon_battles=3, unit_upgrades=2, sold_items=4, trainer_uses=8,
                  merchant_purchases=10, waves_completed=11, capital_guardian_battles=2)
    ordered = [values[name] for name in CAMPAIGN_FIELDS]
    tracking.campaign.add({'events': ordered, 'completed': ordered,
        'participation': [float(v > 0) for v in ordered], 'count': 2,
        'recent': [ordered + [7., 80.], [0.] * len(ordered) + [7., 80.]]})
    tracking.log_training(row(2, 1), 250_000, .5, .6, final=True)
    data = experiment.log_metrics.call_args.args[0] if backend == 'comet' else tracking.run.log.call_args.args[0]
    assert data['campaign/victory_rate'] == .5
    assert data['campaign/timeout_rate'] == .5
    assert data['campaign/battle_win_rate'] == .8
    assert data['campaign/completed/potions_used_mean'] == 3
    assert data['campaign/completed/mercenaries_hired_mean'] == 1
    assert data['campaign/completed/trainer_uses_mean'] == 4
    assert data['campaign/completed/merchant_purchases_mean'] == 5
    assert data['campaign/window/battle_victories_mean'] == 2
    assert data['campaign/window/unit_upgrades_mean'] == 1
    if objective == 'waves':
        assert data['campaign/window/waves_completed_mean'] == 5.5  # includes the timeout with zero waves
    else:
        assert 'campaign/window/waves_completed_mean' not in data
    capital_key = 'campaign/window/capital_guardian_reached_episodes_rate'
    if backend == 'wandb' or objective == 'capital_guardian':
        assert data[capital_key] == .5
    else:
        assert capital_key not in data
    assert data['campaign/window/dragon_reached_episodes_rate'] == .5  # three battles, one reached episode
    assert data['campaign/victories_total'] == 1
    assert data['rollout/ep_rew_mean'] == 7  # rolling episodes, not latest block's reward 4
    reference_keys = json.loads((Path(__file__).resolve().parents[2] /
        'docs/validation/reference-campaign-metric-keys.json').read_text(encoding='utf-8'))
    saved = json.loads((tmp_path / 'charts.jsonl').read_text())
    assert set(reference_keys['supported']) <= saved.keys()
    assert saved['campaign/hired_units_total'] == 3
    assert saved['campaign/mercenaries_hired_total'] == 2
    assert saved['campaign/trainer_uses_total'] == 8
    assert saved['campaign/merchant_purchases_total'] == 10
    assert saved['campaign/merchant_purchase_episodes_rate'] == .5
    assert saved['campaign/window/spell_casts_mean'] == 2.5
    assert saved['campaign/window/enemy_squads_defeated_mean_100'] == 2
    if objective == 'waves':
        assert saved['campaign/window/waves_completed_mean'] == 5.5
    assert saved[capital_key] == .5  # Filtering Comet panels must not remove local telemetry.
    assert 'campaign/scripted_bot_victories_per_episode_mean' not in data
    if backend == 'comet':
        assert experiment.log_metrics.call_args.kwargs['step'] == 250_000
        assert {key for key in data if key.startswith('campaign/')} <= set(CAMPAIGN_CHARTS)
        assert 'campaign/completed/sold_items_mean' not in data
        assert 'campaign/sold_items_total' not in data
    else:
        assert set(reference_keys['supported']) <= data.keys()
        assert data['campaign/completed/sold_items_mean'] == 2
    assert saved['training_steps'] == 250_000 and saved['campaign/hired_units_total'] == 3


def test_campaign_no_completed_episode_omits_undefined_rates_but_keeps_activity():
    from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS
    tracking = tracker()
    tracking.campaign.waves_enabled = True
    events = [float(name == 'spell_casts') for name in CAMPAIGN_FIELDS]
    zero = [0.] * len(events)
    tracking.campaign.add({'events': events, 'completed': zero, 'participation': zero, 'count': 0, 'recent': []})
    metrics = tracking.training_metrics(row(0, 0), 250_000, 1., 1.)
    assert metrics['campaign/spell_casts_total'] == 1
    assert metrics['campaign/episodes_total'] == 0
    assert 'campaign/victory_rate' not in metrics
    assert 'campaign/spell_casts_per_episode_mean' not in metrics
    assert 'campaign/window/dragon_reached_episodes_rate' not in metrics
    assert 'campaign/completed/potions_used_mean' not in metrics
    assert 'campaign/completed/trainer_uses_mean' not in metrics
    assert 'campaign/window/waves_completed_mean' not in metrics


def test_comet_view_shows_requested_progress_and_omits_sold_item_count():
    from numbergrid_comet_dashboard import CAMPAIGN_CHARTS, compact_comet_view
    old = {'panels': {'sections': [
        {'title': 'campaign', 'panels': [{'metricName': 'campaign/completed/sold_items_mean'}]},
        {'title': 'train', 'panels': [{'metricName': 'train/approx_kl'}]},
    ]}}
    view = compact_comet_view(old, {'campaign_objective': 'capital_guardian'})
    sections = view['panels']['sections']
    assert sections[-1] == old['panels']['sections'][-1]  # preserve technical charts
    campaign = next(s for s in sections if s['title'] == 'campaign')
    names = [p['metricName'] for p in campaign['panels']]
    assert len(names) == len(set(names)) == len(CAMPAIGN_CHARTS) == 49
    assert set(names) == set(CAMPAIGN_CHARTS)
    # First row: run-long win/defeat rates and the latest interval's win rate.
    first_row = [names[i] for i, item in enumerate(campaign['layout']) if item['y'] == 0]
    assert first_row == ['campaign/victory_rate', 'campaign/defeat_rate', 'campaign/window/victory_rate']
    assert 'campaign/timeout_rate' in names
    assert not any('sold_items' in name for name in names)
    assert 'campaign/completed/merchant_sale_gold_mean' in names
    assert 'campaign/completed/merchant_purchases_mean' in names
    for name in ('campaign/completed/potions_used_mean', 'campaign/window/battle_victories_mean',
                 'campaign/window/unit_upgrades_mean', 'campaign/window/waves_completed_mean',
                 'campaign/window/dragon_reached_episodes_rate',
                 'campaign/victories_total', 'campaign/completed/mercenaries_hired_mean',
                 'campaign/completed/trainer_uses_mean'):
        panel = next(p for p in campaign['panels'] if p['metricName'] == name)
        assert panel['selectedXAxis'] == 'step' and panel['chartName'] != name


@pytest.mark.parametrize('objective', [None, 'waves', 'cities', 'all_enemies', 'dragon'])
def test_comet_general_view_removes_guardian_panels_even_when_guardian_is_present(objective):
    from numbergrid_comet_dashboard import CAPITAL_CHARTS, compact_comet_view, compact_comet_metrics
    guardian = compact_comet_view(game_map={'campaign_objective': 'capital_guardian'})
    game_map = {'campaign_objective': objective, 'enemy_rosters': [['mizrael']]}
    general = compact_comet_view(guardian, game_map)
    campaign = next(s for s in general['panels']['sections'] if s['title'] == 'campaign')
    names = {p['metricName'] for p in campaign['panels']}
    assert not (names & CAPITAL_CHARTS)
    assert 'campaign/window/waves_completed_mean' in names
    assert 'campaign/completed/potions_used_mean' in names  # General potion usage still applies.
    assert general['template_name'] != guardian['template_name']  # Shared templates cannot overwrite one another.
    assert any(p['metricName'] in CAPITAL_CHARTS for s in guardian['panels']['sections'] for p in s['panels'])
    data = dict.fromkeys(CAPITAL_CHARTS, 1.) | {'training_steps': 1_000_000, 'campaign/victory_rate': .5}
    assert compact_comet_metrics(data, game_map) == {'training_steps': 1_000_000, 'campaign/victory_rate': .5}
    assert compact_comet_metrics(data, {'campaign_objective': 'capital_guardian'}) == data


def test_wandb_view_repeats_comet_sections_titles_and_axis(monkeypatch):
    """The saved W&B view shows the Comet sections, metrics and Russian titles over training_steps."""
    ws = pytest.importorskip('wandb_workspaces.workspaces')
    import numbergrid_wandb_dashboard as dashboard
    from numbergrid_comet_dashboard import compact_comet_view
    saved, views = [], {}
    def from_url(url):
        assert url.endswith('/lentix/numbergrid?nw=disciplesii')
        if 'view' not in views:
            raise ValueError('Workspace `nw-disciplesii-v` not found in project `numbergrid`')
        return views['view']
    monkeypatch.setattr(ws.Workspace, 'from_url', from_url)
    monkeypatch.setattr(ws.Workspace, 'save', lambda self: saved.append(self) or self)
    monkeypatch.setattr(ws.Workspace, 'url', property(lambda self: 'https://wandb.test/view'))
    created = dashboard.configure_wandb_dashboard('lentix', 'numbergrid')
    workspace = saved[-1]
    assert created['status'] == 'created' and created['url'] == 'https://wandb.test/view'
    assert workspace.name == 'Disciples II' and workspace._internal_name == 'nw-disciplesii-v'
    assert workspace.settings.x_axis == 'training_steps' and workspace.settings.smoothing_weight == 0
    comet = compact_comet_view(game_map={'campaign_objective': 'capital_guardian'})['panels']['sections']
    assert [s.name for s in workspace.sections] == [s['title'] for s in comet]
    for section, expected in zip(workspace.sections, comet):
        assert section.is_open == expected['expanded']
        assert [(p.title, p.y, p.x) for p in section.panels] == [
            (p['chartName'], [p['metricName']], 'training_steps') for p in expected['panels']]
        layout = section.layout_settings
        assert layout.columns * layout.rows >= len(section.panels)  # one page, as in Comet
    # A later run regenerates rollout and campaign and keeps the other sections.
    workspace.sections = [ws.Section(name='campaign'), ws.Section(name='notes')]
    workspace._auto_generate_panels = True  # flags a user may have enabled in the view
    workspace.settings.sort_panels_alphabetically = True
    views['view'] = workspace
    updated = dashboard.configure_wandb_dashboard('lentix', 'numbergrid')
    assert updated['status'] == 'updated'
    assert saved[-1].auto_generate_panels is False and not saved[-1].settings.sort_panels_alphabetically
    assert [s.name for s in saved[-1].sections] == ['rollout', 'campaign', 'notes']
    assert len(saved[-1].sections[1].panels) == updated['campaign_charts'] == 49


@pytest.fixture
def comparison_runs(tmp_path):
    """Create complete run metadata without starting a learner."""
    import copy
    game_map = {'name': 'before', 'size': 48, 'enemy_rosters': [['squire']]}
    config = {'arch': {'seed': 42, 'total_timesteps': 1_000_000, 'num_envs': 250},
              'env': {'kwargs': {'map_config': game_map}},
              'system': {'epochs': 4, 'num_minibatches': 10},
              'network': {'layers': [128, 128]}}
    result = dict(seed=42, training_steps=1_000_000, device='test GPU', backend='gpu',
                  jax='0.8', python='3.12', cuda_runtime='13', num_envs=250, rollout_length=100,
                  mean_steps_per_second=200_000, weights_changed=True, checkpoint_roundtrip_verified=True)
    def write(name, *, map_config=None, config_change=None, result_change=None, packages=None):
        directory = tmp_path/name
        directory.mkdir(exist_ok=True)
        settings = copy.deepcopy(config)
        if map_config is not None:
            settings['env']['kwargs']['map_config'] = map_config
        if config_change:
            config_change(settings)
        for filename, data in [('results.json', {**result, **(result_change or {})}),
                               ('config.json', settings), ('map.json', settings['env']['kwargs']['map_config'])]:
            (directory/filename).write_text(json.dumps(data), encoding='utf-8')
        (directory/'packages.txt').write_text(packages or 'jax==0.8\noptax==0.2\nnvidia_cuda_runtime==13\n', encoding='utf-8')
        return directory
    return write, game_map


def test_comparison_accepts_normalized_packages_and_only_declared_map_change(comparison_runs):
    """Expected unit replacements remain comparable without hiding other configuration edits."""
    from scripts.run_unit_action_stage import map_changes, validate_comparison
    write, original = comparison_runs
    changed = {**original, 'name': 'after', 'enemy_rosters': [['witch']]}
    before = write('before')
    after = write('after', map_config=changed, packages=' OPTAX==0.2 \nNVIDIA-CUDA-Runtime==13\nJAX==0.8\n')
    declaration = map_changes(original, changed)
    validate_comparison(before, after, list(reversed(declaration)))
    with pytest.raises(ValueError, match='map changes'):
        validate_comparison(before, after)
    write('after', map_config={**changed, 'size': 49})
    with pytest.raises(ValueError, match='map changes'):
        validate_comparison(before, after, declaration)


def test_comparison_rejects_ppo_seed_budget_dependency_and_metadata_drift(comparison_runs):
    """Settings omitted by the former six-field check must invalidate speed comparisons."""
    from scripts.run_unit_action_stage import validate_comparison
    write, _ = comparison_runs
    before = write('before')
    for section, key, value in [('system', 'epochs', 2), ('system', 'num_minibatches', 5),
                                ('network', 'layers', [64, 64]), ('arch', 'seed', 43),
                                ('arch', 'total_timesteps', 5_000_000)]:
        after = write('after', config_change=lambda c, s=section, k=key, v=value: c[s].update({k: v}))
        with pytest.raises(ValueError, match='config.json'):
            validate_comparison(before, after)
    after = write('after', packages='jax==0.8\noptax==0.3\nnvidia_cuda_runtime==13\n')
    with pytest.raises(ValueError, match='packages.txt'):
        validate_comparison(before, after)
    after = write('after', result_change={'seed': 43})
    with pytest.raises(ValueError, match='seed'):
        validate_comparison(before, after)
    after = write('after')
    (after/'map.json').write_text(json.dumps({'name': 'unrelated'}))
    with pytest.raises(ValueError, match='disagree'):
        validate_comparison(before, after)


@pytest.mark.parametrize('mode', ['--baseline', '--baseline-run'])
def test_both_stage_modes_validate_saved_artifacts_before_reporting_regression(comparison_runs, monkeypatch, mode):
    """Both CLI paths reject incompatible runs before writing a regression percentage."""
    import sys
    from scripts import run_unit_action_stage as stage
    write, _ = comparison_runs
    before = write('before-42' if mode == '--baseline' else 'before')
    write('after-bench-42' if mode == '--baseline' else 'after-1m',
          config_change=lambda c: c['system'].update(epochs=2))
    monkeypatch.setattr(sys, 'argv', ['stage', 'after', '--root', str(before.parent), mode, 'before'])
    def checked_run(command, **kwargs):
        assert '--test-run' in command
        assert int(command[command.index('--total-timesteps')+1]) <= 1_000_000
        return Namespace(returncode=0)
    monkeypatch.setattr(stage.subprocess, 'run', checked_run)
    with pytest.raises(ValueError, match='config.json'):
        stage.main()
    assert not (before.parent/'after-comparison.json').exists()


def test_best_checkpoint_prefers_map_wins_and_exports_the_selected_weights(tmp_path, monkeypatch):
    """A high reward without the map objective must not beat an earlier winning model."""
    import jax.numpy as jnp
    from flax import serialization
    import benchmark_number_grid as benchmark
    from stoix.base_types import ActorCriticParams

    first = ActorCriticParams({'weight': jnp.asarray([1.])}, {'weight': jnp.asarray([11.])})
    last = ActorCriticParams({'weight': jnp.asarray([2.])}, {'weight': jnp.asarray([22.])})
    candidates = [benchmark.save_checkpoint(tmp_path, first, 1_000_000),
                  benchmark.save_checkpoint(tmp_path, last, 2_000_000)]
    for name in ('config.json', 'map.json', 'source_hashes.json', 'packages.txt'):
        (tmp_path / name).write_text('{}')
    def evaluate(params):
        wins = 1 if float(params['weight'][0]) == 1 else 0
        return {'successes': wins, 'success_rate': wins / 128,
                'mean_return': 1. if wins else 100., 'episodes': 128, 'seed': 2042}
    monkeypatch.setattr(benchmark, 'make_evaluator', lambda *args, **kwargs: evaluate)
    result = dict.fromkeys(('environment', 'algorithm', 'recurrent', 'device', 'backend', 'jax',
                           'cuda_runtime', 'python', 'seed', 'mean_steps_per_second', 'parameter_count',
                           'num_envs', 'rollout_length', 'observation_size', 'learner_source_sha256',
                           'source_hashes', 'tracking'), 'test')
    result.update(training_steps=2_000_000, final_evaluation={'successes': 0}, sampled_evaluation={'successes': 0})
    final = Mock(return_value={'successes': 8, 'success_rate': 8 / 1024, 'episodes': 1024, 'seed': 1042})
    sampled = Mock(return_value={'successes': 5, 'success_rate': 5 / 1024, 'episodes': 1024, 'seed': 1042})
    selected = benchmark.select_best_checkpoint(tmp_path, candidates, None, last, {}, result,
                                                final, sampled, Mock(), Mock())
    saved = json.loads((tmp_path / 'best/results.json').read_text())
    weights = serialization.from_bytes(last, (tmp_path / 'best/params.msgpack').read_bytes())
    assert float(weights.actor_params['weight'][0]) == 1.
    assert float(weights.critic_params['weight'][0]) == 11.
    assert selected['training_steps'] == saved['training_steps'] == 1_000_000
    assert saved['source_run_training_steps'] == 2_000_000
    assert saved['final_evaluation']['successes'] == 8
    assert saved['checkpoint_sha256'] == candidates[0]['sha256']
    assert float(final.call_args.args[0]['weight'][0]) == 1.
    records = json.loads((tmp_path / 'checkpoint_selection.json').read_text())
    assert len(records) == 2
    assert benchmark.checkpoint_score(records[0]) > benchmark.checkpoint_score(records[1])
    tied = {**records[0], 'training_steps': 2_000_000,
            'evaluation': {**records[0]['evaluation'], 'mean_return': 2.}}
    assert benchmark.checkpoint_score(tied) > benchmark.checkpoint_score(records[0])
