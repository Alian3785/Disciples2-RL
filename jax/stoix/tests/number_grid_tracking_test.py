"""Regression checks for the meaning of the episode charts (no GPU required)."""
from argparse import Namespace
from unittest.mock import Mock

import pytest

from numbergrid_tracking import NumberGridTracking


def tracker():
    return NumberGridTracking(Namespace(wandb_mode='disabled'))


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


def test_battle_wins_are_not_counted_as_whole_map_wins():
    tracking = tracker()
    metrics = tracking.training_metrics(
        row(10, 1, battle_victory=42, battle_transition=125_000), 250_000, 1.0, 1.1)
    assert metrics['episodes/wins'] == 1
    assert metrics['battles/wins'] == 42
    assert metrics['battles/transition_fraction'] == 0.5
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


def test_disabled_tracking_preserves_metadata_without_creating_wandb_run(tmp_path):
    with tracker() as tracking:
        tracking.start(tmp_path, {}, {}, {})
        tracking.log({'training_steps': 0})
        assert tracking.run is None
    import json
    metadata = json.loads((tmp_path / 'wandb.json').read_text())
    assert metadata['mode'] == 'disabled'
    assert metadata['status'] == 'finished'


def test_million_step_summary_includes_all_episodes_with_weighted_means():
    tracking = tracker()
    tracking.run = Mock()
    for index, (episodes, wins, length) in enumerate(((1, 1, 10), (3, 0, 30), (0, 0, 0), (6, 2, 50))):
        summary = tracking.log_training(
            row(episodes, wins, (index + 1) * 250_000, mean_episode_length=length,
                mean_episode_return=length / 10, battle_victory=10, battle_transition=100_000),
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
    assert logged['battles/transition_fraction'] == pytest.approx(0.4)
    assert logged['performance/steps_per_second'] == 500_000


def test_final_partial_window_is_logged_once_and_counts_are_preserved():
    tracking = tracker()
    tracking.run = Mock()
    for index in range(5):
        tracking.log_training(row(2, 1, (index + 1) * 250_000), 250_000, 0.5,
                              (index + 1) * 0.6, final=index == 4)
    calls = [call.args[0] for call in tracking.run.log.call_args_list]
    assert [call['training_steps'] for call in calls] == [1_000_000, 1_250_000]
    assert calls[0]['episodes/completed'] == 8
    assert calls[1]['episodes/completed'] == 2
    assert calls[1]['episodes/completed_total'] == 10
    assert calls[1]['episodes/wins_total'] == 5
    assert not tracking.pending


def test_smoke_run_shorter_than_interval_still_has_training_chart_point():
    tracking = tracker()
    tracking.run = Mock()
    tracking.log_training(row(4, 1), 250_000, 0.5, 0.6, final=True)
    tracking.run.log.assert_called_once()
    assert tracking.run.log.call_args.args[0]['episodes/wins'] == 1
