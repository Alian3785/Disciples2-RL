"""Do not mistake undefined episode statistics for lost cloud chart points."""
import pytest

from scripts.train_brute_six_maps import expected_metric_steps

TIMING = 'campaign/mean_seconds_per_million_steps'
REWARD = 'rollout/ep_rew_mean'
VICTORY = 'campaign/window/victory_rate'


def test_no_episode_yet_has_timing_but_no_invented_reward_or_victory():
    rows = [dict(training_steps=1_000_000, **{TIMING: 3., 'episodes/completed_total': 0,
                                            'campaign/window/episodes': 0}),
            dict(training_steps=2_000_000, **{TIMING: 3., REWARD: -10., VICTORY: 0.,
                                            'episodes/completed_total': 250,
                                            'campaign/window/episodes': 250})]
    expected = expected_metric_steps(rows, 2_000_000)
    assert expected[TIMING] == {1_000_000, 2_000_000}
    assert expected[REWARD] == expected[VICTORY] == {2_000_000}


def test_empty_interval_keeps_rolling_reward_without_inventing_victory_rate():
    rows = [dict(training_steps=1_000_000, **{TIMING: 3., REWARD: -10., VICTORY: 0.,
                                            'episodes/completed_total': 250,
                                            'campaign/window/episodes': 250}),
            dict(training_steps=2_000_000, **{TIMING: 3., REWARD: -10.,
                                            'episodes/completed_total': 250,
                                            'campaign/window/episodes': 0})]
    expected = expected_metric_steps(rows, 2_000_000)
    assert expected[REWARD] == {1_000_000, 2_000_000}
    assert expected[VICTORY] == {1_000_000}


def test_missing_defined_episode_statistics_is_a_failure():
    with pytest.raises(AssertionError, match='Missing defined victory rate'):
        expected_metric_steps([dict(training_steps=1_000_000, **{
            TIMING: 3., REWARD: -10., 'episodes/completed_total': 250,
            'campaign/window/episodes': 250})], 1_000_000)


def test_missing_summary_is_a_failure():
    with pytest.raises(AssertionError, match='Missing local summaries'):
        expected_metric_steps([dict(training_steps=1_000_000, **{
            TIMING: 3., REWARD: -10., VICTORY: 0.})], 2_000_000)
