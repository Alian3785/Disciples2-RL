import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_campaign import is_better_campaign_eval


def test_victory_rate_beats_higher_reward():
    assert is_better_campaign_eval(
        victory_rate=0.3,
        mean_reward=10.0,
        best_victory_rate=0.2,
        best_mean_reward=100.0,
    )


def test_lower_victory_rate_never_beats_best():
    assert not is_better_campaign_eval(
        victory_rate=0.2,
        mean_reward=1000.0,
        best_victory_rate=0.3,
        best_mean_reward=-1000.0,
    )


def test_mean_reward_breaks_tie_when_victory_rate_matches():
    assert is_better_campaign_eval(
        victory_rate=0.3,
        mean_reward=50.0,
        best_victory_rate=0.3,
        best_mean_reward=40.0,
    )
    assert not is_better_campaign_eval(
        victory_rate=0.3,
        mean_reward=40.0,
        best_victory_rate=0.3,
        best_mean_reward=50.0,
    )
