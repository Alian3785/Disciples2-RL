import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import train_campaign
from train_campaign import CampaignMetricsCallback


class DummyCometExperiment:
    def __init__(self):
        self.logged = []
        self.metric_logged = []
        self.others = {}

    def log_metrics(self, payload, step=None):
        self.logged.append((payload, step))

    def log_metric(self, key, value, step=None):
        self.metric_logged.append((key, value, step))

    def log_other(self, key, value):
        self.others[key] = value


def test_campaign_metrics_callback_logs_custom_metrics(monkeypatch):
    dummy_experiment = DummyCometExperiment()

    callback = CampaignMetricsCallback(
        log_freq=1000,
        episode_window=4,
        experiment=dummy_experiment,
    )
    callback.locals = {
        "infos": [
            {
                "mode": "battle",
                "enemy_id": 68,
                "turns": 10,
                "gold": 125,
                "moves": 4,
            },
            {
                "mode": "grid",
                "enemy_id": 68,
                "battle_result": "victory",
                "campaign_result": "victory",
                "campaign_victory_reason": "objective_cities_cleared",
                "collected_chests": [
                    {"pos": (1, 1), "items": ["Potion of Healing"], "granted_items": []},
                    {"pos": (2, 2), "items": ["Life Potion"], "granted_items": []},
                ],
                "castle_heal_action": True,
                "healed_amount": 37.5,
                "gold_spent": 40.0,
                "blue_hp_ratio": 0.75,
                "unit_upgrades": 2,
                "enemy_defeat_reward": 4.0,
                "blue_exp_reward": 1.5,
                "final_objective_reward": 2.5,
                "merchant_sold_items_count": 3,
                "merchant_sale_gold": 3650.0,
                "hired": True,
                "hired_units_count": 1,
                "episode": {"r": 12.0, "l": 123},
                "turns": 12,
                "gold": 120,
                "moves": 1,
            },
        ]
    }
    callback.num_timesteps = 1000

    assert callback._on_step()
    assert len(dummy_experiment.logged) == 1

    payload, step = dummy_experiment.logged[0]
    assert step == 1000
    assert payload["campaign/episodes_total"] == 1
    assert payload["campaign/victory_rate"] == 1.0
    assert payload["campaign/battle_victories_per_episode_mean"] == 1.0
    assert payload["campaign/battle_win_rate"] == 1.0
    assert payload["campaign/recent_episode_reward_mean"] == 12.0
    assert payload["campaign/recent_blue_hp_ratio_mean"] == 0.75
    assert payload["campaign/castle_heal_uses_total"] == 1.0
    assert payload["campaign/castle_healed_hp_total"] == 37.5
    assert payload["campaign/castle_heal_episodes_rate"] == 1.0
    assert payload["campaign/chests_collected_total"] == 2.0
    assert payload["campaign/chests_collected_per_episode_mean"] == 2.0
    assert payload["campaign/chest_collection_episodes_rate"] == 1.0
    assert payload["campaign/sold_items_total"] == 3.0
    assert payload["campaign/sold_items_per_episode_mean"] == 3.0
    assert payload["campaign/merchant_sale_gold_total"] == 3650.0
    assert payload["campaign/merchant_sale_gold_per_episode_mean"] == 3650.0
    assert payload["campaign/sale_episodes_rate"] == 1.0
    assert payload["campaign/hired_units_total"] == 1.0
    assert payload["campaign/hired_units_per_episode_mean"] == 1.0
    assert payload["campaign/hire_episodes_rate"] == 1.0
    assert payload["campaign/recent_castle_heal_uses_mean"] == 1.0
    assert payload["campaign/recent_castle_heal_episodes_rate"] == 1.0
    assert payload["campaign/recent_castle_healed_hp_mean"] == 37.5
    assert payload["campaign/recent_chests_collected_mean"] == 2.0
    assert payload["campaign/recent_chest_collection_episodes_rate"] == 1.0
    assert payload["campaign/recent_sold_items_mean"] == 3.0
    assert payload["campaign/recent_sale_episodes_rate"] == 1.0
    assert payload["campaign/recent_merchant_sale_gold_mean"] == 3650.0
    assert payload["campaign/recent_hired_units_mean"] == 1.0
    assert payload["campaign/recent_hire_episodes_rate"] == 1.0
    assert payload["campaign/window/chests_collected_mean"] == 2.0
    assert payload["campaign/window/sold_items_mean"] == 3.0
    assert payload["campaign/window/hired_units_mean"] == 1.0
    assert "campaign/window_enemy_encounters/enemy_31" not in payload
    assert "campaign/window_enemy_victories/enemy_31" not in payload
    assert payload["campaign/window_victory_reasons/objective_cities_cleared"] == 1.0
    assert ("campaign/episode_chests_collected", 2.0, 1000) in dummy_experiment.metric_logged
    assert ("campaign/episode_sold_items", 3.0, 1000) in dummy_experiment.metric_logged
    assert ("campaign/episode_merchant_sale_gold", 3650.0, 1000) in dummy_experiment.metric_logged
    assert ("campaign/episode_hired_units", 1.0, 1000) in dummy_experiment.metric_logged
    assert dummy_experiment.others["summary_campaign_victory_rate"] == 1.0


def test_campaign_metrics_callback_flushes_partial_window_on_training_end():
    dummy_experiment = DummyCometExperiment()
    callback = CampaignMetricsCallback(
        log_freq=1000,
        episode_window=4,
        experiment=dummy_experiment,
    )
    callback.locals = {
        "infos": [
            {
                "mode": "battle",
                "enemy_id": 7,
                "turns": 20,
                "gold": 50,
                "moves": 0,
            },
            {
                "mode": "battle",
                "enemy_id": 7,
                "battle_result": "defeat",
                "campaign_result": "defeat",
                "episode": {"r": -4.0, "l": 50},
                "turns": 22,
                "gold": 48,
                "moves": 0,
            },
        ]
    }
    callback.num_timesteps = 750

    assert callback._on_step()
    assert dummy_experiment.logged == []

    callback._on_training_end()
    assert len(dummy_experiment.logged) == 1

    payload, step = dummy_experiment.logged[0]
    assert step == 750
    assert payload["campaign/episodes_total"] == 1
    assert payload["campaign/defeat_rate"] == 1.0
    assert payload["campaign/battle_victories_per_episode_mean"] == 0.0
    assert payload["campaign/castle_heal_uses_total"] == 0.0
    assert payload["campaign/castle_healed_hp_total"] == 0.0
    assert payload["campaign/castle_heal_episodes_rate"] == 0.0
    assert payload["campaign/chests_collected_total"] == 0.0
    assert payload["campaign/chests_collected_per_episode_mean"] == 0.0
    assert payload["campaign/chest_collection_episodes_rate"] == 0.0
    assert payload["campaign/sold_items_total"] == 0.0
    assert payload["campaign/sold_items_per_episode_mean"] == 0.0
    assert payload["campaign/merchant_sale_gold_total"] == 0.0
    assert payload["campaign/merchant_sale_gold_per_episode_mean"] == 0.0
    assert payload["campaign/sale_episodes_rate"] == 0.0
    assert payload["campaign/hired_units_total"] == 0.0
    assert payload["campaign/hired_units_per_episode_mean"] == 0.0
    assert payload["campaign/hire_episodes_rate"] == 0.0
    assert payload["campaign/recent_castle_heal_uses_mean"] == 0.0
    assert payload["campaign/recent_castle_heal_episodes_rate"] == 0.0
    assert payload["campaign/recent_castle_healed_hp_mean"] == 0.0
    assert payload["campaign/recent_chests_collected_mean"] == 0.0
    assert payload["campaign/recent_chest_collection_episodes_rate"] == 0.0
    assert payload["campaign/recent_sold_items_mean"] == 0.0
    assert payload["campaign/recent_sale_episodes_rate"] == 0.0
    assert payload["campaign/recent_merchant_sale_gold_mean"] == 0.0
    assert payload["campaign/recent_hired_units_mean"] == 0.0
    assert payload["campaign/recent_hire_episodes_rate"] == 0.0
    assert payload["campaign/window/chests_collected_mean"] == 0.0
    assert payload["campaign/window/sold_items_mean"] == 0.0
    assert payload["campaign/window/hired_units_mean"] == 0.0
    assert payload["campaign/window/battle_win_rate"] == 0.0
    assert "campaign/window_enemy_encounters/enemy_7" not in payload
    assert "campaign/window_enemy_defeats/enemy_7" not in payload
    assert dummy_experiment.others["summary_campaign_battle_win_rate"] == 0.0


def test_campaign_metrics_callback_saves_chests_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_chests_collected = [0.0, 2.0, 1.0, 3.0]

    output_path = tmp_path / "campaign_chests_per_episode.png"
    saved_path = callback.save_chests_per_episode_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_campaign_metrics_callback_saves_sold_items_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_sold_items = [0.0, 1.0, 3.0, 2.0]

    output_path = tmp_path / "campaign_sold_items_per_episode.png"
    saved_path = callback.save_sold_items_per_episode_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_campaign_metrics_callback_saves_hired_units_per_episode_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_hired_units = [0.0, 1.0, 2.0, 1.0]

    output_path = tmp_path / "campaign_hired_units_per_episode.png"
    saved_path = callback.save_hired_units_per_episode_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_campaign_metrics_callback_saves_cumulative_hired_units_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_hired_units = [0.0, 1.0, 2.0, 1.0]

    output_path = tmp_path / "campaign_hired_units_cumulative.png"
    saved_path = callback.save_cumulative_hired_units_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
