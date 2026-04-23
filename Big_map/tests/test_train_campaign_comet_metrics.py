import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

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


ACTIVE_CAMPAIGN_METRICS = {
    "campaign/battle_item_use_episodes_rate",
    "campaign/battle_items_used_per_episode_mean",
    "campaign/battle_victories_per_episode_mean",
    "campaign/battle_win_rate",
    "campaign/best_recent_victory_rate",
    "campaign/castle_heal_episodes_rate",
    "campaign/chests_collected_per_episode_mean",
    "campaign/defeat_rate",
    "campaign/episodes_total",
    "campaign/hire_episodes_rate",
    "campaign/hired_units_per_episode_mean",
    "campaign/merchant_sale_gold_per_episode_mean",
    "campaign/ruin_clear_episodes_rate",
    "campaign/ruins_cleared_per_episode_mean",
    "campaign/sold_items_per_episode_mean",
    "campaign/spell_cast_episodes_rate",
    "campaign/spell_casts_per_episode_mean",
    "campaign/spells_learned_per_episode_mean",
    "campaign/summon_episodes_rate",
    "campaign/summoned_units_per_episode_mean",
    "campaign/timeout_rate",
    "campaign/unit_upgrades_total",
    "campaign/victories_total",
    "campaign/victory_rate",
    "campaign/window/battle_win_rate",
    "campaign/window/chests_collected_mean",
    "campaign/window/defeat_rate",
    "campaign/window/episodes",
    "campaign/window/hired_units_mean",
    "campaign/window/ruins_cleared_mean",
    "campaign/window/spell_casts_mean",
    "campaign/window/timeout_rate",
    "campaign/window/victory_rate",
}


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
                "mode": "grid",
                "enemy_id": 68,
                "equip_battle_item_action": True,
                "equip_battle_item_applied": True,
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
                "spell_cast_action": True,
                "spell_cast_executed": True,
                "spell_kind": "summon_battle",
                "spell_summon_unit_name": "Белиарх",
                "spell_units_affected": 1,
                "hired": True,
                "hired_units_count": 1,
                "battle_hero_item_action": True,
                "battle_hero_item_applied": True,
                "battle_hero_item_consumed": True,
                "episode": {"r": 12.0, "l": 123},
                "turns": 12,
                "gold": 120,
                "moves": 1,
            }
        ]
    }
    callback.num_timesteps = 1000

    assert callback._on_step()
    assert len(dummy_experiment.logged) == 1

    payload, step = dummy_experiment.logged[0]
    assert step == 1000
    assert set(payload) == ACTIVE_CAMPAIGN_METRICS
    assert payload["campaign/episodes_total"] == 1
    assert payload["campaign/victories_total"] == 1.0
    assert payload["campaign/victory_rate"] == 1.0
    assert payload["campaign/defeat_rate"] == 0.0
    assert payload["campaign/timeout_rate"] == 0.0
    assert payload["campaign/battle_victories_per_episode_mean"] == 1.0
    assert payload["campaign/battle_win_rate"] == 1.0
    assert payload["campaign/unit_upgrades_total"] == 2.0
    assert payload["campaign/castle_heal_episodes_rate"] == 1.0
    assert payload["campaign/chests_collected_per_episode_mean"] == 2.0
    assert payload["campaign/ruins_cleared_per_episode_mean"] == 0.0
    assert payload["campaign/ruin_clear_episodes_rate"] == 0.0
    assert payload["campaign/sold_items_per_episode_mean"] == 3.0
    assert payload["campaign/merchant_sale_gold_per_episode_mean"] == 3650.0
    assert payload["campaign/spells_learned_per_episode_mean"] == 0.0
    assert payload["campaign/spell_casts_per_episode_mean"] == 1.0
    assert payload["campaign/spell_cast_episodes_rate"] == 1.0
    assert payload["campaign/summoned_units_per_episode_mean"] == 1.0
    assert payload["campaign/summon_episodes_rate"] == 1.0
    assert payload["campaign/hired_units_per_episode_mean"] == 1.0
    assert payload["campaign/hire_episodes_rate"] == 1.0
    assert payload["campaign/battle_items_used_per_episode_mean"] == 1.0
    assert payload["campaign/battle_item_use_episodes_rate"] == 1.0
    assert payload["campaign/window/episodes"] == 1.0
    assert payload["campaign/window/victory_rate"] == 1.0
    assert payload["campaign/window/defeat_rate"] == 0.0
    assert payload["campaign/window/timeout_rate"] == 0.0
    assert payload["campaign/window/battle_win_rate"] == 1.0
    assert payload["campaign/window/chests_collected_mean"] == 2.0
    assert payload["campaign/window/ruins_cleared_mean"] == 0.0
    assert payload["campaign/window/spell_casts_mean"] == 1.0
    assert payload["campaign/window/hired_units_mean"] == 1.0
    assert payload["campaign/best_recent_victory_rate"] == 1.0
    assert "campaign/window_enemy_encounters/enemy_31" not in payload
    assert "campaign/window_enemy_victories/enemy_31" not in payload
    assert "campaign/window_victory_reasons/objective_cities_cleared" not in payload
    assert dummy_experiment.metric_logged == []
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
    assert set(payload) == ACTIVE_CAMPAIGN_METRICS
    assert payload["campaign/episodes_total"] == 1
    assert payload["campaign/victories_total"] == 0.0
    assert payload["campaign/victory_rate"] == 0.0
    assert payload["campaign/defeat_rate"] == 1.0
    assert payload["campaign/timeout_rate"] == 0.0
    assert payload["campaign/battle_victories_per_episode_mean"] == 0.0
    assert payload["campaign/battle_win_rate"] == 0.0
    assert payload["campaign/unit_upgrades_total"] == 0.0
    assert payload["campaign/castle_heal_episodes_rate"] == 0.0
    assert payload["campaign/chests_collected_per_episode_mean"] == 0.0
    assert payload["campaign/ruins_cleared_per_episode_mean"] == 0.0
    assert payload["campaign/ruin_clear_episodes_rate"] == 0.0
    assert payload["campaign/sold_items_per_episode_mean"] == 0.0
    assert payload["campaign/merchant_sale_gold_per_episode_mean"] == 0.0
    assert payload["campaign/spells_learned_per_episode_mean"] == 0.0
    assert payload["campaign/spell_casts_per_episode_mean"] == 0.0
    assert payload["campaign/spell_cast_episodes_rate"] == 0.0
    assert payload["campaign/summoned_units_per_episode_mean"] == 0.0
    assert payload["campaign/summon_episodes_rate"] == 0.0
    assert payload["campaign/hired_units_per_episode_mean"] == 0.0
    assert payload["campaign/hire_episodes_rate"] == 0.0
    assert payload["campaign/battle_items_used_per_episode_mean"] == 0.0
    assert payload["campaign/battle_item_use_episodes_rate"] == 0.0
    assert payload["campaign/window/episodes"] == 1.0
    assert payload["campaign/window/victory_rate"] == 0.0
    assert payload["campaign/window/defeat_rate"] == 1.0
    assert payload["campaign/window/timeout_rate"] == 0.0
    assert payload["campaign/window/chests_collected_mean"] == 0.0
    assert payload["campaign/window/ruins_cleared_mean"] == 0.0
    assert payload["campaign/window/spell_casts_mean"] == 0.0
    assert payload["campaign/window/hired_units_mean"] == 0.0
    assert payload["campaign/window/battle_win_rate"] == 0.0
    assert payload["campaign/best_recent_victory_rate"] == 0.0
    assert "campaign/window_enemy_encounters/enemy_7" not in payload
    assert "campaign/window_enemy_defeats/enemy_7" not in payload
    assert dummy_experiment.metric_logged == []
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


def test_campaign_metrics_callback_saves_battle_item_activity_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_battle_items_equipped = [1.0, 0.0, 2.0, 1.0]
    callback.episode_battle_items_used = [0.0, 1.0, 1.0, 2.0]

    output_path = tmp_path / "campaign_battle_item_activity.png"
    saved_path = callback.save_battle_item_activity_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_campaign_metrics_callback_saves_spell_cast_activity_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_spell_casts = [0.0, 1.0, 3.0, 2.0]

    output_path = tmp_path / "campaign_spell_cast_activity.png"
    saved_path = callback.save_spell_cast_activity_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_campaign_metrics_callback_saves_summoned_units_per_episode_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_summoned_units = [0.0, 1.0, 2.0, 1.0]

    output_path = tmp_path / "campaign_summoned_units_per_episode.png"
    saved_path = callback.save_summoned_units_per_episode_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_campaign_metrics_callback_saves_cumulative_summoned_units_plot(tmp_path):
    callback = CampaignMetricsCallback(log_freq=1000, episode_window=4, experiment=None)
    callback.episode_summoned_units = [0.0, 1.0, 2.0, 1.0]

    output_path = tmp_path / "campaign_summoned_units_cumulative.png"
    saved_path = callback.save_cumulative_summoned_units_plot(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
