import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from visualize_campaign import CampaignVisualizer, _format_grid_action_label


def _blue_unit(env: CampaignEnv, position: int) -> dict:
    return next(u for u in env.blue_team_state if int(u.get("position", -1)) == int(position))


def _enable_trainer(env: CampaignEnv, gold: float = 999.0) -> None:
    env.reset(seed=123)
    env.grid_env.agent_pos = (41, 31)
    env.gold = float(gold)


def _clear_slot(env: CampaignEnv, position: int) -> None:
    unit = _blue_unit(env, position)
    unit["name"] = "пусто"
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["maxhp"] = 0.0
    unit["max_health"] = 0.0
    unit["exp_required"] = 0
    unit["exp_current"] = 0
    unit["next_level_exp"] = 0


def test_trainer_interaction_tiles_match_scenario_layout():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)

    assert env.trainer_site_anchors["Тренировочный лагерь"] == (38, 30)
    assert env.trainer_site_interaction_tiles["Тренировочный лагерь"] == (
        (41, 31),
        (41, 32),
        (39, 33),
        (40, 33),
        (41, 33),
    )


def test_grid_info_and_sync_report_trainer_site():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    env.grid_env.agent_pos = (41, 31)

    info = env._trainer_context_info()

    assert info["is_at_trainer"] is True
    assert info["trainer_sites_in_range"] == ["Тренировочный лагерь"]
    assert env.grid_env.trainer_positions == set(env.trainer_interaction_tiles)


def test_grid_render_and_visualizer_mark_trainer_tiles():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    rendered = env.grid_env.render(mode="ansi")

    assert "Trainers:" in rendered
    assert "(41, 31)" in rendered
    assert "R " in rendered

    viz = CampaignVisualizer(grid_size=env.grid_size)
    try:
        viz.draw_grid(
            agent_pos=env.grid_env.agent_pos,
            enemy_positions=env.grid_env.enemy_positions,
            enemies_alive=env.grid_env.enemies_alive,
            visited_cells=env.grid_env.visited_cells,
            obstacle_tiles=env.grid_env.obstacle_positions,
            trainer_positions=env.grid_env.trainer_positions,
            trainer_site_anchors=env.trainer_site_anchors,
        )
        trainer_marker_positions = {
            tuple(text.get_position())
            for text in viz.ax.texts
            if text.get_text() == "R"
        }
        expected_positions = {
            tuple(viz._display_tile(anchor[0] + 2, anchor[1] + 2))
            for anchor in env.trainer_site_anchors.values()
        }
        assert trainer_marker_positions == expected_positions
    finally:
        viz.close()


def test_trainer_action_count_matches_blue_positions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _enable_trainer(env)

    assert env.GRID_MERCHANT_BUY_ACTION_START - env.GRID_TRAINER_ACTION_START == len(
        env.TRAINER_POSITIONS
    )
    assert len(env.TRAINER_POSITIONS) == 6


def test_trainer_trains_faction_unit_to_cap_and_spends_gold():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _enable_trainer(env, gold=100.0)

    unit = _blue_unit(env, 7)
    unit["hero"] = 0
    unit["capital"] = env.Realcapital
    unit["is_neutral_unit"] = False
    unit["Level"] = 1
    unit["exp_current"] = 10
    unit["exp_required"] = 20
    unit["next_level_exp"] = 0

    action = env.GRID_TRAINER_ACTION_START
    mask = env.compute_action_mask()

    assert bool(mask[action]) is True

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert int(unit["exp_current"]) == 19
    assert env.gold == pytest.approx(64.0)
    assert info["trainer_trained"] is True
    assert info["trainer_xp_gained"] == 9
    assert info["trainer_gold_per_xp"] == pytest.approx(4.0)
    assert info["trainer_gold_spent"] == pytest.approx(36.0)
    assert info["trainer_exp_after"] == 19


def test_trainer_uses_partial_gold_when_not_enough_for_full_training():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _enable_trainer(env, gold=10.0)

    unit = _blue_unit(env, 7)
    unit["hero"] = 0
    unit["capital"] = env.Realcapital
    unit["is_neutral_unit"] = False
    unit["Level"] = 1
    unit["exp_current"] = 10
    unit["exp_required"] = 20
    unit["next_level_exp"] = 0

    _, _, _, _, info = env.step(env.GRID_TRAINER_ACTION_START)

    assert int(unit["exp_current"]) == 12
    assert env.gold == pytest.approx(2.0)
    assert info["trainer_trained"] is True
    assert info["trainer_xp_gained"] == 2
    assert info["trainer_gold_spent"] == pytest.approx(8.0)


def test_trainer_uses_five_gold_per_xp_for_neutral_and_hero_units():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _enable_trainer(env, gold=100.0)

    neutral_unit = _blue_unit(env, 9)
    neutral_unit["hero"] = 0
    neutral_unit["capital"] = 0
    neutral_unit["is_neutral_unit"] = True
    neutral_unit["Level"] = 4
    neutral_unit["exp_current"] = 0
    neutral_unit["exp_required"] = 10
    neutral_unit["next_level_exp"] = 0

    hero_unit = _blue_unit(env, 8)
    hero_unit["hero"] = 1
    hero_unit["capital"] = env.Realcapital
    hero_unit["is_neutral_unit"] = False
    hero_unit["Level"] = 4
    hero_unit["exp_current"] = 0
    hero_unit["exp_required"] = 10

    neutral_preview = env._trainer_preview_for_position(9)
    hero_preview = env._trainer_preview_for_position(8)

    assert neutral_preview["gold_per_xp"] == pytest.approx(5.0)
    assert hero_preview["gold_per_xp"] == pytest.approx(5.0)

    neutral_action = env.GRID_TRAINER_ACTION_START + env.TRAINER_POSITIONS.index(9)
    hero_action = env.GRID_TRAINER_ACTION_START + env.TRAINER_POSITIONS.index(8)

    _, _, _, _, neutral_info = env.step(neutral_action)
    assert int(neutral_unit["exp_current"]) == 9
    assert env.gold == pytest.approx(55.0)
    assert neutral_info["trainer_gold_spent"] == pytest.approx(45.0)

    env.gold = 100.0
    hero_unit["exp_current"] = 0
    _, _, _, _, hero_info = env.step(hero_action)
    assert int(hero_unit["exp_current"]) == 9
    assert env.gold == pytest.approx(55.0)
    assert hero_info["trainer_gold_spent"] == pytest.approx(45.0)


def test_trainer_masks_empty_and_dead_positions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    _enable_trainer(env, gold=999.0)

    dead_unit = _blue_unit(env, 11)
    dead_unit["hp"] = 0.0
    dead_unit["health"] = 0.0
    dead_unit["maxhp"] = max(1.0, float(dead_unit.get("maxhp", dead_unit.get("max_health", 1.0)) or 1.0))
    dead_unit["max_health"] = dead_unit["maxhp"]

    _clear_slot(env, 12)

    mask = env.compute_action_mask()
    dead_action = env.GRID_TRAINER_ACTION_START + env.TRAINER_POSITIONS.index(11)
    empty_action = env.GRID_TRAINER_ACTION_START + env.TRAINER_POSITIONS.index(12)

    assert bool(mask[dead_action]) is False
    assert bool(mask[empty_action]) is False


def test_visualize_campaign_formats_trainer_action_label():
    label = _format_grid_action_label(
        57,
        {
            "trainer_action": True,
            "trainer_unit_name": "Сквайр",
            "trainer_xp_gained": 9,
            "trainer_site": "Тренировочный лагерь",
        },
    )

    assert label == "Train: Сквайр +9 XP (Тренировочный лагерь)"
