import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from visualize_campaign import CampaignVisualizer


def _blue_unit(env: CampaignEnv, position: int) -> dict:
    return next(u for u in env.blue_team_state if int(u.get("position", -1)) == int(position))


def _clear_slot(env: CampaignEnv, position: int) -> None:
    unit = _blue_unit(env, position)
    unit["name"] = "РїСѓСЃС‚Рѕ"
    unit["hp"] = 0.0
    unit["health"] = 0.0
    unit["maxhp"] = 0.0
    unit["max_health"] = 0.0
    unit["exp_required"] = 0
    unit["exp_current"] = 0
    unit["next_level_exp"] = 0


def _enable_mercenary_hire(env: CampaignEnv, position: tuple[int, int], gold: float) -> dict:
    env.reset(seed=123)
    env.grid_env.agent_pos = position
    hero = _blue_unit(env, 8)
    env.gold = float(gold)
    return hero


def test_mercenary_interaction_tiles_match_scenario_layout():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)

    assert env.mercenary_site_anchors["Лагерь северных варваров"] == (9, 42)
    assert env.mercenary_site_anchors["Пустой лагерь наёмников"] == (15, 1)
    assert env.mercenary_site_interaction_tiles["Лагерь северных варваров"] == (
        (12, 43),
        (12, 44),
        (10, 45),
        (11, 45),
        (12, 45),
    )
    assert env.mercenary_site_interaction_tiles["Пустой лагерь наёмников"] == (
        (18, 2),
        (18, 3),
        (16, 4),
        (17, 4),
        (18, 4),
    )


def test_grid_info_reports_when_agent_stands_on_mercenary_interaction_tile():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env.grid_env.agent_pos = (12, 43)
    info = env._mercenary_context_info()

    assert info["is_at_mercenary_camp"] is True
    assert info["mercenary_sites_in_range"] == ["Лагерь северных варваров"]
    assert info["mercenary_stock"]["Лагерь северных варваров"][0]["unit_name"] == "Варвар"
    assert info["mercenary_stock"]["Лагерь северных варваров"][0]["stock"] == 1


def test_reset_syncs_mercenary_tiles_into_grid_env():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env.grid_env.mercenary_positions == set(env.mercenary_interaction_tiles)


def test_grid_env_tracks_mercenary_tiles():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert (12, 43) in env.grid_env.mercenary_positions


def test_campaign_visualizer_marks_diagonal_opposite_mercenary_tiles_with_h():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    viz = CampaignVisualizer(grid_size=env.grid_size)

    try:
        viz.draw_grid(
            agent_pos=env.grid_env.agent_pos,
            enemy_positions=env.grid_env.enemy_positions,
            enemies_alive=env.grid_env.enemies_alive,
            visited_cells=env.grid_env.visited_cells,
            obstacle_tiles=env.grid_env.obstacle_positions,
            mercenary_positions=env.grid_env.mercenary_positions,
            mercenary_site_anchors=env.mercenary_site_anchors,
        )
        mercenary_marker_positions = {
            tuple(text.get_position())
            for text in viz.ax.texts
            if text.get_text() == "H"
        }
        expected_positions = {
            tuple(viz._display_tile(anchor[0] + 2, anchor[1] + 2))
            for anchor in env.mercenary_site_anchors.values()
        }
        assert mercenary_marker_positions == expected_positions
    finally:
        viz.close()


def test_mercenary_hire_recruits_barbarian_on_frontline_and_spends_gold():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    hero = _enable_mercenary_hire(env, position=(12, 43), gold=850.0)
    _clear_slot(env, 7)

    assert len(env.active_mercenary_hire_options) == 3
    action = env.GRID_MERCENARY_HIRE_ACTION_START
    mask = env.compute_action_mask()

    assert bool(mask[action]) is True

    _, reward, terminated, truncated, info = env.step(action)

    expected = env._build_unit_from_data(
        env._find_unit_data_by_name("Варвар"),
        "blue",
        7,
    )
    recruited = _blue_unit(env, 7)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(3.0 * float(env.reward_defeat_enemy))
    assert recruited["name"] == expected["name"]
    assert recruited["health"] == expected["health"]
    assert recruited["max_health"] == expected["max_health"]
    assert recruited["damage"] == expected["damage"]
    assert recruited["initiative"] == expected["initiative"]
    assert recruited["unit_type"] == expected["unit_type"]
    assert hero["needaunit"] == 0
    assert env.gold == pytest.approx(0.0)
    assert info["hired"] is True
    assert info["hired_unit_name"] == "Варвар"
    assert info["hired_position"] == 7
    assert info["hire_action_kind"] == "mercenary"
    assert info["mercenary_hire_action"] is True
    assert info["mercenary_hire_site"] == "Лагерь северных варваров"
    assert info["mercenary_hire_stand"] == "ahead"
    assert info["mercenary_hire_stock_before"] == 1
    assert info["mercenary_hire_stock_after"] == 0
    assert env.mercenary_site_rosters["Лагерь северных варваров"][0]["stock"] == 0


def test_second_mercenary_camp_offers_goblin_and_goblin_archer_with_front_and_back_rows():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    _enable_mercenary_hire(env, position=(18, 2), gold=999.0)
    _clear_slot(env, 7)
    _clear_slot(env, 10)

    info = env._mercenary_context_info()
    assert info["mercenary_sites_in_range"] == ["Пустой лагерь наёмников"]
    camp_stock = info["mercenary_stock"]["Пустой лагерь наёмников"]
    assert [entry["unit_name"] for entry in camp_stock] == ["Гоблин", "Гоблин лучник"]
    assert [entry["stand"] for entry in camp_stock] == ["ahead", "behind"]
    assert [entry["gold"] for entry in camp_stock] == [50.0, 50.0]

    mask = env.compute_action_mask()
    option_by_name = {
        str(option["unit_name"]): env.GRID_MERCENARY_HIRE_ACTION_START + idx
        for idx, option in enumerate(env.active_mercenary_hire_options)
    }

    assert bool(mask[option_by_name["Гоблин"]]) is True
    assert bool(mask[option_by_name["Гоблин лучник"]]) is True

    barbarian_action = option_by_name["Варвар"]
    assert bool(mask[barbarian_action]) is False
