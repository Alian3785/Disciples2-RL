import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from visualize_campaign import CampaignVisualizer


def test_merchant_interaction_tiles_match_scenario_layout():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)

    assert any(pos == (21, 34) for pos in env.merchant_site_anchors.values())
    assert any(pos == (36, 1) for pos in env.merchant_site_anchors.values())

    assert any(
        tiles
        == (
            (24, 35),
            (24, 36),
            (22, 37),
            (23, 37),
            (24, 37),
        )
        for tiles in env.merchant_site_interaction_tiles.values()
    )
    assert any(
        tiles
        == (
            (39, 2),
            (39, 3),
            (37, 4),
            (38, 4),
            (39, 4),
        )
        for tiles in env.merchant_site_interaction_tiles.values()
    )


def test_grid_info_reports_when_agent_stands_on_merchant_interaction_tile():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    env.grid_env.agent_pos = (39, 2)
    info = env._merchant_context_info()

    expected_name = next(
        site_name
        for site_name, tiles in env.merchant_site_interaction_tiles.items()
        if (39, 2) in tiles
    )
    assert info["is_at_merchant"] is True
    assert info["merchant_sites_in_range"] == [expected_name]


def test_reset_syncs_merchant_tiles_into_grid_env():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert env.grid_env.merchant_positions == set(env.merchant_interaction_tiles)


def test_grid_env_tracks_merchant_tiles():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)

    assert (39, 2) in env.grid_env.merchant_positions


def test_campaign_visualizer_marks_diagonal_opposite_merchant_tiles_with_t():
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
            merchant_positions=env.grid_env.merchant_positions,
            merchant_site_anchors=env.merchant_site_anchors,
        )
        merchant_marker_positions = {
            tuple(text.get_position())
            for text in viz.ax.texts
            if text.get_text() == "T"
        }
        expected_positions = {
            tuple(viz._display_tile(anchor[0] + 2, anchor[1] + 2))
            for anchor in env.merchant_site_anchors.values()
        }
        assert merchant_marker_positions == expected_positions
    finally:
        viz.close()
