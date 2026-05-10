import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def test_grid_movement_is_blocked_after_all_starting_moves_until_rest():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    start_pos = tuple(env.grid_env.agent_pos)
    far_enemy_id = max(
        env.grid_env.enemy_positions,
        key=lambda enemy_id: (
            abs(env.grid_env.enemy_positions[enemy_id][0] - start_pos[0])
            + abs(env.grid_env.enemy_positions[enemy_id][1] - start_pos[1])
        ),
    )
    env.grid_env.enemies_alive = {
        enemy_id: enemy_id == far_enemy_id
        for enemy_id in env.grid_env.enemy_positions
    }

    assert env.moves == env.moves_per_turn == 20

    move_mask = env.grid_env.compute_action_mask()
    move_action = next(index for index, allowed in enumerate(move_mask[:8]) if allowed)
    env.moves = int(env.GRID_MOVE_COST)

    _, reward, terminated, truncated, info = env.step(move_action)
    assert terminated is False
    assert truncated is False
    assert info.get("blocked_by_moves", False) is False
    assert reward != 0.0
    assert env.moves == 0

    mask_at_zero = env.compute_action_mask()
    assert mask_at_zero[:8].tolist() == [False] * 8

    # Movement attempt must be blocked while moves == 0.
    _, blocked_reward, terminated, truncated, blocked_info = env.step(move_action)
    assert terminated is False
    assert truncated is False
    assert blocked_reward == 0.0
    assert blocked_info.get("blocked_by_moves", False) is True
    assert env.moves == 0

    # REST ends turn, restores moves, and re-enables movement.
    _, rest_reward, terminated, truncated, rest_info = env.step(8)
    assert terminated is False
    assert truncated is False
    assert rest_info.get("rest_action", False) is True
    assert env.moves == env.moves_per_turn == 20
    assert rest_reward < 0.0

    mask_after_rest = env.compute_action_mask()
    assert mask_after_rest[:8].tolist() == env.grid_env.compute_action_mask()[:8].tolist()
