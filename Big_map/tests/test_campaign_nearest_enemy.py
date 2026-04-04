import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _make_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    return env


def test_get_nearest_enemy_stack_prefers_shortest_walk_path():
    env = _make_env()
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        20: (6, 3),
        10: (4, 4),
    }
    env.grid_env.enemies_alive = {20: True, 10: True}
    env.grid_env.obstacle_positions = set()

    nearest = env.get_nearest_enemy_stack()

    assert nearest is not None
    assert nearest["enemy_id"] == 10
    assert nearest["position"] == (4, 4)
    assert nearest["path_distance"] == 1
    assert nearest["reachable"] is True


def test_get_nearest_enemy_stack_breaks_equal_distance_ties_by_enemy_id():
    env = _make_env()
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        30: (2, 3),
        20: (4, 3),
    }
    env.grid_env.enemies_alive = {30: True, 20: True}
    env.grid_env.obstacle_positions = set()

    nearest = env.get_nearest_enemy_stack()

    assert nearest is not None
    assert nearest["enemy_id"] == 20
    assert nearest["path_distance"] == 1


def test_get_nearest_enemy_stack_skips_defeated_enemy():
    env = _make_env()
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (6, 3),
    }
    env.grid_env.enemies_alive = {10: False, 20: True}
    env.grid_env.obstacle_positions = set()

    nearest = env.get_nearest_enemy_stack()

    assert nearest is not None
    assert nearest["enemy_id"] == 20
    assert nearest["position"] == (6, 3)
    assert nearest["path_distance"] == 3


def test_get_nearest_enemy_stack_uses_fallback_when_no_enemy_is_reachable():
    env = _make_env()
    env.grid_env.agent_pos = (0, 0)
    env.grid_env.enemy_positions = {
        20: (3, 0),
        10: (2, 2),
    }
    env.grid_env.enemies_alive = {20: True, 10: True}
    env.grid_env.obstacle_positions = {
        (0, 1),
        (1, 0),
        (1, 1),
    }

    nearest = env.get_nearest_enemy_stack()

    assert nearest is not None
    assert nearest["enemy_id"] == 10
    assert nearest["position"] == (2, 2)
    assert nearest["path_distance"] is None
    assert nearest["fallback_distance"] == 2
    assert nearest["reachable"] is False


def test_get_nearest_enemy_stack_for_spells_skips_ruins_and_city_stacks():
    env = _make_env()
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        70: (4, 3),  # ruins, not spell-targetable
        22: (5, 3),  # city stack, not spell-targetable
        10: (6, 3),  # normal field stack
    }
    env.grid_env.enemies_alive = {70: True, 22: True, 10: True}
    env.grid_env.obstacle_positions = set()

    nearest_any = env.get_nearest_enemy_stack()
    nearest_spell = env.get_nearest_enemy_stack(spell_targetable_only=True)

    assert nearest_any is not None
    assert nearest_any["enemy_id"] == 70

    assert nearest_spell is not None
    assert nearest_spell["enemy_id"] == 10
    assert nearest_spell["position"] == (6, 3)
