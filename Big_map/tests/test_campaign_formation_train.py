import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.formation_train import FORMATION_ENEMY_ID, FORMATION_ENEMY_TILE


def _make_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="formation_train",
        log_enabled=False,
        persist_blue_hp=True,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_units(units) -> list[tuple[int, str]]:
    return [
        (int(unit["position"]), str(unit["name"]))
        for unit in units
        if str(unit.get("name", "")) != "пусто"
    ]


def test_formation_train_is_registered_as_a_clean_single_enemy_map():
    assert "formation_train" in available_maps()
    map_config = get_map("formation_train")

    assert map_config.grid_size == 48
    assert map_config.hero_start == (5, 27)
    assert map_config.default_objective == "all_enemies"
    assert map_config.enemy_positions() == {
        FORMATION_ENEMY_ID: FORMATION_ENEMY_TILE,
    }
    assert map_config.obstacle_blocks == ()
    assert map_config.village_heal_tiles == ()
    assert map_config.chests == ()
    assert map_config.mana_sources == ()
    assert map_config.merchant_sites == {}
    assert map_config.spell_shop_sites == {}
    assert map_config.mercenary_sites == {}
    assert map_config.trainer_sites == {}
    assert map_config.leadership_step_penalty == pytest.approx(0.06)


def test_formation_train_starts_with_requested_undead_formation():
    env = _make_env(
        Realcapital=1,
        typeoflord=1,
        use_boss_starting_roster=True,
    )
    env.reset(seed=123)

    assert env.Realcapital == 4
    assert env.typeoflord == 2
    assert env.grid_env.agent_pos == (5, 27)
    assert _living_units(env.blue_team_state) == [
        (7, "Воин"),
        (8, "Королева лич"),
        (11, "Оборотень"),
    ]

    units_by_position = {
        int(unit["position"]): unit
        for unit in env.blue_team_state
    }
    assert units_by_position[7]["stand"] == "ahead"
    assert units_by_position[8]["stand"] == "ahead"
    assert units_by_position[8]["hero"] is True
    assert units_by_position[11]["stand"] == "behind"
    assert env._available_leadership_points() == 1
    assert units_by_position[8]["needaunit"] == 1


def test_formation_train_enemy_uses_the_requested_front_and_back_rows():
    env = _make_env()
    env.reset(seed=123)

    assert dict(env.grid_env.enemy_positions) == {
        FORMATION_ENEMY_ID: FORMATION_ENEMY_TILE,
    }
    assert _living_units(env._enemy_configs[FORMATION_ENEMY_ID]) == [
        (1, "Ангел"),
        (2, "Мастер клинка"),
        (3, "Защитник Веры"),
        (5, "Ученик"),
    ]


def test_formation_train_uses_a_slightly_larger_leadership_penalty_per_step():
    env = _make_env(reward_needaunit_turn_penalty=0.01)
    env.reset(seed=123)

    assert env.reward_needaunit_turn_penalty == pytest.approx(0.06)
    move_action = next(
        action
        for action, allowed in enumerate(env.compute_action_mask()[:8])
        if bool(allowed)
    )
    _, reward, terminated, truncated, info = env.step(move_action)

    assert terminated is False
    assert truncated is False
    assert info["leadership_points"] == 1
    assert info["leadership_step_penalty"] == pytest.approx(0.06)
    assert reward == pytest.approx(info["grid_reward_scaled"])
