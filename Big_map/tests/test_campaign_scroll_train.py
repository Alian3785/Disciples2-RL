import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import FEATURES_PER_UNIT
from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.scroll_train import (
    GREEN_DRAGON_ENEMY_ID,
    OCCULT_MASTER_ENEMY_ID,
    STARTING_SCROLL_ITEMS,
    UNDEAD_STACK_ENEMY_ID,
)


class _DummyBattleEnv:
    def __init__(self, blue_team):
        self.winner = "blue"
        self.action_space = type("ActionSpace", (), {"n": 15})()
        self.last_battle_exp = 0.0
        self.last_levelups = []
        self.combined = deepcopy(blue_team)

    def _obs(self):
        return np.zeros(FEATURES_PER_UNIT * 12, dtype=np.float32)


def _make_env(**kwargs) -> CampaignEnv:
    params = dict(
        map_name="scroll_train",
        log_enabled=False,
        persist_blue_hp=False,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_units(units) -> list[tuple[int, str]]:
    return [
        (int(unit["position"]), str(unit["name"]))
        for unit in units
        if str(unit.get("name", "")) != "пусто"
    ]


def _finish_battle(env: CampaignEnv, enemy_id: int):
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = enemy_id
    env.battle_env = _DummyBattleEnv(env.blue_team_state)
    return env.step(0)


def test_scroll_train_is_registered_with_exactly_99_unique_supported_scrolls():
    assert "scroll_train" in available_maps()
    map_config = get_map("scroll_train")

    assert map_config.play_grid_size == 24
    assert map_config.default_objective == "all_enemies"
    assert len(STARTING_SCROLL_ITEMS) == 99
    assert len(set(STARTING_SCROLL_ITEMS)) == 99
    assert map_config.starting_scroll_items == STARTING_SCROLL_ITEMS
    assert map_config.starting_scroll_magic_unlocked is False
    assert map_config.passive_mana_income_enabled is False
    assert map_config.turn_penalty == pytest.approx(0.03)
    assert map_config.chests == ()
    assert map_config.mana_sources == ()


def test_scroll_train_starts_with_all_scroll_actions_and_legions_mage_party():
    env = _make_env(Realcapital=1, typeoflord=1, reward_turn_penalty=0.001)
    _, info = env.reset(seed=123)

    assert env.Realcapital == 2
    assert env.typeoflord == 2
    assert env.grid_size == 24
    assert env.grid_env.agent_pos == (2, 12)
    assert _living_units(env.blue_team_state) == [
        (7, "Одержимый"),
        (9, "Одержимый"),
        (10, "Архидьявол"),
        (12, "Сектант"),
    ]
    assert env.scroll_magic_unlocked is True
    assert env.reward_turn_penalty == pytest.approx(0.03)
    assert env.GRID_SCROLL_CAST_ACTION_COUNT == 99
    assert len(env.scenario_scroll_spell_entries) == 99
    assert len(env.scroll_cast_slot_entries()) == 99
    assert all(int(entry["copies"]) == 1 for entry in env.scroll_cast_slot_entries())
    assert info["supported_scroll_types_count"] == 99
    assert info["total_scroll_count"] == 99


def test_scroll_train_enemy_compositions_match_the_requested_three_stacks():
    env = _make_env()
    env.reset(seed=123)

    assert dict(env.grid_env.enemy_positions) == {
        OCCULT_MASTER_ENEMY_ID: (20, 8),
        GREEN_DRAGON_ENEMY_ID: (21, 12),
        UNDEAD_STACK_ENEMY_ID: (20, 16),
    }
    assert _living_units(env._enemy_configs[OCCULT_MASTER_ENEMY_ID]) == [
        (2, "Мастер оккультист"),
    ]
    assert _living_units(env._enemy_configs[GREEN_DRAGON_ENEMY_ID]) == [
        (2, "Зелёный дракон"),
    ]
    assert _living_units(env._enemy_configs[UNDEAD_STACK_ENEMY_ID]) == [
        (1, "Скелет призрак"),
        (3, "Скелет призрак"),
        (4, "Архилич"),
        (6, "Сущий"),
    ]


def test_scroll_train_can_consume_a_supported_scroll_without_spending_mana():
    env = _make_env(
        reward_needaunit_turn_penalty=0.0,
        reward_spell_cast=0.0,
    )
    env.reset(seed=123)
    damage_slot = next(
        idx
        for idx, entry in enumerate(env.scroll_cast_slot_entries())
        if str(entry.get("spell_kind", "")) == "damage"
    )
    slot_entry = env.scroll_cast_slot_entries()[damage_slot]
    item_name = str(slot_entry["item_name"])
    action = env.grid_scroll_cast_action_start + damage_slot
    mana_before = env._current_mana_totals()

    assert env.count_scroll_item(item_name) == 1
    assert bool(env.compute_action_mask()[action]) is True
    _, _, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["spell_cast_executed"] is True
    assert info["scroll_item_consumed"] is True
    assert env.count_scroll_item(item_name) == 0
    assert env._scroll_state_info()["total_scroll_count"] == 98
    assert env._current_mana_totals() == mana_before


def test_scroll_train_mana_never_grows_and_rest_uses_larger_turn_penalty():
    env = _make_env(reward_turn_penalty=0.001)
    env.reset(seed=123)
    env.infernal_mana = 12.0
    env.life_mana = 7.0
    before = env._current_mana_totals()

    reward = env._advance_turns(2)

    assert reward == pytest.approx(-0.06)
    assert env._current_mana_totals() == before
    assert env.mana_income_per_turn == env._empty_mana_dict()


def test_scroll_train_rewards_each_enemy_and_large_final_clear_bonus():
    env = _make_env(
        reward_defeat_enemy=4.0,
        reward_all_enemies=12.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
        reward_needaunit_turn_penalty=0.0,
        battle_reward_scale=0.0,
    )
    env.reset(seed=123)

    for enemy_id in (OCCULT_MASTER_ENEMY_ID, GREEN_DRAGON_ENEMY_ID):
        _, reward, terminated, truncated, info = _finish_battle(env, enemy_id)
        assert reward == pytest.approx(4.0)
        assert terminated is False
        assert truncated is False
        assert info["enemy_defeat_reward"] == pytest.approx(4.0)
        assert "all_enemies_reward" not in info

    _, reward, terminated, truncated, info = _finish_battle(
        env,
        UNDEAD_STACK_ENEMY_ID,
    )

    assert reward == pytest.approx(16.0)
    assert terminated is True
    assert truncated is False
    assert info["enemy_defeat_reward"] == pytest.approx(4.0)
    assert info["all_enemies_reward"] == pytest.approx(12.0)
    assert info["campaign_result"] == "victory"
    assert info["campaign_victory_reason"] == "all_enemies_defeated"
