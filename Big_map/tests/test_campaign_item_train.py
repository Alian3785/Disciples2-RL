import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from battle_env import FEATURES_PER_UNIT, FIRST_HERO_ITEM_ACTION_START, HERO_ITEM_TARGET_SLOTS
from campaign_env import CampaignEnv
from maps import available_maps, get_map
from maps.item_train import (
    ANGEL_ORB_ITEM,
    ARMOR_STACK_ENEMY_ID,
    CENTAUR_STACK_ENEMY_ID,
    DWARF_STACK_ENEMY_ID,
    ELDER_VAMPIRE_TALISMAN_ITEM,
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
        map_name="item_train",
        log_enabled=False,
        persist_blue_hp=False,
        reward_needaunit_turn_penalty=0.0,
    )
    params.update(kwargs)
    return CampaignEnv(**params)


def _living_units(units) -> list[tuple[int, str]]:
    return [
        (int(unit["position"]), str(unit["name"]))
        for unit in units
        if str(unit.get("name", "")) != "пусто"
    ]


def _equip_action(env: CampaignEnv, item_name: str) -> int:
    return (
        env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START
        + env.BATTLE_EQUIPPABLE_ITEM_NAMES.index(item_name)
    )


def _finish_battle(env: CampaignEnv, enemy_id: int):
    env.mode = env.MODE_BATTLE
    env.current_enemy_id = enemy_id
    env.battle_env = _DummyBattleEnv(env.blue_team_state)
    return env.step(0)


def test_item_train_is_a_clean_small_map_with_two_separate_item_chests():
    assert "item_train" in available_maps()
    map_config = get_map("item_train")

    assert map_config.play_grid_size == 24
    assert map_config.default_objective == "all_enemies"
    assert map_config.obstacle_blocks == ()
    assert map_config.village_heal_tiles == ()
    assert map_config.mana_sources == ()
    assert map_config.merchant_sites == {}
    assert map_config.spell_shop_sites == {}
    assert map_config.mercenary_sites == {}
    assert map_config.trainer_sites == {}
    assert map_config.ruin_rewards == {}
    assert dict(map_config.chests) == {
        (18, 14): (ANGEL_ORB_ITEM,),
        (18, 34): (ELDER_VAMPIRE_TALISMAN_ITEM,),
    }

    env = _make_env(Realcapital=1, typeoflord=1)
    _, info = env.reset(seed=123)

    assert env.grid_size == 24
    assert env.grid_env.agent_pos == (2, 12)
    assert env.Realcapital == 2
    assert env.typeoflord == 2
    assert env.campaign_objective == "all_enemies"
    assert env.chests == {
        (9, 7): (ANGEL_ORB_ITEM,),
        (9, 17): (ELDER_VAMPIRE_TALISMAN_ITEM,),
    }
    assert dict(env.grid_env.enemy_positions) == {
        DWARF_STACK_ENEMY_ID: (20, 10),
        ARMOR_STACK_ENEMY_ID: (21, 12),
        CENTAUR_STACK_ENEMY_ID: (20, 14),
    }
    assert info["mana_income_per_turn"] == env._empty_mana_dict()


def test_item_train_starts_with_standard_legions_mage_party_and_item_lore():
    env = _make_env(use_boss_starting_roster=True)
    env.reset(seed=123)

    assert _living_units(env.blue_team_state) == [
        (7, "Одержимый"),
        (9, "Одержимый"),
        (10, "Архидьявол"),
        (12, "Сектант"),
    ]
    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    assert hero["name"] == "Архидьявол"
    assert env.HERO_SORCERY_LORE_ABILITY_KEY in hero["hero_abilities"]
    assert env._hero_has_sorcery_lore() is True
    assert env.scenario_battle_magic_item_names == (
        ANGEL_ORB_ITEM,
        ELDER_VAMPIRE_TALISMAN_ITEM,
    )


def test_item_train_enemy_compositions_match_the_training_scenario():
    env = _make_env()
    env.reset(seed=123)

    assert _living_units(env._enemy_configs[DWARF_STACK_ENEMY_ID]) == [
        (2, "Король гномов"),
        (5, "Арбалетчик"),
    ]
    assert _living_units(env._enemy_configs[ARMOR_STACK_ENEMY_ID]) == [
        (2, "Оживший доспех"),
    ]
    assert _living_units(env._enemy_configs[CENTAUR_STACK_ENEMY_ID]) == [
        (1, "Кентавр дикарь"),
        (3, "Кентавр дикарь"),
    ]


@pytest.mark.parametrize("item_name", [ANGEL_ORB_ITEM, ELDER_VAMPIRE_TALISMAN_ITEM])
def test_item_train_rewards_pickup_equip_and_use(item_name: str):
    env = _make_env(
        reward_turn_penalty=0.0,
        reward_repeat_position_penalty=0.0,
        reward_backtrack_penalty=0.0,
        reward_no_movement_penalty=0.0,
        battle_reward_scale=0.0,
    )
    env.reset(seed=123)

    chest_pos = next(pos for pos, items in env.chests.items() if item_name in items)
    env.grid_env.agent_pos = chest_pos
    _, pickup_reward, terminated, truncated, pickup_info = env.step(8)

    assert terminated is False
    assert truncated is False
    assert pickup_reward > 0.99
    assert pickup_info["item_pickup_reward"] == pytest.approx(1.0)
    assert pickup_info["rewarded_item_pickups"] == {item_name: 1}
    assert env._count_hero_item(item_name) == 1

    _, equip_reward, _, _, equip_info = env.step(_equip_action(env, item_name))

    assert equip_reward == pytest.approx(1.0)
    assert equip_info["equip_battle_item_applied"] is True
    assert equip_info["equip_battle_item_reward"] == pytest.approx(1.0)

    env.current_enemy_id = DWARF_STACK_ENEMY_ID
    env._init_battle(DWARF_STACK_ENEMY_ID)
    env.mode = env.MODE_BATTLE
    env.battle_env._advance_until_blue_turn = lambda: None
    hero = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and bool(unit.get("hero"))
    )
    hero["initiative"] = max(1, int(hero.get("initiative_base", 1) or 1))
    env.battle_env.current_blue_attacker_pos = int(hero["position"])
    env.battle_env.blue_attacks_left = 1
    use_action = FIRST_HERO_ITEM_ACTION_START + HERO_ITEM_TARGET_SLOTS.index(2)

    assert bool(env.battle_env.compute_action_mask()[use_action]) is True
    _, use_reward, _, _, use_info = env.step(use_action)

    assert use_reward == pytest.approx(2.0)
    assert use_info["battle_hero_item_name"] == item_name
    assert use_info["battle_hero_item_applied"] is True
    assert use_info["battle_item_use_reward"] == pytest.approx(2.0)


def test_item_train_never_adds_mana_at_turn_boundaries():
    env = _make_env(reward_turn_penalty=0.0)
    env.reset(seed=123)
    env.infernal_mana = 17.0
    env.life_mana = 9.0
    before = env._current_mana_totals()

    env._advance_turns(3)

    assert env._current_mana_totals() == before
    assert env.mana_income_per_turn == env._empty_mana_dict()


def test_item_train_rewards_each_stack_and_gives_largest_bonus_for_all_three():
    env = _make_env(
        reward_defeat_enemy=4.0,
        reward_all_enemies=10.0,
        reward_exp_weight=0.0,
        reward_survival_alive_weight=0.0,
        reward_survival_hp_weight=0.0,
        reward_unit_upgrade=0.0,
        battle_reward_scale=0.0,
    )
    env.reset(seed=123)

    for enemy_id in (DWARF_STACK_ENEMY_ID, ARMOR_STACK_ENEMY_ID):
        _, reward, terminated, truncated, info = _finish_battle(env, enemy_id)
        assert reward == pytest.approx(4.0)
        assert terminated is False
        assert truncated is False
        assert info["enemy_defeat_reward"] == pytest.approx(4.0)
        assert "all_enemies_reward" not in info

    _, reward, terminated, truncated, info = _finish_battle(
        env,
        CENTAUR_STACK_ENEMY_ID,
    )

    assert reward == pytest.approx(14.0)
    assert terminated is True
    assert truncated is False
    assert info["enemy_defeat_reward"] == pytest.approx(4.0)
    assert info["all_enemies_reward"] == pytest.approx(10.0)
    assert info["all_enemies_reward_granted"] is True
    assert info["campaign_result"] == "victory"
    assert info["campaign_victory_reason"] == "all_enemies_defeated"
    assert info["campaign_objective"] == "all_enemies"
