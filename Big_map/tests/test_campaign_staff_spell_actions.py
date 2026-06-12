from copy import deepcopy
from pathlib import Path
import sys

import pytest
from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS


def _grant_sorcery_lore(env: CampaignEnv) -> None:
    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    hero["hero_abilities"] = [env.HERO_SORCERY_LORE_ABILITY_KEY]
    assert env._hero_has_sorcery_lore()


def _set_spell_use_mana(
    env: CampaignEnv,
    spell_key: str,
    *,
    multiplier: float = 1.0,
) -> dict[str, float]:
    for mana_kind in env.MANA_KIND_ORDER:
        setattr(env, env.MANA_ATTR_BY_KIND[mana_kind], 0.0)

    spell = env._spell_entry_by_id(spell_key)
    assert isinstance(spell, dict)
    costs = env._get_spell_use_costs(spell)
    for mana_kind, amount in costs.items():
        setattr(
            env,
            env.MANA_ATTR_BY_KIND[mana_kind],
            float(amount) * float(multiplier),
        )
    return costs


def _staff_action_index(env: CampaignEnv, item_name: str) -> int:
    for idx, entry in enumerate(env.staff_spell_action_entries()):
        if str(entry.get("item_name", "") or "") == str(item_name):
            return env.grid_staff_spell_action_start + idx
    raise AssertionError(f"staff action not found: {item_name}")


def _hire_scroll_staff_mage(env: CampaignEnv) -> dict:
    env.gold = float(env.SCROLL_MAGE_UNLOCK_GOLD_COST)
    unlock_action = env.GRID_UNLOCK_SCROLL_MAGIC_ACTION
    assert bool(env.compute_action_mask()[unlock_action]) is True
    _, reward, terminated, truncated, info = env.step(unlock_action)
    assert reward == pytest.approx(0.0)
    assert terminated is False
    assert truncated is False
    assert info["scroll_magic_unlock_action"] is True
    assert info["scroll_magic_unlocked_after"] is True
    assert env.scroll_magic_unlocked is True
    return info


class StaffScenarioEnv(CampaignEnv):
    MERCHANT_BUY_ITEMS = CampaignEnv.MERCHANT_BUY_ITEMS + (
        {"name": "Staff of Holiness", "price": 1200.0, "stock": 1, "grant": "inventory"},
    )
    RUIN_REWARD_BY_ENEMY_ID = dict(CampaignEnv.RUIN_REWARD_BY_ENEMY_ID)
    RUIN_REWARD_BY_ENEMY_ID[999] = {
        "title": "Staff test ruin",
        "ruin_pos": (8, 8),
        "marker_pos": (8, 9),
        "item": "Spirit Staff",
        "gold": 0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_chests = dict(self._static_chests)
        self._static_chests[(1, 1)] = ("Staff of Thunder", "Staff of Traveling")
        self.chests = dict(self._static_chests)
        self._refresh_dynamic_action_layout()
        self.action_space = spaces.Discrete(
            self.GRID_SWAP_UNIT_ACTION_START + self.GRID_SWAP_UNIT_ACTION_COUNT
        )


def _make_staff_env() -> StaffScenarioEnv:
    env = StaffScenarioEnv(log_enabled=False, persist_blue_hp=True, Realcapital=1)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {10: (4, 3)}
    env.grid_env.enemies_alive = {10: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[10] = deepcopy(ENEMY_CONFIGS[20])
    return env


def test_staff_spell_slots_are_reserved_from_chests_ruins_and_merchants():
    env = _make_staff_env()

    assert env.scenario_staff_spell_item_names == (
        "Staff of Thunder",
        "Staff of Traveling",
        "Spirit Staff",
        "Staff of Holiness",
    )
    assert [entry["spell_id"] for entry in env.staff_spell_action_entries()] == [
        "emp_d2_s004",
        "emp_d2_s006",
        "emp_d2_s011",
        "emp_d2_s007",
    ]
    assert env.GRID_UNLOCK_SCROLL_MAGIC_ACTION == env.grid_staff_spell_action_start + 4
    assert env.action_space.n == (
        env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_ACTION_COUNT
    )


def test_duplicate_staffs_reserve_duplicate_slots_and_require_matching_count():
    class DuplicateStaffEnv(CampaignEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._static_chests = dict(self._static_chests)
            self._static_chests[(1, 1)] = ("Staff of Thunder", "Staff of Thunder")
            self.chests = dict(self._static_chests)
            self._refresh_dynamic_action_layout()
            self.action_space = spaces.Discrete(
                self.GRID_SWAP_UNIT_ACTION_START + self.GRID_SWAP_UNIT_ACTION_COUNT
            )

    env = DuplicateStaffEnv(log_enabled=False, persist_blue_hp=True, Realcapital=1)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {10: (4, 3)}
    env.grid_env.enemies_alive = {10: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[10] = deepcopy(ENEMY_CONFIGS[20])
    env.scroll_magic_unlocked = True
    _grant_sorcery_lore(env)
    _set_spell_use_mana(env, "emp_d2_s004", multiplier=2.0)

    entries = env.staff_spell_action_entries()
    assert [entry["item_name"] for entry in entries] == [
        "Staff of Thunder",
        "Staff of Thunder",
    ]
    assert [entry["required_item_count"] for entry in entries] == [1, 2]
    assert env.GRID_UNLOCK_SCROLL_MAGIC_ACTION == env.grid_staff_spell_action_start + 2

    first_action = env.grid_staff_spell_action_start
    second_action = env.grid_staff_spell_action_start + 1
    env._append_hero_item("Staff of Thunder")
    mask = env.compute_action_mask()
    assert bool(mask[first_action]) is True
    assert bool(mask[second_action]) is False

    env._append_hero_item("Staff of Thunder")
    mask = env.compute_action_mask()
    assert bool(mask[first_action]) is True
    assert bool(mask[second_action]) is True


def test_staff_spell_action_requires_item_sorcery_lore_and_mana_without_learning():
    env = _make_staff_env()
    item_name = "Staff of Thunder"
    spell_key = "emp_d2_s004"
    action = _staff_action_index(env, item_name)

    assert bool(env.compute_action_mask()[action]) is False

    env._append_hero_item(item_name)
    assert env._is_sell_only_hero_item(env.heroitems[-1]) is False
    assert bool(env.compute_action_mask()[action]) is False
    assert bool(env.compute_action_mask()[env.GRID_UNLOCK_SCROLL_MAGIC_ACTION]) is False

    unlock_info = _hire_scroll_staff_mage(env)
    assert unlock_info["scroll_magic_unlock_scroll_slots"] == 0
    assert unlock_info["scroll_magic_unlock_staff_slots"] == 1
    assert bool(env.compute_action_mask()[action]) is False

    _grant_sorcery_lore(env)
    assert bool(env.compute_action_mask()[action]) is False

    costs = _set_spell_use_mana(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is True
    assert env._is_spell_learned(spell_key) is False

    env.enemy_team_states[10] = [
        {
            "name": "Target",
            "team": "red",
            "position": 1,
            "stand": "ahead",
            "health": 100.0,
            "hp": 100.0,
            "max_health": 100.0,
            "maxhp": 100.0,
            "unit_type": "Warrior",
        }
    ]

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["staff_spell_cast_action"] is True
    assert info["staff_item_name"] == item_name
    assert info["staff_spell_key"] == spell_key
    assert info["staff_magic_unlocked"] is True
    assert info["spell_cast_executed"] is True
    assert info["spell_is_learned"] is False
    assert info["spell_cast_used_this_turn"] is False
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(85.0)
    assert env.life_mana == pytest.approx(0.0)
    assert costs["life"] == pytest.approx(100.0)
    assert env._spell_cast_count_this_turn(spell_key) == 0


def test_staff_spell_can_be_cast_repeatedly_in_same_turn_while_mana_remains():
    env = _make_staff_env()
    item_name = "Staff of Thunder"
    spell_key = "emp_d2_s004"
    action = _staff_action_index(env, item_name)
    env._append_hero_item(item_name)
    _hire_scroll_staff_mage(env)
    _grant_sorcery_lore(env)
    costs = _set_spell_use_mana(env, spell_key, multiplier=2.0)
    env.enemy_team_states[10] = [
        {
            "name": "Durable target",
            "team": "red",
            "position": 1,
            "stand": "ahead",
            "health": 100.0,
            "hp": 100.0,
            "max_health": 100.0,
            "maxhp": 100.0,
            "unit_type": "Warrior",
        }
    ]

    _, first_reward, _, _, first_info = env.step(action)
    assert first_reward == pytest.approx(env.reward_spell_cast)
    assert first_info["spell_cast_executed"] is True
    assert env.life_mana == pytest.approx(float(costs["life"]))
    assert env._spell_cast_count_this_turn(spell_key) == 0
    assert bool(env.compute_action_mask()[action]) is True

    _, second_reward, _, _, second_info = env.step(action)
    assert second_reward == pytest.approx(env.reward_spell_cast)
    assert second_info["spell_cast_executed"] is True
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(70.0)
    assert env.life_mana == pytest.approx(0.0)
    assert env._spell_cast_count_this_turn(spell_key) == 0
    assert bool(env.compute_action_mask()[action]) is False


def test_staff_support_spell_uses_same_effect_and_does_not_spend_turn_counter():
    env = _make_staff_env()
    item_name = "Staff of Traveling"
    spell_key = "emp_d2_s006"
    action = _staff_action_index(env, item_name)
    env._append_hero_item(item_name)
    _hire_scroll_staff_mage(env)
    _grant_sorcery_lore(env)
    _set_spell_use_mana(env, spell_key, multiplier=2.0)
    env.moves = 0

    _, first_reward, _, _, first_info = env.step(action)
    assert first_reward == pytest.approx(env.reward_spell_cast)
    assert first_info["spell_cast_executed"] is True
    assert first_info["spell_moves_restored"] == pytest.approx(10.0)
    assert env.moves == 10
    assert env._spell_cast_count_this_turn(spell_key) == 0
    assert bool(env.compute_action_mask()[action]) is True

    _, second_reward, _, _, second_info = env.step(action)
    assert second_reward == pytest.approx(env.reward_spell_cast)
    assert second_info["spell_cast_executed"] is True
    assert second_info["spell_moves_restored"] == pytest.approx(10.0)
    assert env.moves == 20
    assert env._spell_cast_count_this_turn(spell_key) == 0
