import sys
from copy import deepcopy
from pathlib import Path

import pytest
from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS


@pytest.mark.parametrize(
    ("capital", "hero_name"),
    [
        (1, "Архимаг"),
        (2, "Архидьявол"),
        (3, "Хранитель знаний"),
        (4, "Королева лич"),
        (5, "Дриада"),
    ],
)
def test_every_faction_mage_hero_starts_with_scroll_and_staff_magic_unlocked(
    capital: int,
    hero_name: str,
):
    env = CampaignEnv(
        Realcapital=capital,
        typeoflord=2,
        use_boss_starting_roster=False,
        log_enabled=False,
    )
    env.reset(seed=123)

    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    assert hero["name"] == hero_name
    assert hero["unit_type"] == "Mage"
    assert env._travel_hero_is_mage() is True
    assert env.scroll_magic_unlocked is True
    assert env.gold == pytest.approx(0.0)
    assert bool(env.compute_action_mask()[env.GRID_UNLOCK_SCROLL_MAGIC_ACTION]) is False


def test_nonmage_hero_still_uses_the_paid_scroll_magic_unlock():
    env = CampaignEnv(
        Realcapital=1,
        typeoflord=1,
        use_boss_starting_roster=False,
        log_enabled=False,
    )
    env.reset(seed=123)

    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    assert hero["unit_type"] != "Mage"
    assert env._travel_hero_is_mage() is False
    assert env.scroll_magic_unlocked is False


class _MageStaffScenarioEnv(CampaignEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_chests = dict(self._static_chests)
        self._static_chests[(1, 1)] = ("Staff of Thunder",)
        self.chests = dict(self._static_chests)
        self._refresh_dynamic_action_layout()
        self.action_space = spaces.Discrete(
            self.GRID_SWAP_UNIT_ACTION_START + self.GRID_SWAP_UNIT_ACTION_COUNT
        )


def test_level_one_mage_can_use_staff_without_sorcery_lore_or_paid_unlock():
    env = _MageStaffScenarioEnv(
        Realcapital=1,
        typeoflord=2,
        use_boss_starting_roster=False,
        log_enabled=False,
        reward_needaunit_turn_penalty=0.0,
    )
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {10: (4, 3)}
    env.grid_env.enemies_alive = {10: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[10] = deepcopy(ENEMY_CONFIGS[20])

    hero = env._resolve_travel_hero(units=env.blue_team_state)
    assert hero is not None
    assert hero["Level"] == 1
    # Проверяем именно тип реального героя, а не сохранённый lord-type флаг.
    env.typeoflord = 1
    assert env._travel_hero_is_mage() is True
    assert env._hero_has_sorcery_lore() is False
    assert env.scroll_magic_unlocked is True

    item_name = "Staff of Thunder"
    spell_key = "emp_d2_s004"
    env._append_hero_item(item_name)
    spell = env._spell_entry_by_id(spell_key)
    assert isinstance(spell, dict)
    for mana_kind, amount in env._get_spell_use_costs(spell).items():
        setattr(env, env.MANA_ATTR_BY_KIND[mana_kind], float(amount))

    action = next(
        env.grid_staff_spell_action_start + idx
        for idx, entry in enumerate(env.staff_spell_action_entries())
        if str(entry.get("item_name", "")) == item_name
    )
    assert bool(env.compute_action_mask()[action]) is True

    _, _, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["spell_cast_executed"] is True
    assert info["staff_magic_unlocked"] is True
    assert info["staff_hero_is_mage"] is True
    assert info["staff_requires_sorcery_lore"] is False
    assert info["staff_has_sorcery_lore"] is False
    assert env.gold == pytest.approx(0.0)
