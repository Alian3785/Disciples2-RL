from types import SimpleNamespace

import pytest

from test_campaign_original_item_aliases import AliasEnv
from test_campaign_combat_potions import _blue_state_unit, _battle_blue_unit


def _env(armor=50):
    env = AliasEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2, use_boss_starting_roster=False)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    hero = _blue_state_unit(env, 8)
    hero.update(armor=armor, base_armor=armor)
    return env


def _drink(env, name):
    env._add_hero_item(name)
    action = env._potion_action_start_for_item(name) + env.GRID_POTION_USE_POSITIONS.index(8)
    assert env.compute_action_mask()[action]
    _, _, _, _, info = env.step(action)
    assert info['potion_applied'] and info['potion_consumed']
    assert env._count_hero_item(name) == 0


@pytest.mark.parametrize('name,bonus,permanent', [
    ('g000ig0002', 15, False), ('Treebark Potion', 30, False),
    ('g000ig0017', 50, False), ('Iron Skin Potion', 10, True),
])
def test_armor_potion_damage_same_day_battles_and_expiry(name, bonus, permanent):
    env = _env()
    _drink(env, name)
    for _ in range(2):
        env._init_battle(1)
        hero = _battle_blue_unit(env, 8)
        assert hero['armor'] == 50 + bonus
        assert hero.get('incoming_damage_multiplier', 1) == 1
        env.battle_env.rng = SimpleNamespace(randint=lambda *_: 0)
        assert env.battle_env._apply_damage_with_armor({}, 100, hero) == max(10, 50 - bonus)
        # Battle armor damage must not erase the original armor or permanent potion.
        hero['armor'] = 0
        env._save_blue_state()
        assert _blue_state_unit(env, 8)['armor'] == 50 + (bonus if permanent else 0)
    env._advance_turns(1)
    env._init_battle(1)
    assert _battle_blue_unit(env, 8)['armor'] == 50 + (bonus if permanent else 0)


def test_potion_armor_stacks_with_equipment_spell_and_permanent_bonus():
    env = _env(20)
    hero = _blue_state_unit(env, 8)
    hero['hero_abilities'] = [env.HERO_ARTIFACT_KNOWLEDGE_ABILITY_KEY, env.HERO_BANNER_BEARER_ABILITY_KEY]
    env._add_hero_item(env.RUNESTONE_ARTIFACT_ITEM_NAME)
    env._add_hero_item(env.BANNER_OF_PROTECTION_ITEM_NAME)
    assert hero['armor'] == 40
    _drink(env, 'Iron Skin Potion')
    _drink(env, 'Potion of Protection')
    _drink(env, 'Treebark Potion')
    env._apply_support_spell_to_blue_stack('elf_d2_s005')  # +10 armor
    for _ in range(2):
        env._init_battle(1)
        hero = _battle_blue_unit(env, 8)
        assert hero['armor'] == 105  # 20 + 10 permanent + 20 gear + 15 + 30 + 10 spell
        env.battle_env.rng = SimpleNamespace(randint=lambda *_: 0)
        assert env.battle_env._apply_damage_with_armor({}, 100, hero) == 10
        env._save_blue_state()
        assert _blue_state_unit(env, 8)['armor'] == 50
    env._advance_turns(1)
    env._init_battle(1)
    assert _battle_blue_unit(env, 8)['armor'] == 50
    env._save_blue_state()
    env.heroitems = []
    env._mark_equipment_dirty()
    env._refresh_campaign_equipment_effects()
    assert _blue_state_unit(env, 8)['armor'] == 30


def test_iron_skin_bonus_transfers_to_promoted_unit():
    env = _env()
    _drink(env, 'Iron Skin Potion')
    upgraded = {'armor': 15, 'base_armor': 15}
    env._reapply_persistent_elixir_bonuses_to_promoted_unit(_blue_state_unit(env, 8), upgraded)
    assert upgraded['armor'] == upgraded['base_armor'] == 25
    assert upgraded['campaign_iron_skin_potion_uses'] == 1


@pytest.mark.parametrize('legacy,canonical', [
    ('Зелье защиты (-15%)', 'Potion of Protection'),
    ('Зелье коры дерева (-30%)', 'Treebark Potion'),
    ('Зелье железной кожи (-10%)', 'Iron Skin Potion'),
])
def test_legacy_names_keep_the_same_potion_action(legacy, canonical):
    env = _env()
    assert env._canonical_potion_item_name(legacy) == env._canonical_potion_item_name(canonical)
    assert env._potion_action_start_for_item(legacy) == env._potion_action_start_for_item(canonical)
