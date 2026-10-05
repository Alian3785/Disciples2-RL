"""Growth keeps active combat layers, real fighter identity and spent turns."""
from copy import deepcopy
import json

import pytest

from battle_env import BattleEnv, _apply_hero_levelup_bonuses
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from permanent_unit_stats import persistent_values, rebuild_stat_layers

BY_NAME = {u['кто']: u for u in DATA}
STATS = ('damage', 'damage_secondary', 'accuracy', 'accuracy_secondary',
         'armor', 'initiative_base', 'max_health', 'health')


def unit(name='Герцог', team='blue'):
    u = map_unit_to_battle(BY_NAME[name], team, 7 if team == 'blue' else 1)
    u.update(original_damage=u['damage'], base_armor=u['armor'])
    return u


def apply(battle, target, effect):
    if effect == 'buff':
        battle._apply_hero_item_damage_buff(target_unit=target,
                                           effect={'damage_multiplier': 1.75})
    elif effect == 'debuff':
        battle._apply_hero_item_damage_debuff(source_unit=None, target_unit=target,
                                             effect={'damage_multiplier': 0.5})
    elif effect == 'slow':
        battle._apply_hermit_initiative_slow({'attack_type_secondary': ''}, target)
    elif effect == 'shatter':
        battle._apply_teurg_armor_shred({'attack_type_secondary': ''}, target)


@pytest.mark.parametrize('team', ['blue', 'red'])
@pytest.mark.parametrize('level', [5, 13, 15])
@pytest.mark.parametrize('effects', [('buff',), ('debuff',), ('slow',), ('shatter',),
                                    ('buff', 'debuff'), ('debuff', 'buff'),
                                    ('buff', 'debuff', 'slow', 'shatter')])
def test_growth_replays_current_combat_effects_and_preserves_spent_turn(team, level, effects):
    battle = BattleEnv(log_enabled=False)
    subject = unit(team=team)
    subject.update(Level=level, armor=31, base_armor=31, initiative=0,
                   health=37, hp=37)
    control = deepcopy(subject)
    _apply_hero_levelup_bonuses(control)
    for effect in effects:
        apply(battle, control, effect)
        apply(battle, subject, effect)
    _apply_hero_levelup_bonuses(subject)
    assert {key: subject[key] for key in STATS} == {key: control[key] for key in STATS}
    assert subject['initiative'] == control['initiative'] == 0
    assert persistent_values(subject) == persistent_values(control)
    battle._cleanse_negative_effects(subject)
    battle._cleanse_negative_effects(control)
    battle._reset_powerup(subject)
    battle._reset_powerup(control)
    assert {key: subject[key] for key in STATS} == {key: control[key] for key in STATS}
    assert subject['initiative'] == control['initiative'] == 0


@pytest.mark.parametrize('team', ['blue', 'red'])
@pytest.mark.parametrize('escape', [False, True])
def test_transformed_xp_grows_original_fighter_then_survives_serialization(team, escape):
    battle = BattleEnv(log_enabled=False)
    hero = unit(team=team)
    hero.update(Level=4, health=37, hp=37, exp_current=hero['exp_required'] - 1)
    control = deepcopy(hero)
    battle._apply_exp_award_to_unit(control, 1)
    battle.combined = [hero]
    assert battle._apply_hero_item_lycanthropy(
        source_unit=None, target_unit=hero,
        effect={'damage_type': '', 'transform_unit_name': 'Оборотень'},
    )
    if escape:
        battle._mark_unit_escaped(hero)
        hero = battle.escaped_units[0]
    battle._apply_exp_award_to_unit(hero, 1)
    assert hero['name'] == control['name'] == 'Герцог'
    assert hero['Level'] == 5
    assert hero['campaign_stat_sources'] == control['campaign_stat_sources']
    for key in STATS:
        if key != 'health':
            assert hero[key] == control[key]
    assert 0 < hero['health'] <= hero['max_health']
    restored = json.loads(json.dumps(hero))
    before = deepcopy(restored)
    rebuild_stat_layers(restored)
    assert {k: restored[k] for k in STATS} == {k: before[k] for k in STATS}


@pytest.mark.parametrize('team', ['blue', 'red'])
@pytest.mark.parametrize('elixir', [False, True])
def test_doppelganger_copy_retains_own_growth_over_repeated_fights(team, elixir):
    from permanent_unit_stats import add_permanent_effect
    battle = BattleEnv(log_enabled=False)
    copier = unit('Двойник', team)
    copier['source_unit_id'] = 'own-copier'
    if elixir:
        add_permanent_effect(copier, {'kind': 'damage', 'multiplier': 1.2})
    target = unit('Теург', 'red' if team == 'blue' else 'blue')
    for _ in range(3):
        control = deepcopy(copier)
        battle._apply_exp_award_to_unit(control, control['exp_required'])
        assert battle._apply_doppelganger_copy(copier, target)
        battle._apply_exp_award_to_unit(copier, copier['exp_required'])
        assert copier['unit_type'] == 'Doppelganger'
        assert copier['Level'] == control['Level']
        assert persistent_values(copier) == persistent_values(control)
        assert copier['source_unit_id'] == 'own-copier'
        assert copier['health'] > 0


@pytest.mark.parametrize('armor,bonus', [(0, 20), (83, 30), (105, 30)])
def test_garrison_bonus_is_capped_layer_not_growth_source(armor, bonus):
    subject = unit()
    subject.update(Level=15, armor=armor, base_armor=armor)
    control = deepcopy(subject)
    _apply_hero_levelup_bonuses(control)
    subject.update(garrison_base_armor=armor, garrison_armor_bonus=bonus,
                   armor=min(90, armor + bonus), base_armor=min(90, armor + bonus))
    _apply_hero_levelup_bonuses(subject)
    assert persistent_values(subject) == persistent_values(control)
    assert subject['armor'] == min(90, control['armor'] + bonus)
    assert subject['garrison_base_armor'] == control['armor']


def test_unknown_noop_does_not_create_or_rewrite_sources():
    from unit_dynamic_stats import apply_dynamic_stat_growth
    unknown = unit()
    unknown['unit_id'] = 'not-a-profile'
    unknown['damage'] = 999
    before = deepcopy(unknown)
    assert not apply_dynamic_stat_growth(unknown, 3)
    assert unknown == before


@pytest.mark.parametrize('effects', [('buff', 'debuff'), ('debuff', 'buff')])
@pytest.mark.parametrize('cleanup', ['expire_buff', 'cleanse', 'both'])
@pytest.mark.parametrize('team', ['blue', 'red'])
def test_growth_after_partial_effect_expiry_does_not_reactivate_old_layers(effects, cleanup, team):
    battle = BattleEnv(log_enabled=False)
    subject = unit(team=team)
    subject.update(Level=5, initiative=0)
    control = deepcopy(subject)
    _apply_hero_levelup_bonuses(control)
    for target in (subject, control):
        for effect in effects:
            apply(battle, target, effect)
        if cleanup in {'expire_buff', 'both'}:
            battle._reset_powerup(target)
        if cleanup in {'cleanse', 'both'}:
            battle._cleanse_negative_effects(target)
    _apply_hero_levelup_bonuses(subject)
    assert subject['damage'] == control['damage']
    assert subject['initiative'] == control['initiative'] == 0
    assert persistent_values(subject) == persistent_values(control)


@pytest.mark.parametrize('effect', ['buff', 'debuff'])
def test_effect_only_on_temporary_form_is_not_replayed_on_original_growth(effect):
    battle = BattleEnv(log_enabled=False)
    hero = unit()
    hero.update(Level=4, exp_current=hero['exp_required'] - 1)
    control = deepcopy(hero)
    battle._apply_exp_award_to_unit(control, 1)
    assert battle._apply_hero_item_lycanthropy(
        source_unit=None, target_unit=hero,
        effect={'damage_type': '', 'transform_unit_name': 'Оборотень'},
    )
    apply(battle, hero, effect)
    battle._apply_exp_award_to_unit(hero, 1)
    assert hero['name'] == control['name']
    assert hero['damage'] == control['damage']
    assert persistent_values(hero) == persistent_values(control)


def test_dead_scripted_bot_keeps_pre_shatter_stats_without_revival():
    env = CampaignEnv(map_name='trade_train', observation_version='local5',
                      scripted_capital_bot_enabled=False,
                      use_boss_starting_roster=False, log_enabled=False)
    try:
        env.reset(seed=42)
        fighter = unit()
        fighter.update(armor=35, base_armor=35)
        battle = BattleEnv(log_enabled=False)
        apply(battle, fighter, 'shatter')
        fighter['health'] = fighter['hp'] = 0
        saved = env._normalize_scripted_bot_saved_unit(fighter)
        assert saved['armor'] == saved['base_armor'] == 35
        assert saved['health'] == saved['hp'] == saved['initiative'] == 0
        assert 'shatter_original_armor' not in saved
    finally:
        env.close()
