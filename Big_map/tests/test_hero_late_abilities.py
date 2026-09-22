from copy import deepcopy
from types import SimpleNamespace

import pytest

from battle_env import BattleEnv, UNITS_BLUE
from campaign_env import CampaignEnv
from hero_level_abilities import apply_hero_level_stat_abilities
from test_battle_dynamic_levelups import _unit, _reach_level


@pytest.mark.parametrize('dynamic', [False, True])
def test_actual_levelups_grant_skills_at_13_14_15_once(dynamic):
    hero = deepcopy(next(u for u in UNITS_BLUE if u.get('hero')))
    hero.update(Level=12, boss=dynamic, initiative=50, initiative_base=50,
                accuracy=80, accuracy_secondary=50, armor=5, base_armor=5)
    battle = BattleEnv(log_enabled=False)
    _reach_level(battle, hero)
    assert hero['Level'] == 13
    assert hero['initiative_base'] == 75
    assert hero['accuracy'] == 80
    assert hero['armor'] == 5
    _reach_level(battle, hero)
    assert hero['Level'] == 14
    assert hero['accuracy'] == 96
    assert hero['accuracy_secondary'] == 60
    assert hero['armor'] == 5
    _reach_level(battle, hero)
    assert hero['Level'] == 15
    assert hero['armor'] == hero['base_armor'] == 25
    _reach_level(battle, hero)
    assert hero['initiative_base'] == 75
    assert hero['accuracy'] == 96
    assert hero['armor'] == 25


def test_high_level_regular_unit_does_not_receive_hero_skills():
    unit = _unit('Зелёный дракон')
    unit.update(Level=12, initiative=50, initiative_base=50, accuracy=80, armor=5)
    battle = BattleEnv(log_enabled=False)
    for _ in range(3):
        _reach_level(battle, unit)
    assert (unit['initiative_base'], unit['accuracy'], unit['armor']) == (50, 80, 5)
    assert not unit.get('hero_level_stat_abilities')


def test_starting_high_level_hero_receives_skills_once_and_displays_them():
    env = CampaignEnv(log_enabled=False, use_boss_starting_roster=False)
    env.reset(seed=123)
    hero = env._resolve_travel_hero()
    hero.update(Level=15, initiative=50, initiative_base=50, accuracy=90, armor=0, base_armor=0)
    for _ in range(3):
        env._sync_hero_progression_flags()
    assert (hero['initiative_base'], hero['accuracy'], hero['armor']) == (75, 100, 20)
    labels = env.get_travel_hero_visual_info()['abilities']
    for name in ('Первый удар', 'Точность', 'Природная броня'):
        assert sum(label.startswith(name) for label in labels) == 1


@pytest.mark.parametrize('level,potion,stat,expected', [
    (12, 'HASTE_ELIXIR_ITEM_NAME', 'initiative_base', 75),
    (13, 'HIT_POTION_ITEM_NAME', 'accuracy', 96),
    (14, 'BARK_POTION_ITEM_NAME', 'armor', 25),
])
def test_skill_learned_under_potion_survives_cleanup_and_next_battle(level, potion, stat, expected):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, use_boss_starting_roster=False)
    env.reset(seed=123)
    hero = env._resolve_travel_hero()
    hero['Level'] = level
    env._sync_hero_progression_flags()  # earlier skills have already been learned
    hero.update(initiative=50, initiative_base=50, accuracy=80, armor=5, base_armor=5)
    position = hero['position']
    env._active_potion_positions(getattr(env, potion)).add(position)
    env._init_battle(1)
    battle = env.battle_env
    hero = next(u for u in battle.combined if u.get('team') == 'blue' and u['position'] == position)
    battle._apply_exp_award_to_unit(hero, hero['exp_required'])
    env._save_blue_state()
    assert env._resolve_travel_hero()[stat] == expected
    env._advance_turns(1)
    env._init_battle(1)
    hero = next(u for u in env.battle_env.combined if u.get('team') == 'blue' and u['position'] == position)
    assert hero[stat] == expected


def test_natural_armor_stacks_and_obeys_existing_damage_cap():
    hero = {'Level': 15, 'armor': 75, 'base_armor': 75, 'initiative': 0, 'initiative_base': 50}
    apply_hero_level_stat_abilities(hero)
    assert hero['armor'] == 95
    assert hero['initiative'] == 0  # does not grant another action in a spent turn
    battle = BattleEnv(log_enabled=False)
    battle.rng = SimpleNamespace(randint=lambda *_: 0)
    assert battle._apply_damage_with_armor({}, 100, hero) == 10
