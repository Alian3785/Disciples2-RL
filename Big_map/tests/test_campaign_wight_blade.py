import pytest

from test_campaign_artifacts import _prepare_artifact_attack
from test_wight_level_down import _unit


@pytest.mark.parametrize('name', ['Wight Blade', 'Wight Blade (Artifact)', 'g000ig3018'])
def test_wight_blade_equips_and_drains_through_hero_attack(name):
    env, hero, target, chances = _prepare_artifact_attack(name)
    target.update(_unit('Ангел', 'red', target['position']))
    battle = env.battle_env
    assert env._hero_item_gold_value_by_name(name) == 5000
    assert hero['campaign_active_artifacts'] == [env.WIGHT_BLADE_ARTIFACT_ITEM_NAME]
    assert battle._attack(hero, target['position']) == (True, 'ok')
    assert chances == [75]
    assert target['wight_form_name'] == 'Имперский рыцарь'
    assert target['transform_effect'] == 'wight_level_down'
    assert env._count_hero_item(name) == 1
    assert battle._cleanse_negative_effects(target)
    assert target.get('wight_form_name') is None


@pytest.mark.parametrize('blocked', ['miss', 'failed_roll', 'immunity', 'ward', 'neutral', 'unequipped', 'dead'])
def test_wight_blade_obeys_hit_equipment_and_target_protection(blocked):
    env, hero, target, chances = _prepare_artifact_attack('Wight Blade')
    target.update(_unit('Орк' if blocked == 'neutral' else 'Ангел', 'red', target['position']))
    battle = env.battle_env
    if blocked == 'miss':
        battle._roll_hit = lambda _: False
    elif blocked == 'failed_roll':
        battle._roll_status = lambda _: False
    elif blocked == 'immunity':
        target['immunity'] = ['Death']
    elif blocked == 'ward':
        target['resistance'] = ['Death']
    elif blocked == 'unequipped':
        hero['campaign_active_artifacts'] = []
    elif blocked == 'dead':
        target['health'] = 1
    battle._attack(hero, target['position'])
    assert target.get('wight_form_name') is None
    if blocked == 'ward':
        assert 'Death' in target['resilience_used_types']
        battle._attack(hero, target['position'])
        assert target['wight_form_name'] == 'Имперский рыцарь'
