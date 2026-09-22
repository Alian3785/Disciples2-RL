import pytest

from battle_env import RUN_AWAY_ACTION_INDEX
from test_campaign_artifacts import CampaignEnv, _set_hero_level, _battle_blue_hero
from test_campaign_mercenary_sites import _clear_slot


def _env(item=None):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _set_hero_level(env, 9)
    if item:
        env._add_hero_item(item)
    return env


@pytest.mark.parametrize('name', ['Rusted Shackles', 'g000ig2007', 'Ржавые кандалы',
                                  'Lute of Charming', 'G000IG3022', 'Лютня очарования'])
def test_original_items_are_equippable_and_keep_game_value(name):
    env = _env(name)
    assert env._canonical_item_name(name) in env.equipped_artifact_items
    assert env._hero_item_gold_value_by_name(name) == 1500
    assert env._count_hero_item(name) == 1


def test_lute_requires_equipment_and_discount_does_not_stack():
    env = _env()
    _set_hero_level(env, 8)
    env._add_hero_item('Lute of Charming')
    assert env._shop_buy_price(100) == 100
    _set_hero_level(env, 9)
    assert env._shop_buy_price(100) == 90
    env._add_hero_item('Lute of Charming')
    assert env._shop_buy_price(100) == 90
    env._add_hero_item(env.BETHREZENS_CLAW_ITEM_NAME)
    env._add_hero_item(env.RING_OF_AGES_ITEM_NAME)
    assert env._shop_buy_price(100) == 100


@pytest.mark.parametrize('kind', ['item', 'spell'])
def test_lute_discount_matches_purchase_mask_and_actual_gold(kind):
    env = _env('Lute of Charming')
    if kind == 'item':
        site, stock = next((s, st) for s, st in env.merchant_stocks.items() if any(st.values()))
        name = next(n for n, qty in stock.items() if qty > 0)
        env.grid_env.agent_pos = env.merchant_site_interaction_tiles[site][0]
        price = env._merchant_item_definition(name)['price']
        action = env.GRID_MERCHANT_POTION_BUY_ACTION_START + env.scenario_merchant_item_names.index(name)
        prefix = 'merchant_buy'
    else:
        site, stock = next((s, st) for s, st in env.spell_shop_stocks.items() if any(st.values()))
        idx, spell = next((i, s) for i, s in enumerate(env.SPELL_SHOP_BUY_SPELLS)
                          if stock.get(s['name'], 0) and not env._spell_shop_spell_owned(s['spell_id']))
        env.grid_env.agent_pos = env.spell_shop_site_interaction_tiles[site][0]
        price = spell['price']
        action = env.GRID_SPELL_SHOP_BUY_ACTION_START + idx
        prefix = 'spell_shop_buy'
    discounted = int(price * 90 / 100)
    env.gold = discounted - 1
    assert not env.compute_action_mask()[action]
    env.gold = discounted
    assert env.compute_action_mask()[action]
    _, _, _, _, info = env.step(action)
    assert info[prefix + '_purchased']
    assert info[prefix + '_price'] == discounted
    assert env.gold == 0


def test_lute_discount_applies_to_mercenary_mask_observation_and_hire():
    env = _env('Lute of Charming')
    env.grid_env.agent_pos = (12, 43)
    _clear_slot(env, 7)
    env.gold = 764
    action = env.GRID_MERCENARY_HIRE_ACTION_START
    assert not env.compute_action_mask()[action]
    env.gold = 765
    assert env.compute_action_mask()[action]
    stock = env._mercenary_context_info()['mercenary_stock']
    assert next(iter(stock.values()))[0]['gold'] == 765
    _, _, _, _, info = env.step(action)
    assert info['hired']
    assert env.gold == 0


@pytest.mark.parametrize('equipped,hero_turn', [(True, True), (False, True), (True, False)])
def test_shackles_skip_retreat_delay_only_for_equipped_hero(equipped, hero_turn):
    env = _env('Rusted Shackles' if equipped else None)
    env._init_battle(enemy_id=1)
    battle = env.battle_env
    hero = _battle_blue_hero(env)
    runner = hero if hero_turn else next(u for u in battle.combined if u['team'] == 'blue' and not u.get('hero') and u['health'] > 0)
    position, health = runner['position'], runner['health']
    battle.current_blue_attacker_pos = position
    battle.blue_attacks_left = 1
    battle._advance_until_blue_turn = lambda: None
    battle.step(RUN_AWAY_ACTION_INDEX)
    if equipped and hero_turn:
        assert len(battle.escaped_units) == 1
        assert battle.escaped_units[0]['health'] == health
        assert runner['health'] == 0
        battle.winner = 'red'
        env._save_blue_state()
        saved = next(u for u in env.blue_team_state if u['position'] == position)
        assert saved['health'] == health
        assert saved['running_away'] == 0
        assert env._count_hero_item('Rusted Shackles') == 1
    else:
        assert not battle.escaped_units
        assert runner['running_away'] == 1
        assert runner['health'] == health


def test_shackles_last_survivor_ends_battle_and_preserves_hero():
    env = _env('Rusted Shackles')
    env._init_battle(enemy_id=1)
    battle = env.battle_env
    hero = _battle_blue_hero(env)
    for unit in battle.combined:
        if unit['team'] == 'blue' and unit is not hero:
            unit['health'] = 0
            unit['initiative'] = 0
    battle.current_blue_attacker_pos = hero['position']
    battle.blue_attacks_left = 1
    health = hero['health']
    _, _, terminated, _, _ = battle.step(RUN_AWAY_ACTION_INDEX)
    assert terminated
    assert battle.winner == 'red'
    assert battle.escaped_units[0]['health'] == health
    env._save_blue_state()
    assert env._resolve_travel_hero()['health'] == health
