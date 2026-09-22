from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from unit_revive_costs import known_unit_revive_gold_cost, REVIVE_PROFILE_IDS_BY_NAME, REVIVE_GOLD_BY_UNIT_ID
from test_campaign_start_and_castle_heal_tiles import _mark_temple_built


@pytest.mark.parametrize('name,expected', [
    ('Гаргулья',100),('Мраморная гаргулья',400),('Ониксовая гаргулья',600),
    ('Холмовой гигант',100),('Скальной гигант',400),('Штормовой гигант',600),
    ('Демон',400),('Мизраэль',1000),('Тролль',600),('Кракен',1000),
])
def test_revive_uses_creature_profile_in_mask_and_actual_gold_payment(name,expected):
    env=CampaignEnv(log_enabled=False,persist_blue_hp=True,Realcapital=2)
    env.reset(seed=123)
    _mark_temple_built(env)
    env.grid_env.agent_pos=env.castle_heal_tiles[0]
    unit=next(u for u in env.blue_team_state if u['position']==7)
    raw=next(u for u in DATA if u['кто']==name)
    unit.clear();unit.update(map_unit_to_battle(raw,'blue',7))
    unit['hp']=unit['health']=0
    # Level and scenario HP cannot replace the original creature's tariff.
    unit['Level']=20
    unit['maxhp']=unit['max_health']=5000
    action=env.GRID_CASTLE_REVIVE_ACTION_START+env.CASTLE_REVIVE_POSITIONS.index(7)
    env.gold=expected-1
    assert not env.compute_action_mask()[action]
    env.gold=expected+25
    assert env.compute_action_mask()[action]
    _,_,_,_,info=env.step(action)
    assert info['revived']
    assert info['revive_cost']==expected
    assert info['gold_spent']==expected
    assert env.gold==25
    assert unit['hp']==unit['health']==1


def test_unit_id_is_authoritative_and_every_environment_template_has_a_verified_tariff():
    assert known_unit_revive_gold_cost({'unit_id':'G000UU0167','name':'Гаргулья'})==600
    assert known_unit_revive_gold_cost({'unit_id':'custom','name':'Гаргулья'}) is None
    assert known_unit_revive_gold_cost({'name':'Custom creature'}) is None
    for raw in DATA:
        ids=REVIVE_PROFILE_IDS_BY_NAME[raw['кто']]
        assert len({REVIVE_GOLD_BY_UNIT_ID[key] for key in ids})==1
        assert known_unit_revive_gold_cost(map_unit_to_battle(raw,'blue',7)) is not None


def test_city_upgrade_changes_future_growth_without_retroactive_tiles_or_lost_land():
    env=CampaignEnv(log_enabled=False,persist_blue_hp=True,Realcapital=2)
    env.reset(seed=123)
    name=max(env.legions_settlement_territory_orders_by_name, key=lambda n: len(env.legions_settlement_territory_orders_by_name[n]))
    env.legions_settlement_level_by_name[name]=1
    env._advance_turns(3)
    env.legions_active_settlement_territory_capture_turn_by_name[name]=env.turns
    env._refresh_faction_territories()
    assert len(env.legions_settlement_territory_tiles_by_name[name])==1
    env._advance_turns(3)
    assert len(env.legions_settlement_territory_tiles_by_name[name])==46
    action=env.grid_settlement_upgrade_action_start+env.settlement_upgrade_names.index(name)
    env.gold=10000
    for level,rate in [(2,15),(3,20),(4,20),(5,25)]:
        before=set(env.legions_settlement_territory_tiles_by_name[name])
        day=env.turns
        _,_,_,_,info=env.step(action)
        assert info['settlement_upgrade_applied']
        assert env.turns==day
        assert env.legions_settlement_level_by_name[name]==level
        env._refresh_faction_territories()
        env._refresh_faction_territories()
        assert set(env.legions_settlement_territory_tiles_by_name[name])==before
        env._advance_turns(1)
        after=set(env.legions_settlement_territory_tiles_by_name[name])
        assert before <= after
        assert len(after)-len(before)==rate
        assert env._legions_settlement_territory_info()['legions_settlement_expansion_per_turn_by_name'][name]==rate
        assert after <= env.legions_territory_tile_set
    env.reset(seed=123)
    assert env.legions_settlement_growth_history_by_name=={}
    assert env.legions_settlement_territory_tiles_by_name[name]==()
    assert env.legions_settlement_level_by_name[name]==2


def test_multiple_same_day_upgrades_and_refreshes_cannot_create_extra_territory():
    env=CampaignEnv(log_enabled=False,persist_blue_hp=True,Realcapital=2)
    env.reset(seed=123)
    name=max(env.legions_settlement_territory_orders_by_name, key=lambda n: len(env.legions_settlement_territory_orders_by_name[n]))
    env.legions_settlement_level_by_name[name]=1
    assert env._current_settlement_territory_claim_count(name,1000)==0
    env.legions_active_settlement_territory_capture_turn_by_name[name]=0
    env._advance_turns(2)
    for level in [2,3,4,5]:
        env._set_settlement_level_with_territory_growth(name,level)
        env._refresh_faction_territories()
        assert len(env.legions_settlement_territory_tiles_by_name[name])==31
    assert env.legions_settlement_growth_history_by_name[name]==[(0,15),(2,25)]
    env._advance_turns(2)
    assert len(env.legions_settlement_territory_tiles_by_name[name])==81
    assert env._current_settlement_territory_claim_count(name,40)==40
