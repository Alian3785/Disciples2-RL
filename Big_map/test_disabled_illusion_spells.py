import pytest
from campaign_env import CampaignEnv
from maps import available_maps

@pytest.mark.parametrize('map_name',available_maps())
def test_illusion_learning_and_casting_disabled_on_every_map(map_name):
    env=CampaignEnv(map_name=map_name,scripted_capital_bot_enabled=False,use_boss_starting_roster=False,log_enabled=False)
    env.reset(seed=42)
    shape=env.observation_space.shape;n=env.action_space.n
    for attr in env.MANA_ATTR_BY_KIND.values():setattr(env,attr,100000.0)
    for entry in env.active_buildings.values():
        if isinstance(entry,dict):entry['built']=entry['Build']=1
    env._has_magic_tower_built=lambda:True
    env.spell_learning_locked=False
    for key in ('lod_d2_s002','lod_d2_s007'):
        if key in env.spell_keys:
            action=env.grid_spell_action_start+env.spell_keys.index(key)
            assert not env.compute_action_mask()[action]
            mana=env._current_mana_totals()
            result=env._step_learn_spell(action)
            assert result[-1]['spell_disabled'] and not result[-1]['spell_learned']
            assert env._current_mana_totals()==mana
            assert result[-1]['spell_learning_reward']==0
            env.active_spells[key]['learned']=1 # Old saves must remain blocked.
        assert not env._can_cast_legion_damage_spell(key)
        assert not env._can_cast_support_spell(key)
        assert not env._can_cast_spell_shop_spell(key)
    for key in ('lod_d2_s002','lod_d2_s007','g000ss0042','g000ss0047'):
        entry={'spell_id':key,'supported':True,'spell_kind':'summon_battle','summon_unit_name':'Адская гончая'}
        assert not env._can_cast_scroll_spell(entry)
        assert not env._can_cast_staff_spell(entry)
        mana=env._current_mana_totals()
        result=env._cast_from_spell_spec(source='spell',spell_key=key,spell_description='illusion',spell_spec=entry,mana_costs={'infernal':100},spend_mana=True)
        assert not result['spell_cast_executed'] and result['reward']==0
        assert env._current_mana_totals()==mana
    assert env.action_space.n==n and env.observation_space.shape==shape
    env.close()

def test_real_summons_remain_available():
    env=CampaignEnv(map_name='siege_train',scripted_capital_bot_enabled=False,use_boss_starting_roster=False,log_enabled=False)
    env.reset(seed=42)
    for attr in env.MANA_ATTR_BY_KIND.values():setattr(env,attr,100000.0)
    env._has_magic_tower_built=lambda:True
    for key in ('lod_d2_s001','lod_d2_s006'):
        env.spell_learning_locked=False
        action=env.grid_spell_action_start+env.spell_keys.index(key)
        assert env.compute_action_mask()[action]
        assert env._step_learn_spell(action)[-1]['spell_learned']
        assert env._can_cast_legion_damage_spell(key)
    env.close()
