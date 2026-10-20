"""24 source-backed effects, scenario actions and permanent growth on CUDA."""
from collections import OrderedDict
import json
from pathlib import Path
import threading

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, REST, SHOOT, BattleState
from stoix.envs.number_grid_potions import POTIONS, PotionRules, POTION_START, POTION_BUFF
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


def use(env,state,key,slot=0):
    return compiled_method(env,'step')(state,jnp.int32(POTION_START+6*env.potion_rules.keys.index(key)+slot))[0]


def stocked(env,state,count=5):
    return state.replace(potions=jnp.full(env.potion_rules.count,count,jnp.int32))


def test_scenario_actions_only_include_obtainable_potions(current_game):
    env,state,_,_=current_game
    assert len(POTIONS)==24 and len({p['game_id'] for p in POTIONS})==24
    # The five history columns identify permanent kinds, not a shared limit for all buffs.
    assert sorted(p['stat'] for p in POTIONS if p['duration']=='permanent')==list(range(5))
    assert env.potion_rules.count==18 and env.num_actions==244
    assert env.observation_size==1843 and compiled_method(env,'observation')(state).shape==(1843,)
    assert state.potions[:4].tolist()==[5,5,5,10] and not jnp.any(state.potions[4:])
    assert not {'protection','bark','striking','swiftness','speed','vigor'} & set(env.potion_rules.keys)
    rules=PotionRules(dict(initial_potions={'speed':0},chests=[dict(potions={'celerity':5})]))
    assert rules.keys==('celerity',) and rules.action_count==6
    assert rules.metadata()[0]['action_start']==POTION_START
    assert rules.initial_counts.tolist()==[0]
    assert PotionRules(dict(initial_potions={},chests=[])).count==0
    for value in ([],{'unknown':1},{'healing':-1},{'healing':True},{'healing':2**31}):
        with pytest.raises(ValueError,match='initial_potions'):
            PotionRules(dict(initial_potions=value))


def test_chest_potions_without_initial_inventory_start_at_zero():
    rules=PotionRules(dict(chests=[dict(potions={'healing':2,'celerity':5})]))
    assert rules.keys==('healing','celerity') and rules.action_count==12
    assert rules.initial_counts.tolist()==[0,0]


@pytest.mark.core
def test_catalogue_matches_installed_original_records():
    source=json.loads((Path(__file__).resolve().parents[2]/'docs/validation/potion-sources.json').read_text())
    rows={r['item_id']:r for r in source['items']}
    assert len(rows)==len(POTIONS)==24
    for item in POTIONS:
        row=rows[item['game_id']]
        assert item['price']==row['price']
        assert item['duration']=={4:'temporary',5:'instant',6:'instant',7:'permanent'}[row['category']]
        if row['category']==5:
            assert item['amount']==row['hp']
        elif row['modifiers']:
            mod=row['modifiers'][0]
            if mod['TYPE']=='12':
                assert item['effect']=={'4':'ward_Fire','5':'ward_Water','6':'ward_Earth','7':'ward_Air'}[mod['IMMUNITY']]
            else:
                assert item['effect']=={'3':'accuracy','4':'damage','5':'armor','6':'health','9':'initiative'}[mod['TYPE']]
                assert item['amount']==int(mod['NUMBER'] if mod['TYPE']=='5' else mod['PERCENT'])


def test_every_original_effect_on_all_six_targets_under_jit_vmap(current_game):
    env,initial,_,_=current_game
    rules=PotionRules(dict(initial_potions={p['key']:2 for p in POTIONS}))
    initial=hero_roster_state(env,initial,['possessed','duke','possessed','cultist','cultist','possessed'])
    initial=initial.replace(potions=rules.initial_counts,hp=initial.hp.at[:6].set(1))
    intrinsic=env._potion_intrinsic(initial.unit_ids,initial.unit_levels)
    def apply(state,action):
        return rules.apply(state,action,compiled_method(env,'max_hp')(state),intrinsic[(action-POTION_START)%6])
    actions=jnp.arange(POTION_START,POTION_START+144,dtype=jnp.int32)
    states=jax.tree.map(lambda x:jnp.broadcast_to(x,(144,)+x.shape),initial)
    states=states.replace(hp=states.hp.at[18:24,:6].set(0))
    result=jax.jit(jax.vmap(apply))(states,actions)
    stats=jax.jit(jax.vmap(env.unit_stats))(result)
    traits=jax.jit(jax.vmap(env.unit_traits))(result)
    for kind,p in enumerate(POTIONS):
        for slot in range(6):
            i=kind*6+slot
            assert result.potions[i,kind]==1 and result.last_target[i]==slot
            assert int(jnp.sum(states.potions[i]-result.potions[i]))==1
            if p['effect']=='heal':
                assert result.hp[i,slot]==min(int(intrinsic[slot,0]),1+p['amount'])
            elif p['effect']=='revive':
                assert result.hp[i,slot]==1
            elif p['effect'].startswith('ward_'):
                bit={'ward_Fire':4,'ward_Water':8,'ward_Earth':2,'ward_Air':256}[p['effect']]
                assert int(traits[i,slot,3]) & bit
            else:
                stat=p['stat']; base=float(intrinsic[slot,stat])
                expected=base+p['amount'] if stat==3 else round(base*(1+p['amount']/100))
                if stat==2: expected=min(100,expected)
                assert stats[i,slot,stat]==expected,(p['key'],slot,stats[i,slot].tolist())
                assert result.last_event[i]==POTION_BUFF
    chex.assert_tree_all_finite(result)


@pytest.mark.core
def test_map_only_validation_no_resource_cost_and_recovery(current_game):
    env,state,advance,_=current_game
    wounded=stocked(env,state).replace(hp=state.hp.at[0].set(119).at[1].set(0),movement_points=jnp.int32(0))
    healed=use(env,wounded,'ointment')
    assert healed.hp[0]==120 and healed.potions[2]==4 and healed.last_damage==1
    revived=use(env,healed,'life',1)
    assert revived.hp[1]==1 and revived.potions[3]==4
    for field in ('gold','movement_points','day','buildings','recovery_balance','unit_xp','battle_key'):
        chex.assert_trees_all_equal(getattr(revived,field),getattr(wounded,field))
    invalid=[(state,'healing',0),(state,'life',0),(state,'might',0),(wounded,'healing',1),
             (wounded.replace(summon_owner=wounded.summon_owner.at[1].set(1)),'life',1),
             (wounded.replace(summon_battle=jnp.bool_(True)),'life',1),
             (wounded,'might',5),(wounded.replace(done=jnp.bool_(True)),'might',0),
             (compiled_method(env,'_begin_battle')(wounded.replace(enemy=jnp.int32(0))),'might',0)]
    for before,key,slot in invalid:
        action=POTION_START+6*env.potion_rules.keys.index(key)+slot
        assert not compiled_method(env,'action_mask')(before)[action]
        after,_=advance(before,jnp.int32(action))
        for field in ('hp','potions','potion_doses','potion_active','gold'):
            chex.assert_trees_all_equal(getattr(after,field),getattr(before,field))
    large=hero_roster_state(env,stocked(env,state),['titan','duke','possessed',None,'cultist','cultist'])
    assert not compiled_method(env,'action_mask')(large)[POTION_START+6*env.potion_rules.keys.index('might')+3]


def test_temporary_lifetime_stack_rules_and_secondary_stats(current_game):
    env,state,_,_=current_game
    state=stocked(env,state)
    boosted=use(env,use(env,state,'might'),'celerity')
    boosted=use(env,boosted,'invulnerability')
    assert compiled_method(env,'unit_stats')(boosted)[0,1]==38 and compiled_method(env,'unit_stats')(boosted)[0,4]==80
    assert compiled_method(env,'unit_stats')(boosted)[0,3]==50
    again=use(env,boosted,'might')
    chex.assert_trees_all_equal(again.potions,boosted.potions)
    rested=use(env,state,'titan')
    rested=use(env,rested,'might')
    after,_=compiled_method(env,'step')(rested,jnp.int32(REST))
    assert not jnp.any(after.potion_active) and not jnp.any(after.potion_wards)
    assert compiled_method(env,'unit_stats')(after)[0,1]==28
    chex.assert_trees_all_equal(after.potion_doses,rested.potion_doses)
    chex.assert_trees_all_equal(after.potions,rested.potions)
    rules=PotionRules(dict(initial_potions={p['key']:2 for p in POTIONS}))
    full=state.replace(potions=rules.initial_counts)
    raw=env._potion_intrinsic(full.unit_ids,full.unit_levels)
    apply=jax.jit(lambda s,a:rules.apply(s,a,compiled_method(env,'max_hp')(s),raw[(a-POTION_START)%6]))
    for key in ('vigor','strength','might'):
        full=apply(full,jnp.int32(POTION_START+6*rules.keys.index(key)))
    assert compiled_method(env,'unit_stats')(full)[0,1]==round(25*1.15*1.3*1.5)


@pytest.mark.core
def test_permanent_doses_survive_promotion_and_hero_growth(current_game):
    env,state,_,_=current_game
    state=stocked(env,state)
    for slot in (0,1):
        state=use(env,state,'highfather',slot)
        state=use(env,state,'titan',slot)
    assert compiled_method(env,'max_hp')(state)[0]==round(120*1.15)
    battle=compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0),buildings=jnp.uint32(1))
    battle=battle.replace(hp=battle.hp.at[0].set(1).at[1].set(1).at[6].set(1),
        unit_xp=battle.unit_xp.at[0].set(94).at[1].set(149))
    won,_=compiled_method(env,'_battle_step')(battle,jnp.int32(SHOOT),battle.battle_key,jnp.zeros(env.random_size))
    assert won.unit_ids[0]==env.progression.ids['berserker'] and won.unit_levels[1]==2
    assert won.hp[0]==round(170*1.15) and won.hp[0]==compiled_method(env,'max_hp')(won)[0]
    assert won.hp[1]==compiled_method(env,'max_hp')(won)[1]
    assert compiled_method(env,'unit_stats')(won)[0,1]==round(50*1.1)
    chex.assert_trees_all_equal(won.potion_doses,state.potion_doses)
    mask=compiled_method(env,'action_mask')(won)
    for slot in (0,1):
        for key in ('highfather','titan'):
            assert not mask[POTION_START+6*env.potion_rules.keys.index(key)+slot]


def test_secondary_attacks_copy_and_transformed_stat_layers(current_game):
    # User request 2026-10-19: potions and spells, like banners and artifacts, act only
    # on the primary attack; the assassin's poison and its accuracy keep their growth values.
    env,state,_,_=current_game
    state=hero_roster_state(env,state,['doppelganger','duke','possessed','imperial_assassin','cultist',None])
    state=stocked(env,state)
    base=env.progression.secondary_stats(state.unit_ids[3],state.unit_levels[3])
    natural=compiled_method(env,'unit_stats')(state)[3]
    state=use(env,use(env,state,'titan',3),'fortune',3)
    state=use(env,state,'might',3)
    boosted=compiled_method(env,'unit_stats')(state)[3]
    assert boosted[1]>natural[1] and boosted[2]>natural[2]
    assert not jnp.any(state.potion_bonus[:,5:]) and not jnp.any(state.potion_temporary[:,5:])
    chex.assert_trees_all_equal(env._secondary_stats(state,jnp.int32(3)),base)
    battle=compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    spelled=battle.replace(spell_mods=battle.spell_mods.at[3].set(jnp.array([0.,1.5,1.3,0.,1.])))
    chex.assert_trees_all_equal(env._secondary_stats(spelled,jnp.int32(3)),base)
    copied=compiled_method(env,'_copy_unit')(battle,jnp.int32(3),jnp.bool_(True))
    chex.assert_trees_all_equal(copied.potion_doses,battle.potion_doses)
    chex.assert_trees_all_equal(copied.copy_secondary[0],base)
    chex.assert_trees_all_equal(env._secondary_stats(copied,jnp.int32(0)),base)
    # Witch form hides the potion's combat stats; cleansing restores them exactly.
    native=compiled_method(env,'unit_stats')(battle)
    imp=battle.replace(imp=battle.imp.at[3].set(True))
    assert compiled_method(env,'unit_stats')(imp)[3,1]==20 and env._secondary_stats(imp,jnp.int32(3))[0]==0
    cured,_,_=compiled_method(env,'_cleanse')(imp,imp.hp,imp.wards_used,jnp.arange(12)==3)
    chex.assert_trees_all_equal(cured.potion_doses,battle.potion_doses)
    chex.assert_trees_all_equal(compiled_method(env,'unit_stats')(cured)[3],native[3])

    # Duke's intrinsic damage exceeds the attack cap at high levels. Both
    # permanent and temporary potions precede weakness and the final cap.
    veteran=stocked(env,current_game[1]).replace(unit_levels=current_game[1].unit_levels.at[1].set(80))
    veteran=use(env,use(env,veteran,'titan',1),'might',1)
    original=round(round(512*1.1)*1.5)
    veteran=veteran.replace(weakened=veteran.weakened.at[1].set(True))
    assert env._original_primary(veteran)[1]==original
    assert env._combat_stats(veteran,cap=False)[1,1]==round(original*.68)
    assert compiled_method(env,'_combat_stats')(veteran)[1,1]==400  # Duke's original heavy-strike cap.


def test_armor_and_elemental_wards_affect_actual_attacks(current_game):
    env,state,_,_=current_game
    state=stocked(env,state)
    buffed=use(env,state,'invulnerability')
    attack=compiled_method(env,'_battle_step')
    outcomes=[]
    for s in (state,buffed):
        b=compiled_method(env,'_begin_battle')(s.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(6))
        out,_=attack(b,jnp.int32(SHOOT),b.battle_key,jnp.zeros(env.random_size))
        outcomes.append(int(b.hp[0]-out.hp[0]))
    assert outcomes==[25,12]
    ward=use(env,state,'fire_ward')
    b=compiled_method(env,'_begin_battle')(ward.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(6))
    b=b.replace(unit_ids=b.unit_ids.at[6].set(env.progression.ids['cultist']))
    out,_=attack(b,jnp.int32(SHOOT),b.battle_key,jnp.zeros(env.random_size))
    assert out.hp[0]==b.hp[0] and out.wards_used[0]&4
    out,_=attack(out.replace(actor=jnp.int32(6)),jnp.int32(SHOOT),out.battle_key,jnp.zeros(env.random_size))
    assert out.hp[0]<b.hp[0]
    restarted=compiled_method(env,'_begin_battle')(out.replace(in_battle=jnp.bool_(False)))
    assert restarted.potion_wards[0]&4 and restarted.wards_used[0]==0


def test_autoreset_restores_inventory_and_potion_layers(current_game,training_autoreset):
    training,_,advance=training_autoreset
    state,_=training.reset(jax.random.split(jax.random.PRNGKey(42),2))
    def injure(s):
        if isinstance(s,BattleState):
            return s.replace(hp=s.hp.at[:,0].set(1),potions=s.potions.at[:,0].set(2),
                potion_doses=s.potion_doses.at[:,0,:].set(1),
                potion_active=s.potion_active.at[:,0].set(jnp.uint32(1<<14)))
        return s.replace(base_env_state=injure(s.base_env_state))
    state=injure(state)
    following,ts=advance(state,jnp.array([POTION_START,POTION_START],jnp.int32))
    assert jnp.all(ts.truncated())
    chex.assert_trees_all_equal(following.potions,jnp.stack([current_game[1].potions]*2))
    assert not jnp.any(following.potion_active) and not jnp.any(following.potion_doses)


def test_human_service_uses_shared_potion_effects(current_game):
    from serve_number_grid import GameService
    env,state,advance,_=current_game
    service=GameService.__new__(GameService)
    service.lock=threading.Lock()
    key=(env.construction.faction,env.construction.lord['id'])
    service.environments={key:(env,compiled_method(env,'reset'),advance)}
    state=stocked(env,state)
    service.sessions=OrderedDict({'potions':(key,state,0.)})
    action=POTION_START+6*env.potion_rules.keys.index('celerity')
    result=service.act('potions',action)['snapshot']
    assert result['unit_stats'][0][4]==80 and result['state']['last_event']==POTION_BUFF
    assert not result['action_mask'][action]
