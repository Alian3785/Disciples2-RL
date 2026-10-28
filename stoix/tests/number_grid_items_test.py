"""GPU equipment regressions; reuse current-map transitions and small kernels."""
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state
import copy
import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, REST, POTION_START
from stoix.envs.number_grid_items import ITEMS, ItemRules, VALUABLE
from stoix.envs.number_grid_item_combat import equipment_attack


@pytest.fixture(scope='module')
def all_items(current_game):
    original,state,_,_=current_game
    env=copy.copy(original)
    env._test_jit_methods={}
    env.item_rules=ItemRules(dict(initial_items={p['key']:1 for p in ITEMS}))
    state=state.replace(item_inventory=jnp.full(len(ITEMS),-1,jnp.int32),
                        last_item_loot=jnp.zeros(len(ITEMS),jnp.int32))
    env.random_size=84
    env.has_copies=True
    refresh=jax.jit(env._refresh_equipment)
    attack=jax.jit(lambda s,target,connected,damage,rolls:equipment_attack(
        env,s,s.hp,s.wards_used,target,connected,damage,rolls))
    return env,state,refresh,attack


def inventory(env,state,*keys):
    ids=[env.item_rules.keys.index(key) for key in keys]
    return state.replace(item_inventory=jnp.array(ids+[-1]*(env.item_rules.capacity-len(ids)),jnp.int32))


def test_boot_percentages_scale_with_base_movement(all_items):
    env,state,_,_=all_items
    keys=('boots_speed','boots_traveling','boots_seven_leagues')
    ids=[-1]+[env.item_rules.keys.index(key) for key in keys]
    bases=jnp.repeat(jnp.array([20,25,30,21],jnp.int32),4)
    equipped=jnp.full((16,5),-1,jnp.int32).at[:,4].set(jnp.tile(jnp.array(ids,jnp.int32),4))
    calculate=jax.jit(jax.vmap(lambda base,slots:
        env.item_rules.movement_cap(state.replace(equipped=slots),base)))
    caps=calculate(bases,equipped).reshape(4,4)
    chex.assert_trees_all_equal(caps,jnp.array([
        [20,24,28,32], [25,30,35,40], [30,36,42,48], [21,25,29,33]],jnp.int32))
    assert [env.item_rules.items[i]['movement_percent'] for i in ids[1:]]==[20,40,60]


def test_complete_catalogue_and_scenario_inventory(current_game):
    env,state,_,_=current_game
    assert len(ITEMS)==255 and len({p['key'] for p in ITEMS})==255
    # Usable: 44 orbs, 41 talismans (number_grid_battle_items); 101 scrolls, 14 staves.
    assert [sum(p['category']==c for p in ITEMS) for c in range(8)]==[21,11,8,5,10,85,101,14]
    assert [sum(c['items'].values()) for c in MAP['chests']]==[3]*5
    assert env.item_rules.capacity==15 and env.item_rules.count==15
    assert env.num_actions==244 and compiled_method(env,'observation')(state).shape==(1843,)
    assert not jnp.any(state.item_inventory>=0) and jnp.all(state.equipped==-1)
    valuables=[p for p in ITEMS if p['category']==VALUABLE]
    # Original Tglobal item descriptions give the merchant payout, not GItem.VALUE.
    assert [p['sell_price'] for p in valuables]==[50,100,150,200,250,300,350,400,500,1000]
    assert all(p['price']==p['sell_price']*5 for p in valuables)
    bronze=next(p for p in env.item_rules.metadata['items'] if p['key']=='bronze_ring')
    assert bronze['sell_price']==50 and bronze['price']==250


def test_accuracy_banners_use_relative_percent_and_truncate(all_items):
    env,state,_,_=all_items
    state=hero_roster_state(env,state,['possessed','duke','possessed','cultist','cultist',None])
    ids=[-1]+[env.item_rules.keys.index(key) for key in ('banner_striking','banner_battle')]
    equipped=jnp.full((3,5),-1,jnp.int32).at[:,2].set(jnp.array(ids,jnp.int32))
    intrinsic=env._potion_intrinsic(state.unit_ids,state.unit_levels)
    intrinsic=intrinsic.at[:,2].set(jnp.array([80,86,50,0,95,80,80,80,80,80,80,80]))
    calculate=jax.jit(jax.vmap(lambda slots:intrinsic+env.item_rules.numeric_bonus(
        state.replace(equipped=slots),intrinsic,jnp.int32(1))))
    stats=calculate(equipped)
    # Include rounding, the 100% cap, an empty allied slot, and unaffected enemies.
    chex.assert_trees_all_equal(stats[:,:,2],jnp.array([
        [80,86,50,0,95,80,80,80,80,80,80,80],
        [88,94,55,0,100,80,80,80,80,80,80,80],
        [92,98,57,0,100,80,80,80,80,80,80,80]],jnp.float32))
    other_columns=jnp.array([0,1,3,4,5,6])
    chex.assert_trees_all_equal(stats[:,:,other_columns],
        jnp.broadcast_to(intrinsic[:,other_columns],(3,12,6)))


def test_auto_selection_retains_inventory_and_two_artifact_instances(all_items):
    env,state,refresh,_=all_items
    keys=('runestone','holy_chalice','ring_of_ages','boots_speed','elemental_boots',
          'tome_air','tome_war','banner_protection','banner_strength','imperial_crown')
    state=inventory(env,state,*keys)
    result=refresh(state)
    names=[env.item_rules.keys[int(i)] for i in result.equipped]
    assert names==['ring_of_ages','holy_chalice','banner_strength','tome_war','elemental_boots']
    chex.assert_trees_all_equal(result.item_inventory,state.item_inventory)
    duplicated=refresh(inventory(env,state,'runestone','runestone','dwarven_bracer'))
    assert duplicated.equipped[:2].tolist()==[env.item_rules.keys.index('runestone')]*2
    assert compiled_method(env,'unit_stats')(duplicated)[1,3]==compiled_method(env,'unit_stats')(state)[1,3]+20
    # A valuable never enters an equipment slot, regardless of its price.
    valuable=refresh(inventory(env,state,'imperial_crown'))
    assert jnp.all(valuable.equipped==-1) and valuable.gold==state.gold


def test_numeric_layers_round_and_do_not_accumulate(all_items):
    env,state,refresh,_=all_items
    state=inventory(env,state,'ring_of_ages','holy_chalice','banner_battle')
    base=compiled_method(env,'unit_stats')(state)
    result=refresh(state)
    stats=compiled_method(env,'unit_stats')(result)
    assert stats[1,1]==round(float(base[1,1])*1.4)
    assert stats[1,4]==round(float(base[1,4])*1.25)
    assert stats[1,3]==base[1,3]+15
    chex.assert_trees_all_equal(stats[:5,2],jnp.minimum(100,base[:5,2]+base[:5,2]*15//100))
    chex.assert_trees_all_equal(compiled_method(env,'unit_stats')(refresh(result)),stats)
    assert stats[0,1]==base[0,1] and stats[0,3]==base[0,3]
    # Temporary +15% then artifact +40%: two reference rounding stages.
    active=state.potion_active.at[1].set(jnp.uint32(1<<12))
    intrinsic=env._potion_intrinsic(state.unit_ids,state.unit_levels)
    potion=env.potion_rules.temporary_bonus(intrinsic,active)
    result=refresh(state.replace(potion_active=active,potion_temporary=potion))
    assert compiled_method(env,'unit_stats')(result)[1,1]==round(round(float(base[1,1])*1.15)*1.4)


def test_hero_death_disables_equipment_until_revival(all_items):
    env,state,refresh,_=all_items
    state=refresh(inventory(env,state,'runestone','banner_protection','boots_seven_leagues','tome_air'))
    assert env.movement_cap(state)==32
    assert compiled_method(env,'_combat_traits')(state)[1,3] & 256
    assert not (compiled_method(env,'_combat_traits')(state)[0,3] & 256)
    dead=refresh(state.replace(hp=state.hp.at[1].set(0)))
    assert jnp.all(dead.equipped==-1) and env.movement_cap(dead)==20
    chex.assert_trees_all_equal(dead.item_inventory,state.item_inventory)
    revived=refresh(dead.replace(hp=dead.hp.at[1].set(1)))
    chex.assert_trees_all_equal(revived.equipped,state.equipped)
    assert revived.hp[1]==1


def test_reference_selection_and_skill_thresholds(all_items):
    env,state,_,_=all_items
    state=inventory(env,state,'runestone','banner_war','boots_speed','elemental_boots','tome_air','tome_war')
    rules={lord:ItemRules(dict(initial_items={p['key']:1 for p in ITEMS},lord_type=lord,
                               equipment_auto_equip='reference',equipment_requires_skills=True))
           for lord in ('mage','warrior','guildmaster')}
    choose={lord:jax.jit(r.choose) for lord,r in rules.items()}
    def equipped(lord,level):
        ids=choose[lord](state,jnp.bool_(True),jnp.int32(level))
        return [rules[lord].keys[int(i)] if i>=0 else None for i in ids]
    assert equipped('mage',1)==[None]*5
    assert equipped('mage',11)==['runestone',None,'banner_war','tome_air','boots_speed']
    # Banner 7, boots 8, artifacts 9, book 11; a warrior knows artifacts, a guildmaster boots.
    assert equipped('mage',8)==[None,None,'banner_war',None,'boots_speed']
    assert equipped('warrior',1)==['runestone']+[None]*4
    assert equipped('guildmaster',1)==[None]*4+['boots_speed']


def test_banner_health_regeneration_and_instant_retreat(all_items):
    from stoix.envs.number_grid import RETREAT, ESCAPE, GUARD, FORT_BLUE
    env,state,refresh,_=all_items
    # One additional scalar kernel covers map rest and actual battle withdrawal.
    # Other equipment tests call small shared kernels, avoiding one full env per item.
    env=copy.copy(env)
    env._test_jit_methods={}
    env.chests_enabled=False
    env.random_size=84
    state=refresh(inventory(env,state,'banner_health','rusted_shackles'))
    state=state.replace(hp=state.hp.at[3].set(1))
    resting_hp=jax.jit(lambda s:env._world_step(s,jnp.int32(REST),s.battle_key,jnp.zeros(84))[0].hp)(state)
    assert resting_hp[3]==31  # ceil(45*65%) = 30 HP; capital + warrior lord + banner, rounded once
    start=compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(1))
    @jax.jit
    def retreat(s):
        result=env._battle_step(s,jnp.int32(RETREAT),s.battle_key,jnp.zeros(84))[0]
        return result.escaped[1],result.retreating[1],result.last_event
    escaped,retreating,event=retreat(start)
    assert escaped and not retreating and event==ESCAPE
    defender = start.replace(battle_fort_defender=jnp.int32(FORT_BLUE))
    escaped,retreating,event=retreat(defender)
    assert not escaped and not retreating and event==GUARD


@pytest.mark.core
def test_chests_equip_boots_restore_movement_and_keep_loot_once(current_game):
    env,state,advance,_=current_game
    picked,_=advance(state.replace(position=jnp.array([5,2])),jnp.int32(2))
    assert env.movement_cap(picked)==24 and picked.movement_points==22
    assert jnp.sum(picked.last_item_loot)==3 and jnp.sum(picked.item_inventory>=0)==3
    assert compiled_method(env,'unit_stats')(picked)[1,3]==compiled_method(env,'unit_stats')(state)[1,3]+10
    repeated,_=advance(picked,jnp.int32(2))
    assert not jnp.any(repeated.last_item_loot)
    assert jnp.sum(repeated.item_inventory>=0)==3
    rested,_=advance(repeated,jnp.int32(REST))
    assert rested.movement_points==24
    at_enemy=rested.replace(position=env.opponent_positions[0]-jnp.array([0,1]))
    attacked,_=advance(at_enemy,jnp.int32(2))
    assert attacked.in_battle and attacked.movement_points==12


@pytest.mark.core
def test_permanent_potion_keeps_equipment_and_revival_restores_it(current_game):
    env,state,advance,_=current_game
    picked,_=advance(state.replace(position=jnp.array([5,2])),jnp.int32(2))
    dead=picked.replace(hp=picked.hp.at[1].set(0))
    dead=compiled_method(env,'_refresh_equipment')(dead)
    life=POTION_START+6*env.potion_rules.keys.index('life')+1
    revived,_=advance(dead,jnp.int32(life))
    assert revived.hp[1]==1 and revived.equipped[0]==picked.equipped[0]
    assert revived.movement_points==env.movement_cap(revived)


def test_charmed_child_city_and_day_loot(current_game):
    """Installed S117 vaults, Clans capture events and the day-6 relic (docs/MAPS.md)."""
    from stoix.envs.number_grid import load_map
    from stoix.envs.number_grid_potions import PotionRules
    _,state,_,_=current_game
    config=load_map('the_charmed_child')
    names=[c['name'] for c in config['cities']]
    rules,potions=ItemRules(config),PotionRules(config)
    base=state.replace(item_inventory=jnp.full(rules.capacity,-1,jnp.int32),
                       last_item_loot=jnp.zeros(rules.count,jnp.int32))
    captured=jnp.array([name in ('Akar',"Hasser's Keep") for name in names])
    taken=jax.jit(rules.collect_cities)(base,captured)
    assert sorted(rules.keys[i] for i in taken.item_inventory.tolist() if i>=0)==['orc_orb','staff_of_travelling']
    assert int(jnp.sum(taken.last_item_loot))==2
    vaults={name:{potions.keys[i]:int(n) for i,n in enumerate(row) if n}
            for name,row in zip(names,potions.city_loot.tolist())}
    assert vaults['Fogfell']=={'restoration':1,'life':1} and vaults['Odzark']=={'restoration':1}
    # Only the rest that begins day 6 grants the relic; days grow by one, so once.
    relic=rules.keys.index('ancient_relic')
    collect=jax.jit(rules.collect_days)
    for day,resting,expected in ((6,True,1),(6,False,0),(5,True,0),(12,True,0)):
        after=collect(base.replace(day=jnp.int32(day)),jnp.bool_(resting))
        assert int(jnp.sum(after.item_inventory==relic))==expected
    with pytest.raises(ValueError,match='day_items'):
        ItemRules({**config,'day_items':[dict(day=1,items={'ruby':1})]})


def test_war_book_adds_truncated_bonus_to_personal_xp(all_items):
    env,state,refresh,_=all_items
    state=hero_roster_state(env,state,['possessed','duke','possessed','cultist','cultist',None])
    state=refresh(inventory(env,state,'tome_war'))
    state=state.replace(battle_earned_xp=jnp.array([2,8,3,4,7,0,0,0,0,0,0,0],jnp.int32))
    # Original sub_5840D8: earned + trunc(earned*25/100).
    finish=jax.jit(lambda s:env.progression.finish(s,s.hp,s.escaped,jnp.bool_(True),
        jnp.bool_(False),jnp.bool_(False),env.item_rules.value(s,'experience',3)))
    result=finish(state)
    assert result[5][:5].tolist()==[2,10,3,5,8]


def test_artifact_statuses_respect_sources_and_duplicate_names(all_items):
    env,state,refresh,attack=all_items
    start=compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0)))
    # Source masks are actual Death/Mind bits, distinct from the poison tick bit.
    cases=(('soul_crystal','paralyzed',64),('horn_of_incubus','paralyzed',2),
           ('thanatos_blade','poison_mode',32),('skull_of_thanatos','poison_mode',32),
           ('hags_ring','imp',64))
    for key,field,source in cases:
        before=refresh(inventory(env,start,key,key)).replace(actor=jnp.int32(1))
        zeros=jnp.zeros(84)
        affected,hp,used,immune,ward=attack(before,jnp.int32(6),jnp.bool_(True),jnp.int32(10),zeros)
        assert getattr(affected,field)[6] and not immune and not ward
        # Grant a native book/potion-style ward via copied traits of the target.
        traits=compiled_method(env,'_combat_traits')(before).at[6,3].set(jnp.uint32(source))
        protected=before.replace(copied=before.copied.at[6].set(True),copy_traits=traits,
                                 copy_stats=compiled_method(env,'_combat_stats')(before))
        blocked,_,used,_,ward=attack(protected,jnp.int32(6),jnp.bool_(True),jnp.int32(10),zeros)
        assert not getattr(blocked,field)[6] and (used[6]&source) and (ward&(1<<6))
        immune_state=protected.replace(copy_traits=traits.at[6,2].set(jnp.uint32(source)))
        blocked,_,used,immune,ward=attack(immune_state,jnp.int32(6),jnp.bool_(True),jnp.int32(10),zeros)
        assert not getattr(blocked,field)[6] and (immune&(1<<6)) and not ward and not used[6]
        # The second identical artifact must not bypass the ward just spent.
        missed=attack(before,jnp.int32(6),jnp.bool_(False),jnp.int32(10),zeros)[0]
        assert not getattr(missed,field)[6]


def test_artifact_drain_poison_refresh_and_wight(all_items):
    env,state,refresh,attack=all_items
    start=compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0)))
    start=start.replace(actor=jnp.int32(1),hp=start.hp.at[1].set(10))
    before=refresh(inventory(env,start,'unholy_dagger','unholy_dagger'))
    result=attack(before,jnp.int32(6),jnp.bool_(True),jnp.int32(19),jnp.zeros(84))
    assert result[1][1]==14  # floor(19/4), once per named artifact
    before=refresh(inventory(env,start,'skull_of_thanatos'))
    before=before.replace(poison_mode=before.poison_mode.at[6].set(2),
                          poison_damage=before.poison_damage.at[6].set(20),
                          poison_source=before.poison_source.at[6].set(3))
    result=attack(before,jnp.int32(6),jnp.bool_(True),jnp.int32(10),jnp.zeros(84))[0]
    assert result.poison_damage[6]==35 and result.poison_source[6]==1 and result.poison_mode[6]==2
    before=before.replace(poison_damage=before.poison_damage.at[6].set(50))
    result=attack(before,jnp.int32(6),jnp.bool_(True),jnp.int32(10),jnp.zeros(84))[0]
    assert result.poison_damage[6]==35 and result.poison_mode[6]==2
    # Use an existing upgrade with a known predecessor, instead of a level-1 foe.
    upgraded=env.progression.ids['knight']
    before=refresh(inventory(env,start,'wight_blade')).replace(
        unit_ids=start.unit_ids.at[6].set(upgraded),
        unit_levels=start.unit_levels.at[6].set(env.progression.base_levels[upgraded]))
    result=attack(before,jnp.int32(6),jnp.bool_(True),jnp.int32(10),jnp.zeros(84))[0]
    assert result.decay_form[6]==env.progression.lower_forms[upgraded] and result.decay_form[6]>0


@pytest.mark.parametrize('key,source', [('soul_crystal', 64), ('horn_of_incubus', 2)])
def test_equipment_disable_misses_after_round_40_without_spending_wards(all_items, key, source):
    env, state, refresh, attack = all_items
    start = compiled_method(env, '_begin_battle')(state.replace(enemy=jnp.int32(0)))
    before = refresh(inventory(env, start, key)).replace(actor=jnp.int32(1), round=jnp.int32(40))
    rolls = jnp.zeros(84)
    early = attack(before, jnp.int32(6), jnp.bool_(True), jnp.int32(10), rolls)
    assert early[0].paralyzed[6]
    late = before.replace(round=jnp.int32(41), potion_wards=before.potion_wards.at[6].set(source))
    after, hp, used, immune, ward = attack(late, jnp.int32(6), jnp.bool_(True), jnp.int32(10), rolls)
    assert not after.paralyzed[6] and not after.long_paralyzed[6]
    assert not used[6] and not immune and not ward
    assert jnp.array_equal(hp, late.hp)
