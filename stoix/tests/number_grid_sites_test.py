"""Source-backed sites, economy invariants and JAX mask/step agreement on CUDA."""
import copy
from collections import deque

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, ACTIONS, ACTION_NAMES, REST
from stoix.envs.number_grid_items import ItemRules
from stoix.envs.number_grid_sites import SiteRules, site_layout, BOUGHT, TRAINED
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


def at_site(env, state, kind):
    site = next(s for s in env.site_rules.sites if s['kind'] == kind)
    return state.replace(position=jnp.array(site['interaction_tiles'][0]), gold=jnp.int32(9000))


def inventory(env, state, keys):
    ids = [env.item_rules.keys.index(k) for k in keys]
    return state.replace(item_inventory=jnp.array(ids+[-1]*(env.item_rules.capacity-len(ids))))


def unchanged(before, after):
    for key in ('position', 'gold', 'potions', 'merchant_stock', 'item_inventory', 'equipped',
                'hp', 'unit_xp', 'unit_levels', 'movement_points', 'day'):
        chex.assert_trees_all_equal(getattr(before, key), getattr(after, key))


@pytest.mark.core
def test_original_site_geometry_and_reachable_entrances(current_game):
    env, state, _, _ = current_game
    sites = site_layout(MAP)
    # Explicit original entrance offsets, not a ring around all nine tiles.
    for site in sites:
        r, c = site['position']
        assert set(site['footprint']) == {(r+i,c+j) for i in range(3) for j in range(3)}
        assert set(site['interaction_tiles']) == {(r+1,c+3),(r+2,c+3),(r+3,c+1),(r+3,c+2),(r+3,c+3)}
        positions = jnp.array([(r+i,c+j) for i in range(-1,5) for j in range(-1,5)])
        available = jax.jit(jax.vmap(lambda p,kind=site['kind']:env.site_rules.at(state.replace(position=p),kind)))(positions)
        expected = jnp.array([tuple(p) in site['interaction_tiles'] for p in positions.tolist()])
        chex.assert_trees_all_equal(available,expected)
    blocked = {p for s in sites for p in s['footprint']} | {tuple(p) for p in MAP['opponent_positions']}
    blocked |= {tuple(p) for p in MAP.get('obstacles',[])}
    start = tuple(MAP['agent_position'])
    visited, queue = {start}, deque([start])
    while queue:
        r,c = queue.popleft()
        for dr,dc in ((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)):
            p = r+dr,c+dc
            if all(0 < v < MAP['size']-1 for v in p) and p not in blocked and p not in visited:
                visited.add(p)
                queue.append(p)
    assert all(p in visited for s in sites for p in s['interaction_tiles'])
    for kind in ('merchant','trainer','mercenary'):
        broken=copy.deepcopy(MAP)
        broken[kind]['position']=MAP['agent_position']
        with pytest.raises(ValueError,match='overlap'):
            site_layout(broken)


def test_footprints_block_mask_and_step(current_game):
    env, state, advance, _ = current_game
    for site in env.site_rules.sites:
        r,c=site['position']
        for position,action in (((r-1,c),4),((r+2,c+3),6),((r+3,c+3),7)):
            before=state.replace(position=jnp.array(position))
            assert not compiled_method(env,'action_mask')(before)[action]
            after,_=advance(before,jnp.int32(action))
            unchanged(before,after)


@pytest.mark.core
def test_exact_three_purchase_actions_and_finite_stock(current_game):
    env, initial, advance, _ = current_game
    rules=env.site_rules
    assert rules.buy_keys==('invulnerability','healing','life')
    assert initial.merchant_stock.tolist()==[1,10,10]
    assert rules.buy_start==171 and rules.train_start==174
    assert env.num_actions==ACTIONS==244 and len(ACTION_NAMES)==244
    assert env.observation_size==1844 and compiled_method(env,'observation')(initial).shape==(1844,)
    state=at_site(env,initial,'merchant').replace(movement_points=jnp.int32(0))
    before=state
    purchases=CAMPAIGN_FIELDS.index('merchant_purchases')
    state,ts=advance(state,jnp.int32(rules.buy_start))
    assert state.last_event==BOUGHT and state.gold==before.gold-700
    assert ts.extras['campaign_events'][purchases]==1
    assert state.potions[env.potion_rules.keys.index('invulnerability')]==1
    assert state.merchant_stock.tolist()==[0,10,10]
    assert not compiled_method(env,'action_mask')(state)[rules.buy_start]
    repeated,ts=advance(state,jnp.int32(rules.buy_start))
    unchanged(state,repeated)
    assert ts.extras['campaign_events'][purchases]==0
    state,_=advance(state,jnp.int32(REST))
    assert state.merchant_stock[0]==0  # No daily replenishment.
    for i,price in ((1,150),(2,400)):
        before=at_site(env,initial,'merchant').replace(gold=jnp.int32(price))
        after,_=advance(before,jnp.int32(rules.buy_start+i))
        assert after.gold==0 and after.merchant_stock[i]==9
        assert after.potions[rules.buy_ids[i]]==before.potions[rules.buy_ids[i]]+1
        assert after.day==before.day and after.movement_points==before.movement_points
    reset,_=compiled_method(env,'reset')(jax.random.PRNGKey(4))
    assert reset.merchant_stock.tolist()==[1,10,10]


@pytest.mark.core
def test_trade_and_training_reject_unavailable_actions(current_game):
    env, initial, advance, _ = current_game
    rules=env.site_rules
    buy=rules.buy_start+1
    train=rules.train_start
    variants=[(initial,buy),(initial,train),
        (at_site(env,initial,'merchant').replace(gold=jnp.int32(149)),buy),
        (at_site(env,initial,'merchant').replace(hp=initial.hp.at[1].set(0)),buy),
        (at_site(env,initial,'trainer').replace(hp=initial.hp.at[0].set(0)),train),
        (at_site(env,initial,'trainer').replace(hp=initial.hp.at[1].set(0)),train),
        (at_site(env,initial,'trainer'),rules.train_start+5),
        (at_site(env,initial,'trainer').replace(gold=jnp.int32(3)),train)]
    battle=compiled_method(env,'_begin_battle')(at_site(env,initial,'merchant').replace(enemy=jnp.int32(0)))
    variants += [(battle,buy),(battle,train)]
    for before,action in variants:
        assert not compiled_method(env,'action_mask')(before)[action]
        after, ts = advance(before,jnp.int32(action))
        unchanged(before,after)
        assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('trainer_uses')] == 0


def test_no_manual_sales_and_retained_inventory(current_game):
    env, initial, advance, _ = current_game
    rules = env.site_rules
    assert not any(name.startswith('sell_') for name in ACTION_NAMES)
    assert 'sell' not in rules.metadata
    assert rules.names == ('buy_invulnerability', 'buy_healing', 'buy_life') + tuple('train_'+str(i) for i in range(6))
    state = compiled_method(env,'_refresh_equipment')(
        inventory(env, at_site(env, initial, 'merchant'), ['boots_speed','runestone','runestone']))
    # Former sales outside the reduced action space cannot remove anything.
    # 174 means training and 180 means hiring; neither is available at the merchant.
    for action in (174, 180, 193, 198, 204):
        unchanged(state, advance(state, jnp.int32(action))[0])
    rested, _ = advance(state, jnp.int32(REST))
    for key in ('item_inventory', 'equipped', 'potions'):
        chex.assert_trees_all_equal(getattr(rested,key), getattr(state,key))
    assert rested.gold == state.gold+100 and rested.last_sold_count == 0


def test_autosale_all_valuables_once_and_pickup_after_compaction(current_game):
    env,initial,advance,_=current_game
    # Enter the merchant approach from the east; both duplicate valuables sell.
    state=inventory(env,initial,['bronze_ring','runestone','bronze_ring'])
    state=state.replace(position=jnp.array([9,7]))
    sold,_=advance(state,jnp.int32(6))
    assert sold.position.tolist()==[9,6] and sold.gold==100
    assert sold.last_sold_count==2 and sold.last_sale_gold==100
    assert sold.item_inventory[0]==env.item_rules.keys.index('runestone')
    assert jnp.all(sold.item_inventory[1:]==-1)
    again,_=advance(sold,jnp.int32(4))
    assert again.gold==100 and again.last_sale_gold==0
    # Collect chest 1 after selling holes in the inventory: keep the old runestone.
    picked,_=advance(again.replace(position=jnp.array([5,2])),jnp.int32(2))
    assert jnp.sum(picked.item_inventory>=0)==4
    assert env.item_rules.counts(picked)[env.item_rules.keys.index('runestone')]==2
    trainer=at_site(env,state,'trainer')
    assert env.site_rules.autosell(trainer).gold==trainer.gold


@pytest.mark.core
def test_training_partial_full_hero_growth_and_cap(current_game):
    env,initial,advance,_=current_game
    rules=env.site_rules
    state=at_site(env,initial,'trainer').replace(gold=jnp.int32(11),movement_points=jnp.int32(0))
    state=compiled_method(env,'_refresh_equipment')(inventory(env,state,['tome_war']))  # Battle XP bonuses do not apply.
    result,ts=advance(state,jnp.int32(rules.train_start))
    assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('trainer_uses')] == 1
    assert result.last_event==TRAINED and result.unit_xp[0]==2 and result.gold==3
    assert result.last_service_cost==8 and result.last_xp[0]==2
    assert result.day==state.day and result.movement_points==0
    chex.assert_trees_all_equal(result.unit_levels,state.unit_levels)
    chex.assert_trees_all_equal(result.hp,state.hp)
    for slot in (0,1):
        before=at_site(env,initial,'trainer')
        cap=int(env.progression.required_xp(before.unit_ids[slot],before.unit_levels[slot]))-1
        result,ts=advance(before,jnp.int32(rules.train_start+slot))
        assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('trainer_uses')] == 1
        assert result.unit_xp[slot]==cap and result.gold==before.gold-4*cap
        assert not compiled_method(env,'action_mask')(result)[rules.train_start+slot]
        after, ts = advance(result,jnp.int32(rules.train_start+slot))
        unchanged(result,after)
        assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('trainer_uses')] == 0
    hero=at_site(env,initial,'trainer').replace(unit_levels=initial.unit_levels.at[1].set(2))
    result,_=advance(hero,jnp.int32(rules.train_start+1))
    assert result.unit_xp[1]==649 and result.gold==hero.gold-4*649


def test_lute_discount_does_not_stack_or_discount_training(current_game):
    env,state,_,_=current_game
    items=ItemRules(dict(initial_items={'lute_of_charming':2}))
    rules=SiteRules(MAP,env.potion_rules,items,env.progression,171)
    state=state.replace(equipped=jnp.array([0,0,-1,-1,-1]),item_inventory=items.initial_inventory)
    assert jax.jit(rules.buy_prices)(state).tolist()==[630,135,360]
    assert jax.jit(rules.training_quotes)(state)[1,0]==4


def test_site_metadata_and_quotes_reach_human_service(human_service):
    game=human_service.create(42,'legions')
    assert len(game['sites']['buy'])==3 and len(game['sites']['sites'])==3
    assert 'sell' not in game['sites']
    assert game['snapshot']['state']['merchant_stock']==[1,10,10]
    assert game['snapshot']['site_quotes']['buy']==[700,150,400]
    assert len(game['snapshot']['action_mask'])==244
    assert game['snapshot']['site_quotes']['train'][1][0]==4


def test_batched_site_actions_match_scalar_steps(current_game):
    env,state,advance,_=current_game
    merchant=at_site(env,state,'merchant')
    trainer=at_site(env,state,'trainer')
    camp=hero_roster_state(env,at_site(env,state,'mercenary'),[None,'duke',None,None,None,None])
    inputs=[merchant,merchant,trainer,trainer,state,camp,camp]
    actions=jnp.array([171,172,174,175,171,185,186],jnp.int32)
    batch=jax.tree.map(lambda *xs:jnp.stack(xs),*inputs)
    output=compiled_method(env,'step',batched=True)(batch,actions)
    for i,before in enumerate(inputs):
        scalar=advance(before,actions[i])
        chex.assert_trees_all_close(jax.tree.map(lambda x,i=i:x[i],output),scalar)


def test_autosale_only_on_merchant_approaches_and_only_valuables(current_game):
    env, initial, _, _ = current_game
    state = inventory(env,initial,['bronze_ring','runestone','bronze_ring','boots_speed'])
    merchant = next(s for s in env.site_rules.sites if s['kind']=='merchant')
    r,c = merchant['position']
    positions = merchant['interaction_tiles']+[(r-1,c),(r+1,c-1),(r,c+3)]
    states = [state.replace(position=jnp.array(p)) for p in positions]
    states += [states[0].replace(in_battle=jnp.bool_(True)),
               states[0].replace(done=jnp.bool_(True)),at_site(env,state,'trainer')]
    result = jax.jit(jax.vmap(env.site_rules.autosell))(
        jax.tree.map(lambda *xs:jnp.stack(xs),*states))
    for i,before in enumerate(states):
        after = jax.tree.map(lambda x,i=i:x[i],result)
        assert after.gold == before.gold+(100 if i<5 else 0)
        assert after.last_sold_count == (2 if i<5 else 0)
        chex.assert_trees_all_equal(after.potions,before.potions)
        for key in ('runestone','boots_speed'):
            item = env.item_rules.keys.index(key)
            assert jnp.sum(after.item_inventory==item) == jnp.sum(before.item_inventory==item)
