"""Python-reference mercenaries: finite stock, placement, economy and legality."""
import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, REST
from stoix.envs.number_grid_items import ItemRules
from stoix.envs.number_grid_sites import SiteRules
from stoix.envs.number_grid_recruitment import HIRED, RecruitmentRules
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


def camp_state(env, state):
    return hero_roster_state(env, state, [None, 'duke', None, None, None, None]).replace(
        position=env.site_rules.approaches['mercenary'][0], gold=jnp.int32(1000),
        hire_rewarded_capacity=jnp.int32(0), movement_points=jnp.int32(0))


@pytest.mark.core
def test_mercenary_hiring_stock_rewards_and_reset(current_game):
    env, initial, step, _ = current_game
    rules = env.recruitment
    assert rules.keys[5:] == ('g000uu5040', 'goblin_archer')
    assert rules.costs[5:].tolist() == [850, 50]
    assert rules.sizes[5:].tolist() == [1, 1]
    assert rules.required[5:].tolist() == [-1, -1]
    assert initial.mercenary_stock.tolist() == [1, 1]
    s = camp_state(env, initial)
    for offer, slot, hp, gold, stock, reward in ((5,0,250,150,[0,1],2.899), (6,3,40,100,[0,0],2.949)):
        before = s
        s, ts = step(s, jnp.int32(rules.start+offer))
        assert s.last_event == HIRED and s.last_target == slot
        assert s.gold == gold and s.mercenary_stock.tolist() == stock
        assert s.unit_ids[slot] == rules.ids[offer] and s.native_ids[slot] == rules.ids[offer]
        assert s.hp[slot] == hp and s.unit_levels[slot] == 1 and s.unit_xp[slot] == 0
        assert s.day == before.day and s.movement_points == 0
        assert float(ts.reward) == pytest.approx(reward)
        counts = dict(zip(CAMPAIGN_FIELDS, ts.extras['campaign_events']))
        assert counts['mercenaries_hired'] == counts['hired_units'] == 1
        assert not compiled_method(env,'action_mask')(s)[rules.start+offer]
    s, ts = step(s, jnp.int32(rules.dismiss_start))
    assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('mercenaries_hired')] == 0
    assert s.gold == 100 and s.mercenary_stock.tolist() == [0,0]
    s, _ = step(s, jnp.int32(REST))
    assert s.mercenary_stock.tolist() == [0,0]
    reset, _ = compiled_method(env,'reset')(jax.random.PRNGKey(9))
    assert reset.mercenary_stock.tolist() == [1,1]


@pytest.mark.core
def test_mercenary_mask_and_step_reject_all_unavailable_conditions(current_game):
    env, initial, step, _ = current_game
    rules = env.recruitment
    base = camp_state(env, initial)
    crowded = hero_roster_state(env, base, ['possessed','duke','possessed',None,None,None])
    crowded = crowded.replace(unit_levels=crowded.unit_levels.at[1].set(3), hp=crowded.hp.at[0].set(0))
    rear_full = hero_roster_state(env, base, ['g000uu0055','duke',None,None,'cultist','cultist'])
    rear_full = rear_full.replace(unit_levels=rear_full.unit_levels.at[1].set(6))
    cases = [(base.replace(position=initial.position),5),
             (base.replace(hp=base.hp.at[1].set(0)),5),
             (base.replace(gold=jnp.int32(849)),5),
             (base.replace(gold=jnp.int32(49)),6),
             (base.replace(mercenary_stock=jnp.zeros(2,jnp.int32)),6),
             (initial.replace(position=base.position,gold=base.gold),6),
             (crowded,5), (rear_full,6)]
    for before, offer in cases:
        action = jnp.int32(rules.start+offer)
        assert not compiled_method(env,'action_mask')(before)[action]
        after, ts = step(before, action)
        assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('mercenaries_hired')] == 0
        for key in ('unit_ids','native_ids','hp','unit_xp','gold','mercenary_stock','movement_points','day','hire_rewarded_capacity'):
            chex.assert_trees_all_equal(getattr(before,key),getattr(after,key))
    available = jax.jit(rules.available)
    assert not jnp.any(available(base.replace(in_battle=jnp.bool_(True)))[5:7])
    assert not jnp.any(available(base.replace(done=jnp.bool_(True)))[5:7])
    r,c = next(s['position'] for s in env.site_rules.sites if s['kind']=='mercenary')
    for p in [(r+i,c+j) for i in range(-1,5) for j in range(-1,5)]:
        expected = p in {(r+1,c+3),(r+2,c+3),(r+3,c+1),(r+3,c+2),(r+3,c+3)}
        assert bool(jnp.all(available(base.replace(position=jnp.array(p)))[5:7])) == expected


def test_mercenary_lute_discount_nonstacking_and_observable_quotes(current_game):
    env, initial, _, _ = current_game
    items = ItemRules(dict(initial_items={'lute_of_charming':2}))
    sites = SiteRules(MAP,env.potion_rules,items,env.progression,171)
    rules = RecruitmentRules(MAP,env.progression,env.construction,180,sites)
    lute = items.keys.index('lute_of_charming')
    base = camp_state(env,initial).replace(equipped=jnp.array([lute,lute,-1,-1,-1]),
        item_inventory=jnp.array([lute,lute]+[-1]*(env.item_rules.capacity-2)), gold=jnp.int32(765))
    quotes = jax.jit(rules.quotes)(base)
    assert quotes['prices'].tolist() == [50,60,80,300,100,765,45]
    assert jax.jit(rules.available)(base)[5]
    assert not jax.jit(rules.available)(base.replace(gold=jnp.int32(764)))[5]
    hired,_ = jax.jit(lambda s,a:rules.apply(s,a,env._clear_roster_slots))(base,jnp.int32(rules.start+5))
    assert hired.gold == 0 and hired.last_service_cost == 765
    before = compiled_method(env,'observation')(base)
    after = compiled_method(env,'observation')(base.replace(mercenary_stock=jnp.zeros(2,jnp.int32)))
    assert before.shape == (1843,) and jnp.all(jnp.isfinite(before))
    assert not jnp.array_equal(before,after)


def test_mercenary_metadata_for_manual_game(human_service):
    game = human_service.create(42,'legions')
    site = next(s for s in game['sites']['sites'] if s['kind']=='mercenary')
    assert site['position'] == [6,6] and len(site['footprint']) == 9
    assert len(site['interaction_tiles']) == 5
    offers = [o for o in game['recruitment']['offers'] if o['mercenary']]
    assert [o['action'] for o in offers] == [185,186]
    assert [o['row'] for o in offers] == ['front','back']
    assert game['snapshot']['state']['mercenary_stock'] == [1,1]
    assert game['snapshot']['recruitment_quotes']['prices'][5:] == [850,50]
