"""Reference recruitment, leadership and roster lifecycle on CUDA."""
import jax
import jax.numpy as jnp
import pytest
import chex
from stoix.envs.number_grid import MAP, ACTIONS, ACTION_NAMES, REST, SHOOT
from stoix.envs.number_grid_buildings import BuildingRules
from stoix.envs.number_grid_recruitment import RecruitmentRules, OFFERS, HIRED, DISMISSED
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


def test_initial_roster_leadership_and_observation(current_game):
    env, state, _, _ = current_game
    assert env.map_config['hero_roster'] == ['possessed', 'duke', 'possessed', 'cultist', None, None]
    assert int(jnp.sum(state.hp[:6] > 0)) == 4
    assert ACTIONS == len(ACTION_NAMES) == env.num_actions == 244
    assert env.observation_size == 1843
    assert compiled_method(env, 'observation')(state).shape == (1843,)
    composition = jax.jit(env.recruitment.composition)
    for level, capacity in ((1,3), (2,3), (3,4), (5,4), (6,5), (20,5)):
        s = state.replace(unit_levels=state.unit_levels.at[1].set(level))
        assert composition(s).tolist() == [capacity,3,capacity-3,3]
        assert composition(s.replace(hp=s.hp.at[1].set(0))).tolist() == [capacity,3,capacity-3,3]
    s = hero_roster_state(env,state,['possessed',None,None,None,None,None])
    assert composition(s).tolist() == [0,1,0,3]
    s = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(0)))
    altered = s.replace(unit_ids=s.unit_ids.at[1].set(env.progression.ids['titan']).at[4].set(env.progression.ids['cultist']))
    chex.assert_trees_all_equal(composition(s), composition(altered))


@pytest.mark.parametrize('faction', list(OFFERS))
def test_all_faction_offers_prices_buildings_and_placement(current_game, faction):
    env, state, _, _ = current_game
    rules = RecruitmentRules(MAP, env.progression, BuildingRules({**MAP,'faction':faction}), 180, env.site_rules)
    base = hero_roster_state(env,state,[None,'duke',None,None,None,None]).replace(gold=jnp.int32(9999), hire_rewarded_capacity=jnp.int32(0))
    apply = jax.jit(lambda s,a:rules.apply(s,a,env._clear_roster_slots))
    available = jax.jit(rules.available)
    prices = {'empire':[50,60,40,50,300], 'mountain_clans':[50,60,40,400,100],
              'undead_hordes':[50,60,50,1000,100], 'legions':[50,60,80,300,100], 'elves':[50,60,40,50,300]}
    assert rules.costs[:5].tolist() == prices[faction]
    # Neutral mercenaries are available to all five factions without buildings.
    camp = base.replace(position=env.site_rules.approaches['mercenary'][0])
    assert available(camp)[5:7].tolist() == [True, True]
    for i, slot in ((5,0), (6,3)):
        hired, _ = apply(camp,jnp.int32(rules.start+i))
        assert hired.unit_ids[slot] == rules.ids[i] and hired.mercenary_stock[i-5] == 0
    for i,key in enumerate(OFFERS[faction]):
        s = base
        bit = int(rules.required[i])
        if bit >= 0:
            assert not available(s)[i]
            unchanged, reward = apply(s,jnp.int32(180+i))
            chex.assert_trees_all_equal(s,unchanged)
            assert reward == 0
            s = s.replace(buildings=jnp.uint32(1 << bit))
        assert available(s)[i]
        out,reward = apply(s,jnp.int32(180+i))
        slot = int(rules.targets(s)[i])
        assert slot == (0 if bool(rules.front[i]) or rules.sizes[i] == 2 else 3)
        assert out.last_event == HIRED and out.gold == 9999-prices[faction][i]
        assert out.unit_ids[slot] == env.progression.ids[key] and out.unit_xp[slot] == 0
        assert out.hp[slot] == env.progression.stats(out.unit_ids[slot],out.unit_levels[slot])[0]
        assert out.movement_points == s.movement_points and out.day == s.day
        assert rules.composition(out)[1] == rules.sizes[i] and reward == 3
        assert not available(s.replace(position=jnp.array([3,2])))[i]
        assert not available(s.replace(gold=jnp.int32(prices[faction][i]-1)))[i]
        assert not available(s.replace(in_battle=jnp.bool_(True)))[i]
        assert not available(s.replace(done=jnp.bool_(True)))[i]
        assert available(s.replace(hp=s.hp.at[1].set(0)))[i]


@pytest.mark.core
def test_dismiss_rehire_rewards_and_effect_cleanup(current_game):
    env, initial, advance, _ = current_game
    rules = env.recruitment
    s = initial.replace(gold=jnp.int32(1000), potion_doses=initial.potion_doses.at[0].set(1),
        potion_bonus=initial.potion_bonus.at[0].set(10), potion_active=initial.potion_active.at[0].set(7))
    out,ts = advance(s,jnp.int32(rules.dismiss_start))
    assert out.last_event == DISMISSED and out.gold == 1000 and out.hp[0] == 0 and out.unit_ids[0] == 0
    assert float(ts.reward) == pytest.approx(-.051)
    assert out.hire_rewarded_capacity == 3
    assert not jnp.any(out.potion_doses[0]) and not jnp.any(out.potion_bonus[0]) and out.potion_active[0] == 0
    hired,ts = advance(out,jnp.int32(rules.start))
    assert hired.last_event == HIRED and hired.gold == 950 and hired.hp[0] == initial.hp[0]
    counts = dict(zip(CAMPAIGN_FIELDS, ts.extras['campaign_events']))
    assert counts['hired_units'] == 1 and counts['mercenaries_hired'] == 0
    assert float(ts.reward) == pytest.approx(-.001)
    s = hired.replace(unit_levels=hired.unit_levels.at[1].set(3))
    hired,ts = advance(s,jnp.int32(rules.start+1))
    assert hired.unit_ids[4] == env.progression.ids['cultist']
    assert hired.hire_rewarded_capacity == 4 and float(ts.reward) == pytest.approx(2.999)
    s = hired.replace(unit_levels=hired.unit_levels.at[1].set(6))
    hired,ts = advance(s,jnp.int32(rules.start+1))
    assert hired.unit_ids[5] == env.progression.ids['cultist']
    assert hired.hire_rewarded_capacity == 5 and float(ts.reward) == pytest.approx(2.999)
    battle = compiled_method(env,'_begin_battle')(hired.replace(enemy=jnp.int32(0)))
    battle = battle.replace(actor=jnp.int32(0),hp=battle.hp.at[6].set(1))
    won,_ = compiled_method(env,'_battle_step')(battle,jnp.int32(SHOOT),battle.battle_key,jnp.zeros(env.random_size))
    assert not won.in_battle and jnp.all(won.hp[4:6] > 0)
    reset,_ = compiled_method(env,'reset')(jax.random.PRNGKey(1))
    assert reset.hire_rewarded_capacity == 3


@pytest.mark.core
def test_large_units_dead_slots_and_mask_step_agreement(current_game):
    env,initial,advance,_ = current_game
    rules = env.recruitment
    available = compiled_method(env,'action_mask')
    base = hero_roster_state(env,initial,[None,'duke',None,None,None,None]).replace(gold=jnp.int32(500),hire_rewarded_capacity=jnp.int32(0))
    large,ts = advance(base,jnp.int32(rules.start+2))
    assert large.unit_ids[0] == env.progression.ids['g000uu0055'] and large.unit_ids[3] == 0
    assert rules.composition(large).tolist() == [3,2,1,2]
    assert float(ts.reward) == pytest.approx(2.949)
    dead = large.replace(hp=large.hp.at[0].set(0),position=jnp.array([3,2]))
    assert rules.composition(dead)[1] == 2
    assert available(dead)[rules.dismiss_start] and available(dead)[rules.dismiss_start+3]
    out,_ = advance(dead,jnp.int32(rules.dismiss_start+3))
    assert out.unit_ids[0] == out.unit_ids[3] == 0 and out.gold == dead.gold
    crowded = hero_roster_state(env,initial,['possessed','duke','possessed',None,None,None]).replace(gold=jnp.int32(1000))
    crowded = crowded.replace(unit_levels=crowded.unit_levels.at[1].set(6),hp=crowded.hp.at[0].set(0))
    assert not available(crowded)[rules.start+2]
    cases = [(initial,rules.start),(initial,rules.dismiss_start+1),(initial,rules.dismiss_start+5),
        (dead.replace(hp=dead.hp.at[1].set(0)),rules.dismiss_start),
        (base.replace(position=jnp.array([3,2])),rules.start),
        (base.replace(gold=jnp.int32(49)),rules.start),(crowded,rules.start+2)]
    for s,action in cases:
        assert not available(s)[action]
        out,_ = advance(s,jnp.int32(action))
        for name in ('unit_ids','hp','gold','unit_levels','unit_xp','movement_points','hire_rewarded_capacity'):
            chex.assert_trees_all_equal(getattr(out,name),getattr(s,name))
    _,ts = advance(base.replace(movement_points=jnp.int32(0)),jnp.int32(REST))
    assert float(ts.reward) == pytest.approx(-.15)
