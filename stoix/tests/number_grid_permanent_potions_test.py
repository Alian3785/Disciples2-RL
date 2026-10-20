"""Extended GPU checks for the user's original-game experiment, 2026-10-17.

Each permanent potion kind is allowed once per unit per episode. Different
kinds can coexist. Source and scope: docs/MAP_POTIONS.md; run only on request.
"""
import chex
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import REST
from stoix.envs.number_grid_potions import POTIONS, POTION_START, PotionRules
from stoix.tests.number_grid_fixtures import compiled_method
from stoix.tests.number_grid_potions_test import stocked, use


def action_for(env, key, slot):
    return jnp.int32(POTION_START+6*env.potion_rules.keys.index(key)+slot)


def assert_repeat_rejected(env, state, key, slot):
    """The public mask, quote and actual step all reject the spent permanent kind."""
    action=action_for(env,key,slot)
    assert not compiled_method(env,'action_mask')(state)[action]
    quote=env.potion_rules.quotes(state,compiled_method(env,'max_hp')(state))
    assert quote[env.potion_rules.keys.index(key),slot]==0
    after,ts=compiled_method(env,'step')(state,action)
    for field in ('hp','potions','potion_doses','potion_bonus','potion_temporary',
                  'potion_active','potion_wards'):
        chex.assert_trees_all_equal(getattr(after,field),getattr(state,field))
    assert after.last_potion==-1 and float(ts.reward)<=0.
    return after


def test_each_permanent_kind_once_but_all_five_can_coexist(current_game):
    env,initial,_,_=current_game
    state=stocked(env,initial)
    before=state.potions
    for potion in (p for p in POTIONS if p['duration']=='permanent'):
        key=potion['key']
        assert compiled_method(env,'action_mask')(state)[action_for(env,key,0)]
        state=use(env,state,key,0)
        assert state.potion_doses[0,potion['stat']]==1
        assert state.potions[env.potion_rules.keys.index(key)]==before[env.potion_rules.keys.index(key)]-1
        state=assert_repeat_rejected(env,state,key,0)
        # The restriction belongs to this fighter, not the inventory or party.
        assert compiled_method(env,'action_mask')(state)[action_for(env,key,1)]
        state=use(env,state,key,1)
    assert jnp.all(state.potion_doses[:2]==1)
    assert not jnp.any(state.potion_doses[2:])
    # All five effects are present at once on the starting possessed.
    assert compiled_method(env,'unit_stats')(state)[0,:5].tolist()==[138.,28.,88.,10.,55.]


def test_lifetime_history_survives_rest_death_revival_and_swap(current_game):
    env,initial,_,_=current_game
    state=use(env,stocked(env,initial),'highfather')
    rested,_=compiled_method(env,'step')(state,jnp.int32(REST))
    assert_repeat_rejected(env,rested,'highfather',0)
    dead=rested.replace(hp=rested.hp.at[0].set(0))
    revived=use(env,dead,'life')
    assert revived.hp[0]==1
    assert_repeat_rejected(env,revived,'highfather',0)
    swap=env.unit_swaps.start+env.unit_swaps.pairs.index((0,2))
    moved,_=compiled_method(env,'step')(revived,jnp.int32(swap))
    assert moved.hp[2]==1 and moved.potion_doses[2,0]==1
    assert_repeat_rejected(env,moved,'highfather',2)
    assert compiled_method(env,'action_mask')(moved)[action_for(env,'highfather',0)]
    # A genuinely new recruit in that slot has its own unused lifetime dose.
    dismissed,_=compiled_method(env,'step')(
        moved.replace(gold=jnp.int32(1000)),jnp.int32(env.recruitment.dismiss_start+2))
    assert dismissed.unit_ids[2]==0
    assert compiled_method(env,'action_mask')(dismissed)[env.recruitment.start]
    hired,_=compiled_method(env,'step')(dismissed,jnp.int32(env.recruitment.start))
    assert hired.unit_ids[2]==initial.unit_ids[0] and not jnp.any(hired.potion_doses[2])
    assert compiled_method(env,'action_mask')(hired)[action_for(env,'highfather',2)]
    buffed=use(env,hired,'highfather',2)
    assert buffed.potion_doses[2,0]==1 and buffed.hp[2]==138


@pytest.mark.parametrize('supply', [1, 6, 100])
def test_growth_rebuild_is_independent_of_map_bottle_supply(current_game,supply):
    env,state,_,_=current_game
    # A chest alone must expose an action, but never enable multiple doses.
    rules=PotionRules(dict(initial_potions={},chests=[dict(potions={'highfather':supply})]))
    assert rules.keys==('highfather',) and rules.initial_counts.tolist()==[0]
    intrinsic=env._potion_intrinsic(state.unit_ids,state.unit_levels)
    doses=state.potion_doses.at[0,0].set(1)
    bonus=rules.rebuild(intrinsic,doses)
    assert bonus[0,0]==18 and int(jnp.count_nonzero(bonus))==1
