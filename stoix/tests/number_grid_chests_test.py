"""Mixed five-bottle chests, one-shot adjacent pickup and compact inventory."""
from stoix.tests.number_grid_fixtures import compiled_method
import chex
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, DIRECTIONS, REST
from stoix.envs.number_grid_chests import ChestRules, CHEST_LOOT
from stoix.envs.number_grid_potions import POTION_START


def test_chest_loot_and_observation_contract(current_game):
    env,state,_,_=current_game
    assert state.chest_alive.tolist()==[True]*5
    assert [sum(c['potions'].values()) for c in MAP['chests']]==[5]*5
    assert jnp.all(jnp.sum(env.chest_rules.loot,axis=1)==5)
    assert int(jnp.sum(env.chest_rules.loot))==25
    start=220+5*env.num_opponents
    width=env.potion_rules.count+3
    end=start+env.chest_rules.count*width
    encoded=compiled_method(env,'observation')(state)[start:end].reshape(5,width)
    chex.assert_trees_all_close(encoded[:,:2],env.chest_rules.positions/47)
    chex.assert_trees_all_equal(encoded[:,2:-1],env.chest_rules.loot)
    chex.assert_trees_all_equal(encoded[:,-1],state.chest_alive)
    collected=state.replace(chest_alive=jnp.zeros(5,bool))
    assert not jnp.any(compiled_method(env,'observation')(collected)[start:end].reshape(5,width)[:,-1])
    for chests in (None,[None],[dict(position=[0,4],potions={'life':1})],
                   [dict(position=MAP['agent_position'],potions={'life':1})],
                   [MAP['chests'][0]]*2,[dict(position=[5,4],potions={})],
                   [dict(position=[5,4],potions={'life':-1})],
                   [dict(position=[5,4],potions={'life':2**31-1})]):
        with pytest.raises(ValueError,match='[Cc]hest'):
            ChestRules({**MAP,'chests':chests})


def test_eight_adjacent_tiles_collect_all_contents_once(current_game):
    env,state,advance,_=current_game
    chest=MAP['chests'][4]['position']
    loot=env.chest_rules.loot[4]
    for direction,delta in enumerate(DIRECTIONS):
        before=state.replace(position=jnp.array([chest[i]+2*delta[i] for i in range(2)]))
        action=jnp.int32((direction+4)%8)
        picked,ts=advance(before,action)
        assert picked.last_event==CHEST_LOOT and not picked.chest_alive[4]
        chex.assert_trees_all_equal(picked.potions,state.potions+loot)
        chex.assert_trees_all_equal(picked.last_loot,loot)
        assert picked.movement_points==30 and picked.map_steps==1 and picked.day==1
        assert float(ts.reward)==pytest.approx(env.exploration_bonus-env.step_cost)
        chex.assert_trees_all_equal(picked.hp,state.hp)
        repeated,_=advance(picked,action)
        chex.assert_trees_all_equal(repeated.potions,picked.potions)
        assert not jnp.any(repeated.last_loot)


def test_pickup_makes_new_type_usable_and_nonmovement_does_not_collect(current_game):
    env,state,advance,_=current_game
    beside=state.replace(position=jnp.array([5,2]))
    picked,_=advance(beside,jnp.int32(2))
    kind=env.potion_rules.keys.index('might')
    action=POTION_START+6*kind
    assert picked.potions[kind]==1 and compiled_method(env,'action_mask')(picked)[action]
    used,_=advance(picked,jnp.int32(action))
    assert used.potions[kind]==0 and compiled_method(env,'unit_stats')(used)[0,1]==38
    for action in (REST,-1,POTION_START):
        before=state.replace(position=jnp.array([5,3]),hp=state.hp.at[0].set(1))
        after,_=advance(before,jnp.int32(action))
        chex.assert_trees_all_equal(after.chest_alive,before.chest_alive)
        assert not jnp.any(after.last_loot)
