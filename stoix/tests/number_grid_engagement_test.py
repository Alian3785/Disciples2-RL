"""Enemy-tile attack regressions from the current Python reference; CUDA only."""
from collections import OrderedDict
from stoix.tests.number_grid_fixtures import compiled_method
import threading

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, MAP, SHOOT, CONTINUE, REST, ENGAGE, VICTORY, WITHDRAW
from stoix.envs.number_grid_potions import POTION_START


def batch(state, count):
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (count,)+x.shape), state)


@pytest.mark.core
def test_passing_beside_enemy_and_map_services_do_not_start_combat(current_game):
    env, initial, advance, _ = current_game
    state = initial.replace(position=jnp.array([2,6]), hp=initial.hp.at[0].set(60))
    for col in (7,8,9):
        state, ts = advance(state,jnp.int32(2))
        assert not state.in_battle and state.enemy == initial.enemy
        chex.assert_trees_all_equal(state.position,jnp.array([2,col]))
        chex.assert_trees_all_equal(state.battle_key,initial.battle_key)
        chex.assert_trees_all_equal(state.alive,initial.alive)
        assert float(ts.reward) == pytest.approx(env.exploration_bonus-env.step_cost)
    # Rest and a potion alongside the enemy are still ordinary map actions.
    state, _ = advance(state,jnp.int32(POTION_START))
    assert not state.in_battle and state.hp[0] == 110
    state, _ = advance(state,jnp.int32(REST))
    assert not state.in_battle and state.hp[0] == 120
    assert state.map_steps == 3 and state.day == 2 and state.movement_points == 20
    chex.assert_trees_all_equal(state.battle_key,initial.battle_key)


def test_attack_all_eight_directions_costs_half_cap_clamped_to_positive_remainder(current_game):
    env, initial, _, _ = current_game
    budgets=jnp.array([0,1,2,9,10,11,20],jnp.int32)
    count=len(budgets)*8
    actions=jnp.tile(jnp.arange(8,dtype=jnp.int32),len(budgets))
    states=batch(initial,count).replace(
        position=env.opponent_positions[0]-env.directions[actions],
        movement_points=jnp.repeat(budgets,8), map_steps=jnp.full(count,7,jnp.int32))
    masks=compiled_method(env,'action_mask',batched=True)(states)
    chex.assert_trees_all_equal(masks[jnp.arange(count),actions],states.movement_points>0)
    following,ts=compiled_method(env,'step',batched=True)(states,actions)
    chex.assert_trees_all_equal(following.in_battle,states.movement_points>0)
    chex.assert_trees_all_equal(following.position,states.position)
    chex.assert_trees_all_equal(following.origin[8:],states.position[8:])
    chex.assert_trees_all_equal(following.movement_points,jnp.maximum(0,states.movement_points-10))
    chex.assert_trees_all_equal(following.enemy[8:],jnp.zeros(count-8,jnp.int32))
    chex.assert_trees_all_equal(following.last_event[8:],jnp.full(count-8,ENGAGE,jnp.int32))
    chex.assert_trees_all_close(ts.reward,jnp.full(count,-env.step_cost))
    for field in ('map_steps','visited','gold','day','built_today','buildings','potions','alive'):
        chex.assert_trees_all_equal(getattr(following,field),getattr(states,field))
    chex.assert_trees_all_equal(following.hp[:8],states.hp[:8])
    chex.assert_trees_all_equal(following.battle_key[:8],states.battle_key[:8])
    assert jnp.all(jnp.any(following.battle_key[8:]!=states.battle_key[8:],axis=1))
    chex.assert_trees_all_equal(following.hp[8:,:6],states.hp[8:,:6])
    # Last partial movement consumes the remaining point, as in the reference.
    low=initial.replace(movement_points=jnp.int32(1))
    chex.assert_trees_all_equal(compiled_method(env,'action_mask')(low)[:8],
        compiled_method(env,'action_mask')(initial)[:8])
    stopped,_=compiled_method(env,'step')(low,jnp.int32(2))
    chex.assert_trees_all_equal(stopped.position,low.position+jnp.array([0,1]))
    assert stopped.movement_points == 0
    wall=initial.replace(position=jnp.array([1,1]))
    assert not compiled_method(env,'action_mask')(wall)[0]
    stopped,_=compiled_method(env,'step')(wall,jnp.int32(0))
    assert not stopped.in_battle and stopped.movement_points==20


def test_command_selects_exact_stack_when_two_targets_are_adjacent():
    positions=[list(p) for p in MAP['opponent_positions']]
    positions[1]=[3,9]
    # Target selection is independent of attack effects; keep the formation and
    # current map contract without recompiling every special combat branch.
    rosters=[[('squire' if i < 3 else 'archer') if key else None
              for i,key in enumerate(row)] for row in MAP['enemy_rosters']]
    env=NumberGrid(map_config={**MAP,'opponent_positions':positions,'enemy_rosters':rosters})
    state,_=compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state=state.replace(position=jnp.array([2,8]))
    commands=env.map_commands(state)
    assert commands[4,0]==0 and commands[3,0]==1
    assert commands[4,1]==commands[3,1]==10
    assert jnp.all(compiled_method(env,'action_mask')(state)[jnp.array([4,3])])
    # Public step/mask coupling is exercised for all directions above. Here
    # isolate destination selection and battle initialization from combat turns.
    states=batch(state,2)
    next_states,_=compiled_method(env,'_world_step',batched=True)(states,
        jnp.array([4,3],jnp.int32),states.battle_key,jnp.zeros((2,env.random_size)))
    chex.assert_trees_all_equal(next_states.enemy,jnp.array([0,1]))
    chex.assert_trees_all_equal(next_states.position,jnp.array([[2,8],[2,8]]))
    chex.assert_trees_all_equal(next_states.unit_ids[:,6:],env.progression.enemy_ids[:2])


@pytest.mark.core
def test_victory_and_retreat_stay_at_origin_and_cleared_tile_needs_separate_move(current_game):
    env,initial,advance,battle_step=current_game
    ready=initial.replace(position=jnp.array([3,7]))
    attack,_=advance(ready,jnp.int32(2))
    assert attack.in_battle and attack.movement_points==10
    won,_=battle_step(attack.replace(actor=jnp.int32(0),hp=attack.hp.at[6].set(1)),
                     jnp.int32(SHOOT),attack.battle_key,jnp.zeros(env.random_size))
    assert won.last_event==VICTORY and not won.in_battle and not won.alive[0]
    assert won.movement_points==10 and won.map_steps==0
    chex.assert_trees_all_equal(won.position,ready.position)
    entered,_=advance(won,jnp.int32(2))
    chex.assert_trees_all_equal(entered.position,env.opponent_positions[0])
    assert not entered.in_battle and entered.movement_points==8 and entered.map_steps==1
    escaping=attack.replace(actor=jnp.int32(0),hp=attack.hp.at[1:6].set(0),
        retreating=attack.retreating.at[0].set(True))
    escaped,_=battle_step(escaping,jnp.int32(CONTINUE),escaping.battle_key,jnp.zeros(env.random_size))
    assert escaped.last_event==WITHDRAW and not escaped.in_battle and escaped.alive[0]
    chex.assert_trees_all_equal(escaped.position,ready.position)
    assert escaped.movement_points==10 and escaped.map_steps==0
    passed,_=advance(escaped,jnp.int32(0))
    assert not passed.in_battle and passed.movement_points==6  # Dead Duke: plain costs 4.
    chex.assert_trees_all_equal(passed.position,jnp.array([2,7]))


def test_human_server_uses_jax_target_and_cost_and_does_not_auto_attack_neighbours(current_game):
    from serve_number_grid import GameService
    env,initial,advance,_=current_game
    service=GameService.__new__(GameService)
    service.lock=threading.Lock()
    key=(env.construction.faction,env.construction.lord['id'])
    service.environments={key:(env,compiled_method(env,'reset'),advance)}
    service.sessions=OrderedDict({'attack':(key,initial.replace(position=jnp.array([2,6])),0.)})
    result=service.act('attack',2)['snapshot']
    assert result['state']['position']==[2,7] and not result['state']['in_battle']
    assert result['map_commands'][3]==[0,10] and result['action_mask'][3]
    attacked=service.act('attack',3)
    assert attacked['events'][0]['state']['last_event']==ENGAGE
    assert attacked['snapshot']['state']['in_battle']
    assert attacked['snapshot']['state']['enemy']==0
    assert attacked['snapshot']['state']['position']==[2,7]
    assert attacked['snapshot']['state']['movement_points']==8


def test_expanded_map_edges_and_exploration_bits(current_game):
    env, initial, advance, _ = current_game
    assert env.size == 48 and initial.visited.shape == (72,)
    state = initial.replace(position=jnp.array([45,45]))
    state, _ = advance(state, jnp.int32(2))
    chex.assert_trees_all_equal(state.position, jnp.array([45,46]))
    state, _ = advance(state, jnp.int32(4))
    chex.assert_trees_all_equal(state.position, jnp.array([46,46]))
    assert state.movement_points == 16 and state.map_steps == 2
    cell = 46*48+46
    assert state.visited[cell//32] & jnp.uint32(1 << (cell % 32))
    assert not compiled_method(env,'action_mask')(state)[2] and not compiled_method(env,'action_mask')(state)[4]
    blocked, _ = advance(state, jnp.int32(3))
    chex.assert_trees_all_equal(blocked.position, state.position)
    chex.assert_trees_all_equal(blocked.visited, state.visited)
    assert blocked.movement_points == 16 and not blocked.in_battle
    chex.assert_trees_all_close(compiled_method(env,'observation')(state)[:2], jnp.full(2,46/47))
