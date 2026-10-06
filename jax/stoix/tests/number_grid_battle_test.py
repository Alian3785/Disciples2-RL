"""Combat rules, episode boundaries and batched GPU integration."""
import dataclasses
import json
from collections import deque
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from stoa import AddActionMaskWrapper
from numbergrid_config import make_config
from stoix.utils.make_env import make
from stoix.envs.number_grid import (
    MAP, NumberGrid, wrap_wall_action_mask, SHOOT, DEFEND, WAIT, RETREAT, CONTINUE,
    ENGAGE, HIT, MISS, VICTORY, DEFEAT, WITHDRAW, LIMIT,
)


def fixture(**overrides):
    env = NumberGrid(map_config={**MAP, **overrides})
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(1)))
    # Explicit priorities make rule tests independent of random initiative ties.
    state = state.replace(actor=jnp.int32(0), priority=jnp.linspace(2., 1., 12))
    return env, state, jax.jit(env.step)


def assert_equal_state(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_default_inventory_and_observation():
    env = NumberGrid()
    state, ts = env.reset(jax.random.PRNGKey(3))
    assert state.hp[:6].tolist() == [45] * 6
    assert state.hp[6:].tolist() == [0] * 6
    assert env.action_space().num_values == 18
    assert ts.observation.shape == env.observation_space().shape == (178,)
    assert env.num_opponents == 24
    assert env.mage_slot == 5 and env.mage_damage == 20
    positions = [tuple(p) for p in MAP['opponent_positions']]
    assert len(set(positions)) == 24
    assert all(0 < r < env.size-1 and 0 < c < env.size-1 for r,c in positions)
    assert tuple(MAP['agent_position']) not in positions
    assert set(env.state_space().spaces) == {f.name for f in dataclasses.fields(state)}
    for count, hp in zip(MAP['enemy_units'], MAP['enemy_hp']):
        assert count <= 6 and hp <= 45 and (count < 6 or hp < 45)
    assert np.isfinite(ts.observation).all()
    assert not np.any(env.action_mask(state)[8:])


def test_compact_observation_keeps_queue_status_and_old_checkpoints_supported():
    env, state, _ = fixture()
    state = state.replace(turn_phase=state.turn_phase.at[1].set(1).at[2].set(2),
                          retreating=state.retreating.at[3].set(True),
                          escaped=state.escaped.at[4].set(True))
    offset = 4 + 5 * env.num_opponents
    units = np.asarray(env.observation(state))[offset:offset+48].reshape(12,4)
    assert units[0,1] > 0 and units[1,1] < 0 and units[2,1] == 0
    assert units[3,3] == .5 and units[4,3] == 1 and units[4,1] == 0
    legacy = NumberGrid(map_config={**MAP,'battle_observation_version':1})
    assert legacy.observation(state).shape == (250,)
    saved_map = json.loads((Path(__file__).resolve().parents[2] / 'maps/number_grid-24x24-v6-12-squads.json').read_text())
    for version, size in [(1,178), (2,118)]:
        saved_env = NumberGrid(map_config={**saved_map,'battle_observation_version':version})
        assert saved_env.mage_slot == -1
        _, saved_ts = saved_env.reset(jax.random.PRNGKey(0))
        assert saved_ts.observation.shape == (size,)


def test_fixed_map_has_a_playable_full_route_with_archer_battles():
    # Planning is a preparation/test operation, never called by the PPO learner.
    from stoix.envs.number_grid import DIRECTIONS
    # Prepare a route through actual battle contacts, including all added squads.
    # The CPU search is test-only and supplies no actions to PPO.
    position = tuple(MAP['agent_position'])
    opponents = [tuple(p) for p in MAP['opponent_positions']]
    alive = set(range(len(opponents)))
    route = []
    while alive:
        queue, seen, found = deque([(position, [])]), {position}, None
        occupied = {opponents[i] for i in alive}
        while queue and found is None:
            pos, path = queue.popleft()
            for action, (dr,dc) in enumerate(DIRECTIONS):
                dest = (pos[0]+dr,pos[1]+dc)
                if not all(0 < x < MAP['size']-1 for x in dest) or dest in occupied:
                    continue
                nearby = [i for i in sorted(alive) if max(abs(opponents[i][0]-dest[0]),abs(opponents[i][1]-dest[1])) == 1]
                if nearby:
                    found = (dest, path+[action], nearby[0])
                    break
                if dest not in seen:
                    seen.add(dest)
                    queue.append((dest,path+[action]))
        assert found is not None, 'An enemy cannot be reached'
        position, path, enemy = found
        route.extend(path)
        alive.remove(enemy)
    route = jnp.array(route,jnp.int32)
    env = NumberGrid()
    states,_ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(875),64))
    def run(states):
        def condition(carry):
            return ~jnp.all(carry[0].done)
        def step(carry):
            states,index = carry
            targets = jnp.argmin(jnp.where(states.hp[:,6:]>0,states.hp[:,6:],10000),axis=-1)
            actions = jnp.where(states.in_battle,
                                jnp.where(states.actor<6,SHOOT+targets,CONTINUE),
                                route[jnp.minimum(index,len(route)-1)])
            index += (~states.in_battle & ~states.done).astype(jnp.int32)
            states,_ = jax.vmap(env.step)(states,actions)
            return states,index
        return jax.lax.while_loop(condition,step,(states,jnp.zeros(64,jnp.int32)))
    states,index = jax.jit(run)(states)
    assert int(jnp.sum(states.won)) > 0
    assert np.all(np.asarray(index)[np.asarray(states.won)] == len(route))
    assert np.all(np.asarray(states.hp[:,:6])[np.asarray(states.won)] == 45)


@pytest.mark.parametrize('count,hp', [(6,45), (7,20), (0,20), (2,46), (2,0)])
def test_invalid_enemy_strength_rejected(count, hp):
    counts, health = list(MAP['enemy_units']), list(MAP['enemy_hp'])
    counts[0], health[0] = count, hp
    with pytest.raises(ValueError, match='strictly weaker'):
        NumberGrid(map_config={**MAP, 'enemy_units':counts, 'enemy_hp':health})


def test_diagonal_contact_ignores_numbers_and_selects_one_enemy():
    positions = [[4,4], [4,5]] + MAP['opponent_positions'][2:]
    env = NumberGrid(map_config={**MAP, 'opponent_positions':positions,
                                'opponent_numbers':[12]*len(positions)})
    state, _ = env.reset(jax.random.PRNGKey(0))
    next_state, ts = jax.jit(env.step)(state, jnp.int32(3))  # [2,2] -> [3,3]
    assert next_state.in_battle and not next_state.lost
    assert next_state.enemy == 0 and next_state.last_event == ENGAGE
    assert np.all(next_state.alive) and next_state.number == state.number
    assert next_state.hp[6] == 20 and np.sum(next_state.hp[6:] > 0) == 1
    assert float(ts.reward) == pytest.approx(.009)
    np.testing.assert_array_equal(next_state.origin, [2,2])
    # Occupied destinations are masked before movement.
    near = state.replace(position=jnp.array([3,4]))
    assert not env.action_mask(near)[4]


def test_jitted_reset_and_initiative_are_seeded():
    env, state, _ = fixture()
    start, _ = env.reset(jax.random.PRNGKey(42))
    a = jax.jit(env._begin_battle)(start.replace(enemy=jnp.int32(1)))
    b = jax.jit(env._begin_battle)(start.replace(enemy=jnp.int32(1)))
    assert_equal_state(a, b)
    c = env._begin_battle(start.replace(enemy=jnp.int32(1), battle_key=jax.random.PRNGKey(43)))
    assert not np.array_equal(a.priority, c.priority)
    assert a.actor == jnp.argmax(jnp.where(a.hp > 0, a.priority, -100))


@pytest.mark.parametrize('accuracy,event,remaining', [(1.,HIT,5), (0.,MISS,30)])
def test_rear_archer_shoots_rear_target_hit_or_miss(accuracy, event, remaining):
    env, state, advance = fixture(archer_accuracy=accuracy)
    hp = state.hp.at[10].set(30)
    state = state.replace(actor=jnp.int32(4), hp=hp)
    assert env.action_mask(state)[SHOOT+4]
    result, _ = advance(state, jnp.int32(SHOOT+4))
    assert result.hp[10] == remaining and result.last_event == event
    assert result.turn_phase[4] == 2


def test_mage_hits_all_six_enemies_independently_of_selected_target():
    env, state, _ = fixture(archer_accuracy=1.)
    state = state.replace(actor=jnp.int32(5), hp=state.hp.at[6:].set(35))
    results, timesteps = jax.jit(jax.vmap(env.step, in_axes=(None, 0)))(
        state, jnp.arange(SHOOT, DEFEND, dtype=jnp.int32))
    np.testing.assert_array_equal(results.hp[:, :6], np.full((6,6), 45))
    np.testing.assert_array_equal(results.hp[:, 6:], np.full((6,6), 15))
    assert np.all(results.last_damage == 120) and np.all(results.last_target == -1)
    assert np.all(results.last_event == HIT) and np.all(results.turn_phase[:,5] == 2)
    assert np.all(results.battle_steps == state.battle_steps+1)
    assert np.all(timesteps.extras['player_battle_transition'])
    for index in range(1,6):
        assert_equal_state(jax.tree.map(lambda x: x[0], results),
                           jax.tree.map(lambda x, index=index: x[index], results))


def test_mage_damage_respects_defence_dead_escaped_and_low_hp_targets():
    env, state, advance = fixture(archer_accuracy=1.)
    state = state.replace(actor=jnp.int32(5),
                          hp=state.hp.at[6:].set(jnp.array([35,25,15,0,10,35])),
                          defended=state.defended.at[7].set(True).at[11].set(True),
                          escaped=state.escaped.at[10].set(True))
    result, _ = advance(state, jnp.int32(SHOOT))
    np.testing.assert_array_equal(result.hp[6:], [15,15,0,0,10,25])
    np.testing.assert_array_equal(result.hp[:6], state.hp[:6])
    assert result.last_damage == 55
    assert not env.action_mask(result)[SHOOT+2] and result.actor != 8


def test_mage_spell_misses_all_targets_and_consumes_one_turn():
    env, state, advance = fixture(archer_accuracy=0.)
    state = state.replace(actor=jnp.int32(5))
    result, _ = advance(state, jnp.int32(SHOOT))
    np.testing.assert_array_equal(result.hp, state.hp)
    assert result.last_event == MISS and result.last_damage == 0
    assert result.turn_phase[5] == 2 and result.player_turns == state.player_turns+1
    assert not np.array_equal(result.battle_key, state.battle_key)


@pytest.mark.parametrize('action', [DEFEND, WAIT, RETREAT])
def test_mage_non_attacks_do_not_cast(action):
    _, state, advance = fixture(archer_accuracy=1.)
    state = state.replace(actor=jnp.int32(5))
    result, _ = advance(state, jnp.int32(action))
    np.testing.assert_array_equal(result.hp, state.hp)
    assert result.last_damage == 0 and result.last_target == -1


def test_mage_can_end_battle_with_multiple_kills_and_restore_the_party():
    _, state, advance = fixture(archer_accuracy=1.)
    state = state.replace(actor=jnp.int32(5), hp=state.hp.at[0].set(0).at[5].set(1)
                          .at[6:].set(jnp.array([20,15,10,5,1,0])),
                          alive=jnp.arange(len(state.alive))==1)
    result, ts = advance(state, jnp.int32(SHOOT+3))
    assert result.last_event == VICTORY and result.won and result.done
    assert not result.in_battle and not np.any(result.alive)
    assert result.last_damage == 51 and result.last_target == -1
    np.testing.assert_array_equal(result.hp[:6], [45]*6)
    assert ts.extras['battle_victory'] and float(ts.reward) == pytest.approx(3.999)


def test_enemy_sixth_archer_still_shoots_one_hero_for_25():
    _, state, advance = fixture(archer_accuracy=1.)
    state = state.replace(actor=jnp.int32(11), hp=state.hp.at[11].set(35))
    result, _ = advance(state, jnp.int32(CONTINUE))
    assert np.sum(np.asarray(result.hp[:6]) != 45) == 1
    assert result.last_damage == 25 and 0 <= result.last_target < 6
    np.testing.assert_array_equal(result.hp[6:], state.hp[6:])


def test_saved_all_archer_map_preserves_sixth_archer_single_target_attack():
    old_map = json.loads((Path(__file__).resolve().parents[2] /
                         'maps/number_grid-24x24-v7-24-archer-squads.json').read_text())
    env = NumberGrid(map_config={**old_map, 'archer_accuracy':1.})
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(1)))
    advance = jax.jit(env.step)
    state = state.replace(actor=jnp.int32(5), hp=state.hp.at[6:].set(35))
    result, _ = advance(state, jnp.int32(SHOOT+4))
    assert env.mage_slot == -1
    np.testing.assert_array_equal(result.hp[6:], [35,35,35,35,10,35])
    assert result.last_damage == 25 and result.last_target == 10


@pytest.mark.parametrize('slot,damage', [(6,20), (-2,20), (True,20), (5,0), (5,1.5)])
def test_invalid_mage_configuration_rejected(slot, damage):
    with pytest.raises(ValueError, match='mage'):
        NumberGrid(map_config={**MAP, 'hero_mage_slot':slot, 'mage_damage':damage})


def test_defence_halves_damage_and_expires_at_next_turn():
    env, state, advance = fixture(archer_accuracy=1.)
    priority = jnp.array([2.,1.0,1.0,1.0,1.0,1.0,1.9,1.8,1.,1.,1.,1.])
    state = state.replace(priority=priority, hp=state.hp.at[1:6].set(0))
    result, _ = advance(state, jnp.int32(DEFEND))
    assert result.defended[0] and result.actor == 6
    result, _ = advance(result, jnp.int32(CONTINUE))
    assert result.hp[0] == 32 and result.defended[0]
    # Every remaining actor defends; whenever 0 is scheduled the stance has expired.
    for _ in range(15):
        if int(result.actor) == 0:
            break
        result, _ = advance(result, jnp.int32(CONTINUE if int(result.actor)>=6 else DEFEND))
    assert result.actor == 0 and not result.defended[0]


def test_waiting_reverses_order_and_cannot_repeat():
    env, state, advance = fixture(archer_accuracy=0.)
    hp = jnp.zeros(12,jnp.int32).at[0].set(45).at[1].set(45).at[6].set(25)
    state = state.replace(hp=hp)
    state, _ = advance(state,jnp.int32(WAIT))
    assert state.actor == 1 and state.turn_phase[0] == 1
    state, _ = advance(state,jnp.int32(WAIT))
    assert state.actor == 6
    state, _ = advance(state,jnp.int32(CONTINUE))
    assert state.actor == 1 and state.round == 1
    assert not env.action_mask(state)[WAIT]
    state, _ = advance(state,jnp.int32(DEFEND))
    assert state.actor == 0 and state.round == 1
    state, _ = advance(state,jnp.int32(DEFEND))
    assert state.round == 2 and np.all(state.turn_phase == 0)


def test_killed_units_never_get_a_turn_or_become_targets():
    env, state, advance = fixture(archer_accuracy=1.)
    state = state.replace(priority=state.priority.at[6].set(1.99))
    result, _ = advance(state, jnp.int32(SHOOT))
    assert result.hp[6] == 0 and result.actor != 6
    assert not env.action_mask(result)[SHOOT]
    assert result.actor == 1


def test_enemy_targets_killable_unescaped_units_and_uses_only_continue():
    env, state, advance = fixture(archer_accuracy=1.)
    state = state.replace(actor=jnp.int32(6), hp=state.hp.at[0].set(5).at[1].set(20),
                          escaped=state.escaped.at[0].set(True))
    np.testing.assert_array_equal(np.flatnonzero(env.action_mask(state)), [CONTINUE])
    result, _ = advance(state, jnp.int32(CONTINUE))
    assert result.last_target == 1 and result.hp[1] == 0 and result.hp[0] == 5


def test_retreat_is_delayed_then_restores_the_entire_party():
    env, state, advance = fixture(archer_accuracy=0.)
    state = state.replace(hp=state.hp.at[1:6].set(0), origin=jnp.array([2,2]), position=jnp.array([3,3]))
    state, _ = advance(state, jnp.int32(RETREAT))
    assert state.retreating[0] and not state.escaped[0] and state.in_battle
    for _ in range(15):
        if not bool(state.in_battle):
            break
        state, ts = advance(state, jnp.int32(CONTINUE))
    assert not state.in_battle and not state.done and state.last_event == WITHDRAW
    assert np.all(state.alive)
    np.testing.assert_array_equal(state.hp[:6], [45]*6)
    np.testing.assert_array_equal(state.position,[2,2])
    assert not np.any(state.escaped | state.retreating | state.defended)
    # An invalid combat command on the map cannot immediately restart the fight.
    after, _ = advance(state, jnp.int32(CONTINUE))
    assert not after.in_battle


def test_retreat_can_be_killed_before_escape():
    env, state, advance = fixture(archer_accuracy=1.)
    state = state.replace(hp=state.hp.at[:6].set(0).at[0].set(20))
    state, _ = advance(state,jnp.int32(RETREAT))
    state, ts = advance(state,jnp.int32(CONTINUE))
    assert state.lost and state.done and state.last_event == DEFEAT
    assert ts.terminated() and float(ts.reward) == pytest.approx(-1.001)
    assert not np.any(state.hp[:6])


def test_victory_removes_only_current_enemy_and_restores_casualties():
    env, state, advance = fixture(archer_accuracy=1.)
    hp = state.hp.at[1:6].set(0).at[0].set(2).at[6:].set(0).at[6].set(20)
    state = state.replace(hp=hp)
    result, ts = advance(state,jnp.int32(SHOOT))
    assert result.last_event == VICTORY and not result.in_battle and not result.done
    assert not result.alive[1] and int(jnp.sum(result.alive)) == env.num_opponents-1
    np.testing.assert_array_equal(result.hp[:6], [45]*6)
    assert result.number == state.number+1 and float(ts.reward) == pytest.approx(.999)
    assert ts.extras['battle_victory']
    # Clear the last map opponent: terminal success takes precedence over timeout.
    final = state.replace(alive=jnp.arange(env.num_opponents)==1, step_count=jnp.int32(env.max_steps-1))
    result, ts = advance(final,jnp.int32(SHOOT))
    assert result.won and ts.terminated() and float(ts.reward) == pytest.approx(3.999)
    absorbed, ts = advance(result,jnp.int32(CONTINUE))
    assert_equal_state(absorbed,result)
    assert float(ts.reward) == 0


def test_round_limit_is_truncation_without_victory_reward():
    env, state, advance = fixture(battle_max_rounds=1, archer_accuracy=0.)
    for _ in range(12):
        if state.done:
            break
        state, ts = advance(state,jnp.int32(DEFEND if int(state.actor)<6 else CONTINUE))
    assert state.done and not state.won and not state.lost
    assert state.last_event == LIMIT and ts.truncated() and ts.discount == 1
    assert float(ts.reward) == pytest.approx(-.001)


def test_invalid_actions_do_not_shoot_or_change_random_stream():
    env, state, advance = fixture()
    for action in [-1, 0, 13, 99]:
        result, ts = advance(state,jnp.int32(action))
        np.testing.assert_array_equal(result.hp,state.hp)
        np.testing.assert_array_equal(result.battle_key,state.battle_key)
        assert result.actor == state.actor and result.step_count == state.step_count+1
        assert float(ts.reward) == pytest.approx(-.001)


def test_cached_autoreset_changes_rng_and_preserves_final_observation():
    config = make_config()
    config.env.kwargs.max_steps = 1
    env, eval_env = make(config)
    assert isinstance(eval_env, AddActionMaskWrapper)
    state, ts = env.reset(jax.random.split(jax.random.PRNGKey(2),2))
    original = np.asarray(state.battle_key)
    advance = jax.jit(env.step)
    state, ts = advance(state,jnp.array([7,3]))
    assert np.all(ts.truncated()) and np.all(state.step_count == 0)
    assert not np.array_equal(original,state.battle_key)
    assert np.all(ts.extras['next_obs']['observation'][:,3] == 1)
    assert np.all(ts.observation['observation'][:,3] == 0)
    np.testing.assert_array_equal(ts.observation['action_mask'][:,8:],False)
    again, _ = advance(state,jnp.array([7,3]))
    assert not np.array_equal(state.battle_key,again.battle_key)


def test_batched_random_battle_rollouts_have_valid_actions_and_finite_state():
    env, state, _ = fixture()
    start = jax.vmap(lambda k: env._begin_battle(state.replace(battle_key=k)))(
        jax.random.split(jax.random.PRNGKey(1),64))
    def run(states):
        def step(carry, _):
            states, key = carry
            key, sub = jax.random.split(key)
            masks = jax.vmap(env.action_mask)(states)
            action = jnp.argmax(jnp.where(masks,jax.random.uniform(sub,masks.shape),-1),axis=-1)
            states, ts = jax.vmap(env.step)(states,action)
            good = jnp.all(jnp.any(masks,axis=-1)) & jnp.all(states.hp>=0) & jnp.all(jnp.isfinite(ts.observation))
            return (states,key),good
        return jax.lax.scan(step,(states,jax.random.PRNGKey(5)),None,length=150)
    (states,_),good = jax.jit(run)(start)
    assert np.all(good) and int(jnp.sum(states.battle_steps)) > 100
