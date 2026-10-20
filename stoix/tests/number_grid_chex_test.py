"""Transformation and state-contract checks for the current combat environment."""
from functools import lru_cache

from stoix.tests.number_grid_fixtures import compiled_method
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from stoa import AddActionMaskWrapper
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import CategoricalHead
from stoix.networks.torso import MLPTorso

from stoix.tests.number_grid_fixtures import basic_environment, current_environment
from stoix.envs.number_grid import SHOOT, WAIT, DEFEND, CONTINUE, REST




def scenarios(env):
    world, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    battle = compiled_method(env,'_begin_battle')(world.replace(enemy=jnp.int32(11)))
    hero = battle.replace(actor=jnp.int32(0))
    states = (
        world, hero, hero, hero,
        battle.replace(actor=jnp.int32(6)),
        hero.replace(retreating=hero.retreating.at[0].set(True)),
        hero.replace(turn_phase=hero.turn_phase.at[0].set(1)),
        hero.replace(done=jnp.bool_(True)),
    )
    actions = jnp.asarray([2, SHOOT, WAIT, DEFEND, CONTINUE, CONTINUE, WAIT, SHOOT], jnp.int32)
    return states, actions


def assert_same_result(expected, actual):
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, actual)
    # XLA can fuse float32 normalization; discrete state must still be exact.
    chex.assert_trees_all_close(expected, actual, atol=1e-6, rtol=1e-6)
    chex.assert_tree_all_finite(actual)
    for wanted, got in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        if not jnp.issubdtype(wanted.dtype, jnp.inexact):
            chex.assert_trees_all_equal(wanted, got)


@lru_cache(maxsize=2)
def reference_batch(contract=False):
    # Compare compiled scalar steps with the compiled vmapped production path.
    # Reuse the scalar oracle rather than compiling each individual action.
    """Cache eight scalar transition oracles and their input batch for a selected test map."""
    env = basic_environment() if contract else current_environment()
    states, actions = scenarios(env)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    scalar_step = compiled_method(env,'step')
    expected = jax.tree.map(lambda *xs: jnp.stack(xs),
                            *(scalar_step(state, action) for state, action in zip(states, actions)))
    return env, states, actions, batched, expected


def check_batched_steps(contract):
    # Same eight transitions for the fast core contract and full catalogue case.
    """Compare scalar and vmapped transitions, including exact discrete state and finite features."""
    env, states, actions, batched, expected = reference_batch(contract)
    actual = compiled_method(env, 'step', batched=True)(batched, actions)
    assert_same_result(expected, actual)
    next_states, ts = actual
    chex.assert_shape(next_states.hp, (8, 12))
    chex.assert_shape(ts.observation, (8, env.observation_size))
    chex.assert_type([next_states.hp, next_states.defended, ts.observation],
                     [jnp.int32, jnp.bool_, jnp.float32])


@pytest.mark.core
def test_batched_combat_contract_matches_individual_steps():
    """Cover all eight transition kinds on the compact map within the core compilation budget."""
    check_batched_steps(contract=True)


class CombatTransformsTest(chex.TestCase):
    def test_batched_step_matches_individual_steps(self):
        """Run the same scalar/batch contract on the full catalogue as an extended check."""
        check_batched_steps(contract=False)

    @chex.variants(with_jit=True, without_jit=False)
    def test_reset_and_mask_preserve_batch_contract(self):
        env = current_environment()
        keys = jax.random.split(jax.random.PRNGKey(7), 4)
        states, ts = self.variant(jax.vmap(env.reset))(keys)
        masks = self.variant(jax.vmap(env.action_mask))(states)
        chex.assert_shape(masks, (4, env.num_actions))
        chex.assert_type(masks, jnp.bool_)
        chex.assert_tree_all_finite((states, ts))
        for index, key in enumerate(keys):
            expected = compiled_method(env,'reset')(key)
            actual = jax.tree.map(lambda x, index=index: x[index], (states, ts))
            assert_same_result(expected, actual)


@pytest.mark.parametrize('training_autoreset', [
    pytest.param('contract', marks=pytest.mark.core), 'current',
], indirect=True)
def test_cached_autoreset_changes_rng_and_preserves_final_observation(training_autoreset):
    env, eval_env, advance = training_autoreset
    assert isinstance(eval_env, AddActionMaskWrapper)
    state, ts = compiled_method(env,'reset')(jax.random.split(jax.random.PRNGKey(2),2))
    original = np.asarray(state.battle_key)
    state, ts = advance(state,jnp.array([7,3]))
    assert np.all(ts.truncated()) and np.all(state.step_count == 0)
    assert np.all(ts.discount == 1)
    assert not np.array_equal(original,state.battle_key)
    assert np.all(ts.extras['next_obs']['observation'][:,3] == 1)
    assert np.all(ts.observation['observation'][:,3] == 0)
    np.testing.assert_array_equal(ts.observation['action_mask'][:,8:REST],False)
    assert np.all(ts.observation['action_mask'][:, REST])
    again, _ = advance(state,jnp.array([7,3]))
    assert not np.array_equal(state.battle_key,again.battle_key)


@pytest.mark.core
def test_masked_warrior_policy_sample_argmax_and_ppo_gradient(current_game):
    env, initial, _, _ = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(0))
    # Wrapper/schema integration is covered by batched reset/step above.
    # The gradient check only needs the actual combat observation and mask.
    observation = {'observation':compiled_method(env,'observation')(state),'action_mask':compiled_method(env,'action_mask')(state)}
    obs = jax.tree.map(lambda x:jnp.repeat(x[None],64,axis=0),observation)
    policy = FeedForwardActor(torso=MLPTorso(layer_sizes=[32]), action_head=CategoricalHead(env.num_actions))
    params = policy.init(jax.random.PRNGKey(0), obs)
    dist = policy.apply(params, obs)
    probabilities = np.asarray(dist.probs_parameter())
    mask = np.asarray(obs['action_mask'])
    assert np.all(probabilities[~mask] == 0)
    actions = dist.sample(64, seed=jax.random.PRNGKey(1))
    assert np.all(np.take_along_axis(mask[None], np.asarray(actions)[...,None], axis=-1))
    assert np.all(np.take_along_axis(mask, np.asarray(dist.mode())[:,None], axis=-1))
    old_log = jax.lax.stop_gradient(dist.log_prob(actions[0]))
    def loss(p):
        updated = policy.apply(p, obs)
        ratio = jnp.exp(updated.log_prob(actions[0])-old_log)
        return -jnp.mean(jnp.minimum(ratio,jnp.clip(ratio,.8,1.2))+.01*updated.entropy())
    value, gradients = jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(value) and all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))


def test_human_service_uses_identical_mask_and_rejects_unreachable_target(human_service):
    """Verify the manual service exposes the environment mask and rejects an unreachable attack."""
    service = human_service
    # This regression starts in combat; building an unused new-game snapshot
    # would compile metadata that is unrelated to mask/step agreement.
    session = 'mask-agreement-regression'
    env = service.env
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(11)))
    state = state.replace(actor=jnp.int32(0))
    service.sessions[session] = ((env.construction.faction, env.construction.lord['id']), state, 0.)
    snapshot = service.snapshot(env, state, 0.)
    np.testing.assert_array_equal(snapshot['action_mask'], compiled_method(env,'action_mask')(state))
    with pytest.raises(ValueError, match='недоступно'):
        service.act(session, SHOOT+3)
    assert service.sessions[session][1] is state
    direct, ts = compiled_method(env,'step')(state,jnp.int32(SHOOT))
    first = service.act(session, SHOOT)['events'][0]
    assert first == {**service.snapshot(env, direct, float(ts.reward)), 'reward': float(ts.reward)}
