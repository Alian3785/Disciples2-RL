"""Wall masking across the Stoa wrapper, policy distribution, and auto-reset."""
from stoix.tests.number_grid_fixtures import compiled_method
import jax
import jax.numpy as jnp
import numpy as np
from stoa import AddActionMaskWrapper

from numbergrid_config import make_config
from stoix.envs.number_grid_legacy import MAP, DIRECTIONS, NumberGrid, wrap_wall_action_mask
from stoix.networks.base import FeedForwardActor, FeedForwardCritic
from stoix.networks.heads import CategoricalHead, ScalarCriticHead
from stoix.networks.torso import MLPTorso
from stoix.utils.make_env import make


def test_wall_mask_every_cell_and_dangerous_actions_remain_available():
    env = NumberGrid()
    base, _ = env.reset(jax.random.PRNGKey(0))
    positions = np.array([(r,c) for r in range(1,23) for c in range(1,23)], np.int32)
    masks = jax.jit(jax.vmap(lambda p: env.action_mask(base.replace(position=p))))(positions)
    expected = np.array([[0 < r+dr < 23 and 0 < c+dc < 23 for dr,dc in DIRECTIONS]
                         for r,c in positions])
    np.testing.assert_array_equal(masks, expected)
    assert masks.dtype == jnp.bool_ and np.all(np.sum(masks, axis=1) >= 3)
    # Occupied destinations and adjacency to stronger enemies are not walls.
    assert np.all(env.action_mask(base.replace(position=jnp.array([3,7]))))
    assert np.all(env.action_mask(base.replace(position=jnp.array([18,20]))))
    assert np.all(env.action_mask(base.replace(done=jnp.bool_(True))))


def test_requested_wrapper_and_observation_space():
    env = wrap_wall_action_mask(NumberGrid())
    assert isinstance(env, AddActionMaskWrapper)
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert set(ts.observation) == {'observation', 'action_mask'}
    assert ts.observation['observation'].shape == (52,)
    assert ts.observation['action_mask'].shape == (8,)
    assert jax.tree.structure(ts.observation) == jax.tree.structure(env.observation_space().generate_value())
    state, ts = compiled_method(env,'step')(state, jnp.int32(7))
    np.testing.assert_array_equal(ts.observation['action_mask'], [False,False,True,True,True,False,False,False])
    np.testing.assert_array_equal(ts.observation['action_mask'], ts.extras['action_mask'])
    legacy = wrap_wall_action_mask(NumberGrid(map_config={**MAP, 'mask_walls': False}))
    assert isinstance(legacy, NumberGrid)
    assert legacy.reset(jax.random.PRNGKey(0))[1].observation.shape == (52,)


def test_autoreset_keeps_reset_mask_and_final_mask_separate():
    config = make_config(map_config=MAP)
    config.env.kwargs.max_steps = 1
    env, eval_env = make(config)
    assert isinstance(eval_env, AddActionMaskWrapper)
    state, _ = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    state, ts = compiled_method(env,'step')(state, jnp.array([7,3]))
    assert np.all(ts.truncated()) and np.all(ts.discount == 1)
    assert np.all(ts.observation['action_mask'])  # reset at [2,2]
    final = ts.extras['next_obs']
    np.testing.assert_array_equal(final['action_mask'][0], [False,False,True,True,True,False,False,False])
    assert np.all(final['action_mask'][1])
    assert np.all(final['observation'][:,-1] == 1)
    assert np.all(ts.observation['observation'][:,-1] == 0)


def test_masked_policy_probabilities_sampling_entropy_gradients_and_critic():
    env = wrap_wall_action_mask(NumberGrid())
    state, _ = env.reset(jax.random.PRNGKey(0))
    _, ts = env.step(state, jnp.int32(7))  # [1,1]: only E, SE, S are allowed
    obs = jax.tree.map(lambda x: jnp.repeat(x[None], 128, axis=0), ts.observation)
    actor = FeedForwardActor(torso=MLPTorso(layer_sizes=[32,32]), action_head=CategoricalHead(8))
    params = actor.init(jax.random.PRNGKey(5), obs)
    masked = actor.apply(params, obs)
    raw = actor.apply(params, obs['observation'])
    probs = np.asarray(masked.probs_parameter())
    mask = np.asarray(obs['action_mask'])
    assert np.all(probs[~mask] == 0)
    expected = np.where(mask, np.asarray(raw.probs_parameter()), 0)
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(probs, expected, atol=1e-7)
    actions = jax.jit(lambda k: actor.apply(params, obs).sample(64, seed=k))(jax.random.PRNGKey(9))
    assert np.all(np.take_along_axis(mask[None], np.asarray(actions)[...,None], axis=-1))
    assert np.all(np.take_along_axis(mask, np.asarray(masked.mode())[:,None], axis=-1))
    np.testing.assert_allclose(masked.entropy(), -(probs[:,2:5]*np.log(probs[:,2:5])).sum(-1), atol=1e-6)
    selected = jnp.asarray(actions[0])
    old_log_prob = jax.lax.stop_gradient(masked.log_prob(selected))
    def loss(p):
        # Re-evaluate the stored observation/mask as PPO does on a minibatch.
        policy = actor.apply(p, obs)
        ratio = jnp.exp(policy.log_prob(selected) - old_log_prob)
        return -jnp.mean(jnp.minimum(ratio, jnp.clip(ratio,.8,1.2)) + .01*policy.entropy())
    value, grads = jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(value) and all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    critic = FeedForwardCritic(torso=MLPTorso(layer_sizes=[32,32]), critic_head=ScalarCriticHead())
    critic_params = critic.init(jax.random.PRNGKey(8), obs)
    np.testing.assert_array_equal(critic.apply(critic_params, obs), critic.apply(critic_params, obs['observation']))
