"""Cache the fixed map reset while preserving a fresh combat random stream."""
import dataclasses
import jax
import jax.numpy as jnp
from stoa.core_wrappers.auto_reset import CachedAutoResetWrapper, CachedAutoResetState
from stoa.core_wrappers.wrapper import WrapperState


def restore_random_stream(template, live, key):
    # This recursion walks the static wrapper structure only during tracing.
    if isinstance(template, WrapperState):
        changes = {'base_env_state': restore_random_stream(template.base_env_state, live.base_env_state, key)}
        if 'rng_key' in {f.name for f in dataclasses.fields(template)}:
            changes['rng_key'] = live.rng_key
        return template.replace(**changes)
    return template.replace(battle_key=key)


class NumberGridBattleAutoReset(CachedAutoResetWrapper):
    def step(self, state, action, env_params=None):
        live, ts = self._env.step(state.base_env_state, action, env_params)
        reset = restore_random_stream(state.cached_state, live, live.rng_key)
        selected = jax.tree.map(lambda a, b: jnp.where(ts.done(), a, b), reset, live)
        observation = jax.tree.map(lambda a, b: jnp.where(ts.done(), a, b), state.cached_obs, ts.observation)
        ts = self._maybe_add_obs_to_extras(ts).replace(observation=observation)
        return CachedAutoResetState(selected, state.cached_state, state.cached_obs), ts
