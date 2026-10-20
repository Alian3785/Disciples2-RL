"""Fixed six-unit formation for independent combat regression cases."""
from functools import lru_cache
import json
from pathlib import Path

MAP = json.loads((Path(__file__).parent / 'fixtures/basic_combat_map.json').read_text())
# Mechanics suites keep the party they were written for: the cultist stands back left.
# A new game takes the ruler's reference party (number_grid_starting_party_test.py).
SCENARIO_PARTY = ['possessed', 'duke', 'possessed', 'cultist', None, None]
# They also keep the warrior ruler of that party; the current map itself starts with a mage.
SCENARIO_LORD = 'warrior'


def replace_base_state(state, **changes):
    """Set a test scenario beneath wrappers without altering reset caches."""
    if hasattr(state, 'base_env_state'):
        return state.replace(base_env_state=replace_base_state(state.base_env_state, **changes))
    return state.replace(**changes)


@lru_cache(maxsize=1)
def current_environment():
    from stoix.envs.number_grid import MAP as CURRENT, NumberGrid
    return NumberGrid(map_config={**CURRENT, 'hero_roster': SCENARIO_PARTY, 'lord_type': SCENARIO_LORD})


@lru_cache(maxsize=None)
def batch_layout(count, capacity):
    """Fuse padding/slicing of the whole state tree instead of launching per leaf."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def pad(args):
        return jax.tree.map(
            lambda x: jnp.concatenate((x, jnp.repeat(x[:1], capacity-count, axis=0))), args)

    @jax.jit
    def trim(result):
        return jax.tree.map(lambda x: x[:count], result)

    return pad, trim


def compiled_method(env, name, batched=False):
    """Keep a stable JIT function identity for immutable test environments."""
    import jax
    key = name + ('_batched' if batched else '')
    if not hasattr(env,'_test_jit_methods'):
        env._test_jit_methods = {}
    if key not in env._test_jit_methods:
        method = getattr(env,name)
        compiled = jax.jit(jax.vmap(method) if batched else method)
        import jax.numpy as jnp
        def invoke(*args):
            # jnp.full(..., .99) is weakly typed while jnp.zeros(...) is not.
            # Normalize inputs so equivalent float32/int32 scenarios reuse the
            # same CUDA executable instead of recompiling the entire battle.
            args = jax.tree.map(lambda x:jnp.asarray(x,dtype=jnp.asarray(x).dtype),args)
            return compiled(*args)
        if batched and name in ('step','_battle_step'):
            def shared_batch(*args):
                count = jax.tree.leaves(args)[0].shape[0]
                capacity = max(64,1 << (count-1).bit_length())
                if count == capacity:
                    return invoke(*args)
                pad, trim = batch_layout(count, capacity)
                # Keep the large transition separate: changing the case count
                # must only compile these tiny adapters, not the combat graph.
                return trim(invoke(*pad(args)))
            env._test_jit_methods[key] = shared_batch
        else:
            env._test_jit_methods[key] = invoke
    return env._test_jit_methods[key]


@lru_cache(maxsize=1)
def basic_environment():
    from stoix.envs.number_grid import NumberGrid
    return NumberGrid(map_config=MAP)


def hero_roster_state(env,state,roster):
    """Valid named formation in the existing dynamic battle state, no new JIT graph."""
    import jax.numpy as jnp
    ids = jnp.array([env.progression.ids[key] if key else 0 for key in roster],jnp.int32)
    levels = env.progression.base_levels[ids]
    return state.replace(unit_ids=state.unit_ids.at[:6].set(ids),
        unit_levels=state.unit_levels.at[:6].set(levels),unit_xp=state.unit_xp.at[:6].set(0),
        hp=state.hp.at[:6].set(env.progression.stats(ids,levels)[:,0].astype(jnp.int32)))


def enemy_roster_state(env,state,roster):
    """Pin a mechanic's enemy formation independently of evolving scenario representatives."""
    import jax.numpy as jnp
    ids = jnp.array([env.progression.ids[key] if key else 0 for key in roster],jnp.int32)
    levels = env.progression.base_levels[ids]
    return state.replace(unit_ids=state.unit_ids.at[6:].set(ids),
        unit_levels=state.unit_levels.at[6:].set(levels),unit_xp=state.unit_xp.at[6:].set(0),
        hp=state.hp.at[6:].set(env.progression.stats(ids,levels)[:,0].astype(jnp.int32)))
