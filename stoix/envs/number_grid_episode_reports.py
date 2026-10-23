"""Sparse episode diagnostics computed on device; never used by the policy.

Sources: NumberGrid.step/_battle_step_regular, Progression.finish, CastingRules
finish_battle, and UnitSwapRules.apply. A decision can finish at most one battle;
queued scenario encounters require a separate subsequent world decision.
"""
import jax.numpy as jnp

from stoix.envs.number_grid_episode_schema import HEADER, SCHEMA_VERSION, WIDTH


def empty_event():
    return {'emit': jnp.bool_(False), 'data': jnp.zeros(WIDTH, jnp.int32)}


def permanent_party(state):
    """Six permanent fighters, even underneath copies and battle/map summons."""
    native = state.copied[:6] | (state.summon_owner[:6] >= 0)
    ids = jnp.where(native, state.native_ids[:6], state.unit_ids[:6])
    levels = jnp.where(native, state.native_levels[:6], state.unit_levels[:6])
    xp = jnp.where(native, state.native_xp[:6], state.unit_xp[:6])
    hp = jnp.where(state.summon_owner[:6] >= 0, state.summon_saved_hp[:6], state.hp[:6])
    party = jnp.stack((ids, levels, xp, hp))
    party = jnp.where(state.summon_battle, state.spell_party, party)
    return party.at[3].set((party[3] > 0) & (party[0] != 0))


def episode_event(env, before, world, after, action, valid, battles):
    """Retain terminal data before autoreset and XP before promotion resets it."""
    old = permanent_party(before)
    new = permanent_party(after)
    order = jnp.arange(6, dtype=jnp.int32)
    swapped = jnp.bool_(False)
    if env.unit_swaps_enabled:
        rules = env.unit_swaps
        index = jnp.clip(action - rules.start, 0, rules.count - 1)
        swapped = valid & ~before.in_battle & (action >= rules.start) & (action < rules.end)
        permutation = jnp.where(rules._pairs(before)[1][index], rules.big[index], rules.small[index])
        order = jnp.where(swapped, permutation, order)
    aligned = old[:, order]
    world_party = permanent_party(world)
    # Trainer awards are positive world XP deltas on an existing fighter. A hire
    # must not inherit the former occupant's XP; a swap must not manufacture XP.
    training_xp = jnp.where(valid & ~before.in_battle & (aligned[0] != 0)
        & (world_party[0] == aligned[0]) & (world_party[1] == aligned[1]),
        jnp.maximum(world_party[2] - aligned[2], 0), 0)
    ended = jnp.any(battles[:3] > 0)
    battle_xp = jnp.where(ended & ~world.summon_battle, after.last_xp[:6], 0)
    gains = training_xp + battle_xp
    # world also contains freshly spawned armies; they can be killed immediately.
    defeated = (before.alive | world.alive) & ~after.alive
    defeated_enemy = jnp.where(jnp.any(defeated), jnp.argmax(defeated), -1)
    opponent = jnp.where(ended | world.in_battle, world.enemy, defeated_enemy)
    method = jnp.where(ended, jnp.where(world.summon_battle, 2, 1),
                       jnp.where(defeated_enemy >= 0, 3, 0))
    result = jnp.where(battles[0] > 0, 1, jnp.where(battles[1] > 0, 2,
                        jnp.where(battles[2] > 0, 3, 0)))
    outcome = jnp.where(after.done, jnp.where(after.won, 1, jnp.where(after.lost, 2, 3)), 0)
    required = env.progression.required_xp(new[0], new[1])
    enemy_levels = after.enemy_progress[jnp.maximum(opponent, 0), :, 0]
    header = jnp.array([after.step_count, after.day, outcome, opponent, method, result,
                       defeated_enemy, after.position[0], after.position[1], action], jnp.int32)
    # IDs/levels detect actual permanent promotions; transient forms never appear.
    changed = jnp.any(new[:3] != aligned[:3])
    emit = ~before.done & ((before.step_count == 0) | after.done | ended
                          | jnp.any(defeated) | jnp.any(gains) | swapped | changed)
    data = jnp.concatenate((header, old.reshape(-1), new.reshape(-1), required,
                            gains, order, enemy_levels)).astype(jnp.int32)
    return {'emit': emit, 'data': data}


def compact_episode_events(episodes):
    """Stable GPU compaction, with worst-case capacity and no dropped events.

    PPO metrics are [device, update, batch, time, env]; diagnostic rollouts are
    [time, env]. Pack in [update, time, device, batch, env] order. The host reads
    count once per learner block, then transfers only the populated prefix.
    """
    event = episodes['episode_event']
    shape = event['emit'].shape
    if len(shape) == 5:
        axes = (1, 3, 0, 2, 4)
        streams = shape[0] * shape[2] * shape[4]
    elif len(shape) == 2:
        axes = (0, 1)
        streams = shape[1]
    else:
        raise ValueError(f'Unsupported episode event axes: {shape}')

    def flatten(value):
        tail = value.shape[len(shape):]
        permutation = axes + tuple(range(len(shape), value.ndim))
        return jnp.transpose(value, permutation).reshape((-1,) + tail)

    emit = flatten(event['emit'])
    indices = jnp.nonzero(emit, size=emit.size, fill_value=0)[0]
    rows = flatten(event['data'])[indices]
    returns = flatten(episodes['episode_return'])[indices]
    return jnp.sum(emit, dtype=jnp.int32), {
        'data': rows, 'returns': returns, 'stream': indices % streams, 'offset': indices}
