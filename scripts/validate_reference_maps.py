"""GPU preflight for the requested campaign training; no learning or timing.

Validates source formations, reachability (environment construction), observation
and action dimensions, reset, world transitions, scheduled spawning and goals.
Planner data is never exposed to the learner.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from stoix.envs.number_grid import NumberGrid, REST, action_names, load_map


def validate(name):
    """Check a campaign on CUDA, including reset, REST, and legal objective approaches."""
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Map preflight requires CUDA')
    game_map = load_map(name)
    env = NumberGrid(map_config=game_map)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state, timestep = reset(jax.random.PRNGKey(42))
    assert env.num_actions == len(action_names(game_map))
    assert env.observation(state).shape == (env.observation_size,)
    assert bool(jnp.all(jnp.isfinite(env.observation(state))))
    assert not bool(state.done)
    assert int(state.gold) == game_map.get('starting_gold', 0)
    assert tuple(int(v) for v in state.mana) == env.starting_mana
    assert bool(jnp.all(state.scenario_spawned == ~env.scenario.scheduled))
    for _ in range(3):
        assert bool(env.action_mask(state)[REST])
        state, timestep = step(state, jnp.int32(REST))
        assert bool(jnp.isfinite(timestep.reward))
    if env.bot_attack.enabled:
        # Default Capital's scripted Empire party: absent at reset, then a real battle
        # without withdrawal on the REST that begins its attack day.
        rules = env.bot_attack
        original, _ = reset(jax.random.PRNGKey(42))
        assert not bool(original.alive[rules.enemy]) and not bool(original.scenario_spawned[rules.enemy])
        quiet, _ = step(original.replace(day=jnp.int32(rules.day-2)), jnp.int32(REST))
        assert not bool(quiet.alive[rules.enemy]) and not bool(quiet.in_battle)
        attacked, ts = step(original.replace(day=jnp.int32(rules.day-1)), jnp.int32(REST))
        assert bool(attacked.in_battle) or bool(attacked.done)
        assert int(attacked.enemy) == rules.enemy and bool(attacked.scenario_spawned[rules.enemy])
        assert bool(jnp.isfinite(ts.reward))
        assert not bool(env._can_retreat(attacked, jnp.int32(0)))
        if rules.raid_enemies:
            # Later parties are ordinary stationary encounters, not forced attacks.
            waiting, ts = step(original.replace(day=jnp.int32(rules.first_day-1)), jnp.int32(REST))
            assert not bool(waiting.in_battle) and not bool(waiting.done)
            assert bool(jnp.isfinite(ts.reward)) and not bool(jnp.any(waiting.pending_encounters))
            assert int(jnp.sum(waiting.alive[rules.raids])) == 1
            enemy = int(env.territory.enemy_at(waiting, rules.position))
            assert enemy in rules.raid_enemies and not bool(env.territory.blocked(rules.position))
            calm, _ = step(waiting.replace(day=jnp.int32(rules.first_day+7)), jnp.int32(REST))
            assert not bool(calm.in_battle) and bool(jnp.all(calm.alive == waiting.alive))
            destroyed = rules.schedule_respawn(calm, calm.replace(alive=calm.alive.at[enemy].set(False)))
            assert int(destroyed.bot_respawn_day) == int(calm.day)+rules.respawn_days
            early, _ = step(destroyed.replace(day=destroyed.bot_respawn_day-2), jnp.int32(REST))
            assert not bool(jnp.any(early.alive[rules.raids]))
            returned, _ = step(early, jnp.int32(REST))
            assert int(jnp.sum(returned.alive[rules.raids])) == 1 and not bool(returned.in_battle)
            assert not bool(jnp.any(returned.pending_encounters))
    if env.scenario.enabled:
        event = env.scenario.events[0]
        original, _ = reset(jax.random.PRNGKey(42))
        before = original.replace(day=jnp.int32(event['spawn_turn']))
        advance = jax.jit(env.scenario.advance)
        early = advance(before)
        assert not bool(early.alive[event['enemy']])
        born = advance(before.replace(day=before.day+1))
        assert bool(born.alive[event['enemy']])
        assert bool(born.scenario_spawned[event['enemy']])
        assert bool(jnp.any(born.enemy_positions[event['enemy']] != original.enemy_positions[event['enemy']]))
        assert bool(env._goals_left(jnp.zeros_like(original.alive), original.scenario_spawned))
        assert not bool(env._goals_left(jnp.zeros_like(original.alive), jnp.ones_like(original.scenario_spawned)))
        # Arrival must yield a real battle through the same REST path used in training.
        positions = original.enemy_positions.at[event['enemy']].set(original.position+jnp.array([0, 1]))
        contact = original.replace(day=jnp.int32(event['spawn_turn']), enemy_positions=positions)
        arrived, ts = step(contact, jnp.int32(REST))
        assert bool(arrived.in_battle) or bool(arrived.done)
        assert int(arrived.enemy) == event['enemy']
        assert bool(jnp.isfinite(ts.reward))
    targets = ([env.objective_enemy] if env.objective_enemy is not None
               else game_map.get('target_rewards', {}).get('enemies', []))
    for target in targets if not env.scenario.enabled else []:
        # Static dragon objectives: verify a legal approach and real engagement.
        # This structural preflight does not claim a winning battle trajectory.
        original, _ = reset(jax.random.PRNGKey(42))
        position = game_map['opponent_positions'][target]
        blocked = env.territory.blocked_grid.reshape(env.size, env.size)
        direction, offset = next((i, d) for i, d in enumerate(env.directions.tolist())
            if 0 < position[0]-d[0] < env.size-1 and 0 < position[1]-d[1] < env.size-1
            and not bool(blocked[position[0]-d[0], position[1]-d[1]]))
        near = original.replace(position=jnp.array([position[0]-offset[0], position[1]-offset[1]], jnp.int32))
        assert bool(env.action_mask(near)[direction])
        arrived, ts = step(near, jnp.int32(direction))
        assert bool(arrived.in_battle) or bool(arrived.done)
        assert int(arrived.enemy) == target and bool(jnp.isfinite(ts.reward))
        assert bool(env._goals_left(original.alive, original.scenario_spawned))
        if env.objective_enemy is not None:
            assert not bool(env._goals_left(original.alive.at[target].set(False), original.scenario_spawned))
        else:
            alone = jnp.zeros_like(original.alive).at[target].set(True)
            assert bool(env._goals_left(alone, original.scenario_spawned))
            assert not bool(env._goals_left(jnp.zeros_like(alone), original.scenario_spawned))
    result = dict(map=name, gpu=str(jax.devices()[0]), armies=env.num_opponents,
                  scheduled_armies=len(env.scenario.events), actions=env.num_actions,
                  observation_size=env.observation_size, structural_passability=True,
                  preflight='passed', training_transitions=0,
                  target_rewards=game_map.get('target_rewards'), optional_enemies=game_map.get('optional_enemies', []),
                  absent_enemies=game_map.get('absent_enemies', []), bot_attack=game_map.get('bot_attack'))
    print(json.dumps(result), flush=True)
    jax.clear_caches()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', action='append', dest='maps')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [validate(name) for name in args.maps or
               ('super_last_stand', 'siege_train', 'the_charmed_child', 'trade_train')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
