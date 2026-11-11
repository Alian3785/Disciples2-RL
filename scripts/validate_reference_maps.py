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

from stoix.envs import number_grid_reference_rewards as reference_rewards
from stoix.envs.number_grid import CONTINUE, DEFEND, REST, NumberGrid, action_names, load_map


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
        if bool(state.in_battle):
            # The Salty Dungeon: the human armies reach the capital at the end of day 1,
            # and a hero resting there defends it in an ordinary battle, not its garrison.
            assert not bool(state.garrison_battle)
            break
        if bool(state.done):
            # city_defence_train: the first wave takes the empty defended city on day 2.
            break
    if game_map.get('removed_capital_armies'):
        # The variant uses Default's city victory and base rewards.
        assert not set(game_map['removed_capital_armies']).intersection(
            source.upper() for source in game_map['reference_enemy_ids'])
        assert not env.reference_rewards and not env.defense_rewards.enabled
        assert not env.campaign_events.win_configured
        assert not bool(jnp.any(env.thief.attackers(env, state)))
        default = load_map('default')
        for key in ('step_cost', 'exploration_bonus', 'building_reward', 'rest_penalty_per_point'):
            assert game_map[key] == default[key]
        original, _ = reset(jax.random.PRNGKey(42))
        city, = (i for i, c in enumerate(game_map['cities']) if c['objective'])
        assert game_map['cities'][city]['source_id'] == 'S143FT0006'
        assert not bool(env._campaign_won(original))
        holding = original.replace(city_owned=original.city_owned.at[city].set(True))
        won = env._campaign_won(holding)
        assert bool(won) and float(env._objective_reward(won, jnp.bool_(False))) == 3.
    if game_map.get('defeat') == 'party_or_capital':
        original, _ = reset(jax.random.PRNGKey(42))
        leader = env.leader_hire
        assert leader.terminal_party_loss and not leader.can_hire
        assert 'hire_leader' not in action_names(game_map)
        killed = jax.jit(lambda s: env.campaign_events._cast(
            env, s, jnp.bool_(True), dict(amount=99999, bit=0)))(original)
        assert bool(killed.done & killed.lost) and int(killed.party_losses) == 1
        assert int(killed.capital_captor) == -1 and not bool(leader.available(killed))
        if env.reference_rewards:
            _, penalty = reference_rewards.finish(env, original, killed)
            assert float(penalty) == -10.

    if game_map.get('removed_capital_armies') and game_map.get('defeat') == 'party_or_capital':
        # Exercise both lethal paths with the Duke wounded and a surviving fighter.
        wounded = original.replace(hp=original.hp.at[leader.slot].set(1))
        after_spell = jax.jit(lambda s: env.campaign_events._cast(
            env, s, jnp.bool_(True), dict(amount=1, bit=0)))(wounded)
        assert int(after_spell.hp[leader.slot]) == 0 and bool(jnp.any(after_spell.hp[:6] > 0))
        assert not bool(after_spell.done | after_spell.lost) and int(after_spell.party_losses) == 0

        wounded = wounded.replace(hp=wounded.hp.at[0].set(400))
        dragons = game_map['reference_enemy_ids'].index('S143KC0052')
        battle = jax.jit(env._begin_battle)(wounded.replace(enemy=jnp.int32(dragons)))
        survived_leader = False
        for _ in range(24):
            command = CONTINUE if bool(env.automatic_battle_turn(battle)) else DEFEND
            battle, timestep = step(battle, jnp.int32(command))
            if bool(battle.done):
                break
            if int(battle.hp[leader.slot]) == 0 and bool(jnp.any(battle.hp[:6] > 0)):
                survived_leader = True
                assert not bool(battle.lost) and int(battle.party_losses) == 0
        assert survived_leader
        assert bool(battle.done & battle.lost) and not bool(battle.in_battle | battle.won)
        assert not bool(jnp.any(battle.hp[:6] > 0))
        assert int(battle.party_losses) == 1 and float(timestep.reward) < 0.

    if game_map.get('defeat') == 'party_or_capital' and game_map.get('party_day_rewards'):
        # The capital-attacks variant uses conditional day rewards and excludes attacker farming.
        rules = env.defense_rewards
        assert rules.suppress_attacker_rewards and rules.attacker_xp == 0.
        apply_rewards = jax.jit(lambda before, after: rules.apply(env, before, after))
        for milestone in game_map['party_day_rewards']:
            before = original.replace(day=jnp.int32(milestone['day']-1),
                                      party_upgrade_count=jnp.int32(milestone['upgrades']))
            earned, bonus = apply_rewards(before, before.replace(day=jnp.int32(milestone['day'])))
            assert float(bonus) == milestone['reward']
            _, again = apply_rewards(earned, earned)
            assert float(again) == 0.
            short = before.replace(party_upgrade_count=before.party_upgrade_count-1)
            _, missed = apply_rewards(short, short.replace(day=jnp.int32(milestone['day'])))
            assert float(missed) == 0.
        attacker = game_map['reference_enemy_ids'].index('S143KC0092')
        assert not bool(rules.rewardable[attacker])
        battle = original.replace(enemy=jnp.int32(attacker), in_battle=jnp.bool_(True))
        scored, bonus = reference_rewards.battle(env, battle, battle, battle.hp,
            jnp.bool_(True), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False))
        assert float(bonus) == 0. and not bool(scored.reference_defeated[attacker])
    if env.garrisons.enabled:
        # Every campaign capital has its guardian in the rear middle cell. An army on the
        # capital entrance while the hero is away starts the agent's garrison battle.
        rules = env.garrisons
        original, _ = reset(jax.random.PRNGKey(42))
        capital = rules.positions[rules.capital_index]
        # Scheduled armies walk to their targets before the garrison check (city_defence_train's
        # first wave heads for the city), so use the first attacker still on the entrance after REST.
        for enemy in (int(i) for i in jnp.flatnonzero(rules.attackers)):
            away = original.replace(position=original.position+jnp.array([4, 0], jnp.int32),
                                    alive=original.alive.at[enemy].set(True),
                                    enemy_positions=original.enemy_positions.at[enemy].set(capital))
            attacked, ts = step(away, jnp.int32(REST))
            if bool(jnp.all(attacked.enemy_positions[enemy] == capital)):
                break
        else:
            raise AssertionError('Every attacker leaves the capital entrance on REST')
        assert bool(attacked.in_battle & attacked.garrison_battle) and int(attacked.enemy) == enemy
        assert int(attacked.unit_ids[4]) == rules.guardian
        assert not bool(env._can_retreat(attacked, jnp.int32(0))) and bool(jnp.isfinite(ts.reward))
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
