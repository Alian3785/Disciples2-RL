"""Strategic turns, movement budget and rest are executed only on CUDA."""
from stoix.tests.number_grid_fixtures import compiled_method, basic_environment
import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.tests.number_grid_fixtures import MAP, replace_base_state
from stoix.tests.number_grid_fixtures import enemy_roster_state
from stoix.envs.number_grid import (
    NumberGrid, SHOOT, DEFEND, WAIT, RETREAT, CONTINUE, REST, RESTED,
    WITHDRAW, COPY_ALLY,
)


def last_round_state(env, initial):
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    # The Squire takes the last activation of round 75; both parties survive.
    return state.replace(round=jnp.int32(env.max_rounds), actor=jnp.int32(6),
                         turn_phase=jnp.full(12, 2, jnp.int32).at[6].set(0))


@pytest.mark.core
def test_round_limit_forces_every_survivor_to_retreat_and_penalizes_once(current_game):
    env, initial, advance, _ = current_game
    mask = compiled_method(env, 'action_mask')
    state = last_round_state(env, initial).replace(actor=jnp.int32(0),
        turn_phase=jnp.full(12, 2, jnp.int32).at[0].set(0))
    state = state.replace(hp=jnp.where(state.hp > 0, 1000, 0))
    assert mask(state)[DEFEND]  # Round 75 is still playable.
    starting_hp = state.hp[:6]
    state, ts = advance(state, jnp.int32(DEFEND))
    expected_cost = env.step_cost + .05*env.recruitment.composition(state)[2]
    assert jnp.isclose(ts.reward, -env.round_limit_penalty-expected_cost)
    assert not state.in_battle and state.last_event == WITHDRAW and state.alive[0]
    assert not state.done and state.step_count == initial.step_count+1
    assert ts.extras['player_battle_transition'] == 1
    assert ts.extras['automatic_battle_transition'] >= 8  # select/complete retreat for all four units
    assert jnp.all(state.hp[:6] <= starting_hp)  # Retreat does not heal the party.
    chex.assert_trees_all_equal(state.position, initial.position)
    assert mask(state)[REST] and not env.round_limit_retreat(state)
    second = compiled_method(env, '_begin_battle')(state.replace(enemy=jnp.int32(0)))
    assert second.round == 1 and not env.round_limit_retreat(second)
    assert mask(second.replace(actor=jnp.int32(0)))[DEFEND]


@pytest.mark.core
def test_forced_retreat_mask_and_step_preserve_paralysis_enemy_attacks_and_dots(current_game):
    env, initial, advance, battle_step = current_game
    mask = compiled_method(env, 'action_mask')
    state = last_round_state(env, initial).replace(round=jnp.int32(76), actor=jnp.int32(0))
    assert jnp.array_equal(jnp.flatnonzero(mask(state)), jnp.array([RETREAT]))
    for action in (SHOOT, DEFEND, WAIT, CONTINUE):
        rejected, _ = advance(state, jnp.int32(action))
        chex.assert_trees_all_equal(rejected.hp, state.hp)
        chex.assert_trees_all_equal(rejected.retreating, state.retreating)
        chex.assert_trees_all_equal(rejected.battle_key, state.battle_key)
    for field in ('paralyzed', 'long_paralyzed', 'retreating'):
        waiting = state.replace(**{field: getattr(state, field).at[0].set(True)})
        assert jnp.array_equal(jnp.flatnonzero(mask(waiting)), jnp.array([CONTINUE]))
        out, _ = battle_step(waiting, jnp.int32(CONTINUE), waiting.battle_key, jnp.zeros(env.random_size))
        assert not out.done
        if field == 'paralyzed':
            assert not out.paralyzed[0] and not out.retreating[0] and not out.escaped[0]
    rolls = jnp.zeros(env.random_size)
    enemy = state.replace(actor=jnp.int32(6))
    out, _ = battle_step(enemy, jnp.int32(CONTINUE), enemy.battle_key, rolls)
    assert jnp.sum(out.hp[:6]) < jnp.sum(enemy.hp[:6]) and not out.done
    poisoned = state.replace(actor=jnp.int32(1),
        turn_phase=jnp.full(12, 2, jnp.int32).at[0].set(0).at[1].set(0),
        activation_done=jnp.zeros(12, bool),
        poison_mode=state.poison_mode.at[0].set(1),
        poison_damage=state.poison_damage.at[0].set(7),
        poison_source=state.poison_source.at[0].set(6))
    out, _ = battle_step(poisoned, jnp.int32(RETREAT), poisoned.battle_key, rolls)
    assert out.last_poison_damage[0] == 7 and out.hp[0] == poisoned.hp[0]-7
    assert out.retreating[1] and not out.done


@pytest.mark.core
def test_round_limit_does_not_override_victory_post_healing_or_episode_budget(current_game):
    env, initial, advance, battle_step = current_game
    state = last_round_state(env, initial).replace(actor=jnp.int32(1),
        turn_phase=jnp.full(12, 2, jnp.int32).at[1].set(0))
    state = state.replace(hp=state.hp.at[6:].set(0).at[6].set(1))
    out, reward = battle_step(state, jnp.int32(SHOOT), state.battle_key, jnp.zeros(env.random_size))
    assert not out.in_battle and not out.done and not out.alive[0] and reward == 1
    forced = last_round_state(env, initial).replace(round=jnp.int32(76), actor=jnp.int32(0))
    assert not env.round_limit_retreat(forced.replace(post_victory=jnp.int32(1)))
    last = last_round_state(env, initial).replace(actor=jnp.int32(0),
        turn_phase=jnp.full(12, 2, jnp.int32).at[0].set(0), step_count=jnp.int32(env.max_steps-1))
    last = last.replace(hp=jnp.where(last.hp > 0, 1000, 0))
    out, ts = advance(last, jnp.int32(DEFEND))
    assert out.done and ts.truncated() and not out.won and not out.lost
    assert not out.in_battle and out.step_count == env.max_steps  # responses finish before truncation


def test_human_game_automatically_selects_retreat_after_round_limit(human_service, current_game):
    """Require the manual service to finish forced round-limit retreat without extra player input."""
    env, initial, _, _ = current_game
    state = last_round_state(env, initial).replace(actor=jnp.int32(1),
        turn_phase=jnp.full(12, 2, jnp.int32).at[1].set(0))
    state = state.replace(hp=state.hp.at[:6].set(0).at[1].set(initial.hp[1]))
    key = (env.construction.faction, env.construction.lord['id'])
    token = 'round-limit-retreat-regression'
    human_service.sessions[token] = (key, state, 0.)
    result = human_service.act(token, DEFEND)
    final = result['snapshot']
    assert not final['state']['done'] and not final['state']['in_battle']
    assert not final['forced_retreat']
    assert len(result['events']) == 1 and final['state']['last_event'] == WITHDRAW
    assert final['state']['step_count'] == state.step_count+1


def decision_battle(env, initial):
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    state = enemy_roster_state(env, state, ['squire', 'squire', None, None, None, None])
    return state.replace(actor=jnp.int32(0), hp=jnp.where(state.hp > 0, 1000, 0),
        turn_phase=jnp.full(12, 2, jnp.int32).at[jnp.array([0, 6, 7, 1])].set(0),
        priority=jnp.zeros(12).at[0].set(100).at[6].set(90).at[7].set(80).at[1].set(70))


@pytest.mark.core
def test_one_decision_contains_two_enemy_actions_and_one_cost(current_game):
    env, initial, advance, battle_step = current_game
    state = decision_battle(env, initial)
    out, ts = advance(state, jnp.int32(DEFEND))
    assert out.actor == 1 and out.step_count == state.step_count+1 and not out.done
    assert out.battle_steps == state.battle_steps+3 and out.enemy_turns == state.enemy_turns+2
    assert ts.extras['battle_transition'] == 3 and ts.extras['automatic_battle_transition'] == 2
    assert ts.extras['player_battle_transition'] == 1 and ts.extras['enemy_battle_transition'] == 2
    assert jnp.isclose(ts.reward, -env.step_cost-.05*env.recruitment.composition(out)[2])
    # Independently replay the known three activations with the same random stream.
    replay = state
    draw = jax.jit(lambda key: (jax.random.split(key)[0],
        jax.random.uniform(jax.random.split(key)[1], (env.random_size,))))
    for action in (DEFEND, CONTINUE, CONTINUE):
        key, rolls = draw(replay.battle_key)
        replay, gain = battle_step(replay, jnp.int32(action), key, rolls)
        assert gain == 0
    chex.assert_trees_all_close(out, replay.replace(step_count=out.step_count), atol=1e-6)
    assert not compiled_method(env, 'action_mask')(out)[CONTINUE]
    # Engagement is itself the decision even when RED has the first initiative.
    world = initial.replace(position=env.opponent_positions[0]-jnp.array([0, 1]),
                            potion_bonus=initial.potion_bonus.at[:6, 4].set(-100))
    entered, ts = advance(world, jnp.int32(2))
    assert entered.in_battle and entered.actor < 6 and entered.step_count == world.step_count+1
    assert ts.extras['player_battle_transition'] == 0 and ts.extras['enemy_battle_transition'] > 0
    assert ts.extras['battle_transition'] == ts.extras['automatic_battle_transition']
    assert not compiled_method(env, 'action_mask')(entered)[CONTINUE]


@pytest.mark.core
def test_paralysis_and_retreat_complete_without_additional_steps(current_game):
    env, initial, advance, _ = current_game
    state = decision_battle(env, initial).replace(
        turn_phase=jnp.full(12, 2, jnp.int32).at[jnp.array([0, 1, 2, 3])].set(0),
        priority=jnp.array([100, 90, 80, 70, 0, 0, 0, 0, 0, 0, 0, 0], jnp.float32))
    state = state.replace(paralyzed=state.paralyzed.at[1].set(True),
                          retreating=state.retreating.at[2].set(True))
    out, ts = advance(state, jnp.int32(DEFEND))
    assert out.actor == 3 and not out.paralyzed[1] and out.escaped[2]
    assert out.step_count == state.step_count+1 and ts.extras['automatic_battle_transition'] == 2
    assert jnp.isclose(ts.reward, -env.step_cost-.05*env.recruitment.composition(out)[2])
    # An injected automatic state is drained for free; production steps never return it.
    automatic = state.replace(actor=jnp.int32(1), turn_phase=state.turn_phase.at[0].set(2))
    drained, ts = advance(automatic, jnp.int32(CONTINUE))
    assert drained.actor == 3 and drained.step_count == automatic.step_count and ts.reward == 0


@pytest.mark.core
def test_final_decision_finishes_responses_and_keeps_terminal_reward(current_game):
    env, initial, advance, _ = current_game
    state = decision_battle(env, initial).replace(step_count=jnp.int32(env.max_steps-1))
    out, ts = advance(state, jnp.int32(DEFEND))
    assert out.actor == 1 and out.battle_steps == state.battle_steps+3
    assert out.done and ts.truncated() and out.step_count == env.max_steps
    # With no active blue unit after its escape, retreat and RED's final phase finish inside the decision.
    state = state.replace(step_count=jnp.int32(0), hp=state.hp.at[:6].set(0).at[0].set(1000))
    out, ts = advance(state, jnp.int32(RETREAT))
    assert not out.in_battle and not out.done and out.last_event == WITHDRAW and out.step_count == 1
    assert ts.extras['automatic_battle_transition'] > 0
    # Enemy escape resolves victory inside the same decision, without a pending healer.
    state = decision_battle(env, initial)
    state = state.replace(hp=state.hp.at[3].set(0).at[7].set(0),
                          retreating=state.retreating.at[6].set(True))
    out, ts = advance(state, jnp.int32(DEFEND))
    assert not out.in_battle and not out.alive[0] and not out.done
    assert out.step_count == state.step_count+1 and ts.extras['battle_victory']
    assert jnp.isclose(ts.reward, 1-env.step_cost-.05*env.recruitment.composition(out)[2])
    # The enemy's response can also kill the last survivor and terminate this decision.
    state = decision_battle(env, initial)
    state = state.replace(hp=state.hp.at[:6].set(0).at[0].set(1),
                          potion_bonus=state.potion_bonus.at[6:, 2].set(100))
    out, ts = advance(state, jnp.int32(DEFEND))
    assert out.lost and out.done and ts.last() and not ts.truncated()
    assert out.step_count == state.step_count+1 and ts.extras['enemy_battle_transition'] > 0
    assert jnp.isclose(ts.reward, -1-env.step_cost-.05*env.recruitment.composition(out)[2])
    # The copy preparation branch must also finish RED's response before truncation.
    state = decision_battle(env, initial)
    ids = state.unit_ids.at[jnp.array([0, 6])].set(env.progression.ids['doppelganger'])
    levels = env.progression.base_levels[ids]
    state = state.replace(unit_ids=ids, native_ids=ids, unit_levels=levels, native_levels=levels,
        round=jnp.int32(0), preparation=jnp.zeros(12, bool).at[jnp.array([0, 6])].set(True),
        step_count=jnp.int32(env.max_steps-1))
    out, ts = advance(state, jnp.int32(COPY_ALLY+1))
    assert out.done and ts.truncated() and out.step_count == env.max_steps
    assert out.round == 1 and not jnp.any(out.preparation) and out.copied[0] and out.copied[6]
    assert ts.extras['player_battle_transition'] == 1 and ts.extras['automatic_battle_transition'] >= 1


@pytest.fixture(scope='module')
def env():
    return basic_environment()


def test_ten_moves_exhaust_budget_without_income_or_automatic_turn(env):
    state, initial = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    assert state.gold == state.map_steps == 0 and state.day == 1
    chex.assert_trees_all_close(initial.observation[-3:], jnp.array([0., 1., .01]))
    actions = jnp.tile(jnp.array([2, 6], jnp.int32), 6)
    @jax.jit
    def rollout(start):
        def step(carry, action):
            following, ts = env.step(carry, action)
            return following, (following.movement_points, following.gold, following.day,
                               ts.observation[-3:])
        return jax.lax.scan(step, start, actions)
    final, (points, gold, days, economy) = rollout(state)
    expected = jnp.maximum(20 - 2*jnp.arange(1, 13), 0)
    chex.assert_trees_all_equal(points, expected)
    chex.assert_trees_all_equal(gold, jnp.zeros(12, jnp.int32))
    chex.assert_trees_all_equal(days, jnp.ones(12, jnp.int32))
    chex.assert_trees_all_close(economy[:, 1], expected / 20.)
    chex.assert_trees_all_close(economy[:, 2], jnp.full(12, .01))
    assert final.map_steps == 10 and not final.in_battle
    chex.assert_trees_all_equal(final.position, state.position)
    mask = compiled_method(env,'action_mask')(final)
    assert not jnp.any(mask[:8]) and mask[REST]


def test_rest_penalty_is_linear_for_every_remaining_budget_and_free_at_zero(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    points = jnp.arange(0, 21, 2, dtype=jnp.int32)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (11,)+x.shape), state)
    states = states.replace(movement_points=points, built_today=jnp.ones(11, jnp.bool_))
    assert jnp.all(compiled_method(env,'action_mask',batched=True)(states)[:, REST])
    following, ts = compiled_method(env,'step',batched=True)(states, jnp.full(11, REST, jnp.int32))
    chex.assert_trees_all_close(ts.reward, -.001*points)
    assert float(ts.reward[0]) == 0.
    chex.assert_trees_all_close(ts.extras['rest_penalty'], .001*points)
    assert jnp.all(ts.extras['turn_ended']) and not jnp.any(ts.extras['building_constructed'])
    chex.assert_trees_all_equal(following.movement_points, jnp.full(11, 20, jnp.int32))
    chex.assert_trees_all_equal(following.gold, jnp.full(11, 100, jnp.int32))
    chex.assert_trees_all_equal(following.day, jnp.full(11, 2, jnp.int32))
    chex.assert_trees_all_equal(following.last_event, jnp.full(11, RESTED, jnp.int32))
    assert not jnp.any(following.built_today)
    for field in ('position', 'hp', 'alive', 'map_steps', 'battle_key', 'buildings', 'blocked_buildings'):
        chex.assert_trees_all_equal(getattr(following, field), getattr(states, field))


def test_every_direction_costs_two_with_last_partial_move(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (24,)+x.shape), state)
    # All eight neighboring cells of the initial position are free.
    points = jnp.repeat(jnp.array([2, 1, 0], jnp.int32), 8)
    states = states.replace(movement_points=points)
    actions = jnp.tile(jnp.arange(8, dtype=jnp.int32), 3)
    masks = compiled_method(env,'action_mask',batched=True)(states)
    chex.assert_trees_all_equal(masks[jnp.arange(24), actions], points > 0)
    following, _ = compiled_method(env,'step',batched=True)(states, actions)
    chex.assert_trees_all_equal(following.position[:16], jnp.tile(state.position + env.directions,(2,1)))
    chex.assert_trees_all_equal(following.position[16:], states.position[16:])
    chex.assert_trees_all_equal(following.movement_points, jnp.maximum(0,points-2))


def test_invalid_actions_battle_and_terminal_states_cannot_advance_the_day(env):
    world, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    world = world.replace(day=jnp.int32(101))  # Day observations must not clip at 100.
    battle = compiled_method(env,'_begin_battle')(world.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(0))
    states = (
        world.replace(position=jnp.array([1, 1], jnp.int32)),
        world.replace(position=jnp.array([3, 7], jnp.int32), movement_points=jnp.int32(0)),
        world, world, world.replace(done=jnp.bool_(True)),
        battle, battle, battle.replace(actor=jnp.int32(6)), battle,
    )
    actions = jnp.array([0, 2, -1, 44, REST, SHOOT, DEFEND, CONTINUE, REST], jnp.int32)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    following, ts = compiled_method(env,'step',batched=True)(batched, actions)
    for field in ('gold', 'map_steps', 'day', 'movement_points', 'built_today'):
        chex.assert_trees_all_equal(getattr(following, field), getattr(batched, field))
    assert not jnp.any(ts.extras['turn_ended'])
    chex.assert_trees_all_close(ts.observation[:, -1], jnp.full(len(states), 1.01))
    assert not compiled_method(env,'action_mask')(battle)[REST] and not compiled_method(env,'action_mask')(states[4])[REST]
    chex.assert_trees_all_equal(following.hp[-1], battle.hp)
    chex.assert_trees_all_equal(following.battle_key[-1], battle.battle_key)


def test_last_points_can_attack_and_recovery_does_not_restore_movement(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    state = state.replace(position=jnp.array([3, 7], jnp.int32), movement_points=jnp.int32(2))
    state, _ = compiled_method(env,'step')(state, jnp.int32(2))
    assert state.in_battle and state.gold == 0 and state.movement_points == 0
    state = state.replace(actor=jnp.int32(0))
    won, _ = compiled_method(env,'_battle_step')(state, jnp.int32(SHOOT), state.battle_key, jnp.zeros(36))
    escaping = state.replace(hp=state.hp.at[1:6].set(0), retreating=state.retreating.at[0].set(True))
    escaped, _ = compiled_method(env,'_battle_step')(escaping, jnp.int32(CONTINUE), escaping.battle_key, jnp.zeros(36))
    for recovered in (won, escaped):
        assert not recovered.in_battle and recovered.day == 1 and recovered.map_steps == 0
        assert recovered.movement_points == recovered.gold == 0
        assert not jnp.any(compiled_method(env,'action_mask')(recovered)[:8]) and compiled_method(env,'action_mask')(recovered)[REST]


def test_repeated_rest_and_exhaustion_cycles_pay_once_per_rest(env):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    actions = jnp.concatenate((jnp.array([REST, REST], jnp.int32),
                               jnp.tile(jnp.concatenate((jnp.tile(jnp.array([2, 6], jnp.int32), 5),
                                                         jnp.array([REST], jnp.int32))), 2)))
    def step(carry, action):
        following, ts = env.step(carry, action)
        return following, ts.reward
    final, rewards = jax.jit(lambda start: jax.lax.scan(step, start, actions))(state)
    assert final.day == 5 and final.gold == 400 and final.map_steps == 20
    assert final.movement_points == 20
    chex.assert_trees_all_close(rewards[:2], jnp.full(2, -.02))
    assert rewards[12] == rewards[23] == 0


def test_training_autoreset_rest_keeps_terminal_income_and_new_turn_observation(current_game, training_autoreset):
    training, _, advance = training_autoreset
    state, _ = compiled_method(training,'reset')(jax.random.split(jax.random.PRNGKey(7), 2))
    state = replace_base_state(state, movement_points=jnp.zeros(2,jnp.int32))
    state, ts = advance(state, jnp.full(2, REST, jnp.int32))
    assert jnp.all(ts.truncated())
    chex.assert_trees_all_equal(state.gold, jnp.zeros(2, jnp.int32))
    chex.assert_trees_all_equal(state.day, jnp.ones(2, jnp.int32))
    chex.assert_trees_all_equal(state.movement_points, jnp.full(2, 20, jnp.int32))
    chex.assert_trees_all_equal(ts.reward, jnp.zeros(2))
    base_env, base_state, _, _ = current_game
    delta = compiled_method(base_env,'observation')(base_state.replace(gold=jnp.int32(100),movement_points=jnp.int32(0)))-compiled_method(base_env,'observation')(base_state)
    columns = jnp.concatenate((jnp.nonzero(delta > 0,size=1)[0],jnp.nonzero(delta < 0,size=1)[0]))
    chex.assert_trees_all_close(ts.extras['next_obs']['observation'][:, columns], jnp.array([[.1, 1.], [.1, 1.]]))
    chex.assert_trees_all_equal(ts.observation['observation'][:, columns], jnp.array([[0., 1.], [0., 1.]]))
    chex.assert_trees_all_close(ts.extras['next_obs']['observation'][:, -1], jnp.full(2, .02))
    chex.assert_trees_all_close(ts.observation['observation'][:, -1], jnp.full(2, .01))
    assert jnp.all(ts.observation['action_mask'][:, REST])


def test_bad_rest_penalty_settings_are_rejected():
    for penalty in (-.1, float('nan'), float('inf'), True, '0.001'):
        with pytest.raises(ValueError, match='rest_penalty_per_point'):
            NumberGrid(map_config={**MAP, 'rest_penalty_per_point': penalty})


@pytest.mark.core
def test_hero_level_movement_cap_rest_boots_and_attack_cost(current_game):
    env, initial, advance, _ = current_game
    hero = int(env._equipment_hero(initial))
    caps = compiled_method(env,'movement_cap')
    masks = compiled_method(env,'action_mask')
    quotes = compiled_method(env,'map_commands')
    for level, bonus in ((1,0),(2,1),(9,8),(10,9),(11,9),(30,9)):
        state = initial.replace(unit_levels=initial.unit_levels.at[hero].set(level),
                                movement_points=jnp.int32(0))
        assert caps(state) == 20+bonus
        rested, ts = advance(state,jnp.int32(REST))
        assert rested.movement_points == 20+bonus
        assert jnp.isclose(ts.reward, -.05*env.recruitment.composition(rested)[2])
        # Rest preserves the encoded current movement, including an odd cap.
        delta = compiled_method(env,'observation')(rested)-compiled_method(env,'observation')(
            rested.replace(movement_points=jnp.int32(0)))
        assert jnp.isclose(jnp.max(delta), (20+bonus)/20.)
        enemy = rested.replace(position=env.opponent_positions[0]-jnp.array([0,1]))
        assert masks(enemy)[2] and quotes(enemy)[2,1] == (21+bonus)//2
        attacked, _ = advance(enemy,jnp.int32(2))
        assert attacked.in_battle and attacked.movement_points == (20+bonus)//2
        # A temporary combat form does not erase the native hero's level bonus.
        transformed = attacked.replace(unit_ids=attacked.unit_ids.at[hero].set(0),
                                       unit_levels=attacked.unit_levels.at[hero].set(0))
        assert caps(transformed) == 20+bonus
        # Boots still use the base allowance; they do not multiply the level bonus.
        boots = env.item_rules.keys.index('boots_speed')
        equipped = state.replace(equipped=state.equipped.at[4].set(boots))
        assert caps(equipped) == 24+bonus
    high_unit = initial.replace(unit_levels=initial.unit_levels.at[0].set(50))
    assert caps(high_unit) == 20
    no_hero = high_unit.replace(unit_ids=high_unit.unit_ids.at[hero].set(0))
    assert caps(no_hero) == 20


def test_initial_high_level_hero_and_no_equipment_movement(current_game):
    import copy
    env, initial, _, _ = current_game
    hero = int(env._equipment_hero(initial))
    high = copy.copy(env)
    high._test_jit_methods = {}
    high.progression = copy.copy(env.progression)
    high.progression.initial_levels = env.progression.initial_levels.at[hero].set(10)
    state, _ = compiled_method(high,'reset')(jax.random.PRNGKey(5))
    assert state.movement_points == 29 and high.movement_cap(state) == 29
    plain = copy.copy(high)
    plain._test_jit_methods = {}
    plain.items_enabled = False
    assert compiled_method(plain,'movement_cap')(state) == 29
