"""Basic combat characteristics, Python-reference arithmetic and turn scheduling."""
from stoix.tests.number_grid_fixtures import compiled_method, basic_environment
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.tests.number_grid_fixtures import MAP, hero_roster_state
from stoix.envs.number_grid import (
    NumberGrid, SHOOT, DEFEND, WAIT, RETREAT, CONTINUE, HIT, MISS,
)
from stoix.envs.number_grid_combat import accuracy_hits, HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE


def battle(hero=None, enemy=None, **overrides):
    game_map = copy.deepcopy(MAP)
    game_map['hero_combat_stats'] = hero or [{} for _ in range(6)]
    if enemy:
        game_map['enemy_combat_stats'] = [[{} for _ in range(n)] for n in MAP['enemy_units']]
        game_map['enemy_combat_stats'][11] = enemy
    game_map.update(overrides)
    env = NumberGrid(map_config=game_map)
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(11)))
    return env, state.replace(actor=jnp.int32(0))


def rolls(first=0., second=0., bonus=0.):
    return (jnp.zeros(36).at[12:18].set(first).at[18:24].set(second)
            .at[24:30].set(bonus))


def attack(env, state, values, action=SHOOT):
    return env._battle_step(state, jnp.int32(action), state.battle_key, values)[0]


def test_default_stats_and_observation_are_individual_and_finite():
    env = basic_environment()
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == env.observation_space().shape == (271,)
    np.testing.assert_array_equal(env.unit_stats(state)[:, HP], env.max_hp(state))
    np.testing.assert_array_equal(env.unit_stats(state)[1], [100, 25, 80, 0, 50])
    np.testing.assert_array_equal(env.unit_stats(state)[5], [45, 20, 80, 0, 60])
    assert np.all(env.unit_stats(state)[6:] == 0)
    env, state = battle(hero=[dict(max_hp=120, damage=40, accuracy=75, armor=30, initiative=55)] + [{}]*5)
    obs = np.asarray(env.observation(state))[124:232].reshape(12, 9)
    np.testing.assert_allclose(obs[0, 4:], [1.2, 40/300, .75, .3, .55])
    assert obs[0, 0] == 1 and np.isfinite(obs).all()
    assert env.max_hp(state)[0] == 120


def test_non_progression_recognizes_russian_unit_names():
    game_map = copy.deepcopy(MAP)
    # Catalog units carry the Russian names and unit types; overrides cannot set them.
    # Niddog is large in the catalog, so it is shrunk to fit a full six-unit formation.
    game_map['hero_roster'] = ['abbess', 'prophetess', 'niddog', 'mermaid', 'dark_elf_gast', 'spectre']
    game_map['hero_combat_stats'] = [{}, {}, dict(size=1), {}, {}, {}]
    env = NumberGrid(map_config=game_map)
    assert not env.progression_enabled
    np.testing.assert_array_equal(env.mass_cures[0, :6], [True, True, False, False, False, False])
    np.testing.assert_array_equal(env.cached_poisoners[0, :6], [False, False, True, False, False, False])
    np.testing.assert_array_equal(env.poisoners[0, :6], [False, False, True, False, False, False])
    np.testing.assert_array_equal(env.secondary_paralysis_modes[0, :6], [0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(env.ghost_modes[0, :6], [0, 0, 0, 0, 2, 1])
    assert env.has_cures and env.has_poisoners and env.has_secondary_paralysis and env.has_paralysis


@pytest.mark.core
@pytest.mark.parametrize('chance,count', [(0, 0), (50, 5050), (80, 9220), (100, 10000)])
def test_two_roll_accuracy_matches_exhaustive_discrete_distribution(chance, count):
    values = (jnp.arange(100) + .25) / 100
    hits = jax.jit(accuracy_hits)(chance, values[:, None], values[None, :])
    assert int(jnp.sum(hits)) == count
    expected = (np.arange(100)[:, None] + np.arange(100)[None, :]) / 2 < chance
    np.testing.assert_array_equal(hits, expected)


@pytest.mark.core
@pytest.mark.parametrize('damage,armor,bonus,defended,expected', [
    (25, 0, 0, False, 25), (25, 0, 5, False, 30),
    (25, 0, 0, True, 12), (25, 0, 2, True, 14),
    (25, 40, 5, False, 18), (25, 40, 5, True, 9),
    (15, 70, 0, False, 5),  # Python float/round: 4.500000000000001, not 4.
    (300, 90, 5, True, 15), (400, 100, 5, False, 30),
    (0, 0, 5, False, 0),
])
def test_damage_bonus_cap_armor_defend_and_python_rounding(current_game, damage, armor, bonus, defended, expected):
    env, initial, _, fight = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    # Dynamic combat modifiers cover the same arithmetic without constructing
    # ten legacy maps with different baked-in stats. The base Squire has armor 0.
    state = state.replace(actor=jnp.int32(0),hp=state.hp.at[6].set(1000),
        primary_override=state.primary_override.at[0].set(damage),
        potion_bonus=state.potion_bonus.at[6,ARMOR].set(armor),
        defended=state.defended.at[6].set(defended))
    values = jnp.pad(rolls(bonus=(bonus+.25)/6), (0,env.random_size-36))
    result,_ = fight(state,jnp.int32(SHOOT),state.battle_key,values)
    assert result.last_damage == expected and result.hp[6] == 1000 - expected
    assert result.last_event == HIT


def test_mage_rolls_hit_and_damage_independently_for_each_target():
    env, state = battle(enemy=[dict(max_hp=100)]*6)
    state = state.replace(actor=jnp.int32(5))
    random = rolls().at[12:18].set(jnp.array([0, .99, 0, .99, 0, .99]))
    random = random.at[18:24].set(random[12:18]).at[24:30].set((jnp.arange(6)+.25)/6)
    result = attack(env, state, random)
    np.testing.assert_array_equal(result.hp[6:], [80, 100, 78, 100, 76, 100])
    assert result.last_damage == 66 and result.last_target == -1
    other_target = attack(env, state, random, SHOOT+5)
    np.testing.assert_array_equal(other_target.hp, result.hp)


def test_hp_clamps_at_zero_miss_does_not_damage_and_dead_unit_leaves_queue():
    env, state = battle()
    state = state.replace(hp=state.hp.at[6].set(1))
    result = attack(env, state, rolls())
    assert result.hp[6] == 0 and result.last_damage == 1 and result.actor != 6
    missed = attack(env, state, rolls(.99, .99))
    np.testing.assert_array_equal(missed.hp, state.hp)
    assert missed.last_event == MISS and missed.turn_phase[0] == 2


def test_initiative_is_additive_rerolled_and_resolves_equal_values_without_side_bias():
    env, state = battle()
    low = env._round_priority(jnp.zeros(36), 11)
    high = env._round_priority(jnp.full(36, .999), 11)
    np.testing.assert_array_equal(jnp.floor(low), env.unit_stats(state)[:, INITIATIVE])
    np.testing.assert_array_equal(jnp.floor(high), env.unit_stats(state)[:, INITIATIVE] + 9)
    assert high[1] < low[0]  # 50+9 cannot outrun 60+0.
    r = jnp.zeros(36).at[0].set(.04).at[6].set(.06)
    assert int(jnp.argmax(env._round_priority(r, 11))) == 6
    assert int(jnp.argmax(env._round_priority(r.at[0].set(.08), 11))) == 0
    # Completing the round draws a new priority vector, even if the last unit waited.
    state = state.replace(turn_phase=jnp.full(12, 2).at[0].set(1))
    result = attack(env, state, jnp.full(36, .5), DEFEND)
    assert result.round == 2
    np.testing.assert_array_equal(result.priority, env._round_priority(jnp.full(36, .5), 11))


@pytest.mark.core
def test_wait_is_once_per_round_and_reverses_the_rolled_queue(current_game):
    env, initial, _, fight = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    def attack(env, state, values, action):
        values = jnp.pad(values, (0, env.random_size-values.size))
        return fight(state, jnp.int32(action), state.battle_key, values)[0]
    mask = compiled_method(env,'action_mask')
    state = state.replace(hp=jnp.array([45, 100, 45, 0, 0, 0, 25, 0, 0, 0, 0, 0]),
                          priority=jnp.array([160., 150., 140., 0, 0, 0, 130., 0, 0, 0, 0, 0]))
    for actor in (0, 1, 2):
        assert state.actor == actor
        state = attack(env, state, rolls(), WAIT)
    assert state.actor == 6
    state = attack(env, state, rolls(.99, .99), CONTINUE)
    for actor in (2, 1, 0):
        assert state.actor == actor and not mask(state)[WAIT]
        rejected, _ = compiled_method(env,'step')(state, jnp.int32(WAIT))
        assert rejected.actor == actor
        np.testing.assert_array_equal(rejected.battle_key, state.battle_key)
        state = attack(env, state, rolls(), DEFEND)
    assert state.round == 2 and mask(state)[WAIT]


@pytest.mark.core
def test_defend_survives_round_boundary_and_expires_at_next_activation(current_game):
    env, initial, _, fight = current_game
    initial = hero_roster_state(env,initial,['archer','squire',None,None,None,None])
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    def attack(env, state, values, action):
        values = jnp.pad(values, (0, env.random_size-values.size))
        return fight(state, jnp.int32(action), state.battle_key, values)[0]
    state = state.replace(hp=jnp.array([45, 100, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0]),
                          priority=jnp.array([60., 50., 0, 0, 0, 0, 61., 0, 0, 0, 0, 0]))
    state = attack(env, state, rolls(), DEFEND)
    assert state.defended[0]
    state = attack(env, state, rolls(.99, .99), CONTINUE)  # enemy misses
    assert state.actor == 1 and state.defended[0]
    state = attack(env, state, rolls(), DEFEND)
    assert state.round == 2 and not state.defended[0] and state.defended[1]
    state = attack(env, state, rolls(), WAIT)
    assert not state.defended[0]  # waiting cannot extend the previous defence.


def test_enemy_uses_its_own_accuracy_and_lowest_hp_target():
    env, state = battle(hero=[dict(max_hp=10, armor=90), dict(max_hp=20)] + [{}]*4,
                        enemy=[dict(damage=20, accuracy=0)] + [{}]*5,
                        enemy_warrior_slots=[-1]*24)
    state = state.replace(actor=jnp.int32(6))
    assert env._enemy_action(state, jnp.zeros(6)) == SHOOT
    result = attack(env, state, rolls(), CONTINUE)
    np.testing.assert_array_equal(result.hp, state.hp)
    assert result.last_event == MISS


def test_recovery_restores_individual_hp_and_retreat_does_not_grant_extra_attack():
    env, state = battle(hero=[dict(max_hp=120)] + [{}]*5)
    state = state.replace(hp=state.hp.at[6:].set(jnp.array([1, 0, 0, 0, 0, 0])))
    result = attack(env, state, rolls())
    np.testing.assert_array_equal(result.hp[:6], [120, 100, 45, 45, 45, 45])
    assert not result.in_battle
    state = state.replace(actor=jnp.int32(0), turn_phase=state.turn_phase.at[0].set(1))
    result = attack(env, state, rolls(), RETREAT)
    assert result.retreating[0] and result.turn_phase[0] == 2


@pytest.mark.parametrize('change', [
    dict(max_hp=0), dict(max_hp=True), dict(damage=-1), dict(damage=float('nan')),
    dict(accuracy=101), dict(armor=-1), dict(initiative=-1), dict(initiative=1.5), dict(immunity=1),
])
def test_invalid_unit_stats_fail_before_training(change):
    with pytest.raises(ValueError):
        battle(hero=[change]+[{}]*5)


def test_versions_and_override_shapes_are_validated():
    for change in [dict(combat_rules_version=3), dict(battle_observation_version=3),
                   dict(hero_combat_stats=[]), dict(enemy_combat_stats=[[]])]:
        with pytest.raises(ValueError):
            NumberGrid(map_config={**MAP, **change})
    saved = json.loads((Path(__file__).resolve().parents[2] / 'maps/number_grid-24x24-v10-enemy-warriors.json').read_text())
    env = NumberGrid(map_config=saved)
    assert not env.basic_combat and env.reset(jax.random.PRNGKey(0))[1].observation.shape == (178,)


@pytest.mark.core
def test_every_current_squad_is_reachable_for_an_attack(current_game):
    from collections import deque
    from stoix.envs.number_grid import DIRECTIONS
    env, initial, _, _ = current_game
    position = tuple(env.map_config['agent_position'])
    opponents = [tuple(p) for p in env.map_config['opponent_positions']]
    blocked = set(env.territory.obstacle_cells)
    ruin_approaches = {tuple(r['entrance']): set(r['interaction_tiles']) for r in env.ruins.ruins}
    remaining = set(range(len(opponents)))
    origins, actions, enemies, living = [], [], [], []
    # Map preparation only: the planner is never passed into PPO. This checks
    # geometry and attack commands, not a fictitious guaranteed combat win.
    while remaining:
        occupied = {opponents[i]: i for i in sorted(remaining,reverse=True)}
        queue, seen, found = deque([position]), {position}, None
        while queue and found is None:
            pos = queue.popleft()
            for action, (dr, dc) in enumerate(DIRECTIONS):
                dest = (pos[0]+dr, pos[1]+dc)
                ruin_attack = dest in occupied and pos in ruin_approaches.get(dest,set())
                if (dest in blocked and not ruin_attack) or not all(0 < x < env.size-1 for x in dest):
                    continue
                if dest in occupied:
                    found = pos, action, occupied[dest]
                    break
                if dest not in seen:
                    queue.append(dest)
                    seen.add(dest)
        assert found is not None, 'An enemy has no reachable adjacent attack cell'
        position, action, enemy = found
        living.append([i in remaining for i in range(len(opponents))])
        origins.append(position)
        actions.append(action)
        enemies.append(enemy)
        remaining.remove(enemy)
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(len(enemies),)+x.shape),initial)
    states = states.replace(position=jnp.array(origins,jnp.int32),alive=jnp.array(living,bool))
    commands = compiled_method(env,'map_commands',batched=True)(states)
    selected = commands[jnp.arange(len(enemies)),jnp.array(actions,jnp.int32)]
    np.testing.assert_array_equal(selected[:,0],enemies)
    np.testing.assert_array_equal(selected[:,1],10)


def test_current_map_full_route_through_step_combat_and_rest(current_game):
    """Exercise all 46 encounters and REST under controlled hits, not a PPO win claim."""
    from collections import deque
    from stoix.envs.number_grid import DIRECTIONS, MOVE_COST, REST
    env, initial, _, _ = current_game
    position = tuple(env.map_config['agent_position'])
    opponents = [tuple(p) for p in env.map_config['opponent_positions']]
    blocked = set(env.territory.obstacle_cells)
    ruin_approaches = {tuple(r['entrance']): set(r['interaction_tiles']) for r in env.ruins.ruins}
    remaining, route = set(range(len(opponents))), []
    while remaining:
        occupied = {opponents[i]: i for i in sorted(remaining,reverse=True)}
        queue, seen, found = deque([(position, [])]), {position}, None
        while queue and found is None:
            pos, path = queue.popleft()
            for action, (dr, dc) in enumerate(DIRECTIONS):
                dest = (pos[0]+dr, pos[1]+dc)
                ruin_attack = dest in occupied and pos in ruin_approaches.get(dest,set())
                if (dest in blocked and not ruin_attack) or not all(0 < x < env.size-1 for x in dest):
                    continue
                if dest in occupied:
                    found = pos, path+[action], occupied[dest]
                    break
                if dest not in seen:
                    seen.add(dest)
                    queue.append((dest, path+[action]))
        assert found is not None, 'An enemy cannot be reached'
        position, path, enemy = found
        route.extend(path)
        remaining.remove(enemy)
        if opponents[enemy] in {tuple(c['position']) for c in env.map_config.get('cities',[])} and all(opponents[i] != opponents[enemy] for i in remaining):
            route.append(path[-1])  # Enter only after both defending armies are gone.
            position = opponents[enemy]
    route = jnp.array(route, jnp.int32)
    # A private shallow copy preserves current_game's immutable shared JIT cache.
    # Only the random outcomes are controlled; movement, combat, recovery and
    # terminal transitions all go through the public step on the current map.
    controlled = copy.copy(env)
    def possible_outcomes(state, action, key, values):
        values = jnp.full_like(values, jnp.where(state.actor < 6, 0., .999))
        return env._battle_step(state, action, key, values)
    controlled._battle_step = possible_outcomes

    @jax.jit
    def run(state):
        def advance(carry):
            state, index, battles, rests, legal = carry
            mask = env.action_mask(state)
            targets = mask[SHOOT:DEFEND]
            target = jnp.argmin(jnp.where(targets, state.hp[6:], jnp.iinfo(jnp.int32).max))
            attack = jnp.where(jnp.any(targets), SHOOT+target, DEFEND)
            attack = jnp.where((state.post_victory > 0) & (state.actor < 6), WAIT, attack)
            attack = jnp.where(mask[CONTINUE], CONTINUE, attack)
            maximum = env.max_hp(state)[:6]
            needs_rest = (state.movement_points < MOVE_COST) | jnp.any((state.hp[:6] > 0) & (state.hp[:6] < maximum))
            move = jnp.where(needs_rest, REST, route[jnp.minimum(index, len(route)-1)])
            action = jnp.where(state.in_battle, attack, move)
            following, ts = controlled.step(state, action)
            rested = ~state.in_battle & (action == REST)
            rest_ok = ((following.day == state.day+1) & (following.gold == state.gold+env.territory.income(following)[0])
                       & (following.movement_points == env.movement_cap(following)) & ~following.in_battle)
            legal &= mask[action] & (~rested | rest_ok) & jnp.all(jnp.isfinite(ts.observation))
            index += (~state.in_battle & (action < 8)).astype(jnp.int32)
            battles += jnp.sum(state.alive & ~following.alive)
            rests += rested.astype(jnp.int32)
            return following, index, battles, rests, legal
        return jax.lax.while_loop(lambda carry: ~carry[0].done, advance,
            (state, jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(True)))

    final, index, battles, rests, legal = run(initial)
    assert legal and final.won and not final.lost and not final.in_battle
    assert int(index) == len(route) and int(battles) == env.num_opponents
    assert not np.any(final.alive) and np.all(final.city_owned) and int(rests) > 0
    assert np.all(final.ruin_looted) and final.potions[env.potion_rules.keys.index('strength')] == 1
    occupied = np.asarray(initial.hp[:6]) > 0
    assert np.all(np.asarray(final.hp[:6])[occupied] > 0)
    assert np.all(final.hp[:6] <= env.max_hp(final)[:6])
