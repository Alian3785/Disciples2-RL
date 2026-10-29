"""Basic combat characteristics, original damage arithmetic and turn scheduling."""
from stoix.tests.number_grid_fixtures import compiled_method, basic_environment
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.tests.number_grid_fixtures import MAP, hero_roster_state, enemy_roster_state
from stoix.envs.number_grid import (
    NumberGrid, SHOOT, DEFEND, WAIT, RETREAT, CONTINUE, HIT, MISS,
)
from stoix.envs.number_grid_combat import (
    accuracy_hits, armor_with_defend, build_combat_tables, HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE,
)


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
    assert ts.observation.shape == env.observation_space().shape == (272,)
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
@pytest.mark.parametrize('chance,count', [
    (0, 3), (10, 253), (25, 1378), (50, 5247), (60, 6997),
    (70, 8347), (80, 9297), (90, 9847), (98, 9999), (99, 10000), (100, 10000),
])
def test_two_roll_accuracy_matches_exhaustive_discrete_distribution(chance, count):
    """Installed EXE 0x625F3F; independent discrete counts in docs/COMBAT.md."""
    values = (jnp.arange(100) + .25) / 100
    hits = jax.jit(accuracy_hits)(chance, values[:, None], values[None, :])
    assert int(jnp.sum(hits)) == count
    expected = (np.arange(100)[:, None] + np.arange(100)[None, :]) // 2 <= chance
    np.testing.assert_array_equal(hits, expected)


@pytest.mark.core
def test_accuracy_includes_equal_and_half_integer_boundaries():
    """Truncated averages at the accuracy boundary hit; the next integer misses."""
    pairs = jnp.array([[60, 99], [80, 80], [61, 99], [80, 81], [62, 99],
                       [81, 81], [63, 99], [99, 99]], jnp.float32)
    draws = (pairs+.25)/100
    hit = jax.jit(accuracy_hits)
    assert hit(80, draws[:, 0], draws[:, 1]).tolist() == [True]*5+[False]*3
    assert bool(hit(99, jnp.float32(.9925), jnp.float32(.9925)))


@pytest.mark.core
@pytest.mark.parametrize('damage,armor,bonus,defended,expected', [
    (25, 0, 0, False, 25), (25, 0, 5, False, 30),
    (25, 0, 0, True, 13), (25, 0, 2, True, 14),
    (25, 50, 0, False, 13), (25, 30, 0, False, 18),
    (1, 90, 0, False, 1), (1, 90, 0, True, 1),
    (25, 40, 5, False, 18), (25, 40, 5, True, 9),
    (15, 70, 0, False, 5),  # 15 - trunc(15*70/100), independent of float ties.
    (299, 0, 5, False, 300), (300, 0, 5, False, 300),
    (300, 90, 5, True, 30), (400, 100, 5, False, 30),
    (100, 0, 0, True, 50), (100, 35, 0, True, 33), (100, 50, 0, True, 25),
    (100, 79, 0, True, 11), (100, 80, 0, True, 10),
    (100, 85, 0, True, 10), (100, 90, 0, True, 10),
    (0, 0, 5, False, 0), (0, 90, 5, True, 0),
])
def test_damage_bonus_cap_armor_defend_and_original_rounding(current_game, damage, armor, bonus, defended, expected):
    env, initial, _, fight = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    # Dynamic combat modifiers cover the same arithmetic without constructing
    # separate legacy maps with different baked-in stats. The base Squire has armor 0.
    state = state.replace(actor=jnp.int32(0),hp=state.hp.at[6].set(1000),
        primary_override=state.primary_override.at[0].set(damage),
        potion_bonus=state.potion_bonus.at[6,ARMOR].set(armor),
        defended=state.defended.at[6].set(defended))
    values = jnp.pad(rolls(bonus=(bonus+.25)/6), (0,env.random_size-36))
    result,_ = fight(state,jnp.int32(SHOOT),state.battle_key,values)
    assert result.last_damage == expected and result.hp[6] == 1000 - expected
    assert result.last_event == HIT


@pytest.mark.core
@pytest.mark.parametrize('key,power,expected', [
    ('duke', 399, 400), ('duke', 400, 400), ('duke', 600, 400),
    ('g000uu0072', 299, 300), ('g000uu0072', 600, 300),  # Mage lacks Heavy Strike.
])
def test_final_damage_cap_uses_attacker_profile(current_game, key, power, expected):
    env, initial, _, fight = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    uid = env.progression.ids[key]
    state = state.replace(actor=jnp.int32(1), unit_ids=state.unit_ids.at[1].set(uid),
        unit_levels=state.unit_levels.at[1].set(env.progression.base_levels[uid]),
        hp=state.hp.at[6:].set(0).at[6].set(1000),
        primary_override=state.primary_override.at[1].set(power))
    values = jnp.zeros(env.random_size).at[24:30].set(.99)
    result, _ = fight(state, jnp.int32(SHOOT), state.battle_key, values)
    assert result.last_damage == expected and result.hp[6] == 1000-expected


@pytest.mark.core
@pytest.mark.parametrize('actor,target,power,weakened,expected', [
    (0, 6, 600, False, 150), (0, 6, 600, True, 150),
    (0, 6, 400, True, 137),  # round(400*.67)+5 = 273; armor absorbs 136.
    (6, 0, 600, True, 150),  # The same cap applies to the opponent.
])
def test_final_damage_cap_after_buffs_and_weakening_before_armor(
        current_game, actor, target, power, weakened, expected):
    env, initial, _, fight = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    intrinsic = env.progression.raw_stats(state.unit_ids, state.unit_levels)[actor, DAMAGE]
    state = state.replace(actor=jnp.int32(actor), hp=state.hp.at[target].set(1000),
        potion_bonus=state.potion_bonus.at[actor, DAMAGE].set(power-intrinsic).at[target, ARMOR].set(50),
        weakened=state.weakened.at[actor].set(weakened))
    if actor >= 6:
        # Make the desired front target uniquely weakest for the enemy AI.
        state = state.replace(hp=state.hp.at[1:3].set(2000))
    values = jnp.zeros(env.random_size).at[24:30].set(.99)
    result, _ = fight(state, jnp.int32(SHOOT), state.battle_key, values)
    assert result.last_damage == expected and result.hp[target] == 1000-expected


@pytest.mark.parametrize('power,hero,armor,defended,bonus,expected', [
    (25, False, 50, False, 0., 13),
    (25, False, 0, True, 0., 13),
    (300, False, 0, False, .99, 300),
    (400, True, 0, False, .99, 400),
    (400, True, 90, True, .99, 40),
])
def test_static_attacks_apply_final_cap_and_integer_armor(power, hero, armor, defended, bonus, expected):
    env, state = battle(hero=[dict(damage=power, hero=hero)]+[{}]*5,
                        enemy=[dict(max_hp=1000, armor=armor)]+[{}]*5)
    state = state.replace(defended=state.defended.at[6].set(defended))
    result, _ = compiled_method(env, '_battle_step')(
        state, jnp.int32(SHOOT), state.battle_key, rolls(bonus=bonus))
    assert result.last_damage == expected and result.hp[6] == 1000-expected


@pytest.mark.core
def test_defend_armor_cap_in_static_and_progression_tables(current_game):
    """DFENDBONUS=50 and sub_5D35F3: truncate the armor bonus before the 90 cap."""
    env = current_game[0]
    armors = [0, 35, 79, 80, 85, 90]
    config = copy.deepcopy(MAP)
    config['hero_combat_stats'] = [dict(damage=100, armor=armor) for armor in armors]
    _, damage_ids, armor_ids, table, *_ = build_combat_tables(config)
    for defended, expected in ((0, [100, 65, 21, 20, 15, 10]), (1, [50, 33, 11, 10, 10, 10])):
        np.testing.assert_array_equal(table[damage_ids[0, :6], armor_ids[0, :6], defended, 0],
                                      expected)
        np.testing.assert_array_equal(env.progression.damage_rolls[100, jnp.array(armors), defended, 0],
                                      expected)
    np.testing.assert_array_equal(
        jax.jit(armor_with_defend)(jnp.array(armors), jnp.ones(6, bool)),
        [50, 67, 89, 90, 90, 90])


@pytest.mark.core
@pytest.mark.parametrize('fort,shreds,expected', [(40, 0, 10), (40, 1, 13), (0, 1, 33)])
def test_defend_uses_armor_after_fortification_and_shatter(current_game, fort, shreds, expected):
    env, initial, _, fight = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    state = state.replace(actor=jnp.int32(0), hp=state.hp.at[6].set(1000),
        primary_override=state.primary_override.at[0].set(100),
        potion_bonus=state.potion_bonus.at[6, ARMOR].set(50),
        battle_fort_armor=jnp.int32(fort), armor_shreds=state.armor_shreds.at[6].set(shreds),
        defended=state.defended.at[6].set(True))
    result, _ = fight(state, jnp.int32(SHOOT), state.battle_key, jnp.zeros(env.random_size))
    assert result.last_damage == expected and result.hp[6] == 1000-expected


def test_nonprogression_defend_with_weakening():
    env, state = battle(
        hero=[{}, dict(damage=200), {}, {}],
        enemy=[dict(max_hp=1000, armor=90)] + [{}]*5,
        hero_roster=['tiamat', 'duke', 'possessed', None, 'novice', None], hero_units=4)
    assert env.has_weakening and env.has_powerups and not env.progression_enabled
    state = state.replace(actor=jnp.int32(1), defended=state.defended.at[6].set(True))
    fight = compiled_method(env, '_battle_step')
    for weakened, override, bonus, expected in ((False, -1, 0., 20), (True, -1, 0., 14),
                                               (False, 600, .99, 40)):
        before = state.replace(weakened=state.weakened.at[1].set(weakened),
                               primary_override=state.primary_override.at[1].set(override))
        values = jnp.zeros(env.random_size).at[24:30].set(bonus)
        result, _ = fight(before, jnp.int32(SHOOT), before.battle_key, values)
        assert result.last_damage == expected and result.hp[6] == 1000-expected


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
    state = state.replace(turn_phase=jnp.full(12, 2).at[0].set(1),priority=state.priority.at[0].set(-1.))
    result = attack(env, state, jnp.full(36, .5), DEFEND)
    assert result.round == 2
    np.testing.assert_array_equal(result.priority, env._round_priority(jnp.full(36, .5), 11))


@pytest.mark.core
@pytest.mark.parametrize('priorities,order', [
    ([160., 150., 140.], (0, 1, 2)),
    ([140., 160., 150.], (1, 2, 0)),
    ([150., 150., 150.], (0, 1, 2)),
])
def test_wait_is_once_per_round_and_preserves_wait_order(current_game, priorities, order):
    """EXE 0x65A73E appends waiters; initiative and equal rolls cannot reverse them."""
    env, initial, _, fight = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(order[0]))
    def attack(env, state, values, action):
        values = jnp.pad(values, (0, env.random_size-values.size))
        return fight(state, jnp.int32(action), state.battle_key, values)[0]
    mask = compiled_method(env,'action_mask')
    state = state.replace(hp=jnp.array([45, 100, 45, 0, 0, 0, 25, 0, 0, 0, 0, 0]),
                          priority=jnp.array([*priorities, 0, 0, 0, 130., 0, 0, 0, 0, 0]))
    for actor in order:
        assert state.actor == actor
        state = attack(env, state, rolls(), WAIT)
    assert state.actor == 6
    state = attack(env, state, rolls(.99, .99), CONTINUE)
    for actor in order:
        assert state.actor == actor and not mask(state)[WAIT]
        rejected, _ = compiled_method(env,'step')(state, jnp.int32(WAIT))
        assert rejected.actor == actor
        np.testing.assert_array_equal(rejected.battle_key, state.battle_key)
        state = attack(env, state, rolls(), DEFEND)
    assert state.round == 2 and mask(state)[WAIT]


@pytest.mark.core
def test_asteroth_wait_leaves_one_attack_and_cannot_create_three(current_game):
    """Wait replaces Asteroth's two strikes with one; it is illegal between strikes."""
    env, initial, step, fight = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    state = hero_roster_state(env, state, ['g000uu6011', 'duke', None, None, None, None])
    state = enemy_roster_state(env, state, ['squire', None, None, None, None, None])
    state = state.replace(actor=jnp.int32(0), hp=jnp.where(state.hp > 0, 10000, 0),
        priority=jnp.zeros(12).at[0].set(100.).at[1].set(50.).at[6].set(25.))
    values = jnp.zeros(env.random_size)
    first, _ = fight(state, jnp.int32(SHOOT), state.battle_key, values)
    assert first.actor == 0 and first.second_strike
    assert not compiled_method(env, 'action_mask')(first)[WAIT]
    rejected, _ = step(first, jnp.int32(WAIT))
    np.testing.assert_array_equal(rejected.battle_key, first.battle_key)
    np.testing.assert_array_equal(rejected.hp, first.hp)
    assert rejected.actor == 0 and rejected.second_strike
    second, _ = fight(first, jnp.int32(SHOOT), first.battle_key, values)
    assert not second.second_strike and second.turn_phase[0] == 2 and second.actor == 1
    # Both allies wait before the enemy, then resume in that same order.
    waited, _ = fight(state, jnp.int32(WAIT), state.battle_key, values)
    waited, _ = fight(waited, jnp.int32(WAIT), waited.battle_key, values)
    assert waited.actor == 6
    resumed, _ = fight(waited, jnp.int32(CONTINUE), waited.battle_key, values)
    assert resumed.actor == 0 and resumed.turn_phase[0] == 1
    single, _ = fight(resumed, jnp.int32(SHOOT), resumed.battle_key, values)
    assert single.hp[6] == resumed.hp[6] - first.last_damage
    assert not single.second_strike and single.turn_phase[0] == 2
    assert single.actor == 1 and single.round == 1


def test_wait_position_survives_temporary_forms(current_game):
    """Transforming a waiting unit changes its stats without moving its queued turn."""
    from stoix.envs.number_grid_item_combat import transform_imp, transform_form
    env, initial, _, _ = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    state = hero_roster_state(env, state, ['archer', 'duke', 'archer', None, None, None])
    state = state.replace(turn_phase=state.turn_phase.at[0].set(1),
        waited=state.waited.at[0].set(True),priority=state.priority.at[0].set(-1.).at[2].set(100.))
    slots, selected = jnp.array([0,2]), jnp.ones(2,bool)
    imp, _ = jax.jit(lambda s: transform_imp(env,s,s.wards_used,slots,selected))(state)
    assert imp.priority[0] == -1. and imp.priority[2] == 30.
    forms = jnp.full(2,env.progression.ids['archer'],jnp.int32)
    form, *_ = jax.jit(lambda s: transform_form(env,s,s.hp,s.wards_used,slots,selected,forms))(state)
    assert form.priority[0] == -1.
    assert form.priority[2] == env.progression.base_stats[forms[1],INITIATIVE]
    cured, *_ = compiled_method(env, '_cleanse')(imp,imp.hp,imp.wards_used,jnp.ones(12,bool))
    assert cured.priority[0] == -1. and not cured.imp[0]


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
    state = state.replace(actor=jnp.int32(0), turn_phase=state.turn_phase.at[0].set(1),
        priority=state.priority.at[0].set(-1.))
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
    assert not env.basic_combat and env.reset(jax.random.PRNGKey(0))[1].observation.shape == (179,)


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
