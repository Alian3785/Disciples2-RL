"""Extended GPU contracts for the round-40 disable limit; see ORIGINAL_EFFECTS.md.

User request 2026-10-16; run only when explicitly requested, via scripts/run_gpu.sh.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import SHOOT
from stoix.envs.number_grid_combat import ACCURACY, accuracy_hits
from stoix.envs.number_grid_effects import DISABLE, disable_attack_accuracy, disable_attack_allowed
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state, enemy_roster_state


@pytest.fixture(autouse=True)
def require_gpu():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'


def duel(game, caster, red=False):
    env, initial, _, _ = game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    # RED chooses its own target. Put the highest-damage hero in slot 0 so
    # both ordinary targeting and long-paralysis targeting select that slot.
    hero = ['duke', 'archer'] if red else [caster, 'duke']
    state = hero_roster_state(env, state, [*hero, None, None, None, None])
    state = enemy_roster_state(env, state, [caster if red else 'archer', 'archer', None, None, None, None])
    actor, target = (6, 0) if red else (0, 6)
    return env, state.replace(actor=jnp.int32(actor), round=jnp.int32(40),
        native_ids=state.unit_ids, native_levels=state.unit_levels,
        hp=jnp.where(state.unit_ids != 0, 9999, 0),
        priority=jnp.zeros(12).at[actor+1].set(10000.),
        turn_phase=jnp.zeros(12, jnp.int32), activation_done=jnp.zeros(12, bool)), actor, target


def strike(env, state, target):
    return compiled_method(env, '_battle_step')(state, jnp.int32(SHOOT + target % 6),
        state.battle_key, jnp.zeros(env.random_size))[0]


@pytest.mark.core
def test_round_limit_is_separate_from_zero_accuracy_roll():
    rounds = jnp.array([1, 39, 40, 41, 75, 1000])
    limit = jax.jit(disable_attack_accuracy)
    accuracy = limit(jnp.float32(150), jnp.bool_(True), rounds)
    assert accuracy.tolist() == [150., 150., 150., 0., 0., 0.]
    hits = jax.jit(accuracy_hits)(accuracy, jnp.zeros(6), jnp.zeros(6))
    assert hits.tolist() == [True]*6
    allowed = jax.jit(disable_attack_allowed)(jnp.bool_(True), rounds)
    assert (hits & allowed).tolist() == [True]*3+[False]*3
    assert jnp.all(limit(jnp.float32(95), jnp.bool_(False), rounds) == 95)


@pytest.mark.parametrize('caster', [pytest.param('ghost', marks=pytest.mark.core),
                                  'shadow', 'incubus', pytest.param('medusa', marks=pytest.mark.core)])
@pytest.mark.parametrize('red', [False, True])
def test_primary_paralyze_and_petrify_for_both_sides(current_game, caster, red):
    env, before, actor, target = duel(current_game, caster, red)
    before = before.replace(potion_temporary=before.potion_temporary.at[actor, ACCURACY].set(200.))
    stats = compiled_method(env, 'unit_stats')
    assert stats(before)[actor, ACCURACY] > 0
    early = strike(env, before, target)
    assert early.paralyzed[target] | early.long_paralyzed[target]
    late = before.replace(round=jnp.int32(41))
    assert stats(late)[actor, ACCURACY] == 0
    missed = strike(env, late, target)
    assert not jnp.any(missed.paralyzed | missed.long_paralyzed)
    assert jnp.array_equal(missed.hp, before.hp)
    assert stats(late.replace(in_battle=jnp.bool_(False), actor=jnp.int32(0)))[0, ACCURACY] > 0
    # The explicit round ban rejects even (0, 0) before spending a ward.
    source = compiled_method(env, '_combat_traits')(before)[actor, 1]
    ward = jnp.left_shift(jnp.uint32(1), source-1)
    protected = late.replace(potion_wards=late.potion_wards.at[target].set(ward))
    missed = strike(env, protected, target)
    assert not missed.wards_used[target] and not missed.last_ward
    if not red:
        assert compiled_method(env, 'action_mask')(late)[SHOOT + target % 6]


@pytest.mark.parametrize('caster', [pytest.param('ghoul', marks=pytest.mark.core), 'mermaid', 'abyss_devil'])
@pytest.mark.parametrize('red', [False, True])
def test_secondary_disable_stops_but_primary_damage_remains(current_game, caster, red):
    # Zero random draws hit any positive accuracy; potions no longer raise secondary accuracy.
    env, before, actor, target = duel(current_game, caster, red)
    early = strike(env, before, target)
    assert early.paralyzed[target] | early.long_paralyzed[target]
    late = before.replace(round=jnp.int32(41))
    assert compiled_method(env, '_secondary_stats')(late, jnp.int32(actor))[1] == 0
    missed = strike(env, late, target)
    assert not jnp.any(missed.paralyzed | missed.long_paralyzed)
    assert 0 < missed.last_damage == early.last_damage
    source = env.progression.secondary_sources[before.unit_ids[actor]]
    ward = jnp.left_shift(jnp.uint32(1), source.astype(jnp.uint32)-1)
    protected = late.replace(potion_wards=late.potion_wards.at[target].set(ward))
    assert not (strike(env, protected, target).wards_used[target] & ward)


def test_copy_inherits_limit_and_transformation_uses_its_actual_attack(current_game):
    env, before, actor, target = duel(current_game, 'doppelganger')
    enemy = enemy_roster_state(env, before, ['incubus', 'archer', None, None, None, None])
    copied = compiled_method(env, '_copy_unit')(enemy, jnp.int32(target), jnp.bool_(True))
    assert copied.copied[actor]
    stats = compiled_method(env, 'unit_stats')
    assert stats(copied)[actor, ACCURACY] > 0
    copied = copied.replace(round=jnp.int32(41))
    assert stats(copied)[actor, ACCURACY] == 0
    assert stats(copied.replace(imp=copied.imp.at[actor].set(True)))[actor, ACCURACY] > 0


def test_existing_effects_keep_recovery_and_fear_is_not_disabled(current_game):
    env, before, _, target = duel(current_game, 'baroness')
    late = before.replace(round=jnp.int32(41))
    feared = strike(env, late, target)
    assert feared.feared[target] | feared.paralyzed[target]
    disabled = late.replace(long_paralyzed=late.long_paralyzed.at[0].set(True),
        effect_rounds=late.effect_rounds.at[0, DISABLE].set(40))
    prepare = compiled_method(env, '_prepare_disable')
    assert prepare(disabled, jnp.float32(0), jnp.bool_(True)).long_paralyzed[0]
    assert not prepare(disabled, jnp.float32(.999), jnp.bool_(True)).long_paralyzed[0]
