"""CUDA reward contracts for the 2026-10-18 three-target training objective.

Run only on explicit request through scripts/run_gpu.sh; no policy evaluation.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.tests.number_grid_fixtures import compiled_method


@pytest.fixture(scope='module')
def rewards(current_game):
    env = current_game[0]
    assert jax.default_backend() == 'gpu'
    rules = env.target_rewards
    return (jax.jit(lambda state: rules.potential(env, state)),
            jax.jit(lambda before, after: rules.finish(env, before, after)))


@pytest.mark.core
def test_progress_healing_cycle_and_terminal_correction(current_game, rewards):
    _, initial, _, _ = current_game
    potential, finish = rewards
    wounded = initial.replace(enemy_wounds=initial.enemy_wounds.at[50, 4].set(1000))
    assert float(potential(initial)) == 0
    assert float(potential(wounded)) == pytest.approx(2.5)
    paid, gain = finish(initial, wounded)
    healed, loss = finish(paid, initial)
    assert float(gain) > 0 and float(loss) < 0
    assert float(gain + .99 * loss) == pytest.approx(0., abs=1e-6)
    assert float(healed.target_potential) == 0
    truncated, reward = finish(paid, paid.replace(done=jnp.bool_(True)))
    assert float(truncated.target_potential) == pytest.approx(2.5)
    assert float(reward) == pytest.approx(-.025, abs=1e-6)
    for changes in ({'done': True, 'won': True}, {'done': True, 'lost': True}):
        terminal, correction = finish(paid, paid.replace(**{k: jnp.bool_(v) for k, v in changes.items()}))
        assert float(correction) == pytest.approx(-2.5)
        assert float(terminal.target_potential) == 0


@pytest.mark.core
def test_target_payment_once_for_every_method_and_completion(current_game, rewards):
    _, initial, _, _ = current_game
    _, finish = rewards
    dead = initial.replace(alive=initial.alive.at[50].set(False))
    for summon in (False, True):
        before = initial.replace(summon_battle=jnp.bool_(summon))
        paid, gain = finish(before, dead)
        assert float(gain) == pytest.approx(20. + .99 * 5.)
        assert paid.target_rewarded.tolist() == [False, False, True]
        _, repeat = finish(paid, paid)
        assert float(repeat) == pytest.approx(-.05, abs=1e-6)
    before = paid.replace(alive=paid.alive.at[21].set(False),
        target_rewarded=jnp.array([True, False, True]), target_potential=jnp.float32(10.))
    all_dead = before.replace(alive=before.alive.at[36].set(False))
    final, gain = finish(before, all_dead)
    assert float(gain) == pytest.approx(20. + 30. + .99 * 15. - 10.)
    _, repeat = finish(final, final)
    assert float(repeat) == pytest.approx(-.15, abs=1e-6)


@pytest.mark.core
def test_enemy_summons_do_not_create_progress_or_hide_escapees(current_game, rewards):
    env, initial, _, _ = current_game
    potential, _ = rewards
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(50)))
    assert float(potential(state)) == 0
    # A summoned creature in a previously empty slot is irrelevant to the target.
    summoned = state.replace(summon_owner=state.summon_owner.at[6].set(10),
        hp=state.hp.at[6].set(400), unit_ids=state.unit_ids.at[6].set(env.progression.ids['highvampire']))
    assert float(potential(summoned)) == float(potential(summoned.replace(hp=summoned.hp.at[6].set(0)))) == 0
    # A summon in the original leader's slot must expose the saved leader's HP.
    replaced = summoned.replace(summon_owner=summoned.summon_owner.at[10].set(6),
        summon_saved_hp=summoned.summon_saved_hp.at[10].set(1000), hp=summoned.hp.at[10].set(400))
    assert float(potential(replaced)) == pytest.approx(2.5)
    assert float(potential(replaced.replace(hp=replaced.hp.at[10].set(0)))) == pytest.approx(2.5)


def test_public_step_spell_kill_pays_target_and_group_bonus(current_game, rewards):
    env, initial, step, _ = current_game
    potential, _ = rewards
    rules = env.casting
    # Drulliaan has no fire ward; leave each original fighter at 1 HP.
    maximum = compiled_method(rules, 'enemy_max_hp')(initial)[36]
    before = initial.replace(position=jnp.array([34, 9], jnp.int32),
        mana=jnp.full(5, 5000, jnp.int32), learned_spells=jnp.uint32(1 << 1),
        enemy_wounds=initial.enemy_wounds.at[36].set(jnp.maximum(maximum - 1, 0)),
        alive=initial.alive.at[21].set(False).at[50].set(False),
        target_rewarded=jnp.array([True, False, True]))
    before = before.replace(target_potential=potential(before))
    after, ts = step(before, jnp.int32(rules.start + 1))
    assert after.last_spell_target == 36 and not after.alive[36]
    assert not after.done and after.target_rewarded.all()
    assert float(ts.reward) == pytest.approx(
        50. + .99 * float(potential(after)) - float(before.target_potential) - env.step_cost)
    assert rules.reward == rules.summon_engage_reward == 0
