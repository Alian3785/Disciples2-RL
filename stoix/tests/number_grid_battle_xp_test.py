"""Steam sub_62618B/62632E and sub_665AB7; see original-battle-xp-20261021.json.

These regressions require the project's explicitly requested CUDA test runner.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid_effects import advance_poison_queue, advance_periodic_queue
from stoix.envs.number_grid_summoning import advance_linked_queue
from stoix.tests.number_grid_fixtures import compiled_method


def roster(current_game):
    env, initial, _, _ = current_game
    occupied = jnp.array([True]*4+[False]*2+[True]*4+[False]*2)
    ids = jnp.where(occupied, env.progression.ids['duke'], 0).astype(jnp.int32)
    state = initial.replace(enemy=jnp.int32(0), in_battle=jnp.bool_(True),
        unit_ids=ids, unit_levels=jnp.where(occupied, 10, 0), unit_xp=jnp.zeros(12,jnp.int32),
        hp=jnp.where(occupied, 100, 0), battle_earned_xp=jnp.zeros(12,jnp.int32))
    return env, state, jnp.where(occupied, 200, 0)


def packet(env, state, hp, worth):
    out = compiled_method(env.progression, 'accumulate')(state, state.hp, hp, state.escaped, worth)
    return out.replace(hp=hp)


@pytest.mark.core
@pytest.mark.parametrize('side', [0, 1])
def test_dead_fighters_lose_only_personal_battle_xp_without_redistribution(current_game, side):
    env, state, worth = roster(current_game)
    ally, enemy = 6*side, 6*(1-side)
    state = state.replace(unit_xp=state.unit_xp.at[ally+2:ally+4].set(17))
    state = packet(env, state, state.hp.at[enemy].set(0), worth)
    assert state.battle_earned_xp[ally:ally+4].tolist() == [50]*4
    state = packet(env, state, state.hp.at[ally+2].set(0), worth)
    state = packet(env, state, state.hp.at[ally+3].set(0), worth)
    state = packet(env, state, state.hp.at[enemy+1].set(0), worth)
    assert state.battle_earned_xp[ally:ally+4].tolist() == [150,150,0,0]
    result = compiled_method(env.progression, 'finish')(
        state, state.hp, state.escaped, jnp.bool_(side == 0), jnp.bool_(side == 1), jnp.bool_(False))
    assert result[2][ally:ally+4].tolist() == [150,150,17,17]
    assert result[5][ally:ally+4].tolist() == [150,150,0,0]


@pytest.mark.core
def test_escape_retains_earned_xp_excludes_future_kills_and_cannot_promote(current_game):
    env, state, worth = roster(current_game)
    state = packet(env, state, state.hp.at[6].set(0), worth)
    state = state.replace(escaped=state.escaped.at[0].set(True))
    state = packet(env, state, state.hp.at[7].set(0), worth)
    assert state.battle_earned_xp[:4].tolist() == [50,116,116,116]
    required = env.progression.required_xp(state.unit_ids, state.unit_levels)
    state = state.replace(escaped=state.escaped.at[:4].set(True).at[8].set(True),
        unit_xp=state.unit_xp.at[1].set(required[1]-10).at[8].set(required[8]-3),
        battle_earned_xp=state.battle_earned_xp.at[8].set(20).at[9].set(30))
    result = compiled_method(env.progression, 'finish')(
        state, state.hp, state.escaped, jnp.bool_(False), jnp.bool_(False), jnp.bool_(True))
    assert result[2][:4].tolist() == [50,int(required[1])-1,116,116]
    assert result[2][8] == required[8]-1 and result[2][9] == 30
    assert result[3][0,2,1] == required[8]-1  # fleeing enemy's persisted XP
    assert result[5][1] == 9 and result[5][8] == 2 and result[6] == 0
    assert jnp.array_equal(result[1], state.unit_levels)
    assert jnp.array_equal(result[4], state.hp)  # no promotion heal


@pytest.mark.core
def test_revival_does_not_restore_lost_shares_and_mass_attack_has_one_divisor(current_game):
    env, state, worth = roster(current_game)
    first = packet(env, state, state.hp.at[6].set(0), worth)
    fallen = packet(env, first, first.hp.at[1].set(0), worth)
    revived = packet(env, fallen, fallen.hp.at[1].set(50), worth)
    assert revived.battle_earned_xp[1] == 0
    final = packet(env, revived, revived.hp.at[7].set(0), worth)
    assert final.battle_earned_xp[:4].tolist() == [100,50,100,100]
    # Simultaneous friendly casualty still belongs to this packet's divisor.
    simultaneous = packet(env, state, state.hp.at[3].set(0).at[6].set(0), worth)
    assert simultaneous.battle_earned_xp[:4].tolist() == [50,50,50,0]
    # floor((201+203)/4)=101, rather than floor(201/4)+floor(203/4)=100.
    mass = packet(env, state, state.hp.at[6:8].set(0), worth.at[6].set(201).at[7].set(203))
    assert mass.battle_earned_xp[:4].tolist() == [101]*4
    repeated = packet(env, mass, mass.hp, worth)
    assert jnp.array_equal(repeated.battle_earned_xp, mass.battle_earned_xp)
    assert repeated.battle_xp[1] == 404


@pytest.mark.parametrize('queue', ['poison', 'periodic', 'linked'])
@pytest.mark.parametrize('enemy_first', [False, True])
def test_dot_death_order_changes_only_subsequent_shares(current_game, queue, enemy_first):
    env, state, worth = roster(current_game)
    hp = state.hp.at[3].set(1).at[6].set(1)
    state = state.replace(hp=hp)
    priority = jnp.zeros(12).at[3].set(20 if enemy_first else 30).at[6].set(30 if enemy_first else 20)
    turns = jnp.zeros(12,jnp.int32).at[3].set(1).at[6].set(1)
    damage = jnp.where(turns != 0, 10, 0)
    source = jnp.full(12,-1,jnp.int32)
    common = (hp, state.escaped, jnp.zeros(12,jnp.int32), priority, jnp.int32(1), jnp.zeros(12,bool))
    suffix = (jnp.zeros(12,jnp.uint32), priority, jnp.int32(0), jnp.bool_(False), jnp.int32(100))
    if queue == 'poison':
        resolve = jax.jit(lambda *args: advance_poison_queue(*args, return_death_order=True))
        out = resolve(*common, turns, damage, source, *suffix,
                      jnp.zeros(12,jnp.int32), jnp.ones((2,12)))
    else:
        resolve_fn = advance_linked_queue if queue == 'linked' else advance_periodic_queue
        extra = {'summon_owner':jnp.full(12,-1,jnp.int32)} if queue == 'linked' else {}
        resolve = jax.jit(lambda *args: resolve_fn(*args, return_death_order=True, **extra))
        out = resolve(*common, jnp.stack((turns,turns*0,turns*0),axis=1),
            jnp.stack((damage,damage*0,damage*0),axis=1), jnp.tile(source[:,None],(1,3)), *suffix,
            jnp.array([16,8,4],jnp.uint32), jnp.array([0,10,10]),
            jnp.zeros((12,3),jnp.int32), jnp.ones((2,12,3)))
    assert out[0][3] == out[0][6] == 0
    assert bool(out[-1][6] < out[-1][3]) == enemy_first
    result = compiled_method(env.progression, 'accumulate')(
        state, hp, out[0], state.escaped, worth, out[-1])
    assert result.battle_earned_xp[:3].tolist() == [50 if enemy_first else 66]*3
    assert result.battle_earned_xp[3] == 0


def test_summons_and_empty_cells_do_not_dilute_shares_but_guards_do(current_game):
    env, state, worth = roster(current_game)
    guard = jnp.argmax(env.progression.capital_guards).astype(jnp.int32)
    state = state.replace(unit_ids=state.unit_ids.at[2].set(guard),
        summon_owner=state.summon_owner.at[3].set(0),
        battle_earned_xp=state.battle_earned_xp.at[3].set(17))
    final = packet(env, state, state.hp.at[6].set(0).at[3].set(0), worth)
    assert final.battle_earned_xp[:4].tolist() == [66,66,66,17]
    assert final.battle_xp[0] == 0  # dead summon is worth zero
    result = compiled_method(env.progression, 'finish')(
        final, final.hp, final.escaped, jnp.bool_(True), jnp.bool_(False), jnp.bool_(False))
    assert result[5][:4].tolist() == [66,66,0,0]
