"""Original Steam 3.01 expiry contracts; evidence: docs/COMBAT.md."""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, WAIT, SHOOT
from stoix.envs.number_grid_effects import effect_breaks, advance_poison_queue, advance_periodic_queue, LONG, SHORT
from stoix.envs.number_grid_summoning import advance_linked_queue
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state, enemy_roster_state

pytestmark = pytest.mark.core


def test_inclusive_original_roll_all_1200_cases():
    ages = jnp.arange(12)[:,None]
    rolls = (jnp.arange(100)[None,:]+.5)/100
    actual = jax.jit(effect_breaks)(ages,jnp.zeros_like(ages),rolls)
    expected = jnp.arange(1,101)[None,:] >= 90-10*ages
    assert jnp.array_equal(actual,expected)
    assert jnp.array_equal(actual.sum(axis=1),jnp.array([11,21,31,41,51,61,71,81,91,100,100,100]))


def test_periodic_order_damage_before_recovery_and_wait_guard():
    hp = jnp.full(12,100,jnp.int32).at[6].set(9)
    phases = jnp.full(12,2,jnp.int32).at[6:8].set(0)
    order = jnp.arange(12,dtype=jnp.float32).at[6].set(100)
    modes = jnp.zeros((12,3),jnp.int32).at[6:8].set(LONG)
    damage = jnp.zeros((12,3),jnp.int32).at[6:8].set(jnp.array([3,7,11]))
    args = (hp,jnp.zeros(12,bool),phases,order,jnp.int32(1),jnp.zeros(12,bool),
        modes,damage,jnp.full((12,3),0,jnp.int32),jnp.zeros(12,jnp.uint32),order,
        jnp.int32(6),jnp.bool_(False),jnp.int32(75),jnp.array([16,8,4],jnp.uint32),
        jnp.zeros(3,jnp.int32),jnp.ones((12,3),jnp.int32),jnp.full((2,12,3),.995))
    run = jax.jit(advance_periodic_queue)
    out = run(*args)
    assert out[0][6] == 0 and out[0][7] == 79
    assert jnp.array_equal(out[9][6],jnp.array([3,6,0]))  # frostbite kills before burn
    assert not jnp.any(out[5][7])  # damage still happens on a successful removal roll
    waited = list(args)
    waited[5] = jnp.ones(12,bool)
    out = run(*waited)
    assert jnp.array_equal(out[0],hp) and not jnp.any(out[9])
    short = list(args)
    short[6] = jnp.where(modes>0,SHORT,0)
    short[-1] = jnp.zeros((2,12,3))
    assert not jnp.any(run(*short)[5])


@pytest.mark.parametrize('queue_kind', ['poison', 'periodic', 'linked'])
@pytest.mark.parametrize('lethal', [False, True])
def test_periodic_queues_preserve_wait_fifo_after_a_normal_turn(queue_kind, lethal):
    """Zero-initiative normal turns precede waiters, including after a DOT death."""
    hp = jnp.zeros(12,jnp.int32).at[0:2].set(100).at[6].set(10).at[7].set(100)
    phases = jnp.full(12,2,jnp.int32).at[0:2].set(1).at[6].set(0)
    priority = jnp.zeros(12).at[0].set(-1.).at[1].set(-2.)
    activated = jnp.ones(12,bool).at[6].set(False)
    modes = jnp.zeros(12,jnp.int32).at[0:2].set(LONG).at[6].set(LONG)
    damage = jnp.zeros(12,jnp.int32).at[0:2].set(10).at[6].set(10 if lethal else 0)
    source = jnp.zeros(12,jnp.int32)
    common = (hp,jnp.zeros(12,bool),phases,priority,jnp.int32(1),activated)
    tail = (jnp.zeros(12,jnp.uint32),jnp.ones(12),jnp.int32(6),jnp.bool_(False),jnp.int32(75))
    if queue_kind == 'poison':
        out = jax.jit(advance_poison_queue)(*common,modes,damage,source,*tail,
            jnp.ones(12,jnp.int32),jnp.zeros((2,12)))
    else:
        def matrix(column):
            return jnp.zeros((12,3),jnp.int32).at[:,0].set(column)
        args = (*common,matrix(modes),matrix(damage),matrix(source),*tail,
            jnp.array([16,8,4],jnp.uint32),jnp.zeros(3,jnp.int32),
            jnp.ones((12,3),jnp.int32),jnp.zeros((2,12,3)))
        out = (jax.jit(advance_linked_queue)(*args,jnp.full(12,-1,jnp.int32))
               if queue_kind == 'linked' else jax.jit(advance_periodic_queue)(*args))
    assert out[8] == (0 if lethal else 6)
    assert out[3] == 1  # no new round while a waiter remains
    assert jnp.array_equal(out[0][:2],hp[:2])  # WAIT does not tick poison twice


def battle(current_game):
    env, initial, _, attack = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env,state,['archer','archer','archer','cultist',None,None])
    state = enemy_roster_state(env,state,['archer','archer','archer','archer',None,None])
    state = state.replace(actor=jnp.int32(0),round=jnp.int32(3),hp=jnp.where(state.hp>0,1000,0),
        priority=jnp.arange(12,dtype=jnp.float32),activation_done=jnp.zeros(12,bool))
    return env,state,attack


def test_disable_recovery_preserves_action_and_finite_caster_expiry(current_game):
    env,state,_ = battle(current_game)
    state = state.replace(long_paralyzed=state.long_paralyzed.at[0].set(True),
        effect_rounds=state.effect_rounds.at[0,3].set(2))
    prepare = compiled_method(env,'_prepare_disable')
    recovered = prepare(state,jnp.float32(.795),jnp.bool_(True))  # roll 80, age 1
    assert not recovered.long_paralyzed[0]
    assert compiled_method(env,'action_mask')(recovered)[DEFEND]
    remains = prepare(state,jnp.float32(.785),jnp.bool_(True))
    assert remains.long_paralyzed[0] and not compiled_method(env,'action_mask')(remains)[DEFEND]
    finite = state.replace(paralyzed=state.paralyzed.at[1].set(True),
        disable_source=state.disable_source.at[1].set(0))
    assert not prepare(finite,jnp.float32(0),jnp.bool_(True)).paralyzed[1]


def test_hostile_transform_uses_round_age_but_drain_level_never_rolls(current_game):
    env,state,_ = battle(current_game)
    state = state.replace(imp=state.imp.at[0].set(True),transform_long=state.transform_long.at[0].set(True),
        effect_rounds=state.effect_rounds.at[0,4].set(2),activation_done=state.activation_done.at[0].set(True),
        turn_phase=state.turn_phase.at[0].set(1),priority=state.priority.at[0].set(-1.))
    prepare = compiled_method(env,'_start_activation')
    assert not prepare(state,jnp.float32(.795),jnp.bool_(True)).imp[0]  # also on resumed Wait
    assert prepare(state,jnp.float32(.785),jnp.bool_(True)).imp[0]
    lowered = state.replace(imp=jnp.zeros(12,bool),decay_form=state.decay_form.at[0].set(env.progression.ids['squire']))
    assert prepare(lowered,jnp.float32(.999),jnp.bool_(True)).decay_form[0] == lowered.decay_form[0]


def test_cure_only_dispels_original_six_classes(current_game):
    env,state,_ = battle(current_game)
    state = state.replace(poison_mode=jnp.full(12,LONG),water_mode=jnp.full(12,LONG),burn_mode=jnp.full(12,LONG),
        paralyzed=jnp.ones(12,bool),long_paralyzed=jnp.ones(12,bool),weakened=jnp.ones(12,bool),
        slow_original=jnp.full(12,50),armor_shreds=jnp.full(12,2),retreating=jnp.ones(12,bool),feared=jnp.ones(12,bool),
        decay_form=state.decay_form.at[0].set(env.progression.ids['squire']),imp=state.imp.at[1].set(True))
    out,_,_ = compiled_method(env,'_cleanse')(state,state.hp,state.wards_used,jnp.ones(12,bool))
    assert not jnp.any(out.poison_mode | out.water_mode | out.burn_mode)
    assert not jnp.any(out.paralyzed | out.long_paralyzed | out.imp)
    assert out.decay_form[0] == state.decay_form[0]
    assert jnp.all(out.weakened[jnp.array([0,2,3])]) and jnp.all(out.slow_original == 50)
    assert jnp.all(out.armor_shreds == 2) and jnp.all(out.retreating)


def test_short_powerup_survives_wait_and_long_survives_completed_turn(current_game):
    env,state,attack = battle(current_game)
    state = state.replace(powerup=state.powerup.at[0].set(True),primary_override=state.primary_override.at[0].set(100))
    rolls = jnp.zeros(env.random_size)
    waited,_ = attack(state,jnp.int32(WAIT),state.battle_key,rolls)
    assert waited.powerup[0]
    finished,_ = attack(state,jnp.int32(DEFEND),state.battle_key,rolls)
    assert not finished.powerup[0]
    long = state.replace(powerup_long=state.powerup_long.at[0].set(True))
    finished,_ = attack(long,jnp.int32(DEFEND),state.battle_key,rolls)
    assert finished.powerup[0] and finished.primary_override[0] == 100


def test_hermit_changes_only_next_round_queue(current_game):
    env,state,attack = battle(current_game)
    state = hero_roster_state(env,state,['hermit','archer','archer','cultist',None,None])
    state = state.replace(hp=state.hp.at[6].set(1000),priority=state.priority.at[6].set(100))
    out,_ = attack(state,jnp.int32(SHOOT),state.battle_key,jnp.zeros(env.random_size))
    assert out.slow_original[6] >= 0 and out.priority[6] == 100 and out.round == state.round
    normal = compiled_method(env,'_round_priority')(jnp.full(12,.9),out.enemy,out.replace(slow_original=jnp.full(12,-1)))
    slowed = compiled_method(env,'_round_priority')(jnp.full(12,.9),out.enemy,out)
    assert slowed[6] == normal[6]-jnp.floor(normal[6]*.5)
    end = out.replace(actor=jnp.int32(0),turn_phase=jnp.full(12,2,jnp.int32))
    next_round,_ = attack(end,jnp.int32(DEFEND),end.battle_key,jnp.zeros(env.random_size))
    assert next_round.round == end.round+1 and next_round.slow_original[6] == -1
    assert next_round.priority[6] < normal[6]


def test_native_flags_wards_and_retreating_summoner(current_game):
    env,state,_ = battle(current_game)
    ids = env.progression.ids
    assert env.progression.secondary_paralysis_modes[ids['ghoul']] == LONG
    assert env.progression.ghost_modes[ids['ghost']] == SHORT
    assert env.progression.power_long[ids['dwarfdruid']]
    assert not env.progression.power_long[ids['travnitsa']]
    # GAttacks WARD1..4 through GmodifL: Sundancer fire, Sylfid air, Deva roshi
    # air+fire+earth+water. REACH=1 makes the first two act on every ally.
    assert env.progression.healer_ward_elements[ids['sundancer']] == 4
    assert env.progression.healer_ward_elements[ids['sylfid']] == 256
    assert env.progression.healer_ward_elements[ids['deva_roshi']] == 270
    mass = env.progression.mass_healers
    assert all(bool(mass[ids[key]]) for key in ('cleric','abbess','prophetess','elf_oracle','sundancer','sylfid'))
    assert not any(bool(mass[ids[key]]) for key in ('deva_roshi','g000uu8133','acolyte','priest','patriarch','travnitsa'))
    summoned = state.replace(summon_owner=state.summon_owner.at[1].set(0),escaped=state.escaped.at[0].set(True))
    from types import SimpleNamespace
    from stoix.envs.number_grid_summoning import Summoning
    linked = jax.jit(lambda s: Summoning._linked_summon_hp(
        SimpleNamespace(has_summons=True, has_linked_summons=True),s,s.hp))
    assert linked(summoned)[1] == 0


def test_dot_reapplication_refreshes_age_and_can_affect_another_target(current_game):
    env,state,attack = battle(current_game)
    state = hero_roster_state(env,state,['imperial_assassin','archer','archer','cultist',None,None])
    state = state.replace(priority=state.priority.at[1].set(10000),
        poison_mode=state.poison_mode.at[6].set(LONG),poison_damage=state.poison_damage.at[6].set(1),
        effect_rounds=state.effect_rounds.at[6,0].set(1),poison_source=state.poison_source.at[6].set(0))
    rolls = jnp.zeros(env.random_size)
    first,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    assert first.poison_mode[6] == LONG and first.effect_rounds[6,0] == 3
    assert first.poison_damage[6] > 1
    second,_ = attack(first.replace(actor=jnp.int32(0)),jnp.int32(SHOOT+1),first.battle_key,rolls)
    assert second.poison_mode[6] == second.poison_mode[7] == LONG
    assert second.poison_source[6] == second.poison_source[7] == 0
    observe = compiled_method(env,'observation')
    aged = first.replace(round=first.round+1)
    assert not jnp.array_equal(observe(first),observe(aged))


def test_native_buff_duration_and_weakness_survive_replacement(current_game):
    env,state,attack = battle(current_game)
    state = hero_roster_state(env,state,['archer','archer','archer','dwarfdruid',None,None])
    state = state.replace(actor=jnp.int32(3),priority=state.priority.at[1].set(10000),
        weakened=state.weakened.at[0].set(True))
    rolls = jnp.zeros(env.random_size)
    first,_ = attack(state,jnp.int32(SHOOT),state.battle_key,rolls)
    assert first.powerup_long[0] and first.weakened[0]
    base = compiled_method(env,'_original_primary')(first)[0]
    assert first.primary_override[0] == jnp.rint(base*(1.75-.33))
    replacement = first.replace(actor=jnp.int32(3),unit_ids=first.unit_ids.at[3].set(env.progression.ids['travnitsa']))
    second,_ = attack(replacement,jnp.int32(SHOOT),replacement.battle_key,rolls)
    assert not second.powerup_long[0] and second.weakened[0]
    assert second.primary_override[0] == jnp.rint(base*(1.25-.33))
    spent,_ = attack(second.replace(actor=jnp.int32(0)),jnp.int32(DEFEND),second.battle_key,rolls)
    assert not spent.powerup[0] and spent.weakened[0]
    assert compiled_method(env,'_combat_stats')(spent)[0,1] == jnp.rint(base*.67)
