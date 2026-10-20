"""Siege reward regressions; CUDA-only, run only on an explicit request.

Source: user reward-design requests, 2026-10-13/14; docs/REFERENCE_CAMPAIGNS.md.
These rewards are not claims about original Disciples II mechanics.
"""
from copy import copy

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, REST, load_map
from stoix.envs import number_grid_reference_rewards as rewards
from stoix.envs.number_grid_combat import HP
from stoix.tests.number_grid_fixtures import compiled_method


@pytest.fixture(scope='module')
def siege():
    env = NumberGrid(map_config=load_map('siege_train'))
    state, _ = env.reset(jax.random.PRNGKey(7))
    baseline = copy(env)
    baseline.siege_rewards = None
    return env, baseline, state


def active_battle(env, state):
    target = env.objective_enemy
    active = state.replace(enemy=jnp.int32(target),
        scenario_spawned=state.scenario_spawned.at[target].set(True),
        alive=state.alive.at[target].set(True))
    return env._begin_battle(active, jax.random.PRNGKey(11), jnp.full(env.random_size, .5))


def battle_reward(env, before, after, hp, victory=False, lost=False, withdrawal=False):
    return rewards.battle(env, before, after, hp, jnp.bool_(victory),
                          jnp.bool_(lost), jnp.bool_(withdrawal), jnp.bool_(False))


def test_spawn_rewards_living_levels_above_one_and_ignores_hp(siege):
    """Calculate the siege arrival bonus from living unit levels above one, independently of HP."""
    env, baseline, initial = siege
    # Wounded level-4 fighter + level-3 hero + level-2 cultist = six extra levels.
    # The dead level-7 fighter and an empty level-99 slot do not contribute.
    before = initial.replace(hp=initial.hp.at[0].set(1).at[2].set(0),
        unit_levels=initial.unit_levels.at[0].set(4).at[1].set(3).at[2].set(7).at[3].set(99).at[4].set(2),
        potion_bonus=initial.potion_bonus.at[0, HP].add(100).at[2, HP].add(500),
        day=jnp.int32(27))
    after = env.scenario.advance(before.replace(day=jnp.int32(28)))
    assert after.scenario_spawned[env.objective_enemy] and not after.in_battle
    _, shaped = rewards.world(env, before, after, REST, jnp.bool_(True))
    _, plain = rewards.world(baseline, before, after, REST, jnp.bool_(True))
    assert float(shaped-plain) == pytest.approx(30.)
    # More HP gives no extra reward; one extra hero level gives exactly five.
    for stronger, expected in [(before.replace(potion_bonus=before.potion_bonus+1000), 30.),
                              (before.replace(unit_levels=before.unit_levels.at[1].add(1)), 35.)]:
        born = env.scenario.advance(stronger.replace(day=jnp.int32(28)))
        _, shaped = rewards.world(env, stronger, born, REST, jnp.bool_(True))
        _, plain = rewards.world(baseline, stronger, born, REST, jnp.bool_(True))
        assert float(shaped-plain) == pytest.approx(expected)


def test_spawn_bonus_is_once_per_episode_and_not_deferred_to_contact(siege):
    """Pay the siege bonus on arrival once per episode, without waiting for combat contact."""
    env, baseline, initial = siege
    before = initial.replace(day=jnp.int32(27), unit_levels=initial.unit_levels.at[1].set(4))
    after = env.scenario.advance(before.replace(day=jnp.int32(28)))
    _, shaped = rewards.world(env, before, after, REST, jnp.bool_(True))
    _, plain = rewards.world(baseline, before, after, REST, jnp.bool_(True))
    assert float(shaped-plain) == pytest.approx(15.)
    # Later development, rests, and the eventual engagement cannot repay it.
    later = after.replace(day=jnp.int32(29), unit_levels=after.unit_levels.at[1].set(9))
    contacted = active_battle(env, later)
    for old, new, action in [(after, later, REST), (later, contacted, 0)]:
        _, shaped = rewards.world(env, old, new, action, jnp.bool_(action == REST))
        _, plain = rewards.world(baseline, old, new, action, jnp.bool_(action == REST))
        assert float(shaped) == pytest.approx(float(plain))
    assert env.army_victory_level_reward == 0. and initial.army_victory_rewarded.shape == (0,)
    reset, _ = env.reset(jax.random.PRNGKey(19))
    assert not reset.scenario_spawned[env.objective_enemy]
    fresh = reset.replace(day=jnp.int32(27), unit_levels=reset.unit_levels.at[1].set(4))
    born = env.scenario.advance(fresh.replace(day=jnp.int32(28)))
    _, shaped = rewards.world(env, fresh, born, REST, jnp.bool_(True))
    _, plain = rewards.world(baseline, fresh, born, REST, jnp.bool_(True))
    assert float(shaped-plain) == pytest.approx(15.)


def test_no_spawn_bonus_before_due_turn_or_for_level_one_party(siege):
    """Keep the spawn bonus zero before arrival and for a party with no levels above one."""
    env, baseline, initial = siege
    for before in (initial.replace(day=jnp.int32(26), unit_levels=initial.unit_levels.at[1].set(4)),
                   initial.replace(day=jnp.int32(27))):
        after = env.scenario.advance(before.replace(day=before.day+1))
        _, shaped = rewards.world(env, before, after, REST, jnp.bool_(True))
        _, plain = rewards.world(baseline, before, after, REST, jnp.bool_(True))
        assert float(shaped) == pytest.approx(float(plain))


def test_spawn_reward_precedes_attacks_and_is_not_conditional_on_battle_outcome(siege):
    """Preserve the arrival reward regardless of later combat damage or outcome."""
    env, baseline, initial = siege
    target = env.objective_enemy
    before = initial.replace(day=jnp.int32(27), unit_levels=initial.unit_levels.at[1].set(3),
        enemy_positions=initial.enemy_positions.at[target].set(initial.position+jnp.array([0, 1])))
    key, draws = jax.random.PRNGKey(23), jnp.full(env.random_size, .5)
    arrived, reward = compiled_method(env, '_world_step')(before, jnp.int32(REST), key, draws)
    _, plain = compiled_method(baseline, '_world_step')(before, jnp.int32(REST), key, draws)
    assert arrived.in_battle and arrived.enemy == target and not arrived.won
    assert float(reward-plain) == pytest.approx(10.)
    assert int(arrived.army_battle_level_strength) == 0
    # The paid preparation bonus is independent of loss or retreat rewards.
    for outcome in ('lost', 'withdrawal'):
        _, shaped_end = battle_reward(env, arrived, arrived, arrived.hp, **{outcome: True})
        _, plain_end = battle_reward(baseline, arrived, arrived, arrived.hp, **{outcome: True})
        assert float((reward+shaped_end)-(plain+plain_end)) == pytest.approx(10.)


@pytest.mark.parametrize('value', [-1., True, float('inf'), float('nan'), '5', None])
def test_invalid_spawn_level_coefficient_is_rejected(value):
    """Reject invalid siege spawn-level reward coefficients during environment construction."""
    config = load_map('siege_train')
    config['siege_rewards']['spawn_level'] = value
    with pytest.raises(ValueError, match='siege_rewards.spawn_level'):
        NumberGrid(map_config=config)


def test_spawn_reward_cannot_be_combined_with_victory_level_reward():
    """Reject configuration that would pay both arrival and victory level rewards for the siege."""
    config = {**load_map('siege_train'), 'army_victory_level_reward': 5.}
    with pytest.raises(ValueError, match='mutually exclusive'):
        NumberGrid(map_config=config)


def test_obsolete_siege_hp_reward_is_rejected():
    """Reject the superseded HP-based siege reward instead of silently using an obsolete rule."""
    config = load_map('siege_train')
    config['siege_rewards']['spawn_hp'] = .05
    with pytest.raises(ValueError, match='siege_rewards.spawn_hp was replaced'):
        NumberGrid(map_config=config)


@pytest.mark.parametrize('summon', [False, True])
def test_kills_pay_on_defeat_and_survive_retreat_revival_and_reset(siege, summon):
    env, _, initial = siege
    before = active_battle(env, initial).replace(summon_battle=jnp.bool_(summon))
    hp = before.hp.at[6].set(0).at[9].set(0)
    after, reward = battle_reward(env, before, before, hp, lost=True)
    assert float(reward) == pytest.approx(10.-.25)
    assert int(after.siege_rewarded_kills) == (1 << 0) | (1 << 3)
    # Reward is not tied to battle-local damage credit or final victory.
    repeated, reward = battle_reward(env, after, after, hp, withdrawal=True)
    assert float(reward) == pytest.approx(-.25)
    revived = repeated.replace(hp=before.hp, enemy_killed_credit=jnp.zeros(6, bool))
    _, reward = battle_reward(env, revived, revived, hp)
    assert float(reward) == pytest.approx(.0025)
    reset, _ = env.reset(jax.random.PRNGKey(19))
    assert int(reset.siege_rewarded_kills) == 0


def test_empty_slots_escape_and_other_armies_do_not_pay(siege):
    env, _, initial = siege
    before = active_battle(env, initial)
    # Empty slots are already zero; escaped units still have positive HP.
    escaped = before.replace(escaped=before.escaped.at[6:].set(True))
    _, reward = battle_reward(env, escaped, escaped, escaped.hp, withdrawal=True)
    assert float(reward) == pytest.approx(-.25)
    other = before.replace(enemy=jnp.int32(0))
    _, reward = battle_reward(env, other, other, other.hp.at[6].set(0))
    assert float(reward) == pytest.approx(.0025)


def test_spell_kill_and_later_battle_share_episode_credit(siege):
    env, baseline, initial = siege
    active = active_battle(env, initial).replace(in_battle=jnp.bool_(False))
    target = env.objective_enemy
    maximum = env.casting.enemy_max_hp(active)[target]
    struck = active.replace(enemy_wounds=active.enemy_wounds.at[target, 0].set(maximum[0]))
    credited, shaped = rewards.world(env, active, struck, 0, jnp.bool_(False))
    _, plain = rewards.world(baseline, active, struck, 0, jnp.bool_(False))
    assert float(shaped-plain) == pytest.approx(5.)
    battle = active_battle(env, credited)
    # The spell victim is already dead at entry; only the newly killed ranger pays.
    _, reward = battle_reward(env, battle, battle, battle.hp.at[9].set(0))
    assert float(reward) == pytest.approx(5.0025)


def test_only_fixed_siege_victory_components_are_tripled(siege):
    env, baseline, initial = siege
    before = active_battle(env, initial)
    target = env.objective_enemy
    final = before.replace(alive=before.alive.at[target].set(False),
        hp=before.hp.at[6:].set(0), in_battle=jnp.bool_(False),
        won=jnp.bool_(True), done=jnp.bool_(True))
    shaped_state, shaped = battle_reward(env, before, final, final.hp, victory=True)
    plain_state, plain = battle_reward(baseline, before, final, final.hp, victory=True)
    shaped += rewards.finish(env, before, shaped_state)[1]
    plain += rewards.finish(baseline, before, plain_state)[1]
    # Fixed baseline 16.5 becomes 49.5; four kills add another 20.
    # Identical XP, survival and episode completion fractions cancel.
    assert float(shaped-plain) == pytest.approx(49.5-16.5+20.)


def test_other_reference_maps_do_not_enable_siege_shaping():
    for name in ('super_last_stand', 'wotans_retribution'):
        assert 'siege_rewards' not in load_map(name)
