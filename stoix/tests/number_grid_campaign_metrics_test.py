"""Reference campaign telemetry: successful events, completion boundaries and windows."""
import copy

import jax
import jax.numpy as jnp
import pytest
import chex

from numbergrid_campaign import CampaignMetrics
from stoix.envs.number_grid import BUILD_START, DEFEND, REST, CONTINUE
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS, summarize_campaign, campaign_events
from stoix.envs.number_grid_potions import POTION_START
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state, enemy_roster_state

pytestmark = pytest.mark.core


def counters(timestep):
    return dict(zip(CAMPAIGN_FIELDS, map(float, timestep.extras['campaign_events'])))


def test_executed_build_invalid_repeat_and_timeout(current_game):
    env, initial, step, _ = current_game
    state = initial.replace(gold=jnp.int32(9999))
    mask = compiled_method(env, 'action_mask')(state)
    slot = int(jnp.flatnonzero(mask[BUILD_START:REST])[0])
    built, ts = step(state, jnp.int32(BUILD_START + slot))
    assert counters(ts)['buildings_built'] == 1
    _, rejected = step(built, jnp.int32(BUILD_START + slot))
    assert sum(counters(rejected).values()) == 0
    _, timeout = step(built.replace(step_count=jnp.int32(env.max_steps - 1)), jnp.int32(REST))
    assert counters(timeout)['timeouts'] == 1
    assert counters(timeout)['victories'] == counters(timeout)['defeats'] == 0


def battle(current_game):
    env, initial, step, _ = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env, state, ['archer', None, None, None, None, None])
    state = enemy_roster_state(env, state, ['archer', None, None, None, None, None])
    return env, initial, step, state.replace(actor=jnp.int32(0), round=jnp.int32(3),
        turn_phase=jnp.full(12, 2, jnp.int32).at[0].set(0).at[6].set(0),
        priority=jnp.arange(12, dtype=jnp.float32), activation_done=jnp.zeros(12, bool))


def test_internal_dot_victory_counts_once(current_game):
    _, _, step, state = battle(current_game)
    state = state.replace(poison_mode=state.poison_mode.at[6].set(1),
                          poison_damage=state.poison_damage.at[6].set(10000))
    after, ts = step(state, jnp.int32(DEFEND))
    values = counters(ts)
    assert not after.in_battle and not after.lost
    assert values['battle_victories'] == values['enemy_squads_defeated'] == 1
    assert values['victories'] == values['battle_defeats'] == 0
    # The party's XP shares at the battle end (user request 2026-10-26), counted once.
    assert values['experience_gained'] == pytest.approx(float(jnp.sum(after.last_xp[:6])))
    _, rested = step(after, jnp.int32(REST))
    assert counters(rested)['battle_victories'] == counters(rested)['experience_gained'] == 0


def test_summon_defeat_is_not_campaign_defeat(current_game):
    _, initial, step, state = battle(current_game)
    state = state.replace(actor=jnp.int32(6), summon_battle=jnp.bool_(True),
        spell_party=jnp.stack((initial.unit_ids[:6], initial.unit_levels[:6], initial.unit_xp[:6], initial.hp[:6])),
        poison_mode=state.poison_mode.at[0].set(1), poison_damage=state.poison_damage.at[0].set(10000))
    after, ts = step(state, jnp.int32(CONTINUE))
    assert not after.in_battle and not after.lost
    assert counters(ts)['battle_defeats'] == 1
    assert counters(ts)['defeats'] == counters(ts)['battle_retreats'] == 0


def test_world_spell_survives_automatic_battle_and_stale_fields_are_ignored(current_game):
    env, initial, _, _ = current_game
    index = next(i for i, row in enumerate(env.casting.rows) if row['cast']['kind'] == 'summon_battle')
    world = initial.replace(last_cast_spell=jnp.int32(index))
    final = initial.replace(last_cast_spell=jnp.int32(-1))
    measure = jax.jit(lambda a, b, c, valid: campaign_events(env, a, b, c,
        jnp.int32(env.casting.start + index), valid, jnp.zeros(4, jnp.int32)))
    values = dict(zip(CAMPAIGN_FIELDS, map(float, measure(initial, world, final, jnp.bool_(True)))))
    assert values['summoned_units'] == values['spell_casts'] == 1
    values = measure(world, world, world, jnp.bool_(False))
    assert not jnp.any(values)


def test_potions_count_successful_doses_not_stale_events(current_game):
    env, initial, step, _ = current_game
    healing = env.potion_rules.keys.index('healing')
    state = initial.replace(hp=initial.hp.at[0].set(119),
                            potions=initial.potions.at[healing].set(1))
    action = jnp.int32(POTION_START + 6*healing)
    healed, ts = step(state, action)
    assert healed.potions[healing] == 0 and counters(ts)['potions_used'] == 1
    _, ts = step(healed, action)
    assert counters(ts)['potions_used'] == 0  # empty inventory / full HP
    _, ts = step(healed, jnp.int32(REST))
    assert counters(ts)['potions_used'] == 0


def test_dragon_arrival_uses_hero_entry_before_automatic_combat(current_game):
    env, initial, _, _ = current_game
    # Reuse the existing state shape; the counter depends on the configured
    # objective index, not the creature's appearance or a hard-coded enemy 31.
    goal = copy.copy(env)
    goal.map_config = {**env.map_config, 'campaign_objective': 'blue_dragon'}
    goal.objective_enemy = 11
    entered = initial.replace(in_battle=jnp.bool_(True), enemy=jnp.int32(11))
    dead = initial.replace(done=jnp.bool_(True), lost=jnp.bool_(True))
    measure = jax.jit(lambda before, world, after, valid: campaign_events(
        goal, before, world, after, jnp.int32(1), valid, jnp.zeros(4, jnp.int32)))
    index = CAMPAIGN_FIELDS.index('dragon_battles')
    assert measure(initial, entered, dead, jnp.bool_(True))[index] == 1
    assert measure(initial, entered, entered, jnp.bool_(True))[index] == 1
    for before, world, valid in (
        (entered, entered, True),  # subsequent combat turns are not new arrivals
        (initial, entered.replace(summon_battle=jnp.bool_(True)), True),
        (initial, entered.replace(enemy=jnp.int32(10)), True),
        (initial, initial.replace(enemy=jnp.int32(11)), True),
        (initial, entered, False),
    ):
        assert measure(before, world, dead, jnp.bool_(valid))[index] == 0


def test_gpu_reduction_keeps_every_total_and_latest_100_completions():
    n, fields = 130, len(CAMPAIGN_FIELDS)
    counts = jnp.zeros((n, 2, fields)).at[:, 0, CAMPAIGN_FIELDS.index('hired_units')].set(jnp.arange(n))
    counts = counts.at[:, 0, CAMPAIGN_FIELDS.index('victories')].set(1)
    dragon = CAMPAIGN_FIELDS.index('dragon_battles')
    counts = counts.at[:30, 0, dragon].set(1).at[110:, 0, dragon].set(3)
    counts = counts.at[:, 0, CAMPAIGN_FIELDS.index('battle_victories')].set(jnp.arange(n))
    counts = counts.at[:, 0, CAMPAIGN_FIELDS.index('unit_upgrades')].set(jnp.arange(n) % 4)
    mercenary = CAMPAIGN_FIELDS.index('mercenaries_hired')
    counts = counts.at[:10, 0, mercenary].set(2).at[:, 1, mercenary].set(5)
    trainer = CAMPAIGN_FIELDS.index('trainer_uses')
    counts = counts.at[:, 0, trainer].set(jnp.arange(n) % 3).at[:, 1, trainer].set(7)
    waves = CAMPAIGN_FIELDS.index('waves_completed')
    counts = counts.at[:, 0, waves].set(jnp.arange(n) % 12).at[:, 1, waves].set(11)
    terminal = jnp.zeros((n, 2), bool).at[:, 0].set(True)
    events = jnp.zeros_like(counts).at[:, 0].set(counts[:, 0])
    summary = jax.jit(summarize_campaign)({'is_terminal_step': terminal,
        'campaign_events': events, 'campaign_episode': counts,
        'episode_return': jnp.full(terminal.shape, 7.), 'episode_length': jnp.full(terminal.shape, 100)})
    tracker = CampaignMetrics(waves_enabled=True)
    tracker.add(jax.device_get(summary))
    data = tracker.metrics(reset_window=True)
    assert data['campaign/episodes_total'] == 130
    assert data['campaign/hired_units_total'] == sum(range(130))
    assert data['campaign/mercenaries_hired_total'] == 20
    assert data['campaign/completed/mercenaries_hired_mean'] == pytest.approx(20 / 130)
    assert data['campaign/mercenary_hire_episodes_rate'] == pytest.approx(10 / 130)
    assert data['campaign/window/mercenaries_hired_mean'] == 0
    assert data['campaign/trainer_uses_total'] == sum(i % 3 for i in range(130))
    assert data['campaign/completed/trainer_uses_mean'] == pytest.approx(sum(i % 3 for i in range(130)) / 130)
    assert data['campaign/window/trainer_uses_mean'] == pytest.approx(sum(i % 3 for i in range(30, 130)) / 100)
    assert data['campaign/hire_episodes_rate'] == pytest.approx(129 / 130)
    assert data['campaign/window/hired_units_mean'] == pytest.approx(sum(range(30, 130)) / 100)
    assert data['campaign/window/enemy_squads_defeated_sample_count'] == 100
    assert data['campaign/window/victory_rate'] == 1
    assert data['campaign/window/episode_return_mean'] == 7
    assert data['campaign/victories_total'] == 130
    assert data['campaign/dragon_reached_episodes_rate'] == pytest.approx(50 / 130)
    assert data['campaign/window/dragon_reached_episodes_rate'] == pytest.approx(.2)
    assert data['campaign/window/battle_victories_mean'] == pytest.approx(sum(range(30, 130)) / 100)
    assert data['campaign/window/unit_upgrades_mean'] == pytest.approx(sum(i % 4 for i in range(30, 130)) / 100)
    assert data['campaign/window/waves_completed_mean'] == pytest.approx(sum(i % 12 for i in range(30, 130)) / 100)
    empty = tracker.metrics()
    assert empty['campaign/window/episodes'] == 0
    assert 'campaign/window/victory_rate' not in empty
    assert empty['campaign/window/hired_units_mean'] == data['campaign/window/hired_units_mean']
    assert empty['campaign/window/dragon_reached_episodes_rate'] == pytest.approx(.2)
    assert empty['campaign/window/waves_completed_mean'] == data['campaign/window/waves_completed_mean']
    tracker.waves_enabled = False
    assert not any('waves_completed' in key for key in tracker.metrics())


@pytest.mark.parametrize('training_autoreset', ['contract'], indirect=True)
def test_episode_counters_reset_with_environment(training_autoreset):
    """Keep terminal telemetry/observations while resetting the next episode."""
    env, eval_env, step = training_autoreset
    state, reset_ts = compiled_method(env, 'reset')(jax.random.split(jax.random.PRNGKey(42), 2))
    plain_step = compiled_method(eval_env, 'step', batched=True)
    actions = jnp.full(2, REST, jnp.int32)
    for _ in range(2):
        _, terminal_ts = plain_step(state.unwrapped_state, actions)
        state, ts = step(state, actions)
        assert jnp.all(ts.truncated()) and not jnp.any(ts.terminated())
        assert jnp.all(ts.discount == 1)
        chex.assert_trees_all_equal(ts.observation, reset_ts.observation)
        chex.assert_trees_all_equal(ts.extras['next_obs'], terminal_ts.observation)
        assert jnp.any(ts.extras['next_obs']['observation'] != ts.observation['observation'])
        assert jnp.all(ts.extras['episode_metrics']['is_terminal_step'])
        assert jnp.all(ts.extras['episode_metrics']['campaign_episode'][:, CAMPAIGN_FIELDS.index('timeouts')] == 1)
        assert not jnp.any(state.campaign_counts)
