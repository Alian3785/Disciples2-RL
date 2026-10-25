"""Extended regressions for reference maps; run only on explicit request via run_gpu.sh.

Sources: Python reference map definitions (The Charmed Child uses the legacy source module);
campaign_env_territory.py scheduled armies; train_campaign.py REWARD_CONFIG.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, NumberGrid, load_map
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS, campaign_events
from stoix.envs import number_grid_reference_rewards as rewards


@pytest.fixture(scope='module')
def fortress():
    env = NumberGrid(map_config=load_map('super_last_stand'))
    return env, env.reset(jax.random.PRNGKey(7))[0]


def test_betrezen_identity_and_starting_assets(fortress):
    env, state = fortress
    row = env.progression.rows[int(state.unit_ids[4])]
    assert row['game_id'] == 'g000uu6001'
    assert row['hero'] and row['name'] == 'Бетрезен'
    assert row['exp_required'] == 150 and row['exp_increment'] == 500
    # The scenario's explicit early increment still stops at the native boundary.
    hero = state.unit_ids[4]
    levels = jnp.array([1,10,15,99],jnp.int32)
    assert jax.jit(env.progression.required_xp)(hero,levels).tolist() == [150,4650,4650,4650]
    assert int(state.hp[4]) == 300 and int(state.gold) == 3000
    # User-requested SLS starting setup (2026-10-22): no prebuilt Temple.
    assert not bool(env.capital.temple_built(state))
    assert env.item_rules.skill_levels[:2] == (1, 1)


def test_future_waves_are_not_targets_or_completed_goals(fortress):
    env, state = fortress
    assert int(jnp.sum(state.alive)) == 7
    assert bool(env._goals_left(jnp.zeros_like(state.alive), state.scenario_spawned))
    future = env.scenario.indices[0]
    assert int(env.territory.enemy_at(state, state.enemy_positions[future])) == -1
    assert not bool(env._goals_left(jnp.zeros_like(state.alive), jnp.ones_like(state.scenario_spawned)))


def test_spawn_turn_and_pursuit(fortress):
    env, state = fortress
    advance = jax.jit(env.scenario.advance)
    assert int(jnp.sum(advance(state.replace(day=jnp.int32(5))).alive)) == 7
    after = advance(state.replace(day=jnp.int32(6)))
    assert int(jnp.sum(after.alive)) == 8
    enemy = env.scenario.indices[0]
    assert bool(jnp.any(after.enemy_positions[enemy] != state.enemy_positions[enemy]))


def test_queued_attack_is_the_only_world_action(fortress):
    env, state = fortress
    enemy = env.scenario.indices[0]
    queued = state.replace(alive=state.alive.at[enemy].set(True),
                           pending_encounters=state.pending_encounters.at[enemy].set(True))
    assert int(jnp.sum(env.action_mask(queued))) == 1
    assert bool(env.action_mask(queued)[CONTINUE])


def test_wave_bonus_requires_every_member_and_is_paid_once(fortress):
    env, state = fortress
    members = env.scenario.indices[:3]
    active = state.replace(scenario_spawned=state.scenario_spawned.at[members].set(True),
                            alive=state.alive.at[members].set(True))
    two = active.replace(alive=active.alive.at[members[:2]].set(False))
    complete = two.replace(alive=two.alive.at[members[2]].set(False))
    assert float(env.scenario.wave_bonus(active, two)) == 0
    assert float(env.scenario.wave_bonus(two, complete)) == 4
    assert float(env.scenario.wave_bonus(complete, complete)) == 0
    wave_index = CAMPAIGN_FIELDS.index('waves_completed')
    count_waves = jax.jit(lambda before, after: campaign_events(
        env, before, after, after, CONTINUE, jnp.bool_(True), jnp.zeros(4, jnp.int32))[wave_index])
    assert float(count_waves(state, state)) == 0  # Future armies are not cleared waves.
    assert float(count_waves(state, active)) == 0  # Spawning is not completion.
    assert float(count_waves(active, two)) == 0
    assert float(count_waves(two, complete)) == 1
    assert float(count_waves(complete, complete)) == 0
    assert float(count_waves(two, complete.replace(done=jnp.bool_(True), won=jnp.bool_(True)))) == 1


def test_timeout_penalty_is_not_repeated(fortress):
    env, state = fortress
    terminal, reward = rewards.finish(env, state, state.replace(done=jnp.bool_(True)))
    assert float(reward) == -8
    assert float(rewards.finish(env, terminal, terminal)[1]) == 0


def test_siege_and_charmed_child_starting_rosters_and_goals():
    """Siege and Charmed Child parties and goals; S117 capital potions and original garrisons."""
    siege = load_map('siege_train')
    assert siege['hero_roster'] == ['possessed', 'duke', 'possessed', None, 'cultist', None]
    assert siege['reference_enemy_ids'][siege['objective_enemy']] == 100
    assert siege['scheduled_enemies'][0]['spawn_turn'] == 27
    assert sum(siege['initial_potions'].values()) == 20
    charmed_child = load_map('the_charmed_child')
    assert charmed_child['hero_roster'][0] == 'reference_g000uu0036'
    assert len(charmed_child['enemy_rosters']) == 71
    assert sum(c['objective'] for c in charmed_child['cities']) == 3
    # Installed S117: the capital vault's potions; the Empire capital's Cleric and Hasser's Keep's Holy Avenger.
    assert charmed_child['initial_potions'] == {'restoration': 3, 'life': 1}
    assert charmed_child['enemy_rosters'][4][4] == 'cleric' and charmed_child['enemy_rosters'][7][5] == 'blade_master'
    assert charmed_child['enemy_hp'][7] == 250 and not any(
        key in r for r in charmed_child['enemy_rosters'] for key in ('priest', 'defender_of_faith'))
