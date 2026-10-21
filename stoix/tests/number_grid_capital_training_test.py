"""Extended GPU cases for user-requested Capital rewards; not part of core.

Source: user request 2026-10-16 and docs/DEFAULT_MAP.md. These are RL rewards,
not claims about original Disciples II mechanics. Run only on explicit request.
"""
from copy import deepcopy

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import REST, NumberGrid, load_map
from stoix.envs.number_grid_capital_training import CapitalTrainingRules
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS
from stoix.envs.number_grid_potions import POTION_START
from stoix.envs.number_grid_recruitment import RecruitmentRules
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state
from stoix.tests.number_grid_target_level_rewards_test import developed_party, enter, finish, near_enemy


@pytest.fixture(scope='module')
def capital():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'
    env = NumberGrid(map_config=load_map('default-capital'))
    return env, compiled_method(env, 'reset')(jax.random.PRNGKey(42))[0]


def defended_battle(env, initial):
    state = initial.replace(potions=jnp.full_like(initial.potions, 5))
    for key in ('invulnerability', 'highfather'):
        action = jnp.int32(POTION_START + 6 * env.potion_rules.keys.index(key) + 1)
        state, _ = compiled_method(env, 'step')(state, action)
    near, action = near_enemy(env, state, env.objective_enemy)
    return enter(env, near, action)[0]


def test_reset_legions_and_reward_memory_observation(capital):
    env, state = capital
    assert env.construction.faction == 'legions'
    assert env.construction.lord['id'] == 'warrior'
    assert env.map_config['hero_roster'] == ['possessed', 'duke', 'possessed', None, 'cultist', None]
    assert env.round_limit_penalty == env.episode_timeout_penalty == 100.
    assert not state.capital_defense_acquired and not state.capital_rounds_rewarded
    assert not jnp.any(state.capital_defense_peak)
    assert int(state.capital_entry_strength) == -1
    assert env.capital_training.entry_level_reward == 2.
    assert float(env.capital_training.observation(state)[-1]) == pytest.approx(-.01)
    assert compiled_method(env, 'observation')(state).shape == (env.observation_size,)


def test_actual_chest_step_pays_acquisition_and_emits_metric(capital):
    env, state = capital
    chest = next(c for c in env.map_config['chests'] if c.get('potions', {}).get('invulnerability'))
    position = jnp.array(chest['position'], jnp.int32)
    blocked = env.territory.blocked_grid.reshape(env.size, env.size)
    action, direction = next((i, d) for i, d in enumerate(env.directions.tolist())
                            if not bool(blocked[position[0]-d[0], position[1]-d[1]]))
    state = state.replace(position=position-jnp.array(direction), movement_points=jnp.int32(100))
    assert compiled_method(env, 'action_mask')(state)[action]
    after, ts = compiled_method(env, 'step')(state, jnp.int32(action))
    assert int(after.capital_defense_acquired) == 1
    assert float(ts.reward) >= 3. - env.step_cost
    assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('capital_defense_acquired')] == 1
    _, no_repeat = env.capital_training.acquire(after, after)
    assert float(no_repeat) == 0.
    healed = after.replace(potions=after.potions.at[env.potion_rules.keys.index('healing')].add(5))
    assert float(env.capital_training.acquire(after, healed)[1]) == 0.


def test_effects_include_health_and_cannot_be_farmed_by_reentry(capital):
    env, initial = capital
    rules = env.capital_training
    battle = defended_battle(env, initial)
    reward = jax.jit(rules.battle)
    credited, amount = reward(battle, battle.hp, battle.escaped, battle.round)
    assert float(amount) == 4. and int(credited.capital_defense_peak.sum()) == 2
    assert int(credited.capital_defense_peak[rules.keys.index('highfather')]) == 1
    repeated, amount = reward(credited, credited.hp, credited.escaped, credited.round)
    assert float(amount) == 0.
    # Moving the same buffed fighter to another slot leaves all peaks unchanged.
    from stoix.envs.number_grid_unit_swaps import SLOT_FIELDS
    swapped = repeated.replace(**{name: getattr(repeated, name).at[jnp.array([0, 1])].set(
        getattr(repeated, name)[jnp.array([1, 0])]) for name in SLOT_FIELDS})
    assert float(reward(swapped, swapped.hp, swapped.escaped, swapped.round)[1]) == 0.


def test_dead_summoned_and_non_goal_effects_do_not_pay(capital):
    env, initial = capital
    battle = defended_battle(env, initial)
    reward = jax.jit(env.capital_training.battle)
    for excluded in (battle.replace(enemy=jnp.int32(30)), battle.replace(summon_battle=jnp.bool_(True)),
                     battle.replace(hp=battle.hp.at[1].set(0)), battle.replace(post_victory=jnp.int32(1)),
                     battle.replace(summon_owner=battle.summon_owner.at[1].set(0))):
        assert float(reward(excluded, excluded.hp, excluded.escaped, excluded.round)[1]) == 0.
    # It was active during this activation even if the first incoming hit kills the fighter.
    assert float(reward(battle, battle.hp.at[1].set(0), battle.escaped, battle.round)[1]) == 4.


def test_new_rounds_are_once_per_episode_and_never_pay_for_timeout(capital):
    env, initial = capital
    near, action = near_enemy(env, initial, env.objective_enemy)
    battle = enter(env, near, action)[0]
    reward = jax.jit(env.capital_training.battle)
    after, amount = reward(battle, battle.hp, battle.escaped, jnp.int32(2))
    assert float(amount) == .5 and int(after.capital_rounds_rewarded) == 1
    assert float(reward(after, after.hp, after.escaped, jnp.int32(2))[1]) == 0.
    assert float(reward(after, after.hp, after.escaped, jnp.int32(env.max_rounds+1))[1]) == 0.
    assert float(reward(battle, battle.hp.at[:6].set(0), battle.escaped, jnp.int32(2))[1]) == 0.


def test_episode_timeout_penalty(capital):
    env, initial = capital
    ending = initial.replace(step_count=jnp.int32(env.max_steps-1))
    after, ts = compiled_method(env, 'step')(ending, jnp.int32(REST))
    assert after.done and not after.lost and not after.won
    assert float(ts.reward) <= -100.
    assert float(compiled_method(env, 'step')(after, jnp.int32(REST))[1].reward) == 0.


@pytest.mark.parametrize('levels,expected', [([1, 1, 1, 1], 0.), ([2, 2, 3, 1], 8.),
                                          ([4, 4, 4, 4], 24.)])
def test_first_guardian_entry_pays_strength_immediately(capital, levels, expected):
    """Pay on contact, not victory; even a zero payout consumes the first encounter."""
    env, initial = capital
    party = hero_roster_state(env, initial, ['possessed', 'duke', 'possessed', 'cultist', None, None])
    party = party.replace(unit_levels=party.unit_levels.at[:4].set(jnp.array(levels)))
    before, action = near_enemy(env, party, env.objective_enemy)
    assert compiled_method(env, 'action_mask')(before)[action]
    battle, amount = enter(env, before, action)
    _, plain = enter(env, before.replace(capital_entry_strength=jnp.int32(0)), action)
    assert float(amount-plain) == pytest.approx(expected)
    assert int(battle.capital_entry_strength) == int(expected/2.)
    assert not battle.done and not battle.won
    assert float(env.capital_training.observation(battle)[-1]) == pytest.approx(expected/200.)


def test_entry_counts_living_levels_and_emits_step_metrics(capital):
    """A wounded level-5 elite counts; a dead mage, empty slot and potion HP do not."""
    env, initial = capital
    party = developed_party(env, initial)
    party = party.replace(hp=party.hp.at[0].set(1).at[3].set(0),
        unit_levels=party.unit_levels.at[2].set(99), potion_bonus=party.potion_bonus.at[0, 0].add(100))
    before, action = near_enemy(env, party, env.objective_enemy)
    after, ts = compiled_method(env, 'step')(before, action)
    _, plain = compiled_method(env, 'step')(before.replace(capital_entry_strength=jnp.int32(0)), action)
    assert float(ts.reward-plain.reward) == pytest.approx(8.)
    # The snapshot and telemetry survive automatic enemy attacks, including a fatal hit.
    assert int(after.capital_entry_strength) == 4
    events = ts.extras['campaign_events']
    assert events[CAMPAIGN_FIELDS.index('capital_first_contacts')] == 1
    assert events[CAMPAIGN_FIELDS.index('capital_entry_strength')] == 4


@pytest.mark.parametrize('developed', [False, True])
def test_retreat_reentry_and_later_levels_cannot_repeat_entry_reward(capital, developed):
    env, initial = capital
    party = developed_party(env, initial) if developed else initial
    before, action = near_enemy(env, party, env.objective_enemy)
    battle, _ = enter(env, before, action)
    retreated, _ = finish(env, battle, 'retreat')
    stronger = retreated.replace(unit_levels=retreated.unit_levels.at[:6].add(3))
    before, action = near_enemy(env, stronger, env.objective_enemy)
    again, reward = enter(env, before, action)
    _, fresh_reward = enter(env, before.replace(capital_entry_strength=jnp.int32(-1)), action)
    assert int(again.capital_entry_strength) == int(battle.capital_entry_strength)
    assert float(fresh_reward-reward) == pytest.approx(2.*int(again.army_battle_level_strength))
    lost, _ = finish(env, again, 'defeat')
    assert lost.done and int(lost.capital_entry_strength) == int(battle.capital_entry_strength)
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(43))
    assert int(reset.capital_entry_strength) == -1


def test_other_armies_invalid_actions_and_separate_summons_cannot_claim_entry(capital):
    env, initial = capital
    before, action = near_enemy(env, developed_party(env, initial), 30)
    other, _ = enter(env, before, action)
    assert int(other.capital_entry_strength) == -1
    before, action = near_enemy(env, initial, env.objective_enemy)
    invalid, _ = compiled_method(env, 'step')(before.replace(movement_points=jnp.int32(0)), action)
    assert not invalid.in_battle and int(invalid.capital_entry_strength) == -1
    battle, _ = enter(env, before, action)
    unclaimed = battle.replace(capital_entry_strength=jnp.int32(-1))
    for excluded in (unclaimed.replace(summon_battle=jnp.bool_(True)),
                     unclaimed.replace(bot_autobattle=jnp.bool_(True)),
                     unclaimed.replace(done=jnp.bool_(True)),
                     unclaimed.replace(hp=unclaimed.hp.at[6:].set(0))):
        after, reward = env.capital_training.enter(excluded, jnp.bool_(True), jnp.int32(10))
        assert int(after.capital_entry_strength) == -1 and float(reward) == 0.


@pytest.mark.parametrize('value', [-1., True, float('inf'), float('nan')])
def test_entry_level_coefficient_must_be_finite_nonnegative(capital, value):
    env, _ = capital
    with pytest.raises(ValueError, match='capital_entry_level_reward'):
        CapitalTrainingRules(env, {**load_map('default-capital'), 'capital_entry_level_reward': value})


def at_camp(env, state, offer):
    site = int(env.recruitment.offer_sites[offer - 5])
    return state.replace(position=jnp.array(env.site_rules.sites[site]['interaction_tiles'][0], jnp.int32),
                         gold=jnp.int32(9999))


def test_capital_camp_hires_pay_zero_and_keep_vacancy_penalty(capital):
    """Every native camp offer consumes real leadership but fills no reward slots."""
    env, initial = capital
    rules = env.recruitment
    empty = hero_roster_state(env, initial, [None, 'duke', None, None, None, None]).replace(
        hire_rewarded_capacity=jnp.int32(0))
    for offer in range(5, rules.offer_count):
        before = at_camp(env, empty, offer)
        action = jnp.int32(rules.start + offer)
        assert compiled_method(env, 'action_mask')(before)[action]
        after, ts = compiled_method(env, 'step')(before, action)
        slot = int(after.last_target)
        assert after.mercenary_units[slot] and jnp.sum(after.mercenary_units) == 1
        assert int(rules.composition(after)[1]) == int(rules.sizes[offer])
        assert rules.reward_occupancy(after) == after.hire_rewarded_capacity == 0
        assert rules.reward_vacancies(after) == rules.reward_vacancies(before) == 3
        assert float(ts.reward) == pytest.approx(-env.step_cost - .15)
        assert rules.quotes(after)['reward_vacancies'] == 3
    assert rules.metadata['mercenary_hire_reward'] == 0.
    assert not rules.metadata['mercenaries_reduce_vacancy_penalty']


def test_regular_replacement_fills_reward_slot_without_mercenary_consuming_bonus(capital):
    env, initial = capital
    rules = env.recruitment
    # Level 3 unlocks a fourth companion; the original three already claimed their places.
    party = initial.replace(unit_levels=initial.unit_levels.at[1].set(3))
    offer = rules.keys.index('goblin_archer')
    hired, ts = compiled_method(env, 'step')(at_camp(env, party, offer), jnp.int32(rules.start + offer))
    slot = int(hired.last_target)
    assert rules.composition(hired)[2] == 0 and rules.reward_vacancies(hired) == 1
    assert hired.hire_rewarded_capacity == 3 and float(ts.reward) == pytest.approx(-env.step_cost - .05)
    assert not jnp.any(rules.available(hired)[:rules.offer_count])  # no extra real leadership
    dismissed, _ = compiled_method(env, 'step')(hired, jnp.int32(rules.dismiss_start + slot))
    assert not jnp.any(dismissed.mercenary_units) and rules.reward_vacancies(dismissed) == 1
    base = dismissed.replace(position=initial.position)
    action = jnp.int32(rules.start + 1)  # ordinary faction cultist
    assert compiled_method(env, 'action_mask')(base)[action]
    regular, ts = compiled_method(env, 'step')(base, action)
    assert not jnp.any(regular.mercenary_units) and rules.reward_vacancies(regular) == 0
    assert regular.hire_rewarded_capacity == 4 and float(ts.reward) == pytest.approx(3. - env.step_cost)
    dismissed, _ = compiled_method(env, 'step')(regular, jnp.int32(rules.dismiss_start + slot))
    rehired, ts = compiled_method(env, 'step')(dismissed, action)
    assert rehired.hire_rewarded_capacity == 4 and float(ts.reward) == pytest.approx(-env.step_cost)


def test_camp_origin_survives_promotion_swaps_battle_death_and_reset(capital):
    env, initial = capital
    rules = env.recruitment
    party = initial.replace(unit_levels=initial.unit_levels.at[1].set(3))
    offer = rules.keys.index('goblin_archer')
    hired, _ = compiled_method(env, 'step')(at_camp(env, party, offer), jnp.int32(rules.start + offer))
    slot = int(hired.last_target)
    before, action = near_enemy(env, hired, 30)
    battle, _ = enter(env, before, action)
    # A real XP award at victory promotes the mercenary without changing its origin.
    threshold = env.unit_experience(battle)[slot, 2]
    battle = battle.replace(unit_xp=battle.unit_xp.at[slot].set(threshold - 1),
        native_xp=battle.native_xp.at[slot].set(threshold - 1),
        battle_earned_xp=battle.battle_earned_xp.at[slot].set(100))
    promoted, _ = finish(env, battle)
    assert promoted.unit_levels[slot] > hired.unit_levels[slot]
    assert promoted.mercenary_units[slot] and rules.reward_vacancies(promoted) == 1
    swaps = env.unit_swaps
    swap = jnp.int32(swaps.start + swaps.pairs.index((0, slot)))
    moved, _ = compiled_method(env, 'step')(promoted, swap)
    assert moved.mercenary_units[0] and not moved.mercenary_units[slot]
    assert rules.reward_vacancies(moved) == 1
    assert rules.observation(moved)[-6:].tolist() == [1., 0., 0., 0., 0., 0.]
    assert compiled_method(env, 'observation')(moved).shape == (env.observation_size,)
    dead = moved.replace(hp=moved.hp.at[0].set(0))
    assert rules.reward_vacancies(dead) == 1
    before, action = near_enemy(env, dead, env.objective_enemy)
    battle, _ = enter(env, before, action)
    lost, _ = finish(env, battle, 'defeat')
    assert lost.mercenary_units[0] and rules.reward_vacancies(lost) == 1
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(44))
    assert not jnp.any(reset.mercenary_units)


@pytest.mark.parametrize('faction_only', [False, True])
def test_hire_origin_not_unit_type_controls_reward_and_other_maps_keep_rule(capital, faction_only):
    env, initial = capital
    config = deepcopy(env.map_config)
    config['recruitment_reward_faction_only'] = faction_only
    config['mercenaries'][0]['roster'][0]['unit'] = 'possessed'
    rules = RecruitmentRules(config, env.progression, env.construction, env.recruitment.start, env.site_rules)
    rules.territory = env.territory
    apply = jax.jit(lambda s, a: rules.apply(s, a, env._clear_roster_slots))
    empty = hero_roster_state(env, initial, [None, 'duke', None, None, None, None]).replace(
        hire_rewarded_capacity=jnp.int32(0), gold=jnp.int32(9999))
    mercenary, reward = apply(at_camp(env, empty, 5), jnp.int32(rules.start + 5))
    regular, regular_reward = apply(empty, jnp.int32(rules.start))
    assert mercenary.unit_ids[0] == regular.unit_ids[0]  # identical unit, different hiring source
    assert mercenary.mercenary_units[0] and not regular.mercenary_units[0]
    assert float(reward) == (0. if faction_only else 3.)
    assert rules.reward_vacancies(mercenary) == (3 if faction_only else 2)
    assert float(regular_reward) == 3. and rules.reward_vacancies(regular) == 2
