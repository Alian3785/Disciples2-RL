"""Target-only level rewards, requested 2026-10-14; Default Capital added 2026-10-16.

Extended CUDA regressions. Run only on explicit request through scripts/run_gpu.sh.
These rewards are training rules, not original game rules. Sources: docs/DEFAULT_MAP.md
(reference maps/default.py for Mizrael and the green dragon), docs/REFERENCE_CAMPAIGNS.md.
"""
import json

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, REST, NumberGrid, action_names, load_map
from stoix.envs.number_grid_casting import SUMMON
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


@pytest.fixture(scope='module', params=['default-blue-dragon', 'default-capital'])
def target_game(request):
    """Share each target campaign and reset state across the extended reward cases."""
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'
    env = NumberGrid(map_config=load_map(request.param))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, state


def near_enemy(env, state, enemy):
    """Place the hero on a passable approach with the target spawned and movement available."""
    position = state.enemy_positions[enemy]
    blocked = env.territory.blocked_grid.reshape(env.size, env.size)
    direction, offset = next((i, d) for i, d in enumerate(env.directions.tolist())
        if not bool(blocked[position[0]-d[0], position[1]-d[1]]))
    return state.replace(position=position-jnp.array(offset, jnp.int32),
        movement_points=jnp.int32(100), alive=state.alive.at[enemy].set(True),
        scenario_spawned=state.scenario_spawned.at[enemy].set(True)), jnp.int32(direction)


def enter(env, before, action):
    """Execute a deterministic world transition to capture the battle-entry reward snapshot."""
    return compiled_method(env, '_world_step')(before, action,
        jax.random.PRNGKey(19), jnp.full(env.random_size, .5))


def finish(env, battle, outcome='victory'):
    """Resolve a prepared victory, retreat, or defeat without pending healing turns."""
    if outcome == 'victory':
        battle = battle.replace(hp=battle.hp.at[6:].set(0))
    elif outcome == 'retreat':
        battle = battle.replace(escaped=battle.escaped.at[:6].set(True))
    else:
        battle = battle.replace(hp=battle.hp.at[:6].set(0))
    battle = battle.replace(actor=jnp.int32(1), round=jnp.int32(1),
        post_victory=jnp.int32(1 if outcome == 'victory' else 2),
        pending_healers=jnp.zeros(12, bool))
    return compiled_method(env, '_battle_step')(battle, jnp.int32(DEFEND),
        jax.random.PRNGKey(20), jnp.full(env.random_size, .5))


def developed_party(env, initial):
    """Build a mixed-level hero party that can expose incorrect level-strength accounting."""
    return hero_roster_state(env, initial,
        ['defender_of_faith', 'duke', None, 'white_mage', None, None])


def test_victory_snapshot_is_observable_and_paid_only_on_victory(target_game):
    """Freeze living entry levels, expose the snapshot, and pay it exactly once on target victory."""
    env, initial = target_game
    target = env.objective_enemy
    assert env.army_victory_level_reward == 5.
    assert env.army_victory_level_objective_only
    assert jnp.flatnonzero(env.army_victory_level_targets).tolist() == [target]
    base = developed_party(env, initial)
    # Wounded level-5 mercenary + level-1 hero; dead and empty slots add nothing.
    base = base.replace(hp=base.hp.at[0].set(1).at[3].set(0),
        unit_levels=base.unit_levels.at[2].set(99),
        potion_bonus=base.potion_bonus.at[0, 0].add(100).at[3, 0].add(500))
    before, action = near_enemy(env, base, target)
    assert compiled_method(env, 'action_mask')(before)[action]
    arrived, entry_reward = enter(env, before, action)
    claimed, plain_entry = enter(env, before.replace(
        army_victory_rewarded=before.army_victory_rewarded.at[target].set(True)), action)
    assert arrived.in_battle and int(arrived.enemy) == target
    assert float(entry_reward) == pytest.approx(float(plain_entry))
    assert int(arrived.army_battle_level_strength) == 4
    assert not jnp.any(arrived.army_victory_rewarded)
    observation = compiled_method(env, 'observation')(arrived)
    assert observation.shape == (env.observation_size,)
    assert float(observation[-1]) == pytest.approx(.04)
    won, reward = finish(env, arrived)
    _, plain = finish(env, claimed)
    assert won.won and won.done and not won.alive[target]
    assert float(reward-plain) == pytest.approx(20.)
    assert jnp.flatnonzero(won.army_victory_rewarded).tolist() == [target]
    assert int(won.army_battle_level_strength) == 0
    # HP changes, casualties and level gains after entry cannot change the snapshot.
    changed = arrived.replace(hp=arrived.hp.at[0].set(0),
        unit_levels=arrived.unit_levels.at[1].add(3),
        potion_bonus=arrived.potion_bonus.at[1, 0].add(500))
    _, changed_reward = finish(env, changed)
    _, changed_plain = finish(env, changed.replace(army_victory_rewarded=won.army_victory_rewarded))
    assert float(changed_reward-changed_plain) == pytest.approx(20.)
    _, repeated = finish(env, claimed.replace(army_battle_level_strength=jnp.int32(100)))
    assert float(repeated) == pytest.approx(float(plain))
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(43))
    assert not jnp.any(reset.army_victory_rewarded)
    assert int(reset.army_battle_level_strength) == 0


@pytest.mark.parametrize('outcome', ['retreat', 'defeat'])
def test_unsuccessful_target_battle_discards_snapshot(target_game, outcome):
    """Discard unpaid strength after retreat or defeat and recalculate it on a later entry."""
    env, initial = target_game
    before, action = near_enemy(env, developed_party(env, initial), env.objective_enemy)
    arrived, _ = enter(env, before, action)
    after, reward = finish(env, arrived, outcome)
    _, plain = finish(env, arrived.replace(army_battle_level_strength=jnp.int32(0)), outcome)
    assert float(reward) == pytest.approx(float(plain))
    assert int(after.army_battle_level_strength) == 0
    assert not jnp.any(after.army_victory_rewarded)
    if outcome == 'retreat':
        stronger = after.replace(unit_levels=after.unit_levels.at[1].add(2))
        before, action = near_enemy(env, stronger, env.objective_enemy)
        retried, _ = enter(env, before, action)
        assert int(retried.army_battle_level_strength) == int(arrived.army_battle_level_strength)+2
        _, reward = finish(env, retried)
        _, plain = finish(env, retried.replace(army_battle_level_strength=jnp.int32(0)))
        assert float(reward-plain) == pytest.approx(5.*int(retried.army_battle_level_strength))


def test_other_armies_and_invalid_moves_cannot_earn_target_bonus(target_game):
    """Prevent unrelated encounters, stale snapshots, and rejected movement from earning the bonus."""
    env, initial = target_game
    # Capital keeps the green dragon as an ordinary enemy, with no goal bonus.
    other = 30 if env.map_config['campaign_objective'] == 'capital_guardian' else 0
    before, action = near_enemy(env, developed_party(env, initial), other)
    arrived, _ = enter(env, before, action)
    assert compiled_method(env, 'action_mask')(before)[action]
    assert arrived.in_battle and int(arrived.enemy) == other
    assert int(arrived.army_battle_level_strength) == 0
    # The payment boundary also checks the target, even with a stale snapshot.
    after, reward = finish(env, arrived.replace(army_battle_level_strength=jnp.int32(99)))
    _, plain = finish(env, arrived)
    assert float(reward) == pytest.approx(float(plain))
    assert float(reward) == pytest.approx(1.)
    assert not after.won and not after.done and not after.alive[other]
    assert not jnp.any(after.army_victory_rewarded)
    near, action = near_enemy(env, developed_party(env, initial), env.objective_enemy)
    invalid, _ = compiled_method(env, 'step')(near.replace(movement_points=jnp.int32(0)), action)
    assert not invalid.in_battle and not jnp.any(invalid.army_victory_rewarded)
    assert int(invalid.army_battle_level_strength) == 0


def test_summoned_army_cannot_earn_target_level_bonus(target_game):
    """Exclude spell-created armies from the hero's objective-level victory reward."""
    env, initial = target_game
    near, _ = near_enemy(env, initial, env.objective_enemy)
    spell = next(i for i in range(env.casting.book_count) if int(env.casting.kinds[i]) == SUMMON)
    near = near.replace(learned_spells=jnp.uint32(1 << spell), mana=jnp.full(5, 5000, jnp.int32))
    action = jnp.int32(env.casting.start+spell)
    if not bool(env.casting.field[env.objective_enemy]):
        # Capital guardians remain protected from map spells and separate summons.
        near = near.replace(alive=jnp.zeros_like(near.alive).at[env.objective_enemy].set(True))
        assert not compiled_method(env, 'action_mask')(near)[action]
        after, _ = compiled_method(env, 'step')(near, action)
        assert not after.in_battle and not after.won and not after.done
        assert after.alive.tolist() == near.alive.tolist()
        assert not jnp.any(after.army_victory_rewarded)
        return
    battle, _ = enter(env, near, action)
    assert battle.summon_battle and int(battle.enemy) == env.objective_enemy
    assert int(battle.army_battle_level_strength) == 0
    after, reward = finish(env, battle.replace(army_battle_level_strength=jnp.int32(99)))
    _, plain = finish(env, battle)
    assert float(reward) == pytest.approx(float(plain))
    assert not jnp.any(after.army_victory_rewarded)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_invalid_target_scope_is_rejected(value):
    """Reject non-boolean values for the objective-only level reward option."""
    with pytest.raises(ValueError, match='army_victory_level_objective_only'):
        NumberGrid(map_config={**load_map('default-blue-dragon'), 'army_victory_level_objective_only': value})


def test_target_scope_requires_an_objective_and_enabled_reward():
    """Require an army objective and a positive reward when objective-only scope is requested."""
    with pytest.raises(ValueError, match='requires objective_enemy'):
        NumberGrid(map_config={**load_map('trade_train'), 'army_victory_level_objective_only': True})
    with pytest.raises(ValueError, match='requires objective_enemy'):
        NumberGrid(map_config={**load_map('default-blue-dragon'), 'army_victory_level_reward': 0.})


def test_unrelated_maps_keep_level_rewards_disabled():
    """Keep target-level shaping disabled on campaigns that did not opt into it."""
    for name in ('default', 'default-green-dragon', 'super_last_stand', 'the_charmed_child', 'siege_train'):
        assert not load_map(name).get('army_victory_level_reward', 0.)
        assert not load_map(name).get('capital_entry_level_reward', 0.)
        assert not load_map(name).get('capital_guardian_battle_reward', 0.)
        assert not load_map(name).get('capital_incubus_reward', 0.)
        assert not load_map(name).get('capital_incubus_building_reward', 0.)
        assert not load_map(name).get('capital_guardian_rounds_bonus', 0.)
        assert not load_map(name).get('recruitment_reward_faction_only', False)
    assert not load_map('default-blue-dragon').get('capital_entry_level_reward', 0.)
    assert not load_map('default-blue-dragon').get('capital_guardian_battle_reward', 0.)
    assert not load_map('default-blue-dragon').get('recruitment_reward_faction_only', False)
    assert not load_map('trade_train').get('army_victory_level_objective_only', False)


@pytest.mark.parametrize('levels,expected', [([1, 1, 1, 1], 0.), ([2, 2, 3, 1], 20.),
                                          ([4, 4, 4, 4], 60.)])
def test_target_bonus_grows_with_entry_levels(target_game, levels, expected):
    """Verify exact victory payouts for representative party levels at battle entry."""
    env, initial = target_game
    base = hero_roster_state(env, initial, ['possessed', 'duke', 'possessed', 'cultist', None, None])
    base = base.replace(unit_levels=base.unit_levels.at[:4].set(jnp.array(levels)))
    before, action = near_enemy(env, base, env.objective_enemy)
    battle, _ = enter(env, before, action)
    assert int(battle.army_battle_level_strength) == expected/5.
    won, reward = finish(env, battle)
    _, plain = finish(env, battle.replace(army_victory_rewarded=won.army_victory_rewarded))
    assert float(reward-plain) == pytest.approx(expected)
    assert float(reward) == pytest.approx(4.+expected)
    assert bool(won.army_victory_rewarded[env.objective_enemy])
    assert not jnp.any(compiled_method(env, 'action_mask')(won))
    again, ts = compiled_method(env, 'step')(won, jnp.int32(REST))
    assert again.won and again.done and float(ts.reward) == 0.


def test_default_capital_preserves_simpler_times_and_blue_reward_settings(tmp_path):
    """User request 2026-10-16: only Mizrael is the goal; no events or dragon replacement."""
    base, capital, blue = (load_map(name) for name in ('default', 'default-capital', 'default-blue-dragon'))
    changed = {'name', 'display_name', 'campaign_objective', 'objective_enemy',
               'objective_reward_hero_only', 'army_victory_level_reward',
               'army_victory_level_objective_only', 'scenario_events_enabled', 'cities', 'faction', 'lord_type',
               'capital_guardian_battle_reward', 'capital_guardian_battle_min_defeated',
               'capital_defense_acquisition_reward', 'capital_incubus_reward', 'capital_incubus_building_reward',
               'capital_guardian_rounds_bonus', 'capital_guardian_rounds_bonus_rounds',
               'episode_timeout_penalty', 'battle_round_limit_penalty', 'recruitment_reward_faction_only',
               'enemy_rosters', 'enemy_units', 'enemy_hp', 'enemy_initial_levels', 'absent_enemies', 'bot_attack',
               'hero_roster'}
    assert {k: v for k, v in capital.items() if k not in changed} == {
        k: v for k, v in base.items() if k not in changed}
    assert capital['cities'] == [{**city, 'objective': False} for city in base['cities']]
    assert sum(c['objective'] for c in load_map('default')['cities']) == 2
    assert capital['display_name'] == 'Default — Capital'
    assert capital['objective_enemy'] == capital['enemy_capitals'][0]['guard'] == 74
    assert capital['opponent_positions'][74] == [11, 27]
    assert capital['enemy_rosters'][74] == [None, None, None, None, 'g000uu3001', None]
    assert capital['enemy_rosters'][30] == [None, 'g000uu5023', None, None, None, None]
    assert capital['enemy_full_regeneration'] == [30]
    assert capital['scenario_events_enabled'] is False
    assert not capital.get('scenario_events') and not capital.get('scheduled_enemies')
    assert not capital.get('scripted_bot')
    for key in ('objective_reward_hero_only', 'army_victory_level_reward', 'army_victory_level_objective_only'):
        assert capital[key] == blue[key]
    assert capital['faction'] == 'legions' and capital['lord_type'] == 'warrior'
    # User request 2026-10-26: no Mizrael reward besides victory, except the +10 battle bonus
    # once more than 50 other armies were defeated.
    for key in ('capital_defense_effect_reward', 'capital_guardian_round_reward', 'capital_entry_level_reward'):
        assert key not in capital
    assert capital['capital_guardian_battle_reward'] == 10.
    # Later the same day: the bonus needs more than 70, +3 per defensive bottle again, +25 per new Incubus.
    assert capital['capital_guardian_battle_min_defeated'] == 71
    assert capital['capital_defense_acquisition_reward'] == 3. and capital['capital_incubus_reward'] == 25.
    assert capital['capital_incubus_building_reward'] == 12.5
    assert (capital['capital_guardian_rounds_bonus'], capital['capital_guardian_rounds_bonus_rounds']) == (12.5, 10)
    assert capital['recruitment_reward_faction_only'] is True
    # User request 2026-10-26: the armies the Empire hero Edwin destroyed by turn 21 of
    # SaveGame/6789.sg are absent; the Wolf's slot 9 holds his party, attacking on day 23.
    # From day 31 a stationary screenshot party waits near the Empire capital, respawning
    # seven days after destruction (user request 2026-10-27).
    bot = 9
    assert capital['bot_attack'] == {'enemy': bot, 'day': 23,
                                     'raids': {'enemies': [37, 41], 'first_day': 31,
                                               'respawn_days': 7, 'position': [10, 28]}}
    assert capital['absent_enemies'] == [9, 37, 41, 42, 44, 46, 47, 49, 50, 51, 52]
    for key in ('enemy_rosters', 'enemy_units', 'enemy_hp'):
        assert ([v for i, v in enumerate(capital[key]) if i not in (bot, 37, 41)]
                == [v for i, v in enumerate(base[key]) if i not in (bot, 37, 41)])
    assert base['enemy_rosters'][bot] == [None, 'g000uu5039', None, None, None, None]
    assert capital['enemy_rosters'][bot] == ['g000uu0019', 'knight', 'knight', None, 'marksman', None]
    assert capital['enemy_rosters'][37] == ['g000uu0020', None, 'squire', 'archer', None, 'apprentice']
    assert capital['enemy_rosters'][41] == [None, 'squire', None, 'g000uu0020', 'acolyte', 'archer']
    assert [(capital['enemy_units'][i], capital['enemy_hp'][i]) for i in (bot, 37, 41)] == [(4, 150), (4, 100), (4, 100)]
    assert capital['enemy_initial_levels'][bot] == [2, 2, 2, 0, 2, 0]
    assert capital['enemy_initial_levels'][37] == [1, 0, 1, 1, 0, 1]
    assert capital['enemy_initial_levels'][41] == [0, 1, 0, 1, 1, 1]
    assert len(capital['enemy_initial_levels']) == len(capital['enemy_rosters'])
    for name in ('default', 'default-blue-dragon', 'default-green-dragon'):
        assert not any(key in load_map(name) for key in ('bot_attack', 'absent_enemies', 'enemy_initial_levels',
                                                         'hero_roster'))
    # User request 2026-10-26: Capital's own warrior party, the Duke and a Possessed on the front
    # edges and two Cultists on the back edges; unit-swap actions follow its slot order.
    assert capital['hero_roster'] == ['duke', None, 'possessed', 'cultist', None, 'cultist']
    assert action_names(capital) == action_names({**base, 'faction': 'legions', 'lord_type': 'warrior',
                                                  'hero_roster': capital['hero_roster']})
    assert len(action_names(capital)) == len(action_names(base))
    saved = tmp_path / 'map.json'
    saved.write_text(json.dumps(capital), encoding='utf-8')
    assert load_map(saved) == capital


def test_only_the_selected_target_finishes_the_campaign(target_game):
    """All other armies and cities are optional, including the green dragon in Capital."""
    env, initial = target_game
    target = env.objective_enemy
    assert jnp.flatnonzero(env.goal_enemies).tolist() == [target]
    assert not jnp.any(env.territory.goal_cities)
    assert not env.script_events.enabled and not env.scenario.enabled and not env.bot.enabled
    won = compiled_method(env, '_campaign_won')
    only_target_left = initial.replace(city_owned=jnp.ones_like(initial.city_owned),
        alive=jnp.zeros_like(initial.alive).at[target].set(True))
    assert not bool(won(only_target_left))
    after, _ = compiled_method(env, 'step')(only_target_left, jnp.int32(REST))
    assert not after.won and not after.done
    only_target_defeated = initial.replace(alive=initial.alive.at[target].set(False))
    assert bool(won(only_target_defeated))
    assert not bool(won(only_target_defeated.replace(lost=jnp.bool_(True))))
