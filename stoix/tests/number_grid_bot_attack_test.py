"""Default Capital's minimal scripted Empire attacks, requested 2026-10-26; not part of core.

Sources: user requests 2026-10-26/27, the user's save SaveGame/6789.sg (turn 21: Edwin, a
level-2 Pegasus Knight with two Knights and a Marksman, 11 destroyed armies in
CMidStackDestroyed), the user's two Empire screenshots for the raids from day 31,
docs/DEFAULT_MAP.md. From 2026-10-27 later parties wait near the Empire capital and
respawn seven days after destruction. A training rule, not the original AI.
Extended CUDA regressions; run only on explicit request through scripts/run_gpu.sh.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, REST, RETREAT, NumberGrid, load_map
from stoix.envs.number_grid_bot_attack import BotAttackRules
from stoix.envs.number_grid_casting import DAMAGE, SUMMON
from stoix.envs.number_grid_combat import HP
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS, campaign_events
from stoix.envs.number_grid_scenarios import ScenarioRules
from stoix.tests.number_grid_capital_training_test import defeat_armies
from stoix.tests.number_grid_fixtures import compiled_method
from stoix.tests.number_grid_target_level_rewards_test import finish, near_enemy

PARTY = ['g000uu0019', 'knight', 'knight', None, 'marksman', None]
RAIDERS = {37: ['g000uu0020', None, 'squire', 'archer', None, 'apprentice'],
           41: [None, 'squire', None, 'g000uu0020', 'acolyte', 'archer']}


@pytest.fixture(scope='module')
def capital():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'
    env = NumberGrid(map_config=load_map('default-capital'))
    return env, compiled_method(env, 'reset')(jax.random.PRNGKey(42))[0]


def attacked(env, state):
    """REST into the attack day on a field tile beside the green dragon."""
    before, _ = near_enemy(env, state.replace(day=jnp.int32(env.bot_attack.day-1)), 30)
    assert compiled_method(env, 'action_mask')(before)[REST]
    return before, compiled_method(env, 'step')(before, jnp.int32(REST))[0]


def test_victims_are_absent_and_the_party_waits_for_its_day(capital):
    env, state = capital
    rules = env.bot_attack
    absent = jnp.array(env.map_config['absent_enemies'])
    assert rules.enabled and (rules.enemy, rules.day) == (9, 23)
    assert int(state.bot_respawn_day) == 31
    assert not jnp.any(state.alive[absent]) and not jnp.any(state.scenario_spawned[absent])
    assert int(jnp.sum(state.alive)) == env.num_opponents-11
    assert int(env.capital_training.defeated(state)) == 0
    ids = env.progression.enemy_ids[rules.enemy]
    assert ids.tolist() == [env.progression.ids[key] if key else 0 for key in PARTY]
    levels = state.enemy_progress[rules.enemy, :, 0]
    assert levels.tolist() == [2, 2, 2, 0, 2, 0]
    assert env.progression.stats(ids, levels)[:, 0].astype(jnp.int32).tolist() == [165, 150, 150, 0, 90, 0]
    for slot, roster in RAIDERS.items():
        assert env.progression.enemy_ids[slot].tolist() == [env.progression.ids[key] if key else 0 for key in roster]
        assert state.enemy_progress[slot, :, 0].tolist() == [int(key is not None) for key in roster]
    assert rules.observation(state).tolist() == pytest.approx([22/23, 30/31])
    assert compiled_method(env, 'observation')(state).shape == (env.observation_size,)
    quiet, _ = compiled_method(env, 'step')(state.replace(day=jnp.int32(rules.day-2)), jnp.int32(REST))
    assert int(quiet.day) == rules.day-1 and not quiet.alive[rules.enemy] and not quiet.in_battle


def test_rest_into_day_23_starts_the_battle_on_the_spot_without_retreat(capital):
    env, initial = capital
    rules = env.bot_attack
    before, battle = attacked(env, initial)
    assert int(battle.day) == rules.day
    assert battle.in_battle and int(battle.enemy) == rules.enemy
    assert battle.alive[rules.enemy] and battle.scenario_spawned[rules.enemy]
    assert not battle.pending_encounters[rules.enemy]
    assert battle.enemy_positions[rules.enemy].tolist() == before.position.tolist()
    assert battle.position.tolist() == before.position.tolist()
    assert int(battle.battle_fort_defender) == 0 and int(battle.battle_hero_fort_armor) == 0
    assert not bool(compiled_method(env, 'action_mask')(battle)[RETREAT])
    assert not bool(env._can_retreat(battle, jnp.int32(0)))
    assert rules.observation(battle).tolist() == pytest.approx([0., 8/31])


def test_attack_inside_the_capital_keeps_the_fort_rules(capital):
    env, initial = capital
    home, _ = compiled_method(env, 'step')(initial.replace(day=jnp.int32(env.bot_attack.day-1)), jnp.int32(REST))
    assert home.in_battle and int(home.enemy) == env.bot_attack.enemy
    assert int(home.battle_hero_fort_armor) == 50 and int(home.battle_fort_defender) == 1


def test_victory_is_an_ordinary_battle_and_feeds_the_chart_and_threshold(capital):
    env, initial = capital
    rules = env.bot_attack
    _, battle = attacked(env, defeat_armies(env, initial, 50))
    assert int(env.capital_training.defeated(battle)) == 50
    assert not bool(rules.raid_battle(battle))
    won, reward = finish(env, battle)
    assert not won.in_battle and not won.done and not won.lost and not won.alive[rules.enemy]
    assert float(reward) >= 1.  # Edwin pays the ordinary +1 army victory, plus any level-up rewards
    assert int(env.capital_training.defeated(won)) == 51
    field = CAMPAIGN_FIELDS.index('capital_bot_heroes_defeated')
    events = campaign_events(env, battle, battle, won, jnp.int32(DEFEND), jnp.bool_(True), jnp.zeros(4, jnp.int32))
    assert float(events[field]) == 1.
    lost, _ = finish(env, battle, 'defeat')
    events = campaign_events(env, battle, battle, lost, jnp.int32(DEFEND), jnp.bool_(True), jnp.zeros(4, jnp.int32))
    assert float(events[field]) == 0.


def test_defeat_ends_the_episode_as_a_lost_battle(capital):
    env, initial = capital
    _, battle = attacked(env, initial)
    lost, reward = finish(env, battle, 'defeat')
    assert lost.done and lost.lost and not lost.won
    assert float(reward) == pytest.approx(-1.)


def test_stationary_party_from_day_31_draws_either_screenshot_party(capital):
    env, initial = capital
    rules = env.bot_attack
    assert (rules.raid_enemies, rules.first_day, rules.respawn_days) == ([37, 41], 31, 7)
    assert rules.position.tolist() == [10, 28]
    assert not bool(env.territory.blocked(rules.position))
    assert int(jnp.max(jnp.abs(rules.position-env.territory.enemy_capital_positions[0]))) == 1
    spawn = jax.jit(rules.spawn)
    chosen = set()
    for seed in range(16):
        raided = spawn(initial.replace(day=jnp.int32(31)), jnp.bool_(True), jax.random.PRNGKey(seed))
        present = [i for i in rules.raid_enemies if bool(raided.alive[i])]
        assert len(present) == 1 and not bool(jnp.any(raided.pending_encounters))
        assert raided.enemy_positions[present[0]].tolist() == [10, 28]
        assert int(raided.bot_respawn_day) == 0
        assert float(rules.observation(raided)[1]) == 0.
        chosen.add(present[0])
    assert chosen == {37, 41}
    for day in (23, 30):
        after = spawn(initial.replace(day=jnp.int32(day)), jnp.bool_(True), jax.random.PRNGKey(0))
        assert not bool(jnp.any(after.alive[rules.raids]))
    # An existing party is never replaced or refreshed by the former periodic schedule.
    worn = raided.replace(enemy_wounds=raided.enemy_wounds.at[present[0]].set(50))
    for day in (32, 39, 47, 95):
        after = spawn(worn.replace(day=jnp.int32(day)), jnp.bool_(True), jax.random.PRNGKey(day))
        assert after.alive.tolist() == worn.alive.tolist()
        assert after.enemy_positions.tolist() == worn.enemy_positions.tolist()
        assert after.enemy_wounds.tolist() == worn.enemy_wounds.tolist()


def test_stationary_party_waits_for_a_free_tile_and_never_forces_combat(capital):
    env, initial = capital
    rules = env.bot_attack
    step = compiled_method(env, 'step')
    # REST beside the spawn point does not start a battle.
    before = initial.replace(day=jnp.int32(30), position=jnp.array([9, 28], jnp.int32))
    waiting, _ = step(before, jnp.int32(REST))
    assert not waiting.in_battle and not waiting.done and not jnp.any(waiting.pending_encounters)
    enemy = int(jnp.argmax(waiting.alive & rules.raid_mask))
    assert bool(waiting.alive[enemy]) and enemy in rules.raid_enemies
    assert int(env.territory.enemy_at(waiting, rules.position)) == enemy
    assert int(env.territory.enemy_at(waiting, initial.enemy_positions[enemy])) == -1
    later, _ = step(waiting.replace(day=jnp.int32(38)), jnp.int32(REST))
    assert not later.in_battle and later.alive.tolist() == waiting.alive.tolist()
    # Occupying the spawn point cannot cause a forced fight; leaving releases the arrival.
    blocked, _ = step(before.replace(position=rules.position), jnp.int32(REST))
    assert not jnp.any(blocked.alive[rules.raids]) and not blocked.in_battle
    assert int(blocked.bot_respawn_day) == 31
    released, _ = step(blocked, jnp.int32(0))  # move north to [9, 28]
    assert jnp.any(released.alive[rules.raids]) and not released.in_battle
    assert not jnp.any(released.pending_encounters)
    # Nor may reinforcements overwrite another living army on their tile.
    occupied = before.replace(day=jnp.int32(31), enemy_positions=before.enemy_positions.at[30].set(rules.position))
    held = jax.jit(rules.spawn)(occupied, jnp.bool_(True), jax.random.PRNGKey(0))
    assert not jnp.any(held.alive[rules.raids])


def test_stationary_party_allows_retreat_and_respawns_seven_days_after_victory(capital):
    env, initial = capital
    rules = env.bot_attack
    waiting, _ = compiled_method(env, 'step')(initial.replace(day=jnp.int32(30)), jnp.int32(REST))
    enemy = int(jnp.argmax(waiting.alive & rules.raid_mask))
    before = waiting.replace(day=jnp.int32(35), position=jnp.array([9, 28], jnp.int32),
                             movement_points=jnp.int32(100))
    action = jnp.int32(4)  # south to the stationary party, from a free field tile
    assert compiled_method(env, 'action_mask')(before)[action]
    battle, _ = compiled_method(env, 'step')(before, action)
    enemy = int(battle.enemy)
    assert battle.in_battle and enemy in rules.raid_enemies
    assert bool(compiled_method(env, 'action_mask')(battle)[RETREAT])
    assert bool(rules.raid_battle(battle))
    escaped, _ = finish(env, battle, 'retreat')
    assert escaped.alive[enemy] and not escaped.in_battle and not escaped.done
    assert int(escaped.bot_respawn_day) == 0
    assert escaped.enemy_positions[enemy].tolist() == [10, 28]
    calm, _ = compiled_method(env, 'step')(escaped, jnp.int32(REST))
    assert not calm.in_battle and calm.alive[enemy] and int(calm.bot_respawn_day) == 0
    # Raids pay no +1 and no +2 per promotion (user request 2026-10-26), but XP still promotes.
    duke = 0
    threshold = env.unit_experience(battle)[duke, 2]
    boosted = battle.replace(unit_xp=battle.unit_xp.at[duke].set(threshold-1),
        native_xp=battle.native_xp.at[duke].set(threshold-1),
        battle_earned_xp=battle.battle_earned_xp.at[duke].set(100))
    won, reward = finish(env, boosted)
    assert won.unit_levels[duke] > battle.unit_levels[duke]
    assert not won.alive[enemy] and not won.done and float(reward) <= 0.
    assert int(won.bot_hero_victories) == 1 and int(env.capital_training.defeated(won)) == 1
    assert int(won.bot_respawn_day) == 42  # killed on day 35, not a fixed 31 + 8*k schedule
    spawn = jax.jit(rules.spawn)
    for day in range(35, 42):
        early = spawn(won.replace(day=jnp.int32(day)), jnp.bool_(True), jax.random.PRNGKey(day))
        assert not jnp.any(early.alive[rules.raids])
        assert float(rules.observation(early)[1]) == pytest.approx((42-day)/31)
    # The next raid brings a party at full strength, even into a slot used before.
    worn = won.replace(day=jnp.int32(42), enemy_wounds=won.enemy_wounds.at[rules.raids].set(50),
                       enemy_progress=won.enemy_progress.at[rules.raids].set(0),
                       enemy_spell_effects=won.enemy_spell_effects.at[rules.raids].set(jnp.uint32(1)))
    again = spawn(worn, jnp.bool_(True), jax.random.PRNGKey(5))
    back = [i for i in rules.raid_enemies if bool(again.alive[i])]
    assert len(back) == 1 and not jnp.any(again.enemy_wounds[back[0]])
    assert again.enemy_progress[back[0]].tolist() == env.progression.initial_enemies[back[0]].tolist()
    assert not jnp.any(again.enemy_spell_effects[back[0]]) and not jnp.any(again.pending_encounters)
    assert int(again.bot_respawn_day) == 0
    assert int(env.capital_training.defeated(again)) == 1


@pytest.mark.parametrize('kind', [DAMAGE, SUMMON])
def test_magic_and_summon_destruction_start_the_same_respawn_clock(capital, kind):
    env, initial = capital
    rules = env.bot_attack
    born = jax.jit(rules.spawn)(initial.replace(day=jnp.int32(31)), jnp.bool_(True), jax.random.PRNGKey(0))
    enemy = int(jnp.argmax(born.alive & rules.raid_mask))
    spell = next(i for i in range(env.casting.book_count) if int(env.casting.kinds[i]) == kind)
    ids = env.progression.enemy_ids[enemy]
    maximum = env.progression.stats(ids, born.enemy_progress[enemy, :, 0])[:, HP].astype(jnp.int32)
    ready = born.replace(day=jnp.int32(50), position=jnp.array([9, 28], jnp.int32),
        learned_spells=jnp.uint32(1 << spell), mana=jnp.full(5, 5000, jnp.int32),
        enemy_wounds=born.enemy_wounds.at[enemy].set(jnp.where(ids != 0, maximum-1, 0)))
    action = jnp.int32(env.casting.start+spell)
    assert compiled_method(env, 'action_mask')(ready)[action]
    if kind == DAMAGE:
        dead, _ = compiled_method(env, 'step')(ready, action)
        assert int(dead.last_spell_target) == enemy
    else:
        battle, _ = compiled_method(env, '_world_step')(
            ready, action, jax.random.PRNGKey(20), jnp.full(env.random_size, .5))
        assert battle.summon_battle and int(battle.enemy) == enemy
        dead, _ = finish(env, battle)
    assert not dead.in_battle and not dead.alive[enemy] and not dead.done
    assert not jnp.any(dead.alive[rules.raids]) and int(dead.bot_respawn_day) == 57
    assert int(dead.bot_hero_victories) == 0  # only hero-army victories count in this metric


def test_destruction_clock_does_not_restart_on_later_actions_or_edwin_death(capital):
    env, initial = capital
    rules = env.bot_attack
    born = jax.jit(rules.spawn)(initial.replace(day=jnp.int32(31)), jnp.bool_(True), jax.random.PRNGKey(0))
    dead = born.replace(day=jnp.int32(50), alive=born.alive.at[rules.raids].set(False))
    schedule = jax.jit(rules.schedule_respawn)
    scheduled = schedule(born, dead)
    assert int(scheduled.bot_respawn_day) == 57
    later = schedule(scheduled, scheduled.replace(day=jnp.int32(51)))
    assert int(later.bot_respawn_day) == 57
    edwin = initial.replace(alive=initial.alive.at[rules.enemy].set(True))
    after_edwin = schedule(edwin, edwin.replace(alive=edwin.alive.at[rules.enemy].set(False)))
    assert int(after_edwin.bot_respawn_day) == 31
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(10))
    assert int(reset.bot_respawn_day) == 31 and not jnp.any(reset.alive[rules.raids])


def test_invalid_bot_attack_and_absent_armies_are_rejected():
    base = load_map('default-capital')
    for spec in ({'enemy': 9, 'day': 1}, {'enemy': 9, 'day': 23.}, {'enemy': 75, 'day': 23},
                 {'enemy': 74, 'day': 23}, {'enemy': 30, 'day': 23}):
        with pytest.raises(ValueError, match='bot_attack'):
            BotAttackRules({**base, 'bot_attack': spec})
    for change in ({'enemies': [37, 37]}, {'enemies': [9]}, {'enemies': [30]}, {'enemies': '37'},
                   {'first_day': 1}, {'respawn_days': 0}, {'respawn_days': 7.},
                   {'position': [0, 28]}, {'position': [11, 28]}, {'position': [10]}, {'position': None}):
        spec = {**base['bot_attack'], 'raids': {**base['bot_attack']['raids'], **change}}
        with pytest.raises(ValueError, match='bot_attack'):
            BotAttackRules({**base, 'bot_attack': spec})
    for absent in ([9, 9], [75], 'x'):
        with pytest.raises(ValueError, match='absent_enemies'):
            ScenarioRules({**base, 'absent_enemies': absent}, None)
