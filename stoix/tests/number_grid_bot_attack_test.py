"""Default Capital's minimal scripted Empire attacks, requested 2026-10-26; not part of core.

Sources: user requests 2026-10-26, the user's save SaveGame/6789.sg (turn 21: Edwin, a
level-2 Pegasus Knight with two Knights and a Marksman, 11 destroyed armies in
CMidStackDestroyed), the user's two Empire screenshots for the raids from day 31,
docs/DEFAULT_MAP.md. A training rule, not the original AI.
Extended CUDA regressions; run only on explicit request through scripts/run_gpu.sh.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, REST, RETREAT, NumberGrid, load_map
from stoix.envs.number_grid_bot_attack import BotAttackRules
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


def test_raids_from_day_31_every_8_days_draw_either_screenshot_party(capital):
    env, initial = capital
    rules = env.bot_attack
    assert (rules.raid_enemies, rules.first_day, rules.interval) == ([37, 41], 31, 8)
    spawn = jax.jit(rules.spawn)
    chosen = set()
    for seed in range(16):
        raided = spawn(initial.replace(day=jnp.int32(31)), jnp.bool_(True), jax.random.PRNGKey(seed))
        present = [i for i in rules.raid_enemies if bool(raided.alive[i])]
        assert len(present) == 1 and bool(raided.pending_encounters[present[0]])
        assert raided.enemy_positions[present[0]].tolist() == initial.position.tolist()
        chosen.add(present[0])
    assert chosen == {37, 41}
    for day, due in ((23, False), (30, False), (32, False), (38, False), (39, True), (47, True), (95, True)):
        after = spawn(initial.replace(day=jnp.int32(day)), jnp.bool_(True), jax.random.PRNGKey(0))
        assert bool(jnp.any(after.alive[rules.raids])) == due
    resting_only = spawn(initial.replace(day=jnp.int32(39)), jnp.bool_(False), jax.random.PRNGKey(0))
    assert not jnp.any(resting_only.alive[rules.raids])
    for day, left in ((32, 7), (38, 1), (39, 0)):
        assert float(rules.observation(initial.replace(day=jnp.int32(day)))[1]) == pytest.approx(left/31)


def test_raid_battles_are_fresh_without_retreat_and_each_victory_counts(capital):
    env, initial = capital
    rules = env.bot_attack
    before, _ = near_enemy(env, initial.replace(day=jnp.int32(rules.first_day-1)), 30)
    battle, _ = compiled_method(env, 'step')(before, jnp.int32(REST))
    enemy = int(battle.enemy)
    assert battle.in_battle and enemy in rules.raid_enemies
    assert not bool(compiled_method(env, 'action_mask')(battle)[RETREAT])
    assert bool(rules.raid_battle(battle))
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
    # The next raid brings a party at full strength, even into a slot used before.
    worn = won.replace(day=jnp.int32(rules.first_day+rules.interval),
                       enemy_wounds=won.enemy_wounds.at[rules.raids].set(50))
    again = jax.jit(rules.spawn)(worn, jnp.bool_(True), jax.random.PRNGKey(5))
    back = [i for i in rules.raid_enemies if bool(again.alive[i])]
    assert len(back) == 1 and not jnp.any(again.enemy_wounds[back[0]])
    assert again.enemy_progress[back[0]].tolist() == env.progression.initial_enemies[back[0]].tolist()
    assert int(env.capital_training.defeated(again)) == 1


def test_invalid_bot_attack_and_absent_armies_are_rejected():
    base = load_map('default-capital')
    for spec in ({'enemy': 9, 'day': 1}, {'enemy': 9, 'day': 23.}, {'enemy': 75, 'day': 23},
                 {'enemy': 74, 'day': 23}, {'enemy': 30, 'day': 23},
                 {'enemy': 9, 'day': 23, 'raids': {'enemies': [37, 37], 'first_day': 31, 'interval': 8}},
                 {'enemy': 9, 'day': 23, 'raids': {'enemies': [37], 'first_day': 23, 'interval': 8}},
                 {'enemy': 9, 'day': 23, 'raids': {'enemies': [37], 'first_day': 31, 'interval': 0}},
                 {'enemy': 9, 'day': 23, 'raids': {'enemies': [9], 'first_day': 31, 'interval': 8}},
                 {'enemy': 9, 'day': 23, 'raids': {'enemies': [30], 'first_day': 31, 'interval': 8}}):
        with pytest.raises(ValueError, match='bot_attack'):
            BotAttackRules({**base, 'bot_attack': spec})
    for absent in ([9, 9], [75], 'x'):
        with pytest.raises(ValueError, match='absent_enemies'):
            ScenarioRules({**base, 'absent_enemies': absent}, None)
