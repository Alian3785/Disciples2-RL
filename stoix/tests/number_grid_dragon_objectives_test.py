"""Extended CUDA cases; sources and reward differences: docs/DEFAULT_MAP.md.

Run only on explicit request, through scripts/run_gpu.sh. These two full Default
variants deliberately stay outside core to preserve its compilation budget.
"""
import json

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, REST, SHOOT, WAIT, NumberGrid, action_names, load_map
from stoix.envs.number_grid_casting import CAST_REWARD, SPELL_KILL, SUMMON
from stoix.envs.number_grid_combat import ACCURACY, AREA, DAMAGE, HP, INITIATIVE
from stoix.tests.number_grid_fixtures import compiled_method

DRAGON = 30  # maps/default.py enemy_id=31, at reference (30, 46).


@pytest.fixture(scope='module', params=['default-green-dragon', 'default-blue-dragon'])
def dragon_game(request):
    config = load_map(request.param)
    env = NumberGrid(map_config=config)
    initial, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, initial, compiled_method(env, 'step'), compiled_method(env, 'action_mask')


def test_default_variants_keep_layout_and_original_dragon_profiles(dragon_game, tmp_path):
    """Reference roster replacement; installed Gunits/Gattacks, not invented stats."""
    env, initial, _, _ = dragon_game
    config, base = env.map_config, load_map('default')
    blue = config['campaign_objective'] == 'blue_dragon'
    for key in ('opponent_positions', 'obstacles', 'terrain', 'chests', 'mines', 'ruins'):
        assert config[key] == base[key]
    assert config['opponent_positions'][DRAGON] == [47, 31]
    assert action_names(config) == action_names(base)
    assert env.observation_size == 2661+(env.num_opponents+1 if blue else 0)
    assert config.get('army_victory_level_reward', 0.) == (5. if blue else 0.)
    assert config.get('army_victory_level_objective_only', False) == blue
    uid = env.progression.enemy_ids[DRAGON, 1]
    row = env.progression.rows[int(uid)]
    assert row['game_id'] == ('g000uu5021' if blue else 'g000uu5023')
    stats = env.progression.stats(uid, initial.enemy_progress[DRAGON, 1, 0])
    assert stats[jnp.array([HP, DAMAGE, ACCURACY, INITIATIVE])].tolist() == [
        700 if blue else 600, 100 if blue else 60, 75, 40]
    assert row['size'] == 2 and row['role'] == 'area'
    assert row['attack_type'] == ('water' if blue else 'fire')
    assert row['immunities'] == ['water' if blue else 'fire']
    assert int(env.progression.traits[uid, 0]) == AREA
    assert jnp.flatnonzero(env.goal_enemies).tolist() == [DRAGON]
    assert not jnp.any(env.territory.goal_cities)
    # Variant loading does not mutate the original Default; saved runs are standalone.
    assert sum(c['objective'] for c in base['cities']) == 2
    assert base['enemy_rosters'][DRAGON][1] == 'g000uu5023'
    saved = tmp_path / 'map.json'
    saved.write_text(json.dumps(config), encoding='utf-8')
    assert load_map(saved) == config


def test_cities_and_other_defeated_stacks_do_not_finish_the_hunt(dragon_game):
    env, initial, step, mask = dragon_game
    # Even complete city ownership and the defeat of every other stack are insufficient.
    state = initial.replace(city_owned=jnp.ones_like(initial.city_owned),
                            alive=jnp.zeros_like(initial.alive).at[DRAGON].set(True))
    assert mask(state)[REST]
    after, _ = step(state, jnp.int32(REST))
    assert not after.won and not after.done and after.alive[DRAGON]
    # Capturing the final former objective city must still work, without victory.
    city = next(i for i, c in enumerate(env.map_config['cities']) if c['position'] == [26, 41])
    target = env.map_config['cities'][city]['position']
    blocked = env.territory.blocked_grid.reshape(env.size, env.size)
    direction, (dr, dc) = next((i, d) for i, d in enumerate(env.directions.tolist())
                              if not blocked[target[0]-d[0], target[1]-d[1]])
    near = state.replace(position=jnp.array([target[0]-dr, target[1]-dc], jnp.int32),
                         city_owned=state.city_owned.at[city].set(False))
    assert mask(near)[direction]
    after, _ = step(near, jnp.int32(direction))
    assert after.city_owned[city] and not after.won and not after.done


def test_target_regenerates_fully_each_day(dragon_game):
    _, initial, step, _ = dragon_game
    wounds = initial.enemy_wounds.at[DRAGON, 1].set(500)
    rested, _ = step(initial.replace(enemy_wounds=wounds), jnp.int32(REST))
    assert rested.enemy_wounds[DRAGON, 1] == 0 and rested.alive[DRAGON]


def test_map_spell_kill_finishes_without_cities_and_obeys_bonus_rule(dragon_game):
    """Reference test_blue_dragon_objective_reward.py: map_spell is ineligible for blue."""
    env, initial, step, mask = dragon_game
    spell = next(i for i, row in enumerate(env.casting.rows) if row['id'] == 'g000ss0067')  # tower Plague
    maximum = int(env.casting.enemy_max_hp(initial)[DRAGON, 1])
    state = initial.replace(position=jnp.array([46, 31], jnp.int32),
                            enemy_wounds=initial.enemy_wounds.at[DRAGON, 1].set(maximum-1),
                            learned_spells=jnp.uint32(1 << spell), mana=jnp.full(5, 5000, jnp.int32))
    action = jnp.int32(env.casting.start+spell)
    assert mask(state)[action]
    after, ts = step(state, action)
    assert after.last_spell_target == DRAGON and after.last_event == SPELL_KILL
    assert after.won and after.done and not after.alive[DRAGON]
    assert not jnp.any(after.army_victory_rewarded)
    assert jnp.sum(after.alive) == len(initial.alive)-1 and not jnp.any(after.city_owned)
    bonus = 0 if env.objective_reward_hero_only else 3
    penalty = .05*float(env.recruitment.composition(after)[2])
    assert float(ts.reward) == pytest.approx(CAST_REWARD+bonus-env.step_cost-penalty)
    assert not jnp.any(mask(after))
    again, ts = step(after, jnp.int32(REST))
    assert again.done and not again.alive[DRAGON] and float(ts.reward) == 0


@pytest.mark.parametrize('summon', [False, True])
def test_hero_and_summon_victories_finish_with_the_correct_bonus(dragon_game, summon):
    """Hero gets +3 in both modes; a separate summon gets it only against green."""
    env, initial, _, _ = dragon_game
    prepared = initial.replace(enemy=jnp.int32(DRAGON), position=jnp.array([46, 31], jnp.int32))
    actor = 1
    if summon:
        spell = next(i for i in range(env.casting.book_count) if int(env.casting.kinds[i]) == SUMMON)
        prepared = compiled_method(env.casting, '_summon')(
            prepared, jnp.bool_(True), jnp.int32(spell), jnp.int32(DRAGON))
        actor = int(env.casting.summon_slots[spell])
    battle = compiled_method(env, '_begin_battle')(prepared)
    battle = battle.replace(actor=jnp.int32(actor), hp=battle.hp.at[7].set(1))
    advance = compiled_method(env, '_battle_step')
    draws = jnp.zeros(env.random_size, jnp.float32)  # deterministic successful attack
    after, reward = advance(battle, jnp.int32(SHOOT+1), jax.random.PRNGKey(7), draws)
    for key in range(8, 14):  # post-victory healing turns, only while a healer is alive
        if not after.in_battle:
            break
        after, extra = advance(after, jnp.int32(WAIT), jax.random.PRNGKey(key), draws)
        reward += extra
    assert after.won and after.done and not after.in_battle and not after.alive[DRAGON]
    assert not jnp.any(after.city_owned) and jnp.sum(after.alive) == len(initial.alive)-1
    # Either dragon gives enough XP for the starting hero's first level (+2).
    # Two Possessed and the Cultist lack their buildings and each lose XP (-.05).
    assert float(reward) == pytest.approx(5.85 if not summon else 0 if env.objective_reward_hero_only else 3)
    if not summon:
        assert after.unit_levels[1] == 2 and after.last_promoted == 2
        assert after.unit_xp[jnp.array([0, 2, 4])].tolist() == [94, 94, 74]  # one below 95 / 75
    if summon:
        assert after.hp[:6].tolist() == initial.hp[:6].tolist()


def test_losing_to_the_target_is_defeat(dragon_game):
    env, initial, _, _ = dragon_game
    battle = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(DRAGON)))
    battle = battle.replace(hp=battle.hp.at[:6].set(0), actor=jnp.int32(1))
    after, _ = compiled_method(env, '_battle_step')(
        battle, jnp.int32(DEFEND), jax.random.PRNGKey(9), jnp.zeros(env.random_size, jnp.float32))
    assert after.lost and after.done and not after.won and after.alive[DRAGON]


@pytest.mark.parametrize('target', [-1, 75, True, 30.0])
def test_invalid_objective_enemy_is_rejected(target):
    with pytest.raises(ValueError, match='objective_enemy'):
        NumberGrid(map_config={**load_map('default'), 'objective_enemy': target})
