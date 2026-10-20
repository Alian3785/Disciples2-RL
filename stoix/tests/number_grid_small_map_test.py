"""Small source: Big_map/maps/small.py and maps/small/layout.json (seed=0).

Geometry adaptations and sources: docs/SMALL_MAP.md. Extended CUDA tests;
run only on explicit request through scripts/run_gpu.sh, outside the core budget.
"""
import json

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, REST, action_names, load_map
from stoix.envs.number_grid_sites import TRAINED
from stoix.envs.number_grid_territory import growth_ranks
from stoix.tests.number_grid_fixtures import compiled_method

VARIANTS = ('small', 'small-green-dragon', 'small-blue-dragon')


@pytest.fixture(scope='module', params=VARIANTS)
def small_game(request):
    """Share a reproducible Small environment for each city or dragon objective variant."""
    env = NumberGrid(map_config=load_map(request.param))
    initial, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, initial


def test_small_reference_armies_and_independent_variants(tmp_path):
    """Pin source army IDs and verify objective variants preserve geometry and independent data."""
    base = load_map('small')
    expected = sorted((set(range(1, 31, 2)) | set(range(37, 66, 2))
                       | {31, 34, 35, 68, 69, 70, 74, 75}) - {55})
    assert base['reference_enemy_ids'] == expected and len(expected) == 37
    assert base['size'] == 26 and base['agent_position'] == [14, 3]
    assert type(base['starting_gold']) is int and base['starting_gold'] == 0
    assert len(base['chests']) == 9 and len(base['ruins']) == 2 and len(base['mines']) == 8
    assert [c['name'] for c in base['cities']] == ['Порт Полонис', 'Соругирилла']
    assert [base['city_defender_roles'][str(expected.index(i))] for i in (34, 35, 68, 69, 75)] == [
        'guard', 'guard', 'garrison', 'garrison', 'capital']
    assert base['enemy_full_regeneration'] == [expected.index(31)]
    assert base['opponent_positions'][expected.index(31)] == [24, 16]
    for name in VARIANTS:
        variant = load_map(name)
        target = expected.index(31)
        for key in ('opponent_positions', 'obstacles', 'terrain', 'chests', 'mines', 'ruins'):
            assert variant[key] == base[key]
        assert action_names(variant) == action_names(base)
        assert variant.get('objective_enemy') == (None if name == 'small' else target)
        assert all(c['objective'] == (name == 'small') for c in variant['cities'])
        blue = name == 'small-blue-dragon'
        assert variant['enemy_rosters'][target][1] == ('g000uu5021' if blue else 'g000uu5023')
        assert variant.get('army_victory_level_reward', 0.) == (5. if blue else 0.)
        assert variant.get('army_victory_level_objective_only', False) == blue
        saved = tmp_path/(name+'.json')
        saved.write_text(json.dumps(variant), encoding='utf-8')
        assert load_map(saved) == variant
    assert load_map('small') == base


def test_small_geometry_and_observations(small_game):
    """Check collision-free objects, reachable services, finite features, and variant objectives."""
    env, initial = small_game
    config = env.map_config
    assert env.num_actions == len(action_names(config))
    assert env.observation(initial).shape == (env.observation_size,)
    assert jnp.all(jnp.isfinite(env.observation(initial)))
    assert not initial.done and not initial.in_battle
    # Unintended object collisions are forbidden, including chest/mine and chest/army.
    loot = [tuple(c['position']) for c in config['chests']+config['mines']]
    assert len(set(loot)) == len(loot)
    assert not set(loot) & set(map(tuple, config['opponent_positions']))
    ranks = growth_ranks(env.size, (tuple(config['agent_position']),), env.territory.obstacle_cells)[0]
    for site in env.site_rules.sites+env.ruins.ruins:
        assert any(ranks[r*env.size+c] < env.size**2 for r, c in site['interaction_tiles'])
    target = config['reference_enemy_ids'].index(31)
    if config['campaign_objective'] == 'cities':
        assert env.territory.goal_cities.tolist() == [True, True]
        # Killing every army alone never captures a city.
        cleared = initial.replace(alive=jnp.zeros_like(initial.alive))
        after, _ = compiled_method(env, 'step')(cleared, jnp.int32(REST))
        assert not after.won and not after.done
        captured = initial.replace(city_owned=jnp.ones_like(initial.city_owned))
        after, _ = compiled_method(env, 'step')(captured, jnp.int32(REST))
        assert after.won and after.done
    else:
        assert jnp.flatnonzero(env.goal_enemies).tolist() == [target]
        assert not jnp.any(env.territory.goal_cities)
        assert not env._goals_left(initial.alive.at[target].set(False), initial.scenario_spawned)
        assert env._goals_left(jnp.zeros_like(initial.alive).at[target].set(True), initial.scenario_spawned)


def test_small_trainer_mask_and_step(small_game):
    """Require trainer proximity and payment, with XP stopping short of an automatic level-up."""
    env, initial = small_game
    mask, step = compiled_method(env, 'action_mask'), compiled_method(env, 'step')
    trainer = next(site for site in env.site_rules.sites if site['kind'] == 'trainer')
    action = env.site_rules.train_start  # the first living Squire
    rich = initial.replace(gold=jnp.int32(9999))
    assert not mask(rich)[action]
    unchanged, _ = step(rich, jnp.int32(action))
    assert unchanged.gold == rich.gold and jnp.array_equal(unchanged.unit_xp, rich.unit_xp)
    there = rich.replace(position=jnp.array(trainer['interaction_tiles'][0], jnp.int32))
    assert mask(there)[action]
    trained, _ = step(there, jnp.int32(action))
    assert trained.last_event == TRAINED and trained.gold < there.gold
    assert trained.unit_xp[0] > there.unit_xp[0]
    assert trained.unit_levels[0] == there.unit_levels[0]  # trainer stops before a level-up


def test_small_dragon_has_a_legal_entry(small_game):
    """Find an in-bounds approach and enter the dragon battle through a legal public action."""
    env, initial = small_game
    target = env.map_config['reference_enemy_ids'].index(31)
    position = env.map_config['opponent_positions'][target]
    blocked = env.territory.blocked_grid.reshape(env.size, env.size)
    direction, offset = next((i, d) for i, d in enumerate(env.directions.tolist())
        if 0 < position[0]-d[0] < env.size-1 and 0 < position[1]-d[1] < env.size-1
        and not bool(blocked[position[0]-d[0], position[1]-d[1]]))
    near = initial.replace(position=jnp.array([position[0]-offset[0], position[1]-offset[1]], jnp.int32))
    assert compiled_method(env, 'action_mask')(near)[direction]
    battle, _ = compiled_method(env, 'step')(near, jnp.int32(direction))
    assert battle.in_battle and battle.enemy == target and not battle.won
