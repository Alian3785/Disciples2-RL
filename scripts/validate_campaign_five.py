"""CUDA readiness of the five requested scenarios, with no training or reference engine.

Original-game numeric provenance is checked separately by audit_campaign_five.cjs.
These checks pin the user's scenario goals and exercise actual public JAX steps.
"""
import argparse
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid import NumberGrid, REST, load_map

MAPS = ('mirror_match', 'siege_train', 'default', 'default-green-dragon', 'current')


def validate(name):
    env = NumberGrid(map_config=load_map(name))
    reset, step, mask, won = map(jax.jit, (env.reset, env.step, env.action_mask, env._campaign_won))
    state, ts = reset(jax.random.PRNGKey(42))
    initial = state
    assert not won(initial)
    if name == 'mirror_match':
        assert env.bot.enabled and env.map_config['campaign_objective'] == 'scripted_bot'
        assert not won(initial.replace(alive=jnp.zeros_like(initial.alive)))
        assert won(initial.replace(bot_hero_victories=jnp.int32(1)))
        assert not won(initial.replace(bot_hero_victories=jnp.int32(1), lost=jnp.bool_(True)))
    elif name == 'siege_train':
        enemy = env.objective_enemy
        advance = jax.jit(env.scenario.advance)
        assert not initial.alive[enemy] and not initial.scenario_spawned[enemy]
        assert not won(initial.replace(alive=jnp.zeros_like(initial.alive)))
        before = advance(initial.replace(day=jnp.int32(27)))
        after = advance(initial.replace(day=jnp.int32(28)))
        assert not before.alive[enemy] and after.alive[enemy] and after.scenario_spawned[enemy]
        assert not won(after)
        assert won(after.replace(alive=after.alive.at[enemy].set(False)))
    elif name == 'default':
        goals = np.flatnonzero(np.asarray(env.territory.goal_cities))
        assert len(goals) == 2
        for index in goals:
            assert not won(initial.replace(city_owned=initial.city_owned.at[index].set(True)))
        assert won(initial.replace(city_owned=initial.city_owned.at[goals].set(True)))
    elif name == 'default-green-dragon':
        enemy = env.objective_enemy
        assert env.map_config['enemy_rosters'][enemy][1] == 'g000uu5023'
        assert not won(initial.replace(alive=jnp.zeros_like(initial.alive).at[enemy].set(True)))
        assert won(initial.replace(alive=initial.alive.at[enemy].set(False)))
        assert not jnp.any(env.territory.goal_cities)
    else:
        assert env.size == 48 and env.num_opponents == 51
        assert not env.map_config.get('optional_enemies')
        units = json.loads((Path(__file__).resolve().parents[1]/'stoix/envs/data/units.json').read_text())
        present = {units[k]['unit_type'] for r in env.map_config['enemy_rosters'] for k in r if k}
        assert len(present) == 52 and present == {r['unit_type'] for r in units.values()}
        for enemy in range(51):
            assert not won(initial.replace(alive=jnp.zeros_like(initial.alive).at[enemy].set(True)))
        assert won(initial.replace(alive=jnp.zeros_like(initial.alive)))
    # Public transitions include a world day (bot/wave processing), movement,
    # action masking and terminal handling, with reproducible random legal actions.
    rng = np.random.default_rng(42)
    for i in range(64):
        legal = np.flatnonzero(np.asarray(mask(state)))
        assert len(legal)
        action = REST if i % 8 == 0 and REST in legal else int(rng.choice(legal))
        state, ts = step(state, jnp.int32(action))
        assert np.isfinite(np.asarray(ts.reward)).all()
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(ts.observation))
        assert np.asarray(state.hp).min() >= 0
        if state.done:
            state, ts = reset(jax.random.PRNGKey(i+43))
    report = dict(map=name, passed=True, public_steps=64, armies=env.num_opponents,
                  actions=env.num_actions, observation_size=env.observation_size)
    jax.clear_caches()
    gc.collect()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == 'gpu'
    reports = []
    for name in MAPS:
        args.output.write_text(json.dumps(dict(passed=False, backend='cuda', checking=name, maps=reports), indent=2)+'\n')
        print(json.dumps(dict(phase='checking', map=name)), flush=True)
        reports.append(validate(name))
        print(json.dumps(reports[-1]), flush=True)
    args.output.write_text(json.dumps(dict(passed=True, backend='cuda', maps=reports), indent=2)+'\n')
