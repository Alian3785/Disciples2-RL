"""S101 source fidelity and reach-position objective; extended, CUDA-only.

Run only on an explicit user request, through scripts/run_gpu.sh.
Sources and intentional adaptations: docs/BETHREZENS_RETREAT.md.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, NumberGrid, load_map
from stoix.envs import number_grid_reference_rewards as rewards
from stoix.tests.number_grid_fixtures import compiled_method

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module', autouse=True)
def cuda_only():
    assert jax.default_backend() == 'gpu', 'Run this suite through scripts/run_gpu.sh on CUDA'


@pytest.fixture(scope='module')
def retreat():
    env = NumberGrid(map_config={**load_map('bethrezens_retreat'), 'scenario_events_enabled': False})
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(101))
    return env, state


def test_native_armies_and_unique_large_fighters():
    from scripts.build_bethrezens_retreat import formation
    snapshot = json.loads((ROOT/'docs/maps/bethrezens_retreat/layout.json').read_text(encoding='utf-8'))
    config = load_map('bethrezens_retreat')
    armies = [a for a in snapshot['armies'] if a['owner'] != 'S101PL0001']
    assert len(armies) == 105 and len(config['enemy_rosters']) == 113
    assert config['reference_enemy_ids'][:105] == [a['source_id'] for a in armies]
    for army, roster, levels in zip(armies, config['enemy_rosters'], config['enemy_initial_levels']):
        assert formation(army) == (roster, levels)
        assert sum(k is not None for k in roster) == len(army['units'])
    hero = next(a for a in snapshot['armies'] if a['source_id'] == 'S101KC0013')
    assert formation(hero)[0] == ['reference_g000uu0057', None, 'berserker', None, 'uter_betrezen', None]
    assert 'S101KC0001' not in config['reference_enemy_ids']
    assert len(config['cities']) == 8 and len(config['ruins']) == 8
    assert len(config['chests']) == 35 and len(config['mines']) == 15
    assert config['scheduled_enemies'] == []


def test_initial_party_lord_and_separate_capital(retreat):
    env, state = retreat
    assert state.position.tolist() == [22, 36]
    assert list(env.capital.position) == [18, 57]
    assert env.territory.capital == (18, 57)
    rows = [env.progression.rows[int(i)] for i in state.unit_ids[:6]]
    assert [i for i, row in enumerate(rows) if row.get('hero')] == [4]
    assert rows[4]['name'] == 'Бетрезен' and rows[4]['exp_required'] == 5750
    assert state.hp[:6].tolist() == [170, 0, 170, 0, 300, 0]
    assert int(state.gold) == 300 and int(state.buildings) == 0
    assert env.map_config['lord_type'] == 'warrior'
    # Original unit regen 5% + warrior lord 15%, away from owned land/fort.
    assert int(env.territory.regeneration_percent(state, 0)[4]) == 20
    assert env.recruitment.composition(state)[:3].tolist() == [3, 3, 0]
    assert not env.scenario.enabled and not env.bot.enabled
    assert not env.territory.objective_cities
    assert int(jnp.sum(env.territory.capital_guards)) == 2


def test_exit_centre_requires_living_genuine_bethrezen(retreat):
    env, initial = retreat
    won = compiled_method(env, '_campaign_won')
    state = initial.replace(position=jnp.array([2, 30], jnp.int32))
    assert bool(won(state))
    assert env.objective_radius == 0
    for invalid in (
        state.replace(position=jnp.array([3, 27], jnp.int32)),  # old winning replay's premature exit
        state.replace(position=jnp.array([5, 33], jnp.int32)),  # native zone edge is no longer sufficient
        state.replace(position=jnp.array([3, 30], jnp.int32)),
        state.replace(hp=state.hp.at[4].set(0)),
        state.replace(unit_ids=state.unit_ids.at[4].set(env.progression.ids['duke'])),
        state.replace(copied=state.copied.at[4].set(True)),
        state.replace(summon_owner=state.summon_owner.at[4].set(0)),
        state.replace(in_battle=jnp.bool_(True)),
        state.replace(summon_battle=jnp.bool_(True)),
        initial.replace(alive=jnp.zeros_like(initial.alive)),
    ):
        assert not bool(won(invalid))
    assert bool(env._goals_left(jnp.zeros_like(state.alive), state.scenario_spawned))


def test_goal_reward_is_once_and_preserves_reference_terminal_rewards(retreat):
    env, before = retreat
    assert env.reference_rewards and env.siege_rewards is None
    assert env.army_victory_level_reward == 0.
    after = before.replace(position=env.objective_position, done=jnp.bool_(True), won=jnp.bool_(True))
    paid, bonus = rewards.finish(env, before, after)
    # Python Default dragon: +6 first contact + 2*(8+40) objective.
    assert float(bonus) == pytest.approx(102.)
    assert float(rewards.finish(env, paid, paid)[1]) == 0.
    timeout = before.replace(done=jnp.bool_(True))
    assert float(rewards.finish(env, before, timeout)[1]) == -8.
    assert float(rewards.finish(env, before, timeout.replace(lost=jnp.bool_(True)))[1]) == -6.
    victorious = after.replace(reference_defeated=~env.scenario.scheduled)
    assert float(rewards.finish(env, before, victorious)[1]) == pytest.approx(107.)


def test_quest_totem_survives_merchant_autosale(retreat):
    env, state = retreat
    totem = env.item_rules.keys.index('agshlisga_totem')
    scroll = env.item_rules.keys.index('sanctuera_scroll')
    merchant = next(s for s in env.site_rules.sites if s['kind'] == 'merchant')
    visiting = state.replace(position=jnp.array(merchant['interaction_tiles'][0], jnp.int32),
        item_inventory=state.item_inventory.at[0].set(totem).at[1].set(scroll))
    sold = env.site_rules.autosell(visiting)
    assert bool(jnp.any(sold.item_inventory == totem))
    assert not bool(jnp.any(sold.item_inventory == scroll))
    assert int(sold.gold-state.gold) == 120 and int(sold.last_sold_count) == 1


def test_step_mask_arrival_terminal_observation_and_reset(retreat):
    env, initial = retreat
    # The source tile immediately south of the goal is clear. No route/actions
    # from this fixture are ever supplied to training.
    before = initial.replace(position=jnp.array([3, 30], jnp.int32))
    mask = compiled_method(env, 'action_mask')
    step = compiled_method(env, 'step')
    assert bool(mask(before)[0])  # north
    after, timestep = step(before, jnp.int32(0))
    assert after.position.tolist() == [2, 30] and bool(after.won) and bool(after.done)
    assert float(timestep.reward) == pytest.approx(102.089, abs=1e-4)  # fresh movement, full leadership
    again, repeated = step(after, jnp.int32(CONTINUE))
    assert bool(again.won) and float(repeated.reward) == 0.
    assert int(jnp.sum(mask(after))) == 1 and bool(mask(after)[CONTINUE])
    obs = compiled_method(env, 'observation')(initial)
    assert obs.shape == (env.observation_size,)
    assert jnp.allclose(obs[-5:-1], jnp.array([2, 30, -20, -6])/(env.size-1))
    fresh, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(102))
    assert not bool(fresh.done) and fresh.position.tolist() == [22, 36]
    # No movement points: mask and step both reject arrival and do not pay the goal.
    exhausted = before.replace(movement_points=jnp.int32(0))
    assert not bool(mask(exhausted)[0])
    blocked, _ = step(exhausted, jnp.int32(0))
    assert not bool(blocked.won) and blocked.position.tolist() == [3, 30]


def test_scenario_limits_and_source_modifiers(retreat):
    env, state = retreat
    source = json.loads((ROOT/'docs/maps/bethrezens_retreat/restrictions.json').read_text())
    assert source['scenario_limits'] == dict(MAX_UNIT=3, MAX_LEADER=5, MAX_CITY=5, MAX_SPELL=3)
    assert source['leadership']['scenario_delta'] == -2
    assert len(source['modifiers']) == 2
    for i, row in enumerate(env.construction.rows):
        if row['level'] > 3:
            assert int(state.blocked_buildings) & (1 << i)
    rich = state.replace(gold=jnp.int32(9999), mana=jnp.full_like(state.mana, 9999),
                         buildings=state.buildings | env.spell_research.tower_bit)
    mask = compiled_method(env, 'action_mask')(rich)
    for i, row in enumerate(env.spell_research.rows):
        if row['level'] > 3:
            action = env.spell_research.start+i
            assert not bool(mask[action])
            rejected, _ = env.spell_research.apply(rich, jnp.int32(action))
            assert int(rejected.learned_spells) == int(rich.learned_spells)
            assert jnp.array_equal(rejected.mana, rich.mana)
            assert int(env.spell_research.status(rich)[i]) == 8
    for i, row in enumerate(env.construction.rows):
        if row['level'] > 3:
            assert not bool(mask[18+i])
            rejected, _ = compiled_method(env, 'step')(rich, jnp.int32(18+i))
            assert int(rejected.buildings) == int(rich.buildings)
            assert int(rejected.gold) == int(rich.gold)


def test_hero_level_ceiling_and_tier_limited_numerical_growth(retreat):
    env, initial = retreat
    progression = env.progression
    hero = int(initial.unit_ids[4])
    # The existing fixed skill ladder adds one place at level 3; level 6 is unreachable.
    level_three = initial.replace(unit_levels=initial.unit_levels.at[4].set(3))
    assert int(env.recruitment.composition(level_three)[0]) == 4
    assert env.recruitment.metadata['leadership_levels'] == [1, 3]

    def award(uid, level, buildings=0):
        ids = jnp.zeros_like(initial.unit_ids).at[4].set(uid)
        levels = jnp.zeros_like(initial.unit_levels).at[4].set(level)
        state = initial.replace(unit_ids=ids, unit_levels=levels, unit_xp=jnp.zeros_like(ids),
                                buildings=jnp.uint32(buildings), in_battle=jnp.bool_(True),
                                battle_earned_xp=jnp.zeros_like(ids).at[4].set(100000))
        hp = jnp.zeros_like(initial.hp).at[4].set(1)
        return progression.finish(state, hp, jnp.zeros(12, bool), jnp.bool_(True),
            jnp.bool_(False), jnp.bool_(False))

    assert int(award(hero, 4)[1][4]) == 5
    for level in (5, 8):  # imported leaders are not reduced to the ceiling
        result = award(hero, level)
        assert int(result[1][4]) == level and int(result[4][4]) == 1
        assert int(result[5][4]) == 0
    paladin = next(i for i, r in enumerate(progression.rows) if r['name'] == 'Темный паладин')
    result = award(paladin, 3)
    assert int(result[0][4]) == paladin and int(result[1][4]) == 4
    assert int(result[4][4]) > 1
    berserker = progression.ids['berserker']
    result = award(berserker, 2)  # permitted next building has not been built
    assert int(result[0][4]) == berserker and int(result[1][4]) == 2
    assert int(result[4][4]) == 1
