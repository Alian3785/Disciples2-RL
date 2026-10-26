"""Requested hard limits (2026-10-19); GPU-only regressions, no core budget added."""
import copy

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, REST, SHOOT
from stoix.envs.number_grid_combat import HP, ACCURACY, ARMOR, INITIATIVE
from stoix.envs.number_grid_limits import add_resource
from stoix.envs.number_grid_potions import POTIONS, POTION_START, PotionRules
from stoix.tests.number_grid_fixtures import MAP as BASIC_MAP, compiled_method


@pytest.fixture(autouse=True)
def require_gpu():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'


def test_initial_profiles_and_starting_treasury_are_capped():
    """Starting gold, mana and unit profiles are capped; malformed starting_mana is refused."""
    config = copy.deepcopy(BASIC_MAP)
    config['starting_gold'] = 2**40  # Cap before conversion to the state's int32.
    config['starting_mana'] = {'infernal': 2**40, 'elves': 7}  # MANA_KINDS order
    config['hero_combat_stats'] = [dict(max_hp=20000, initiative=500, armor=100)
                                   for _ in range(config['hero_units'])]
    env = NumberGrid(map_config=config)
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    stats = compiled_method(env, 'unit_stats')(state)
    occupied = state.hp[:6] > 0
    assert state.gold == 9999 and state.mana.tolist() == [9999, 0, 0, 0, 7]
    for invalid in ({'gold': 1}, {'infernal': -1}, {'infernal': 1.5}, [200]):
        with pytest.raises(ValueError, match='starting_mana'):
            NumberGrid(map_config={**BASIC_MAP, 'starting_mana': invalid})
    assert jnp.all(state.hp[:6][occupied] == 9999)
    assert jnp.all(stats[:6, INITIATIVE][occupied] == 150)
    assert jnp.all(stats[:6, ARMOR][occupied] == 90)


def test_level_99_blocks_victory_xp_healing_and_training(current_game):
    env, initial, advance, fight = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    ids = state.unit_ids.at[0].set(env.progression.ids['infernal_knight'])
    levels = state.unit_levels.at[:2].set(99)
    xp = state.unit_xp.at[:2].set(env.progression.required_xp(ids[:2], levels[:2])-1)
    state = state.replace(unit_ids=ids, unit_levels=levels, unit_xp=xp,
        actor=jnp.int32(1), hp=state.hp.at[:].set(0).at[:2].set(1).at[6].set(1))
    won, _ = fight(state, jnp.int32(SHOOT), state.battle_key, jnp.zeros(env.random_size))
    assert not won.in_battle and not won.alive[0]
    chex.assert_trees_all_equal(won.unit_levels[:2], jnp.array([99, 99]))
    chex.assert_trees_all_equal(won.unit_xp[:2], xp[:2])
    chex.assert_trees_all_equal(won.hp[:2], jnp.ones(2, jnp.int32))
    assert not jnp.any(won.last_xp[:2]) and won.last_promoted == 0
    trainer = next(s for s in env.site_rules.sites if s['kind'] == 'trainer')
    before = won.replace(position=jnp.array(trainer['interaction_tiles'][0]), gold=jnp.int32(9999))
    quotes = compiled_method(env.site_rules, 'training_quotes')(before)
    mask = compiled_method(env, 'action_mask')(before)
    assert not jnp.any(quotes[:2, 3:])
    for slot in (0, 1):
        action = env.site_rules.train_start+slot
        assert not mask[action]
        after, _ = advance(before, jnp.int32(action))
        assert after.gold == before.gold
        chex.assert_trees_all_equal(after.unit_xp, before.unit_xp)


def test_enemy_levels_and_lower_scenario_ceiling(current_game):
    env, state, _, _ = current_game
    rules = copy.copy(env.progression)
    rules.max_hero_level = 6
    ids = jnp.array([rules.ids['duke'], rules.ids['infernal_knight']])
    assert jax.jit(rules.level_up_allowed)(ids, jnp.array([6, 98])).tolist() == [False, True]
    state = state.replace(enemy=jnp.int32(0),
        unit_ids=state.unit_ids.at[6:8].set(ids),
        unit_levels=state.unit_levels.at[6:8].set(99),
        hp=state.hp.at[:].set(0).at[6:8].set(10))
    result = jax.jit(rules.finish)(state, state.hp, state.escaped, False, True, False,
                                   jnp.array([100000, 0]))
    _, levels, _, enemies, hp, gains, promoted = result
    assert levels[6:8].tolist() == [99, 99]
    assert enemies[0, :2, 0].tolist() == [99, 99]
    assert hp[6:8].tolist() == [10, 10] and not jnp.any(gains) and promoted == 0


def test_all_catalogue_growth_obeys_limits(current_game):
    env = current_game[0]
    rules = env.progression
    ids = jnp.arange(len(rules.rows), dtype=jnp.int32)
    at_limit = jax.jit(rules.stats)(ids, jnp.full_like(ids, 99))
    assert jnp.all(at_limit[:, HP] <= 9999)
    assert jnp.all(at_limit[:, ACCURACY] <= 100)
    assert jnp.all(at_limit[:, ARMOR] <= 90)
    assert jnp.all(at_limit[:, INITIATIVE] <= 150)
    chex.assert_trees_all_equal(at_limit, jax.jit(rules.stats)(ids, jnp.full_like(ids, 100)))


def test_stacked_potions_final_stats_and_health(current_game):
    source, state, _, _ = current_game
    env = copy.copy(source)
    env.progression = copy.copy(source.progression)
    hero = int(state.unit_ids[1])
    # A legal custom profile near the HP ceiling and with initiative 90.
    env.progression.stat_levels = env.progression.stat_levels.at[:, hero, HP].set(9000)
    env.progression.stat_levels = env.progression.stat_levels.at[:, hero, INITIATIVE].set(90)
    rules = PotionRules(dict(initial_potions={p['key']: 1 for p in POTIONS}))
    state = state.replace(potions=rules.initial_counts, hp=state.hp.at[1].set(9000))
    keys = ('highfather', 'quicksilver', 'swiftness', 'speed', 'celerity',
            'protection', 'bark', 'invulnerability', 'striking', 'accuracy')
    actions = jnp.array([POTION_START+6*rules.keys.index(k)+1 for k in keys], jnp.int32)

    @jax.jit
    def apply_all(s):
        def apply_one(s, action):
            intrinsic = env._potion_intrinsic(s.unit_ids, s.unit_levels)[1]
            s = rules.apply(s, action, env.max_hp(s), intrinsic)
            return s, env.unit_stats(s)[1]
        return jax.lax.scan(apply_one, s, actions)

    result, history = apply_all(state)
    assert history[1:5, INITIATIVE].tolist() == [99, 114, 148, 150]
    assert result.hp[1] == 9999 and history[-1, HP] == 9999
    assert history[-1, ARMOR] == 90 and history[-1, ACCURACY] == 100
    assert jnp.all(result.potions[jnp.array([rules.keys.index(k) for k in keys])] == 0)
    expired = rules.expire(result, True)
    assert jax.jit(env.unit_stats)(expired)[1, INITIATIVE] == 99
    # Equipment and fortification are also covered by the final cap.
    fortified = result.replace(in_battle=jnp.bool_(True), battle_fort_armor=jnp.int32(90),
        equipment_bonus=result.equipment_bonus.at[1, INITIATIVE].set(100),
        potion_bonus=result.potion_bonus.at[6:, ARMOR].set(80))
    stats = jax.jit(env.unit_stats)(fortified)
    assert jnp.all(stats[:, ARMOR] <= 90) and stats[1, INITIATIVE] == 150
    copied = jax.jit(env._copy_form_stats)(result, jnp.int32(1))
    assert copied[HP] == 9999 and copied[INITIATIVE] == 99
    # Return a half-wounded copy to a natural form capped at 9999, not raw 10350.
    transformed = result.replace(copied=result.copied.at[1].set(True),
        copy_stats=result.copy_stats.at[1, HP].set(8000), hp=result.hp.at[1].set(4000),
        native_ids=result.unit_ids, native_levels=result.unit_levels)
    _, restored = jax.jit(env._restore_roster)(transformed, transformed.hp, True)
    assert restored[1] == 5000


def test_income_caps_gold_and_each_mana_and_does_not_bank_overflow(current_game):
    env, state, advance, _ = current_game
    before = state.replace(territory_claims=env.territory.reachable_counts,
        gold=jnp.int32(9990), mana=jnp.array([9990, 9999, 9980, 0, 9998], jnp.int32))
    after, _ = advance(before, jnp.int32(REST))
    assert after.gold == 9999 and after.mana.tolist() == [9999, 9999, 9999, 50, 9999]
    # A subsequent spend cannot recover the discarded income.
    credited = jax.jit(add_resource)(jnp.int32(9990), jnp.int32(2**31-1))
    assert credited-100 == 9899


def test_ruin_and_autosale_overflow_are_discarded(current_game):
    env, state, _, _ = current_game
    ruins = copy.copy(env.ruins)
    ruins.gold = jnp.full_like(ruins.gold, 10000)
    before = state.replace(enemy=ruins.enemies[0], gold=jnp.int32(9990))
    looted, _ = jax.jit(ruins.grant)(before, True)
    assert looted.gold == 9999 and looted.ruin_looted[0]
    again, _ = jax.jit(ruins.grant)(looted.replace(gold=jnp.int32(0)), True)
    assert again.gold == 0
    merchant = next(s for s in env.site_rules.sites if s['kind'] == 'merchant')
    before = state.replace(position=jnp.array(merchant['interaction_tiles'][0]), gold=jnp.int32(9990),
        item_inventory=state.item_inventory.at[0].set(env.item_rules.keys.index('bronze_ring')))
    sold = compiled_method(env.site_rules, 'autosell')(before)
    assert sold.gold == 9999 and sold.last_sale_gold == 9 and sold.last_sold_count == 1
