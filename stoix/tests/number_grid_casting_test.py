"""CUDA contracts: map casts of researched spells, Python reference rules."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.envs.number_grid import (
    MAP, ACTION_NAMES, NumberGrid, SHOOT, DEFEND, CONTINUE, REST, VICTORY, WITHDRAW,
)
from stoix.envs.number_grid_buildings import BuildingRules, FACTIONS
from stoix.envs.number_grid_casting import (
    KINDS, NEUTRAL, SPELL_CAST, SPELL_KILL, SUMMON_CAST,
    SpellCastRules, cast_action_names,
)
from stoix.envs.number_grid_combat import ATTACK_TYPES, HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE
from stoix.envs.number_grid_spells import CATALOG, SpellResearchRules
from stoix.envs.number_grid_terrain import FOREST, WATER
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state

EAST = 2
MANA = jnp.full(5, 5000, jnp.int32)


def ready(env, initial, *spells, **changes):
    """Day-one map state with the listed legion spells learned and ample mana."""
    bits = sum(1 << i for i in spells)
    return initial.replace(learned_spells=jnp.uint32(bits), mana=MANA, **changes)


def faction_rules(env, faction, lord='warrior'):
    config = {**MAP, 'faction': faction, 'lord_type': lord}
    construction = BuildingRules(config)
    research = SpellResearchRules(config, construction, 195)
    return SpellCastRules(config, research, construction, env.progression, env.territory, env.ruins, 195)


def run_battle(step, state, policy, limit=400):
    rewards = []
    for _ in range(limit):
        if not bool(state.in_battle):
            break
        state, ts = step(state, jnp.int32(policy(state)))
        rewards.append(float(ts.reward))
    assert not bool(state.in_battle)
    return state, rewards


def test_catalog_cast_data_actions_and_summons(current_game):
    env = current_game[0]
    rows = [row for faction in CATALOG['factions'].values() for row in faction]
    assert {row['cast']['kind'] for row in rows} == set(KINDS)
    for row in rows:
        cast = row['cast']
        assert len(cast['cast_mana']) == 5 and sum(cast['cast_mana']) > 0
        if cast['kind'] in ('damage', 'ward'):
            assert cast['element'] in ATTACK_TYPES
        if cast['kind'] == 'summon_battle':
            assert cast['unit'] in env.progression.ids
    # Gspells.dbf, Storm: 50 earth damage (the Python reference lists 60).
    storm = next(row for row in rows if row['id'] == 'g000ss0033')
    assert storm['cast'] == dict(kind='damage', cast_mana=[0, 150, 0, 150, 0], amount=50, element='earth')
    names = cast_action_names(MAP)
    assert ACTION_NAMES[212:229] == names and len(names) == 17 and env.num_actions == 244
    assert env.casting.start == 212 and env.casting.limit == 1
    meta = env.casting.metadata['spells']
    assert [m['unit_name'] for m in meta if m['kind'] == 'summon_battle'] == ['Адская гончая', 'Белиарх', 'Мститель']
    # _entry_stand: weapon or large summons stand at position 7, others at 10.
    slots = {m['id']: m['summon_slot'] for f in ('mountain_clans', 'undead_hordes')
             for m in faction_rules(env, f).metadata['spells'] if m['kind'] == 'summon_battle'}
    assert slots == {'g000ss0025': 0, 'g000ss0031': 3, 'g000ss0038': 0, 'g000ss0061': 0,
                     'g000ss0066': 0, 'g000ss0071': 3, 'g000ss0078': 0}
    clans = faction_rules(env, 'mountain_clans')
    valkyrie = next(i for i, r in enumerate(clans.rows) if r['id'] == 'g000ss0031')
    state = current_game[1].replace(spell_casts=jnp.zeros(clans.count, jnp.int32), mana=MANA,
                                    learned_spells=jnp.uint32(1 << valkyrie))
    out, _, summon = jax.jit(clans.apply)(state, jnp.int32(195+valkyrie), env.max_hp(state), 20)
    assert summon and out.unit_ids[3] == env.progression.ids['g000uu0050'] and jnp.sum(out.unit_ids[:6] != 0) == 1
    assert out.hp[3] == env.progression.stats(out.unit_ids[3], out.unit_levels[3])[HP] and out.unit_xp[3] == 0
    # Only the mage casts the same spell twice per turn; level V needs the mage.
    for lord, limit in (('warrior', 1), ('mage', 2), ('guildmaster', 1)):
        rules = faction_rules(env, 'legions', lord)
        assert rules.limit == limit
        assert bool(rules.allowed[15]) == (lord == 'mage')


def test_nearest_target_is_one_vector_chebyshev_argmin(current_game):
    env = current_game[0]
    rules = env.casting
    positions = np.asarray(rules.positions)
    field = np.asarray(rules.field)
    rng = np.random.default_rng(3)
    points = rng.integers(1, 47, size=(256, 2)).astype(np.int32)
    alive = rng.random((256, rules.enemy_count)) < .3
    alive[0] = False  # no living stack at all
    initial = current_game[1]
    batch = jax.vmap(lambda p, a: initial.replace(position=p, alive=a))(jnp.asarray(points), jnp.asarray(alive))
    any_target, field_target, distance = jax.jit(jax.vmap(rules.targets))(batch)
    for k in range(len(points)):
        d = np.max(np.abs(positions - points[k]), axis=1)
        np.testing.assert_array_equal(distance[k], d)
        for target, eligible in ((any_target, alive[k]), (field_target, alive[k] & field)):
            # Python reference: min over (distance, enemy id), no path search.
            expected = min(((d[i], i) for i in range(len(d)) if eligible[i]), default=(0, -1))[1]
            assert int(target[k]) == expected


@pytest.mark.core
def test_damage_hits_nearest_field_stack_spends_mana_and_daily_limit(current_game):
    env, initial, step, _ = current_game
    rules = env.casting
    mask = compiled_method(env, 'action_mask')
    state = ready(env, initial, 1, 2)
    assert mask(state)[rules.start+1] and mask(state)[rules.start+2] and not mask(state)[rules.start]
    burned, ts = step(state, jnp.int32(rules.start+1))
    assert float(ts.reward) == pytest.approx(rules.reward-env.step_cost)
    assert burned.last_event == SPELL_CAST and burned.last_spell_target == 0 and burned.last_spell_amount == 15
    chex.assert_trees_all_equal(burned.enemy_wounds[0], jnp.array([15, 0, 0, 0, 0, 0], jnp.int32))
    assert not jnp.any(burned.enemy_wounds[1:])
    chex.assert_trees_all_equal(burned.mana, MANA.at[0].add(-100))
    assert burned.spell_casts[1] == 1 and burned.alive[0] and not burned.in_battle
    for field in ('position', 'movement_points', 'hp', 'unit_ids', 'gold', 'day'):
        chex.assert_trees_all_equal(getattr(burned, field), getattr(state, field))
    # Same spell once per turn for a warrior lord; another learned spell stays open.
    assert not mask(burned)[rules.start+1] and mask(burned)[rules.start+2]
    denied, ts = step(burned, jnp.int32(rules.start+1))
    chex.assert_trees_all_equal(denied.enemy_wounds, burned.enemy_wounds)
    chex.assert_trees_all_equal(denied.mana, burned.mana)
    assert float(ts.reward) == pytest.approx(-env.step_cost)
    both, _ = step(burned, jnp.int32(rules.start+2))
    assert both.enemy_wounds[0, 0] == 30
    tail = compiled_method(env, 'observation')(both)[:-1][-rules.observation_size-env.unit_swaps.observation_size:-env.unit_swaps.observation_size]
    np.testing.assert_allclose(tail[:rules.count], np.eye(rules.count)[1]+np.eye(rules.count)[2])
    assert float(tail[2*rules.count]) == pytest.approx(.3)  # 30 of the Squire's 100 HP
    np.testing.assert_allclose(tail[-5:], [0, 1, 1, 6/47, 6/47], rtol=1e-6)
    # REST: limits expire; the Squire's own 5% +10% on land nobody owns (none on the player's).
    rested, _ = step(both, jnp.int32(REST))
    owned = bool(jax.jit(env.territory.owned)(rested, rules.positions)[0])
    assert rested.enemy_wounds[0, 0] == 30-(5 if owned else 15)
    assert not jnp.any(rested.spell_casts) and mask(rested)[rules.start+1]
    # Status codes shared by the viewer: ready, unknown, used, mana, no target, battle, done.
    quotes = jax.jit(env.cast_quotes)
    status = quotes(burned)['status']
    assert status[1] == 2 and status[2] == 0 and status[0] == 1
    assert quotes(burned.replace(mana=jnp.zeros(5, jnp.int32)))['status'][2] == 3
    assert quotes(burned.replace(alive=jnp.zeros_like(burned.alive)))['status'][2] == 4
    assert quotes(burned.replace(in_battle=jnp.bool_(True)))['status'][2] == 5
    assert quotes(burned.replace(done=jnp.bool_(True)))['status'][2] == 6


def test_kill_by_spell_removes_stack_without_victory_reward(current_game):
    env, initial, step, _ = current_game
    rules = env.casting
    state = ready(env, initial, 1, enemy_wounds=initial.enemy_wounds.at[0, 0].set(90))
    killed, ts = step(state, jnp.int32(rules.start+1))
    assert killed.last_event == SPELL_KILL and killed.last_spell_amount == 10
    assert not killed.alive[0] and killed.number == state.number+1 and not killed.won
    assert float(ts.reward) == pytest.approx(rules.reward-env.step_cost)
    retarget, _ = step(killed.replace(spell_casts=jnp.zeros_like(killed.spell_casts)), jnp.int32(rules.start+1))
    assert retarget.last_spell_target != 0


def test_immunity_and_ward_block_spell_damage_but_not_other_units(current_game):
    env, initial, step, _ = current_game
    rules = env.casting
    # Stack 27 (Uter) is immune to fire and wards against mind; it is the nearest one here.
    state = ready(env, initial, 1, 2, 12, position=jnp.array([19, 26], jnp.int32))
    blocks = np.asarray(rules.blocks[27])
    for spell, element in ((1, 'fire'), (2, 'mind'), (12, 'earth')):
        out, _ = step(state, jnp.int32(rules.start+spell))
        bit = 1 << ATTACK_TYPES.index(element)
        expected = np.where((np.asarray(rules.occupied[27])) & ((blocks & bit) == 0),
                            np.minimum(rules.metadata['spells'][spell]['amount'], np.asarray(rules.enemy_max_hp(state)[27])), 0)
        assert out.last_spell_target == 27
        np.testing.assert_array_equal(out.enemy_wounds[27], expected)
    assert blocks[0] & 4 and blocks[0] & 64


def test_damage_skips_cities_and_ruins_summons_reach_them(current_game):
    env, initial, _, _ = current_game
    quotes = jax.jit(env.cast_quotes)
    # Two city defenders (41, 42) share [14,16]; the nearest field stack is 32 at [16,14].
    state = ready(env, initial, 0, 1, position=jnp.array([14, 15], jnp.int32))
    target = quotes(state)['target']
    assert target[0] == 41 and target[1] == 32 and target[3] == 32
    # The ruin guard [9,25] is not a field stack either.
    state = ready(env, initial, 0, 1, position=jnp.array([10, 26], jnp.int32))
    target = quotes(state)['target']
    assert target[0] == 45 and target[1] == 13 and not bool(env.casting.field[45])


def test_debuffs_change_the_next_battle_of_that_stack_only(current_game):
    env, initial, step, _ = current_game
    rules = env.casting
    state = ready(env, initial, 6, 7, position=jnp.array([3, 7], jnp.int32))
    state, _ = step(state, jnp.int32(rules.start+6))
    state, _ = step(state, jnp.int32(rules.start+7))
    assert state.enemy_spell_effects[0] == rules.bits[6] | rules.bits[7]
    battle, ts = step(state, jnp.int32(EAST))
    assert battle.in_battle and battle.enemy == 0 and not battle.summon_battle
    stats = compiled_method(env, 'unit_stats')(battle)
    # Squire: damage 25 x0.85 and initiative 50 x0.85, reference round-half-even.
    assert stats[6, DAMAGE] == 21 and stats[6, INITIATIVE] == 42
    chex.assert_trees_all_equal(battle.spell_mods[:6], jnp.tile(jnp.array(NEUTRAL), (6, 1)))
    assert float(ts.reward) == pytest.approx(-env.step_cost)


def test_blue_modifier_layer_order_potions_spell_equipment(current_game):
    env, initial, step, _ = current_game
    battle, _ = step(initial.replace(position=jnp.array([3, 7], jnp.int32)), jnp.int32(EAST))
    stats = compiled_method(env, 'unit_stats')
    layered = battle.replace(potion_temporary=battle.potion_temporary.at[1, DAMAGE].set(5.),
                             equipment_bonus=battle.equipment_bonus.at[1, DAMAGE].set(10.))
    base = stats(layered)
    mods = jnp.array([50., 1.1, 1.2, 10., 1.5])
    buffed = stats(layered.replace(spell_mods=layered.spell_mods.at[1].set(mods)))
    natural = env.progression.stats(battle.unit_ids[1], battle.unit_levels[1])
    assert buffed[1, HP] == base[1, HP]+50 and buffed[1, ARMOR] == base[1, ARMOR]+10
    assert buffed[1, DAMAGE] == np.rint((natural[DAMAGE]+5)*1.1)+10
    assert buffed[1, ACCURACY] == min(100, np.rint(natural[ACCURACY]*1.2))
    assert buffed[1, INITIATIVE] == np.rint(natural[INITIATIVE]*1.5)
    chex.assert_trees_all_equal(buffed[2:], base[2:])


def test_support_spells_heal_moves_buffs_wards_health_and_terrain(current_game):
    env, initial, _, _ = current_game
    max_hp = jnp.array([120, 150, 120, 45]+[0]*8, jnp.int32)
    for faction, checks in (('empire', ('heal', 'moves', 'buff', 'ward')), ('elves', ('health_bonus',)),
                            ('mountain_clans', ('terrain',))):
        rules = faction_rules(env, faction)
        apply, available = jax.jit(rules.apply), jax.jit(rules.available)
        base = initial.replace(spell_casts=jnp.zeros(rules.count, jnp.int32), mana=MANA,
                               learned_spells=jnp.uint32((1 << rules.count)-1),
                               enemy_spell_effects=jnp.zeros_like(initial.enemy_spell_effects))
        index = {kind: next(i for i, r in enumerate(rules.rows) if r['cast']['kind'] == kind
                            and r['cast'].get('stat') != 'terrain') for kind in checks if kind != 'terrain'}
        if 'heal' in checks:
            i = index['heal']  # Healing: 30
            wounded = base.replace(hp=base.hp.at[:4].set(jnp.array([100, 0, 110, 45])))
            assert available(base, max_hp, 20)[i] == False and available(wounded, max_hp, 20)[i]
            out, reward, summon = apply(wounded, jnp.int32(195+i), max_hp, 20)
            np.testing.assert_array_equal(out.hp[:4], [120, 0, 120, 45])
            assert out.last_spell_amount == 30 and reward == pytest.approx(rules.reward) and not summon
        if 'moves' in checks:
            i = index['moves']  # Acceleration: 50% of the full allowance, 20 here
            assert rules.metadata['spells'][i]['percent'] == 50
            out, _, _ = apply(base.replace(movement_points=jnp.int32(4)), jnp.int32(195+i), max_hp, 20)
            assert out.movement_points == 14
            out, _, _ = apply(base.replace(movement_points=jnp.int32(15)), jnp.int32(195+i), max_hp, 20)
            assert out.movement_points == 20 and not available(out, max_hp, 20)[i]
            assert not available(base.replace(movement_points=jnp.int32(20)), max_hp, 20)[i]
        if 'buff' in checks:
            i = index['buff']  # Haste: initiative x1.1, no renewal while active
            out, _, _ = apply(base, jnp.int32(195+i), max_hp, 20)
            assert out.spell_effects == rules.bits[i] and not available(out.replace(spell_casts=base.spell_casts), max_hp, 20)[i]
            started, _ = rules.begin_battle(out.replace(hp=out.hp.at[1].set(0)), out.hp.at[1].set(0))
            assert started.spell_mods[0, INITIATIVE] == pytest.approx(1.1)
            chex.assert_trees_all_equal(started.spell_mods[1], jnp.array(NEUTRAL))
        if 'ward' in checks:
            i = index['ward']  # Ward against Air magic, living hero units only
            out, _, _ = apply(base, jnp.int32(195+i), max_hp, 20)
            started, _ = rules.begin_battle(out, out.hp)
            np.testing.assert_array_equal(started.spell_wards, [256]*4+[0]*8)
        if 'health_bonus' in checks:
            i = index['health_bonus']  # Galean's blessing: +50 HP until REST
            out, _, _ = apply(base, jnp.int32(195+i), max_hp, 20)
            hp = out.hp.at[1].set(0)
            started, battle_hp = rules.begin_battle(out.replace(enemy=jnp.int32(0)), hp)
            np.testing.assert_array_equal(battle_hp[:4], [170, 0, 170, 95])
            # Reference restore: max(0, HP-bonus); a promoted unit keeps its new HP.
            after = started.replace(hp=jnp.array([30, 0, 160, 95]+[0]*8, jnp.int32), last_promoted=jnp.uint32(4))
            done = rules.finish_battle(started, after, after.hp, jnp.bool_(True), jnp.bool_(True))
            np.testing.assert_array_equal(done.hp[:4], [0, 0, 160, 45])
        if 'terrain' in checks:
            # Use a walking leader: the Duke now flies without a terrain spell.
            walking = hero_roster_state(env,base,['possessed','g000uu0044','possessed','cultist',None,None])
            forest = next(i for i, r in enumerate(rules.rows) if r['cast'].get('terrain') == 'forest')
            out, _, _ = apply(walking, jnp.int32(195+forest), env.max_hp(walking), 20)
            assert tuple(map(bool, rules.terrain(out))) == (True, False)
            costs = jax.jit(env.terrain.costs, static_argnums=1)(out, None, rules.terrain(out))
            assert costs[FOREST] == 2 and costs[WATER] == 6
            dead = out.replace(hp=out.hp.at[1].set(0))  # King's Guard leads the party
            assert jax.jit(env.terrain.costs, static_argnums=1)(dead, None, rules.terrain(dead))[FOREST] == 4


@pytest.mark.core
def test_summon_battle_victory_returns_party_and_pays_no_defeat_reward(current_game):
    env, initial, step, _ = current_game
    rules = env.casting
    mask = compiled_method(env, 'action_mask')
    # Squire with 10 HP left: the lone Hellhound is fixed at slot 0 at its native level.
    state = ready(env, initial, 0, enemy_wounds=initial.enemy_wounds.at[0, 0].set(90))
    summoned, ts = step(state, jnp.int32(rules.start))
    hound = int(rules.summon_ids[0])
    assert summoned.in_battle and summoned.summon_battle and summoned.enemy == 0
    assert summoned.last_event == SUMMON_CAST and float(ts.reward) == pytest.approx(rules.reward-env.step_cost)
    np.testing.assert_array_equal(summoned.unit_ids[:6], [hound, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(summoned.hp, [125, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0])
    chex.assert_trees_all_equal(summoned.spell_party, jnp.stack((state.unit_ids[:6], state.unit_levels[:6],
                                                              state.unit_xp[:6], state.hp[:6])))
    chex.assert_trees_all_equal(summoned.position, state.position)
    chex.assert_trees_all_equal(summoned.movement_points, state.movement_points)
    policy = lambda s: SHOOT if bool(mask(s)[SHOOT]) else CONTINUE
    final, rewards = run_battle(step, summoned, policy)
    assert final.last_event == VICTORY and not final.alive[0] and final.number == state.number+1
    assert sum(rewards) == pytest.approx(-env.step_cost*len(rewards))
    # Red slots keep the last opponent's ids after any battle; the party is exact.
    for field in ('unit_ids', 'unit_levels', 'unit_xp', 'hp'):
        chex.assert_trees_all_equal(getattr(final, field)[:6], getattr(state, field)[:6])
    for field in ('position', 'recovery_balance', 'potions', 'potion_bonus', 'equipment_bonus'):
        chex.assert_trees_all_equal(getattr(final, field), getattr(state, field))
    assert not final.summon_battle and not final.lost and not final.done and final.last_promoted == 0
    chex.assert_trees_all_equal(final.spell_mods, jnp.tile(jnp.array(NEUTRAL), (12, 1)))


@pytest.mark.core
def test_summon_defeat_keeps_wounds_hero_bonus_and_never_ends_the_game(current_game):
    env, initial, step, _ = current_game
    rules = env.casting
    mask = compiled_method(env, 'action_mask')
    state = ready(env, initial, 4, position=jnp.array([3, 7], jnp.int32))  # Beliarch, 200 HP
    summoned, _ = step(state, jnp.int32(rules.start+4))
    assert summoned.summon_battle and summoned.enemy == 0
    # One strike, then the summon only defends with 1 HP until the Squire kills it.
    weakened = summoned.replace(hp=summoned.hp.at[0].set(1))
    first = {'done': False}
    def policy(s):
        if s.actor >= 6 or not bool(mask(s)[DEFEND]):
            return CONTINUE
        if not first['done'] and bool(mask(s)[SHOOT]):
            first['done'] = True
            return SHOOT
        return DEFEND
    final, _ = run_battle(step, weakened, policy)
    assert final.last_event == WITHDRAW and final.alive[0] and not final.lost and not final.done
    for field in ('unit_ids', 'unit_levels', 'unit_xp', 'hp'):
        chex.assert_trees_all_equal(getattr(final, field)[:6], getattr(state, field)[:6])
    chex.assert_trees_all_equal(final.position, state.position)
    assert final.summon_marks[0] and not final.summon_battle
    # Hero engagement consumes the mark; its configured action bonus is zero on v89.
    battle, ts = step(final, jnp.int32(EAST))
    assert battle.in_battle and not battle.summon_battle and not battle.summon_marks[0]
    assert float(ts.reward) == pytest.approx(rules.summon_engage_reward-env.step_cost)
    squire = int(rules.enemy_max_hp(final)[0, 0])
    assert battle.hp[6] == squire-final.enemy_wounds[0, 0] and battle.enemy_initial_hp[0] == battle.hp[6]


@pytest.mark.core
def test_summon_round_limit_keeps_existing_return_without_retreat_penalty(current_game):
    env, initial, step, _ = current_game
    state = ready(env, initial, 0)
    summoned, _ = step(state, jnp.int32(env.casting.start))
    summoned = summoned.replace(round=jnp.int32(env.max_rounds), actor=jnp.int32(0),
        turn_phase=jnp.full(12, 2, jnp.int32).at[0].set(0))
    out, ts = step(summoned, jnp.int32(DEFEND))
    assert out.last_event == WITHDRAW and not out.in_battle and not out.done
    assert not out.summon_battle and not env.round_limit_retreat(out)
    assert out.step_count == summoned.step_count+1
    for field in ('unit_ids', 'unit_levels', 'unit_xp', 'hp'):
        chex.assert_trees_all_equal(getattr(out, field)[:6], getattr(state, field)[:6])
    assert float(ts.reward) == pytest.approx(-env.step_cost-.05*env.recruitment.composition(out)[2])


@pytest.mark.parametrize('compare_batch', [
    pytest.param(False, id='scalar', marks=pytest.mark.core), pytest.param(True, id='full-batch'),
])
def test_batched_casts_match_single_transitions_and_mask_equals_step(current_game, compare_batch):
    """Check cast legality against step and optionally compare full-map batch transitions."""
    env, initial, step, _ = current_game
    rules = env.casting
    mask = compiled_method(env, 'action_mask')
    states = [ready(env, initial, 0, 1, 3), ready(env, initial, 1, position=jnp.array([19, 26], jnp.int32)),
              ready(env, initial, 0, 1).replace(mana=jnp.zeros(5, jnp.int32)),
              ready(env, initial, 3, enemy_spell_effects=initial.enemy_spell_effects.at[0].set(rules.bits[3]))]
    actions = [rules.start, rules.start+1, rules.start+1, rules.start+3]
    if compare_batch:
        batched = compiled_method(env, 'step', batched=True)
        batch = jax.tree.map(lambda *x: jnp.stack(x), *states)
        out, ts = batched(batch, jnp.array(actions, jnp.int32))
        for k, (state, action) in enumerate(zip(states, actions)):
            single, single_ts = step(state, jnp.int32(action))
            chex.assert_trees_all_equal(jax.tree.map(lambda x, k=k: x[k], out), single)
            assert float(ts.reward[k]) == pytest.approx(float(single_ts.reward))
    for state in states:
        allowed = mask(state)
        for i in range(rules.count):
            out, _ = step(state, jnp.int32(rules.start+i))
            assert bool(allowed[rules.start+i]) == bool(out.spell_casts[i] == state.spell_casts[i]+1)
    # A debuff already on the nearest stack may be cast again (the reference only
    # limits casts per day), and the same spell's bit simply stays set.
    assert mask(states[3])[rules.start+3]


def test_human_cast_metadata_quotes_and_summon_event(human_service):
    service = human_service
    game = service.create(42, 'legions', 'warrior')
    key, state, total = service.sessions[game['session']]
    env, _, _ = service.environment(*key)
    casting = game['spell_casting']
    assert casting['daily_limit'] == 1 and len(casting['spells']) == 17
    assert casting['spells'][1] == {**casting['spells'][1], 'kind': 'damage', 'amount': 15, 'element': 'fire', 'action': 213}
    assert game['snapshot']['cast_quotes']['status'] == [1]*17
    service.sessions[game['session']] = (key, state.replace(learned_spells=jnp.uint32(1), mana=MANA), total)
    event = service.act(game['session'], env.casting.start)['snapshot']
    assert event['state']['last_event'] == SUMMON_CAST and event['state']['summon_battle']
    assert event['state']['in_battle'] and event['state']['last_spell_target'] == 0
