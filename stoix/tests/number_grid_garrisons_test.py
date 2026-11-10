"""City and capital garrisons (docs/WORLD.md#garrisons) on city_defence_train; CUDA, by request only."""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, DEFEND, REST, SHOOT, NumberGrid, action_names, load_map
from stoix.envs.number_grid_garrisons import (CITY_LOST, GARRISON_HIRED, GARRISON_REVIVED, GARRISON_SWAPPED,
                                              GUARDIANS, GUARDIAN_SLOT, SWAP_LIMIT, SWAP_PAIRS, UNIT_STORED, UNIT_TAKEN)
from stoix.tests.number_grid_fixtures import compiled_method


@pytest.fixture(scope='module', autouse=True)
def cuda_only():
    assert jax.default_backend() == 'gpu', 'Run only on request through scripts/run_gpu.sh'


@pytest.fixture(scope='module')
def defence():
    env = NumberGrid(map_config=load_map('city_defence_train'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(0))
    return env, state


def step(env, state, action):
    return compiled_method(env, 'step')(state, jnp.int32(action))[0]


def fight(env, state, limit=400):
    mask = compiled_method(env, 'action_mask')
    for _ in range(limit):
        if not bool(state.in_battle) or bool(state.done):
            break
        legal = mask(state)
        choices = [a for a in range(SHOOT, DEFEND+1) if bool(legal[a])]
        state = step(env, state, choices[0] if choices else int(jnp.argmax(legal)))
    return state


def hire(env, state, site, offer):
    return step(env, state, env.garrisons.start+site*5+offer)


@pytest.mark.parametrize('name', ['default', 'the_charmed_child', 'mirror_match', 'trade_train'])
def test_every_campaign_capital_has_its_faction_guardian(name):
    game_map = load_map(name)
    env = NumberGrid(map_config=game_map)
    g = env.garrisons
    assert g.enabled and env.num_actions == len(action_names(game_map)) == g.end
    row = env.progression.rows[g.guardian]
    assert row['game_id'] == GUARDIANS[env.construction.faction] and row['max_hp'] == 900
    assert int(g.initial_ids[g.capital_index, GUARDIAN_SLOT]) == g.guardian
    assert int(jnp.sum(g.initial_ids != 0)) == 1 and g.hire_end-g.start == 5*(g.cities+1)
    assert g.end == g.swap_start+len(SWAP_PAIRS) and g.swap_start == g.take_start+6 == g.store_start+12
    assert g.heal_start == g.hire_end+g.revive_count and g.store_start == g.heal_start+g.revive_count
    assert len(SWAP_PAIRS) == 10


def test_start_owned_empty_city_and_capital_guardian(defence):
    env, state = defence
    g = env.garrisons
    assert state.city_owned.tolist() == [True] and g.defended == 0 and int(state.gold) == 1000
    assert state.garrison_ids[0].tolist() == [0]*6
    capital = env.progression.rows[int(state.garrison_ids[g.capital_index, GUARDIAN_SLOT])]
    assert capital['name'] == 'Мизраэль' and int(state.garrison_hp[g.capital_index, GUARDIAN_SLOT]) == 900
    assert not bool(jnp.any(state.alive))  # both assaults appear later


def test_hire_rows_capacity_columns_and_gold(defence):
    env, state = defence
    g = env.garrisons
    keys = [o['key'] for o in g.metadata['offers']]
    assert keys == ['squire', 'apprentice', 'archer', 'acolyte', 'titan']
    s = hire(env, state, 0, 0)  # squire: first front cell
    assert int(s.garrison_ids[0, 0]) == int(g.offer_ids[0]) and int(s.last_event) == GARRISON_HIRED
    assert int(s.gold) == 1000-int(g.offer_costs[0])
    s = hire(env, s, 0, 2)  # archer: first rear cell
    assert int(s.garrison_ids[0, 3]) == int(g.offer_ids[2])
    assert int(g.targets(s)[g.capital_index, 4]) == 0 and not bool(g.available(s)[g.capital_index*5+4])
    # The Titan needs its capital building, as for the party.
    building = jnp.uint32(1 << int(g.offer_required[4]))
    rich = s.replace(gold=jnp.int32(99999), buildings=s.buildings | building)
    # The capital's middle column holds the guardian behind: a Titan takes the first free column.
    titan = hire(env, rich, g.capital_index, 4)
    assert int(titan.garrison_ids[g.capital_index, 0]) == int(g.offer_ids[4])
    assert int(titan.garrison_ids[g.capital_index, 3]) == 0
    titan = hire(env, titan, g.capital_index, 4)
    assert int(titan.garrison_ids[g.capital_index, 2]) == int(g.offer_ids[4])
    assert int(g.targets(titan)[g.capital_index, 4]) == -1  # no free column left
    # A level-5 city holds five cells.
    full = rich
    for offer in (0, 0, 0, 2, 2, 2):
        full = hire(env, full, 0, offer)
    assert int(jnp.sum(g.occupied(full.garrison_ids)[0])) == 5
    assert not bool(jnp.any(g.available(full)[:5]))


def test_empty_defended_city_falls_on_the_second_rest(defence):
    env, state = defence
    first = step(env, state, REST)
    assert bool(first.alive[0]) and bool(first.city_owned[0]) and not bool(first.done)
    second = step(env, first, REST)
    assert not bool(second.city_owned[0]) and int(second.last_event) == CITY_LOST
    assert bool(second.lost & second.done)


def test_garrison_beats_both_assaults(defence):
    env, state = defence
    g = env.garrisons
    s = state
    for offer in (0, 0, 2, 2, 3):
        s = hire(env, s, 0, offer)
    # Level-1 recruits can lose to the first wave; veterans make the outcome certain.
    levels = jnp.where(s.garrison_ids != 0, 10, 0).at[g.capital_index].set(s.garrison_levels[g.capital_index])
    s = s.replace(garrison_levels=levels, garrison_hp=g.max_hp(s.garrison_ids, levels))
    party = s.unit_ids[:6]
    battles = []
    for _ in range(12):
        s = step(env, s, REST)
        if bool(s.in_battle):
            assert bool(s.garrison_battle) and int(s.garrison_site) == 0
            assert not bool(env._can_retreat(s, jnp.int32(0)))
            battles.append(int(s.day))
            s = fight(env, s)
            assert jnp.array_equal(s.unit_ids[:6], party) and bool(s.city_owned[0])
        if bool(s.done):
            break
    assert len(battles) == 2 and battles[1]-battles[0] == 4
    assert bool(s.won) and not bool(s.lost)


def test_garrison_battle_adds_the_fort_armour(defence):
    """The city's fort armour reaches its garrison, with or without an encounter queue."""
    env, state = defence
    g = env.garrisons
    s = hire(env, state, 0, 0)
    for _ in range(4):
        s = step(env, s, REST)
        if bool(s.in_battle):
            break
    assert bool(s.garrison_battle)
    fort = int(s.battle_hero_fort_armor)
    assert fort == int(env.territory.fort_bonus[s.city_levels[0]]) > 0
    base = env.progression.stats(s.unit_ids[0], s.unit_levels[0])[3]
    assert float(env._combat_stats(s, cap=False)[0, 3]) == float(base)+fort
    assert g.capital_index == 1
    assert bool(jnp.any((s.garrison_ids[0] != 0) & (s.garrison_hp[0] > 0)))


def test_waiting_garrison_battle_forces_continue_and_lost_city_goes_on(defence):
    # Without the defended flag a fallen city only changes hands.
    game_map = load_map('city_defence_train')
    game_map['cities'][0]['defended'] = False
    open_env = NumberGrid(map_config=game_map)
    s, _ = compiled_method(open_env, 'reset')(jax.random.PRNGKey(0))
    s = hire(open_env, s, 0, 3)  # one acolyte
    s = step(open_env, step(open_env, s, REST), REST)
    if not bool(s.in_battle):
        legal = compiled_method(open_env, 'action_mask')(s)
        assert bool(legal[CONTINUE]) and int(jnp.sum(legal)) == 1
        s = step(open_env, s, CONTINUE)
    assert bool(s.garrison_battle)
    s = fight(open_env, s)
    assert not bool(s.city_owned[0]) and not bool(s.lost | s.done | s.in_battle)
    assert int(jnp.sum(s.garrison_ids[0] != 0)) == 0
    assert bool(s.alive[0]) and s.enemy_positions[0].tolist() == open_env.map_config['cities'][0]['position']


def test_regeneration_heals_owned_garrisons_only(defence):
    env, state = defence
    g = env.garrisons
    s = hire(env, state, 0, 0)
    wounded = s.replace(garrison_hp=s.garrison_hp.at[0, 0].set(10).at[g.capital_index, GUARDIAN_SLOT].set(100))
    healed = g.regenerate(wounded, jnp.bool_(True))
    assert int(healed.garrison_hp[0, 0]) > 10 and int(healed.garrison_hp[g.capital_index, GUARDIAN_SLOT]) > 100
    lost = g.regenerate(wounded.replace(city_owned=jnp.zeros_like(wounded.city_owned)), jnp.bool_(True))
    assert int(lost.garrison_hp[0, 0]) == 10
    dead = g.regenerate(wounded.replace(garrison_hp=wounded.garrison_hp.at[0, 0].set(0)), jnp.bool_(True))
    assert int(dead.garrison_hp[0, 0]) == 0


@pytest.fixture(scope='module')
def wild():
    env = NumberGrid(map_config=load_map('wild_mountains'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(143))
    return env, state


def potion_soldier(env, state):
    """The Possessed (party slot 0) with a permanent HP potion and a little experience."""
    slot = 0
    uid, level = state.unit_ids[slot], state.unit_levels[slot]
    doses = state.potion_doses.at[slot, 0].set(1)
    bonus = env.potion_rules.rebuild(env._potion_intrinsic(uid[None], level[None]), doses[slot][None])[0]
    s = state.replace(potion_doses=doses, potion_bonus=state.potion_bonus.at[slot].set(bonus),
                      unit_xp=state.unit_xp.at[slot].set(7))
    return s.replace(hp=s.hp.at[slot].set(int(env.max_hp(s)[slot])-3)), slot, bonus


def test_transfers_carry_the_unit_both_ways(wild):
    env, state = wild
    g, cap = env.garrisons, env.garrisons.capital_index
    s, slot, bonus = potion_soldier(env, state)
    full = int(env.max_hp(s)[slot])
    uid = int(s.unit_ids[slot])
    legal = compiled_method(env, 'action_mask')(s)
    leader = [x for x in range(6) if bool(env.progression.heroes[s.unit_ids[x]])]
    assert leader and not any(bool(legal[g.store_start+x]) for x in leader)
    assert not bool(legal[g.take_start+GUARDIAN_SLOT])
    stored = step(env, s, g.store_start+slot)
    assert int(stored.last_event) == UNIT_STORED and int(stored.unit_ids[slot]) == 0
    assert int(stored.garrison_ids[cap, slot]) == uid  # its own cell was free
    assert int(stored.garrison_hp[cap, slot]) == full-3 and int(stored.garrison_xp[cap, slot]) == 7
    assert stored.garrison_doses[cap, slot].tolist()[0] == 1 and int(g.garrison_max_hp(stored)[cap, slot]) == full
    assert int(stored.potion_doses[slot].sum()) == 0
    back = step(env, stored, g.take_start+slot)
    assert int(back.last_event) == UNIT_TAKEN and int(back.garrison_ids[cap, slot]) == 0
    assert int(back.unit_ids[slot]) == uid and int(back.hp[slot]) == full-3 and int(back.unit_xp[slot]) == 7
    assert bool(jnp.allclose(back.potion_bonus[slot], bonus)) and int(env.max_hp(back)[slot]) == full
    # The Cultist's own rear-middle cell holds the guardian: the first free rear cell is taken.
    cultist = next(x for x in range(3, 6) if int(s.unit_ids[x]) and x == GUARDIAN_SLOT)
    assert int(g.store_targets(s)[cultist]) == 3
    # Away from the capital nothing moves.
    away = s.replace(position=jnp.array([12, 20], jnp.int32))
    assert not bool(jnp.any(g.available(away)[g.store_start-g.start:]))


def test_capital_keeps_its_dead_and_the_temple_revives_them(wild):
    env, state = wild
    g, cap = env.garrisons, env.garrisons.capital_index
    s, slot, _ = potion_soldier(env, state)
    stored = step(env, s.replace(hp=s.hp.at[slot].set(0)), g.store_start+slot)
    assert int(stored.garrison_ids[cap, slot]) != 0 and int(stored.garrison_hp[cap, slot]) == 0
    rested = step(env, stored, REST)
    assert int(rested.garrison_hp[cap, slot]) == 0
    revive = g.hire_end+slot
    assert not bool(compiled_method(env, 'action_mask')(rested)[revive])  # no Temple yet
    ready = rested.replace(buildings=rested.buildings | env.capital.temple_bit, gold=jnp.int32(5000))
    away = ready.replace(position=jnp.array([12, 20], jnp.int32))
    assert bool(compiled_method(env, 'action_mask')(away)[revive])  # remote
    price = int(g.revive_price(away)[slot])
    raised = step(env, away, revive)
    assert int(raised.last_event) == GARRISON_REVIVED and int(raised.garrison_hp[cap, slot]) == 1
    assert int(raised.gold) == 5000-price


def test_temple_heals_wounded_capital_defenders_only(wild):
    env, state = wild
    g, cap = env.garrisons, env.garrisons.capital_index
    mask = compiled_method(env, 'action_mask')
    temple = state.buildings | env.capital.temple_bit
    heal = g.heal_start+GUARDIAN_SLOT
    full = state.replace(buildings=temple, gold=jnp.int32(5000))
    assert not bool(mask(full)[heal])  # Ashkael is unhurt
    hurt = full.replace(garrison_hp=full.garrison_hp.at[cap, GUARDIAN_SLOT].set(100))
    assert not bool(mask(hurt.replace(buildings=state.buildings))[heal])  # no Temple
    assert bool(mask(hurt)[heal]) and not bool(mask(hurt.replace(gold=jnp.int32(0)))[heal])
    price = int(g.heal_price(hurt)[GUARDIAN_SLOT])
    healed = step(env, hurt, heal)
    assert int(healed.garrison_hp[cap, GUARDIAN_SLOT]) == 190 and int(healed.gold) == 5000-90*price
    poor = step(env, hurt.replace(gold=jnp.int32(10*price+price-1)), heal)  # whole HP only
    assert int(poor.garrison_hp[cap, GUARDIAN_SLOT]) == 110 and int(poor.gold) == price-1
    dead = hurt.replace(garrison_hp=hurt.garrison_hp.at[cap, GUARDIAN_SLOT].set(0))
    assert not bool(mask(dead)[heal]) and bool(mask(dead)[g.hire_end+GUARDIAN_SLOT])  # revive instead


def test_capital_garrison_swaps_three_a_day_around_the_guardian(wild):
    env, state = wild
    g, cap = env.garrisons, env.garrisons.capital_index
    mask = compiled_method(env, 'action_mask')
    s, slot, _ = potion_soldier(env, state)
    s = step(env, s, g.store_start+slot)  # the Possessed to capital cell 0, with its potion
    assert all(GUARDIAN_SLOT not in pair for pair in SWAP_PAIRS)
    uid, doses = int(s.garrison_ids[cap, 0]), s.garrison_doses[cap, 0].tolist()
    legal = mask(s)[g.swap_start:g.end]
    assert legal.tolist() == [0 in pair for pair in SWAP_PAIRS]  # only pairs touching the unit
    moves = [SWAP_PAIRS.index(p) for p in ((0, 2), (2, 5), (3, 5))]
    for count, k in enumerate(moves, 1):
        s = step(env, s, g.swap_start+k)
        assert int(s.last_event) == GARRISON_SWAPPED and int(s.garrison_swaps_today) == count
    assert int(s.garrison_ids[cap, 3]) == uid and s.garrison_doses[cap, 3].tolist() == doses
    assert int(s.garrison_ids[cap, GUARDIAN_SLOT]) == g.guardian
    assert not bool(jnp.any(mask(s)[g.swap_start:g.end]))  # three a day
    rested = step(env, s, REST)
    assert int(rested.garrison_swaps_today) == 0 and int(rested.garrison_swaps_used) == 0
    assert int(SWAP_LIMIT) == 3 and bool(mask(rested)[g.swap_start+SWAP_PAIRS.index((0, 3))])


def test_capital_battle_keeps_dead_and_returns_party_potions(wild):
    env, state = wild
    g, cap = env.garrisons, env.garrisons.capital_index
    s, slot, _ = potion_soldier(env, state)
    stored = step(env, s.replace(hp=s.hp.at[slot].set(0)), g.store_start+slot)
    goblins = env.map_config['reference_enemy_ids'].index('S143KC0033')
    marked = stored.potion_doses.at[1, 1].set(1)  # the Duke's permanent damage potion
    away = stored.replace(position=jnp.array([12, 20], jnp.int32), potion_doses=marked,
                          enemy_positions=stored.enemy_positions.at[goblins].set(g.positions[cap]))
    battle = step(env, away, REST)
    assert bool(battle.garrison_battle) and int(battle.potion_doses[1].sum()) == 0
    assert int(battle.unit_ids[slot]) == int(stored.garrison_ids[cap, slot]) and int(battle.hp[slot]) == 0
    after = fight(env, battle)
    assert not bool(after.in_battle) and int(after.potion_doses[1, 1]) == 1
    assert int(after.garrison_ids[cap, slot]) != 0 and int(after.garrison_hp[cap, slot]) == 0
