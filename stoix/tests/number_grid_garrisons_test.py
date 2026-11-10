"""City and capital garrisons (docs/WORLD.md#garrisons) on city_defence_train; CUDA, by request only."""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, DEFEND, REST, SHOOT, NumberGrid, action_names, load_map
from stoix.envs.number_grid_garrisons import (CITY_LOST, GARRISON_HEALED, GARRISON_HIRED, GARRISON_REVIVED,
                                              GARRISON_SELECTED, GARRISON_SWAPPED,
                                              GUARDIANS, GUARDIAN_SLOT, SWAP_LIMIT, SWAP_PAIRS, UNIT_STORED, UNIT_TAKEN)
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


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
    g = env.garrisons
    if int(g.selected_site(state)) != site:
        state = step(env, state, g.select_start+site)
    return step(env, state, g.start+offer)


@pytest.mark.parametrize('name', ['default', 'the_charmed_child', 'mirror_match', 'trade_train'])
def test_every_campaign_capital_has_its_faction_guardian(name):
    game_map = load_map(name)
    env = NumberGrid(map_config=game_map)
    g = env.garrisons
    assert g.enabled and env.num_actions == len(action_names(game_map)) == g.end
    row = env.progression.rows[g.guardian]
    assert row['game_id'] == GUARDIANS[env.construction.faction] and row['max_hp'] == 900
    assert int(g.initial_ids[g.capital_index, GUARDIAN_SLOT]) == g.guardian
    assert int(jnp.sum(g.initial_ids != 0)) == 1 and g.hire_end-g.start == 5
    assert action_names(game_map)[g.start:g.hire_end] == tuple(f'garrison_hire_{k}' for k in range(5))
    assert g.action_count == 5+2*g.revive_count+12+len(SWAP_PAIRS)+g.count
    assert g.select_start == g.swap_start+len(SWAP_PAIRS) and g.end == g.select_start+g.count
    assert g.swap_start == g.take_start+6 == g.store_start+12
    assert g.heal_start == g.hire_end+g.revive_count and g.store_start == g.heal_start+g.revive_count
    assert len(SWAP_PAIRS) == 15


def test_start_owned_empty_city_and_capital_guardian(defence):
    env, state = defence
    g = env.garrisons
    assert state.city_owned.tolist() == [True] and g.defended == 0 and int(state.gold) == 1000
    assert state.garrison_ids[0].tolist() == [0]*6
    capital = env.progression.rows[int(state.garrison_ids[g.capital_index, GUARDIAN_SLOT])]
    assert capital['name'] == 'Мизраэль' and int(state.garrison_hp[g.capital_index, GUARDIAN_SLOT]) == 900
    assert not bool(jnp.any(state.alive))  # both assaults appear later
    assert int(state.selected_garrison) == g.capital_index


def test_garrison_selection_actions_observation_and_loss(defence):
    env, state = defence
    g = env.garrisons
    names = action_names(env.map_config)
    mask = compiled_method(env, 'action_mask')
    assert names[g.select_start:g.end] == ('select_city_0', 'select_capital')
    assert int(jnp.sum(mask(state)[g.select_start:g.end])) == int(jnp.sum(state.city_owned))
    assert g.observation(state)[-g.count:].tolist() == [0., 1.]
    selected = step(env, state, g.select_start)
    assert int(selected.last_event) == GARRISON_SELECTED and int(selected.selected_garrison) == 0
    assert g.observation(selected)[-g.count:].tolist() == [1., 0.]
    assert g.observation(selected).shape == (g.observation_size,)
    assert env.observation(selected).shape == (env.observation_size,)
    assert selected.position.tolist() == state.position.tolist()
    assert int(selected.gold) == int(state.gold) and int(selected.day) == int(state.day)
    assert int(selected.movement_points) == int(state.movement_points)
    moved = step(env, selected, 2)
    assert moved.position.tolist() != selected.position.tolist() and int(moved.selected_garrison) == 0
    assert int(g.selected_site(moved)) == 0
    lost = g.lose(selected, jnp.arange(g.count) == 0)
    assert int(lost.selected_garrison) == g.capital_index
    assert g.observation(lost)[-g.count:].tolist() == [0., 1.]
    unowned = state.replace(city_owned=jnp.zeros_like(state.city_owned))
    assert not bool(mask(unowned)[g.select_start])
    unchanged = step(env, unowned, g.select_start)
    assert int(unchanged.selected_garrison) == g.capital_index
    for blocked in (state.replace(done=jnp.bool_(True)), state.replace(in_battle=jnp.bool_(True)),
                    state.replace(alive=state.alive.at[0].set(True),
                                  garrison_pending=state.garrison_pending.at[0].set(True))):
        assert not bool(jnp.any(g.available(blocked)[g.select_start-g.start:]))


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
    assert int(g.targets(s)[g.capital_index, 4]) == 0 and not bool(g.available(s)[4])
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


def test_five_shared_hire_actions_use_selection_independently_of_party_position(defence):
    env, state = defence
    g, cap = env.garrisons, env.garrisons.capital_index
    mask = compiled_method(env, 'action_mask')
    action = g.start
    assert g.hire_end-g.start == 5 and g.metadata['hire_count'] == 5
    assert env.num_actions == len(action_names(env.map_config)) == g.end
    # The hero stands in the city, but the capital is selected initially.
    city_position = state.replace(position=g.positions[0])
    home = step(env, city_position, action)
    assert int(home.last_event) == GARRISON_HIRED and int(home.garrison_ids[cap, 0]) == int(g.offer_ids[0])
    assert not bool(jnp.any(home.garrison_ids[0]))
    assert bool(jnp.array_equal(home.unit_ids, city_position.unit_ids))
    # Selecting the city redirects exactly the same action while the hero is elsewhere.
    selected = step(env, home.replace(position=g.positions[cap]), g.select_start)
    city = step(env, selected, action)
    assert int(city.garrison_ids[0, 0]) == int(g.offer_ids[0]) and int(city.selected_garrison) == 0
    assert bool(jnp.array_equal(city.garrison_ids[cap], home.garrison_ids[cap]))
    assert int(city.gold) == int(selected.gold)-int(g.offer_costs[0])
    assert int(city.day) == int(selected.day) and int(city.movement_points) == int(selected.movement_points)
    assert not bool(mask(selected.replace(gold=jnp.int32(int(g.offer_costs[0])-1)))[action])
    for blocked in (selected.replace(done=jnp.bool_(True)), selected.replace(in_battle=jnp.bool_(True)),
                    selected.replace(alive=selected.alive.at[0].set(True),
                                     garrison_pending=selected.garrison_pending.at[0].set(True))):
        assert not bool(jnp.any(g.available(blocked)[:5]))
    # A full selected city masks hiring even when the capital still has room.
    full = city.replace(city_levels=city.city_levels.at[0].set(1))
    assert not bool(jnp.any(mask(full)[g.start:g.hire_end]))
    capital = step(env, full, g.select_start+cap)
    assert bool(mask(capital)[action])
    hired = step(env, capital, action)
    assert int(jnp.sum(hired.garrison_ids[cap] != 0)) == int(jnp.sum(capital.garrison_ids[cap] != 0))+1
    assert bool(jnp.array_equal(hired.garrison_ids[0], full.garrison_ids[0]))


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
    """The soldier in party slot 0 with a permanent HP potion and a little experience."""
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
    # Away from owned settlements nothing moves.
    away = s.replace(position=jnp.array([12, 20], jnp.int32))
    assert not bool(jnp.any(g.available(away)[g.store_start-g.start:g.swap_start-g.start]))


@pytest.mark.parametrize('dead', [False, True])
def test_city_transfers_reuse_actions_and_preserve_unit(defence, dead):
    env, state = defence
    g, city, slot = env.garrisons, 0, 0
    s = state.replace(position=g.positions[city],
                      unit_levels=state.unit_levels.at[slot].set(3),
                      native_levels=state.native_levels.at[slot].set(3),
                      mercenary_units=state.mercenary_units.at[slot].set(True))
    s, _, bonus = potion_soldier(env, s)
    if dead:
        s = s.replace(hp=s.hp.at[slot].set(0))
    uid, hp = int(s.unit_ids[slot]), int(s.hp[slot])
    names = action_names(env.map_config)
    assert names[g.store_start+slot] == 'party_to_capital_0'
    assert names[g.take_start+slot] == 'capital_to_party_0'
    assert g.take_start-g.store_start == g.swap_start-g.take_start == 6
    assert env.num_actions == len(names) == g.end
    leader = next(c for c in range(6) if bool(env.progression.heroes[s.unit_ids[c]]))
    assert not bool(compiled_method(env, 'action_mask')(s)[g.store_start+leader])
    stored = step(env, s, g.store_start+slot)
    assert int(stored.last_event) == UNIT_STORED and int(stored.unit_ids[slot]) == 0
    assert int(stored.garrison_ids[city, slot]) == uid
    assert int(stored.garrison_levels[city, slot]) == 3 and int(stored.garrison_xp[city, slot]) == 7
    assert int(stored.garrison_hp[city, slot]) == hp and bool(stored.garrison_mercenary[city, slot])
    assert stored.garrison_doses[city, slot].tolist() == s.potion_doses[slot].tolist()
    assert bool(jnp.array_equal(stored.garrison_ids[g.capital_index], s.garrison_ids[g.capital_index]))
    sites = g.snapshot(stored)
    assert sites[city]['at_party'] and not sites[g.capital_index]['at_party']
    unit = next(u for u in sites[city]['units'] if u['cell'] == slot)
    assert unit['take'] and unit['take_slot'] == slot
    assert not any(sites[g.capital_index]['store'])
    back = step(env, stored, g.take_start+slot)
    assert int(back.last_event) == UNIT_TAKEN and int(back.garrison_ids[city, slot]) == 0
    assert int(back.unit_ids[slot]) == uid and int(back.unit_levels[slot]) == 3
    assert int(back.hp[slot]) == hp and int(back.unit_xp[slot]) == 7 and bool(back.mercenary_units[slot])
    assert bool(jnp.allclose(back.potion_bonus[slot], bonus))
    assert int(back.day) == int(s.day) and int(back.gold) == int(s.gold)
    assert int(back.movement_points) == int(s.movement_points)


def test_city_transfers_require_ownership_and_select_only_the_local_garrison(defence):
    env, state = defence
    g = env.garrisons
    # A city soldier in rear-middle cell 4 is not the capital guardian.
    s = hire(env, state, 0, 2)
    s = hire(env, s, 0, 2)
    clear = (jnp.arange(12) < 6) & ~env.progression.heroes[s.unit_ids]
    s = env._clear_roster_slots(s, clear).replace(position=g.positions[0])
    mask = compiled_method(env, 'action_mask')
    assert bool(mask(s)[g.take_start+4])
    taken = step(env, s, g.take_start+4)
    assert int(taken.last_event) == UNIT_TAKEN and int(taken.garrison_ids[0, 4]) == 0
    assert int(taken.garrison_ids[g.capital_index, GUARDIAN_SLOT]) == g.guardian
    unit = next(u for u in g.snapshot(s)[0]['units'] if u['cell'] == 4)
    assert unit['take']
    # Neither foreign city guards nor capital reserves can be taken from another tile.
    capital = s.replace(position=g.positions[g.capital_index])
    foreign = s.replace(city_owned=jnp.zeros_like(s.city_owned))
    outside = s.replace(position=jnp.array([10, 10], jnp.int32))
    for blocked in (capital, foreign, outside):
        assert not bool(jnp.any(mask(blocked)[g.store_start:g.swap_start]))
        unchanged = step(env, blocked, g.take_start+4)
        assert bool(jnp.array_equal(unchanged.garrison_ids, blocked.garrison_ids))
        assert bool(jnp.array_equal(unchanged.unit_ids, blocked.unit_ids))
    # Lack of leadership also blocks taking a city recruit when a physical slot is free.
    assert not bool(mask(state.replace(position=g.positions[0],
        garrison_ids=s.garrison_ids, garrison_levels=s.garrison_levels,
        garrison_hp=s.garrison_hp))[g.take_start+4])


def test_city_transfer_capacity_counts_large_units_and_free_columns(defence):
    env, state = defence
    g = env.garrisons
    mask = compiled_method(env, 'action_mask')
    city = state.replace(position=g.positions[0], city_levels=state.city_levels.at[0].set(1))
    full = step(env, city, g.store_start)
    assert int(full.garrison_ids[0, 0]) == int(city.unit_ids[0])
    assert not bool(jnp.any(mask(full)[g.store_start:g.take_start]))
    # Even a free column cannot fit a two-cell unit into a level-1 city.
    hero = env.progression.rows[int(state.unit_ids[1])]['key']
    big = hero_roster_state(env, city, ['titan', hero, None, None, None, None])
    assert not bool(mask(big)[g.store_start])
    room = big.replace(city_levels=big.city_levels.at[0].set(2))
    assert bool(mask(room)[g.store_start])
    stored = step(env, room, g.store_start)
    assert int(jnp.sum(g.occupied(stored.garrison_ids)[0])) == 2
    assert not bool(jnp.any(mask(stored)[g.store_start:g.take_start]))
    back = step(env, stored, g.take_start)
    assert int(back.unit_ids[0]) == env.progression.ids['titan'] and int(back.unit_ids[3]) == 0
    # Three rear guards leave enough level-5 capacity, but no whole column for a Titan.
    blocked = room.replace(city_levels=room.city_levels.at[0].set(5))
    for _ in range(3):
        blocked = hire(env, blocked, 0, 2)
    assert int(jnp.sum(g.occupied(blocked.garrison_ids)[0])) == 3
    assert not bool(mask(blocked)[g.store_start])


def test_transfers_use_party_location_independently_of_selected_garrison(defence):
    env, state = defence
    g, cap = env.garrisons, env.garrisons.capital_index
    selected = step(env, state, g.select_start)  # city selected, hero at the capital
    home = step(env, selected, g.store_start)
    assert int(home.garrison_ids[cap, 0]) == int(state.unit_ids[0])
    assert not bool(jnp.any(home.garrison_ids[0])) and int(home.selected_garrison) == 0
    capital = step(env, home, g.select_start+cap)
    city = capital.replace(position=g.positions[0])  # capital selected, hero in the city
    stored = step(env, city, g.store_start+2)
    assert int(stored.garrison_ids[0, 2]) == int(city.unit_ids[2])
    assert bool(jnp.array_equal(stored.garrison_ids[cap], city.garrison_ids[cap]))
    sites = g.snapshot(stored)
    assert sites[0]['at_party'] and not sites[0]['selected']
    assert sites[cap]['selected'] and not sites[cap]['at_party']
    assert next(u for u in sites[0]['units'] if u['cell'] == 2)['take']
    assert not next(u for u in sites[cap]['units'] if u['cell'] == 0)['take']
    back = step(env, stored, g.take_start+2)
    assert int(back.unit_ids[2]) == int(city.unit_ids[2]) and int(back.garrison_ids[0, 2]) == 0
    assert int(back.selected_garrison) == cap
    outside = back.replace(position=jnp.array([10, 10], jnp.int32))
    assert not bool(jnp.any(compiled_method(env, 'action_mask')(outside)[g.store_start:g.swap_start]))


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


def test_temple_heals_wounded_capital_defenders(wild):
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


def test_city_garrison_recovery_reuses_actions_and_requires_temple_and_gold(defence):
    env, state = defence
    g, city, living, dead = env.garrisons, 0, 3, 4
    s = hire(env, hire(env, state, city, 2), city, 2)
    assert int(s.selected_garrison) == city  # hiring selected it remotely; hero is at the capital
    s = s.replace(garrison_hp=s.garrison_hp.at[city, living].set(1).at[city, dead].set(0)
                    .at[g.capital_index, GUARDIAN_SLOT].set(100))
    heal, revive = g.heal_start+living, g.hire_end+dead
    mask = compiled_method(env, 'action_mask')
    names = action_names(env.map_config)
    assert names[heal] == 'garrison_capital_heal_3'
    assert names[revive] == 'garrison_capital_revive_4'
    assert g.heal_start-g.hire_end == g.store_start-g.heal_start == 6
    assert env.num_actions == len(names) == g.end
    assert not bool(mask(s)[heal] | mask(s)[revive])  # no Temple
    ready = s.replace(buildings=s.buildings | env.capital.temple_bit, gold=jnp.int32(5000))
    assert bool(mask(ready)[heal] & mask(ready)[revive])
    poor = ready.replace(gold=jnp.int32(0))
    assert not bool(mask(poor)[heal] | mask(poor)[revive])
    assert not bool(mask(ready)[g.heal_start+dead] | mask(ready)[g.hire_end+living])
    assert not bool(mask(ready)[g.heal_start] | mask(ready)[g.hire_end])  # empty city cell
    prices = g.snapshot(ready)
    assert prices[city]['recovery'] and not prices[g.capital_index]['recovery']
    units = {u['cell']: u for u in prices[city]['units']}
    assert units[living]['heal'] and units[dead]['revive']
    assert not any(u['heal'] or u['revive'] for u in prices[g.capital_index]['units'])
    price = int(g.heal_price(ready)[living])
    healed = step(env, ready.replace(gold=jnp.int32(10*price+price-1)), heal)
    assert int(healed.last_event) == GARRISON_HEALED and int(healed.garrison_hp[city, living]) == 11
    assert int(healed.gold) == price-1 and int(healed.last_service_cost) == 10*price
    assert int(healed.garrison_hp[city, dead]) == 0
    revive_price = int(g.revive_price(ready)[dead])
    assert not bool(mask(ready.replace(gold=jnp.int32(revive_price-1)))[revive])
    raised = step(env, ready, revive)
    assert int(raised.last_event) == GARRISON_REVIVED and int(raised.garrison_hp[city, dead]) == 1
    assert int(raised.gold) == 5000-revive_price
    assert int(raised.garrison_hp[g.capital_index, GUARDIAN_SLOT]) == 100
    assert bool(jnp.array_equal(raised.unit_ids, ready.unit_ids))
    assert bool(jnp.array_equal(raised.recovery_balance, ready.recovery_balance))
    assert int(raised.day) == int(ready.day) and int(raised.movement_points) == int(ready.movement_points)
    # Moving does not switch recovery away from the selected city.
    elsewhere = ready.replace(position=jnp.array([10, 10], jnp.int32))
    assert int(g.selected_site(elsewhere)) == city and bool(mask(elsewhere)[revive])
    # Selecting the capital while standing in the city changes only the recovery target.
    capital = step(env, ready.replace(position=g.positions[city]), g.select_start+g.capital_index)
    assert bool(mask(capital)[g.heal_start+GUARDIAN_SLOT]) and not bool(mask(capital)[revive])
    restored = step(env, capital, g.heal_start+GUARDIAN_SLOT)
    assert bool(jnp.array_equal(restored.garrison_hp[city], capital.garrison_hp[city]))
    rate = int(g.heal_price(capital)[GUARDIAN_SLOT])
    amount = min(g.guardian_max_hp-100, int(capital.gold)//rate)
    assert int(restored.garrison_hp[g.capital_index, GUARDIAN_SLOT]) == 100+amount
    assert int(restored.gold) == int(capital.gold)-amount*rate


def test_city_battle_keeps_dead_for_revival_and_city_loss_clears_them(defence):
    env, state = defence
    g, city, cell = env.garrisons, 0, 3
    s = hire(env, hire(env, state, city, 0), city, 2)
    s = s.replace(alive=s.alive.at[0].set(True), enemy_positions=s.enemy_positions.at[0].set(g.positions[city]),
                  garrison_pending=s.garrison_pending.at[0].set(True),
                  garrison_doses=s.garrison_doses.at[city, cell, 0].set(1),
                  garrison_mercenary=s.garrison_mercenary.at[city, cell].set(True))
    battle, started = jax.jit(g.engage)(s, jnp.bool_(True))
    assert bool(started) and int(battle.garrison_site) == city
    after = battle.replace(hp=battle.hp.at[cell].set(0), unit_xp=battle.unit_xp.at[cell].set(7))
    kept = jax.jit(g.finish)(battle, after, jnp.bool_(True), jnp.bool_(False))
    assert bool(kept.city_owned[city]) and not bool(kept.garrison_battle)
    assert int(kept.garrison_ids[city, cell]) == int(s.garrison_ids[city, cell])
    assert int(kept.garrison_hp[city, cell]) == 0 and int(kept.garrison_xp[city, cell]) == 7
    assert int(kept.garrison_doses[city, cell, 0]) == 1 and bool(kept.garrison_mercenary[city, cell])
    rested = g.regenerate(kept, jnp.bool_(True))
    assert int(rested.garrison_hp[city, cell]) == 0  # REST does not resurrect
    ready = kept.replace(selected_garrison=jnp.int32(city), buildings=kept.buildings | env.capital.temple_bit,
                         gold=jnp.int32(5000))
    raised = jax.jit(lambda world: g.apply(env, world, jnp.int32(g.hire_end+cell), jnp.bool_(False))[0])(ready)
    assert int(raised.garrison_hp[city, cell]) == 1 and int(raised.garrison_xp[city, cell]) == 7
    assert int(raised.garrison_doses[city, cell, 0]) == 1 and bool(raised.garrison_mercenary[city, cell])
    fallen = jax.jit(g.finish)(battle, after, jnp.bool_(True), jnp.bool_(True))
    assert not bool(fallen.city_owned[city]) and not bool(jnp.any(fallen.garrison_ids[city]))
    assert not bool(jnp.any(fallen.garrison_doses[city]))


def test_capital_garrison_swaps_three_a_day_around_the_guardian(wild):
    env, state = wild
    g, cap = env.garrisons, env.garrisons.capital_index
    mask = compiled_method(env, 'action_mask')
    s, slot, _ = potion_soldier(env, state)
    s = step(env, s, g.store_start+slot)  # the Possessed to capital cell 0, with its potion
    assert all(GUARDIAN_SLOT not in pair for pair in SWAP_PAIRS[:10])
    uid, doses = int(s.garrison_ids[cap, 0]), s.garrison_doses[cap, 0].tolist()
    legal = mask(s)[g.swap_start:g.select_start]
    assert legal.tolist() == [0 in pair and GUARDIAN_SLOT not in pair for pair in SWAP_PAIRS]
    moves = [SWAP_PAIRS.index(p) for p in ((0, 2), (2, 5), (3, 5))]
    for count, k in enumerate(moves, 1):
        s = step(env, s, g.swap_start+k)
        assert int(s.last_event) == GARRISON_SWAPPED and int(s.garrison_swaps_today[cap]) == count
    assert int(s.garrison_ids[cap, 3]) == uid and s.garrison_doses[cap, 3].tolist() == doses
    assert int(s.garrison_ids[cap, GUARDIAN_SLOT]) == g.guardian
    assert not bool(jnp.any(mask(s)[g.swap_start:g.select_start]))  # three a day
    rested = step(env, s, REST)
    assert not bool(jnp.any(rested.garrison_swaps_today)) and not bool(jnp.any(rested.garrison_swaps_used))
    assert int(SWAP_LIMIT) == 3 and bool(mask(rested)[g.swap_start+SWAP_PAIRS.index((0, 3))])


def test_selected_city_swaps_all_cells_and_keeps_separate_daily_limits(defence):
    env, state = defence
    g, cap = env.garrisons, env.garrisons.capital_index
    mask = compiled_method(env, 'action_mask')
    s = hire(env, hire(env, state, 0, 2), 0, 2)
    s = step(env, s, g.store_start)  # one capital reserve too
    s = s.replace(garrison_hp=s.garrison_hp.at[0, 4].set(0), garrison_xp=s.garrison_xp.at[0, 4].set(7),
                  garrison_doses=s.garrison_doses.at[0, 4, 0].set(1),
                  garrison_mercenary=s.garrison_mercenary.at[0, 4].set(True))
    assert int(s.selected_garrison) == 0
    swap = g.swap_start+SWAP_PAIRS.index((0, 4))
    assert bool(mask(s)[swap])  # cell 4 is an ordinary city cell
    moved = step(env, s, swap)
    assert int(moved.garrison_ids[0, 0]) == int(s.garrison_ids[0, 4])
    assert int(moved.garrison_hp[0, 0]) == 0 and int(moved.garrison_xp[0, 0]) == 7
    assert int(moved.garrison_doses[0, 0, 0]) == 1 and bool(moved.garrison_mercenary[0, 0])
    assert moved.garrison_swaps_today.tolist() == [1, 0]
    capital = step(env, moved, g.select_start+cap)
    assert not bool(mask(capital)[swap])  # the capital's guardian still cannot move
    capital = step(env, capital, g.swap_start+SWAP_PAIRS.index((0, 2)))
    assert capital.garrison_swaps_today.tolist() == [1, 1]
    city = step(env, capital, g.select_start)
    assert not bool(mask(city)[swap])  # selection does not reset used pairs
    for pair in ((0, 2), (2, 5)):
        city = step(env, city, g.swap_start+SWAP_PAIRS.index(pair))
    assert city.garrison_swaps_today.tolist() == [3, 1]
    assert not bool(jnp.any(mask(city)[g.swap_start:g.select_start]))
    capital = step(env, city, g.select_start+cap)
    assert bool(mask(capital)[g.swap_start+SWAP_PAIRS.index((2, 5))])
    rested = step(env, capital, REST)
    assert not bool(jnp.any(rested.garrison_swaps_today)) and not bool(jnp.any(rested.garrison_swaps_used))
    assert int(rested.selected_garrison) == cap


def test_city_big_unit_swaps_whole_column_without_capital_guardian_restrictions(defence):
    env, state = defence
    g, cap = env.garrisons, env.garrisons.capital_index
    building = jnp.uint32(1 << int(g.offer_required[4]))
    rich = state.replace(buildings=state.buildings | building, gold=jnp.int32(99999))
    s = hire(env, hire(env, rich, 0, 4), 0, 2)  # Titan in 0, Archer in 4
    assert int(s.selected_garrison) == 0
    swap = g.swap_start+SWAP_PAIRS.index((0, 1))
    assert bool(compiled_method(env, 'action_mask')(s)[swap])
    moved = step(env, s, swap)
    assert int(moved.garrison_ids[0, 1]) == int(g.offer_ids[4])
    assert int(moved.garrison_ids[0, 3]) == int(g.offer_ids[2])
    assert int(moved.garrison_ids[0, 0]) == int(moved.garrison_ids[0, 4]) == 0
    assert bool(jnp.array_equal(moved.garrison_ids[cap], s.garrison_ids[cap]))


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
