"""Terrain, live flight and dead leader travel: CUDA JIT/vmap regressions."""
import chex
import jax
import jax.numpy as jnp
import pytest
from stoix.envs.number_grid import MAP, REST
from stoix.envs.number_grid_items import ItemRules
from stoix.envs.number_grid_terrain import TerrainRules
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


def test_lake_connected_empty_and_terrain_validated(current_game):
    env, _, _, _ = current_game
    water = {tuple(p) for p in MAP['terrain']['water']}
    assert len(water) == 50
    seen, pending = {next(iter(water))}, [next(iter(water))]
    while pending:
        r,c = pending.pop()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            p = (r+dr,c+dc)
            if p in water and p not in seen:
                seen.add(p)
                pending.append(p)
    assert seen == water
    assert not water & set(env.territory.obstacle_cells)
    objects = {tuple(MAP['agent_position']), *map(tuple, MAP['opponent_positions'])}
    objects.update(tuple(o['position']) for kind in ('chests','cities','mines') for o in MAP[kind])
    assert not water & objects
    for terrain in ({'bog':[[2,3]]}, {'water':[MAP['agent_position']]},
                    {'forest':[[0,3]]}, {'road':[[2,3]],'forest':[[2,3]]},
                    {'water':[[8,3]]}):
        with pytest.raises(ValueError):
            TerrainRules(dict(MAP,terrain=terrain),env.progression)


def test_living_flying_heroes_and_dead_leader_prices(current_game):
    """Keep explicit leader cases so an omitted flyer cannot disappear from coverage."""
    env, initial, _, _ = current_game
    cases, expected = [], []
    # Installed GMabi.dbf: Duke, Counselor and Dryad share Pegasus Knight's 0/1/3.
    # King's Guard (g000uu0044) remains the explicit walking control.
    keys = ['duke', 'g000uu0071', 'g000uu8010', 'g000uu0019', 'g000uu0022',
            'g000uu0072', 'g000uu0096', 'g000uu0099', 'g000uu0044']
    for key in keys:
        living = hero_roster_state(env,initial,['possessed',key,None,'cultist',None,None])
        cases += [living,living.replace(hp=living.hp.at[1].set(0))]
        expected += [[2,1,4,6] if key=='g000uu0044' else [2]*4,[4,2,8,12]]
    batch = jax.tree.map(lambda *xs:jnp.stack(xs),*cases)
    chex.assert_trees_all_equal(compiled_method(env,'travel_costs',batched=True)(batch),jnp.array(expected))
    fake = cases[1].replace(unit_ids=cases[1].unit_ids.at[3].set(env.progression.ids['g000uu0022']),
        copied=cases[1].copied.at[3].set(True))
    assert compiled_method(env,'travel_costs')(fake).tolist() == [4,2,8,12]
    no_hero = initial.replace(unit_ids=initial.unit_ids.at[1].set(0))
    assert compiled_method(env,'travel_costs')(no_hero).tolist() == [2,1,4,6]


@pytest.mark.parametrize('leader_key,flying,compare_batch', [
    pytest.param('duke', True, False, id='scalar', marks=pytest.mark.core),
    pytest.param('g000uu0096', True, False, id='death-knight', marks=pytest.mark.core),
    pytest.param('g000uu0071', True, False, id='counselor'),
    pytest.param('g000uu8010', True, False, id='dryad'),
    pytest.param('g000uu0044', False, False, id='walking'),
    pytest.param('duke', True, True, id='full-batch'),
])
def test_actual_step_mask_quotes_and_last_partial_move(current_game, leader_key, flying, compare_batch):
    """Check terrain costs and partial final moves against masks, quotes, and actual transitions."""
    env, initial, _, _ = current_game
    initial = hero_roster_state(env,initial,['possessed',leader_key,'possessed','cultist',None,None])
    cases, actions, expected, destinations = [], [], [], []
    for position, action, cost in (([2,2],2,1),([2,3],2,2),([4,16],3,4),([12,4],4,6)):
        for dead in (False,True):
            for points in (20,1,0):
                s = initial.replace(position=jnp.array(position),movement_points=jnp.int32(points))
                if dead:
                    s = s.replace(hp=s.hp.at[1].set(0))
                cases.append(s)
                actions.append(action)
                price = cost*2 if dead else (2 if flying else cost)
                expected.append(min(points,price))
                destinations.append(jnp.array(position)+env.directions[action]*(points>0))
    if not compare_batch:
        # Reuse scalar programs. Stacking every leaf of 24 whole campaign states
        # only to test scalar moves compiled many unrelated padding kernels.
        step = compiled_method(env, 'step')
        mask = compiled_method(env, 'action_mask')
        quotes = compiled_method(env, 'map_commands')
        for state, action, spent, destination in zip(cases, actions, expected, destinations):
            assert bool(mask(state)[action]) == bool(state.movement_points > 0)
            assert quotes(state)[action, 1] == spent
            following, _ = step(state, jnp.int32(action))
            chex.assert_trees_all_equal(following.position, destination)
            assert following.movement_points == state.movement_points-spent
            assert following.day == state.day and not following.in_battle
        return
    states = jax.tree.map(lambda *xs:jnp.stack(xs),*cases)
    actions, expected = jnp.array(actions), jnp.array(expected)
    masks = compiled_method(env,'action_mask',batched=True)(states)
    quotes = compiled_method(env,'map_commands',batched=True)(states)
    chex.assert_trees_all_equal(masks[jnp.arange(len(cases)),actions],states.movement_points>0)
    chex.assert_trees_all_equal(quotes[jnp.arange(len(cases)),actions,1],expected)
    following,_ = compiled_method(env,'step',batched=True)(states,actions)
    chex.assert_trees_all_equal(following.position,jnp.stack(destinations))
    chex.assert_trees_all_equal(following.movement_points,states.movement_points-expected)
    chex.assert_trees_all_equal(following.day,states.day)
    assert not jnp.any(following.in_battle)


def test_terrain_boots_require_living_leader_and_flight_stays_two(current_game):
    env, initial, _, _ = current_game
    initial = hero_roster_state(env,initial,['possessed','g000uu0044',None,'cultist',None,None])
    items = ItemRules(dict(MAP,initial_items={'elven_boots':1,'elemental_boots':1}))
    costs = jax.jit(lambda s:env.terrain.costs(s,items))
    for key, expected in (('elven_boots',[2,1,2,6]),('elemental_boots',[2,1,4,2])):
        s = initial.replace(equipped=initial.equipped.at[4].set(items.keys.index(key)))
        assert costs(s).tolist() == expected
        assert costs(s.replace(hp=s.hp.at[1].set(0))).tolist() == [4,2,8,12]
        flyer = hero_roster_state(env,s,['possessed','duke',None,'cultist',None,None])
        assert costs(flyer).tolist() == [2]*4


@pytest.mark.parametrize('leader_key,equipped_cap', [
    pytest.param('duke', 24, id='duke'),
    pytest.param('g000uu0071', 42, id='counselor'),
    pytest.param('g000uu8010', 30, id='dryad'),
    pytest.param('g000uu0096', 24, id='g000uu0096'),
])
def test_death_revival_and_rest_restore_correct_movement(current_game, leader_key, equipped_cap):
    """Revival restores flight and boots (+20% of each leader's base allowance)."""
    env, initial, advance, _ = current_game
    initial = hero_roster_state(env,initial,['possessed',leader_key,'possessed','cultist',None,None])
    boot = env.item_rules.keys.index('boots_speed')
    equipped = compiled_method(env,'_refresh_equipment')(initial.replace(item_inventory=initial.item_inventory.at[0].set(boot)))
    dead = compiled_method(env,'_refresh_equipment')(equipped.replace(hp=equipped.hp.at[1].set(0)))
    assert env.movement_cap(dead) == 20 and dead.equipped[4] == -1
    assert compiled_method(env,'travel_costs')(dead).tolist() == [4,2,8,12]
    rested,_ = advance(dead,jnp.int32(REST))
    assert rested.movement_points == 20 and rested.hp[1] == 0
    revive = env.potion_rules.metadata()[env.potion_rules.keys.index('life')]['action_start']+1
    revived,_ = advance(rested,jnp.int32(revive))
    assert revived.hp[1] == 1 and revived.equipped[4] == boot
    assert env.movement_cap(revived) == equipped_cap
    assert compiled_method(env,'travel_costs')(revived).tolist() == [2]*4


@pytest.mark.core
def test_water_excludes_land_but_not_travel_and_flight_cannot_cross_obstacles(current_game):
    env, initial, _, _ = current_game
    flooded = initial.replace(territory_claims=jnp.full_like(initial.territory_claims,env.size**2))
    water = jnp.array(MAP['terrain']['water'])
    @jax.jit
    def check(s):
        return env.territory.owned(s,water),env.territory.blocked(water),env.territory.regeneration_percent(s.replace(position=water[0]))
    owned,blocked,regen = check(flooded)
    assert not jnp.any(owned) and not jnp.any(blocked)
    assert regen[1] == 20  # 5 native + 15 warrior lord, no +10 land.
    flyer = hero_roster_state(env,initial,['possessed','g000uu0019',None,'cultist',None,None])
    assert not compiled_method(env,'action_mask')(flyer)[0]  # Capital wall at [1,2].
    assert not compiled_method(env,'action_mask')(flyer.replace(position=jnp.array([6,5])))[2]


def test_observation_reset_and_human_quotes(current_game,human_service):
    env, initial, _, _ = current_game
    observe = compiled_method(env,'observation')
    assert observe(initial).shape == (1844,) and env.num_actions == 244
    research_context = 4+env.spell_research.observation_size+env.casting.observation_size+env.unit_swaps.observation_size  # lord, research, casts, swaps, day
    terrain_columns = slice(-35-research_context,-research_context)
    view = observe(initial)[terrain_columns]
    assert view[-2:].tolist() == [1.,0.]
    assert jnp.allclose(view[25:33],jnp.full(8,2/12))
    walking = hero_roster_state(env,initial,['possessed','g000uu0044',None,'cultist',None,None])
    walking_view = observe(walking)[terrain_columns]
    assert walking_view[-2:].tolist() == [0.,0.]
    dead = observe(initial.replace(hp=initial.hp.at[1].set(0)))[terrain_columns]
    assert dead[-2:].tolist() == [0.,1.]
    assert jnp.allclose(dead[25:33],walking_view[25:33]*2)
    flying = hero_roster_state(env,initial,['possessed','g000uu0019',None,'cultist',None,None])
    assert observe(flying)[terrain_columns][-2:].tolist() == [1.,0.]
    reset,_ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    chex.assert_trees_all_equal(observe(reset),observe(initial))
    response = human_service.create(42,'legions')
    assert response['snapshot']['travel_costs'] == [2]*4
    assert response['turn_rules']['dead_hero_cost_multiplier'] == 2
    assert len(response['map']['terrain']['water']) == 50
