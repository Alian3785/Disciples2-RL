"""Territory/city contracts from the reference and installed game DBFs/EXE."""
import chex
import jax
import jax.numpy as jnp
import pytest
from stoix.envs.number_grid import (
    load_map, MAP, REST, SHOOT, DEFEND, RETREAT, CONTINUE, HEAL_START, REVIVE_START,
    FORT_NONE, FORT_BLUE, FORT_RED, ROUND_LIMIT_RETREAT,
)
from stoix.envs.number_grid_territory import (
    TerritoryRules, CITY_CAPTURED, CITY_UPGRADED, OBSTACLE, fort_center, land_order,
)
from stoix.envs.number_grid_combat import ARMOR
from stoix.tests.number_grid_fixtures import compiled_method, full_land, hero_roster_state, enemy_roster_state


def cleared_city(env, state, index=0):
    return state.replace(alive=state.alive & ~env.territory.city_enemies[index])


def land_count(state):
    return int(jnp.sum(state.territory_owners == 1))


def take_city(rules):
    """Capture as env.step does: the city first, its land at the end of the step."""
    def run(state, entered):
        after = rules.capture(state, entered)
        return rules.grow_captured(after, after.city_owned & ~state.city_owned)
    return jax.jit(run)


def painted_up_to(rules, state, count):
    """The capital's front has painted its first count tiles, and no mine yet."""
    return state.replace(territory_owners=state.territory_owners.at[rules.land_order[0,:count]].set(1),
                         territory_claims=state.territory_claims.at[0].set(count))


@pytest.mark.core
def test_growth_income_acquisition_and_obstacles(current_game):
    """Daily land growth acquires mine income without crossing blocked cells."""
    env,s,step,_ = current_game
    rules = env.territory
    assert s.territory_claims.tolist() == [0,0,0] and s.mana.tolist() == [0]*5
    assert land_count(s) == 1
    assert rules.income(s).tolist() == [100,25,0,0,0,0]
    assert len(rules.mines) == 6 and len(set(m['kind'] for m in rules.mines)) == 6
    turn = compiled_method(rules,'next_turn')
    start = s
    s = turn(s)
    # Ten new tiles touch the gold mine and the infernal source: their rings come outside the quota.
    assert s.territory_claims.tolist() == [10,0,0] and s.gold == 150
    assert s.mana.tolist() == [75,0,0,0,0]
    assert rules.mine_ownership(s).tolist() == [True,True,False,False,False,False]
    assert jnp.sum(rules.quotes(s)['territory']) == 27
    # A mine starts paying on the same turn that the front touches it. No visit.
    touching = rules.touch_bits[rules.land_order[0,:rules.land_lengths[0]]]
    for i in range(6):
        rank = int(jnp.argmax(touching == 1 << i))
        assert touching[rank] == 1 << i
        before = painted_up_to(rules,start,rank).replace(gold=jnp.int32(0),mana=jnp.zeros(5,jnp.int32))
        assert not rules.mine_ownership(before)[i]
        after = turn(before)
        assert rules.mine_ownership(after)[i]
        assert after.gold == rules.income(after)[0]
        chex.assert_trees_all_equal(after.mana,rules.income(after)[1:])
        chex.assert_trees_all_equal(after.position,s.position)
    full = full_land(env,s)
    assert rules.income(full).tolist() == [150,75,50,50,50,50]
    # Only a mine's ring paints the map's obstacles; sites do not stop the land at all.
    others = jnp.array([p for p in env.map_config['obstacles'] if rules.paint_bits[p[0]*env.size+p[1]] == 0])
    assert not jnp.any(rules.owned(full,others))
    assert jnp.any(rules.owned(full,jnp.array(env.site_rules.sites[0]['footprint'])))
    chex.assert_trees_all_equal(turn(full).territory_owners,full.territory_owners)
    mask = compiled_method(env,'action_mask')(s)
    assert not mask[0] and not mask[6] and mask[2] and mask[3]
    bumped,ts = step(s,jnp.int32(0))
    assert bumped.last_event == OBSTACLE and float(ts.reward) == pytest.approx(-.251)
    chex.assert_trees_all_equal(bumped.position,s.position)
    assert bumped.movement_points == s.movement_points
    diagonal,_ = step(s,jnp.int32(3))
    assert diagonal.position.tolist() == [3,3]


@pytest.mark.core
def test_mines_block_movement_but_land_still_reaches_them(current_game):
    """Original Crystal tiles are impassable, for gold mines and mana sources alike."""
    env,s,step,_ = current_game
    rules = env.territory
    assert {m['kind'] for m in rules.mines} >= {'gold','infernal'}
    assert jnp.all(rules.mine_ownership(full_land(env,s)))
    mask = compiled_method(env,'action_mask')
    for i, mine in enumerate(rules.mines):
        tile = tuple(mine['position'])
        assert tile in rules.impassable_cells and tile not in rules.obstacle_cells
        assert not jnp.any(rules.land_order[0,:rules.land_lengths[0]] == rules.mine_cells[i])
        # Stand north of the mine: action 4 steps south.
        start =s.replace(position=jnp.array([tile[0]-1,tile[1]]),movement_points=jnp.int32(20))
        assert not mask(start)[4]
        bumped,ts = step(start,jnp.int32(4))
        assert bumped.last_event == OBSTACLE and float(ts.reward) == pytest.approx(-.251)
        chex.assert_trees_all_equal(bumped.position,start.position)
        assert bumped.movement_points == 20


@pytest.mark.parametrize('faction,mana', [('legions',0),('empire',1),('undead_hordes',2),('mountain_clans',3),('elves',4)])
def test_faction_native_mana_and_lord_data(current_game,faction,mana):
    env,s,_,_ = current_game
    rules = TerritoryRules({**MAP,'faction':faction,'lord_type':'guildmaster'},env.progression,193,env.site_rules,env.ruins)
    expected = [100]+[25 if i == mana else 0 for i in range(5)]
    assert rules.base_income.tolist() == expected
    assert rules.costs.tolist() == [0,75,125,250,375,0]
    assert rules.lord_regen == 0
    assert jax.jit(rules.regeneration_percent)(s)[0] == 40


@pytest.mark.core
def test_last_defender_takes_the_city_and_moves_the_party_in(current_game):
    """CBattleResultVillage: the battle clearing a city captures it, even if the leader fell."""
    env,s,step,fight = current_game
    city = env.territory.positions[0]
    outside = city+jnp.array([1,0])

    def beat(state, enemy, leader_falls=False):
        """Enter the fight and land one deterministic final blow (normal casualty/XP cleanup)."""
        assert env.map_commands(state)[0,0] == enemy
        battle,_ = step(state,jnp.int32(0))
        assert battle.in_battle and battle.enemy == enemy and battle.battle_fort_armor == 10
        assert not battle.city_owned[0]
        target = int(jnp.argmax(battle.hp[6:]>0))
        hp = battle.hp.at[6:].set(0).at[6+target].set(1)
        battle = battle.replace(actor=jnp.int32(0),hp=hp.at[1].set(0) if leader_falls else hp)
        after,_ = fight(battle,jnp.int32(SHOOT+target),battle.battle_key,jnp.zeros(env.random_size))
        assert not after.alive[enemy] and not after.in_battle
        return after.replace(movement_points=jnp.int32(20))

    first = beat(s.replace(position=outside),41)
    assert first.alive[42] and not first.city_owned[0] and first.position.tolist() == outside.tolist()
    points = first.movement_points
    taken = beat(first,42)
    assert taken.city_owned.tolist() == [True,False]
    assert taken.last_event == CITY_CAPTURED and taken.position.tolist() == city.tolist()
    # env.step grows the city's land once, at the end of the public step.
    grown = compiled_method(env.territory,'grow_captured')(taken,taken.city_owned & ~first.city_owned)
    assert grown.territory_claims[1] == 15 and land_count(grown) >= land_count(first)+15
    assert env.territory.owned(grown,outside)
    assert taken.day == first.day and taken.gold == first.gold
    chex.assert_trees_all_equal(taken.mana,first.mana)
    # The leader falling in that same battle changes nothing about the capture.
    fallen = beat(first,42,leader_falls=True)
    assert fallen.hp[1] == 0 and fallen.city_owned[0] and fallen.position.tolist() == city.tolist()
    # A dead leader cannot start a fight with city defenders; field armies stay open to him.
    mask = compiled_method(env,'action_mask')
    leaderless = first.replace(hp=first.hp.at[1].set(0),movement_points=points)
    assert not mask(leaderless)[0]
    refused,_ = step(leaderless,jnp.int32(0))
    assert not refused.in_battle and refused.position.tolist() == outside.tolist()
    assert refused.movement_points == points
    field = int(jnp.argmax(env.territory.enemy_city<0))
    assert not env.territory.city_attack_refused(leaderless,leaderless.enemy_positions[field])
    assert mask(first)[0]
    # Defenders gone without a fight of the party (a summon): entry needs a living hero.
    s = first.replace(alive=first.alive.at[42].set(False))
    dead = s.replace(hp=s.hp.at[1].set(0))
    inside,_ = step(dead,jnp.int32(0))
    assert not inside.city_owned[0]
    revived = inside.replace(hp=inside.hp.at[1].set(1))
    rested,_ = step(revived,jnp.int32(REST))
    assert not rested.city_owned[0]
    outside,_ = step(rested,jnp.int32(4))
    captured,_ = step(outside,jnp.int32(0))
    assert captured.city_owned[0] and captured.territory_claims[1] == 15
    capture = compiled_method(env.territory,'capture')
    for bad in (revived.replace(unit_ids=revived.unit_ids.at[1].set(0)),
                revived.replace(summon_owner=revived.summon_owner.at[1].set(0)),
                revived.replace(copied=revived.copied.at[1].set(True)),
                revived.replace(alive=revived.alive.at[42].set(True))):
        rejected = capture(bad,True)
        assert not rejected.city_owned[0]
        chex.assert_trees_all_equal(rejected.territory_claims,bad.territory_claims)


@pytest.mark.core
@pytest.mark.parametrize('level,budget', [(1,15),(2,15),(3,20),(4,20),(5,25)])
def test_city_capture_spreads_once_before_the_next_day(current_game,level,budget):
    """EXE sub_4343DA: trunc(T_CAPTURE/100) * T_CITY[level] on capture."""
    env,initial,_,_ = current_game
    rules = env.territory
    before = cleared_city(env,initial).replace(position=rules.positions[0],
        city_levels=initial.city_levels.at[0].set(level),gold=jnp.int32(173),
        mana=jnp.arange(5,dtype=jnp.int32))
    capture = take_city(rules)
    captured = capture(before,jnp.bool_(True))
    assert captured.territory_claims.tolist() == [0,budget,0]
    assert rules.owned(captured,rules.positions[0]+jnp.array([1,0]))
    assert captured.last_captured_cities.tolist() == [True,False]
    for name in ('gold','mana','day','movement_points','opponent_claims','hp'):
        chex.assert_trees_all_equal(getattr(captured,name),getattr(before,name))
    repeated = capture(captured,jnp.bool_(True))
    chex.assert_trees_all_equal(repeated.territory_claims,captured.territory_claims)
    assert not jnp.any(repeated.last_captured_cities)
    # Daily growth remains an additional budget, not a delayed capture bonus.
    tomorrow = compiled_method(rules,'next_turn')(repeated)
    assert tomorrow.territory_claims[0] == 10 and tomorrow.territory_claims[1] > budget
    assert land_count(tomorrow) >= land_count(repeated)+10+budget


def capture_land_case(env,initial,contested=False,island=False):
    """Small territory-only fixture; reuse the real unit catalogue and hero."""
    city, mine = [4,4], [4,3]
    water = ([[r,c] for r in range(3,6) for c in range(2,6) if [r,c] not in (city,mine)]
             if island else [[3,4]])
    config = dict(size=9,capital_position=[1,1],agent_position=[1,1],
        faction='legions',lord_type='warrior',cities=[dict(position=city,level=1)],
        mines=[dict(position=mine,kind='gold')],opponent_positions=[[6,6]],
        obstacles=[] if island else [[3,3]],terrain=dict(water=water),
        enemy_territory_sources=[dict(position=[4,6],growth=10)] if contested else [])
    rules = TerritoryRules(config,env.progression,193)
    state = initial.replace(position=jnp.array(city),alive=jnp.ones(1,bool),
        city_owned=jnp.zeros(1,bool),city_levels=rules.initial_levels,
        territory_claims=rules.initial_claims,last_captured_cities=jnp.zeros(1,bool),
        territory_owners=rules.initial_owners,
        opponent_claims=rules.initial_opponent_claims,
        gold=jnp.int32(173),mana=jnp.arange(5,dtype=jnp.int32))
    if contested:
        # The opposing source has painted everything its front reaches.
        state = compiled_method(rules,'_spread')(state,jnp.array([0,0,81]))
    return rules,state,jnp.array(water)


@pytest.mark.core
@pytest.mark.parametrize('contested', [False,True])
def test_capture_connects_mine_immediately_without_income_or_other_source_growth(current_game,contested):
    """Capture claims nearby mines immediately but pays their income the next day."""
    env,initial,_,_ = current_game
    rules,before,water = capture_land_case(env,initial,contested)
    capture = take_city(rules)
    assert not rules.mine_ownership(before)[0] and rules.income(before)[0] == 100
    rejected = capture(before,jnp.bool_(False))
    for name in ('territory_claims','territory_owners','opponent_claims'):
        chex.assert_trees_all_equal(getattr(rejected,name),getattr(before,name))
    after = capture(before,jnp.bool_(True))
    assert after.territory_claims[0] == before.territory_claims[0] and rules.mine_ownership(after)[0]
    assert rules.income(after)[0] == 150
    assert not jnp.any(rules.owned(after,water))
    # Discipl2.exe 0x60c411 paints the mine's whole ring, obstacles included, outside the quota.
    assert rules.owned(after,jnp.array([3,3]))
    assert not rules.owned(after,jnp.array([0,4]))  # outer border
    for name in ('gold','mana','day','movement_points','opponent_claims'):
        chex.assert_trees_all_equal(getattr(after,name),getattr(before,name))
    if contested:
        # A new front repaints foreign land at once; the opposing source's own tile keeps its owner.
        assert rules.owner(before,jnp.array([4,3])) == 2
        assert rules.owner(after,jnp.array([4,3])) == 1
        assert rules.owner(after,jnp.array([4,6])) == 2
    # Income is paid after every front has grown: the opposing one takes the mine back that day.
    tomorrow = compiled_method(rules,'next_turn')(after)
    assert tomorrow.gold == after.gold+(100 if contested else 150)


@pytest.mark.core
def test_capture_growth_is_clamped_to_reachable_land(current_game):
    """A city on an island cannot spend its capture budget across water: 0x5e1408 walks four ways."""
    env,initial,_,_ = current_game
    rules,before,water = capture_land_case(env,initial,island=True)
    before = before.replace(city_levels=jnp.array([5],jnp.int32))
    after = take_city(rules)(before,jnp.bool_(True))
    assert rules.reachable_counts[1] == 0 and after.territory_claims.tolist() == [0,0]
    assert not rules.mine_ownership(after)[0] and not jnp.any(rules.owned(after,water))
    tomorrow = compiled_method(rules,'next_turn')(after)
    assert tomorrow.territory_claims[1] == 0


def land_rules(env,**extra):
    """A 12x12 map with its capital in the corner; the tests place the other objects."""
    return TerritoryRules({**dict(size=12,capital_position=[1,1],agent_position=[1,1],faction='legions',
        lord_type='warrior',opponent_positions=[[10,10]],obstacles=[],mines=[],cities=[]),**extra},env.progression,193)


def land_state(rules,initial):
    return initial.replace(territory_owners=rules.initial_owners,territory_claims=rules.initial_claims,
                           opponent_claims=rules.initial_opponent_claims,city_owned=jnp.zeros(rules.count,bool),
                           city_levels=rules.initial_levels,last_captured_cities=jnp.zeros(rules.count,bool))


@pytest.mark.core
def test_land_order_follows_the_original_heap():
    """Discipl2.exe 0x5e1499: four directions, by depth, with the MSVC heap's order among equals."""
    assert land_order(5,(1,1),frozenset()) == tuple(5*r+c for r,c in (
        (2,1),(1,2),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)))
    # A rejected tile is neither painted nor spread through.
    assert land_order(5,(1,1),frozenset({(1,2),(2,1)})) == ()
    assert 5*3+1 not in land_order(5,(1,1),frozenset({(2,1),(2,2),(1,3)}))
    assert fort_center({(r,c) for r in range(3,8) for c in range(4,9)}) == (5,6)
    assert fort_center({(r,c) for r in range(3,7) for c in range(4,8)}) == (5,6)
    assert fort_center({(3,4)}) == (3,4)


@pytest.mark.core
def test_daily_quota_counts_new_tiles_only(current_game):
    """Own tiles cost a front nothing, so painted land does not waste the day's growth."""
    env,initial,_,_ = current_game
    rules = land_rules(env,starting_land=[[1,1],[1,2],[2,1],[2,2],[1,3],[3,1]])
    state = land_state(rules,initial)
    assert land_count(state) == 6 and state.territory_claims[0] > 0
    turn = compiled_method(rules,'next_turn')
    state = turn(state)
    assert land_count(state) == 16
    assert land_count(turn(state)) == 26


@pytest.mark.core
@pytest.mark.parametrize('blocker,stops',[('land_blocked',True),('water',True),('real_fort',True),
                                          ('neutral_city',False),('foreign_land',False)])
def test_what_stops_the_land_front(current_game,blocker,stops):
    """0x5e119b: water, mountains and real players' forts stop it; neutral cities and foreign paint do not."""
    env,initial,_,_ = current_game
    wall = [[r,3] for r in range(1,11)]
    if blocker in ('real_fort','neutral_city'):
        rules = land_rules(env,land_blocked=[p for p in wall if p != [5,3]],cities=[dict(position=[5,3],level=1,
            footprint=[[5,3]],neutral=blocker == 'neutral_city')])
    else:
        rules = land_rules(env,**(dict(terrain=dict(water=wall)) if blocker == 'water' else {blocker:wall}))
    amounts = jnp.zeros(rules.land_order.shape[0],jnp.int32).at[0].set(144)
    full = compiled_method(rules,'_spread')(land_state(rules,initial),amounts)
    assert bool(rules.owned(full,jnp.array([5,6]))) == (not stops)


def owned_cells(state):
    return {int(c) for c in jnp.nonzero(state.territory_owners == 1)[0]}


@pytest.mark.core
def test_touched_mine_paints_its_ring_outside_the_quota(current_game):
    """0x60c3b5/0x60c411: the mine and the eight tiles around it, as soon as the front is beside it."""
    env,initial,_,_ = current_game
    rules = land_rules(env,mines=[dict(position=[1,4],kind='gold')])
    state = land_state(rules,initial)
    order = [int(c) for c in rules.land_order[0,:rules.land_lengths[0]]]
    assert int(rules.mine_cells[0]) not in order  # The mine itself stops the front.
    first = next(i for i,c in enumerate(order) if rules.touch_bits[c] == 1)
    grow = compiled_method(rules,'_spread')
    assert not rules.mine_ownership(grow(state,jnp.array([first])))[0]
    touched = grow(state,jnp.array([first+1]))
    assert touched.territory_claims[0] == first+1
    assert owned_cells(touched) == {12+1} | set(order[:first+1]) | {r*12+c for r in (1,2) for c in (3,4,5)}


@pytest.mark.core
@pytest.mark.parametrize('capital,chained',[([1,10],True),([1,1],False)],ids=['from-east','from-west'])
def test_a_mine_ring_can_touch_the_next_mine(current_game,capital,chained):
    """0x60c2ff walks the set while 0x60c411 grows it; a tile takes the first mine of table 0x6f1690, westmost."""
    env,initial,_,_ = current_game
    rules = land_rules(env,capital_position=capital,agent_position=capital,opponent_positions=[[10,5]],
                       mines=[dict(position=[1,4],kind='gold'),dict(position=[1,6],kind='life')])
    state = land_state(rules,initial)
    order = rules.land_order[0,:rules.land_lengths[0]]
    first = int(jnp.argmax(rules.touch_bits[order] != 0))
    after = compiled_method(rules,'_spread')(state,jnp.array([first+1]))
    assert rules.mine_ownership(after).tolist() == ([True,True] if chained else [True,False])


@pytest.mark.core
@pytest.mark.parametrize('name,start,capital,union,never',[
    ('default',140,139,902,{(5,25),(12,7),(17,26),(41,30),(47,30),(47,42)}),
    ('the_charmed_child',190,287,1140,{(17,34)}),
    ('bethrezens_retreat',259,177,2322,{(11,4),(19,6),(40,22),(69,29)}),
])
def test_original_scenarios_paint_the_start_and_reach_what_the_exe_reaches(current_game,name,start,capital,union,never):
    """Pinned from an independent model of 0x5e1499 on the scenarios' own tiles."""
    env,initial,_,_ = current_game
    config = load_map(name)
    rules = TerritoryRules(config,env.progression,193)
    state = land_state(rules,initial)
    assert land_count(state) == start and rules.land_lengths[0] == capital
    reach = {int(c) for s,length in enumerate(rules.land_lengths[:1+rules.count])
             for c in rules.land_order[s,:length]}
    assert len(reach) == union
    # Sites and ordinary landmarks do not stop the land, so the front covers some of them.
    full = compiled_method(rules,'_spread')(state,jnp.full(rules.land_order.shape[0],config['size']**2))
    owned = rules.mine_ownership(full)
    assert {tuple(m['position']) for m,o in zip(rules.mines,owned.tolist()) if not o} == never


def test_city_upgrade_remote_costs_growth_history_and_masks(current_game):
    env,start,step,_ = current_game
    rules = env.territory
    start = start.replace(city_owned=jnp.array([True,False]),gold=jnp.int32(3000))
    mask = compiled_method(env,'action_mask')
    assert mask(start)[193] and not mask(start)[194]
    s = start
    for level,cost in ((2,150),(3,250),(4,500),(5,750)):
        before = s
        s,_ = step(s,jnp.int32(193))
        assert s.city_levels[0] == level and s.gold == before.gold-cost and s.last_event == CITY_UPGRADED
        chex.assert_trees_all_equal(s.territory_claims,before.territory_claims)
        assert s.movement_points == before.movement_points and s.day == before.day
        s,_ = step(s,jnp.int32(REST))  # the next level has to wait for the next turn
    # The turn's limit is clear here, so only the level cap closes the action.
    assert not s.city_upgraded_today[0] and not mask(s)[193]
    failed,_ = step(s,jnp.int32(193))
    assert failed.gold == s.gold and failed.city_levels[0] == 5
    rested,_ = step(start.replace(city_levels=s.city_levels),jnp.int32(REST))
    assert rested.territory_claims.tolist() == [10,25,0]
    for invalid in (s.replace(city_owned=jnp.zeros(2,bool)),s.replace(city_levels=jnp.array([1,3]),gold=jnp.int32(149))):
        assert not mask(invalid)[193]
        out,_ = step(invalid,jnp.int32(193))
        chex.assert_trees_all_equal(out.city_levels,invalid.city_levels)
        assert out.gold == invalid.gold
    # Each fort's front starts at its own middle, which it never paints itself.
    assert rules.land_order.shape[0] == 3
    for source,cell in enumerate((2*48+2,14*48+16,32*48+36)):
        assert not jnp.any(rules.land_order[source,:rules.land_lengths[source]] == cell)


@pytest.mark.core
def test_city_upgrade_once_per_turn_for_each_city(current_game):
    """Each city gains one level per turn: another city stays open and REST starts the next turn."""
    env,start,step,_ = current_game
    mask = compiled_method(env,'action_mask')
    observe = compiled_method(env,'observation')
    s = start.replace(city_owned=jnp.array([True,True]),gold=jnp.int32(3000))
    assert s.city_levels.tolist() == [1,3] and not jnp.any(s.city_upgraded_today)
    assert mask(s)[193] and mask(s)[194]
    first,_ = step(s,jnp.int32(193))
    assert first.city_levels.tolist() == [2,3] and first.city_upgraded_today.tolist() == [True,False]
    assert not mask(first)[193] and mask(first)[194]
    # A repeat in the same turn changes nothing and costs nothing.
    repeat,_ = step(first,jnp.int32(193))
    assert repeat.city_levels.tolist() == [2,3] and repeat.gold == first.gold
    assert repeat.city_upgraded_today.tolist() == [True,False]
    second,_ = step(first,jnp.int32(194))
    assert second.city_levels.tolist() == [2,4] and second.gold == s.gold-150-500
    assert not mask(second)[193] and not mask(second)[194]
    # A refused attempt does not use up the turn's upgrade.
    poor = s.replace(gold=jnp.int32(149))
    refused,_ = step(poor,jnp.int32(193))
    assert refused.city_levels.tolist() == [1,3] and not jnp.any(refused.city_upgraded_today)
    rested,_ = step(second,jnp.int32(REST))
    assert rested.day == second.day+1 and not jnp.any(rested.city_upgraded_today)
    assert mask(rested)[193] and mask(rested)[194]
    next_turn,_ = step(rested,jnp.int32(193))
    assert next_turn.city_levels.tolist() == [3,4]
    # The agent sees the limit: exactly one feature tells the two states apart.
    delta = observe(s.replace(city_upgraded_today=jnp.array([True,False])))-observe(s)
    assert jnp.count_nonzero(delta) == 1 and jnp.sum(delta) == 1


@pytest.mark.core
def test_regeneration_uses_original_fort_instead_of_land_and_does_not_revive(current_game):
    env,s,step,_ = current_game
    rules = env.territory
    regen = jax.jit(rules.regeneration_percent)
    assert regen(s).tolist() == [55,55,55,55,0,0,0,0,0,0,0,0]
    neutral = s.replace(position=jnp.array([8,9]))
    assert regen(neutral)[0] == 20
    own = neutral.replace(territory_owners=jnp.ones_like(neutral.territory_owners))
    assert regen(own)[0] == 30
    for level,percent in ((1,30),(2,35),(3,40),(4,45),(5,50)):
        city = own.replace(position=rules.positions[0],city_owned=jnp.array([True,False]),
                           city_levels=jnp.array([level,3]))
        assert regen(city)[0] == percent and regen(city,1)[0] == percent+10
    injured = s.replace(hp=s.hp.at[:4].set(jnp.array([1,50,44,0])))
    rested,_ = step(injured,jnp.int32(REST))
    assert rested.hp[:6].tolist() == [67,133,110,0,0,0]
    assert rested.gold == 150 and rested.mana.tolist() == [75,0,0,0,0]
    # Regen uses the land before growth, while income uses newly grown land.
    cell = neutral.position[0]*MAP['size']+neutral.position[1]
    rank = int(jnp.argmax(rules.land_order[0] == cell))
    before = painted_up_to(rules,neutral,rank).replace(hp=neutral.hp.at[0].set(1))
    after,_ = step(before,jnp.int32(REST))
    assert rules.owned(after,after.position) and after.hp[0] == 25


def test_captured_city_services_require_ownership_and_capital_buildings(current_game):
    env,s,step,_ = current_game
    s = cleared_city(env,s).replace(position=env.territory.positions[0],gold=jnp.int32(1000))
    s = hero_roster_state(env,s,[None,'duke',None,'cultist',None,None])
    mask = compiled_method(env,'action_mask')
    assert not mask(s)[env.recruitment.start]
    owned = s.replace(city_owned=jnp.array([True,False]))
    hired,_ = step(owned,jnp.int32(env.recruitment.start))
    assert hired.unit_ids[0] == env.progression.ids['possessed'] and hired.gold == 950
    injured = hired.replace(hp=hired.hp.at[0].set(1).at[1].set(0))
    assert not mask(injured)[HEAL_START] and not mask(injured)[REVIVE_START+1]
    temple = env.capital.temple_slot
    ready = injured.replace(buildings=jnp.uint32(1 << temple))
    assert mask(ready)[HEAL_START] and mask(ready)[REVIVE_START+1]
    healed,_ = step(ready,jnp.int32(HEAL_START))
    assert healed.hp[0] == env.max_hp(ready)[0] and healed.gold < ready.gold
    revived,_ = step(ready,jnp.int32(REVIVE_START+1))
    assert revived.hp[1] == 1
    assert not mask(ready.replace(city_owned=jnp.zeros(2,bool)))[HEAL_START]


def test_city_armor_and_fear_protect_defenders_only(current_game):
    env,s,_,fight = current_game
    begin = compiled_method(env,'_begin_battle')
    stats = compiled_method(env,'_combat_stats')
    city = begin(s.replace(enemy=jnp.int32(43)))
    assert city.battle_fort_armor == 20
    assert city.battle_fort_defender == FORT_RED
    assert compiled_method(env,'action_mask')(city.replace(actor=jnp.int32(1)))[RETREAT]
    chex.assert_trees_all_equal(stats(city)[6:,ARMOR]-stats(city.replace(battle_fort_armor=jnp.int32(0)))[6:,ARMOR],jnp.full(6,20.))
    assert stats(city)[1,ARMOR] == stats(s)[1,ARMOR]  # attacking from capital grants no defensive armor
    field = begin(s.replace(enemy=jnp.int32(0)))
    assert field.battle_fort_armor == 0 and field.battle_fort_defender == FORT_NONE
    caster = hero_roster_state(env,s,['baroness','duke',None,None,None,None])
    fear = begin(caster.replace(enemy=jnp.int32(41))).replace(actor=jnp.int32(0))
    hit,_ = fight(fear,jnp.int32(SHOOT),fear.battle_key,jnp.zeros(env.random_size))
    assert hit.paralyzed[6] and not hit.retreating[6] and not hit.feared[6]


@pytest.mark.parametrize('fort_side', [FORT_NONE, FORT_BLUE, FORT_RED])
def test_fort_retreat_mask_step_and_escape_for_both_sides(current_game, fort_side):
    """Steam sub_5F5A4B: a settlement blocks its defenders, never its attackers."""
    env, initial, step, fight = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    state = enemy_roster_state(env,state,['squire','archer',None,None,None,None])
    # The restriction is independent of the armour value, including zero.
    state = state.replace(actor=jnp.int32(1), battle_fort_defender=jnp.int32(fort_side),
                          battle_fort_armor=jnp.int32(0), battle_hero_fort_armor=jnp.int32(0))
    mask = compiled_method(env,'action_mask')
    assert bool(mask(state)[RETREAT]) == (fort_side != FORT_BLUE)
    rolls = jnp.zeros(env.random_size).at[36].set(.99).at[67].set(.99)
    if fort_side == FORT_BLUE:
        denied, _ = step(state,jnp.int32(RETREAT))
        for field in ('hp','retreating','escaped','battle_key','turn_phase','actor'):
            chex.assert_trees_all_equal(getattr(denied,field),getattr(state,field))
        late = state.replace(round=jnp.int32(env.max_rounds+1))
        assert not env.round_limit_retreat(late) and not env.automatic_battle_turn(late)
        assert mask(late)[DEFEND] and not mask(late)[RETREAT]
        boundary = state.replace(round=jnp.int32(env.max_rounds),
            turn_phase=jnp.full(12,2,jnp.int32).at[1].set(0))
        following, _ = fight(boundary,jnp.int32(DEFEND),boundary.battle_key,rolls)
        assert following.round == env.max_rounds+1 and following.last_event != ROUND_LIMIT_RETREAT
    else:
        following, _ = fight(state,jnp.int32(RETREAT),state.battle_key,rolls)
        assert following.retreating[1]
    # A queued escape cannot bypass the restriction, for either actor colour.
    for actor, blocked in ((1,fort_side == FORT_BLUE),(6,fort_side == FORT_RED)):
        queued = state.replace(actor=jnp.int32(actor),retreating=state.retreating.at[actor].set(True))
        following, _ = fight(queued,jnp.int32(CONTINUE),queued.battle_key,rolls)
        assert bool(following.escaped[actor]) == (not blocked)
        assert not following.retreating[actor]


@pytest.mark.parametrize('fort_side', [FORT_NONE, FORT_BLUE, FORT_RED])
@pytest.mark.parametrize('target', [1, 7], ids=['blue-target','red-target'])
def test_fort_fear_skips_one_activation_but_attackers_escape(current_game, fort_side, target):
    """Fear onHit 0x65D06F selects Retreating or Paralyze using sub_5F5A4B."""
    env, initial, _, fight = current_game
    state = compiled_method(env,'_begin_battle')(initial.replace(enemy=jnp.int32(0)))
    state = hero_roster_state(env,state,['baroness','duke',None,None,None,None])
    state = enemy_roster_state(env,state,['baroness','squire',None,None,None,None])
    actor = 6 if target < 6 else 0
    state = state.replace(actor=jnp.int32(actor),battle_fort_defender=jnp.int32(fort_side),
        battle_fort_armor=jnp.int32(0),battle_hero_fort_armor=jnp.int32(0),
        priority=jnp.zeros(12).at[target].set(10000.))
    rolls = jnp.zeros(env.random_size).at[36].set(.99).at[67].set(.99)
    hit, _ = fight(state,jnp.int32(SHOOT+target%6),state.battle_key,rolls)
    protected = fort_side == (FORT_BLUE if target < 6 else FORT_RED)
    assert bool(hit.paralyzed[target]) == protected
    assert not hit.long_paralyzed[target]
    assert bool(hit.feared[target]) == (not protected)
    assert bool(hit.retreating[target]) == (not protected)
    assert hit.actor == target
    following, _ = fight(hit,jnp.int32(CONTINUE),hit.battle_key,rolls)
    assert not following.paralyzed[target]
    assert bool(following.escaped[target]) == (not protected)
    assert not following.retreating[target]


@pytest.mark.core
def test_current_victory_needs_no_city_visit_and_reset_clears_resources(current_game):
    env,s,step,_ = current_game
    s = s.replace(alive=jnp.zeros_like(s.alive),position=env.territory.positions[1]+jnp.array([1,0]),
                  city_owned=jnp.array([True,False]))
    won,ts = step(s,jnp.int32(REST))
    assert won.done and won.won and won.city_owned.tolist() == [True,False]
    assert float(ts.discount) == 0 and jnp.isfinite(ts.reward)
    reset,_ = compiled_method(env,'reset')(jax.random.PRNGKey(7))
    assert not jnp.any(reset.mana) and not jnp.any(reset.city_owned)
    assert reset.territory_claims.tolist() == [0,0,0] and land_count(reset) == 1
    obs = compiled_method(env,'observation')(reset)
    assert obs.shape == (env.observation_size,) and jnp.all(jnp.isfinite(obs))


def test_cell_lookup_matches_coordinate_oracle_for_every_tile(current_game):
    env,s,_,_ = current_game
    positions = jnp.stack(jnp.meshgrid(jnp.arange(env.size),jnp.arange(env.size),indexing='ij'),axis=-1).reshape(-1,2)
    alive = jnp.stack((s.alive,jnp.zeros_like(s.alive),jnp.arange(env.num_opponents)%2==0,jnp.arange(env.num_opponents)%2==1))
    @jax.jit
    def compare(alive):
        states = jax.tree.map(lambda x:jnp.broadcast_to(x,(4,)+x.shape),s).replace(alive=alive)
        actual = jax.vmap(lambda state:env.territory.enemy_at(state,positions))(states)
        matches = alive[:,None,:] & jnp.all(positions[None,:,None,:]==env.opponent_positions[None,None,:,:],axis=-1)
        expected = jnp.where(jnp.any(matches,axis=-1),jnp.argmax(matches,axis=-1),-1)
        movement = jax.vmap(lambda p:env.territory.movement_mask(s.replace(position=p)))(positions)
        permitted = ~env.territory.blocked(positions[:,None,:]+env.directions[None,:,:])
        grids = jnp.stack((jnp.zeros_like(s.territory_owners),jnp.ones_like(s.territory_owners),
                           (jnp.arange(env.size**2)%3).astype(jnp.int8),s.territory_owners))
        owned = jax.vmap(lambda o:env.territory.owned(s.replace(territory_owners=o),positions))(grids)
        inside = jnp.all((positions > 0) & (positions < env.size-1),axis=-1)
        expected_owned = (grids == 1) & inside
        return actual,expected,movement,permitted,owned,expected_owned
    actual,expected,movement,permitted,owned,expected_owned = compare(alive)
    chex.assert_trees_all_equal(actual,expected)
    chex.assert_trees_all_equal(movement,permitted)
    chex.assert_trees_all_equal(owned,expected_owned)
