"""Territory/city contracts from the reference and installed game DBFs/EXE."""
import chex
import jax
import jax.numpy as jnp
import pytest
from stoix.envs.number_grid import MAP, REST, SHOOT, HEAL_START, REVIVE_START
from stoix.envs.number_grid_territory import TerritoryRules, CITY_CAPTURED, CITY_UPGRADED, OBSTACLE
from stoix.envs.number_grid_combat import ARMOR
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


def cleared_city(env, state, index=0):
    return state.replace(alive=state.alive & ~env.territory.city_enemies[index])


@pytest.mark.core
def test_growth_income_acquisition_and_obstacles(current_game):
    env,s,step,_ = current_game
    rules = env.territory
    assert s.territory_claims.tolist() == [1,0,0] and s.mana.tolist() == [0]*5
    assert rules.income(s).tolist() == [100,25,0,0,0,0]
    assert len(rules.mines) == 6 and len(set(m['kind'] for m in rules.mines)) == 6
    turn = compiled_method(rules,'next_turn')
    s = turn(s)
    assert s.territory_claims.tolist() == [11,0,0] and s.gold == 100
    assert s.mana.tolist() == [25,0,0,0,0]
    assert jnp.sum(rules.quotes(s)['territory']) == 11
    # A mine starts paying on the same turn that growth reaches it. No visit.
    for i in range(6):
        rank = rules.ranks[0,rules.mine_cells[i]]
        before = s.replace(territory_claims=jnp.array([rank,0,0]),gold=jnp.int32(0),mana=jnp.zeros(5,jnp.int32))
        assert not rules.mine_ownership(before)[i]
        after = turn(before)
        assert rules.mine_ownership(after)[i]
        assert after.gold == rules.income(after)[0]
        chex.assert_trees_all_equal(after.mana,rules.income(after)[1:])
        chex.assert_trees_all_equal(after.position,s.position)
    full = turn(s.replace(territory_claims=rules.reachable_counts))
    assert rules.income(full).tolist() == [150,75,50,50,50,50]
    assert not jnp.any(rules.owned(full,jnp.array(rules.obstacle_cells)))
    chex.assert_trees_all_equal(full.territory_claims,rules.reachable_counts)
    mask = compiled_method(env,'action_mask')(s)
    assert not mask[0] and not mask[6] and mask[2] and mask[3]
    bumped,ts = step(s,jnp.int32(0))
    assert bumped.last_event == OBSTACLE and float(ts.reward) == pytest.approx(-.251)
    chex.assert_trees_all_equal(bumped.position,s.position)
    assert bumped.movement_points == s.movement_points
    diagonal,_ = step(s,jnp.int32(3))
    assert diagonal.position.tolist() == [3,3]


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
def test_two_city_armies_capture_live_hero_and_no_remote_capture(current_game):
    env,s,step,fight = current_game
    city = env.territory.positions[0]
    s = s.replace(position=city+jnp.array([1,0]))
    for enemy in (41,42):
        assert env.map_commands(s)[0,0] == enemy
        battle,_ = step(s,jnp.int32(0))
        assert battle.in_battle and battle.enemy == enemy and battle.battle_fort_armor == 10
        assert not battle.city_owned[0]
        target = int(jnp.argmax(battle.hp[6:]>0))
        battle = battle.replace(actor=jnp.int32(0),hp=battle.hp.at[6:].set(0).at[6+target].set(1))
        # One deterministic final blow exercises normal casualty/XP cleanup.
        s,_ = fight(battle,jnp.int32(SHOOT+target),battle.battle_key,jnp.zeros(env.random_size))
        assert not s.alive[enemy] and not s.in_battle and not s.city_owned[0]
        assert s.position.tolist() == (city+jnp.array([1,0])).tolist()
        if enemy == 41:
            assert s.alive[42]
        s = s.replace(movement_points=jnp.int32(20))
    entered,_ = step(s,jnp.int32(0))
    assert entered.city_owned.tolist() == [True,False]
    assert entered.territory_claims.tolist() == [1,16,0]
    assert entered.last_event == CITY_CAPTURED and entered.position.tolist() == city.tolist()
    assert env.territory.owned(entered,city+jnp.array([1,0]))
    assert entered.day == s.day and entered.gold == s.gold
    chex.assert_trees_all_equal(entered.mana,s.mana)
    # A dead leader cannot capture; resurrection/rest on the tile is not entry.
    dead = s.replace(hp=s.hp.at[1].set(0))
    inside,_ = step(dead,jnp.int32(0))
    assert not inside.city_owned[0]
    revived = inside.replace(hp=inside.hp.at[1].set(1))
    rested,_ = step(revived,jnp.int32(REST))
    assert not rested.city_owned[0]
    outside,_ = step(rested,jnp.int32(4))
    captured,_ = step(outside,jnp.int32(0))
    assert captured.city_owned[0]
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
    capture = compiled_method(rules,'capture')
    captured = capture(before,jnp.bool_(True))
    # The reference grid counts the city's own cell separately from the budget.
    assert captured.territory_claims.tolist() == [1,1+budget,0]
    assert rules.owned(captured,rules.positions[0]+jnp.array([1,0]))
    assert captured.last_captured_cities.tolist() == [True,False]
    for name in ('gold','mana','day','movement_points','opponent_claims','hp'):
        chex.assert_trees_all_equal(getattr(captured,name),getattr(before,name))
    repeated = capture(captured,jnp.bool_(True))
    chex.assert_trees_all_equal(repeated.territory_claims,captured.territory_claims)
    assert not jnp.any(repeated.last_captured_cities)
    # Daily growth remains an additional budget, not a delayed capture bonus.
    tomorrow = compiled_method(rules,'next_turn')(repeated)
    assert tomorrow.territory_claims.tolist() == [11,1+2*budget,0]


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
        opponent_claims=rules.opponent_counts if contested else jnp.zeros(0,jnp.int32),
        gold=jnp.int32(173),mana=jnp.arange(5,dtype=jnp.int32))
    if contested:
        state = compiled_method(rules,'repaint')(state)
    return rules,state,jnp.array(water)


@pytest.mark.parametrize('contested', [False,True])
def test_capture_connects_mine_immediately_without_income_or_other_source_growth(current_game,contested):
    env,initial,_,_ = current_game
    rules,before,water = capture_land_case(env,initial,contested)
    capture = compiled_method(rules,'capture')
    assert not rules.mine_ownership(before)[0] and rules.income(before)[0] == 100
    rejected = capture(before,jnp.bool_(False))
    for name in ('territory_claims','territory_owners','opponent_claims'):
        chex.assert_trees_all_equal(getattr(rejected,name),getattr(before,name))
    after = capture(before,jnp.bool_(True))
    assert after.territory_claims.tolist() == [1,16]
    assert rules.mine_ownership(after)[0] and rules.income(after)[0] == 150
    assert not jnp.any(rules.owned(after,water))
    assert not rules.owned(after,jnp.array([3,3]))  # obstacle
    assert not rules.owned(after,jnp.array([0,4]))  # outer border
    for name in ('gold','mana','day','movement_points','opponent_claims'):
        chex.assert_trees_all_equal(getattr(after,name),getattr(before,name))
    if contested:
        # A closer new city source wins immediately; equal distances retain
        # the existing owner, and the opposing source itself is not repainted.
        assert rules.owner(before,jnp.array([4,3])) == 2
        assert rules.owner(after,jnp.array([4,4])) == 1
        assert rules.owner(after,jnp.array([4,5])) == 2
        assert rules.owner(after,jnp.array([4,6])) == 2
    tomorrow = compiled_method(rules,'next_turn')(after)
    assert tomorrow.gold == after.gold+150


def test_capture_growth_is_clamped_to_reachable_land(current_game):
    env,initial,_,_ = current_game
    rules,before,water = capture_land_case(env,initial,island=True)
    before = before.replace(city_levels=jnp.array([5],jnp.int32))
    after = compiled_method(rules,'capture')(before,jnp.bool_(True))
    assert rules.reachable_counts[1] == 2 and after.territory_claims.tolist() == [1,2]
    assert rules.mine_ownership(after)[0] and not jnp.any(rules.owned(after,water))
    tomorrow = compiled_method(rules,'next_turn')(after)
    assert tomorrow.territory_claims[1] == 2


def test_city_upgrade_remote_costs_growth_history_and_masks(current_game):
    env,s,step,_ = current_game
    rules = env.territory
    s = s.replace(city_owned=jnp.array([True,False]),territory_claims=jnp.array([11,1,0]),gold=jnp.int32(3000))
    mask = compiled_method(env,'action_mask')
    assert mask(s)[193] and not mask(s)[194]
    for level,cost in ((2,150),(3,250),(4,500),(5,750)):
        before = s
        s,_ = step(s,jnp.int32(193))
        assert s.city_levels[0] == level and s.gold == before.gold-cost and s.last_event == CITY_UPGRADED
        chex.assert_trees_all_equal(s.territory_claims,before.territory_claims)
        assert s.movement_points == before.movement_points and s.day == before.day
    assert not mask(s)[193]
    failed,_ = step(s,jnp.int32(193))
    assert failed.gold == s.gold and failed.city_levels[0] == 5
    rested,_ = step(s,jnp.int32(REST))
    assert rested.territory_claims.tolist() == [21,26,0]
    for invalid in (s.replace(city_owned=jnp.zeros(2,bool)),s.replace(city_levels=jnp.array([1,3]),gold=jnp.int32(149))):
        assert not mask(invalid)[193]
        out,_ = step(invalid,jnp.int32(193))
        chex.assert_trees_all_equal(out.city_levels,invalid.city_levels)
        assert out.gold == invalid.gold
    # Independent sources count their overlaps rather than spending them elsewhere.
    assert rules.ranks.shape == (3,48*48)
    assert jnp.all(rules.ranks[jnp.arange(3),jnp.array([2*48+2,14*48+16,32*48+36])] == 0)


@pytest.mark.core
def test_regeneration_uses_original_fort_instead_of_land_and_does_not_revive(current_game):
    env,s,step,_ = current_game
    rules = env.territory
    regen = jax.jit(rules.regeneration_percent)
    assert regen(s).tolist() == [55,55,55,55,0,0,0,0,0,0,0,0]
    neutral = s.replace(position=jnp.array([8,9]))
    assert regen(neutral)[0] == 20
    own = neutral.replace(territory_claims=jnp.array([2304,0,0]))
    assert regen(own)[0] == 30
    for level,percent in ((1,30),(2,35),(3,40),(4,45),(5,50)):
        city = s.replace(position=rules.positions[0],city_owned=jnp.array([True,False]),
                         territory_claims=jnp.array([2304,1,0]),city_levels=jnp.array([level,3]))
        assert regen(city)[0] == percent and regen(city,1)[0] == percent+10
    injured = s.replace(hp=s.hp.at[:4].set(jnp.array([1,50,44,0])))
    rested,_ = step(injured,jnp.int32(REST))
    assert rested.hp[:6].tolist() == [67,133,110,0,0,0]
    assert rested.gold == 100 and rested.mana.tolist() == [25,0,0,0,0]
    # Regen uses the land before growth, while income uses newly grown land.
    cell = neutral.position[0]*MAP['size']+neutral.position[1]
    before = neutral.replace(territory_claims=jnp.array([rules.ranks[0,cell],0,0]),hp=neutral.hp.at[0].set(1))
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
    chex.assert_trees_all_equal(stats(city)[6:,ARMOR]-stats(city.replace(battle_fort_armor=jnp.int32(0)))[6:,ARMOR],jnp.full(6,20.))
    assert stats(city)[1,ARMOR] == stats(s)[1,ARMOR]  # attacking from capital grants no defensive armor
    field = begin(s.replace(enemy=jnp.int32(0)))
    assert field.battle_fort_armor == 0
    caster = hero_roster_state(env,s,['baroness','duke',None,None,None,None])
    fear = begin(caster.replace(enemy=jnp.int32(41))).replace(actor=jnp.int32(0))
    hit,_ = fight(fear,jnp.int32(SHOOT),fear.battle_key,jnp.zeros(env.random_size))
    assert hit.paralyzed[6] and not hit.retreating[6] and not hit.feared[6]


def test_map_victory_requires_entry_into_both_cities_and_reset_clears_resources(current_game):
    env,s,step,_ = current_game
    s = s.replace(alive=jnp.zeros_like(s.alive),position=env.territory.positions[1]+jnp.array([1,0]),
                  city_owned=jnp.array([True,False]))
    waiting,_ = step(s,jnp.int32(REST))
    assert not waiting.done and not waiting.won
    won,ts = step(waiting,jnp.int32(0))
    assert won.done and won.won and won.city_owned.tolist() == [True,True]
    assert float(ts.reward) == pytest.approx(3.009)
    reset,_ = compiled_method(env,'reset')(jax.random.PRNGKey(7))
    assert not jnp.any(reset.mana) and not jnp.any(reset.city_owned)
    assert reset.territory_claims.tolist() == [1,0,0]
    obs = compiled_method(env,'observation')(reset)
    assert obs.shape == (1843,) and jnp.all(jnp.isfinite(obs))


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
        claims = jnp.array([[1,0,0],[11,25,0],[900,0,70],[2304,2304,2304]])
        owned = jax.vmap(lambda c:env.territory.owned(s.replace(territory_claims=c),positions))(claims)
        expected_owned = jnp.any(env.territory.ranks[None,:,:] < claims[:,:,None],axis=1)
        return actual,expected,movement,permitted,owned,expected_owned
    actual,expected,movement,permitted,owned,expected_owned = compare(alive)
    chex.assert_trees_all_equal(actual,expected)
    chex.assert_trees_all_equal(movement,permitted)
    chex.assert_trees_all_equal(owned,expected_owned)
