"""Reference ruin geometry, living leader, guarded loot and cleared obstacle."""
import copy
import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, SHOOT, CONTINUE, REST, POTION_START
from stoix.envs.number_grid_combat import DAMAGE
from stoix.envs.number_grid_items import ItemRules
from stoix.envs.number_grid_potions import PotionRules
from stoix.envs.number_grid_progression import BLOCKED_XP_PENALTY
from stoix.envs.number_grid_ruins import RuinRules
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


def at_ruin(env,state):
    return state.replace(position=jnp.array([10,25]),movement_points=jnp.int32(20))


def finish(env,state,fight):
    battle = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(45)))
    battle = battle.replace(actor=jnp.int32(0),hp=battle.hp.at[6:].set(0).at[7].set(1))
    return fight(battle,jnp.int32(SHOOT+1),battle.battle_key,jnp.zeros(env.random_size))


def test_ruin_guardians_geometry_and_batched_access(current_game):
    env,s,step,_ = current_game
    ruin = env.ruins.ruins[0]
    assert ruin['position'] == [7,23] and ruin['entrance'] == [9,25]
    assert len(ruin['footprint']) == 9 and len(ruin['interaction_tiles']) == 5
    assert MAP['enemy_rosters'][45] == [None,'g000uu5016',None,'goblin_archer',None,None]
    troll = env.progression.rows[env.progression.ids['g000uu5016']]
    assert (troll['size'],troll['max_hp'],troll['damage'],troll['initiative']) == (2,350,120,40)
    # All eight neighbours of the entrance: only five lie outside its footprint.
    positions = env.ruins.entrances[0]-env.directions
    states = [s.replace(position=p,movement_points=jnp.int32(1)) for p in positions]
    batch = jax.tree.map(lambda *xs:jnp.stack(xs),*states)
    mask = compiled_method(env,'action_mask',batched=True)(batch)
    expected = [tuple(p.tolist()) in ruin['interaction_tiles'] for p in positions]
    assert mask[jnp.arange(8),jnp.arange(8)].tolist() == expected
    following,_ = compiled_method(env,'step',batched=True)(batch,jnp.arange(8,dtype=jnp.int32))
    assert following.in_battle.tolist() == expected
    assert following.movement_points.tolist() == [0 if v else 1 for v in expected]
    chex.assert_trees_all_equal(following.position,batch.position)
    for i,before in enumerate(states):
        scalar,_ = step(before,jnp.int32(i))
        chex.assert_trees_all_equal(scalar,jax.tree.map(lambda x,i=i:x[i],following))
    assert jnp.all(env.territory.blocked(jnp.array(ruin['footprint'])))


def test_ruin_dead_missing_copied_leader_and_cleared_entry_rejected(current_game):
    env,s,step,_ = current_game
    base = at_ruin(env,s)
    cases = [base.replace(hp=base.hp.at[1].set(0)),
        base.replace(unit_ids=base.unit_ids.at[1].set(0)),
        base.replace(copied=base.copied.at[1].set(True)),
        base.replace(summon_owner=base.summon_owner.at[1].set(0)),
        base.replace(movement_points=jnp.int32(0)),
        base.replace(ruin_looted=jnp.array([True]),alive=base.alive.at[45].set(False))]
    mask = compiled_method(env,'action_mask')
    for before in cases:
        assert not mask(before)[0]
        after,_ = step(before,jnp.int32(0))
        assert not after.in_battle
        for key in ('position','hp','potions','item_inventory','gold','ruin_looted','movement_points','alive'):
            chex.assert_trees_all_equal(getattr(after,key),getattr(before,key))
    assert mask(base)[0]


@pytest.mark.core
def test_ruin_victory_loot_once_strength_effect_and_reset(current_game):
    env,s,step,fight = current_game
    before = at_ruin(env,s)
    won,reward = finish(env,before,fight)
    strength = env.potion_rules.keys.index('strength')
    # 1 for the battle and 0.25 for the ruin, minus the 2026-10-16 shaping penalty
    # for each fighter whose victory XP is capped by an unbuilt upgrade building.
    blocked = int(jnp.sum(env.progression.blocked_xp_observation(won)))
    assert reward == pytest.approx(1.25-BLOCKED_XP_PENALTY*blocked) and won.last_ruin == 0
    assert won.ruin_looted.tolist() == [True] and not won.alive[45]
    assert won.potions[strength] == 1 and won.last_loot[strength] == 1
    assert won.gold == before.gold and not won.in_battle
    chex.assert_trees_all_equal(won.position,before.position)
    assert not compiled_method(env,'action_mask')(won)[0]
    twice,bonus = jax.jit(env.ruins.grant)(won,True)
    assert bonus == 0
    chex.assert_trees_all_equal(twice.potions,won.potions)
    damaged = compiled_method(env,'unit_stats')(won)[0,DAMAGE]
    used,_ = step(won,jnp.int32(POTION_START+6*strength))
    assert used.potions[strength] == 0
    assert compiled_method(env,'unit_stats')(used)[0,DAMAGE] == round(float(damaged)*1.3)
    rested,_ = step(used,jnp.int32(REST))
    assert compiled_method(env,'unit_stats')(rested)[0,DAMAGE] == damaged
    assert rested.ruin_looted[0]
    reset,_ = compiled_method(env,'reset')(jax.random.PRNGKey(8))
    assert not reset.ruin_looted[0] and reset.alive[45] and reset.potions[strength] == 0


def test_ruin_no_loot_before_victory_or_on_defeat_and_terminal_loot(current_game):
    env,s,step,fight = current_game
    start,_ = step(at_ruin(env,s),jnp.int32(0))
    assert start.in_battle and not start.ruin_looted[0] and not jnp.any(start.last_loot)
    assert start.movement_points == 10  # half the full 20-point allowance
    escaped = start.replace(actor=jnp.int32(0),escaped=jnp.arange(12)<6,retreating=jnp.arange(12)<6)
    withdrawn,_ = fight(escaped,jnp.int32(CONTINUE),escaped.battle_key,jnp.zeros(env.random_size))
    assert not withdrawn.in_battle and not withdrawn.ruin_looted[0] and withdrawn.alive[45]
    chex.assert_trees_all_equal(withdrawn.potions,s.potions)
    dead = start.replace(actor=jnp.int32(7),hp=start.hp.at[:6].set(0))
    lost,_ = fight(dead,jnp.int32(CONTINUE),dead.battle_key,jnp.zeros(env.random_size))
    assert lost.lost and not lost.ruin_looted[0]
    chex.assert_trees_all_equal(lost.potions,s.potions)
    last = at_ruin(env,s).replace(alive=jnp.zeros_like(s.alive).at[45].set(True),city_owned=jnp.ones(2,bool))
    won,reward = finish(env,last,fight)
    assert won.won and won.done and won.ruin_looted[0] and reward == pytest.approx(4.25)
    assert won.potions[env.potion_rules.keys.index('strength')] == 1
    # The hero may fall during a victorious fight; surviving companions keep loot.
    leaderless = at_ruin(env,s).replace(hp=s.hp.at[1].set(0))
    survived,_ = finish(env,leaderless,fight)
    assert survived.ruin_looted[0] and not survived.hp[1] and not survived.lost


def test_ruin_generic_equipment_loot_capacity_and_auto_equip(current_game):
    env,s,_,_ = current_game
    config = copy.deepcopy(MAP)
    config['ruins'][0].update(items={'runestone':1,'boots_speed':1},gold=125)
    items = ItemRules(config)
    rules = RuinRules(config,env.potion_rules,items,env.progression,env.site_rules)
    other = copy.copy(env)
    other.item_rules = items
    before = s.replace(enemy=jnp.int32(45),item_inventory=items.initial_inventory)
    @jax.jit
    def collect(state,victory):
        state,reward = rules.grant(state,victory)
        return other._refresh_equipment(state),reward
    untouched,bonus = collect(before,False)
    assert untouched.gold == 0 and bonus == 0 and jnp.all(untouched.item_inventory<0)
    result,bonus = collect(before,True)
    assert items.capacity == env.item_rules.capacity+2 and items.chest_loot.shape == env.item_rules.chest_loot.shape
    assert result.gold == 125 and bonus == .25
    assert result.equipped[0] == items.keys.index('runestone')
    assert result.equipped[4] == items.keys.index('boots_speed')
    assert other.movement_cap(result) > other.movement_cap(before)
    assert result.last_item_loot[items.keys.index('runestone')] == 1
    assert result.last_item_loot[items.keys.index('boots_speed')] == 1
    twice,_ = collect(result,True)
    chex.assert_trees_all_equal(twice.item_inventory,result.item_inventory)
    assert twice.gold == 125


def test_ruin_fear_pins_guardians_without_city_armor(current_game):
    env,s,_,fight = current_game
    caster = hero_roster_state(env,s,['baroness','duke',None,None,None,None])
    battle = compiled_method(env,'_begin_battle')(caster.replace(enemy=jnp.int32(45))).replace(actor=jnp.int32(0))
    assert battle.battle_fort_armor == 0
    hit,_ = fight(battle,jnp.int32(SHOOT+1),battle.battle_key,jnp.zeros(env.random_size))
    assert hit.paralyzed[7] and not hit.retreating[7] and not hit.feared[7]


def test_ruin_validation_and_observable_metadata(current_game,human_service):
    env,s,_,_ = current_game
    for position in ([0,0],[5,10],[8,23]):
        invalid = copy.deepcopy(MAP)
        invalid['ruins'][0]['position'] = position
        with pytest.raises(ValueError):
            RuinRules(invalid,env.potion_rules,env.item_rules,env.progression,env.site_rules)
    with pytest.raises(ValueError):
        PotionRules(dict(initial_potions={},ruins=[dict(potions={'unknown':1})]))
    observation = compiled_method(env,'observation')
    assert observation(s).shape == (1844,) and env.num_actions == 244
    assert not jnp.array_equal(observation(s),observation(s.replace(ruin_looted=jnp.array([True]))))
    game = human_service.create(42,'legions')
    assert game['ruins']['ruins'][0]['enemy'] == 45
    assert game['ruins']['ruins'][0]['potions'] == {'strength':1}
    assert game['snapshot']['state']['ruin_looted'] == [False]
