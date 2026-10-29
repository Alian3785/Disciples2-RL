"""CUDA contracts: reference research rules and original ruler bonuses."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.envs.number_grid import MAP, NumberGrid, ACTION_NAMES, BUILD_START, REST
from stoix.envs.number_grid_buildings import BuildingRules, FACTIONS
from stoix.envs.number_grid_lords import LORDS, STARTING_PARTIES
from stoix.envs.number_grid_spells import CATALOG, SPELL_LEARNED, SpellResearchRules, research_action_names
from stoix.envs.number_grid_casting import cast_action_names
from stoix.envs.number_grid_territory import TerritoryRules
from stoix.tests.number_grid_fixtures import compiled_method, replace_base_state


@pytest.mark.core
def test_catalog_is_exact_supported_reference_subset_and_preserves_original_text():
    assert {f:len(rows) for f,rows in CATALOG['factions'].items()} == {
        'empire':22,'mountain_clans':22,'legions':17,'undead_hordes':20,'elves':20}
    # spells.py has 24 entries per race; these 19 are absent from the reference supported specs.
    excluded = {'emp':{9,20},'mcl':{10,22},'lod':{2,7,11,12,13,22,24},
                'und':{8,14,19,20},'elf':{3,11,23,24}}
    prefixes = {'empire':'emp','mountain_clans':'mcl','legions':'lod','undead_hordes':'und','elves':'elf'}
    for faction,prefix in prefixes.items():
        expected = {f'{prefix}_d2_s{i:03d}' for i in range(1,25) if i not in excluded[prefix]}
        assert {r['reference_id'] for r in CATALOG['factions'][faction]} == expected
    rows = [row for faction in CATALOG['factions'].values() for row in faction]
    assert len(rows) == len({r['id'] for r in rows}) == len({r['reference_id'] for r in rows}) == 101
    assert all(r['original_description'] and r['description'] and r['original_name'] for r in rows)
    assert all(1 <= r['level'] <= 5 and len(r['research_mana']) == 5 for r in rows)
    assert all(sum(r['research_mana']) > 0 and all(x >= 0 for x in r['research_mana']) for r in rows)
    # DBF GspellR: Great Chronos uses infernal/death/runes; reference wrongly uses life.
    chronos = next(r for r in rows if r['id'] == 'g000ss0091')
    assert chronos['research_mana'] == [1200,0,600,600,0]
    assert len(ACTION_NAMES) == 244 and ACTION_NAMES[195:212] == research_action_names(MAP)
    assert ACTION_NAMES[212:229] == cast_action_names(MAP) == tuple('cast_'+n[6:] for n in research_action_names(MAP))


@pytest.mark.parametrize('faction', FACTIONS)
@pytest.mark.parametrize('lord', LORDS)
def test_every_spell_exact_mana_all_gates_and_compact_faction_actions(current_game, faction, lord):
    _,initial,_,_ = current_game
    config = {**MAP,'faction':faction,'lord_type':lord}
    buildings = BuildingRules(config)
    rules = SpellResearchRules(config,buildings,195)
    names = research_action_names(config)
    assert len(names) == rules.count and rules.end == 195+rules.count
    assert names == tuple('learn_'+r['id'] for r in rules.rows)
    assert all(r['cost'] == [x//LORDS[lord]['research_divisor'] for x in r['research_mana']]
               for r in rules.metadata['spells'])
    states,actions,expected = [],[],[]
    for i,row in enumerate(rules.rows):
        ready = initial.replace(buildings=rules.tower_bit,mana=rules.costs[i])
        allowed = row['level'] <= LORDS[lord]['max_spell_level']
        cases = [(ready,allowed), (ready.replace(buildings=jnp.uint32(0)),False),
                 (ready.replace(learned_spells=rules.bits[i]),False),
                 (ready.replace(spell_researched_today=jnp.bool_(True)),False),
                 (ready.replace(in_battle=jnp.bool_(True)),False),
                 (ready.replace(done=jnp.bool_(True)),False)]
        for kind,cost in enumerate(rules.metadata['spells'][i]['cost']):
            if cost:
                cases.append((ready.replace(mana=ready.mana.at[kind].add(-1)),False))
        for state,valid in cases:
            states.append(state)
            actions.append(rules.start+i)
            expected.append(valid)
    batch = jax.tree.map(lambda *xs:jnp.stack(xs),*states)
    actions = jnp.array(actions,jnp.int32)
    @jax.jit
    @jax.vmap
    def evaluate(state,action):
        out,reward = rules.apply(state,action)
        index = action-rules.start
        return (rules.available(state)[index], rules.available(state,index), rules.status(state)[index],
                out.learned_spells,out.mana,out.spell_researched_today,out.last_researched_spell,reward)
    mask,selected,status,learned,mana,locked,last,reward = evaluate(batch,actions)
    np.testing.assert_array_equal(mask,expected)
    np.testing.assert_array_equal(selected,expected)
    np.testing.assert_array_equal(status == 0,expected)
    valid = jnp.array(expected)
    indices = actions-rules.start
    chex.assert_trees_all_equal(learned,batch.learned_spells | jnp.where(valid,rules.bits[indices],jnp.uint32(0)))
    chex.assert_trees_all_equal(mana,batch.mana-jnp.where(valid[:,None],rules.costs[indices],0))
    chex.assert_trees_all_equal(locked,batch.spell_researched_today | valid)
    chex.assert_trees_all_equal(last,jnp.where(valid,indices,-1))
    np.testing.assert_array_equal(reward,expected)


@pytest.mark.core
def test_research_full_step_daily_reset_independent_build_limit_and_observation(current_game):
    env,initial,step,_ = current_game
    rules = env.spell_research
    observe = compiled_method(env,'observation')
    mask = compiled_method(env,'action_mask')
    # No location, movement or living leader gate in the Python reference.
    ready = initial.replace(position=jnp.array([8,9]),movement_points=jnp.int32(0),
        hp=initial.hp.at[1].set(0),buildings=rules.tower_bit,mana=jnp.full(5,5000,jnp.int32))
    assert mask(ready)[rules.start]
    learned,ts = step(ready,jnp.int32(rules.start))
    assert learned.learned_spells == 1 and learned.last_event == SPELL_LEARNED
    assert learned.spell_researched_today and not learned.built_today
    assert float(ts.reward) == pytest.approx(1-env.step_cost)
    for field in ('gold','day','movement_points','hp','unit_ids','unit_levels','position','buildings'):
        chex.assert_trees_all_equal(getattr(learned,field),getattr(ready,field))
    chex.assert_trees_all_equal(learned.mana,ready.mana-rules.costs[0])
    assert not jnp.any(mask(learned)[rules.start:rules.end])
    denied,_ = step(learned,jnp.int32(rules.start+1))
    assert denied.learned_spells == learned.learned_spells
    chex.assert_trees_all_equal(denied.mana,learned.mana)
    assert observe(learned).shape == (1844,) and env.num_actions == 244
    research = observe(learned)[:-1-env.casting.observation_size-env.unit_swaps.observation_size]  # casts, swaps and day follow research
    np.testing.assert_array_equal(research[-rules.count-4:-rules.count-1],[1,0,0])
    np.testing.assert_array_equal(research[-rules.count-1:], [1]+[0]*(rules.count-1)+[1])
    rested,_ = step(learned,jnp.int32(REST))
    assert rested.day == ready.day+1 and not rested.spell_researched_today
    assert rested.learned_spells == 1 and mask(rested)[rules.start+1] and not mask(rested)[rules.start]
    # Research does not spend the construction turn, and construction does not spend research.
    funded = ready.replace(gold=jnp.int32(5000))
    built,_ = step(funded,jnp.int32(BUILD_START))
    assert built.built_today and not built.spell_researched_today and mask(built)[rules.start]
    researched,_ = step(built,jnp.int32(rules.start))
    assert researched.built_today and researched.spell_researched_today
    first,_ = step(funded,jnp.int32(rules.start))
    assert mask(first)[BUILD_START]
    for invalid in (ready.replace(buildings=jnp.uint32(0)),ready.replace(mana=jnp.zeros(5,jnp.int32))):
        assert not mask(invalid)[rules.start]
        out,_ = step(invalid,jnp.int32(rules.start))
        assert out.learned_spells == 0 and not out.spell_researched_today
        chex.assert_trees_all_equal(out.mana,invalid.mana)
    fifth = rules.start+next(i for i,r in enumerate(rules.rows) if r['level']==5)
    assert not mask(ready)[fifth]
    out,_ = step(ready,jnp.int32(fifth))
    assert out.learned_spells == 0
    for action in (-1,env.num_actions,999):
        out,_ = step(ready,jnp.int32(action))
        chex.assert_trees_all_equal(out.mana,ready.mana)
        assert out.learned_spells == 0


def test_lords_free_buildings_regeneration_city_prices_and_reset(current_game):
    env,initial,_,_ = current_game
    for faction in FACTIONS:
        for lord in LORDS:
            config = {**MAP,'faction':faction,'lord_type':lord}
            rules = BuildingRules(config)
            starting = LORDS[lord]['starting_building']
            assert int(rules.initial_built) == (0 if starting is None else
                1 << next(i for i,r in enumerate(rules.rows) if r['name']==starting))
    # Full reset for each new ruler: free building, no gold/day spent, the ruler's own party.
    for lord in ('mage','guildmaster'):
        other = NumberGrid(map_config={**MAP,'lord_type':lord})
        state,ts = compiled_method(other,'reset')(jax.random.PRNGKey(42))
        assert state.buildings == other.construction.initial_built and state.gold == 0
        assert not state.built_today and not state.spell_researched_today and state.learned_spells == 0
        assert state.day == 1 and ts.observation.shape == (1844,)
        rows = other.progression.rows
        assert [rows[i]['key'] if i else None for i in state.unit_ids[:6].tolist()] == STARTING_PARTIES['legions'][lord]
        np.testing.assert_array_equal(ts.observation[:-1-other.casting.observation_size-other.unit_swaps.observation_size][-21:-18],other.construction.lord_observation)
        rules = other.spell_research
        funded = state.replace(mana=jnp.full(5,5000,jnp.int32),buildings=state.buildings | rules.tower_bit)
        mask = compiled_method(other,'action_mask')(funded)
        assert bool(jnp.all(mask[rules.start:rules.end])) == (lord == 'mage')
        fifth = next(i for i,r in enumerate(rules.rows) if r['level']==5)
        out,reward = jax.jit(rules.apply)(funded,jnp.int32(rules.start+fifth))
        assert bool(reward) == (lord=='mage')
        if lord == 'mage':
            chex.assert_trees_all_equal(out.mana,funded.mana-rules.costs[fifth])
    neutral = initial.replace(position=jnp.array([8,9]))
    base_percent = jax.jit(env.territory.regeneration_percent)(neutral)
    for lord in ('mage','guildmaster'):
        rules = TerritoryRules({**MAP,'lord_type':lord},env.progression,193,env.site_rules,env.ruins)
        assert jax.jit(rules.regeneration_percent)(neutral)[0] == base_percent[0]-15
        expected = [0,75,125,250,375,0] if lord=='guildmaster' else [0,150,250,500,750,0]
        assert rules.costs.tolist() == expected
        city = initial.replace(city_owned=jnp.array([True,False]),gold=jnp.int32(expected[1]))
        upgraded = jax.jit(rules.upgrade)(city,jnp.int32(193))
        assert upgraded.city_levels[0] == 2 and upgraded.gold == 0
    with pytest.raises(ValueError,match='lord_type'):
        BuildingRules({**MAP,'lord_type':'unknown'})


def test_research_autoreset_preserves_final_observation_and_changes_prng(current_game,training_autoreset):
    env,initial,_,_ = current_game
    training,_,advance = training_autoreset
    state,_ = training.reset(jax.random.split(jax.random.PRNGKey(7),2))
    state = replace_base_state(state,buildings=jnp.full(2,env.spell_research.tower_bit,jnp.uint32),
                              mana=jnp.full((2,5),5000,jnp.int32))
    before_key = state.battle_key
    state,ts = advance(state,jnp.full(2,env.spell_research.start,jnp.int32))
    assert jnp.all(ts.truncated()) and jnp.all(state.learned_spells == 0)
    assert not jnp.any(state.spell_researched_today) and not jnp.any(state.mana)
    assert not jnp.array_equal(state.battle_key,before_key)
    cast = env.casting.observation_size+env.unit_swaps.observation_size+1  # casts, swaps and day follow research
    assert jnp.all(ts.extras['next_obs']['observation'][:,-18-cast] == 1)
    assert jnp.all(ts.extras['next_obs']['observation'][:,-1-cast] == 1)
    assert jnp.all(ts.observation['observation'][:,-18-cast:-cast] == 0)


def test_human_spell_book_masks_and_ruler_session_isolation(human_service):
    service = human_service
    game = service.create(42,'legions','warrior')
    key,state,total = service.sessions[game['session']]
    env,_,_ = service.environment(*key)
    assert len(game['spell_research']['spells']) == len(game['snapshot']['spell_status']) == 17
    assert game['construction']['lord']['id'] == 'warrior'
    with pytest.raises(ValueError,match='недоступно'):
        service.act(game['session'],env.spell_research.start)
    state = state.replace(buildings=env.spell_research.tower_bit,mana=jnp.full(5,5000,jnp.int32))
    service.sessions[game['session']] = (key,state,total)
    learned = service.act(game['session'],env.spell_research.start)['snapshot']
    assert learned['state']['learned_spells'] == 1 and learned['spell_status'][0] == 1
    assert not any(learned['action_mask'][env.spell_research.start:env.spell_research.end])
    # The learned summon is castable at once on the nearest stack.
    assert learned['action_mask'][env.casting.start] and learned['cast_quotes']['status'][0] == 0
    for lord in ('mage','guildmaster'):
        other = service.create(42,'legions',lord)
        assert other['map']['lord_type'] == lord and other['snapshot']['state']['learned_spells'] == 0
        assert other['map']['hero_roster'] == STARTING_PARTIES['legions'][lord]  # the ruler's own hero
        assert other['snapshot']['state']['buildings'] != 0
        assert not other['snapshot']['state']['built_today']
        assert service.sessions[other['session']][0] == ('legions',lord)
    assert service.sessions[game['session']][1].learned_spells == 1
    with pytest.raises(ValueError,match='тип правителя'):
        service.create(42,'legions','bad')
