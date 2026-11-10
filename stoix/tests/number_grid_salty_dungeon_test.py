"""S103 source, screenshot party and key-delivery contract; CUDA, by request only."""
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, REST, NumberGrid, load_map
from stoix.envs import number_grid_reference_rewards as rewards
from stoix.tests.number_grid_fixtures import compiled_method

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module', autouse=True)
def cuda_only():
    assert jax.default_backend() == 'gpu', 'Run only on request through scripts/run_gpu.sh'


@pytest.fixture(scope='module')
def dungeon():
    env = NumberGrid(map_config=load_map('the_salty_dungeon'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(103))
    return env, state


def test_source_preserves_static_armies_and_contents():
    from scripts.build_salty_dungeon import build
    from scripts.build_bethrezens_retreat import formation
    source = json.loads((ROOT/'docs/maps/the_salty_dungeon/source.json').read_text(encoding='utf-8'))
    config = load_map('the_salty_dungeon')
    assert build(source) == config
    armies = [a for a in source['armies'] if a['owner'] != source['player']]
    assert len(armies) == 97
    assert config['reference_enemy_ids'] == [a['source_id'] for a in armies]
    for army, roster, levels in zip(armies, config['enemy_rosters'], config['enemy_initial_levels']):
        assert formation(army) == (roster, levels)
    assert len(config['cities']) == 8 and len(config['ruins']) == 7
    assert len(config['chests']) == 37 and len(config['mines']) == 14
    assert len(config['merchants']) == 3 and len(config['spell_shops']) == 1
    assert config['scheduled_enemies'] == []
    assert {e['id'] for e in config['scenario_events']} == {
        'S103EV000D', 'S103EV000E', 'S103EV0010', 'S103EV0014', 'S103EV001B'}
    assert len(config['scenario_order_armies']) == 11
    assert all(m['moves'] == 20 for m in config['scenario_order_armies'])
    assert source['goal']['ruin'] == 'S103RU0005'
    assert config['objective_position'] == [11, 40]
    assert config['objective_radius'] == 1


def test_screenshot_party_and_capital_potions(dungeon):
    env, state = dungeon
    assert state.position.tolist() == [11, 16]
    assert state.unit_levels[:6].tolist() == [1, 4, 2, 1, 0, 0]
    assert state.hp[:6].tolist() == [64, 195, 150, 45, 0, 0]
    assert env.max_hp(state)[:6].tolist() == [120, 195, 150, 45, 0, 0]
    assert int(state.potions[env.potion_rules.keys.index('restoration')]) == 2
    assert int(state.potions[env.potion_rules.keys.index('life')]) == 1
    assert int(jnp.sum(state.potions)) == 3
    assert int(state.gold) == 650 and state.mana.tolist() == [25, 0, 0, 0, 0]
    assert env.recruitment.composition(state)[:3].tolist() == [4, 4, 0]
    assert env.script_events.enabled and env.garrisons.enabled
    assert not env.scenario.enabled and not env.bot.enabled
    capital = env.garrisons.capital_index
    assert int(state.garrison_ids[capital, 4]) == env.garrisons.guardian  # Ashkael, rear middle
    assert int(state.garrison_hp[capital, 4]) == 900 and int(jnp.sum(state.garrison_ids != 0)) == 1
    assert not env.territory.objective_cities
    assert not any(s['key'] == 'endurance' for s in env.progression.rows[int(state.unit_ids[1])]['hero_skills'])


def test_gate_requires_key_and_living_permanent_hero(dungeon):
    env, initial = dungeon
    won = compiled_method(env, '_campaign_won')
    at_gate = initial.replace(position=env.objective_position)
    assert not bool(won(at_gate))
    carried = at_gate.replace(item_inventory=at_gate.item_inventory.at[0].set(env.objective_item))
    assert bool(won(carried))
    for state in (
        initial.replace(alive=jnp.zeros_like(initial.alive)),
        at_gate.replace(alive=jnp.zeros_like(initial.alive)),
        carried.replace(position=initial.position),
        carried.replace(hp=carried.hp.at[1].set(0)),
        carried.replace(copied=carried.copied.at[1].set(True)),
        carried.replace(summon_owner=carried.summon_owner.at[1].set(0)),
        carried.replace(in_battle=jnp.bool_(True)),
        carried.replace(summon_battle=jnp.bool_(True)),
        carried.replace(pending_encounters=carried.pending_encounters.at[0].set(True)),
    ):
        assert not bool(won(state))
    assert bool(env._goals_left(jnp.zeros_like(initial.alive), initial.scenario_spawned))
    states = jax.tree.map(lambda *xs: jnp.stack(xs), at_gate, carried)
    assert jax.jit(jax.vmap(env._campaign_won))(states).tolist() == [False, True]


def test_key_loot_and_delivery_rewards_are_separate_and_once(dungeon):
    env, initial = dungeon
    index = next(i for i, r in enumerate(env.ruins.ruins) if r['items'].get('runic_key'))
    ruin = env.ruins.ruins[index]
    before = initial.replace(position=jnp.array(ruin['interaction_tiles'][0], jnp.int32),
                             enemy=jnp.int32(ruin['enemy']))
    grant = jax.jit(env.ruins.grant)
    after, _ = grant(before, jnp.bool_(True))
    assert bool(after.ruin_looted[index])
    assert bool(env._has_objective_item(after))
    assert not bool(env._campaign_won(after)) and not bool(after.done)
    paid, bonus = rewards.finish(env, before, after)
    assert float(bonus) == pytest.approx(100.)
    repeated, _ = grant(paid, jnp.bool_(True))
    assert float(rewards.finish(env, paid, repeated)[1]) == 0.
    assert int(jnp.sum(repeated.item_inventory == env.objective_item)) == 1
    at_gate = paid.replace(position=env.objective_position)
    victory = env._campaign_won(at_gate)
    finished = at_gate.replace(done=victory, won=victory)
    final, bonus = rewards.finish(env, paid, finished)
    assert float(bonus) == pytest.approx(102.)
    assert float(rewards.finish(env, final, final)[1]) == 0.
    timeout = initial.replace(done=jnp.bool_(True))
    assert float(rewards.finish(env, initial, timeout)[1]) == -8.
    assert float(rewards.finish(env, initial, timeout.replace(lost=jnp.bool_(True)))[1]) == -6.


def test_key_is_not_sold_and_cache_requires_all_defenders(dungeon):
    env, initial = dungeon
    merchant = next(s for s in env.site_rules.sites if s['kind'] == 'merchant')
    state = initial.replace(position=jnp.array(merchant['interaction_tiles'][0], jnp.int32),
                            item_inventory=initial.item_inventory.at[0].set(env.objective_item))
    sold = jax.jit(env.site_rules.autosell)(state)
    assert bool(env._has_objective_item(sold))
    assert int(sold.gold) == int(initial.gold)
    i = next(i for i, chest in enumerate(env.map_config['chests']) if len(chest.get('guards', [])) > 1)
    chest = env.map_config['chests'][i]
    visiting = initial.replace(position=jnp.array(chest['position'], jnp.int32))
    collect = jax.jit(env.chest_rules.collect)
    for guard in chest['guards']:
        alive = jnp.zeros_like(initial.alive).at[guard].set(True)
        assert bool(collect(visiting.replace(alive=alive), jnp.bool_(True)).chest_alive[i])
    collected = collect(visiting.replace(alive=jnp.zeros_like(initial.alive)), jnp.bool_(True))
    assert not bool(collected.chest_alive[i])
    assert collect(collected, jnp.bool_(True)).potions.tolist() == collected.potions.tolist()


def enemy_index(env, gid):
    return env.map_config['reference_enemy_ids'].index(gid)


HUMANS = ('S103KC0057', 'S103KC0058', 'S103KC0059', 'S103KC005A')


def humans_of(env):
    return jnp.array([enemy_index(env, gid) for gid in HUMANS])


def test_humans_start_once_and_walk_to_capital_not_remote_hero(dungeon):
    env, initial = dungeon
    rules = env.script_events
    humans = humans_of(env)
    assert jnp.all(initial.script_orders[jnp.array([rules.order_slot[int(i)] for i in humans])] == 6)
    assert jnp.array_equal(initial.enemy_positions, jnp.array(env.map_config['opponent_positions']))
    assert int(jnp.sum(initial.script_events_fired)) == 1
    # Changing a distant hero's location must not change the capital attackers' route.
    advance = jax.jit(rules.advance)
    a = advance(initial.replace(position=jnp.array([1, 1], jnp.int32)))
    b = advance(initial.replace(position=env.objective_position))
    assert jnp.array_equal(a.enemy_positions[humans], b.enemy_positions[humans])
    distances = rules.distances[11*env.size+16]
    old = initial.enemy_positions[humans]
    new = a.enemy_positions[humans]
    assert jnp.all(distances[new[:, 0]*env.size+new[:, 1]] < distances[old[:, 0]*env.size+old[:, 1]])
    fixed = jnp.array([m['enemy'] for m in env.map_config['scenario_order_armies'] if m['order'] != 0])
    assert jnp.array_equal(initial.enemy_positions[fixed], a.enemy_positions[fixed])
    assert not bool(jnp.any(a.pending_encounters))


def test_dwarf_area_kill_and_day_triggers(dungeon):
    env, initial = dungeon
    rules = env.script_events
    process = jax.jit(lambda s, day: rules.process(env, s, new_day=day))
    group = [enemy_index(env, gid) for gid in ('S103KC002D', 'S103KC002E', 'S103KC006B')]
    before = initial.replace(position=jnp.array([57, 41], jnp.int32))
    after = process(before, jnp.bool_(False))
    assert all(int(after.script_orders[rules.order_slot[i]]) == 4 for i in group)
    assert jnp.array_equal(after.enemy_positions, before.enemy_positions)
    repeat = process(after, jnp.bool_(False))
    assert jnp.array_equal(repeat.script_events_fired, after.script_events_fired)
    target = enemy_index(env, 'S103KC0072')
    killed = enemy_index(env, 'S103KC0052')
    after_kill = process(initial.replace(alive=initial.alive.at[killed].set(False)), jnp.bool_(False))
    assert int(after_kill.script_orders[rules.order_slot[target]]) == 4
    alderic = rules.order_slot[group[0]]
    assert int(process(initial.replace(day=jnp.int32(19)), jnp.bool_(True)).script_orders[alderic]) == 3
    assert int(process(initial.replace(day=jnp.int32(20)), jnp.bool_(True)).script_orders[alderic]) == 4
    dead = before.replace(alive=before.alive.at[group[0]].set(False))
    assert not bool(process(dead, jnp.bool_(False)).alive[group[0]])


def test_undead_ambush_fights_during_entry_without_rest(dungeon):
    env, initial = dungeon
    first = enemy_index(env, 'S103KC0014')
    others = [enemy_index(env, gid) for gid in ('S103KC0016', 'S103KC007A')]
    # Enter the trigger from the east with unrelated zones removed.
    alive = jnp.zeros_like(initial.alive).at[jnp.array([first]+others)].set(True)
    before = initial.replace(position=jnp.array([38, 24], jnp.int32), alive=alive)
    action = env.directions.tolist().index([0, -1])
    assert bool(compiled_method(env, 'action_mask')(before)[action])
    after, _ = compiled_method(env, 'step')(before, jnp.int32(action))
    assert int(after.day) == int(before.day)
    assert int(after.enemy) == first and bool(after.in_battle | after.lost)
    assert all(int(after.script_orders[env.script_events.order_slot[i]]) == 4 for i in [first]+others)
    assert jnp.array_equal(after.enemy_positions[jnp.array(others)], before.enemy_positions[jnp.array(others)])


def test_pursuers_attack_on_rest_and_ignore_dead_armies(dungeon):
    env, initial = dungeon
    enemy = enemy_index(env, 'S103KC002D')
    # An open approach immediately outside the capital, within one day's budget.
    before = initial.replace(alive=jnp.zeros_like(initial.alive).at[enemy].set(True),
        enemy_positions=initial.enemy_positions.at[enemy].set(jnp.array([12, 17])),
        script_orders=initial.script_orders.at[env.script_events.order_slot[enemy]].set(4))
    after, _ = compiled_method(env, 'step')(before, jnp.int32(REST))
    assert int(after.day) == 2 and bool(after.in_battle | after.lost)
    assert int(after.enemy) == enemy
    dead = jax.jit(env.script_events.advance)(before.replace(alive=jnp.zeros_like(before.alive)))
    assert not bool(jnp.any(dead.pending_encounters))
    assert jnp.array_equal(dead.enemy_positions, before.enemy_positions)


def test_capital_garrison_queues_arrivals_without_the_hero(dungeon):
    env, initial = dungeon
    humans = humans_of(env)
    capital = env.garrisons.positions[env.garrisons.capital_index]
    before = initial.replace(position=env.objective_position,
        enemy_positions=initial.enemy_positions.at[humans].set(capital))
    after = jax.jit(lambda s: env.garrisons.advance(s))(before)
    # Every human army waits for a garrison battle the agent plays; nothing is resolved yet.
    assert bool(jnp.all(after.garrison_pending[humans])) and not bool(after.lost | after.done)
    for name in ('hp', 'unit_ids', 'alive', 'garrison_hp', 'battle_key', 'position'):
        assert jnp.array_equal(getattr(after, name), getattr(before, name)), name
    legal = compiled_method(env, 'action_mask')(after)
    assert bool(legal[CONTINUE]) and int(jnp.sum(legal)) == 1
    battle, _ = compiled_method(env, 'step')(after, jnp.int32(CONTINUE))
    assert bool(battle.in_battle & battle.garrison_battle) and int(battle.enemy) in humans.tolist()
    assert int(battle.unit_ids[4]) == env.garrisons.guardian and int(battle.battle_hero_fort_armor) == 50


def test_capital_present_hero_uses_normal_encounter_queue(dungeon):
    env, initial = dungeon
    humans = humans_of(env)
    capital = env.garrisons.positions[env.garrisons.capital_index]
    before = initial.replace(enemy_positions=initial.enemy_positions.at[humans].set(capital))
    moved = jax.jit(env.script_events.advance)(before)
    assert bool(jnp.all(moved.pending_encounters[humans]))
    after = jax.jit(lambda s: env.garrisons.advance(s))(moved)
    assert not bool(jnp.any(after.garrison_pending))  # the party defends first
    assert jnp.array_equal(after.hp, moved.hp)


def test_capital_loss_and_vmapped_inactive_lane(dungeon):
    env, initial = dungeon
    g = env.garrisons
    enemy = int(humans_of(env)[0])
    # A capital with a dead guardian and no other defender falls at once.
    before = initial.replace(position=env.objective_position,
        garrison_hp=initial.garrison_hp.at[g.capital_index].set(0),
        enemy_positions=initial.enemy_positions.at[enemy].set(g.positions[g.capital_index]))
    batch = jax.tree.map(lambda x: jnp.stack((x, x)), before)
    result = jax.jit(jax.vmap(lambda s, active: g.advance(s, active)))(batch, jnp.array([True, False]))
    assert bool(result.lost[0] & result.done[0])
    assert not bool(result.lost[1]) and not bool(jnp.any(result.garrison_pending[1]))


@pytest.mark.parametrize('kind,steps', [(0, 10), (1, 20), (2, 5), (3, 4)])
def test_twenty_movement_points_in_straight_corridor(dungeon, kind, steps):
    env, initial = dungeon
    rules = copy.copy(env.script_events)
    rules.terrain = copy.copy(env.terrain)
    rules.size = rules.terrain.size = 32
    rules.terrain.grid = jnp.full(32*32, kind, jnp.int32)
    # A single straight reachable corridor has an unambiguous shortest route.
    columns = jnp.arange(1, 31)
    rules.distances = jnp.full((32*32, 32*32), 65535, jnp.uint16).at[
        32+25, 32+columns].set(jnp.abs(columns-25).astype(jnp.uint16))
    enemy = enemy_index(env, 'S103KC006B')  # No flying leader: ordinary terrain costs.
    before = initial.replace(position=jnp.array([1, 25], jnp.int32),
        alive=jnp.zeros_like(initial.alive).at[enemy].set(True),
        script_orders=jnp.zeros_like(initial.script_orders).at[rules.order_slot[enemy]].set(4),
        enemy_positions=initial.enemy_positions.at[enemy].set(jnp.array([1, 1])))
    after = jax.jit(rules.advance)(before)
    assert after.enemy_positions[enemy].tolist() == [1, 1+steps]
    assert not bool(jnp.any(after.pending_encounters))
