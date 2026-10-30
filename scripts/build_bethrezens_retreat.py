"""Convert the pinned Steam S101 snapshot into a JAX campaign.

No environment, JAX backend, original game, or scenario events are executed.
See docs/BETHREZENS_RETREAT.md for source coordinates and adaptations.
"""
import argparse
import ast
from collections import deque
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_default_map import CURRENT, ROOT, RULES, UNITS, cell, dump

SNAPSHOT = ROOT/'docs/maps/bethrezens_retreat/layout.json'
RESTRICTIONS = SNAPSHOT.parent/'restrictions.json'
PLAYER = 'S101PL0001'
HERO = 'S101KC0013'
UNIT_KEYS = {row['game_id'].upper(): key for key, row in UNITS.items()}


def item_catalog():
    """Read catalogue literals without importing modules that initialize JAX."""
    result = {}
    for filename, factory, category in (('number_grid_items.py', '_item', 'items'),
                                        ('number_grid_potions.py', '_p', 'potions')):
        tree = ast.parse((ROOT/'stoix/envs'/filename).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == factory
                    and isinstance(node.args[0], ast.Constant)):
                key, gid = ast.literal_eval(node.args[0]), ast.literal_eval(node.args[2])
                result['G000IG'+gid] = (category, key)
        for node in ast.walk(tree):
            if (isinstance(node, ast.GeneratorExp) and isinstance(node.elt, ast.Call)
                    and isinstance(node.elt.func, ast.Name) and node.elt.func.id == '_item'):
                for index, row in enumerate(ast.literal_eval(node.generators[0].iter.args[0]), 7001):
                    result['G000IG'+str(index)] = ('items', row[0])
    tree = ast.parse((ROOT/'stoix/envs/number_grid_battle_items.py').read_text(encoding='utf-8'))
    literals = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)}
    for index, row in enumerate(literals['SUMMONS'].elts, 1):
        key = ast.literal_eval(row.elts[0])
        for suffix, prefix in (('orb', '90'), ('talisman', '91')):
            result[f'G000IG{prefix}{index:02d}'] = ('items', key+'_'+suffix)
    for row in literals['EFFECTS'].elts:
        key, index = ast.literal_eval(row.elts[0]), ast.literal_eval(row.elts[2])
        for suffix, prefix in (('orb', '90'), ('talisman', '91')):
            result[f'G000IG{prefix}{index:02d}'] = ('items', suffix+'_of_'+key)
    for row in literals['ORB_ONLY'].elts:
        key, gid = ast.literal_eval(row.elts[0]), ast.literal_eval(row.elts[2])
        result[gid] = ('items', 'orb_of_'+key)
    result['G000IG9141'] = ('items', 'talisman_of_nightmare')
    spells = json.loads((ROOT/'stoix/envs/data/spell_items.json').read_text(encoding='utf-8'))
    for row in spells['items']:
        result[row['game_id'].upper()] = ('items', row['key'])
    return result


def formation(army):
    """Native even/odd front/rear pairs become JAX front 0..2, rear 3..5."""
    roster, levels = [None]*6, [0]*6
    occupied = set()
    for unit in army['units']:
        cells = unit['cells']
        if occupied.intersection(cells) or len(cells) != (2 if unit['big'] else 1):
            raise ValueError('Invalid native formation: '+army['source_id'])
        occupied.update(cells)
        first = min(cells)
        if unit['big'] and (first % 2 or sorted(cells) != [first, first+1]):
            raise ValueError('Large unit must occupy a front/rear pair')
        slot = first//2 + 3*(first % 2)
        key = UNIT_KEYS[unit['game_id'].upper()]
        roster[slot] = key
        levels[slot] = max(UNITS[key]['level'], unit['level'])
    return roster, levels


def footprint(obj):
    x, y = obj['position']
    size = obj['size']
    width, height = (size, size) if isinstance(size, int) else size
    return {(x+dx, y+dy) for dx in range(width) for dy in range(height)}


def build(snapshot):
    raw, catalog = snapshot['raw_tables'], item_catalog()
    economy = json.loads((SNAPSHOT.parent/'economy.json').read_text(encoding='utf-8'))
    restrictions = json.loads(RESTRICTIONS.read_text(encoding='utf-8'))
    limits = restrictions['scenario_limits']
    item_types = {r['ITEM_ID']: r['ITEM_TYPE'] for r in raw['SItem.DBF']}

    def loot(gids):
        result = dict(potions={}, items={})
        for gid in gids:
            if gid == 'G000000000':
                continue
            category, key = catalog[gid.upper()]
            result[category][key] = result[category].get(key, 0)+1
        return result

    def inventory(source_id):
        return loot(item_types[r['ITEM_ID']] for r in raw['SInventr.DBF'] if r['INV_ID'] == source_id)

    enemies = [a for a in snapshot['armies'] if a['owner'] != PLAYER]
    hero = next(a for a in snapshot['armies'] if a['source_id'] == HERO)
    home = next(f for f in snapshot['forts'] if f['owner'] == PLAYER)
    rosters, levels = zip(*(formation(a) for a in enemies))
    hero_roster, _ = formation(hero)
    # Object geometry follows the existing JAX campaign convention: all building
    # tiles blocked except a fort's schematic entrance; ruins/sites own their 3x3.
    blocked = {(x, y) for y, row in enumerate(snapshot['terrain']) for x, value in enumerate(row)
               if (value >> 3) & 7 == 4}
    for landmark in snapshot['landmarks']:
        blocked.update(footprint(landmark))
    for fort in snapshot['forts']:
        blocked.update(footprint(fort))
        blocked.discard(tuple(fort['marker_position']))
    managed = set().union(*(footprint(r) for r in snapshot['ruins']),
                          *(footprint(dict(s, size=3)) for s in snapshot['sites']))
    blocked -= managed
    terrain = {}
    for name, kind in (('forest', 1), ('water', 3)):
        terrain[name] = [cell((x, y)) for y, row in enumerate(snapshot['terrain'])
                         for x, value in enumerate(row) if (value >> 3) & 7 == kind
                         and (x, y) not in blocked | managed]
    terrain['road'] = [cell(p) for p in snapshot['roads'] if tuple(p) not in blocked | managed]
    # Road entries take precedence over underlying forest/water in TerrainRules.
    roads = {tuple(p) for p in terrain['road']}
    for name in ('forest', 'water'):
        terrain[name] = [p for p in terrain[name] if tuple(p) not in roads]
    game_map = dict(
        name='NumberGrid-74x74-bethrezens_retreat', display_name='Бегство Бетрезена', size=snapshot['size']+2,
        compact_observations=True,
        agent_position=cell(hero['source_position']), capital_position=cell(home['marker_position']),
        agent_number=1, opponent_positions=[cell(a['marker_position']) for a in enemies],
        opponent_numbers=[1]*len(enemies), **{k: CURRENT[k] for k in RULES},
        faction='legions', battle_items=True, hero_skills=True,
        hero_roster=hero_roster, hero_units=sum(k is not None for k in hero_roster), hero_hp=300,
        hero_combat_stats=[dict(name='Бетрезен', hero=True) if k == 'uter_betrezen' else {}
                           for k in hero_roster if k],
        hero_leadership_capacity=restrictions['leadership']['displayed_companion_capacity'],
        enemy_rosters=list(rosters), enemy_initial_levels=list(levels),
        enemy_units=[sum(k is not None for k in roster) for roster in rosters],
        enemy_hp=[max(UNITS[k]['max_hp'] for k in roster if k) for roster in rosters],
        starting_gold=300, starting_built_buildings=[], starting_hero_abilities=[],
        initial_potions=inventory(HERO)['potions'], initial_items=inventory(HERO)['items'],
        obstacles=sorted(cell(p) for p in blocked), terrain=terrain, allow_water_objects=True,
        campaign_objective='reach_position', objective_position=cell(snapshot['goal']['position']),
        # User-approved adaptation (2026-10-16): reach the centre of the native
        # exit zone, after the ambush has had an enemy turn. Native S101 itself
        # uses radius 3 and has no response prerequisite; preserve it in raw data.
        objective_radius=0, objective_after_event='S101EV0008',
        objective_hero='uter_betrezen', objective_position_reward=102., reward_profile='python_reference',
        scheduled_enemies=[],
        cities=[dict(name=f['name'], position=cell(f['marker_position']), level=f['level'], objective=False)
                for f in snapshot['forts'] if f['kind'] == 'city'],
        mines=[dict(position=cell(r['position']), kind=('gold', 'infernal', 'life', 'death', 'runes', 'elves')[r['resource']])
               for r in snapshot['resources']],
        chests=[dict(position=cell((int(r['POS_X']), int(r['POS_Y']))), **inventory(r['BAG_ID']))
                for r in raw['SBag.dbf']],
        ruins=[dict(name=r['name'], position=cell(r['position']), entrance=cell(r['marker_position']),
                    gold=r['gold'], **loot([next(v['ITEM'] for v in raw['SRuin.DBF'] if v['RUIN_ID'] == r['source_id'])]))
               for r in snapshot['ruins']],
        merchants=[], spell_shops=[], enemy_capitals=[],
        reference_enemy_ids=[a['source_id'] for a in enemies],
        reference_source='Steam Rise of the Elves Scens/S101; docs/maps/bethrezens_retreat/layout.json',
        reconstruction_note='Static original placement before intro events; see docs/BETHREZENS_RETREAT.md.',
    )
    game_map.update(lord_type='warrior', step_cost=0., exploration_bonus=.08, building_reward=0.,
                    equipment_requires_skills=True,
                    max_building_level=limits['MAX_UNIT'], max_spell_level=limits['MAX_SPELL'],
                    max_hero_level=limits['MAX_LEADER'],
                    battle_observation_version=max(44, CURRENT['battle_observation_version']))
    for site in snapshot['sites']:
        entry = dict(name=site['name'], position=cell(site['position']))
        if site['kind'] == 'merchant':
            gids = [r['ITEM_ID'] for r in raw['SMerch.DBF'] if r['SITE_ID'] == site['source_id']
                    for _ in range(int(r['ITEM_COUNT']))]
            entry.update(loot(gids))
            entry['prices'] = {catalog[gid][1]: int(economy['items'][gid].split(':')[0][1:]) for gid in gids}
            game_map['merchants'].append(entry)
        elif site['kind'] == 'mage':
            entry['spells'] = [dict(spell=r['SPELL_ID'].lower(), stock=1,
                                   price=int(economy['spells'][r['SPELL_ID']].split(':')[0][1:]))
                               for r in raw['SMage.DBF'] if r['SITE_ID'] == site['source_id']]
            game_map['spell_shops'].append(entry)
        else:
            raise ValueError('Unsupported site: '+site['kind'])
    for fort in snapshot['forts']:
        if fort['kind'] == 'capital' and fort['owner'] != PLAYER:
            guards = [i for i, a in enumerate(enemies) if a['building'] == fort['code']]
            game_map['enemy_capitals'].append(dict(name=fort['name'], position=cell(fort['marker_position']),
                guard=guards[0], guards=guards, footprint=sorted(cell(p) for p in footprint(fort))))
    from scripts.build_bethrezens_retreat_events import add_events
    add_events(game_map, snapshot, catalog, UNIT_KEYS, UNITS)
    validate_geometry(game_map)
    return game_map


def validate_geometry(game_map):
    """Static JSON geometry validation, without running the environment or a policy."""
    blocked = {tuple(p) for p in game_map['obstacles']}
    reserved = {tuple(game_map[k]) for k in ('agent_position', 'capital_position', 'objective_position') if k in game_map}
    reserved.update(tuple(p) for p in game_map['opponent_positions'])
    reserved.update(tuple(o['position']) for kind in ('cities', 'chests', 'mines') for o in game_map[kind])
    if reserved & blocked:
        raise ValueError(f'Obstacle overlaps an object: {sorted(reserved & blocked)}')
    approaches = []
    for kind in ('ruins', 'merchants', 'spell_shops'):
        for obj in game_map[kind]:
            r, c = obj['position']
            cells = {(r+i, c+j) for i in range(3) for j in range(3)}
            allowed = {tuple(obj['entrance'])} if kind == 'ruins' else set()
            if cells & (blocked | (reserved-allowed)):
                raise ValueError('Building footprint overlaps another object: '+obj['name'])
            blocked.update(cells)
            reserved -= allowed
            if kind == 'ruins':
                candidates = [(r+3, obj['entrance'][1]+dc) for dc in (-1, 0, 1)]
            else:
                candidates = [(r+i, c+j) for i, j in ((1, 3), (2, 3), (3, 1), (3, 2), (3, 3))]
            approaches.append((obj['name'], candidates))
    origin = tuple(game_map['capital_position'])
    reached, queue = {origin}, deque([origin])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            p = (r+dr, c+dc)
            if (all(0 < n < game_map['size']-1 for n in p) and p not in blocked and p not in reached):
                reached.add(p)
                queue.append(p)
    if reserved-reached:
        raise ValueError(f'Objects outside the capital terrain component: {sorted(reserved-reached)}')
    for name, candidates in approaches:
        if not reached.intersection(candidates):
            raise ValueError('No accessible building approach: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--output', type=Path, default=ROOT/'maps/bethrezens_retreat.json')
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
    game_map = build(snapshot)
    args.output.write_text(dump(game_map), encoding='utf-8')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
