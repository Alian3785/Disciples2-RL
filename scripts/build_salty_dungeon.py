"""Build the S103 JAX campaign from its pinned source snapshot.

No JAX backend, training, game process or scenario events are executed.
The player must loot the Runic Key and carry it to the dungeon entrance.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import CURRENT, RULES, UNITS, cell, dump
from scripts.build_bethrezens_retreat import formation, footprint, item_catalog, UNIT_KEYS

SNAPSHOT = ROOT/'docs/maps/the_salty_dungeon/source.json'


def enemy_overrides(army, snapshot):
    """Native names, XP and per-instance modifiers; no automatic player skill ladder."""
    result = []
    for unit in sorted(army['units'], key=lambda u: min(u['cells'])//2+3*(min(u['cells']) % 2)):
        profile = UNITS[UNIT_KEYS[unit['game_id']]]
        override = dict(exp_current=int(unit['source']['XP']))
        # A blank native name means the unit's stock name.
        if unit['custom_name'].strip() and unit['custom_name'] != unit['source_id']:
            override['name'] = unit['custom_name']
        if profile.get('hero'):
            override['level_bonuses'] = []
        effects = [effect for mod in snapshot['raw_tables']['SUnitM']
                   if mod['UNIT_ID'].upper() == unit['source_id']
                   for effect in snapshot['modifiers']
                   if effect['BELONGS_TO'].upper() == mod['MODIF_ID'].upper()]
        growth = json.loads(json.dumps(profile['growth']))
        stat_fields = ('max_hp', 'damage', 'accuracy', 'armor', 'initiative')
        for effect in effects:
            category = int(effect['TYPE'])
            if category == 2:  # Leadership cannot change a fixed native enemy formation.
                continue
            field = {6: 'max_hp', 4: 'damage', 9: 'initiative'}[category]
            factor = 1+int(effect['PERCENT'])/100
            override[field] = int(override.get(field, profile[field])*factor)
            for phase in ('early', 'late'):
                growth[phase][stat_fields.index(field)] *= factor
            override['growth'] = growth
        result.append(override)
    return result


def build(snapshot):
    raw, catalog, player = snapshot['raw_tables'], item_catalog(), snapshot['player']
    instances = {r['ITEM_ID'].upper(): r['ITEM_TYPE'].upper() for r in raw['SItem']}

    def loot(gids):
        result = dict(potions={}, items={})
        for gid in gids:
            if gid == 'G000000000':
                continue
            category, key = catalog[gid]
            result[category][key] = result[category].get(key, 0)+1
        return result

    def inventory(source):
        return loot(instances[r['ITEM_ID'].upper()] for r in raw['SInventr']
                    if r['INV_ID'].upper() == source)

    enemies = [a for a in snapshot['armies'] if a['owner'] != player]
    home = next(f for f in snapshot['forts'] if f['owner'] == player and f['kind'] == 'capital')
    rosters, levels = zip(*(formation(a) for a in enemies))
    blocked = {(x, y) for y, row in enumerate(snapshot['terrain']) for x, code in enumerate(row)
               if (code >> 3) & 7 == 4}
    for mark in snapshot['landmarks']:
        blocked.update(footprint(mark))
    for fort in snapshot['forts']:
        blocked.update(footprint(fort))
        blocked.discard(tuple(fort['marker_position']))
    managed = set().union(*(footprint(r) for r in snapshot['ruins']),
                          *(footprint(dict(s, size=3)) for s in snapshot['sites']))
    blocked -= managed
    terrain = {name: [cell((x, y)) for y, row in enumerate(snapshot['terrain']) for x, code in enumerate(row)
                      if (code >> 3) & 7 == kind and (x, y) not in blocked | managed]
               for name, kind in (('forest', 1), ('water', 3))}
    terrain['road'] = [cell(p) for p in snapshot['roads'] if tuple(p) not in blocked | managed]
    roads = {tuple(p) for p in terrain['road']}
    for name in ('forest', 'water'):
        terrain[name] = [p for p in terrain[name] if tuple(p) not in roads]
    chests = [dict(source_id=r['source_id'], position=cell(r['position']), **inventory(r['source_id']))
              for r in snapshot['chests']]
    # Permanent enemy inventories are guarded caches on the stack/fort tile.
    # They become reachable after its defenders die; no scripted loot event.
    for source in [a for a in enemies if a['kind'] != 'capital']+[
            f for f in snapshot['forts'] if f['kind'] == 'capital' and f['owner'] != player]:
        contents = inventory(source['source_id'])
        if any(contents.values()):
            guards = [i for i, a in enumerate(enemies) if a['marker_position'] == source['marker_position']]
            chests.append(dict(source_id=source['source_id'], position=cell(source['marker_position']),
                               guards=guards, **contents))
    game_map = dict(
        name='NumberGrid-74x74-the_salty_dungeon', display_name='Соляная темница', size=snapshot['size']+2,
        agent_position=cell(home['marker_position']), capital_position=cell(home['marker_position']),
        agent_number=1, opponent_positions=[cell(a['marker_position']) for a in enemies],
        opponent_numbers=[1]*len(enemies), **{k: CURRENT[k] for k in RULES},
        faction='legions', battle_items=True, hero_skills=True,
        # Screenshot: right column is front; a large unit occupies both bottom cells.
        hero_roster=['possessed', 'duke', 'g000uu0056', 'cultist', None, None], hero_units=4,
        hero_initial_levels=[1, 4, 2, 1, 0, 0], hero_initial_hp=[64, 195, 150, 45, 0, 0], hero_hp=195,
        # The pictured Duke has 195 max HP, not 234: he has not taken Endurance.
        hero_combat_stats=[{}, dict(level_bonuses=[b for b in UNITS['duke']['level_bonuses'] if b['level'] != 4]), {}, {}],
        hero_leadership_capacity=3,
        enemy_rosters=list(rosters), enemy_initial_levels=list(levels),
        enemy_combat_stats=[enemy_overrides(a, snapshot) for a in enemies],
        enemy_units=[sum(k is not None for k in roster) for roster in rosters],
        enemy_hp=[max(UNITS[k]['max_hp'] for k in roster if k) for roster in rosters],
        starting_built_buildings=[], starting_hero_abilities=[],
        initial_potions=inventory(home['source_id'])['potions'], initial_items=inventory(home['source_id'])['items'],
        obstacles=sorted(cell(p) for p in blocked), terrain=terrain, allow_water_objects=True,
        campaign_objective='reach_position', objective_position=cell(snapshot['goal']['position']),
        objective_radius=snapshot['goal']['radius'], objective_hero='duke', objective_item='runic_key',
        objective_label='Доставить рунический ключ ко входу в темницу',
        objective_position_reward=102., reward_profile='python_reference', item_pickup_rewards={'runic_key': 100.},
        scheduled_enemies=[],
        # Asteroth remains behind the native sealed prison walls. The requested
        # finish is at the reachable outer gate; the prison-opening event is omitted.
        enforce_enemy_reachability=False,
        cities=[dict(source_id=f['source_id'], name=f['name'], position=cell(f['marker_position']),
                     level=f['level'], objective=False, **inventory(f['source_id']))
                for f in snapshot['forts'] if f['kind'] == 'city'],
        mines=[dict(position=cell(r['position']), kind=('gold', 'infernal', 'life', 'death', 'runes', 'elves')[r['resource']])
               for r in snapshot['resources']],
        chests=chests,
        ruins=[dict(source_id=r['source_id'], name=r['name'], position=cell(r['position']),
                    entrance=cell(r['marker_position']), gold=r['gold'], **loot([r['item']])) for r in snapshot['ruins']],
        merchants=[], spell_shops=[], enemy_capitals=[],
        reference_enemy_ids=[a['source_id'] for a in enemies],
        reference_source='Installed Rise of the Elves Scens/S103; docs/maps/the_salty_dungeon/source.json',
        reconstruction_note='S103 static map with selected pursuit/ambush events and capital defense; victory requires the Runic Key at the gate.',
    )
    info = snapshot['scenario']
    # KC0027's Ghosts and Shade can paralyse a lone Doppelganger copying its weapon-immune
    # Werewolf for 40 rounds (~245 automatic activations). Cap them per step so one lane
    # does not stall the batch; docs/COMBAT.md#automatic-turn-limit.
    game_map.update(battle_automatic_turn_limit=16)
    # Telemetry only: the Mountain Clans' Green Dragon guarding the way into the dwarf lands.
    game_map.update(tracked_enemy=game_map['reference_enemy_ids'].index('S103KC002B'))
    game_map.update(lord_type='warrior', step_cost=0., exploration_bonus=.08, building_reward=0.,
        equipment_requires_skills=True, max_building_level=int(info['MAX_UNIT']),
        max_spell_level=int(info['MAX_SPELL']), max_hero_level=int(info['MAX_LEADER']))
    for site in snapshot['sites']:
        source = site['source_id']
        entry = dict(source_id=source, name=site['name'], position=cell(site['position']))
        if site['kind'] == 'merchant':
            gids = [r['ITEM_ID'].upper() for r in raw['SMerch'] if r['SITE_ID'].upper() == source
                    for _ in range(int(r['ITEM_COUNT']))]
            entry.update(loot(gids))
            entry['prices'] = {catalog[gid][1]: int(snapshot['economy']['items'][gid]['price'][1:5]) for gid in gids}
            game_map['merchants'].append(entry)
        else:
            entry['spells'] = [dict(spell=r['SPELL_ID'].lower(), stock=1,
                price=int(snapshot['economy']['spells'][r['SPELL_ID'].upper()][1:5]))
                for r in raw['SMage'] if r['SITE_ID'].upper() == source]
            game_map['spell_shops'].append(entry)
    for fort in snapshot['forts']:
        if fort['kind'] == 'capital' and fort['owner'] != player:
            guards = [i for i, a in enumerate(enemies) if a['building'] == fort['source_id']]
            game_map['enemy_capitals'].append(dict(name=fort['name'], position=cell(fort['marker_position']),
                guard=guards[0], guards=guards, footprint=sorted(cell(p) for p in footprint(fort))))
    own, neutral = 3, 5
    tiles = [((x, y), code) for y, row in enumerate(snapshot['terrain']) for x, code in enumerate(row)]
    game_map['starting_land'] = sorted(cell(p) for p, code in tiles if code & 7 == own)
    game_map['foreign_land'] = sorted(cell(p) for p, code in tiles if code & 7 not in (own, neutral))
    mountains = set().union(*(footprint(m) for m in snapshot['landmarks'] if m['mountain']))
    game_map['land_blocked'] = sorted(cell(p) for p in mountains | {p for p, code in tiles if code >> 3 in (3, 4)})
    game_map['capital_footprint'] = sorted(cell(p) for p in footprint(home))
    races = {r['PLAYER_ID'].upper(): r['RACE_ID'].upper() for r in raw['Splayer']}
    forts = {f['source_id']: f for f in snapshot['forts']}
    for city in game_map['cities']:
        fort = forts[city['source_id']]
        city.update(footprint=sorted(cell(p) for p in footprint(fort)), neutral=races[fort['owner']] == 'G000RR0004')
    # As for S101, credit the first strategic morning before the first move.
    bank = next(r['BANK'] for r in raw['Splayer'] if r['PLAYER_ID'].upper() == player)
    owned_mines = [r['resource'] for r in snapshot['resources']
                   if snapshot['terrain'][r['position'][1]][r['position'][0]] & 7 == own]
    game_map['starting_gold'] = int(bank[1:5])+100+50*owned_mines.count(0)
    game_map['starting_mana'] = {name: int(part[1:])+25*(index == 1)+50*owned_mines.count(index)
        for index, (name, part) in enumerate(zip(('infernal', 'life', 'death', 'runes'), bank.split(':')[1:]), 1)}
    add_army_events(game_map, snapshot)
    validate_geometry(game_map)
    return game_map



def add_army_events(game_map, snapshot):
    """Compile only the user-selected S103 attacks; all other scripts stay omitted."""
    raw = snapshot['raw_tables']
    ids = game_map['reference_enemy_ids']
    stacks = {r['STACK_ID'].upper(): r for r in raw['Sstack']}
    locations = {r['LOC_ID'].upper(): dict(
        position=cell((int(r['POS_X']), int(r['POS_Y']))), radius=int(r['RADIUS']))
        for r in raw['SLoc']}
    humans = ['S103KC0057', 'S103KC0058', 'S103KC0059', 'S103KC005A']
    selected = {'S103EV000D', 'S103EV000E', 'S103EV0010', 'S103EV0014', 'S103EV001B'}
    events, movers = [], set(humans)
    for native in raw['SEvent']:
        eid = native['ID'].upper()
        if eid not in selected:
            continue
        if native['ENABLED'] != 'T' or native['OCCUR_ONCE'] != 'T' or native['CHANCE'] != '100':
            raise ValueError('Expected enabled deterministic one-shot event: '+eid)
        entry = dict(id=eid, conditions=[], effects=[])
        for row in raw['SEvCond']:
            if row['ID'].upper() != eid:
                continue
            category = int(row['CATEGORY'])
            if category == 0:
                condition = dict(kind='FREQUENCY', day=int(row['FREQUENCY']))
            elif category == 2:
                condition = dict(kind='LOCATION', **locations[row['ID_LOC'].upper()])
            elif category == 5:
                condition = dict(kind='KILL_STACK', enemy=ids.index(row['ID_STACK'].upper()))
            else:
                raise ValueError('Unexpected army-event condition: '+str(category))
            entry['conditions'].append(condition)
        if eid == 'S103EV000E':
            # Requested adaptation: walk from the source positions, without the
            # native introductory relocation. First movement is on ending day 1.
            entry['effects'] = [dict(kind='CHANGE_ORDER', enemy=ids.index(gid), order=6)
                                for gid in humans]
        else:
            for row in sorted((r for r in raw['SEvEffec'] if r['ID'].upper() == eid),
                              key=lambda r: int(r['NUM'] or 0)):
                category = int(row['CATEGORY'])
                if category == 17:  # Dialogue has no gameplay effect.
                    continue
                gid = row['ID_STACK'].upper()
                movers.add(gid)
                if category == 18:
                    # Native normal/roam orders become shortest-path pursuit.
                    effect = dict(kind='CHANGE_ORDER', enemy=ids.index(gid), order=4)
                elif category == 6:
                    effect = dict(kind='BATTLE', enemy=ids.index(gid))
                else:
                    raise ValueError('Unexpected army-event effect: '+str(category))
                entry['effects'].append(effect)
        events.append(entry)
    game_map.update(scenario_events=events, scenario_events_enabled=True,
        scenario_normal_ai='pursuit', scenario_hero='duke', scenario_movement_points=True,
        scenario_order_armies=[dict(enemy=ids.index(gid), order=0 if gid in humans else int(stacks[gid]['ORDER']),
                                    moves=int(stacks[gid]['MOVE'])) for gid in sorted(movers)],
        # Any army on the capital entrance attacks its garrison (docs/WORLD.md#garrisons).
        capital_defender=dict(unit=UNIT_KEYS['G000UU3003'], level=1))

def validate_geometry(game_map):
    """Validate source conversion only; this does not execute a JAX environment."""
    blocked = {tuple(p) for p in game_map['obstacles']}
    reserved = {tuple(game_map[k]) for k in ('agent_position', 'capital_position', 'objective_position')}
    reserved.update(tuple(p) for p in game_map['opponent_positions'])
    reserved.update(tuple(o['position']) for kind in ('cities', 'chests', 'mines') for o in game_map[kind])
    if blocked & reserved:
        raise ValueError(f'Object overlaps obstacle: {sorted(blocked & reserved)}')
    for kind in ('ruins', 'merchants', 'spell_shops'):
        for obj in game_map[kind]:
            r, c = obj['position']
            cells = {(r+i, c+j) for i in range(3) for j in range(3)}
            entrance = {tuple(obj['entrance'])} if kind == 'ruins' else set()
            if cells & (blocked | (reserved-entrance)):
                raise ValueError('Building footprint overlaps object: '+obj['name'])
            blocked.update(cells)
    # The native prison is sealed until the mountain-removal event; retain its geometry.
    # The requested gate and an approach to every ruin must remain reachable.
    blocked.update(tuple(m['position']) for m in game_map['mines'])
    reached = {tuple(game_map['capital_position'])}
    queue = deque(reached)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            point = r+dr, c+dc
            if all(0 < v < game_map['size']-1 for v in point) and point not in blocked | reached:
                reached.add(point)
                queue.append(point)
    if tuple(game_map['objective_position']) not in reached:
        raise ValueError('Dungeon entrance is unreachable without events')
    ruins = {tuple(r['entrance']) for r in game_map['ruins']}
    enclosed = {source for source, position in zip(game_map['reference_enemy_ids'], game_map['opponent_positions'])
                if tuple(position) not in reached | ruins}
    if enclosed != {'S103KC002C'}:
        raise ValueError('Only the native imprisoned Asteroth may be enclosed: '+str(enclosed))
    for ruin in game_map['ruins']:
        r, c = ruin['entrance']
        if not reached.intersection((r+1, c+dc) for dc in (-1, 0, 1)):
            raise ValueError('Ruin is unreachable: '+ruin['name'])
    for site in game_map['merchants']+game_map['spell_shops']:
        r, c = site['position']
        if not reached.intersection((r+dr, c+dc) for dr, dc in ((1,3), (2,3), (3,1), (3,2), (3,3))):
            raise ValueError('Site is unreachable: '+site['name'])
    for obj in game_map['cities']+game_map['chests']:
        if tuple(obj['position']) not in reached:
            raise ValueError('Map object is unreachable: '+str(obj['position']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--output', type=Path, default=ROOT/'maps/the_salty_dungeon.json')
    args = parser.parse_args()
    game_map = build(json.loads(args.snapshot.read_text(encoding='utf-8')))
    args.output.write_text(dump(game_map), encoding='utf-8')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
