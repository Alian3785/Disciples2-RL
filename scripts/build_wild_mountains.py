"""Build the «Дикие горы» (S143) JAX campaign from its pinned source snapshot.

User request 2026-11-08: no thieves. The agent plays the Legions (Duke level 1, Possessed
and Cultist start in Vayrest) with the capital's potions, the original treasury and the
weakened guardian Ashkael. An army that reaches the capital while the Duke's party is
elsewhere fights Ashkael in a battle the agent controls; with the party inside, the party
defends first, as in the original. Losing the Duke's whole party or the capital ends
the episode in both variants. The Undead (Vaast) do not act: Sentinel keeps its
weakened Ashgan, its garrison and the Death Knight as a passive enemy capital.

User request 2026-11-09: the scenario's events on the Legions' side (scouts, waves against
Vayrest and captured cities, blockades, magic, land and ruin events). Armies walk the
shortest path around obstacles and ignore other neutrals. With capital attacks, victory
requires holding Shornhelm and defeating Sir Allemon and Gumtik. The variant without
capital attacks uses Default's rewards and ends on capture of Shornhelm.

No JAX backend, training, game process or scenario event is executed.
"""
import argparse
from collections import defaultdict, deque
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import CURRENT, RULES, UNITS, cell, dump  # noqa: E402
from scripts.build_bethrezens_retreat import formation, footprint, item_catalog, UNIT_KEYS  # noqa: E402

SNAPSHOT = ROOT/'docs/maps/wild_mountains/source.json'
NULL = 'G000000000'
HERO_STACK = 'S143KC0000'
THIEVES = ('S143KC007b', 'S143KC007c')  # Undead and Legions thieves: removed by request
GOAL_CITY = 'S143FT0006'  # Shornhelm
NEUTRAL_RACE = 'G000RR0004'
# GmodifL TYPE -> catalogue stat. 2 (leadership) and 11 (abilities: equipment use) do not
# change a fixed native enemy formation that never equips items.
STAT_MODIFIERS = {6: 'max_hp', 5: 'armor', 4: 'damage', 3: 'accuracy', 9: 'initiative'}
STATS = ('max_hp', 'damage', 'accuracy', 'armor', 'initiative')  # catalogue growth order
IGNORED_MODIFIERS = {2, 11}
# LAttackSource -> catalogue attack type; IMMUNECAT 2 = once (ward), 3 = always.
SOURCES = {0: 'weapon', 1: 'mind', 2: 'life', 3: 'death', 4: 'fire', 5: 'water', 6: 'earth', 7: 'air'}
RESOURCES = ('gold', 'infernal', 'life', 'death', 'runes', 'elves')
OWN_TERRAIN, NEUTRAL_TERRAIN = 3, 5  # Lterrain: Legions, neutral
# Anderic's Tower also sells Daylight, which lifts the fog of war; the environment has no
# fog, and the spell is not in its catalogue (stoix/envs/data/spells.json).
UNMODELED_SPELLS = {'G000SS0083'}
SPELL_CATALOG = {row['id'] for rows in json.loads((ROOT/'stoix/envs/data/spells.json').read_text(
    encoding='utf-8'))['factions'].values() for row in rows}


def truncated_percent(value, percent):
    """Discipl2.exe stacks each percent modifier on the running value, truncating toward zero.

    Seven -20% HP modifiers take Ashkael from 900 to the 190 HP stored in the .sg.
    """
    product = value*percent
    return value + (abs(product)//100)*(1 if product >= 0 else -1)


def modified(unit, snapshot):
    """Catalogue overrides of one native unit instance: modifiers, XP and its name."""
    key = UNIT_KEYS[unit['game_id']]
    profile = UNITS[key]
    override = dict(exp_current=int(unit['xp']))
    if unit['custom_name'].strip():
        override['name'] = unit['custom_name']
    stats = {}
    for mid in unit['modifiers']:
        for effect in snapshot['modifiers'][mid]:
            kind = int(effect['TYPE'])
            if kind in IGNORED_MODIFIERS:
                continue
            if kind == 12:
                field = {'2': 'protections', '3': 'immunities'}[effect['IMMUNECAT']]
                source = SOURCES[int(effect['IMMUNITY'])]
                values = override.setdefault(field, list(profile[field]))
                if source not in values:
                    values.append(source)
                continue
            field = STAT_MODIFIERS[kind]
            value = stats.get(field, profile[field])
            if 'PERCENT' in effect:
                value = truncated_percent(value, int(effect['PERCENT']))
            else:
                value += int(effect['NUMBER'])
            stats[field] = max(0, value)
    for field in stats:
        # A modifier on a levelled unit is exact only for a stat its level growth leaves alone.
        index = STATS.index(field)
        if unit['level'] > profile['level'] and (profile['growth']['early'][index] or profile['growth']['late'][index]):
            raise ValueError('Stat modifiers on a levelled unit need scaled growth: '+unit['source_id'])
    override.update(stats)
    return key, override


def army_units(army):
    """Formation order used by build_bethrezens_retreat.formation and roster overrides."""
    return sorted(army['units'], key=lambda u: min(u['cells'])//2+3*(min(u['cells']) % 2))


# Scripted events (user request 2026-11-09). Only the Legions' side is played: the passive
# Undead's own events, the alliance and every army sent against Sentinel are left out.
LEGIONS, CAPITAL_FORT, SENTINEL = 'S143PL0001', 'S143FT0000', 'S143FT0001'
SECURE_CITY, MOVE_TO_LOCATION, DEFEND_LOCATION = 6, 8, 9  # LOrder.dbf
CREATE, SPELL, ENABLE, POPUP, ORDER, WIN = 1, 2, 7, 17, 18, 0  # LEvEffct.dbf
FREQUENCY, LOCATION, ENTER_CITY, OWN_CITY, KILL_STACK, ALLIANCE, LOOT_RUIN, LAND, ZONE, EXISTS = (
    0, 2, 3, 4, 5, 9, 10, 11, 14, 17)  # LEvCond.dbf


def played(event):
    """Events the Legions or the neutrals trigger; Undead-only and alliance events are omitted."""
    return (('HERETIC' in event['races'] or 'NEUTRAL' in event['races']) and event['verify'] != ['UNDEAD']
            and all(c['category'] != ALLIANCE for c in event['conditions']))


def event_templates(snapshot):
    """Stack templates that a played event creates, except armies sent against Sentinel."""
    templates = {t['source_id']: t for t in snapshot['stack_templates']}
    used = {e['id_stktemp'] for event in snapshot['events'] if played(event)
            for e in event['effects'] if e['category'] == CREATE}
    return [templates[t] for t in sorted(used) if templates[t]['target'] != SENTINEL]


def without_capital_attacks(snapshot):
    """Remove Vayrest-bound armies, including scouts and blockades near its entrances.

    Filter source objects before compiling so every enemy, loot guard and event index is
    rebuilt together. Other city counterattacks and the remote blockades are retained.
    """
    snapshot = deepcopy(snapshot)
    destinations = {CAPITAL_FORT, 'S143LO0000', 'S143LO0009', 'S143LO000F',
                    'S143LO0013', 'S143LO0016'}
    removed = {t['source_id'].upper() for t in event_templates(snapshot)
               if t['target'].upper() in destinations}
    removed.update(e['id_stack'].upper() for event in snapshot['events'] if played(event)
                   for e in event['effects'] if e['category'] == ORDER
                   and e['order_targ'].upper() in destinations)
    snapshot['stacks'] = [s for s in snapshot['stacks'] if s['source_id'].upper() not in removed]
    events = []
    for event in snapshot['events']:
        references = [c for c in event['conditions'] if c.get('id_stack', '').upper() in removed]
        victory = any(e['category'] == WIN and e.get('id_player1') == LEGIONS for e in event['effects'])
        if references or victory:
            # The city objective replaces the scripted win; removed armies cannot trigger orders.
            continue
        event['effects'] = [e for e in event['effects']
                            if not (e['category'] == CREATE and e['id_stktemp'].upper() in removed)
                            and not (e['category'] == ORDER and e['id_stack'].upper() in removed)]
        events.append(event)
    snapshot['events'] = events
    return snapshot, sorted(removed)


def template_army(template, location):
    """POS_<cell> names the first UNIT_<index> of a unit type; repeated types take the next index.
    A large unit fills both cells of its column; UNIT_POS modifiers attach to the unit in that cell."""
    free = defaultdict(deque)
    for index, gid in enumerate(template['units']):
        if gid != NULL:
            free[gid].append(index)
    cells_of, taken = {}, set()
    for cell_index, index in enumerate(template['cells']):
        if index < 0 or cell_index in taken:
            continue
        gid = template['units'][index]
        cells = [cell_index]
        if UNITS[UNIT_KEYS[gid]].get('size', 1) == 2:
            if cell_index % 2 or template['cells'][cell_index+1] != index:
                raise ValueError('Large template unit outside a column: '+template['source_id'])
            cells.append(cell_index+1)
        cells_of[free[gid].popleft()] = cells
        taken.update(cells)
    if any(free.values()):
        raise ValueError('Template formation leaves units unplaced: '+template['source_id'])
    modifiers = defaultdict(list)
    for cell_index, mid in template['modifiers']:
        modifiers[next(i for i, cells in cells_of.items() if cell_index in cells)].append(mid)
    units = [dict(source_id=f"{template['source_id']}#{i}", game_id=template['units'][i], level=template['levels'][i],
                  cells=cells, hp=UNITS[UNIT_KEYS[template['units'][i]]]['max_hp'], xp=0, custom_name='',
                  modifiers=modifiers[i]) for i, cells in sorted(cells_of.items())]
    return dict(source_id=template['source_id'], owner='S143PL0000', units=units, marker_position=location,
                kind='event', building=None, inventory=[])


def free_tile(tile, blocked):
    """The unblocked tile nearest to tile: Chebyshev rings, then x and y."""
    x, y = tile
    for ring in range(0, 8):
        ring_tiles = sorted((x+dx, y+dy) for dx in range(-ring, ring+1) for dy in range(-ring, ring+1)
                            if max(abs(dx), abs(dy)) == ring)
        for candidate in ring_tiles:
            if candidate not in blocked:
                return list(candidate)
    raise ValueError(f'No free tile near {tile}')


def add_campaign_events(game_map, snapshot, enemies, templates):
    """Compile played events into campaign_events (stoix/envs/number_grid_campaign_events.py)."""
    ids = [a['source_id'].upper() for a in enemies]
    locations = {loc['source_id']: loc for loc in snapshot['locations']}
    forts = {f['source_id'].upper(): f for f in snapshot['forts']}
    cities = [f['source_id'].upper() for f in snapshot['forts'] if f['kind'] == 'city']
    ruins = [r['source_id'].upper() for r in snapshot['ruins']]
    by_template = {t['source_id']: t for t in templates}

    def enemy(gid):
        if gid.upper() not in ids:
            raise ValueError('Event army missing from the map: '+gid)
        return ids.index(gid.upper())

    def target(order, place):
        if order == SECURE_CITY and place in forts:
            return cell(forts[place]['marker_position'])
        if order in (MOVE_TO_LOCATION, DEFEND_LOCATION) and place in locations:
            return cell(locations[place]['position'])
        return None  # stand, guard, roam and berserk orders stand

    def zone(location):
        loc = locations[location]
        return dict(position=cell(loc['position']), radius=loc['radius'])

    def condition(c):
        kind = c['category']
        if kind == FREQUENCY:
            return dict(kind='FREQUENCY', days=c['frequency'])
        if kind == LOCATION:
            return dict(kind='HERO_IN_ZONE', **zone(c['id_loc']))
        if kind == ZONE:
            return dict(kind='STACK_IN_ZONE', enemy=enemy(c['id_stack']), **zone(c['id_loc']))
        if kind in (ENTER_CITY, OWN_CITY):
            return dict(kind='CITY_OWNED', city=cities.index(c['id_city']))
        if kind == KILL_STACK:
            return dict(kind='STACK_KILLED', enemy=enemy(c['id_stack']))
        if kind == EXISTS:
            return dict(kind='STACK_ALIVE', enemy=enemy(c['id_stack']), alive=c['misc_int'] == 0)
        if kind == LAND:
            return dict(kind='LAND_PERCENT', percent=c['pct_land'])
        if kind == LOOT_RUIN:
            return dict(kind='RUIN_LOOTED', ruin=ruins.index(c['id_ruin']))
        raise ValueError('Unsupported event condition: '+str(kind))

    def effects(event, kept):
        result = []
        for e in event['effects']:
            kind = e['category']
            if kind == CREATE and e['id_stktemp'] in by_template:
                t = by_template[e['id_stktemp']]
                result.append(dict(kind='CREATE_STACK', enemy=enemy(t['source_id']), **zone(e['id_loc']),
                                   target=target(t['order'], t['target'])))
            elif kind == ORDER and e['order_targ'] != SENTINEL:
                result.append(dict(kind='CHANGE_ORDER', enemy=enemy(e['id_stack']), target=target(e['order'], e['order_targ'])))
            elif kind == ENABLE and e['id_event'] in kept:
                result.append(dict(kind='ENABLE_EVENT', event=e['id_event'], enable=bool(e['enable'])))
            elif kind == SPELL:
                result.append(dict(kind='CAST_SPELL', spell=e['type_spell'].lower()))
            elif kind == WIN and e['id_player1'] == LEGIONS:
                result.append(dict(kind='WIN'))
            elif kind not in (CREATE, ORDER, ENABLE, POPUP, WIN):
                raise ValueError('Unsupported event effect: '+str(kind))
        return result

    # Drop events left without effects, then the switches that named them, until stable.
    kept = {e['source_id'] for e in snapshot['events'] if played(e)}
    while True:
        compiled = {e['source_id']: effects(e, kept) for e in snapshot['events'] if e['source_id'] in kept}
        remaining = {eid for eid, found in compiled.items() if found}
        if remaining == kept:
            break
        kept = remaining
    order = [e for e in snapshot['events'] if e['source_id'] in kept]
    index = {e['source_id']: i for i, e in enumerate(order)}
    events = []
    for e in order:
        found = [dict(f, event=index[f['event']]) if f['kind'] == 'ENABLE_EVENT' else f for f in compiled[e['source_id']]]
        events.append(dict(id=e['source_id'], name=e['name'], enabled=e['enabled'], once=e['once'],
                           chance=e['chance'], conditions=[condition(c) for c in e['conditions']], effects=found))
    ordered = sorted({f['enemy'] for e in events for f in e['effects'] if f['kind'] in ('CREATE_STACK', 'CHANGE_ORDER')})

    def leader(gid):
        """Movement points and a label: the template's name, else the stack leader's."""
        army = enemies[ids.index(gid)]
        if army['kind'] == 'event':
            template = by_template[army['source_id']]
            return UNITS[UNIT_KEYS[template['leader']]]['movement'], template['name']
        unit = next(u for u in army['units'] if u['source_id'] == army['leader'])
        profile = UNITS[UNIT_KEYS[unit['game_id']]]
        return profile['movement'], unit['custom_name'].strip() or profile['name']
    armies = [dict(enemy=i, moves=leader(ids[i])[0], name=leader(ids[i])[1]) for i in ordered]
    game_map['campaign_events'] = dict(events=events, armies=armies)
    game_map['scenario_event_enemies'] = [enemy(t['source_id']) for t in templates]
    # Some tiles near the river at (56,0) and behind the dwarves at (71,55) are cut off: an event
    # army's reset tile moves to the nearest tile walkable from the capital. The env snaps every
    # creation and target the same way (number_grid_campaign_events.nearest).
    reached = reachable(game_map)
    reserved = ({tuple(p) for p in game_map['opponent_positions']} | {tuple(game_map['capital_position'])}
                | {tuple(o['position']) for kind in ('cities', 'chests', 'mines') for o in game_map[kind]})
    for slot in game_map['scenario_event_enemies']:
        point = game_map['opponent_positions'][slot]
        if tuple(point) not in reached:
            best = min(reached - reserved, key=lambda t: (max(abs(t[0]-point[0]), abs(t[1]-point[1])), t))
            game_map['opponent_positions'][slot] = list(best)
            reserved.add(best)


def build(snapshot, *, no_capital_attacks=False):
    removed = []
    if no_capital_attacks:
        snapshot, removed = without_capital_attacks(snapshot)
    catalog = item_catalog()
    player = snapshot['player']
    players = {p['source_id']: p for p in snapshot['players']}
    home =next(f for f in snapshot['forts'] if f['owner'] == player and f['kind'] == 'capital')

    def loot(gids):
        result = dict(potions={}, items={})
        for gid in gids:
            if gid != NULL:
                category, key = catalog[gid]
                result[category][key] = result[category].get(key, 0)+1
        return result

    # Every native group as an army: stacks, fort garrisons and ruin guards.
    armies = []
    for stack in snapshot['stacks']:
        armies.append(dict(stack, source_id=stack['source_id']))
    for fort in snapshot['forts']:
        if fort['garrison']:
            armies.append(dict(source_id=fort['source_id'], owner=fort['owner'], units=fort['garrison'],
                               marker_position=fort['marker_position'], kind=fort['kind'],
                               building=fort['source_id'], inventory=[]))
    for ruin in snapshot['ruins']:
        armies.append(dict(source_id=ruin['source_id'], owner='S143PL0000', units=ruin['guard'],
                           marker_position=ruin['marker_position'], kind='ruin',
                           building=ruin['source_id'], inventory=[]))
    # One slot per event template, absent at reset, after every native group (S143TM > S143RU).
    templates = event_templates(snapshot)
    locations = {loc['source_id']: loc for loc in snapshot['locations']}
    for template in templates:
        spawn = next(e['id_loc'] for event in snapshot['events'] if played(event) for e in event['effects']
                     if e['category'] == CREATE and e['id_stktemp'] == template['source_id'])
        armies.append(template_army(template, locations[spawn]['position']))
    armies.sort(key=lambda a: a['source_id'])
    for army in armies:
        for unit in army['units']:
            unit['big'] = UNITS[UNIT_KEYS[unit['game_id']]].get('size', 1) == 2
    own = {a['source_id'] for a in armies if a['owner'] == player}
    if own != {HERO_STACK, home['source_id'], 'S143KC007c'}:
        raise ValueError(f'Unexpected Legions groups: {sorted(own)}')
    enemies = [a for a in armies if a['owner'] != player and a['source_id'] not in THIEVES]

    # The Duke keeps his cell; the Possessed and the Cultist leave the garrison for his party.
    hero = next(a for a in armies if a['source_id'] == HERO_STACK)
    garrison = next(a for a in armies if a['source_id'] == home['source_id'])
    guardian = next(u for u in garrison['units'] if UNITS[UNIT_KEYS[u['game_id']]]['name'] == 'Ашкаэль')
    party = dict(source_id='party', units=hero['units']+[u for u in garrison['units'] if u is not guardian])
    roster, levels = formation(party)
    ordered = army_units(party)
    hero_stats = [modified(u, snapshot)[1] for u in ordered]
    if any(set(o) - {'exp_current', 'name'} for o in hero_stats):
        raise ValueError('The starting party has no native modifiers')
    for override in hero_stats:
        override.pop('exp_current')
    hp = [0]*6
    for u in ordered:
        hp[min(u['cells'])//2+3*(min(u['cells']) % 2)] = u['hp']
    _, guardian_stats = modified(guardian, snapshot)
    guardian_stats.pop('exp_current')
    if guardian_stats.get('max_hp') != guardian['hp']:
        raise ValueError('Ashkael HP must match the stored HP after its modifiers')

    rosters, enemy_levels = zip(*(formation(a) for a in enemies))
    enemy_stats = [[modified(u, snapshot)[1] for u in army_units(a)] for a in enemies]
    for army, stats in zip(enemies, enemy_stats):
        for unit, override in zip(army_units(army), stats):
            if UNITS[UNIT_KEYS[unit['game_id']]].get('hero'):
                override['level_bonuses'] = []  # no automatic player skill ladder
            if 'max_hp' in override and override['max_hp'] != unit['hp']:
                raise ValueError('Modified HP differs from the stored HP: '+unit['source_id'])

    blocked = {(x, y) for y, row in enumerate(snapshot['terrain']) for x, code in enumerate(row)
               if (code >> 3) & 7 == 4}
    mountain_objects = {p for m in snapshot['mountains'] for p in footprint(m)}
    if mountain_objects - blocked:
        raise ValueError('Mountain objects outside mountain ground: '+str(sorted(mountain_objects-blocked)[:5]))
    for mark in snapshot['landmarks']:
        blocked.update(footprint(mark))
    for fort in snapshot['forts']:
        blocked.update(footprint(fort))
        blocked.discard(tuple(fort['marker_position']))
    sites = [s for s in snapshot['sites']]
    managed = set().union(*(footprint(r) for r in snapshot['ruins']),
                          *(footprint(dict(s, size=3)) for s in sites))
    blocked -= managed
    # An event army's reset tile is its first spawn location, moved off obstacles, sources and
    # other groups' tiles; the env places each creation near its own location.
    taken = (blocked | managed | {tuple(r['position']) for r in snapshot['resources']}
             | {tuple(c['position']) for c in snapshot['chests']} | {tuple(f['marker_position']) for f in snapshot['forts']}
             | {tuple(a['marker_position']) for a in enemies if a['kind'] != 'event'})
    for army in enemies:
        if army['kind'] == 'event':
            army['marker_position'] = free_tile(army['marker_position'], taken)
            taken.add(tuple(army['marker_position']))
    terrain = {name: [cell((x, y)) for y, row in enumerate(snapshot['terrain']) for x, code in enumerate(row)
                      if (code >> 3) & 7 == kind and (x, y) not in blocked | managed]
               for name, kind in (('forest', 1), ('water', 3))}
    terrain['road'] = [cell(p) for p in snapshot['roads'] if tuple(p) not in blocked | managed]
    roads = {tuple(p) for p in terrain['road']}
    for name in ('forest', 'water'):
        terrain[name] = [p for p in terrain[name] if tuple(p) not in roads]

    chests = [dict(source_id=c['source_id'], position=cell(c['position']), **loot(c['inventory']))
              for c in snapshot['chests']]
    # Permanent enemy inventories are guarded caches on the stack or fort tile: they open
    # after its defenders die (as for S103). Neutral city vaults go to the captor instead.
    undead_capitals = [f for f in snapshot['forts'] if f['kind'] == 'capital' and f['owner'] != player]
    for source in [a for a in enemies if a['kind'] != 'capital']+undead_capitals:
        contents = loot(source['inventory'])
        if any(contents.values()):
            guards = [i for i, a in enumerate(enemies) if a['marker_position'] == source['marker_position']]
            chests.append(dict(source_id=source['source_id'], position=cell(source['marker_position']),
                               guards=guards, **contents))

    info = snapshot['scenario']
    bank = players[player]['bank']
    owned_mines = [r['resource'] for r in snapshot['resources']
                   if snapshot['terrain'][r['position'][1]][r['position'][0]] & 7 == OWN_TERRAIN]
    ids = [a['source_id'] for a in enemies]
    game_map = dict(
        name='NumberGrid-74x74-wild_mountains', display_name='Дикие горы', size=snapshot['size']+2,
        agent_position=cell(home['marker_position']), capital_position=cell(home['marker_position']),
        agent_number=1, opponent_positions=[cell(a['marker_position']) for a in enemies],
        opponent_numbers=[1]*len(enemies), **{k: CURRENT[k] for k in RULES},
        faction='legions', battle_items=True, hero_skills=True,
        hero_roster=roster, hero_units=len(ordered), hero_initial_levels=levels,
        hero_initial_hp=hp, hero_hp=max(hp), hero_combat_stats=hero_stats, hero_leadership_capacity=3,
        enemy_rosters=list(rosters), enemy_initial_levels=list(enemy_levels), enemy_combat_stats=enemy_stats,
        enemy_units=[sum(k is not None for k in r) for r in rosters],
        enemy_hp=[max(UNITS[k]['max_hp'] for k in r if k) for r in rosters],
        starting_built_buildings=[], starting_hero_abilities=[],
        initial_potions=loot(home['inventory'])['potions'], initial_items=loot(home['inventory'])['items'],
        obstacles=sorted(cell(p) for p in blocked), terrain=terrain, allow_water_objects=True,
        scheduled_enemies=[],
        cities=[dict(source_id=f['source_id'], name=f['name'], position=cell(f['marker_position']),
                     level=f['level'], objective=f['source_id'] == GOAL_CITY, **loot(f['inventory']),
                     footprint=sorted(cell(p) for p in footprint(f)),
                     neutral=players[f['owner']]['race'] == NEUTRAL_RACE)
                for f in snapshot['forts'] if f['kind'] == 'city'],
        mines=[dict(position=cell(r['position']), kind=RESOURCES[r['resource']]) for r in snapshot['resources']],
        chests=chests,
        ruins=[dict(source_id=r['source_id'], name=r['name'], position=cell(r['position']),
                    entrance=cell(r['marker_position']), gold=r['gold'], **loot([r['item']]))
               for r in snapshot['ruins']],
        merchants=[], spell_shops=[], mercenaries=[], trainers=[], enemy_capitals=[],
        reference_enemy_ids=ids,
        reference_source='nevendaar.3dn.ru load 393, Exports/393_QRI.sg; docs/maps/wild_mountains/source.json',
        reconstruction_note=('S143 with the Legions-side scripted events, without thieves; armies walk the '
                             'shortest path to capitals and cities; agent-controlled garrison defense; '+
                             ('victory: capture Shornhelm. Rewards as on Default.' if no_capital_attacks else
                              'original victory: hold Shornhelm, destroy Sir Allemon and Gumtik.')))
    # Category 1 of the scenario's lord G000LR0007 is L_WARRIOR.
    game_map.update(lord_type='warrior',
        equipment_requires_skills=True, max_building_level=int(info['MAX_UNIT']),
        max_spell_level=int(info['MAX_SPELL']), max_hero_level=int(info['MAX_LEADER']))
    if not no_capital_attacks:
        game_map.update(reward_profile='python_reference', step_cost=0., exploration_bonus=.08, building_reward=0.)
    for site in sites:
        entry = dict(source_id=site['source_id'], name=site['name'], position=cell(site['position']))
        if site['kind'] == 'merchant':
            gids = [gid for gid, count in site['stock'] for _ in range(count)]
            entry.update(loot(gids))
            entry['prices'] = {catalog[gid][1]: snapshot['economy']['items'][gid] for gid in gids}
            game_map['merchants'].append(entry)
        elif site['kind'] == 'mage':
            missing = {s for s in site['spells'] if s.lower() not in SPELL_CATALOG}
            if missing - UNMODELED_SPELLS:
                raise ValueError(f'Spells missing from the catalogue: {sorted(missing)}')
            entry['spells'] = [dict(spell=s.lower(), stock=1, price=snapshot['economy']['spells'][s])
                               for s in site['spells'] if s not in UNMODELED_SPELLS]
            game_map['spell_shops'].append(entry)
        elif site['kind'] == 'mercenary':
            # UNIT_UNIQ is false: the original rehires; the environment sells one of each.
            entry['roster'] = []
            for offer in site['units']:
                key = UNIT_KEYS[offer['game_id']]
                if offer['level'] != UNITS[key]['level']:
                    raise ValueError('Mercenaries are hired at their base level: '+offer['game_id'])
                entry['roster'].append(dict(unit=key, stock=1))
            game_map['mercenaries'].append(entry)
        else:
            game_map['trainers'].append(entry)
    for fort in undead_capitals:
        guards = [i for i, a in enumerate(enemies) if a['building'] == fort['source_id']]
        game_map['enemy_capitals'].append(dict(name=fort['name'], position=cell(fort['marker_position']),
            guard=guards[0], guards=guards, footprint=sorted(cell(p) for p in footprint(fort))))
    tiles = [((x, y), code) for y, row in enumerate(snapshot['terrain']) for x, code in enumerate(row)]
    game_map['starting_land'] = sorted(cell(p) for p, code in tiles if code & 7 == OWN_TERRAIN)
    game_map['foreign_land'] = sorted(cell(p) for p, code in tiles if code & 7 not in (OWN_TERRAIN, NEUTRAL_TERRAIN))
    mountains = set().union(*(footprint(m) for m in snapshot['landmarks'] if m['mountain']))
    game_map['land_blocked'] = sorted(cell(p) for p in mountains | {p for p, code in tiles if code >> 3 in (3, 4)})
    game_map['capital_footprint'] = sorted(cell(p) for p in footprint(home))
    # As for S101/S103, credit the first strategic morning before the first move.
    game_map['starting_gold'] = int(bank[1:5])+100+50*owned_mines.count(0)
    game_map['starting_mana'] = {name: int(part[1:])+25*(index == 1)+50*owned_mines.count(index)
        for index, (name, part) in enumerate(zip(('infernal', 'life', 'death', 'runes'), bank.split(':')[1:]), 1)}
    # Ashkael fights arriving armies under agent control; no army moves without events.
    # The native cell 2 would be the front middle; by user rule every guardian stands in the
    # rear middle cell (docs/WORLD.md#garrisons). The garrison battle is the agent's.
    game_map['capital_defender'] = dict(unit=UNIT_KEYS[guardian['game_id']], level=guardian['level'],
                                        combat_stats=guardian_stats)
    game_map.update(defeat='party_or_capital', leader_hire=dict(unit='duke'))
    # User rule 2026-11-09: the gold mine beside the capital and the mana sources lie beyond
    # the land. A Baroness hired once for 300 gold plants a rod for 150 gold (sites below);
    # the source then pays as if Legions land covered it (stoix/envs/number_grid_rods.py).
    game_map['rod_planter'] = dict(name='Баронесса', cost=300, rod_cost=150)
    if not no_capital_attacks:
        game_map['rod_planter']['reward'] = 1.0
    # User rule 2026-11-09: a hired thief tries to poison, with a 50% chance, the army that
    # attacks the capital next turn; a successful first try grants one free repeat that day.
    # A failed or spent thief can be replaced for 200 gold while strictly more than 200 remains;
    # each replacement gets its own tries. Every unit loses POISON_S = 20% of its current HP
    # (installed GVars.dbf; Discipl2.exe 0x41d372), to no less than 1 HP.
    game_map['thief_poison'] = dict(cost=200, chance=50, percent=20)
    # User request 2026-11-09: damage and summoning spells may also target that same army, one
    # turn before its attack (number_grid_attacker_spells.py).
    game_map['attacker_spells'] = True
    # The full map uses the original scripted victory. The variant uses the objective city.
    add_campaign_events(game_map, snapshot, enemies, templates)
    # User rule 2026-11-09: rods only on the five nearest mana sources, each after its guard falls.
    game_map['rod_planter'].update(rod_sites(game_map))
    if no_capital_attacks:
        game_map.update(name='NumberGrid-74x74-wild_mountains_no_capital_attacks',
                        display_name='Дикие горы — без атак на столицу',
                        removed_capital_armies=removed)
        game_map['reconstruction_note'] += (
            ' Capital attackers, scouts and nearby blockades removed.')
    else:
        # 2026-11-10: defending the capital is not a source of farming rewards.
        # The three day checks instead require actual promotions in the Duke's party.
        game_map.update(defeat_reward=-10., timeout_reward=0., capital_attacker_rewards=False,
                        defense_rewards=dict(attacker_xp=0., party_level_record=2., mana_guard=.5,
                                             level2_multiplier=1.5, level3_multiplier=2.,
                                             party_victory=5., party_guard_victory=10.,
                                             level2_bonus=10., level3_bonus=20.,
                                             repeat_wave_multiplier=.1, repeat_wave_max_xp=1500.),
                        party_day_rewards=[dict(day=20, upgrades=4, reward=10.),
                                           dict(day=50, upgrades=8, reward=20.),
                                           dict(day=80, upgrades=12, reward=30.)])
    validate_geometry(game_map)
    return game_map


def walking_distances(game_map, blocked=None):
    """Steps from the capital to every JAX tile a party walks to: obstacles, site footprints and
    sources block, eight directions."""
    if blocked is None:
        blocked = {tuple(p) for p in game_map['obstacles']}
        for kind in ('ruins', 'merchants', 'spell_shops', 'mercenaries', 'trainers'):
            for obj in game_map[kind]:
                r, c = obj['position']
                blocked.update((r+i, c+j) for i in range(3) for j in range(3))
    blocked = blocked | {tuple(m['position']) for m in game_map['mines']}
    start = tuple(game_map['capital_position'])
    distance = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            point = r+dr, c+dc
            if all(0 < v < game_map['size']-1 for v in point) and point not in blocked and point not in distance:
                distance[point] = distance[(r, c)]+1
                queue.append(point)
    return distance


def reachable(game_map, blocked=None):
    return set(walking_distances(game_map, blocked))


def rod_sites(game_map, count=5):
    """User rule 2026-11-09: rods only on the `count` mana sources nearest to the capital by walking
    (a source is reached from a neighbouring tile) and not already on the starting land, each once
    the native army nearest to it (Chebyshev distance, then index) is defeated."""
    distance = walking_distances(game_map)
    land = {tuple(p) for p in game_map['starting_land']}
    native = [i for i in range(len(game_map['opponent_positions'])) if i not in set(game_map['scenario_event_enemies'])]

    def walk(mine):
        r, c = mine['position']
        steps = [distance[(r+dr, c+dc)]+1 for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (r+dr, c+dc) in distance]
        return min(steps) if steps else None
    candidates = [(walk(m), k) for k, m in enumerate(game_map['mines'])
                  if m['kind'] != 'gold' and tuple(m['position']) not in land and walk(m) is not None]
    sources = [k for _, k in sorted(candidates)[:count]]

    def guard(k):
        r, c = game_map['mines'][k]['position']
        return min(native, key=lambda i: (max(abs(game_map['opponent_positions'][i][0]-r),
                                              abs(game_map['opponent_positions'][i][1]-c)), i))
    return dict(sources=sources, guards=[guard(k) for k in sources])


def validate_geometry(game_map):
    """Validate source conversion only; this does not execute a JAX environment."""
    blocked = {tuple(p) for p in game_map['obstacles']}
    reserved = {tuple(game_map[k]) for k in ('agent_position', 'capital_position')}
    reserved.update(tuple(p) for p in game_map['opponent_positions'])
    reserved.update(tuple(o['position']) for kind in ('cities', 'chests', 'mines') for o in game_map[kind])
    if blocked & reserved:
        raise ValueError(f'Object overlaps obstacle: {sorted(blocked & reserved)}')
    for kind in ('ruins', 'merchants', 'spell_shops', 'mercenaries', 'trainers'):
        for obj in game_map[kind]:
            r, c = obj['position']
            cells = {(r+i, c+j) for i in range(3) for j in range(3)}
            entrance = {tuple(obj['entrance'])} if kind == 'ruins' else set()
            if cells & (blocked | (reserved-entrance)):
                raise ValueError('Building footprint overlaps object: '+obj['name'])
            blocked.update(cells)
    reached = reachable(game_map, blocked)
    goal = next(c for c in game_map['cities'] if c['objective'])
    if tuple(goal['position']) not in reached:
        raise ValueError('Shornhelm is unreachable')
    entrances = {tuple(r['entrance']) for r in game_map['ruins']}
    enclosed = [i for i, p in zip(game_map['reference_enemy_ids'], game_map['opponent_positions'])
                if tuple(p) not in reached | entrances]
    if enclosed:
        raise ValueError('Unreachable enemy armies: '+str(enclosed))
    for ruin in game_map['ruins']:
        r, c = ruin['entrance']
        if not reached.intersection((r+1, c+dc) for dc in (-1, 0, 1)):
            raise ValueError('Ruin is unreachable: '+ruin['name'])
    for obj in game_map['cities']+game_map['chests']+game_map['enemy_capitals']:
        if tuple(obj['position']) not in reached:
            raise ValueError('Map object is unreachable: '+str(obj['position']))
    for site in game_map['merchants']+game_map['spell_shops']+game_map['mercenaries']+game_map['trainers']:
        r, c = site['position']
        if not reached.intersection((r+dr, c+dc) for dr, dc in ((1, 3), (2, 3), (3, 1), (3, 2), (3, 3))):
            raise ValueError('Site is unreachable: '+site['name'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--no-capital-attacks', action='store_true',
                        help='Build a separate variant without Vayrest attackers, scouts and nearby blockades')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    name = 'wild_mountains_no_capital_attacks' if args.no_capital_attacks else 'wild_mountains'
    output = args.output or ROOT/f'maps/{name}.json'
    game_map = build(json.loads(args.snapshot.read_text(encoding='utf-8')),
                     no_capital_attacks=args.no_capital_attacks)
    output.write_text(dump(game_map), encoding='utf-8')
    print('Wrote', output, len(game_map['opponent_positions']), 'armies')
    if args.no_capital_attacks:
        print('Removed:', ', '.join(game_map['removed_capital_armies']))


if __name__ == '__main__':
    main()
