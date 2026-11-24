"""Check a Mirror match scenario (.sg) against maps/mirror_match.json tile by tile and stack by stack.

The .sg is read with scripts/sg_stream.py, not with the generator. Unit rules come from the
installed Globals tables. Allowed differences are only those of docs/MIRROR_MATCH.md, section
«Сценарий Disciples II»: one leader per D2 stack, no road under a fort or a resource, and the
Tiamat objective event. Exit status 1 lists every mismatch.

    python scripts/validate_mirror_match_sg.py [--sg "<game>/Exports/mirror match.sg"]
"""
import argparse
import collections
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import cell  # noqa: E402
from scripts.build_mirror_match_sg import (  # noqa: E402
    EMPIRE, FOREST, GAME_CANDIDATES, GUARDIAN_SLOT, KILL_STACK, LEGIONS, NEAREST_LEADERS, NEUTRALS, NULL,
    OUTPUT_NAME, PLAIN, RACES, RESOURCES, ROAD_INDEX, SIZE, STAND, WARRIOR, WATER, WIN, Game, game_id,
    garrison_constant)
from scripts.sg_stream import Obj, Reader  # noqa: E402
from stoix.envs.number_grid_lords import STARTING_PARTIES  # noqa: E402

# Independent of the generator: the inverse of the JAX import cell() and the D2 formation
# order of build_wild_mountains.army_units (cell // 2 is the column, odd cells are the back row).
TILE = {tuple(cell((x, y))): (x, y) for x in range(SIZE) for y in range(SIZE)}


def original(jax):
    return TILE[tuple(jax)]


def slot_of(d2_cell):
    return d2_cell//2+3*(d2_cell % 2)


class Check:
    def __init__(self):
        self.failures, self.counts = [], collections.Counter()

    def __call__(self, ok, what, kind='check'):
        self.counts[kind] += 1
        if not ok:
            self.failures.append(what)
        return ok


def read_header(data):
    """Offsets of ScenEdit exports (see build_mirror_match_sg.header)."""
    def string(at, width):
        return data[at:at+width].split(b'\0')[0].decode('cp1251')
    playable = []
    for i in range(struct.unpack_from('<i', data, 0x25c)[0]):
        at = 0x260+269*i
        playable.append((string(at+1, 256),)+struct.unpack_from('<iii', data, at+257))
    players = struct.unpack_from('<i', data, 0xa84)[0]
    return dict(signature=data[:10], body=struct.unpack_from('<i', data, 0x10)[0], name=string(0x141, 256),
                size=struct.unpack_from('<i', data, 0x241)[0], playable=playable,
                races=[struct.unpack_from('<i', data, 0xa88+40*i)[0] for i in range(players)])


def group_units(obj, units):
    """cell -> unit object of a CMidStack/CCapital group (POS_<cell> holds the UNIT_<index>)."""
    cells = {}
    for cell in range(6):
        index = obj[f'POS_{cell}']
        if index >= 0:
            cells[cell] = units[obj[f'UNIT_{index}']]
    declared = {obj[f'UNIT_{i}'] for i in range(6)}-{NULL}
    return cells, declared == {u['UNIT_ID'] for u in cells.values()}


def check_party(check, game, label, roster, levels, obj, units, allow_substitute):
    """Every JAX slot holds its unit; one leader; at most one documented leader substitute.

    Returns the substitutions as (JAX type, D2 type).
    """
    cells, consistent = group_units(obj, units)
    check(consistent, f'{label}: group lists units without cells')
    used, substitutions = set(), []
    for slot, key in enumerate(roster):
        if key is None:
            continue
        want = game_id(key)
        own = [c for c in range(6) if slot_of(c) == slot or game.big(want) and slot_of(c) == slot+3]
        found = {cells[c]['UNIT_ID'] for c in own if c in cells}
        if not check(len(found) == 1 and all(c in cells for c in own), f'{label} slot {slot}: cells {own} hold {found}', 'unit'):
            continue
        unit = cells[own[0]]
        used.update(own)
        kind = unit['TYPE'].upper()
        if kind == want:
            check(unit['LEVEL'] == levels[slot], f'{label} slot {slot}: level {unit["LEVEL"]} != JAX {levels[slot]}')
        else:
            row = game.units[kind]
            copy = row['BASE_UNIT'].upper() == want and game.same_unit(want, kind)
            demoted = game.units[want]['BASE_UNIT'].upper() == kind and game.same_unit(kind, want)
            nearest = NEAREST_LEADERS.get(want) == kind
            check(allow_substitute and (copy or nearest) or demoted,
                  f'{label} slot {slot}: {kind} is no allowed substitute for {want}')
            check(unit['LEVEL'] == game.level(kind), f'{label} slot {slot}: substitute level')
            substitutions.append((want, kind, 'copy' if copy else 'nearest' if nearest else 'base soldier'))
        check(unit['HP'] == game.hp(kind) and unit['XP'] == 0, f'{label} slot {slot}: HP/XP {unit["HP"]}/{unit["XP"]}')
        check(not unit.get('#'+unit['UNIT_ID']), f'{label} slot {slot}: modifiers')
    check(set(cells) == used, f'{label}: units in cells the JAX roster leaves empty: {sorted(set(cells)-used)}')
    leaders = [u for u in cells.values() if game.leader(u['TYPE'].upper())]
    leaders = list({u['UNIT_ID']: u for u in leaders}.values())
    check(len(leaders) == 1 and leaders[0]['UNIT_ID'] == obj['LEADER_ID'], f'{label}: exactly one leader, the stack leader')
    replaced = [s for s in substitutions if s[2] != 'base soldier']
    check(len(replaced) <= 1, f'{label}: more than one leader substitute')
    if replaced:
        check(game.units[replaced[0][1]]['UNIT_ID'].upper() == units[obj['LEADER_ID']]['TYPE'].upper(),
              f'{label}: the substitute must lead the stack')
    return substitutions


def validate(data, game_map, game):
    check = Check()
    parsed = Reader(game.path/'Scens').objects(data)
    check(all('error' not in o for o in parsed), 'unparsed objects: '+str([o['cls'] for o in parsed if 'error' in o]))
    by = collections.defaultdict(list)
    for o in parsed:
        if 'error' not in o:
            by[o['cls']].append(Obj(o))
    head = read_header(data)
    count = struct.unpack_from('<i', data, 16+head['body']+10)[0]
    check(head['signature'] == b'D2EESFISIG' and data[16+head['body']:16+head['body']+10] == b'S143OB0000'
          and count == len(parsed), 'header/object count')
    check(head['size'] == SIZE and by['CScenarioInfo'][0]['MAP_SIZE'] == SIZE
          and by['CMidgardMap'][0]['#'+by['CMidgardMap'][0].id] == SIZE, 'map size 48')
    check(head['races'] == [RACES[p][1] for p in (NEUTRALS, LEGIONS, EMPIRE)], 'header players')
    check(head['playable'] == [(by['CMidPlayer'][1]['NAME_TXT'], RACES[LEGIONS][1], WARRIOR, 0)], 'only the Legions are playable')

    # Tiles: 72 blocks of 8x4.
    tiles = {}
    for block in by['CMidgardMapBlock']:
        by_, bx = int(block['BLOCKID'][-4:-2], 16), int(block['BLOCKID'][-2:], 16)
        for i, word in enumerate(struct.unpack('<32I', bytes.fromhex(block['BLOCKDATA']))):
            tiles[bx+i % 8, by_+i//8] = word
    check(set(tiles) == {(x, y) for x in range(SIZE) for y in range(SIZE)}, 'blocks cover the 48x48 field')
    water = {original(c) for c in game_map['terrain']['water']}
    forest = {original(c) for c in game_map['terrain']['forest']}
    legions = {original(c) for c in game_map['starting_land']}
    source = next(s for s in game_map['enemy_territory_sources'] if s['position'] == game_map['scripted_bot']['home'])
    empire = {original(c) for c in source['footprint']}
    for (x, y), word in sorted(tiles.items()):
        ground = WATER if (x, y) in water else FOREST if (x, y) in forest else PLAIN
        owner = RACES[LEGIONS][3] if (x, y) in legions else RACES[EMPIRE][3] if (x, y) in empire else RACES[NEUTRALS][3]
        check(word & 7 == owner and word >> 3 & 7 == ground and (word >> 6 == 0 or ground == FOREST and word & 0xfffc0 == 0),
              f'tile {(x, y)}: word {word:#x}, JAX ground {ground}, land {owner}', 'tile')

    # Capitals, obstacles and plan.
    plan_obj = by['CMidgardPlan'][0]
    xs, ys, elements = plan_obj.all('POS_X'), plan_obj.all('POS_Y'), plan_obj.all('ELEMENT')
    check(len(elements) == plan_obj.all('#'+plan_obj.id)[1], 'plan entry count')
    plan = collections.defaultdict(set)
    for x, y, e in zip(xs, ys, elements):
        plan[e].add((x, y))
    capitals = {c['OWNER']: c for c in by['CCapital']}
    footprint = {o: {(c['POS_X']+i, c['POS_Y']+j) for i in range(5) for j in range(5)} for o, c in capitals.items()}
    check(set(capitals) == {LEGIONS, EMPIRE}, 'two capitals: Legions and Empire')
    check(footprint[LEGIONS] == {original(c) for c in game_map['capital_footprint']}, 'Legions capital footprint')
    check(footprint[EMPIRE] == empire, 'Empire capital footprint')
    check(original(game_map['agent_position']) in footprint[LEGIONS]
          and original(game_map['scripted_bot']['home']) in footprint[EMPIRE], 'JAX capital tiles inside the forts')
    entrances = {original(game_map['agent_position']), original(game_map['scripted_bot']['home'])}
    check({original(c) for c in game_map['obstacles']} == (footprint[LEGIONS] | footprint[EMPIRE])-entrances,
          'JAX obstacles are the fort tiles except the entrances')
    for owner, cap in capitals.items():
        check(plan[cap['CITY_ID']] == footprint[owner], f'plan of {cap["CITY_ID"]}')
    crystals = {(c['POS_X'], c['POS_Y']): c['RESOURCE'] for c in by['CMidCrystal']}
    check(crystals == {original(m['position']): RESOURCES[m['kind']] for m in game_map['mines']}, 'mines and mana sources')
    for c in by['CMidCrystal']:
        check(plan[c['CRYSTAL_ID']] == {(c['POS_X'], c['POS_Y'])}, 'plan of '+c['CRYSTAL_ID'])
    forts = footprint[LEGIONS] | footprint[EMPIRE]
    roads = {(r['POS_X'], r['POS_Y']): r for r in by['CMidRoad']}
    jax_roads = {original(c) for c in game_map['terrain']['road']}
    omitted = sorted(jax_roads-set(roads), key=lambda p: (p[1], p[0]))
    check(set(roads) <= jax_roads and all(p in forts or p in crystals for p in omitted), 'roads: JAX roads except fort/resource tiles')
    for (x, y), r in roads.items():
        around = tuple(int(p in roads) for p in ((x, y-1), (x+1, y), (x, y+1), (x-1, y)))
        check(r['INDEX'] == ROAD_INDEX[around] and plan[r['ROAD_ID']] == {(x, y)}, f'road {(x, y)} image/plan')

    # Players.
    players = {p['PLAYER_ID']: p for p in by['CMidPlayer']}
    check(set(players) == {NEUTRALS, LEGIONS, EMPIRE}, 'three players')
    for pid, p in players.items():
        check(p['RACE_ID'] == RACES[pid][0] and p['LORD_ID'] == RACES[pid][5], f'{pid} race and lord')
    check(game.lords[players[LEGIONS]['LORD_ID']]['CATEGORY'] == str(WARRIOR), 'Legions lord is a warrior')
    check(players[LEGIONS]['BANK'].startswith('G%04d:' % game_map['starting_gold']), 'starting gold')
    check(players[EMPIRE]['ALWAYSAI'] == 1 and players[LEGIONS]['ALWAYSAI'] == 0, 'the Empire is always the AI')

    # Stacks.
    units = {u['UNIT_ID']: u for u in by['CMidUnit']}
    stacks = by['CMidStack']
    at = collections.defaultdict(list)
    for s in stacks:
        at[s['POS_X'], s['POS_Y']].append(s)
        check(plan[s['STACK_ID']] == {(s['POS_X'], s['POS_Y'])}, 'plan of '+s['STACK_ID'])
    substitutions = []
    bot = game_map['scripted_bot']['enemy']
    tiamat = next(i for i, r in enumerate(game_map['enemy_rosters']) if 'tiamat' in r)
    tiamat_stack = None
    for index, (cell, roster) in enumerate(zip(game_map['opponent_positions'], game_map['enemy_rosters'])):
        if index == bot:
            continue
        found = [s for s in at[original(cell)] if s['OWNER'] == NEUTRALS]
        if not check(len(found) == 1, f'neutral stack {index} at {original(cell)}', 'stack'):
            continue
        s = found[0]
        check(s['ORDER'] == STAND and s['INSIDE'] == NULL, f'neutral stack {index}: standing, outside forts')
        substitutions += [(index,)+x for x in check_party(check, game, f'stack {index}', roster,
                                                         game_map['enemy_initial_levels'][index], s, units, True)]
        if index == tiamat:
            tiamat_stack = s['STACK_ID']
    neutral_count = sum(s['OWNER'] == NEUTRALS for s in stacks)
    check(neutral_count == 39 == len(game_map['opponent_positions'])-1, f'39 neutral stacks, found {neutral_count}')
    for owner, roster, levels, label in (
            (EMPIRE, game_map['enemy_rosters'][bot], game_map['enemy_initial_levels'][bot], 'Empire'),
            (LEGIONS, STARTING_PARTIES[game_map['faction']][game_map['lord_type']], None, 'hero')):
        own = [s for s in stacks if s['OWNER'] == owner]
        if not check(len(own) == 1, f'one {label} stack', 'stack'):
            continue
        s = own[0]
        check(s['INSIDE'] == capitals[owner]['CITY_ID'] and capitals[owner]['STACK'] == s['STACK_ID']
              and (s['POS_X'], s['POS_Y']) == (capitals[owner]['POS_X'], capitals[owner]['POS_Y']), f'{label} stack inside its capital')
        levels = levels or [game.level(game_id(k)) if k else 0 for k in roster]
        substitutions += [(label,)+x for x in check_party(check, game, label, roster, levels, s, units, False)]
    placed = ({c['CITY_ID'] for c in by['CCapital']} | {c['CRYSTAL_ID'] for c in by['CMidCrystal']}
              | set(r['ROAD_ID'] for r in roads.values()) | {s['STACK_ID'] for s in stacks})
    check(set(plan) == placed, 'plan lists exactly the forts, resources, roads and stacks')
    hero = units[[s for s in stacks if s['OWNER'] == LEGIONS][0]['LEADER_ID']]
    check(game.hp(hero['TYPE'].upper()) == game_map['hero_hp'], 'hero HP')
    # Garrisons: the JAX player capital keeps its faction guardian in GUARDIAN_SLOT; the JAX
    # bot capital has none, so the Empire may hold only the scenario-default guardian in cell 2.
    jax_guardian = garrison_constant('GUARDIANS')[game_map['faction']].upper()
    for owner, cap in capitals.items():
        cells, consistent = group_units(cap, units)
        found = {c: (u['TYPE'].upper(), u['LEVEL'], u['HP']) for c, u in cells.items()}
        guard = game.races[RACES[owner][0]]['GUARDIAN'].upper()
        if owner == LEGIONS:
            want = [{c: (jax_guardian, 1, game.hp(jax_guardian)) for c in range(6) if slot_of(c) == GUARDIAN_SLOT}]
        else:
            want = [{}, {2: (guard, 1, game.hp(guard))}]
        check(consistent and found in want, f'{owner} garrison {found}, expected one of {want}')

    # Objective.
    events = by['CMidEvent']
    check(len(events) == 1 and tiamat_stack is not None, 'one objective event')
    if events and tiamat_stack:
        e = events[0]
        pairs = e.pairs
        names = [k for k, _ in pairs]
        effects = pairs[names.index('EFFECT_QTY'):]
        check(e['HERETIC'] == 1 and e['VERHERETIC'] == 1 and e['ENABLED'] == 1 and e['COND_QTY'] == 1
              and e['CATEGORY'] == KILL_STACK and e['ID_STACK'] == tiamat_stack
              and dict(effects[1:]).get('CATEGORY') == WIN and dict(effects).get('ID_PLAYER1') == LEGIONS
              and dict(effects).get('WIN_SCEN') == 1, 'objective: the Legions win when Tiamat\'s stack is destroyed')
    return check, dict(tiles=len(tiles), stacks=len(stacks), substitutions=substitutions, roads=len(roads),
                       roads_omitted=omitted, tiamat_stack=tiamat_stack, objects=len(parsed),
                       classes=dict(collections.Counter(o['cls'] for o in parsed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--map', type=Path, default=ROOT/'maps/mirror_match.json')
    parser.add_argument('--game', type=Path, default=next((p for p in GAME_CANDIDATES if p.exists()), GAME_CANDIDATES[0]))
    parser.add_argument('--sg', type=Path, help=f'default: <game>/{OUTPUT_NAME}')
    args = parser.parse_args()
    game = Game(args.game)
    path = args.sg or args.game/OUTPUT_NAME
    check, report = validate(path.read_bytes(), json.loads(args.map.read_text(encoding='utf-8')), game)
    print(path)
    print('  objects', report['objects'], 'tiles', report['tiles'], 'stacks', report['stacks'], 'roads', report['roads'])
    print('  checks', dict(check.counts))
    print('  roads left out (fort or resource tile):', report['roads_omitted'])
    kinds = collections.Counter((want, kind, how) for _, want, kind, how in report['substitutions'])
    for (want, kind, how), n in sorted(kinds.items()):
        print(f'  {n:2d} x {want} -> {kind} ({how})')
    for failure in check.failures:
        print('FAIL', failure)
    print('OK' if not check.failures else f'{len(check.failures)} mismatches')
    sys.exit(1 if check.failures else 0)


if __name__ == '__main__':
    main()
