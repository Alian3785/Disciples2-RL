"""Write Mirror match (maps/mirror_match.json) as a Disciples II scenario (.sg) for the game.

The JAX field is 50x50 with an impassable frame; the scenario is the inner 48x48 field,
original (x, y) = (column-1, row-1) of a JAX [row, column]. The file follows the scenarios that
ScenEdit exports: a fixed header, then objects WHAT/OBJ_ID/BEGOBJECT/fields/ENDOBJECT with the
field layout that scripts/sg_stream.py reads. Unit types, levels and HP come from the installed
Globals tables; nothing of the game is executed. Differences from the JAX map and the reason for
each one: docs/MIRROR_MATCH.md, section «Сценарий Disciples II».

    python scripts/build_mirror_match_sg.py
    python scripts/validate_mirror_match_sg.py
"""
import argparse
import ast
import json
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_unit_sources import dbf  # noqa: E402
from stoix.envs.number_grid_lords import STARTING_PARTIES  # noqa: E402

GAME_CANDIDATES = (Path('C:/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves'),
                   Path('/mnt/c/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves'))
OUTPUT_NAME = 'Exports/mirror match.sg'
UNITS = json.loads((ROOT/'stoix/envs/data/units.json').read_text(encoding='utf-8'))
NULL = 'G000000000'
PREFIX = 'S143'  # scenario id of every ScenEdit export
SIZE = 48
NEUTRALS, LEGIONS, EMPIRE = 'S143PL0000', 'S143PL0001', 'S143PL0002'
# Grace RACE_ID, Lrace, LSubRace, Lterrain, subrace banner, Glord lord with Llord CATEGORY 1 (L_WARRIOR)
RACES = {NEUTRALS: ('G000RR0004', 4, 5, 5, 4, 'G000LR0013'),
         LEGIONS: ('G000RR0002', 2, 3, 3, 2, 'G000LR0007'),
         EMPIRE: ('G000RR0000', 0, 1, 1, 0, 'G000LR0001')}
WARRIOR = 1
PLAIN, FOREST, WATER = 0, 1, 3  # Lground
# Lres: L_GOLD, L_RED, L_YELLOW, L_ORANGE, L_WHITE, L_BLUE (bank letters G R Y E W B)
RESOURCES = dict(gold=0, infernal=1, life=2, death=3, runes=4, elves=5)


def garrison_constant(name):
    """A constant of the JAX garrison rules, read without importing JAX."""
    tree = ast.parse((ROOT/'stoix/envs/number_grid_garrisons.py').read_text(encoding='utf-8'))
    value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, 'id', None) == name for t in node.targets))
    if isinstance(value, ast.Call) and getattr(value.func, 'id', None) == 'dict' and not value.args:
        return {k.arg: ast.literal_eval(k.value) for k in value.keywords}
    return ast.literal_eval(value)


# The JAX capital of the player keeps its faction guardian in this slot (docs/WORLD.md#garrisons);
# the JAX bot capital has no garrison. Saved scenarios put the guardian in front middle cell 2.
GUARDIAN_SLOT = garrison_constant('GUARDIAN_SLOT')
SCENARIO_GUARDIAN_CELL = 2
NORMAL, STAND = 1, 2  # LOrder
KILL_STACK, WIN = 5, 0  # LEvCond, LEvEffct
NEUTRAL_SUBRACES = range(6, 14)  # LSubRace L_NEUTRAL_HUMAN .. L_NEUTRAL_WOLF, banners 5..12
TREE_IMAGES = 19  # forest image index (bits 20+ of a tile); 0..18 occur on neutral land of saved maps
# Road image by road neighbours (north, east, south, west); y grows to the south. Every index
# except 1 and 4 occurs in saved maps with exactly this neighbourhood; 1 and 4 complete the set.
ROAD_INDEX = {(0, 0, 0, 0): 0, (1, 1, 1, 1): 1, (0, 1, 0, 1): 2, (1, 0, 1, 0): 3, (1, 0, 1, 1): 4,
              (1, 1, 0, 1): 5, (0, 1, 1, 1): 6, (1, 1, 1, 0): 7, (1, 1, 0, 0): 8, (0, 1, 1, 0): 9,
              (0, 0, 1, 1): 10, (1, 0, 0, 1): 11, (0, 1, 0, 0): 12, (0, 0, 1, 0): 13, (1, 0, 0, 0): 14,
              (0, 0, 0, 1): 15}
LEADER_CATEGORIES = ('1', '2')  # LunitC L_NOBLE, L_LEADER: every stack of the Scens tables has one
# Every D2 stack needs exactly one leader. A JAX formation without one gets the official neutral
# leader copy of one of its units (Gunits BASE_UNIT, see leader_copies). These four types have no
# copy, so one of them is replaced by the closest neutral leader of the same size:
NEAREST_LEADERS = {
    'G000UU5002': 'G000UU5201',  # Ополченец -> Скваер-лидер: та же атака 25/80/50, HP 100 вместо 95
    'G000UU5040': 'G000UU6006',  # Варвар -> Ятаа Халли: те же 250 HP и атака 80/80/40; броня 20, опыт 200 вместо 160
    'G000UU0101': 'G000UU5105',  # Скелет -> Лесной эльф: 100 HP, атака 40/80; инициатива 65 вместо 60, нет защиты от Смерти
    'G000UU0043': 'G000UU5137',  # Йети -> Белый медведь: большой, защита от Воды; 300 HP, 80 по одной цели вместо 230 HP, 30 по всем
}
# Fields a leader copy must share with its soldier (stats, attacks, immunities are compared too).
COPY_FIELDS = ('NAME_TXT', 'SIZE_SMALL', 'HIT_POINT', 'ARMOR', 'REGEN', 'ATTACK_ID', 'ATTACK2_ID',
               'ATCK_TWICE', 'XP_KILLED')
TEXTS = dict(
    name='Mirror match',
    desc='Зеркальная карта 48x48 JAX-среды Disciples2-RL. Цель: уничтожить отряд Тиамат в центре карты.',
    briefing='Уничтожить отряд Тиамат в центре карты.',
    brieflong=('Легионы и Империя начинают в противоположных углах зеркальной карты. На каждой стороне '
               'стоят одинаковые нейтральные отряды, в центре - Тиамат. Уничтожьте её отряд._'),
    win='Отряд Тиамат уничтожен. Победа!_',
    lose='Вы потерпели поражение._',
    author='Disciples2-RL')


def game_id(key):
    return UNITS[key]['game_id'].upper()


def original(cell):
    """JAX [row, column] inside the frame -> original (x, y)."""
    row, column = cell
    if not (1 <= row <= SIZE and 1 <= column <= SIZE):
        raise ValueError(f'JAX cell outside the 48x48 field: {cell}')
    return column-1, row-1


def d2_cells(slot):
    """JAX slot (0-2 front, 3-5 back) -> D2 group cell: even cells front, odd back, per column."""
    return 2*slot if slot < 3 else 2*(slot-3)+1


class Game:
    """Tables of the installed game that the scenario refers to."""

    def __init__(self, path):
        self.path = path
        globals_dir = path/'Globals'

        def table(name):
            return dbf(next(p for p in globals_dir.iterdir() if p.name.lower() == name.lower()+'.dbf'),
                       encoding='cp866')
        self.units = {r['UNIT_ID'].upper(): r for r in table('Gunits')}
        self.races = {r['RACE_ID'].upper(): r for r in table('Grace')}
        self.lords = {r['LORD_ID'].upper(): r for r in table('Glord')}
        self.immunities = {}
        for name in ('Gimmu', 'GimmuC'):
            for r in table(name):
                self.immunities.setdefault(r['UNIT_ID'].upper(), set()).add((name, r['IMMUNITY'], r['IMMUNECAT']))
        self.copies = {}
        for uid, row in self.units.items():
            base = row['BASE_UNIT'].upper()
            if row['UNIT_CAT'] == '2' and base in self.units and self.same_unit(base, uid):
                self.copies.setdefault(base, uid)

    def same_unit(self, a, b):
        ra, rb = self.units[a], self.units[b]
        return (all(ra[k].upper() == rb[k].upper() for k in COPY_FIELDS)
                and self.immunities.get(a, set()) == self.immunities.get(b, set()))

    def level(self, uid):
        return int(self.units[uid]['LEVEL'])

    def hp(self, uid):
        return int(self.units[uid]['HIT_POINT'])

    def big(self, uid):
        return self.units[uid]['SIZE_SMALL'] != 'T'

    def leader(self, uid):
        return self.units[uid]['UNIT_CAT'] in LEADER_CATEGORIES

    def movement(self, uid):
        return int(self.units[uid]['MOVE'] or 0)


def party(game, roster, levels):
    """D2 units of one JAX roster: [{slot, cells, type, jax}], the leader index and notes.

    JAX levels must be the base levels of the types: a higher D2 LEVEL would add dynamic upgrades.
    """
    members = []
    for slot, key in enumerate(roster):
        if key is None:
            continue
        uid = game_id(key)
        if levels[slot] != game.level(uid):
            raise ValueError(f'{key}: JAX level {levels[slot]} is not the D2 base level {game.level(uid)}')
        cells = [d2_cells(slot)]+([d2_cells(slot)+1] if game.big(uid) else [])
        if game.big(uid) and slot > 2:
            raise ValueError(f'{key}: a large unit must stand in the front row')
        members.append(dict(slot=slot, cells=cells, type=uid, jax=uid))
    notes = []
    leaders = [i for i, m in enumerate(members) if game.leader(m['type'])]
    if leaders:
        # One leader per stack: a real hero leads; other leader copies become their soldier.
        lead = min(leaders, key=lambda i: (game.units[members[i]['type']]['BASE_UNIT'].upper() in game.units, i))
        for i in leaders:
            if i != lead:
                base = game.units[members[i]['type']]['BASE_UNIT'].upper()
                if not game.same_unit(base, members[i]['type']):
                    raise ValueError('A second leader without an identical soldier: '+members[i]['type'])
                members[i]['type'] = base
                notes.append(f"{members[i]['jax']} -> {base} (second leader -> its base soldier)")
        return members, lead, notes
    copies = [i for i, m in enumerate(members) if m['type'] in game.copies]
    if copies:
        lead = max(copies, key=lambda i: (int(game.units[members[i]['type']]['XP_KILLED']), -i))
        members[lead]['type'] = game.copies[members[lead]['type']]
        kind = 'official neutral leader copy'
    else:
        lead = next(i for i, m in enumerate(members) if m['type'] in NEAREST_LEADERS)
        members[lead]['type'] = NEAREST_LEADERS[members[lead]['type']]
        kind = 'closest neutral leader'
    if game.big(members[lead]['type']) != game.big(members[lead]['jax']):
        raise ValueError('The leader substitute must keep the unit size')
    notes.append(f"{members[lead]['jax']} -> {members[lead]['type']} ({kind})")
    return members, lead, notes


# Object stream ---------------------------------------------------------------------------------

def text(value):
    raw = value.encode('cp1251')+b'\0'
    return struct.pack('<I', len(raw))+raw


def encode(name, kind, value):
    """One field: bare name, then a string, int32, bool byte or raw bytes ('x').

    A count ('#', named '#<object id>') is the bare id and an int32, as sg_stream returns it.
    """
    if kind == '#':
        return name.lstrip('#').encode('ascii')+struct.pack('<i', value)
    head = name.encode('ascii')
    if kind == 's':
        return head+text(value)
    if kind == 'x':
        return head+struct.pack('<I', len(value))+value
    if kind == 'i':
        return head+struct.pack('<i', value)
    if kind == 'b':
        return head+bytes([int(value)])
    raise ValueError(kind)


def serialize(cls, oid, fields):
    body = b''.join(encode(*f) for f in fields)
    return b'WHAT'+text('.?AV'+cls+'@@')+b'OBJ_ID'+text(oid)+b'BEGOBJECT\0'+body+b'ENDOBJECT\0'


def fixed(value, width):
    raw = value.encode('cp1251')
    if len(raw) >= width:
        raise ValueError(f'Header text too long: {value!r}')
    return raw+bytes(width-len(raw))


def header(players, playable, name, desc, author, rng_state):
    """Scenario header of ScenEdit exports (offsets as in Exports/*.sg).

    players: Lrace of every player in object order; playable: (lord name, Lrace, Llord, face) of
    the races offered to a human (up to four); rng_state: index and 250 words of the editor RNG.
    """
    body = 0xa88-16+40*len(players)
    out = bytearray(b'D2EESFISIG\0\0')
    out += struct.pack('<iiiii', -1, body, 1, 35, 0)
    out += fixed(PREFIX+'SC0000', 11)+fixed(desc, 256)+fixed(author, 21)+b'\0'+fixed(name, 256)
    out += struct.pack('<ii', SIZE, 1)+bytes(8)+fixed('C000CC0001', 11)
    out += struct.pack('<i', len(playable))
    for i in range(4):
        if i < len(playable):
            lord, race, category, face = playable[i]
            out += b'\1'+fixed(lord, 256)+struct.pack('<iii', race, category, face)
        else:
            out += bytes(269)
    index, words = rng_state
    out += struct.pack('<i', index)+struct.pack('<250i', *words)
    out += struct.pack('<ii', 0, len(players))
    for race in players:
        out += struct.pack('<i', race)+bytes(36)
    if len(out) != 16+body:
        raise AssertionError('Header layout')
    return bytes(out)


def ids(kind):
    count = iter(range(1 << 16))
    return lambda: f'{PREFIX}{kind}{next(count):04x}'


def build(game_map, game, empire_guardian=True):
    """(file bytes, summary) of the scenario."""
    if game_map['size'] != SIZE+2 or game_map.get('cities') or game_map.get('ruins') or game_map.get('chests'):
        raise ValueError('Expected the Mirror match map: 50x50 with a frame, no cities, ruins or chests')
    bot = game_map['scripted_bot']['enemy']
    tiamat = next(i for i, r in enumerate(game_map['enemy_rosters']) if 'tiamat' in r)
    objects, plan, summary = [], [], dict(substitutions=[], roads_omitted=[])
    unit_id, stack_id, road_id, crystal_id = ids('UN'), ids('KC'), ids('RA'), ids('CR')
    units = []

    def add(cls, oid, fields):
        objects.append((cls, oid, fields))

    def group(members, gid):
        """GROUP_ID, UNIT_0..5 and POS_0..5 of a party: POS_<cell> holds the UNIT_<index>."""
        slots, cells = [NULL]*6, [-1]*6
        for index, m in enumerate(members):
            slots[index] = m['id']
            for c in m['cells']:
                cells[c] = index
        return ([('GROUP_ID', 's', gid)]+[(f'UNIT_{i}', 's', slots[i]) for i in range(6)]
                + [(f'POS_{i}', 'i', cells[i]) for i in range(6)])

    def make_units(members, names=None):
        for m in members:
            m['id'] = unit_id()
            units.append(('CMidUnit', m['id'], [
                ('UNIT_ID', 's', m['id']), ('TYPE', 's', m['type']), ('LEVEL', 'i', game.level(m['type'])),
                ('#'+m['id'], '#', 0), ('CREATION', 'i', 0), ('NAME_TXT', 's', (names or {}).get(m['type'], '')),
                ('TRANSF', 'b', 0), ('DYNLEVEL', 'b', 0), ('HP', 'i', game.hp(m['type'])), ('XP', 'i', 0)]))

    # Terrain: neutral plain, water, forests, the 25 start tiles of each capital.
    land = {original(c): RACES[LEGIONS][3] for c in game_map['starting_land']}
    source = next(s for s in game_map['enemy_territory_sources'] if s['position'] == game_map['scripted_bot']['home'])
    land.update({original(c): RACES[EMPIRE][3] for c in source['footprint']})
    ground = {original(c): WATER for c in game_map['terrain']['water']}
    ground.update({original(c): FOREST for c in game_map['terrain']['forest']})
    tiles = {}
    for y in range(SIZE):
        for x in range(SIZE):
            g = ground.get((x, y), PLAIN)
            tiles[x, y] = land.get((x, y), RACES[NEUTRALS][3]) | g << 3 | ((x*7+y*11) % TREE_IMAGES << 20 if g == FOREST else 0)
    for by in range(0, SIZE, 4):
        for bx in range(0, SIZE, 8):
            bid = f'{PREFIX}MB{by:02x}{bx:02x}'
            words = [tiles[bx+i % 8, by+i//8] for i in range(32)]
            add('CMidgardMapBlock', bid, [('BLOCKID', 's', bid), ('BLOCKDATA', 'x', struct.pack('<32I', *words))])

    # Capitals: 5x5 at the upper-left tile of the JAX footprint; its JAX entrance lies inside.
    capitals = {}
    for owner, fid, footprint, entrance, title in (
            (LEGIONS, f'{PREFIX}FT0000', game_map['capital_footprint'], game_map['agent_position'], 'Столица Легионов'),
            (EMPIRE, f'{PREFIX}FT0001', source['footprint'], game_map['scripted_bot']['home'], 'Столица Империи')):
        cells = sorted(original(c) for c in footprint)
        x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
        if set(cells) != {(x0+i, y0+j) for i in range(5) for j in range(5)} or original(entrance) not in cells:
            raise ValueError('A capital footprint must be the 5x5 square that contains its entrance')
        capitals[owner] = dict(id=fid, pos=(x0, y0), cells=cells, title=title, entrance=original(entrance))
    obstacles = {original(c) for c in game_map['obstacles']}
    fort_tiles = {c for cap in capitals.values() for c in cap['cells']}
    if obstacles != fort_tiles-{cap['entrance'] for cap in capitals.values()}:
        raise ValueError('JAX obstacles are expected to be the capital footprints without entrances')

    # Parties: hero, the Empire, then the neutrals in JAX order.
    hero_roster = STARTING_PARTIES[game_map['faction']][game_map['lord_type']]
    hero_members, hero_lead, notes = party(game, hero_roster, [game.level(game_id(k)) if k else 0 for k in hero_roster])
    if notes or game.hp(hero_members[hero_lead]['type']) != game_map['hero_hp']:
        raise ValueError('Unexpected hero party')
    make_units(hero_members, {hero_members[hero_lead]['type']: 'Герцог'})
    bot_members, bot_lead, notes = party(game, game_map['enemy_rosters'][bot], game_map['enemy_initial_levels'][bot])
    summary['substitutions'] += [('Empire', n) for n in notes]
    make_units(bot_members, {bot_members[bot_lead]['type']: 'Рыцарь на пегасе'})
    # The player's guardian stands where JAX keeps it; the Empire gets the scenario default.
    garrisons = {}
    for owner, cell_index in ((LEGIONS, d2_cells(GUARDIAN_SLOT)), (EMPIRE, SCENARIO_GUARDIAN_CELL)):
        garrisons[owner] = []
        if owner == EMPIRE and not empire_guardian:
            continue
        guard = game.races[RACES[owner][0]]['GUARDIAN'].upper()
        if owner == LEGIONS and guard != garrison_constant('GUARDIANS')[game_map['faction']].upper():
            raise ValueError('The JAX capital guardian differs from Grace.dbf')
        if game.big(guard):
            raise ValueError('A large guardian does not fit one cell')
        garrisons[owner] = [dict(slot=None, cells=[cell_index], type=guard, jax=None)]
        make_units(garrisons[owner])

    subraces = [(NEUTRALS, 5, 4), (LEGIONS, 3, 2), (EMPIRE, 1, 0)]+[(NEUTRALS, s, s-1) for s in NEUTRAL_SUBRACES]
    subrace_id = {(owner, sub): f'{PREFIX}SR{i:04x}' for i, (owner, sub, _) in enumerate(subraces)}

    def stack(sid, owner, members, lead, pos, inside, order, facing=0):
        leader = members[lead]
        sub = int(game.units[leader['type']]['SUBRACE'] or 0)
        sub = sub if owner == NEUTRALS and sub in NEUTRAL_SUBRACES else RACES[owner][2]
        add('CMidStack', sid, group(members, sid)+[
            ('#'+sid, '#', 0), ('STACK_ID', 's', sid), ('SRCTMPL_ID', 's', NULL), ('LEADER_ID', 's', leader['id']),
            ('LEADR_ALIV', 'b', 1), ('POS_X', 'i', pos[0]), ('POS_Y', 'i', pos[1]), ('MORALE', 'i', 0),
            ('MOVE', 'i', game.movement(leader['type'])), ('FACING', 'i', facing)]
            + [(k, 's', NULL) for k in ('BANNER', 'TOME', 'BATTLE1', 'BATTLE2', 'ARTIFACT1', 'ARTIFACT2', 'BOOTS')]
            + [('OWNER', 's', owner), ('INSIDE', 's', inside), ('SUBRACE', 's', subrace_id[owner, sub]),
               ('INVISIBLE', 'b', 0), ('AI_IGNORE', 'b', 0), ('UPGCOUNT', 'i', 0), ('ORDER', 'i', order),
               ('ORDER_TARG', 's', NULL), ('AIORDER', 'i', 1), ('AIORDERTAR', 's', NULL), ('AIPRIORITY', 'i', 3),
               ('CREAT_LVL', 'i', 1), ('NBBATTLE', 'i', 0)])
        plan.append((pos, sid))

    hero_stack, bot_stack = stack_id(), stack_id()
    neutral_stacks = []
    for index, (cell, roster) in enumerate(zip(game_map['opponent_positions'], game_map['enemy_rosters'])):
        if index == bot:
            continue
        members, lead, notes = party(game, roster, game_map['enemy_initial_levels'][index])
        summary['substitutions'] += [(index, n) for n in notes]
        make_units(members)
        neutral_stacks.append((index, stack_id(), members, lead, original(cell)))
    if len(neutral_stacks) != 39:
        raise ValueError('Mirror match has 39 neutral stacks')

    for owner, cap in capitals.items():
        inside = hero_stack if owner == LEGIONS else bot_stack
        add('CCapital', cap['id'], [
            ('CITY_ID', 's', cap['id']), ('NAME_TXT', 's', cap['title']), ('DESC_TXT', 's', ''),
            ('OWNER', 's', owner), ('SUBRACE', 's', subrace_id[owner, RACES[owner][2]]), ('STACK', 's', inside),
            ('POS_X', 'i', cap['pos'][0]), ('POS_Y', 'i', cap['pos'][1])]
            + group(garrisons[owner], cap['id'])+[('#'+cap['id'], '#', 0), ('AIPRIORITY', 'i', 3)])
        plan += [(c, cap['id']) for c in cap['cells']]

    crystals = {}
    for mine in game_map['mines']:
        cid = crystal_id()
        pos = original(mine['position'])
        crystals[pos] = cid
        add('CMidCrystal', cid, [('CRYSTAL_ID', 's', cid), ('RESOURCE', 'i', RESOURCES[mine['kind']]),
                                 ('POS_X', 'i', pos[0]), ('POS_Y', 'i', pos[1]), ('AIPRIORITY', 'i', 3)])
        plan.append((pos, cid))

    # ScenEdit puts no road under a fort or a resource; those JAX road tiles are left out.
    roads = set()
    for cell in game_map['terrain']['road']:
        pos = original(cell)
        if pos in fort_tiles or pos in crystals:
            summary['roads_omitted'].append(pos)
        else:
            roads.add(pos)
    for x, y in sorted(roads, key=lambda p: (p[1], p[0])):
        rid = road_id()
        around = tuple(int(p in roads) for p in ((x, y-1), (x+1, y), (x, y+1), (x-1, y)))
        add('CMidRoad', rid, [('ROAD_ID', 's', rid), ('INDEX', 'i', ROAD_INDEX[around]), ('VAR', 'i', 0),
                              ('POS_X', 'i', x), ('POS_Y', 'i', y)])
        plan.append(((x, y), rid))

    stack(hero_stack, LEGIONS, hero_members, hero_lead, capitals[LEGIONS]['pos'], capitals[LEGIONS]['id'], NORMAL)
    stack(bot_stack, EMPIRE, bot_members, bot_lead, capitals[EMPIRE]['pos'], capitals[EMPIRE]['id'], NORMAL)
    occupied = fort_tiles | set(crystals)
    tiamat_stack = None
    for index, sid, members, lead, pos in neutral_stacks:
        if pos in occupied or ground.get(pos) == WATER and not game_map.get('allow_water_objects'):
            raise ValueError(f'Neutral stack {index} on a blocked tile {pos}')
        occupied.add(pos)
        stack(sid, NEUTRALS, members, lead, pos, NULL, STAND)
        if index == tiamat:
            tiamat_stack = sid

    # Objective: the Legions win once Tiamat's stack no longer exists.
    event = f'{PREFIX}EV0000'
    add('CMidEvent', event, [('ID', 's', event), ('NAME_TXT', 's', 'Победа: Тиамат')]
        + [(r, 'b', int(r == 'HERETIC')) for r in ('HUMAN', 'DWARF', 'UNDEAD', 'HERETIC', 'NEUTRAL', 'ELF')]
        + [('VER'+r, 'b', int(r == 'HERETIC')) for r in ('HUMAN', 'DWARF', 'UNDEAD', 'HERETIC', 'NEUTRAL', 'ELF')]
        + [('ENABLED', 'b', 1), ('OCCUR_ONCE', 'b', 1), ('CHANCE', 'i', 100), ('ORDER', 'i', 0),
           ('COND_QTY', 'i', 1), ('CATEGORY', 'i', KILL_STACK), ('ID_STACK', 's', tiamat_stack),
           ('EFFECT_QTY', 'i', 1), ('CATEGORY', 'i', WIN), ('NUM', 'i', 0), ('WIN_SCEN', 'b', 1),
           ('ID_PLAYER1', 's', LEGIONS)])

    players = [(NEUTRALS, 'Нейтралы', 'Нет описания расы'),
               (LEGIONS, 'Лорд Легионов', 'Повелитель Легионов проклятых.'),
               (EMPIRE, 'Лорд Империи', 'Повелитель Империи. Ходит только компьютер.')]
    head = []
    for index, (pid, name, desc) in enumerate(players):
        race_id, race, _, _, _, lord = RACES[pid]
        if pid != NEUTRALS and game.lords[lord]['CATEGORY'] != str(WARRIOR):
            raise ValueError('Expected a warrior lord: '+lord)
        suffix = f'{index:04x}'
        head.append(('CMidPlayer', pid, [
            ('PLAYER_ID', 's', pid), ('NAME_TXT', 's', name), ('DESC_TXT', 's', desc), ('LORD_ID', 's', lord),
            ('RACE_ID', 's', race_id), ('FOG_ID', 's', f'{PREFIX}FG{suffix}'), ('KNOWN_ID', 's', f'{PREFIX}KS{suffix}'),
            ('BUILDS_ID', 's', f'{PREFIX}PB{suffix}'), ('FACE', 'i', 0), ('QTY_BREAKS', 'i', 0),
            ('BANK', 's', 'G%04d:R0000:Y0000:E0000:W0000:B0000' % game_map['starting_gold']), ('IS_HUMAN', 'b', 0),
            ('SPELL_BANK', 's', 'G0000:R0000:Y0000:E0000:W0000:B0000'), ('ATTITUDE', 'i', 1), ('RESEAR_T', 'i', 0),
            ('CONSTR_T', 'i', 0), ('SPY_1', 's', NULL), ('SPY_2', 's', NULL), ('SPY_3', 's', NULL),
            ('CAPT_BY', 's', NULL), ('ALWAYSAI', 'b', int(pid == EMPIRE)), ('EXMAPID1', 's', NULL),
            ('EXMAPTURN1', 'i', 0), ('EXMAPID2', 's', NULL), ('EXMAPTURN2', 'i', 0), ('EXMAPID3', 's', NULL),
            ('EXMAPTURN3', 'i', 0)]))
    races = [RACES[pid][1] for pid, _, _ in players]
    rng = random.Random(game_map['map_seed'])
    info = [('INFO_ID', 's', f'{PREFIX}IF0000'), ('CAMPAIGN', 's', 'C000CC0001'), ('SOURCE_M', 'b', 0),
            ('QTY_CITIES', 'i', 0), ('NAME', 's', TEXTS['name']), ('DESC', 's', TEXTS['desc']),
            ('BRIEFING', 's', TEXTS['briefing']), ('DEBUNKW', 's', TEXTS['win'])]
    info += [(f'DEBUNKW{i}', 's', '') for i in range(2, 6)]+[('DEBUNKL', 's', TEXTS['lose'])]
    info += [('BRIEFLONG1', 's', TEXTS['brieflong'])]+[(f'BRIEFLONG{i}', 's', '') for i in range(2, 6)]
    info += [('O', 'b', 0), ('CUR_TURN', 'i', 0), ('MAX_UNIT', 'i', 5), ('MAX_SPELL', 'i', 5), ('MAX_LEADER', 'i', 99),
             ('MAX_CITY', 'i', 5), ('MAP_SIZE', 'i', SIZE), ('DIFFSCEN', 'i', 1), ('DIFFGAME', 'i', 1),
             ('CREATOR', 's', TEXTS['author'])]
    info += [(f'PLAYER_{i+1}', 'i', races[i] if i < len(races) else 99) for i in range(13)]
    info += [('SUGG_LVL', 'i', 1), ('MAP_SEED', 'i', rng.randrange(1, 1 << 30))]

    def counter(cls, kind, n=0, extra=()):
        oid = f'{PREFIX}{kind}0000'
        return (cls, oid, [('#'+oid, '#', n)]+list(extra))

    fogs = [('CMidgardMapFog', f'{PREFIX}FG{i:04x}', [(f'#{PREFIX}FG{i:04x}', '#', SIZE)]
             + [f for y in range(SIZE) for f in (('POS_Y', 'i', y), ('FOG', 'x', bytes(SIZE//8)))])
            for i in range(len(players))]
    pairs = [(a, b) for i, a in enumerate(races) for b in races[i+1:]]
    plan.sort(key=lambda e: (e[0][1], e[0][0]))
    plan_id = f'{PREFIX}PN0000'
    ordered = ([counter('CMidStackDestroyed', 'SD'), ('CScenarioInfo', f'{PREFIX}IF0000', info),
                counter('CMidMountains', 'ML')]+head+[counter('CMidQuestLog', 'QL')]
               + [('CPlayerBuildings', f'{PREFIX}PB{i:04x}', [(f'#{PREFIX}PB{i:04x}', '#', 0)]) for i in range(len(players))]
               + [counter('CMidSpellCast', 'ST', 0, [(f'#{PREFIX}ST0000', '#', 0)])]+units
               + [o for o in objects if o[0] == 'CMidgardMapBlock']
               + [o for o in objects if o[0] in ('CCapital', 'CMidCrystal', 'CMidRoad', 'CMidEvent')]+fogs
               + [counter('CMidDiplomacy', 'DP', len(pairs), [f for a, b in pairs for f in
                                                                (('RACE_1', 'i', a), ('RACE_2', 'i', b), ('RELATION', 'i', 0))]),
                  ('CMidgardPlan', plan_id, [('#'+plan_id, '#', SIZE), ('#'+plan_id, '#', len(plan))]
                   + [f for (x, y), element in plan for f in (('POS_X', 'i', x), ('POS_Y', 'i', y), ('ELEMENT', 's', element))]),
                  counter('CMidScenVariables', 'SV')]
               + [o for o in objects if o[0] == 'CMidStack']
               + [counter('CMidgardMap', 'MP', SIZE), counter('CMidTalismanCharges', 'TC'), counter('CMidSpellEffects', 'ET')]
               + [('CMidSubRace', subrace_id[owner, sub], [
                   ('SUBRACE_ID', 's', subrace_id[owner, sub]), ('SUBRACE', 'i', sub), ('PLAYER_ID', 's', owner),
                   ('NUMBER', 'i', 0), ('NAME_TXT', 's', ''), ('BANNER', 'i', banner)]) for owner, sub, banner in subraces]
               + [('CPlayerKnownSpells', f'{PREFIX}KS{i:04x}', [(f'#{PREFIX}KS{i:04x}', '#', 0)]) for i in range(len(players))]
               + [counter('CTurnSummary', 'TS')])
    rng_state = (rng.randrange(250), [rng.randrange(1 << 16) for _ in range(250)])
    playable = [('Лорд Легионов', RACES[LEGIONS][1], WARRIOR, 0)]
    data = header(races, playable, TEXTS['name'], TEXTS['desc'], TEXTS['author'], rng_state)
    data += encode(f'{PREFIX}OB0000', '#', len(ordered))+b''.join(serialize(*o) for o in ordered)
    summary.update(objects=len(ordered), stacks=2+len(neutral_stacks), units=len(units), roads=len(roads),
                   tiamat_stack=tiamat_stack, empire_guardian=empire_guardian,
                   on_water=sorted(pos for _, _, _, _, pos in neutral_stacks if ground.get(pos) == WATER))
    return data, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--map', type=Path, default=ROOT/'maps/mirror_match.json')
    parser.add_argument('--game', type=Path, default=next((p for p in GAME_CANDIDATES if p.exists()), GAME_CANDIDATES[0]))
    parser.add_argument('--output', type=Path, help=f'default: <game>/{OUTPUT_NAME}')
    parser.add_argument('--no-empire-guardian', action='store_true',
                        help='leave the Empire capital empty, as the JAX bot capital (ScenEdit always adds a guardian)')
    args = parser.parse_args()
    data, summary = build(json.loads(args.map.read_text(encoding='utf-8')), Game(args.game), not args.no_empire_guardian)
    output = args.output or args.game/OUTPUT_NAME
    output.write_bytes(data)
    print('Wrote', output, len(data), 'bytes')
    for where, note in summary.pop('substitutions'):
        print('  leader', where, note)
    print('  roads left out (fort or resource tile):', summary.pop('roads_omitted'))
    print(' ', json.dumps(summary))


if __name__ == '__main__':
    main()
