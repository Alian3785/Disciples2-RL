"""Extract the custom scenario «Дикие горы» (S143) from its .sg file into a pinned snapshot.

The scenario is not one of the installed Scens tables: it is the nevendaar.3dn.ru
download 393 (Exports/393_QRI.sg). scripts/sg_stream.py reads its object stream.
Coordinates are original (x, y). Scripted events and stack templates are counted,
not extracted; no game, JAX or event is executed.

    python3 scripts/extract_wild_mountains.py --game "/path/to/Disciples II Rise of the Elves"
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_unit_sources import dbf  # noqa: E402
from scripts.extract_land import rows_of  # noqa: E402
from scripts.build_default_map import dump  # noqa: E402
from scripts.sg_stream import Obj, Reader  # noqa: E402

NULL = 'G000000000'
SCENARIO_FILE = 'Exports/393_QRI.sg'
GAME_CANDIDATES = (Path('C:/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves'),
                   Path('/mnt/c/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves'))
INFO_FIELDS = ('NAME', 'DESC', 'BRIEFING', 'CREATOR', 'MAX_UNIT', 'MAX_SPELL', 'MAX_LEADER', 'MAX_CITY',
               'DIFFSCEN', 'DIFFGAME', 'SUGG_LVL', 'MAP_SEED', 'QTY_CITIES')


def group(obj):
    """Native formation: unit id -> its cells (POS_<cell> holds the UNIT_<index> slot)."""
    cells = defaultdict(list)
    for cell in range(6):
        index = obj[f'POS_{cell}']
        if index >= 0:
            cells[obj[f'UNIT_{index}']].append(cell)
    declared = {obj[f'UNIT_{i}'] for i in range(6)}-{NULL}
    if declared != set(cells):
        raise ValueError('Invalid native formation: '+obj.id)
    return dict(cells)


RACES = ('HUMAN', 'DWARF', 'UNDEAD', 'HERETIC', 'NEUTRAL', 'ELF')


def event_entry(obj):
    """CMidEvent: header, then COND_QTY conditions and EFFECT_QTY effects, each opened by CATEGORY."""
    pairs = obj.pairs
    names = [k for k, _ in pairs]
    start, middle = names.index('COND_QTY'), names.index('EFFECT_QTY')
    def parts(chunk):
        found = []
        for key, value in chunk:
            if key == 'CATEGORY':
                found.append(dict(category=value))
            elif key not in ('MUSIC', 'SOUND', 'IMAGE', 'IMAGE2', 'LEFT_SIDE', 'POPUP_SHOW', 'AFF', 'BOOLVALUE'):
                found[-1][key.lower()] = value.upper() if isinstance(value, str) and key != 'POPUP_TXT' else value
        return found
    conditions, effects = parts(pairs[start+1:middle]), parts(pairs[middle+1:])
    if len(conditions) != obj['COND_QTY'] or len(effects) != obj['EFFECT_QTY']:
        raise ValueError('Event part count mismatch: '+obj.id)
    return dict(source_id=obj.id.upper(), name=obj['NAME_TXT'], races=[r for r in RACES if obj[r]],
                verify=[r for r in RACES if obj['VER'+r]], enabled=bool(obj['ENABLED']),
                once=bool(obj['OCCUR_ONCE']), chance=obj['CHANCE'], order=obj['ORDER'],
                conditions=conditions, effects=sorted(effects, key=lambda e: e.get('num', 0)))


def template_entry(obj):
    """CMidStackTemplate: POS_<cell> names the first UNIT_<index> of that unit type (a large
    unit fills both cells of its column); UNIT_POS/MODIF_ID pairs attach modifiers to cells."""
    pairs = obj.pairs
    modifiers = [[pairs[i][1], pairs[i+1][1].upper()] for i, (k, _) in enumerate(pairs) if k == 'UNIT_POS']
    if len(modifiers) != obj.get('#'+obj.id, 0):
        raise ValueError('Template modifier count mismatch: '+obj.id)
    return dict(source_id=obj.id.upper(), name=obj['NAME_TXT'], leader=obj['LEADER'].upper(),
                order=obj['ORDER'], target=obj['ORDER_TARG'].upper(),
                units=[obj[f'UNIT_{i}'].upper() for i in range(6)], levels=[obj[f'UNIT_{i}_LVL'] for i in range(6)],
                cells=[obj[f'POS_{c}'] for c in range(6)], modifiers=modifiers)


def inventory(obj, items):
    """Item types of an object's '#<id>' inventory list, in stored order."""
    count = obj.get('#'+obj.id, 0)
    instances = obj.all('ITEM_ID')
    if len(instances) != count:
        raise ValueError('Inventory count mismatch: '+obj.id)
    return [items[i] for i in instances]


def extract(game):
    sg_path = game/SCENARIO_FILE
    data = sg_path.read_bytes()
    parsed = Reader(game/'Scens').objects(data)
    failed = defaultdict(int)
    by_class = defaultdict(list)
    for entry in parsed:
        if 'error' in entry:
            failed[entry['cls']] += 1
        else:
            by_class[entry['cls']].append(Obj(entry))
    if failed:
        raise ValueError(f'Unparsed scenario objects: {dict(failed)}')
    hashes = {SCENARIO_FILE: hashlib.sha256(data).hexdigest()}

    def table(name):
        path = game/'Globals'/(name+'.dbf')
        if not path.exists():
            path = next(p for p in (game/'Globals').iterdir() if p.name.lower() == name.lower()+'.dbf')
        hashes['Globals/'+path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dbf(path, encoding='cp866')

    marks = {r['LMARK_ID'].upper(): r for r in table('GLmark')}
    item_rows = {r['ITEM_ID'].upper(): r for r in table('GItem')}
    spell_rows = {r['SPELL_ID'].upper(): r for r in table('Gspells')}
    lords = {r['LORD_ID'].upper(): r for r in table('Glord')}
    info = by_class['CScenarioInfo'][0]
    size = by_class['CMidgardMap'][0]['#'+by_class['CMidgardMap'][0].id]
    items = {o['ITEM_ID']: o['ITEM_TYPE'].upper() for o in by_class['CMidItem']}
    units = {o['UNIT_ID']: o for o in by_class['CMidUnit']}

    def unit_entry(uid, cells):
        unit = units[uid]
        modifiers = unit.all('MODIF_ID')
        if len(modifiers) != unit.get('#'+uid, 0):
            raise ValueError('Modifier count mismatch: '+uid)
        return dict(source_id=uid, game_id=unit['TYPE'].upper(), level=unit['LEVEL'], cells=cells,
                    hp=unit['HP'], xp=unit['XP'], custom_name=unit.get('NAME_TXT', ''),
                    modifiers=[m.upper() for m in modifiers])

    def pos(obj):
        return [obj['POS_X'], obj['POS_Y']]

    forts = []
    for cls in ('CCapital', 'CMidVillage'):
        for o in by_class[cls]:
            capital = cls == 'CCapital'
            x, y = pos(o)
            forts.append(dict(source_id=o.id, name=o['NAME_TXT'], position=[x, y],
                marker_position=[x+(4 if capital else 3), y+(2 if capital else 3)],
                size=5 if capital else 4, level=None if capital else o['SIZE'],
                kind='capital' if capital else 'city', owner=o['OWNER'], stack=o['STACK'],
                inventory=inventory(o, items),
                garrison=[unit_entry(u, c) for u, c in group(o).items()]))
    ruins = [dict(source_id=o.id, name=o['TITLE'], position=pos(o), marker_position=[o['POS_X']+1, o['POS_Y']+2],
                  size=3, kind='ruin', gold=int(o['CASH'][1:5]), item=o['ITEM'].upper(),
                  guard=[unit_entry(u, c) for u, c in group(o).items()]) for o in by_class['CMidRuin']]
    buildings = {r['source_id']: r for r in forts+ruins}
    stacks = []
    for o in by_class['CMidStack']:
        inside = buildings.get(o['INSIDE'])
        stacks.append(dict(source_id=o.id, owner=o['OWNER'], position=pos(o),
            marker_position=inside['marker_position'] if inside else pos(o),
            kind=inside['kind'] if inside else 'field', building=inside['source_id'] if inside else None,
            leader=o['LEADER_ID'], order=o['ORDER'], move=o['MOVE'], template=o['SRCTMPL_ID'],
            inventory=inventory(o, items), units=[unit_entry(u, c) for u, c in group(o).items()]))
    sites = []
    for o in by_class['CMidSiteMerchant']:
        sites.append(dict(source_id=o.id, name=o['TXT_TITLE'], position=pos(o), kind='merchant',
                          stock=[[gid.upper(), n] for gid, n in zip(o.all('ITEM_ID'), o.all('ITEM_COUNT'))]))
    for o in by_class['CMidSiteMage']:
        sites.append(dict(source_id=o.id, name=o['TXT_TITLE'], position=pos(o), kind='mage',
                          spells=[s.upper() for s in o.all('SPELL_ID')]))
    for o in by_class['CMidSiteMercs']:
        sites.append(dict(source_id=o.id, name=o['TXT_TITLE'], position=pos(o), kind='mercenary',
                          units=[dict(game_id=u.upper(), level=lv, unique=bool(q)) for u, lv, q in
                                 zip(o.all('UNIT_ID'), o.all('UNIT_LEVEL'), o.all('UNIT_UNIQ'))]))
    for o in by_class['CMidSiteTrainer']:
        sites.append(dict(source_id=o.id, name=o['TXT_TITLE'], position=pos(o), kind='trainer'))
    landmarks = [dict(source_id=o.id, type=o['TYPE'].upper(), position=pos(o),
                      size=[int(marks[o['TYPE'].upper()][k]) for k in ('CX', 'CY')],
                      mountain=marks[o['TYPE'].upper()]['MOUNTAIN'] == 'T') for o in by_class['CMidLandmark']]
    mountains_obj = by_class['CMidMountains'][0]
    mountains = [dict(position=[x, y], size=[sx, sy]) for x, y, sx, sy in zip(
        mountains_obj.all('POS_X'), mountains_obj.all('POS_Y'), mountains_obj.all('SIZE_X'), mountains_obj.all('SIZE_Y'))]
    blocks = [(o['BLOCKID'], struct.unpack('<32I', bytes.fromhex(o['BLOCKDATA']))) for o in by_class['CMidgardMapBlock']]
    players = [dict(source_id=o.id, name=o['NAME_TXT'], race=o['RACE_ID'].upper(), lord=o['LORD_ID'].upper(),
                    lord_category=lords[o['LORD_ID'].upper()]['CATEGORY'], bank=o['BANK'])
               for o in by_class['CMidPlayer']]
    events = sorted((event_entry(o) for o in by_class['CMidEvent']), key=lambda e: e['source_id'])
    templates = sorted((template_entry(o) for o in by_class['CMidStackTemplate']), key=lambda t: t['source_id'])
    locations = sorted((dict(source_id=o.id.upper(), name=o['NAME_TXT'], position=pos(o), radius=o['RADIUS'])
                        for o in by_class['CMidLocation']), key=lambda loc: loc['source_id'])
    modifier_ids = sorted({m for army in stacks for u in army['units'] for m in u['modifiers']}
                          | {m for f in forts for u in f['garrison'] for m in u['modifiers']}
                          | {m for r in ruins for u in r['guard'] for m in u['modifiers']}
                          | {m for t in templates for _, m in t['modifiers']})
    modifiers = {mid: [{k: v for k, v in r.items() if v} for r in table('Gmodifl')
                       if r['BELONGS_TO'].upper() == mid] for mid in modifier_ids}
    item_ids = sorted(({i for s in stacks for i in s['inventory']} | {i for f in forts for i in f['inventory']}
                       | {r['item'] for r in ruins} | {gid for s in sites for gid, _ in s.get('stock', [])}
                       | {items[i] for o in by_class['CMidBag'] for i in o.all('ITEM_ID')}) - {NULL})
    return dict(schema_version=1, map_id='wild_mountains', scenario={k: info.get(k) for k in INFO_FIELDS},
        player='S143PL0001', size=size,
        source=dict(file=SCENARIO_FILE, page='https://nevendaar.3dn.ru/load/2-1-0-393', sha256=hashes,
                    units_sha256=hashlib.sha256((ROOT/'stoix/envs/data/units.json').read_bytes()).hexdigest(),
                    note='S143 objects and scripted events as stored; no game, JAX or event execution'),
        players=players, terrain=rows_of(blocks, size), mountains=mountains,
        events=events, stack_templates=templates, locations=locations,
        roads=[pos(o) for o in by_class['CMidRoad']], landmarks=landmarks,
        forts=forts, ruins=ruins, stacks=stacks, sites=sites,
        resources=[dict(source_id=o.id, position=pos(o), resource=o['RESOURCE']) for o in by_class['CMidCrystal']],
        chests=[dict(source_id=o.id, position=pos(o), inventory=inventory(o, items)) for o in by_class['CMidBag']],
        modifiers=modifiers,
        economy=dict(items={gid: int(item_rows[gid]['VALUE'][1:5]) for gid in item_ids},
                     spells={sid: int(spell_rows[sid]['BUY_C'][1:5]) for s in sites for sid in s.get('spells', [])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--game', type=Path, default=next((p for p in GAME_CANDIDATES if p.exists()), GAME_CANDIDATES[0]))
    parser.add_argument('--output', type=Path, default=ROOT/'docs/maps/wild_mountains/source.json')
    args = parser.parse_args()
    snapshot = extract(args.game)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump(snapshot), encoding='utf-8')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
