"""Extract the installed S103 static scenario without running the game or JAX.

The native event records document the key and gate, but are never executed.
Coordinates in the snapshot are original (x, y), before turn-one events.
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_unit_sources import dbf
from scripts.extract_land import decode_block, rows_of
from scripts.build_default_map import dump, UNITS

SCENARIO, PLAYER = 'S103SC0000', 'S103PL0000'
NULL = 'G000000000'
DEFAULT_GAME = Path('C:/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves')


def extract(game):
    hashes = {}

    def read(folder, name):
        path = game/folder/(name+'.dbf')
        hashes[folder+'/'+name+'.dbf'] = hashlib.sha256(path.read_bytes()).hexdigest()
        # The installed Russian global texts use OEM-866; scenario text is ASCII.
        # Unlike cp1251, this also losslessly decodes every byte in binary fields.
        rows = dbf(path, encoding='cp866')
        return [r for r in rows if r['NO_SCEN'].upper() == SCENARIO] if folder == 'Scens' else rows

    raw = {name: read('Scens', name) for name in (
        'Sinfo', 'SMapBlck', 'SMounts', 'SLmark', 'SRoad', 'SObjMap', 'SFort',
        'Sstack', 'SUGroup', 'SUnit', 'SCrys', 'SBag', 'SRuin', 'SSite', 'Splayer',
        'SInventr', 'SItem', 'SMerch', 'SMage', 'SBuild', 'Sspell', 'SUnitM',
        'Tscen', 'SEvent', 'SEvCond', 'SEvEffec', 'SLoc')}
    marks = {r['LMARK_ID'].upper(): r for r in read('Globals', 'GLmark')}
    items = {r['ITEM_ID'].upper(): r for r in read('Globals', 'GItem')}
    spells = {r['SPELL_ID'].upper(): r for r in read('Globals', 'Gspells')}
    global_text = {r['TXT_ID'].upper(): r['TEXT'] for r in read('Globals', 'Tglobal')}
    modifier_ids = {r['MODIF_ID'].upper() for r in raw['SUnitM']}
    modifiers = [r for r in read('Globals', 'Gmodifl') if r['BELONGS_TO'].upper() in modifier_ids]
    modifier_types = read('Globals', 'LmodifE')
    info = raw['Sinfo'][0]
    size = int(info['MAP_SIZE'])
    texts = defaultdict(list)
    for r in raw['Tscen']:
        texts[r['ID_OBJECT'].upper(), r['ID_STRING']].append((int(r['SEQ_NUM']), r['TEXT']))

    def title(source):
        return next((''.join(v for _, v in sorted(texts[source.upper(), key]))
                     for key in ('NAME_TXT', 'TXT_TITLE', 'TITLE') if texts[source.upper(), key]), source)

    def pos(row):
        return [int(row['POS_X']), int(row['POS_Y'])]

    classes = {r['OBJ_ID'].upper(): r['WHAT'] for r in raw['SObjMap']}
    forts = []
    for r in raw['SFort']:
        capital = classes[r['CITY_ID'].upper()] == '.?AVCCapital@@'
        x, y = pos(r)
        forts.append(dict(source_id=r['CITY_ID'].upper(), name=title(r['CITY_ID']),
            position=[x, y], marker_position=[x+(4 if capital else 3), y+(2 if capital else 3)],
            size=5 if capital else 4, level=int(r['SIZE'] or 0),
            kind='capital' if capital else 'city', owner=r['OWNER'].upper()))
    ruins = [dict(source_id=r['RUIN_ID'].upper(), name=title(r['RUIN_ID']),
        position=pos(r), marker_position=[int(r['POS_X'])+1, int(r['POS_Y'])+2],
        size=3, kind='ruin', gold=int(r['CASH'][1:5]), item=r['ITEM'].upper()) for r in raw['SRuin']]
    buildings = {r['source_id']: r for r in forts+ruins}
    stacks = {r['STACK_ID'].upper(): r for r in raw['Sstack']}
    units = {r['UNIT_ID'].upper(): r for r in raw['SUnit']}
    catalog = {r['game_id'].upper(): r for r in UNITS.values()}
    armies = []
    for group in sorted(raw['SUGroup'], key=lambda r: r['GROUP_ID'].upper()):
        cells = defaultdict(list)
        for cell in range(6):
            index = int(group[f'POS_{cell}'])
            if index >= 0:
                cells[group[f'UNIT_{index}'].upper()].append(cell)
        declared = {group[f'UNIT_{i}'].upper() for i in range(6)}-{NULL}
        if declared != set(cells):
            raise ValueError('Invalid native formation: '+group['GROUP_ID'])
        if not declared:
            continue
        source = group['GROUP_ID'].upper()
        stack = stacks.get(source)
        building = buildings.get(stack['INSIDE'].upper() if stack else source)
        roster = []
        for uid, slots in cells.items():
            unit = units[uid]
            gid = unit['TYPE'].upper()
            roster.append(dict(source_id=uid, game_id=gid, name=catalog[gid]['name'],
                custom_name=title(uid), level=int(unit['LEVEL']), cells=slots,
                big=catalog[gid]['size'] == 2, source=unit))
        armies.append(dict(source_id=source, units=roster,
            owner=stack['OWNER'].upper() if stack else building.get('owner', ''),
            source_position=pos(stack) if stack else building['position'],
            marker_position=building['marker_position'] if building else pos(stack),
            kind=building['kind'] if building else 'field',
            building=building['source_id'] if building else None))
    sites = [dict(source_id=r['SITE_ID'].upper(), name=title(r['SITE_ID']), position=pos(r),
        kind={'.?AVCMidSiteMerchant@@': 'merchant', '.?AVCMidSiteMage@@': 'mage'}[
            classes[r['SITE_ID'].upper()]]) for r in raw['SSite']]
    landmarks = [dict(source_id=r['LMARK_ID'], position=pos(r),
        size=[int(marks[r['TYPE'].upper()][k]) for k in ('CX', 'CY')],
        mountain=marks[r['TYPE'].upper()]['MOUNTAIN'] == 'T') for r in raw['SLmark']]
    terrain = rows_of([(r['BLOCKID'], decode_block(r['BLOCKDATA'])) for r in raw['SMapBlck']], size)
    item_ids = ({r['ITEM_TYPE'].upper() for r in raw['SItem']} |
                {r['ITEM'].upper() for r in raw['SRuin']} |
                {r['ITEM_ID'].upper() for r in raw['SMerch']})-{NULL}
    locations = {r['LOC_ID'].upper(): dict(position=pos(r), radius=int(r['RADIUS'])) for r in raw['SLoc']}
    key_ruin = next(r for r in ruins if r['item'] == 'G000IG7114')
    gate = next(r for r in raw['SEvCond'] if r['CATEGORY'] == '16' and r['TYPE_ITEM'] == 'G000IG7114')
    return dict(schema_version=1, map_id='the_salty_dungeon', scenario=info, player=PLAYER, size=size,
        source=dict(game=str(game), sha256=hashes,
            units_sha256=hashlib.sha256((ROOT/'stoix/envs/data/units.json').read_bytes()).hexdigest(),
            note='Static S103 before events; no game or JAX execution'),
        terrain=terrain, roads=[pos(r) for r in raw['SRoad']], landmarks=landmarks,
        forts=forts, ruins=ruins, armies=armies, sites=sites,
        resources=[dict(position=pos(r), resource=int(r['RESOURCE'])) for r in raw['SCrys']],
        chests=[dict(source_id=r['BAG_ID'].upper(), position=pos(r)) for r in raw['SBag']],
        goal=dict(item='G000IG7114', ruin=key_ruin['source_id'], event=gate['ID'],
                  location=gate['ID_LOC'], **locations[gate['ID_LOC'].upper()]),
        economy=dict(items={gid: dict(price=items[gid]['VALUE'],
                    name=global_text[items[gid]['NAME_TXT'].upper()]) for gid in sorted(item_ids)},
            spells={r['SPELL_ID'].upper(): spells[r['SPELL_ID'].upper()]['BUY_C'] for r in raw['SMage']}),
        modifiers=modifiers, modifier_types=modifier_types, raw_tables=raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', type=Path, default=DEFAULT_GAME)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/maps/the_salty_dungeon/source.json')
    args = parser.parse_args()
    snapshot = extract(args.game)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump(snapshot), encoding='utf-8')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
