"""Extract the original start terrain and forts for Default, The Charmed Child and S101.

Writes docs/maps/<map>/land.json: the tile code of every map tile (terrain = code & 7,
ground = code >> 3), mountain landmark tiles and fort footprints with their neutrality.
scripts/land_data.py turns them into the land fields of the maps.

    python scripts/extract_land.py --game "path/to/Disciples II Rise of the Elves" \
        --reference "C:/Disciples 2 python reference"
"""
import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

from extract_landmarks import DEFAULT_SG, sg_text
from inspect_unit_sources import dbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import dump  # noqa: E402

NEUTRAL_RACE = 'G000RR0004'  # Grace.dbf, RACE_TYPE 4


def decode_block(text):
    """DBF BLOCKDATA: 6-bit ASCII-33, most significant bit first, into little-endian words."""
    bits = count = 0
    data = bytearray()
    for char in text.encode('ascii'):
        bits = (bits << 6) | (char-33)
        count += 6
        while count >= 8:
            count -= 8
            data.append((bits >> count) & 255)
    return struct.unpack(f'<{len(data)//4}I', bytes(data[:len(data)//4*4]))


def rows_of(blocks, size):
    """Tile rows from (BLOCKID, words) of 8x4 blocks; the id ends in hex y, then hex x."""
    rows = [[None]*size for _ in range(size)]
    for block_id, words in blocks:
        by, bx = int(block_id[-4:-2], 16), int(block_id[-2:], 16)
        for i, word in enumerate(words[:32]):
            rows[by+i//8][bx+i % 8] = word & 255
    if any(v is None for row in rows for v in row):
        raise ValueError('Incomplete terrain')
    return rows


def sg_tiles(data, size):
    """Tile rows of a .sg scenario, whose BLOCKDATA holds 32 raw little-endian words."""
    blocks = []
    for match in re.finditer(rb'\.\?AVCMidgardMapBlock@@\x00', data):
        chunk = data[match.end():data.index(b'ENDOBJECT', match.end())]
        at = chunk.index(b'BLOCKDATA')+len(b'BLOCKDATA')
        blocks.append((sg_text(chunk, b'BLOCKID'), struct.unpack_from('<32I', chunk, at+4)))
    return rows_of(blocks, size)


def write(name, scenario, sources, rows, mountains, capital, cities):
    """Write one snapshot with the SHA-256 of what it was read from."""
    path = ROOT/'docs/maps'/name/'land.json'
    path.write_text(dump(dict(
        schema_version=1, scenario=scenario,
        note='Original (x, y). Tile code: terrain = code & 7 (Lterrain), ground = code >> 3 (Lground).',
        sha256={label: hashlib.sha256(source.read_bytes()).hexdigest() for label, source in sources.items()},
        tiles=rows, mountain_landmarks=sorted(mountains), capital=sorted(capital), cities=cities)), encoding='utf-8')
    print('Wrote', path)


def main():
    """Extract the three original scenarios among the baseline maps."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--game', type=Path, required=True, help='Installed Rise of the Elves folder')
    parser.add_argument('--reference', type=Path, required=True,
                        help='Python reference root (contains Big_map/) or its Big_map folder')
    args = parser.parse_args()
    base = args.reference/'Big_map' if (args.reference/'Big_map').is_dir() else args.reference
    glmark = args.game/'Globals/GLmark.dbf'
    blocks_path, forts_path, players_path = (args.game/'Scens'/n for n in ('SMapBlck.DBF', 'SFort.DBF', 'Splayer.DBF'))
    kinds = {r['LMARK_ID'].upper(): r for r in dbf(glmark)}
    blocks = dbf(blocks_path)
    races = {r['PLAYER_ID'].upper(): r['RACE_ID'].upper() for r in dbf(players_path)}

    def scens_tiles(scenario, size):
        return rows_of([(r['BLOCKID'], decode_block(r['BLOCKDATA'])) for r in blocks
                        if r['NO_SCEN'].upper().startswith(scenario)], size)

    def mountains_of(path):
        marks = json.loads(path.read_text(encoding='utf-8'))['landmarks']
        return [[m['position'][0]+i, m['position'][1]+j] for m in marks if m['mountain']
                for i in range(m['size'][0]) for j in range(m['size'][1])]

    def own_capital(layout):
        return next(c['footprint'] for c in layout['capitals'] if c['position'] == layout['hero'])

    # Default: every village of the .sg is neutral (docs/DEFAULT_MAP.md).
    scenario = base/DEFAULT_SG
    layout_path = base/'maps/default/layout.json'
    layout = json.loads(layout_path.read_text(encoding='utf-8'))
    write('default', 'A Return to Simpler Times',
          {'Big_map/'+DEFAULT_SG: scenario, 'Big_map/maps/default/layout.json': layout_path},
          sg_tiles(scenario.read_bytes(), layout['grid_size']),
          mountains_of(ROOT/'docs/maps/default/landmarks.json'), own_capital(layout),
          [dict(name=c['name'], footprint=sorted(c['footprint']), neutral=True) for c in layout['cities']])

    # The Charmed Child: S117 map blocks, and each village's owner in SFort.
    layout_path = base/'maps/wotans_retribution/layout.json'
    layout = json.loads(layout_path.read_text(encoding='utf-8'))
    owners = {(int(r['POS_X']), int(r['POS_Y'])): r['OWNER'].upper() for r in dbf(forts_path)
              if r['NO_SCEN'].upper().startswith('S117')}

    def s117_neutral(city):
        return races[owners[min(p[0] for p in city['footprint']), min(p[1] for p in city['footprint'])]] == NEUTRAL_RACE

    write('the_charmed_child', 'S117SC0000 The Charmed Child',
          {'Scens/SMapBlck.DBF': blocks_path, 'Scens/SFort.DBF': forts_path, 'Scens/Splayer.DBF': players_path,
           'Big_map/maps/wotans_retribution/layout.json': layout_path},
          scens_tiles('S117', layout['grid_size']),
          mountains_of(ROOT/'docs/maps/the_charmed_child/landmarks.json'), own_capital(layout),
          [dict(name=c['name'], footprint=sorted(c['footprint']), neutral=s117_neutral(c)) for c in layout['cities']])

    # Bethrezen's Retreat: forts and landmarks of the S101 snapshot; the player is S101PL0001.
    snapshot_path = ROOT/'docs/maps/bethrezens_retreat/layout.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    mountains = []
    for mark in snapshot['raw_tables']['SLmark.dbf']:
        kind = kinds[mark['TYPE'].upper()]
        if kind['MOUNTAIN'] == 'T':
            mountains += [[int(mark['POS_X'])+i, int(mark['POS_Y'])+j]
                          for i in range(int(kind['CX'])) for j in range(int(kind['CY']))]

    def square(fort):
        x, y = fort['position']
        return [[x+i, y+j] for i in range(fort['size']) for j in range(fort['size'])]

    home = next(f for f in snapshot['forts'] if f['kind'] == 'capital' and f['owner'] == 'S101PL0001')
    write('bethrezens_retreat', "S101SC0000 Bethrezen's Retreat",
          {'Scens/SMapBlck.DBF': blocks_path, 'Scens/Splayer.DBF': players_path, 'Globals/GLmark.dbf': glmark,
           'docs/maps/bethrezens_retreat/layout.json': snapshot_path},
          scens_tiles('S101', snapshot['size']), mountains, square(home),
          [dict(name=f['name'], footprint=square(f), neutral=races[f['owner'].upper()] == NEUTRAL_RACE)
           for f in snapshot['forts'] if f['kind'] == 'city'])


if __name__ == '__main__':
    main()
