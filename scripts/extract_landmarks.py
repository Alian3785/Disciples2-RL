"""Extract original landmark footprints for Default and The Charmed Child.

Landmarks (Scens SLmark, .sg CMidLandmark) are map objects that block movement; their
size is GLmark CX x CY of the installed game. Positions are the original (x, y) of the
upper-left tile. Outputs docs/maps/{default,the_charmed_child}/landmarks.json, which
build_default_map.py and build_reference_maps.py turn into obstacles.

    python scripts/extract_landmarks.py --game "path/to/Disciples II Rise of the Elves" \
        --reference "C:/Disciples 2 python reference"
"""
import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

from inspect_unit_sources import dbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import dump  # noqa: E402

DEFAULT_SG = 'A Return To Simpler Times.sg'
CHARMED = 'S117'


def sg_text(chunk, field):
    """.sg string field: NAME + uint32 length + bytes."""
    at = chunk.index(field)+len(field)
    return chunk[at+4:at+4+struct.unpack_from('<I', chunk, at)[0]].rstrip(b'\0').decode('ascii')


def sg_number(chunk, field):
    """.sg number field: NAME + int32."""
    return struct.unpack_from('<i', chunk, chunk.index(field)+len(field))[0]


def sg_landmarks(data):
    """(LMARK_ID, TYPE, POS_X, POS_Y) of every CMidLandmark object."""
    found = []
    for match in re.finditer(rb'\.\?AVCMidLandmark@@\x00', data):
        chunk = data[match.end():data.index(b'ENDOBJECT', match.end())]
        found.append((sg_text(chunk, b'LMARK_ID'), sg_text(chunk, b'TYPE'),
                      sg_number(chunk, b'POS_X'), sg_number(chunk, b'POS_Y')))
    return found


def write(path, scenario, sources, rows, kinds, names):
    """Write one extract: (id, type, x, y) rows with GLmark sizes and names, and source SHA-256."""
    landmarks = []
    for lmark_id, kind, x, y in sorted(rows):
        row = kinds[kind.upper()]
        landmarks.append(dict(id=lmark_id, type=kind.upper(), name=names.get(row['NAME_TXT'].lower(), ''),
                              position=[x, y], size=[int(row['CX']), int(row['CY'])],
                              mountain=row['MOUNTAIN'] == 'T'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(dict(
        schema_version=1, scenario=scenario,
        note='Original (x, y) of the upper-left tile; size is GLmark CX x CY; every tile blocks movement.',
        sha256={label: hashlib.sha256(source.read_bytes()).hexdigest() for label, source in sources.items()},
        landmarks=landmarks)), encoding='utf-8')
    print('Wrote', path, len(landmarks), 'landmarks')


def main():
    """Extract the Default (.sg) and The Charmed Child (Scens S117) landmarks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--game', type=Path, required=True, help='Installed Rise of the Elves folder')
    parser.add_argument('--reference', type=Path, required=True,
                        help='Python reference root (contains Big_map/) or its Big_map folder')
    args = parser.parse_args()
    base = args.reference/'Big_map' if (args.reference/'Big_map').is_dir() else args.reference
    glmark, tglobal = args.game/'Globals/GLmark.dbf', args.game/'Globals/Tglobal.dbf'
    slmark, scenario = args.game/'Scens/SLmark.dbf', base/DEFAULT_SG
    kinds = {r['LMARK_ID'].upper(): r for r in dbf(glmark)}
    names = {r['TXT_ID'].lower(): r['TEXT'] for r in dbf(tglobal)}
    write(ROOT/'docs/maps/default/landmarks.json', 'A Return to Simpler Times',
          {'Big_map/'+DEFAULT_SG: scenario, 'Globals/GLmark.dbf': glmark, 'Globals/Tglobal.dbf': tglobal},
          sg_landmarks(scenario.read_bytes()), kinds, names)
    write(ROOT/'docs/maps/the_charmed_child/landmarks.json', 'S117SC0000 The Charmed Child',
          {'Scens/SLmark.dbf': slmark, 'Globals/GLmark.dbf': glmark, 'Globals/Tglobal.dbf': tglobal},
          [(r['LMARK_ID'], r['TYPE'], int(r['POS_X']), int(r['POS_Y'])) for r in dbf(slmark)
           if r['LMARK_ID'].upper().startswith(CHARMED)], kinds, names)


if __name__ == '__main__':
    main()
