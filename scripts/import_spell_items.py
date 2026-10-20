"""Rebuild scroll and staff data from the installed edition; no game/JAX code runs.

Usage: python scripts/import_spell_items.py --game GAME
GItem.dbf supplies item IDs, names, prices and staff casting costs; Tglobal.dbf
the English texts. Only items whose spell is in stoix/envs/data/spells.json (the
101 spells that can be researched and cast) enter the JAX catalogue; the source
audit keeps every scroll (ITEM_CAT 8) and staff (ITEM_CAT 9) with the reason.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from inspect_unit_sources import dbf

SCROLL, STAFF = '8', '9'  # LmagItm.dbf: L_SCROLL, L_WAND
MANA_LETTERS = ('r', 'y', 'e', 'w', 'b')  # infernal, life, death, runes, elves


def currency(text):
    """'g0600:r0000:y0100:e0000:w0000[:b0000]' -> gold, [infernal, life, death, runes, elves]."""
    values = {part[0].lower(): int(part[1:]) for part in text.split(':') if part}
    return values.get('g', 0), [values.get(letter, 0) for letter in MANA_LETTERS]


def snake(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = {p.stem.lower(): p for p in (args.game/'Globals').iterdir() if p.suffix.lower() == '.dbf'}
    texts = {r['TXT_ID'].lower(): r['TEXT'] for r in dbf(files['tglobal'])}
    spells = json.loads((root/'stoix/envs/data/spells.json').read_text(encoding='utf-8'))
    book = {row['id']: (faction, row) for faction, rows in spells['factions'].items() for row in rows}
    rows = [r for r in dbf(files['gitem']) if r['ITEM_CAT'] in (SCROLL, STAFF)]
    names = [texts[r['NAME_TXT'].lower()] for r in rows]
    repeated = {n.lower() for n in names if sum(m.lower() == n.lower() for m in names) > 1}
    catalogue, audit = [], []
    for row, name in zip(rows, names):
        spell = row['SPELL_ID'].lower()
        price, _ = currency(row['VALUE'])
        category = 'scroll' if row['ITEM_CAT'] == SCROLL else 'staff'
        faction, entry = book.get(spell, (None, None))
        key = snake(name)+('_'+faction if name.lower() in repeated else '')
        record = dict(key=key, game_id=row['ITEM_ID'].upper(), name=name, category=category, price=price,
                      spell=spell, description=texts[row['DESC_TXT'].lower()])
        if category == 'staff':
            record['cast_mana'] = currency(row['CASTING'])[1]
        audit.append(dict(record, supported=entry is not None,
                          reason=None if entry else 'spell is not researchable/castable in JAX',
                          spell_name=entry['name'] if entry else None, spell_level=entry['level'] if entry else None,
                          spell_cast_mana=entry['cast']['cast_mana'] if entry else None))
        if entry is not None:
            catalogue.append(record)
    if len({r['key'] for r in catalogue}) != len(catalogue):
        raise SystemExit('Item keys must be unique')
    data = dict(edition=spells['edition'], source='Globals/GItem.dbf (ITEM_CAT 8 scrolls, 9 staves), Tglobal.dbf',
                mana_kinds=spells['mana_kinds'], items=catalogue)
    (root/'stoix/envs/data/spell_items.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    hashes = {f'Globals/{files[n].name}': hashlib.sha256(files[n].read_bytes()).hexdigest() for n in ('gitem', 'tglobal')}
    (root/'docs/validation/spell-item-sources.json').write_text(json.dumps(dict(
        edition=spells['edition'], source_hashes=hashes,
        counts=dict(scrolls=sum(r['category'] == 'scroll' for r in audit),
                    staves=sum(r['category'] == 'staff' for r in audit),
                    supported_scrolls=sum(r['category'] == 'scroll' and r['supported'] for r in audit),
                    supported_staves=sum(r['category'] == 'staff' and r['supported'] for r in audit)),
        items=audit), ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(len(catalogue), 'items;', len(audit)-len(catalogue), 'left out')


if __name__ == '__main__':
    main()
