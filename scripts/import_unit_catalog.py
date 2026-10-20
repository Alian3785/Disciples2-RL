"""Rebuild the complete installed-edition catalog; no game/JAX code is executed.

Usage: python scripts/import_unit_catalog.py --game GAME --reference REFERENCE
The installed DBF values take precedence. Python DATA supplies behavioral types.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy

from inspect_unit_sources import dbf, reference_data

SOURCES = ('weapon', 'mind', 'life', 'death', 'fire', 'water', 'earth', 'air')
FACTIONS = ('empire', 'mountain_clans', 'legions', 'undead_hordes', 'neutral', 'elves')


def integer(row, field):
    return int(row.get(field) or 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = {p.stem.lower(): p for p in (args.game/'Globals').iterdir() if p.suffix.lower() == '.dbf'}
    tables = {name: dbf(files[name]) for name in
              ('gunits', 'gattacks', 'gdynupgr', 'gimmu', 'gimmuc', 'gtransf', 'tglobal')}
    units = {r['UNIT_ID'].lower(): r for r in tables['gunits']}
    attacks = {r['ATT_ID'].lower(): r for r in tables['gattacks']}
    growth = {r['UPGRADE_ID'].lower(): r for r in tables['gdynupgr']}
    texts = {r['TXT_ID'].lower(): r['TEXT'] for r in tables['tglobal']}
    ref_file = args.reference/'Big_map/data_dicts_compact_lines.py'
    reference = reference_data(ref_file)
    bindings_file = args.reference/'Big_map/unit_revive_costs.py'
    bindings = runpy.run_path(str(bindings_file))['REVIVE_PROFILE_IDS_BY_NAME']
    ref_by_id = {uid: row for row in reference for uid in bindings.get(row['кто'], ())}
    catalog_file = root/'stoix/envs/data/units.json'
    old = json.loads(catalog_file.read_text(encoding='utf-8'))
    keys = {row['game_id']: key for key, row in old.items()}
    # Stable readable keys for the newly enabled actions and their principal summons.
    preferred = {'g000uu0171': 'doppelganger', 'g000uu0153': 'elementalist',
                 'g000uu8012': 'sage', 'g000uu6013': 'occultist',
                 'g000uu6014': 'occultmaster', 'g000uu6007': 'lyf',
                 'g000uu8007': 'laclaan'}
    keys.update({uid: key for uid, key in preferred.items() if uid not in keys})
    keys.update({uid: uid for uid in units if uid not in keys})
    unsupported = []

    def classify(uid, attack, secondary):
        if uid in ref_by_id:
            return ref_by_id[uid]['тип']
        base = units[uid]['BASE_UNIT'].lower()
        if base in ref_by_id:
            return ref_by_id[base]['тип']
        # Additional Gunits variants share attack definitions with a reference template.
        candidates = [r for other, r in ref_by_id.items() if other in units
                      and units[other]['ATTACK_ID'].lower() == units[uid]['ATTACK_ID'].lower()
                      and units[other]['ATCK_TWICE'] == units[uid]['ATCK_TWICE']]
        types = {r['тип'] for r in candidates}
        if len(types) == 1:
            return types.pop()
        kind, reach, second = integer(attack, 'CLASS'), integer(attack, 'REACH'), integer(secondary, 'CLASS')
        if units[uid]['ATCK_TWICE'] == 'T':
            return 'Demon' if reach == 3 else 'Elfarcher'
        if kind == 1:
            if second == 12:
                return 'Dead dragon' if reach == 1 else 'Death' if reach == 2 else 'Spider'
            if second == 13:
                return 'Ismir son' if reach == 1 else 'Sentry'
            if second == 23:
                return 'Lord' if reach == 1 else 'Watcher'
            if second == 25:
                return 'Teurg' if reach == 1 else 'Aleman'
            if second == 18:
                return 'Wight'
            if second in (3, 9):
                return 'Betrezen' if reach == 1 else 'Abyss Devil'
            return {1: 'Mage', 2: 'Archer', 3: 'Warrior'}[reach]
        known = {2: 'Vampire' if reach == 1 else 'Dregazul',
                 3: 'Shadow' if reach == 1 else 'Ghost', 6: 'Profit' if reach == 1 else 'Cliric',
                 7: 'Baroness', 9: 'Incub', 10: 'Tiamat', 11: 'Hermit',
                 15: 'Highvampire' if reach == 1 else 'Bone Lord', 17: 'Summoner',
                 19: 'Alchemist', 20: 'Doppelganger', 21: 'Wolf Lord',
                 22: 'Succub' if reach == 1 else 'Witch'}
        if kind in known:
            return known[kind]
        unsupported.append((uid, texts.get(units[uid]['NAME_TXT'].lower()), kind, second))
        return 'UNKNOWN'

    catalog, differences = {}, []
    for uid, source in units.items():
        attack = attacks[source['ATTACK_ID'].lower()]
        secondary = attacks.get(source['ATTACK2_ID'].lower(), {})
        ref = ref_by_id.get(uid, ref_by_id.get(source['BASE_UNIT'].lower(), {}))
        key = keys[uid]
        previous = old.get(key, {})
        kind = classify(uid, attack, secondary)
        native_attack = attack
        alternate = attacks.get(attack.get('ALT_ATTACK', '').lower(), {})
        if kind == 'Wolf Lord':
            attack = alternate
        primary_class = integer(attack, 'CLASS')
        reach = integer(attack, 'REACH')
        role = ('healer' if kind in ('Cliric', 'Profit', 'Patriach', 'Deva roshi', 'Sundancer',
                 'Sylfid', 'Travnitsa', 'Novice', 'Dwarfdruid', 'Arhidruid', 'Alchemist')
                else 'area' if reach == 1 else 'melee' if reach == 3 else 'ranged')
        amount_field = 'QTY_HEAL' if primary_class in (6, 24) else 'QTY_DAM'
        damage = integer(attack, amount_field)
        if primary_class == 24:
            damage = integer(secondary, 'QTY_HEAL')
        # Buff/effect magnitude is encoded in CLASS/LEVEL, not damage.
        if primary_class not in (1, 2, 6, 12, 13, 15, 23, 24):
            damage = int(ref.get('урон', 0))
        row = dict(name=ref.get('кто', texts[source['NAME_TXT'].lower()]),
            game_id=uid, faction=FACTIONS[int(source['RACE_ID'][-1])],
            level=integer(source, 'LEVEL'), unit_type=kind, role=role,
            size=1 if source['SIZE_SMALL'] == 'T' else 2,
            max_hp=integer(source, 'HIT_POINT'), damage=damage, accuracy=integer(attack, 'POWER'),
            armor=integer(source, 'ARMOR'), initiative=integer(attack, 'INITIATIVE'),
            attack_type=SOURCES[integer(attack, 'SOURCE')],
            secondary_attack_type=SOURCES[integer(secondary, 'SOURCE')] if secondary else '',
            secondary_accuracy=integer(secondary, 'POWER'),
            secondary_damage=integer(secondary, 'QTY_DAM') or integer(secondary, 'QTY_HEAL'),
            immunities=[], protections=[], attack_class_immunities=[],
            exp_kill=integer(source, 'XP_KILLED'), exp_required=integer(source, 'XP_NEXT'),
            exp_current=0, upgrades=[], upgrade_unavailable_reason='',
            heal_gold_per_hp=int(source['HEAL_C'][1:5]), revive_gold=int(source['REVIVE_C'][1:5]),
            neutral=FACTIONS[int(source['RACE_ID'][-1])] == 'neutral', lower_form='',
            game_data=dict(source, primary_attack=native_attack, secondary_attack=secondary, alternate_attack=alternate),
            regeneration=integer(source, 'REGEN'), movement=integer(source, 'MOVE'),
            scouting=integer(source, 'SCOUT'), leadership=integer(source, 'LEADERSHIP'),
            lifetime=integer(source, 'LIFE_TIME'), double_attack=source['ATCK_TWICE'] == 'T',
            recruit_cost=source['ENROLL_C'], recruit_building=source['ENROLL_B'],
            upgrade_building=source['UPGRADE_B'], attack_class=primary_class,
            secondary_attack_class=integer(secondary, 'CLASS'), attack_reach=reach,
            english_name=texts[source['NAME_TXT'].lower()])
        for immunity in tables['gimmu']:
            if immunity['UNIT_ID'].lower() == uid:
                row['immunities' if immunity['IMMUNECAT'] == '3' else 'protections'].append(SOURCES[int(immunity['IMMUNITY'])])
        for immunity in tables['gimmuc']:
            if immunity['UNIT_ID'].lower() == uid:
                row['attack_class_immunities'].append(int(immunity['IMMUNITY']))
                if immunity['IMMUNITY'] == '12':
                    row['immunities'].append('poison')
        primary_growth = 'HEAL' if primary_class in (6, 24) else 'DAMAGE' if primary_class in (1, 2, 12, 13, 15, 23) and kind != 'Wolf Lord' else None
        secondary_growth = 'HEAL' if integer(secondary, 'CLASS') in (6, 24) else 'DAMAGE' if integer(secondary, 'CLASS') in (1, 2, 12, 13, 15, 23, 25) else None
        row['growth'] = {'threshold': integer(source, 'DYN_UPG_LV')}
        for phase, field in (('early', 'DYN_UPG1'), ('late', 'DYN_UPG2')):
            g = growth.get(source[field].lower(), {})
            row['growth'][phase] = [integer(g, 'HIT_POINT'), integer(g, primary_growth),
                                    integer(g, 'POWER'), integer(g, 'ARMOR'), integer(g, 'INITIATIVE')]
            row['growth']['kill_'+phase] = integer(g, 'XP_KILLED')
            row['growth']['secondary_'+phase] = [integer(g, secondary_growth), integer(g, 'POWER') if secondary else 0]
        row['leader'] = integer(source, 'UNIT_CAT') in (2, 6)
        row['hero'] = int(ref.get('следуровень', 0)) > 0
        if row['hero']:
            row['exp_increment'] = int(ref.get('следуровень', integer(growth.get(source['DYN_UPG1'].lower(), {}), 'XP_NEXT')))
            # Preserve the user-requested reference hero skill progression.
            row['level_bonuses'] = previous.get('level_bonuses') or old['duke']['level_bonuses']
        for field, ru, scale in (('max_hp', 'здоровье', 1), ('damage', 'урон', 1), ('accuracy', 'точн', 100),
                                  ('armor', 'броня', 100), ('initiative', 'инит', 1), ('exp_kill', 'опыт убийства', 1)):
            if uid in ref_by_id and row[field] != round(ref[ru]*scale):
                differences.append(dict(game_id=uid, name=row['name'], field=field, reference=ref[ru]*scale, installed=row[field]))
        catalog[key] = row
    if unsupported:
        raise SystemExit('Unclassified installed attacks: '+json.dumps(unsupported, ensure_ascii=False))
    for uid, source in units.items():
        row = catalog[keys[uid]]
        parent = source['PREV_ID'].lower()
        if parent in units and integer(source, 'UNIT_CAT') == 0:
            row['lower_form'] = keys[parent]
            catalog[keys[parent]]['upgrades'].append(keys[uid])
        row['summon_pool'] = [keys[r['TRANSF_ID'].lower()] for r in tables['gtransf']
                              if r['ATTACK_ID'].lower() == source['ATTACK_ID'].lower() and row['attack_class'] != 20]
    # Preserve former catalog order to avoid gratuitous identity changes on existing maps.
    catalog = {key: catalog[key] for key in list(old)+list(catalog) if key in catalog}
    catalog_file.write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest = dict(edition='Steam Disciples II Rise of the Elves', unit_count=len(catalog),
        reference_count=len(reference), types=dict(sorted(Counter(r['unit_type'] for r in catalog.values()).items())),
        files={name: hashlib.sha256(files[name].read_bytes()).hexdigest() for name in tables},
        reference_sha256=hashlib.sha256(ref_file.read_bytes()).hexdigest(),
        bindings_sha256=hashlib.sha256(bindings_file.read_bytes()).hexdigest(), differences=differences)
    (root/'docs/validation/unit-catalog.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(units=len(catalog), types=len(manifest['types']), differences=len(differences))))


if __name__ == '__main__':
    main()
