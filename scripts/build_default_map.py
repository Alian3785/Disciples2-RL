"""Build maps/default.json from the Python reference Default map.

Source: <reference>/Big_map/maps/default/layout.json (a reset(seed=0) snapshot of
maps/default.py) plus the shop stock in campaign_env_data.py, the capital and
city inventories of the original A Return To Simpler Times.sg and its landmarks
(docs/maps/default/landmarks.json, scripts/extract_landmarks.py). Reference (x, y)
becomes JAX [y+1, x+1]: the JAX grid adds an impassable outer ring, so 48x48
becomes 50x50. Rule flags, except the ruler, are copied from number_grid_map.json. See
docs/DEFAULT_MAP.md for every difference from the reference.

    python scripts/build_default_map.py --reference "C:/Disciples 2 python reference"
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS = json.loads((ROOT/'stoix/envs/data/units.json').read_text(encoding='utf-8'))
CURRENT = json.loads((ROOT/'number_grid_map.json').read_text(encoding='utf-8'))
LANDMARKS = ROOT/'docs/maps/default/landmarks.json'

# Reference loot names -> JAX potion or item keys.
POTIONS = {
    'Potion of Healing': 'healing', 'Life Potion': 'life', 'Эликсир Жизни': 'life',
    'Potion of Strength': 'strength', 'Potion of Invulnerability': 'invulnerability',
    'Эликсир неуязвимости': 'invulnerability', 'Эликсир Силы титана': 'titan',
    'Эликсир Всевышнего': 'highfather',
}
ITEMS = {
    'Banner of War': 'banner_war', 'Banner of Might': 'banner_might',
    'Tome of Arcanum': 'tome_arcanum', 'Ruby (Valuable)': 'ruby',
    'Diamond (Valuable)': 'diamond', 'Bronze Ring (Valuable)': 'bronze_ring',
    'Кольцо веков (Артефакт)': 'ring_of_ages', 'Vampire Orb': 'vampire_orb',
    'Orb of Life': 'orb_of_life', 'Lich Orb': 'lich_orb',
    'Zombie Talisman': 'zombie_talisman', 'Mind ward scroll': 'mind_ward_scroll',
}
# campaign_env_data.py MERCHANT_BUY_ITEMS, the same as Allar's Shop in the original.
ALAR = {'life': 10, 'healing': 10, 'restoration': 10, 'ointment': 10, 'might': 1,
        'fire_ward': 1, 'earth_ward': 1, 'water_ward': 1, 'air_ward': 1, 'celerity': 1}
# User request: one artifact for sale as well (docs/ECONOMY.md#предметы-у-торговца).
ALAR_ITEMS = {'runestone': 1}
# Installed game, the user's saves of this scenario (SaveGame/12345.sg, turn 11). The
# reference .sg, a modified Russian copy, leaves these three sites empty.
# Tralar's Shop:
TRALAR = {'life': 10, 'healing': 10, 'restoration': 10, 'ointment': 10,
          'accuracy': 1, 'swiftness': 1, 'vigor': 1, 'bark': 1}
# Erarick's Camp: a Wolf (G000UU5039), level 1.
ERARICK = [{'unit': 'g000uu5039', 'stock': 1}]
# Twin Island Mages: four level II spells at their Gspells BUY_C, 400 gold each.
TOWER_SPELLS = [{'spell': spell, 'price': 400, 'stock': 1}
                for spell in ('g000ss0028', 'g000ss0067', 'g000ss0070', 'g000ss0029')]
# Original Big_map/A Return To Simpler Times.sg: the Legions player S001PL0002 starts with
# BANK G0400:R0200 (the Empire's S001PL0001 with G0400:Y0200). In the user's saves of this
# scenario (SaveGame/12345.sg, turn 11) the human Legions player holds G1150:R0725.
STARTING_GOLD, STARTING_MANA = 400, {'infernal': 200}
# Original Big_map/A Return To Simpler Times.sg: both CCapital inventories hold
# 3x G000IG0005, 3x G000IG0006 and 3x G000IG0001; layout.json omits them.
CAPITAL_POTIONS = {'healing': 3, 'restoration': 3, 'life': 3}
# Same file, CMidVillage inventories S001FT0004-6: G000IG6003, G000IG8003, G000IG9107.
# Brigham's Port (S001FT0003): event S001EV0025 gives the Legions G000IG3002 on capture;
# user request 2026-10-25: any faction gets it, like a vault item.
CITY_ITEMS = {'Порт Бирмингем': {'unholy_chalice': 1}, 'Кордилия': {'staff_of_holiness': 1},
              'Порт Полонис': {'boots_speed': 1}, 'Соругирилла': {'zombie_talisman': 1}}
# Reference chest X11 at (36, 22), beside Соругирилла, holds that city's vault Zombie Talisman;
# the original .sg has no bag there (17 bags), and CITY_ITEMS gives the talisman on capture.
DUPLICATE_CHESTS = {'X11'}
# Same file: original units that the reference's name-based IDs replaced, by reference stack id.
# The Undead Warlock G000UU0081 (death attack) became the Legions Sorcerer G000UU0063 ("Колдун",
# fire attack) in 7 at (6, 22), 9 at (24, 23), 34 Порт Полонис' stack, 38 at (39, 3) and 73, the
# guard of Руины Бритона. The neutral Elf Ranger G000UU5009 (25 damage) became the Elves' Scout
# G000UU8018 (30) in 64 at (28, 41) and 65 at (30, 41); 65's neutral Centaur Lancer G000UU5111
# (165 HP, initiative 55, 65 damage, 80 XP) became the Elves' G000UU8014 (150, 40, 30, 36).
# No other enemy holds a replaced id.
SORCERER, WARLOCK, SCOUT, RANGER, LANCER, NEUTRAL_LANCER = (
    'g000uu0063', 'g000uu0081', 'g000uu8018', 'g000uu5009', 'g000uu8014', 'g000uu5111')
ORIGINAL_UNITS = {**{i: {SORCERER: WARLOCK} for i in (7, 9, 34, 38, 73)},
                  64: {SCOUT: RANGER}, 65: {SCOUT: RANGER, LANCER: NEUTRAL_LANCER}}
# Small's requested shop (build_small_map.py): Armageddon and the strongest Ent summon
# (Verdant), 1000 gold each. Default sells TOWER_SPELLS since 2026-10-25.
SPELLS = [{'spell': 'g000ss0019', 'price': 1000, 'stock': 1},
          {'spell': 'g000ss0117', 'price': 1000, 'stock': 1}]
GREEN_DRAGON = 31  # campaign_env_magic.py: heals fully every day.
CAPITAL_GUARD = 75
ROLES = {22: 'guard', 32: 'guard', 33: 'guard', 34: 'guard', 35: 'guard',
         67: 'garrison', 68: 'garrison', 69: 'garrison', CAPITAL_GUARD: 'capital'}
# Copied unchanged from the current map so both maps train with the same rules.
RULES = ('max_steps', 'step_cost', 'exploration_bonus', 'mask_walls', 'map_seed',
         'battle_mode', 'battle_max_rounds', 'battle_recovery', 'battle_observation_version',
         'combat_rules_version', 'building_reward', 'blocked_buildings', 'max_building_level',
         'rest_penalty_per_point', 'unit_progression', 'capital_services', 'battle_engagement',
         'equipment_auto_equip', 'equipment_requires_skills', 'faction_recruitment',
         'territory', 'lord_type', 'spell_research', 'spell_casting', 'unit_swaps')


def cell(point):
    x, y = point
    return [y+1, x+1]


def site_cells(anchor):
    x, y = anchor
    return {(x+i, y+j) for i in range(3) for j in range(3)}


def landmark_cells(path):
    """Original (x, y) tiles of every landmark: houses, skeletons, landmark mountains, etc.

    The layout snapshots omit landmarks, but they block movement in the original
    (SLmark / CMidLandmark with GLmark CX x CY), as in build_bethrezens_retreat.py.
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    return {(x+i, y+j) for mark in data['landmarks'] for x, y in [mark['position']]
            for i in range(mark['size'][0]) for j in range(mark['size'][1])}


# Each original unit id has exactly one catalogue key.
UNIT_KEYS = {row['game_id']: key for key, row in UNITS.items()}


def roster(stack):
    slots = [None]*6
    for unit in stack['units']:
        slot = unit['position']-1  # red positions 1-3 front, 4-6 back
        if slots[slot] is not None or (unit['big'] and (slot > 2 or slots[slot+3] is not None)):
            raise ValueError(f"Stack {stack['id']}: overlapping units")
        slots[slot] = UNIT_KEYS[unit['unit_id']]
    return slots


def loot(names):
    potions, items = {}, {}
    for name in names:
        target = potions if name in POTIONS else items
        key = POTIONS.get(name) or ITEMS[name]
        target[key] = target.get(key, 0)+1
    return potions, items


def restore_original_units(enemies, table):
    """Reference stacks with the original unit ids of table {stack id: {reference id: original id}}.

    Fails unless each replaced id occurs in exactly the stacks the table names.
    """
    for replaced in {r for swap in table.values() for r in swap}:
        if ({e['id'] for e in enemies if any(u['unit_id'] == replaced for u in e['units'])}
                != {i for i, swap in table.items() if replaced in swap}):
            raise ValueError('Reference stacks with '+replaced+' differ from the original-unit table')
    return [dict(e, units=[dict(u, unit_id=table.get(e['id'], {}).get(u['unit_id'], u['unit_id']))
                           for u in e['units']]) for e in enemies]


def build(layout, landmarks):
    """JAX map from the reference layout, with the original shops, loot, units and landmark tiles."""
    stacks = sorted(restore_original_units(layout['enemies'], ORIGINAL_UNITS), key=lambda e: e['id'])
    if [e['id'] for e in stacks] != list(range(1, len(stacks)+1)):
        raise ValueError('Expected enemy ids 1..N')
    bags = [c for c in layout['chests'] if c['code'] not in DUPLICATE_CHESTS]
    if len(bags) != len(layout['chests'])-len(DUPLICATE_CHESTS):
        raise ValueError('A duplicate chest is missing from the layout')
    index = {e['id']: i for i, e in enumerate(stacks)}
    rosters = [roster(e) for e in stacks]
    sites = {s['code']: s for s in layout['sites']}
    removed = set().union(*(site_cells(s['position']) for s in layout['sites']),
                          *(site_cells(r['ruin_pos']) for r in layout['ruins']))
    objects = ({tuple(e['position']) for e in stacks} | {tuple(c['position']) for c in bags}
               | {tuple(p) for p in layout['gold_mines']} | {tuple(m['position']) for m in layout['mana']}
               | {tuple(c['position']) for c in layout['cities']+layout['capitals']})
    if landmarks & (objects | removed):
        raise ValueError('A landmark overlaps a map object, site or ruin')
    obstacles = sorted(cell(p) for p in {tuple(p) for p in layout['obstacles']} | landmarks if p not in removed)
    # Landmark tiles are obstacles. Three chests and a stack stand on water, as in the
    # original .sg (allow_water_objects below).
    terrain = {kind: [cell(p) for p in points if tuple(p) not in landmarks]
               for kind, points in (('road', layout['terrain']['road']),
                                    ('forest', layout['terrain']['forest']),
                                    ('water', layout['terrain']['water']))}
    capital, enemy_capital = layout['capitals']
    chests = []
    for chest in bags:
        potions, items = loot(chest['items'])
        chests.append(dict(position=cell(chest['position']), potions=potions, items=items))
    ruins = []
    for ruin in layout['ruins']:
        potions, items = loot([ruin['item']])
        ruins.append(dict(name=ruin['title'], position=cell(ruin['ruin_pos']),
                          potions=potions, items=items, gold=ruin['gold']))

    def site(code, **extra):
        return dict(name=sites[code]['name'], position=cell(sites[code]['position']), **extra)
    units = [[UNITS[k]['max_hp'] for k in r if k] for r in rosters]
    game_map = dict(
        name='NumberGrid-50x50-default',
        size=layout['grid_size']+2,
        agent_position=cell(capital['position']),
        agent_number=1,
        opponent_positions=[cell(e['position']) for e in stacks],
        opponent_numbers=[1]*len(stacks),
        # Default sets its own ruler below; the other builders keep copying it.
        **{key: CURRENT[key] for key in RULES if key != 'lord_type'},
        # Reference hero battle slots: orbs, talismans and battle potions (docs/ITEMS.md).
        battle_items=True,
        # Reference skill ladder: Pathfinding, Weapon Master, the lord's Might (docs/UNITS.md).
        hero_skills=True,
        number_generation='Legacy labels; numbers do not affect battles.',
        reconstruction_note=('Python reference Default map "'+layout['title']+'" ('+layout['map_id']
                             + '), generated by scripts/build_default_map.py; see docs/DEFAULT_MAP.md.'),
        hero_hp=150,
        enemy_units=[len(u) for u in units],
        enemy_hp=[max(u) for u in units],
        # No hero_roster: faction and lord_type choose the ruler's party (docs/UNITS.md).
        # User request 2026-10-25: every Default mode plays the Legions of the Damned. In the
        # original .sg the hero's western capital S001FT0001 belongs to S001PL0002 (G000RR0002,
        # Legions); the eastern one with Mizrael to the Empire. The reference played the Empire.
        faction='legions',
        # The reference's warrior ruler; fixed here because the current map's own ruler is a mage.
        lord_type='warrior',
        # Same .sg, S001PL0002's BANK G0400:R0200: 400 gold and 200 infernal mana at the start.
        starting_gold=STARTING_GOLD,
        starting_mana=STARTING_MANA,
        enemy_rosters=rosters,
        initial_potions=CAPITAL_POTIONS,
        initial_items={},
        chests=chests,
        merchants=[site('T1', potions=ALAR, items=ALAR_ITEMS), site('T2', potions=TRALAR)],
        trainer=site('U1'),
        # The reference calls H2 an empty camp; the original Erarick's Camp hires a Wolf.
        mercenaries=[site('H1', roster=[{'unit': 'g000uu5040', 'stock': 1}]),
                     dict(site('H2', roster=ERARICK), name='Лагерь Эрарика')],
        spell_shop=site('M1', spells=TOWER_SPELLS),
        # Reference goal "cities": capture FINAL_OBJECTIVE_CITIES (Порт Полонис, Соругирилла).
        cities=[dict(name=c['name'], position=cell(c['position']), level=c['level'], objective=c['objective'],
                     items=CITY_ITEMS.get(c['name'], {}))
                for c in layout['cities']],
        mines=([dict(kind='gold', position=cell(p)) for p in layout['gold_mines']]
               + [dict(kind=m['kind'], position=cell(m['position'])) for m in layout['mana']]),
        obstacles=obstacles,
        city_defender_roles={str(index[i]): role for i, role in ROLES.items()},
        enemy_capitals=[dict(name=enemy_capital['name'], position=cell(enemy_capital['position']),
                             guard=index[CAPITAL_GUARD],
                             footprint=sorted(cell(p) for p in enemy_capital['footprint']))],
        enemy_full_regeneration=[index[GREEN_DRAGON]],
        ruins=ruins,
        terrain=terrain,
        allow_water_objects=True,
    )
    game_map['battle_observation_version'] = max(42, game_map['battle_observation_version'])
    # Reference equipment lore: banner 7, boots 8, artifacts 9, book 11, the lord's own lore at once.
    game_map['equipment_requires_skills'] = True
    if game_map['opponent_positions'][index[CAPITAL_GUARD]] != game_map['enemy_capitals'][0]['position']:
        raise ValueError('Capital guard is not on the capital')
    if set(CITY_ITEMS) - {c['name'] for c in layout['cities']}:
        raise ValueError('City items name a city missing from the layout')
    return game_map


def dump(game_map):
    """One top-level key per line; lists of objects or cells one element per line."""
    def compact(value):
        return json.dumps(value, ensure_ascii=False)
    lines = []
    for key, value in game_map.items():
        if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
            body = ',\n'.join('    '+compact(v) for v in value)
            lines.append(f'  {compact(key)}: [\n{body}\n  ]')
        else:
            lines.append(f'  {compact(key)}: {compact(value)}')
    return '{\n'+',\n'.join(lines)+'\n}\n'


def main():
    """Read the reference layout and the landmark extract; write maps/default.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reference', type=Path, required=True,
                        help='Python reference root (contains Big_map/) or its Big_map folder')
    parser.add_argument('--output', type=Path, default=ROOT/'maps/default.json')
    args = parser.parse_args()
    base = args.reference/'Big_map' if (args.reference/'Big_map').is_dir() else args.reference
    layout = json.loads((base/'maps/default/layout.json').read_text(encoding='utf-8'))
    args.output.write_text(dump(build(layout, landmark_cells(LANDMARKS))), encoding='utf-8')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
