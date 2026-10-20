"""Convert the reference Small snapshot to the existing JAX campaign rules.

Reads JSON only; no environment, training, evaluation or GPU work. Small uses
the same shops/rules as Default. Explicit 3x3 footprint adaptations are recorded
in the output and documented in docs/SMALL_MAP.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import (
    ALAR, ALAR_ITEMS, CAPITAL_GUARD, CURRENT, GREEN_DRAGON, ROLES, RULES,
    SPELLS, UNITS, cell, dump, loot, roster, site_cells,
)

# Reference (x, y). Keep the existing JAX 3x3 objects and southeast approaches.
# Source small/layout.json stores scaled anchors/markers; they cannot be used
# directly as unscaled 3x3 buildings. These moves avoid armies and settlements.
SITE_ANCHORS = {'T1': (9, 17), 'M1': (19, 0), 'H1': (6, 20), 'U1': (18, 14)}
RUIN_ANCHORS = {70: (6, 3), 74: (20, 17)}
RUIN_ENTRANCES = {70: (8, 5), 74: (22, 19)}
OPEN_PASSAGES = {(21, 15), (19, 20)}
CHEST_POSITIONS = {'X1': (14, 22)}  # X1 originally shares (14, 23) with a gold mine.


def build(layout):
    """Convert Small while recording footprint relocations and preserving source army IDs."""
    if layout['map_id'] != 'small' or layout['grid_size'] != 24:
        raise ValueError('Expected the reference 24x24 Small snapshot')
    stacks = sorted(layout['enemies'], key=lambda e: e['id'])
    index = {e['id']: i for i, e in enumerate(stacks)}
    if len(index) != len(stacks):
        raise ValueError('Duplicate reference army IDs')
    rosters = [roster(e) for e in stacks]
    sites = {s['code']: s for s in layout['sites']}
    footprints = set().union(*(site_cells(p) for p in SITE_ANCHORS.values()),
                             *(site_cells(p) for p in RUIN_ANCHORS.values()))
    obstacles = {tuple(p) for p in layout['obstacles']} - footprints - OPEN_PASSAGES
    # Reference _sync_grid_terrain_positions: road > water > forest. Scaling
    # produces duplicates and overlaps; preserve the effective terrain type.
    terrain_by_cell = {tuple(p): kind for kind in ('forest', 'water', 'road')
                       for p in layout['terrain'][kind] if tuple(p) not in obstacles | footprints}
    terrain = {kind: sorted(cell(p) for p, k in terrain_by_cell.items() if k == kind)
               for kind in ('road', 'forest', 'water')}
    changes = [dict(kind='site', name=s['name'], source=s['position'],
                    position=list(SITE_ANCHORS[s['code']])) for s in layout['sites']]
    changes += [dict(kind='open_passage', position=list(p)) for p in sorted(OPEN_PASSAGES)]
    chests, ruins = [], []
    for chest in layout['chests']:
        potions, items = loot(chest['items'])
        position = CHEST_POSITIONS.get(chest['code'], chest['position'])
        chests.append(dict(position=cell(position), potions=potions, items=items))
        if list(position) != chest['position']:
            changes.append(dict(kind='chest', name=chest['code'], source=chest['position'],
                                position=list(position)))
    for ruin in layout['ruins']:
        eid = ruin['id']
        potions, items = loot([ruin['item']])
        ruins.append(dict(name=ruin['title'], position=cell(RUIN_ANCHORS[eid]),
                          entrance=cell(RUIN_ENTRANCES[eid]), gold=ruin['gold'],
                          potions=potions, items=items))
        changes.append(dict(kind='ruin', reference_id=eid, source=ruin['ruin_pos'],
                            position=list(RUIN_ANCHORS[eid]), source_entrance=ruin['marker_pos'],
                            entrance=list(RUIN_ENTRANCES[eid])))

    def site(code, **extra):
        """Build a service record at its collision-free anchor with the supplied inventory."""
        return dict(name=sites[code]['name'], position=cell(SITE_ANCHORS[code]), **extra)

    capital = next(c for c in layout['capitals'] if c['position'] != layout['hero'])
    game_map = dict(
        name='NumberGrid-26x26-small', display_name='Small — города', size=26,
        agent_position=cell(layout['hero']), agent_number=1,
        opponent_positions=[cell(RUIN_ENTRANCES.get(e['id'], e['position'])) for e in stacks],
        opponent_numbers=[1]*len(stacks), **{k: CURRENT[k] for k in RULES},
        battle_items=True, hero_skills=True, faction='empire', hero_hp=150,
        starting_gold=int(layout['starting_gold']), campaign_objective='cities',
        enemy_rosters=rosters, enemy_units=[sum(k is not None for k in r) for r in rosters],
        enemy_hp=[max(UNITS[k]['max_hp'] for k in r if k) for r in rosters],
        enemy_initial_levels=[[max(UNITS[k]['level'], next(u['Level'] for u in e['units']
                                if u['position'] == i+1)) if k else 0
                               for i, k in enumerate(r)] for e, r in zip(stacks, rosters)],
        reference_enemy_ids=[e['id'] for e in stacks],
        reference_source='Big_map/maps/small.py; Big_map/maps/small/layout.json',
        reference_layout_adjustments=changes,
        initial_potions={}, initial_items={}, chests=chests, ruins=ruins,
        merchants=[site('T1', potions=ALAR, items=ALAR_ITEMS)], trainer=site('U1'),
        mercenaries=[site('H1', roster=[dict(unit='g000uu5040', price=850, stock=1)])],
        spell_shop=site('M1', spells=SPELLS),
        cities=[dict(name=c['name'], position=cell(c['position']), level=c['level'], objective=True)
                for c in layout['cities']],
        mines=[dict(kind='gold', position=cell(p)) for p in layout['gold_mines']]
              + [dict(kind=m['kind'], position=cell(m['position'])) for m in layout['mana']],
        city_defender_roles={str(index[eid]): role for eid, role in ROLES.items() if eid in index},
        enemy_capitals=[dict(name=capital['name'], position=cell(capital['position']),
                             guard=index[CAPITAL_GUARD],
                             footprint=sorted({tuple(cell(p)) for p in capital['footprint']}))],
        enemy_full_regeneration=[index[GREEN_DRAGON]],
        obstacles=sorted(cell(p) for p in obstacles), terrain=terrain, allow_water_objects=True,
    )
    game_map['battle_observation_version'] = 44
    game_map['equipment_requires_skills'] = True
    return game_map


def main():
    """Write the adapted Small map from a local reference layout without running the game."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'maps/small.json')
    args = parser.parse_args()
    base = args.reference/'Big_map' if (args.reference/'Big_map').is_dir() else args.reference
    layout = json.loads((base/'maps/small/layout.json').read_text(encoding='utf-8'))
    args.output.write_text(dump(build(layout)), encoding='utf-8')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
