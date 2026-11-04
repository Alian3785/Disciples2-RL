"""Land fields of a map JSON: start paint, tiles that stop the land front, fort footprints.

Rules: docs/WORLD.md#territory-владения-и-доход. Original scenarios use the snapshot that
scripts/extract_land.py writes to docs/maps/<map>/land.json; maps made from a reference
layout have no original terrain, so only their capital starts painted.
"""
import json
from pathlib import Path

from scripts.build_default_map import cell

ROOT = Path(__file__).resolve().parents[1]
WATER, MOUNTAIN, NEUTRAL = 3, 4, 5  # Lground L_WATER, L_MOUNTAIN; Lterrain L_NE
TERRAIN = dict(empire=1, mountain_clans=2, legions=3, undead_hordes=4, elves=6)  # Lterrain


def scenario_land(game_map, name):
    """Add the land of an original scenario from its docs/maps/<name>/land.json snapshot."""
    land = json.loads((ROOT/'docs/maps'/name/'land.json').read_text(encoding='utf-8'))
    own = TERRAIN[game_map['faction']]
    tiles = [((x, y), code) for y, row in enumerate(land['tiles']) for x, code in enumerate(row)]
    game_map['land_blocked'] = sorted(map(list, {tuple(cell(t)) for t, code in tiles if code >> 3 in (WATER, MOUNTAIN)}
                                          | {tuple(cell(t)) for t in land['mountain_landmarks']}))
    game_map['starting_land'] = sorted(cell(t) for t, code in tiles if code & 7 == own)
    game_map['foreign_land'] = sorted(cell(t) for t, code in tiles if code & 7 not in (own, NEUTRAL))
    game_map['capital_footprint'] = sorted(cell(t) for t in land['capital'])
    forts = {c['name']: c for c in land['cities']}
    for city in game_map['cities']:
        fort = forts[city['name']]
        city.update(footprint=sorted(cell(t) for t in fort['footprint']), neutral=fort['neutral'])


def layout_land(game_map, layout):
    """Add the land of a map made from a reference layout: its capital alone starts painted."""
    # Scaled layouts such as Small repeat footprint tiles.
    def footprint(fort):
        return sorted(map(list, {tuple(cell(p)) for p in fort['footprint']}))

    forts, home = [], game_map.get('capital_position', game_map['agent_position'])
    for capital in layout['capitals']:
        forts += footprint(capital)
        if cell(capital['position']) == home:
            game_map['capital_footprint'] = game_map['starting_land'] = footprint(capital)
    if 'capital_footprint' not in game_map:
        raise ValueError('The layout has no capital at the party start')
    for city, source in zip(game_map['cities'], layout['cities']):
        city['footprint'] = footprint(source)
        forts += city['footprint']
    forts = {tuple(p) for p in forts}
    game_map['land_blocked'] = sorted(p for p in game_map['obstacles'] if tuple(p) not in forts)
