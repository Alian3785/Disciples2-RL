"""Give the current 48x48 map every unit behaviour type (user request 2026-10-17).

Offline JSON edit: no JAX, environment, policy or original game runs. The map
had 47 of the 52 catalogue types; each missing type gets one new single-unit
army. The representative is the minimum-HP mobile catalogue unit of the type
(the g000uu61xx/g000uu8107 copies are immobile guards with the same stats). All
five attack at range or by area, so they stand in the centre of the back row and
leave free cells for copies and summons. New armies take the most open passable
cells; weaker armies stand closer to the capital. Dark Laclaan stays on the map
but is not required for victory (optional_enemies). The ruler becomes a mage.
"""
import argparse
from collections import deque
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'number_grid_map.json'
UNITS = json.loads((ROOT / 'stoix/envs/data/units.json').read_text(encoding='utf-8'))
BEFORE, AFTER = 'NumberGrid-48x48-v87-original-effects', 'NumberGrid-48x48-v88-all-unit-types'
ADDED = ('doppelganger', 'occultist', 'occultmaster', 'lyf', 'laclaan')
OPTIONAL = ('laclaan',)
BACK_CENTRE = 4
PER_ARMY = ('opponent_positions', 'opponent_numbers', 'enemy_units', 'enemy_hp', 'enemy_rosters')


def square(anchor):
    r, c = anchor
    return {(r+i, c+j) for i in range(3) for j in range(3)}


def check_representatives(game_map):
    present = {UNITS[k]['unit_type'] for roster in game_map['enemy_rosters'] for k in roster if k}
    missing = {row['unit_type'] for row in UNITS.values()} - present
    if {UNITS[k]['unit_type'] for k in ADDED} != missing or len(ADDED) != len(missing):
        raise ValueError(f'Representatives must cover exactly the missing types: {sorted(missing)}')
    for key in ADDED:
        row = UNITS[key]
        same = [r for r in UNITS.values() if r['unit_type'] == row['unit_type']]
        mobile = [r['max_hp'] for r in same if r.get('movement')] or [r['max_hp'] for r in same]
        if row['max_hp'] != min(mobile) or row.get('size', 1) != 1 or row['role'] == 'melee':
            raise ValueError(f'{key} is not a small back-row minimum-HP representative')


def open_cells(game_map, count):
    """Most open passable cells, greedily spread; deterministic tie order."""
    size = game_map['size']
    blocked = {tuple(p) for p in game_map['obstacles']}
    for kind in ('merchant', 'trainer', 'mercenary'):
        if game_map.get(kind):
            blocked |= square(game_map[kind]['position'])
    for ruin in game_map.get('ruins', []):
        blocked |= square(ruin['position'])
    objects = {tuple(game_map['agent_position']), *(tuple(p) for p in game_map['opponent_positions'])}
    objects |= {tuple(o['position']) for kind in ('cities', 'mines', 'chests') for o in game_map[kind]}
    objects |= blocked
    terrain = {tuple(p) for kind in ('water', 'forest') for p in game_map.get('terrain', {}).get(kind, [])}
    start = tuple(game_map['agent_position'])
    reached, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r+dr, c+dc)
                if 0 < p[0] < size-1 and 0 < p[1] < size-1 and p not in blocked and p not in reached:
                    reached.add(p)
                    queue.append(p)
    candidates = sorted(p for p in reached if p not in objects and p not in terrain)
    chosen = []
    for _ in range(count):
        best = max(candidates, key=lambda p: min(max(abs(p[0]-q[0]), abs(p[1]-q[1])) for q in objects))
        chosen.append(best)
        objects.add(best)
    return sorted(chosen, key=lambda p: max(abs(p[0]-start[0]), abs(p[1]-start[1])))


def extend(game_map):
    if game_map['name'] != BEFORE:
        raise ValueError('Expected the unmodified v87 map')
    if any(len(game_map[key]) != len(game_map['enemy_rosters']) for key in PER_ARMY):
        raise ValueError('Per-army lists disagree')
    check_representatives(game_map)
    added = sorted(ADDED, key=lambda key: UNITS[key]['max_hp'])
    first = len(game_map['enemy_rosters'])
    for key, position in zip(added, open_cells(game_map, len(added))):
        roster = [None]*6
        roster[BACK_CENTRE] = key
        game_map['opponent_positions'].append(list(position))
        game_map['opponent_numbers'].append(1)
        game_map['enemy_units'].append(1)
        game_map['enemy_hp'].append(UNITS[key]['max_hp'])
        game_map['enemy_rosters'].append(roster)
    game_map['optional_enemies'] = [first+added.index(key) for key in OPTIONAL]
    game_map.update(name=AFTER, lord_type='mage')
    game_map['reconstruction_note'] = (game_map.get('reconstruction_note', '') + ' v88: armies 47-51 add '
        'Doppelganger, Summoner, Occultmaster, Lyf and Dark Laclaan; Laclaan is optional for victory; '
        'the default ruler is a Legions mage. See scripts/add_current_map_unit_types.py.').strip()
    return game_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MAP)
    args = parser.parse_args()
    game_map = extend(json.loads(MAP.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(game_map, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for index in range(len(game_map['enemy_rosters'])-len(ADDED), len(game_map['enemy_rosters'])):
        key = game_map['enemy_rosters'][index][BACK_CENTRE]
        print(index, key, UNITS[key]['name'], game_map['opponent_positions'][index],
              'optional' if index in game_map['optional_enemies'] else 'required')


if __name__ == '__main__':
    main()
