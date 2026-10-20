"""Import Mirror match's checked-in layout; no environment or GPU execution.

The snapshot omits the optional bot, so its ordinary roster is taken from
maps/mirror_match.py. Coordinates (x,y) become [y+1,x+1] inside the JAX wall.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import CURRENT, RULES, UNITS, cell, dump, roster


def build(layout, source_hash):
    """Convert the pinned 48x48 layout and Empire roster into a bordered JAX map."""
    if layout['map_id'] != 'mirror_match' or layout['grid_size'] != 48:
        raise ValueError('Expected the reference 48x48 Mirror match layout')
    stacks = sorted(layout['enemies'], key=lambda e: e['id'])
    rosters = [roster(e) for e in stacks]
    keys = {r['name']: k for k, r in UNITS.items()}
    bot = [keys[n] if n else None for n in
           ('Скваер', 'Рыцарь на пегасе', 'Скваер', None, 'Служка', None)]
    levels = [[max(UNITS[k]['level'], next(u['Level'] for u in e['units']
               if u['position'] == i+1)) if k else 0 for i, k in enumerate(r)]
              for e, r in zip(stacks, rosters)]
    rosters.append(bot)
    levels.append([UNITS[k]['level'] if k else 0 for k in bot])
    home = cell((42, 42))
    result = dict(
        name='NumberGrid-50x50-mirror-match', display_name='Mirror match', size=50,
        **{k: CURRENT[k] for k in RULES},
        agent_position=cell(layout['hero']), agent_number=1,
        opponent_positions=[cell(e['position']) for e in stacks]+[home],
        opponent_numbers=[1]*len(rosters), enemy_rosters=rosters,
        enemy_units=[sum(k is not None for k in r) for r in rosters],
        enemy_hp=[max(UNITS[k]['max_hp'] for k in r if k) for r in rosters],
        enemy_initial_levels=levels, hero_hp=150, faction='legions',
        starting_gold=int(layout['starting_gold']), initial_potions={}, initial_items={},
        battle_items=True, hero_skills=True, chests=[], cities=[], ruins=[],
        obstacles=[cell(p) for p in layout['obstacles']],
        terrain={kind: [cell(p) for p in layout['terrain'][kind]]
                 for kind in ('water', 'forest', 'road')},
        mines=[dict(kind='gold', position=cell(p)) for p in layout['gold_mines']]
              + [dict(kind=m['kind'], position=cell(m['position'])) for m in layout['mana']],
        enemy_territory_sources=[dict(position=home, growth=10,
            footprint=[cell(p) for c in layout['capitals'] if cell(c['position']) == home
                       for p in c['footprint']])],
        scripted_bot=dict(enemy=len(stacks), home=home, movement=20,
                          respawn_turns=2, battle_decisions=300, faction='empire'),
        campaign_objective='scripted_bot', reward_profile='python_reference',
        # User request 2026-10-14: twice the reference +8 for beating the bot.
        campaign_victory_reward=16.,
        reference_enemy_ids=[e['id'] for e in stacks]+[-75],
        reference_source='Big_map/maps/mirror_match.py; Big_map/maps/mirror_match/layout.json',
        reference_commit='2df84cbed7fc1e2a1848aa5d00cde9db271b520b',
        reference_layout_sha256=source_hash,
    )
    result.update(step_cost=0., exploration_bonus=.08, building_reward=0., lord_type='warrior',
                  battle_observation_version=45)
    return result


def main():
    """Read the reference snapshot, retain its digest, and write the requested map JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'maps/mirror_match.json')
    args = parser.parse_args()
    base = args.reference/'Big_map' if (args.reference/'Big_map').is_dir() else args.reference
    data = (base/'maps/mirror_match/layout.json').read_bytes()
    args.output.write_text(dump(build(json.loads(data), hashlib.sha256(data).hexdigest())),
                           encoding='utf-8')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
