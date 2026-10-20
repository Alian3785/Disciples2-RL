"""Convert reference campaigns to standalone, versioned JAX map files.

Only reads reference map definitions and layout snapshots; never runs its environment.
Run with scripts/run_gpu.sh and --reference pointing to the reference directory.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import CURRENT, RULES, UNITS, cell, dump, roster, site_cells
from stoix.envs.number_grid_items import ITEMS
from stoix.envs.number_grid_potions import POTIONS

NAMES = ('super_last_stand', 'siege_train', 'wotans_retribution', 'trade_train')
ALIASES = {
    'Potion of Healing': 'healing', 'Potion of Restoration': 'restoration', 'Life Potion': 'life',
    'Эликсир Жизни': 'life', 'Эликсир восстановления': 'restoration',
    'Potion of Vigor': 'vigor', 'Potion of Strength': 'strength', 'Potion of Might': 'might',
    'Potion of Protection': 'protection', 'Treebark Potion': 'bark', 'Potion of Accuracy': 'accuracy',
    'Potion of Invulnerability': 'invulnerability', 'Potion of Celerity': 'celerity',
    'Potion of Swiftness': 'swiftness', 'Potion of Speed': 'speed', 'Potion of Striking': 'striking',
    'Potion of Water Warding': 'water_ward', 'Potion of Fire Warding': 'fire_ward',
    'Potion of Earth Warding': 'earth_ward', 'Potion of Air Warding': 'air_ward',
    'Titan Might Potion': 'titan', 'Highfather Potion': 'highfather',
    "Titan's Might Potion": 'titan',
    'Кольцо веков (Артефакт)': 'ring_of_ages', 'Staff of Travelling': 'staff_of_travelling',
    'Свиток "Призывание II: Белиарх"': 'incantare_beliarh_scroll',
    'Зелье силы (+30%)': 'strength', 'Зелье точности (+30%)': 'accuracy',
}


def item_key(name):
    name = name.strip()
    name = name.replace('защиты от магии ', 'защиты от ')
    if name in ALIASES:
        return ALIASES[name]
    clean = name.replace(' (Valuable)', '').replace(' (Artifact)', '')
    candidates = [p['key'] for p in (*POTIONS, *ITEMS) if p['name'].casefold() == clean.casefold()]
    if len(candidates) != 1:
        raise ValueError('Unmapped reference item: '+name)
    return candidates[0]


def loot(names):
    potions, items = {}, {}
    potion_keys = {p['key'] for p in POTIONS}
    for name in names:
        key = item_key(name)
        target = potions if key in potion_keys else items
        target[key] = target.get(key, 0)+1
    return potions, items


def build(base, name):
    """Translate a reference scenario and its layout into JAX armies, events, and rewards."""
    module = importlib.import_module('maps.'+name)
    from unit_revive_costs import REVIVE_PROFILE_IDS_BY_NAME
    config = module.MAP
    layout = json.loads((base/'maps'/name/'layout.json').read_text(encoding='utf-8'))
    stacks = sorted(layout['enemies'], key=lambda e: e['id'])
    index = {e['id']: i for i, e in enumerate(stacks)}
    rosters = [roster(e) for e in stacks]
    by_id = {r['game_id']: k for k, r in UNITS.items()}
    keys = {unit_name: by_id[ids[0]] for unit_name, ids in REVIVE_PROFILE_IDS_BY_NAME.items()
            if ids and ids[0] in by_id}
    hero = [None]*6
    overrides = {}
    for slot, unit in config.starting_roster.items():
        hero[slot-7] = keys[unit]
        override = dict((config.starting_unit_overrides or {}).get(slot, {}))
        if 'dynamic_level_exp_increment' in override:
            override['exp_increment'] = override.pop('dynamic_level_exp_increment')
        override.pop('next_level_exp', None)
        overrides[slot-7] = override
    potions, items = loot(config.starting_hero_items)
    removed = set().union(*(site_cells(s['position']) for s in layout['sites']),
                          *(site_cells(r['ruin_pos']) for r in layout['ruins']))
    obstacles = [cell(p) for p in layout['obstacles'] if tuple(p) not in removed]
    blocked = {tuple(p) for p in layout['obstacles']} | removed
    terrain = {k: [cell(p) for p in layout['terrain'][k] if tuple(p) not in blocked]
               for k in ('water', 'forest', 'road')}
    size = layout['grid_size']+2
    game_map = dict(name=f'NumberGrid-{size}x{size}-'+name, size=size,
        agent_position=cell(layout['hero']), agent_number=1,
        opponent_positions=[cell(e['position']) for e in stacks], opponent_numbers=[1]*len(stacks),
        **{k: CURRENT[k] for k in RULES}, battle_items=True, hero_skills=True,
        faction={1:'empire', 2:'legions', 3:'mountain_clans', 4:'undead_hordes', 5:'elves'}[config.starting_capital_id],
        hero_roster=hero, hero_units=sum(k is not None for k in hero), hero_hp=150,
        hero_combat_stats=[overrides[i] for i, k in enumerate(hero) if k],
        starting_gold=int(config.starting_gold or 0),
        starting_built_buildings=list(config.starting_built_buildings),
        starting_hero_abilities=list(config.starting_hero_abilities),
        starting_scroll_magic_unlocked=config.starting_scroll_magic_unlocked,
        enemy_rosters=rosters, enemy_units=[sum(k is not None for k in r) for r in rosters],
        enemy_hp=[max(UNITS[k]['max_hp'] for k in r if k) for r in rosters],
        enemy_initial_levels=[[max(UNITS[key]['level'], next(u['Level'] for u in e['units']
                                if u['position']==i+1)) if key else 0
                               for i, key in enumerate(squad)] for e, squad in zip(stacks, rosters)],
        initial_potions=potions, initial_items=items, obstacles=obstacles, terrain=terrain,
        allow_water_objects=True,
        chests=[], ruins=[], cities=[dict(name=c['name'], position=cell(c['position']),
                                         level=c['level'], objective=c['objective']) for c in layout['cities']],
        mines=[dict(kind='gold', position=cell(p)) for p in layout['gold_mines']]+
              [dict(kind=m['kind'], position=cell(m['position'])) for m in layout['mana']],
        campaign_objective=layout['objective'], reward_profile='python_reference',
        reference_enemy_ids=[e['id'] for e in stacks],
        reference_source='Big_map/maps/'+name+'.py; maps/'+name+'/layout.json',
        scheduled_enemies=[dict(enemy=index[e['id']], **{k:v for k,v in e['wave'].items() if k!='enemy_id'})
                           for e in stacks if e.get('wave')],
    )
    game_map['lord_type'] = {1:'warrior', 2:'mage', 3:'guildmaster'}[config.starting_lord_type]
    game_map.update(step_cost=0., exploration_bonus=.08, building_reward=0., battle_observation_version=44)
    if layout.get('objective_enemy_id') is not None:
        game_map['objective_enemy'] = index[layout['objective_enemy_id']]
    if name == 'siege_train':
        # User-requested JAX training shaping, not an original game mechanic.
        game_map['siege_rewards'] = dict(victory_multiplier=3., unit_kill=5., spawn_level=5.)
    if name == 'trade_train':
        # User-requested entry levels above 1, paid only for the hero army's victory.
        game_map['army_victory_level_reward'] = 5.
    for chest in layout['chests']:
        p, it = loot(chest['items'])
        game_map['chests'].append(dict(position=cell(chest['position']), potions=p, items=it))
    for ruin in layout['ruins']:
        p, it = loot(ruin.get('items') or [ruin['item']])
        game_map['ruins'].append(dict(name=ruin['title'], position=cell(ruin['ruin_pos']),
                                     entrance=cell(ruin['marker_pos']), gold=ruin['gold'], potions=p, items=it))
    for kind in ('merchant', 'trainer', 'spell_shop', 'mercenary'):
        definitions = getattr(config, kind+'_sites')
        sites = []
        for site_name, spec in definitions.items():
            # Layout snapshots already contain scaled/conflict-resolved coordinates.
            # trade_train's source anchors are on 48x48, but its playable map is 24x24.
            snapshot = next(s for s in layout['sites'] if s['name'] == site_name)
            site = dict(name=site_name, position=cell(snapshot['position']))
            if kind == 'merchant':
                offers = (config.merchant_buy_items_by_site or {}).get(site_name, config.merchant_buy_items)
                site.update(potions={}, items={}, prices={})
                for offer in offers:
                    key = item_key(offer['name'])
                    target = 'potions' if any(p['key']==key for p in POTIONS) else 'items'
                    site[target][key] = int(offer['stock'])
                    site['prices'][key] = int(offer['price'])
            if kind == 'mercenary':
                site['roster'] = [dict(unit=keys[offer['unit_name']],
                                       price=int(offer['gold']), stock=int(offer['stock']))
                                  for offer in spec['roster']]
            if kind == 'spell_shop':
                site['spells'] = [dict(spell='g000ss0019', price=1000, stock=1),
                                  dict(spell='g000ss0058', price=1000, stock=1)]
            sites.append(site)
        if sites:
            game_map[dict(merchant='merchants', trainer='trainers', spell_shop='spell_shops', mercenary='mercenaries')[kind]] = sites
    capitals = [c for c in layout['capitals'] if cell(c['position']) != game_map['agent_position']]
    game_map['enemy_capitals'] = [dict(name=c['name'], position=cell(c['position']),
        guard=next(i for i,e in enumerate(stacks) if e['position']==c['position']),
        footprint=[cell(p) for p in c['footprint']]) for c in capitals]
    return game_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--map', action='append', choices=NAMES, dest='maps',
                        help='Only rebuild the selected map(s); by default rebuild all reference campaigns')
    args = parser.parse_args()
    base = args.reference/'Big_map' if (args.reference/'Big_map').is_dir() else args.reference
    sys.path.insert(0, str(base))
    for name in args.maps or NAMES:
        target = ROOT/'maps'/f'{name}.json'
        target.write_text(dump(build(base, name)), encoding='utf-8')
        print('Wrote', target)


if __name__ == '__main__':
    main()
