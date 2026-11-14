"""Convert reference campaigns to standalone, versioned JAX map files.

Only reads reference map definitions and layout snapshots; never runs its environment.
Run with scripts/run_gpu.sh and --reference pointing to the reference directory.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_map import (CURRENT, RULES, UNITS, cell, dump, landmark_cells,
                                       restore_original_formations, restore_original_units, roster,
                                       site_cells)
from scripts.land_data import layout_land, scenario_land
from stoix.envs.number_grid_items import ITEMS
from stoix.envs.number_grid_potions import POTIONS
from stoix.envs.number_grid_spell_items import SPELL_ITEMS

NAMES = ('super_last_stand', 'siege_train', 'the_charmed_child', 'trade_train')
FOCUSED_NAMES = ('scroll_train', 'magic_train', 'item_train', 'hire_train', 'orc_duel', 'formation_train',
                 'city_defence_train')
# Reference campaign_env_city_defence.py: the scripted bot walks ten plain tiles a day.
CITY_DEFENCE_MOVES, CITY_DEFENCE_RESPAWN_DAYS = 10, 2
SOURCE_NAMES = {'the_charmed_child': 'wotans_retribution'}  # Legacy Python-reference filename.
# Installed Scens SLmark, extracted by scripts/extract_landmarks.py; the layout omits landmarks.
LANDMARKS = {'the_charmed_child': ROOT/'docs/maps/the_charmed_child/landmarks.json'}
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


# Installed Scens S117 (SInventr/SItem vaults; SEvCond category 4 "own city" with SEvEffec
# category 9 "give item", all for the Mountain Clans player). User request 2026-10-25: grant
# them on capture, except quest items (Idun's Prosperity Rune, S117EV0005); the victory
# event's Talisman of Regeneration (S117EV0002) would arrive with the scenario's end.
CHARMED_CITY_LOOT = {
    'Fogfell': dict(potions={'restoration': 1, 'life': 1}),            # S117FT0002 vault
    'Odzark': dict(potions={'restoration': 1}, items={'emerald': 1}),  # S117FT0007 vault
    'Akar': dict(items={'orc_orb': 1}),                                 # S117EV0008
    "Hasser's Keep": dict(items={'staff_of_travelling': 1}),            # S117EV000E
}
# S117EV001E: Frequency 6, occur once -> the Ancient Relic on day 6.
CHARMED_DAY_ITEMS = [dict(day=6, items={'ancient_relic': 1})]
# Vault of the player's capital S117FT0001: 3x G000IG0006 and 1x G000IG0001; the layout omits it.
CHARMED_CAPITAL_POTIONS = {'restoration': 3, 'life': 1}
# S117 garrisons that the reference's name-based profiles replaced, by reference stack id:
# 5, the Empire capital S117FT0000, holds the Cleric G000UU0016 (heals all allies by 20), not the
# Priest G000UU0012 (one ally by 40); 8, Hasser's Keep S117FT0006, the Holy Avenger G000UU0155
# (250 HP, armor 0, initiative 50, 75 damage twice), not the Defender of Faith G000UU0156
# (225, 30, 70, 125 once). No other S117 stack holds either replaced id.
CHARMED_ORIGINAL_UNITS = {5: {'g000uu0012': 'g000uu0016'}, 8: {'g000uu0156': 'g000uu0155'}}
ORIGINAL_UNITS = {'the_charmed_child': CHARMED_ORIGINAL_UNITS}
# User request 2026-11-05: the hero starts at level 2; S117 SUnit records S117UN0003 at level 1.
CHARMED_HERO_LEVEL = 2
# Treasury on turn 1 of S117: SPlayer BANK G0100 plus the income the game credits before the first
# move (saves at CUR_TURN 1: BANK G0250:W0075). Source: docs/validation/first-turn-treasury-20261106.json.
CHARMED_TREASURY = dict(starting_gold=250, starting_mana={'runes': 75})
# User request 2026-11-05. Mines block movement as in the original, and the reference puts all
# four sources on the capital's east exit, which would lock the hero in. Here they touch its walls.
MAGIC_MANA_SOURCES = {'life': [26, 7], 'infernal': [30, 7], 'runes': [25, 4], 'death': [31, 4]}
# User request 2026-11-05: Hire plays the Empire, so its hero is the Empire warrior's Pegasus
# Knight, not the reference's Duke of the Legions. Same statistics, and both fly.
HIRE_HERO = 'g000uu0019'


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
    source_name = SOURCE_NAMES.get(name, name)
    module = importlib.import_module('maps.'+source_name)
    from unit_revive_costs import REVIVE_PROFILE_IDS_BY_NAME
    config = module.MAP
    layout = json.loads((base/'maps'/source_name/'layout.json').read_text(encoding='utf-8'))
    stacks = sorted(restore_original_units(layout['enemies'], ORIGINAL_UNITS.get(name, {})), key=lambda e: e['id'])
    index = {e['id']: i for i, e in enumerate(stacks)}
    rosters = [roster(e) for e in stacks]
    by_id = {r['game_id']: k for k, r in UNITS.items()}
    keys = {unit_name: by_id[ids[0]] for unit_name, ids in REVIVE_PROFILE_IDS_BY_NAME.items()
            if ids and ids[0] in by_id}
    hero = [None]*6
    overrides = {}
    for slot, unit in (config.starting_roster or {}).items():
        hero[slot-7] = keys[unit]
        override = dict((config.starting_unit_overrides or {}).get(slot, {}))
        if 'dynamic_level_exp_increment' in override:
            override['exp_increment'] = override.pop('dynamic_level_exp_increment')
        override.pop('next_level_exp', None)
        overrides[slot-7] = override
    if name == 'the_charmed_child':
        # S117FT0001's S117UN00EC is a level-2 Crossbowman with 110 HP.
        # The reference put a level-1 Axe Thrower in the party's rear slot 3.
        hero[3] = by_id['g000uu0027']
        overrides[3] = {}
    potions, items = loot(config.starting_hero_items)
    if config.starting_scroll_items:
        from scroll_spell_data import SCROLL_ITEM_DEFINITIONS
        reference_scrolls = {item['item_name']: item for item in SCROLL_ITEM_DEFINITIONS}
        scroll_keys = {item['spell']: item['key'] for item in SPELL_ITEMS if item['category'] == 'scroll'}
        for name_in_inventory in config.starting_scroll_items:
            key = scroll_keys[reference_scrolls[name_in_inventory]['spell_id'].lower()]
            items[key] = items.get(key, 0)+1
    removed = set().union(*(site_cells(s['position']) for s in layout['sites']),
                          *(site_cells(r['ruin_pos']) for r in layout['ruins']))
    landmarks = landmark_cells(LANDMARKS[name]) if name in LANDMARKS else set()
    objects = ({tuple(layout['hero'])} | {tuple(e['position']) for e in stacks}
               | {tuple(c['position']) for c in layout['chests']+layout['cities']+layout['capitals']}
               | {tuple(p) for p in layout['gold_mines']} | {tuple(m['position']) for m in layout['mana']}
               | {tuple(r['marker_pos']) for r in layout['ruins']})
    if landmarks & (objects | removed):
        raise ValueError('A landmark overlaps a map object, site or ruin')
    obstacles = [cell(p) for p in layout['obstacles'] if tuple(p) not in removed]
    obstacles += sorted(cell(p) for p in landmarks - {tuple(p) for p in layout['obstacles']})
    blocked = {tuple(p) for p in layout['obstacles']} | removed | landmarks
    terrain = {k: [cell(p) for p in layout['terrain'][k] if tuple(p) not in blocked]
               for k in ('water', 'forest', 'road')}
    size = layout['grid_size']+2
    game_map = dict(name=f'NumberGrid-{size}x{size}-'+name, size=size,
        agent_position=cell(layout['hero']), agent_number=1,
        opponent_positions=[cell(e['position']) for e in stacks], opponent_numbers=[1]*len(stacks),
        **{k: CURRENT[k] for k in RULES}, battle_items=True, hero_skills=True,
        faction={1:'empire', 2:'legions', 3:'mountain_clans', 4:'undead_hordes', 5:'elves'}[
            config.starting_capital_id or config.default_capital_id],
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
        reference_source='Big_map/maps/'+source_name+'.py; maps/'+source_name+'/layout.json',
        scheduled_enemies=[dict(enemy=index[e['id']], **{k:v for k,v in e['wave'].items() if k!='enemy_id'})
                           for e in stacks if e.get('wave')],
    )
    if name in ('super_last_stand', 'trade_train', 'the_charmed_child'):
        game_map['equipment_requires_skills'] = True
    if name == 'super_last_stand':
        # User-requested difficulty setting: the Temple must be built during play.
        game_map['starting_built_buildings'] = [
            building for building in game_map['starting_built_buildings'] if building != 'Храм']
    if name == 'the_charmed_child':
        game_map.update(display_name='Проклятое дитя', original_name='The Charmed Child',
                        scenario_id='S117SC0000', day_items=CHARMED_DAY_ITEMS, **CHARMED_TREASURY)
        if set(CHARMED_CITY_LOOT) - {c['name'] for c in game_map['cities']}:
            raise ValueError('Charmed Child loot names a city missing from the layout')
        for city in game_map['cities']:
            vault = CHARMED_CITY_LOOT.get(city['name'], {})
            city.update(potions=vault.get('potions', {}), items=vault.get('items', {}))
        for key, count in CHARMED_CAPITAL_POTIONS.items():
            game_map['initial_potions'][key] = game_map['initial_potions'].get(key, 0)+count
        game_map['hero_initial_levels'] = [(CHARMED_HERO_LEVEL if UNITS[key].get('hero') else UNITS[key]['level']) if key else 0
                                           for key in hero]
    if name == 'magic_train':
        game_map['mines'] = [dict(kind=m['kind'], position=MAGIC_MANA_SOURCES[m['kind']]) for m in game_map['mines']]
    if name == 'hire_train':
        game_map['hero_roster'] = [HIRE_HERO if key else None for key in hero]
    game_map['lord_type'] = {1:'warrior', 2:'mage', 3:'guildmaster'}[config.starting_lord_type or 1]
    if config.starting_roster is None:
        for key in ('hero_roster', 'hero_units', 'hero_combat_stats'):
            game_map.pop(key)
    game_map.update(step_cost=0., exploration_bonus=.08, building_reward=0., battle_observation_version=44)
    if name in FOCUSED_NAMES:
        game_map.update(display_name=layout['title'], equipment_requires_skills=True,
                        passive_mana_income_enabled=config.passive_mana_income_enabled,
                        enforce_enemy_reachability=config.enforce_enemy_reachability)
        game_map['reference_sha256'] = {
            suffix: hashlib.sha256((base/'maps'/suffix).read_bytes()).hexdigest()
            for suffix in (name+'.py', name+'/layout.json')}
        for key in ('item_pickup_rewards', 'battle_item_equip_reward', 'battle_item_use_reward',
                    'leadership_step_penalty', 'turn_penalty', 'spell_cast_reward', 'new_cell_reward',
                    'rest_with_moves_penalty', 'timeout_reward', 'magic_enemy_defeat_reward'):
            value = getattr(config, key)
            if value is not None:
                game_map[key] = ({item_key(k): v for k, v in value.items()}
                                 if key == 'item_pickup_rewards' else value)
        if config.all_enemies_reward is not None:
            game_map['campaign_victory_reward'] = config.all_enemies_reward
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
    if name == 'the_charmed_child':
        scenario_land(game_map, name)
        restore_original_formations(game_map, name, game_map['reference_enemy_ids'])
    else:
        layout_land(game_map, layout)
    if name == 'city_defence_train':
        add_city_assaults(game_map, module, keys)
    return game_map


def add_city_assaults(game_map, module, keys):
    """Two assaults on the player's empty level-5 city (maps/city_defence_train.py).

    The reference's scripted bot becomes two scheduled armies with the city as their target:
    the first appears on the first REST and walks ten tiles a day; the second appears two
    RESTs after the first is destroyed. Win: both destroyed; loss: the city falls.
    """
    city = next(c for c in game_map['cities'] if c['name'] == module.CITY_NAME)
    city.update(owned=True, defended=True)
    for wave, roster_by_position in enumerate(module.ASSAULT_ROSTERS):
        squad = [None]*6
        for position, unit_name in roster_by_position.items():
            squad[position-7] = keys[unit_name]  # blue 7..12 become red 1..6 in the city battle
        game_map['opponent_positions'].append(cell(module.BOT_SPAWN_TILE))
        game_map['opponent_numbers'].append(1)
        game_map['enemy_rosters'].append(squad)
        game_map['enemy_units'].append(sum(k is not None for k in squad))
        game_map['enemy_hp'].append(max(UNITS[k]['max_hp'] for k in squad if k))
        game_map['enemy_initial_levels'].append([UNITS[k]['level'] if k else 0 for k in squad])
        game_map['reference_enemy_ids'].append(f'assault_{wave+1}')
        event = dict(enemy=len(game_map['opponent_positions'])-1, moves_per_turn=CITY_DEFENCE_MOVES,
                     target=city['position'])
        if wave == 0:
            event['spawn_turn'] = 0
        else:
            event['after'] = dict(enemy=event['enemy']-1, days=CITY_DEFENCE_RESPAWN_DAYS)
        game_map['scheduled_enemies'].append(event)
    game_map.update(campaign_objective='waves', city_defence=dict(
        city=module.CITY_NAME, factions=list(module.ASSAULT_FACTIONS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--map', action='append', choices=NAMES+FOCUSED_NAMES, dest='maps',
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
