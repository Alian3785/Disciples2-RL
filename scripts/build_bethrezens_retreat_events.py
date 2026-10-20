"""Compile S101 event DBFs into fixed-size device tables; no JAX execution.

Sources: pinned layout.json, installed scenario editor manual, CMidStackTemplate.
Only gameplay effects survive. Enabling/disabling dialogue-only events is also
discarded. Original win/loss events are handled by the campaign's objective.
"""
from scripts.build_default_map import cell

HERO = 'S101KC0013'
NULL = 'G000000000'


def template_formation(row, unit_keys, units):
    """Template positions refer to unit *types*, unlike SUGroup instances.

    Repeated small types are separate fighters; a large type occupies two cells.
    See D2ModdingToolset mss32/include/midstacktemplate.h: positions[6].
    """
    roster, levels = [None]*6, [0]*6
    for pos in range(6):
        index = int(row[f'POS_{pos}'])
        if index < 0:
            continue
        key = unit_keys[row[f'UNIT_{index}'].upper()]
        big = units[key]['size'] == 2
        if big:
            partner = int(row[f'POS_{pos ^ 1}'])
            if partner < 0 or row[f'UNIT_{partner}'] != row[f'UNIT_{index}']:
                raise ValueError('Broken large template unit: '+row['ID'])
            if pos % 2:
                continue
        slot = pos//2+3*(pos % 2)
        roster[slot] = key
        levels[slot] = max(units[key]['level'], int(row[f'UNIT_{index}_LVL']))
    return roster, levels


def add_events(game_map, snapshot, catalog, unit_keys, units):
    raw = snapshot['raw_tables']
    conditions = {r['ID']: r['TEXT'][2:] for r in raw['LEvCond']}
    effects = {r['ID']: r['TEXT'][2:] for r in raw['LEvEffct']}
    locations = {r['LOC_ID']: dict(position=cell((int(r['POS_X']), int(r['POS_Y']))),
                                      radius=int(r['RADIUS'])) for r in raw['SLoc.dbf']}
    templates = {r['ID']: r for r in raw['SStkTemp.dbf']}
    ids = game_map['reference_enemy_ids']
    compiled, omitted, spawned = [], [], []

    def army(gid):
        return -1 if gid == HERO else ids.index(gid)

    def location(gid):
        return dict(location_id=gid, **locations[gid])

    for event in raw['SEvent.dbf']:
        eid = event['ID']
        rows = sorted((r for r in raw['SEvEffec.dbf'] if r['ID'] == eid), key=lambda r: int(r['NUM'] or 0))
        gameplay = [r for r in rows if effects[r['CATEGORY']] not in ('POPUP', 'UNFOG', 'ENABLE_EVENT')]
        if not gameplay:
            omitted.append(eid)
            continue
        if event['ENABLED'] != 'T' or event['OCCUR_ONCE'] != 'T' or event['CHANCE'] != '100':
            raise ValueError('S101 compiler expects enabled, deterministic, one-time events')
        entry = dict(id=eid, conditions=[], effects=[])
        for row in raw['SEvCond.DBF']:
            if row['ID'] != eid:
                continue
            kind = conditions[row['CATEGORY']]
            cond = dict(kind=kind)
            if kind == 'FREQUENCY':
                cond['day'] = int(row['FREQUENCY'])
            elif kind in ('LOCATION', 'LEADER_TO_ZONE', 'ITEM_TO_LOCATION'):
                cond.update(location(row['ID_LOC']))
                if kind == 'LEADER_TO_ZONE':
                    if row['ID_STACK'] != HERO:
                        raise ValueError('Unexpected location-trigger leader')
                    cond['hero'] = True
                if kind == 'ITEM_TO_LOCATION':
                    cond['item'] = catalog[row['TYPE_ITEM']][1]
            elif kind == 'STACK_EXISTS':
                if row['MISC_INT'] != '0':
                    raise ValueError('Unexpected existence condition')
                cond['enemy'] = army(row['ID_STACK'])
            elif kind == 'KILL_STACK' and row['ID_STACK'] == HERO:
                pass
            else:
                raise ValueError('Unsupported gameplay condition: '+kind)
            entry['conditions'].append(cond)
        for row in gameplay:
            kind = effects[row['CATEGORY']]
            effect = dict(kind=kind, source_order=int(row['NUM']))
            if kind == 'CREATE_STACK':
                template = templates[row['ID_STKTEMP']]
                if template['ORDER'] == '4' and template['ORDER_TARG'] != HERO:
                    raise ValueError('Attack stack must target Bethrezen: '+template['ID'])
                roster, levels = template_formation(template, unit_keys, units)
                index = len(ids)
                ids.append(template['ID'])
                spawned.append(index)
                game_map['opponent_positions'].append(locations[row['ID_LOC']]['position'])
                game_map['opponent_numbers'].append(1)
                game_map['enemy_rosters'].append(roster)
                game_map['enemy_initial_levels'].append(levels)
                game_map['enemy_units'].append(sum(k is not None for k in roster))
                game_map['enemy_hp'].append(max(units[k]['max_hp'] for k in roster if k))
                effect.update(enemy=index, template=template['ID'], order=int(template['ORDER']),
                              moves=int(units[unit_keys[template['LEADER']]]['movement']),
                              **location(row['ID_LOC']))
            elif kind == 'MOVE_STACK':
                effect.update(enemy=army(row['ID_STKTEMP']), **location(row['ID_LOC']))
            elif kind in ('MOVE_BESIDE', 'REMOVE_STACK', 'CHANGE_ORDER'):
                effect['enemy'] = army(row['ID_STACK'])
                if kind == 'CHANGE_ORDER':
                    effect['order'] = int(row['ORDER'])
                    if row['ORDER_TARG'] not in ('', NULL):
                        effect.update(location(row['ORDER_TARG']))
            elif kind in ('CAST_SPELL', 'CAST_SPELL_MAP'):
                effect['spell'] = row['TYPE_SPELL'].lower()
                if kind == 'CAST_SPELL_MAP':
                    effect.update(location(row['ID_LOC']))
            elif kind in ('GIVE_ITEM', 'DESTROY_ITEM'):
                effect['item'] = catalog[row['TYPE_ITEM']][1]
                if kind == 'GIVE_ITEM' and row['GIVETO'] != '0':
                    raise ValueError('Only triggerer inventory is supported')
            elif kind == 'WIN':
                effect['win'] = row['WIN_SCEN'] == 'T'
            else:
                raise ValueError('Unsupported gameplay effect: '+kind)
            entry['effects'].append(effect)
        compiled.append(entry)
    stacks = {r['STACK_ID']: r for r in raw['Sstack.dbf']}
    changed = {e['enemy'] for event in compiled for e in event['effects'] if e['kind'] == 'CHANGE_ORDER'}
    # Initial Attack stack orders are effective without a CHANGE_ORDER event.
    # Sstack contains the six intro attackers and KC0072, all targeting Bethrezen.
    attackers = set()
    for index, gid in enumerate(ids):
        stack = stacks.get(gid)
        if stack and stack['ORDER'] == '4':
            if stack['ORDER_TARG'] != HERO:
                raise ValueError('Attack stack must target Bethrezen: '+gid)
            attackers.add(index)
    movers = []
    for index in sorted(changed | attackers):
        stack = stacks[ids[index]]
        movers.append(dict(enemy=index, order=int(stack['ORDER']), moves=int(stack['MOVE']),
                           target=stack['ORDER_TARG']))
    game_map.update(scenario_events=compiled, scenario_events_enabled=True,
                    scenario_event_enemies=spawned, scenario_order_armies=movers,
                    scenario_normal_ai='stationary', scenario_hero='uter_betrezen',
                    event_items={'runestone': 1}, omitted_dialogue_events=omitted,
                    reconstruction_note='Original S101 functional events; see docs/BETHREZENS_RETREAT.md.')
