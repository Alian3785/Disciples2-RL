"""Validate real exported episodes and their zone-of-control explanations on CPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
def distance(a, b):
    return max(abs(x-y) for x, y in zip(a, b))
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def validate(path, training):
    data = read(path)
    assert data['format'] == 'shared-replay-values-v1'
    meta, pool, record = data['metadata'], data['pool'], data['records'][0]
    result, game, zones = meta['result'], meta['map'], meta['zones']
    states = [dict(zip(record['state_keys'], (pool[i] for i in refs))) for refs in record['states']]
    fields = [dict(zip(record['frame_keys'], (pool[i] if i >= 0 else None for i in refs))) for refs in record['fields']]
    assert result['checkpoint_sha256'] == training['checkpoint_sha256']
    assert result['training_steps'] == training['training_steps'] == result['checkpoint_steps']
    assert len(states) == len(fields) == result['episode']['steps'] + 1
    scripted_start = any(
        states[0]['script_events_fired'][i] and effect['kind'] in ('MOVE_STACK', 'MOVE_BESIDE')
        and effect.get('enemy') == -1
        and distance(states[0]['position'], effect.get('position', game['agent_position'])) <= effect.get('radius', 0)
        for i, event in enumerate(game.get('scenario_events', [])) for effect in event['effects'])
    assert (states[0]['position'] == game['agent_position'] or scripted_start) and not states[0]['done'], result['id']
    assert states[-1]['done'] and states[-1]['step_count'] == result['episode']['steps']
    assert states[-1]['won'] == (result['episode']['outcome'] == 'victory')
    assert zones['source_sha256'] == result['zone_source_sha256']
    assert [tuple(p) for p in zones['neighbours']] == NEIGHBOURS
    inside, forts, kin = set(map(tuple, zones['inside'])), set(map(tuple, zones['forts'])), set(zones['kin_armies'])
    transitions, held, direct, previews, chest_steps = 0, 0, 0, 0, 0
    def first_neighbour(s, center, holders):
        for dy, dx in NEIGHBOURS:
            pos = [center[0]+dy, center[1]+dx]
            found = [i for i in holders if s['enemy_positions'][i] == pos]
            if found:
                return min(found)
        return -1
    for index, s in enumerate(states):
        holders = {i for i, alive in enumerate(s['alive'])
                   if alive and i not in kin and tuple(s['enemy_positions'][i]) not in inside}
        assert [i for i, flag in enumerate(s['zone_holders']) if flag] == sorted(holders)
        assert len(s['map_commands']) == len(s['move_legal']) == len(s['move_enters']) == 8
        if not s['in_battle'] and not s['done']:
            for action, offset in enumerate(zones['directions']):
                if not s['move_legal'][action]:
                    continue
                previews += 1
                destination = [x+y for x, y in zip(s['position'], offset)]
                target, cost = s['map_commands'][action]
                assert 0 < cost <= s['movement_points']
                if s['move_direct'][action]:
                    assert target >= 0 and s['alive'][target] and not s['move_enters'][action]
                    continue
                common = (tuple(s['position']) not in forts and tuple(destination) not in forts
                    and any(distance(s['enemy_positions'][i], s['position']) == 1
                            and distance(s['enemy_positions'][i], destination) == 1 for i in holders))
                expected = (-1 if tuple(destination) in forts else
                            first_neighbour(s, s['position'] if common else destination, holders))
                assert target == expected and s['move_enters'][action] == (not common)
        if not index:
            continue
        before, action, entry = states[index-1], fields[index]['action'], fields[index].get('engagement')
        assert s['day'] >= before['day']
        if not before['in_battle'] and 0 <= action < 8:
            assert before['move_legal'][action]
            target, cost = before['map_commands'][action]
            assert before['movement_points'] - s['movement_points'] == cost
            destination = [x+y for x, y in zip(before['position'], zones['directions'][action])]
            assert s['position'] == (destination if before['move_enters'][action] else before['position'])
            if target >= 0:
                assert entry and entry['enemy'] == target and entry['cost'] == cost
                assert before['chest_alive'] == s['chest_alive']
                if before['move_direct'][action]:
                    direct += 1
                    assert entry['kind'] == 'direct'
                elif before['move_enters'][action]:
                    transitions += 1
                    assert entry['kind'] == 'zone_enter' and distance(s['position'], before['enemy_positions'][target]) == 1
                else:
                    held += 1
                    assert entry['kind'] == 'zone_held' and distance(before['position'], before['enemy_positions'][target]) == 1
            else:
                assert entry is None
        removed = [i for i, alive in enumerate(before['chest_alive']) if alive and not s['chest_alive'][i]]
        if removed:
            chest_steps += 1
            assert not before['in_battle'] and 0 <= action < 8 and before['map_commands'][action][0] < 0
            assert all(distance(s['position'], game['chests'][i]['position']) <= 1 for i in removed)
    assert transitions == result['engagement_counts']['zone_enter']
    assert held == result['engagement_counts']['zone_held'] and direct == result['engagement_counts']['direct']
    if states[-1]['won']:
        target, position = game.get('objective_enemy'), game.get('objective_position')
        if target is not None:
            assert not states[-1]['alive'][target] and states[-1]['scenario_spawned'][target]
        elif position is not None:
            assert states[-1]['position'] == position
            assert not states[-1]['in_battle'] and not states[-1]['summon_battle'] and not states[-1]['lost']
            if game.get('objective_after_event') is not None:
                event = next(i for i, e in enumerate(game['scenario_events']) if e['id'] == game['objective_after_event'])
                assert states[-1]['script_events_fired'][event]
        elif game.get('campaign_objective') == 'waves':
            required = {e['enemy'] for e in game['scheduled_enemies']}
            assert required and all(not states[-1]['alive'][i] and states[-1]['scenario_spawned'][i]
                                    for i in required), result['id'] + ': unfinished wave'
        elif any(c.get('objective') for c in game.get('cities', [])):
            assert all(states[-1]['city_owned'][i] for i, c in enumerate(game['cities']) if c.get('objective'))
        else:
            optional = set(game.get('optional_enemies', [])) | set(game.get('absent_enemies', []))
            assert all(not alive and spawned for i, (alive, spawned) in
                       enumerate(zip(states[-1]['alive'], states[-1]['scenario_spawned'])) if i not in optional), result['id'] + ': living required army'
    def finite(v):
        if isinstance(v, float):
            assert math.isfinite(v)
        elif isinstance(v, list):
            for x in v:
                finite(x)
        elif isinstance(v, dict):
            for x in v.values():
                finite(x)
    finite(pool)
    return dict(map=result['id'], frames=len(states), outcome=result['episode']['outcome'],
        previews_checked=previews, zone_entries=transitions, held_entries=held,
        direct_attacks=direct, chest_pickup_steps=chest_steps, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--map', action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training = read(args.batch.parent / 'baseline-six-maps-20261103.json')['maps']
    reports = [validate(args.directory / (r['map'] + '.json'), r) for r in training
               if args.map is None or r['map'] in args.map]
    args.output.write_text(json.dumps(dict(passed=True, maps=reports), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(reports, ensure_ascii=False))
