"""Check exported replay outcomes against the actual map objective, without using GPU."""
import argparse
import json
import math
from pathlib import Path

MAPS = ('scroll_train', 'magic_train', 'item_train', 'hire_train', 'orc_duel', 'formation_train')
ALL_MAPS = MAPS + ('mirror_match', 'siege_train', 'default', 'default-green-dragon', 'current')


def validate(directory, names=MAPS):
    reports = []
    for name in names:
        raw = json.loads((directory/(name+'.json')).read_text())
        assert raw['format'] == 'shared-replay-values-v1'
        metadata, pool, record = raw['metadata'], raw['pool'], raw['records'][0]
        result, game_map = metadata['result'], metadata['map']
        states = [dict(zip(record['state_keys'], (pool[i] for i in row))) for row in record['states']]
        assert result['id'] == name and result['training_steps'] == 50_000_000
        assert 0 < result['checkpoint_steps'] <= result['training_steps']
        assert len(states) == result['episode']['steps']+1
        assert not states[0]['done'] and states[-1]['done']
        assert states[0]['position'] == game_map['agent_position']
        assert states[-1]['step_count'] == result['episode']['steps']
        assert states[-1]['won'] == (result['episode']['outcome'] == 'victory')
        if states[-1]['won']:
            target = game_map.get('objective_enemy')
            if game_map.get('campaign_objective') == 'scripted_bot':
                assert states[-1]['bot_hero_victories'] > 0 and not states[-1]['lost']
            elif target is not None:
                assert states[-1]['scenario_spawned'][target] and not states[-1]['alive'][target]
            elif any(c.get('objective', False) for c in game_map.get('cities', [])):
                assert all(states[-1]['city_owned'][i] for i, c in enumerate(game_map['cities']) if c.get('objective'))
            else:
                assert not any(states[-1]['alive']) and all(states[-1]['scenario_spawned'])
        defeated = sum(not alive and spawned for alive, spawned in
                       zip(states[-1]['alive'], states[-1]['scenario_spawned']))
        if game_map.get('scripted_bot'):
            defeated -= states[-1]['bot_defeated_enemies']
        assert defeated == result['episode']['defeated']
        def finite(value):
            if isinstance(value, float):
                assert math.isfinite(value)
            elif isinstance(value, list):
                for item in value:
                    finite(item)
            elif isinstance(value, dict):
                for item in value.values():
                    finite(item)
        finite(pool)
        reports.append(dict(map=name, frames=len(states), outcome=result['episode']['outcome'],
                            defeated=defeated, checkpoint_steps=result['checkpoint_steps']))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--map', action='append', choices=ALL_MAPS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = validate(args.directory, args.map or MAPS)
    args.output.write_text(json.dumps(dict(passed=True, maps=reports), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(reports, ensure_ascii=False))
