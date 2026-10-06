"""Independent scalar rules and breadth-first search, for tests (not PPO training)."""
from collections import deque
import json
from pathlib import Path

MAP = json.loads((Path(__file__).parent / 'number_grid_map.json').read_text())
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def initial_state(config=MAP):
    visited = [0] * ((config['size']**2 + 31) // 32) if config.get('exploration_bonus', 0) else []
    if visited:
        cell = config['agent_position'][0] * config['size'] + config['agent_position'][1]
        visited[cell // 32] |= 1 << (cell % 32)
    return dict(position=config['agent_position'][:], number=config['agent_number'],
                alive=[True]*len(config['opponent_numbers']),
                step_count=0, done=False, won=False, lost=False, visited=visited)


def step(state, action, config=MAP):
    if state['done']:
        return {**state}, 0.0
    r, c = state['position']
    if 0 <= action < 8:
        dr, dc = DIRECTIONS[action]
        rr, cc = r + dr, c + dc
        occupied = any(a and p == [rr, cc] for a, p in zip(state['alive'], config['opponent_positions']))
        if 0 < rr < config['size'] - 1 and 0 < cc < config['size'] - 1 and not occupied:
            r, c = rr, cc
    adjacent = [i for i, (p, a) in enumerate(zip(config['opponent_positions'], state['alive']))
                if a and max(abs(p[0]-r), abs(p[1]-c)) == 1]
    lost = any(config['opponent_numbers'][i] >= state['number'] for i in adjacent)
    alive = state['alive'][:]
    captures = 0
    if not lost:
        for i in adjacent:
            alive[i] = False
            captures += 1
    won = not any(alive) and not lost
    count = state['step_count'] + 1
    visited = state['visited'][:]
    bonus = 0.0
    if visited:
        cell = r * config['size'] + c
        word, bit = cell // 32, 1 << (cell % 32)
        if not (visited[word] & bit) and [r, c] != state['position'] and not lost:
            bonus = config['exploration_bonus']
        visited[word] |= bit
    return dict(position=[r, c], number=state['number'] + captures, alive=alive,
                step_count=count, done=lost or won or count >= config['max_steps'],
                won=won, lost=lost, visited=visited), float(-1 if lost else captures + 3 * won) + bonus - config.get('step_cost', 0.0)


def shortest_path(config=MAP):
    # This exact reference solver is only intended for the small original map.
    # Large maps use ordinary rollouts; no expensive combinatorial search in export.
    if len(config['opponent_numbers']) > 4:
        raise ValueError('Exact shortest_path is limited to at most four opponents')
    start = initial_state(config)
    def key(s):
        return (*s['position'], s['number'], *s['alive'])
    seen = {key(start)}
    queue = deque([(start, [])])
    while queue:
        state, path = queue.popleft()
        for action in range(8):
            nxt, _ = step(state, action, config)
            if nxt['won']:
                return path + [action]
            k = key(nxt)
            if not nxt['done'] and k not in seen:
                seen.add(k)
                queue.append((nxt, path + [action]))
    raise RuntimeError('Fixed map has no winning path')


def winning_path(config=MAP):
    """Construct a legal route by finding the next safe capture; not an optimum.

    A test-only reachability check. It is never used by the learner or its policy.
    """
    current = initial_state(config)
    actions = []
    while not current['won']:
        queue = deque([(current, [])])
        seen = {tuple(current['position'])}
        found = None
        while queue and found is None:
            state, path = queue.popleft()
            for action in range(8):
                nxt, _ = step(state, action, config)
                if not nxt['lost'] and nxt['number'] > current['number']:
                    found = (nxt, path + [action])
                    break
                key = tuple(nxt['position'])
                if not nxt['done'] and key not in seen:
                    seen.add(key)
                    queue.append((nxt, path + [action]))
        if found is None:
            raise RuntimeError('No safe route to another capture')
        current, path = found
        actions.extend(path)
        if current['done'] and not current['won']:
            raise RuntimeError('Winning route exceeds episode limit')
    return actions


if __name__ == '__main__':
    exact = len(MAP['opponent_numbers']) <= 4
    actions = shortest_path() if exact else winning_path()
    print(json.dumps({'steps': len(actions), 'proven_shortest': exact, 'actions': actions}))
