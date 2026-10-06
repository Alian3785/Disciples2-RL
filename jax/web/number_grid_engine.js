(function (root) {
  'use strict';
  const directions = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]];
  function actionMask(state, map) {
    return directions.map(([dr,dc]) => {
      const r=state.position[0]+dr, c=state.position[1]+dc;
      return r>0 && c>0 && r<map.size-1 && c<map.size-1;
    });
  }
  function initial(map) {
    const visited = map.exploration_bonus ? Array(Math.ceil(map.size * map.size / 32)).fill(0) : [];
    if (visited.length) {
      const cell = map.agent_position[0] * map.size + map.agent_position[1];
      visited[cell >>> 5] = (1 << (cell & 31)) >>> 0;
    }
    return {position: [...map.agent_position], number: map.agent_number,
      alive: map.opponent_numbers.map(() => true), step_count: 0, done: false, won: false, lost: false, visited};
  }
  function step(previous, action, map) {
    const state = {...previous, position: [...previous.position], alive: [...previous.alive], visited: [...previous.visited]};
    if (state.done) return {state, reward: 0};
    if (Number.isInteger(action) && action >= 0 && action < 8) {
      const r = state.position[0] + directions[action][0];
      const c = state.position[1] + directions[action][1];
      const occupied = map.opponent_positions.some((p, i) => state.alive[i] && p[0] === r && p[1] === c);
      if (r > 0 && c > 0 && r < map.size - 1 && c < map.size - 1 && !occupied) state.position = [r, c];
    }
    const adjacent = map.opponent_positions.map((p, i) => state.alive[i] &&
      Math.max(Math.abs(p[0] - state.position[0]), Math.abs(p[1] - state.position[1])) === 1);
    state.lost = adjacent.some((hit, i) => hit && map.opponent_numbers[i] >= state.number);
    let captured = 0;
    if (!state.lost) adjacent.forEach((hit, i) => { if (hit) { state.alive[i] = false; captured++; } });
    state.number += captured;
    state.won = !state.alive.some(Boolean) && !state.lost;
    state.step_count++;
    state.done = state.lost || state.won || state.step_count >= map.max_steps;
    const baseReward = state.lost ? -1 : captured + (state.won ? 3 : 0);
    let bonus = 0;
    if (state.visited.length) {
      const cell = state.position[0] * map.size + state.position[1], word = cell >>> 5, bit = 1 << (cell & 31);
      const moved = state.position.some((v, i) => v !== previous.position[i]);
      if (!(state.visited[word] & bit) && moved && !state.lost) bonus = Math.fround(map.exploration_bonus);
      state.visited[word] = (state.visited[word] | bit) >>> 0;
    }
    return {state, reward: Math.fround(Math.fround(baseReward + bonus) - Math.fround(map.step_cost || 0))};
  }
  const api = {directions, initial, step, actionMask};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NumberGridEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
