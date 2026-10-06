'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const engine = require('./web/number_grid_engine.js');
const html = fs.readFileSync(path.join(__dirname, 'viewer.html'), 'utf8');
const match = html.match(/<script id="run-data" type="application\/json">([\s\S]*?)<\/script>/);
assert.ok(match, 'Viewer must contain embedded run data');
assert.ok(!/<(?:script|link)[^>]+(?:src|href)=["']https?:/i.test(html), 'Viewer must work offline');
const data = JSON.parse(match[1]);
if (data.map.battle_mode) {
  let transitions = 0, combatTransitions = 0;
  for (const record of data.records) {
    assert.equal(record.frames[0].state.step_count, 0);
    let totalReward = 0;
    for (let i = 1; i < record.frames.length; i++) {
      const before = record.frames[i-1], after = record.frames[i];
      assert.ok(before.action_mask[after.action], 'Recorded action must be legal');
      assert.equal(after.state.step_count, before.state.step_count+1);
      assert.equal(after.state.hp.length, 12);
      assert.ok(after.state.hp.every(h=>h>=0));
      assert.equal(after.action_mask.length, 18);
      totalReward += after.reward;
      assert.equal(after.total_reward, totalReward);
      transitions++; combatTransitions += Number(before.state.in_battle);
    }
    assert.equal(record.frames.at(-1).state.done, true);
  }
  assert.ok(html.includes('/api/step'), 'Human mode must call the JAX service');
  assert.ok(!html.includes('NumberGridEngine.step'), 'Browser must not simulate the archer battle');
  const report = {passed:true, mode:'JAX archer battle replays', records:data.records.length,
    transitions, combatTransitions, validation:'Replay structure and masks; rules tested in JAX pytest'};
  console.log(JSON.stringify(report,null,2));
  if (process.argv[2]) fs.writeFileSync(process.argv[2],JSON.stringify(report,null,2));
  return;
}
let transitions = 0;
for (const record of data.records) {
  assert.deepEqual(record.frames[0].state, engine.initial(data.map));
  let totalReward = 0;
  for (let i = 1; i < record.frames.length; i++) {
    const previous = record.frames[i - 1], next = record.frames[i];
    if (data.map.mask_walls) assert.ok(engine.actionMask(previous.state, data.map)[next.action],
      `${record.label}, transition ${i}: policy selected a wall`);
    const result = engine.step(previous.state, next.action, data.map);
    assert.deepEqual(result.state, next.state, `${record.label}, transition ${i}`);
    assert.equal(result.reward, next.reward);
    totalReward += result.reward;
    assert.equal(totalReward, next.total_reward);
    transitions++;
  }
  assert.equal(record.frames.at(-1).state.done, true);
}
for (const c of data.validation_cases) {
  const result = engine.step(c.before, c.action, c.map);
  assert.deepEqual(result.state, c.after);
  assert.equal(result.reward, c.reward);
}
for (const c of data.wall_mask_cases || []) {
  assert.deepEqual(engine.actionMask({position:c.position}, data.map), c.mask);
}
if (data.shortest_path_actions) {
  let state = engine.initial(data.map), reward = 0;
  for (const action of data.shortest_path_actions) {
    const result = engine.step(state, action, data.map);
    state = result.state; reward += result.reward;
  }
  assert.equal(state.won, true);
  const newCells = state.visited.length ? state.visited.reduce((sum, word) => sum +
    Array.from({length: 32}, (_, bit) => (word >>> bit) & 1).reduce((a,b) => a+b, 0), 0) - 1 : 0;
  assert.ok(Math.abs(reward - (data.map.opponent_numbers.length + 3 -
    data.shortest_path_steps * (data.map.step_cost || 0) + newCells * (data.map.exploration_bonus || 0))) < 1e-4);
  assert.equal(state.step_count, data.shortest_path_steps);
}
const report = {passed: true, records: data.records.length, transitions,
  edge_and_random_cases: data.validation_cases.length, shortest_path_steps: data.shortest_path_steps,
  wall_mask_cases: (data.wall_mask_cases || []).length,
  map_size: data.map.size, enemy_count: data.map.opponent_numbers.length};
console.log(JSON.stringify(report, null, 2));
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify(report, null, 2));
