const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const nodes = new Map();
const node = key => { if (!nodes.has(key)) nodes.set(key, {innerHTML:'',textContent:'',addEventListener(){},showModal(){},close(){}}); return nodes.get(key); };
const ctx = vm.createContext({window:{},document:{querySelector:node,addEventListener(){}},location:{hash:''},history:{replaceState(){}}});
const base = 'outputs/disciples-audit-site/dist/';
vm.runInContext(fs.readFileSync(base+'data.js','utf8'),ctx);
vm.runInContext(fs.readFileSync(base+'app.js','utf8'),ctx);
assert(node('#main').innerHTML.includes('Принятая модель обучения'));
assert(node('#main').innerHTML.includes('number">2</div>'));
for (const [view,filter,count] of [['mechanics','intentional',2],['mechanics','fixed',16],['mechanics','gaps',11],['spells','intentional',16]]) {
  vm.runInContext(`state={view:'${view}',filter:'${filter}',group:'all',query:''};render()`,ctx);
  assert.equal(node('#result-count').textContent,`${count} записей`);
  assert(!node('#results').innerHTML.includes('undefined'));
  if(filter==='gaps') for(const id of ['target','summon','research_cost','unit_stats','scroll_alias','might','cast_cost']) assert(!node('#results').innerHTML.includes(`data-record="findings:${id}"`));
}
vm.runInContext("showDetail('findings:target')",ctx);
assert(node('#detail-body').innerHTML.includes('Почему принято такое решение'));
assert(node('#detail-body').innerHTML.includes('Решение для текущей среды'));
assert(!node('#detail-body').innerHTML.includes('Как исправить'));
vm.runInContext("showDetail('findings:research_cost')",ctx);
assert(node('#detail-body').innerHTML.includes('Результат исправления'));
assert.equal(vm.runInContext("A.spells.filter(r=>r.rationale&&r.status==='partial').length",ctx),0);
console.log('PASS: overview counts, mechanics filters, spell classification and detail cards');
