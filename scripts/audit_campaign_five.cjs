// Compare the five scenarios directly with the installed original game's DBF tables.
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const game = 'C:/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves/Globals';
const output = process.argv[2];
if (!output) throw Error('Specify the output JSON path');
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const hashes = {};
function table(name) {
  const file = fs.readdirSync(game).find(x => x.toLowerCase() === name.toLowerCase()+'.dbf');
  const b = fs.readFileSync(path.join(game, file));
  hashes[file] = crypto.createHash('sha256').update(b).digest('hex');
  const fields = []; let offset = 1;
  for (let i=32; i<b.readUInt16LE(8)-1 && b[i]!==13; i+=32) {
    const name=b.subarray(i,i+11).toString('ascii').split('\0')[0], width=b[i+16];
    fields.push({name,width,offset}); offset+=width;
  }
  const rows=[];
  for (let i=0;i<b.readUInt32LE(4);i++) {
    const pos=b.readUInt16LE(8)+i*b.readUInt16LE(10);
    if (b[pos]===42) continue;
    rows.push(Object.fromEntries(fields.map(f=>[f.name,new TextDecoder('windows-1251').decode(b.subarray(pos+f.offset,pos+f.offset+f.width)).trim()])));
  }
  return rows;
}
const original=table('Gunits'), attacks=table('Gattacks'), units=read('stoix/envs/data/units.json');
const maps = Object.fromEntries(['mirror_match','siege_train','default','default-green-dragon','current'].map(name=>
  [name,read(name==='current'?'number_grid_map.json':'maps/'+(name==='default-green-dragon'?'default':name)+'.json')]));
const keys = new Set(['possessed','duke','cultist']);
for (const m of Object.values(maps)) for (const roster of [m.hero_roster||[], ...m.enemy_rosters]) for (const k of roster) if(k) keys.add(k);
const mismatches=[], checked=[], specialProfiles=[];
for (const k of keys) {
  const u=units[k], o=original.find(x=>x.UNIT_ID.toLowerCase()===u.game_id);
  if (!o) throw Error('No original unit '+k);
  const a=attacks.find(x=>x.ATT_ID===o.ATTACK_ID);
  const expected={level:+o.LEVEL,max_hp:+o.HIT_POINT,armor:+o.ARMOR,regeneration:+o.REGEN,exp_kill:+o.XP_KILLED,exp_required:+o.XP_NEXT,movement:+o.MOVE,scouting:+o.SCOUT,leadership:+o.LEADERSHIP,lifetime:+o.LIFE_TIME,size:o.SIZE_SMALL==='T'?1:2,double_attack:o.ATCK_TWICE==='T'};
  if(a) {
    Object.assign(expected,{initiative:+a.INITIATIVE});
    if (+a.QTY_DAM || +a.QTY_HEAL) Object.assign(expected,{damage:(+a.QTY_DAM || +a.QTY_HEAL),accuracy:+a.POWER});
    else {
      specialProfiles.push({key:k,attack_class:a.CLASS,original_power:+a.POWER,catalogue_damage:u.damage,catalogue_accuracy:u.accuracy,
        note:'Special-effect profile: placeholder damage/accuracy is not compared to a direct damage attack. Effect-specific CUDA contracts apply.'});
      const stored=u.game_data?.primary_attack;
      for(const field of ['CLASS','POWER','REACH','QTY_DAM','QTY_HEAL','INITIATIVE','WARD1','WARD2','WARD3','WARD4']) {
        if (!(field in a)) continue;
        checked.push({key:k,field:'game_data.primary_attack.'+field});
        if(String(stored?.[field]).toLowerCase()!==a[field].toLowerCase()) mismatches.push({key:k,field:'game_data.primary_attack.'+field,jax:stored?.[field],original:a[field]});
      }
    }
  }
  for(const [field,value] of Object.entries(expected)) {
    checked.push({key:k,field});
    if(u[field]!==value) mismatches.push({key:k,game_id:u.game_id,field,jax:u[field],original:value});
  }
}
const report={passed:!mismatches.length,authority:'Installed Disciples II: Rise of the Elves original DBF tables; no Python reference',original_hashes:hashes,
  units_checked:keys.size,fields_checked:checked.length,mismatches,special_attack_profiles:specialProfiles,
  scope:'Base numeric unit/primary attack fields for starting and enemy rosters. Scenario goals are user-defined; this is not a full original-engine equivalence test.',
  maps:Object.fromEntries(Object.entries(maps).map(([n,m])=>[n,{size:m.size,armies:m.enemy_rosters.length,behavior_types:new Set(m.enemy_rosters.flat().filter(Boolean).map(k=>units[k].unit_type)).size}]))};
fs.mkdirSync(path.dirname(output),{recursive:true}); fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2)); if(!report.passed) process.exitCode=1;
