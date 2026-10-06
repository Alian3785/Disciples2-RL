import json,collections
from pathlib import Path
s=json.loads(Path('tmp/d2_audit_snapshot.json').read_text(encoding='utf8'))
t=s['tables'];att={a['ATT_ID'].lower():a for a in t['gattacks.dbf']}
def num(x):return int(x or 0)
def original(r):
 a=att.get(r['ATTACK_ID'].lower(),{})
 return {'health':num(r['HIT_POINT']),'armor':num(r['ARMOR']),'accuracy':num(a.get('POWER')),'damage':num(a.get('QTY_DAM')) or num(a.get('QTY_HEAL')),'initiative_base':num(a.get('INITIATIVE')),'exp_kill':num(r['XP_KILLED']),'exp_required':num(r['XP_NEXT']),'Level':num(r['LEVEL']),'big':r['SIZE_SMALL']=='F'}
matches=[]
for unit in s['units']:
 b=unit['battle'];rank=[]
 for r in t['gunits.dbf']:
  o=original(r);diff={k:[v,b[k]] for k,v in o.items() if v!=b[k]}
  rank.append((len(diff),r,diff))
 rank.sort(key=lambda x:x[0]);best=rank[0][0];c=[x for x in rank if x[0]==best]
 matches.append({'unit':b['name'],'diff_count':best,'candidates':[{'id':r['UNIT_ID'],'name':r['name'],'diff':d} for _,r,d in c]})
Path('tmp/d2_unit_matches.json').write_text(json.dumps(matches,ensure_ascii=False,indent=2),encoding='utf8')
print('COUNTS',collections.Counter(r['diff_count'] for r in matches))
print('MISMATCHES',json.dumps([r for r in matches if r['diff_count']>0],ensure_ascii=False))
