import sys,json,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.verify_data_vs_dbf import load_dbf_units,compare_unit,DEFAULT_GLOBALS
from tools.inspect_sg_map import parse_dbf_rows
from data_dicts_compact_lines import DATA
db=load_dbf_units(DEFAULT_GLOBALS)
profiles={c['unit_id']:c for group in db.values() for c in group}
raw={r['UNIT_ID']:r for r in parse_dbf_rows(DEFAULT_GLOBALS/'Gunits.dbf')}
resolved={}
for u in DATA:
    scores=[(len(compare_unit(u,c)),k) for k,c in profiles.items()]
    score=min(x[0] for x in scores)
    ids=[k for n,k in scores if n==score]
    if u.get('unit_id'):ids=[u['unit_id']];score=0
    resolved[u['кто']]={'score':score,'candidates':[{'id':k,'name':profiles[k]['name'],'revive':raw[k]['REVIVE_C'],'race':raw[k]['RACE_ID'],'category':raw[k]['UNIT_CAT']} for k in ids]}
(ROOT/'tmp/revive_candidates.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2),encoding='utf8')
for n,r in resolved.items():
    if r['score'] or len(r['candidates'])!=1: print(n,json.dumps(r,ensure_ascii=False))
print('TOTAL',len(resolved),'UNIQUE EXACT',sum(r['score']==0 and len(r['candidates'])==1 for r in resolved.values()))
