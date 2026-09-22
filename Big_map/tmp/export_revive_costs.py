"""Create a portable, reviewed economy snapshot; no game installation is needed at runtime."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.inspect_sg_map import parse_dbf_rows
from data_dicts_compact_lines import DATA
GAME=Path(r'C:\Program Files (x86)\Steam\steamapps\common\Disciples II Rise of the Elves\Globals')
raw={r['UNIT_ID'].lower():r for r in parse_dbf_rows(GAME/'Gunits.dbf')}
dyn={r['UPGRADE_ID'].lower():r for r in parse_dbf_rows(GAME/'GDynUpgr.dbf')}
candidates=json.loads((ROOT/'tmp/revive_candidates.json').read_text(encoding='utf8'))
race={1:'g000rr0000',2:'g000rr0002',3:'g000rr0001',4:'g000rr0003',5:'g000rr0005',0:'g000rr0004'}
def gold(s):
    parts=s.lower().split(':');assert all(int(x[1:])==0 for x in parts[1:]);return int(parts[0][1:])
costs={k:gold(r['REVIVE_C']) for k,r in raw.items()}
by_name={}
for u in DATA:
    name=u['кто'];options=candidates[name]['candidates']
    filtered=[c for c in options if c['race']==race[int(u['столица'])]]
    if filtered:options=filtered
    if name in {'Тролль','Кракен'}:
        options=[c for c in options if c['name']=={'Тролль':'Troll','Кракен':'Kraken'}[name]]
    assert len({c['name'] for c in options})==1,(name,options)
    ids=tuple(sorted(c['id'].lower() for c in options))
    assert len({costs[k] for k in ids})==1,(name,ids)
    # Dynamic levels do not change REVIVE_C in these original profiles.
    for key in ids:
        for field in ('DYN_UPG1','DYN_UPG2'):
            d=dyn.get(raw[key][field].lower())
            assert d is None or gold(d['REVIVE_C'])==0,(key,field)
    by_name[name]=ids
lines=['"""Original Rise of the Elves resurrection prices, keyed by stable game IDs.',
       '', 'Russian name bindings cover the 252 environment templates. Multiple IDs are',
       'same-name game variants with identical revival prices, not inferred identities.',
       'Runtime never guesses identity from level, health, or combat statistics.',
       'Source: installed Steam Globals/Gunits.dbf; exported 2026-09-16.',
       'SHA256: '+hashlib.sha256((GAME/'Gunits.dbf').read_bytes()).hexdigest(), '"""','', 'REVIVE_GOLD_BY_UNIT_ID = {']
lines += [f'    {k!r}: {costs[k]},' for k in sorted(costs)]
lines += ['}', '', 'REVIVE_PROFILE_IDS_BY_NAME = {']
lines += [f'    {n!r}: {ids!r},' for n,ids in by_name.items()]
lines += ['}', '', '', 'def known_unit_revive_gold_cost(unit):',
          '    """Return a verified price, or None for a genuinely unknown profile."""',
          '    unit_id = str(unit.get("unit_id") or unit.get("game_unit_id") or "").strip().lower()',
          '    if unit_id:',
          '        return REVIVE_GOLD_BY_UNIT_ID.get(unit_id)',
          '    name = str(unit.get("name") or unit.get("кто") or "").strip()',
          '    ids = REVIVE_PROFILE_IDS_BY_NAME.get(name, ())',
          '    return REVIVE_GOLD_BY_UNIT_ID[ids[0]] if ids else None', '']
(ROOT/'unit_revive_costs.py').write_text('\n'.join(lines),encoding='utf8')
print('Exported',len(costs),'game profiles and',len(by_name),'reviewed environment name bindings')
