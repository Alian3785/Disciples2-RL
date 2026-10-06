import json, re
from collections import defaultdict, Counter

with open('tmp/d2_audit_snapshot.json', encoding='utf-8') as f:
    s = json.load(f)

def parse_cost(x):
    out = {k: 0 for k in 'ryewb'}
    for k, v in re.findall(r'([gryewb])(\d+)', x or ''):
        if k in out:
            out[k] = int(v)
    return out

def fmt(c):
    return '/'.join(str(c[k]) for k in 'ryewb')

lords = s['tables'].get('glord.dbf', [])
cats = defaultdict(list)
for row in lords:
    cats[str(row.get('LORD_ID'))].append(row.get('CATEGORY'))
print('LORD CATEGORIES', {k: sorted(set(v), key=lambda x: str(x)) for k, v in cats.items()})
print('CATEGORY COUNTS', Counter(str(row.get('CATEGORY')) for row in lords))

gspellr = s['tables'].get('gspellr.dbf', [])
by_spell = defaultdict(list)
for row in gspellr:
    by_spell[str(row.get('SPELL_ID'))].append(row)

level5 = []
for b in s['books']:
    o = b['original']
    if int(o.get('LEVEL') or 0) == 5:
        level5.append(b)

print('LEVEL5 COUNT', len(level5))
for b in level5:
    o, e = b['original'], b['env']
    sid = str(o.get('SPELL_ID'))
    rows = by_spell[sid]
    by_cat = defaultdict(list)
    for row in rows:
        for lord_id, cat_values in cats.items():
            if str(row.get('LORD_ID')) == lord_id:
                by_cat[str(cat_values[0])].append(row)
                break
    uniq = {}
    for cat, rs in by_cat.items():
        uniq[cat] = sorted(set(str(r.get('RESEARCH')) for r in rs))
    learn = {k: e.get('learn_' + k) for k in ['rune', 'life', 'death', 'hell', 'nature']}
    # GspellR letters map r->hell, y->life, e->death, w->rune, b->nature.
    game = parse_cost(rows[0].get('RESEARCH'))
    game_env_order = {'rune': game['w'], 'life': game['y'], 'death': game['e'], 'hell': game['r'], 'nature': game['b']}
    ratio = {k: (learn[k] / game_env_order[k] if game_env_order[k] else None) for k in learn}
    print('\\n', sid, o.get('NAME_TXT'), 'env=', e.get('id'), 'faction=', b.get('faction'))
    print('  GspellR by category:', uniq)
    print('  GspellR mapped:', game_env_order, 'env/game:', ratio)
    print('  env learn raw:', learn, 'effective mage x0.5:', {k: (learn[k] or 0) * 0.5 for k in learn})
