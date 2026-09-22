import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inspect_sg_map import parse_dbf_rows
GAME = Path(r'C:\Program Files (x86)\Steam\steamapps\common\Disciples II Rise of the Elves\Globals')
snapshot = json.loads((ROOT/'tmp/d2_audit_snapshot.json').read_text(encoding='utf8'))
ids = {'g000ss0085','g000ss0087','g000ss0055','g000ss0056','g000ss0091'}
rows = {r['SPELL_ID'].lower():r for r in parse_dbf_rows(GAME/'Gspells.dbf')}
for b in snapshot['books']:
    key=b['original']['SPELL_ID'].lower()
    if key in ids:
        print(key, b['env']['id'], b['scroll']['spell_name'], rows[key]['CASTING_C'], {k:v for k,v in b['env'].items() if k.startswith('use_')})
units = {r['UNIT_ID'].lower():r for r in parse_dbf_rows(GAME/'Gunits.dbf')}
for key in ['g000uu8033','g000uu8035','g000uu3001']:
    u=units[key]
    print(key, {k:v for k,v in u.items() if k in ['HIT_POINT','XP_NEXT','ATTACK_ID']})
    if key=='g000uu3001':
        print([r for r in parse_dbf_rows(GAME/'Gattacks.dbf') if r['ATT_ID'].lower()==u['ATTACK_ID'].lower()])
print([r for r in parse_dbf_rows(GAME/'GmodifL.dbf') if r['BELONGS_TO'].lower()=='g000um2020'])
print([r for r in parse_dbf_rows(GAME/'GItem.dbf') if r['ITEM_ID'].lower()=='g000ig0020'])
from campaign_env_data import SPELLS_D2
import re
all_spells = {key: value for faction in SPELLS_D2.values() for key,value in faction.items()}
colors = {'hell':'r','life':'y','death':'e','rune':'w','nature':'b'}
checked = 0
for b in snapshot['books']:
    original = rows[b['original']['SPELL_ID'].lower()]
    actual = all_spells[b['env']['id']]
    encoded = dict((k,int(v)) for k,v in re.findall(r'([a-z])(\d+)',original['CASTING_C']))
    for suffix, color in colors.items():
        assert actual['use_'+suffix] == encoded.get(color,0), (actual['id'],suffix)
    checked += 1
print('PASS: all', checked, 'book casting costs match installed Gspells.dbf')
