from pathlib import Path
import sys,json,hashlib,collections,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.inspect_sg_map import parse_dbf_rows
from campaign_env_data import CampaignConstantsMixin, SPELLS_D2
from campaign_env_inventory import CampaignInventoryMixin
from data_dicts_compact_lines import DATA,map_unit_to_battle
from scroll_spell_data import SCROLL_ITEM_DEFINITIONS
class C(CampaignInventoryMixin,CampaignConstantsMixin): pass
GAME=Path(r'C:\Program Files (x86)\Steam\steamapps\common\Disciples II Rise of the Elves')
tables={p.name.lower():parse_dbf_rows(p) for p in (GAME/'Globals').glob('*') if p.suffix.lower()=='.dbf'}
t={r['TXT_ID'].lower():r['TEXT'] for r in tables['tglobal.dbf']}
def txt(k):return t.get(k.lower(),k)
for key in ['gspells.dbf','gitem.dbf','gunits.dbf']:
 for row in tables[key]:
  row['name']=txt(row['NAME_TXT']);row['description']=txt(row['DESC_TXT'])
sc={s['spell_id']:s for s in SCROLL_ITEM_DEFINITIONS}
fact=['empire','mountain_clans','legions','undead_hordes','elves']
book=[]
for i,f in enumerate(fact):
 orig=[r for r in tables['gspells.dbf'] if (i<4 and (i*20<int(r['SPELL_ID'][-4:])<=i*20+20 or 81+i*4<=int(r['SPELL_ID'][-4:])<=84+i*4)) or (i==4 and int(r['SPELL_ID'][-4:])>=97)]
 orig.sort(key=lambda r:(int(r['LEVEL']),r['SPELL_ID']))
 entries=list(SPELLS_D2[f].values())
 for a,b in zip(orig,entries):
  book.append({'original':a,'env':b,'scroll':sc.get(a['SPELL_ID']),'faction':f})
inv=[]
effects=C._battle_item_effect_definitions()
for r in tables['gitem.dbf']:
 n=r['name'];cat=int(r['ITEM_CAT']);effect={};matched=n
 if cat==8:effect=sc.get(r['SPELL_ID'],{})
 elif cat in [4,5,6,7]:effect=C._potion_item_definition(n)
 elif cat in [0,2]:effect=C._artifact_effect_definition(n)
 elif cat==1:effect=C._book_effect_definition(n)
 elif cat==3:effect=C._banner_effect_definition(n)
 elif cat==13:effect=C._boot_effect_definition(n)
 elif cat in [11,12]:effect=effects.get(n,{})
 elif cat==9:effect=next((x for x in C.STAFF_SPELL_ITEM_DEFINITIONS if x['item_name']==n),{})
 inv.append({'original':r,'effect':effect,'matched_name':matched})
units=[{'raw':u,'battle':map_unit_to_battle(u,'blue',1)} for u in DATA]
out={'tables':tables,'books':book,'items':inv,'units':units,'battle_item_effects':effects,'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (GAME/'Globals').glob('*') if p.suffix.lower()=='.dbf'}}
(ROOT/'tmp/d2_audit_snapshot.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('ITEM NO EFFECT',json.dumps([(r['original']['ITEM_ID'],r['original']['ITEM_CAT'],r['original']['name']) for r in inv if not r['effect'] and int(r['original']['ITEM_CAT']) not in [8,10,14]],ensure_ascii=False))
print('SPELL MAPPING',json.dumps([(r['original']['SPELL_ID'],r['env']['id'],r['scroll']['spell_name'],r['env']['description']) for r in book],ensure_ascii=False))
print('UNIT SAMPLE',json.dumps(units[0],ensure_ascii=False))
