from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from campaign_env import CampaignEnv
from campaign_env_data import CampaignConstantsMixin as C
from campaign_env_battle import CampaignBattleMixin
from campaign_env_inventory import CampaignInventoryMixin
from battle_env import _apply_dynamic_unit_levelup
from copy import deepcopy
class P(CampaignBattleMixin,CampaignInventoryMixin,C):pass
p=P();p._map=None
checks={}
for name in ['Хуорн','Энт Малый','Энт Большой','Вердант']:
 checks['unresolved_'+name]=p._build_summoned_blue_team(name) is None
checks['might_wrong_30']=p._potion_item_definition('Potion of Might')['effect']['multiplier']==1.3
checks['treebark_unresolved']=not p._potion_item_definition('Treebark Potion')
checks['tome_thought_unresolved']=not p._book_effect_definition('Tome of Thought')
checks['tome_mind_exists']=bool(p._book_effect_definition('Tome of Mind'))
checks['all_missing_scroll_aliases_absent_from_maps']=all(name not in '\n'.join(x.read_text(encoding='utf8') for x in Path('maps').glob('*.py')) for name in ['Хуорн','Энт Малый','Энт Большой','Вердант'])
u={'Level':1,'damage':60,'original_damage':60,'health':600,'max_health':600,'armor':20,'accuracy':80,'exp_kill':1030,'hero':False}
_apply_dynamic_unit_levelup(u,1000);_apply_dynamic_unit_levelup(u,1000)
checks['growth_compounds']=u['max_health']==726 and u['damage']==73
checks['dynamic_xp_unchanged']=u['exp_kill']==1030
print(json.dumps(checks,ensure_ascii=False,indent=2));assert all(checks.values())
Path('tmp/d2_audit_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
