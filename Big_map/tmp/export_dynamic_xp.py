"""Export only the original per-level kill XP increments; no runtime DBF dependency."""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inspect_sg_map import parse_dbf_rows
from unit_revive_costs import REVIVE_PROFILE_IDS_BY_NAME
GAME = Path(r'C:\Program Files (x86)\Steam\steamapps\common\Disciples II Rise of the Elves\Globals')
dyn = {r['UPGRADE_ID'].lower(): r for r in parse_dbf_rows(GAME/'GDynUpgr.dbf')}
profiles = {r['UNIT_ID'].lower(): (int(r['DYN_UPG_LV']), *[int(dyn[r[k].lower()]['XP_KILLED'] or 0) for k in ('DYN_UPG1', 'DYN_UPG2')]) for r in parse_dbf_rows(GAME/'Gunits.dbf')}
for name, ids in REVIVE_PROFILE_IDS_BY_NAME.items():
    assert len({profiles[i] for i in ids}) == 1, name
groups = defaultdict(list)
for key, profile in sorted(profiles.items()):
    groups[profile].append(key)
lines = ['"""Gunits.DYN_UPG_LV and GDynUpgr.XP_KILLED, installed Rise of the Elves.',
         'Exported 2026-09-16; identical profiles share a row to keep the table compact.',
         'GDynUpgr SHA256: '+hashlib.sha256((GAME/'GDynUpgr.dbf').read_bytes()).hexdigest(),
         '"""', 'from unit_revive_costs import REVIVE_PROFILE_IDS_BY_NAME', '',
         '# Each profile is (last early level, early increment, late increment).',
         'DYNAMIC_KILL_XP_PROFILES = {', '    unit_id: profile', '    for profile, ids in (']
lines += [f'        ({profile!r}, {" ".join(ids)!r}),' for profile, ids in sorted(groups.items())]
lines += ['    )', '    for unit_id in ids.split()', '}', '', '',
          'def dynamic_kill_xp_increment(unit, next_level):',
          '    unit_id = str(unit.get("unit_id") or unit.get("game_unit_id") or "").strip().lower()',
          '    if not unit_id:',
          '        ids = REVIVE_PROFILE_IDS_BY_NAME.get(str(unit.get("name") or unit.get("кто") or ""), ())',
          '        unit_id = ids[0] if ids else ""',
          '    threshold, early, late = DYNAMIC_KILL_XP_PROFILES.get(unit_id, (0, 0, 0))',
          '    return early if next_level <= threshold else late', '']
(ROOT/'unit_dynamic_xp.py').write_text('\n'.join(lines), encoding='utf8')
print(f'Exported {len(profiles)} profiles in {len(groups)} grouped rows; all 252 name bindings verified.')
