from pathlib import Path
root=Path(__file__).resolve().parents[1]
site=root/'outputs/disciples-audit-site'
p=site/'dist/app.js'
s=p.read_text(encoding='utf8')
needle='<section class="source-card scope-panel"><h2>${esc(A.trainingScope.title)}</h2>'
panel='<section class="source-card"><h2>${esc(A.recentFixes.title)}</h2><p>${esc(A.recentFixes.text)}</p><div class="scope-chips">${A.recentFixes.ids.map(id=>A.findings.find(r=>r.id===id)).map(r=>`<button class="scope-link" data-record="findings:${r.id}">✓ ${esc(r.title)}</button>`).join(\'\')}</div></section>'
assert s.count(needle)==1
s=s.replace(needle,panel+needle)
needle="['intentional','Осознанные ограничения']];return heading"
assert needle in s
s=s.replace(needle,"['intentional','Осознанные ограничения'],['fixed','Исправлено']];return heading")
p.write_text(s,encoding='utf8')
p=site/'dist/index.html'
s=p.read_text(encoding='utf8').replace('rev=20260916-scope','rev=20260916-fixes')
s=s.replace('<span>Текущая рабочая копия</span>','<span>Исправления: 16 сентября 2026</span>')
p.write_text(s,encoding='utf8')
p=root/'tmp/check_audit_scope.cjs'
s=p.read_text(encoding='utf8').replace("['mechanics','fixed',1]","['mechanics','fixed',5]").replace("['mechanics','gaps',26]","['mechanics','gaps',22]").replace("['spells','intentional',12]","['spells','intentional',16]").replace("['target','summon','research_cost']","['target','summon','research_cost','unit_stats','scroll_alias','might','cast_cost']").replace("r.status==='partial').length\",ctx),4)","r.status==='partial').length\",ctx),0)")
p.write_text(s,encoding='utf8')
# The generator runs these durable corrections after recreating its base audit.
p=root/'tmp/build_d2_audit.py'
s=p.read_text(encoding='utf8')
line="runpy.run_path(str(ROOT / 'outputs/disciples-audit-site/tools/apply_completed_fixes.py'))"
if line not in s:s+='\n'+line+'\n'
# The evidence locator must follow the corrected summon reference.
s=s.replace('ev(\'scroll_spell_data.py\',"\'summon_unit_name\': \'Хуорн\'",2)', 'ev(\'scroll_spell_data.py\',"\'summon_unit_name\': \'Темный энт\'",2)')
p.write_text(s,encoding='utf8')
