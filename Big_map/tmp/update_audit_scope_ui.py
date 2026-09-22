from pathlib import Path
site = Path("outputs/disciples-audit-site")
path = site / "dist/app.js"
src = path.read_text(encoding="utf8")

def replace(old, new):
    global src
    old, new = old.replace("BT", chr(96)), new.replace("BT", chr(96))
    assert old in src, old
    src = src.replace(old, new)

replace("fixed:'Исправлено'", "fixed:'Исправлено',intentional:'Осознанное ограничение'")
replace("const card=(f,i)=>", "const isIssue=r=>!['intentional','fixed','excluded'].includes(r.status);\nconst recordBadge=r=>['intentional','fixed'].includes(r.status)?badge(statuses[r.status],r.status):r.priority?badge(priorities[r.priority],r.priority):badge(statuses[r.status],r.status);\nconst card=(f,i)=>")
replace("${badge(priorities[f.priority],f.priority)}", "${recordBadge(f)}")
replace("${r.priority?badge(priorities[r.priority],r.priority):badge(statuses[r.status],r.status)}", "${recordBadge(r)}")
replace("A.findings.filter(f=>f.priority==='high'),quick=A.findings.filter(f=>f.quick)", "A.findings.filter(f=>isIssue(f)&&f.priority==='high'),quick=A.findings.filter(f=>isIssue(f)&&f.quick)")
replace("Влияют на стратегию обучения", "Требуют исправления в текущей модели")
replace('<div class="overview-grid">', '<section class="source-card scope-panel"><h2>${esc(A.trainingScope.title)}</h2><p>${esc(A.trainingScope.text)}</p><div class="scope-chips">${A.findings.filter(r=>r.status===\'intentional\').map(r=>BT<button class="scope-link" data-record="findings:${r.id}">${esc(r.title)} ↗</button>BT).join(\'\')}</div><p>${esc(A.trainingScope.note)}</p></section><div class="overview-grid">')
replace("Сначала — влияние на игру", "Приоритетные исправления")
replace("Приоритет — правила, которые меняют решения агента.", "Приоритет — ошибки внутри принятой модели обучения. Осознанные ограничения и уже исправленные пункты не входят в счётчики проблем.")
replace("Подтверждённые различия, влияние на обучение и конкретный следующий шаг.", "Ошибки, осознанные ограничения и выполненные исправления. Рекомендации учитывают компактный набор действий и один управляемый отряд.")
replace("[['all','Все приоритеты'],['high','Значительные'],['medium','Менее значительные'],['quick','Быстро поправить']]", "[['all','Все записи'],['gaps','Требуют исправления'],['high','Значительные'],['medium','Менее значительные'],['quick','Быстро поправить'],['intentional','Осознанные ограничения'],['fixed','Исправлено']]")
replace("['review','Требует проверки']]", "['review','Требует проверки'],['intentional','Осознанные ограничения']]")
replace("f==='quick'?r.quick:f==='gaps'?['missing','partial','alias','review'].includes(r.status):v==='mechanics'?r.priority===f:r.status===f", "f==='quick'?isIssue(r)&&r.quick:f==='gaps'?isIssue(r)&&['missing','partial','alias','review'].includes(r.status):v==='mechanics'?(['intentional','fixed'].includes(f)?r.status===f:isIssue(r)&&r.priority===f):r.status===f")
replace("${v==='mechanics'?'различий':'записей'}", "записей")
replace("«Упрощено» — эффект работает с другим поведением.", "«Упрощено» — остаётся несоответствие, требующее исправления. «Осознанное ограничение» — автор подтвердил такое поведение как часть модели обучения; это не ошибка. «Исправлено» — отмеченная проблема устранена.")
replace("${r.impact?BT<section", "${r.rationale?BT<section class=\"detail-section\"><h3>Почему принято такое решение</h3><p>${esc(r.rationale)}</p></section>BT:''}${r.impact?BT<section")
replace("<h3>Как приблизить к игре${r.effort?", "<h3>${r.status==='intentional'?'Решение для текущей среды':r.status==='fixed'?'Результат исправления':'Как исправить в текущей модели'}${r.effort&&isIssue(r)?")
path.write_text(src, encoding="utf8")
css = site / "dist/style.css"
css.write_text(css.read_text(encoding="utf8") + '\n.badge.intentional{color:var(--blue);background:#243240;border-color:#3c5166}.badge.fixed{color:var(--green);background:#23362f;border-color:#36554a}.scope-panel{border-left:3px solid var(--blue)}.scope-link{background:#243240;color:#bdd7ed;border:1px solid #3c5166;border-radius:6px;padding:9px 12px;text-align:left;font-size:13px}\n', encoding="utf8")
index = site / "dist/index.html"
text = index.read_text(encoding="utf8")
for name in ("style.css", "data.js", "app.js"):
    text = text.replace(f'"{name}"', f'"{name}?rev=20260916-scope"')
index.write_text(text, encoding="utf8")
