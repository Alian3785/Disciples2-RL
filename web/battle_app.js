(() => {
  'use strict';
  // Exported replays share repeated frame values (export_battle_viewer.py, shared-replay-values-v1).
  const raw=JSON.parse(document.getElementById('run-data').textContent);
  const data=raw.format==='shared-replay-values-v1'?{...raw.metadata,records:raw.records.map(r=>({label:r.label,seed:r.seed,policy:r.policy,
    frames:r.states.map((row,i)=>{
      const frame={};
      r.frame_keys.forEach((key,j)=>{const index=r.fields[i][j];if(index>=0)frame[key]=raw.pool[index];});
      frame.state=Object.fromEntries(r.state_keys.map((key,j)=>[key,raw.pool[row[j]]]));
      return frame;
    })}))}:raw;
  if(raw.format)document.getElementById('run-data').textContent='';
  const $=id=>document.getElementById(id), fmt=n=>new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3}).format(n);
  const canvas=$('board'), ctx=canvas.getContext('2d');
  const manaMarkers={
    infernal:{letter:'П',color:'#c65320'},life:{letter:'Ж',color:'#25833c'},
    death:{letter:'С',color:'#7940b0'},runes:{letter:'Р',color:'#246bb1'},elves:{letter:'Э',color:'#117c77'}
  };
  const manaLegend=document.createElement('div');
  manaLegend.className='legend mana-legend';manaLegend.setAttribute('aria-label','Источники маны');
  $('chest-status').before(manaLegend);
  let mode='manual', session=null, manual=null, manualMap=null, busy=false, frame=0, recordIndex=0, timer=null, messages=[];
  let manualConstruction=null, manualTurnRules=null, manualCombat=null, manualCapital=null, manualPotions=null, capitalOpen=false, potionsOpen=false, selectedPotion=0, capitalKey=null;
  let manualEquipment=null, manualBattleItems=null, manualSites=null, manualRecruitment=null, manualTerritory=null, manualRuins=null, manualSpells=null, manualCasting=null, manualSwaps=null, spellsOpen=false;
  const records=data.records||[];
  const combatInfo=()=>mode==='manual'?(manualCombat||data.combat):data.combat;
  const heroLevel=snap=>Math.max(0,...[0,1,2,3,4,5].filter(i=>combatInfo()?.catalogue?.[snap.state.unit_ids?.[i]]?.hero).map(i=>snap.unit_experience?.[i]?.[0]||0));
  const profile=i=>{
    const original=i>=0&&i<12?(current()?.state.unit_ids?combatInfo()?.catalogue?.[current().state.unit_ids[i]]:(i<6?combatInfo()?.heroes[i]:combatInfo()?.enemies[current()?.state.enemy]?.[i-6])):null;
    const lowered=combatInfo()?.catalogue?.[current()?.state.decay_form?.[i]];
    let base=lowered?{...lowered,name:original.name+' → '+lowered.name}:original;
    if(base&&current()?.state.copied?.[i])base={...base,name:'Двойник → '+base.name,size:1};
    if(base&&(current()?.state.summon_owner?.[i]??-1)>=0)base={...base,name:base.name+' · призван'};
    if(base&&current()?.state.imp?.[i])base={...base,name:base.name+' · '+(base.size===2?'Толстый бес':'Бес'),role:'melee',attack_type:'weapon',immunities:0,protections:0,unit_type:'Warrior'};
    const owners=current()?.state.healer_wards?.[i]||[];
    const temporary=[1,256,8,2].reduce((bits,bit,j)=>bits|(owners[j]?bit:0),0);
    const s=current()?.state,potionWard=s&&!s.imp?.[i]&&!s.decay_form?.[i]&&(s.summon_owner?.[i]??-1)<0?(s.copied?.[i]?(s.copy_traits?.[i]?.[3]||0):(s.potion_wards?.[i]||0)):0;
    const bookWard=base?.hero&&i<6&&!s?.imp?.[i]&&!s?.decay_form?.[i]&&!s?.copied?.[i]?(equipmentInfo()?.items?.[s?.equipped?.[3]]?.ward||0):0;
    if(base)base={...base,protections:(base.protections||0)|temporary|potionWard|bookWard};
    return base&&current()?.state.fenrir?.[i]&&!current()?.state.imp?.[i]?{...base,name:'Дух Фенрира',role:'melee',attack_type:'weapon',unit_type:'Warrior'}:base;
  };
  const isMage=i=>profile(i)?.role==='area';
  const isWarrior=i=>profile(i)?.role==='melee';
  const isHealer=i=>profile(i)?.role==='healer';
  const isCopier=i=>profile(i)?.unit_type==='Doppelganger';
  const isSummoner=i=>['Summoner','Occultmaster','Lyf','Laclaan'].includes(profile(i)?.unit_type);
  const targetAction=i=>i<6&&isCopier(current()?.state.actor)?57+i:8+i%6;
  const isPowerSupport=i=>['Travnitsa','Novice','Dwarfdruid','Arhidruid'].includes(profile(i)?.unit_type);
  const isMassHealer=i=>['Profit','Deva roshi'].includes(profile(i)?.unit_type);
  const role=i=>profile(i)?.name||'Пусто';
  const unitName=i=>(i<6?'Ваш ':'Вражеский ')+role(i).toLowerCase()+' '+(i%6+1);
  const partyText=units=>{const counts=new Map();for(const u of units||[])if(u)counts.set(u.name,(counts.get(u.name)||0)+1);return [...counts].map(([name,count])=>name+' × '+count).join(' · ');};
  const enemyParty=(map,enemy)=>(map.city_defender_roles?.[enemy]?({guard:'Охрана · ',garrison:'Гарнизон · ',capital:'Страж столицы · '}[map.city_defender_roles[enemy]]||''):'')+partyText(current()?.state.in_battle?Array.from({length:6},(_,i)=>profile(i+6)):combatInfo()?.enemies[enemy]);
  const sourceName=key=>combatInfo()?.attack_types.find(t=>t.key===key)?.name||key;
  const protectionNames=bits=>(combatInfo()?.attack_types||[]).filter(t=>bits&t.bit).map(t=>t.name).join(', ')||'нет';
  const blockText=s=>{const parts=[];for(const [key,label] of [['last_immune','Иммунитет'],['last_ward','Защита поглотила удар']]){const names=Array.from({length:12},(_,i)=>i).filter(i=>s[key]&(1<<i)).map(unitName);if(names.length)parts.push(label+': '+names.join(', ')+'.');}return parts.length?' '+parts.join(' '):'';};
  function current(){return mode==='manual'?manual:records[recordIndex]?.frames[frame];}
  function currentMap(){return mode==='manual'?(manualMap||data.map):data.map;}
  function mapTitle(map){
    const name=(map.name||'').replace(/^NumberGrid-\d+x\d+-/,'');
    return map.display_name||({
      default:'Default', super_last_stand:'Super Last Stand',
      siege_train:'Siege Train', wotans_retribution:'Возмездие Вотана', trade_train:'Торговое обучение'
    })[name]||name;
  }
  function construction(){return mode==='manual'?(manualConstruction||data.construction):data.construction;}
  function turnRules(){return mode==='manual'?(manualTurnRules||data.turn_rules):data.turn_rules;}
  const potionInfo=()=>mode==='manual'?(manualPotions||data.potions):data.potions;
  const equipmentInfo=()=>mode==='manual'?(manualEquipment||data.equipment):data.equipment;
  const battleItemInfo=()=>mode==='manual'?(manualBattleItems||data.battle_items):data.battle_items;
  const siteInfo=()=>mode==='manual'?(manualSites||data.sites):data.sites;
  const recruitmentInfo=()=>mode==='manual'?(manualRecruitment||data.recruitment):data.recruitment;
  const territoryInfo=()=>mode==='manual'?(manualTerritory||data.territory):data.territory;
  const spellInfo=()=>mode==='manual'?(manualSpells||data.spell_research):data.spell_research;
  const castInfo=()=>mode==='manual'?(manualCasting||data.spell_casting):data.spell_casting;
  const swapInfo=()=>mode==='manual'?(manualSwaps||data.unit_swaps):data.unit_swaps;
  const manaNames=['Преисподняя','Жизнь','Смерть','Руны','Природа'];
  const elementNames={weapon:'оружия',earth:'Земли',fire:'Огня',water:'Воды',poison:'яда',death:'Смерти',mind:'Разума',life:'Жизни',air:'Воздуха'};
  const statNames={armor:'броня',damage:'урон',accuracy:'точность',initiative:'инициатива'};
  function castEffect(c){
    const change=c.stat==='armor'?(c.amount>0?'+':'')+c.amount:'×'+c.multiplier;
    switch(c.kind){
      case 'damage':return 'Урон '+c.amount+' магией '+elementNames[c.element]+' каждому бойцу ближайшего отряда вне городов, руин и столиц; иммунитет и защита блокируют.';
      case 'debuff':return 'Ближайший отряд вне городов, руин и столиц: '+statNames[c.stat]+' '+change+' до следующего хода.';
      case 'summon_battle':return 'Призыв: '+c.unit_name+' один сражается с ближайшим отрядом. Отряд героя не участвует.';
      case 'moves':return '+'+c.percent+'% от максимума очков перемещения с учётом сапог и уровня героя, дробная часть отбрасывается.';
      case 'heal':return 'Лечение '+c.amount+' HP каждому живому бойцу героя.';
      case 'buff':return c.stat==='terrain'?(c.terrain==='forest'?'Лес':'Вода')+' стоит 2 очка, как равнина, до следующего хода.':'Отряд героя: '+statNames[c.stat]+' '+change+' в боях до следующего хода.';
      case 'ward':return 'Отряд героя: защита от '+elementNames[c.element]+' в боях до следующего хода.';
      case 'health_bonus':return 'Отряд героя: +'+c.amount+' HP в боях до следующего хода.';
    }
    return '';
  }
  function renderSpells(snap){
    const info=spellInfo(),lord=construction()?.lord,map=currentMap();
    $('spells-view').hidden=!info;
    $('lord-summary').textContent=lord?'Правитель: '+lord.name+'. '+lord.description:'';
    const choice=construction()?.lords?.find(l=>l.id===$('lord-type').value);
    $('lord-choice-hint').textContent=choice?'Новая игра — '+choice.name+': '+choice.description:'';
    $('lord-type').disabled=busy||mode!=='manual';
    if(!info)return;
    $('spells-title').textContent='Заклинания · '+construction().name;
    $('spell-rules').textContent=lord.description+(map.max_spell_level?' Ограничение сценария: изучение в башне до '+map.max_spell_level+' уровня.':'');
    const learned=(snap.spell_status||[]).filter(c=>c===1).length;
    const casting=castInfo(),quotes=snap.cast_quotes;
    $('spell-status').textContent='Изучено '+learned+' / '+info.spells.length+' · '+(snap.state.spell_researched_today?'Сегодня заклинание уже изучено.':'Дневной лимит свободен.')+(casting?' · Одно заклинание можно применить '+(casting.daily_limit===2?'дважды':'один раз')+' за ход.':'');
    const castLabels=['Можно применить','Не изучено','Лимит применений на этот ход исчерпан','Недостаточно маны','Нет цели или эффект уже действует','Недоступно во время боя','Игра завершена'];
    const labels=['Можно изучить','Изучено','V уровень доступен только правителю-магу','Нужна башня магии','Недостаточно маны','Сегодня уже изучено заклинание','Недоступно во время боя','Игра завершена','Выше предела магии сценария'];
    $('spell-list').replaceChildren(...info.spells.map((spell,i)=>{
      const card=document.createElement('article');card.className='spell-card'+(snap.spell_status?.[i]===1?' is-learned':'');
      card.hidden=$('spell-level').value!=='all'&&Number($('spell-level').value)!==spell.level;
      const title=document.createElement('h3');title.textContent=spell.name+' · '+['','I','II','III','IV','V'][spell.level];
      const description=document.createElement('p');description.textContent=spell.description;
      const original=document.createElement('details'),summary=document.createElement('summary'),originalText=document.createElement('p');
      summary.textContent='Описание оригинала';originalText.textContent=spell.original_name+': '+spell.original_description;original.append(summary,originalText);
      const cost=document.createElement('p');cost.className='spell-cost';cost.textContent=spell.cost.map((n,j)=>n?manaNames[j]+': '+fmt(n)+' / есть '+fmt(snap.state.mana[j]):null).filter(Boolean).join(' · ');
      const status=document.createElement('p');status.className='spell-state';status.textContent=labels[snap.spell_status?.[i]??7];
      const button=document.createElement('button');button.dataset.action=spell.action;button.textContent=snap.spell_status?.[i]===1?'Изучено':'Изучить';button.setAttribute('aria-label','Изучить: '+spell.name);
      button.disabled=mode!=='manual'||busy||!session||!snap.action_mask[spell.action];button.onclick=()=>sendAction(spell.action);
      card.append(title,description,original,cost,status,button);
      if(casting?.spells[i])castBlock(card,casting.spells[i],i);
      return card;
    }));
    // Spells of another faction bought in a spell shop: no research, one cast per day.
    $('spell-list').append(...(casting?.spells||[]).slice(info.spells.length).filter(c=>!c.item_only).map((cast,k)=>{
      const i=info.spells.length+k,learned=(snap.state.learned_spells>>>i)&1;
      const card=document.createElement('article');card.className='spell-card'+(learned?' is-learned':'');
      card.hidden=$('spell-level').value!=='all'&&Number($('spell-level').value)!==cast.level;
      const title=document.createElement('h3');title.textContent=cast.name+' · '+['','I','II','III','IV','V'][cast.level];
      const description=document.createElement('p');description.textContent=cast.description;
      const status=document.createElement('p');status.className='spell-state';status.textContent=learned?'Куплено в лавке заклинаний':'Продаётся в лавке заклинаний; изучить нельзя';
      card.append(title,description,status);castBlock(card,cast,i);return card;
    }));
    function castBlock(card,cast,i){
      const effect=document.createElement('p');effect.className='spell-effect';effect.textContent=castEffect(cast);
      const price=document.createElement('p');price.className='spell-cost';price.textContent='Применение: '+cast.cast_mana.map((n,j)=>n?manaNames[j]+' '+fmt(n):null).filter(Boolean).join(' · ');
      const code=quotes?.status?.[i]??6,target=quotes?.target?.[i]??-1;
      const castState=document.createElement('p');castState.className='spell-state';
      castState.textContent=castLabels[code]+(code===0&&target>=0?' · цель: отряд № '+(target+1):'')+(snap.state.spell_casts?.[i]?' · применено сегодня: '+snap.state.spell_casts[i]:'');
      const castButton=document.createElement('button');castButton.dataset.action=cast.action;castButton.textContent='Применить';castButton.setAttribute('aria-label','Применить: '+cast.name);
      castButton.disabled=mode!=='manual'||busy||!session||!snap.action_mask[cast.action];castButton.onclick=()=>sendAction(cast.action);
      card.append(effect,price,castState,castButton);
    }
    renderScrolls(snap,casting,quotes);
  }
  // Scrolls and staves: docs/SCROLLS_AND_STAVES.md.
  function renderScrolls(snap,casting,quotes){
    const items=casting?.items||[],s=snap.state;
    $('scroll-section').hidden=!items.length;
    if(!items.length)return;
    $('scroll-status').textContent=(s.scroll_magic?'Свитки и посохи читает отряд героя.':'Читать свитки некому: нужен герой-маг или нанятый маг ('+fmt(casting.hire_cost)+' золота, один раз за игру).')+' Свиток не требует маны и сгорает. Посох остаётся, тратит свою ману и требует героя-мага, правителя-мага или мудрость магии ('+casting.lore_level+' уровень героя). Дневного лимита нет; одно и то же заклинание не складывается.';
    const hire=document.createElement('button');hire.dataset.action=casting.hire_action;hire.textContent='Нанять мага · '+fmt(casting.hire_cost)+' золота';
    hire.hidden=!!s.scroll_magic;hire.disabled=mode!=='manual'||busy||!session||!snap.action_mask[casting.hire_action];hire.onclick=()=>sendAction(casting.hire_action);
    $('scroll-hire').replaceChildren(hire);
    const labels=['Можно применить','Нужен маг для чтения свитков','Нет в инвентаре','Недостаточно маны','Нет цели или эффект уже действует','Недоступно во время боя','Игра завершена','Нужен герой-маг, правитель-маг или мудрость магии'];
    $('scroll-list').replaceChildren(...items.map((item,k)=>{
      const spell=casting.spells[item.spell],owned=quotes?.owned?.[k]??0,code=quotes?.item_status?.[k]??6,target=quotes?.item_target?.[k]??-1;
      const card=document.createElement('article');card.className='spell-card'+(owned?' is-learned':'');
      const title=document.createElement('h3');title.textContent=item.name+' · '+owned+' шт.';
      const label=document.createElement('p');label.textContent=item.label;
      const effect=document.createElement('p');effect.className='spell-effect';effect.textContent=castEffect(spell);
      const price=document.createElement('p');price.className='spell-cost';price.textContent=item.cast_mana?'Применение: '+item.cast_mana.map((n,j)=>n?manaNames[j]+' '+fmt(n):null).filter(Boolean).join(' · '):'Применение: без маны, свиток сгорает';
      const state=document.createElement('p');state.className='spell-state';state.textContent=labels[code]+(code===0&&target>=0?' · цель: отряд № '+(target+1):'');
      const button=document.createElement('button');button.dataset.action=item.action;button.textContent=item.category==='scroll'?'Прочитать':'Применить';button.setAttribute('aria-label',button.textContent+': '+item.name);
      button.disabled=mode!=='manual'||busy||!session||!snap.action_mask[item.action];button.onclick=()=>sendAction(item.action);
      card.append(title,label,effect,price,state,button);return card;
    }));
  }
  const ruinInfo=()=>mode==='manual'?(manualRuins||data.ruins):data.ruins;
  const lootText=ruin=>[...Object.entries(ruin.potions||{}),...Object.entries(ruin.items||{})].filter(([,n])=>n).map(([key,n])=>(potionInfo()?.find(p=>p.key===key)?.name||equipmentInfo()?.items.find(p=>p.key===key)?.name||key)+' × '+n).concat(ruin.gold?[fmt(ruin.gold)+' золота']:[]).join(', ');
  function renderRuins(snap){
    const ruins=ruinInfo()?.ruins||[],s=snap.state;
    $('ruins-panel').hidden=!ruins.length;
    $('ruin-list').replaceChildren(...ruins.map((ruin,i)=>{
      const card=document.createElement('article');card.className='city-card';
      const title=document.createElement('b');title.textContent=ruin.name+' · 3 × 3';
      const status=document.createElement('p');status.textContent=s.ruin_looted[i]?'Разграблены. Непроходимое препятствие.':partyText(combatInfo()?.enemies[ruin.enemy]);
      const loot=document.createElement('small');loot.textContent=(s.ruin_looted[i]?'Награда получена: ':'Награда за победу: ')+lootText(ruin);
      card.append(title,status,loot);
      if(!s.ruin_looted[i]){
        const action=(snap.map_commands||[]).findIndex(c=>c[0]===ruin.enemy);
        const button=document.createElement('button');button.textContent='Атаковать руины'+(action>=0?' · '+snap.map_commands[action][1]+' очков движения':'');
        if(action>=0){button.dataset.action=action;button.onclick=()=>sendAction(action);}else button.disabled=true;
        card.append(button);
      }
      return card;
    }));
  }
  function renderTerritory(snap){
    const info=territoryInfo(),q=snap.territory_quotes,s=snap.state;
    $('city-services').hidden=!info;$('mana-resources').hidden=!info;
    const manaKinds=(info?.resource_kinds||[]).filter(kind=>manaMarkers[kind]&&info.mines.some(mine=>mine.kind===kind));
    manaLegend.hidden=!manaKinds.length;
    manaLegend.replaceChildren(...manaKinds.map(kind=>{
      const marker=manaMarkers[kind],entry=document.createElement('span'),icon=document.createElement('i');
      icon.textContent=marker.letter;icon.style.backgroundColor=marker.color;
      entry.append(icon,document.createTextNode(info.resource_names[info.resource_kinds.indexOf(kind)]));return entry;
    }));
    $('income-summary').textContent='Текущий доход: +'+fmt(q?.income[0]??turnRules().income)+' золота / ход';
    if(!info||!q)return;
    $('mana-resources').replaceChildren(...info.resource_names.slice(1).map((name,i)=>{
      const row=document.createElement('div'),label=document.createElement('span'),value=document.createElement('b');
      label.textContent=name;value.textContent=fmt(s.mana[i])+' (+'+fmt(q.income[i+1])+')';row.append(label,value);return row;
    }));
    $('city-list').replaceChildren(...info.cities.map((city,i)=>{
      const card=document.createElement('article');card.className='city-card';
      const heading=document.createElement('b');heading.textContent=city.name+' · уровень '+s.city_levels[i];
      const count=(s.enemy_positions||currentMap().opponent_positions).filter((p,j)=>s.alive[j]&&p.every((v,k)=>v===city.position[k])).length;
      const status=document.createElement('p');status.textContent='Клетка '+city.position.join(', ')+' · '+(s.city_owned[i]?'Ваш город':count?'Защитников: '+count+' отряда':'Для захвата войдите с живым героем');
      const detail=document.createElement('small');detail.textContent='Рост земли +'+info.city_growth[s.city_levels[i]]+' клеток/ход · регенерация и броня +'+info.city_bonus[s.city_levels[i]]+'%';
      const button=document.createElement('button');button.dataset.action=city.action;button.onclick=()=>sendAction(city.action);
      button.textContent=!s.city_owned[i]?'Сначала захватите город':s.city_levels[i]>=5?'Максимальный уровень':'Улучшить · '+fmt(q.city_costs[i])+' золота';
      card.append(heading,status,detail,button);return card;
    }));
  }
  function renderRecruitment(snap){
    const info=recruitmentInfo(),s=snap.state,q=snap.recruitment_quotes;
    $('recruitment-services').hidden=!info;$('leadership').hidden=!info;
    if(!info||!q)return;
    const [capacity,used,free]=q.leadership;
    const vacancies=q.reward_vacancies??free;
    $('leadership').textContent='Лидерство: '+used+' / '+capacity+' · свободно '+free+(vacancies?' · штраф −'+fmt(vacancies*.05)+' за шаг':'');
    const at=snap.territory_quotes?.at_owned_fort??info.positions.some(p=>p.every((v,i)=>v===s.position[i]));
    $('recruitment-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Найм доступен после боя.':!at?'Для найма войдите в столицу или захваченный город.':'Найм за золото, без расхода очков перемещения. Герой не занимает лидерство; большой юнит занимает 2 очка и обе клетки своей линии.';
    $('recruitment-offers').replaceChildren(...info.offers.map((offer,i)=>{
      if(offer.mercenary)return null;
      const card=document.createElement('article');card.className='site-offer';
      const name=document.createElement('b');name.textContent=offer.name+' · лидерство '+offer.size;
      const detail=document.createElement('small');detail.textContent=offer.building?'Нужна постройка: '+offer.building:'Без дополнительной постройки';
      const b=document.createElement('button');b.dataset.action=offer.action;b.onclick=()=>sendAction(offer.action);
      const blocked=offer.building_action!=null&&snap.building_status?.[offer.building_action-18]!==1;
      b.textContent='Нанять · '+fmt(offer.cost)+' золота';
      const reason=document.createElement('small');reason.textContent=blocked?'Постройка не открыта':free<offer.size?'Недостаточно лидерства':q.targets[i]<0?'Нет места в нужном ряду':s.gold<offer.cost?'Недостаточно золота':'';
      card.append(name,detail,b,reason);return card;
    }).filter(Boolean));
  }
  const movementCap=snap=>snap.movement_cap;
  const closedShopGroups=new Set();
  function renderSites(snap){
    const info=siteInfo(),s=snap.state,panel=$('site-services');
    panel.hidden=!info;if(!info)return;
    const here=kind=>info.sites.find(site=>site.kind===kind&&site.interaction_tiles.some(p=>p.every((v,i)=>v===s.position[i])));
    const button=(text,action)=>{const b=document.createElement('button');b.textContent=text;b.dataset.action=action;b.onclick=()=>sendAction(action);return b;};
    const ready=kind=>s.done?'Эпизод завершён.':s.in_battle?'Завершите бой.':here(kind)?'Вы у входа: '+here(kind).name+'.':'Подойдите на одну из пяти подсвеченных клеток у входа.';
    // With several sites of a kind, every offer names its own site.
    const siteName=(kind,site)=>info.sites.filter(x=>x.kind===kind).length>1&&info.sites[site]?' · '+info.sites[site].name:'';
    for(const [id,kind] of [['merchant-site','merchant'],['trainer-site','trainer'],['spell-shop','spell_shop']])$(id).hidden=!info.sites.some(x=>x.kind===kind);
    const recruits=recruitmentInfo(),quotes=snap.recruitment_quotes;
    $('mercenary-camp').hidden=!recruits?.offers.some(o=>o.mercenary);
    $('mercenary-status').textContent=ready('mercenary')+' Нужен живой герой; запас не пополняется.';
    $('mercenary-offers').replaceChildren(...(recruits?.offers||[]).flatMap((offer,i)=>{
      if(!offer.mercenary||!quotes)return [];
      const card=document.createElement('article');card.className='site-offer';
      const stock=s.mercenary_stock[offer.stock_index],price=quotes.prices[i];
      const name=document.createElement('b');name.textContent=offer.name+' · осталось '+stock+siteName('mercenary',offer.camp);
      const detail=document.createElement('small');detail.textContent='Лидерство '+offer.size+' · '+(offer.row==='front'?'передний ряд':'задний ряд');
      const reason=document.createElement('small');reason.textContent=!stock?'Все бойцы наняты':quotes.leadership[2]<offer.size?'Недостаточно лидерства':quotes.targets[i]<0?'Нет места в нужном ряду':s.gold<price?'Недостаточно золота':'';
      card.append(name,detail,button('Нанять · '+fmt(price)+' золота',offer.action),reason);return [card];
    }));
    $('merchant-status').textContent=ready('merchant')+' Драгоценности здесь продаются автоматически.';
    const offers=info.buy.map((p,i)=>{
      const card=document.createElement('div');card.className='site-offer';
      const stock=s.merchant_stock?.[i]??0,price=snap.site_quotes?.buy?.[i]??p.price;
      const name=document.createElement('b');name.textContent=p.name+' · '+stock+' шт.'+siteName('merchant',p.site);
      const detail=document.createElement('small');detail.textContent=p.label;
      const reason=document.createElement('small');reason.textContent=!stock?'Продано':s.gold<price?'Недостаточно золота':'';
      card.append(name,detail,button('Купить · '+fmt(price)+' золота',p.action),reason);return card;
    });
    // One collapsible section per category: potions, artifacts, scrolls...
    const groups=[...new Set(info.buy.map(p=>p.group||'Зелья'))];
    $('merchant-stock').replaceChildren(...(groups.length<2?offers:groups.map(group=>{
      const box=document.createElement('details'),title=document.createElement('summary');box.className='site-group';
      const cards=offers.filter((_,i)=>(info.buy[i].group||'Зелья')===group);
      title.textContent=group+' · '+cards.length;box.open=!closedShopGroups.has(group);
      box.ontoggle=()=>box.open?closedShopGroups.delete(group):closedShopGroups.add(group);
      box.append(title,...cards);return box;
    })));
    const book=castInfo()?.spells||[];
    $('spell-shop-status').textContent=ready('spell_shop')+' Купленное заклинание сразу изучено. Его может применять любой правитель, по полной цене маны.';
    $('spell-shop-offers').replaceChildren(...(info.spells||[]).map((offer,i)=>{
      const card=document.createElement('article');card.className='site-offer';
      const stock=s.spell_stock?.[i]??0,index=book.findIndex(c=>c.id===offer.spell),cast=book[index];
      const learned=index>=0&&((s.learned_spells>>>index)&1);
      const name=document.createElement('b');name.textContent=offer.name+' · '+['','I','II','III','IV','V'][cast?.level||0]+' · осталось '+stock;
      const detail=document.createElement('small');detail.textContent=cast?castEffect(cast)+' Применение: '+cast.cast_mana.map((n,j)=>n?manaNames[j]+' '+fmt(n):null).filter(Boolean).join(' · ')+'.':'';
      const reason=document.createElement('small');reason.textContent=!stock?'Продано':learned?'Уже изучено':s.gold<offer.price?'Недостаточно золота':'';
      card.append(name,detail,button('Купить · '+fmt(offer.price)+' золота',info.spell_start+i),reason);return card;
    }));
    $('trainer-status').textContent=ready('trainer')+' Последний XP до повышения нужно заработать в бою.';
    $('trainer-units').replaceChildren(...Array.from({length:6},(_,slot)=>{
      const q=snap.site_quotes?.train?.[slot];if(!snap.max_hp[slot]||!q||info.train_start==null)return null;
      const card=document.createElement('div');card.className='site-offer';
      const name=document.createElement('b');name.textContent=role(slot)+' · '+(slot+1);
      const detail=document.createElement('small');detail.textContent='Опыт '+s.unit_xp[slot]+' / '+(q[1]+1)+' · '+q[0]+' золота / XP';
      const text=!s.hp[slot]?'Боец погиб':q[2]===0?'Достигнут предел обучения':q[3]===0?'Не хватает золота':'Обучить +'+q[3]+' XP · '+fmt(q[4])+' золота';
      card.append(name,detail,button(text,info.train_start+slot));return card;
    }).filter(Boolean));
  }
  function renderEquipment(snap){
    const info=equipmentInfo(),s=snap.state;
    $('equipment-panel').hidden=!info&&!battleItemInfo();
    if(!info)return;
    $('equipment-rule').textContent=(info.policy==='price'?'Более дорогие находки автоматически заменяют дешёвые в своём слоте.':'Автовыбор по правилам референса: сапоги — по движению, книга сохраняется, артефакты — по цене.')+' Заменённые вещи остаются в инвентаре.';
    $('equipment-slots').replaceChildren(...info.slots.map((name,slot)=>{
      const item=info.items[s.equipped?.[slot]],row=document.createElement('div');row.className='equipment-slot';
      const title=document.createElement('span');title.textContent=name;
      const value=document.createElement('b');value.textContent=item?item.name:'Пусто';
      const level=info.requires_skills?info.skill_levels?.[slot]:0,hero=heroLevel(snap);
      const detail=document.createElement('small');detail.textContent=item?item.label+' · '+fmt(item.price)+' золота':level>1&&hero<level?'Навык героя с '+level+' уровня':'';
      row.append(title,value,detail);return row;
    }));
    const counts=new Map();for(const id of s.item_inventory||[])if(id>=0)counts.set(id,(counts.get(id)||0)+1);
    $('equipment-count').textContent='Все предметы · '+[...counts.values()].reduce((a,b)=>a+b,0);
    $('equipment-stock').replaceChildren(...[...counts].map(([id,count])=>{
      const item=info.items[id],row=document.createElement('li');
      const worn=(s.equipped||[]).filter(i=>i===id).length;
      const cost=item.sell_price!=null?'автопродажа: '+fmt(item.sell_price):fmt(item.price);
      row.textContent=item.name+' × '+count+' · '+cost+' золота'+(worn?' · надето: '+worn:'');
      row.title=item.label;return row;
    }));
  }
  function battleItemSlot(info,s,k){
    const item=info.items[s.battle_items?.[k]];if(!item)return 'Пусто';
    return item.name+(item.category==='talisman'?' · зарядов: '+s.battle_item_charges[k]+(s.battle_item_used?.[k]?' · уже применён в этом бою':''):item.category==='potion'?' · зелье, одно применение':' · сфера, одно применение');
  }
  function renderBattleItems(snap){
    const info=battleItemInfo(),s=snap.state,mask=snap.action_mask,q=snap.battle_item_quotes;
    $('battle-items').hidden=!info;$('battle-item-actions').hidden=!info||!s.in_battle;
    if(!info)return;
    $('battle-item-rule').textContent='Положить предмет в свободный слот можно на карте один раз за ход, снять нельзя. В бою герой применяет предмет вместо своего действия. Зелья доступны всегда; сферы и талисманы — с 10 уровня героя, лорду-магу или с книгой: Tome of Arcanum для сфер, Tome of Sorcery для талисманов. Сфера и зелье расходуются; у талисмана '+info.talisman_uses+' зарядов, не больше одного применения за бой.';
    $('battle-item-slots').replaceChildren(...Array.from({length:info.slots},(_,k)=>{
      const row=document.createElement('div');row.className='equipment-slot';
      const title=document.createElement('span');title.textContent='Боевой слот '+(k+1);
      const value=document.createElement('b');value.textContent=battleItemSlot(info,s,k);
      const detail=document.createElement('small');detail.textContent=info.items[s.battle_items?.[k]]?.label||'';
      row.append(title,value,detail);return row;
    }));
    $('battle-item-equip').replaceChildren(...info.items.map((item,i)=>{
      if(!(q?.stock?.[i]>0))return null;
      const b=document.createElement('button');b.dataset.action=item.equip_action;
      b.textContent='В слот: '+item.name+' · есть '+q.stock[i];
      b.title=item.label+(q.access[i]?'':' · нужен 10 уровень героя, лорд-маг или книга');
      b.onclick=()=>sendAction(item.equip_action);return b;
    }).filter(Boolean));
    if(!s.in_battle)return;
    const groups=[];
    for(let k=0;k<info.slots;k++){
      const item=info.items[s.battle_items?.[k]];if(!item)continue;
      const box=document.createElement('div'),title=document.createElement('b');
      title.textContent='Предмет '+(k+1)+': '+battleItemSlot(info,s,k)+(item.party?' · действует на весь отряд':'');box.append(title);
      for(const [side,label] of [[0,'союзник'],[1,'враг']])for(let t=0;t<info.targets;t++){
        const action=info.use_start+(side*info.slots+k)*info.targets+t;if(!mask[action])continue;
        const b=document.createElement('button');b.dataset.action=action;b.textContent=label+' '+(t+1);
        b.setAttribute('aria-label',item.name+' → '+unitName(t+6*side));b.onclick=()=>sendAction(action);box.append(b);
      }
      groups.push(box);
    }
    $('battle-item-actions').replaceChildren(...groups);
  }
  function battleItemText(s){
    const item=battleItemInfo()?.items?.[s.last_battle_item];
    const target=s.last_target>=0?unitName(s.last_target):item?.enemy?'вражеский отряд':'свой отряд';
    return unitName(s.last_actor)+': '+(item?.name||'боевой предмет')+' → '+target.toLowerCase()+(s.last_damage?' (эффект: '+s.last_damage+')':'')+'.'+blockText(s);
  }
  const capitalInfo=()=>mode==='manual'?(manualCapital||data.capital):data.capital;
  const branchNames={
    empire:['Воины','Стрелки','Маги','Целители','Титаны'],
    mountain_clans:['Воины','Стрелки','Маги','Великаны','Йети'],
    undead_hordes:['Воины','Призраки','Маги','Драконы','Оборотни'],
    legions:['Воины','Гаргульи','Маги','Демоны','Сатиры'],
    elves:['Кентавры','Стрелки','Маги','Поддержка','Грифоны'],
  };
  const buildStatus=['Доступно для строительства','Построено','Ветка или здание заблокированы',
    'Сначала постройте предшественника','Не хватает золота','Завершите ход отдыхом',
    'Строительство недоступно во время боя','Эпизод завершён','Нет у этой фракции'];
  function renderConstruction(snap){
    const info=construction();if(!info)return;
    const s=snap.state;
    $('capital-title').textContent=info.name;
    $('capital-day').textContent='Ход '+s.day+' · золото '+fmt(s.gold);
    $('capital-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Строить можно после боя.':s.built_today?'Постройка этого хода использована. Отдых откроет следующий ход.':'В этом ходу можно построить одно здание.';
    if(!$('faction').options.length){
      for(const faction of info.factions){const o=document.createElement('option');o.value=faction.id;o.textContent=faction.name;$('faction').append(o);}
      $('faction').value=currentMap().faction||'legions';
    }
    $('faction').disabled=busy||mode!=='manual';
    if(capitalKey!==info.faction){
      capitalKey=info.faction;$('building-list').replaceChildren();
      $('building-branch').replaceChildren();
      const all=document.createElement('option');all.value='all';all.textContent='Все здания';$('building-branch').append(all);
      for(const branch of [0,1,2,3,4,-1]){
        const rows=info.buildings.filter(b=>b.branch===branch);if(!rows.length)continue;
        const label=branch===-1?'Общие здания':branchNames[info.faction][branch];
        const option=document.createElement('option');option.value=String(branch);option.textContent=label;$('building-branch').append(option);
        const section=document.createElement('section');section.className='building-group';section.dataset.branch=branch;
        const heading=document.createElement('h3');heading.textContent=label;section.append(heading);
        const grid=document.createElement('div');grid.className='building-grid';section.append(grid);
        for(const row of rows){
          const card=document.createElement('article');card.className='building-card';card.dataset.building=row.action-18;
          const name=document.createElement('h4');name.textContent=row.name;card.append(name);
          const price=document.createElement('span');price.className='building-price';price.textContent=fmt(row.gold)+' золота';card.append(price);
          const unit=document.createElement('p');unit.className='building-unit';unit.textContent=row.unit?'Ветка юнита: '+row.unit:row.name==='Храм'?'Храм столицы':'Магическая служба';card.append(unit);
          const requirements=document.createElement('p');requirements.className='building-requires';requirements.textContent='Требует: '+(row.requires.join(', ')||'нет');card.append(requirements);
          if(row.blocks.length){const excludes=document.createElement('p');excludes.className='building-excludes';excludes.textContent='Закроет: '+row.blocks.join(', ');card.append(excludes);}
          const status=document.createElement('p');status.className='building-status';card.append(status);
          const button=document.createElement('button');button.dataset.action=row.action;button.textContent='Построить';button.setAttribute('aria-label','Построить '+row.name+' за '+row.gold+' золота');button.onclick=()=>sendAction(row.action);card.append(button);
          grid.append(card);
        }
        $('building-list').append(section);
      }
    }
    document.querySelectorAll('[data-building]').forEach(card=>{
      const i=Number(card.dataset.building),code=snap.building_status[i];
      card.className='building-card'+(code===1?' is-built':code===2?' is-blocked':code===0?' is-ready':'');
      card.querySelector('.building-status').textContent=info.buildings[i].unavailable_reason||buildStatus[code];
      const button=card.querySelector('button');button.textContent=code===1?'Построено':'Построить';
      button.disabled=mode!=='manual'||busy||!session||!snap.action_mask[Number(button.dataset.action)];
    });
    document.querySelectorAll('[data-branch]').forEach(group=>group.hidden=$('building-branch').value!=='all'&&group.dataset.branch!==$('building-branch').value);
  }
  function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Смотреть';}
  function roundRect(x,y,w,h,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();}
  function renderCapitalServices(snap){
    const info=capitalInfo(),section=$('capital-services');section.hidden=!info;if(!info)return;
    const s=snap.state,atCapital=snap.territory_quotes?.at_owned_fort??s.position.every((n,i)=>n===info.position[i]);
    const city=territoryInfo()?.cities.find((c,i)=>s.city_owned[i]&&c.position.every((v,j)=>v===s.position[j]));
    const temple=snap.building_status?.[info.temple_action-18]===1;
    $('capital-location').textContent=city?city.name+' · ваш город':'Столица · клетка '+info.position.join(', ');
    $('service-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Услуги доступны после боя.':!atCapital?'Войдите в столицу или захваченный город, чтобы лечить и воскрешать бойцов.':!temple?'Постройте Храм в столице за 300 золота, чтобы открыть лечение и воскрешение.':'Храм открыт; услуги не расходуют очки перемещения и не завершают ход.';
    const cards=[];
    for(let i=0;i<6;i++){
      if(!snap.max_hp[i])continue;
      const q=snap.capital_quotes?.[i];if(!q)continue;
      const card=document.createElement('article');card.className='recovery-card'+(!s.hp[i]?' fallen':'');
      const name=document.createElement('h3');name.textContent=role(i)+' '+(i+1);card.append(name);
      const hp=document.createElement('p');hp.className='recovery-health';hp.textContent=s.hp[i]+' / '+snap.max_hp[i]+' HP'+(!s.hp[i]?' · погиб':'');card.append(hp);
      const prices=document.createElement('p');prices.className='recovery-price';prices.textContent='Лечение: '+q[0]+' золото / HP · воскрешение: '+q[4]+' золота';card.append(prices);
      const action=(!s.hp[i]?info.revive_start:info.heal_start)+i;
      const button=document.createElement('button');button.dataset.action=action;
      button.textContent=!s.hp[i]?'Воскресить · '+q[4]+' золота':q[1]===0?'Здоровье полное':q[2]>0?'Лечить +'+q[2]+' HP · '+q[3]+' золота':'Не хватает золота';
      button.dataset.agentAction=action;button.onclick=()=>sendAction(action);card.append(button);
      cards.push(card);
    }
    $('recovery-list').replaceChildren(...cards);
  }
  function renderPotions(snap){
    const items=potionInfo()||[],s=snap.state;
    $('potion-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Зелья можно использовать после завершения боя.':'Доступны на любой клетке карты, даже без очков перемещения. Выберите зелье и бойца.';
    $('potion-stock').replaceChildren(...items.map((item,i)=>{
      const button=document.createElement('button');button.className='potion-choice';
      button.setAttribute('aria-pressed',String(selectedPotion===i));
      button.textContent=item.name+' · '+item.label+' · '+s.potions[i]+' шт.';
      button.onclick=()=>{selectedPotion=i;render();};return button;
    }));
    const item=items[selectedPotion];if(!item)return;
    $('potion-detail').textContent=item.effect==='revive'?'Воскрешает одного погибшего бойца с 1 HP.':item.effect==='heal'?'Восстанавливает до '+item.amount+' HP одному живому раненому бойцу. Здоровье не превышает максимум.':item.label+' · '+(item.duration==='permanent'?'Постоянный эффект; сохраняется при повышении.':'До следующего отдыха. Один раз на бойца за ход для каждого вида зелья.');
    const cards=[];
    for(let i=0;i<6;i++){
      if(!snap.max_hp[i])continue;
      const card=document.createElement('article');card.className='recovery-card'+(!s.hp[i]?' fallen':'');
      const name=document.createElement('h3');name.textContent=role(i)+' '+(i+1);card.append(name);
      const hp=document.createElement('p');hp.className='recovery-health';hp.textContent=s.hp[i]+' / '+snap.max_hp[i]+' HP'+(!s.hp[i]?' · погиб':'');card.append(hp);
      const button=document.createElement('button'),amount=snap.potion_quotes?.[selectedPotion]?.[i]||0;
      button.dataset.action=item.action_start+i;
      button.textContent=!s.potions[selectedPotion]?'Зелья закончились':item.effect==='revive'?(s.hp[i]?'Боец жив':'Воскресить · 1 бутылка'):!s.hp[i]?'Сначала воскресите бойца':item.effect==='heal'?(!amount?'Здоровье полное':'Лечить +'+amount+' HP · 1 бутылка'):snap.action_mask[item.action_start+i]?'Применить · 1 бутылка':s.in_battle?'Завершите бой':'Эффект уже действует';
      button.dataset.agentAction=item.action_start+i;button.onclick=()=>sendAction(item.action_start+i);card.append(button);cards.push(card);
    }
    $('potion-targets').replaceChildren(...cards);
  }
  function renderMapCommands(snap){
    const attacks=[];
    for(const [action,command] of (snap.map_commands||[]).entries()){
      const [enemy,cost]=command, direction=document.querySelector('.dpad [data-action="'+action+'"]');
      direction.classList.toggle('attack-direction',enemy>=0);
      let badge=direction.querySelector('small');if(!badge){badge=document.createElement('small');direction.append(badge);}badge.textContent=cost;
      direction.title=enemy>=0?'Атаковать отряд № '+(enemy+1)+' · '+cost+' очков':'Перемещение · '+cost+' очк.';
      if(enemy<0)continue;
      const button=document.createElement('button');button.dataset.action=action;
      button.textContent='Атаковать отряд № '+(enemy+1)+' · '+enemyParty(currentMap(),enemy)+' · '+cost+' очков';
      button.dataset.agentAction=action;button.onclick=()=>sendAction(action);attacks.push(button);
    }
    $('map-targets').replaceChildren(...attacks);
    $('map-targets').hidden=!attacks.length;
  }
  const terrainNames=['Равнина','Дорога','Лес','Вода'];
  // Python reference GridVisualizer._display_tile/_display_rect, adapted from
  // (x, y) to the JAX [row, column] grid including its outer wall.
  const displayTile=([row,col],size)=>({x:size-1-row,y:col});
  const displayRect=([row,col],rows,cols,size)=>({x:size-row-rows,y:col,width:rows,height:cols});
  const worldTile=(x,y,size)=>[size-1-x,y];
  const directionAction=screenDirection=>(screenDirection+6)%8;
  const directionLabels=['↑ север','↗ северо-восток','→ восток','↘ юго-восток','↓ юг','↙ юго-запад','← запад','↖ северо-запад'];
  function mapTileAtPointer(event){
    const rect=canvas.getBoundingClientRect(),size=currentMap().size;
    return worldTile(Math.floor((event.clientX-rect.left)/rect.width*size),Math.floor((event.clientY-rect.top)/rect.height*size),size);
  }
  const terrainCache=new WeakMap();
  function terrainGrid(map){
    if(!terrainCache.has(map)){
      const grid=new Uint8Array(map.size*map.size);
      for(const [i,key] of ['road','forest','water'].entries())for(const [r,c] of map.terrain?.[key]||[])grid[r*map.size+c]=i+1;
      terrainCache.set(map,grid);
    }
    return terrainCache.get(map);
  }
  function drawMap(s){
    const map=currentMap();
    const width=canvas.clientWidth;if(!width)return;
    const ratio=Math.min(devicePixelRatio||1,2);canvas.width=width*ratio;canvas.height=width*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);
    const cell=width/map.size;
    for(let r=0;r<map.size;r++)for(let c=0;c<map.size;c++){
      const wall=r===0||c===0||r===map.size-1||c===map.size-1,index=r*map.size+c;
      const seen=s.visited&&(s.visited[index>>>5]&(1<<(index&31)));
      const {x,y}=displayTile([r,c],map.size);
      roundRect(x*cell+1,y*cell+1,cell-2,cell-2,Math.max(2,cell*.12),wall?'#d3decc':current()?.territory_quotes?.territory[r][c]?'#e6cfb5':seen?'#dfecd9':(r+c)%2?'#f3f6ef':'#ecf2e6');
    }
    const terrain=terrainGrid(map);
    for(let r=1;r<map.size-1;r++)for(let c=1;c<map.size-1;c++){
      const kind=terrain[r*map.size+c];if(!kind)continue;
      const tile=displayTile([r,c],map.size),x=tile.x*cell,y=tile.y*cell;
      roundRect(x+1,y+1,cell-2,cell-2,cell*.12,['','#dbc7a6','#b4cca2','#a9d4e2'][kind]);
      ctx.strokeStyle=kind===3?'#6ea6bb':kind===2?'#6c9061':'#af9672';ctx.lineWidth=Math.max(1,cell*.045);
      if(kind===3){for(const dy of [.37,.66]){ctx.beginPath();ctx.moveTo(x+cell*.2,y+cell*dy);ctx.quadraticCurveTo(x+cell*.4,y+cell*(dy-.12),x+cell*.55,y+cell*dy);ctx.quadraticCurveTo(x+cell*.7,y+cell*(dy+.12),x+cell*.83,y+cell*dy);ctx.stroke();}}
      if(kind===2){ctx.fillStyle='#6c9061';ctx.beginPath();ctx.moveTo(x+cell*.22,y+cell*.73);ctx.lineTo(x+cell*.5,y+cell*.22);ctx.lineTo(x+cell*.78,y+cell*.73);ctx.fill();}
      if(kind===1){ctx.setLineDash([cell*.12,cell*.12]);ctx.beginPath();ctx.moveTo(x+cell*.18,y+cell*.5);ctx.lineTo(x+cell*.82,y+cell*.5);ctx.stroke();ctx.setLineDash([]);}
      if(current()?.territory_quotes?.territory[r][c]){ctx.strokeStyle='#b58d64';ctx.strokeRect(x+1,y+1,cell-2,cell-2);}
    }
    for(const position of territoryInfo()?.obstacles||[]){
      const {x:c,y:r}=displayTile(position,map.size);
      roundRect(c*cell+1,r*cell+1,cell-2,cell-2,cell*.14,'#899088');
      ctx.fillStyle='#626c63';ctx.beginPath();ctx.moveTo((c+.16)*cell,(r+.8)*cell);ctx.lineTo((c+.48)*cell,(r+.2)*cell);ctx.lineTo((c+.84)*cell,(r+.8)*cell);ctx.fill();
    }
    for(const capital of territoryInfo()?.enemy_capitals||[]){
      const cells=capital.footprint||[];if(!cells.length)continue;
      const rows=cells.map(p=>p[0]),cols=cells.map(p=>p[1]),top=Math.min(...rows),left=Math.min(...cols);
      const {x:c,y:r,width:w,height:h}=displayRect([top,left],Math.max(...rows)-top+1,Math.max(...cols)-left+1,map.size);
      roundRect(c*cell+1,r*cell+1,w*cell-2,h*cell-2,cell*.2,'#ead0cd');
      ctx.strokeStyle='#a5524f';ctx.lineWidth=1.5;ctx.strokeRect(c*cell+2,r*cell+2,w*cell-4,h*cell-4);
      ctx.fillStyle='#7d2f2c';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font='650 '+Math.round(cell*.7)+'px system-ui';ctx.fillText('Столица',(c+w/2)*cell,(r+h/2-.9)*cell);
      ctx.font=Math.round(cell*.5)+'px system-ui';ctx.fillText('врага',(c+w/2)*cell,(r+h/2-.2)*cell);
    }
    for(const [i,mine] of (territoryInfo()?.mines||[]).entries()){
      const {x:c,y:r}=displayTile(mine.position,map.size),marker=manaMarkers[mine.kind];
      if(marker){
        roundRect(c*cell+1,r*cell+1,cell-2,cell-2,cell*.12,marker.color);
        ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.strokeRect(c*cell+1.5,r*cell+1.5,cell-3,cell-3);
      }else{
        ctx.fillStyle='#d1a737';ctx.beginPath();ctx.moveTo((c+.5)*cell,(r+.07)*cell);ctx.lineTo((c+.91)*cell,(r+.5)*cell);ctx.lineTo((c+.5)*cell,(r+.93)*cell);ctx.lineTo((c+.09)*cell,(r+.5)*cell);ctx.closePath();ctx.fill();
      }
      ctx.fillStyle='#fff';ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='800 '+Math.round(cell*(marker ? .72 : .48))+'px system-ui';ctx.fillText(marker?.letter||'З',(c+.5)*cell,(r+.52)*cell);
      if(current()?.territory_quotes?.mine_owned[i]){ctx.strokeStyle='#795a37';ctx.lineWidth=1.5;ctx.strokeRect(c*cell+1,r*cell+1,cell-2,cell-2);}
    }
    for(const site of siteInfo()?.sites||[]){
      const style={merchant:['#b58242','#e3cba5','#775129','Торг'],trainer:['#688da4','#c3d7df','#3f637b','Тренер'],mercenary:['#8e7b59','#e5dcc7','#665238','Наём'],spell_shop:['#7b68a8','#ddd5ef','#4c3d78','Магия']}[site.kind];
      const [color,fill,ink,label]=style;
      ctx.fillStyle=color+'29';
      for(const position of site.interaction_tiles){const {x,y}=displayTile(position,map.size);ctx.fillRect(x*cell+1,y*cell+1,cell-2,cell-2);}
      const bounds=displayRect(site.position,3,3,map.size),x=bounds.x*cell,y=bounds.y*cell;
      roundRect(x+1,y+1,cell*3-2,cell*3-2,cell*.2,fill);
      ctx.strokeStyle=color;ctx.lineWidth=1;ctx.strokeRect(x+2,y+2,cell*3-4,cell*3-4);
      ctx.fillStyle=ink;ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font='650 '+Math.round(cell*.7)+'px system-ui';ctx.fillText(label,x+cell*1.5,y+cell*1.35);
      ctx.font=Math.round(cell*.52)+'px system-ui';ctx.fillText('3 × 3',x+cell*1.5,y+cell*2);
      ctx.fillStyle=color;ctx.fillRect(x+cell*.2,y+cell*2.25,cell*.55,cell*.55);
    }
    for(const [i,ruin] of (ruinInfo()?.ruins||[]).entries()){
      const cleared=s.ruin_looted?.[i],bounds=displayRect(ruin.position,3,3,map.size),x=bounds.x*cell,y=bounds.y*cell;
      if(!cleared){ctx.fillStyle='#97795324';for(const position of ruin.interaction_tiles){const tile=displayTile(position,map.size);ctx.fillRect(tile.x*cell+1,tile.y*cell+1,cell-2,cell-2);}}
      roundRect(x+1,y+1,cell*3-2,cell*3-2,cell*.16,cleared?'#c6c9c3':'#d6c9b4');
      ctx.strokeStyle=cleared?'#8a9086':'#9a7b56';ctx.lineWidth=1;ctx.strokeRect(x+2,y+2,cell*3-4,cell*3-4);
      ctx.fillStyle=cleared?'#697164':'#725738';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font='650 '+Math.round(cell*.7)+'px system-ui';ctx.fillText('Руины',x+cell*1.5,y+cell*1.2);
      ctx.font=Math.round(cell*.46)+'px system-ui';ctx.fillText(cleared?'Разграблены':'Стража: '+map.enemy_units[ruin.enemy],x+cell*1.5,y+cell*1.92);
      if(!cleared)ctx.fillRect(x+cell*.2,y+cell*2.25,cell*.55,cell*.55);
    }
    // Cities occupy one tile. Keep the larger label transparent and below
    // neighboring units/chests so the annotation cannot hide a selectable target.
    (territoryInfo()?.cities||[]).forEach((city,i)=>{
      const tile=displayTile(city.position,map.size),x=(tile.x+.5)*cell,y=(tile.y+.5)*cell;
      const owned=s.city_owned[i],size=cell*3;
      ctx.save();
      roundRect(x-cell/2+1,y-cell/2+1,cell-2,cell-2,cell*.12,owned?'#eadcbb':'#e2d8ec');
      ctx.strokeStyle=owned?'#a48649':'#9676af';ctx.lineWidth=1;
      ctx.strokeRect(x-size/2+2,y-size/2+2,size-4,size-4);
      ctx.fillStyle=owned?'#745522':'#654777';
      ctx.strokeStyle='#f3f6ef';ctx.lineWidth=Math.max(2,cell*.1);ctx.lineJoin='round';
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font='650 '+Math.round(cell*.7)+'px system-ui';
      ctx.strokeText('Город',x,y-cell*.35);ctx.fillText('Город',x,y-cell*.35);
      const label=(owned?'ваш · ':'')+'ур. '+s.city_levels[i];
      ctx.font=Math.round(cell*.5)+'px system-ui';
      ctx.strokeText(label,x,y+cell*.35);ctx.fillText(label,x,y+cell*.35);
      ctx.restore();
    });
    const token=(p,n)=>{const {x,y}=displayTile(p,map.size);roundRect(x*cell+cell*.09,y*cell+cell*.09,cell*.82,cell*.82,cell*.2,'#d5c0e8');ctx.fillStyle='#654777';ctx.font='650 '+Math.round(cell*.55)+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n,(x+.5)*cell,(y+.52)*cell);};
    (s.enemy_positions||map.opponent_positions).forEach((p,i)=>{if(s.alive[i]&&!(ruinInfo()?.ruins||[]).some(r=>r.enemy===i)&&!(territoryInfo()?.cities||[]).some(c=>c.position.every((v,j)=>v===p[j])))token(p,map.enemy_units[i]);});
    (map.chests||[]).forEach((chest,i)=>{
      if(!s.chest_alive?.[i])return;
      const tile=displayTile(chest.position,map.size),x=tile.x*cell,y=tile.y*cell;
      roundRect(x+cell*.12,y+cell*.24,cell*.76,cell*.6,cell*.1,'#9b6230');
      roundRect(x+cell*.12,y+cell*.17,cell*.76,cell*.28,cell*.1,'#d8ad64');
      ctx.fillStyle='#f5dfa1';ctx.fillRect(x+cell*.44,y+cell*.35,cell*.12,cell*.24);
    });
    const capital=capitalInfo()?.position;
    if(capital){
      const {x,y}=displayTile(capital,map.size);
      roundRect(x*cell+1,y*cell+1,cell-2,cell-2,cell*.15,'#dfb963');
      ctx.fillStyle='#604515';ctx.font='650 '+Math.round(cell*.62)+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('♜',(x+.5)*cell,(y+.52)*cell);
      ctx.strokeStyle='#aa7623';ctx.lineWidth=2;ctx.strokeRect(x*cell+1,y*cell+1,cell-2,cell-2);
    }
    if(map.campaign_objective==='reach_position'){
      const {x,y}=displayTile(map.objective_position,map.size);
      ctx.save();ctx.strokeStyle='#197d60';ctx.fillStyle='#b4f0ce';ctx.lineWidth=2;
      const radius=map.objective_radius||0;
      if(radius){
        ctx.globalAlpha=.25;
        for(let row=Math.max(1,map.objective_position[0]-radius);row<=Math.min(map.size-2,map.objective_position[0]+radius);row++){
          for(let col=Math.max(1,map.objective_position[1]-radius);col<=Math.min(map.size-2,map.objective_position[1]+radius);col++){
            const tile=displayTile([row,col],map.size);ctx.fillRect(tile.x*cell,tile.y*cell,cell,cell);
          }
        }
        ctx.globalAlpha=1;
      }
      ctx.fillRect(x*cell,y*cell,cell,cell);ctx.strokeRect(x*cell,y*cell,cell,cell);
      ctx.fillStyle='#07523d';ctx.font='800 '+Math.max(12,Math.round(cell*.85))+'px system-ui';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('⚑',(x+.5)*cell,(y+.5)*cell);ctx.restore();
    }
    const agent=displayTile(s.position,map.size),x=(agent.x+.5)*cell,y=(agent.y+.5)*cell,radius=Math.max(16,cell*.85);
    ctx.save();
    ctx.shadowColor='#ff182899';ctx.shadowBlur=12;
    ctx.fillStyle='#ff1828';ctx.strokeStyle='#fff';ctx.lineWidth=3;
    ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.shadowBlur=0;ctx.fillStyle='#fff';
    ctx.font='900 '+Math.round(radius*1.4)+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText('А',x,y+1);
    ctx.restore();
  }
  const slots=[3,0,4,1,5,2,6,9,7,10,8,11];
  for(const i of slots){
    const button=document.createElement('button');button.className='unit'+(i>=6?' foe':'');button.dataset.slot=i;
    button.innerHTML='<span class="unit-name"></span><span class="archer-icon" aria-hidden="true">➶</span><span class="health"></span><span class="unit-stats"></span><span class="unit-xp"></span><span class="health-bar"><span class="health-fill"></span></span><span class="status"></span>';
    button.dataset.agentAction=targetAction(i);button.onclick=()=>sendAction(targetAction(i));$(i<6?'allies':'enemies').append(button);
  }
  function experienceText(snap,i){
    const xp=snap.unit_experience?.[i];
    return xp?'Ур. '+xp[0]+' · опыт '+xp[3]+' / '+xp[2]+'\nЗа победу: '+xp[1]+' опыта':'';
  }
  function upgradeText(snap,i){
    const unit=profile(i),choices=unit?.upgrades;
    if(unit?.hero){
      const level=snap.unit_experience?.[i]?.[0]||1,skills=unit.hero_skills?.length?unit.hero_skills:unit.level_bonuses||[];
      const earned=skills.filter(b=>b.level<=level).map(b=>b.name);
      const next=skills.find(b=>b.level>level&&(!unit.max_level||b.level<=unit.max_level));
      return (unit.max_level&&level>=unit.max_level?'Герой: достигнут предел сценария ('+unit.max_level+').':'Герой: повышение без здания.'+(unit.max_level?' Предел сценария: '+unit.max_level+'.':''))+(earned.length?(unit.hero_skills?.length?' Навыки: ':' Бонусы: ')+earned.join('; ')+'.':'')+
        (next?' На уровне '+next.level+': '+next.name+'.':'');
    }
    if(!choices)return '';
    if(unit.scenario_dynamic)return 'Дальнейшие формы запрещены сценарием. Следующий уровень: рост характеристик без смены формы.';
    if(!choices.length)return 'Следующий уровень: рост характеристик без здания.';
    return choices.map(u=>{
      if(!u.supported)return u.name+': недоступно — '+u.reason.toLowerCase()+'.';
      const b=construction()?.buildings.find(b=>b.name===u.building);
      const bit=b?1<<(b.action-18):0;
      const status=!b?'нужна столица родной фракции':snap.state.buildings&bit?'построено':snap.state.blocked_buildings&bit?'заблокировано':'не построено';
      return u.name+' ← '+u.building+' ('+status+').';
    }).join(' ');
  }
  function renderExperience(snap){
    const cards=[];
    for(let i=0;i<6;i++){
      if(!snap.max_hp[i])continue;
      const card=document.createElement('article');card.className='experience-card';
      const heading=document.createElement('b');heading.textContent=role(i)+' '+(i+1);card.append(heading);
      const hp=document.createElement('span');hp.textContent=snap.state.hp[i]+' / '+snap.max_hp[i]+' HP';card.append(hp);
      const xp=document.createElement('p');xp.textContent=experienceText(snap,i);card.append(xp);
      const next=document.createElement('small');next.textContent=upgradeText(snap,i);card.append(next);
      const recruitment=recruitmentInfo();
      if(recruitment&&!profile(i)?.hero){
        const dismiss=document.createElement('button');dismiss.textContent='Уволить без возврата золота';
        dismiss.dataset.action=recruitment.dismiss_start+i;dismiss.onclick=()=>sendAction(recruitment.dismiss_start+i);
        card.append(dismiss);
      }
      cards.push(card);
    }
    $('hero-experience').replaceChildren(...cards);
  }
  // Slots 0-2: front row, 3-5: back row; slot s and s+3 form a column.
  const swapCell=i=>(i<3?'Перед ':'Тыл ')+(i%3+1);
  const swapUnit=i=>current()?.state.unit_ids?.[i]?role(i):i>=3&&profile(i-3)?.size===2?'('+role(i-3)+')':'пусто';
  function renderSwaps(snap){
    const info=swapInfo(),s=snap.state;$('unit-swaps').hidden=!info;if(!info)return;
    $('swap-status').textContent=s.done?'Эпизод завершён.':s.in_battle?'Перестановка доступна после боя.':'Сегодня осталось '+(info.limit-s.unit_swaps_today)+' из '+info.limit+'. Каждая пара — один раз за ход. Очки перемещения не тратятся; штраф −'+fmt(info.penalty)+' за перестановку. Большой боец переносит всю свою линию.';
    $('swap-formation').replaceChildren(...[0,1,2].flatMap(c=>[c+3,c]).map(i=>{
      const cell=document.createElement('span');cell.className=s.unit_ids?.[i]?'':'empty';
      cell.textContent=swapCell(i)+': '+swapUnit(i)+(s.unit_ids?.[i]?' · '+s.hp[i]+' HP':'');return cell;
    }));
    $('swap-actions').replaceChildren(...info.pairs.map(([a,b],k)=>{
      const button=document.createElement('button');button.dataset.action=info.start+k;
      button.textContent=swapCell(a)+' ⇄ '+swapCell(b);button.title=swapUnit(a)+' ⇄ '+swapUnit(b);
      button.onclick=()=>sendAction(info.start+k);return button;
    }));
  }
  function swapText(s){
    const info=swapInfo(),[a,b]=info?.pairs?.[s.last_swap]||[];if(a==null)return null;
    const big=[a,b].some(i=>profile(i%3)?.size===2);
    return 'Перестановка: '+swapCell(a)+' ⇄ '+swapCell(b)+(big?', большой боец перенёс всю линию':'')+'. Осталось сегодня: '+(info.limit-s.unit_swaps_today)+'. Штраф −'+fmt(info.penalty)+'.';
  }
  function xpSummary(snap){
    const parts=[];
    for(let i=0;i<12;i++){
      if(snap.state.last_promoted&(1<<i))parts.push(unitName(i)+' → уровень '+snap.unit_experience[i][0]+', полное лечение, опыт сброшен');
      else if(snap.state.last_xp?.[i])parts.push(unitName(i)+': опыт '+snap.unit_experience[i][3]+' / '+snap.unit_experience[i][2]);
    }
    return parts.length?' '+parts.join('; ')+'.':'';
  }
  function eventText(snapshot){
    const ticks=[...['poison','burn','water'].flatMap(kind=>(snapshot.state['last_'+kind+'_damage']||[]).map((amount,i)=>amount?unitName(i)+': '+({poison:'яд',burn:'огонь',water:'вода'}[kind])+' −'+amount+' HP':'').filter(Boolean))];
    const sale=snapshot.state.last_sale_gold>0?' Драгоценности проданы автоматически: '+snapshot.state.last_sold_count+' шт., +'+fmt(snapshot.state.last_sale_gold)+' золота.':'';
    return (actionEventText(snapshot)||'')+(ticks.length?' '+ticks.join('; ')+'.':'')+sale;
  }
  // A summon battle ends before any world step clears the cast record.
  const summonEnded=s=>castInfo()?.spells[s.last_cast_spell]?.kind==='summon_battle';
  function castText(s){
    const spell=castInfo()?.spells[s.last_cast_spell];if(!spell)return '';
    const stack='отряд № '+(s.last_spell_target+1),item=s.last_cast_item>=0?castInfo()?.items?.[s.last_cast_item]:null;
    const name=(item?(item.category==='scroll'?'Свиток':'Посох')+' «'+item.name+'»: з':'З')+'аклинание «'+spell.name+'»';
    if(s.last_event===50)return name+': '+spell.unit_name+' вступает в отдельный бой, '+stack+'.';
    if(s.last_event===49)return name+' уничтожило '+stack+' (−'+s.last_spell_amount+' HP). Награды за победу нет.';
    switch(spell.kind){
      case 'damage':return name+': '+stack+' теряет '+s.last_spell_amount+' HP. Раны сохраняются, враг восстанавливает их каждый ход.';
      case 'debuff':return name+' ослабило '+stack+' до следующего хода.';
      case 'heal':return name+': восстановлено '+s.last_spell_amount+' HP.';
      case 'moves':return name+': +'+s.last_spell_amount+' очков перемещения.';
    }
    return name+' действует на отряд героя до следующего хода.';
  }
  function actionEventText(snapshot){
    const s=snapshot.state,a=unitName(s.last_actor),t=unitName(s.last_target);
    switch(s.last_event){
      case 47:return 'Изучено заклинание: '+spellInfo()?.spells[s.last_researched_spell]?.name+'. Следующее исследование доступно завтра.';
      case 48:case 49:case 50:return castText(s);
      case 55:return swapText(s);
      case 56:return 'Превышен лимит '+(snapshot.battle_max_rounds??currentMap().battle_max_rounds??75)+' раундов: весь отряд отступает. Штраф −'+fmt(snapshot.battle_round_limit_penalty??currentMap().battle_round_limit_penalty??1)+'.';
      case 54:return 'Нанят маг для чтения свитков · −'+fmt(castInfo()?.hire_cost??500)+' золота. Свитки и посохи доступны до конца игры.';
      case 43:return 'Город захвачен: '+(territoryInfo()?.cities||[]).filter((c,i)=>s.last_captured_cities[i]).map(c=>c.name).join(', ')+'. Доступны найм, храм и рост земли.';
      case 44:return 'Город улучшен: '+territoryInfo()?.cities[s.last_upgraded_city]?.name+' · −'+fmt(s.last_service_cost)+' золота.';
      case 45:return 'Путь закрыт препятствием.';
      case 52:return battleItemText(s);
      case 53:return 'Боевой слот героя: '+(battleItemInfo()?.items?.[s.last_battle_item]?.name||'предмет')+'. Следующий предмет можно положить после отдыха.';
      case 41:return 'Боец нанят: −'+fmt(s.last_service_cost)+' золота.';
      case 42:return 'Боец уволен. Место и лидерство освобождены.';
      case 1:return 'Атака отряда № '+(s.enemy+1)+'. Отряд остаётся на своей клетке; очков перемещения: '+s.movement_points+'.';
      case 2:return isMage(s.last_actor)?a+': заклинание по всем противникам (суммарный урон '+s.last_damage+').'+blockText(s):a+(isWarrior(s.last_actor)?': удар мечом по ':': попадание в ')+t.toLowerCase()+' (−'+s.last_damage+').'+blockText(s);
      case 3:return a+': промах.';
      case 22:return a+': превращение в Дух Фенрира. Доля здоровья сохранена.';
      case 33:return a+': принял облик выбранного бойца.';
      case 34:return a+': призвал подкрепление. Очередь показывает готовность новых бойцов.';
      case 23:return a+(s.last_target<0?': массовое превращение в бесов.':': '+t+' превращён в беса.')+' Здоровье сохранено.'+blockText(s);
      case 25:return a+(s.last_target<0?': массовый паралич.':': паралич — '+t+'.')+blockText(s);
      case 26:return a+': пропускает ход из-за паралича.';
      case 27:return a+(s.last_target<0?': массовый страх.':': страх — '+t+'.')+blockText(s);
      case 28:return 'Противник побеждён. Лекари получают по одному действию перед выдачей опыта.';
      case 24:return a+': понижение формы '+t+' (урон '+s.last_damage+'). Доля HP пересчитана.';
      case 4:return a+' встал в защиту.';
      case 5:return a+(s.post_victory?' пропускает заключительное лечение.':' ждёт конца раунда.');
      case 6:return a+' готовится отступить.';
      case 7:return a+' покинул бой.';
      case 8:return (summonEnded(s)?'Призванный боец победил: отряд № '+(s.enemy+1)+' уничтожен. Отряд героя вернулся без изменений.':'Победа! Ранения и потери сохранены.'+xpSummary(snapshot))+((s.last_ruin??-1)>=0?' Руины разграблены. Получено: '+lootText(ruinInfo().ruins[s.last_ruin])+'.':'');
      case 9:return 'Ваш отряд погиб. Игра завершена.';
      case 10:return summonEnded(s)?'Бой призванного бойца окончен без победы. Отряд № '+(s.enemy+1)+' сохраняет раны до регенерации.':'Отступление завершено. Ранения и потери сохранены.';
      case 11:return 'Достигнут общий лимит переходов. Эпизод завершён.';
      case 12:{const b=construction()?.buildings[s.last_building];return b?'Построено: '+b.name+' (−'+b.gold+' золота).':null;}
      case 13:return 'Отдых. Начался ход '+s.day+', +'+(snapshot.territory_quotes?.income[0]??turnRules().income)+' золота; очки перемещения восстановлены, живые бойцы получили регенерацию. Штраф: '+fmt(Math.max(0,-snapshot.reward))+'.';
      case 14:return 'Атака заблокирована иммунитетом.'+blockText(s);
      case 32:return a+': атака и замедление инициативы противников (урон '+s.last_damage+').';
      case 31:return a+': дополнительный ход — '+t+'.';
      case 30:return a+': воскрешение — '+t+' (+'+s.last_damage+' HP). Повторное воскрешение в этом бою недоступно.';
      case 29:return a+': усиление — '+t+'. Сила атаки: '+s.last_damage+'.';
      case 16:return a+(s.last_target<0?': лечение союзников':': лечение — '+t.toLowerCase())+' (+'+s.last_damage+' HP).';
      case 17:return 'Храм: '+t.toLowerCase()+' восстановил '+s.last_damage+' HP за '+s.last_service_cost+' золота.';
      case 18:return 'Храм: '+t.toLowerCase()+' воскрешён с 1 HP за '+s.last_service_cost+' золота.';
      case 19:return potionInfo()?.[s.last_potion]?.name+': '+t.toLowerCase()+' восстановил '+s.last_damage+' HP. Осталось: '+s.potions[s.last_potion]+'.';
      case 21:return 'Сундук: '+[...s.last_loot.map((n,i)=>n?(potionInfo()?.[i]?.name+' × '+n):''),...(s.last_item_loot||[]).map((n,i)=>n?(equipmentInfo()?.items?.[i]?.name+' × '+n):'')].filter(Boolean).join(', ')+'. Предметы получены, экипировка обновлена.';
      case 20:return 'Зелье воскрешения: '+t.toLowerCase()+' вернулся с 1 HP. Осталось: '+s.potions[s.last_potion]+'.';
      case 38:return 'Куплено: '+siteInfo()?.buy?.[s.last_trade_item]?.name+' · −'+fmt(s.last_service_cost)+' золота.';
      case 51:return 'Куплено заклинание: '+siteInfo()?.spells?.[s.last_trade_item]?.name+' · −'+fmt(s.last_service_cost)+' золота. Оно сразу изучено.';
      case 40:return 'Тренер: '+t+' получил '+s.last_xp[s.last_target]+' XP за '+fmt(s.last_service_cost)+' золота.';
      case 37:return potionInfo()?.[s.last_potion]?.name+': '+t.toLowerCase()+' получил усиление. Осталось: '+s.potions[s.last_potion]+'.';
      case 15:return 'Атака поглощена защитой.'+blockText(s);
      default:return null;
    }
  }
  function render(){
    const snap=current();if(!snap)return;const s=snap.state,mask=snap.action_mask,map=currentMap();
    $('version').textContent=map.name;
    $('map-name').textContent=mapTitle(map);
    $('map-panel').hidden=s.in_battle||capitalOpen||potionsOpen||spellsOpen;$('battle-panel').hidden=!s.in_battle||capitalOpen||potionsOpen||spellsOpen;
    $('capital-panel').hidden=!capitalOpen;$('potions-panel').hidden=!potionsOpen;$('spells-panel').hidden=!spellsOpen;
    $('spells-view').setAttribute('aria-pressed',String(spellsOpen));
    $('world-view').setAttribute('aria-pressed',String(!capitalOpen&&!potionsOpen&&!spellsOpen));$('capital-view').setAttribute('aria-pressed',String(capitalOpen));$('potions-view').setAttribute('aria-pressed',String(potionsOpen));
    renderConstruction(snap);
    renderCapitalServices(snap);
    renderPotions(snap);
    renderEquipment(snap);
    renderBattleItems(snap);
    renderSites(snap);
    renderRecruitment(snap);
    renderTerritory(snap);
    renderRuins(snap);
    renderSpells(snap);
    renderMapCommands(snap);
    $('phase-label').textContent=s.in_battle?'Бой · отряд № '+(s.enemy+1):'Карта '+map.size+' × '+map.size;
    // Capital guards and optional_enemies are not victory goals; count only the stacks that must fall.
    // With objective cities (reference "cities" goal) count those cities instead.
    const guards=new Set([...(territoryInfo()?.enemy_capitals||[]).flatMap(c=>c.guards||[c.guard]),...(map.optional_enemies||[])]);
    const targetEnemy=Number.isInteger(map.objective_enemy)?map.objective_enemy:null;
    const targetName=targetEnemy===null?'':partyText(combatInfo()?.enemies[targetEnemy]);
    const goalCities=targetEnemy===null?(territoryInfo()?.cities||[]).map((c,i)=>c.objective?i:-1).filter(i=>i>=0):[];
    const remainingEnemy=i=>s.alive[i]||s.scenario_spawned?.[i]===false;
    const waves=map.campaign_objective==='waves';
    $('remaining').textContent=targetEnemy!==null?Number(remainingEnemy(targetEnemy)):waves?(map.scheduled_enemies||[]).filter(e=>remainingEnemy(e.enemy)).length:goalCities.length?goalCities.filter(i=>!s.city_owned?.[i]).length:s.alive.filter((a,i)=>a&&!guards.has(i)).length;
    $('remaining-label').textContent=targetEnemy!==null?'цель: '+targetName+' · отряд № '+(targetEnemy+1):waves?'армий волн осталось, включая ещё не появившиеся':goalCities.length?'целевых городов осталось ('+goalCities.map(i=>territoryInfo().cities[i].name).join(', ')+')':'отрядов осталось'+((map.optional_enemies||[]).length?' (без необязательных)':'');
    if(map.campaign_objective==='reach_position'){
      $('remaining').textContent=Number(!s.won);
      $('remaining-label').textContent='цель: привести живого героя '+(map.objective_radius?'в зону ⚑ радиуса '+map.objective_radius:'к ⚑')+' · строка '+map.objective_position[0]+', столбец '+map.objective_position[1];
      if(map.objective_after_event&&map.scenario_events_enabled!==false){
        const event=map.scenario_events.findIndex(e=>e.id===map.objective_after_event);
        const responded=s.script_events_fired?.[event]&&s.script_objective_ready;
        const pending=(s.pending_encounters||[]).some((v,i)=>v&&s.alive[i]);
        $('remaining-label').textContent+=!responded?' · нужно пережить ответный ход засады':pending||s.in_battle?' · завершите начатые бои':' · ответный ход засады состоялся';
      }
    }
    $('chest-status').hidden=!(map.chests||[]).length;
    $('chest-status').textContent=(s.last_event===21?eventText(snap)+' ':'')+'Сундуков осталось: '+(s.chest_alive||[]).filter(Boolean).length+' / '+(map.chests||[]).length+'. Подойдите на соседнюю клетку, включая диагональ: всё содержимое попадёт в инвентарь.';
    $('wins').textContent=s.alive.filter((v,i)=>!v&&s.scenario_spawned?.[i]!==false).length;
    if(map.scenario_events?.length)$('wins').textContent=(s.reference_defeated||[]).filter(Boolean).length;
    $('steps').textContent=fmt(s.step_count);$('reward').textContent=fmt(snap.total_reward);
    $('gold').textContent=fmt(s.gold);
    const turns=turnRules(),points=s.movement_points;
    $('turn-summary').textContent='Ход '+s.day+' · '+points+' / '+movementCap(snap)+' очков перемещения';
    if(map.scenario_events?.length)$('turn-summary').textContent+=map.scenario_events_enabled===false?' · события отключены':' · событий: '+(s.script_events_fired||[]).filter(Boolean).length+' / '+map.scenario_events.length;
    $('movement-summary').textContent=points+' / '+movementCap(snap)+' очков движения';
    const travel=snap.travel_costs||[2,1,4,6];
    $('terrain-summary').textContent=terrainNames.map((name,i)=>name+': '+travel[i]).join(' · ')+' очков. '+(travel.every(n=>n===2)?'Полёт: любая местность стоит 2 очка. ':travel[0]===4?'Герой погиб: стоимость движения удвоена. ':'')+'Последний шаг расходует остаток, если очков меньше цены клетки.';
    $('rest').dataset.action=turns.rest_action;
    $('rest-hint').textContent=s.in_battle?'Отдых доступен после завершения боя.':snap.rest_penalty>0?'Штраф за оставшиеся очки: −'+fmt(snap.rest_penalty)+'. Следующий ход: +'+(snap.territory_quotes?.income[0]??turns.income)+' золота при текущих владениях и полный запас очков.':'Все очки использованы: штраф за неиспользованное движение равен нулю. Следующий ход: +'+(snap.territory_quotes?.income[0]??turns.income)+' золота при текущих владениях и полный запас очков.';
    if(!s.in_battle)$('rest-hint').textContent+=' Живым бойцам: +'+(snap.territory_quotes?[...new Set(snap.territory_quotes.regeneration.filter((v,i)=>snap.max_hp[i]&&s.hp[i]>0))].join('–'):(turns.regeneration_percent+10*(equipmentInfo()?.items?.[s.equipped?.[2]]?.regeneration||0)))+'% максимального HP с округлением вверх (до максимума).';
    const heroes=Array.from({length:6},(_,i)=>profile(i));
    renderExperience(snap);
    renderSwaps(snap);
    $('party-health').textContent=partyText(heroes.filter((u,i)=>u&&s.hp[i]>0))+' · '+s.hp.slice(0,6).reduce((a,b)=>a+b,0)+' здоровья';
    $('party-title').textContent=partyText(heroes)+'.';
    $('mage-rule').hidden=!heroes.some(u=>u?.role==='area');
    $('warrior-rule').hidden=!heroes.some(u=>u?.role==='melee');
    $('enemy-warrior-rule').hidden=!combatInfo()?.enemies.some(row=>row.some(u=>u?.role==='melee'));
    const basic=true;
    $('basic-combat-rule').hidden=false;
    $('mage-rule').innerHTML='<b>Массовая атака.</b> Маг атакует всех живых противников. Попадание, прибавка к урону, иммунитет и защита проверяются отдельно для каждой цели.';
    $('battle-hint').textContent='Источник атаки, иммунитеты и оставшиеся защиты указаны на карточках. Точность 80% соответствует фактическому шансу попадания 92,2%. Броня и действие защиты уменьшают прошедший урон. Титан занимает обе клетки линии. Служка лечит одного живого союзника на 20 HP.';
    $('fenrir').hidden=!s.in_battle||profile(s.actor)?.unit_type!=='Wolf Lord';
    if(s.in_battle&&s.round===0)$('battle-hint').textContent='Подготовка двойников перед первым раундом: выберите доступного бойца любой стороны или пропустите копирование кнопкой защиты.';
    $('attack-hint').textContent=profile(s.actor)?.unit_type==='Succub'?'Массовое превращение в бесов без урона: выберите любого доступного живого противника.':profile(s.actor)?.unit_type==='Witch'?'Превращение: выберите живого противника.':profile(s.actor)?.unit_type==='Alchemist'?'Дополнительный ход: выберите живого союзника, кроме алхимика и отступающего.':profile(s.actor)?.unit_type==='Patriach'?'Выберите союзника: лечение живого или однократное воскрешение погибшего с 50% HP.':isPowerSupport(s.actor)?(['Dwarfdruid','Arhidruid'].includes(profile(s.actor)?.unit_type)?'Выберите союзника для усиления и очищения; себя — только для очищения.':'Усиление: выберите другого живого союзника.'):isMassHealer(s.actor)?'Массовое лечение всех живых союзников, кроме самого лекаря. Выберите любую доступную карточку.':isHealer(s.actor)?'Лечение: выберите живого союзника.':isMage(s.actor)?'Массовая атака: выберите любого живого врага.':isWarrior(s.actor)?(mask.slice(8,14).some(Boolean)?'Ход воина: выберите подсвеченного врага для удара мечом.':'Воин не достаёт до врагов. Можно защищаться, ждать или отступить.'):'Нажмите на живого противника, чтобы выстрелить.';
    if(isCopier(s.actor))$('attack-hint').textContent='Копирование: выберите доступного бойца любой стороны. Большие юниты и стражи недоступны.';
    if(isSummoner(s.actor))$('attack-hint').textContent='Призыв: выберите свободную клетку своего отряда. Большое существо занимает переднюю и заднюю клетки линии.';
    const winText=map.campaign_objective==='reach_position'?'Победа! Герой достиг цели ⚑.'
      :targetEnemy!==null?'Победа! Цель повержена: '+targetName+'.':goalCities.length?'Победа! Целевые города захвачены.':'Победа! Карта очищена.';
    const message=s.won?winText:s.lost?'Ваш отряд погиб. Начните новую игру.':s.done?'Достигнут лимит. Начните новую игру.':s.in_battle?'Выбирайте цели и берегите свой отряд.':!s.movement_points?'Очки перемещения закончились. Можно использовать зелья, строить или отдохнуть.':'Шаг на клетку врага — атака. Рядом с отрядом можно пройти без боя.';
    $('message').textContent=message;
    $('map-controls').hidden=s.in_battle||capitalOpen||potionsOpen||spellsOpen;$('battle-controls').hidden=!s.in_battle||capitalOpen||potionsOpen||spellsOpen;
    $('continue').hidden=!s.in_battle||!mask[17];
    $('wave-continue').hidden=s.in_battle||!mask[17]||s.done;
    document.querySelectorAll('[data-action]').forEach(b=>b.disabled=mode!=='manual'||busy||s.done||!session||!mask[Number(b.dataset.action)]);
    $('reset').disabled=busy;
    if(s.in_battle){
      $('battle-heading').textContent='Отряд № '+(s.enemy+1)+' · '+enemyParty(map,s.enemy);
      $('round').textContent='Раунд '+s.round+(snap.forced_retreat?' · отступление':'');
      $('turn-message').textContent=s.done?message:s.actor<6&&(!s.retreating[s.actor]||s.post_victory)?'Ваш ход: '+role(s.actor).toLowerCase()+' '+(s.actor+1):s.actor>=6?'Ход противника: '+role(s.actor).toLowerCase()+' '+(s.actor-5):unitName(s.actor)+' завершает отступление';
      if(s.post_victory&&!s.done)$('turn-message').textContent='Лечение после боя: '+unitName(s.actor)+(s.actor<6?' · выберите союзника или пропустите действие кнопкой ожидания.':'');
      if(s.second_strike&&!s.done)$('turn-message').textContent+=' · второй удар';
      document.querySelectorAll('[data-slot]').forEach(b=>{
        const i=Number(b.dataset.slot),max=snap.max_hp[i],hp=s.hp[i];
        const local=i%6,big=profile(i)?.size===2&&(s.hp[i]>0||!s.hp[i+3]),covered=local>=3&&profile(i-3)?.size===2&&s.hp[i-3]>0;
        b.hidden=covered;
        b.style.gridRow=String(local%3+1);
        b.style.gridColumn=big?'1 / span 2':String(i<6?(local<3?2:1):(local<3?1:2));
        const action=targetAction(i);
        const targetTurn=(isCopier(s.actor)?true:(isHealer(s.actor)||isSummoner(s.actor))?i<6:i>=6)&&s.actor<6&&(!s.retreating[s.actor]||s.post_victory)&&!s.done;
        const unreachable=targetTurn&&hp>0&&!s.escaped[i]&&!mask[action];
        b.className='unit'+(i>=6?' foe':'')+(big?' large':'')+(isHealer(i)?' healer':'')+(isMage(i)?' mage':'')+(isWarrior(i)?' warrior':'')+(i===s.actor&&!s.done?' active':'')+(!max?' empty':!hp?' dead':s.escaped[i]?' escaped':'')+(unreachable?' unreachable':targetTurn&&mask[action]?' reachable':'');
        b.querySelector('.unit-name').textContent=max?role(i)+' '+(i%6+1)+(big?' · 2 клетки':''):'Пусто';
        b.querySelector('.archer-icon').textContent=isHealer(i)?'✚':isMage(i)?'✦':isWarrior(i)?'⚔':'➶';
        b.querySelector('.health').textContent=max?hp+' / '+max+' HP':'—';
        const values=snap.unit_stats?.[i]??[max,isMage(i)?map.mage_damage:isWarrior(i)?map.warrior_damage:map.archer_damage,100*(isWarrior(i)?(map.warrior_accuracy??.8):map.archer_accuracy),0,isWarrior(i)?map.warrior_initiative:60];
        b.querySelector('.unit-stats').textContent=max?(profile(i)?.unit_type==='Succub'?'Массовое превращение · без урона':profile(i)?.unit_type==='Witch'?'Превращение · без урона':(profile(i)?.unit_type==='Alchemist'?'Дополнительный ход':isPowerSupport(i)?'Усиление ×'+({Travnitsa:1.25,Novice:1.5,Dwarfdruid:1.75,Arhidruid:2}[profile(i).unit_type]):(isMassHealer(i)?'Массовое лечение ':isHealer(i)?'Лечение ':'Урон ')+fmt(values[1])))+(profile(i)?.unit_type==='Centaur Savage'?' + критический удар 5% (целые HP)':['Demon','Elfarcher'].includes(profile(i)?.unit_type)?' × 2 удара':'')+' · Точн. '+fmt(values[2])+'%\nБроня '+fmt(values[3])+'% · Иниц. '+fmt(values[4])+'\nИсточник: '+sourceName(profile(i)?.attack_type)+'\nИммунитеты: '+protectionNames(profile(i)?.immunities||0)+'\nЗащиты: '+protectionNames((profile(i)?.protections||0)&~s.wards_used[i])+((profile(i)?.protections||0)&s.wards_used[i]?' (израсходовано: '+protectionNames(profile(i).protections&s.wards_used[i])+')':''):'';
        b.querySelector('.unit-xp').textContent=max?experienceText(snap,i):'';
        b.querySelector('.health-fill').style.width=(max?hp/max*100:0)+'%';
        b.querySelector('.status').textContent=!max?'':!hp?'Погиб':s.escaped[i]?'Отступил':unreachable?'Вне досягаемости':s.paralyzed?.[i]?'Паралич':s.long_paralyzed?.[i]?'Долгий паралич':s.feared?.[i]?'Страх':s.retreating[i]?'Побег':s.defended[i]?'Защита':s.imp?.[i]?'Временное превращение':s.decay_form?.[i]?'Понижение формы':s.weakened?.[i]?'Урон ослаблен':s.armor_shreds?.[i]?'Броня повреждена':s.turn_phase[i]===1?'Ожидание':s.turn_phase[i]===2?'Ход завершён':'';
        if((s.slow_original?.[i]??-1)>=0)b.querySelector('.status').textContent+=' · Инициатива снижена';
        if(s.powerup?.[i])b.querySelector('.status').textContent+=' · Урон усилен';
        if(s.poison_mode?.[i]){const chance=s.poison_mode[i]===1?100:Math.min(100,11+10*Math.max(0,s.round-(s.effect_rounds?.[i]?.[0]??s.round)));b.querySelector('.status').textContent+=' · Яд '+s.poison_damage[i]+' HP, шанс снятия в этом раунде: '+chance+'%';}
        if(s.water_mode?.[i]){const chance=s.water_mode[i]===1?100:Math.min(100,11+10*Math.max(0,s.round-(s.effect_rounds?.[i]?.[1]??s.round)));b.querySelector('.status').textContent+=' · Обморожение '+(s.water_damage[i]||10)+' HP, шанс снятия в этом раунде: '+chance+'%';}
        if(s.burn_mode?.[i]){const chance=s.burn_mode[i]===1?100:Math.min(100,11+10*Math.max(0,s.round-(s.effect_rounds?.[i]?.[2]??s.round)));b.querySelector('.status').textContent+=' · Ожог '+(s.burn_damage[i]||10)+' HP, шанс снятия в этом раунде: '+chance+'%';}
        b.disabled=!targetTurn||mode!=='manual'||busy||s.done||!session||!mask[action];
        b.setAttribute('aria-label',(isHealer(s.actor)&&i<6?(profile(s.actor)?.unit_type==='Alchemist'?'Дать дополнительный ход: ':profile(s.actor)?.unit_type==='Patriach'&&!hp?'Воскресить: ':isPowerSupport(s.actor)?'Усилить: ':'Лечить: ')+unitName(i):i>=6?(profile(s.actor)?.unit_type==='Succub'?'Превращение всех доступных врагов: противник ':profile(s.actor)?.unit_type==='Witch'?'Превращение: противник ':isMage(s.actor)?'Заклинание по всем врагам: противник ':isWarrior(s.actor)?'Удар мечом: противник ':'Стрелять: противник ')+(i%6+1):unitName(i))+', '+hp+' из '+max+' здоровья'+(unreachable?', вне досягаемости':''));
      });
      const queue=Array.from({length:12},(_,i)=>i).filter(i=>s.round===0?s.preparation?.[i]:s.post_victory?s.pending_healers[i]:s.hp[i]>0&&!s.escaped[i]&&s.turn_phase[i]<2);
      const priority=i=>s.turn_phase[i]===0?s.priority[i]:-s.priority[i];queue.sort((a,b)=>s.post_victory?a-b:priority(b)-priority(a)||a-b);
      $('queue').replaceChildren(...queue.map(i=>{const e=document.createElement('span');e.className='queue-unit'+(i>=6?' foe':'')+(i===s.actor?' current':'');e.textContent=(isHealer(i)?'✚':isMage(i)?'М':isWarrior(i)?'⚔':i<6?'Л':'П')+(i%6+1);e.title=unitName(i)+(basic?' · инициатива раунда '+Math.floor(s.priority[i]):'');return e;}));
    }else drawMap(s);
    $('events').replaceChildren(...messages.slice(-6).reverse().map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
    if(records.length){const last=records[recordIndex].frames.length-1;$('scrubber').max=last;$('scrubber').value=frame;$('frame-count').textContent=frame+' / '+last;$('next').disabled=frame>=last;}
  }
  async function request(path,payload){
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const value=await response.json();if(!response.ok)throw Error(value.error||'Не удалось выполнить действие.');return value;
  }
  function showError(error){$('connection').textContent='Игра недоступна';$('message').textContent=error.message+' Для игры запустите open-numbergrid.cmd.';}
  async function resetGame(){
    if(busy)return;busy=true;render();
    try{const result=await request('/api/reset',{seed:42,faction:$('faction').value||undefined,lord_type:$('lord-type').value||data.map.lord_type||'warrior'});session=result.session;manualMap=result.map;$('version').textContent=manualMap.name;manualConstruction=result.construction;manualTurnRules=result.turn_rules;manualCombat=result.combat;manualCapital=result.capital;manualPotions=result.potions;manualEquipment=result.equipment;manualBattleItems=result.battle_items;manualSites=result.sites;manualRecruitment=result.recruitment;manualTerritory=result.territory;manualRuins=result.ruins;manualSpells=result.spell_research;manualCasting=result.spell_casting;manualSwaps=result.unit_swaps;manual=result.snapshot;messages=[];$('connection').textContent='Игра готова · локально';}
    catch(error){busy=false;render();showError(error);return;}
    busy=false;render();
  }
  async function sendAction(action){
    if(mode!=='manual'||busy||!session||!manual||manual.state.done||!manual.action_mask[action])return false;
    busy=true;render();
    try{const result=await request('/api/step',{session,action});
      for(const snapshot of result.events){manual=snapshot;const text=eventText(snapshot);if(text)messages.push(text);render();if(result.events.length>1)await new Promise(r=>setTimeout(r,180));}
      manual=result.snapshot;
    }catch(error){busy=false;render();showError(error);return false;}
    busy=false;render();
    // The server bounds each automatic batch. Continue a forced retreat even
    // when paralysis keeps it running for more than one response.
    if(manual.forced_retreat&&!manual.state.done){
      const retreatSession=session;
      setTimeout(()=>{if(session===retreatSession&&manual?.forced_retreat&&!busy)sendAction(manual.action_mask[16]?16:17);},0);
    }
    return true;
  }
  function setMode(next){if(busy)return;stop();mode=next;messages=[];$('manual-tab').setAttribute('aria-selected',String(mode==='manual'));$('replay-tab').setAttribute('aria-selected',String(mode==='replay'));$('manual-controls').hidden=mode!=='manual';$('replay-controls').hidden=mode!=='replay';render();}
  function nextFrame(){const last=records[recordIndex].frames.length-1;if(frame<last){frame++;const text=eventText(current());if(text)messages.push(text);}if(frame>=last)stop();render();}
  $('map-zoom').onchange=()=>{
    canvas.style.width=(Number($('map-zoom').value)*100)+'%';
    const snap=current();if(!snap)return;drawMap(snap.state);
    const viewport=$('map-viewport'),cell=canvas.clientWidth/currentMap().size;
    const agent=displayTile(snap.state.position,currentMap().size);
    viewport.scrollLeft=(agent.x+.5)*cell-viewport.clientWidth/2;
    viewport.scrollTop=(agent.y+.5)*cell-viewport.clientHeight/2;
  };
  canvas.onmousemove=event=>{
    const snap=current();if(!snap)return;
    const map=currentMap(),[row,col]=mapTileAtPointer(event);
    const chest=(map.chests||[]).find((c,i)=>snap.state.chest_alive?.[i]&&c.position[0]===row&&c.position[1]===col);
    const territory=territoryInfo(),cityIndex=territory?.cities.findIndex(c=>c.position[0]===row&&c.position[1]===col)??-1;
    const mineIndex=territory?.mines.findIndex(m=>m.position[0]===row&&m.position[1]===col)??-1;
    const ruinIndex=ruinInfo()?.ruins.findIndex(r=>r.footprint.some(p=>p[0]===row&&p[1]===col))??-1;
    if(ruinIndex>=0){const ruin=ruinInfo().ruins[ruinIndex];canvas.title=ruin.name+' · '+(snap.state.ruin_looted[ruinIndex]?'Разграблены · непроходимы':partyText(combatInfo()?.enemies[ruin.enemy])+' · награда: '+lootText(ruin));return;}
    if(cityIndex>=0){
      const city=territory.cities[cityIndex],defenders=(snap.state.enemy_positions||map.opponent_positions).map((p,i)=>snap.state.alive[i]&&p[0]===row&&p[1]===col?enemyParty(map,i):null).filter(Boolean);
      canvas.title=city.name+' · уровень '+snap.state.city_levels[cityIndex]+' · '+(snap.state.city_owned[cityIndex]?'Ваш город':defenders.join('; ')||'Войдите с живым героем для захвата');return;
    }
    if(mineIndex>=0){
      const mine=territory.mines[mineIndex],kind=territory.resource_kinds.indexOf(mine.kind);
      canvas.title=territory.resource_names[kind]+' · +50 за ход на своей земле · '+(snap.territory_quotes?.mine_owned[mineIndex]?'Ваш источник':'Нейтральный источник');return;
    }
    const kind=terrainGrid(map)[row*map.size+col]||0;
    const terrainTip=terrainNames[kind]+' · '+(snap.travel_costs||[2,1,4,6])[kind]+' очков движения';
    canvas.title=chest?'Сундук: '+[...Object.entries(chest.potions||{}),...Object.entries(chest.items||{})].filter(([,n])=>n).map(([key,n])=>(potionInfo()?.find(p=>p.key===key)?.name||equipmentInfo()?.items.find(p=>p.key===key)?.name||key)+' × '+n).join(', '):terrainTip;
  };
  canvas.ondblclick=event=>{
    const snap=current();if(mode!=='manual'||!snap||snap.state.in_battle||busy||snap.state.done)return;
    const map=currentMap(),[row,col]=mapTileAtPointer(event);
    const ruin=ruinInfo()?.ruins.find((r,i)=>!snap.state.ruin_looted[i]&&r.footprint.some(p=>p[0]===row&&p[1]===col));
    const enemy=ruin?.enemy??(snap.state.enemy_positions||map.opponent_positions).findIndex((p,i)=>snap.state.alive[i]&&p[0]===row&&p[1]===col);
    if(enemy<0)return;
    const action=(snap.map_commands||[]).findIndex(command=>command[0]===enemy);
    if(action>=0&&snap.action_mask[action])sendAction(action);
  };
  $('manual-tab').onclick=()=>setMode('manual');$('replay-tab').onclick=()=>setMode('replay');$('replay-tab').disabled=!records.length;
  $('world-view').onclick=()=>{capitalOpen=false;potionsOpen=false;spellsOpen=false;render();};$('capital-view').onclick=()=>{capitalOpen=true;potionsOpen=false;spellsOpen=false;render();};
  $('potions-view').onclick=()=>{capitalOpen=false;potionsOpen=true;spellsOpen=false;render();};
  $('spells-view').onclick=()=>{capitalOpen=false;potionsOpen=false;spellsOpen=true;render();};
  $('spell-level').onchange=render;
  $('lord-type').value=data.map.lord_type||'warrior';
  $('lord-type').onchange=render;
  $('building-branch').onchange=render;
  document.querySelectorAll('.dpad [data-action]').forEach(b=>{b.dataset.action=directionAction(Number(b.dataset.action));});
  $('reset').onclick=resetGame;document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>sendAction(Number(b.dataset.action)));
  $('play').onclick=()=>{if(timer){stop();return;}if(frame===records[recordIndex].frames.length-1){frame=0;messages=[];}timer=setInterval(nextFrame,230);$('play').textContent='Ⅱ Пауза';};
  $('next').onclick=nextFrame;$('scrubber').oninput=()=>{stop();frame=Number($('scrubber').value);messages=[];render();};
  $('record').onchange=()=>{stop();recordIndex=Number($('record').value);frame=0;messages=[];render();};
  records.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=r.label;$('record').append(option);});
  const keys={ArrowUp:0,ArrowRight:2,ArrowDown:4,ArrowLeft:6,KeyW:0,KeyE:1,KeyD:2,KeyC:3,KeyX:4,KeyS:4,KeyZ:5,KeyA:6,KeyQ:7};
  window.addEventListener('keydown',event=>{if(event.target.matches('select,input,textarea')||event.ctrlKey||event.metaKey||event.altKey)return;if(mode==='manual'&&!capitalOpen&&!potionsOpen&&!spellsOpen&&!manual?.state.in_battle&&keys[event.code]!==undefined){event.preventDefault();sendAction(directionAction(keys[event.code]));}});
  window.addEventListener('resize',()=>{const snap=current();if(snap&&!snap.state.in_battle)drawMap(snap.state);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  $('version').textContent=data.map.name;
  $('map-name').textContent=mapTitle(data.map);
  if(data.result){
    $('run-card').hidden=false;
    $('run-info').textContent='Запись: '+data.map.opponent_positions.length+' вражеских отрядов'+' · '+fmt(data.result.training_steps)+' шагов · '+fmt(Math.round(data.result.mean_steps_per_second))+' шагов/с';
    const evaluation=data.result.final_evaluation;
    $('run-evaluation').textContent=evaluation?'Победы argmax на карте записи: '+evaluation.successes+' / '+evaluation.episodes+'.':'Оценка модели не проводилась.';
  }
  let agentPick=null,agentAuto=false,actionCount=0,autoRun=0;
  function showPick(){
    document.querySelectorAll('.agent-pick').forEach(b=>b.classList.remove('agent-pick'));
    if(mode!=='manual'||!agentPick||agentPick.count!==actionCount)return;
    document.querySelectorAll('[data-action="'+agentPick.action+'"],[data-agent-action="'+agentPick.action+'"]').forEach(b=>b.classList.add('agent-pick'));
  }
  const baseRender=render;render=function(){baseRender();showPick();};
  const baseSend=sendAction;sendAction=async function(action){const ok=await baseSend(action);if(ok)actionCount++;showPick();return ok;};
  async function askAgent(){
    if(!session||!manual||manual.state.done)return null;
    const askedSession=session,askedCount=actionCount;
    try{const pick=await request('/api/agent',{session});
      if(session!==askedSession||actionCount!==askedCount)return null;
      agentPick={...pick,count:actionCount};
      const label=pick.action>=0&&pick.action<8?directionLabels[(pick.action+2)%8]:pick.label;
      $('agent-hint').textContent='Агент выбирает: '+label+' (вероятность '+Math.round(pick.probability*100)+'%).';showPick();return agentPick;}
    catch(error){if(session===askedSession)$('agent-hint').textContent=error.message;return null;}
  }
  async function agentMove(){if(busy)return false;const pick=await askAgent();if(!pick||busy)return false;return await sendAction(pick.action);}
  function setAuto(on){agentAuto=on;$('agent-auto').textContent=on?'Ⅱ Остановить агента':'▶ Агент играет сам';}
  $('reset').addEventListener('click',()=>{agentPick=null;setAuto(false);showPick();});
  $('agent-suggest').onclick=askAgent;$('agent-move').onclick=agentMove;
  $('agent-auto').onclick=async()=>{if(agentAuto){setAuto(false);return;}const run=++autoRun;setAuto(true);
    while(run===autoRun&&agentAuto&&mode==='manual'&&manual&&!manual.state.done){if(busy){await new Promise(r=>setTimeout(r,60));continue;}if(!await agentMove())break;await new Promise(r=>setTimeout(r,120));}
    if(run===autoRun)setAuto(false);};
  fetch('/api/health').then(r=>r.json()).then(h=>{if(!h.agent)return;$('agent-controls').hidden=false;
    const evaluation=h.agent.argmax_episodes==null?'оценка не проводилась':'побед argmax '+h.agent.argmax_wins+' / '+h.agent.argmax_episodes;
    $('agent-info').textContent='Модель: '+fmt(h.agent.training_steps)+' шагов обучения · '+evaluation+'.';}).catch(()=>{});
  window.numberGridApp={snapshot:()=>JSON.parse(JSON.stringify({mode,frame,...current()}))};
  if(records.length){manual=records[0].frames[0];render();}
  resetGame();
})();
