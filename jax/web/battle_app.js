(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('run-data').textContent);
  const $=id=>document.getElementById(id), fmt=n=>new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3}).format(n);
  const canvas=$('board'), ctx=canvas.getContext('2d');
  let mode='manual', session=null, manual=null, manualMap=null, busy=false, frame=0, recordIndex=0, timer=null, messages=[];
  const records=data.records||[];
  const archerCount=n=>n+' '+(n===1?'лучник':n>=2&&n<=4?'лучника':'лучников');
  const isMage=i=>i>=0&&i<6&&i===currentMap().hero_mage_slot;
  const unitName=i=>isMage(i)?'Ваш маг '+(i+1):(i<6?'Ваш лучник ':'Вражеский лучник ')+(i%6+1);
  function current(){return mode==='manual'?manual:records[recordIndex]?.frames[frame];}
  function currentMap(){return mode==='manual'?(manualMap||data.map):data.map;}
  function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Смотреть';}
  function roundRect(x,y,w,h,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();}
  function drawMap(s){
    const map=currentMap();
    const width=canvas.clientWidth;if(!width)return;
    const ratio=Math.min(devicePixelRatio||1,2);canvas.width=width*ratio;canvas.height=width*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);
    const cell=width/map.size;
    for(let r=0;r<map.size;r++)for(let c=0;c<map.size;c++){
      const wall=r===0||c===0||r===map.size-1||c===map.size-1,index=r*map.size+c;
      const seen=s.visited&&(s.visited[index>>>5]&(1<<(index&31)));
      roundRect(c*cell+1,r*cell+1,cell-2,cell-2,Math.max(2,cell*.12),wall?'#d3decc':seen?'#dfecd9':(r+c)%2?'#f3f6ef':'#ecf2e6');
    }
    const token=(p,n,hero)=>{roundRect(p[1]*cell+cell*.09,p[0]*cell+cell*.09,cell*.82,cell*.82,cell*.2,hero?'#a5d6ad':'#d5c0e8');ctx.fillStyle=hero?'#244c30':'#654777';ctx.font='650 '+Math.round(cell*.55)+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n,(p[1]+.5)*cell,(p[0]+.52)*cell);};
    map.opponent_positions.forEach((p,i)=>{if(s.alive[i])token(p,map.enemy_units[i],false);});
    token(s.position,s.hp.slice(0,6).filter(h=>h>0).length,true);
  }
  const slots=[3,0,4,1,5,2,6,9,7,10,8,11];
  for(const i of slots){
    const button=document.createElement('button');button.className='unit'+(i>=6?' foe':'');button.dataset.slot=i;
    button.innerHTML='<span class="unit-name"></span><span class="archer-icon" aria-hidden="true">➶</span><span class="health"></span><span class="health-bar"><span class="health-fill"></span></span><span class="status"></span>';
    button.onclick=()=>sendAction(8+i-6);$(i<6?'allies':'enemies').append(button);
  }
  function eventText(snapshot){
    const s=snapshot.state,a=unitName(s.last_actor),t=unitName(s.last_target);
    switch(s.last_event){
      case 1:return 'Начался бой с отрядом № '+(s.enemy+1)+'.';
      case 2:return isMage(s.last_actor)?a+': заклинание по всем противникам (суммарный урон '+s.last_damage+').':a+': попадание в '+t.toLowerCase()+' (−'+s.last_damage+').';
      case 3:return a+': промах.';
      case 4:return a+' встал в защиту.';
      case 5:return a+' ждёт конца раунда.';
      case 6:return a+' готовится отступить.';
      case 7:return a+' покинул бой.';
      case 8:return 'Победа! Весь отряд восстановлен.';
      case 9:return 'Ваш отряд погиб. Игра завершена.';
      case 10:return 'Отступление завершено. Отряд полностью восстановлен.';
      case 11:return 'Бой достиг лимита раундов. Эпизод завершён.';
      default:return null;
    }
  }
  function render(){
    const snap=current();if(!snap)return;const s=snap.state,mask=snap.action_mask,map=currentMap();
    $('version').textContent=map.name;
    $('map-panel').hidden=s.in_battle;$('battle-panel').hidden=!s.in_battle;
    $('phase-label').textContent=s.in_battle?'Бой · отряд № '+(s.enemy+1):'Карта '+map.size+' × '+map.size;
    $('remaining').textContent=s.alive.filter(Boolean).length;
    $('wins').textContent=s.alive.filter(v=>!v).length;
    $('steps').textContent=fmt(s.step_count);$('reward').textContent=fmt(snap.total_reward);
    const mageAlive=s.hp.slice(0,6).some((hp,i)=>hp>0&&isMage(i));
    $('party-health').textContent=archerCount(s.hp.slice(0,6).filter((hp,i)=>hp>0&&!isMage(i)).length)+(mageAlive?' + маг':'')+' · '+s.hp.slice(0,6).reduce((a,b)=>a+b,0)+' здоровья';
    const hasMage=Number.isInteger(map.hero_mage_slot)&&map.hero_mage_slot>=0;
    $('party-title').textContent=hasMage?'Пять лучников и маг.':'Шесть лучников.';
    $('mage-rule').hidden=!hasMage;
    $('battle-hint').textContent='Лучник: '+map.archer_damage+' урона одной цели.'+(hasMage?' Маг: '+map.mage_damage+' урона всем врагам.':'')+' Попадание '+Math.round(map.archer_accuracy*100)+'%.';
    $('attack-hint').textContent=isMage(s.actor)?'Ход мага: нажмите на любого живого врага — заклинание поразит всех противников.':'Нажмите на живого противника, чтобы выстрелить.';
    const message=s.won?'Победа! Карта очищена.':s.lost?'Ваш отряд погиб. Начните новую игру.':s.done?'Достигнут лимит. Начните новую игру.':s.in_battle?'Выбирайте цели и берегите свой отряд.':'Подойдите к любому вражескому отряду.';
    $('message').textContent=message;
    $('map-controls').hidden=s.in_battle;$('battle-controls').hidden=!s.in_battle;
    $('continue').hidden=!s.in_battle||!mask[17];
    document.querySelectorAll('[data-action]').forEach(b=>b.disabled=mode!=='manual'||busy||s.done||!session||!mask[Number(b.dataset.action)]);
    $('reset').disabled=busy;
    if(s.in_battle){
      $('battle-heading').textContent='Отряд № '+(s.enemy+1)+' · '+archerCount(map.enemy_units[s.enemy]);
      $('round').textContent='Раунд '+Math.min(s.round,snap.battle_max_rounds ?? map.battle_max_rounds);
      $('turn-message').textContent=s.done?message:s.actor<6&&!s.retreating[s.actor]?'Ваш ход: '+(isMage(s.actor)?'маг ':'лучник ')+(s.actor+1):s.actor>=6?'Ход противника: лучник '+(s.actor-5):unitName(s.actor)+' завершает отступление';
      document.querySelectorAll('[data-slot]').forEach(b=>{
        const i=Number(b.dataset.slot),max=snap.max_hp[i],hp=s.hp[i];
        b.className='unit'+(i>=6?' foe':'')+(isMage(i)?' mage':'')+(i===s.actor&&!s.done?' active':'')+(!max?' empty':!hp?' dead':s.escaped[i]?' escaped':'');
        b.querySelector('.unit-name').textContent=max?(isMage(i)?'Маг ':'Лучник ')+(i%6+1):'Пусто';
        b.querySelector('.archer-icon').textContent=isMage(i)?'✦':'➶';
        b.querySelector('.health').textContent=max?hp+' / '+max+' HP':'—';
        b.querySelector('.health-fill').style.width=(max?hp/max*100:0)+'%';
        b.querySelector('.status').textContent=!max?'':!hp?'Погиб':s.escaped[i]?'Отступил':s.retreating[i]?'Побег':s.defended[i]?'Защита':s.turn_phase[i]===1?'Ожидание':s.turn_phase[i]===2?'Ход завершён':'';
        b.disabled=i<6||mode!=='manual'||busy||s.done||!session||!mask[8+i-6];
        b.setAttribute('aria-label',(i>=6?(isMage(s.actor)?'Заклинание по всем врагам: противник ':'Стрелять: противник ')+(i%6+1):unitName(i))+', '+hp+' из '+max+' здоровья');
      });
      const queue=Array.from({length:12},(_,i)=>i).filter(i=>s.hp[i]>0&&!s.escaped[i]&&s.turn_phase[i]<2);
      const priority=i=>s.turn_phase[i]===0?s.priority[i]:-s.priority[i];queue.sort((a,b)=>priority(b)-priority(a)||a-b);
      $('queue').replaceChildren(...queue.map(i=>{const e=document.createElement('span');e.className='queue-unit'+(i>=6?' foe':'')+(i===s.actor?' current':'');e.textContent=(isMage(i)?'М':i<6?'В':'П')+(i%6+1);e.title=unitName(i);return e;}));
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
    try{const result=await request('/api/reset',{seed:42});session=result.session;manualMap=result.map;manual=result.snapshot;messages=[];$('connection').textContent='Игра готова · локально';}
    catch(error){busy=false;render();showError(error);return;}
    busy=false;render();
  }
  async function sendAction(action){
    if(mode!=='manual'||busy||!session||!manual||manual.state.done||!manual.action_mask[action])return;
    busy=true;render();
    try{const result=await request('/api/step',{session,action});
      for(const snapshot of result.events){manual=snapshot;const text=eventText(snapshot);if(text)messages.push(text);render();if(result.events.length>1)await new Promise(r=>setTimeout(r,180));}
      manual=result.snapshot;
    }catch(error){busy=false;render();showError(error);return;}
    busy=false;render();
  }
  function setMode(next){if(busy)return;stop();mode=next;messages=[];$('manual-tab').setAttribute('aria-selected',String(mode==='manual'));$('replay-tab').setAttribute('aria-selected',String(mode==='replay'));$('manual-controls').hidden=mode!=='manual';$('replay-controls').hidden=mode!=='replay';render();}
  function nextFrame(){const last=records[recordIndex].frames.length-1;if(frame<last){frame++;const text=eventText(current());if(text)messages.push(text);}if(frame>=last)stop();render();}
  $('manual-tab').onclick=()=>setMode('manual');$('replay-tab').onclick=()=>setMode('replay');$('replay-tab').disabled=!records.length;
  $('reset').onclick=resetGame;document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>sendAction(Number(b.dataset.action)));
  $('play').onclick=()=>{if(timer){stop();return;}if(frame===records[recordIndex].frames.length-1){frame=0;messages=[];}timer=setInterval(nextFrame,230);$('play').textContent='Ⅱ Пауза';};
  $('next').onclick=nextFrame;$('scrubber').oninput=()=>{stop();frame=Number($('scrubber').value);messages=[];render();};
  $('record').onchange=()=>{stop();recordIndex=Number($('record').value);frame=0;messages=[];render();};
  records.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=r.label;$('record').append(option);});
  const keys={ArrowUp:0,ArrowRight:2,ArrowDown:4,ArrowLeft:6,KeyW:0,KeyE:1,KeyD:2,KeyC:3,KeyX:4,KeyS:4,KeyZ:5,KeyA:6,KeyQ:7};
  window.addEventListener('keydown',event=>{if(event.target.matches('select,input,textarea')||event.ctrlKey||event.metaKey||event.altKey)return;if(mode==='manual'&&!manual?.state.in_battle&&keys[event.code]!==undefined){event.preventDefault();sendAction(keys[event.code]);}});
  window.addEventListener('resize',()=>{const snap=current();if(snap&&!snap.state.in_battle)drawMap(snap.state);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  $('version').textContent=data.map.name;
  if(data.result){$('run-card').hidden=false;$('run-info').textContent='Запись: '+data.map.opponent_positions.length+' отрядов · '+fmt(data.result.training_steps)+' шагов · '+fmt(Math.round(data.result.mean_steps_per_second))+' шагов/с';$('run-evaluation').textContent='Победы argmax на карте записи: '+data.result.final_evaluation.successes+' / '+data.result.final_evaluation.episodes+'.';}
  window.numberGridApp={snapshot:()=>JSON.parse(JSON.stringify({mode,frame,...current()}))};
  if(records.length){manual=records[0].frames[0];render();}
  resetGame();
})();
