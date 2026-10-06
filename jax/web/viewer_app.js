(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('run-data').textContent);
  const engine = window.NumberGridEngine, map = data.map;
  const $ = id => document.getElementById(id);
  const canvas = $('board'), ctx = canvas.getContext('2d');
  let mode = 'replay', recordIndex = 0, frame = 0, timer = null;
  let manual = engine.initial(map), manualReward = 0;
  const fmt = n => new Intl.NumberFormat('ru-RU').format(n);
  const signed = n => (n > 0 ? '+' : '') + fmt(n);
  data.records.forEach((record, i) => {
    const option = document.createElement('option'); option.value = i; option.textContent = record.label;
    $('record').append(option);
  });
  $('gpu').textContent = data.result.device.replace('NVIDIA GeForce ', '');
  $('steps').textContent = fmt(data.result.training_steps);
  $('throughput').textContent = fmt(Math.round(data.result.mean_steps_per_second)) + ' / с';
  $('evaluation').textContent = `${data.result.final_evaluation.successes} / ${data.result.final_evaluation.episodes}`;
  $('map-size-badge').textContent = map.size;
  $('map-size').textContent = `${map.size} × ${map.size}`;
  $('initial-number').textContent = map.agent_number;
  $('enemy-count').textContent = map.opponent_numbers.length;
  $('max-reward').textContent = map.opponent_numbers.length + 3;
  $('episode-limit').textContent = map.max_steps;
  $('step-cost').textContent = fmt(map.step_cost || 0);
  $('exploration-bonus').textContent = fmt(map.exploration_bonus || 0);
  $('wall-rule').textContent = map.mask_walls ?
    'Ходы в стены недоступны, включая диагонали. Попытка войти в занятую клетку расходует ход.' :
    'Стены и занятые клетки не пропускают, но попытка тратит ход.';
  $('map-version').textContent = map.name;
  $('optimal-note').textContent = data.shortest_path_steps == null ?
    'Фиксированная карта · числа врагов сохраняются при сбросе' : `Кратчайший путь: ${data.shortest_path_steps} ходов.`;
  function current() { return mode === 'manual' ? {state: manual, total_reward: manualReward} : data.records[recordIndex].frames[frame]; }
  function stop() { if (timer) clearInterval(timer); timer = null; $('play').textContent = '▶ Смотреть'; $('play').setAttribute('aria-label','Воспроизвести запись'); }
  function roundedRect(x,y,w,h,r,color) { ctx.fillStyle=color; ctx.beginPath(); ctx.roundRect(x,y,w,h,r); ctx.fill(); }
  function draw() {
    const {state} = current();
    const width = canvas.clientWidth, ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(width*ratio);
    ctx.setTransform(ratio,0,0,ratio,0,0);
    const cell = width/map.size;
    ctx.fillStyle = '#f3f5ed'; ctx.fillRect(0,0,width,width);
    for (let r=0;r<map.size;r++) for (let c=0;c<map.size;c++) {
      const wall = r===0 || c===0 || r===map.size-1 || c===map.size-1;
      roundedRect(c*cell+1,r*cell+1,cell-2,cell-2,Math.max(2,cell*.13),wall?'#d2dcce':((r+c)%2===0?'#f9faf5':'#f0f3e9'));
      const index = r*map.size+c;
      if (!wall && state.visited && (state.visited[index>>>5] & (1<<(index&31)))) {
        roundedRect(c*cell+1,r*cell+1,cell-2,cell-2,Math.max(2,cell*.13),'#dcebdc');
      }
    }
    if (mode==='replay' && frame>0) {
      const frames=data.records[recordIndex].frames.slice(0,frame+1);
      ctx.strokeStyle='#8bb99e75';ctx.lineWidth=Math.max(2,cell*.09);ctx.lineJoin='round';ctx.beginPath();
      frames.forEach(({state:s},i)=>{const x=(s.position[1]+.5)*cell,y=(s.position[0]+.5)*cell;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    }
    const [ar,ac]=state.position;
    ctx.fillStyle='#91c8a322';
    for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++)if(ar+dr>0&&ar+dr<map.size-1&&ac+dc>0&&ac+dc<map.size-1)ctx.fillRect((ac+dc)*cell,(ar+dr)*cell,cell,cell);
    const colors=['#d4c5ef','#f4cd8d','#efb7ad'];
    function token(p,n,color,agent=false) {
      const x=p[1]*cell+cell*.1,y=p[0]*cell+cell*.1;
      roundedRect(x,y,cell*.8,cell*.8,cell*.2,color);
      if(agent){ctx.strokeStyle='#285c40';ctx.lineWidth=1.7;ctx.beginPath();ctx.roundRect(x,y,cell*.8,cell*.8,cell*.2);ctx.stroke();}
      ctx.fillStyle='#233a31';ctx.font=`650 ${Math.max(8,Math.round(cell*(n>=10 ? .50 : .55)))}px system-ui`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n,(p[1]+.5)*cell,(p[0]+.51)*cell);
    }
    map.opponent_positions.forEach((p,i)=>{if(state.alive[i])token(p,map.opponent_numbers[i],map.opponent_numbers[i]<state.number?colors[0]:map.opponent_numbers[i]===state.number?colors[1]:colors[2]);});
    token(state.position,state.number,state.lost?'#e8988b':'#a8deb6',true);
  }
  function render() {
    const {state,total_reward} = current();
    $('power').textContent=state.number;$('moves').textContent=fmt(state.step_count);
    $('reward').textContent=signed(total_reward);
    $('remaining').textContent=state.alive.filter(Boolean).length;
    const hasWeaker=map.opponent_numbers.some((n,i)=>state.alive[i]&&n<state.number);
    const text=state.won?'Победа! Все оппоненты собраны.':state.lost?'Поражение: сосед оказался не слабее.':state.done?'Время вышло. Начните новую игру.':hasWeaker?`Ищите число меньше ${state.number}.`:'Слабых врагов нет: захват сейчас невозможен.';
    $('message').textContent=text;
    $('board-status').textContent=state.won?`Победа · ${signed(total_reward)}`:state.lost?'Поражение':state.done?'Лимит ходов':state.step_count===0?'Готов к старту':'Игра идёт';
    const last=data.records[recordIndex].frames.length-1;
    $('scrubber').max=last;$('scrubber').value=frame;$('frame-count').textContent=`${frame} / ${last}`;
    $('next').disabled=frame>=last;
    const mask = engine.actionMask(state,map);
    document.querySelectorAll('[data-action]').forEach(b=>b.disabled=mode!=='manual'||state.done||
      (map.mask_walls && !mask[Number(b.dataset.action)]));
    draw();
  }
  function next() { const last=data.records[recordIndex].frames.length-1;if(frame<last)frame++;if(frame===last)stop();render(); }
  function play() { if(timer){stop();return;}if(frame===data.records[recordIndex].frames.length-1)frame=0;render();timer=setInterval(next,Number($('speed').value));$('play').textContent='Ⅱ Пауза';$('play').setAttribute('aria-label','Пауза'); }
  function setMode(nextMode) {
    stop();mode=nextMode;
    $('replay-tab').setAttribute('aria-selected',String(mode==='replay'));
    $('manual-tab').setAttribute('aria-selected',String(mode==='manual'));
    $('replay-controls').hidden=mode!=='replay';$('manual-controls').hidden=mode!=='manual';
    $('mode-label').textContent=mode==='manual'?'ВАША ИГРА':'ОБУЧЕННЫЙ АГЕНТ';render();
  }
  function move(action) {if(mode!=='manual'||manual.done||(map.mask_walls&&!engine.actionMask(manual,map)[action]))return;const result=engine.step(manual,action,map);manual=result.state;manualReward+=result.reward;render();}
  function resetManual(){manual=engine.initial(map);manualReward=0;render();}
  $('play').onclick=play;$('next').onclick=next;$('restart').onclick=()=>{stop();frame=0;render();};
  $('scrubber').oninput=()=>{stop();frame=Number($('scrubber').value);render();};
  $('speed').onchange=()=>{if(timer){stop();play();}};
  $('record').onchange=()=>{stop();recordIndex=Number($('record').value);frame=0;render();};
  $('replay-tab').onclick=()=>setMode('replay');$('manual-tab').onclick=()=>setMode('manual');
  $('reset-manual').onclick=resetManual;
  document.querySelectorAll('[data-action]').forEach(button=>button.onclick=()=>move(Number(button.dataset.action)));
  const keys={KeyW:0,KeyE:1,KeyD:2,KeyC:3,KeyX:4,KeyS:4,KeyZ:5,KeyA:6,KeyQ:7,ArrowUp:0,ArrowRight:2,ArrowDown:4,ArrowLeft:6,Numpad8:0,Numpad9:1,Numpad6:2,Numpad3:3,Numpad2:4,Numpad1:5,Numpad4:6,Numpad7:7};
  window.addEventListener('keydown',event=>{if(event.target.matches('select,input,textarea')||event.ctrlKey||event.altKey||event.metaKey)return;if(mode==='manual'&&keys[event.code]!==undefined){event.preventDefault();move(keys[event.code]);}else if(event.code==='KeyR'&&mode==='manual'){event.preventDefault();resetManual();}else if(event.code==='Space'&&mode==='replay'&&!event.target.matches('button')){event.preventDefault();play();}});
  window.addEventListener('resize',draw);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  window.numberGridApp={snapshot:()=>JSON.parse(JSON.stringify({mode,frame,...current()}))};
  render();
})();
