"""Record one policy episode as a self-contained HTML replay (no desktop needed)."""
import argparse
import json
import os
import random
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from battle_env import BattleEnv
from maps.super_last_stand import SCHEDULED_WAVES
from train_campaign import MaskablePPO, make_env


HTML = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Super Last Stand — повтор эпизода</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10151e;color:#e8eef8;font:15px system-ui,sans-serif}
header,main,footer{max-width:1450px;margin:auto;padding:20px}h1{font-size:25px;margin:0 0 8px}h2{font-size:17px}
.muted{color:#a8b7cc}.layout{display:grid;grid-template-columns:1.15fr 1fr;gap:20px}.panel{background:#1b2432;border-radius:12px;padding:18px}
canvas{width:100%;background:#111925;border-radius:8px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0}
button,select{border:1px solid #4c607a;border-radius:7px;background:#26344a;color:white;padding:9px;cursor:pointer}input{flex:1;min-width:180px}
.units{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.unit{padding:9px;background:#243951;min-height:82px;border-radius:8px;font-size:12px}.red .unit{background:#482d37}.dead{opacity:.4}
.hp{height:5px;background:#10151e;margin:8px 0}.hp span{display:block;background:#65c796;height:100%}.red .hp span{background:#f2878a}
pre{white-space:pre-wrap;font:12px/1.5 monospace;max-height:220px;overflow:auto}#stats{line-height:1.8;margin-bottom:10px}summary{cursor:pointer}#events{max-height:180px;overflow:auto}
@media(max-width:850px){.layout{grid-template-columns:1fr}header,main,footer{padding:12px}}
</style>
<header><h1>Super Last Stand · повтор эпизода</h1><div id="meta" class="muted"></div>
<div class="controls"><button id="play">▶ Смотреть</button><button id="prev">←</button><button id="next">→</button>
<input id="seek" type="range" min="0" value="0"><span id="counter"></span>
<select id="speed"><option value="1">1 шаг/с</option><option value="3" selected>3 шага/с</option><option value="10">10 шагов/с</option><option value="30">30 шагов/с</option></select></div></header>
<main class="layout"><section class="panel"><canvas id="map" width="800" height="800"></canvas>
<p class="muted">● Голубой — герой · красный — враг · зелёный — побеждён · жёлтый — сундук · серый — препятствие. Координаты клетки и номер врага — при наведении.</p></section>
<section class="panel"><div id="stats"></div><h2>Отряд героя</h2><div id="blue" class="units"></div>
<h2 id="enemyTitle">Противник</h2><div id="red" class="units red"></div>
<h2>События шага</h2><pre id="logs"></pre><details><summary>Постройки и заклинания</summary><pre id="inventory"></pre></details>
<h2>Переходы к боям</h2><div id="events"></div></section></main>
<footer class="muted">Запись одного эпизода модели, без изменения её весов. Пауза, перемотка и скорость управляют просмотром записи.</footer>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent()),F=D.frames;
const $=id=>document.getElementById(id), c=$('map'),ctx=c.getContext('2d');let i=0,playing=false,timer;
$('meta').textContent=`Модель ${D.model_steps.toLocaleString('ru')} шагов · ${D.deterministic?'наиболее вероятное действие':'выбор по вероятностям политики'} · seed ${D.seed} · итог: ${D.result} · пройдено волн: ${D.waves}`;
$('seek').max=F.length-1;
function units(id,arr){$(id).replaceChildren();for(const u of arr){let el=document.createElement('div');el.className='unit'+(u.hp<=0?' dead':'');let title=document.createElement('strong');title.textContent=u.name;el.append(title);let bar=document.createElement('div');bar.className='hp';let fill=document.createElement('span');fill.style.width=Math.max(0,Math.min(100,100*u.hp/(u.maxhp||1)))+'%';bar.append(fill);el.append(bar);let text=document.createElement('div');text.textContent=`HP ${Math.round(u.hp)}/${Math.round(u.maxhp)} · слот ${u.pos}`;el.append(text);$(id).append(el)}}
function draw(){const f=F[i],s=800/D.grid_size;ctx.clearRect(0,0,800,800);ctx.fillStyle='#111925';ctx.fillRect(0,0,800,800);
ctx.fillStyle='#354153';for(const [x,y] of D.obstacles)ctx.fillRect(x*s,y*s,s,s);
ctx.strokeStyle='#263144';ctx.lineWidth=.4;for(let a=0;a<=D.grid_size;a++){ctx.beginPath();ctx.moveTo(a*s,0);ctx.lineTo(a*s,800);ctx.stroke();ctx.beginPath();ctx.moveTo(0,a*s);ctx.lineTo(800,a*s);ctx.stroke()}
ctx.strokeStyle='#64cafc';ctx.lineWidth=2;ctx.beginPath();for(let j=0;j<=i;j++){const p=F[j].pos;ctx.lineTo((p[0]+.5)*s,(p[1]+.5)*s)}ctx.stroke();
ctx.fillStyle='#e8c36b';for(const p of f.chests)ctx.fillRect((p[0]+.2)*s,(p[1]+.2)*s,s*.6,s*.6);
for(const e of f.enemies){ctx.fillStyle=e[3]?'#fa7c8b':'#477b61';ctx.beginPath();ctx.arc((e[1]+.5)*s,(e[2]+.5)*s,Math.max(2,s*.36),0,Math.PI*2);ctx.fill()}
ctx.fillStyle='#84dcff';ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.arc((f.pos[0]+.5)*s,(f.pos[1]+.5)*s,Math.max(5,s*.5),0,Math.PI*2);ctx.fill();ctx.stroke();
$('stats').textContent=`Шаг ${f.step} · ${f.mode==='battle'?'Бой':'Карта'} · ход ${f.turn} · золото ${Math.round(f.gold)} · награда ${f.reward.toFixed(1)} · пройдено волн ${f.waves}`;
units('blue',f.blue);units('red',f.red);$('enemyTitle').textContent=f.red.length?'Противник — бой':'Противник — вне боя';$('logs').textContent=f.logs.join('\n')||'Нет событий';
$('inventory').textContent='Постройки: '+f.buildings.join(', ')+'\n\nЗаклинания: '+f.spells.join(', ');
$('seek').value=i;$('counter').textContent=`${i} / ${F.length-1}`;}
function stop(){playing=false;clearTimeout(timer);$('play').textContent='▶ Смотреть'}
function tick(){if(!playing)return;if(i>=F.length-1){stop();return}i++;draw();timer=setTimeout(tick,1000/Number($('speed').value))}
$('play').onclick=()=>{if(playing){stop();return}if(i===F.length-1)i=0;playing=true;$('play').textContent='❚❚ Пауза';tick()};
$('prev').onclick=()=>{stop();i=Math.max(0,i-1);draw()};$('next').onclick=()=>{stop();i=Math.min(F.length-1,i+1);draw()};$('seek').oninput=()=>{stop();i=Number($('seek').value);draw()};
c.onmousemove=e=>{const b=c.getBoundingClientRect(),x=Math.floor((e.clientX-b.left)/b.width*D.grid_size),y=Math.floor((e.clientY-b.top)/b.height*D.grid_size);const en=F[i].enemies.filter(v=>v[1]===x&&v[2]===y);c.title=`(${x}, ${y})`+en.map(v=>` · враг ${v[0]}`).join('')};
for(let j=1;j<F.length;j++)if(F[j].battle_start){const b=document.createElement('button');b.textContent=`Шаг ${j}: враг ${F[j].battle_start}`;b.onclick=()=>{stop();i=j;draw()};$('events').append(b)}draw();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--vecnormalize', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    battle_seeds = random.Random(args.seed)
    original_init = BattleEnv.__init__

    def seeded_init(battle, *a, **kw):
        original_init(battle, *a, **kw)
        battle.seed(battle_seeds.getrandbits(64))

    BattleEnv.__init__ = seeded_init
    raw = make_env(log_enabled=True, map_name='super_last_stand', eval_max_episode_steps=5000)
    norm = VecNormalize.load(args.vecnormalize, DummyVecEnv([lambda: raw]))
    norm.training = False
    norm.norm_reward = False
    model = MaskablePPO.load(args.model, env=norm, device='cpu')
    set_random_seed(args.seed)
    battle_seeds.seed(args.seed)
    obs, info = raw.reset(seed=args.seed)
    base = raw.unwrapped
    wave_by_enemy = {int(enemy): int(wave['wave_index']) for wave in SCHEDULED_WAVES for enemy in wave['enemy_ids']}
    frames = []
    reward_total = 0.0
    waves = reached = 0

    def units(rows):
        return [dict(name=str(u.get('name', '?')), pos=int(u.get('position', 0)),
                     hp=float(u.get('health', 0)), maxhp=float(u.get('max_health', u.get('health', 0)))) for u in rows]

    def capture(step, info):
        logs = list(base.pop_campaign_logs())
        battle = base.battle_env
        if battle is not None:
            logs += list(battle.pop_pretty_events())
        rows = battle.combined if battle is not None and base.mode == base.MODE_BATTLE else []
        blue = [u for u in rows if str(u.get('team')).lower() == 'blue'] if rows else base._get_blue_state()
        red = [u for u in rows if str(u.get('team')).lower() == 'red']
        if info.get('campaign_result'):
            logs.append('Итог: ' + str(info['campaign_result']))
        frames.append(dict(step=step, pos=list(base.grid_env.agent_pos), mode='battle' if base.mode == base.MODE_BATTLE else 'grid',
                           turn=base.turns, gold=base.gold, reward=reward_total, waves=waves,
                           blue=units(blue), red=units(red), logs=logs,
                           enemies=[[int(k), *v, bool(base.grid_env.enemies_alive.get(k, False))]
                                    for k, v in base.grid_env.enemy_positions.items()],
                           chests=list(getattr(base, 'chests', {})),
                           buildings=base.get_built_building_names(), spells=base.get_learned_spell_descriptions(),
                           battle_start=info.get('enemy_id') if info.get('battle_triggered') else None))

    capture(0, info)
    for step in range(1, 5001):
        normalized = norm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        action, _ = model.predict(normalized, deterministic=args.deterministic,
                                  action_masks=base.compute_action_mask().reshape(1, -1))
        obs, reward, terminated, truncated, info = raw.step(int(action[0]))
        reward_total += float(reward)
        waves = max(waves, int(info.get('waves_defeated_count', 0)))
        if info.get('battle_triggered') or info.get('battle_result') or info.get('spell_enemy_defeated'):
            reached = max(reached, wave_by_enemy.get(int(info.get('enemy_id', info.get('target_enemy_id', 0)) or 0), 0))
        capture(step, info)
        if terminated or truncated:
            break
    data = dict(model_steps=int(model.num_timesteps), seed=args.seed, deterministic=args.deterministic,
                grid_size=base.grid_size, obstacles=list(base.grid_env.obstacle_positions),
                result=info.get('campaign_result', 'timeout'), waves=waves, reached_wave=reached,
                reward=reward_total, length=step, frames=frames)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else list(x))
    output.write_text(HTML.replace('__DATA__', payload.replace('<', '\\u003c')), encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps({k:v for k,v in data.items() if k not in ('frames', 'obstacles')}, ensure_ascii=False, indent=2))
    norm.close()
    print(f'REPLAY SAVED {output}: steps={step}, result={data["result"]}, waves={waves}, reached={reached}', flush=True)


if __name__ == '__main__':
    main()
