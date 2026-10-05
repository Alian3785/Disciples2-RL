#!/usr/bin/env python3
"""Generate documented runtime map diagrams. Never starts training or a replay.

Run from Big_map: python tools/generate_map_layouts.py
Requires the game's normal dependencies and Pillow. Asset directories deliberately
have no __init__.py: maps/<name>.py remains the importable module.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont
from campaign_env import CampaignEnv
from maps import available_maps
from maps.base import capital_footprint_block, city_footprint_block, footprint_tiles
from grid import scale_static_tiles

BG = '#0d1524'
FG = '#eff3f8'
MUTED = '#9cacc1'
COLORS = {'plain':'#182637', 'forest':'#355d49', 'water':'#2d6e94',
          'road':'#756a5b', 'obstacle':'#465063', 'field':'#add771',
          'city':'#f1b25c', 'ruin':'#c397ef', 'capital':'#ed8796',
          'future':'#7dd9e1', 'chest':'#ffd789', 'merchant':'#f6c175',
          'spell_shop':'#cca7f7', 'mercenary':'#f1959f', 'trainer':'#8ac6e8'}
TIERS = {1:'#a6d672',2:'#d6d866',3:'#f1b25c',4:'#ee835d',5:'#e45666'}
MANA = {'infernal':('#d87392','П'), 'life':('#83acef','Ж'),
        'death':('#d6d8de','С'), 'runes':('#8de0df','Р'), 'grove':('#8acb86','Л')}
FACTIONS = {1:'Империя',2:'Легионы Проклятых',3:'Горные Кланы',4:'Орды Нежити',5:'Эльфы'}
TITLES = {'default':'A Return to Simpler Times', 'small':'Small · уменьшенная карта',
 'builder':'Builder · строительство', 'city_defence_train':'City Defence · оборона города',
 'formation_train':'Formation · построение', 'green_dragon_minimal':'Green Dragon Minimal',
 'hire_train':'Hire · найм', 'item_train':'Item · предметы', 'magic_train':'Magic · магия',
 'mirrow_match':'Mirrow match', 'orc_duel':'Orc Duel · дуэль с орком',
 'scroll_train':'Scroll · свитки', 'siege_train':'Siege · осада',
 'super_last_stand':'Super Last Stand', 'trade_train':'Trade · торговля',
 'wotans_retribution':"Wotan’s Retribution"}
OBJECTIVES = {'cities':'Захват целевых городов','city_defence':'Отразить два штурма города',
 'build_all':'Построить все здания','all_enemies':'Победить всех врагов',
 'dragon':'Победить зелёного дракона','orc':'Победить орка',
 'scripted_bot':'Победить скриптового бота','target_enemy':'Победить целевой отряд',
 'waves':'Отразить все волны'}


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + name, size)


def real_units(units):
    # Never deduplicate by unit_id (often a unit TYPE id) or name. Runtime
    # armies already contain one fighter per source unit and placeholders.
    return [dict(u) for u in units if u.get('name') not in (None,'','пусто')
            and float(u.get('max_health',u.get('health',0)) or 0) > 0]


def unit_summary(units):
    count = Counter((str(u['name']), int(u.get('Level',1) or 1)) for u in units)
    return ' + '.join((f'{n} × ' if n > 1 else '') + name +
                    (f' (ур. {level})' if level > 1 else '')
                    for (name,level),n in count.items()) or 'Пустой отряд'


def extract(name):
    env = CampaignEnv(map_name=name, observation_version='local5',
                      scripted_capital_bot_enabled=False, use_boss_starting_roster=False)
    env.reset(seed=0)
    try:
        m = env._map
        module = importlib.import_module('maps.' + name)
        raw = {int(r['enemy_id']):dict(r) for r in m.enemy_stacks}
        waves = {int(r['enemy_id']):dict(r) for r in env._scheduled_enemy_wave_configs()}
        cities = []
        city_ids = set()
        for i,(title,row) in enumerate(env.legions_settlement_territory_source_by_name.items(),1):
            ids = [int(x) for x in row.get('required_enemy_ids',())]
            city_ids.update(ids)
            cities.append(dict(code=f'C{i}',name=title,position=list(row['source_tile']),
                               enemy_ids=ids,level=int(row.get('settlement_level',1)),
                               objective=title in m.final_objective_cities,
                               owned=title in env.legions_active_settlement_territory_capture_turn_by_name))
        enemies = []
        bot_ids = set(m.scripted_capital_bot_home_enemy_ids)
        if m.empire_territory_source_enemy_id is not None:
            bot_ids.add(int(m.empire_territory_source_enemy_id))
        for eid,pos in env.grid_env.enemy_positions.items():
            units = real_units(env.enemy_team_states.get(eid,()))
            kind = 'future' if eid in waves else ('ruin' if eid in m.ruin_rewards else
                    'city' if eid in city_ids else 'capital' if eid in bot_ids else 'field')
            r = raw.get(eid,{})
            enemies.append(dict(id=int(eid),position=list(pos),alive=bool(env.grid_env.enemies_alive.get(eid)),
              kind=kind,description=env._enemy_descriptions.get(eid,r.get('description','')),
              units=[{k:u.get(k) for k in ('name','Level','big','position','source_unit_id','unit_id') if k in u}
                     for u in units],summary=unit_summary(units),
              fighter_count=len(units),formation_cells=sum(2 if u.get('big') else 1 for u in units),
              difficulty_tier=r.get('difficulty_tier'),formation_key=r.get('formation_key'),
              region=r.get('region'),wave=waves.get(eid)))
        sites=[]
        for kind,prefix in [('merchant','T'),('spell_shop','M'),('mercenary','H'),('trainer','U')]:
            for i,(title,pos) in enumerate(getattr(env,kind+'_site_anchors').items(),1):
                sites.append(dict(code=f'{prefix}{i}',kind=kind,name=title,position=list(pos),
                     interaction_tiles=[list(t) for t in getattr(env,kind+'_site_interaction_tiles')[title]]))
        caps = [dict(code='A',name='Столица: '+FACTIONS.get(env.Realcapital,str(env.Realcapital)),
                     position=list(env.CASTLE_POS),color='#b85869')]
        if m.empire_territory_source_tile is not None or m.empire_territory_source_enemy_id is not None:
            caps.append(dict(code='B',name='Вражеская столица: '+m.scripted_capital_bot_faction,
                             position=list(env.empire_territory_source_tile),color='#598cd9'))
        def scaled_tiles(block):
            return [list(t) for t in sorted(scale_static_tiles(env.grid_size, footprint_tiles(block)))]
        for cap in caps:
            if cap['code']=='A':
                block=getattr(module,'AGENT_CAPITAL_BLOCK',capital_footprint_block(m.hero_start))
            else:
                block=getattr(module,'BOT_CAPITAL_BLOCK',capital_footprint_block(
                    m.empire_territory_source_tile or m.enemy_positions()[m.empire_territory_source_enemy_id]))
            cap['footprint']=scaled_tiles(block)
        source_cities={str(r['name']):r['source_tile'] for r in m.legions_settlement_territory_data}
        for c in cities:
            c['footprint']=scaled_tiles(city_footprint_block(source_cities[c['name']]))
        data = dict(schema_version=1,map_id=name,title=TITLES.get(name,name),grid_size=env.grid_size,
            source_grid_size=m.grid_size,seed=0,observation_version='local5',
            scripted_bot_enabled=False,boss_roster_enabled=False,objective=m.default_objective,
            objective_enemy_id=m.objective_enemy_id,turn=int(env.turns),
            faction=FACTIONS.get(env.Realcapital,str(env.Realcapital)),
            starting_gold=float(env.gold),starting_army=unit_summary(real_units(env.blue_team_state)),
            starting_roster_mode='fixed' if m.starting_roster else 'ordinary',
            hero=list(env.grid_env.agent_pos),capitals=caps,cities=cities,enemies=enemies,sites=sites,
            terrain={k:[list(p) for p in sorted(getattr(env,k+'_tiles'))] for k in ('water','forest','road')},
            obstacles=[list(p) for p in sorted(env.grid_env.obstacle_positions)],
            gold_mines=[list(p) for p in env.gold_mine_tiles],
            mana=[dict(position=list(p),kind=r['kind'],name=r['name']) for p,r in env.mana_sources.items()],
            chests=[dict(code=f'X{i}',position=list(p),items=list(items)) for i,(p,items) in enumerate(env.chests.items(),1)],
            ruins=[dict(id=int(eid),**r) for eid,r in m.ruin_rewards.items()],
            notes=[],conditional_assaults=[])
        for ruin in data['ruins']:
            for key in ('ruin_pos','marker_pos'):
                if key in ruin:
                    ruin['source_'+key]=list(ruin[key])
                    ruin[key]=list(scale_static_tiles(env.grid_size,(tuple(ruin[key]),),base_grid_size=m.grid_size)[0])
            ruin['guardian_position']=list(env.grid_env.enemy_positions[ruin['id']])
        if name=='mirrow_match':
            keys={r[0]:i for i,r in enumerate(module.FORMATIONS,1)}
            for e in enemies:e['marker']='★' if e['region']=='center' else str(keys[e['formation_key']])
            data['notes'].append('19 зеркальных пар нейтральных отрядов и Тиамат в центре. Цвета: сложность 1–5 из конфигурации.')
        else:
            for e in enemies:e['marker']=str(e['id'])
        if any(n > 1 for n in Counter(tuple(e['position']) for e in enemies if e['kind']!='future').values()):
            data['notes'].append('Знак + у номера: на клетке несколько армий; все составы перечислены справа.')
        if m.starting_roster:
            data['notes'].append('Стартовый отряд закреплён сценарием и сохраняется при отключённом боссовом ростере.')
        if name=='city_defence_train':
            data['conditional_assaults']=[dict(wave=i,position=list(module.BOT_SPAWN_TILE),faction=f,
                  roster=list(r.values())) for i,(f,r) in enumerate(zip(module.ASSAULT_FACTIONS,module.ASSAULT_ROSTERS),1)]
            data['notes'].append('При включённом scripted bot: штурм из (24,44) к C1 за 2 хода; после первой победы вторая волна появляется через 2 хода.')
            data['notes'].append('C1 принадлежит игроку, уровень 5; гарнизон на старте пуст.')
        if m.scripted_capital_bot_supported:
            data['notes'].append('Скриптовый бот показан только как условие сценария; при снимке выключен.')
        if m.boss_starting_roster_default and not m.starting_roster:
            data['notes'].append('Для снимка выбран обычный стартовый отряд; боссовый ростер не включён.')
        if env.grid_size != m.grid_size:
            data['notes'].append(f'Игровая сетка {env.grid_size}×{env.grid_size}: координаты взяты после масштабирования и разрешения конфликтов.')
        if len(m.enemy_stacks)!=len(raw):
            data['notes'].append(f'Повторяющиеся enemy_id: действует последняя запись; {len(enemies)} уникальных отрядов.')
        # Guard that the visual inventory is complete and every fighter is counted once.
        assert set(raw)=={e['id'] for e in enemies}
        assert all(e['formation_cells'] <= 6 for e in enemies), name
        assert all(0<=v<env.grid_size for e in enemies for v in e['position']),name
        assert len(data['mana'])==len(env.mana_sources)
        return data
    finally:
        env.close()


def wrap(text, f, max_width):
    result=[]
    for paragraph in text.split('\n'):
        line=''
        for word in paragraph.split():
            candidate=(line+' '+word).strip()
            if f.getlength(candidate)>max_width and line:
                result.append(line);line=word
            else:line=candidate
        result.append(line)
    return result


def roster_entries(data):
    result=[]
    seen=set()
    for e in data['enemies']:
        if e['kind']=='future':continue
        if data['map_id']=='mirrow_match':
            if e['marker'] in seen:continue
            seen.add(e['marker'])
        suffix = ' · руины' if e['kind']=='ruin' else ' · гарнизон' if e['kind'] in ('city','capital') else ''
        color='#c397ef' if e['marker']=='★' else TIERS.get(e['difficulty_tier'],COLORS[e['kind']])
        result.append((e['marker'],e['summary']+suffix,color))
    return result


def draw(data, destination):
    dense=len(roster_entries(data))>38
    map_size=1728
    margin=86; top=208
    sidebar_x=margin+map_size+90
    col_width=540 if dense else 720
    cols=2 if dense else 1
    width=sidebar_x+cols*col_width+60
    f=font(24);small=font(21);bold=font(25,True)
    entries=roster_entries(data)
    prepared=[(code,wrap(txt,f,col_width-64),color) for code,txt,color in entries]
    # Balanced contiguous columns, preserving numeric registry order.
    heights=[max(42,len(lines)*31+14) for _,lines,_ in prepared]
    split=len(prepared)
    if dense:
        half=sum(heights)/2;acc=0
        split=0
        for h in heights:
            if acc>=half:break
            acc+=h;split+=1
    groups=[prepared[:split]]+([prepared[split:]] if dense else [])
    roster_h=max([sum(max(42,len(ls)*31+14) for _,ls,_ in g) for g in groups] or [0])
    future=[e for e in data['enemies'] if e['kind']=='future']
    info_items=[f"{c['code']}  {c['name']}" for c in data['capitals']]
    for c in data['cities']:
        ids=', '.join(str(x) for x in c['enemy_ids']) or 'пустой гарнизон'
        info_items.append(f"{c['code']}  {c['name']} · ур. {c['level']}"+(' · цель' if c['objective'] else '')+f' · {ids}')
    for s in data['sites']:info_items.append(f"{s['code']}  {s['name']}")
    for r in data['ruins']:info_items.append(f"R{r['id']}  {r['title']} · отряд {r['id']}")
    # Full object/chest loot is also in the neighboring README and JSON.
    notes=list(data['notes'])
    if future:
        turns=[e['wave']['spawn_turn'] for e in future]
        notes.insert(0,f"Будущие отряды: {len(future)}; появление на ходах {min(turns)}–{max(turns)}. Полное расписание: waves.png.")
    if not entries:prepared=[]
    info_lines=[line for text in info_items for line in wrap(text,small,cols*col_width-10)]
    notes_lines=[line for text in notes for line in wrap(text,small,cols*col_width-10)]
    sidebar_end=top+55+roster_h+70+len(info_lines)*30+50+len(notes_lines)*30
    height=max(top+map_size+540,sidebar_end+80)
    im=Image.new('RGB',(width,height),BG);d=ImageDraw.Draw(im)
    # All labels are explicitly wrapped and checked against the canvas.
    def text(x,y,t,ft=f,color=FG):
        box=d.textbbox((x,y),t,font=ft)
        assert box[2]<=width-12 and box[3]<=height-12,(data['map_id'],t,box,(width,height))
        d.text((x,y),t,font=ft,fill=color)
    text(margin,42,data['title'],font(52,True))
    active=sum(e['alive'] for e in data['enemies'])
    text(margin,115,f"{data['grid_size']} × {data['grid_size']}   ·   Активных отрядов: {active}"+
        (f"   ·   Будущих: {len(future)}" if future else '')+f"   ·   Городов: {len(data['cities'])}",font(27),MUTED)
    text(margin,158,OBJECTIVES.get(data['objective'],data['objective'])+f"   /   {data['map_id']}",font(22),MUTED)
    n=data['grid_size'];cell=map_size/n
    def center(p):return (margin+(p[0]+.5)*cell,top+(p[1]+.5)*cell)
    def tile(p,color,outline=None):
        x,y=center(p);d.rectangle((x-cell/2,y-cell/2,x+cell/2,y+cell/2),fill=color,outline=outline)
    d.rectangle((margin,top,margin+map_size,top+map_size),fill=COLORS['plain'])
    for kind in ('forest','water','road'):
        for p in data['terrain'][kind]:tile(p,COLORS[kind])
    obstacles={tuple(t) for t in data['obstacles']}
    for p in data['obstacles']:tile(p,COLORS['obstacle'])
    for c in data['capitals']+data['cities']:
        color=c.get('color','#8d7449')
        for p in c['footprint']:
            if tuple(p) in obstacles:tile(p,color)
    for i in range(n+1):
        x=margin+i*cell;y=top+i*cell
        d.line((x,top,x,top+map_size),fill='#29384b',width=1)
        d.line((margin,y,margin+map_size,y),fill='#29384b',width=1)
    d.rectangle((margin,top,margin+map_size,top+map_size),outline='#52647b',width=2)
    ticks=2 if n<=32 else 4
    for i in range(0,n,ticks):
        x,y=center((i,i));d.text((x,top+map_size+16),str(i),font=small,fill=MUTED,anchor='mt')
        d.text((margin-15,y),str(i),font=small,fill=MUTED,anchor='rm')
    text(margin+map_size-30,top+map_size+49,'x',small,MUTED)
    text(margin-42,top-35,'y ↓',small,MUTED)
    radius=min(16,cell*.42)
    def symbol(p,label,color,shape='circle',scale=1.0):
        x,y=center(p);r=radius*scale
        if shape=='diamond':d.polygon(((x,y-r),(x+r,y),(x,y+r),(x-r,y)),fill=color,outline=BG)
        elif shape=='box':d.rectangle((x-r,y-r,x+r,y+r),fill=color,outline=BG,width=2)
        elif shape=='star':
            pts=[(x+math.sin(i*math.pi/5)*r*(1 if i%2==0 else .42),y-math.cos(i*math.pi/5)*r*(1 if i%2==0 else .42)) for i in range(10)]
            d.polygon(pts,fill=color,outline=FG)
        elif shape=='ring':d.ellipse((x-r,y-r,x+r,y+r),fill=BG,outline=color,width=3)
        else:d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline=BG,width=2)
        if label:
            fs=min(18,int(2*r*.77)) if len(label)==1 else 18 if len(label)==2 else 14
            d.text((x,y),label,font=font(fs,True),fill=color if shape=='ring' else BG,anchor='mm')
    for p in data['gold_mines']:symbol(p,'', '#f3ca56','diamond',.75)
    for m in data['mana']:
        color,letter=MANA.get(m['kind'],('#c397ef','?'));symbol(m['position'],letter,color,'box',.66)
    for c in data['chests']:symbol(c['position'],'X',COLORS['chest'],'box',.62)
    for s in data['sites']:
        symbol(s['position'],s['code'],COLORS[s['kind']],'box',.98)
        # Dots are every actual usable interaction tile, not guessed entrances.
        for p in s['interaction_tiles']:
            x,y=center(p);d.ellipse((x-3,y-3,x+3,y+3),fill=COLORS[s['kind']])
    for c in data['capitals']:
        tile(c['position'],'#f4eddf')
        # The letter is centered inside the colored footprint, away from entrance armies.
        points=c['footprint'];avg=(sum(p[0] for p in points)/len(points),sum(p[1] for p in points)/len(points))
        x,y=center(avg);d.text((x,y),c['code'],font=font(30,True),fill=FG,anchor='mm')
    for c in data['cities']:
        points=c['footprint'];avg=(sum(p[0] for p in points)/len(points),sum(p[1] for p in points)/len(points))
        x,y=center(avg);d.text((x,y),c['code'],font=font(23,True),fill=FG,anchor='mm')
        if not c['enemy_ids']:symbol(c['position'],'+', '#eed5a4','box',.6)
    groups_at=defaultdict(list)
    for e in data['enemies']:
        if e['kind']!='future':groups_at[tuple(e['position'])].append(e)
    for p,group in groups_at.items():
        e=group[0];color=TIERS.get(e['difficulty_tier'],COLORS[e['kind']])
        label=e['marker'] if len(group)==1 else str(e['id'])+'+'
        if label=='★':symbol(p,'','#c397ef','star',1.35)
        else:symbol(p,label,color,'circle')
    future_pos=sorted({tuple(e['position']) for e in future})
    for i,p in enumerate(future_pos,1):symbol(p,'W'+str(i),COLORS['future'],'ring',1)
    if data['conditional_assaults']:
        p=data['conditional_assaults'][0]['position'];symbol(p,'S',COLORS['future'],'ring',1)
        tx,ty=center(data['cities'][0]['position']);sx,sy=center(p)
        for yy in range(int(ty+cell),int(sy-cell),28):d.line((sx,yy,sx,yy+12),fill=COLORS['future'],width=3)
        d.polygon(((tx,ty+cell*.7),(tx-8,ty+cell*.7+18),(tx+8,ty+cell*.7+18)),fill=COLORS['future'])
    # The hero gets a small bright triangle on the exact start tile, including map corners.
    x,y=center(data['hero']);d.polygon(((x,y-11),(x-8,y+7),(x+8,y+7)),fill=BG)
    sy=top+map_size+82
    legend=[('A / B','Столица игрока / вражеская',FG),('C1','Город; оранжевый номер — защитник',COLORS['city']),
       ('●','Номер отряда; фиолетовый — руины',COLORS['field']),('◆','Золотая шахта', '#f3ca56'),
       ('X','Сундук; содержимое в README.md',COLORS['chest']),('T / M','Торговец / магическая лавка',COLORS['merchant']),
       ('H / U','Наёмники / тренер',COLORS['trainer']),('W / S','Будущее появление / условный штурм',COLORS['future'])]
    for i,(code,desc,color) in enumerate(legend):
        col=i%2;row=i//2;xx=margin+col*850;yy=sy+row*39
        text(xx,yy,code,font(24,True),color);text(xx+102,yy,desc,font(22),MUTED)
    sy+=180
    text(margin,sy,'Мана:  П — преисподняя   Ж — жизнь   С — смерть   Р — руны   Л — роща',font(23),MUTED)
    sy+=42
    for j,(kind,title) in enumerate([('forest','Лес'),('water','Вода'),('road','Дорога'),('obstacle','Препятствия')]):
        xx=margin+j*310;d.rectangle((xx,sy+3,xx+23,sy+26),fill=COLORS[kind]);text(xx+37,sy,title,small,MUTED)
    sy+=55
    for line in wrap(f"Старт: {data['faction']} · {data['starting_gold']:g} золота · {data['starting_army']}",small,map_size):
        text(margin,sy,line,small,MUTED);sy+=29
    roster_note='заданный отряд карты' if data['starting_roster_mode']=='fixed' else 'обычный ростер'
    text(margin,sy+8,'Снимок reset(seed=0) · '+roster_note+' · scripted bot выключен',small,MUTED)
    text(sidebar_x,top,'ОТРЯДЫ НА СТАРТЕ',font(30,True))
    for ci,g in enumerate(groups):
        yy=top+60;xx=sidebar_x+ci*col_width
        for code,lines,color in g:
            text(xx,yy,code,bold,color)
            for j,line in enumerate(lines):text(xx+58,yy+j*31,line,f,'#d5dfeb')
            yy+=max(42,len(lines)*31+14)
    yy=top+60+roster_h
    if not entries:text(sidebar_x,yy,'Активных полевых отрядов нет.',f,MUTED);yy+=45
    yy+=36
    if info_lines:
        text(sidebar_x,yy,'ГОРОДА И ОБЪЕКТЫ',font(26,True));yy+=46
        for line in info_lines:text(sidebar_x,yy,line,small,MUTED);yy+=30
    yy+=36
    for line in notes_lines:text(sidebar_x,yy,line,small,MUTED);yy+=30
    if data['map_id']=='mirrow_match':
        yy+=30;text(sidebar_x,yy,'Крупное существо: 2 клетки, 1 боец.',small,MUTED)
    im.save(destination,optimize=True)
    return im


def draw_waves(data,path):
    future=[e for e in data['enemies'] if e['wave']]
    if not future and not data['conditional_assaults']:return
    entries=[]
    positions=sorted({tuple(e['position']) for e in future})
    for e in future:
        w=e['wave'];code='W'+str(positions.index(tuple(e['position']))+1)
        entries.append(f"{code} · отряд {e['id']} · волна {w['wave_index']} · ход {w['spawn_turn']} · {w['moves_per_turn']} очков хода/ход\n{e['summary']}")
    for a in data['conditional_assaults']:
        entries.append(f"S · штурм {a['wave']} · {a['faction']} · из {tuple(a['position'])}\n"+' + '.join(a['roster']))
    f=font(25);width=1780;lines=[]
    for entry in entries:lines.append(wrap(entry,f,width-150))
    height=210+sum(len(row)*36+32 for row in lines)+100
    im=Image.new('RGB',(width,height),BG);d=ImageDraw.Draw(im)
    d.text((70,48),data['title']+' · будущие волны',font=font(37,True),fill=FG)
    d.text((70,110),'Эти отряды не показаны активными на стартовой схеме.',font=font(25),fill=MUTED)
    y=190
    for row in lines:
        for j,line in enumerate(row):d.text((70,y),line,font=f,fill=COLORS['future'] if j==0 else FG);y+=36
        y+=32
    im.save(path,optimize=True)


def write_readme(data,path):
    def coord(p):return f"({p[0]}, {p[1]})"
    out=[f"# {data['title']}", '', '![Схема карты](layout.png)', '',
         f"Карта: `{data['map_id']}`. Игровая сетка: **{data['grid_size']} × {data['grid_size']}**.",
         'Координаты `(x, y)`: x вправо, y вниз, отсчёт от нуля.',
         'Снимок реальной среды после `reset(seed=0)`: `local5`, `use_boss_starting_roster=False`, scripted bot выключен. Закреплённый картой отряд сохраняется.',
         'Это документация карты; генератор не запускает обучение, оценку или replay.',
         '','## Старт',f"- Фракция: {data['faction']}",f"- Герой: {coord(data['hero'])}",
         f"- Золото: {data['starting_gold']:g}",f"- Отряд: {data['starting_army']}",
         f"- Цель: {OBJECTIVES.get(data['objective'],data['objective'])}",
         '','## Обозначения','- A / B: столица игрока / вражеская столица; треугольник — старт героя.',
         '- C: город. T: торговец. M: магическая лавка. H: наёмники. U: тренер.',
         '- Точки возле магазина показывают все действующие клетки взаимодействия.',
         '- X: сундук; ромб: шахта. П/Ж/С/Р/Л: преисподняя/жизнь/смерть/руны/роща.',
         '- Круг: отряд. Зелёный: полевой; оранжевый: город; фиолетовый: руины; красный: столица.',
         '- Для Mirrow match цвета — заданные уровни сложности; номер общий для зеркальной пары.',
         '- W: место будущего появления; S: условный штурм. Знак + после номера: несколько армий на одной клетке.',
         '- Большое существо занимает 2 клетки, но считается одним бойцом.', '', '## Города и объекты']
    for c in data['capitals']:out.append(f"- {c['code']}: {c['name']}, {coord(c['position'])}")
    for c in data['cities']:
        out.append(f"- {c['code']}: {c['name']}, {coord(c['position'])}, уровень {c['level']}, отряды {c['enemy_ids'] or 'нет'}"+(' — цель кампании' if c['objective'] else '')+(' — принадлежит игроку' if c['owned'] else ''))
    for s in data['sites']:out.append(f"- {s['code']}: {s['name']}, якорь {coord(s['position'])}; взаимодействие: "+', '.join(coord(t) for t in s['interaction_tiles']))
    out+=['','## Отряды','| № на схеме | enemy_id | Координаты | Статус | Состав | Бойцов / клеток |','|---|---:|---|---|---|---:|']
    for e in data['enemies']:
        status={'future':'Будущая волна','ruin':'Руины','city':'Город','capital':'Столица','field':'Полевой'}[e['kind']]
        if e['wave']:status+=f", ход {e['wave']['spawn_turn']}"
        out.append(f"| {e['marker']} | {e['id']} | {coord(e['position'])} | {status} | {e['summary']} | {e['fighter_count']} / {e['formation_cells']} |")
    out+=['','## Ресурсы и сундуки']
    for i,p in enumerate(data['gold_mines'],1):out.append(f'- Шахта {i}: {coord(p)}')
    for m in data['mana']:out.append(f"- {m['name']}: {coord(m['position'])}")
    for c in data['chests']:out.append(f"- {c['code']}: {coord(c['position'])} — "+', '.join(c['items']))
    for r in data['ruins']:
        items=r.get('items',[]) or ([r['item']] if r.get('item') else [])
        out.append(f"- Руины {r['id']} «{r['title']}»: {r.get('gold',0)} золота; "+(', '.join(items) or 'без предметов'))
    out+=['','## Примечания']+['- '+note for note in data['notes']]
    if any(e['wave'] for e in data['enemies']) or data['conditional_assaults']:
        out+=['','![Расписание волн](waves.png)']
    out+=['','## Обновление','Из папки `Big_map`: `python tools/generate_map_layouts.py --map '+data['map_id']+'`.',
         'Точные данные снимка и все координаты сохранены в `layout.json`.',
         'В этой папке намеренно нет `__init__.py`: игровые импорты продолжают использовать соседний `'+data['map_id']+'.py`.','']
    path.write_text('\n'.join(out),encoding='utf-8')


def generate(names,out_root):
    results=[]
    for name in names:
        data=extract(name);folder=out_root/name;folder.mkdir(parents=True,exist_ok=True)
        draw(data,folder/'layout.png');draw_waves(data,folder/'waves.png')
        (folder/'layout.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        write_readme(data,folder/'README.md')
        results.append(data)
        print(f"{name}: {data['grid_size']}x{data['grid_size']}, {len(data['enemies'])} armies, saved {folder/'layout.png'}",flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map',action='append',choices=available_maps(),dest='maps')
    parser.add_argument('--output',type=Path,default=ROOT/'maps')
    args=parser.parse_args();generate(args.maps or list(available_maps()),args.output)

if __name__=='__main__':main()
