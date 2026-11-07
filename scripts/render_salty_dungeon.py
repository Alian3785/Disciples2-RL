"""Render the current S103 JAX layout in the style of the earlier S101 diagram.

Also audit the dragon bottleneck with an eight-neighbour graph, reproducing
TerritoryRules footprints/mines and ZoneRules' fort exceptions. No JAX runtime.
Other armies may be fought; this audit isolates whether the green dragon can
remain alive while the player collects the key and reaches the finish.
"""
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from render_bethrezens_retreat import BG, FG, MUTED, COLORS, RESOURCES, font, wrap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'docs/maps/the_salty_dungeon'
DIRECTIONS = ((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1))
ROUTE = '#efce76'
DANGER = '#ff7f89'


def audit(game_map):
    blocked = {tuple(p) for p in game_map['obstacles']}
    blocked.update(tuple(m['position']) for m in game_map['mines'])
    for obj in game_map['ruins']+game_map['merchants']+game_map['spell_shops']:
        r, c = obj['position']
        blocked.update((r+dr,c+dc) for dr in range(3) for dc in range(3))
    index = game_map['reference_enemy_ids'].index('S103KC002B')
    dragon = tuple(game_map['opponent_positions'][index])
    forts = {tuple(game_map['capital_position']), *(tuple(c['position']) for c in game_map['cities'])}
    zone = {(dragon[0]+dr, dragon[1]+dc) for dr in (-1,0,1) for dc in (-1,0,1)}-forts
    zone.add(dragon)
    key = next(r for r in game_map['ruins'] if r['items'].get('runic_key'))
    approaches = {(key['position'][0]+3,key['entrance'][1]+dc) for dc in (-1,0,1)}-blocked
    r,c = game_map['objective_position']
    radius = game_map['objective_radius']
    goals = {(r+dr,c+dc) for dr in range(-radius,radius+1) for dc in range(-radius,radius+1)}-blocked

    def search(start, targets, forbidden):
        start = tuple(start)
        parents, queue = {start: None}, deque([start])
        while queue:
            point = queue.popleft()
            if point in targets:
                path = []
                while point is not None:
                    path.append(point)
                    point = parents[point]
                return path[::-1], set(parents)
            for dr,dc in DIRECTIONS:
                other = point[0]+dr,point[1]+dc
                if (all(0 < v < game_map['size']-1 for v in other)
                        and other not in forbidden and other not in parents):
                    parents[other] = point
                    queue.append(other)
        return None, set(parents)

    to_key, _ = search(game_map['agent_position'], approaches, blocked|zone)
    if to_key is None:
        raise ValueError('Key unexpectedly unreachable without fighting the dragon')
    alive_path, region = search(to_key[-1], goals, blocked|zone)
    dead_path, _ = search(to_key[-1], goals, blocked)
    cell_only_path, _ = search(to_key[-1], goals, blocked|{dragon})
    if alive_path is not None or dead_path is None:
        raise ValueError('Dragon bottleneck differs from the verified map geometry')
    for route, excluded in ((to_key,blocked|zone),(dead_path,blocked)):
        if any(tuple(p) in excluded for p in route):
            raise ValueError('Route crosses a blocked tile')
        if any(max(abs(a-b) for a,b in zip(p,q)) != 1 for p,q in zip(route,route[1:])):
            raise ValueError('Non-adjacent route steps')
    result = dict(map_sha256=hashlib.sha256((ROOT/'maps/the_salty_dungeon.json').read_bytes()).hexdigest(),
        coordinates='JAX [row,column]; original (x,y)=(column-1,row-1)',
        scope='Static topology; other armies may be defeated. No claim about tactical success or magic kills.',
        dragon_source_id='S103KC002B', dragon_enemy_index=index, dragon_position=dragon,
        dragon_zone=sorted(zone), key_approaches=sorted(approaches), goal_tiles=sorted(goals),
        key_reachable_with_dragon_alive=True, goal_reachable_with_dragon_alive=alive_path is not None,
        goal_reachable_after_dragon_defeat=dead_path is not None,
        blocking_only_dragon_tile_is_insufficient=cell_only_path is not None,
        reachable_cells_with_dragon_alive=len(region),
        start_to_key_steps=len(to_key)-1, key_to_goal_steps_after_defeat=len(dead_path)-1,
        start_to_key_path=to_key, key_to_goal_path=dead_path,
        rules_sha256={name:hashlib.sha256((ROOT/'stoix/envs'/name).read_bytes()).hexdigest()
            for name in ('number_grid_territory.py','number_grid_ruins.py','number_grid_zones.py')})
    return result, blocked, region


def render(game_map, source, report, blocked):
    units = json.loads((ROOT/'stoix/envs/data/units.json').read_text(encoding='utf-8'))
    source_armies = {a['source_id']:a for a in source['armies']}
    buildings, codes = [], {}
    ncity = 0
    for fort in source['forts']:
        if fort['kind'] == 'city':
            ncity += 1
            code = 'C'+str(ncity)
        else:
            code = 'A' if fort['owner'] == source['player'] else 'B'
        buildings.append(dict(fort,code=code))
        codes[fort['source_id']] = code
    for i,ruin in enumerate(source['ruins'],1):
        buildings.append(dict(ruin,code='R'+str(i),kind='ruin',size=3))
        codes[ruin['source_id']] = 'R'+str(i)
    shops = []
    counts = Counter()
    for site in source['sites']:
        counts[site['kind']] += 1
        shops.append(dict(site,code=('T' if site['kind']=='merchant' else 'M')+str(counts[site['kind']])))
    armies = []
    normal, small, bold = font(25),font(23),font(26,True)
    for index,gid in enumerate(game_map['reference_enemy_ids']):
        army = source_armies[gid]
        group = Counter((units[key]['name'],level) for key,level in zip(
            game_map['enemy_rosters'][index],game_map['enemy_initial_levels'][index]) if key)
        summary = ' + '.join((str(count)+' × ' if count>1 else '')+name+
            (' (ур. '+str(level)+')' if level>1 else '') for (name,level),count in group.items())
        code = codes.get(army.get('building'))
        if code:
            summary += ' · '+code
        if gid == 'S103KC002B':
            summary += ' · обязательный проход'
        lines = wrap(summary,normal,628)
        armies.append(dict(number=index+1,source_id=gid,kind=army['kind'],summary=summary,
                           lines=lines,height=max(48,len(lines)*32+14)))
    target,used,split = sum(a['height'] for a in armies)/2,0,0
    while used<target:
        used += armies[split]['height']
        split += 1
    columns = armies[:split],armies[split:]
    margin,top,map_px,col_width = 86,220,2160,710
    side = margin+map_px+90
    width = side+2*col_width+86
    height = max(top+map_px+1490,top+100+max(sum(a['height'] for a in col) for col in columns)+90)
    im = Image.new('RGB',(width,height),BG)
    draw = ImageDraw.Draw(im)

    def text(x,y,value,ft=normal,color=FG):
        bounds = draw.textbbox((x,y),value,font=ft)
        if bounds[2]>width-20 or bounds[3]>height-20:
            raise ValueError('Text outside canvas: '+value)
        draw.text((x,y),value,font=ft,fill=color)

    text(margin,40,'Соляная темница',font(58,True))
    text(margin,112,'The Salty Dungeon · Легионы Проклятых, миссия 4 · S103 · текущая карта JAX',font(29),MUTED)
    text(margin,164,'72 × 72 · 97 вражеских отрядов · 8 городов · 7 руин · 37 тайников · 14 шахт',font(27),MUTED)
    cell = map_px/72
    def center(p):
        return margin+(p[1]-.5)*cell,top+(p[0]-.5)*cell
    def rectangle(p,size,color,outline=None,line=1):
        x,y = margin+(p[1]-1)*cell,top+(p[0]-1)*cell
        draw.rectangle((x,y,x+size[1]*cell,y+size[0]*cell),fill=color,outline=outline,width=line)
    terrain = {tuple(p):kind for kind,points in game_map['terrain'].items() for p in points}
    for r in range(1,73):
        for c in range(1,73):
            kind = 'mountain' if (r,c) in blocked else terrain.get((r,c),'plain')
            rectangle((r,c),(1,1),COLORS[kind])
    for b in buildings:
        x,y = b['position']
        color = '#b85869' if b['code']=='A' else '#598cd9' if b['code']=='B' else '#8d7449' if b['kind']=='city' else '#675282'
        rectangle((y+1,x+1),(b['size'],b['size']),color)
    for shop in shops:
        x,y = shop['position']
        rectangle((y+1,x+1),(3,3),'#615747' if shop['kind']=='merchant' else '#56496a')
    for i in range(73):
        x,y = margin+i*cell,top+i*cell
        draw.line((x,top,x,top+map_px),fill='#29384b')
        draw.line((margin,y,margin+map_px,y),fill='#29384b')
    for i in range(0,72,4):
        x,y = center((i+1,i+1))
        draw.text((x,top+map_px+16),str(i),font=small,fill=MUTED,anchor='mt')
        draw.text((margin-14,y),str(i),font=small,fill=MUTED,anchor='rm')
    text(margin-50,top-36,'y ↓',small,MUTED)
    text(margin+map_px-22,top+map_px+43,'x',small,MUTED)
    for route,color in ((report['start_to_key_path'],COLORS['friendly']),(report['key_to_goal_path'],ROUTE)):
        points = [center(p) for p in route]
        draw.line(points,fill=color,width=4)
    for p in report['dragon_zone']:
        if tuple(p) not in blocked:
            rectangle(p,(1,1),None,DANGER,3)

    def marker(p,label,color,shape='circle',radius=14):
        x,y = center(p)
        box = x-radius,y-radius,x+radius,y+radius
        if shape=='box':
            draw.rectangle(box,fill=color,outline=BG,width=2)
        elif shape=='diamond':
            draw.polygon(((x,y-radius),(x+radius,y),(x,y+radius),(x-radius,y)),fill=color,outline=BG)
        elif shape=='ring':
            draw.ellipse(box,fill=BG,outline=color,width=3)
        else:
            draw.ellipse(box,fill=color,outline=BG,width=2)
        if label:
            draw.text((x,y),label,font=font(17 if len(label)<=2 else 13,True),fill=color if shape=='ring' else BG,anchor='mm')
    for resource in source['resources']:
        _,label,_,color = RESOURCES[resource['resource']]
        x,y = resource['position']
        marker((y+1,x+1),label,color,'diamond' if resource['resource']==0 else 'box',11)
    for chest in game_map['chests']:
        marker(chest['position'],'X',COLORS['chest'],'box',10)
    for b in buildings:
        x,y = b['position']
        draw.text(center((y+1+(b['size']-1)/2,x+1+(b['size']-1)/2)),b['code'],font=font(25,True),fill=FG,anchor='mm')
    for shop in shops:
        x,y = shop['position']
        marker((y+2,x+2),shop['code'],COLORS[shop['kind']],'box',18)
    groups = defaultdict(list)
    for index,position in enumerate(game_map['opponent_positions']):
        groups[tuple(position)].append(armies[index])
    for p,group in groups.items():
        army = group[0]
        color = DANGER if army['source_id']=='S103KC002B' else COLORS[army['kind']]
        marker(p,str(army['number'])+('+' if len(group)>1 else ''),color)
    marker(game_map['agent_position'],'H',COLORS['friendly'],'ring',20)
    marker(game_map['objective_position'],'G',COLORS['goal'],'diamond',20)
    key = next(r for r in game_map['ruins'] if r['items'].get('runic_key'))
    marker((key['position'][0]+1,key['position'][1]+1),'K',ROUTE,'diamond',19)
    for event,label in (('S103EV000D','D'),('S103EV0014','Z')):
        location = next(e for e in game_map['scenario_events'] if e['id']==event)['conditions'][0]
        marker(location['position'],label,COLORS['event'],'ring',18)

    y = top+map_px+92
    legend = [('H / A','Герой / столица Легионов',COLORS['friendly']),('B','Столица Горных Кланов',COLORS['capital']),
        ('C / R','Город / руины; гарнизоны пронумерованы',COLORS['city']),('T / M','Торговец / башня магии',COLORS['mage']),
        ('K / G','Рунический ключ / место доставки',ROUTE),('D / Z','Активация гномов / засада нежити',COLORS['event']),
        ('40','Зелёный дракон и его зона контроля',DANGER),('X / +','Тайник / несколько отрядов в объекте',COLORS['chest'])]
    for i,(code,label,color) in enumerate(legend):
        xx,yy = margin+i%2*1060,y+i//2*42
        text(xx,yy,code,bold,color)
        text(xx+130,yy,label,small,MUTED)
    y += 190
    for i,(kind,label) in enumerate((('forest','Лес'),('water','Вода'),('road','Дорога'),('mountain','Непроходимо'))):
        xx = margin+i*500
        draw.rectangle((xx,y+3,xx+24,y+27),fill=COLORS[kind])
        text(xx+38,y,label,small,MUTED)
    y += 52
    text(margin,y,'ПУТЬ К ЦЕЛИ И ПРОХОД ДРАКОНА',font(28,True))
    y += 48
    notes = [
        'Бирюзовая линия: старт → подход к руинам с ключом. Золотая: руины → финиш после устранения дракона.',
        'Линии показывают геометрию маршрута; другие встреченные отряды тоже могут потребовать боя.',
        'Живой отряд №40, (43,56), перекрывает путь к финишу зоной 3 × 3. Одну клетку дракона обойти можно.',
        'Ключ можно получить до дракона. В руинах R6 нужно победить охрану; затем доставить ключ к G, (39,10), радиус 1.',
        'Проверено: с зоной дракона финиш недоступен; после её удаления — доступен. Ограничение действует для пути, а не как отдельный флаг победы.',
        'H: Герцог ур.4, Одержимый, Сектант и Мраморная гаргулья ур.2. Столицу без героя защищает Ашкаэль.',
        'Координаты исходные: x вправо, y вниз, от нуля. Номера отрядов совпадают с интерфейсом JAX.',
    ]
    for note in notes:
        for line in wrap(note,small,map_px):
            text(margin,y,line,small,MUTED)
            y += 32
        y += 6
    y += 24
    text(margin,y,'ГОРОДА И ОБЪЕКТЫ',font(28,True))
    y += 48
    rows = []
    for b in buildings+shops:
        label = b['code']+'  '+b['name']
        if b.get('level'):
            label += ' · ур. '+str(b['level'])
        guards = [str(a['number']) for a in armies if source_armies[a['source_id']].get('building')==b['source_id']]
        if guards:
            label += ' · '+', '.join(guards)
        if b.get('source_id')=='S103RU0005':
            label += ' · КЛЮЧ'
        rows.append(label)
    half = (len(rows)+1)//2
    for i,label in enumerate(rows):
        text(margin+(i//half)*1060,y+(i%half)*34,label,small,MUTED)
    y += half*34+40
    text(margin,y,'Источник: maps/the_salty_dungeon.json + снимок Scens/Globals S103; 07.11.2026.',small,MUTED)
    text(margin,y+34,'Показаны стартовые позиции текущей JAX-версии; активируемые отряды позднее перемещаются.',small,MUTED)
    text(side,top,'ВРАЖЕСКИЕ ОТРЯДЫ И ГАРНИЗОНЫ JAX',font(29,True))
    text(side,top+45,'Номер → состав · объект. Крупный юнит считается один раз.',small,MUTED)
    for col,entries in enumerate(columns):
        yy = top+100
        for army in entries:
            xx = side+col*col_width
            color = DANGER if army['source_id']=='S103KC002B' else COLORS[army['kind']]
            text(xx,yy,str(army['number']),bold,color)
            for i,line in enumerate(army['lines']):
                text(xx+65,yy+i*32,line,normal,'#d5dfeb')
            yy += army['height']
    im.save(OUTPUT/'layout.png',optimize=True)
    # Full-resolution crop of the actual pass, magnified with nearest pixels.
    left,upper,right,lower = 34,47,53,65
    crop = im.crop((margin+left*cell,top+upper*cell,margin+right*cell,top+lower*cell)).resize((1140,1080),Image.Resampling.NEAREST)
    detail = Image.new('RGB',(1300,1270),BG)
    detail.paste(crop,(80,110))
    d = ImageDraw.Draw(detail)
    d.text((80,25),'Проход зелёного дракона · отряд №40',font=font(36,True),fill=FG)
    d.text((80,1210),'Красные клетки — зона контроля; золотая линия — путь после победы.',font=font(25),fill=MUTED)
    detail.save(OUTPUT/'dragon-pass.png',optimize=True)
    return im.size


def main():
    game_map = json.loads((ROOT/'maps/the_salty_dungeon.json').read_text(encoding='utf-8'))
    source = json.loads((OUTPUT/'source.json').read_text(encoding='utf-8'))
    report,blocked,_ = audit(game_map)
    report_file = ROOT/'results/the-salty-dungeon-route-audit-20261107.json'
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    size = render(game_map,source,report,blocked)
    print(json.dumps(dict(layout=str(OUTPUT/'layout.png'),size=size,report=str(report_file),
                         key_steps=report['start_to_key_steps'],goal_steps=report['key_to_goal_steps_after_defeat']),ensure_ascii=False))


if __name__=='__main__':
    main()
