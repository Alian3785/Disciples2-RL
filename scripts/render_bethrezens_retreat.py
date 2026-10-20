"""Render S101 from installed Rise of the Elves DBFs, without running a game/env.

Style follows Big_map/tools/generate_map_layouts.py in the Python reference.
Run: python scripts/render_bethrezens_retreat.py --game "path/to/game"
Outputs: docs/maps/bethrezens_retreat/{layout.png,events.png,layout.json}.
This is an illustration of source data before turn-one events, not a JAX map.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

from PIL import Image, ImageDraw, ImageFont

from inspect_unit_sources import dbf

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "S101SC0000"
NULL_ID = "G000000000"
PLAYER = "S101PL0001"
BETHREZEN = "S101KC0013"
BG, FG, MUTED = "#0d1524", "#eff3f8", "#9cacc1"
COLORS = {
    "plain": "#182637", "forest": "#355d49", "water": "#2d6e94",
    "mountain": "#465063", "landmark": "#3c4657", "road": "#756a5b",
    "field": "#add771", "city": "#f1b25c", "ruin": "#c397ef",
    "capital": "#ed8796", "friendly": "#78dfcf", "event": "#7dd9e1",
    "chest": "#ffd789", "merchant": "#f6c175", "mage": "#cca7f7",
    "gold": "#f3ca56", "goal": "#f1e6a6",
}
RESOURCES = {
    0: ("gold", "", "Золото", COLORS["gold"]),
    1: ("infernal", "П", "Мана преисподней", "#d87392"),
    2: ("life", "Ж", "Мана жизни", "#83acef"),
    3: ("death", "С", "Мана смерти", "#d6d8de"),
    4: ("runes", "Р", "Мана рун", "#8de0df"),
    5: ("grove", "Л", "Мана рощи", "#8acb86"),
}


def pos(row):
    return [int(row["POS_X"]), int(row["POS_Y"])]


def decode_block(encoded):
    """DBF binary fields use six bits/character, ASCII 33 offset, MSB first.

    171 characters encode 128 bytes = 32 little-endian tile words. The same
    decoding reproduces the existing S117 reference snapshot's tile codes.
    """
    acc = bits = 0
    result = bytearray()
    for char in encoded:
        value = ord(char) - 33
        if not 0 <= value < 64:
            raise ValueError("Invalid DBF binary character")
        acc = (acc << 6) | value
        bits += 6
        if bits >= 8:
            bits -= 8
            result.append((acc >> bits) & 255)
            acc &= (1 << bits) - 1
    if len(result) != 128:
        raise ValueError("Terrain block must contain 128 bytes")
    return struct.unpack("<32I", result)


def extract(game):
    files = {str(p.relative_to(game)).replace("\\", "/").lower(): p
             for folder in ("Scens", "Globals") for p in (game / folder).iterdir()
             if p.suffix.lower() == ".dbf"}
    hashes = {}
    raw = {}

    def read(name):
        path = files[name.lower()]
        hashes[str(path.relative_to(game)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = dbf(path)
        if name.lower().startswith("scens/"):
            rows = [r for r in rows if r["NO_SCEN"] == SCENARIO]
            raw[path.name] = rows
        return rows

    tables = {name: read("Scens/" + name + ".dbf") for name in (
        "Sinfo", "SMapBlck", "SMounts", "SLmark", "SRoad", "SObjMap", "SFort",
        "Sstack", "SUGroup", "SUnit", "SCrys", "SBag", "SRuin", "SSite", "Splayer",
        "SInventr", "SItem", "SMerch", "SMage", "Tscen", "SEvent", "SEvCond",
        "SEvEffec", "SLoc", "SStkTemp",
    )}
    info = tables["Sinfo"][0]
    size = int(info["MAP_SIZE"])
    global_text = {r["TXT_ID"].lower(): r["TEXT"] for r in read("Globals/Tglobal.dbf")}
    profiles = {r["UNIT_ID"].lower(): r for r in read("Globals/Gunits.dbf")}
    items = {r["ITEM_ID"].lower(): r for r in read("Globals/GItem.dbf")}
    landmarks = {r["LMARK_ID"].lower(): r for r in read("Globals/GLmark.dbf")}
    # Preserve the original category dictionaries beside the raw event records.
    for name in ("LEvCond", "LEvEffct", "Lground", "Lterrain", "Lres"):
        raw[name] = read("Globals/" + name + ".dbf")
    catalog_path = ROOT / "stoix/envs/data/units.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    unit_names = {r["game_id"].lower(): r["name"] for r in catalog.values()}
    text_parts = defaultdict(list)
    for r in tables["Tscen"]:
        text_parts[r["ID_OBJECT"], r["ID_STRING"]].append((int(r["SEQ_NUM"]), r["TEXT"]))
    texts = {key: "".join(s for _, s in sorted(parts)) for key, parts in text_parts.items()}

    def title(source_id):
        return next((texts[source_id, k] for k in ("NAME_TXT", "TXT_TITLE", "TITLE")
                     if texts.get((source_id, k))), source_id)

    def item_name(item_type):
        return global_text[items[item_type.lower()]["NAME_TXT"].lower()]

    item_instances = {r["ITEM_ID"]: r["ITEM_TYPE"] for r in tables["SItem"]}
    inventories = defaultdict(list)
    for r in tables["SInventr"]:
        inventories[r["INV_ID"]].append(item_name(item_instances[r["ITEM_ID"]]))

    grid = [[None] * size for _ in range(size)]
    for r in tables["SMapBlck"]:
        by, bx = int(r["BLOCKID"][-4:-2], 16), int(r["BLOCKID"][-2:], 16)
        for i, word in enumerate(decode_block(r["BLOCKDATA"])):
            x, y = bx + i % 8, by + i // 8
            if grid[y][x] is not None:
                raise ValueError("Overlapping terrain blocks")
            grid[y][x] = word & 255
    if any(v is None for row in grid for v in row):
        raise ValueError("Incomplete terrain")

    classes = {r["OBJ_ID"]: r["WHAT"] for r in tables["SObjMap"]}
    forts = []
    city_number = 0
    for r in tables["SFort"]:
        capital = classes[r["CITY_ID"]] == ".?AVCCapital@@"
        if not capital:
            city_number += 1
        code = ("A" if r["OWNER"] == PLAYER else "B") if capital else f"C{city_number}"
        x, y = pos(r)
        forts.append(dict(source_id=r["CITY_ID"], code=code, name=title(r["CITY_ID"]),
                          position=[x, y], marker_position=[x + (4 if capital else 3), y + (2 if capital else 3)],
                          size=5 if capital else 4, level=int(r["SIZE"] or 0),
                          kind="capital" if capital else "city", owner=r["OWNER"]))
    ruins = [dict(source_id=r["RUIN_ID"], code=f"R{i}", name=title(r["RUIN_ID"]),
                  position=pos(r), marker_position=[int(r["POS_X"]) + 1, int(r["POS_Y"]) + 2],
                  size=3, kind="ruin", gold=int(r["CASH"][1:5]),
                  items=[] if r["ITEM"] == NULL_ID else [item_name(r["ITEM"])])
             for i, r in enumerate(tables["SRuin"], 1)]
    buildings = {b["source_id"]: b for b in forts + ruins}
    stacks = {r["STACK_ID"]: r for r in tables["Sstack"]}
    units = {r["UNIT_ID"]: r for r in tables["SUnit"]}

    def describe_unit(unit_type, level, source_id=None, cells=None):
        profile = profiles[unit_type.lower()]
        name = unit_names[unit_type.lower()]
        custom = title(source_id) if source_id else ""
        if source_id == "S101UN0064":
            name = "Бетрезен"
        return dict(source_id=source_id, game_id=unit_type, name=name, level=int(level),
                    custom_name=custom if custom != source_id else "",
                    big=profile["SIZE_SMALL"] != "T", cells=cells or [])

    def summary(roster):
        counts = Counter((u["name"], u["level"]) for u in roster)
        return " + ".join((f"{n} × " if n > 1 else "") + name +
                          (f" (ур. {level})" if level > 1 else "")
                          for (name, level), n in counts.items())

    armies = []
    for r in sorted(tables["SUGroup"], key=lambda r: r["GROUP_ID"]):
        uid_cells = defaultdict(list)
        for cell in range(6):
            index = int(r[f"POS_{cell}"])
            if index >= 0:
                uid = r[f"UNIT_{index}"]
                if uid == NULL_ID:
                    raise ValueError("Formation refers to an empty unit")
                uid_cells[uid].append(cell)
        declared = {r[f"UNIT_{i}"] for i in range(6)} - {NULL_ID}
        if declared != set(uid_cells):
            raise ValueError("Formation unit IDs differ from declared unit IDs")
        if not declared:
            continue
        roster = [describe_unit(units[uid]["TYPE"], units[uid]["LEVEL"], uid, cells)
                  for uid, cells in uid_cells.items()]
        if sum(2 if u["big"] else 1 for u in roster) > 6:
            raise ValueError("Formation capacity exceeded")
        source_id = r["GROUP_ID"]
        stack = stacks.get(source_id)
        building = buildings.get(stack["INSIDE"] if stack else source_id)
        source_position = pos(stack) if stack else building["position"]
        owner = stack["OWNER"] if stack else building.get("owner", "")
        kind = building["kind"] if building else "field"
        armies.append(dict(number=len(armies) + 1, source_id=source_id,
                           source_position=source_position,
                           marker_position=building["marker_position"] if building else source_position,
                           kind=kind, owner=owner, building=building["code"] if building else None,
                           units=roster, summary=summary(roster), inventory=inventories[source_id],
                           raw_formation=r))
    for building in buildings.values():
        building["armies"] = [a["number"] for a in armies if a["building"] == building["code"]]

    sites = []
    site_counts = Counter()
    for r in tables["SSite"]:
        kind = {".?AVCMidSiteMerchant@@": "merchant", ".?AVCMidSiteMage@@": "mage"}[classes[r["SITE_ID"]]]
        site_counts[kind] += 1
        sites.append(dict(source_id=r["SITE_ID"], position=pos(r), name=title(r["SITE_ID"]),
                          kind=kind, code=("T" if kind == "merchant" else "M") + str(site_counts[kind]),
                          items=[dict(name=item_name(v["ITEM_ID"]), count=int(v["ITEM_COUNT"]))
                                 for v in tables["SMerch"] if v["SITE_ID"] == r["SITE_ID"]]))
    locations = {r["LOC_ID"]: dict(source_id=r["LOC_ID"], position=pos(r), radius=int(r["RADIUS"]))
                 for r in tables["SLoc"]}
    templates = {r["ID"]: r for r in tables["SStkTemp"]}
    appearances = []
    spawn_codes = {}
    for effect in tables["SEvEffec"]:
        if effect["CATEGORY"] != "1":
            continue
        loc_id = effect["ID_LOC"]
        spawn_codes.setdefault(loc_id, "W" + str(len(spawn_codes) + 1))
        template = templates[effect["ID_STKTEMP"]]
        # Templates store global types in their cells (CMidStackTemplate), not
        # unique scenario-unit instances. Repeated small types remain separate.
        roster = []
        for position in range(6):
            index = int(template[f"POS_{position}"])
            if index < 0:
                continue
            unit = describe_unit(template[f"UNIT_{index}"], template[f"UNIT_{index}_LVL"])
            if unit['big'] and position % 2:
                continue
            roster.append(unit)
        appearances.append(dict(code=spawn_codes[loc_id], location=locations[loc_id],
                                event=effect["ID"], template=effect["ID_STKTEMP"],
                                units=roster, summary=summary(roster)))
    goal_effect = next(r for r in tables["SEvEffec"] if r["CATEGORY"] == "0" and r["WIN_SCEN"] == "T")
    goal_condition = next(r for r in tables["SEvCond"] if r["ID"] == goal_effect["ID"])
    goal = dict(locations[goal_condition["ID_LOC"]], event=goal_effect["ID"])
    ambush_condition = next(r for r in tables["SEvCond"] if r["ID"] == "S101EV0008")
    moves = []
    for effect in tables["SEvEffec"]:
        if effect["CATEGORY"] == "10" and any(c["ID"] == effect["ID"] and c["FREQUENCY"] == "1"
                                               for c in tables["SEvCond"]):
            army = next(a for a in armies if a["source_id"] == effect["ID_STKTEMP"])
            moves.append(dict(number=army["number"], event=effect["ID"],
                              source_position=army["source_position"], location=locations[effect["ID_LOC"]]))
    return dict(schema_version=1, scenario=info, map_id="bethrezens_retreat", size=size,
                source=dict(game=str(game), sha256=hashes,
                            names_catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                            snapshot="Original DBF positions before turn-one scripted events; no runtime initialization",
                            style_reference="C:/Disciples 2 python reference/Big_map/tools/generate_map_layouts.py",
                            terrain_encoding="6-bit ASCII-33 MSB-first; 32 little-endian words per 8x4 block",
                            marker_note="Buildings use schematic entrance offsets: capital (4,2), city (3,3), ruin (1,2). These are display coordinates, not pathfinding rules."),
                terrain=grid, roads=[pos(r) for r in tables["SRoad"]],
                landmarks=[dict(position=pos(r), size=[int(landmarks[r["TYPE"].lower()][k]) for k in ("CX", "CY")])
                           for r in tables["SLmark"]],
                forts=forts, ruins=ruins, sites=sites, armies=armies,
                resources=[dict(position=pos(r), resource=int(r["RESOURCE"])) for r in tables["SCrys"]],
                chests=[dict(code=f"X{i}", position=pos(r), items=inventories[r["BAG_ID"]])
                        for i, r in enumerate(tables["SBag"], 1)],
                appearances=appearances, goal=goal,
                ambush=locations[ambush_condition["ID_LOC"]], intro_moves=moves, raw_tables=raw)


def font(size, bold=False):
    for path in (["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
                 if bold else ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("A Cyrillic TrueType font is required")


def wrap(text, ft, width):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if line and ft.getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def army_color(army):
    return COLORS["friendly"] if army["owner"] == PLAYER else COLORS[army["kind"]]


def render(data, output):
    margin, top, map_px = 86, 220, 2160
    side, col_width = margin + map_px + 90, 710
    width = side + 2 * col_width + 86
    normal, small, bold = font(25), font(23), font(26, True)
    prepared = []
    for a in data["armies"]:
        suffix = f" · {a['building']}" if a["building"] else ""
        if a["owner"] == PLAYER:
            suffix += " · Легионы"
        lines = wrap(a["summary"] + suffix, normal, col_width - 82)
        prepared.append((a, lines, max(48, len(lines) * 32 + 14)))
    target, used, split = sum(r[2] for r in prepared) / 2, 0, 0
    while used < target:
        used += prepared[split][2]
        split += 1
    columns = [prepared[:split], prepared[split:]]
    roster_height = max(sum(r[2] for r in c) for c in columns)
    height = max(top + map_px + 1260, top + 100 + roster_height + 90)
    im = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(im)

    def text(x, y, value, ft=normal, color=FG):
        bounds = draw.textbbox((x, y), value, font=ft)
        if bounds[2] > width - 20 or bounds[3] > height - 20:
            raise ValueError(f"Text outside canvas: {value}")
        draw.text((x, y), value, font=ft, fill=color)

    text(margin, 40, "Бегство Бетрезена", font(58, True))
    text(margin, 112, "Bethrezen’s Retreat  ·  Сага Легионов Проклятых  ·  S101", font(29), MUTED)
    text(margin, 164, f"72 × 72  ·  {len(data['armies'])} отрядов и гарнизонов  ·  8 городов  ·  8 руин  ·  35 сундуков", font(27), MUTED)
    cell = map_px / data["size"]

    def center(p):
        return margin + (p[0] + 0.5) * cell, top + (p[1] + 0.5) * cell

    def rect(p, size, fill, outline=None):
        x, y = margin + p[0] * cell, top + p[1] * cell
        draw.rectangle((x, y, x + size[0] * cell, y + size[1] * cell), fill=fill, outline=outline)

    for y, row in enumerate(data["terrain"]):
        for x, code in enumerate(row):
            kind = {1: "forest", 3: "water", 4: "mountain"}.get((code >> 3) & 7, "plain")
            rect([x, y], [1, 1], COLORS[kind])
    for p in data["roads"]:
        rect(p, [1, 1], COLORS["road"])
    for landmark in data["landmarks"]:
        rect(landmark["position"], landmark["size"], COLORS["landmark"])
    for b in data["forts"] + data["ruins"]:
        color = "#b85869" if b["code"] == "A" else "#598cd9" if b["code"] == "B" else "#8d7449" if b["kind"] == "city" else "#675282"
        rect(b["position"], [b["size"]] * 2, color)
    for s in data["sites"]:
        rect(s["position"], [3, 3], "#615747" if s["kind"] == "merchant" else "#56496a")
    for i in range(data["size"] + 1):
        x, y = margin + i * cell, top + i * cell
        draw.line((x, top, x, top + map_px), fill="#29384b")
        draw.line((margin, y, margin + map_px, y), fill="#29384b")
    draw.rectangle((margin, top, margin + map_px, top + map_px), outline="#52647b", width=2)
    for i in range(0, data["size"], 4):
        x, y = center([i, i])
        draw.text((x, top + map_px + 16), str(i), font=small, fill=MUTED, anchor="mt")
        draw.text((margin - 14, y), str(i), font=small, fill=MUTED, anchor="rm")
    text(margin - 50, top - 36, "y ↓", small, MUTED)
    text(margin + map_px - 22, top + map_px + 43, "x", small, MUTED)

    def marker(p, label, color, shape="circle", radius=14):
        x, y = center(p)
        box = (x - radius, y - radius, x + radius, y + radius)
        if shape == "diamond":
            draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)), fill=color, outline=BG)
        elif shape == "box":
            draw.rectangle(box, fill=color, outline=BG, width=2)
        elif shape == "ring":
            draw.ellipse(box, fill=BG, outline=color, width=3)
        else:
            draw.ellipse(box, fill=color, outline=BG, width=2)
        if label:
            size = 17 if len(label) <= 2 else 13
            draw.text((x, y), label, font=font(size, True), fill=color if shape == "ring" else BG, anchor="mm")

    for r in data["resources"]:
        _, label, _, color = RESOURCES[r["resource"]]
        marker(r["position"], label, color, "diamond" if r["resource"] == 0 else "box", 11)
    for c in data["chests"]:
        marker(c["position"], "X", COLORS["chest"], "box", 10)
    for b in data["forts"] + data["ruins"]:
        x, y = center([b["position"][0] + (b["size"] - 1) / 2, b["position"][1] + (b["size"] - 1) / 2])
        draw.text((x, y - 5), b["code"], font=font(28 if b["kind"] == "capital" else 21, True), fill=FG, anchor="mm")
    for s in data["sites"]:
        marker([s["position"][0] + 1, s["position"][1] + 1], s["code"], COLORS[s["kind"]], "box", 18)
    groups = defaultdict(list)
    for a in data["armies"]:
        groups[tuple(a["marker_position"])].append(a)
    for p, group in groups.items():
        a = group[0]
        label = str(a["number"]) + ("+" if len(group) > 1 else "")
        marker(p, label, army_color(a))
    for code, p in {a["code"]: a["location"]["position"] for a in data["appearances"]}.items():
        marker(p, code, COLORS["event"], "ring", 18)
    marker(data["goal"]["position"], "G", COLORS["goal"], "diamond", 20)
    marker(data["ambush"]["position"], "Z", COLORS["event"], "ring", 18)
    beth = next(a for a in data["armies"] if a["source_id"] == BETHREZEN)
    x, y = center(beth["marker_position"])
    draw.ellipse((x - 19, y - 19, x + 19, y + 19), outline=FG, width=3)

    y = top + map_px + 90
    legends = [
        ("A / B", "Столицы Легионов / Орд Нежити", FG),
        ("C / R", "Город / руины; номера защитников справа", COLORS["city"]),
        ("●", "Полевой отряд; бирюзовый — Легионы", COLORS["field"]),
        ("X /", "Сундук / золотая шахта", COLORS["chest"]),
        ("T / M", "Торговая лавка / башня магии", COLORS["mage"]),
        ("G / Z", "Цель сопровождения / триггер засады", COLORS["goal"]),
        ("W1–W4", "Центры зон появления по событиям", COLORS["event"]),
        ("+", "Несколько армий в одном объекте", FG),
    ]
    for i, (code, label, color) in enumerate(legends):
        xx, yy = margin + i % 2 * 1060, y + i // 2 * 42
        text(xx, yy, code, bold, color)
        if code == "X /":
            cx, cy = xx + 80, yy + 17
            draw.polygon(((cx, cy - 10), (cx + 10, cy), (cx, cy + 10), (cx - 10, cy)),
                         fill=COLORS["gold"])
        text(xx + 130, yy, label, small, MUTED)
    y += 192
    text(margin, y, "Мана: П — преисподняя · Ж — жизнь · С — смерть · Р — руны · Л — роща", small, MUTED)
    y += 44
    for i, (kind, label) in enumerate((("forest", "Лес"), ("water", "Вода"), ("road", "Дорога"),
                                        ("mountain", "Горы"), ("landmark", "Декорации"))):
        xx = margin + i * 385
        draw.rectangle((xx, y + 3, xx + 24, y + 27), fill=COLORS[kind])
        text(xx + 38, y, label, small, MUTED)
    y += 60
    text(margin, y, "ЦЕЛЬ И ИСХОДНЫЙ СТАРТ", font(28, True))
    notes = [
        "Сопроводить Бетрезена в безопасную зону G: центр (29, 1), радиус 3. Он не должен погибнуть.",
        f"Отряд {beth['number']} — {beth['summary']}. Исходная позиция: (35, 21), обведена белым.",
        "Z: центр (19, 2), радиус 2. При входе Бетрезена появляются три армии в зоне W1.",
        "Показаны записи сценария ДО событий первого хода. Вступление перемещает семь отрядов.",
        "W обозначает центр зоны, а не гарантированную клетку появления. Подробности — events.png.",
        "Координаты от нуля: x вправо, y вниз. Это вид сверху, без изометрического поворота.",
    ]
    y += 48
    for note in notes:
        for line in wrap(note, small, map_px):
            text(margin, y, line, small, MUTED)
            y += 32
    y += 35
    text(margin, y, "ГОРОДА И ОБЪЕКТЫ", font(28, True))
    y += 47
    objects = data["forts"] + data["ruins"] + data["sites"]
    rows = []
    for b in objects:
        label = f"{b['code']}  {b['name']}"
        if b.get("level"):
            label += f" · ур. {b['level']}"
        if b.get("armies"):
            label += " · " + ", ".join(map(str, b["armies"]))
        rows.append(label)
    half = (len(rows) + 1) // 2
    for i, label in enumerate(rows):
        text(margin + (i // half) * 1060, y + (i % half) * 34, label, small, MUTED)
    y += half * 34 + 40
    text(margin, y, "Источник: Steam Rise of the Elves · Scens / Globals · S101SC0000", small, MUTED)
    text(margin, y + 33, "Имена объектов сохранены из английской редакции; юниты подписаны по каталогу JAX.", small, MUTED)
    text(side, top, "ОТРЯДЫ И ГАРНИЗОНЫ ИСХОДНОГО СЦЕНАРИЯ", font(29, True))
    text(side, top + 45, "Номер на карте → состав · объект. Крупный юнит считается один раз.", small, MUTED)
    for col, entries in enumerate(columns):
        yy = top + 100
        for a, lines, h in entries:
            xx = side + col * col_width
            text(xx, yy, str(a["number"]), bold, army_color(a))
            for i, line in enumerate(lines):
                text(xx + 65, yy + i * 32, line, normal, "#d5dfeb")
            yy += h
    im.save(output / "layout.png", optimize=True)
    return im.size


def render_events(data, output):
    width, padding = 2050, 75
    ft, small = font(28), font(24)
    rows = []
    descriptions = {
        "S101EV0008": "Засада: Бетрезен входит в Z (19, 2), радиус 2",
        "S101EV000e": "Подкрепления: событие с FREQUENCY = 50, однократное",
        "S101EV000f": "Подкрепления: событие с FREQUENCY = 20, однократное",
    }
    for event in descriptions:
        rows.append((descriptions[event], True))
        for a in data["appearances"]:
            if a["event"] == event:
                loc = a["location"]
                rows.append((f"{a['code']} · центр {tuple(loc['position'])}, радиус {loc['radius']} · {a['template']}", False))
                rows.append((a["summary"], False))
        rows.append(("", False))
    rows.append(("ПЕРЕМЕЩЕНИЯ ВО ВСТУПЛЕНИИ ПЕРВОГО ХОДА", True))
    for move in data["intro_moves"]:
        loc = move["location"]
        rows.append((f"Отряд {move['number']}: {tuple(move['source_position'])} → зона {tuple(loc['position'])}, радиус {loc['radius']}", False))
    rows += [("", False), ("Основная схема показывает исходные позиции до исполнения событий. "
              "Для зоны с ненулевым радиусом точная клетка выбирается игрой. "
              "FREQUENCY сохранено как в таблицах сценария: изображение не моделирует очередность событий и ИИ.", False),
             ("Источники: SStkTemp, SLoc, SEvent, SEvCond, SEvEffec; категории — LEvCond / LEvEffct. "
              "Полные записи, исходные ID и контрольные суммы файлов находятся в layout.json.", False)]
    prepared = [(wrap(t, ft, width - padding * 2), heading) for t, heading in rows]
    height = 210 + sum(len(lines) * 40 + (24 if heading else 8) for lines, heading in prepared) + 60
    im = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(im)
    d.text((padding, 45), "Бегство Бетрезена · события и подкрепления", font=font(42, True), fill=FG)
    d.text((padding, 110), "Отдельно от отрядов, уже записанных в исходной расстановке", font=small, fill=MUTED)
    y = 200
    for lines, heading in prepared:
        for line in lines:
            d.text((padding, y), line, font=font(28, True) if heading else ft,
                   fill=COLORS["event"] if heading else FG)
            y += 40
        y += 24 if heading else 8
    im.save(output / "events.png", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=Path(
        "C:/Program Files (x86)/Steam/steamapps/common/Disciples II Rise of the Elves"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/maps/bethrezens_retreat")
    args = parser.parse_args()
    data = extract(args.game)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "layout.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dimensions = render(data, args.output)
    render_events(data, args.output)
    print(f"Saved {args.output / 'layout.png'} ({dimensions[0]} x {dimensions[1]}); {len(data['armies'])} groups")


if __name__ == "__main__":
    main()
