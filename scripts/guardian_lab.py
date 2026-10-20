"""Search parties and tactics against a capital guardian of the shared NumberGrid environment.

Subcommands:
  capabilities  guardian profile against the reachable catalogue: what works and why
  candidates    parties within leadership, formation, buildings, map loot and item access
  screen        analytic win estimate: the binomial formula and a Markov chain over rounds
  cost          battle XP, trainer gold, buildings, hires and purchases of a party
  archive       MAP-Elites style elites over controllers, damage, extra lives and cost
  simulate      batched GPU Monte Carlo of the best tactics in the real JAX environment

The static subcommands read map data and catalogue modules only and create no JAX device
arrays. simulate needs CUDA: bash scripts/run_gpu.sh scripts/guardian_lab.py simulate ...
Scripted tactics are checks of the environment; they are never given to the agent.
Rules, their sources, approximations and open questions: docs/GUARDIAN_LAB.md.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

# Importing the environment never allocates; keep any accidental backend start small.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / 'stoa-src', ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

from stoix.envs.number_grid_battle_items import (
    BUFF, DOT, DRAIN, EXTRA_TURN, FEAR, GAME_IDS, HEAL, KIND_NAMES, LYCANTHROPY, PARALYSIS, REVIVE,
    STRIKE, SUMMON, TRANSFORM, WEAKEN, scenario_battle_items)
from stoix.envs.number_grid_buildings import CATALOG
from stoix.envs.number_grid_combat import ACCURACY, ARMOR, ATTACK_TYPES, DAMAGE, HP, INITIATIVE, UNITS
from stoix.envs.number_grid_effects import (
    BAT_BMODIF, BAT_BREAK, CAPITAL_GUARDS, DISABLE_ALLOWED_ROUND_MAX, LONG, attack_class, attack_long,
    disable_mode)
from stoix.envs.number_grid_hero_skills import enabled as hero_skills_enabled, lore_levels, might_bonus
from stoix.envs.number_grid_items import ARTIFACT, BANNER, BOOK, BOOTS, ITEM_BY_KEY, VALUABLE
from stoix.envs.number_grid_lords import DEFAULT_FACTION, lord_settings, with_starting_party
from stoix.envs.number_grid_potions import POTIONS, POTION_BY_KEY, SECONDARY_COLUMNS, merchants
from stoix.envs.number_grid_recruitment import OFFERS, mercenary_offers
from stoix.envs.number_grid_territory import CAPITAL_ARMOR, CAPITAL_REGENERATION, CITY_BONUS

STATS = ('max_hp', 'damage', 'accuracy', 'armor', 'initiative')
ARMOR_CAP = 90  # number_grid_progression.damage clips armour to 0..90
WEAKEN_FACTOR = .67  # number_grid.py:_weakening_factor, GVars BATLOWERD2=33
DOT_CLASSES = {12: 'poison', 13: 'water', 23: 'burn'}  # number_grid_effects.effect_tables
DOT_SOURCES = {'poison': 'poison', 'water': 'water', 'burn': 'fire'}  # immunity masks 16/8/4
DOT_FALLBACK = {'poison': 0, 'water': 10, 'burn': 10}  # advance_periodic_queue fallback damage
DOT_BREAK_AGE1 = .21  # first removal check of a refreshed DOT, age 1 (ORIGINAL_EFFECTS.md)
# number_grid_progression.ProgressionRules: damage caps and zero-damage support types.
HEAVY_STRIKE_IDS = {'g000uu0019', 'g000uu0020', 'g000uu0044', 'g000uu0045', 'g000uu0047', 'g000uu0070',
                    'g000uu0071', 'g000uu0096', 'g000uu8009', 'g000uu8011'}
SUPPORT_TYPES = {'Cliric', 'Profit', 'Patriach', 'Deva roshi', 'Sundancer', 'Sylfid', 'Ghost', 'Witch', 'Shadow',
                 'Incub', 'Succub', 'Baroness', 'Travnitsa', 'Novice', 'Alchemist', 'Dwarfdruid', 'Arhidruid'}
SURVIVAL_POTIONS = ('invulnerability', 'bark', 'protection', 'iron_skin', 'highfather')
DAMAGE_POTIONS = ('titan', 'might', 'strength', 'vigor')
ACCURACY_POTIONS = ('fortune', 'accuracy', 'striking')
INITIATIVE_POTIONS = ('celerity', 'speed', 'swiftness', 'quicksilver')
WARD_POTIONS = {'fire_ward': 'fire', 'water_ward': 'water', 'earth_ward': 'earth', 'air_ward': 'air'}
TEMPORARY_ORDER = POTIONS[4:15]  # PotionRules.modifiers bit order
ITEM_SOURCES = {i + 1: name for i, name in enumerate(ATTACK_TYPES)}  # one-based item sources
# Battle items the hero uses at the first activation, not to save a fallen or wounded ally.
OPENING_KINDS = (WEAKEN, PARALYSIS, FEAR, SUMMON, STRIKE, DRAIN, DOT, BUFF, TRANSFORM, LYCANTHROPY, EXTRA_TURN)


def _rule(text, source, where, status, check=''):
    return dict(text=text, source=source, where=where, status=status, check=check)


# Every tactic lists the rules it relies on. status: original (installed DBF/EXE, manual or an
# in-game check by the user), reference (Python reference only), user (user request, not a claim
# about the original), open (explicit open question). Only "original" counts as confirmed.
RULES = {
    'guardian_attack': _rule(
        'Удар стража: класс, источник, сила, точность и охват атаки из GAttacks/GUnits; Мизраэль — '
        'массовая атака Жизнью 250, точность 95, инициатива 90', 'DBF/EXE',
        'stoix/envs/data/units.json (g000uu3001.game_data.primary_attack)', 'original'),
    'capital_armor': _rule(
        'Страж столицы получает +50 брони в бою (GVars PROT_CAP), город — бонус укрепления', 'DBF/EXE',
        'number_grid_territory.py:CAPITAL_ARMOR, defender_armor; docs/DEFAULT_MAP.md', 'original'),
    'armor_cap': _rule(
        'Эффективная броня, включая Defend, не выше 90%', 'DBF/EXE',
        'Discipl2.exe sub_5D35F3; GVars DFENDBONUS=50; docs/ARCHER_BATTLES.md', 'original'),
    'damage_roll': _rule(
        'D = min(сила после модификаторов + 0…5, предел); урон = D − trunc(D × броня/100)', 'EXE',
        'Discipl2.exe sub_5D3715, sub_65C047, sub_5840EC; docs/ARCHER_BATTLES.md', 'original'),
    'pair_roll': _rule(
        'Попадание: два целых броска 0…99, успех при a + b < 2 × точность', 'Python-референс',
        'number_grid_combat.py:accuracy_hits', 'reference',
        'IDA: функция броска точности — одно или два случайных числа; влияет на все вероятности контроля.'),
    'area_rolls': _rule(
        'Массовая атака бросает попадание отдельно для каждой цели', 'Python-референс',
        'number_grid.py:_battle_step_regular (hits per target)', 'reference'),
    'initiative_roll': _rule(
        'Очерёдность раунда: инициатива + случайные 0…9 (дробная часть разбивает ничьи)', 'Python-референс',
        'number_grid.py:_round_priority', 'reference'),
    'hero_first': _rule(
        'Боец с инициативой ≥ 100 всегда ходит раньше стража с инициативой 90 (90 + 0…9,99)', 'Python-референс',
        'следствие initiative_roll', 'reference'),
    'short_disable': _rule(
        'Короткий паралич/окаменение: пропуск ближайшего хода цели и снятие; снимается и при следующем '
        'ходе источника', 'DBF/EXE', 'docs/ORIGINAL_EFFECTS.md (EXE 0x626AA7); number_grid.py:_prepare_disable',
        'original'),
    'long_disable': _rule(
        'Длительный паралич: перед ходом снимается с шансом 11, 21, … 100% по возрасту в раундах',
        'DBF/EXE', 'GVars BAT_BREAK/BAT_BMODIF, EXE 0x626CB3; number_grid_effects.py:effect_breaks', 'original'),
    'effect_breaks': _rule(
        'Длительный DoT после тика снимается с шансом 11, 21, … 100% по возрасту; экран берёт среднюю долю '
        'активных раундов при обновлении каждый раунд', 'DBF/EXE + приближение lab',
        'number_grid_effects.py:effect_breaks; guardian_lab.stationary_dot', 'original'),
    'disable_round40': _rule(
        'С 41-го раунда точность паралича и окаменения равна 0 (страх не ограничен)',
        'D2ModdingToolset + запрос пользователя 16.10.2026', 'number_grid_effects.py:disable_attack_accuracy',
        'open', 'Бой дольше 40 раундов: окаменяет ли Инкуб на 41-м раунде; в IDA — ветка обнуления точности '
        'paralyze/petrify в расчёте силы атаки установленного Discipl2.exe.'),
    'damage_vs_disabled': _rule(
        'Парализованный и окаменевший страж получает обычный урон', 'приближение JAX (особого правила нет)',
        'number_grid.py:_battle_step_regular', 'open',
        'Окаменить врага Инкубом и ударить бойцом: уменьшается ли HP; по описаниям окаменевшие неуязвимы. '
        'Если неуязвимы — тактики «контроль + урон» в JAX завышены.'),
    'secondary_after_primary': _rule(
        'Вторичный эффект (окаменение Дьявола, ожог Владыки, ослабление Тиамат) — только после попадания '
        'основной атаки по выжившей цели и отдельного броска точности', 'Python-референс',
        'number_grid.py:_battle_step_regular (connected_status)', 'reference'),
    'dot_tick': _rule(
        'Яд, обморожение, ожог тикают перед действием цели, в том числе перед пропуском из-за паралича',
        'DBF/EXE', 'docs/ORIGINAL_EFFECTS.md (EXE 0x628B60, 0x628D12, 0x628EC4)', 'original'),
    'dot_armor': _rule(
        'Тик DoT не уменьшается бронёй и Defend', 'Python-референс',
        'number_grid_effects.py:advance_periodic_queue; docs/REFERENCE_UNIT_ACTIONS.md', 'reference',
        'Ожог Владыки по цели с бронёй 90: снимает ли полные 30/47 HP за тик.'),
    'dot_refresh': _rule(
        'Повторный DoT обновляет урон и возраст без суммирования', 'DBF/EXE', 'docs/ORIGINAL_EFFECTS.md',
        'original'),
    'dot_potions': _rule(
        'Зелья и заклинания, как знамёна и артефакты, не меняют вторичную атаку: силу тика DoT и точность '
        'вторичных эффектов (окаменение, паралич, страх, ослабление); Python-референс усиливал обе атаки',
        'запрос пользователя 19.10.2026; DBF (один класс L_ATTACK) + D2ModdingToolset',
        'number_grid_potions.py:SECONDARY_COLUMNS; number_grid.py:_secondary_stats; docs/MAP_POTIONS.md', 'open',
        'Тик ожога Владыки 5-го уровня (30) без бонусов, с Эликсиром энергии и со Знаменем мощи; доля '
        'окаменений Дьявола Бездны с Зельем точности и без.'),
    'weakening': _rule(
        'Ослабление: −33% урона цели до конца боя (BATLOWERD2=33)', 'DBF/EXE',
        'docs/ORIGINAL_EFFECTS.md (EXE 0x65E591); number_grid.py:_weakening_factor', 'original'),
    'transform_guard': _rule(
        'Превращение юнитами (Ведьма, Колдунья, Суккуб) и Hag\'s Ring не действует на стража столицы',
        'Python-референс', 'number_grid.py:2645-2646; number_grid_item_combat.py:103', 'open',
        'В GImmuC у Мизраэля только иммунитет к Shatter (25). Проверить в игре, превращает ли Суккуб '
        'стража столицы (или показывает «Иммунитет»).'),
    'transform_orb_guard': _rule(
        'Сфера/талисман Ведьм и Сфера ликантропии в JAX не исключают стража: превращение сработает',
        'несогласованность JAX', 'number_grid_battle_items.py:use (TRANSFORM, LYCANTHROPY)', 'open',
        'Решить вместе с transform_guard; Сферы ликантропии в установленном GItem нет.'),
    'shatter_guard': _rule(
        'Разрушение брони не действует на стража: GImmuC IMMUNITY 25, IMMUNECAT 3', 'DBF/EXE',
        'number_grid.py:2645-2646 (~guards)', 'original'),
    'spell_guard': _rule(
        'Заклинания карты и отдельные призывы не выбирают стража столицы целью', 'Python-референс',
        'number_grid_casting.py:148 (_spell_untargetable_enemy_ids)', 'open',
        'Можно ли наложить урон/ослабление заклинанием на стража столицы в игре.'),
    'copy_guard': _rule(
        'Двойник не копирует юнитов категории UNIT_CAT=8 (стражи)', 'Python-референс/DBF',
        'number_grid_progression.py:106 (copy_forbidden)', 'reference'),
    'fear_fort': _rule(
        'Страх по защитнику укрепления (столица, город, руины) становится коротким параличом',
        'Python-референс + EXE (замена при запрете бегства)', 'number_grid.py:_fear_protected', 'open',
        'Запрещено ли бегство стражу столицы в оригинале.'),
    'wounds_persist': _rule(
        'После отступления отряда страж сохраняет раны; при отдыхе восстанавливает REGEN юнита + 35% столицы',
        'Python-референс + D2ModdingToolset getUnitRegen', 'number_grid_casting.py:finish_battle, rest', 'open',
        'Отступить от Мизраэля и напасть снова в тот же день: сохранились ли раны; сколько он лечит за ход.'),
    'potion_until_rest': _rule(
        'Временные зелья действуют до отдыха; каждый вид — один раз на юнита', 'Python-референс',
        'number_grid_potions.py:available, expire', 'reference'),
    'permanent_once': _rule(
        'Постоянное зелье каждого вида — один раз на юнита за эпизод', 'проверка пользователя в игре 17.10.2026',
        'number_grid_potions.py:available', 'original'),
    'invulnerability': _rule(
        'Эликсир неуязвимости: +50 брони до отдыха; не превышает общий предел 90%', 'Python-референс/GItem',
        'number_grid_potions.py:POTIONS', 'reference'),
    'equip_daily': _rule(
        'Боевой предмет кладётся в слот не чаще раза в день, снятия нет', 'Python-референс',
        'docs/BATTLE_ITEMS.md; number_grid_battle_items.py:equip_available', 'reference'),
    'magic_access': _rule(
        'Сферы и талисманы — с 10-го уровня героя, у лорда-мага или с книгой в слоте книги', 'Python-референс',
        'number_grid_battle_items.py:access', 'reference'),
    'item_turn': _rule(
        'Применение боевого предмета тратит ход героя', 'Python-референс', 'docs/BATTLE_ITEMS.md', 'reference'),
    'life_potion': _rule(
        'Зелье воскрешения в бою возвращает павшего с 1 HP, Сфера/талисман жизни — с 50% HP', 'Python-референс',
        'number_grid_battle_items.py:POTION_EFFECTS, EFFECTS', 'reference'),
    'summon_owner': _rule(
        'Призыв сферой исчезает при гибели или бегстве героя', 'проверка пользователя в игре 14.10.2026',
        'docs/BATTLE_ITEMS.md', 'original'),
    'leadership': _rule(
        'Лидерство: 3 места, +1 на 3-м и 6-м уровне героя, не более 5; большой боец занимает 2',
        'Python-референс', 'number_grid_recruitment.py:composition', 'reference'),
    'hero_levels': _rule(
        'Герой: порог 150, +500 за уровень; Выносливость (HP ×1,2) на 4-м, Сила (урон ×1,25) на 5-м, '
        'Мощь лорда-воина на 9-м', 'Python-референс', 'docs/LEGIONS_DUKE.md, docs/HERO_SKILLS.md', 'reference'),
    'equipment_skills': _rule(
        'Экипировка героя: знамя с 7-го, сапоги с 8-го, книга с 11-го уровня, артефакты у воина сразу; '
        'надевается самая дорогая вещь', 'Python-референс + автовыбор JAX',
        'number_grid_items.py:choose; number_grid_hero_skills.py:lore_levels', 'reference'),
    'trainer': _rule(
        'Тренер продаёт XP до порога − 1 по цене TRAINING_C; последний XP — из боя', 'DBF + руководство',
        'docs/MAP_SITES.md', 'original'),
    'promotion_once': _rule(
        'Одно повышение за бой, избыток опыта не переносится', 'Python-референс', 'docs/UNIT_PROGRESSION.md',
        'reference'),
    'xp_split': _rule(
        'Опыт боя делится поровну между живыми не отступившими победителями', 'Python-референс',
        'number_grid_progression.py:finish', 'reference'),
    'guard_no_xp': _rule(
        'Стражи столиц не получают боевой опыт', 'запрос пользователя', 'docs/UNIT_PROGRESSION.md', 'user'),
    'round_limit': _rule(
        'После 75-го раунда отряд обязан отступить', 'Python-референс', 'number_grid.py:round_limit_retreat',
        'reference'),
    'recruit_slots': _rule(
        'Найм: малые бойцы ближнего боя — в передние слоты 0/2, остальные назад; большой — передний слот с '
        'пустым задним', 'Python-референс', 'number_grid_recruitment.py:targets', 'reference'),
}


# ------------------------------------------------------------------------------- map and catalogue

def _gold(cost):
    """Gunits ENROLL_C 'g0050:r0000:…' → 50 gold."""
    text = str(cost or '').split(':')[0]
    return int(text[1:]) if text[:1] == 'g' and text[1:].isdigit() else 0


def _totals(sources, field):
    counts = defaultdict(int)
    for source in sources:
        for key, count in (source.get(field) or {}).items():
            counts[key] += count
    return dict(counts)


def _merge(*parts):
    counts = defaultdict(int)
    for part in parts:
        for key, count in part.items():
            counts[key] += count
    return dict(counts)


def kill_xp(key, level):
    """Experience a dead unit puts into the battle bank (ordinary growth rows)."""
    row = UNITS[key]
    extra = max(level - row['level'], 0)
    early = min(extra, max(row['growth']['threshold'] - row['level'], 0))
    return row['exp_kill'] + early * row['growth']['kill_early'] + (extra - early) * row['growth']['kill_late']


class World:
    """Static scenario facts for one guardian: loot, offers, buildings, XP. No JAX arrays."""

    def __init__(self, map_name='default-capital', guardian=None):
        from stoix.envs.number_grid import load_map  # the module import builds no device arrays
        self.map_name = map_name
        game_map = with_starting_party(load_map(map_name))
        self.game_map = game_map
        self.faction = game_map.get('faction', DEFAULT_FACTION)
        self.lord = lord_settings(game_map)
        self.mage_lord = self.lord['id'] == 'mage'
        self.hero_skills = hero_skills_enabled(game_map)
        self.requires_skills = bool(game_map.get('equipment_requires_skills', False))
        self.skill_levels = lore_levels(game_map)
        self.equip_policy = game_map.get('equipment_auto_equip', 'price')
        self.max_rounds = int(game_map.get('battle_max_rounds', 75))
        self.base_capacity = game_map.get('hero_leadership_capacity', 3)
        self.max_hero_level = game_map.get('max_hero_level')
        roster = game_map['hero_roster']
        heroes = [s for s, k in enumerate(roster) if k and UNITS[k].get('hero')]
        if not heroes:
            raise SystemExit('The starting party has no hero; guardian_lab needs a leader')
        self.hero_slot = heroes[0]
        self.hero_key = roster[self.hero_slot]
        might = might_bonus(game_map)
        self.might = (might['level'], might['multiplier']) if self.hero_skills else None
        self.starting = [(s, k) for s, k in enumerate(roster) if k and s != self.hero_slot]
        self.training = json.loads((ROOT / 'stoix/envs/data/training_prices.json').read_text(encoding='utf-8'))
        self._guardian(guardian)
        self._loot()
        self._catalogue()

    def _guardian(self, guardian):
        game_map = self.game_map
        guards = [g for c in game_map.get('enemy_capitals', []) for g in c.get('guards', [c.get('guard')])]
        if guardian is None:
            guardian = game_map.get('objective_enemy')
        if guardian is None and guards:
            guardian = guards[0]
        if guardian is None:
            raise SystemExit('Pass --guardian: the map has neither objective_enemy nor a capital guard')
        roster = game_map['enemy_rosters'][guardian]
        occupied = [s for s, k in enumerate(roster) if k]
        named = [s for s in occupied if UNITS[roster[s]]['name'] in CAPITAL_GUARDS]
        slot = (named or sorted(occupied, key=lambda s: -UNITS[roster[s]]['max_hp']))[0]
        self.guardian, self.guardian_slot, self.guardian_key = guardian, slot, roster[slot]
        self.escort = [(s, roster[s]) for s in occupied if s != slot]
        levels = game_map.get('enemy_initial_levels')
        self.enemy_levels = levels
        self.guardian_level = levels[guardian][slot] if levels else UNITS[self.guardian_key]['level']
        self.capital_guard = guardian in guards
        self.all_guards = set(guards)
        position = list(game_map['opponent_positions'][guardian])
        city = next((c for c in game_map.get('cities', []) if list(c['position']) == position), None)
        self.city_armor = CITY_BONUS[city.get('level', 1)] if city is not None and game_map.get('territory') else 0
        self.fort_armor = self.city_armor + (CAPITAL_ARMOR if self.capital_guard else 0)
        self.fort = self.fort_armor > 0  # number_grid.py:_fear_protected

    def _loot(self):
        game_map = self.game_map
        free = ([dict(potions=game_map.get('initial_potions', {}), items=game_map.get('initial_items', {}))]
                + list(game_map.get('chests', [])) + list(game_map.get('ruins', []))
                + [dict(items=game_map.get('event_items', {}))])
        shops = merchants(game_map)
        self.free_potions, self.shop_potions = _totals(free, 'potions'), _totals(shops, 'potions')
        self.free_items, self.shop_items = _totals(free, 'items'), _totals(shops, 'items')
        self.potions = _merge(self.free_potions, self.shop_potions)
        self.items = _merge(self.free_items, self.shop_items)
        self.battle_items = {item['key']: item for item in scenario_battle_items(game_map)}
        valuables = sum(ITEM_BY_KEY[k].get('sell_price', 0) * n for k, n in self.free_items.items()
                        if ITEM_BY_KEY.get(k, {}).get('category') == VALUABLE)
        self.gold_sources = dict(starting=game_map.get('starting_gold', 0), valuables=valuables,
                                 ruins=sum(r.get('gold', 0) for r in game_map.get('ruins', [])))

    def _catalogue(self):
        game_map = self.game_map
        rows = CATALOG['factions'][self.faction]['buildings']
        self.buildings = {row['name']: row for row in rows}
        ids = {row['id']: row['name'] for row in rows}
        self.building_by_unit = {row['unit']: row['name'] for row in rows if row.get('unit')}
        self.building_by_original = {row['original_id'].lower(): row['name'] for row in rows}
        self.ancestors = {}
        for name in self.buildings:
            reached, pending = set(), list(self.buildings[name]['requires'])
            while pending:
                parent = pending.pop()
                if parent not in reached:
                    reached.add(parent)
                    pending.extend(self.buildings[parent]['requires'])
            self.ancestors[name] = reached
        descendants = {name: {d for d, parents in self.ancestors.items() if name in parents} for name in self.buildings}
        # BuildingRules: a building blocks the listed rivals and all their descendants.
        self.excludes = {name: set().union(*[{b} | descendants[b] for b in row['blocks']])
                         for name, row in self.buildings.items()}
        max_level = game_map.get('max_building_level', 5)
        blocked = {ids.get(b, b) for b in game_map.get('blocked_buildings', [])}
        blocked |= {name for name, row in self.buildings.items() if row['level'] > max_level}
        self.unusable = set(blocked).union(*[descendants[b] for b in blocked if b in descendants])
        offers = {k: dict(mercenary=False, price=_gold(UNITS[k]['game_data'].get('ENROLL_C')), stock=None)
                  for k in OFFERS.get(self.faction, ()) if k in UNITS}
        for _, row, _ in mercenary_offers(game_map):
            price = row.get('price', _gold(UNITS[row['unit']]['game_data'].get('ENROLL_C')))
            offers[row['unit']] = dict(mercenary=True, price=price, stock=row['stock'])
        self.offers = offers
        starting = {k for _, k in self.starting}
        self.forms = {}
        roots = [(k, info['mercenary']) for k, info in offers.items()] + [(k, False) for k in sorted(starting)]
        for root, mercenary in roots:
            if root in self.forms:
                continue
            need = set()
            if not mercenary and root not in starting:
                building = self.building_by_original.get(str(UNITS[root]['game_data'].get('ENROLL_B', '')).lower())
                if building in self.unusable:
                    continue
                if building:
                    need = {building} | self.ancestors[building]
            queue = [(root, [root], need)]
            while queue:
                key, chain, need = queue.pop(0)
                if key in self.forms:
                    continue
                children = []
                for target in UNITS[key].get('upgrades', []):
                    row = UNITS.get(target)
                    building = self.building_by_unit.get(row['name']) if row else None
                    if (row is None or row.get('upgrade_unavailable_reason') or mercenary
                            or UNITS[key]['faction'] != self.faction or building is None or building in self.unusable):
                        continue
                    children.append(target)
                    queue.append((target, chain + [target], need | {building} | self.ancestors[building]))
                self.forms[key] = dict(chain=chain, root=root, mercenary=mercenary, buildings=need,
                                       final=not children, starting_root=root in starting)
        self.xp_pool = 0
        for index, roster in enumerate(game_map['enemy_rosters']):
            if index == self.guardian or index in self.all_guards:
                continue
            levels = self.enemy_levels[index] if self.enemy_levels else [UNITS[k]['level'] if k else 0 for k in roster]
            self.xp_pool += sum(kill_xp(k, level) for k, level in zip(roster, levels) if k)

    def conflict(self, keys):
        """Mutually exclusive buildings needed by these forms, or None."""
        needed = set().union(*[self.forms[k]['buildings'] for k in keys if k in self.forms])
        for name in sorted(needed):
            clash = self.excludes[name] & needed
            if clash:
                return name, sorted(clash)[0]
        return None

    def train_price(self, key, level):
        """Trainer gold per XP of a form at a level (docs/MAP_SITES.md, TRAINING_C)."""
        entry = self.training.get(UNITS[key].get('game_id'))
        if entry is None:
            return None
        native, threshold, base, early, late = entry
        return base + max(0, min(level, threshold) - native) * early + max(0, level - max(native, threshold)) * late

    def capacity(self, hero_level):
        """RecruitmentRules.composition: base places, +1 at hero levels 3 and 6, at most 5."""
        return min(5, self.base_capacity + (hero_level >= 3) + (hero_level >= 6))

    def name(self, key):
        return UNITS[key]['name'] if key else '—'


# ------------------------------------------------------------------------------- unit statistics

def grown_stats(row, level):
    """ProgressionRules.grown_stats: level growth and one-time level bonuses."""
    extra = max(level - row['level'], 0)
    early = min(extra, max(row['growth']['threshold'] - row['level'], 0))
    late = extra - early
    if not row.get('level_bonuses'):
        return [row[k] + early * a + late * b for k, a, b in zip(STATS, row['growth']['early'], row['growth']['late'])]
    values = [row[k] for k in STATS]
    for following in range(row['level'] + 1, level + 1):
        increment = row['growth']['early' if following <= row['growth']['threshold'] else 'late']
        values = [v + i for v, i in zip(values, increment)]
        for bonus in row['level_bonuses']:
            if bonus['level'] == following:
                index = STATS.index(bonus['stat'])
                value = values[index] * bonus['multiplier'] + bonus['bonus']
                values[index] = int(value + .5) if bonus['rounding'] == 'half_up' else round(value)
        values[2] = min(values[2], 100)
    return values


def stat_caps(row):
    damage = math.inf if row.get('unit_type') in SUPPORT_TYPES else 400 if row.get('game_id') in HEAVY_STRIKE_IDS else 300
    return [math.inf, damage, 100, ARMOR_CAP, math.inf]


@lru_cache(maxsize=None)
def unit_row(key, might=None):
    """Catalogue row; the player's hero rows receive the lord's Might (number_grid_progression)."""
    row = UNITS[key]
    if might is None:
        return row
    level, multiplier = might
    bonus = dict(level=level, name='Мощь', stat='damage', multiplier=multiplier, bonus=0, rounding='even')
    row = dict(row, level_bonuses=sorted(list(row.get('level_bonuses', [])) + [bonus], key=lambda b: b['level']))
    if row['level'] >= level:
        row['damage'] = round(row['damage'] * multiplier)
    return row


@lru_cache(maxsize=None)
def base_profile(key, level, might=None):
    """_potion_intrinsic: capped stats with raw damage, then secondary damage/accuracy."""
    row = unit_row(key, might)
    raw = grown_stats(row, level)
    values = [float(min(v, c)) for v, c in zip(raw, stat_caps(row))]
    values[DAMAGE] = float(raw[DAMAGE])
    extra = max(level - row['level'], 0)
    early = min(extra, max(row['growth']['threshold'] - row['level'], 0))
    first, second = row['growth'].get('secondary_early', [0, 0]), row['growth'].get('secondary_late', [0, 0])
    secondary = [row.get('secondary_damage', 0) + early * first[0] + (extra - early) * second[0],
                 row.get('secondary_accuracy', 0) + early * first[1] + (extra - early) * second[1]]
    return tuple(values + [float(min(secondary[0], 300.)), float(min(secondary[1], 100.))])


def _drink(base, bonus, key):
    """PotionRules.apply for one permanent dose over intrinsic + earlier doses (7 columns)."""
    potion = POTION_BY_KEY[key]
    stat = potion['stat']
    current = [b + x for b, x in zip(base, bonus)]
    if stat == ARMOR:
        grown = [c + potion['amount'] for c in current]
    else:
        grown = [float(round(c * (1.15 if stat == HP else 1.1))) for c in current]
    grown[HP] = max(current[HP] + 1, grown[HP])
    grown[ACCURACY] = min(grown[ACCURACY], 100.)
    # Primary attack only: SECONDARY_COLUMNS never change (rule dot_potions).
    return [x + (g - c if i == stat else 0.) for i, (x, g, c) in enumerate(zip(bonus, grown, current))]


def _temporary(values, active):
    """PotionRules.temporary_bonus: distinct temporary doses multiply before one rounding."""
    factors = [1., 1., 1., 0., 1., 1., 1.]
    for potion in TEMPORARY_ORDER:
        if potion['key'] in active:
            stat = potion['stat']
            if stat == ARMOR:
                factors[stat] += potion['amount']
            else:
                factors[stat] *= 1 + potion['amount'] / 100.  # SECONDARY_COLUMNS stay x1
    grown = [float(round(v * f)) for v, f in zip(values, factors)]
    grown[ARMOR] = values[ARMOR] + factors[ARMOR]
    grown[ACCURACY] = min(grown[ACCURACY], 100.)
    return [g - v for g, v in zip(grown, values)]


def _equip(values, hero, worn):
    """ItemRules.numeric_bonus: the hero's artifacts, then the banner for every party slot."""
    values = list(values)
    if hero:
        artifacts = [p for p in worn[:2] if p]
        damage = math.prod(p.get('damage', 1.) for p in artifacts)
        initiative = math.prod(p.get('initiative', 1.) for p in artifacts)
        values = [float(round(v * f)) for v, f in zip(values, (1., damage, 1., 1., initiative, 1., 1.))]
        values[ARMOR] += sum(p.get('armor', 0) for p in artifacts)
    banner = worn[2] or {}
    values = [float(round(v * f)) for v, f in
              zip(values, (1., banner.get('damage', 1.), 1., 1., banner.get('initiative', 1.), 1., 1.))]
    accuracy = int(values[ACCURACY])
    values[ACCURACY] = float(min(100, accuracy + accuracy * banner.get('accuracy_percent', 0) // 100))
    values[ARMOR] += banner.get('armor', 0)
    return values


@lru_cache(maxsize=None)
def unit_profile(key, level, permanent=(), temporary=(), worn=(None,) * 5, hero=False, might=None):
    """Battle values of one party slot as _combat_stats and _secondary_stats compute them."""
    base = list(base_profile(key, level, might))
    bonus = [0.] * 7
    for potion in permanent:
        bonus = _drink(base, bonus, potion)
    current = [b + x for b, x in zip(base, bonus)]
    temp = _temporary(current, frozenset(temporary))
    values = [c + t for c, t in zip(current, temp)]
    enhanced = _equip(values, hero, [ITEM_BY_KEY[k] if k else None for k in worn])
    cap = stat_caps(unit_row(key, might))[DAMAGE]
    stats = list(enhanced[:5])
    stats[ACCURACY] = min(stats[ACCURACY], 100.)
    stats[DAMAGE] = min(enhanced[DAMAGE], cap)
    # _secondary_stats: form and level only; potions, spells, banners and artifacts
    # change the primary attack alone (rule dot_potions, docs/MAP_POTIONS.md).
    secondary = tuple(base[c] for c in SECONDARY_COLUMNS)
    copy = tuple(base[:5][i] + bonus[i] for i in range(5))  # _copy_form_stats: permanent values only
    return dict(stats=tuple(stats), secondary=secondary, enhanced_damage=enhanced[DAMAGE], cap=cap, copy=copy)


def mechanics(row):
    """Behaviour switches as ProgressionRules and effect_tables derive them for the JAX step."""
    kind = row.get('unit_type', '')
    return dict(
        type=kind, role=row['role'], size=row.get('size', 1), source=row['attack_type'],
        secondary_source=row.get('secondary_attack_type') or '', disable=disable_mode(row),
        secondary_disable=disable_mode(row, True), dot=DOT_CLASSES.get(attack_class(row, True)),
        dot_long=attack_long(row, True), fear=kind == 'Baroness', secondary_fear=kind == 'Shamanka',
        weaken=kind == 'Tiamat', witch=kind in ('Witch', 'Succub'), shatter=kind in ('Teurg', 'Aleman'),
        wight=kind == 'Wight', double=kind in ('Demon', 'Elfarcher'), centaur=kind == 'Centaur Savage',
        copy=kind == 'Doppelganger', healer=row['role'] == 'healer',
        summoner=kind in ('Summoner', 'Occultmaster', 'Lyf', 'Laclaan'),
        leech={'Bone Lord': 2, 'Highvampire': 2, 'Dregazul': 1, 'Vampire': 1}.get(kind, 0))


def choose_equipment(world, items, hero_level):
    """ItemRules.choose, 'price' policy: two artifacts, banner, book, boots; skills gate slots."""
    def score(key):
        item = ITEM_BY_KEY[key]
        return item['price'] * 16 + (item.get('priority', 0) if item['category'] == BANNER else 0)
    pool = [k for k, n in sorted(items.items()) for _ in range(n)
            if ITEM_BY_KEY.get(k, {}).get('category') in (ARTIFACT, BANNER, BOOK, BOOTS)]
    worn = []
    for slot, category in enumerate((ARTIFACT, ARTIFACT, BANNER, BOOK, BOOTS)):
        options = [k for k in pool if ITEM_BY_KEY[k]['category'] == category]
        best = max(options, key=score) if options else None
        if slot == 0 and best:
            pool.remove(best)  # the second artifact slot uses another copy
        allowed = not world.requires_skills or hero_level >= world.skill_levels[slot]
        worn.append(best if allowed else None)
    return tuple(worn)


def magic_access(world, hero_level, worn):
    """BattleItemRules.access: sorcery lore at level 10, a mage lord, or a book in the book slot."""
    book = ITEM_BY_KEY.get(worn[3] or '', {})
    lore = world.mage_lord or hero_level >= 10
    return dict(orb=lore or bool(book.get('orbs')), talisman=lore or bool(book.get('talismans')))


def cand_items(world, cand):
    inventory = cand.get('inventory', 'map')
    return dict(world.items) if inventory == 'map' else dict(inventory.get('items', {}))


def cand_potions(world, cand):
    inventory = cand.get('inventory', 'map')
    return dict(world.potions) if inventory == 'map' else dict(inventory.get('potions', {}))


def party_profiles(world, cand):
    """Static battle values of every occupied party slot of a candidate."""
    roster = cand['roster']
    levels = {int(k): int(v) for k, v in (cand.get('levels') or {}).items()}
    worn = choose_equipment(world, cand_items(world, cand), int(cand['hero_level']))
    permanent, temporary = defaultdict(list), defaultdict(list)
    for key, slot in cand.get('potions', []):
        duration = POTION_BY_KEY[key]['duration']
        if duration == 'permanent':
            permanent[int(slot)].append(key)
        elif duration == 'temporary':
            temporary[int(slot)].append(key)
    units = []
    for slot, key in enumerate(roster):
        if not key:
            continue
        hero = slot == world.hero_slot
        row = UNITS[key]
        level = int(cand['hero_level']) if hero else levels.get(slot, row['level'])
        might = world.might if hero and key == world.hero_key else None
        profile = unit_profile(key, level, tuple(permanent[slot]), tuple(sorted(temporary[slot])), worn, hero, might)
        wards = set(row.get('protections', []))
        wards |= {WARD_POTIONS[k] for k in temporary[slot] if k in WARD_POTIONS}
        if hero and worn[3]:
            bits = ITEM_BY_KEY[worn[3]].get('ward', 0)
            wards |= {name for i, name in enumerate(ATTACK_TYPES) if bits & (1 << i)}
        units.append(dict(slot=slot, key=key, name=row['name'], level=level, hero=hero, size=row.get('size', 1),
                          mech=mechanics(row), immunities=set(row.get('immunities', [])), wards=wards,
                          summon=False, appears=0, **profile))
    return units, worn


def guardian_model(world, wounds=0):
    row = UNITS[world.guardian_key]
    base = base_profile(world.guardian_key, world.guardian_level)
    cap = stat_caps(row)[DAMAGE]
    armor = base[ARMOR] + world.fort_armor
    game = row.get('game_data', {})
    regen = int(row.get('regeneration', game.get('REGEN', 0)) or 0)
    return dict(
        key=world.guardian_key, name=row['name'], level=world.guardian_level, max_hp=int(base[HP]),
        hp=max(1, int(base[HP]) - int(wounds)), damage=int(min(base[DAMAGE], cap)), cap=cap,
        weak_damage=int(min(round(base[DAMAGE] * WEAKEN_FACTOR), cap)), accuracy=base[ACCURACY],
        armor=armor, effective_armor=int(min(max(armor, 0), ARMOR_CAP)), initiative=base[INITIATIVE],
        source=row['attack_type'], role=row['role'], immunities=set(row.get('immunities', [])),
        wards=set(row.get('protections', [])), class_immunities=row.get('attack_class_immunities', []),
        category=game.get('UNIT_CAT'), capital=world.capital_guard, fort=world.fort, fort_armor=world.fort_armor,
        regen_percent=min(100, regen + (CAPITAL_REGENERATION if world.capital_guard else 0)),
        exp_kill=row['exp_kill'], primary_class=attack_class(row), lower_form=row.get('lower_form', ''))


# ------------------------------------------------------------------------------- combat arithmetic

@lru_cache(maxsize=None)
def hit_chance(accuracy):
    """accuracy_hits: two integer draws 0..99 hit when a + b < 2 * accuracy."""
    limit = 2 * float(accuracy)
    return sum(min(s, 198 - s) + 1 for s in range(199) if s < limit) / 10000


def hit_damage(power, armor, defended=False, cap=300):
    """Six capped raw rolls, minus the integer absorbed part, as in JAX combat."""
    power, armor = int(min(power, cap)), int(min(max(armor, 0), ARMOR_CAP))
    if power <= 0:
        return [0] * 6
    if defended:
        armor = min(ARMOR_CAP, armor+(100-armor)*50//100)
    raw = [min(power+b, cap) for b in range(6)]
    return [int(d-d*armor//100) for d in raw]


def before_chance(initiative, other):
    """P(initiative + 10u beats other + 10u'): _round_priority adds a uniform 0..10 draw."""
    x = (other - initiative) / 10.
    if x >= 1:
        return 0.
    if x <= -1:
        return 1.
    return (1 - x) ** 2 / 2 if x >= 0 else 1 - (1 + x) ** 2 / 2


def recovery(age):
    """effect_breaks: roll 1..100 >= 90 - 10 * age."""
    return min(1., max(0., (101 - (BAT_BREAK - BAT_BMODIF * age)) / 100))


def _convolve(a, b):
    out = defaultdict(float)
    for x, p in a.items():
        for y, q in b.items():
            out[x + y] += p * q
    return dict(out)


def _mean(dist):
    return sum(x * p for x, p in dist.items())


def stationary_dot(p, long):
    """Share of rounds with an active DOT that is refreshed with probability p each round."""
    if p <= 0:
        return 0.
    return p if not long else p / (1 - (1 - p) * (1 - DOT_BREAK_AGE1))


def vs_guardian(world, unit, guard, worn, defend=False, buff=1.):
    """What one party member does to the guardian per activation, and what one strike does to it."""
    m = dict(unit)
    mech, stats, secondary = unit['mech'], unit['stats'], unit['secondary']
    hit = hit_chance(stats[ACCURACY])
    blocked = mech['source'] in guard['immunities']
    zero = mech['witch'] or mech['disable'] > 0 or mech['fear'] or mech['healer'] or blocked
    power = stats[DAMAGE]
    if buff != 1 and power > 0 and not zero:
        power = min(round(unit['enhanced_damage'] * buff), unit['cap'])  # Vigor/Strength primary_override
    notes = []
    dist = {0: 1.}
    if not zero and power > 0:
        values = hit_damage(power, guard['effective_armor'], cap=unit['cap'])
        if mech['centaur']:
            values = [v + round(power / 20) for v in values]  # bypasses armour on a connected hit
        single = defaultdict(float, {0: 1 - hit})
        for v in values:
            single[v] += hit / 6
        for _ in range(2 if mech['double'] else 1):
            dist = _convolve(dist, dict(single))
    sec_hit = hit_chance(secondary[1]) if secondary[1] > 0 else 0.
    sec_blocked = bool(mech['secondary_source']) and mech['secondary_source'] in guard['immunities']
    control = None
    if mech['disable'] and not blocked:
        control = dict(kind='long' if mech['disable'] == LONG else 'short', p=hit, round40=True, how='основная атака')
    elif mech['secondary_disable'] and not blocked and not sec_blocked and sec_hit > 0:
        control = dict(kind='long' if mech['secondary_disable'] == LONG else 'short', p=hit * sec_hit,
                       round40=True, how='вторичная атака')
    elif (mech['fear'] or (mech['secondary_fear'] and sec_hit > 0)) and not blocked:
        p = hit if mech['fear'] else hit * sec_hit
        control = dict(kind='short' if guard['fort'] else 'flee', p=p, round40=False,
                       how='страх → паралич в укреплении' if guard['fort'] else 'страх: бегство')
    dot = None
    amount = int(secondary[0]) or DOT_FALLBACK.get(mech['dot'], 0)
    if (mech['dot'] and not blocked and DOT_SOURCES[mech['dot']] not in guard['immunities'] and sec_hit > 0
            and amount > 0):
        dot = dict(kind=mech['dot'], amount=amount, p=hit * sec_hit, long=mech['dot_long'])
    weaken = hit * sec_hit if mech['weaken'] and not blocked and not sec_blocked else 0.
    if unit['hero'] and mech['role'] != 'area' and not zero and power > 0:
        # Equipment procs follow a connected single-target hero hit (equipment_attack).
        for key in dict.fromkeys(k for k in worn[:2] if k):
            item = ITEM_BY_KEY[key]
            status, source = item.get('status'), ITEM_SOURCES.get(item.get('source', 0), '')
            if not status or source in guard['immunities']:
                continue
            if status == 'paralysis':
                p = hit * item['chance']
                if control and control['kind'] == 'short':
                    control = dict(control, p=1 - (1 - control['p']) * (1 - p))
                else:
                    control = dict(kind='short', p=p, round40=True, how=item['name'])
            elif status == 'poison':
                dot = dict(kind='poison', amount=item['poison'], p=hit * item['chance'], long=True)
            elif status == 'imp' and guard['capital']:
                notes.append(f"{item['name']}: страж исключён (number_grid_item_combat.py:103)")
            elif status == 'decay' and not guard['lower_form']:
                notes.append(f"{item['name']}: у стража нет пониженной формы")
    if mech['witch']:
        notes.append('превращение: стражи столиц исключены' if guard['capital'] else 'превращение врага')
    if mech['shatter'] and guard['capital']:
        notes.append('разрушение брони: страж исключён')
    if mech['wight'] and not guard['lower_form']:
        notes.append('понижение уровня: у стража нет пониженной формы')
    if blocked:
        notes.append(f"иммунитет стража к источнику {mech['source']}")
    strike = {}
    for kind, power_g in (('n', guard['damage']), ('w', guard['weak_damage'])):
        for guarded in (False, True):
            values = [0] * 6 if guard['source'] in unit['immunities'] else hit_damage(
                power_g, stats[ARMOR], guarded, cap=guard['cap'])
            strike[kind, guarded] = values
    expected = _mean(dist)
    dot_round = stationary_dot(dot['p'], dot['long']) * dot['amount'] if dot else 0.
    m.update(hit=hit, dist=dist, expected=expected, control=control, dot=dot, weaken=weaken, notes=notes,
             strike=strike, defend=defend, fast=before_chance(stats[INITIATIVE], guard['initiative']),
             ward_first=guard['source'] in unit['wards'], value=expected + dot_round,
             hp=float(stats[HP]), max_hp=float(stats[HP]))
    return m


def strikes_to_die(member, kind='n'):
    """Strikes of the guardian that kill a member at the mean bonus roll (min, mean, max)."""
    values = member['strike'][kind, False]
    if max(values) <= 0:
        return math.inf, math.inf, math.inf
    hp = member['max_hp']
    counts = [math.ceil(hp / v) if v > 0 else math.inf for v in (max(values), sum(values) / 6, min(values))]
    return tuple(counts)


# ------------------------------------------------------------------------------- party model

def copy_target(members, chosen=None):
    """Doppelganger target: the configured ally or the best small non-copy ally."""
    small = [m for m in members if m['size'] == 1 and not m['mech']['copy'] and not m.get('summon')]
    if chosen is not None:
        return next((m for m in small if m['slot'] == int(chosen)), None)
    return max(small, key=lambda m: ((m['control'] or {}).get('p', 0), m['value']), default=None)


def member_models(world, cand, guard, buff=1.):
    units, worn = party_profiles(world, cand)
    tactic = cand.get('tactic') or {}
    defend = {int(s) for s in tactic.get('defend_slots', [])}
    members = [vs_guardian(world, u, guard, worn, u['slot'] in defend, buff) for u in units]
    for i, m in enumerate(members):
        if not m['mech']['copy']:
            continue
        target = copy_target(members, (tactic.get('copy') or {}).get('ally'))
        if target is None:
            m['notes'].append('двойник: нет малой цели копирования')
            continue
        row = UNITS[target['key']]
        stats = list(unit_profile(target['key'], target['level'])['copy'])
        copy = dict(m, key=target['key'], name=f"Двойник → {target['name']}", mech=mechanics(row),
                    stats=tuple(stats), secondary=base_profile(target['key'], target['level'])[5:],
                    enhanced_damage=stats[DAMAGE], cap=stat_caps(row)[DAMAGE], copied_from=target['slot'])
        members[i] = vs_guardian(world, copy, guard, (None,) * 5, m['defend'], buff)
    return members, worn


def summon_member(world, item, guard, appears):
    key = GAME_IDS[item['summon']]
    row = UNITS[key]
    profile = unit_profile(key, row['level'])
    unit = dict(slot=None, key=key, name=f"{row['name']} (призыв)", level=row['level'], hero=False,
                size=row.get('size', 1), mech=mechanics(row), immunities=set(row.get('immunities', [])),
                wards=set(row.get('protections', [])), summon=True, appears=appears, **profile)
    return vs_guardian(world, unit, guard, (None,) * 5)


def timeline(members, sequence, items, rules):
    """Alive flags after each strike of a sequence ('n' normal, 'w' weakened), with hero items.

    The hero uses at most one battle item between strikes, choosing rules in priority order
    and item slots in order, like the simulate policy: Life fits a fallen ally, a heal fits a
    living ally at or below the rule's HP limit.
    """
    hp = [m['hp'] if m['appears'] == 0 else 0. for m in members]
    present = [m['appears'] == 0 for m in members]
    ward = [m['ward_first'] for m in members]
    hero = next((i for i, m in enumerate(members) if m['hero']), None)
    by_slot = {m['slot']: i for i, m in enumerate(members) if m['slot'] is not None}
    used = [False] * len(items)
    rows = [tuple(h > 0 for h in hp)]
    for step, kind in enumerate(sequence, 1):
        for i, m in enumerate(members):
            if hp[i] <= 0 or not present[i]:
                continue
            if ward[i]:
                ward[i] = False
                continue
            guarded = m['defend'] and (step >= 2 or m['fast'] >= .5)
            hp[i] = max(0., hp[i] - sum(m['strike'][kind, guarded]) / 6)
        for i, m in enumerate(members):
            if m['summon'] and m['appears'] == step:
                present[i], hp[i] = True, m['max_hp']
        if hero is not None and hp[hero] <= 0:
            for i, m in enumerate(members):
                if m['summon']:
                    hp[i], present[i] = 0., False
        if hero is not None and hp[hero] > 0:
            done = False
            for rule in rules:
                target = by_slot.get(rule.get('target')) if rule.get('side', 'ally') == 'ally' else None
                if target is None:
                    continue
                for j, item in enumerate(items):
                    if used[j] or item is None:
                        continue
                    t = members[target]
                    if item['kind'] == REVIVE and hp[target] <= 0:
                        restored = round(t['max_hp'] * item['percent'] / 100) if item.get('percent') else item['amount']
                        hp[target], used[j], done = max(1., float(restored)), True, True
                    elif (item['kind'] == HEAL and 0 < hp[target] <= rule.get('max_hp', math.inf)
                          and hp[target] < t['max_hp']):
                        receivers = [i for i in range(len(members)) if hp[i] > 0] if item.get('party') else [target]
                        for i in receivers:
                            hp[i] = min(members[i]['max_hp'], hp[i] + item['amount'])
                        used[j], done = True, True
                    if done:
                        break
                if done:
                    break
        rows.append(tuple(h > 0 and present[i] for i, h in enumerate(hp)))
    return rows


def _first_wipe(rows, permanent):
    for index, alive in enumerate(rows):
        if not any(a and p for a, p in zip(alive, permanent)):
            return index
    return None


def _group_dist(members, alive, fast):
    dist = {0: 1.}
    for m, a in zip(members, alive):
        if a and (m['fast'] >= .5) == fast and m['expected'] > 0:
            dist = _convolve(dist, m['dist'])
    return dist


def _control(members, alive, fast, kind, late):
    miss = 1.
    for m, a in zip(members, alive):
        c = m['control']
        if a and c and (m['fast'] >= .5) == fast and c['kind'] == kind and not (late and c['round40']):
            miss *= 1 - c['p']
    return 1 - miss


def _dot_dist(members, alive):
    dist = {0: 1.}
    for kind in ('poison', 'water', 'burn'):
        casters = [m['dot'] for m, a in zip(members, alive) if a and m['dot'] and m['dot']['kind'] == kind]
        if not casters:
            continue
        p = 1 - math.prod(1 - d['p'] for d in casters)
        share = stationary_dot(p, any(d['long'] for d in casters))
        amount = max(d['amount'] for d in casters)
        dist = _convolve(dist, {0: 1 - share, amount: share})
    return dist


def _kernel(dist, width):
    size = int(max(dist)) // width + 2
    out = np.zeros(size)
    for d, p in dist.items():
        low, rest = divmod(int(d), width)
        out[low] += p * (1 - rest / width)
        if rest:
            out[low + 1] += p * rest / width
    return out


def _stack(kernels):
    width = max(len(k) for k in kernels)
    return np.array([np.pad(k, (0, width - len(k))) for k in kernels])


def build_model(world, cand, opts=None):
    """Static party model, the binomial estimate and the Markov-chain tables of one candidate."""
    opts = opts or {}
    guard = guardian_model(world, cand.get('guardian_wounds', 0))
    catalog = world.battle_items
    items = [catalog[k] for k in cand.get('battle_items', []) if k]
    buff = max([i['multiplier'] for i in items if i['kind'] == BUFF], default=1.)
    members, worn = member_models(world, cand, guard, buff)
    hero = next((m for m in members if m['hero']), None)
    hero_fast = hero is not None and hero['fast'] >= .5
    gap_items = [i for i in items if i['kind'] in (HEAL, REVIVE)]
    opening = dict(weaken=None, long=None, short=None, damage=0., flags=[])
    for item in items:
        kind = item['kind']
        if kind not in OPENING_KINDS or hero is None:
            continue
        source = ITEM_SOURCES.get(item.get('source', 0), '')
        immune = source in guard['immunities']
        when = 'fast' if hero_fast else 'slow'
        if kind == WEAKEN and not immune:
            opening['weaken'] = when
        elif kind == PARALYSIS and not immune:
            opening['long'] = when
        elif kind == FEAR and not immune:
            opening['short' if guard['fort'] else 'flee'] = when
        elif kind == SUMMON:
            members.append(summon_member(world, item, guard, 0 if hero_fast else 1))
        elif kind in (STRIKE, DRAIN):
            values = hit_damage(item['amount'], guard['effective_armor'])
            opening['damage'] += hit_chance(item['accuracy']) * sum(values) / 6
        elif kind == DOT and DOT_SOURCES[('poison', 'burn', 'water')[item['dot']]] not in guard['immunities']:
            ticks, alive = 0., 1.
            for age in range(1, 11):
                ticks += alive
                alive *= 1 - recovery(age)
            opening['damage'] += item['amount'] * ticks
        elif kind in (TRANSFORM, LYCANTHROPY) and not immune:
            opening['flags'].append('guard_transform')
    rules = (cand.get('tactic') or {}).get('item_rules') or default_item_rules(members)
    permanent = [not m['summon'] for m in members]
    has_weaken = any(m['weaken'] > 0 for m in members) or opening['weaken'] is not None
    has_long = any(m['control'] and m['control']['kind'] == 'long' for m in members) or opening['long'] is not None
    cap = int(opts.get('max_strikes', 12))
    rows_normal = timeline(members, 'n' * cap, gap_items, rules)
    wipe = _first_wipe(rows_normal, permanent)
    strikes = cap if wipe is None else wipe
    if has_weaken:
        weak_wipe = _first_wipe(timeline(members, 'w' * cap, gap_items, rules), permanent)
        strikes = max(strikes, cap if weak_wipe is None else weak_wipe)
    strikes = max(1, min(strikes, cap))
    rows = {a: timeline(members, 'n' * a + 'w' * (strikes - a), gap_items, rules)
            for a in (range(strikes + 1) if has_weaken else [strikes])}
    states = [(a, 0, 0) for a in range(strikes + 1)]
    if has_weaken:
        states += [(a, b, 1) for a in range(strikes + 1) for b in range(strikes + 1 - a)]
    index = {s: i for i, s in enumerate(states)}

    def alive_at(state):
        a, b, w = state
        return rows[a][a + b] if w else rows[strikes if not has_weaken else a][a]

    alive = [alive_at(s) for s in states]
    width = int(opts.get('hp_bin', 5))
    cache = {}

    def tables(flags):
        if flags not in cache:
            fast, slow, dot = _group_dist(members, flags, True), _group_dist(members, flags, False), _dot_dist(members, flags)
            cache[flags] = dict(
                fast=fast, slow=slow, dot=dot, kernels=(_kernel(fast, width), _kernel(slow, width), _kernel(dot, width)),
                control={(f, k, late): _control(members, flags, f, k, late)
                         for f in (True, False) for k in ('short', 'long', 'flee') for late in (False, True)},
                weaken=1 - math.prod(1 - m['weaken'] for m, a in zip(members, flags) if a and m['weaken'] > 0))
        return cache[flags]

    info = [tables(flags) for flags in alive]
    next_strike = []
    for a, b, w in states:
        if a + b >= strikes:
            next_strike.append(index[a, b, w])
        else:
            next_strike.append(index[(a + 1, b, 0) if not w else (a, b + 1, 1)])
    wipes = np.array([not any(x and p for x, p in zip(alive[n], permanent)) for n in next_strike])
    weaken_next = [index[(a, 0, 1)] if not w and has_weaken else index[a, b, w] for a, b, w in states]
    hero_index = next((i for i, m in enumerate(members) if m['hero']), None)
    hero_alive = np.array([hero_index is not None and flags[hero_index] for flags in alive], float)
    model = dict(
        guard=guard, members=members, worn=worn, rules=rules, opening=opening, states=states, alive=alive,
        hp=max(1., guard['hp'] - opening['damage']), width=width, has_long=has_long, has_weaken=has_weaken,
        max_rounds=world.max_rounds, strikes=strikes, permanent=permanent,
        fast=_stack([i['kernels'][0] for i in info]), slow=_stack([i['kernels'][1] for i in info]),
        dot=_stack([i['kernels'][2] for i in info]), next_strike=np.array(next_strike), wipes=wipes,
        weaken_next=np.array(weaken_next),
        weaken=np.array([0. if s[2] else i['weaken'] for s, i in zip(states, info)]),
        fast_short={late: np.array([i['control'][True, 'short', late] for i in info]) for late in (False, True)},
        fast_long={late: np.array([i['control'][True, 'long', late] for i in info]) for late in (False, True)},
        slow_short={late: np.array([i['control'][False, 'short', late] for i in info]) for late in (False, True)},
        slow_long={late: np.array([i['control'][False, 'long', late] for i in info]) for late in (False, True)},
        flee={late: np.array([1 - (1 - i['control'][True, 'flee', late]) * (1 - i['control'][False, 'flee', late])
                              for i in info]) for late in (False, True)},
        expected={s: _mean(i['fast']) + _mean(i['slow']) + _mean(i['dot']) for s, i in zip(states, info)},
        hero_alive=hero_alive, start=index[(0, 0, 1 if opening['weaken'] == 'fast' else 0)])
    model['binomial'] = binomial(model)
    return model


def binomial(model):
    """P ≈ Σ_{j≤k} C(R−1, j) q^j (1−q)^(R−1−j) after the opening strike.

    q: all controllers alive after the opening miss in a round (pair rolls); R: rounds to remove
    the guardian's HP at the expected damage per round with armour, accuracy and DOT; k: extra
    strikes absorbed without losing a controller. p repeats the sum with q = 1 for paralysis and
    petrify after round 40; formula is the plain textbook sum.
    """
    members, states = model['members'], model['states']
    start, opening = model['start'], model['opening']
    if opening['long'] == 'fast' or opening['short'] == 'fast':
        first = 0.
    else:
        first = (1 - model['fast_short'][False][start]) * (1 - model['fast_long'][False][start])
    after = model['next_strike'][start] if first >= .5 else start
    result = dict(p=0., formula=0., q=1., R=None, k=0, damage=0., controllers=0, opening_strike=round(first, 4))
    if first >= .5 and model['wipes'][start]:
        return dict(result, reason='отряд гибнет от первого удара')
    alive = model['alive'][after]
    controllers = [m for m, a in zip(members, alive) if a and m['control'] and m['control']['kind'] != 'flee']
    q = math.prod(1 - m['control']['p'] for m in controllers)
    q_late = math.prod(1 - m['control']['p'] for m in controllers if not m['control']['round40'])
    damage = model['expected'][states[after]]
    rounds = math.ceil(model['hp'] / damage) if damage > 0 else None
    # Extra strikes the party absorbs without losing a controller (or everyone, if none).
    k, state = 0, after
    while True:
        following = model['next_strike'][state]
        if following == state or model['wipes'][state]:
            break
        count = sum(1 for m, a in zip(members, model['alive'][following]) if a and m['control'])
        if controllers and count < len(controllers):
            break
        k, state = k + 1, following
    result.update(q=q, R=rounds, k=k, damage=damage, controllers=len(controllers))
    if rounds is None or rounds > model['max_rounds']:
        return dict(result, reason='урона не хватает до лимита раундов')
    counts = np.zeros(k + 2)
    counts[0] = 1.
    for r in range(2, rounds + 1):
        strike = q_late if r > DISABLE_ALLOWED_ROUND_MAX else q
        counts = np.concatenate(([counts[0] * (1 - strike)], counts[1:] * (1 - strike) + counts[:-1] * strike))
    textbook = sum(math.comb(rounds - 1, j) * q ** j * (1 - q) ** (rounds - 1 - j) for j in range(k + 1))
    return dict(result, p=float(counts[:k + 1].sum()), formula=float(textbook))


def _shift(P, kernels):
    """Move remaining-HP mass down by kernel offsets; mass at or below zero lands in bin 0."""
    out = np.zeros_like(P)
    cum = np.cumsum(P, axis=2)
    size = P.shape[2]
    for offset in range(kernels.shape[1]):
        w = kernels[:, offset]
        if not w.any():
            continue
        if offset == 0:
            out += w[:, None, None] * P
        elif offset < size:
            out[:, :, 1:size - offset] += w[:, None, None] * P[:, :, 1 + offset:]
            out[:, :, 0] += w[:, None] * cum[:, :, offset]
        else:
            out[:, :, 0] += w[:, None] * cum[:, :, -1]
    return out


def markov(model):
    """Round-by-round chain over strike history, guardian disable state and remaining HP bins.

    Round order: units faster than the guardian, DOT ticks, the guardian's area strike unless
    paralysed (short) or still petrified/paralysed long, then the slower units, whose controls
    and weakening apply to the next round. Battle items act through the alive tables.
    """
    states = model['states']
    K = len(states)
    ages = [None] + (list(range(10)) if model['has_long'] else [])
    D = [(s, a) for a in ages for s in (0, 1)]
    index = {d: i for i, d in enumerate(D)}
    bins = max(1, math.ceil(model['hp'] / model['width']))
    P = np.zeros((K, len(D), bins + 1))
    opening = model['opening']
    start_d = (1 if opening['short'] == 'fast' else 0, 0 if opening['long'] == 'fast' and model['has_long'] else None)
    P[model['start'], index[start_d], bins] = 1.
    rounds = model['max_rounds']
    win, lost, fled = np.zeros(rounds + 1), np.zeros(rounds + 1), 0.
    ident = np.arange(K)
    for r in range(1, rounds + 1):
        late = r > DISABLE_ALLOWED_ROUND_MAX
        P = _shift(P, model['fast'])
        win[r] += P[:, :, 0].sum()
        P[:, :, 0] = 0
        if r >= 2:
            P = _shift(P, model['dot'])
            win[r] += P[:, :, 0].sum()
            P[:, :, 0] = 0
        # The guardian's activation.
        out = np.zeros_like(P)
        fs, fl = model['fast_short'][late], model['fast_long'][late]
        for di, (short, age) in enumerate(D):
            mass = P[:, di, :]
            if not mass.any():
                continue
            for fs_flag in (0, 1):
                w_short = fs if fs_flag else 1 - fs
                for fl_flag in ((0, 1) if age is None and model['has_long'] else (0,)):
                    weight = w_short * (fl if fl_flag else 1 - fl)
                    if not weight.any():
                        continue
                    current = age if age is not None else (0 if fl_flag else None)
                    skip = bool(short or fs_flag)
                    if current is None:
                        branches = [(1., None, not skip)]
                    else:
                        rec = recovery(current)
                        branches = [(1 - rec, min(current + 1, 9), False), (rec, None, not skip)]
                    for prob, next_age, strike in branches:
                        if prob <= 0:
                            continue
                        part = mass * (weight * prob)[:, None]
                        target = out[:, index[(0, next_age)], :]
                        if not strike:
                            target += part
                            continue
                        keep = ~model['wipes']
                        lost[r] += part[~keep].sum()
                        np.add.at(target, model['next_strike'][keep], part[keep])
        P = out
        P = _shift(P, model['slow'])
        win[r] += P[:, :, 0].sum()
        P[:, :, 0] = 0
        # Slower units' controls and weakening, for the next round.
        flee = model['flee'][late]
        if flee.any():
            fled += float((P.sum(axis=(1, 2)) * flee).sum())
            P = P * (1 - flee)[:, None, None]
        out = np.zeros_like(P)
        ss, sl = model['slow_short'][late], model['slow_long'][late]
        pw = model['weaken'].copy()
        if r == 1:
            if opening['weaken'] == 'slow':
                pw = np.where(np.array([s[2] for s in states]) == 0, model['hero_alive'], 0.)
            if opening['long'] == 'slow':
                sl = 1 - (1 - sl) * (1 - model['hero_alive'])
            if opening['short'] == 'slow':
                ss = 1 - (1 - ss) * (1 - model['hero_alive'])
        for di, (_, age) in enumerate(D):  # the activation has consumed any short paralysis
            mass = P[:, di, :]
            if not mass.any():
                continue
            for s_flag in (0, 1):
                w_short = ss if s_flag else 1 - ss
                for l_flag in ((0, 1) if age is None and model['has_long'] else (0,)):
                    w_long = w_short * (sl if l_flag else 1 - sl)
                    target = out[:, index[(s_flag, 1 if l_flag else age)], :]
                    for w_flag in ((0, 1) if model['has_weaken'] else (0,)):
                        weight = w_long * (pw if w_flag else 1 - pw)
                        if weight.any():
                            np.add.at(target, model['weaken_next'] if w_flag else ident, mass * weight[:, None])
        P = out
    total_win = float(win.sum()) + fled
    by_round = np.arange(rounds + 1)
    return dict(p=total_win, loss=float(lost.sum()), timeout=float(P.sum()), fled=fled,
                win_by_40=float(win[:DISABLE_ALLOWED_ROUND_MAX + 1].sum()),
                rounds_when_won=float((win * by_round).sum() / win.sum()) if win.sum() > 0 else None,
                rounds_when_lost=float((lost * by_round).sum() / lost.sum()) if lost.sum() > 0 else None)


def tactic_rules(model, cand):
    """Rule ids a tactic relies on, from the mechanisms present in its model."""
    ids = ['guardian_attack', 'capital_armor', 'armor_cap', 'damage_roll', 'pair_roll', 'initiative_roll',
           'round_limit', 'area_rolls', 'leadership', 'hero_levels', 'equipment_skills', 'recruit_slots']
    members = model['members']
    controls = [m['control'] for m in members if m['control']]
    if controls:
        ids += ['short_disable', 'damage_vs_disabled']
        if any(c['round40'] for c in controls):
            ids.append('disable_round40')
        if any(c['kind'] == 'long' for c in controls) or model['opening']['long']:
            ids.append('long_disable')
        if any(c['how'] == 'вторичная атака' for c in controls):
            ids.append('secondary_after_primary')
        if any('страх' in c['how'] for c in controls):
            ids.append('fear_fort')
    if any(m['dot'] for m in members):
        ids += ['dot_tick', 'dot_armor', 'dot_refresh', 'effect_breaks', 'secondary_after_primary']
    # A damage/accuracy potion on a unit whose DOT, secondary disable or weakening matters:
    # the estimate assumes the potion leaves that secondary attack unchanged.
    secondary = {m['slot'] for m in members if m['dot'] or m['weaken'] > 0
                 or (m['control'] and m['control']['how'] == 'вторичная атака')}
    if any(POTION_BY_KEY[k]['stat'] in (DAMAGE, ACCURACY) and int(s) in secondary for k, s in cand.get('potions', [])):
        ids.append('dot_potions')
    if any(m['weaken'] > 0 for m in members) or model['opening']['weaken']:
        ids += ['weakening', 'secondary_after_primary']
    potions = [POTION_BY_KEY[k] for k, _ in cand.get('potions', [])]
    if any(p['duration'] == 'temporary' for p in potions):
        ids.append('potion_until_rest')
    if any(p['duration'] == 'permanent' for p in potions):
        ids.append('permanent_once')
    if any(p['key'] == 'invulnerability' for p in potions):
        ids.append('invulnerability')
    if [k for k in cand.get('battle_items', []) if k]:
        ids += ['equip_daily', 'item_turn']
        kinds = {model_item['kind'] for model_item in
                 (WORLD_ITEMS.get(k, {'kind': None}) for k in cand.get('battle_items', []) if k)}
        if REVIVE in kinds:
            ids.append('life_potion')
        if SUMMON in kinds:
            ids.append('summon_owner')
    if any(c for c in cand.get('battle_items', []) if c and WORLD_ITEMS.get(c, {}).get('category') in ('orb', 'talisman')):
        ids.append('magic_access')
    if any(m['hero'] and m['fast'] >= 1 for m in members):
        ids.append('hero_first')
    if any(m.get('copied_from') is not None or m['mech']['copy'] for m in members):
        ids.append('copy_guard')
    if 'guard_transform' in model['opening']['flags']:
        ids.append('transform_orb_guard')
    if cand.get('guardian_wounds'):
        ids.append('wounds_persist')
    return list(dict.fromkeys(ids))


WORLD_ITEMS = {}  # battle-item catalogue of the loaded scenario, filled by load_world


def load_world(map_name, guardian):
    world = World(map_name, guardian)
    WORLD_ITEMS.clear()
    WORLD_ITEMS.update(world.battle_items)
    return world


# ------------------------------------------------------------------------------- plans and candidates

def default_item_rules(members):
    """Controllers first, then the hero, then the strongest damage dealers.

    Revive a fallen one; heal a living one only when the next strike would kill it and a full
    heal lets it survive one (limit 0: revive only). The policy also requires a wounded target.
    """
    def limit(m):
        strike = int(max(m['strike']['n', False]))
        return strike if m['max_hp'] > strike else 0
    order = sorted([m for m in members if m['control'] and m['slot'] is not None], key=lambda m: -m['control']['p'])
    order += [m for m in members if m['hero']]
    order += sorted([m for m in members if not m['control'] and not m['hero'] and m['slot'] is not None
                     and m['value'] > 0], key=lambda m: -m['value'])
    slots = list(dict.fromkeys(m['slot'] for m in order))
    by_slot = {m['slot']: m for m in members}
    return [dict(side='ally', target=s, max_hp=limit(by_slot[s])) for s in slots]


def _member(world, cand, slot, extra, guard):
    """Model of one slot with its planned potions plus extra ones (the hero kept for equipment)."""
    potions = [p for p in cand.get('potions', []) if int(p[1]) == slot] + [[k, slot] for k in extra]
    roster = [k if i in (slot, world.hero_slot) else None for i, k in enumerate(cand['roster'])]
    units, worn = party_profiles(world, dict(cand, potions=potions, roster=roster))
    unit = next(u for u in units if u['slot'] == slot)
    return vs_guardian(world, unit, guard, worn)


def allocate_potions(world, cand, guard):
    """Heuristic plan: survival potions to controllers, hero, dealers; damage to the best dealer."""
    stock = {k: n for k, n in cand_potions(world, cand).items() if n > 0 and POTION_BY_KEY[k]['duration'] != 'instant'}
    plan = dict(cand, potions=[])
    units, worn = party_profiles(world, plan)
    members = {u['slot']: vs_guardian(world, u, guard, worn) for u in units}

    def give(key, slot):
        plan['potions'].append([key, slot])
        stock[key] -= 1
        members[slot] = _member(world, plan, slot, [], guard)

    def has(key, slot):
        return any(k == key and int(s) == slot for k, s in plan['potions'])

    ctrl = sorted((m for m in members.values() if m['control']), key=lambda m: -m['control']['p'])
    hero = [m for m in members.values() if m['hero']]
    rest = sorted((m for m in members.values() if not m['control'] and not m['hero']), key=lambda m: -m['value'])
    order = list(dict.fromkeys(m['slot'] for m in ctrl + hero + rest))
    for key in SURVIVAL_POTIONS:
        for slot in order:
            if stock.get(key, 0) <= 0:
                break
            if has(key, slot):
                continue
            before = strikes_to_die(members[slot])[1]
            after = strikes_to_die(_member(world, plan, slot, [key], guard))[1]
            if after > before:
                give(key, slot)
        # Leftovers still raise HP/armour for heals and later strikes: the hero first.
        for slot in [h['slot'] for h in hero] + order:
            if stock.get(key, 0) > 0 and not has(key, slot):
                give(key, slot)
    dealers = [s for s in order if members[s]['value'] > 0]
    damage = [k for k in DAMAGE_POTIONS if stock.get(k, 0) > 0]
    if dealers and damage:
        best = max(dealers, key=lambda s: _member(world, plan, s, damage, guard)['value'] - members[s]['value'])
        for key in damage:
            if not has(key, best):
                give(key, best)
    accuracy = [k for k in ACCURACY_POTIONS if stock.get(k, 0) > 0]
    if accuracy:
        def gain(slot):
            new = _member(world, plan, slot, accuracy, guard)
            old = members[slot]
            return ((new['control'] or {}).get('p', 0) - (old['control'] or {}).get('p', 0),
                    new['value'] - old['value'])
        best = max(order, key=gain)
        for key in accuracy:
            if not has(key, best):
                give(key, best)
    speed = [k for k in INITIATIVE_POTIONS if stock.get(k, 0) > 0]
    if speed and hero:
        for key in speed:
            if not has(key, hero[0]['slot']) and stock[key] > 0:
                give(key, hero[0]['slot'])
    for key, source in WARD_POTIONS.items():
        if source == guard['source']:
            for slot in order:
                if stock.get(key, 0) > 0 and not has(key, slot):
                    give(key, slot)
    permanent = [p for p in plan['potions'] if POTION_BY_KEY[p[0]]['duration'] == 'permanent']
    return permanent + [p for p in plan['potions'] if POTION_BY_KEY[p[0]]['duration'] != 'permanent']


def battle_item_options(world, cand, worn):
    access = magic_access(world, int(cand['hero_level']), worn)
    potions, items = cand_potions(world, cand), cand_items(world, cand)
    options = []
    for key, item in world.battle_items.items():
        stock = potions.get(key, 0) if item['category'] == 'potion' else items.get(key, 0)
        if stock and (item['category'] == 'potion' or access[item['category']]):
            options.append((key, stock))
    return options


def default_battle_items(world, cand, members, worn):
    options = dict(battle_item_options(world, cand, worn))
    catalog = world.battle_items
    used = defaultdict(int)
    for key, _ in cand.get('potions', []):
        used[key] += 1
    picks = []

    def take(key):
        if key and options.get(key, 0) - used[key] > 0 and len(picks) < 2:
            picks.append(key)
            used[key] += 1
            return True
        return False

    def best(kind, score):
        keys = [k for k in options if catalog[k]['kind'] == kind and options[k] - used[k] > 0]
        return max(keys, key=score) if keys else None

    def heal_score(key):
        return not catalog[key].get('party'), catalog[key]['amount']

    def revive_score(key):
        return catalog[key].get('percent', 0), -catalog[key].get('price', 0)

    take(best(WEAKEN, len))  # any key: there is at most one kind of each
    take(best(PARALYSIS, len))
    if any(m['control'] for m in members):
        take(best(REVIVE, revive_score))
    while len(picks) < 2:
        if not take(best(HEAL, heal_score)) and not take(best(REVIVE, revive_score)):
            break
    return picks


def opening_rules(world, items):
    """Simulate rules for items used at the hero's first chance: enemy target or any ally slot."""
    rules = []
    for index, key in enumerate(items):
        item = world.battle_items[key]
        if item['kind'] in OPENING_KINDS:
            enemy = bool(item.get('enemy'))
            rules.append(dict(side='enemy' if enemy else 'ally', target='guardian' if enemy else 'any',
                              item=key, slot_index=index + 1))
    return rules


def default_plan(world, cand):
    guard = guardian_model(world, cand.get('guardian_wounds', 0))
    potions = allocate_potions(world, cand, guard)
    planned = dict(cand, potions=potions, battle_items=[], tactic={})
    members, worn = member_models(world, planned, guard)
    items = default_battle_items(world, planned, members, worn)
    tactic = dict(item_rules=opening_rules(world, items) + default_item_rules(members), defend_slots=[])
    for m in members:
        if m.get('copied_from') is not None:
            tactic['copy'] = dict(slot=m['slot'], ally=m['copied_from'])
    return potions, items, tactic


def formation(world, units):
    """Recruitment placement: large units front with an empty rear, melee front, the rest behind."""
    roster = [None] * 6
    roster[world.hero_slot] = world.hero_key
    covered = set()
    for key in [k for k in units if UNITS[k].get('size', 1) == 2]:
        slot = next((s for s in range(3) if roster[s] is None and roster[s + 3] is None and s + 3 not in covered), None)
        if slot is None:
            return None
        roster[slot] = key
        covered.add(slot + 3)
    small = [k for k in units if UNITS[k].get('size', 1) == 1]
    for key in [k for k in small if UNITS[k]['role'] == 'melee']:
        slot = next((s for s in range(3) if roster[s] is None), None)
        if slot is None:
            return None
        roster[slot] = key
    for key in [k for k in small if UNITS[k]['role'] != 'melee']:
        slot = next((s for s in range(3, 6) if roster[s] is None and s not in covered), None)
        if slot is None:
            return None
        roster[slot] = key
    return roster


def unit_pool(world, mode='final', explicit=None):
    if explicit:
        missing = [k for k in explicit if k not in world.forms]
        if missing:
            raise SystemExit('Not reachable for this faction/map: ' + ', '.join(missing))
        return list(explicit)
    keys = [k for k, form in world.forms.items() if not UNITS[k].get('hero') and (mode == 'all' or form['final'])]
    return sorted(keys)


def multisets(pool, budget, limits):
    def walk(start, budget, chosen):
        yield list(chosen)
        for i in range(start, len(pool)):
            key = pool[i]
            size = UNITS[key].get('size', 1)
            if size <= budget and chosen.count(key) < limits.get(key, 5):
                chosen.append(key)
                yield from walk(i, budget - size, chosen)
                chosen.pop()
    yield from walk(0, budget, [])


def make_candidate(world, roster, hero_level, levels=None, potions=None, battle_items=None, tactic=None,
                   name=None, wounds=0):
    cand = dict(map=world.map_name, guardian=world.guardian, hero_level=int(hero_level), roster=list(roster),
                levels={str(k): int(v) for k, v in (levels or {}).items()}, guardian_wounds=int(wounds))
    if potions is None or battle_items is None or tactic is None:
        plan = default_plan(world, cand)
        potions = plan[0] if potions is None else potions
        battle_items = plan[1] if battle_items is None else battle_items
        tactic = plan[2] if tactic is None else tactic
    cand.update(potions=[list(p) for p in potions], battle_items=list(battle_items), tactic=tactic)
    cand['name'] = name or describe(world, cand)
    return cand


def describe(world, cand):
    counts = defaultdict(int)
    for slot, key in enumerate(cand['roster']):
        if key and slot != world.hero_slot:
            level = (cand.get('levels') or {}).get(str(slot))
            counts[world.name(key) + (f' ур.{level}' if level else '')] += 1
    party = ', '.join(f'{n}×{c}' if c > 1 else n for n, c in counts.items())
    return f"{world.name(world.hero_key)} ур.{cand['hero_level']} + {party or 'без отряда'}"


def generate(world, hero_levels, pool, include_partial=False, allow_transform=False):
    limits = {k: (world.offers[k]['stock'] or 5) for k in pool if k in world.offers and world.offers[k]['mercenary']}
    out, seen = [], set()
    for level in hero_levels:
        capacity = world.capacity(level)
        for units in multisets(pool, capacity, limits):
            used = sum(UNITS[k].get('size', 1) for k in units)
            if not units or (not include_partial and used < capacity - 1):
                continue
            if world.conflict(units):
                continue
            roster = formation(world, units)
            if roster is None:
                continue
            signature = (level, tuple(roster))
            if signature in seen:
                continue
            seen.add(signature)
            cand = make_candidate(world, roster, level)
            if not allow_transform and any(world.battle_items[k]['kind'] in (TRANSFORM, LYCANTHROPY)
                                           for k in cand['battle_items'] if k):
                continue
            out.append(cand)
    return out


def variants(world, cand, limit_items=None):
    """Small search over invulnerability recipients, the damage-potion recipient and battle items."""
    guard = guardian_model(world, cand.get('guardian_wounds', 0))
    members, worn = member_models(world, dict(cand, potions=[], battle_items=[], tactic={}), guard)
    slots = [m['slot'] for m in members]
    potions = cand_potions(world, cand)
    base = [p for p in cand['potions'] if p[0] not in ('invulnerability',) + DAMAGE_POTIONS]
    invulnerable = min(potions.get('invulnerability', 0), len(slots))
    damage = [k for k in DAMAGE_POTIONS if potions.get(k, 0) > 0]
    dealers = [m['slot'] for m in members if m['value'] > 0] or [world.hero_slot]
    plans = []
    for guarded in itertools.combinations(slots, invulnerable):
        for dealer in dealers:
            plan = [p for p in base if POTION_BY_KEY[p[0]]['duration'] == 'permanent']
            plan += [[k, dealer] for k in damage if POTION_BY_KEY[k]['duration'] == 'permanent']
            plan += [p for p in base if POTION_BY_KEY[p[0]]['duration'] != 'permanent']
            plan += [['invulnerability', s] for s in guarded]
            plan += [[k, dealer] for k in damage if POTION_BY_KEY[k]['duration'] != 'permanent']
            plans.append(plan)
    options = battle_item_options(world, cand, worn)
    keys = [k for k, _ in options]
    pairs = [list(p) for p in itertools.combinations_with_replacement(keys, 2)
             if p[0] != p[1] or dict(options)[p[0]] >= 2] + [[k] for k in keys]
    if limit_items:
        pairs = pairs[:limit_items]
    return plans, pairs


# ------------------------------------------------------------------------------- cost model

def chain_steps(world, key, level, hero=False):
    """XP thresholds and trainer prices from the chain root to a form at a level."""
    steps = []
    chain = [key] if hero else world.forms.get(key, dict(chain=[key]))['chain']
    for form in chain[:-1]:
        row = UNITS[form]
        steps.append(dict(form=form, level=row['level'], xp=row['exp_required'],
                          price=world.train_price(form, row['level'])))
    row = UNITS[key]
    for current in range(row['level'], level):
        xp = row['exp_required'] + (current - row['level']) * row.get('exp_increment', 0)
        steps.append(dict(form=key, level=current, xp=xp, price=world.train_price(key, current)))
    return steps


def cost(world, cand, pool=None, chunk=5):
    """XP flow with hires as leadership grows, trainer deficits and gold; optimistic (see docs)."""
    pool = world.xp_pool if pool is None else pool
    level_map = {int(k): int(v) for k, v in (cand.get('levels') or {}).items()}
    hero_steps = chain_steps(world, world.hero_key, int(cand['hero_level']), hero=True)
    thresholds = list(itertools.accumulate(s['xp'] for s in hero_steps))
    starting = [k for _, k in world.starting]
    members = [dict(key=world.hero_key, slot=world.hero_slot, size=0, steps=hero_steps, origin='герой', xp=0)]
    pending = []
    for slot, key in enumerate(cand['roster']):
        if not key or slot == world.hero_slot:
            continue
        level = level_map.get(slot, UNITS[key]['level'])
        form = world.forms.get(key, dict(root=key, mercenary=False))
        entry = dict(key=key, slot=slot, size=UNITS[key].get('size', 1), steps=chain_steps(world, key, level),
                     root=form['root'], xp=0)
        if form['root'] in starting:
            starting.remove(form['root'])
            entry['origin'] = 'стартовый ' + world.name(form['root'])
            members.append(entry)
        else:
            offer = world.offers.get(form['root'], {})
            entry['origin'] = ('наёмник ' if offer.get('mercenary') else 'найм ') + world.name(form['root'])
            entry['price'] = offer.get('price', 0) or 0
            pending.append(entry)
    pending.sort(key=lambda e: -sum(s['xp'] for s in e['steps']))
    decisions = [f"уволить стартового {world.name(k)}" for k in starting]
    trained = [False]  # the trainer has finished the hero: his levels open the remaining places

    def hero_level():
        if trained[0]:
            return int(cand['hero_level'])
        return 1 + sum(1 for t in thresholds if members[0]['xp'] >= t)

    def hire(label):
        free = world.capacity(hero_level()) - sum(m['size'] for m in members)
        for entry in list(pending):
            if entry['size'] <= free:
                members.append(entry)
                pending.remove(entry)
                free -= entry['size']
                source = entry['origin'][len('найм '):] if entry['origin'].startswith('найм ') else entry['origin']
                decisions.append(f"{label}: нанять {source} → {world.name(entry['key'])}")

    hire('ур.1')
    spent = 0
    # Battle XP is shared equally by every member, including those already at their target.
    while spent < pool and any(m['xp'] < sum(s['xp'] for s in m['steps']) for m in members):
        for m in members:
            m['xp'] += chunk
        spent += chunk * len(members)
        if pending:
            hire(f'ур.{hero_level()}')
    if pending:
        trained[0] = True
        hire('после тренера героя')
        if pending:
            decisions.append('не хватает лидерства: ' + ', '.join(world.name(e['key']) for e in pending))
    rows, trainer_gold, missing_price = [], 0, False
    for m in members:
        target = sum(s['xp'] for s in m['steps'])
        received, start, gold = min(m['xp'], target), 0, 0
        for step in m['steps']:
            covered = min(max(received - start, 0), step['xp'])
            if step['price'] is None:
                missing_price = True
            else:
                gold += (step['xp'] - covered) * step['price']
            start += step['xp']
        trainer_gold += gold
        rows.append(dict(unit=world.name(m['key']), slot=m['slot'], origin=m['origin'], target_xp=target,
                         battle_xp=received, trainer_xp=max(target - received, 0), trainer_gold=gold,
                         recruit_gold=m.get('price', 0)))
    keys = [k for k in cand['roster'] if k]
    buildings = sorted(set().union(*[world.forms[k]['buildings'] for k in keys if k in world.forms]))
    building_gold = sum(world.buildings[b]['gold'] for b in buildings)
    shop_gold, shop = 0, []
    need = defaultdict(int)
    for key, _ in cand.get('potions', []):
        need[key] += 1
    for key in cand.get('battle_items', []):
        if key:
            need[key] += 1
    for key, count in need.items():
        if key in POTION_BY_KEY:
            bought = max(0, count - world.free_potions.get(key, 0))
            price = POTION_BY_KEY[key]['price']
            available = world.shop_potions.get(key, 0)
        else:
            bought = max(0, count - world.free_items.get(key, 0))
            price = ITEM_BY_KEY.get(key, {}).get('price', 0)
            available = world.shop_items.get(key, 0)
        if bought:
            shop.append(dict(item=key, count=bought, gold=bought * price, in_stock=available >= bought))
            shop_gold += bought * price
    worn = choose_equipment(world, cand_items(world, cand), int(cand['hero_level']))
    for key in dict.fromkeys(k for k in worn if k):
        if world.free_items.get(key, 0) == 0 and world.shop_items.get(key, 0):
            price = ITEM_BY_KEY[key]['price']
            shop.append(dict(item=key, count=1, gold=price, in_stock=True))
            shop_gold += price
    recruit_gold = sum(r['recruit_gold'] for r in rows)
    total = trainer_gold + building_gold + recruit_gold + shop_gold
    return dict(xp_pool=pool, xp_pool_used=min(spent, pool), xp_needed=sum(r['target_xp'] for r in rows),
                trainer_xp=sum(r['trainer_xp'] for r in rows), trainer_gold=trainer_gold,
                building_gold=building_gold, buildings=buildings, recruit_gold=recruit_gold, shop_gold=shop_gold,
                shop=shop, total_gold=total, members=rows, decisions=decisions, gold_sources=world.gold_sources,
                trainer_price_missing=missing_price, rules=['xp_split', 'promotion_once', 'trainer', 'leadership',
                                                            'guard_no_xp', 'recruit_slots'],
                notes=['оптимистично: одно повышение за бой и XP сверх порога не учтены',
                       'последний XP каждого порога должен прийти из боя (тренер — до порога − 1)',
                       'золото дохода по дням не учтено: см. gold_sources и доход столицы'])


# ------------------------------------------------------------------------------- screening

def evaluate(world, cand, opts, chain=True):
    model = build_model(world, cand, opts)
    result = dict(binomial=model['binomial'])
    if chain:
        result['markov'] = markov(model)
    members = model['members']
    result.update(
        p=result['markov']['p'] if chain else model['binomial']['p'],
        controllers=sum(1 for m in members if m['control']),
        damage_round=model['binomial'].get('damage', 0.),
        extra_lives=model['binomial'].get('k', 0),
        rules=tactic_rules(model, cand),
        members=[dict(slot=m['slot'], name=m['name'], level=m['level'], hp=m['max_hp'],
                      armor=m['stats'][ARMOR], damage=m['stats'][DAMAGE], accuracy=m['stats'][ACCURACY],
                      initiative=m['stats'][INITIATIVE], hit=round(m['hit'], 4), per_activation=round(m['expected'], 2),
                      control=m['control'], dot=m['dot'], weaken=round(m['weaken'], 4), before_guardian=round(m['fast'], 3),
                      strikes_to_die=strikes_to_die(m), notes=m['notes']) for m in members],
        guardian={k: v for k, v in model['guard'].items() if not isinstance(v, set)},
        warnings=screen_warnings(world, model))
    result['unverified'] = [r for r in result['rules'] if RULES[r]['status'] != 'original']
    return result


def screen_warnings(world, model):
    out = []
    if world.escort:
        out.append('в отряде стража есть другие бойцы: экран их не учитывает, нужен simulate')
    if model['guard']['role'] != 'area':
        out.append('экран предполагает массовый удар стража; у этого стража другой охват')
    mixed = [m['name'] for m in model['members'] if 0 < m['fast'] < 1]
    if mixed:
        out.append('порядок хода относительно стража случайный: ' + ', '.join(mixed))
    if 'guard_transform' in model['opening']['flags']:
        out.append('предмет превращает стража (несогласованность JAX), экран это не моделирует')
    return out


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value))


def _clean(value):
    """Plain JSON: tuples to lists, infinite or NaN numbers to null."""
    if isinstance(value, dict):
        return {(k if isinstance(k, str) else str(k)): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_clean(v) for v in value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(data), ensure_ascii=False, indent=1, default=_json_default, allow_nan=False)
    path.write_text(text + '\n', encoding='utf-8')


def _levels(text):
    if '-' in str(text):
        low, high = str(text).split('-')
        return list(range(int(low), int(high) + 1))
    return [int(x) for x in str(text).split(',')]


def _load_candidates(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data.get('tactics', data.get('candidates', [])) if isinstance(data, dict) else data


def _roster(text):
    slots = [None if s.strip() in ('', '-', 'none') else s.strip() for s in text.split(',')]
    if len(slots) != 6:
        raise SystemExit('--roster needs six comma-separated slots (- for empty)')
    return slots


def _summary_line(rank, world, cand, result):
    b = result['binomial']
    m = result.get('markov') or {}
    items = '+'.join(k for k in cand.get('battle_items', []) if k) or '—'
    return (f"{rank:>3} P={result['p']:.3f} (бином {b['p']:.3f}, q={b['q']:.3f}, R={b['R']}, k={b['k']}, "
            f"урон/раунд {b.get('damage', 0):.1f}; цепь: поражение {m.get('loss', 0):.3f}, "
            f"таймаут {m.get('timeout', 0):.3f}) | {cand['name']} | предметы {items}")


def cmd_candidates(args):
    world = load_world(args.map, args.guardian)
    pool = unit_pool(world, args.pool, args.units.split(',') if args.units else None)
    cands = generate(world, _levels(args.hero_levels), pool, args.include_partial, args.allow_guard_transform)
    print(f'{len(cands)} кандидатов; пул: ' + ', '.join(world.name(k) for k in pool))
    for cand in cands[:args.show]:
        print(' ', cand['name'], '| зелья', cand['potions'], '| предметы', cand['battle_items'])
    if args.output:
        _write(args.output, dict(map=world.map_name, guardian=world.guardian, candidates=cands))
        print('записано', args.output)


def cmd_screen(args):
    world = load_world(args.map, args.guardian)
    opts = dict(hp_bin=args.hp_bin)
    if args.candidates:
        cands = _load_candidates(args.candidates)
    elif args.roster:
        cands = [make_candidate(world, _roster(args.roster), args.hero_level, wounds=args.wounds)]
    else:
        pool = unit_pool(world, args.pool, args.units.split(',') if args.units else None)
        cands = generate(world, _levels(args.hero_levels), pool, args.include_partial, args.allow_guard_transform)
    cands = [{k: v for k, v in c.items() if k not in ('analytic', 'cost')} for c in cands]
    started = time.time()
    scored = [(cand, evaluate(world, cand, opts, chain=False)) for cand in cands]
    scored.sort(key=lambda item: -item[1]['p'])
    top = [(cand, evaluate(world, cand, opts)) for cand, _ in scored[:args.markov_top]]
    expanded = []
    for cand, _ in sorted(top, key=lambda item: -item[1]['p'])[:args.expand_top]:
        # Potion plans first with the party's items, then every item pair for the best plan.
        plans, pairs = variants(world, cand)
        trials = [dict(cand, potions=plan) for plan in plans]
        trials.sort(key=lambda trial: -evaluate(world, trial, opts, chain=False)['p'])
        best = trials[0] if trials else cand
        guard = guardian_model(world, cand.get('guardian_wounds', 0))
        members, _ = member_models(world, best, guard)
        for pair in pairs:
            rules = opening_rules(world, pair) + default_item_rules(members)
            expanded.append(dict(best, battle_items=pair, tactic=dict(cand.get('tactic') or {}, item_rules=rules)))
    seen = {json.dumps(c, sort_keys=True, default=_json_default) for c, _ in top}
    for trial in expanded:
        signature = json.dumps(trial, sort_keys=True, default=_json_default)
        if signature not in seen:
            seen.add(signature)
            trial = dict(trial, name=describe(world, trial))
            top.append((trial, evaluate(world, trial, opts)))
    top.sort(key=lambda item: -item[1]['p'])
    print(f'Экран: {len(cands)} кандидатов, цепь Маркова для {len(top)}, {time.time() - started:.1f} с')
    for rank, (cand, result) in enumerate(top[:args.show], 1):
        print(_summary_line(rank, world, cand, result))
    tactics = []
    for cand, result in top[:args.export]:
        entry = dict(cand, analytic=result)
        if args.with_cost:
            entry['cost'] = cost(world, cand)
        tactics.append(entry)
    if args.output:
        rules = sorted({r for t in tactics for r in t['analytic']['rules']})
        _write(args.output, dict(map=world.map_name, guardian=world.guardian, tactics=tactics,
                                 rules={r: RULES[r] for r in rules},
                                 verify_in_game={r: RULES[r] for r in rules if RULES[r]['status'] != 'original'}))
        print('записано', args.output, '— для simulate: --tactics', args.output)


def cmd_cost(args):
    world = load_world(args.map, args.guardian)
    if args.tactics:
        cands = _load_candidates(args.tactics)[:args.top]
    else:
        cands = [make_candidate(world, _roster(args.roster), args.hero_level)]
    out = []
    for cand in cands:
        result = cost(world, cand, args.pool, args.chunk)
        out.append(dict(name=cand.get('name') or describe(world, cand), cost=result))
        print(f"{cand.get('name') or describe(world, cand)}: XP {result['xp_needed']} (из боёв "
              f"{result['xp_needed'] - result['trainer_xp']}, тренер {result['trainer_xp']} = {result['trainer_gold']} з.), "
              f"здания {result['building_gold']}, найм {result['recruit_gold']}, лавка {result['shop_gold']}, "
              f"всего {result['total_gold']} золота")
        for decision in result['decisions']:
            print('   ', decision)
    if args.output:
        _write(args.output, out)


def cmd_archive(args):
    world = load_world(args.map, args.guardian)
    opts = dict(hp_bin=args.hp_bin)
    pool = unit_pool(world, args.pool, args.units.split(',') if args.units else None)
    cands = generate(world, _levels(args.hero_levels), pool, args.include_partial, args.allow_guard_transform)
    cells = {}
    for cand in cands:
        result = evaluate(world, cand, opts, chain=False)
        gold = cost(world, cand)['total_gold']
        key = (result['controllers'], int(result['damage_round'] // args.damage_bin), min(result['extra_lives'], 3),
               int(gold // args.gold_bin))
        if key not in cells or result['p'] > cells[key][1]['p']:
            cells[key] = (cand, result, gold)
    elites = sorted(cells.items(), key=lambda item: -item[1][1]['p'])[:args.chain_top]
    rows = []
    for key, (cand, _, gold) in elites:
        result = evaluate(world, cand, opts)
        rows.append(dict(cell=dict(controllers=key[0], damage_bin=key[1] * args.damage_bin, extra_lives=key[2],
                                   gold_bin=key[3] * args.gold_bin), p=result['p'], gold=gold, tactic=dict(cand, analytic=result)))
    rows.sort(key=lambda r: -r['p'])
    print(f'MAP-Elites: {len(cands)} кандидатов, {len(cells)} ячеек; цепь для {len(rows)} элит')
    for row in rows[:args.show]:
        c = row['cell']
        print(f"P={row['p']:.3f} контроль {c['controllers']}, урон {c['damage_bin']}+, жизни {c['extra_lives']}, "
              f"золото {c['gold_bin']}+ ({row['gold']}) | {row['tactic']['name']}")
    if args.output:
        _write(args.output, dict(map=world.map_name, guardian=world.guardian, elites=rows,
                                 tactics=[r['tactic'] for r in rows]))


# ------------------------------------------------------------------------------- capabilities

def cmd_capabilities(args):
    world = load_world(args.map, args.guardian)
    guard = guardian_model(world)
    lines = []
    say = lines.append
    g = guard
    say(f"Страж: {g['name']} ({g['key']}), отряд #{world.guardian} (индекс {world.guardian}), слот {world.guardian_slot}, "
        f"уровень {g['level']}, категория UNIT_CAT={g['category']}")
    say(f"  HP {g['max_hp']}, урон {g['damage']} ({g['source']}, охват {g['role']}), точность {g['accuracy']:.0f} "
        f"(попадание по цели {hit_chance(g['accuracy']):.4f}), инициатива {g['initiative']:.0f}")
    say(f"  броня {g['armor'] - g['fort_armor']:.0f} + укрепление {g['fort_armor']} = {g['armor']:.0f}, "
        f"действует {g['effective_armor']}% (предел 90)")
    say(f"  иммунитеты к источникам: {sorted(g['immunities']) or 'нет'}; защиты: {sorted(g['wards']) or 'нет'}; "
        f"иммунитеты к классам атак (GImmuC): {g['class_immunities'] or 'нет'} — JAX их не читает, "
        'действуют только явные исключения ниже')
    say(f"  ослабленный удар {g['weak_damage']}; восстановление за отдых {g['regen_percent']}%; опыт за убийство {g['exp_kill']}")
    if world.escort:
        say('  в отряде стража есть другие бойцы: ' + ', '.join(world.name(k) for _, k in world.escort))
    say('')
    say('Исключения JAX для стража и особые правила:')
    for rule in ('transform_guard', 'transform_orb_guard', 'shatter_guard', 'spell_guard', 'copy_guard', 'fear_fort',
                 'wounds_persist', 'guard_no_xp', 'disable_round40', 'damage_vs_disabled', 'dot_armor', 'dot_potions'):
        r = RULES[rule]
        say(f"  [{r['status']}] {r['text']} — {r['source']}; {r['where']}")
    say('  Понижение уровня (Сущий, Wight Blade): у стража нет пониженной формы — эффекта нет.')
    say('  Заклинания-усиления своего отряда в бою действуют (casting.begin_battle); урон/ослабление по стражу — нет.')
    say('')
    pool = sorted((k for k in world.forms if not UNITS[k].get('hero')), key=lambda k: (UNITS[k]['level'], k))
    entries = []
    for key in [world.hero_key] + pool:
        hero = key == world.hero_key
        levels = sorted({UNITS[key]['level'], 5, 6, 10}) if hero else [UNITS[key]['level']]
        for level in levels:
            cand = dict(roster=[key if s == world.hero_slot else None for s in range(6)] if hero else
                        [world.hero_key if s == world.hero_slot else None for s in range(6)], hero_level=level if hero else 1,
                        potions=[], inventory={'items': {}, 'potions': {}})
            if not hero:
                roster = formation(world, [key])
                if roster is None:
                    continue
                cand['roster'] = roster
            units, worn = party_profiles(world, cand)
            unit = next(u for u in units if u['key'] == key and u['hero'] == hero)
            m = vs_guardian(world, unit, guard, worn)
            armored = dict(unit, stats=tuple(v + (50 if i == ARMOR else 0) for i, v in enumerate(unit['stats'])))
            m_inv = vs_guardian(world, armored, guard, worn)
            form = world.forms.get(key, {})
            entries.append(dict(key=key, name=world.name(key), level=level, size=unit['size'], hero=hero,
                                hp=unit['stats'][HP], armor=unit['stats'][ARMOR], initiative=unit['stats'][INITIATIVE],
                                before=m['fast'], hit=m['hit'], per_activation=m['expected'], control=m['control'],
                                dot=m['dot'], weaken=m['weaken'], value=m['value'],
                                strikes=strikes_to_die(m)[1], strikes_invulnerable=strikes_to_die(m_inv)[1],
                                weak_strikes=strikes_to_die(m, 'w')[1], notes=m['notes'],
                                buildings=sorted(form.get('buildings', [])), chain=form.get('chain', [key]),
                                building_gold=sum(world.buildings[b]['gold'] for b in form.get('buildings', []))))
    say('Бойцы (уровень формы; удары до гибели: обычные / с Эликсиром неуязвимости / после ослабления):')
    for e in entries:
        parts = [f"{e['name']} ур.{e['level']}{' (герой)' if e['hero'] else ''}, размер {e['size']}, HP {e['hp']:.0f}, "
                 f"броня {e['armor']:.0f}, инициатива {e['initiative']:.0f} (раньше стража {e['before']:.2f})",
                 f"удары {e['strikes']}/{e['strikes_invulnerable']}/{e['weak_strikes']}",
                 f"попадание {e['hit']:.3f}, урон за ход {e['per_activation']:.1f}"]
        if e['control']:
            c = e['control']
            parts.append(f"контроль {c['kind']} p={c['p']:.3f} ({c['how']}{', до 40-го раунда' if c['round40'] else ''})")
        if e['dot']:
            d = e['dot']
            parts.append(f"DoT {d['kind']} {d['amount']} p={d['p']:.3f} {'длительный' if d['long'] else 'короткий'}, мимо брони")
        if e['weaken']:
            parts.append(f"ослабление p={e['weaken']:.3f}")
        if e['notes']:
            parts.append('; '.join(e['notes']))
        if not e['hero']:
            parts.append(f"здания {e['building_gold']} з.")
        say('  ' + ' | '.join(parts))
    say('')
    controllers = sorted((e for e in entries if e['control']), key=lambda e: -e['control']['p'])
    say('Контроль (вероятность за ход): ' + ', '.join(f"{e['name']} {e['control']['p']:.2f}" for e in controllers[:8]))
    dots = sorted((e for e in entries if e['dot']), key=lambda e: -stationary_dot(e['dot']['p'], e['dot']['long']) * e['dot']['amount'])
    say('DoT (средний тик за раунд): ' + ', '.join(
        f"{e['name']} {stationary_dot(e['dot']['p'], e['dot']['long']) * e['dot']['amount']:.1f}" for e in dots[:8]))
    dealers = sorted((e for e in entries if e['value'] > 0), key=lambda e: -e['value'] / max(e['size'], 1))
    say('Урон с DoT на единицу лидерства: ' + ', '.join(
        f"{e['name']} {e['value'] / max(e['size'], 1):.1f}" for e in dealers[:8]))
    tanks = sorted(entries, key=lambda e: -(e['strikes_invulnerable'] if math.isfinite(e['strikes_invulnerable']) else 99) / max(e['size'], 1))
    say('Выживаемость с неуязвимостью (удары на единицу лидерства): ' + ', '.join(
        f"{e['name']} {e['strikes_invulnerable']}" for e in tanks[:8]))
    say('')
    say('Зелья на карте (бесплатно + у торговцев):')
    for key in sorted(world.potions, key=lambda k: [p['key'] for p in POTIONS].index(k)):
        p = POTION_BY_KEY[key]
        useful = ('бесполезно против источника ' + guard['source'] if key in WARD_POTIONS and WARD_POTIONS[key] != guard['source']
                  else '')
        say(f"  {p['name']} ({key}): {world.free_potions.get(key, 0)} + {world.shop_potions.get(key, 0)}, {p['label']}, "
            f"{p['duration']}, {p['price']} з. {useful}")
    say('Снаряжение на карте:')
    for key, count in sorted(world.items.items()):
        item = ITEM_BY_KEY.get(key, {})
        if item.get('category') in (ARTIFACT, BANNER, BOOK, BOOTS):
            slot = (ARTIFACT, ARTIFACT, BANNER, BOOK, BOOTS).index(item['category'])
            say(f"  {item['name']} ×{count}: {item['label']} (с {world.skill_levels[slot]}-го уровня героя)")
    say('Боевые предметы (2 слота героя, одна экипировка в день):')
    for key, item in world.battle_items.items():
        stock = world.potions.get(key, 0) if item['category'] == 'potion' else world.items.get(key, 0)
        access = 'всегда' if item['category'] == 'potion' else 'с 10-го уровня, у лорда-мага или с книгой'
        guard_note = ''
        if item['kind'] in (TRANSFORM, LYCANTHROPY):
            guard_note = ' — в JAX превращает и стража (transform_orb_guard)'
        elif item['kind'] == PARALYSIS:
            guard_note = ' — длительный паралич без броска точности до 40-го раунда'
        say(f"  {item['name']} ({KIND_NAMES[item['kind']]}) ×{stock}: {item.get('label', '')}; доступ {access}{guard_note}")
    spells = json.loads((ROOT / 'stoix/envs/data/spells.json').read_text(encoding='utf-8'))['factions'].get(world.faction, [])
    support = [s for s in spells if (s.get('cast') or {}).get('kind') in ('buff', 'ward', 'health', 'armor')]
    if support:
        say('Заклинания-усиления фракции (до отдыха, нужна мана и изучение):')
        for s in support:
            cast = s['cast']
            say(f"  {s['name']} ур.{s['level']}: {cast.get('kind')} {cast.get('stat', cast.get('element', ''))} "
                f"{cast.get('multiplier', cast.get('amount', ''))}")
    say('')
    say(f"Опыт карты без стража: {world.xp_pool}; золото: {world.gold_sources}")
    say('Неподтверждённые в оригинале правила, на которых держатся тактики, — docs/GUARDIAN_LAB.md и вывод screen.')
    print('\n'.join(lines))
    if args.output:
        _write(args.output, dict(guardian={k: v for k, v in guard.items()}, units=entries,
                                 rules=RULES, xp_pool=world.xp_pool, gold_sources=world.gold_sources))


# ------------------------------------------------------------------------------- GPU simulation

def cmd_simulate(args):
    import jax
    import jax.numpy as jnp
    from stoix.envs.number_grid import CONTINUE, DEFEND, RETREAT, SHOOT, WAIT, NumberGrid, action_names, load_map

    if jax.default_backend() != 'gpu':
        raise SystemExit('simulate runs on CUDA only: bash scripts/run_gpu.sh scripts/guardian_lab.py simulate ...')
    if args.cache:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
        jax.config.update('jax_compilation_cache_dir', str(Path(args.cache).resolve()))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)
    world = load_world(args.map, args.guardian)
    if args.tactics:
        tactics = _load_candidates(args.tactics)[:args.top]
    else:
        tactics = [make_candidate(world, _roster(args.roster), args.hero_level, wounds=args.wounds)]
    if any(t.get('map', world.map_name) != world.map_name or int(t.get('guardian', world.guardian)) != world.guardian
           for t in tactics):
        raise SystemExit('All tactics must target the same map and guardian as --map/--guardian')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'guardian_lab.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    _write(out / 'tactics.json', tactics)
    timing = {}
    started = time.time()
    env = NumberGrid(map_config=load_map(world.map_name))
    config = env.map_config
    names = action_names(config)
    A = {name: i for i, name in enumerate(names)}
    step = jax.jit(env.step)
    mask_fn = jax.jit(env.action_mask)
    stats_fn = jax.jit(env.unit_stats)
    max_hp_fn = jax.jit(env.max_hp)
    refresh = jax.jit(env._refresh_equipment)
    reset = jax.jit(env.reset)
    timing['environment_s'] = round(time.time() - started, 1)
    G = 6 + world.guardian_slot
    target_action = SHOOT + world.guardian_slot
    p = env.progression
    rules_max, copies_max = 24, 6

    def build(tactic):
        roster = tactic['roster']
        warnings = []
        missing = [k for k in roster if k and k not in p.ids]
        if missing:
            raise SystemExit('not in the progression catalogue: ' + ', '.join(missing))
        ids = [p.ids[k] if k else 0 for k in roster]
        if roster[world.hero_slot] == world.hero_key:
            # The starting hero has its own row with the lord's skills and Might (hero_combat_stats).
            ids[world.hero_slot] = int(p.initial_ids[world.hero_slot])
        for slot, key in enumerate(roster):
            if not key:
                continue
            if ids[slot] not in p.used_ids:
                raise SystemExit(f'{key}: abilities are pruned from this map (not in used_ids); simulate would lie')
            if UNITS[key].get('size', 1) == 2 and (slot >= 3 or roster[slot + 3] is not None):
                raise SystemExit(f'{key}: a large unit needs a front slot and an empty rear slot')
        state, _ = reset(jax.random.PRNGKey(args.seed))
        levels = [int(p.base_levels[i]) for i in ids]
        levels[world.hero_slot] = int(tactic['hero_level'])
        for slot, level in (tactic.get('levels') or {}).items():
            levels[int(slot)] = int(level)
            if not world.forms.get(roster[int(slot)], {}).get('final', True):
                warnings.append(f'слот {slot}: уровень выше базового у формы с дальнейшим улучшением')
        items = cand_items(world, tactic)
        copies = [env.item_rules.keys.index(k) for k, n in items.items() if k in env.item_rules.keys for _ in range(n)]
        if len(copies) > env.item_rules.capacity:
            raise SystemExit('inventory exceeds the scenario item capacity')
        potions = cand_potions(world, tactic)
        inventory = jnp.full_like(state.item_inventory, -1).at[:len(copies)].set(jnp.array(copies, jnp.int32))
        state = state.replace(unit_ids=state.unit_ids.at[:6].set(jnp.array(ids, jnp.int32)),
                              unit_levels=state.unit_levels.at[:6].set(jnp.array(levels, jnp.int32)),
                              unit_xp=state.unit_xp.at[:6].set(0), item_inventory=inventory,
                              potions=jnp.array([potions.get(k, 0) for k in env.potion_rules.keys], jnp.int32))
        state = refresh(state)
        state = state.replace(hp=state.hp.at[:6].set(max_hp_fn(state)[:6]))
        log = []

        def act(state, name):
            if name not in A:
                raise SystemExit(f'{name}: no such action on this map')
            if not bool(mask_fn(state)[A[name]]):
                raise SystemExit(f'{name} is not allowed by the action mask (setup {log})')
            log.append(name)
            return step(state, jnp.int32(A[name]))[0]

        equips = [k for k in tactic.get('battle_items', []) if k]
        for i, key in enumerate(equips):
            if i:
                state = act(state, 'rest')  # one battle-item equip per day
            state = act(state, 'equip_' + key)
        for key, slot in tactic.get('potions', []):
            index = env.potion_rules.keys.index(key)
            before = int(state.potions[index])
            state = act(state, f'potion_{key}_{slot}')
            if int(state.potions[index]) != before - 1:
                raise SystemExit(f'potion {key} on slot {slot} was not consumed')
        if tactic.get('guardian_wounds'):
            state = state.replace(enemy_wounds=state.enemy_wounds.at[world.guardian, world.guardian_slot].set(
                int(tactic['guardian_wounds'])))
        composition = np.asarray(env.recruitment.composition(state)).tolist() if env.recruitment_enabled else None
        if composition and composition[1] > composition[0]:
            warnings.append(f'лидерство превышено: занято {composition[1]} из {composition[0]}')
        target = state.enemy_positions[world.guardian]
        blocked = env.territory.blocked_grid.reshape(env.size, env.size)
        for direction, offset in enumerate(env.directions.tolist()):
            origin = target - jnp.array(offset, jnp.int32)
            if bool(blocked[origin[0], origin[1]]):
                continue
            candidate = state.replace(position=origin)
            if bool(mask_fn(candidate)[direction]):
                return candidate, direction, log, warnings, composition
        raise SystemExit('No legal approach to the guardian')

    def tables(tactic, model):
        hero = world.hero_slot
        actions, slots, limits, wounded = [], [], [], []
        for rule in (tactic.get('tactic') or {}).get('item_rules', []):
            which = [rule['slot_index']] if rule.get('slot_index') else [1, 2]
            if rule.get('side', 'ally') == 'enemy':
                target = world.guardian_slot if rule.get('target', 'guardian') == 'guardian' else int(rule['target'])
                for k in which:
                    actions.append(A[f'item{k}_enemy_{target}'])
                    slots.append(6 + target)
                    limits.append(2 ** 30)
                    wounded.append(False)
            else:
                # A named ally gets Life when fallen, a heal when wounded and at or below max_hp;
                # 'any' (summons, Orb of Rage) takes the first slot the mask allows.
                named = rule.get('target') != 'any'
                for t in ([int(rule['target'])] if named else range(6)):
                    for k in which:
                        actions.append(A[f'item{k}_ally_{t}'])
                        slots.append(t)
                        limits.append(int(rule.get('max_hp', 2 ** 30)) if named else 2 ** 30)
                        wounded.append(named)
        if len(actions) > rules_max:
            raise SystemExit(f'too many item rules ({len(actions)} actions, at most {rules_max})')
        enabled = [True] * len(actions) + [False] * (rules_max - len(actions))
        pad = rules_max - len(actions)
        copy = (tactic.get('tactic') or {}).get('copy')
        copy_actions = [A[f"copy_ally_{int(copy['ally'])}"]] if copy and f"copy_ally_{int(copy['ally'])}" in A else []
        copy_actors = [int(copy['slot'])] if copy_actions else []
        defend = [False] * 6
        for slot in (tactic.get('tactic') or {}).get('defend_slots', []):
            defend[int(slot)] = True
        controllers = [False] * 6
        for m in model['members']:
            if m['control'] and m['slot'] is not None:
                controllers[m['slot']] = True
        return dict(
            hero=jnp.int32(hero), item_actions=jnp.array(actions + [0] * pad, jnp.int32),
            item_slots=jnp.array(slots + [0] * pad, jnp.int32), item_limits=jnp.array(limits + [0] * pad, jnp.int32),
            item_wounded=jnp.array(wounded + [False] * pad), item_enabled=jnp.array(enabled), copy_actions=jnp.array(copy_actions + [0] * (copies_max - len(copy_actions)), jnp.int32),
            copy_actors=jnp.array(copy_actors + [-1] * (copies_max - len(copy_actors)), jnp.int32),
            copy_enabled=jnp.array([True] * len(copy_actions) + [False] * (copies_max - len(copy_actions))),
            defend=jnp.array(defend), controllers=jnp.array(controllers),
            retreat_below=jnp.int32(int((tactic.get('tactic') or {}).get('retreat_below_controllers') or 0)))

    healers = env.basic_combat and env.has_healers

    def policy(state, t):
        mask = env.action_mask(state)
        actor = state.actor
        fighting = state.post_victory == 0
        own = jnp.clip(actor, 0, 5)
        hp = state.hp[t['item_slots']]
        hurt = ~t['item_wounded'] | (hp < env.max_hp(state)[t['item_slots']])
        ready = (mask[t['item_actions']] & t['item_enabled'] & (actor == t['hero']) & fighting
                 & (hp <= t['item_limits']) & hurt)
        copying = mask[t['copy_actions']] & t['copy_enabled'] & (actor == t['copy_actors'])
        alive = jnp.sum(t['controllers'] & (state.hp[:6] > 0) & ~state.escaped[:6])
        retreat = (t['retreat_below'] > 0) & (alive < t['retreat_below']) & (actor < 6) & fighting
        attack = jnp.array([target_action, DEFEND, WAIT, RETREAT, CONTINUE], jnp.int32)
        guard_first = jnp.array([DEFEND, target_action, WAIT, RETREAT, CONTINUE], jnp.int32)
        leave = jnp.array([RETREAT, DEFEND, WAIT, CONTINUE, target_action], jnp.int32)
        order = jnp.where(retreat, leave, jnp.where(t['defend'][own] & (actor < 6), guard_first, attack))
        allowed = mask[order] & (fighting | (order != target_action))
        action = jnp.where(jnp.any(allowed), order[jnp.argmax(allowed)], jnp.argmax(mask))
        if healers:
            # A healer's SHOOT actions target allies: heal the most wounded one, also after victory.
            maximum = env.max_hp(state)[:6]
            wounded = mask[SHOOT:SHOOT + 6] & (state.hp[:6] < maximum)
            need = jnp.where(wounded, maximum - state.hp[:6], -1)
            action = jnp.where(env._actor_is_healer(state) & jnp.any(wounded), SHOOT + jnp.argmax(need), action)
        action = jnp.where(jnp.any(ready), t['item_actions'][jnp.argmax(ready)], action)
        action = jnp.where(jnp.any(copying), t['copy_actions'][jnp.argmax(copying)], action)
        return action.astype(jnp.int32), jnp.any(ready) & ~jnp.any(copying)

    limit = args.decision_limit

    def simulate_one(entry, direction, t, key):
        state, _ = env.step(entry.replace(battle_key=key), direction)
        opening = (jnp.sum(state.hp[:6]) < jnp.sum(entry.hp[:6])).astype(jnp.int32)

        def cond(c):
            return c['s'].in_battle & ~c['s'].done & (c['n'] < limit)

        def body(c):
            s = c['s']
            action, using = policy(s, t)
            after, _ = env.step(s, action)
            hurt = jnp.sum(after.hp[:6]) < jnp.sum(s.hp[:6])
            g_hp = jnp.where(after.in_battle, after.hp[G], c['g_min'])
            closing = (c['round'] <= 1) & ((after.round >= 2) | ~after.in_battle)
            return dict(s=after, n=c['n'] + 1, g_min=jnp.minimum(c['g_min'], g_hp),
                        round=jnp.maximum(c['round'], after.round), strikes=c['strikes'] + hurt,
                        items=c['items'] + using,
                        weak=jnp.where((c['weak'] < 0) & after.weakened[G], after.round, c['weak']),
                        open_hp=jnp.where(closing, after.hp[:6], c['open_hp']))

        init = dict(s=state, n=jnp.int32(0), g_min=state.hp[G], round=state.round, strikes=opening,
                    items=jnp.int32(0), weak=jnp.int32(-1), open_hp=state.hp[:6])
        c = jax.lax.while_loop(cond, body, init)
        s = c['s']
        victory = ~s.alive[world.guardian]
        return dict(won=victory, lost=s.lost, stuck=s.in_battle, withdrew=~s.in_battle & ~victory & ~s.lost,
                    decisions=c['n'], guardian_hp=jnp.where(victory, 0, c['g_min']), rounds=c['round'],
                    strikes=c['strikes'], items=c['items'], weak_round=c['weak'],
                    survivors=jnp.sum(s.hp[:6] > 0), open_hp=c['open_hp'])

    run = jax.jit(jax.vmap(simulate_one, in_axes=(None, None, None, 0)))
    policy_jit = jax.jit(policy)
    report = dict(map=world.map_name, guardian=world.guardian, guardian_slot=world.guardian_slot,
                  sims_per_tactic=args.sims, decision_limit=limit, seed=args.seed, jax=jax.__version__,
                  device=str(jax.devices()[0]), tactics=[])
    first_call = None
    for index, tactic in enumerate(tactics):
        t0 = time.time()
        entry, direction, log, warnings, composition = build(tactic)
        model = build_model(world, tactic)
        t = tables(tactic, model)
        setup_s = time.time() - t0
        peek, _ = step(entry.replace(battle_key=jax.random.PRNGKey(args.seed + 7)), jnp.int32(direction))
        values = np.asarray(stats_fn(peek))
        diffs = []
        for m in model['members']:
            if m['slot'] is None:
                continue
            for i, stat in enumerate(STATS):
                if abs(values[m['slot'], i] - m['stats'][i]) > .5:
                    diffs.append(dict(slot=m['slot'], unit=m['name'], stat=stat, env=float(values[m['slot'], i]),
                                      static=float(m['stats'][i])))
        guard = model['guard']
        for i, (stat, static) in enumerate(zip(STATS, (guard['max_hp'], guard['damage'], guard['accuracy'],
                                                       guard['armor'], guard['initiative']))):
            if abs(values[G, i] - static) > .5:
                diffs.append(dict(slot=G, unit=guard['name'], stat=stat, env=float(values[G, i]), static=float(static)))
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), index), args.sims)
        t1 = time.time()
        r = jax.tree.map(np.asarray, run(entry, jnp.int32(direction), t, keys))
        elapsed = time.time() - t1
        if first_call is None:
            first_call = elapsed
        won, lost = r['won'], r['lost']
        weak = r['weak_round'] >= 0
        summary = dict(
            name=tactic.get('name') or describe(world, tactic), seconds=round(elapsed, 2), setup_seconds=round(setup_s, 2),
            decisions=int(r['decisions'].sum()), win_rate=float(won.mean()), defeat_rate=float(lost.mean()),
            withdraw_rate=float(r['withdrew'].mean()), undecided=int(r['stuck'].sum()),
            rounds_when_won=float(r['rounds'][won].mean()) if won.any() else None,
            rounds_when_lost=float(r['rounds'][lost].mean()) if lost.any() else None,
            guardian_hp_quantiles={q: float(np.quantile(r['guardian_hp'], q / 100)) for q in (10, 50, 90)},
            strikes_mean=float(r['strikes'].mean()),
            strikes_hist={int(k): int(v) for k, v in zip(*np.unique(np.minimum(r['strikes'], 6), return_counts=True))},
            item_uses_mean=float(r['items'].mean()), weakened_rate=float(weak.mean()),
            survivors_when_won=float(r['survivors'][won].mean()) if won.any() else None,
            hp_after_round_1=r['open_hp'].mean(axis=0).round(1).tolist(), setup=log, warnings=warnings,
            leadership=composition, static_vs_env=diffs,
            analytic=dict(binomial=model['binomial'], markov=markov(model)),
            rules=tactic_rules(model, tactic))
        report['tactics'].append(summary)
        print(index, summary['name'], json.dumps({k: summary[k] for k in (
            'win_rate', 'defeat_rate', 'undecided', 'rounds_when_won', 'strikes_mean', 'seconds')}, ensure_ascii=False),
            'аналитика', round(summary['analytic']['markov']['p'], 3), 'расхождений', len(diffs), flush=True)
    order = sorted(range(len(report['tactics'])), key=lambda i: -report['tactics'][i]['win_rate'])
    report['ranking'] = [dict(index=i, name=report['tactics'][i]['name'], win_rate=report['tactics'][i]['win_rate'])
                         for i in order]
    traces = []
    for i in order[:args.trace]:
        entry, direction, _, _, _ = build(tactics[i])
        t = tables(tactics[i], build_model(world, tactics[i]))
        state, _ = step(entry.replace(battle_key=jax.random.PRNGKey(args.seed + 5)), jnp.int32(direction))
        steps = []
        for _ in range(limit):
            if not bool(state.in_battle) or bool(state.done):
                break
            action, _ = policy_jit(state, t)
            actor, rnd = int(state.actor), int(state.round)
            state, _ = step(state, action)
            steps.append(dict(round=rnd, actor=actor, action=names[int(action)], guardian_hp=int(state.hp[G]),
                              party_hp=np.asarray(state.hp[:6]).tolist(), event=int(state.last_event),
                              paralyzed=bool(state.paralyzed[G] | state.long_paralyzed[G]),
                              weakened=bool(state.weakened[G]), burn=int(state.burn_mode[G]),
                              poison=int(state.poison_mode[G])))
        traces.append(dict(index=i, key=args.seed + 5, won=bool(~state.alive[world.guardian]), lost=bool(state.lost),
                           steps=steps))
    report['traces'] = traces
    timing.update(first_batch_s=round(first_call or 0., 1), wall_s=round(time.time() - started, 1))
    report['timing'] = timing
    _write(out / 'report.json', report)
    print('ранжирование:', json.dumps(report['ranking'][:10], ensure_ascii=False))
    print('timing', json.dumps(timing), '→', out / 'report.json')


# ------------------------------------------------------------------------------- command line

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--map', default='default-capital')
    common.add_argument('--guardian', type=int, default=None, help='enemy index; default: objective or capital guard')
    sub = parser.add_subparsers(dest='command', required=True)

    def command(name, **kwargs):
        return sub.add_parser(name, parents=[common], **kwargs)

    def search_options(p):
        p.add_argument('--hero-levels', default='4-8', help='e.g. 4-8 or 5,6')
        p.add_argument('--pool', choices=('final', 'all'), default='final', help='final forms only or every reachable form')
        p.add_argument('--units', help='explicit comma-separated unit keys instead of the pool')
        p.add_argument('--include-partial', action='store_true', help='also parties that leave 2+ leadership free')
        p.add_argument('--allow-guard-transform', action='store_true',
                       help='keep parties whose battle items transform the guardian (JAX inconsistency)')

    p = command('capabilities', help='guardian profile versus the reachable catalogue')
    p.add_argument('--output')
    p.set_defaults(func=cmd_capabilities)
    p = command('candidates', help='enumerate parties with default plans')
    search_options(p)
    p.add_argument('--output')
    p.add_argument('--show', type=int, default=10)
    p.set_defaults(func=cmd_candidates)
    p = command('screen', help='analytic estimate; exports tactics for simulate')
    search_options(p)
    p.add_argument('--candidates', help='candidates/tactics JSON instead of generating')
    p.add_argument('--roster', help='six slots, e.g. infernal_knight,duke,infernal_knight,-,incubus,incubus')
    p.add_argument('--hero-level', type=int, default=6)
    p.add_argument('--wounds', type=int, default=0, help='guardian HP already lost (same-day second battle)')
    p.add_argument('--markov-top', type=int, default=200)
    p.add_argument('--expand-top', type=int, default=10, help='potion/item variants for the best parties')
    p.add_argument('--hp-bin', type=int, default=5)
    p.add_argument('--show', type=int, default=20)
    p.add_argument('--export', type=int, default=20)
    p.add_argument('--with-cost', action='store_true')
    p.add_argument('--output')
    p.set_defaults(func=cmd_screen)
    p = command('cost', help='XP, trainer, buildings, hires and purchases')
    p.add_argument('--tactics')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--roster')
    p.add_argument('--hero-level', type=int, default=6)
    p.add_argument('--pool', type=int, default=None, help='battle XP available; default: all map armies')
    p.add_argument('--chunk', type=int, default=5)
    p.add_argument('--output')
    p.set_defaults(func=cmd_cost)
    p = command('archive', help='MAP-Elites style elites')
    search_options(p)
    p.add_argument('--damage-bin', type=float, default=10.)
    p.add_argument('--gold-bin', type=int, default=5000)
    p.add_argument('--chain-top', type=int, default=60)
    p.add_argument('--hp-bin', type=int, default=5)
    p.add_argument('--show', type=int, default=30)
    p.add_argument('--output')
    p.set_defaults(func=cmd_archive)
    p = command('simulate', help='GPU Monte Carlo in the shared JAX environment (CUDA)')
    p.add_argument('--tactics', help='screen output or a JSON list of tactics')
    p.add_argument('--roster')
    p.add_argument('--hero-level', type=int, default=6)
    p.add_argument('--wounds', type=int, default=0)
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--sims', type=int, default=2048)
    p.add_argument('--decision-limit', type=int, default=1600)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--trace', type=int, default=1, help='record one readable battle for the best N tactics')
    p.add_argument('--cache', default=str(ROOT / 'results/guardian-lab/compilation-cache'))
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_simulate)
    args = parser.parse_args()
    if args.command in ('simulate', 'cost') and not args.tactics and not args.roster:
        parser.error(f'{args.command} needs --tactics or --roster')
    args.func(args)


if __name__ == '__main__':
    main()
