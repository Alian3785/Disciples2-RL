"""Battle XP and building-backed promotions; fixed-shape JAX state, no host steps.

Rules and source pointers are recorded in docs/UNIT_PROGRESSION.md.
"""
from functools import lru_cache
import json

import jax
import jax.numpy as jnp
from stoix.envs.number_grid_limits import MAX_LEVEL, MAX_HP, MAX_INITIATIVE, MAX_ACCURACY, MAX_ARMOR

from stoix.envs.number_grid_combat import (
    UNITS, ROLES, HP, DAMAGE, ARMOR, STAT_NAMES,
    _validated_stats, _source, _protection_mask, uses_aoe_accuracy_falloff, build_damage_rolls,
    capped_damage_bonus, support_reaches_all, healer_ward_bits,
)
from stoix.envs.number_grid_buildings import CATALOG
from stoix.envs.number_grid_effects import CAPITAL_GUARDS
from stoix.envs.number_grid_casting import cast_summon_keys
from stoix.envs.number_grid_battle_items import battle_item_unit_keys
from stoix.envs.number_grid_recruitment import OFFERS, mercenary_offers
from stoix.envs import number_grid_hero_skills as hero_skills

# User-requested current-profile shaping (2026-10-16), not original game rules.
LEVEL_UP_REWARD = 2.
BLOCKED_XP_PENALTY = .05
BLOCKED_XP_FEATURES = 6


@lru_cache(maxsize=8)
def _damage_rolls(armors):
    """Immutable device lookup shared by scenarios with the same armor values."""
    return build_damage_rolls(range(401), armors)


class ProgressionRules:
    def __init__(self, game_map, construction):
        """Build unit statistics, starting experience, and upgrade requirements for the scenario."""
        if 'hero_roster' not in game_map or 'enemy_rosters' not in game_map:
            raise ValueError('Unit progression requires named rosters')
        self.ids = {}
        rows = [dict(name='Пусто', level=0, exp_kill=0, exp_required=0,
                     exp_current=0, upgrades=[], faction='', role='ranged',
                     max_hp=0, damage=0, accuracy=0, armor=0, initiative=0,
                     attack_type='weapon', immunities=[], protections=[],
                     growth=dict(threshold=10, early=[0]*5, late=[0]*5,
                                 kill_early=0, kill_late=0, required_early=0, required_late=0))]
        for key, values in UNITS.items():
            if not values.get('upgrade_unavailable_reason'):
                self.ids[key] = len(rows)
                rows.append(dict(values, key=key))
        cache = {}

        def roster_ids(roster, overrides):
            result, occupied = [], 0
            for key in roster:
                if key is None:
                    result.append(0)
                    continue
                override = overrides[occupied]
                occupied += 1
                if not override:
                    result.append(self.ids[key])
                    continue
                signature = json.dumps([key, override], sort_keys=True)
                if signature not in cache:
                    cache[signature] = len(rows)
                    rows.append({**UNITS[key], **override, 'key': key})
                result.append(cache[signature])
            return result

        hero_overrides = game_map.get('hero_combat_stats', [{} for _ in range(game_map['hero_units'])])
        if hero_skills.enabled(game_map):
            # The player's heroes get their own rows: Might follows the lord, not the unit.
            might, skills = hero_skills.might_bonus(game_map), hero_skills.skill_list(game_map)

            def with_skills(key, override):
                row = {**UNITS[key], **override}
                bonuses = row.get('level_bonuses', [])
                if not UNITS[key].get('hero') and row.get('hero'):
                    # Scenario-promoted heroes (Betrezen) take the same stat ladder.
                    bonuses = next(r['level_bonuses'] for r in UNITS.values()
                                   if r.get('game_id') == 'g000uu0044')
                result = dict(override, hero_skills=skills, level_bonuses=sorted(
                    bonuses+[might], key=lambda b: b['level']))
                if row['level'] >= might['level']:
                    # Growth starts above the row's level, so a hero starting past Might gets it now.
                    result['damage'] = round(row['damage']*might['multiplier'])
                return result

            keys = [key for key in game_map['hero_roster'] if key is not None]
            hero_overrides = [with_skills(key, o) if o.get('hero', UNITS[key].get('hero')) else o
                              for key, o in zip(keys, hero_overrides)]
        hero = roster_ids(game_map['hero_roster'], hero_overrides)
        enemy_overrides = game_map.get('enemy_combat_stats',
                                      [[{} for _ in range(n)] for n in game_map['enemy_units']])
        enemies = [roster_ids(roster, overrides) for roster, overrides in
                   zip(game_map['enemy_rosters'], enemy_overrides)]
        levels = [r['level'] for r in rows]
        if any(type(level) is not int or not 1 <= level <= MAX_LEVEL for level in levels[1:]):
            raise ValueError('Unit and hero base levels must be between 1 and 99')
        self.rows = rows
        self.max_hero_level = game_map.get('max_hero_level')
        if self.max_hero_level is not None and (type(self.max_hero_level) is not int or self.max_hero_level < 1):
            raise ValueError('max_hero_level must be a positive integer')
        self.max_hero_level = min(self.max_hero_level or MAX_LEVEL, MAX_LEVEL)
        self.heroes = jnp.array([bool(r.get('hero')) for r in rows])
        self.doppelgangers = jnp.array([r.get('unit_type') == 'Doppelganger' for r in rows])
        self.copy_forbidden = jnp.array([r.get('game_data', {}).get('UNIT_CAT') == '8'
            or r.get('name') in ('Драллиаан','Лаклаан','Drulliaan','Laclaan') for r in rows])
        self.imp_copy_id = next(i for i,r in enumerate(rows) if r.get('game_id') == 'g000uu5034')
        self.fenrir_copy_id = self.ids.get('g000uu0190', next(i for i,r in enumerate(rows) if r.get('game_id') == 'g000uu0190'))
        summon_types = {'Summoner':1, 'Occultmaster':2, 'Lyf':2, 'Laclaan':2}
        self.summon_modes = jnp.array([summon_types.get(r.get('unit_type'),0) for r in rows],jnp.int32)
        self.immediate_dragons = jnp.array([r.get('unit_type') == 'Laclaan' for r in rows])
        self.summon_small = jnp.array([( [self.ids[k] for k in r.get('summon_pool',[])
            if UNITS[k].get('size',1) == 1]+[0]*8)[:8] for r in rows],jnp.int32)
        self.summon_large = jnp.array([( [self.ids[k] for k in r.get('summon_pool',[])
            if UNITS[k].get('size',1) == 2]+[0]*8)[:8] for r in rows],jnp.int32)
        self.hermits = jnp.array([r.get('unit_type') == 'Hermit' for r in rows])
        self.alchemists = jnp.array([r.get('unit_type') == 'Alchemist' for r in rows])
        self.patriarchs = jnp.array([r.get('unit_type') == 'Patriach' for r in rows])
        power_types = {'Travnitsa':1.25,'Novice':1.5,'Dwarfdruid':1.75,'Arhidruid':2.}
        self.power_factors = jnp.array([power_types.get(r.get('unit_type'),0.) for r in rows])
        self.power_cures = jnp.array([r.get('unit_type') in ('Dwarfdruid','Arhidruid') for r in rows])
        # Installed GAttacks: ward elements and L_ALL reach come from each unit's own
        # primary attack (Sundancer/Sylfid all allies, Deva roshi one target).
        ward_bits = [healer_ward_bits(r) for r in rows]
        self.healer_ward_elements = jnp.array(ward_bits,jnp.uint32)
        self.mass_healers = jnp.array([support_reaches_all(r) for r in rows])
        self.mass_cures = jnp.array([r.get('unit_type') == 'Profit' and r['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess') for r in rows])
        post_names = {'Служка','Жрец','Священник','Архангел','Медиум','Оракул','Дева рощи','Патриарх','Клирик','Аббатиса','Прорицательница','Нейтральный эльфийский оракул','Солнечная танцовщица','Сильфида'}
        self.post_healers = jnp.array([r['name'] in post_names and r['role'] == 'healer' for r in rows])
        self.wights = jnp.array([r.get('unit_type') == 'Wight' for r in rows])
        self.has_wights = any(r.get('unit_type') == 'Wight' for r in rows)
        self.lower_forms = jnp.array([self.ids.get(r.get('lower_form', ''), 0) for r in rows], jnp.int32)
        self.neutrals = jnp.array([r.get('neutral', r['faction'] == 'neutral') for r in rows])
        self.aoe_accuracy_falloff = jnp.array([uses_aoe_accuracy_falloff(r) for r in rows])
        self.has_aoe_accuracy_falloff = any(uses_aoe_accuracy_falloff(r) for r in rows)
        self.wolf_lord = jnp.array([r.get('unit_type') == 'Wolf Lord' for r in rows])
        self.has_wolf_lord = any(r.get('unit_type') == 'Wolf Lord' for r in rows)
        self.shatterers = jnp.array([r.get('unit_type') in ('Teurg','Aleman') for r in rows])
        self.has_shatterers = any(r.get('unit_type') in ('Teurg','Aleman') for r in rows)
        self.witches = jnp.array([r.get('unit_type') in ('Witch','Succub') for r in rows])
        self.has_witches = any(r.get('unit_type') in ('Witch','Succub') for r in rows)
        self.centaurs = jnp.array([r.get('unit_type') == 'Centaur Savage' for r in rows])
        self.has_centaurs = any(r.get('unit_type') == 'Centaur Savage' for r in rows)
        self.double_strike = jnp.array([r.get('unit_type') in ('Demon', 'Elfarcher') for r in rows])
        self.has_double_strike = any(r.get('unit_type') in ('Demon', 'Elfarcher') for r in rows)
        self.cached_poisoners = jnp.array([(r['name'] == 'Ниддог' or r.get('unit_type') in ('Dregazul','Spider')) for r in rows])
        self.poisoners = jnp.array([(r['name'] == 'Ниддог' or r.get('unit_type') in ('Dregazul','Spider')) or r.get('unit_type') in ('Death','Dead dragon') for r in rows])
        used_ids = set(hero) | {i for squad in enemies for i in squad}
        # Feature pruning must include every recruit and its later promotions.
        # This host-only closure runs at construction, never in the JAX step.
        reached = set(hero)
        self.bot_enabled = bool(game_map.get('scripted_bot'))
        if self.bot_enabled:
            reached.update(enemies[game_map['scripted_bot']['enemy']])
        if game_map.get('faction_recruitment'):
            recruits = OFFERS[construction.faction] + tuple(
                row['unit'] for _, row, _ in mercenary_offers(game_map))
            reached.update(self.ids[key] for key in recruits)
        pending = list(reached)
        while pending:
            row = rows[pending.pop()]
            for target in row['upgrades']:
                target_id = self.ids.get(target)
                if target_id is not None and target_id not in reached:
                    reached.add(target_id)
                    pending.append(target_id)
        used_ids |= reached
        # Summon spells bring their unit into a battle of its own.
        used_ids |= {self.ids[key] for key in cast_summon_keys(game_map)}
        # Hero orbs and talismans summon or polymorph into catalogue units.
        used_ids |= {self.ids[key] for key in battle_item_unit_keys(game_map)}
        # Enemy forms never promote, but their summons can introduce additional
        # combat behavior. Follow those pools without enabling enemy upgrades.
        pending = list(used_ids)
        while pending:
            for target in rows[pending.pop()].get('summon_pool',[]):
                target_id = self.ids[target]
                if target_id not in used_ids:
                    used_ids.add(target_id)
                    pending.append(target_id)
        self.used_ids = used_ids
        self.has_healers = any(rows[i]['role'] == 'healer' for i in used_ids)
        self.has_copies = any(rows[i].get('unit_type') == 'Doppelganger' for i in used_ids)
        self.has_summons = any(rows[i].get('unit_type') in summon_types for i in used_ids)
        self.has_hermits = any(rows[i].get('unit_type') == 'Hermit' for i in used_ids)
        self.has_alchemists = any(rows[i].get('unit_type') == 'Alchemist' for i in used_ids)
        self.has_patriarchs = any(rows[i].get('unit_type') == 'Patriach' for i in used_ids)
        self.has_powerups = any(rows[i].get('unit_type') in power_types for i in used_ids)
        self.has_healer_wards = any(ward_bits[i] for i in used_ids)
        self.has_cures = any(rows[i].get('unit_type') in ('Patriach','Dwarfdruid','Arhidruid') or (rows[i].get('unit_type') == 'Profit' and rows[i]['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess')) for i in used_ids)
        self.has_poisoners = any((rows[i]['name'] == 'Ниддог' or rows[i].get('unit_type') in ('Dregazul','Spider')) or rows[i].get('unit_type') in ('Death','Dead dragon') for i in used_ids)
        self.cached_water = jnp.array([r.get('unit_type') == 'Ismir son' for r in rows])
        self.water_casters = jnp.array([r.get('unit_type') in ('Sentry','Ismir son','Drulliaan') for r in rows])
        self.has_water = any(rows[i].get('unit_type') in ('Sentry','Ismir son','Drulliaan') for i in used_ids)
        self.secondary_fear = jnp.array([r.get('unit_type') == 'Shamanka' for r in rows])
        self.has_secondary_fear = any(rows[i].get('unit_type') == 'Shamanka' for i in used_ids)
        self.fear_casters = jnp.array([r.get('unit_type') == 'Baroness' for r in rows])
        self.has_fear = any(rows[i].get('unit_type') in ('Baroness','Shamanka') for i in used_ids)
        self.weakeners = jnp.array([r.get('unit_type') == 'Tiamat' for r in rows])
        self.has_weakening = any(rows[i].get('unit_type') == 'Tiamat' for i in used_ids)
        self.secondary_paralysis_modes = jnp.array([1 if (r.get('game_id') in ('g000uu5026','g000uu5126') or r.get('unit_type') == 'Abyss Devil') else 2 if r.get('unit_type') in ('Betrezen','Uter','Uter Demon','Abyss Devil') else 0 for r in rows],jnp.int32)
        self.has_secondary_paralysis = any(rows[i].get('unit_type') in ('Betrezen','Uter','Uter Demon','Abyss Devil') for i in used_ids)
        self.ghost_modes = jnp.array([2 if r.get('game_id') in ('g000uu8044','g000uu8144') else 1 if r.get('unit_type') in ('Ghost','Shadow','Incub') else 0 for r in rows],jnp.int32)
        self.has_paralysis = self.has_fear or self.has_secondary_paralysis or any(rows[i].get('unit_type') in ('Ghost','Shadow','Incub') for i in used_ids)
        self.leech_modes = jnp.array([2 if r.get('unit_type') in ('Bone Lord','Highvampire') else 1 if r.get('unit_type') in ('Dregazul','Vampire') else 0 for r in rows],jnp.int32)
        self.has_leech = any(rows[i].get('unit_type') in ('Bone Lord','Dregazul','Vampire','Highvampire') for i in used_ids)
        self.cached_fire = jnp.array([r.get('unit_type') == 'Lord' for r in rows])
        self.fire_casters = jnp.array([r.get('unit_type') in ('Watcher','Lord','Gumtic') for r in rows])
        self.has_fire = any(rows[i].get('unit_type') in ('Watcher','Lord','Gumtic') for i in used_ids)
        self.secondary_base = jnp.array([[r.get('secondary_damage', 0), r.get('secondary_accuracy', 0)] for r in rows], jnp.float32)
        self.secondary_early = jnp.array([r['growth'].get('secondary_early', [0, 0]) for r in rows], jnp.float32)
        self.secondary_late = jnp.array([r['growth'].get('secondary_late', [0, 0]) for r in rows], jnp.float32)
        self.capital_guards = jnp.array([r['name'] in CAPITAL_GUARDS for r in rows])
        self.secondary_sources = jnp.array([_source(r['secondary_attack_type'])+1
            if r.get('secondary_attack_type') else 0 for r in rows], jnp.uint32)
        self.sizes = jnp.array([0]+[r.get('size', 1) for r in rows[1:]], jnp.int32)
        self.base_levels = jnp.array(levels, jnp.int32)
        self.initial_ids = jnp.array(hero+[0]*6, jnp.int32)
        self.initial_levels = self.base_levels[self.initial_ids]
        initial_xp = jnp.array([r.get('exp_current', 0) for r in rows], jnp.int32)
        self.initial_xp = initial_xp[self.initial_ids]
        enemy_ids = jnp.array(enemies, jnp.int32)
        # Opponents have no capital, so their forms are immutable scenario data.
        self.enemy_ids = enemy_ids
        self.initial_enemies = jnp.stack((self.base_levels[enemy_ids], initial_xp[enemy_ids]), axis=-1)
        if 'enemy_initial_levels' in game_map:
            starting_levels = game_map['enemy_initial_levels']
            if (len(starting_levels) != len(enemies) or any(len(row) != 6 for row in starting_levels)
                    or any(type(level) is not int or level < levels[unit] or level > MAX_LEVEL
                           or (unit == 0 and level != 0)
                           for squad, row in zip(enemies, starting_levels) for unit, level in zip(squad, row))):
                raise ValueError('Enemy starting levels must match the roster and be at least the base level')
            self.initial_enemies = self.initial_enemies.at[:, :, 0].set(jnp.array(starting_levels, jnp.int32))
        self.base_stats = jnp.array([[0]*5]+[_validated_stats(r) for r in rows[1:]], jnp.float32)
        self.traits = jnp.array([[0]*4]+[[ROLES[r['role']], _source(r['attack_type'])+1,
            _protection_mask(r['immunities']), _protection_mask(r['protections'])] for r in rows[1:]], jnp.uint32)
        self.thresholds = jnp.array([r['growth']['threshold'] for r in rows], jnp.int32)
        self.early = jnp.array([r['growth']['early'] for r in rows], jnp.float32)
        self.late = jnp.array([r['growth']['late'] for r in rows], jnp.float32)
        # A tiny lookup covers the early levels; later growth is linear. This
        # avoids gathering five separate profile arrays on every environment step.
        self.level_anchor = max(max(r['growth']['threshold'],
            max((b['level'] for b in r.get('level_bonuses', [])), default=0)) for r in rows)

        def grown_stats(row, level):
            extra = max(level-row['level'], 0)
            early = min(extra, max(row['growth']['threshold']-row['level'], 0))
            late = extra-early
            if not row.get('level_bonuses'):
                return [row[k]+early*a+late*b for k, a, b in zip(
                    STAT_NAMES, row['growth']['early'], row['growth']['late'])]
            values = [row[k] for k in STAT_NAMES]
            for next_level in range(row['level']+1, level+1):
                increment = row['growth']['early' if next_level <= row['growth']['threshold'] else 'late']
                values = [v+inc for v, inc in zip(values, increment)]
                for bonus in row['level_bonuses']:
                    if bonus['level'] == next_level:
                        index = STAT_NAMES.index(bonus['stat'])
                        value = values[index]*bonus['multiplier']+bonus['bonus']
                        values[index] = int(value+.5) if bonus['rounding'] == 'half_up' else round(value)
                values[2] = min(values[2], 100)
            return values

        def xp_value(row, level, field, growth):
            # Gunits + GDynUpgr, applied by Steam EXE sub_599335/sub_599417.
            # Heroes use the same two-stage additive rule as other fighters.
            extra = max(level-row['level'], 0)
            early = min(extra, max(row['growth']['threshold']-row['level'], 0))
            first = row['growth'][growth+'_early']
            if growth == 'required':
                # Explicit scenario overrides (e.g. playable Betrezen) replace
                # only the early XP_NEXT increment; late growth stays native.
                first = row.get('exp_increment', first)
            return row[field]+early*first+(extra-early)*row['growth'][growth+'_late']

        self.stat_levels = jnp.array([[grown_stats(r, level) for r in rows]
                                     for level in range(self.level_anchor+1)], jnp.float32)
        # All supported levels are immutable device lookups, including 98 -> 99.
        self.kill_levels = jnp.array([[xp_value(r, level, 'exp_kill', 'kill') for r in rows]
                                     for level in range(MAX_LEVEL+1)], jnp.int32)
        self.required_levels = jnp.array([[xp_value(r, level, 'exp_required', 'required') for r in rows]
                                         for level in range(MAX_LEVEL+1)], jnp.int32)
        heavy_strike_ids = {'g000uu0019','g000uu0020','g000uu0044','g000uu0045','g000uu0047','g000uu0070','g000uu0071','g000uu0096','g000uu8009','g000uu8011'}
        support_types = {'Cliric','Profit','Patriach','Deva roshi','Sundancer','Sylfid','Ghost','Witch','Shadow','Incub','Succub','Baroness','Travnitsa','Novice','Alchemist','Dwarfdruid','Arhidruid'}
        self.stat_caps = jnp.array([[MAX_HP, jnp.inf if r.get('unit_type') in support_types else 400 if r.get('game_id') in heavy_strike_ids else 300, MAX_ACCURACY, MAX_ARMOR, MAX_INITIATIVE]
                                    for r in rows], jnp.float32)
        self.required = jnp.array([r['exp_required'] for r in rows], jnp.int32)
        self.dynamic = jnp.array([i > 0 and not r['upgrades'] for i, r in enumerate(rows)])
        targets, bits = [], []
        for row in rows:
            choices, required_bits = [], []
            for key in row['upgrades']:
                target = UNITS[key]
                building = next((i for i, b in enumerate(construction.rows)
                                 if b['unit'] == target['name']), None)
                usable = (key in self.ids and row['faction'] == construction.faction
                          and building is not None)
                choices.append(self.ids[key] if usable else 0)
                required_bits.append((1 << building) if usable else 0)
            # The reference's order decides between multiple built alternatives.
            targets.append((choices+[0, 0])[:2])
            bits.append((required_bits+[0, 0])[:2])
            if len(choices) > 2:
                raise ValueError('Progression catalogue supports at most two direct evolutions')
        self.targets = jnp.array(targets, jnp.int32)
        self.bot_targets = jnp.array([next((self.ids[k] for k in r['upgrades'] if k in self.ids), 0)
                                      for r in rows], jnp.int32)
        self.building_bits = jnp.array(bits, jnp.uint32)
        # Sinfo.MAX_UNIT limits buildable evolution tiers, not numerical levels.
        # A form whose every next tier exceeds that limit can grow in place.
        # Merely lacking gold or an unbuilt, allowed building never enables this.
        tier_limit = game_map.get('max_building_level', 5)
        building_levels = {b['unit']: b['level'] for b in construction.rows}
        self.scenario_dynamic = jnp.array([bool(r['upgrades'])
            and r['faction'] == construction.faction and not r.get('hero')
            and all(building_levels.get(UNITS[k]['name'], 0) > tier_limit for k in r['upgrades'])
            for r in rows])
        # Precompute integer absorbed damage for all supported primary powers.
        # Integer armour growth reaches its cap within this lookup. The first
        # ten levels use the early increment, all later levels the late one.
        armor_rows = []
        self.armor_max_level = self.level_anchor+100
        for level in range(self.armor_max_level+1):
            armor_rows.append([min(90, grown_stats(row, level)[ARMOR]) for row in rows])
        armors = list(range(91))
        self.armor_ids = jnp.array([[armors.index(a) for a in row] for row in armor_rows], jnp.int32)
        self.shatter_armor_ids = jnp.array([[[armors.index(a if k == 0 else max(0, int(a)-15*k))
                                            for k in range(7)] for a in row] for row in armor_rows], jnp.int32)
        self.damage_rolls = _damage_rolls(tuple(armors))
        self.has_protections = any(r['immunities'] or r['protections'] for r in rows)
        self.metadata = []
        for i, row in enumerate(rows):
            options = []
            for target in row['upgrades']:
                target_row = UNITS[target]
                buildings = CATALOG['factions'][row['faction']]['buildings']
                building = next((b for b in buildings if b['unit'] == target_row['name']), None)
                options.append(dict(name=target_row['name'],
                    building=building['name'] if building else None,
                    supported=not bool(target_row.get('upgrade_unavailable_reason')),
                    reason=target_row.get('upgrade_unavailable_reason', '')))
            self.metadata.append(None if i == 0 else dict(
                name=row['name'], key=row['key'], faction=row['faction'], level=row['level'],
                hero=row.get('hero', False), level_bonuses=row.get('level_bonuses', []),
                max_level=self.max_hero_level if row.get('hero') else MAX_LEVEL,
                scenario_dynamic=bool(self.scenario_dynamic[i]),
                hero_skills=row.get('hero_skills', []),
                unit_type=row.get('unit_type', ''), mass_support=support_reaches_all(row),
                ward_bits=ward_bits[i],
                secondary_attack_type=row.get('secondary_attack_type', ''),
                role=row['role'], size=row.get('size', 1), attack_type=row['attack_type'],
                immunities=_protection_mask(row['immunities']),
                protections=_protection_mask(row['protections']), upgrades=options,
                exp_required=row['exp_required'], exp_kill=row['exp_kill']))

    def raw_stats(self, ids, levels):
        levels = jnp.minimum(levels, MAX_LEVEL)
        anchor = jnp.minimum(levels, self.level_anchor)
        return self.stat_levels[anchor, ids] + jnp.maximum(levels-self.level_anchor, 0)[..., None]*self.late[ids]

    def stats(self, ids, levels):
        return jnp.minimum(self.raw_stats(ids,levels), self.stat_caps[ids])

    def secondary_stats(self, ids, levels):
        levels = jnp.minimum(levels, MAX_LEVEL)
        extra = jnp.maximum(levels-self.base_levels[ids], 0)
        early = jnp.minimum(extra, jnp.maximum(self.thresholds[ids]-self.base_levels[ids], 0))
        values = self.secondary_base[ids] + early[..., None]*self.secondary_early[ids] + (extra-early)[..., None]*self.secondary_late[ids]
        return jnp.minimum(values, jnp.array([300., 100.]))

    def required_xp(self, ids, levels):
        levels = jnp.minimum(levels, MAX_LEVEL)
        return self.required_levels[levels, ids]

    def missing_upgrade_building(self, state, ids, levels):
        """An own-faction, permitted evolution is waiting for its building."""
        bits = self.building_bits[ids]
        allowed = (bits != 0) & ((state.blocked_buildings & bits) == 0)
        built = allowed & ((state.buildings & bits) == bits)
        return ((ids != 0) & ~self.capital_guards[ids] & self.level_up_allowed(ids, levels)
                & ~self.dynamic[ids] & ~self.scenario_dynamic[ids]
                & jnp.any(allowed, axis=-1) & ~jnp.any(built, axis=-1))

    def blocked_xp_observation(self, state):
        """Six flags for living permanent fighters capped by an unbuilt building."""
        native = state.copied[:6] | (state.summon_owner[:6] >= 0)
        ids = jnp.where(native, state.native_ids[:6], state.unit_ids[:6])
        levels = jnp.where(native, state.native_levels[:6], state.unit_levels[:6])
        xp = jnp.where(native, state.native_xp[:6], state.unit_xp[:6])
        hp = jnp.where(state.summon_owner[:6] >= 0, state.summon_saved_hp[:6], state.hp[:6])
        party = jnp.where(state.summon_battle, state.spell_party, jnp.stack((ids, levels, xp, hp)))
        ids, levels, xp, hp = party
        required = self.required_xp(ids, levels)
        blocked = (self.missing_upgrade_building(state, ids, levels)
                   & (required > 0) & (xp >= required-1) & (hp > 0) & ~state.bot_autobattle)
        return blocked.astype(jnp.float32)

    def training_reward(self, state, gains, promoted, victory):
        """Pay real party promotions; charge once per fighter/battle for discarded XP."""
        ids, levels = state.unit_ids[:6], state.unit_levels[:6]
        required = self.required_xp(ids, levels)
        wasted = (self.missing_upgrade_building(state, ids, levels)
                  & (required > 0) & (gains[:6] > 0)
                  & (state.unit_xp[:6]+gains[:6] >= required))
        reward = (LEVEL_UP_REWARD*jax.lax.population_count(promoted & jnp.uint32(63))
                  - BLOCKED_XP_PENALTY*jnp.sum(wasted))
        return jnp.where(victory & ~state.summon_battle & ~state.bot_autobattle, reward, 0.)

    def level_up_allowed(self, ids, levels):
        """All units stop at 99; a scenario may impose an earlier hero ceiling."""
        return (levels < MAX_LEVEL) & (~self.heroes[ids] | (levels < self.max_hero_level))

    def experience(self, ids, levels, current):
        levels = jnp.minimum(levels, MAX_LEVEL)
        killed = self.kill_levels[levels, ids]
        return jnp.stack((levels, killed, self.required_xp(ids, levels), current), axis=-1)

    def damage(self, state, actor, targets, guarded, bonuses, damage_value=None, armor_shreds=None, imps=None, armor_values=None):
        if damage_value is None:
            damage_value = self.stats(state.unit_ids, state.unit_levels)[actor, DAMAGE]
        cap = self.stat_caps[state.unit_ids[actor], DAMAGE]
        cap = jnp.where(state.imp[actor] | state.fenrir[actor], 300., cap)
        damage_value = jnp.minimum(damage_value, cap)
        bonuses = capped_damage_bonus(damage_value, bonuses, cap)
        levels, ids = jnp.minimum(state.unit_levels[targets], min(MAX_LEVEL, self.armor_max_level)), state.unit_ids[targets]
        armor_ids = (self.armor_ids[levels, ids] if armor_shreds is None
                     else self.shatter_armor_ids[levels, ids, jnp.minimum(armor_shreds[targets], 6)])
        if imps is not None:
            armor_ids = jnp.where(imps[targets], 0, armor_ids)
        if armor_values is not None:
            armor_ids = jnp.clip(armor_values,0,90).astype(jnp.int32)
        return self.damage_rolls[damage_value.astype(jnp.int32), armor_ids,
                                 guarded.astype(jnp.int32), bonuses]

    def accumulate(self, state, before_hp, hp, escaped, kill_xp, death_order=None):
        """Original sub_62618B/62632E: distribute each death packet immediately.

        A regular attack is one packet, including all its targets. DOT deaths
        are separate packets in queue order. Newly dead allies still count in
        their packet's divisor; their own accumulated share is then erased.
        Summons never participate, including a summon covering an escapee.
        """
        permanent = (state.unit_ids != 0) & (state.summon_owner < 0)
        killed = (before_hp > 0) & (hp <= 0) & permanent & ~escaped
        participants = permanent & ~escaped & ((before_hp > 0) | (hp > 0))
        survivors = permanent & ~escaped & (hp > 0)
        worth = jnp.where(killed, kill_xp, 0)
        totals = jnp.sum(worth.reshape(2, 6), axis=1)
        side = jnp.arange(12) // 6
        if death_order is None:
            counts = jnp.sum(participants.reshape(2, 6), axis=1)
            shares = totals[1-side] // jnp.maximum(counts[side], 1)
        else:
            # Rows are recipients, columns are successive victims. A later
            # casualty remains in earlier divisors, even though its XP is lost.
            present = participants[:, None] & (death_order[:, None] >= death_order[None, :])
            counts = jnp.sum(present.reshape(2, 6, 12), axis=1)
            portions = worth[None, :] // jnp.maximum(counts[side], 1)
            shares = jnp.sum(jnp.where(side[:, None] != side[None, :], portions, 0), axis=1)
        earned = state.battle_earned_xp + jnp.where(survivors, shares, 0)
        earned = jnp.where(permanent & (hp <= 0), 0, earned)
        return state.replace(battle_earned_xp=earned, battle_xp=state.battle_xp+totals)

    def finish(self, state, hp, escaped, victory, lost, withdrawal, experience_book=False):
        """Award once, before automatic resurrection; one promotion per battle."""
        ended = victory | lost | withdrawal
        recipients = (hp > 0) & ended & (state.unit_ids != 0) & (state.summon_owner < 0)
        # Both sides retain their personal awards, including escaped fighters.
        recipients &= (self.required[state.unit_ids] > 0) & ~self.capital_guards[state.unit_ids]
        scale = jnp.where((jnp.arange(12) < 6) & experience_book, 5, 4)
        # sub_5840D8 truncates the percentage bonus on the personal integer XP.
        award = state.battle_earned_xp*scale//4
        # sub_665AB7 grants at least one XP to an otherwise eligible survivor.
        gains = jnp.where(recipients, jnp.maximum(award, 1), 0)
        gains = jnp.where(self.level_up_allowed(state.unit_ids, state.unit_levels), gains, 0)
        ids, levels = state.unit_ids, state.unit_levels
        required = self.required_xp(ids, levels)
        retreat_room = jnp.maximum(required-1-state.unit_xp, 0)
        gains = jnp.where(escaped, jnp.minimum(gains, retreat_room), gains)
        xp = state.unit_xp + gains
        reached = (gains > 0) & (required > 0) & (xp >= required)
        choices, bits = self.targets[ids], self.building_bits[ids]
        # Opponents have no capital in this scenario. They still retain XP and
        # can gain dynamic levels if their form has no further evolution.
        built = ((state.buildings & bits) == bits) & (bits != 0) & (jnp.arange(12)[:, None] < 6)
        built &= (state.blocked_buildings & bits) == 0
        target = jnp.where(built[:, 0], choices[:, 0], jnp.where(built[:, 1], choices[:, 1], 0))
        if self.bot_enabled:
            target = jnp.where(state.bot_autobattle & (jnp.arange(12) < 6), self.bot_targets[ids], target)
        evolved = reached & (target != 0)
        scenario_growth = self.scenario_dynamic[ids] & (jnp.arange(12) < 6) & ~state.bot_autobattle
        grown = reached & (self.dynamic[ids] | scenario_growth) & ~evolved
        changed = evolved | grown
        ids = jnp.where(evolved, target, ids)
        levels = jnp.where(evolved, self.base_levels[ids], levels + grown.astype(jnp.int32))
        xp = jnp.where(changed, 0, jnp.where(reached, required-1, xp))
        new_hp = jnp.where(changed, self.stats(ids, levels)[:, HP].astype(jnp.int32), hp)
        saved = jnp.stack((levels[6:], xp[6:]), axis=-1)
        enemies = state.enemy_progress.at[state.enemy].set(
            jnp.where(ended, saved, state.enemy_progress[state.enemy]))
        promoted = jnp.sum(jnp.where(changed, jnp.left_shift(jnp.uint32(1), jnp.arange(12, dtype=jnp.uint32)), jnp.uint32(0)))
        return ids, levels, xp, enemies, new_hp, gains, promoted
