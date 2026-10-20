"""Static combat profiles and pure JAX hit checks; see docs/ATTACK_PROTECTIONS.md."""
import json
import math
from pathlib import Path

import jax.numpy as jnp
from stoix.envs.number_grid_effects import CAPITAL_GUARDS
from stoix.envs.number_grid_limits import MAX_HP, MAX_INITIATIVE, MAX_ARMOR

HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE = range(5)
STAT_NAMES = ('max_hp', 'damage', 'accuracy', 'armor', 'initiative')
ATTACK_TYPES = ('weapon', 'earth', 'fire', 'water', 'poison', 'death', 'mind', 'life', 'air')
ATTACK_LABELS = ('Оружие', 'Земля', 'Огонь', 'Вода', 'Яд', 'Смерть', 'Разум', 'Жизнь', 'Воздух')
EMPTY, MELEE, RANGED, AREA, HEALER = range(5)
ROLES = {'melee': MELEE, 'ranged': RANGED, 'area': AREA, 'healer': HEALER}
UNITS = json.loads((Path(__file__).parent / 'data/units.json').read_text(encoding='utf-8'))
DEFEND_BONUS = 50  # Installed GVars.dbf: DFENDBONUS; see docs/ARCHER_BATTLES.md.
PROFILE_FIELDS = set(STAT_NAMES) | {'size', 'role', 'attack_type', 'immunities', 'protections', 'exp_kill', 'exp_required', 'exp_current', 'hero',
                                  'secondary_attack_type', 'secondary_accuracy', 'secondary_damage'}


def armor_with_defend(armor, defended):
    """Original Defend adds a truncated fraction of missing armor, capped at 90."""
    bonus = jnp.trunc((100-armor)*DEFEND_BONUS/100)
    return jnp.where(defended, jnp.minimum(MAX_ARMOR, armor+bonus), armor)


def capped_damage_bonus(damage, bonus, cap):
    """Limit the random addition so capped power plus bonus cannot exceed cap."""
    return jnp.minimum(bonus, jnp.maximum(cap-damage, 0)).astype(jnp.int32)


def damage_after_armor(damage, armor):
    """Subtract the integer absorbed part, as in Discipl2.exe sub_65C047."""
    damage = damage.astype(jnp.int32)
    return damage - jnp.trunc(damage*armor/100).astype(jnp.int32)


def build_damage_rolls(damages, armors):
    """Build armor reductions before JIT; callers cap power and the random bonus."""
    # Scalar equivalent of armor_with_defend; no device operations inside these
    # construction loops. The learner only indexes the resulting device table.
    effective = [(armor, min(MAX_ARMOR, armor+int((100-armor)*DEFEND_BONUS/100)))
                 for armor in armors]
    return jnp.asarray([[[[damage+bonus-int((damage+bonus)*armor/100) if damage else 0
                          for bonus in range(6)] for armor in pair]
                        for pair in effective] for damage in damages], jnp.int32)


def can_revive(state, slots):
    """Shared resurrection eligibility; callers also check occupancy and access.

    Battle summons carry an owner (their own slot for item summons). World
    summons instead replace the player party under the summon_battle flag.
    Unit identity is irrelevant: a permanent Zombie can still be resurrected.
    """
    return ((state.hp[slots] <= 0) & (state.summon_owner[slots] < 0)
            & ~(state.summon_battle & (slots < 6)))


def uses_aoe_accuracy_falloff(profile):
    """Reference Mage heroes and Nosferatu lose 10 points per living target."""
    unit_type = profile.get('unit_type', 'Mage' if profile.get('role') == 'area' else '')
    return bool(profile.get('hero') and (unit_type == 'Mage' or profile.get('name') == 'Носферату'))


def _validated_stats(values):
    stats = {name: values[name] for name in STAT_NAMES}
    for name, value in stats.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Invalid combat {name}')
    if (type(stats['max_hp']) is not int or not 1 <= stats['max_hp'] <= 1_000_000
            or type(stats['damage']) is not int or not 0 <= stats['damage'] <= 1_000_000
            or not 0 <= stats['accuracy'] <= 100 or not 0 <= stats['armor'] <= 100
            or type(stats['initiative']) is not int or not 0 <= stats['initiative'] <= 1000):
        raise ValueError('Combat stats outside supported ranges')
    stats['damage'] = min(stats['damage'], 400 if values.get('hero') else 300)
    stats['armor'] = min(stats['armor'], MAX_ARMOR)
    stats['max_hp'] = min(stats['max_hp'], MAX_HP)
    stats['initiative'] = min(stats['initiative'], MAX_INITIATIVE)
    return [stats[name] for name in STAT_NAMES]


def _source(value):
    if not isinstance(value, str) or value.lower() not in ATTACK_TYPES:
        raise ValueError('Unknown attack type')
    return ATTACK_TYPES.index(value.lower())


def _protection_mask(values):
    if not isinstance(values, list):
        raise ValueError('Immunities/protections must be lists of attack types')
    mask = 0
    for value in values:
        mask |= 1 << _source(value)
    return mask


def build_combat_tables(game_map):
    """Resolve named six-slot formations or explicit test profiles once before jit."""
    hero_count = game_map.get('hero_units', 6)
    counts = game_map['enemy_units']
    warriors = game_map.get('enemy_warrior_slots', [-1] * len(counts))
    hero_overrides = game_map.get('hero_combat_stats', [{} for _ in range(hero_count)])
    enemy_overrides = game_map.get('enemy_combat_stats', [[{} for _ in range(n)] for n in counts])
    if not isinstance(hero_overrides, list) or len(hero_overrides) != hero_count:
        raise ValueError('hero_combat_stats must contain one entry per hero unit')
    if (not isinstance(enemy_overrides, list) or len(enemy_overrides) != len(counts)
            or any(not isinstance(s, list) or len(s) != n for s, n in zip(enemy_overrides, counts))):
        raise ValueError('enemy_combat_stats must match the squad sizes')
    hero_roster = game_map.get('hero_roster')
    enemy_rosters = game_map.get('enemy_rosters')
    if enemy_rosters is not None and (not isinstance(enemy_rosters, list) or len(enemy_rosters) != len(counts)):
        raise ValueError('enemy_rosters must match all squads')

    def formation(count, overrides, roster, enemy=None):
        if roster is not None:
            if (not isinstance(roster, list) or len(roster) != 6
                    or sum(u is not None for u in roster) != count
                    or any(u is not None and (not isinstance(u, str) or u not in UNITS) for u in roster)):
                raise ValueError('Roster must have six slots and the declared count of known units')
        profiles, rows, traits = [], [], []
        index = 0
        for slot in range(6):
            occupied = roster[slot] is not None if roster is not None else slot < count
            if not occupied:
                profiles.append(None)
                rows.append([0]*5)
                traits.append([EMPTY, 0, 0, 0])
                continue
            if roster is not None:
                defaults = dict(UNITS[roster[slot]])
            else:
                warrior = slot == (game_map.get('hero_warrior_slot', -1) if enemy is None else warriors[enemy])
                mage = enemy is None and slot == game_map.get('hero_mage_slot', -1)
                role = 'warrior' if warrior else 'mage' if mage else 'archer'
                health = (game_map.get('warrior_hp', 100) if warrior else
                          game_map.get('mage_hp', game_map.get('hero_hp', 45)) if mage else
                          game_map.get('hero_hp', 45)) if enemy is None else game_map['enemy_hp'][enemy]
                defaults = dict(name='Воин' if warrior else 'Маг' if mage else 'Лучник', level=1,
                    role='melee' if warrior else 'area' if mage else 'ranged',
                    max_hp=health, damage=game_map.get(role+'_damage', 20 if mage else 25),
                    accuracy=100*game_map.get(role+'_accuracy', game_map.get('archer_accuracy', .8)),
                    armor=game_map.get(role+'_armor', 0), initiative=game_map.get(role+'_initiative', 50 if warrior else 60),
                    attack_type='fire' if mage else 'weapon', immunities=[], protections=[])
            override = overrides[index]
            index += 1
            if not isinstance(override, dict) or set(override) - (PROFILE_FIELDS | {'exp_increment', 'name'}):
                raise ValueError('Unknown combat profile fields')
            values = {**defaults, **override}
            if not isinstance(values['name'], str) or not values['name'].strip():
                raise ValueError('Combat name must be a nonempty string')
            if type(values.get('hero', False)) is not bool:
                raise ValueError('Combat hero flag must be boolean')
            if values.get('upgrade_unavailable_reason'):
                raise ValueError(values['upgrade_unavailable_reason'])
            if not isinstance(values['role'], str) or values['role'] not in ROLES:
                raise ValueError('Unknown combat role')
            for field in ('exp_kill', 'exp_required', 'exp_current', 'exp_increment'):
                value = values.get(field, 0)
                if type(value) is not int or not 0 <= value <= 1_000_000:
                    raise ValueError('Invalid combat '+field)
            if values.get('exp_required', 0) > 0 and values.get('exp_current', 0) >= values['exp_required']:
                raise ValueError('Initial experience must be below the level threshold')
            size = values.get('size', 1)
            if type(size) is not int or size not in (1, 2):
                raise ValueError('Combat size must be 1 or 2')
            stats = _validated_stats(values)
            source = _source(values['attack_type'])
            secondary_source = _source(values['secondary_attack_type'])+1 if values.get('secondary_attack_type') else 0
            for field, limit in (('secondary_accuracy', 100), ('secondary_damage', 1_000_000)):
                v = values.get(field, 0)
                if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= limit:
                    raise ValueError('Invalid combat '+field)
            immune, wards = _protection_mask(values['immunities']), _protection_mask(values['protections'])
            profiles.append(dict(name=values['name'], level=values['level'], size=size, role=values['role'],
                                 damage_cap=400 if values.get('hero') else 300,
                                 unit_type=values.get('unit_type', ''), game_data=values.get('game_data', {}),
                                 secondary_source=secondary_source,
                                 secondary_accuracy=values.get('secondary_accuracy', 0),
                                 secondary_damage=values.get('secondary_damage', 0),
                                 capital_guard=values['name'] in CAPITAL_GUARDS,
                                 attack_type=ATTACK_TYPES[source], immunities=immune, protections=wards,
                                 aoe_accuracy_falloff=uses_aoe_accuracy_falloff(values)))
            rows.append(stats)
            traits.append([ROLES[values['role']], source+1, immune, wards])
        for slot, profile in enumerate(profiles):
            if profile and profile['size'] == 2 and (slot >= 3 or profiles[slot+3] is not None):
                raise ValueError('Large units need a front slot and an empty paired rear slot')
        return profiles, rows, traits

    hero_names, heroes, hero_traits = formation(hero_count, hero_overrides, hero_roster)
    squads, traits, enemies = [], [], []
    for enemy, count in enumerate(counts):
        names, rows, properties = formation(count, enemy_overrides[enemy],
            enemy_rosters[enemy] if enemy_rosters is not None else None, enemy)
        squads.append(heroes+rows)
        traits.append(hero_traits+properties)
        enemies.append(names)
    damages = sorted({s[DAMAGE] for squad in squads for s in squad} | {20, 30, 90})
    original_armors = {s[ARMOR] for squad in squads for s in squad}
    armors = sorted(original_armors | {max(0, int(a)-15*k) for a in original_armors for k in range(1,7)})
    damage_ids = [[damages.index(s[DAMAGE]) for s in squad] for squad in squads]
    armor_ids = [[armors.index(s[ARMOR]) for s in squad] for squad in squads]
    rolls = build_damage_rolls(damages, armors)
    metadata = dict(heroes=hero_names, enemies=enemies, fenrir_damage_id=damages.index(90),
                    imp_damage_ids=[damages.index(20), damages.index(30)],
                    _shatter_armor_ids=[[[armors.index(s[ARMOR] if k == 0 else max(0, int(s[ARMOR])-15*k))
                                         for k in range(7)] for s in squad] for squad in squads],
                    attack_types=[dict(key=key, name=label, bit=1 << i)
                                  for i, (key, label) in enumerate(zip(ATTACK_TYPES, ATTACK_LABELS))])
    return (jnp.asarray(squads, jnp.float32), jnp.asarray(damage_ids, jnp.int32),
            jnp.asarray(armor_ids, jnp.int32), rolls,
            max(s[INITIATIVE] for squad in squads for s in squad)+10.,
            jnp.asarray(traits, jnp.uint32), metadata,
            any(t[2] or t[3] for squad in traits for t in squad))


def accuracy_hits(accuracy, first, second):
    """Python reference: average two independent integer draws from 0..99."""
    a = jnp.clip(jnp.floor(first*100), 0, 99)
    b = jnp.clip(jnp.floor(second*100), 0, 99)
    return a+b < 2*accuracy
