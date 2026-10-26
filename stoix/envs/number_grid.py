"""Incremental archer/warrior/mage battles, simulated entirely in JAX.

A step is one player decision. A JAX loop advances enemy turns, forced skips
and completed retreats until the next decision. Saved maps can use numeric rules.
"""
import dataclasses
import json
import math
import os
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import struct
from stoix.envs.number_grid_combat import (
    HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE, MELEE, AREA, HEALER, build_combat_tables, accuracy_hits, can_revive,
    armor_with_defend, capped_damage_bonus, damage_after_armor,
)
from stoix.envs.number_grid_updates import set_column, set_rows
from stoix.envs.number_grid_capital import CapitalRules, HEAL_START, REVIVE_START, CAPITAL_ACTIONS
from stoix.envs.number_grid_potions import PotionRules, POTION_START, merchants, scenario_potions
from stoix.envs.number_grid_progression import ProgressionRules, BLOCKED_XP_FEATURES
from stoix.envs.number_grid_limits import MAX_HP, MAX_RESOURCE, cap_stats, add_resource
from stoix.envs.number_grid_summoning import Summoning, COPY_ALLY, COPY_ACTIONS, advance_linked_queue
from stoix.envs.number_grid_chests import ChestRules
from stoix.envs.number_grid_capital_training import CapitalTrainingRules
from stoix.envs.number_grid_items import ItemRules
from stoix.envs.number_grid_sites import BOUGHT, SiteRules, sites_present, trade_action_names
from stoix.envs.number_grid_recruitment import RecruitmentRules, mercenary_camps, recruit_action_names, HIRED, DISMISSED
from stoix.envs.number_grid_ruins import RuinRules
from stoix.envs.number_grid_terrain import TerrainRules
from stoix.envs.number_grid_spells import SpellResearchRules, research_action_names
from stoix.envs.number_grid_casting import SpellCastRules, SUMMON_CAST, cast_action_names, casting_enabled
from stoix.envs.number_grid_territory import MANA_KINDS, TerritoryRules, OBSTACLE
from stoix.envs.number_grid_item_combat import equipment_attack, transform_imp, transform_decay
from stoix.envs import number_grid_battle_items as battle_items
from stoix.envs import number_grid_hero_skills as hero_skills
from stoix.envs.number_grid_battle_items import (BattleItemRules, battle_item_action_names, battle_items_enabled,
    BATTLE_ITEM)
from stoix.envs.number_grid_unit_swaps import UnitSwapRules, unit_swap_action_names, UNITS_SWAPPED
from stoix.envs.number_grid_lords import with_starting_party
from stoix.envs import number_grid_unit_swaps as unit_swaps
from stoix.envs.number_grid_wards import ward_tags, grant_wards, expire_wards
from stoix.envs.number_grid_metrics import campaign_events, empty_campaign_counts
from stoix.envs.number_grid_episode_reports import empty_event, episode_event
from stoix.envs.number_grid_scenarios import ScenarioRules
from stoix.envs.number_grid_script_events import ScriptEventRules
from stoix.envs.number_grid_bot import ScriptedBotRules
from stoix.envs.number_grid_target_rewards import TargetRewardRules
from stoix.envs import number_grid_reference_rewards as reference_rewards
from stoix.envs.number_grid_effects import effect_tables, effect_breaks, POISON, FROSTBITE, BLISTER, DISABLE, TRANSFORM, SHORT, LONG, armor_after_shreds, source_protection, advance_poison_queue, advance_periodic_queue, vampiric_heal
from stoix.envs.number_grid_effects import disable_attack_accuracy, disable_attack_allowed
from stoix.envs.number_grid_buildings import (
    BuildingRules, BUILD_START, BUILD_SLOTS, DAILY_GOLD,
)
from stoa import AddActionMaskWrapper, ArraySpace, DictSpace, DiscreteSpace
from stoix.envs.number_grid_legacy import (
    DIRECTIONS, NumberGrid as NumericNumberGrid, NumberGridState,
)

_ROOT = Path(__file__).resolve().parents[2]
# Named maps for training and human mode; anything else is a JSON path.
MAP_FILES = {'current': _ROOT / 'number_grid_map.json', 'default': _ROOT / 'maps/default.json'}
MAP_FILES.update({name: _ROOT / 'maps' / (name+'.json')
                 for name in ('small', 'super_last_stand', 'siege_train', 'the_charmed_child', 'trade_train',
                              'mirror_match', 'bethrezens_retreat')})
MAP_FILES['mirrow_match'] = MAP_FILES['mirror_match']
MAP_FILES['mirror-match'] = MAP_FILES['mirror_match']
DRAGON_MAPS = {base+'-'+color+'-dragon': (base, objective)
               for base in ('default', 'small')
               for color, objective in (('green', 'dragon'), ('blue', 'blue_dragon'))}


def load_map(name=None):
    """Named campaign (including Default/Small goals), or a standalone JSON map."""
    if name == 'default-capital':
        # User-requested Simpler Times variant (2026-10-16): keep the map,
        # including its green dragon and its Legions warrior, and pay the
        # blue-dragon rewards for Mizrael.
        # User request 2026-10-26: no other Mizrael reward (bottles, effects, entry
        # strength, rounds); a fixed +10 for a guardian battle once more than 50
        # other armies have been defeated in the episode.
        game_map = load_map('default')
        target = game_map['enemy_capitals'][0]['guard']
        if game_map['enemy_rosters'][target] != [None, None, None, None, 'g000uu3001', None]:
            raise ValueError('Default Capital objective must be the Mizrael guard stack')
        game_map.update(name=f"NumberGrid-{game_map['size']}x{game_map['size']}-default-capital",
                        display_name='Default — Capital', campaign_objective='capital_guardian',
                        objective_enemy=target, objective_reward_hero_only=True,
                        army_victory_level_reward=5., army_victory_level_objective_only=True,
                        scenario_events_enabled=False,
                        capital_guardian_battle_reward=10., capital_guardian_battle_min_defeated=51,
                        episode_timeout_penalty=100., battle_round_limit_penalty=100.,
                        recruitment_reward_faction_only=True)
        for city in game_map['cities']:
            city['objective'] = False
        return game_map
    if name in DRAGON_MAPS:
        # Reference maps/default.py and enemy_configs.build_blue_dragon_enemy_configs:
        # enemy 31 keeps its position and formation slot; Small has sparse source IDs.
        base, objective = DRAGON_MAPS[name]
        game_map = load_map(base)
        source_ids = game_map.get('reference_enemy_ids', list(range(1, len(game_map['enemy_rosters'])+1)))
        target = source_ids.index(31)
        if game_map['enemy_rosters'][target] != [None, 'g000uu5023', None, None, None, None]:
            raise ValueError(f'{base} objective enemy 31 must be the Green Dragon')
        blue = objective == 'blue_dragon'
        if blue:
            units = json.loads((_ROOT / 'stoix/envs/data/units.json').read_text(encoding='utf-8'))
            for i, roster in enumerate(game_map['enemy_rosters']):
                if 'g000uu5023' in roster:
                    roster[:] = ['g000uu5021' if unit == 'g000uu5023' else unit for unit in roster]
                    game_map['enemy_hp'][i] = max(units[unit]['max_hp'] for unit in roster if unit)
        size = game_map['size']
        game_map.update(name=f'NumberGrid-{size}x{size}-'+name, campaign_objective=objective,
                        objective_enemy=target, objective_reward_hero_only=blue)
        if base == 'small':
            game_map['display_name'] = 'Small — '+('синий дракон' if blue else 'зелёный дракон')
        if blue:
            game_map.update(army_victory_level_reward=5., army_victory_level_objective_only=True)
        for city in game_map['cities']:
            city['objective'] = False
        return game_map
    return json.loads(Path(MAP_FILES.get(name or 'current', name)).read_text(encoding='utf-8'))


# NUMBER_GRID_MAP selects the module map; it fixes ACTION_NAMES for human mode.
MAP = load_map(os.environ.get('NUMBER_GRID_MAP'))
SHOOT, DEFEND, WAIT, RETREAT, CONTINUE = 8, 14, 15, 16, 17
REST = BUILD_START + BUILD_SLOTS
BASE_ACTIONS, FENRIR = REST + 1, CAPITAL_ACTIONS
def action_names(game_map):
    game_map = with_starting_party(game_map)
    return ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW') + tuple(
        f'shoot_{i}' for i in range(6)) + ('defend', 'wait', 'retreat', 'continue') + tuple(
        f'build_{i}' for i in range(BUILD_SLOTS)) + ('rest',) + tuple(
        f'heal_{i}' for i in range(6)) + tuple(f'revive_{i}' for i in range(6)) + ('transform_fenrir',) + tuple(
        f'copy_ally_{slot}' for slot in range(6)) + tuple(
        f'potion_{item["key"]}_{slot}' for item in scenario_potions(game_map) for slot in range(6)) + trade_action_names(
        game_map) + recruit_action_names(game_map) + (tuple('upgrade_city_'+str(i) for i in range(len(game_map.get('cities', []))))
        if game_map.get('territory') else ()) + research_action_names(game_map) + cast_action_names(game_map) + battle_item_action_names(game_map) + unit_swap_action_names(
        game_map)


ACTION_NAMES = action_names(MAP)
ACTIONS = len(ACTION_NAMES)
# Reference MOVES_PER_TURN: the allowance without a living hero; heroes have their own.
MAX_MOVEMENT_POINTS, MOVE_COST = 20, 2
MOVE, ENGAGE, HIT, MISS, GUARD, DELAY, FLEE, ESCAPE, VICTORY, DEFEAT, WITHDRAW, LIMIT = range(12)
BUILD, RESTED, IMMUNE, WARD, HEAL = 12, 13, 14, 15, 16
TRANSFORMED = 22
IMP_TRANSFORMED = 23
DECAY_TRANSFORMED = 24
FEARED = 27
POST_HEAL = 28
BUFFED = 29
BATTLE_REVIVED = 30
EXTRA_TURN = 31
SLOWED = 32
PARALYZED = 25
PARALYSIS_SKIP = 26
ROUND_LIMIT_RETREAT = 56
FORT_NONE, FORT_BLUE, FORT_RED = range(3)


@struct.dataclass
class BattleState(NumberGridState):
    bot_ids: jax.Array
    bot_mode: jax.Array
    bot_timer: jax.Array
    bot_target: jax.Array
    bot_moves_spent: jax.Array
    bot_defeated_enemies: jax.Array
    bot_hero_victories: jax.Array
    bot_autobattle: jax.Array
    opponent_claims: jax.Array
    territory_owners: jax.Array
    enemy_positions: jax.Array
    scenario_spawned: jax.Array
    script_events_fired: jax.Array
    script_objective_ready: jax.Array
    script_orders: jax.Array
    pending_encounters: jax.Array
    battle_hero_fort_armor: jax.Array
    battle_fort_defender: jax.Array  # 0: field, 1: blue defends a settlement, 2: red
    reference_visits: jax.Array
    reference_previous: jax.Array
    reference_city_reached: jax.Array
    reference_defeated: jax.Array
    reference_hp_milestones: jax.Array
    reference_potion_pending: jax.Array
    capital_defense_acquired: jax.Array  # newly obtained defensive bottles, not initial stock
    capital_defense_peak: jax.Array  # peak simultaneous living potion recipients per kind at guardian
    capital_rounds_rewarded: jax.Array  # highest guardian round minus one, sticky across attempts
    capital_entry_strength: jax.Array  # -1 until first hero contact; then frozen living entry level sum
    capital_guardian_bonus_paid: jax.Array  # the guardian battle bonus was paid this episode
    siege_rewarded_kills: jax.Array  # six original siege slots, sticky until episode reset
    army_victory_rewarded: jax.Array  # one victory payment per enemy, or empty when disabled
    army_battle_level_strength: jax.Array  # living permanent party's levels above 1 at battle entry
    native_ids: jax.Array
    native_levels: jax.Array
    native_xp: jax.Array
    copied: jax.Array
    copy_stats: jax.Array
    copy_secondary: jax.Array
    copy_traits: jax.Array
    summon_owner: jax.Array
    summon_saved_hp: jax.Array  # escaped permanent fighter beneath a battle summon
    preparation: jax.Array
    initiative_override: jax.Array  # native or exact temporary form/slow base
    saved_initiative_override: jax.Array
    slow_original: jax.Array  # -1: no Hermit slow; otherwise pre-slow base
    waited: jax.Array  # WAIT is still spent after an Alchemist grants another turn
    bonus_turns: jax.Array
    battle_revived: jax.Array  # one Patriarch revival per recipient per battle
    battle_earned_xp: jax.Array  # personal XP earned since the last death, retained on escape
    primary_override: jax.Array  # -1 means native/current-form damage
    preweak_damage: jax.Array
    powerup: jax.Array
    powerup_long: jax.Array
    saved_powerup_long: jax.Array
    effect_rounds: jax.Array  # [12,5]: DOTs, disable, hostile transform application rounds
    disable_source: jax.Array  # caster slot; finite disable also ends on its next activation
    transform_long: jax.Array
    powerup_layered: jax.Array  # Lower Damage was applied over the current buff
    saved_primary_override: jax.Array
    saved_preweak_damage: jax.Array
    saved_powerup: jax.Array
    saved_powerup_layered: jax.Array
    healer_wards: jax.Array  # [12,4] caster ownership bits, shared elemental blocks
    ward_native_used: jax.Array  # sticky spent native wards while grants are active
    saved_healer_wards: jax.Array
    saved_ward_native_used: jax.Array
    post_victory: jax.Array  # 0 combat/map; 1 blue healing; 2 red healing
    pending_healers: jax.Array  # one native support action each, in formation order
    fenrir: jax.Array  # temporary Wolf Lord form; permanent identity/XP is retained
    armor_shreds: jax.Array  # successful 15-point Shatters, capped once armour reaches zero
    imp: jax.Array  # Witch/Succub form overlays Fenrir without changing HP or size
    imp_wards: jax.Array  # spent wards of the suspended natural form
    decay_form: jax.Array  # native catalogue ID of a Wight-lowered temporary form
    decay_shreds: jax.Array  # armour effects applied only to the temporary form
    feared: jax.Array  # distinguish magical fear from voluntary retreat for Cure
    enemy_initial_hp: jax.Array
    enemy_damage_credit: jax.Array
    enemy_killed_credit: jax.Array
    weakened: jax.Array  # Tiamat: current-form primary amount x0.67, no stacking
    saved_weakened: jax.Array  # debuff of the form suspended by Witch/Wight
    item_weakened: jax.Array  # the debuff came from Orb of Weakening: x0.67
    saved_item_weakened: jax.Array
    lycanthrope: jax.Array  # hostile transformation; Drain Level has no random recovery
    paralyzed: jax.Array  # finite: skip one activation
    long_paralyzed: jax.Array  # round-age recovery checked before the target can act
    second_strike: jax.Array  # remaining second attack of the current Demon/Elfarcher
    poison_mode: jax.Array
    poison_damage: jax.Array
    poison_source: jax.Array  # caster slot, -1 when absent
    water_mode: jax.Array
    water_damage: jax.Array
    water_source: jax.Array
    last_water_damage: jax.Array
    burn_mode: jax.Array
    burn_damage: jax.Array
    burn_source: jax.Array
    last_burn_damage: jax.Array
    last_poison_damage: jax.Array
    activation_done: jax.Array  # once-per-round effects, unaffected by WAIT
    potions: jax.Array  # dense inventory of the scenario's obtainable potion types
    potion_doses: jax.Array  # [12,5] 0/1: each permanent kind used once, retained across growth
    potion_bonus: jax.Array  # [12,7] cached permanent deltas; secondary columns 5-6 stay zero
    potion_temporary: jax.Array  # [12,7] cached temporary stat deltas; secondary columns 5-6 stay zero
    potion_active: jax.Array  # one bit per temporary catalogue item, until rest
    potion_wards: jax.Array
    last_potion: jax.Array
    learned_spells: jax.Array  # faction-local uint32 bitset
    spell_researched_today: jax.Array
    last_researched_spell: jax.Array
    mana: jax.Array
    spell_casts: jax.Array  # casts of each faction spell since the last REST
    spell_effects: jax.Array  # active hero-stack support spells, uint32 bitset
    enemy_spell_effects: jax.Array  # active debuff bitset per enemy stack
    enemy_wounds: jax.Array  # [stacks,6] HP lost by map stacks between battles
    summon_marks: jax.Array  # stacks a summon fought today: one hero-engage bonus
    target_rewarded: jax.Array  # once-per-episode destruction bonuses for configured targets
    target_potential: jax.Array  # cached progress at the previous decision boundary
    summon_battle: jax.Array  # the blue side is a lone summoned unit
    spell_party: jax.Array  # [4,6] stashed hero ids/levels/xp/hp during a summon battle
    spell_mods: jax.Array  # [12,5] battle modifiers fixed at engagement
    spell_wards: jax.Array  # [12] source bits granted by ward spells
    last_cast_spell: jax.Array
    last_spell_target: jax.Array
    last_spell_amount: jax.Array
    scroll_magic: jax.Array  # scrolls/staves readable: map flag, mage hero or the hired reader
    last_cast_item: jax.Array  # scenario scroll/staff of the latest cast, -1 for the book
    unit_swaps_today: jax.Array  # successful party swaps since the last REST
    unit_swaps_used: jax.Array  # swap actions already used since the last REST, one bit each
    last_swap: jax.Array  # swap pair of the latest map transition, -1 for none
    territory_claims: jax.Array
    city_owned: jax.Array
    city_levels: jax.Array
    last_captured_cities: jax.Array
    last_upgraded_city: jax.Array
    battle_fort_armor: jax.Array
    hire_rewarded_capacity: jax.Array
    mercenary_units: jax.Array  # permanent camp-hire provenance, attached to the twelve roster slots
    mercenary_stock: jax.Array
    merchant_stock: jax.Array
    spell_stock: jax.Array  # spell shop offers left
    last_sale_gold: jax.Array
    last_sold_count: jax.Array
    last_trade_item: jax.Array
    ruin_looted: jax.Array
    last_ruin: jax.Array
    chest_alive: jax.Array
    last_loot: jax.Array  # potion counts granted by the most recent map transition
    item_inventory: jax.Array  # acquisition order, scenario item IDs; -1 empty
    item_charges_spent: jax.Array  # per inventory copy; zero for fresh items and empty cells
    equipped: jax.Array  # artifact x2, banner, book, boots; inventory retains all items
    equipment_bonus: jax.Array  # cached artifact/banner layer, after permanent/temporary potions
    last_item_loot: jax.Array
    battle_items: jax.Array  # two hero battle slots, scenario battle-item kinds; -1 empty
    battle_item_charges: jax.Array  # talisman charges left in each slot
    battle_item_used: jax.Array  # talisman slot already used in this battle
    battle_item_equipped_today: jax.Array  # one equip action per strategic turn
    last_battle_item: jax.Array
    recovery_balance: jax.Array  # victory-funded HP/kill allowances; debt is retained
    last_service_cost: jax.Array
    gold: jax.Array
    map_steps: jax.Array  # successful map movements, excluding combat/invalid actions
    movement_points: jax.Array
    day: jax.Array  # starts at 1; only REST begins the next strategic turn
    buildings: jax.Array  # uint32 bit set: one bit per construction action
    blocked_buildings: jax.Array
    built_today: jax.Array
    last_building: jax.Array
    battle_key: jax.Array
    in_battle: jax.Array
    enemy: jax.Array
    origin: jax.Array
    hp: jax.Array
    wards_used: jax.Array  # source bits consumed per unit during this battle
    last_immune: jax.Array  # bit set of unit slots blocked by immunity
    last_ward: jax.Array  # bit set of unit slots blocked by a one-hit ward
    priority: jax.Array  # normal initiative, or negative FIFO position after WAIT
    turn_phase: jax.Array  # 0: normal turn, 1: waiting, 2: finished
    defended: jax.Array
    retreating: jax.Array
    escaped: jax.Array
    actor: jax.Array
    round: jax.Array
    last_event: jax.Array
    last_actor: jax.Array
    last_target: jax.Array  # -1 for area attacks or non-attacks
    last_damage: jax.Array  # HP removed (or restored for HEAL), summed over targets
    unit_ids: jax.Array
    unit_levels: jax.Array
    unit_xp: jax.Array
    enemy_progress: jax.Array
    battle_xp: jax.Array  # killed-unit XP totals by victim side
    last_xp: jax.Array  # actual XP shares before caps/reset
    last_promoted: jax.Array  # promoted slot bits
    battle_steps: jax.Array
    player_turns: jax.Array
    enemy_turns: jax.Array


class NumberGrid(Summoning, NumericNumberGrid):
    def __init__(self, max_steps=None, map_config=None, episode_reports=False):
        """Validate the campaign configuration and prepare static rules for JAX transitions."""
        self.episode_reports = False
        # Without an explicit hero_roster the ruler chooses the starting hero and party.
        game_map = with_starting_party(MAP if map_config is None else map_config)
        super().__init__(max_steps=max_steps, map_config=game_map)
        self.battle_mode = bool(game_map.get('battle_mode', False))
        if not self.battle_mode:
            return
        self.reference_rewards = game_map.get('reward_profile') == 'python_reference'
        self.starting_gold = game_map.get('starting_gold', 0)
        if type(self.starting_gold) is not int or self.starting_gold < 0:
            raise ValueError('starting_gold must be a nonnegative integer')
        self.starting_gold = min(self.starting_gold, MAX_RESOURCE)
        # A scenario player's BANK also holds mana (R infernal, Y life, E death, W runes, B grove).
        starting_mana = game_map.get('starting_mana', {})
        if (not isinstance(starting_mana, dict) or any(kind not in MANA_KINDS for kind in starting_mana)
                or any(type(amount) is not int or amount < 0 for amount in starting_mana.values())):
            raise ValueError('starting_mana must map mana kinds to nonnegative integers')
        self.starting_mana = tuple(min(starting_mana.get(kind, 0), MAX_RESOURCE) for kind in MANA_KINDS)
        # Removed with Empire-Alliance War; refuse instead of silently changing the loot.
        if 'enemy_loot' in game_map:
            raise ValueError('enemy_loot is no longer supported')
        self.combat_rules_version = game_map.get('combat_rules_version', 1)
        if type(self.combat_rules_version) is not int or self.combat_rules_version not in (1, 2):
            raise ValueError('Unsupported combat_rules_version')
        self.basic_combat = self.combat_rules_version == 2
        self.has_copies = False
        self.has_summons = False
        self.has_wolf_lord = False
        self.has_shatterers = False
        self.has_witches = False
        self.has_wights = False
        self.has_centaurs = False
        self.has_double_strike = False
        self.has_poisoners = False
        self.has_paralysis = False
        self.has_secondary_paralysis = False
        self.has_weakening = False
        self.has_fear = False
        self.has_secondary_fear = False
        fear_teams = game_map.get('fear_paralysis_teams',[])
        if not isinstance(fear_teams,list) or any(t not in ('blue','red') for t in fear_teams):
            raise ValueError('fear_paralysis_teams must contain blue/red team names')
        self.fear_protected = jnp.array([('blue' in fear_teams)]*6+[('red' in fear_teams)]*6)
        self.has_leech = False
        self.has_cures = False
        self.has_powerups = False
        self.has_patriarchs = False
        self.has_alchemists = False
        self.has_hermits = False
        self.has_healer_wards = False
        self.has_water = False
        self.has_fire = False
        self.progression_enabled = bool(game_map.get('unit_progression', False))
        self.episode_reports = bool(episode_reports) and self.progression_enabled
        if self.progression_enabled and not self.basic_combat:
            raise ValueError('Unit progression requires basic combat')
        # Reference skill ladder: Pathfinding, Weapon Master, the lord's Might (docs/HERO_SKILLS.md).
        self.hero_skills = hero_skills.enabled(game_map)
        if self.hero_skills and not self.progression_enabled:
            raise ValueError('Hero skills require unit progression')
        self.capital_enabled = bool(game_map.get('capital_services', False))
        if self.capital_enabled and not self.progression_enabled:
            raise ValueError('Capital services require named units and progression')
        self.potions_enabled = 'initial_potions' in game_map
        if self.potions_enabled and not self.capital_enabled:
            raise ValueError('Map potions require the current capital/progression environment')
        self.has_potion_buffs = False
        if self.potions_enabled:
            self.potion_rules = PotionRules(game_map)
            self.has_potion_buffs = self.potion_rules.has_buffs
        self.chests_enabled = 'chests' in game_map
        self.items_enabled = 'initial_items' in game_map or any(
            c.get('items') for c in game_map.get('chests',[])+game_map.get('ruins',[])+game_map.get('cities',[])
            + game_map.get('day_items',[]) + merchants(game_map) if isinstance(c,dict))
        if self.items_enabled:
            if not self.potions_enabled:
                raise ValueError('Equipment requires the current map inventory/progression environment')
            self.item_rules = ItemRules(game_map)
        if self.chests_enabled:
            if not self.potions_enabled:
                raise ValueError('Chests require the map potion inventory')
            self.chest_rules = ChestRules(game_map,self.potion_rules,
                                         self.item_rules if self.items_enabled else None)
        self.num_actions = (POTION_START+self.potion_rules.action_count if self.potions_enabled else CAPITAL_ACTIONS if self.capital_enabled
                            else BASE_ACTIONS if self.basic_combat else BUILD_START)
        if self.basic_combat:
            self.construction = BuildingRules(game_map)
            penalty = game_map.get("rest_penalty_per_point", .001)
            if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty < 0:
                raise ValueError("rest_penalty_per_point must be finite and nonnegative")
            self.rest_penalty_per_point = float(penalty)
        self.random_size = 36 if self.basic_combat else 13
        self.mask_walls = True
        self.hero_count = int(game_map.get('hero_units', 6))
        self.hero_hp = int(game_map.get('hero_hp', 45))
        self.damage = int(game_map.get('archer_damage', 25))
        self.accuracy = float(game_map.get('archer_accuracy', .8))
        # Missing role fields preserve the rules of saved all-archer maps.
        self.mage_slot = game_map.get('hero_mage_slot', -1)
        self.mage_damage = game_map.get('mage_damage', 20)
        if (type(self.mage_slot) is not int
                or not (-1 <= self.mage_slot < self.hero_count)
                or type(self.mage_damage) is not int or self.mage_damage <= 0):
            raise ValueError('Invalid mage battle configuration')
        self.warrior_slot = game_map.get('hero_warrior_slot', -1)
        self.warrior_hp = game_map.get('warrior_hp', 100)
        self.warrior_damage = game_map.get('warrior_damage', 25)
        self.warrior_accuracy = game_map.get('warrior_accuracy', .8)
        self.warrior_initiative = game_map.get('warrior_initiative', 50)
        if (type(self.warrior_slot) is not int
                or not (-1 <= self.warrior_slot < self.hero_count)
                or (self.warrior_slot >= 0 and self.warrior_slot == self.mage_slot)
                or any(type(value) is not int or value <= 0 for value in
                       (self.warrior_hp, self.warrior_damage, self.warrior_initiative))
                or type(self.warrior_accuracy) not in (int, float)
                or not 0 <= self.warrior_accuracy <= 1):
            raise ValueError('Invalid warrior battle configuration')
        self.max_rounds = int(game_map.get('battle_max_rounds', 75))
        self.round_limit_penalty = game_map.get('battle_round_limit_penalty', 1.)
        if (type(self.round_limit_penalty) not in (int, float)
                or not math.isfinite(self.round_limit_penalty) or self.round_limit_penalty < 0):
            raise ValueError('battle_round_limit_penalty must be finite and nonnegative')
        self.episode_timeout_penalty = game_map.get('episode_timeout_penalty', 0.)
        if (type(self.episode_timeout_penalty) not in (int, float)
                or not math.isfinite(self.episode_timeout_penalty) or self.episode_timeout_penalty < 0):
            raise ValueError('episode_timeout_penalty must be finite and nonnegative')
        counts, health = game_map['enemy_units'], game_map['enemy_hp']
        if not (1 <= self.hero_count <= 6 and self.hero_hp > 0 and self.damage > 0
                and 0 <= self.accuracy <= 1 and self.max_rounds > 0):
            raise ValueError('Invalid archer battle configuration')
        if len(counts) != self.num_opponents or len(health) != self.num_opponents:
            raise ValueError('Every enemy needs troop count and hit points')
        for count, hp in zip(counts, health):
            if not (type(count) is int and type(hp) is int and 1 <= count <= 6 and hp > 0):
                raise ValueError('Invalid enemy count or HP')
            if not self.basic_combat and not (count <= self.hero_count and hp <= self.hero_hp
                    and (count < self.hero_count or hp < self.hero_hp)):
                raise ValueError('Every enemy must be strictly weaker in count or HP')
        enemy_roles = game_map.get('enemy_warrior_slots', [-1] * self.num_opponents)
        if (not isinstance(enemy_roles, list) or len(enemy_roles) != self.num_opponents
                or any(type(slot) is not int or not -1 <= slot < count
                       for slot, count in zip(enemy_roles, counts))):
            raise ValueError('Enemy warrior slots must identify an existing unit or -1 per squad')
        self.enemy_warrior_slots = jnp.asarray(enemy_roles, jnp.int32)
        self.has_enemy_warriors = any(slot >= 0 for slot in enemy_roles)
        self.enemy_counts = jnp.asarray(counts, jnp.int32)
        self.enemy_health = jnp.asarray(health, jnp.int32)
        self.slots = jnp.arange(12)
        self.hero_full = jnp.where(jnp.arange(6) < self.hero_count, self.hero_hp, 0)
        if self.warrior_slot >= 0:
            self.hero_full = self.hero_full.at[self.warrior_slot].set(self.warrior_hp)
            # Front slots 0..2, rear slots 3..5, ordered top to bottom.
            # First nonempty tier: near front, far front, near rear, far rear.
            enemy_slots = jnp.arange(6)
            near = jnp.abs(enemy_slots % 3 - self.warrior_slot % 3) <= 1
            self.melee_tiers = (enemy_slots // 3) * 2 + (~near).astype(jnp.int32)
            self.initiative_scale = jnp.ones(12).at[self.warrior_slot].set(
                self.warrior_initiative / 60)
        if self.has_enemy_warriors:
            unit_slots = jnp.arange(6)
            near = jnp.abs(unit_slots[None, :] % 3 - unit_slots[:, None] % 3) <= 1
            self.enemy_melee_tiers = (unit_slots[None, :] // 3) * 2 + (~near).astype(jnp.int32)
            warriors = (self.slots[None, :] == self.warrior_slot) | (
                (self.slots[None, :] >= 6)
                & (self.slots[None, :] - 6 == self.enemy_warrior_slots[:, None]))
            self.squad_initiative_scale = jnp.where(warriors, self.warrior_initiative / 60, 1.)
        if self.basic_combat:
            (self.stats_table, self.damage_ids, self.armor_ids,
             self.damage_rolls, self.priority_scale, self.combat_traits, self.combat_info,
             self.has_protections) = build_combat_tables(game_map)
            self.shatter_armor_ids = jnp.array(self.combat_info.pop('_shatter_armor_ids'), jnp.int32)
            profiles = [self.combat_info['heroes']+squad for squad in self.combat_info['enemies']]
            self.damage_caps = jnp.array([[p['damage_cap'] if p else 300 for p in squad] for squad in profiles], jnp.float32)
            self.size_table = jnp.array([[p['size'] if p else 0 for p in squad] for squad in profiles], jnp.int32)
            self.mass_healers = jnp.array([[bool(p and p['mass_support']) for p in squad] for squad in profiles])
            self.mass_cures = jnp.array([[bool(p and p['unit_type'] == 'Profit' and p['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess')) for p in squad] for squad in profiles])
            power_types = {'Travnitsa':1.25,'Novice':1.5,'Dwarfdruid':1.75,'Arhidruid':2.}
            self.power_factors = jnp.array([[power_types.get(p['unit_type'],0.) if p else 0. for p in squad] for squad in profiles])
            self.power_cures = jnp.array([[bool(p and p['unit_type'] in ('Dwarfdruid','Arhidruid')) for p in squad] for squad in profiles])
            self.hermits = jnp.array([[bool(p and p['unit_type'] == 'Hermit') for p in squad] for squad in profiles])
            self.has_hermits = any(p and p['unit_type'] == 'Hermit' for squad in profiles for p in squad)
            self.alchemists = jnp.array([[bool(p and p['unit_type'] == 'Alchemist') for p in squad] for squad in profiles])
            self.has_alchemists = any(p and p['unit_type'] == 'Alchemist' for squad in profiles for p in squad)
            self.patriarchs = jnp.array([[bool(p and p['unit_type'] == 'Patriach') for p in squad] for squad in profiles])
            self.has_patriarchs = any(p and p['unit_type'] == 'Patriach' for squad in profiles for p in squad)
            self.has_powerups = any(p and p['unit_type'] in power_types for squad in profiles for p in squad)
            self.healer_ward_elements = jnp.array([[p['ward_bits'] if p else 0 for p in squad] for squad in profiles],jnp.uint32)
            self.has_healer_wards = any(p and p['ward_bits'] for squad in profiles for p in squad)
            self.has_protections |= self.has_healer_wards
            self.has_healers = any(p and p['role'] == 'healer' for squad in profiles for p in squad)
            self.has_cures = any(p and (p['unit_type'] in ('Patriach','Dwarfdruid','Arhidruid') or (p['unit_type'] == 'Profit' and p['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess'))) for squad in profiles for p in squad)
            if not self.progression_enabled and any(p and p['unit_type'] == 'Wight' for squad in profiles for p in squad):
                raise ValueError('Wight requires named unit progression for its temporary forms')
            self.aoe_accuracy_falloff = jnp.array([[bool(p and p['aoe_accuracy_falloff']) for p in squad]
                                                  for squad in profiles])
            self.has_aoe_accuracy_falloff = any(p and p['aoe_accuracy_falloff'] for squad in profiles for p in squad)
            self.wolf_lord = jnp.array([[bool(p and p['unit_type'] == 'Wolf Lord') for p in squad] for squad in profiles])
            self.has_wolf_lord = any(p and p['unit_type'] == 'Wolf Lord' for squad in profiles for p in squad)
            self.shatterers = jnp.array([[bool(p and p['unit_type'] in ('Teurg','Aleman')) for p in squad] for squad in profiles])
            self.has_shatterers = any(p and p['unit_type'] in ('Teurg','Aleman') for squad in profiles for p in squad)
            self.witches = jnp.array([[bool(p and p['unit_type'] in ('Witch','Succub')) for p in squad] for squad in profiles])
            self.has_witches = any(p and p['unit_type'] in ('Witch','Succub') for squad in profiles for p in squad)
            self.centaurs = jnp.array([[bool(p and p['unit_type'] == 'Centaur Savage') for p in squad] for squad in profiles])
            self.has_centaurs = any(p and p['unit_type'] == 'Centaur Savage' for squad in profiles for p in squad)
            self.double_strike = jnp.array([[bool(p and p['unit_type'] in ('Demon', 'Elfarcher')) for p in squad] for squad in profiles])
            self.has_double_strike = any(p and p['unit_type'] in ('Demon', 'Elfarcher') for squad in profiles for p in squad)
            self.cached_poisoners = jnp.array([[bool(p and (p['name'] == 'Ниддог' or p['unit_type'] in ('Dregazul','Spider'))) for p in squad] for squad in profiles])
            self.poisoners = jnp.array([[bool(p and ((p['name'] == 'Ниддог' or p['unit_type'] in ('Dregazul','Spider')) or p['unit_type'] in ('Death','Dead dragon'))) for p in squad] for squad in profiles])
            self.has_poisoners = any(p and ((p['name'] == 'Ниддог' or p['unit_type'] in ('Dregazul','Spider')) or p['unit_type'] in ('Death','Dead dragon')) for squad in profiles for p in squad)
            self.cached_water = jnp.array([[bool(p and p['unit_type'] == 'Ismir son') for p in squad] for squad in profiles])
            self.water_casters = jnp.array([[bool(p and p['unit_type'] in ('Sentry','Ismir son','Drulliaan')) for p in squad] for squad in profiles])
            self.has_water = any(p and p['unit_type'] in ('Sentry','Ismir son','Drulliaan') for squad in profiles for p in squad)
            self.secondary_fear = jnp.array([[bool(p and p['unit_type'] == 'Shamanka') for p in squad] for squad in profiles])
            self.has_secondary_fear = any(p and p['unit_type'] == 'Shamanka' for squad in profiles for p in squad)
            self.fear_casters = jnp.array([[bool(p and p['unit_type'] == 'Baroness') for p in squad] for squad in profiles])
            self.has_fear = any(p and p['unit_type'] in ('Baroness','Shamanka') for squad in profiles for p in squad)
            self.weakeners = jnp.array([[bool(p and p['unit_type'] == 'Tiamat') for p in squad] for squad in profiles])
            self.has_weakening = any(p and p['unit_type'] == 'Tiamat' for squad in profiles for p in squad)
            self.secondary_paralysis_modes = jnp.array([[1 if p and (p['name'] == 'Русалка' or p['unit_type'] == 'Abyss Devil') else 2 if p and p['unit_type'] in ('Betrezen','Uter','Uter Demon','Abyss Devil') else 0 for p in squad] for squad in profiles],jnp.int32)
            self.has_secondary_paralysis = any(p and p['unit_type'] in ('Betrezen','Uter','Uter Demon','Abyss Devil') for squad in profiles for p in squad)
            self.ghost_modes = jnp.array([[2 if p and p['name'] == 'Тёмный эльф призрак' else 1 if p and p['unit_type'] in ('Ghost','Shadow','Incub') else 0 for p in squad] for squad in profiles],jnp.int32)
            self.has_paralysis = self.has_fear or self.has_secondary_paralysis or any(p and p['unit_type'] in ('Ghost','Shadow','Incub') for squad in profiles for p in squad)
            self.leech_modes = jnp.array([[2 if p and p['unit_type'] in ('Bone Lord','Highvampire') else 1 if p and p['unit_type'] in ('Dregazul','Vampire') else 0 for p in squad] for squad in profiles],jnp.int32)
            self.has_leech = any(p and p['unit_type'] in ('Bone Lord','Dregazul','Vampire','Highvampire') for squad in profiles for p in squad)
            self.cached_fire = jnp.array([[bool(p and p['unit_type'] == 'Lord') for p in squad] for squad in profiles])
            self.fire_casters = jnp.array([[bool(p and p['unit_type'] in ('Watcher', 'Lord', 'Gumtic')) for p in squad] for squad in profiles])
            self.has_fire = any(p and p['unit_type'] in ('Watcher', 'Lord', 'Gumtic') for squad in profiles for p in squad)
            self.secondary_values = jnp.array([[[p['secondary_damage'],p['secondary_accuracy']] if p else [0,0] for p in squad] for squad in profiles], jnp.float32)
            self.secondary_sources = jnp.array([[p['secondary_source'] if p else 0 for p in squad] for squad in profiles], jnp.uint32)
            self.capital_guards = jnp.array([[bool(p and p['capital_guard']) for p in squad] for squad in profiles])
            unit_slots = jnp.arange(6)
            near = jnp.abs(unit_slots[None, :] % 3 - unit_slots[:, None] % 3) <= 1
            self.all_melee_tiers = (unit_slots[None, :] // 3) * 2 + (~near).astype(jnp.int32)
            self.hero_full = self.stats_table[0, :6, HP].astype(jnp.int32)
        if self.progression_enabled:
            self.progression = ProgressionRules(game_map, self.construction)
            self.has_healers = self.progression.has_healers
            self.has_copies = self.progression.has_copies
            self.has_summons = self.progression.has_summons
            self.has_cures = self.progression.has_cures
            self.has_powerups = self.progression.has_powerups
            self.has_patriarchs = self.progression.has_patriarchs
            self.has_alchemists = self.progression.has_alchemists
            self.has_hermits = self.progression.has_hermits
            self.has_healer_wards = self.progression.has_healer_wards
            self.has_protections |= self.has_healer_wards
            self.combat_info['catalogue'] = self.progression.metadata
            self.has_protections = self.progression.has_protections or self.has_healer_wards
            self.has_aoe_accuracy_falloff = self.progression.has_aoe_accuracy_falloff
            self.has_wolf_lord = self.progression.has_wolf_lord
            self.has_shatterers = self.progression.has_shatterers
            self.has_witches = self.progression.has_witches
            self.has_wights = self.progression.has_wights
            self.has_centaurs = self.progression.has_centaurs
            self.has_double_strike = self.progression.has_double_strike
            self.has_poisoners = self.progression.has_poisoners
            self.has_paralysis = self.progression.has_paralysis
            self.has_secondary_paralysis = self.progression.has_secondary_paralysis
            self.has_weakening = self.progression.has_weakening
            self.has_fear = self.progression.has_fear
            self.has_secondary_fear = self.progression.has_secondary_fear
            self.has_leech = self.progression.has_leech
            self.has_water = self.progression.has_water
            self.has_fire = self.progression.has_fire
            self.priority_scale = 110.
        if self.basic_combat:
            owner = self.progression if self.progression_enabled else self
            rows = owner.rows if self.progression_enabled else [p for squad in profiles for p in squad]
            native = effect_tables(rows)
            for name, values in native.items():
                array = jnp.asarray(values)
                if not self.progression_enabled:
                    array = array.reshape((len(profiles),12)+array.shape[1:])
                setattr(owner,name,array)
            used = owner.used_ids if self.progression_enabled else range(len(rows))
            self.has_poisoners = any(native['poisoners'][i] for i in used)
            self.has_water = any(native['water_casters'][i] for i in used)
            self.has_fire = any(native['fire_casters'][i] for i in used)
            self.has_secondary_paralysis = any(native['secondary_paralysis_modes'][i] for i in used)
            self.has_paralysis = self.has_fear or self.has_secondary_paralysis or any(native['ghost_modes'][i] for i in used)
        if self.progression_enabled:
            self.equipment_heroes = jnp.array([bool(p and p['hero']) for p in self.progression.metadata])
            # Reference _hero_base_moves: a hero's own allowance (MOVE in Gunits.dbf), and
            # Pathfinding adds round(20%) of it.
            moves = [int(r.get('movement', 0)) if r.get('hero') else 0 for r in self.progression.rows]
            self.hero_moves = jnp.array(moves, jnp.int32)
            self.pathfinding_points = jnp.array(
                [round(m*hero_skills.PATHFINDING_PERCENT/100) for m in moves], jnp.int32)
            starting = [moves[i] for i in self.progression.initial_ids[:6].tolist() if moves[i]]
            self.base_movement = max(starting, default=MAX_MOVEMENT_POINTS)  # as movement_cap
        else:
            self.base_movement = MAX_MOVEMENT_POINTS
        if self.items_enabled:
            statuses = {p['status'] for _,p in self.item_rules.status_items}
            self.has_witches |= 'imp' in statuses
            self.has_wights |= 'decay' in statuses
            self.has_poisoners |= 'poison' in statuses
            self.has_paralysis |= 'paralysis' in statuses
            self.has_protections = True
        self.battle_items_enabled = battle_items_enabled(game_map)
        self.has_item_summons = self.has_item_weakening = self.has_lycanthropy = False
        if self.battle_items_enabled:
            if not self.potions_enabled or not self.progression_enabled:
                raise ValueError('Battle items require the map potion inventory and unit progression')
            rules = self.battle_item_rules = BattleItemRules(game_map, self.potion_rules,
                self.item_rules if self.items_enabled else None, self.progression, self.construction)
            kinds, dots = rules.kinds, rules.dot_kinds
            # Item effects reuse the native status machinery, compiled only when present.
            self.has_poisoners |= battle_items.POISON in dots
            self.has_fire |= battle_items.BURN in dots
            self.has_water |= battle_items.WATER in dots
            self.has_paralysis |= bool(kinds & {battle_items.PARALYSIS, battle_items.FEAR})
            self.has_witches |= battle_items.TRANSFORM in kinds
            self.has_wights |= battle_items.LYCANTHROPY in kinds
            self.has_lycanthropy = battle_items.LYCANTHROPY in kinds
            self.has_powerups |= battle_items.BUFF in kinds
            self.has_weakening |= battle_items.WEAKEN in kinds
            self.has_item_weakening = battle_items.WEAKEN in kinds
            self.has_item_summons = battle_items.SUMMON in kinds
            if battle_items.REVIVE in kinds:
                self.has_cures = True
            self.has_protections = True
        if self.has_wolf_lord or self.has_copies or self.has_summons:
            self.num_actions = POTION_START+(self.potion_rules.action_count if self.potions_enabled else 0)
        if self.has_witches:
            self.random_size = 37
        if self.has_poisoners or self.has_water or self.has_fire or self.has_wights:
            self.random_size = 67
        if self.has_paralysis:
            self.random_size = 68
        if self.has_copies or self.has_summons:
            self.random_size = 80
        if self.items_enabled and self.item_rules.status_items:
            self.random_size = max(self.random_size,80)+4
        if self.has_poisoners or self.has_water or self.has_fire or self.has_summons:
            # Independent removal draws for 12 targets x 3 DOTs x two rounds.
            self.random_size = max(self.random_size,156)
        if self.has_hermits or self.has_secondary_fear or self.has_weakening:
            self.random_size = max(self.random_size,67)
        if self.capital_enabled:
            self.capital = CapitalRules(self.progression, self.construction,
                                        game_map.get('capital_position', game_map['agent_position']))
        self.sites_enabled = sites_present(game_map)
        if self.sites_enabled:
            if not self.potions_enabled or not self.progression_enabled:
                raise ValueError('Sites require potion inventory and unit progression')
            self.site_rules = SiteRules(game_map, self.potion_rules,
                self.item_rules if self.items_enabled else None, self.progression, self.num_actions)
            self.num_actions = self.site_rules.end
        self.recruitment_enabled = bool(game_map.get('faction_recruitment', False))
        if mercenary_camps(game_map) and not self.recruitment_enabled:
            raise ValueError('Mercenary camps require recruitment')
        if self.recruitment_enabled:
            if not self.capital_enabled:
                raise ValueError('Recruitment requires capital services and progression')
            self.recruitment = RecruitmentRules(game_map, self.progression, self.construction, self.num_actions,
                self.site_rules if self.sites_enabled else None)
            self.num_actions = self.recruitment.end
        self.ruins_enabled = bool(game_map.get('ruins',[]))
        if self.ruins_enabled:
            if not self.potions_enabled or not self.progression_enabled:
                raise ValueError('Ruins require potion inventory and unit progression')
            self.ruins = RuinRules(game_map,self.potion_rules,self.item_rules if self.items_enabled else None,
                self.progression,self.site_rules if self.sites_enabled else None)
        self.territory_enabled = bool(game_map.get('territory',False))
        if self.territory_enabled:
            if not self.recruitment_enabled:
                raise ValueError('Territory requires progression and recruitment')
            self.territory = TerritoryRules(game_map,self.progression,self.num_actions,
                self.site_rules if self.sites_enabled else None,self.ruins if self.ruins_enabled else None)
            self.num_actions = self.territory.end
            self.capital.territory = self.territory
            self.recruitment.territory = self.territory
        self.scenario = ScenarioRules(game_map, self.territory if self.territory_enabled else None)
        if game_map.get('scripted_bot') and not (self.territory_enabled and casting_enabled(game_map)):
            raise ValueError('Scripted bot requires territory, progression and persistent spell wounds')
        self.bot = ScriptedBotRules(game_map, self.territory if self.territory_enabled else None,
                                    self.progression if self.progression_enabled else None)
        self.scripted_bot_objective = game_map.get('campaign_objective') == 'scripted_bot'
        if self.scripted_bot_objective and not self.bot.enabled:
            raise ValueError('The scripted_bot objective requires an enabled bot')
        # The Bethrezen Capital objective was removed. Refuse saved maps that still
        # use it instead of silently making their capital guardians optional.
        if game_map.get('campaign_objective') == 'capital' or any(k in game_map for k in (
                'objective_capital', 'capital_objective_reward', 'capital_survival_round_reward',
                'capital_guardian_reached_reward', 'capital_guardian_damage_rewards', 'reference_timeout_penalty')):
            raise ValueError('The removed capital campaign objective and its reward fields are no longer supported')
        # Capital guards stay on the map but are never required for victory. With
        # objective cities only their capture counts, so no stack has to fall.
        self.goal_enemies = None
        self.objective_enemy = game_map.get('objective_enemy')
        self.position_objective = game_map.get('campaign_objective') == 'reach_position'
        self.objective_position_reward = game_map.get('objective_position_reward', 8.)
        if not self.position_objective and any(k in game_map for k in (
                'objective_position', 'objective_radius', 'objective_position_reward', 'objective_hero',
                'objective_after_event')):
            raise ValueError('Position objective fields require campaign_objective=reach_position')
        if self.position_objective:
            point = game_map.get('objective_position')
            if (not self.progression_enabled or not self.territory_enabled or self.objective_enemy is not None
                    or self.bot.enabled or self.scenario.enabled):
                raise ValueError('Reach-position campaigns require progression and territory without another objective')
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(type(v) is not int or not 0 < v < self.size-1 for v in point)
                    or tuple(point) in self.territory.obstacle_cells
                    or list(point) == list(game_map['agent_position'])):
                raise ValueError('Objective position must be a passable interior cell distinct from the start')
            if (type(self.objective_position_reward) not in (int, float)
                    or not 0 <= self.objective_position_reward < float('inf')):
                raise ValueError('Position objective reward must be finite and nonnegative')
            self.objective_position = jnp.array(point, jnp.int32)
            self.objective_radius = game_map.get('objective_radius', 0)
            if (type(self.objective_radius) is not int or not 0 <= self.objective_radius < self.size
                    or max(abs(a-b) for a, b in zip(point, game_map['agent_position'])) <= self.objective_radius):
                raise ValueError('Objective radius must be nonnegative and exclude the initial position')
            hero_key = game_map.get('objective_hero')
            eligible = [bool(r.get('hero')) and (hero_key is None or r.get('key') == hero_key)
                        for r in self.progression.rows]
            if not any(eligible[int(i)] for i in self.progression.initial_ids[:6]):
                raise ValueError('Position objective requires its hero in the starting party')
            self.objective_heroes = jnp.array(eligible, bool)
        self.objective_reward_hero_only = game_map.get('objective_reward_hero_only', False)
        if type(self.objective_reward_hero_only) is not bool:
            raise ValueError('objective_reward_hero_only must be true or false')
        if self.objective_enemy is not None:
            if type(self.objective_enemy) is not int or not 0 <= self.objective_enemy < len(self.opponent_positions):
                raise ValueError('objective_enemy must be a valid zero-based enemy index')
            self.goal_enemies = jnp.arange(len(self.opponent_positions)) == self.objective_enemy
        elif self.objective_reward_hero_only:
            raise ValueError('objective_reward_hero_only requires objective_enemy')
        elif game_map.get('campaign_objective') == 'waves':
            self.goal_enemies = self.scenario.scheduled
        elif self.territory_enabled and self.territory.objective_cities:
            self.goal_enemies = jnp.zeros(self.territory.enemy_count, bool)
        elif self.territory_enabled and self.territory.has_capital_guards:
            self.goal_enemies = ~self.territory.capital_guards
        # User request 2026-10-17: listed armies stay on the map, like capital
        # guards, but clearing the map does not require defeating them.
        optional = game_map.get('optional_enemies', [])
        count = len(self.opponent_positions)
        if (not isinstance(optional, list) or len(set(optional)) != len(optional)
                or any(type(i) is not int or not 0 <= i < count for i in optional)):
            raise ValueError('optional_enemies must list distinct valid enemy indices')
        if optional:
            if (self.objective_enemy is not None or self.position_objective or self.scripted_bot_objective
                    or game_map.get('campaign_objective') == 'waves'
                    or (self.territory_enabled and self.territory.objective_cities)):
                raise ValueError('optional_enemies apply only to maps won by defeating every army')
            required = jnp.ones(count, bool) if self.goal_enemies is None else self.goal_enemies
            self.goal_enemies = required.at[jnp.array(optional)].set(False)
            if not bool(jnp.any(self.goal_enemies)):
                raise ValueError('At least one enemy army must remain a victory goal')
        self.reference_preserve_target_reward = game_map.get('reference_preserve_target_reward', False)
        if type(self.reference_preserve_target_reward) is not bool:
            raise ValueError('reference_preserve_target_reward must be true or false')
        if self.reference_preserve_target_reward and (
                not self.reference_rewards or self.objective_enemy is None):
            raise ValueError('reference_preserve_target_reward requires reference rewards and objective_enemy')
        for obsolete in ('army_engage_hp_reward', 'army_engage_level_reward', 'objective_engage_hp_reward'):
            if obsolete in game_map:
                raise ValueError(f'{obsolete} was replaced by army_victory_level_reward; '
                                 'old engagement-reward maps must not silently use victory rewards')
        self.army_victory_level_reward = game_map.get('army_victory_level_reward', 0.)
        if (type(self.army_victory_level_reward) not in (int, float)
                or not 0 <= self.army_victory_level_reward < float('inf')):
            raise ValueError('army_victory_level_reward must be a finite nonnegative number')
        if self.army_victory_level_reward and not self.progression_enabled:
            raise ValueError('army_victory_level_reward requires unit progression')
        self.army_victory_level_objective_only = game_map.get('army_victory_level_objective_only', False)
        if type(self.army_victory_level_objective_only) is not bool:
            raise ValueError('army_victory_level_objective_only must be true or false')
        if self.army_victory_level_objective_only and (
                self.objective_enemy is None or not self.army_victory_level_reward):
            raise ValueError('army_victory_level_objective_only requires objective_enemy and a positive level reward')
        self.army_victory_level_targets = (jnp.arange(self.num_opponents) == self.objective_enemy
            if self.army_victory_level_objective_only else jnp.ones(self.num_opponents, bool))
        self.capital_training = CapitalTrainingRules(self, game_map)
        self.siege_rewards = game_map.get('siege_rewards')
        if self.siege_rewards is not None:
            if (not self.reference_rewards or game_map.get('campaign_objective') != 'target_enemy'
                    or len(self.scenario.events) != 1
                    or self.scenario.events[0]['enemy'] != self.objective_enemy):
                raise ValueError('Siege rewards require one scheduled objective army and reference rewards')
            if 'spawn_hp' in self.siege_rewards:
                raise ValueError('siege_rewards.spawn_hp was replaced by siege_rewards.spawn_level; '
                                 'old HP-reward maps must not silently use level rewards')
            for key in ('victory_multiplier', 'unit_kill', 'spawn_level'):
                value = self.siege_rewards.get(key)
                if type(value) not in (int, float) or not 0 <= value < float('inf'):
                    raise ValueError(f'siege_rewards.{key} must be a finite nonnegative number')
            if self.siege_rewards['spawn_level'] and self.army_victory_level_reward:
                raise ValueError('siege_rewards.spawn_level and army_victory_level_reward are mutually exclusive')
        self.spell_research_enabled = bool(game_map.get('spell_research', False))
        if self.spell_research_enabled:
            if not self.territory_enabled:
                raise ValueError('Spell research requires territory and mana resources')
            self.spell_research = SpellResearchRules(game_map, self.construction, self.num_actions)
            self.num_actions = self.spell_research.end
        self.spell_casting_enabled = casting_enabled(game_map)
        self.has_spell_mods = self.has_spell_wards = False
        if self.spell_casting_enabled:
            if not self.spell_research_enabled:
                raise ValueError('Spell casting requires spell research')
            self.casting = SpellCastRules(game_map, self.spell_research, self.construction, self.progression,
                self.territory, self.ruins if self.ruins_enabled else None, self.num_actions,
                self.item_rules if self.items_enabled else None)
            self.num_actions = self.casting.end
            self.has_spell_mods = self.casting.has_mods
            self.has_spell_wards = self.casting.has_wards
            self.has_protections |= self.has_spell_wards
        if self.battle_items_enabled:
            self.battle_item_rules.place(self.num_actions)
            self.num_actions = self.battle_item_rules.end
        self.unit_swaps_enabled = unit_swaps.enabled(game_map)
        if self.unit_swaps_enabled:
            if not self.progression_enabled:
                raise ValueError('Unit swaps require unit progression')
            self.unit_swaps = UnitSwapRules(game_map, self.progression, self.num_actions)
            self.num_actions = self.unit_swaps.end
        if self.basic_combat:
            self.terrain = TerrainRules(game_map, self.progression if self.progression_enabled else None)
        self.script_events = ScriptEventRules(game_map, self)
        self.target_rewards = TargetRewardRules(self, game_map)
        self.reference_enemy_count = self.num_opponents - (
            len(game_map.get('scenario_event_enemies', [])) if not self.script_events.enabled else 0)
        self.restored_hp = jnp.concatenate((self.hero_full, jnp.zeros(6, jnp.int32)))
        # User-defined prototype rule: ceil(10% max HP) once per strategic rest.
        self.rest_healing = jnp.concatenate(((self.hero_full + 9) // 10, jnp.zeros(6, jnp.int32)))
        self.observation_version = int(game_map.get('battle_observation_version', 1))
        if (self.bot.enabled or (self.territory_enabled and self.territory.contested)) and self.observation_version < 45:
            raise ValueError('Scripted bot and contested territory require observation version 45')
        if self.basic_combat and self.observation_version not in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45):
            raise ValueError('Combat rules version 2 requires observation version 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 or 18')
        if not self.basic_combat and self.observation_version in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45):
            raise ValueError('Observation versions 7вЂ“18 require combat rules version 2')
        if (self.warrior_slot >= 0 or self.has_enemy_warriors) and self.observation_version not in (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45):
            raise ValueError('Warrior maps require battle_observation_version 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 or 18')
        self.observation_size = (4 + 5 * self.num_opponents + (141 if self.basic_combat else 48) + 6 if self.observation_version in (2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)
                                 else self.observation_size + 2 * self.num_opponents + 12 * 8 + 6)
        self.observation_size += self.scenario.observation_size
        self.observation_size += self.script_events.observation_size
        self.observation_size += self.bot.observation_size
        self.observation_size += 4 if self.position_objective else 0
        self.observation_size += self.capital_training.observation_size
        self.observation_size += self.num_opponents+1 if self.army_victory_level_reward else 0

        if self.observation_version in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45):
            self.observation_size += 48
        if self.progression_enabled != (self.observation_version in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)):
            raise ValueError('Unit progression requires observation version 9, 10, 11, 12, 13, 14, 15, 16, 17 or 18')
        if self.progression_enabled:
            self.observation_size += 60 + BLOCKED_XP_FEATURES
            if self.observation_version < 25:
                self.observation_size += 12
        if self.observation_version >= 10:
            self.observation_size += 12
        if self.capital_enabled != (self.observation_version in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)):
            raise ValueError("Capital services require observation version 11, 12, 13, 14, 15, 16, 17 or 18")
        if self.capital_enabled:
            self.observation_size += 17
        if self.potions_enabled != (self.observation_version in (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)):
            raise ValueError("Map potions require observation version 12, 13, 14, 15, 16, 17 or 18")
        if self.potions_enabled:
            self.observation_size += self.potion_rules.count+(36 if self.has_potion_buffs else 0)
        if self.chests_enabled != (self.observation_version in (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)):
            raise ValueError('Map chests require observation version 13, 14, 15, 16, 17 or 18')
        if self.chests_enabled:
            self.observation_size += (1+self.chest_rules.features.shape[1])*self.chest_rules.count
        if self.items_enabled:
            if self.item_rules.charge_capacity and self.observation_version < 43:
                raise ValueError('Inventory talisman charges require observation version 43')
            self.observation_size += self.item_rules.observation_size
        if self.sites_enabled:
            if self.observation_version < 31:
                raise ValueError('Sites require observation version 31 or later')
            self.observation_size += self.site_rules.observation_size
        if self.recruitment_enabled:
            if self.observation_version < 33:
                raise ValueError('Recruitment requires observation version 33')
            self.observation_size += self.recruitment.observation_size
        if self.territory_enabled:
            if self.observation_version < 34:
                raise ValueError('Territory requires observation version 34')
            self.observation_size += self.territory.observation_size
        if self.ruins_enabled:
            if self.observation_version < 36:
                raise ValueError('Ruins require observation version 36')
            self.observation_size += self.ruins.observation_size
        if self.observation_version >= 37:
            self.observation_size += self.terrain.observation_size
        if self.observation_version >= 38:
            self.observation_size += 3  # ruler one-hot observation
        if self.spell_research_enabled:
            if self.observation_version < 38:
                raise ValueError('Spell research requires observation version 38')
            self.observation_size += self.spell_research.observation_size
        if self.spell_casting_enabled:
            if self.observation_version < 39:
                raise ValueError('Spell casting requires observation version 39')
            if self.casting.item_count and self.observation_version < 41:
                raise ValueError('Scrolls and staves require observation version 41')
            self.observation_size += self.casting.observation_size
        if self.battle_items_enabled:
            if self.observation_version < 40:
                raise ValueError('Battle items require observation version 40')
            self.observation_size += self.battle_item_rules.observation_size
        if self.unit_swaps_enabled:
            if self.observation_version < 42:
                raise ValueError('Unit swaps require observation version 42')
            self.observation_size += self.unit_swaps.observation_size
        if self.observation_version >= 14:
            self.observation_size += 12
        if self.observation_version >= 15:
            self.observation_size += 36
        if self.observation_version >= 16:
            self.observation_size += 12
        if self.observation_version >= 17:
            self.observation_size += 36
        if self.observation_version >= 18:
            self.observation_size += 36
        if self.observation_version >= 21:
            self.observation_size += 30
        if self.observation_version >= 22:
            self.observation_size += 13
        if self.observation_version >= 23:
            self.observation_size += 120
        if self.observation_version >= 24:
            self.observation_size += 72
        if self.observation_version >= 25:
            self.observation_size += 24
        if self.observation_version >= 26:
            self.observation_size += 24
        if self.observation_version >= 27:
            self.observation_size += 36
        if self.observation_version >= 28:
            self.observation_size += 48
        if self.observation_version >= 44:
            self.observation_size += 108
        if self.observation_version >= 29:
            self.observation_size += 24

    def reset(self, rng_key, env_params=None):
        """Create a fresh campaign state and FIRST timestep from the supplied PRNG key."""
        if not self.battle_mode:
            return super().reset(rng_key, env_params)
        visited = jnp.zeros(self.visited_words, jnp.uint32)
        if self.exploration_bonus:
            r, c = self.map_config['agent_position']
            cell = r * self.size + c
            visited = visited.at[cell // 32].set(jnp.uint32(1 << (cell % 32)))
        zero = jnp.int32(0)
        state = BattleState(
            bot_ids=self.bot.initial_ids if self.bot.enabled else jnp.zeros(0, jnp.int32),
            bot_mode=zero, bot_timer=zero, bot_target=jnp.int32(-1), bot_moves_spent=zero,
            bot_defeated_enemies=zero, bot_hero_victories=zero, bot_autobattle=jnp.bool_(False),
            opponent_claims=jnp.ones(len(self.territory.opponent_sources) if self.territory_enabled else 0, jnp.int32),
            territory_owners=self.territory.initial_owners if self.territory_enabled else jnp.zeros(0, jnp.int8),
            enemy_positions=self.opponent_positions,
            scenario_spawned=~self.scenario.scheduled,
            script_events_fired=jnp.zeros(self.script_events.count, bool),
            script_objective_ready=jnp.bool_(False),
            script_orders=self.script_events.initial_orders,
            pending_encounters=jnp.zeros(self.num_opponents, bool),
            battle_hero_fort_armor=zero,
            battle_fort_defender=zero,
            reference_visits=jnp.zeros(self.size*self.size if self.reference_rewards else 0, jnp.uint8),
            reference_previous=jnp.full(2, -1, jnp.int32),
            reference_city_reached=jnp.zeros(self.territory.count if self.territory_enabled else 0, bool),
            reference_defeated=jnp.zeros(self.num_opponents, bool),
            capital_defense_acquired=jnp.int32(0),
            capital_defense_peak=jnp.zeros(self.capital_training.count, jnp.int32),
            capital_rounds_rewarded=jnp.int32(0),
            capital_entry_strength=jnp.int32(-1),
            capital_guardian_bonus_paid=jnp.bool_(False),
            reference_hp_milestones=jnp.array([False, False]),
            reference_potion_pending=jnp.bool_(False),
            siege_rewarded_kills=jnp.uint32(0),
            army_victory_rewarded=jnp.zeros(self.num_opponents if self.army_victory_level_reward else 0, bool),
            army_battle_level_strength=jnp.int32(0),
            target_rewarded=jnp.zeros(self.target_rewards.count, bool),
            target_potential=jnp.float32(0.),
            native_ids=jnp.zeros(12,jnp.int32), native_levels=jnp.zeros(12,jnp.int32),
            native_xp=jnp.zeros(12,jnp.int32), copied=jnp.zeros(12,bool),
            copy_stats=jnp.zeros((12,5),jnp.float32), copy_traits=jnp.zeros((12,4),jnp.uint32),
            copy_secondary=jnp.zeros((12,2),jnp.float32),
            summon_owner=jnp.full(12,-1,jnp.int32), preparation=jnp.zeros(12,bool),
            summon_saved_hp=jnp.zeros(12,jnp.int32),
            fenrir=jnp.zeros(12, jnp.bool_),
            armor_shreds=jnp.zeros(12, jnp.int32),
            imp=jnp.zeros(12, jnp.bool_), imp_wards=jnp.zeros(12, jnp.uint32),
            activation_done=jnp.zeros(12, jnp.bool_),
            post_victory=jnp.int32(0),pending_healers=jnp.zeros(12,jnp.bool_),
            initiative_override=jnp.full(12,-1,jnp.int32),saved_initiative_override=jnp.full(12,-1,jnp.int32),
            slow_original=jnp.full(12,-1,jnp.int32),waited=jnp.zeros(12,bool),bonus_turns=jnp.zeros(12,jnp.int32),
            battle_revived=jnp.zeros(12,bool),battle_earned_xp=jnp.zeros(12,jnp.int32),
            primary_override=jnp.full(12,-1,jnp.int32),preweak_damage=jnp.full(12,-1,jnp.int32),
            powerup=jnp.zeros(12,bool),powerup_layered=jnp.zeros(12,bool),
            powerup_long=jnp.zeros(12,bool),saved_powerup_long=jnp.zeros(12,bool),
            effect_rounds=jnp.zeros((12,5),jnp.int32),disable_source=jnp.full(12,-1,jnp.int32),
            transform_long=jnp.zeros(12,bool),
            saved_primary_override=jnp.full(12,-1,jnp.int32),saved_preweak_damage=jnp.full(12,-1,jnp.int32),
            saved_powerup=jnp.zeros(12,bool),saved_powerup_layered=jnp.zeros(12,bool),
            healer_wards=jnp.zeros((12,4),jnp.uint32),ward_native_used=jnp.zeros(12,jnp.uint32),
            saved_healer_wards=jnp.zeros((12,4),jnp.uint32),saved_ward_native_used=jnp.zeros(12,jnp.uint32),
            feared=jnp.zeros(12,jnp.bool_),enemy_initial_hp=jnp.zeros(6,jnp.int32),
            enemy_damage_credit=jnp.zeros(6,jnp.int32),enemy_killed_credit=jnp.zeros(6,jnp.bool_),
            weakened=jnp.zeros(12,jnp.bool_),saved_weakened=jnp.zeros(12,jnp.bool_),
            item_weakened=jnp.zeros(12,jnp.bool_),saved_item_weakened=jnp.zeros(12,jnp.bool_),
            lycanthrope=jnp.zeros(12,jnp.bool_),
            paralyzed=jnp.zeros(12,jnp.bool_),long_paralyzed=jnp.zeros(12,jnp.bool_),
            decay_form=jnp.zeros(12, jnp.int32), decay_shreds=jnp.zeros(12, jnp.int32),
            position=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            number=jnp.int32(self.initial_number), alive=~self.scenario.scheduled,
            step_count=zero, done=jnp.bool_(False), won=jnp.bool_(False), lost=jnp.bool_(False),
            visited=visited, battle_key=rng_key, in_battle=jnp.bool_(False), enemy=jnp.int32(-1),
            origin=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            second_strike=jnp.bool_(False), poison_mode=jnp.zeros(12, jnp.int32),
            poison_damage=jnp.zeros(12, jnp.int32), poison_source=jnp.full(12, -1, jnp.int32),
            last_poison_damage=jnp.zeros(12, jnp.int32),
            water_mode=jnp.zeros(12,jnp.int32), water_damage=jnp.zeros(12,jnp.int32),
            water_source=jnp.full(12,-1,jnp.int32), last_water_damage=jnp.zeros(12,jnp.int32),
            burn_mode=jnp.zeros(12,jnp.int32),burn_damage=jnp.zeros(12,jnp.int32),
            burn_source=jnp.full(12,-1,jnp.int32),last_burn_damage=jnp.zeros(12,jnp.int32),
            hp=self.restored_hp, wards_used=jnp.zeros(12, jnp.uint32),
            last_immune=jnp.uint32(0), last_ward=jnp.uint32(0), priority=jnp.zeros(12), turn_phase=jnp.zeros(12, jnp.int32),
            defended=jnp.zeros(12, bool), retreating=jnp.zeros(12, bool), escaped=jnp.zeros(12, bool),
            actor=zero, round=zero, last_event=jnp.int32(MOVE), last_actor=jnp.int32(-1),
            last_target=jnp.int32(-1), last_damage=zero, battle_steps=zero,
            player_turns=zero, enemy_turns=zero,
            unit_ids=(self.progression.initial_ids if self.progression_enabled else jnp.zeros(0, jnp.int32)),
            unit_levels=(self.progression.initial_levels if self.progression_enabled else jnp.zeros(0, jnp.int32)),
            unit_xp=(self.progression.initial_xp if self.progression_enabled else jnp.zeros(0, jnp.int32)),
            enemy_progress=(self.progression.initial_enemies if self.progression_enabled else jnp.zeros((0, 6, 2), jnp.int32)),
            battle_xp=jnp.zeros(2, jnp.int32), last_xp=jnp.zeros(12, jnp.int32), last_promoted=jnp.uint32(0),
            potions=(self.potion_rules.initial_counts if self.potions_enabled else jnp.zeros(0, jnp.int32)),
            potion_doses=jnp.zeros((12,5),jnp.int32),potion_bonus=jnp.zeros((12,7)),
            potion_temporary=jnp.zeros((12,7)),
            potion_active=jnp.zeros(12,jnp.uint32),potion_wards=jnp.zeros(12,jnp.uint32),
            learned_spells=jnp.uint32(0), spell_researched_today=jnp.bool_(False),
            last_researched_spell=jnp.int32(-1), mana=jnp.array(self.starting_mana,jnp.int32),
            spell_casts=jnp.zeros(self.casting.book_count if self.spell_casting_enabled else 0,jnp.int32),
            spell_effects=jnp.uint32(0),
            enemy_spell_effects=jnp.zeros(self.num_opponents if self.spell_casting_enabled else 0,jnp.uint32),
            enemy_wounds=jnp.zeros((self.num_opponents if self.spell_casting_enabled else 0,6),jnp.int32),
            summon_marks=jnp.zeros(self.num_opponents if self.spell_casting_enabled else 0,bool),
            summon_battle=jnp.bool_(False), spell_party=jnp.zeros((4,6),jnp.int32),
            spell_mods=jnp.tile(jnp.array((0.,1.,1.,0.,1.),jnp.float32),(12,1)), spell_wards=jnp.zeros(12,jnp.uint32),
            last_cast_spell=jnp.int32(-1), last_spell_target=jnp.int32(-1), last_spell_amount=zero,
            scroll_magic=jnp.bool_(self.spell_casting_enabled and self.casting.scroll_magic_start),
            last_cast_item=jnp.int32(-1),
            unit_swaps_today=zero, unit_swaps_used=jnp.uint32(0), last_swap=jnp.int32(-1),
            territory_claims=self.territory.initial_claims if self.territory_enabled else jnp.zeros(0,jnp.int32),
            city_owned=jnp.zeros(self.territory.count if self.territory_enabled else 0,bool),
            city_levels=self.territory.initial_levels if self.territory_enabled else jnp.zeros(0,jnp.int32),
            last_captured_cities=jnp.zeros(self.territory.count if self.territory_enabled else 0,bool),
            last_upgraded_city=jnp.int32(-1),battle_fort_armor=zero,
            hire_rewarded_capacity=zero,
            mercenary_units=jnp.zeros(12, bool),
            mercenary_stock=(self.recruitment.initial_stock if self.recruitment_enabled else jnp.zeros(0, jnp.int32)),
            merchant_stock=(self.site_rules.initial_stock if self.sites_enabled else jnp.zeros(0, jnp.int32)),
            spell_stock=(self.site_rules.spell_initial_stock if self.sites_enabled else jnp.zeros(0, jnp.int32)),
            last_sale_gold=zero, last_sold_count=zero, last_trade_item=jnp.int32(-1),
            ruin_looted=jnp.zeros(self.ruins.count if self.ruins_enabled else 0,bool),last_ruin=jnp.int32(-1),
            chest_alive=jnp.ones(self.chest_rules.count if self.chests_enabled else 0, bool),
            last_loot=jnp.zeros(self.potion_rules.count if self.potions_enabled else 0, jnp.int32),
            item_inventory=(self.item_rules.initial_inventory if self.items_enabled else jnp.zeros(0,jnp.int32)),
            item_charges_spent=jnp.zeros(self.item_rules.charge_capacity if self.items_enabled else 0,jnp.int32),
            equipped=jnp.full(5,-1,jnp.int32),equipment_bonus=jnp.zeros((12,7)),
            last_item_loot=jnp.zeros(self.item_rules.count if self.items_enabled else 0,jnp.int32),
            battle_items=jnp.full(2,-1,jnp.int32),battle_item_charges=jnp.zeros(2,jnp.int32),
            battle_item_used=jnp.zeros(2,bool),battle_item_equipped_today=jnp.bool_(False),
            last_battle_item=jnp.int32(-1),
            last_potion=jnp.int32(-1), recovery_balance=jnp.zeros(2, jnp.int32), last_service_cost=zero,
            gold=jnp.int32(self.starting_gold), map_steps=zero,
            movement_points=jnp.int32(MAX_MOVEMENT_POINTS), day=jnp.int32(1),
            buildings=(self.construction.initial_built if self.basic_combat else jnp.uint32(0)),
            blocked_buildings=(self.construction.initial_blocked if self.basic_combat else jnp.uint32(0)),
            built_today=jnp.bool_(False), last_building=jnp.int32(-1),
        )
        if self.items_enabled:
            state=self._refresh_equipment(state)
        if self.progression_enabled:
            state=state.replace(movement_points=self.movement_cap(state))
        if self.sites_enabled:
            state=self.site_rules.autosell(state)
        if self.recruitment_enabled:
            state=state.replace(hire_rewarded_capacity=self.recruitment.reward_occupancy(state))
        if self.script_events.enabled:
            state = self.script_events.process(self, state, new_day=True, locations=False)
        if self.reference_rewards:
            state = state.replace(reference_visits=state.reference_visits.at[
                state.position[0]*self.size+state.position[1]].set(1),
                reference_hp_milestones=jnp.max(self.max_hp(state)[:6]) > jnp.array([200, 250]))
        return state, self._timestep(state, jnp.float32(0), first=True)

    def _clear_roster_slots(self, state, affected):
        fields = self._empty_summon_fields(state)
        zero = ('unit_ids', 'unit_levels', 'unit_xp', 'native_ids', 'native_levels', 'native_xp', 'summon_saved_hp',
                'hp', 'copy_stats', 'copy_traits', 'copy_secondary', 'preparation', 'pending_healers',
                'potion_doses', 'potion_bonus', 'potion_temporary', 'potion_active', 'potion_wards',
                'equipment_bonus', 'last_poison_damage', 'last_burn_damage', 'last_water_damage',
                'activation_done', 'priority', 'turn_phase', 'last_xp', 'battle_earned_xp', 'mercenary_units')
        fields.update({name: jnp.zeros_like(getattr(state, name)) for name in zero})
        fields['summon_owner'] = jnp.full_like(state.summon_owner, -1)
        return state.replace(**{name: jnp.where(affected.reshape((12,)+(1,)*(value.ndim-1)),
            value, getattr(state, name)) for name, value in fields.items()})

    def _equipment_hero(self,state):
        heroes=self.equipment_heroes[state.unit_ids[:6]]
        return jnp.where(jnp.any(heroes),jnp.argmax(heroes),-1)

    def _refresh_equipment(self,state):
        hero=self._equipment_hero(state)
        equipped=self.item_rules.choose(state,(hero>=0)&(state.hp[hero]>0),state.unit_levels[hero])
        old_cap=self.movement_cap(state)
        state=state.replace(equipped=equipped)
        cap=self.movement_cap(state)
        intrinsic=self._potion_intrinsic(state.unit_ids,state.unit_levels)
        bonus=self.item_rules.numeric_bonus(state,intrinsic,hero)
        return state.replace(equipment_bonus=bonus,
            movement_points=jnp.minimum(cap,state.movement_points+jnp.maximum(0,cap-old_cap)))

    def _hero_level_movement_bonus(self, ids, levels):
        # User rule: +1 for levels 2..10, then no further growth.
        bonus = jnp.clip(levels[:6]-1, 0, 9)
        if self.hero_skills:
            # Pathfinding: round(20% of the hero's own allowance) from level 2.
            bonus += jnp.where(levels[:6] >= hero_skills.PATHFINDING_LEVEL, self.pathfinding_points[ids[:6]], 0)
        return jnp.max(jnp.where(self.equipment_heroes[ids[:6]], bonus, 0))

    def _hero_base_movement(self, ids, hp):
        """Reference _hero_base_moves: the living hero's allowance, else MOVES_PER_TURN."""
        moves = jnp.max(jnp.where(hp[:6] > 0, self.hero_moves[ids[:6]], 0))
        return jnp.where(moves > 0, moves, MAX_MOVEMENT_POINTS).astype(jnp.int32)

    def _party_experience_bonus(self, state):
        """Tome of War or Weapon Master: +25% battle XP; they do not stack."""
        bonus = self.item_rules.value(state,'experience',3) if self.items_enabled else False
        if self.hero_skills:
            # Reference _hero_has_weapon_master: any hero of the party, alive or not.
            level = jnp.max(jnp.where(self.equipment_heroes[state.unit_ids[:6]], state.unit_levels[:6], 0))
            bonus = (level >= hero_skills.WEAPON_MASTER_LEVEL) | bonus
        # Reference bot autoresolution creates a bare BattleEnv whose
        # exp_multiplier stays 1.0; campaign hero/Tome bonuses are not applied.
        return bonus & ~state.bot_autobattle

    def movement_cap(self,state):
        base = jnp.int32(MAX_MOVEMENT_POINTS)
        if self.progression_enabled:
            ids, levels = state.unit_ids, state.unit_levels
            if self.has_copies or self.has_summons or self.has_item_summons:
                ids = jnp.where(state.in_battle, state.native_ids, ids)
                levels = jnp.where(state.in_battle, state.native_levels, levels)
            base = self._hero_base_movement(ids, state.hp)
        cap = self.item_rules.movement_cap(state,base) if self.items_enabled else base
        if self.progression_enabled:
            cap += self._hero_level_movement_bonus(ids, levels)
        return cap

    def _equipment_wards(self,state):
        return jnp.where(self.slots==self._equipment_hero(state),self.item_rules.value(state,'ward',3),jnp.uint32(0))

    def cast_quotes(self, state):
        if not self.spell_casting_enabled:
            return {}
        return self.casting.quotes(state, self.max_hp(state), self.movement_cap(state))

    def territory_quotes(self, state):
        if not self.territory_enabled:
            return {}
        banner = self.item_rules.value(state, 'regeneration', 2) if self.items_enabled else 0
        return self.territory.quotes(state, banner)

    def turn_metadata(self):
        return {"movement_points": self.base_movement, "dead_hero_movement_points": MAX_MOVEMENT_POINTS,
                "move_cost": MOVE_COST,
                "hero_level_movement_per_level": 1, "hero_level_movement_last_level": 10,
                "battle_entry_cost": (self.base_movement + 1) // 2, "attack_requires_points": 1,
                 "terrain_costs": dict(plain=2, road=1, forest=4, water=6),
                 "flying_cost": 2, "dead_hero_cost_multiplier": 2, "last_partial_move": True,
                "income": DAILY_GOLD, "rest_action": REST,
                "regeneration_percent": None if self.territory_enabled else 10, "regeneration_rounding": "ceil", "automatic_revive": False,
                "fenrir_action": FENRIR}

    def _spell_terrain(self, state):
        return self.casting.terrain(state) if self.spell_casting_enabled and self.casting.has_terrain else None

    def travel_costs(self, state):
        return self.terrain.costs(state, self.item_rules if self.items_enabled else None, self._spell_terrain(state))

    def move_cost(self, state, destination):
        return self.terrain.move_cost(state, destination, self.item_rules if self.items_enabled else None,
                                      self._spell_terrain(state))

    def map_commands(self, state):
        """UI/export quotes: exact target stack (-1 for a move) and point cost."""
        destinations = state.position[None] + self.directions
        if self.territory_enabled:
            targets = self.territory.enemy_at(state, destinations)
            attacking = targets>=0
        else:
            matches = state.alive[None, :] & jnp.all(
                destinations[:, None, :] == self.opponent_positions[None, :, :], axis=-1)
            attacking = jnp.any(matches, axis=1)
            targets = jnp.where(attacking, jnp.argmax(matches, axis=1), -1)
        costs = jnp.where(attacking, jnp.minimum(state.movement_points, (self.movement_cap(state)+1)//2), jnp.minimum(state.movement_points,self.move_cost(state,destinations)))
        return jnp.stack((targets, costs), axis=1)

    def rest_penalty(self, state):
        if self.reference_rewards:
            return jnp.float32(.01)+.20*(state.movement_points > 0)
        return state.movement_points * jnp.float32(self.rest_penalty_per_point)

    def unit_stats(self, state):
        """Five effective characteristics per slot; empty/world enemy slots are zero."""
        values = self._combat_stats(state)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], values, 0.)

    def _effective_ids(self, state):
        return jnp.where(state.decay_form > 0, state.decay_form, state.unit_ids) if self.has_wights else state.unit_ids

    def _effective_levels(self, state):
        return jnp.where(state.decay_form > 0, self.progression.base_levels[state.decay_form], state.unit_levels) if self.has_wights else state.unit_levels

    def _effective_shreds(self, state):
        return jnp.where(state.decay_form > 0, state.decay_shreds, state.armor_shreds) if self.has_wights else state.armor_shreds

    def _potion_intrinsic(self, ids, levels):
        values=self.progression.stats(ids,levels)
        # Potions change original power before weakening and the final damage cap.
        values=set_column(values,DAMAGE,self.progression.raw_stats(ids,levels)[...,DAMAGE])
        return jnp.concatenate((values,
                                self.progression.secondary_stats(ids,levels)),axis=-1)

    def _potion_stats(self,state,values,slot=None,temporary=True):
        if not self.has_potion_buffs and not self.items_enabled and not self.has_spell_mods:
            return values
        bonus = state.potion_bonus[:,:5] if slot is None else state.potion_bonus[slot,:5]
        active = state.summon_owner < 0 if slot is None else state.summon_owner[slot] < 0
        if self.spell_casting_enabled:
            active &= ~state.summon_battle  # the hero party's layers stay with the stashed party
        values = values+jnp.where(active[...,None],bonus,0)
        if temporary:
            delta = state.potion_temporary[:,:5] if slot is None else state.potion_temporary[slot,:5]
            values = values+jnp.where(active[...,None],delta,0)
            if self.has_spell_mods:
                values = self.casting.stats(state,values,slot)
            if self.items_enabled:
                delta = state.equipment_bonus[:,:5] if slot is None else state.equipment_bonus[slot,:5]
                values = values+jnp.where(active[...,None],delta,0)
        return set_column(values, ACCURACY, jnp.minimum(values[...,ACCURACY], 100.))

    def _natural_stats(self,state):
        return cap_stats(self._potion_stats(state,self.progression.stats(state.unit_ids,state.unit_levels)))

    def _secondary_stats(self,state,slot):
        """Secondary damage/accuracy of the current form, with growth but no buffs.

        Potions and spells, like banners and artifacts, modify only the primary
        attack (user request 2026-10-19, docs/MAP_POTIONS.md).
        """
        ids,levels=self._effective_ids(state)[slot],self._effective_levels(state)[slot]
        values=self.progression.secondary_stats(ids,levels)
        if self.has_copies:
            values=jnp.where(state.copied[slot] & (state.decay_form[slot]==0),state.copy_secondary[slot],values)
        values = jnp.where(state.imp[slot] | state.fenrir[slot],0.,values)
        values = set_column(values, 1, jnp.minimum(values[1], 100.))
        if self.has_secondary_paralysis:
            disabling = state.in_battle & (self._effect_value(state, 'secondary_paralysis_modes')[slot] > 0)
            values = set_column(values, 1, disable_attack_accuracy(values[1], disabling, state.round))
        return values

    def _combat_stats(self, state, cap=True):
        """Combine unit progression, battle modifiers, and applicable fortification armor."""
        if self.progression_enabled:
            # Lowered forms always have their native level. Fetching their
            # base row avoids a dependent level lookup through the growth table.
            values = self.progression.stats(state.unit_ids, state.unit_levels)
            if self.has_weakening or self.has_powerups or self.has_potion_buffs or self.items_enabled or self.has_spell_mods:
                values = set_column(values, DAMAGE, self.progression.raw_stats(state.unit_ids,state.unit_levels)[:,DAMAGE])
            values = self._potion_stats(state,values)
            if self.territory_enabled:
                enemies = self.slots >= 6
                values = set_column(values, ARMOR, jnp.where(
                    enemies, values[:,ARMOR]+state.battle_fort_armor.astype(values.dtype), values[:,ARMOR]))
                if self.scenario.enabled or self.bot.enabled or self.script_events.enabled:
                    values = set_column(values, ARMOR, jnp.where(
                        enemies, values[:,ARMOR], values[:,ARMOR]+state.battle_hero_fort_armor.astype(values.dtype)))
            if self.has_copies:
                values = jnp.where(state.copied[:,None],state.copy_stats,values)
                uncopied = self.progression.doppelgangers[state.unit_ids]
                fallback = uncopied & ~jnp.any(self._copy_targets(state))
                alternate_accuracy = jnp.minimum(100.,self.progression.raw_stats(state.unit_ids,state.unit_levels)[:,ACCURACY]-20)
                values = set_column(values, ACCURACY, jnp.where(fallback,alternate_accuracy,values[:,ACCURACY]))
            if self.has_wights:
                values = jnp.where((state.decay_form > 0)[:, None],
                                   self.progression.base_stats[state.decay_form], values)
        else:
            values = self.stats_table[jnp.maximum(state.enemy, 0)]
        if self.has_wolf_lord:
            values = set_column(values, HP, jnp.where(state.fenrir, 275, values[:, HP]))
            values = set_column(values, DAMAGE, jnp.where(state.fenrir, 90, values[:, DAMAGE]))
            values = set_column(values, INITIATIVE, jnp.where(state.fenrir, 65, values[:, INITIATIVE]))
        if self.has_shatterers:
            values = set_column(values, ARMOR, armor_after_shreds(values[:, ARMOR], self._effective_shreds(state)))
        if self.has_witches:
            big = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled
                   else self.size_table[jnp.maximum(state.enemy, 0)]) == 2
            values = set_column(values, DAMAGE, jnp.where(state.imp, jnp.where(big, 30, 20), values[:, DAMAGE]))
            values = set_column(values, ACCURACY, jnp.where(state.imp, jnp.where(big, 70, 80), values[:, ACCURACY]))
            values = set_column(values, ARMOR, jnp.where(state.imp, 0, values[:, ARMOR]))
            values = set_column(values, INITIATIVE, jnp.where(state.imp, jnp.where(big, 50, 30), values[:, INITIATIVE]))
        if self.has_weakening or self.has_powerups or self.has_potion_buffs or self.items_enabled or self.has_spell_mods:
            damage = jnp.rint(values[:,DAMAGE]*jnp.where(state.weakened,self._weakening_factor(state),1.))
            caps = (self.progression.stat_caps[:,DAMAGE][self._effective_ids(state)] if self.progression_enabled
                    else self.damage_caps[state.enemy])
            caps = jnp.where(state.imp | state.fenrir,300.,caps)
            if self.has_powerups:
                damage = jnp.where(state.primary_override >= 0,state.primary_override,damage)
            values = set_column(values, DAMAGE, jnp.minimum(damage,caps) if cap else damage)
        if self.has_hermits:
            values = set_column(values, INITIATIVE, jnp.where(
                state.initiative_override >= 0,state.initiative_override,values[:,INITIATIVE]))
        if self.has_paralysis:
            disabling = (state.in_battle & (self._effect_value(state, 'ghost_modes') > 0)
                         & ~state.imp & ~state.fenrir)
            values = set_column(values, ACCURACY, disable_attack_accuracy(values[:,ACCURACY], disabling, state.round))
        return cap_stats(values)

    def _original_primary(self,state,saved=False):
        if self.progression_enabled:
            ids = state.unit_ids if saved else self._effective_ids(state)
            levels = state.unit_levels if saved else self._effective_levels(state)
            amount = self.progression.raw_stats(ids,levels)[:,DAMAGE]
            if self.has_potion_buffs or self.items_enabled or self.has_spell_mods:
                enhanced = amount+state.potion_bonus[:,DAMAGE]+state.potion_temporary[:,DAMAGE]
                if self.has_spell_mods:
                    enhanced = jnp.rint(enhanced*state.spell_mods[:,DAMAGE])
                if self.items_enabled:
                    enhanced += state.equipment_bonus[:,DAMAGE]
                natural = (state.summon_owner < 0) & ~state.copied & ((state.decay_form == 0) | saved)
                if self.spell_casting_enabled:
                    natural &= ~state.summon_battle
                amount = jnp.where(natural,enhanced,amount)
            if self.has_copies:
                amount=jnp.where(state.copied & ((state.decay_form==0) | saved),state.copy_stats[:,DAMAGE],amount)
        else:
            amount = self.stats_table[state.enemy,:,DAMAGE]
        if not saved and self.has_witches:
            big = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled else self.size_table[state.enemy]) == 2
            amount = jnp.where(state.imp,jnp.where(big,30,20),amount)
        # Fenrir retains the Wolf Lord original_damage field in the reference.
        return amount.astype(jnp.int32)

    def _capture_initiative_form(self,state,mask,first):
        if not self.has_hermits:
            return state
        original = self._combat_stats(state)[:,INITIATIVE].astype(jnp.int32)
        return state.replace(saved_initiative_override=jnp.where(first,original,state.saved_initiative_override),
            initiative_override=jnp.where(mask,-1,state.initiative_override))

    def _restore_initiative_form(self,state,mask):
        if not self.has_hermits:
            return state
        return state.replace(initiative_override=jnp.where(mask,state.saved_initiative_override,state.initiative_override),
            saved_initiative_override=jnp.where(mask,-1,state.saved_initiative_override))

    def _alchemist_units(self,state):
        flags = (self.progression.alchemists[self._effective_ids(state)] if self.progression_enabled else self.alchemists[state.enemy])
        return flags & ~state.imp

    def _alchemist_targets(self,state):
        return (state.hp > 0) & ~state.escaped & ~state.retreating & ~self._alchemist_units(state) & (self.slots != state.actor) & ((self.slots < 6) == (state.actor < 6))

    def _actor_is_patriarch(self,state):
        if not self.has_patriarchs:
            return jnp.bool_(False)
        flag = (self.progression.patriarchs[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.patriarchs[state.enemy,state.actor])
        return flag & ~state.imp[state.actor]

    def _patriarch_targets(self,state):
        alive = (state.hp > 0) & ~state.escaped
        sizes = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled else self.size_table[state.enemy])
        natural_sizes = self.progression.sizes[state.unit_ids] if self.progression_enabled else sizes
        pair = jnp.where(self.slots%6 < 3,self.slots+3,self.slots-3)
        back = self.slots%6 >= 3
        covered = back & alive[pair] & (sizes[pair] == 2)
        footprint_free = ~covered & ~((natural_sizes == 2) & alive[pair])
        occupied = (state.unit_ids != 0) if self.progression_enabled else (self._combat_stats(state)[:,HP] > 0)
        revive = can_revive(state, self.slots) & ~state.battle_revived & footprint_free
        # Dead slots have zero effective initiative in the JAX scheduler.
        return occupied & ~state.escaped & ~covered & (alive | revive) & ((self.slots < 6) == (state.actor < 6))

    def _actor_power_cures(self,state):
        if not self.has_powerups or not self.has_cures:
            return jnp.bool_(False)
        flag = (self.progression.power_cures[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.power_cures[state.enemy,state.actor])
        return flag & ~state.imp[state.actor]

    def _actor_power_factor(self,state):
        if not self.has_powerups:
            return jnp.float32(0)
        factor = (self.progression.power_factors[self._effective_ids(state)[state.actor]] if self.progression_enabled
                  else self.power_factors[state.enemy,state.actor])
        return jnp.where(state.imp[state.actor],0.,factor)

    def _capture_damage_forms(self,state,mask,first):
        if not self.has_powerups:
            return state
        update = {}
        for name,empty in (('primary_override',-1),('preweak_damage',-1),('powerup',False),('powerup_long',False),('powerup_layered',False)):
            update['saved_'+name] = jnp.where(first,getattr(state,name),getattr(state,'saved_'+name))
            update[name] = jnp.where(mask,empty,getattr(state,name))
        return state.replace(**update)

    def _restore_damage_forms(self,state,mask):
        if not self.has_powerups:
            return state
        update = {}
        for name,empty in (('primary_override',-1),('preweak_damage',-1),('powerup',False),('powerup_long',False),('powerup_layered',False)):
            update[name] = jnp.where(mask,getattr(state,'saved_'+name),getattr(state,name))
            update['saved_'+name] = jnp.where(mask,empty,getattr(state,'saved_'+name))
        return state.replace(**update)

    def _weakening_factor(self,state,saved=False):
        # GVars BATLOWER level 2: subtract 33 percent of the base amount.
        return .67

    def _expire_powerups(self,state,spent):
        if not self.has_powerups:
            return state
        update = {}
        for prefix in ('','saved_'):
            active = spent & getattr(state,prefix+'powerup') & ~getattr(state,prefix+'powerup_long')
            original = self._original_primary(state,saved=bool(prefix))
            factor = self._weakening_factor(state,saved=bool(prefix))
            restored = jnp.rint(original*jnp.where(getattr(state,prefix+'weakened'),factor,1.)).astype(jnp.int32)
            update[prefix+'primary_override'] = jnp.where(active,restored,getattr(state,prefix+'primary_override'))
            update[prefix+'preweak_damage'] = jnp.where(active,original,getattr(state,prefix+'preweak_damage'))
            update[prefix+'powerup'] = getattr(state,prefix+'powerup') & ~active
            update[prefix+'powerup_layered'] = getattr(state,prefix+'powerup_layered') & ~active
        return state.replace(**update)

    def _base_combat_traits(self, state):
        if self.progression_enabled:
            traits = self.progression.traits[self._effective_ids(state)]
            if self.has_copies:
                traits = jnp.where((state.copied & (state.decay_form == 0))[:,None],state.copy_traits,traits)
                fallback = self.progression.doppelgangers[self._effective_ids(state)] & ~jnp.any(self._copy_targets(state))
                traits = set_column(traits, 0, jnp.where(fallback,MELEE,traits[:,0]))
                traits = set_column(traits, 1, jnp.where(fallback,1,traits[:,1]))
        else:
            traits = self.combat_traits[jnp.maximum(state.enemy, 0)]
        natural = ~state.copied & (state.decay_form == 0) & (state.summon_owner < 0)
        if self.spell_casting_enabled:
            natural &= ~state.summon_battle
        if self.has_potion_buffs:
            traits = set_column(traits, 3, traits[:,3] | jnp.where(natural,state.potion_wards,jnp.uint32(0)))
        if self.items_enabled:
            traits = set_column(traits, 3, traits[:,3] | jnp.where(natural,self._equipment_wards(state),jnp.uint32(0)))
        if self.has_spell_wards:
            traits = set_column(traits, 3, traits[:,3] | jnp.where(natural,state.spell_wards,jnp.uint32(0)))
        if self.has_wolf_lord:
            traits = set_column(traits, 0, jnp.where(state.fenrir, MELEE, traits[:, 0]))
            traits = set_column(traits, 1, jnp.where(state.fenrir, 1, traits[:, 1]))
        if self.has_witches:
            traits = jnp.where(state.imp[:, None], jnp.array([MELEE, 1, 0, 0], jnp.uint32), traits)
        return traits

    def _combat_traits(self,state):
        traits = self._base_combat_traits(state)
        if self.has_healer_wards:
            traits = set_column(traits, 3, traits[:,3] | ward_tags(state.healer_wards))
        return traits

    def _expire_healer_wards(self,state,used,casters):
        if not self.has_healer_wards:
            return state,used
        native = self._base_combat_traits(state)[:,3]
        saved_native = (self.progression.traits[state.unit_ids,3] if self.progression_enabled
                        else self.combat_traits[state.enemy,:,3])
        if self.has_potion_buffs:
            saved_native |= jnp.where(~state.copied & (state.summon_owner < 0),state.potion_wards,jnp.uint32(0))
        if self.items_enabled:
            saved_native |= jnp.where(~state.copied & (state.summon_owner < 0),self._equipment_wards(state),jnp.uint32(0))
        if self.has_spell_wards:
            saved_native |= jnp.where(~state.copied & (state.summon_owner < 0),state.spell_wards,jnp.uint32(0))
        owners,sticky,used = expire_wards(state.healer_wards,state.ward_native_used,used,native,casters)
        saved,saved_sticky,saved_used = expire_wards(state.saved_healer_wards,state.saved_ward_native_used,state.imp_wards,saved_native,casters)
        return state.replace(healer_wards=owners,ward_native_used=sticky,
            saved_healer_wards=saved,saved_ward_native_used=saved_sticky,imp_wards=saved_used),used

    def _capture_ward_forms(self,state,mask,first):
        if not self.has_healer_wards:
            return state
        return state.replace(
            saved_healer_wards=jnp.where(first[:,None],state.healer_wards,state.saved_healer_wards),
            saved_ward_native_used=jnp.where(first,state.ward_native_used,state.saved_ward_native_used),
            healer_wards=jnp.where(mask[:,None],jnp.uint32(0),state.healer_wards),
            ward_native_used=jnp.where(mask,jnp.uint32(0),state.ward_native_used))

    def _restore_ward_forms(self,state,mask):
        if not self.has_healer_wards:
            return state
        return state.replace(healer_wards=jnp.where(mask[:,None],state.saved_healer_wards,state.healer_wards),
            ward_native_used=jnp.where(mask,state.saved_ward_native_used,state.ward_native_used),
            saved_healer_wards=jnp.where(mask[:,None],jnp.uint32(0),state.saved_healer_wards),
            saved_ward_native_used=jnp.where(mask,jnp.uint32(0),state.saved_ward_native_used))

    def _actor_is_wolf_lord(self, state):
        if not self.has_wolf_lord:
            return jnp.bool_(False)
        wolf = (self.progression.wolf_lord[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.wolf_lord[jnp.maximum(state.enemy, 0), state.actor])
        return wolf & ~state.fenrir[state.actor] & ~state.imp[state.actor]

    def _actor_is_witch(self, state):
        if not self.has_witches:
            return jnp.bool_(False)
        witch = (self.progression.witches[self._effective_ids(state)[state.actor]] if self.progression_enabled
                 else self.witches[jnp.maximum(state.enemy, 0), state.actor])
        return witch & ~state.imp[state.actor]

    def _actor_ghost_mode(self,state):
        if not self.has_paralysis:
            return jnp.int32(0)
        mode = (self.progression.ghost_modes[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.ghost_modes[state.enemy,state.actor])
        return jnp.where(state.imp[state.actor],0,mode)

    def _can_retreat(self, state, slots):
        """Settlement restriction for either side; turn/status checks stay with callers.

        Snapshot the defending side at battle entry, independently of armour.
        Steam sub_5F5A4B; sources: docs/validation/fort-defenders-20261021.json.
        """
        return ~jnp.where(slots < 6, state.battle_fort_defender == FORT_BLUE,
                          state.battle_fort_defender == FORT_RED)

    def _fear_protected(self, state, target_slots):
        """Fear paralyses instead of routing: map teams, defended cities and ruins."""
        protected = self.fear_protected[target_slots] | ~self._can_retreat(state, target_slots)
        if self.ruins_enabled:
            protected |= (target_slots>=6) & (state.enemy>=0) & (self.ruins.enemy_ruin[jnp.maximum(state.enemy,0)]>=0)
        return protected

    def _actor_is_fear_caster(self, state):
        if not self.has_fear:
            return jnp.bool_(False)
        caster = (self.progression.fear_casters[self._effective_ids(state)[state.actor]] if self.progression_enabled
                  else self.fear_casters[state.enemy,state.actor])
        return caster & ~state.imp[state.actor]

    def _actor_secondary_paralysis(self, state):
        if not self.has_secondary_paralysis:
            return jnp.int32(0)
        mode = (self.progression.secondary_paralysis_modes[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.secondary_paralysis_modes[state.enemy,state.actor])
        return jnp.where(state.imp[state.actor],0,mode)

    def _actor_double_strike(self, state):
        if not self.has_double_strike:
            return jnp.bool_(False)
        double = (self.progression.double_strike[self._effective_ids(state)[state.actor]] if self.progression_enabled
                  else self.double_strike[state.enemy, state.actor])
        return double & ~state.imp[state.actor]

    def _start_activation(self, state, roll, enabled=True):
        if not self.has_witches and not self.has_poisoners and not self.has_water and not self.has_fire and not self.has_wights:
            if self.has_hermits:
                return state.replace(activation_done=set_rows(state.activation_done, state.actor,
                    state.activation_done[state.actor] | enabled))
            return state
        actor = state.actor
        lowered = state.decay_form[actor] > 0
        leaving = state.retreating[actor] & ~state.paralyzed[actor] & ~state.long_paralyzed[actor]
        hostile = state.imp[actor] | state.lycanthrope[actor]
        recover = enabled & ~leaving & hostile & ~state.second_strike & (
            ~state.transform_long[actor] | effect_breaks(state.round,state.effect_rounds[actor,TRANSFORM],roll))
        natural = (self._natural_stats(state)[actor]
                   if self.progression_enabled else self.stats_table[state.enemy, actor])
        if self.has_copies:
            natural = jnp.where(state.copied[actor],state.copy_stats[actor],natural)
        initiative, maximum = natural[INITIATIVE], natural[HP]
        if self.has_wolf_lord:
            initiative = jnp.where(state.fenrir[actor], 65., initiative)
            maximum = jnp.where(state.fenrir[actor], 275., maximum)
        hp = state.hp
        if self.has_wights:
            current_max = self.progression.base_stats[:, HP][state.decay_form[actor]]
            restored = jnp.rint(hp[actor] / jnp.maximum(current_max, 1) * maximum).astype(jnp.int32)
            restored = jnp.where(hp[actor] > 0, jnp.clip(restored, 1, MAX_HP), 0)
            hp = set_rows(hp, actor, jnp.where(recover & lowered, restored, hp[actor]))
        state = self._restore_initiative_form(state,(self.slots == actor) & recover)
        initiative = jnp.where(state.initiative_override[actor] >= 0,state.initiative_override[actor],initiative)
        priority = jnp.where(recover & (state.turn_phase[actor] != 1), initiative, state.priority[actor])
        state = self._restore_damage_forms(state,(self.slots == actor) & recover)
        state = self._restore_ward_forms(state,(self.slots == actor) & recover)
        if self.has_item_weakening:
            state = state.replace(item_weakened=set_rows(state.item_weakened, actor,
                jnp.where(recover,state.saved_item_weakened[actor],state.item_weakened[actor])))
        if self.has_lycanthropy:
            state = state.replace(lycanthrope=set_rows(state.lycanthrope, actor, state.lycanthrope[actor] & ~recover))
        return state.replace(hp=hp, weakened=set_rows(state.weakened, actor, jnp.where(recover,state.saved_weakened[actor],state.weakened[actor])),
            transform_long=set_rows(state.transform_long, actor, state.transform_long[actor] & ~recover),
            imp=set_rows(state.imp, actor, state.imp[actor] & ~recover),
            decay_form=set_rows(state.decay_form, actor, jnp.where(recover, 0, state.decay_form[actor])),
            decay_shreds=set_rows(state.decay_shreds, actor, jnp.where(recover, 0, state.decay_shreds[actor])),
            wards_used=set_rows(state.wards_used, actor, jnp.where(recover, state.imp_wards[actor], state.wards_used[actor])),
            activation_done=set_rows(state.activation_done, actor, state.activation_done[actor] | enabled),
            priority=set_rows(state.priority, actor, priority))

    def unit_experience(self, state):
        if not self.progression_enabled:
            return jnp.zeros((12, 4), jnp.int32)
        values = self.progression.experience(state.unit_ids, state.unit_levels, state.unit_xp)
        if self.has_copies or self.has_summons or self.has_item_summons:
            native = self.progression.experience(state.native_ids,state.native_levels,state.native_xp)
            values = jnp.where(state.copied[:,None],native,values)
            values = jnp.where((state.summon_owner >= 0)[:,None],0,values)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], values, 0)

    def unit_traits(self, state):
        traits = self._combat_traits(state)
        traits = traits.at[:, 3].set(traits[:, 3] & ~state.wards_used)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], traits, 0)

    def unit_sizes(self, state):
        sizes = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled
                 else self.size_table[jnp.maximum(state.enemy, 0)])
        if self.has_copies:
            sizes = jnp.where(state.copied,1,sizes)
        return jnp.where((self.slots < 6) | state.in_battle, sizes, 0)

    def _actor_mass_healer(self, state):
        if not self.basic_combat or not self.has_healers:
            return jnp.bool_(False)
        flag = (self.progression.mass_healers[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.mass_healers[state.enemy,state.actor])
        return flag & ~state.imp[state.actor]

    def _actor_is_healer(self, state):
        return self._combat_traits(state)[state.actor, 0] == HEALER

    def _actor_is_melee(self, state):
        return self._combat_traits(state)[state.actor, 0] == MELEE

    def max_hp(self, state):
        if self.basic_combat:
            return self.unit_stats(state)[:, HP].astype(jnp.int32)
        enemy = jnp.maximum(state.enemy, 0)
        opponents = jnp.where(jnp.arange(6) < self.enemy_counts[enemy], self.enemy_health[enemy], 0)
        return jnp.concatenate((self.hero_full, jnp.where(state.in_battle, opponents, 0)))

    def observation(self, state):
        """Encode the current world and battle state into the configured flat feature vector."""
        if not self.battle_mode:
            return super().observation(state)
        context = jnp.asarray([
            state.in_battle, (state.enemy + 1) / self.num_opponents,
            state.round / self.max_rounds, state.actor / 11,
            state.origin[0] / (self.size - 1), state.origin[1] / (self.size - 1)
        ], jnp.float32)
        if self.basic_combat:
            context = jnp.concatenate((context, self.construction.observation(state), jnp.asarray(
                [state.gold / 1000., state.movement_points / MAX_MOVEMENT_POINTS], jnp.float32)))
        if self.observation_version in (2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45):
            # No duplicate max-HP arrays or obsolete numeric battle strengths.
            # Signed queue priority contains both order and waiting/acted status.
            world = jnp.stack((state.enemy_positions[:,0] / (self.size-1),
                               state.enemy_positions[:,1] / (self.size-1), state.alive,
                               self.enemy_counts / 6, self.enemy_health / self.hero_hp),axis=1)
            active = (state.hp > 0) & ~state.escaped
            queue = jnp.where(active & (state.turn_phase < 2), state.priority, 0.)
            hp_scale = (jnp.maximum(self.max_hp(state), 1) if self.observation_version in (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)
                        else self.hero_hp)
            if self.basic_combat:
                queue /= self.priority_scale
            units = jnp.stack((state.hp / hp_scale, queue, state.defended,
                               jnp.where(state.escaped, 1., state.retreating * .5)),axis=1)
            if self.basic_combat:
                units = jnp.concatenate((units, self.unit_stats(state) /
                                         jnp.asarray([100., 300., 100., 100., 100.])), axis=1)
            if self.potions_enabled:
                context = jnp.concatenate((self.potion_rules.observation(state), context))
            if self.capital_enabled:
                context = jnp.concatenate((self.capital.observation(state, self.size), context))
            if self.chests_enabled:
                context = jnp.concatenate((self.chest_rules.observation(state), context))
            if self.progression_enabled:
                xp = self.unit_experience(state).astype(jnp.float32)
                ids = jnp.where((self.slots < 6) | state.in_battle, state.unit_ids, 0)
                growth = jnp.stack((ids / (len(self.progression.rows)-1), xp[:, 0]/100.,
                                   xp[:, 1]/1000., xp[:, 2]/10000., xp[:, 3]/jnp.maximum(xp[:, 2], 1)), axis=1)
                context = jnp.concatenate((growth.reshape(-1), context,
                                          self.progression.blocked_xp_observation(state)))
                if self.observation_version < 25:
                    context = jnp.concatenate((context, state.battle_earned_xp/10000.))
            if self.observation_version in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45):
                # Compact exact source/bitset encoding; keeps the GPU policy input small.
                context = jnp.concatenate((self.unit_traits(state).reshape(-1) /
                                           jnp.tile(jnp.array([4. if self.observation_version >= 10 else 3., 9., 511., 511.]), 12), context))
            if self.observation_version >= 10:
                context = jnp.concatenate((context, self.unit_sizes(state) / 2.))
            if self.observation_version >= 14:
                flags = state.imp.astype(jnp.int32) + 2*state.fenrir + 4*state.activation_done + 8*((self.slots == state.actor) & state.second_strike)
                flags += 16*state.paralyzed + 32*state.long_paralyzed + 64*state.weakened + 128*state.saved_weakened
                context = jnp.concatenate((context, flags / 255.))
            if self.observation_version >= 15:
                poison = jnp.stack((state.poison_mode / 2., state.poison_damage / 300., (state.poison_source+1) / 12.), axis=1)
                context = jnp.concatenate((context, poison.reshape(-1)))
            if self.observation_version >= 16:
                context = jnp.concatenate((context, state.decay_form / len(self.progression.rows)))
            if self.observation_version >= 17:
                water = jnp.stack((state.water_mode / 2., state.water_damage / 300., (state.water_source+1) / 12.),axis=1)
                context = jnp.concatenate((context,water.reshape(-1)))
            if self.observation_version >= 18:
                burn = jnp.stack((state.burn_mode/2.,state.burn_damage/300.,(state.burn_source+1)/12.),axis=1)
                context = jnp.concatenate((context,burn.reshape(-1)))
            if self.observation_version >= 21:
                credit = jnp.stack((state.enemy_initial_hp/1000.,state.enemy_damage_credit/1000.,state.enemy_killed_credit),axis=1)
                context = jnp.concatenate((context,state.feared,jnp.where(state.in_battle,credit.reshape(-1),0.)))
            if self.observation_version >= 22:
                context = jnp.concatenate((context,jnp.array([state.post_victory/2.]),state.pending_healers))
            if self.observation_version >= 23:
                context = jnp.concatenate((context,state.healer_wards.reshape(-1)/4095.,state.saved_healer_wards.reshape(-1)/4095.,
                    state.ward_native_used/511.,state.saved_ward_native_used/511.))
            if self.observation_version >= 24:
                for prefix in ('','saved_'):
                    flags = getattr(state,prefix+'powerup').astype(jnp.int32)+2*getattr(state,prefix+'powerup_layered')
                    context = jnp.concatenate((context,getattr(state,prefix+'primary_override')/1000.,getattr(state,prefix+'preweak_damage')/1000.,flags/3.))
            if self.observation_version >= 25:
                context = jnp.concatenate((context,state.battle_revived,state.battle_earned_xp/10000.))
            if self.observation_version >= 26:
                context = jnp.concatenate((context,state.waited,state.bonus_turns/10.))
            if self.observation_version >= 27:
                context = jnp.concatenate((context,state.initiative_override/100.,state.saved_initiative_override/100.,state.slow_original/100.))
            if self.observation_version >= 28:
                context = jnp.concatenate((context,state.copied,(state.summon_owner+1)/12.,state.preparation,
                    jnp.where(state.copied | (state.summon_owner >= 0),state.native_ids,0)/(len(self.progression.rows)-1)))
            if self.observation_version >= 44:
                present = jnp.stack((state.poison_mode>0,state.water_mode>0,state.burn_mode>0,
                    state.paralyzed | state.long_paralyzed,state.imp | state.lycanthrope),axis=1)
                age = jnp.where(present & state.in_battle,jnp.clip(state.round-state.effect_rounds,0,9)/9.,0.)
                context=jnp.concatenate((context,age.reshape(-1),(state.disable_source+1)/12.,
                    state.transform_long,state.powerup_long,state.saved_powerup_long))
            if self.observation_version >= 29:
                context=jnp.concatenate((context,jnp.where(state.copied[:,None],state.copy_secondary,0).reshape(-1)/300.))
            if self.items_enabled:
                context=jnp.concatenate((context,self.item_rules.observation(state)))
            if self.sites_enabled:
                context=jnp.concatenate((context,self.site_rules.observation(state)))
            if self.recruitment_enabled:
                context=jnp.concatenate((context,self.recruitment.observation(state)))
            if self.territory_enabled:
                context=jnp.concatenate((context,self.territory.observation(state)))
            if self.ruins_enabled:
                context=jnp.concatenate((context,self.ruins.observation(state)))
            if self.observation_version >= 37:
                context=jnp.concatenate((context,self.terrain.observation(state,self.directions,
                    self.item_rules if self.items_enabled else None,self._spell_terrain(state))))
            if self.observation_version >= 38:
                context=jnp.concatenate((context,self.construction.lord_observation))
            if self.spell_research_enabled:
                context=jnp.concatenate((context,self.spell_research.observation(state)))
            if self.spell_casting_enabled:
                context=jnp.concatenate((context,self.casting.observation(state)))
            if self.battle_items_enabled:
                context=jnp.concatenate((context,self.battle_item_rules.observation(state)))
            if self.unit_swaps_enabled:
                context=jnp.concatenate((context,self.unit_swaps.observation(state)))
            if self.scenario.enabled:
                context=jnp.concatenate((context,self.scenario.observation(state)))
            if self.script_events.configured:
                context=jnp.concatenate((context,self.script_events.observation(state)))
            if self.bot.enabled:
                context=jnp.concatenate((context,self.bot.observation(state)))
            if self.position_objective:
                context=jnp.concatenate((context, self.objective_position/(self.size-1),
                                         (self.objective_position-state.position)/(self.size-1)))
            if self.capital_training.enabled:
                context=jnp.concatenate((context, self.capital_training.observation(state)))
            if self.army_victory_level_reward:
                context=jnp.concatenate((context,state.army_victory_rewarded.astype(jnp.float32),
                                         jnp.asarray([state.army_battle_level_strength/100.], jnp.float32)))
            return jnp.concatenate((state.position / (self.size-1),
                                    jnp.asarray([state.number / self.number_scale,
                                                 state.step_count / self.max_steps],jnp.float32),
                                    world.reshape(-1), units.reshape(-1), context))
        base = super().observation(state)
        units = jnp.stack((state.hp / self.hero_hp, self.max_hp(state) / self.hero_hp,
                           state.priority, state.turn_phase / 2, state.defended,
                           state.retreating, state.escaped,
                           (self.slots == state.actor) & state.in_battle), axis=1)
        return jnp.concatenate((base, self.enemy_counts / 6, self.enemy_health / self.hero_hp,
                                units.reshape(-1), context))

    def _melee_targets(self, state, enemy_side=False):
        """Same reachability for both sides; pending retreat still occupies a slot."""
        if self.basic_combat:
            side = 0 if enemy_side else 6
            own = 6 if enemy_side else 0
            active = (state.hp[side:side+6] > 0) & ~state.escaped[side:side+6]
            tiers = self.all_melee_tiers[state.actor % 6]
            blocked = (state.actor % 6 >= 3) & jnp.any(
                (state.hp[own:own+3] > 0) & ~state.escaped[own:own+3])
        elif enemy_side:
            active = (state.hp[:6] > 0) & ~state.escaped[:6]
            tiers = self.enemy_melee_tiers[jnp.clip(state.actor - 6, 0, 5)]
            blocked = (state.actor >= 9) & jnp.any((state.hp[6:9] > 0) & ~state.escaped[6:9])
        else:
            active = (state.hp[6:] > 0) & ~state.escaped[6:]
            tiers = self.melee_tiers
            blocked = (jnp.any((state.hp[:3] > 0) & ~state.escaped[:3])
                       if self.warrior_slot >= 3 else False)
        first_tier = jnp.min(jnp.where(active, tiers, 4))
        return active & (tiers == first_tier) & ~jnp.bool_(blocked)

    def _enemy_is_warrior(self, state):
        return (state.actor >= 6) & (state.actor - 6 == self.enemy_warrior_slots[state.enemy])

    def _round_priority(self, random_values, enemy=0, state=None):
        if self.basic_combat:
            base = self._combat_stats(state)[:,INITIATIVE] if state is not None else self.stats_table[enemy,:,INITIATIVE]
            random = random_values[:12]*10
            value = base+jnp.floor(random)
            if state is not None and self.has_hermits:
                value = jnp.where(state.slow_original >= 0,value-jnp.floor(value*.5),value)
            # Keep the fractional tie key after the original integer calculation.
            return value+random-jnp.floor(random)
        # Preserve existing random initiative (1..2), scaled by base initiative.
        # Archers/mage stay at 60; a warrior at 50 is slower, but not always last.
        priority = random_values[1:] + 1
        if self.has_enemy_warriors:
            priority *= self.squad_initiative_scale[enemy]
        elif self.warrior_slot >= 0:
            priority *= self.initiative_scale
        return priority

    def action_mask(self, state):
        """Mark legal commands using the current party, dynamic enemy positions, and resources."""
        if not self.battle_mode:
            return super().action_mask(state)
        destination = state.position[None] + self.directions
        if self.territory_enabled:
            movement = self.territory.movement_mask(state) & ~state.in_battle
        else:
            movement = super().action_mask(state) & ~state.in_battle
            if self.sites_enabled:
                movement &= ~self.site_rules.blocked(destination)
        if self.ruins_enabled:
            if not self.territory_enabled:
                movement &= ~self.ruins.blocked(destination)
            movement = (movement | self.ruins.attackable(state,destination)) & ~state.in_battle
        if self.basic_combat:
            movement &= state.movement_points > 0
        controlled = (state.in_battle & (state.actor < 6) & (state.hp[state.actor] > 0)
                      & ~state.escaped[state.actor] & ~state.retreating[state.actor]
                      & ~state.paralyzed[state.actor] & ~state.long_paralyzed[state.actor])
        targets = (state.hp[6:] > 0) & ~state.escaped[6:] & controlled
        if self.basic_combat:
            if self.has_healers:
                targets = jnp.where(self._actor_is_healer(state),
                    (state.hp[:6] > 0) & ~state.escaped[:6] & controlled, targets)
            if self.has_patriarchs:
                targets = jnp.where(self._actor_is_patriarch(state),self._patriarch_targets(state)[:6] & controlled,targets)
            if self.has_alchemists:
                targets = jnp.where(self._alchemist_units(state)[state.actor],self._alchemist_targets(state)[:6] & controlled,targets)
            targets &= (self._actor_power_factor(state) == 0) | (self.slots[:6] != state.actor) | self._actor_power_cures(state)
            targets &= ~self._actor_is_melee(state) | self._melee_targets(state)
        elif self.warrior_slot >= 0:
            targets &= (state.actor != self.warrior_slot) | self._melee_targets(state)
        mask = jnp.concatenate((movement, targets, jnp.asarray([
            controlled & ~state.second_strike, controlled & ~state.second_strike & (state.turn_phase[state.actor] == 0) & ~state.waited[state.actor],
            controlled & ~state.second_strike, state.in_battle & ~controlled])))
        if self.basic_combat:
            mask = jnp.concatenate((mask, self.construction.available(state),
                                    jnp.asarray([~state.in_battle & ~state.done])))
        if self.capital_enabled:
            heal, revive = self.capital.available(state, self.max_hp(state))
            mask = jnp.concatenate((mask, heal, revive))
        if self.num_actions >= POTION_START:
            mask = jnp.concatenate((mask, jnp.zeros(FENRIR-mask.shape[0], jnp.bool_),
                                    jnp.asarray([controlled & ~state.second_strike & self._actor_is_wolf_lord(state)]),
                                    jnp.zeros(COPY_ACTIONS,bool)))
        if self.potions_enabled:
            mask = jnp.concatenate((mask, self.potion_rules.available(state, self.max_hp(state)).reshape(-1)))
        if self.sites_enabled:
            mask = jnp.concatenate((mask, self.site_rules.available(state)))
        if self.recruitment_enabled:
            mask = jnp.concatenate((mask, self.recruitment.available(state)))
        if self.territory_enabled:
            mask = jnp.concatenate((mask, self.territory.available(state)))
        if self.spell_research_enabled:
            mask = jnp.concatenate((mask, self.spell_research.available(state)))
        if self.spell_casting_enabled:
            mask = jnp.concatenate((mask, self.casting.available(state, self.max_hp(state), self.movement_cap(state))))
        if self.battle_items_enabled and self.battle_item_rules.count:
            rules = self.battle_item_rules
            mask = jnp.concatenate((mask, rules.equip_available(self, state), rules.use_available(self, state)))
        if self.unit_swaps_enabled:
            mask = jnp.concatenate((mask, self.unit_swaps.available(state)))
        if self.has_copies:
            copy = self._actor_copies(state) & controlled
            valid_copy = self._copy_targets(state)
            mask = mask.at[SHOOT:DEFEND].set(jnp.where(copy,valid_copy[6:],mask[SHOOT:DEFEND]))
            mask = mask.at[COPY_ALLY:COPY_ALLY+COPY_ACTIONS].set(copy & valid_copy[:6])
            preparing = jnp.zeros(self.num_actions,bool).at[SHOOT:DEFEND].set(copy & valid_copy[6:])
            preparing = preparing.at[COPY_ALLY:COPY_ALLY+COPY_ACTIONS].set(copy & valid_copy[:6]).at[DEFEND].set(controlled).at[CONTINUE].set(~controlled)
            mask = jnp.where(state.in_battle & (state.round == 0),preparing,mask)
        if self.has_summons:
            mask = mask.at[SHOOT:DEFEND].set(jnp.where(self._actor_summons(state) > 0,
                controlled & self._summon_targets(state)[:6],mask[SHOOT:DEFEND]))
        if self.progression_enabled and self.has_healers:
            post = jnp.zeros(self.num_actions,jnp.bool_).at[SHOOT:DEFEND].set(
                (state.hp[:6] > 0) & ~state.escaped[:6] & (state.actor < 6))
            if self.has_patriarchs:
                post = post.at[SHOOT:DEFEND].set(jnp.where(self._actor_is_patriarch(state),self._patriarch_targets(state)[:6] & (state.actor < 6),post[SHOOT:DEFEND]))
            post = post.at[WAIT].set(state.actor < 6).at[CONTINUE].set(state.actor >= 6)
            mask = jnp.where(state.post_victory > 0,post,mask)
        forced_action = jnp.where(controlled, RETREAT, CONTINUE)
        mask = jnp.where(self.round_limit_retreat(state), jnp.arange(self.num_actions) == forced_action, mask)
        mask = mask.at[RETREAT].set(mask[RETREAT] & self._can_retreat(state, state.actor))
        if self.scenario.enabled or self.bot.enabled or self.script_events.enabled:
            mask = jnp.where(~state.in_battle & jnp.any(state.pending_encounters & state.alive),
                             jnp.arange(self.num_actions) == CONTINUE, mask)
        return jnp.where(state.done, jnp.arange(self.num_actions) == CONTINUE, mask)

    def round_limit_retreat(self, state):
        """After the round limit, a player party allowed to withdraw must retreat.

        Spell summons keep their existing timeout/return path. Post-battle
        healing must finish normally. Settlement defenders keep fighting;
        the episode decision budget still applies.
        """
        return (state.in_battle & (state.round > self.max_rounds)
                & ~state.summon_battle & ~state.bot_autobattle & (state.post_victory == 0)
                & self._can_retreat(state, jnp.int32(0)))

    def automatic_battle_turn(self, state):
        """Reference _advance_until_blue_turn: no player choice on this activation."""
        forced = (state.retreating[state.actor] | state.paralyzed[state.actor]
                  | state.long_paralyzed[state.actor] | (state.hp[state.actor] <= 0)
                  | state.escaped[state.actor]) & (state.post_victory == 0)
        return (state.in_battle & ~state.done
                & ((state.actor >= 6) | forced | self.round_limit_retreat(state)))

    def _automatic_battle_action(self, state):
        controlled = ((state.actor < 6) & ~state.retreating[state.actor]
                      & (state.hp[state.actor] > 0) & ~state.escaped[state.actor]
                      & ~state.paralyzed[state.actor] & ~state.long_paralyzed[state.actor])
        return jnp.where(self.round_limit_retreat(state) & controlled, RETREAT, CONTINUE).astype(jnp.int32)

    def _goals_left(self, alive, spawned):
        """Check army goals; location and bot objectives have separate completion conditions."""
        if self.scripted_bot_objective or self.position_objective:
            return jnp.bool_(True)  # clearing the army table cannot complete these objectives
        alive = alive | ~spawned
        return jnp.any(alive if self.goal_enemies is None else alive & self.goal_enemies)

    def _capture_cities(self, state, entered):
        """Capture a cleared city; its vault items and potions go to the party at once."""
        state = self.territory.capture(state, entered)
        if self.items_enabled:
            state = self.item_rules.collect_cities(state, state.last_captured_cities)
        if self.potions_enabled and self.territory.count:
            loot = jnp.sum(jnp.where(state.last_captured_cities[:, None], self.potion_rules.city_loot, 0), axis=0)
            state = state.replace(potions=state.potions+loot, last_loot=state.last_loot+loot)
        return state

    def _campaign_won(self, state):
        """Check the configured army, city, location or bot objective without awarding rewards."""
        if self.position_objective:
            alive = (self.objective_heroes[state.unit_ids[:6]] & (state.hp[:6] > 0)
                     & (state.summon_owner[:6] < 0) & ~state.copied[:6])
            return ((jnp.max(jnp.abs(state.position-self.objective_position)) <= self.objective_radius) & jnp.any(alive)
                    & self.script_events.objective_ready(state) & ~jnp.any(state.pending_encounters & state.alive)
                    & ~state.in_battle & ~state.summon_battle & ~state.lost)
        if self.scripted_bot_objective:
            return (state.bot_hero_victories > 0) & ~state.lost & ~state.bot_autobattle
        return (~self._goals_left(state.alive, state.scenario_spawned) & ~state.lost
                & (self.territory.goals_captured(state) if self.territory_enabled else True))

    def enemy_unit_ids(self, state):
        """Return enemy formations with the mobile bot's current upgraded roster substituted."""
        return self.bot.ids(state) if self.bot.enabled else self.progression.enemy_ids

    def _objective_reward(self, won, hero_battle):
        # Reference blue-dragon bonus: magic can win, but only the hero army earns it.
        eligible = hero_battle if self.objective_reward_hero_only else jnp.bool_(True)
        amount = self.objective_position_reward if self.position_objective else 3.
        return amount * (won & eligible).astype(jnp.float32)

    def _timestep(self, state, reward, first=False):
        ts = super()._timestep(state, reward, first)
        if self.battle_mode:
            ts = ts.replace(extras={**ts.extras, 'battle_transition': jnp.int32(0),
                                    'player_battle_transition': jnp.int32(0),
                                    'enemy_battle_transition': jnp.int32(0),
                                    'automatic_battle_transition': jnp.int32(0),
                                    'battle_victory': jnp.bool_(False),
                                    'building_constructed': jnp.bool_(False),
                                    'turn_ended': jnp.bool_(False),
                                    'campaign_events': empty_campaign_counts(),
                                    'rest_penalty': jnp.float32(0)})
            if self.episode_reports:
                ts = ts.replace(extras={**ts.extras, 'episode_event': empty_event()})
        return ts

    def _begin_battle(self, state, key=None, random_values=None):
        """Initialize tactical state and snapshot entry strength for an eligible victory reward."""
        state = state.replace(battle_fort_defender=jnp.int32(FORT_NONE), battle_hero_fort_armor=jnp.int32(0),
                              copied=jnp.zeros(12,bool),summon_owner=jnp.full(12,-1,jnp.int32),
                              summon_saved_hp=jnp.zeros(12,jnp.int32),
                              preparation=jnp.zeros(12,bool),
                              primary_override=jnp.full(12,-1,jnp.int32),preweak_damage=jnp.full(12,-1,jnp.int32),
                              powerup=jnp.zeros(12,bool),powerup_layered=jnp.zeros(12,bool),
            powerup_long=jnp.zeros(12,bool),saved_powerup_long=jnp.zeros(12,bool),
            effect_rounds=jnp.zeros((12,5),jnp.int32),disable_source=jnp.full(12,-1,jnp.int32),
            transform_long=jnp.zeros(12,bool),
                              saved_primary_override=jnp.full(12,-1,jnp.int32),saved_preweak_damage=jnp.full(12,-1,jnp.int32),
                              saved_powerup=jnp.zeros(12,bool),saved_powerup_layered=jnp.zeros(12,bool),
                              healer_wards=jnp.zeros((12,4),jnp.uint32),ward_native_used=jnp.zeros(12,jnp.uint32),
                              saved_healer_wards=jnp.zeros((12,4),jnp.uint32),saved_ward_native_used=jnp.zeros(12,jnp.uint32),
                              fenrir=jnp.zeros(12, jnp.bool_), armor_shreds=jnp.zeros(12, jnp.int32),
                              imp=jnp.zeros(12, jnp.bool_), imp_wards=jnp.zeros(12, jnp.uint32),
                              activation_done=jnp.zeros(12, jnp.bool_),
            post_victory=jnp.int32(0),pending_healers=jnp.zeros(12,jnp.bool_),
            initiative_override=jnp.full(12,-1,jnp.int32),saved_initiative_override=jnp.full(12,-1,jnp.int32),
            slow_original=jnp.full(12,-1,jnp.int32),waited=jnp.zeros(12,bool),bonus_turns=jnp.zeros(12,jnp.int32),
            battle_revived=jnp.zeros(12,bool),battle_earned_xp=jnp.zeros(12,jnp.int32),
            feared=jnp.zeros(12,jnp.bool_),enemy_initial_hp=jnp.zeros(6,jnp.int32),
            enemy_damage_credit=jnp.zeros(6,jnp.int32),enemy_killed_credit=jnp.zeros(6,jnp.bool_),
            weakened=jnp.zeros(12,jnp.bool_),saved_weakened=jnp.zeros(12,jnp.bool_),
            item_weakened=jnp.zeros(12,jnp.bool_),saved_item_weakened=jnp.zeros(12,jnp.bool_),
            lycanthrope=jnp.zeros(12,jnp.bool_),
            paralyzed=jnp.zeros(12,jnp.bool_),long_paralyzed=jnp.zeros(12,jnp.bool_),
                              decay_form=jnp.zeros(12, jnp.int32), decay_shreds=jnp.zeros(12, jnp.int32),
                              poison_mode=jnp.zeros(12,jnp.int32),poison_damage=jnp.zeros(12,jnp.int32),
                              poison_source=jnp.full(12,-1,jnp.int32),
                              water_mode=jnp.zeros(12,jnp.int32), water_damage=jnp.zeros(12,jnp.int32),
                              water_source=jnp.full(12,-1,jnp.int32), last_water_damage=jnp.zeros(12,jnp.int32),
                              burn_mode=jnp.zeros(12,jnp.int32),burn_damage=jnp.zeros(12,jnp.int32),
                              burn_source=jnp.full(12,-1,jnp.int32),last_burn_damage=jnp.zeros(12,jnp.int32))
        if key is None:
            key, random_key = jax.random.split(state.battle_key)
            random_values = jax.random.uniform(random_key, (self.random_size,))
        if self.progression_enabled:
            enemy_progress = state.enemy_progress[state.enemy]
            state = state.replace(unit_ids=state.unit_ids.at[6:].set(self.enemy_unit_ids(state)[state.enemy]),
                                  unit_levels=state.unit_levels.at[6:].set(enemy_progress[:, 0]),
                                  unit_xp=state.unit_xp.at[6:].set(enemy_progress[:, 1]),
                                  battle_xp=jnp.zeros(2, jnp.int32),
                                  last_xp=jnp.zeros(12, jnp.int32), last_promoted=jnp.uint32(0))
        if self.territory_enabled:
            defended_fort = self.territory.defender_in_fort(state)
            state = state.replace(battle_fort_armor=self.territory.defender_armor(state))
            if self.bot.enabled:
                at_home = jnp.all(state.enemy_positions[self.bot.enemy] == self.bot.home)
                defended_fort |= (state.enemy == self.bot.enemy) & at_home
                state = state.replace(battle_fort_armor=jnp.where(
                    (state.enemy == self.bot.enemy) & at_home, 50, state.battle_fort_armor))
            state = state.replace(battle_fort_defender=jnp.where(defended_fort, FORT_RED, FORT_NONE))
        if self.spell_casting_enabled:
            state, spell_hp = self.casting.begin_battle(state, state.hp)
            state = state.replace(hp=spell_hp)
        priority = self._round_priority(random_values, state.enemy, state)
        enemy_hp = jnp.where(jnp.arange(6) < self.enemy_counts[state.enemy],
                             self.enemy_health[state.enemy], 0)
        if self.basic_combat:
            enemy_hp = self._combat_stats(state)[6:, HP].astype(jnp.int32)
        if self.spell_casting_enabled:
            enemy_hp = jnp.maximum(enemy_hp - state.enemy_wounds[state.enemy], 0)
        hp = jnp.concatenate((state.hp[:6] if self.basic_combat else self.hero_full, enemy_hp))
        actor = jnp.argmax(jnp.where(hp > 0, priority, -100)).astype(jnp.int32)
        state = state.replace(
            battle_key=key, in_battle=jnp.bool_(True), hp=hp, priority=priority,
            enemy_initial_hp=enemy_hp,
            activation_done=state.activation_done.at[actor].set(True),
            wards_used=jnp.zeros(12, jnp.uint32), last_immune=jnp.uint32(0), last_ward=jnp.uint32(0),
            turn_phase=jnp.zeros(12, jnp.int32), defended=jnp.zeros(12, bool),
            retreating=jnp.zeros(12, bool), escaped=jnp.zeros(12, bool),
            actor=actor, round=jnp.int32(1), last_event=jnp.int32(ENGAGE),
            last_actor=jnp.int32(-1), last_target=jnp.int32(-1), last_damage=jnp.int32(0),
        )
        if self.has_copies or self.has_summons or self.has_item_summons:
            state = state.replace(native_ids=state.unit_ids,native_levels=state.unit_levels,native_xp=state.unit_xp)
        if self.battle_items_enabled:
            state = self.battle_item_rules.begin_battle(state)
        if self.has_copies:
            pending = self.progression.doppelgangers[state.unit_ids] & (hp > 0)
            # All uncopied doppelgangers are excluded as targets independently of actor.
            targets = (hp > 0) & (self.progression.sizes[state.unit_ids] == 1) & ~self.progression.copy_forbidden[state.unit_ids] & ~pending
            pending &= jnp.any(targets)
            preparing = jnp.any(pending)
            state = state.replace(preparation=pending,round=jnp.where(preparing,0,1),
                actor=jnp.where(preparing,jnp.argmax(jnp.where(pending,priority,-1e9)),actor).astype(jnp.int32),
                activation_done=jnp.where(preparing,False,state.activation_done))
        return state

    def _world_step(self, state, action, key, random_values, spell_targets=None):
        """Apply one strategic command, including REST growth, bot movement, and queued encounters."""
        before = state
        state = state.replace(last_ruin=jnp.int32(-1), last_sale_gold=jnp.int32(0), last_sold_count=jnp.int32(0), last_trade_item=jnp.int32(-1),
                              last_poison_damage=jnp.zeros(12, jnp.int32),
                              last_item_loot=jnp.zeros_like(state.last_item_loot),
                              last_loot=jnp.zeros_like(state.last_loot),
                              last_water_damage=jnp.zeros(12,jnp.int32),
                              last_burn_damage=jnp.zeros(12,jnp.int32))
        destination = state.position + self.directions[jnp.clip(action, 0, 7)]
        # A directional command aimed at an occupied tile attacks that exact stack.
        # The attacker stays on its own tile, including after victory/retreat.
        if self.territory_enabled:
            enemy = self.territory.enemy_at(state, destination)
            engage = (action < 8) & (enemy>=0)
        else:
            targeted = state.alive & jnp.all(self.opponent_positions == destination, axis=1)
            enemy = jnp.argmax(targeted).astype(jnp.int32)
            engage = (action < 8) & jnp.any(targeted)
        moved = (action < 8) & ~engage  # step() checks bounds and movement points.
        position = jnp.where(moved, destination, state.position)
        map_steps = state.map_steps + moved.astype(jnp.int32)
        resting = (action == REST) if self.basic_combat else jnp.bool_(False)
        gold = add_resource(state.gold, jnp.where(resting & ~jnp.bool_(self.territory_enabled), DAILY_GOLD, 0))
        next_state = state.replace(position=position, origin=state.position,
                                   gold=gold, map_steps=map_steps, last_service_cost=jnp.int32(0),
                                   last_captured_cities=jnp.zeros_like(state.city_owned),last_upgraded_city=jnp.int32(-1),
                                   last_xp=jnp.zeros(12, jnp.int32), last_promoted=jnp.uint32(0),
                                   last_immune=jnp.uint32(0), last_ward=jnp.uint32(0),
                                   enemy=jnp.where(engage, enemy, state.enemy),
                                   last_event=jnp.int32(MOVE), last_actor=jnp.int32(-1),
                                   last_target=jnp.int32(-1), last_damage=jnp.int32(0))
        bonus = jnp.float32(0)
        if self.basic_combat:
            rest_max = self.max_hp(state) if self.progression_enabled else self.restored_hp
            rest_healing = (rest_max+9)//10 if self.progression_enabled else self.rest_healing
            if self.items_enabled:
                rest_healing = (rest_max*(1+self.item_rules.value(state,'regeneration',2))+9)//10
            if self.territory_enabled:
                banner = self.item_rules.value(state,'regeneration',2) if self.items_enabled else 0
                percent = self.territory.regeneration_percent(state,banner)
                rest_healing = (rest_max*percent+99)//100
            building = jnp.clip(action - BUILD_START, 0, BUILD_SLOTS - 1)
            building_action = (action >= BUILD_START) & (action < REST)
            next_state = next_state.replace(
                gold=gold-jnp.where(building_action, self.construction.costs[building], 0),
                buildings=state.buildings | jnp.where(building_action, self.construction.bits[building], jnp.uint32(0)),
                blocked_buildings=state.blocked_buildings | jnp.where(
                    building_action, self.construction.blocks[building], jnp.uint32(0)),
                built_today=~resting & (state.built_today | building_action),
                spell_researched_today=~resting & state.spell_researched_today,
                last_researched_spell=jnp.int32(-1),
                movement_points=jnp.where(resting, self.movement_cap(state),
                                          jnp.maximum(0, state.movement_points - jnp.where(
                                              engage, (self.movement_cap(state)+1)//2, moved.astype(jnp.int32) * self.move_cost(state, destination)))),
                day=state.day + resting.astype(jnp.int32),
                hp=jnp.where(resting & (state.hp > 0),
                             jnp.minimum(state.hp + rest_healing, rest_max), state.hp),
                last_building=jnp.where(building_action, building, -1),
                last_event=jnp.where(resting, RESTED, jnp.where(building_action, BUILD, MOVE)),
            )
            bonus += jnp.where(building_action, jnp.float32(self.construction.reward), 0.)
            bonus -= jnp.where(resting, self.rest_penalty(state), 0.)
        summon = jnp.bool_(False)
        if self.territory_enabled:
            next_state = jax.lax.cond(resting,self.territory.next_turn,lambda s:s,next_state)
            if self.spell_casting_enabled:
                next_state = self.casting.rest(next_state, resting)
                next_state, cast_bonus, summon = self.casting.apply(
                    next_state, action, rest_max, self.movement_cap(state), spell_targets)
                next_state, engage_bonus = self.casting.engage_bonus(next_state, engage)
                bonus += cast_bonus + engage_bonus
            next_state = self.territory.upgrade(next_state,action)
            next_state = self._capture_cities(next_state,moved)
            if self.bot.enabled:
                next_state = self.bot.sync_defeat(next_state)
            if not self.position_objective:
                won = self._campaign_won(next_state)
                next_state = next_state.replace(won=won,done=next_state.done|won)
                bonus += self._objective_reward(won, jnp.bool_(False))
        if self.spell_research_enabled:
            next_state, research_bonus = self.spell_research.apply(next_state, action)
            bonus += research_bonus
        if self.capital_enabled:
            next_state, recovery_bonus = self.capital.apply(next_state, action, rest_max)
            bonus += recovery_bonus
        if self.potions_enabled:
            next_state = self.potion_rules.expire(next_state,resting)
            potion_slot=jnp.clip(action-POTION_START,0,max(0,self.potion_rules.action_count-1))%6
            next_state = self.potion_rules.apply(next_state, action, rest_max,
                self._potion_intrinsic(state.unit_ids[potion_slot],state.unit_levels[potion_slot]))
        if self.chests_enabled:
            next_state = self.chest_rules.collect(next_state, moved)
        if self.sites_enabled:
            next_state = self.site_rules.apply(next_state, action)
            next_state = self.site_rules.autosell(next_state)
        if self.recruitment_enabled:
            next_state, hire_bonus = self.recruitment.apply(next_state, action, self._clear_roster_slots)
            bonus += hire_bonus
        if self.unit_swaps_enabled:
            next_state, swap_bonus = self.unit_swaps.apply(next_state, action, resting)
            bonus += swap_bonus
        if self.script_events.enabled:
            next_state = self.script_events.process(self, next_state, new_day=resting,
                                                    locations=~(engage | summon))
        if self.items_enabled:
            next_state = self.item_rules.collect_days(next_state, resting)
            changed=resting | jnp.any(next_state.last_item_loot>0) | (next_state.last_potion>=0) | (action>=REVIVE_START)&(action<REVIVE_START+6)
            if self.recruitment_enabled:
                changed |= (next_state.last_event == HIRED) | (next_state.last_event == DISMISSED)
            if self.unit_swaps_enabled:
                changed |= next_state.last_event == UNITS_SWAPPED  # the equipment holder may change
            if self.sites_enabled and self.site_rules.item_offers:
                changed |= next_state.last_event == BOUGHT  # a purchase may beat the worn item
            next_state=jax.lax.cond(changed,self._refresh_equipment,lambda s:s,next_state)
        if self.battle_items_enabled and self.battle_item_rules.count:
            rules = self.battle_item_rules
            next_state = rules.rest(rules.sync(self, next_state), resting)
            next_state = rules.equip(self, next_state.replace(last_battle_item=jnp.int32(-1)), action)
        if self.exploration_bonus:
            cell = position[0] * self.size + position[1]
            word, bit = cell // 32, jnp.left_shift(jnp.uint32(1), (cell % 32).astype(jnp.uint32))
            first = (state.visited[word] & bit) == 0
            bonus += jnp.where(first & moved, jnp.float32(self.exploration_bonus), 0.)
            next_state = next_state.replace(visited=state.visited.at[word].set(state.visited[word] | jnp.where(moved, bit, jnp.uint32(0))))
        if self.scenario.enabled:
            next_state = jax.lax.cond(resting, self.scenario.advance, lambda s: s, next_state)
        if self.script_events.enabled:
            next_state = jax.lax.cond(resting & ~next_state.done,
                                      self.script_events.advance, lambda s: s, next_state)
        if self.bot.enabled:
            # Under vmap the cond runs for every environment; only a real map
            # REST may enter the bot's search and auto-battle loops.
            next_state = jax.lax.cond(resting, lambda s: self.bot.advance(self, s, resting & ~before.in_battle),
                                      lambda s: s, next_state)
            # Autoresolution consumes the same device PRNG stream; do not reuse
            # the old world-action key for a subsequent encounter.
            key, random_key = jax.random.split(next_state.battle_key)
            random_values = jax.random.uniform(random_key, (self.random_size,))
            next_state = next_state.replace(battle_key=key)
        if self.scenario.enabled or self.bot.enabled or self.script_events.enabled:
            pending = next_state.pending_encounters & next_state.alive
            attacked = jnp.any(pending) & ~(engage | summon) & ~next_state.done
            enemy = jnp.argmax(pending)
            next_state = next_state.replace(enemy=jnp.where(attacked, enemy, next_state.enemy),
                pending_encounters=next_state.pending_encounters.at[enemy].set(
                    next_state.pending_encounters[enemy] & ~attacked))
            engage |= attacked
        next_state = jax.lax.cond(engage | summon, lambda s: self._begin_battle(s, key, random_values), lambda s: s, next_state)
        if self.scenario.enabled or self.bot.enabled or self.script_events.enabled:
            inside_fort = (jnp.all(next_state.position == jnp.array(self.territory.capital))
                           | jnp.any(self.territory.city_at(next_state) & next_state.city_owned))
            fort = jnp.where(jnp.all(next_state.position == jnp.array(self.territory.capital)), 50,
                jnp.max(jnp.where(self.territory.city_at(next_state) & next_state.city_owned,
                                  self.territory.fort_bonus[next_state.city_levels], 0), initial=0))
            next_state = next_state.replace(battle_hero_fort_armor=jnp.where(attacked, fort, 0),
                battle_fort_armor=jnp.where(attacked, 0, next_state.battle_fort_armor),
                battle_fort_defender=jnp.where(attacked,
                    jnp.where(inside_fort, FORT_BLUE, FORT_NONE), next_state.battle_fort_defender))
        if self.spell_casting_enabled:
            next_state = next_state.replace(last_event=jnp.where(summon, SUMMON_CAST, next_state.last_event))
        if self.position_objective:
            # Resolve events, the enemy turn and its battle queue before checking
            # arrival. Reaching the flag cannot suppress a pending ambush attack.
            won = self._campaign_won(next_state)
            next_state = next_state.replace(won=won, done=next_state.done | won)
            bonus += self._objective_reward(won, jnp.bool_(False))
        if self.reference_rewards:
            next_state, bonus = reference_rewards.world(self, before, next_state, action, resting)
        if self.army_victory_level_reward or self.capital_training.tracks_entry:
            # Snapshot before automatic enemy attacks. Wounded living fighters
            # still count; empty/dead slots and separate summoned armies do not.
            living = (next_state.hp[:6] > 0) & (next_state.unit_ids[:6] != 0)
            # Compare each fighter to level 1, including hired elites.
            # Filling slots with level-1 recruits adds no victory reward.
            developed = jnp.maximum(next_state.unit_levels[:6] - 1, 0)
            strength = jnp.sum(jnp.where(living, developed, 0))
            if self.army_victory_level_reward:
                enemy = jnp.clip(next_state.enemy, 0, self.num_opponents-1)
                contact = (engage & next_state.in_battle & ~next_state.summon_battle
                           & (next_state.enemy >= 0) & self.army_victory_level_targets[enemy])
                # This separate victory bonus keeps a fresh snapshot per attempt.
                # Temporary transformations, battle losses and XP cannot change it.
                next_state = next_state.replace(army_battle_level_strength=jnp.where(contact, strength, 0))
            if self.capital_training.tracks_entry:
                next_state, entry_bonus = self.capital_training.enter(next_state, engage, strength)
                bonus += entry_bonus
        return next_state, bonus

    def _enemy_action(self, state, random_values):
        valid = (state.hp[:6] > 0) & ~state.escaped[:6]
        base_damage = self.damage
        if self.basic_combat:
            valid &= ~self._actor_is_melee(state) | self._melee_targets(state, enemy_side=True)
            traits = self._combat_traits(state)
            source = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[state.actor, 1], 1)-1)
            # Python reference: ordinary attacks choose the lowest-HP reachable
            # non-immune opponent. Wards do not disqualify a target. Area spells
            # still act when everyone is immune; the attack resolves each target.
            valid &= (traits[state.actor, 0] == AREA) | ((traits[:6, 2] & source) == 0)
        elif self.has_enemy_warriors:
            warrior = self._enemy_is_warrior(state)
            valid &= ~warrior | self._melee_targets(state, enemy_side=True)
            base_damage = jnp.where(warrior, self.warrior_damage, base_damage)
        damage = jnp.where(state.defended[:6], (base_damage + 1) // 2, base_damage)
        kill = valid & (state.hp[:6] <= damage)
        candidates = valid if self.basic_combat else valid & jnp.where(jnp.any(kill), kill, True)
        # Uniform random keys break equal-HP ties without consulting hit rolls.
        score = jnp.where(candidates, state.hp[:6] + random_values[:6] * .5, 1e9)
        attack = SHOOT + jnp.argmin(score).astype(jnp.int32)
        action = jnp.where(jnp.any(valid), attack, DEFEND) if self.basic_combat or self.has_enemy_warriors else attack
        if self.has_secondary_paralysis:
            candidates = valid & ~state.paralyzed[:6] & ~state.long_paralyzed[:6]
            # Reference ranks stored damage, before the battle damage cap.
            raw = self._combat_stats(state,cap=False)[:6,DAMAGE]
            score = jnp.where(candidates,-raw+random_values[:6]*.5,1e9)
            smart = jnp.where(jnp.any(candidates),SHOOT+jnp.argmin(score),action)
            action = jnp.where((self._actor_secondary_paralysis(state) == 2) & (traits[state.actor,0] != AREA),smart,action)
        if self.has_witches or self.has_paralysis or self.has_fear:
            living = (state.hp[:6] > 0) & ~state.escaped[:6]
            weapon_immune = living & ((self._combat_traits(state)[:6, 2] & jnp.uint32(1)) != 0)
            prefer_immune = jnp.any(weapon_immune) & ~self._actor_is_fear_caster(state)
            candidates = jnp.where(prefer_immune, weapon_immune, living)
            score = jnp.where(candidates, jnp.where(prefer_immune, 0., -state.hp[:6]) + random_values[:6]*.5, 1e9)
            witch_target = jnp.where(jnp.any(living), SHOOT+jnp.argmin(score), DEFEND)
            action = jnp.where(self._actor_is_witch(state) | (self._actor_ghost_mode(state) > 0) | self._actor_is_fear_caster(state), witch_target, action)
        if self.basic_combat and self.has_healers:
            # Reference: lowest absolute HP among wounded living allies, self included.
            wounded = (state.hp[6:] > 0) & ~state.escaped[6:] & (state.hp[6:] < self._combat_stats(state)[6:, HP])
            score = jnp.where(wounded, state.hp[6:] + random_values[:6] * .5, 1e9)
            heal = jnp.where(jnp.any(wounded), SHOOT + jnp.argmin(score).astype(jnp.int32), DEFEND)
            if self.has_healer_wards:
                # A single-target ward healer (Deva roshi) still bestows its wards
                # when nobody is wounded: lowest-HP living ally, caster included.
                living = (state.hp[6:] > 0) & ~state.escaped[6:]
                warder = self._effect_value(state, 'healer_ward_elements')[state.actor] > 0
                spare = jnp.where(living, state.hp[6:] + random_values[:6] * .5, 1e9)
                heal = jnp.where(~jnp.any(wounded) & warder & jnp.any(living),
                                 SHOOT + jnp.argmin(spare).astype(jnp.int32), heal)
            heal = jnp.where(self._actor_mass_healer(state),SHOOT,heal)
            action = jnp.where(self._actor_is_healer(state), heal, action)
            if self.has_powerups:
                living = (state.hp[6:] > 0) & ~state.escaped[6:] & (self.slots[6:] != state.actor)
                raw = self._combat_stats(state,cap=False)[6:,DAMAGE]
                score = jnp.where(living,-raw+random_values[:6]*.5,1e9)
                buff = jnp.where(jnp.any(living),SHOOT+jnp.argmin(score),DEFEND)
                action = jnp.where(self._actor_power_factor(state) > 0,buff,action)
        if self.has_patriarchs:
            valid = self._patriarch_targets(state)[6:]
            dead = valid & (state.hp[6:] <= 0)
            wounded = valid & (state.hp[6:] > 0) & (state.hp[6:] < self._combat_stats(state)[6:,HP])
            targets = jnp.where(jnp.any(dead),dead,wounded)
            score = jnp.where(targets,jnp.where(jnp.any(dead),0.,state.hp[6:])+random_values[:6]*.5,1e9)
            support = jnp.where(jnp.any(targets),SHOOT+jnp.argmin(score),DEFEND)
            action = jnp.where(self._actor_is_patriarch(state),support,action)
        if self.has_alchemists:
            raw = self._combat_stats(state,cap=False)[6:]
            current = jnp.where(state.turn_phase[6:] == 0,state.priority[6:],0.)
            valid = self._alchemist_targets(state)[6:] & (current < raw[:,INITIATIVE]) & (raw[:,DAMAGE] > 0)
            score = jnp.where(valid,-raw[:,DAMAGE]+random_values[:6]*.5,1e9)
            support = jnp.where(jnp.any(valid),SHOOT+jnp.argmin(score),DEFEND)
            action = jnp.where(self._alchemist_units(state)[state.actor],support,action)
        if self.has_copies or self.has_summons:
            action = self._special_enemy_action(state,action,random_values)
        return action

    def _effect_value(self,state,name):
        return (getattr(self.progression,name)[self._effective_ids(state)] if self.progression_enabled
                else getattr(self,name)[state.enemy])

    def _record_effect(self,state,slots,selected,kind):
        column = state.effect_rounds[:,kind]
        column = set_rows(column,slots,jnp.where(selected,state.round,column[slots]))
        return state.replace(effect_rounds=set_column(state.effect_rounds,kind,column))

    def _record_disable(self,state,slots,selected):
        state = self._record_effect(state,slots,selected,DISABLE)
        return state.replace(disable_source=set_rows(state.disable_source,slots,
            jnp.where(selected,state.actor,state.disable_source[slots])))

    def _prepare_disable(self,state,roll,enabled):
        # Finite effects also expire when their caster starts its next turn.
        expired = enabled & (state.disable_source == state.actor)
        recover = enabled & effect_breaks(state.round,state.effect_rounds[state.actor,DISABLE],roll)
        return state.replace(paralyzed=state.paralyzed & ~expired,
            long_paralyzed=set_rows(state.long_paralyzed,state.actor,state.long_paralyzed[state.actor] & ~recover))

    def _cleanse(self,state,hp,wards_used,mask):
        """Cure living recipients, restoring original forms before healing."""
        if not self.has_cures:
            return state,hp,wards_used
        # Original Cure removes six hostile classes; Drain Level, Slow,
        # Lower Damage, Shatter and fleeing are deliberately not dispelled.
        transformed = mask & (state.imp | state.lycanthrope)
        state = self._restore_initiative_form(state,transformed)
        natural = (self._natural_stats(state) if self.progression_enabled
                   else self.stats_table[state.enemy])
        if self.has_copies:
            natural = jnp.where(state.copied[:,None],state.copy_stats,natural)
        maximum = jnp.where(state.fenrir,275.,natural[:,HP])
        if self.has_wights:
            current_max = self.progression.base_stats[:,HP][state.decay_form]
            restored = jnp.clip(jnp.rint(hp/jnp.maximum(current_max,1)*maximum).astype(jnp.int32),1,MAX_HP)
            hp = jnp.where(transformed & (state.decay_form > 0),restored,hp)
        wards_used = jnp.where(transformed,state.imp_wards,wards_used)
        state = self._restore_damage_forms(state,transformed)
        state = self._restore_ward_forms(state,transformed)
        priority = jnp.where(transformed & (state.turn_phase < 2) & (state.turn_phase != 1),
            jnp.where(state.initiative_override >= 0,state.initiative_override,jnp.where(state.fenrir,65.,natural[:,INITIATIVE])),state.priority)
        updates = dict(imp=state.imp & ~mask,decay_form=jnp.where(transformed,0,state.decay_form),
            decay_shreds=jnp.where(transformed,0,state.decay_shreds),
            weakened=jnp.where(transformed,state.saved_weakened,state.weakened),
            item_weakened=jnp.where(transformed,state.saved_item_weakened,state.item_weakened),
            lycanthrope=state.lycanthrope & ~mask,transform_long=state.transform_long & ~transformed,
            paralyzed=state.paralyzed & ~mask,long_paralyzed=state.long_paralyzed & ~mask,
            disable_source=jnp.where(mask,-1,state.disable_source),priority=priority)
        for kind in ('poison','water','burn'):
            updates[kind+'_mode'] = jnp.where(mask,0,getattr(state,kind+'_mode'))
            updates[kind+'_damage'] = jnp.where(mask,0,getattr(state,kind+'_damage'))
            updates[kind+'_source'] = jnp.where(mask,-1,getattr(state,kind+'_source'))
        return state.replace(**updates),hp,wards_used

    def _battle_step(self, state, action, key, random_values, with_metrics=False):
        state = state.replace(last_sale_gold=jnp.int32(0), last_sold_count=jnp.int32(0), last_trade_item=jnp.int32(-1))
        if not self.has_copies:
            return self._battle_step_regular(state,action,key,random_values,with_metrics=with_metrics)
        # Round 0 prepares copies, later rounds resolve an activation. Both sides need one
        # special-action call; it is made once, on the inputs of the environment's own round.
        # Batched environments evaluate both sides of the cond (a vmapped cond is a select)
        # and share that call; a single environment runs only its own side.
        preparing = state.round == 0
        prelude = self._activation_prelude(state, action, random_values)
        prepare_action = jnp.where(state.actor >= 6,
                                   self._special_enemy_action(state, DEFEND, random_values[30:36]), action)
        special = self._special_action(jax.tree.map(lambda a, b: jnp.where(preparing, a, b), state, prelude[0]),
                                       jnp.where(preparing, prepare_action, prelude[1]), random_values)

        def prepare(_):
            prepared = self._prepare_copies(state, action, key, random_values, special)
            return (*prepared, jnp.zeros(4, jnp.int32)) if with_metrics else prepared

        def regular(_):
            return self._battle_step_regular(state, action, key, random_values, with_metrics, prelude, special)
        return jax.lax.cond(preparing, prepare, regular, None)

    def _activation_prelude(self, state, action, random_values):
        """The acting unit's skip/flee status and its effective command, before special actions."""
        actor = state.actor
        post_turn = state.post_victory > 0
        can_retreat = self._can_retreat(state, actor)
        skipping = (state.paralyzed[actor] | state.long_paralyzed[actor]) & ~post_turn
        if self.has_paralysis:
            state = state.replace(paralyzed=set_rows(state.paralyzed, actor, state.paralyzed[actor] & post_turn))
        escaping = state.retreating[actor] & can_retreat & ~skipping & ~post_turn
        action = jnp.where(actor >= 6, self._enemy_action(
            state, random_values[30:36] if self.basic_combat else random_values[1:]), action)
        # Public step rejects this command; internal AI/autobattles must also obey.
        action = jnp.where((action == RETREAT) & ~can_retreat, DEFEND, action)
        action = jnp.where(escaping | skipping, CONTINUE, action)
        return state, action, post_turn, can_retreat, skipping, escaping

    def _battle_step_regular(self, state, action, key, random_values, with_metrics=False, prelude=None, special=None):
        """Resolve one tactical activation, its aftermath, and optional combat event counters.

        prelude/special: _activation_prelude and _special_action results already computed by
        _battle_step for these inputs.
        """
        actor = state.actor
        state, action, post_turn, can_retreat, skipping, escaping = (
            self._activation_prelude(state, action, random_values) if prelude is None else prelude)
        special_event = jnp.int32(-1)
        if special is not None:
            state,action,special_event = special
        elif self.has_copies or self.has_summons:
            state,action,special_event = self._special_action(state,action,random_values)
        attack = (action >= SHOOT) & (action < DEFEND)
        healer = self._actor_is_healer(state) if self.basic_combat and self.has_healers else jnp.bool_(False)
        target = jnp.clip(action - SHOOT, 0, 5) + jnp.where((actor < 6) ^ healer, 6, 0)
        accuracy, single_damage = self.accuracy, self.damage
        if self.warrior_slot >= 0 or self.has_enemy_warriors:
            warrior = actor == self.warrior_slot
            if self.has_enemy_warriors:
                warrior |= self._enemy_is_warrior(state)
            accuracy = jnp.where(warrior, self.warrior_accuracy, accuracy)
            single_damage = jnp.where(warrior, self.warrior_damage, single_damage)
        hit = attack & (random_values[0] < accuracy)
        area_attack = jnp.bool_(False)
        wards_used = state.wards_used
        immune_slots, ward_slots = jnp.uint32(0), jnp.uint32(0)
        transformed_slots = jnp.zeros(6, jnp.bool_)
        decayed_slots = jnp.zeros(6, jnp.bool_)
        paralyzed_slots = jnp.zeros(6,jnp.bool_)
        feared_slots = jnp.zeros(6,jnp.bool_)
        damage_credit = jnp.zeros(6,jnp.int32)
        revived = jnp.zeros(12,bool)
        extra_turn = jnp.zeros(12,bool)
        slowed = jnp.zeros(6,bool)
        if self.basic_combat:
            effective_ids = self._effective_ids(state) if self.progression_enabled else None
            effective_levels = self._effective_levels(state) if self.progression_enabled else None
            stats = self._combat_stats(state)
            target_slots = jnp.arange(6) + jnp.where(actor < 6, 6, 0)
            traits = self._combat_traits(state)
            area_attack = traits[actor, 0] == AREA
            targets = ((area_attack | (target_slots == target))
                       & (state.hp[target_slots] > 0) & ~state.escaped[target_slots])
            accuracy = stats[actor, ACCURACY]
            if self.has_aoe_accuracy_falloff:
                falloff = (self.progression.aoe_accuracy_falloff[effective_ids[actor]]
                           if self.progression_enabled else self.aoe_accuracy_falloff[state.enemy, actor])
                index = jnp.maximum(jnp.cumsum(targets.astype(jnp.int32))-1, 0)
                accuracy = jnp.maximum(0., accuracy-jnp.where(area_attack & falloff, index*10, 0))
            hits = accuracy_hits(accuracy, random_values[12:18], random_values[18:24])
            bonuses = jnp.minimum((random_values[24:30] * 6).astype(jnp.int32), 5)
            if self.progression_enabled:
                damage = self.progression.damage(
                    state.replace(unit_ids=effective_ids, unit_levels=effective_levels), actor, target_slots, state.defended[target_slots], bonuses,
                                                 stats[actor, DAMAGE], self._effective_shreds(state) if self.has_shatterers else None,
                                                 state.imp if self.has_witches else None,
                                                 stats[target_slots,ARMOR] if self.has_potion_buffs or self.items_enabled or self.has_spell_mods else None)
            else:
                cap = jnp.where(state.imp[actor] | state.fenrir[actor], 300., self.damage_caps[state.enemy, actor])
                bonuses = capped_damage_bonus(stats[actor, DAMAGE], bonuses, cap)
                damage_id = self.damage_ids[state.enemy, actor]
                if self.has_wolf_lord:
                    damage_id = jnp.where(state.fenrir[actor], self.combat_info['fenrir_damage_id'], damage_id)
                if self.has_witches:
                    imp_damage = jnp.where(self.size_table[state.enemy, actor] == 2,
                                          self.combat_info['imp_damage_ids'][1], self.combat_info['imp_damage_ids'][0])
                    damage_id = jnp.where(state.imp[actor], imp_damage, damage_id)
                armor_id = (self.shatter_armor_ids[state.enemy, target_slots, jnp.minimum(state.armor_shreds[target_slots], 6)]
                            if self.has_shatterers else self.armor_ids[state.enemy, target_slots])
                if self.has_witches:
                    armor_id = jnp.where(state.imp[target_slots], 0, armor_id)
                damage = self.damage_rolls[damage_id, armor_id,
                                           state.defended[target_slots].astype(jnp.int32), bonuses]
            if (self.has_weakening or self.has_powerups) and not self.progression_enabled:
                armor = armor_with_defend(stats[target_slots,ARMOR], state.defended[target_slots])
                damage = jnp.where(stats[actor,DAMAGE] > 0,
                                   damage_after_armor(stats[actor,DAMAGE]+bonuses, armor), 0)
            connected = attack & ~healer & targets & hits
            ghost_mode = self._actor_ghost_mode(state)
            # Accuracy zero can pass the original roll; the round-40 ban must
            # reject the attack explicitly, before spending a target's ward.
            connected &= disable_attack_allowed(ghost_mode > 0, state.round)
            # AOE status-only attacks reject an existing matching flag before
            # checking wards; single Ghost attacks keep their primary hit path.
            already = jnp.where(ghost_mode == 2,state.long_paralyzed[target_slots],state.paralyzed[target_slots])
            connected &= ~(area_attack & (ghost_mode > 0) & already)
            effective = connected
            check_primary = connected & ~(area_attack & self._actor_is_witch(state))
            if self.has_protections:
                source_bit = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[actor, 1], 1)-1)
                immune = check_primary & ((traits[target_slots, 2] & source_bit) != 0)
                warded = check_primary & ~immune & ((traits[target_slots, 3] &
                                                  ~wards_used[target_slots] & source_bit) != 0)
                wards_used = set_rows(wards_used, target_slots,
                    wards_used[target_slots] | jnp.where(warded, source_bit, jnp.uint32(0)))
                slot_bits = jnp.left_shift(jnp.uint32(1), target_slots.astype(jnp.uint32))
                immune_slots = jnp.sum(jnp.where(immune, slot_bits, jnp.uint32(0)))
                ward_slots = jnp.sum(jnp.where(warded, slot_bits, jnp.uint32(0)))
                effective &= ~immune & ~warded
            ghost_mode = self._actor_ghost_mode(state)
            damage = jnp.where(self._actor_is_witch(state) | (ghost_mode > 0) | self._actor_is_fear_caster(state), 0, damage)
            removed = jnp.where(effective, jnp.minimum(damage, state.hp[target_slots]), 0)
            if self.has_centaurs:
                centaur = (self.progression.centaurs[effective_ids[actor]] if self.progression_enabled
                           else self.centaurs[state.enemy, actor]) & ~state.imp[actor]
                # Reference: a connected hit adds 5% of effective base damage
                # only to a survivor, bypassing armour, defend and secondary
                # POWER/source checks. User: round this addition to integer HP.
                extra = jnp.rint(stats[actor, DAMAGE] / 20.).astype(jnp.int32)
                extra = jnp.minimum(extra, state.hp[target_slots]-removed)
                removed += jnp.where(centaur & effective, extra, 0)
            damage_credit = jnp.where(actor < 6,removed,0)
            if self.has_paralysis:
                # Ghost uses its primary source and primary accuracy only; those
                # immunity/ward checks were already applied to effective.
                paralyzed_slots = effective & (ghost_mode > 0) & ~jnp.where(ghost_mode == LONG,state.long_paralyzed[target_slots],state.paralyzed[target_slots])
                state = self._record_disable(state,target_slots,paralyzed_slots)
                state = state.replace(
                    paralyzed=set_rows(state.paralyzed, target_slots, state.paralyzed[target_slots] | (paralyzed_slots & (ghost_mode == 1))),
                    long_paralyzed=set_rows(state.long_paralyzed, target_slots, state.long_paralyzed[target_slots] | (paralyzed_slots & (ghost_mode == 2))))
            zeros = jnp.zeros(6, jnp.int32)
            hp = state.hp - jnp.where(actor < 6, jnp.concatenate((zeros, removed)),
                                     jnp.concatenate((removed, zeros)))
            applied = jnp.sum(removed)
            leech_mode = jnp.int32(0)
            if self.has_leech:
                leech_mode = (self.progression.leech_modes[effective_ids[actor]] if self.progression_enabled
                              else self.leech_modes[state.enemy,actor])
                pool = jnp.where((leech_mode > 0) & ~state.imp[actor], applied//2, 0)
                hp = vampiric_heal(hp,stats[:,HP],state.escaped,actor,pool,leech_mode == 2)
            hit = attack & jnp.any(targets & hits)
            if self.has_healers:
                # An L_ALL support (GAttacks REACH=1) heals, cures and wards every
                # living ally, caster included: Discipl2.exe target lists and the
                # heal/cure/ward checks never exclude the attacker (see
                # support_reaches_all). Only the two native Profit cure forms
                # cleanse; the ordinary Cleric does not remove effects.
                mass = self._actor_mass_healer(state)
                allies = (self.slots < 6) == (actor < 6)
                recipients = allies & (hp > 0) & ~state.escaped & (mass | (self.slots == target))
                cure_caster = (self.progression.mass_cures[effective_ids[actor]] if self.progression_enabled
                               else self.mass_cures[state.enemy,actor])
                cleanse = recipients & attack & healer & cure_caster
                state,hp,wards_used = self._cleanse(state,hp,wards_used,cleanse)
                # Cure can restore a Wight form with a different maximum HP.
                maximum = self._combat_stats(state)[:,HP] if self.has_cures else stats[:,HP]
                healed = jnp.where(attack & healer & recipients,
                    jnp.minimum(stats[actor,DAMAGE],jnp.maximum(maximum-hp,0)),0).astype(jnp.int32)
                hp += healed
                if self.has_alchemists:
                    extra_turn = self._alchemist_targets(state) & (self.slots == target) & attack & self._alchemist_units(state)[actor]
                    state = state.replace(turn_phase=jnp.where(extra_turn,0,state.turn_phase),
                        priority=jnp.where(extra_turn,stats[:,INITIATIVE],state.priority),
                        bonus_turns=state.bonus_turns+extra_turn.astype(jnp.int32))
                if self.has_patriarchs:
                    revived = self._patriarch_targets(state) & (state.hp <= 0) & (self.slots == target) & attack & self._actor_is_patriarch(state)
                    restored = jnp.maximum(1,jnp.rint(stats[:,HP]*.5)).astype(jnp.int32)
                    hp = jnp.where(revived,restored,hp)
                    state = state.replace(battle_revived=state.battle_revived | revived,
                        battle_earned_xp=jnp.where(revived,0,state.battle_earned_xp),
                        turn_phase=jnp.where(revived,2,state.turn_phase),priority=jnp.where(revived,0.,state.priority),
                        retreating=state.retreating & ~revived,feared=state.feared & ~revived)
                    state,hp,wards_used = self._cleanse(state,hp,wards_used,revived)
                    healed += jnp.where(revived,hp,0)
                if self.has_powerups:
                    factor = self._actor_power_factor(state)
                    buffed = recipients & attack & healer & (factor > 0)
                    power_cure = (self.progression.power_cures[effective_ids[actor]] if self.progression_enabled
                                  else self.power_cures[state.enemy,actor])
                    state,hp,wards_used = self._cleanse(state,hp,wards_used,buffed & power_cure)
                    original = self._original_primary(state)
                    buffed &= (original > 0) & (self.slots != actor)
                    amount = jnp.rint(original*(factor-jnp.where(state.weakened,.33,0.))).astype(jnp.int32)
                    state = state.replace(primary_override=jnp.where(buffed,amount,state.primary_override),
                        powerup=state.powerup | buffed,powerup_layered=state.powerup_layered & ~buffed,
                        powerup_long=jnp.where(buffed,self._effect_value(state,'power_long')[actor],state.powerup_long))
                if self.has_healer_wards:
                    elements = (self.progression.healer_ward_elements[effective_ids[actor]] if self.progression_enabled
                                else self.healer_ward_elements[state.enemy,actor])
                    granted = recipients & attack & healer & (stats[actor,DAMAGE] > 0)
                    owners,sticky,wards_used = grant_wards(state.healer_wards,state.ward_native_used,wards_used,
                        self._base_combat_traits(state)[:,3],granted,elements,actor)
                    state = state.replace(healer_wards=owners,ward_native_used=sticky)
                applied = jnp.where(healer,jnp.sum(healed),applied)
                if self.has_powerups:
                    applied = jnp.where(attack & (factor > 0),amount[target],applied)
                area_attack |= healer & mass
                hit |= attack & healer
            if self.has_shatterers or self.has_witches or self.has_wights or self.has_poisoners or self.has_water or self.has_fire or self.has_secondary_paralysis or self.has_weakening or self.has_fear or self.has_hermits:
                secondary = (self.progression.secondary_sources[effective_ids[actor]] if self.progression_enabled
                             else self.secondary_sources[state.enemy, actor])
                guards = (self.progression.capital_guards[state.unit_ids] if self.progression_enabled
                          else self.capital_guards[state.enemy])
                secondary_fear = jnp.bool_(False)
                hermit = jnp.bool_(False)
                shatterer = jnp.bool_(False)
                wight = jnp.bool_(False)
                poisoner = jnp.bool_(False)
                water_caster = jnp.bool_(False)
                fire_caster = jnp.bool_(False)
                paralysis_mode = self._actor_secondary_paralysis(state)
                weakener = ((self.progression.weakeners[effective_ids[actor]] if self.progression_enabled else self.weakeners[state.enemy,actor]) & ~state.imp[actor]) if self.has_weakening else jnp.bool_(False)
                if self.has_shatterers:
                    shatterer = (self.progression.shatterers[effective_ids[actor]] if self.progression_enabled
                                 else self.shatterers[state.enemy, actor]) & ~state.imp[actor]
                witch = self._actor_is_witch(state)
                fear_caster = self._actor_is_fear_caster(state)
                eligible = ((shatterer & (removed > 0) & (hp[target_slots] > 0))
                            | (witch & effective)) & ~guards[target_slots]
                eligible |= fear_caster & effective & ~state.retreating[target_slots]
                if self.has_poisoners or self.has_water or self.has_fire or self.has_wights or self.has_secondary_paralysis or self.has_weakening or self.has_hermits or self.has_secondary_fear:
                    secondary_stats = (self._secondary_stats(state,actor)
                                       if self.progression_enabled else self.secondary_values[state.enemy, actor])
                    if not self.progression_enabled and self.has_secondary_paralysis:
                        secondary_stats = secondary_stats.at[1].set(disable_attack_accuracy(
                            secondary_stats[1], paralysis_mode > 0, state.round))
                    status_hit = accuracy_hits(secondary_stats[1], random_values[37:43], random_values[43:49])
                    connected_status = effective & status_hit & (hp[target_slots] > 0)
                    eligible |= weakener & connected_status
                    if self.has_secondary_fear:
                        secondary_fear = (self.progression.secondary_fear[effective_ids[actor]] if self.progression_enabled else self.secondary_fear[state.enemy,actor]) & ~state.imp[actor]
                        eligible |= secondary_fear & (secondary_stats[1] > 0) & connected_status & ~state.retreating[target_slots]
                    if self.has_secondary_paralysis:
                        eligible |= ((paralysis_mode > 0) & connected_status
                                     & disable_attack_allowed(paralysis_mode > 0, state.round)
                                     & ~state.long_paralyzed[target_slots])
                    if self.has_wights:
                        wight = self.progression.wights[effective_ids[actor]] & ~state.imp[actor]
                        eligible |= wight & connected_status & ~self.progression.neutrals[state.unit_ids[target_slots]]
                    if self.has_poisoners:
                        poisoner = (self.progression.poisoners[effective_ids[actor]] if self.progression_enabled
                                    else self.poisoners[state.enemy, actor]) & ~state.imp[actor]
                        eligible |= poisoner & connected_status & ((leech_mode != 1) | (removed > 0))
                    if self.has_water:
                        water_caster = (self.progression.water_casters[effective_ids[actor]] if self.progression_enabled
                                        else self.water_casters[state.enemy,actor]) & ~state.imp[actor]
                        eligible |= water_caster & connected_status
                    if self.has_fire:
                        fire_caster = (self.progression.fire_casters[effective_ids[actor]] if self.progression_enabled
                                       else self.fire_casters[state.enemy,actor]) & ~state.imp[actor]
                        eligible |= fire_caster & connected_status
                if self.has_hermits:
                    hermit = (self.progression.hermits[effective_ids[actor]] if self.progression_enabled else self.hermits[state.enemy,actor]) & ~state.imp[actor]
                    eligible |= hermit & connected_status
                # Mutually exclusive caster types share the same source/ward
                # pipeline; only Teurg preserves an explicitly absent source.
                effect_source = jnp.where(shatterer | weakener | hermit | (paralysis_mode > 0) | (secondary > 0), secondary, traits[actor, 1])
                affected, wards_used, blocked, warded = source_protection(
                    eligible, effect_source, traits, wards_used, target_slots)
                immune_slots |= blocked
                ward_slots |= warded
                if self.has_hermits:
                    slowed = affected & hermit & (state.slow_original[target_slots] < 0)
                    base = self._combat_stats(state)[target_slots,INITIATIVE].astype(jnp.int32)
                    state = state.replace(slow_original=set_rows(state.slow_original, target_slots, jnp.where(slowed,base,state.slow_original[target_slots])))
                if self.has_fear:
                    feared_slots = affected & (fear_caster | secondary_fear)
                    protected = self._fear_protected(state,target_slots)
                    state = self._record_disable(state,target_slots,feared_slots & protected)
                    paralyzed_slots |= feared_slots & protected & ~state.paralyzed[target_slots]
                    state = state.replace(
                        paralyzed=set_rows(state.paralyzed, target_slots, state.paralyzed[target_slots] | (feared_slots & protected)),
                        retreating=set_rows(state.retreating, target_slots, state.retreating[target_slots] | (feared_slots & ~protected)),
                        feared=set_rows(state.feared, target_slots, state.feared[target_slots] | (feared_slots & ~protected)))
                if self.has_weakening:
                    if self.has_powerups:
                        newly = affected & weakener & ~state.weakened[target_slots]
                        raw = self._combat_stats(state,cap=False)[target_slots,DAMAGE].astype(jnp.int32)
                        state = state.replace(
                            primary_override=set_rows(state.primary_override, target_slots, jnp.where(newly,jnp.rint(raw-self._original_primary(state)[target_slots]*.33).astype(jnp.int32),state.primary_override[target_slots])),
                            preweak_damage=set_rows(state.preweak_damage, target_slots, jnp.where(newly,raw,state.preweak_damage[target_slots])),
                            powerup_layered=set_rows(state.powerup_layered, target_slots, jnp.where(newly,state.powerup[target_slots],state.powerup_layered[target_slots])))
                    state = state.replace(weakened=set_rows(state.weakened, target_slots, state.weakened[target_slots] | (affected & weakener)))
                if self.has_secondary_paralysis:
                    secondary_paralyzed = affected & (paralysis_mode > 0)
                    state = self._record_disable(state,target_slots,secondary_paralyzed)
                    paralyzed_slots |= secondary_paralyzed
                    state = state.replace(
                        paralyzed=set_rows(state.paralyzed, target_slots, state.paralyzed[target_slots] | (secondary_paralyzed & (paralysis_mode == 1))),
                        long_paralyzed=set_rows(state.long_paralyzed, target_slots, state.long_paralyzed[target_slots] | (secondary_paralyzed & (paralysis_mode == 2))))
                if self.has_shatterers:
                    shreds = jnp.minimum(6, self._effective_shreds(state)[target_slots] +
                        (affected & shatterer & (stats[target_slots, ARMOR] > 0)).astype(jnp.int32))
                    lowered = state.decay_form[target_slots] > 0
                    state = state.replace(
                        armor_shreds=set_rows(state.armor_shreds, target_slots, jnp.where(lowered, state.armor_shreds[target_slots], shreds)),
                        decay_shreds=set_rows(state.decay_shreds, target_slots, jnp.where(lowered, shreds, state.decay_shreds[target_slots])))
                if self.has_witches:
                    transformed_slots = affected & witch
                    state,wards_used=transform_imp(self,state,wards_used,target_slots,transformed_slots,self._effect_value(state,'transform_long')[actor])
                if self.has_wights:
                    state,hp,wards_used,decayed_slots=transform_decay(
                        self,state,hp,wards_used,target_slots,affected & wight)
                if self.has_poisoners:
                    poisoned = affected & poisoner
                    duration = self._effect_value(state,'dot_modes')[actor,POISON]
                    state = self._record_effect(state,target_slots,poisoned,POISON)
                    state = state.replace(
                        poison_mode=set_rows(state.poison_mode, target_slots, jnp.where(poisoned, duration, state.poison_mode[target_slots])),
                        poison_damage=set_rows(state.poison_damage, target_slots, jnp.where(poisoned, secondary_stats[0].astype(jnp.int32), state.poison_damage[target_slots])),
                        poison_source=set_rows(state.poison_source, target_slots, jnp.where(poisoned, actor, state.poison_source[target_slots])))
                if self.has_water:
                    soaked = affected & water_caster
                    duration = self._effect_value(state,'dot_modes')[actor,FROSTBITE]
                    state = self._record_effect(state,target_slots,soaked,FROSTBITE)
                    state = state.replace(
                        water_mode=set_rows(state.water_mode, target_slots, jnp.where(soaked,duration,state.water_mode[target_slots])),
                        water_damage=set_rows(state.water_damage, target_slots, jnp.where(soaked,secondary_stats[0].astype(jnp.int32),state.water_damage[target_slots])),
                        water_source=set_rows(state.water_source, target_slots, jnp.where(soaked,actor,state.water_source[target_slots])))
                if self.has_fire:
                    burning = affected & fire_caster
                    duration = self._effect_value(state,'dot_modes')[actor,BLISTER]
                    state = self._record_effect(state,target_slots,burning,BLISTER)
                    state = state.replace(
                        burn_mode=set_rows(state.burn_mode, target_slots, jnp.where(burning,duration,state.burn_mode[target_slots])),
                        burn_damage=set_rows(state.burn_damage, target_slots, jnp.where(burning,secondary_stats[0].astype(jnp.int32),state.burn_damage[target_slots])),
                        burn_source=set_rows(state.burn_source, target_slots, jnp.where(burning,actor,state.burn_source[target_slots])))
            if self.items_enabled and (self.item_rules.status_items or self.item_rules.has_drain):
                connected=attack & ~area_attack & jnp.any(effective)
                state,hp,wards_used,item_immune,item_ward=equipment_attack(
                    self,state,hp,wards_used,target,connected,applied,random_values)
                immune_slots |= item_immune
                ward_slots |= item_ward
        elif self.mage_slot >= 0:
            # A fixed hero role is inferable from the actor/HP slots already in
            # observation v2. One hit roll covers the entire spell, without
            # extra RNG, loops, or host operations in the vmapped learner.
            area_attack = actor == self.mage_slot
            targets = jnp.where(area_attack, self.slots >= 6, self.slots == target)
            targets &= (state.hp > 0) & ~state.escaped
            base_damage = jnp.where(area_attack, self.mage_damage, single_damage)
            damage = jnp.where(state.defended, (base_damage + 1) // 2, base_damage)
            removed = jnp.where(hit & targets, jnp.minimum(damage, state.hp), 0)
            hp = state.hp - removed
            applied = jnp.sum(removed)
        else:
            damage = jnp.where(state.defended[target], (single_damage + 1) // 2, single_damage)
            applied = jnp.where(hit, jnp.minimum(damage, state.hp[target]), 0)
            hp = state.hp.at[target].add(-applied)
        item_using, item_extra = jnp.bool_(False), jnp.zeros(12,bool)
        item_target, item_applied = jnp.int32(-1), jnp.int32(0)
        if self.battle_items_enabled and self.battle_item_rules.count:
            # The mask/step predicate already admitted the hero; using an item spends the turn.
            state,hp,wards_used,item = self.battle_item_rules.use(
                self,state,hp,wards_used,action,random_values,stats,traits)
            item_using, item_extra = item['using'], item['extra']
            item_target, item_applied = item['target'], item['applied']
            immune_slots |= item['immune']
            ward_slots |= item['ward']
            damage_credit = damage_credit+item['credit']
        transformed = (action == FENRIR) & self._actor_is_wolf_lord(state)
        if self.has_wolf_lord:
            new_hp = jnp.rint(state.hp[actor] * 275. / jnp.maximum(self._combat_stats(state)[actor, HP], 1)).astype(jnp.int32)
            hp = set_rows(hp, actor, jnp.where(transformed, new_hp, hp[actor]))
            if self.has_hermits:
                state = state.replace(initiative_override=set_rows(state.initiative_override, actor, jnp.where(transformed,65,state.initiative_override[actor])))
            if self.has_powerups:
                state = state.replace(primary_override=set_rows(state.primary_override, actor, jnp.where(transformed,90,state.primary_override[actor])),
                    preweak_damage=set_rows(state.preweak_damage, actor, jnp.where(transformed,-1,state.preweak_damage[actor])),
                    powerup_layered=set_rows(state.powerup_layered, actor, state.powerup_layered[actor] & ~transformed))
            state = state.replace(fenrir=set_rows(state.fenrir, actor, state.fenrir[actor] | transformed),
                weakened=set_rows(state.weakened, actor, state.weakened[actor] & ~transformed),
                item_weakened=set_rows(state.item_weakened, actor, state.item_weakened[actor] & ~transformed))
        # Wait inserts one attack into the original turnsOrder. A separate
        # Alchemist activation restores phase 0 even if WAIT was already spent.
        another_strike = (attack & self._actor_double_strike(state) & ~state.second_strike & ~escaping
                          & (state.turn_phase[actor] == 0))
        if self.has_linked_summons:
            hp = self._linked_summon_hp(state,hp)
        # Append behind existing waiters, independent of initiative or slot ties.
        # Normal priorities are nonnegative; -1, -2, ... retain actual wait order.
        wait_priority = jnp.min(jnp.where(state.turn_phase == 1, state.priority, 0.)) - 1.
        state = state.replace(waited=set_rows(state.waited, actor, state.waited[actor] | (action == WAIT)),
            priority=set_rows(state.priority, actor, jnp.where(action == WAIT, wait_priority, state.priority[actor])))
        state = self._expire_powerups(state,(self.slots == actor) & ~another_strike & (action != WAIT))
        phases = set_rows(state.turn_phase, actor, jnp.where(another_strike, state.turn_phase[actor], jnp.where(action == WAIT, 1, 2)))
        # Orb of Rage on the hero: the bonus activation follows the spent one.
        phases = jnp.where(item_extra, 0, phases)
        defended = set_rows(state.defended, actor, (action == DEFEND) & (special_event < 0))
        retreating = set_rows(state.retreating, actor,
            jnp.where(skipping,state.retreating[actor],action == RETREAT) & can_retreat)
        escaped = set_rows(state.escaped, actor, escaping)
        if self.items_enabled:
            instant=(action==RETREAT)&(actor==self._equipment_hero(state))&jnp.any(self.item_rules.value(state,'instant_retreat')[:2])
            escaped=set_rows(escaped,actor,escaped[actor]|instant)
            retreating=set_rows(retreating,actor,retreating[actor]&~instant)
        if self.has_linked_summons:
            hp = self._linked_summon_hp(state.replace(escaped=escaped),hp)
        if self.progression_enabled:
            kill_xp = self.unit_experience(state)[:,1]
            state = self.progression.accumulate(state, state.hp, hp, escaped, kill_xp)
        before_periodic_hp = hp
        death_order = jnp.full(12,24,jnp.int32)
        active = (hp > 0) & ~escaped
        normal, waiting = active & (phases == 0), active & (phases == 1)
        new_round = ~jnp.any(normal | waiting)
        phases = jnp.where(new_round, jnp.zeros(12, jnp.int32), phases)
        priority = jnp.where(new_round, self._round_priority(random_values, state.enemy, state), state.priority)
        round_number = state.round + new_round.astype(jnp.int32)
        scores = jnp.where(active & (phases < 2), priority, -1e9 if self.basic_combat else -100)
        next_actor = jnp.where(another_strike, actor, jnp.argmax(scores)).astype(jnp.int32)
        defended = set_rows(defended, next_actor, False)
        state = state.replace(activation_done=jnp.where(new_round, False, state.activation_done),
                              last_poison_damage=jnp.zeros(12, jnp.int32),
                              last_water_damage=jnp.zeros(12,jnp.int32),
                              last_burn_damage=jnp.zeros(12,jnp.int32))
        extra_priority = self._round_priority(random_values[55:67],state.enemy,
            state.replace(slow_original=jnp.where(new_round,-1,state.slow_original))) if (
                self.has_poisoners or self.has_water or self.has_fire or self.has_summons) else priority
        # Retreating units remain exposed to normal DOTs after the round limit.
        queue_limit = jnp.where(state.summon_battle, self.max_rounds, jnp.iinfo(jnp.int32).max)
        visited = set_rows(jnp.zeros(12,bool), next_actor, ~another_strike & jnp.any(active[:6]) & jnp.any(active[6:]))
        if self.has_summons or self.has_water or self.has_fire:
            # Effects tick in original order: poison, frostbite, then burn; a lethal
            # earlier effect suppresses later ticks and any following actor.
            turns = jnp.stack((state.poison_mode,state.water_mode,state.burn_mode),axis=1)
            damage = jnp.stack((state.poison_damage,state.water_damage,state.burn_damage),axis=1)
            source = jnp.stack((state.poison_source,state.water_source,state.burn_source),axis=1)
            queue = advance_linked_queue if self.has_summons else advance_periodic_queue
            linked = {'summon_owner':state.summon_owner} if self.has_summons else {}
            (hp,phases,priority,round_number,activated,turns,damage,source,next_actor,losses,visited,death_order) = queue(
                hp,escaped,phases,priority,round_number,state.activation_done,turns,damage,source,
                self._combat_traits(state)[:,2],extra_priority,
                next_actor,another_strike,queue_limit,jnp.array([16,8,4],jnp.uint32),jnp.array([0,10,10],jnp.int32),
                state.effect_rounds[:,:3],random_values[84:156].reshape(2,12,3),return_death_order=True,**linked)
            state = state.replace(activation_done=activated,poison_mode=turns[:,0],poison_damage=damage[:,0],
                poison_source=source[:,0],last_poison_damage=losses[:,0],
                water_mode=turns[:,1],water_damage=damage[:,1],water_source=source[:,1],last_water_damage=losses[:,1],
                burn_mode=turns[:,2],burn_damage=damage[:,2],burn_source=source[:,2],last_burn_damage=losses[:,2])
            defended = set_rows(defended, next_actor, False)
            active = (hp > 0) & ~escaped
        elif self.has_poisoners:
            (hp, phases, priority, round_number, activated, poison_mode, poison_damage,
             poison_source, next_actor, poison_losses, visited, death_order) = advance_poison_queue(
                hp, escaped, phases, priority, round_number, state.activation_done,
                state.poison_mode, state.poison_damage, state.poison_source,
                self._combat_traits(state)[:, 2], extra_priority,
                next_actor, another_strike, queue_limit, state.effect_rounds[:,POISON],random_values[84:108].reshape(2,12),
                return_death_order=True)
            state = state.replace(activation_done=activated, poison_mode=poison_mode,
                                  poison_damage=poison_damage, poison_source=poison_source, last_poison_damage=poison_losses)
            defended = set_rows(defended, next_actor, False)
            active = (hp > 0) & ~escaped
        state = state.replace(slow_original=jnp.where(round_number != state.round,-1,state.slow_original))
        if self.has_linked_summons:
            # Also catches an owner killed by a DOT tick when only item summons
            # are present (the linked DOT queue runs only with summoner units).
            hp = self._linked_summon_hp(state,hp)
            active = (hp > 0) & ~escaped
            next_actor = jnp.where(active[next_actor],next_actor,jnp.argmax(jnp.where(
                active & (phases < 2),priority,-1e9))).astype(jnp.int32)
        state = state.replace(waited=jnp.where(round_number != state.round,False,state.waited))
        if self.progression_enabled and (self.has_poisoners or self.has_water or self.has_fire or self.has_summons):
            state = self.progression.accumulate(state, before_periodic_hp, hp, escaped, kill_xp, death_order)
        party_defeated = ~jnp.any((hp[:6] > 0) | (state.summon_saved_hp[:6] > 0))
        lost = party_defeated
        summon_battle = state.summon_battle if self.spell_casting_enabled else jnp.bool_(False)
        lost &= ~summon_battle  # a fallen summon only ends its own battle
        victory = ~jnp.any(active[6:]) & ~lost
        withdrawal = ~jnp.any(active[:6]) & ~lost
        # Reference: a summon battle without a winner ends; the stack keeps its wounds.
        withdrawal |= summon_battle & (round_number > self.max_rounds) & ~victory
        if self.has_summons or self.has_item_summons:
            # Battle summons leave with the battle (_restore_roster). If no permanent
            # fighter is alive (an escapee beneath a summon counts), a battle the
            # summon won or retreated from still ends in the party's defeat.
            permanent_hp = jnp.where(state.summon_owner[:6] >= 0, state.summon_saved_hp[:6], hp[:6])
            wiped = (victory | withdrawal) & ~summon_battle & ~jnp.any(permanent_hp > 0)
            lost |= wiped
            victory &= ~wiped
            withdrawal &= ~wiped
        # The decision boundary enforces max_steps after all automatic responses.
        # Internal unit activations never consume that budget.
        raw_ended = victory | withdrawal | lost
        capital_training_bonus = jnp.float32(0.)
        if self.capital_training.enabled:
            state, capital_training_bonus = self.capital_training.battle(state, hp, escaped, round_number)
        round_limit_reached = ((state.round <= self.max_rounds) & (round_number > self.max_rounds)
                               & ~raw_ended & ~post_turn & ~state.bot_autobattle
                               & self._can_retreat(state, jnp.int32(0)))
        pending = jnp.zeros(12,jnp.bool_)
        post_team = jnp.int32(0)
        if self.progression_enabled and self.has_healers:
            eligible = self.progression.post_healers[state.unit_ids] & (hp > 0) & ~escaped
            if self.has_copies or self.has_summons or self.has_item_summons:
                eligible &= ~state.copied & (state.summon_owner < 0)
            eligible &= jnp.where(victory,self.slots < 6,self.slots >= 6) & (victory | withdrawal | lost)
            pending = jnp.where(post_turn,set_rows(state.pending_healers,actor,False),eligible)
            healing = jnp.any(pending)
            post_team = jnp.where(healing,jnp.where(victory,1,2),0).astype(jnp.int32)
            victory &= ~healing
            lost &= ~healing
            withdrawal &= ~healing
            next_actor = jnp.where(healing,jnp.argmax(pending),next_actor).astype(jnp.int32)
        alive = set_rows(state.alive, state.enemy, ~victory)
        won = ~self._goals_left(alive, state.scenario_spawned) & ~lost
        if self.territory_enabled:
            won &= self.territory.goals_captured(state)
        back = victory | withdrawal
        event = jnp.where(attack, jnp.where(hit, HIT, MISS),
                            jnp.where(action == DEFEND, GUARD,
                            jnp.where(action == WAIT, DELAY,
                            jnp.where(action == RETREAT, FLEE, ESCAPE))))
        if self.items_enabled:
            event=jnp.where(instant,ESCAPE,event)
        event = jnp.where(attack & healer, HEAL, event)
        event = jnp.where(attack & (self._actor_power_factor(state) > 0),BUFFED,event)
        event = jnp.where(jnp.any(revived),BATTLE_REVIVED,event)
        event = jnp.where(jnp.any(extra_turn),EXTRA_TURN,event)
        event = jnp.where(jnp.any(slowed),SLOWED,event)
        event = jnp.where((post_team > 0) & ~post_turn,POST_HEAL,event)
        event = jnp.where(jnp.any(feared_slots),FEARED,event)
        event = jnp.where(jnp.any(paralyzed_slots),PARALYZED,event)
        event = jnp.where(skipping,PARALYSIS_SKIP,event)
        event = jnp.where(transformed, TRANSFORMED, event)
        event = jnp.where(jnp.any(transformed_slots), IMP_TRANSFORMED, event)
        event = jnp.where(jnp.any(decayed_slots), DECAY_TRANSFORMED, event)
        event = jnp.where(special_event >= 0,special_event,event)
        event = jnp.where(attack & (applied == 0) & (immune_slots != 0), IMMUNE, event)
        event = jnp.where(attack & (applied == 0) & (ward_slots != 0), WARD, event)
        event = jnp.where(item_using & (item_applied == 0) & (immune_slots != 0), IMMUNE,
            jnp.where(item_using & (item_applied == 0) & (ward_slots != 0), WARD,
            jnp.where(item_using, BATTLE_ITEM, event)))
        event = jnp.where(victory, VICTORY, jnp.where(lost, DEFEAT,
                            jnp.where(withdrawal, WITHDRAW, event)))
        event = jnp.where(round_limit_reached, ROUND_LIMIT_RETREAT, event)
        progress = {}
        ended_status = victory | withdrawal | lost
        state = self._restore_initiative_form(state,raw_ended & (state.imp | (state.decay_form > 0)))
        if self.has_hermits:
            state = state.replace(initiative_override=jnp.where(ended_status,-1,state.initiative_override),
                saved_initiative_override=jnp.where(ended_status,-1,state.saved_initiative_override),
                slow_original=jnp.where(ended_status,-1,state.slow_original))
        state = self._restore_damage_forms(state,raw_ended & (state.imp | (state.decay_form > 0)))
        if self.has_powerups:
            updates = {}
            for prefix in ('','saved_'):
                for name,empty in (('primary_override',-1),('preweak_damage',-1),('powerup',False),('powerup_long',False),('powerup_layered',False)):
                    updates[prefix+name] = jnp.where(ended_status,empty,getattr(state,prefix+name))
            state = state.replace(**updates)
        if self.has_healer_wards:
            caster_bits = jnp.sum(jnp.where(visited,jnp.left_shift(jnp.uint32(1),self.slots.astype(jnp.uint32)),jnp.uint32(0)))
            expire = caster_bits | jnp.where(ended_status | (raw_ended & state.fenrir),jnp.uint32(4095),jnp.uint32(0))[:,None]
            state,wards_used = self._expire_healer_wards(state,wards_used,expire)
            # Escaped recipients lose grants, but a caster leaving does not expire other recipients.
            state = state.replace(healer_wards=jnp.where(escaped[:,None],jnp.uint32(0),state.healer_wards),
                saved_healer_wards=jnp.where(escaped[:,None],jnp.uint32(0),state.saved_healer_wards),
                ward_native_used=jnp.where(escaped,jnp.uint32(0),state.ward_native_used),
                saved_ward_native_used=jnp.where(escaped,jnp.uint32(0),state.saved_ward_native_used))
            state = self._restore_ward_forms(state,raw_ended & (state.imp | (state.decay_form > 0)))
        natural_weakening = jnp.where(raw_ended & (state.imp | (state.decay_form > 0)),state.saved_weakened,state.weakened)
        if self.has_item_weakening:
            natural_item = jnp.where(raw_ended & (state.imp | (state.decay_form > 0)),state.saved_item_weakened,state.item_weakened)
            state = state.replace(item_weakened=natural_item & ~ended_status,
                                  saved_item_weakened=state.saved_item_weakened & ~ended_status)
        if self.has_lycanthropy:
            state = state.replace(lycanthrope=state.lycanthrope & ~raw_ended)
        state = state.replace(weakened=natural_weakening & ~ended_status,saved_weakened=state.saved_weakened & ~ended_status,
            feared=state.feared & ~ended_status & (hp > 0) & ~escaped)
        state = state.replace(paralyzed=state.paralyzed & ~ended_status & (hp > 0) & ~escaped,
                              long_paralyzed=state.long_paralyzed & ~ended_status & (hp > 0) & ~escaped)
        state = state.replace(disable_source=jnp.where(state.paralyzed | state.long_paralyzed,state.disable_source,-1),
            transform_long=state.transform_long & ~ended_status & (hp > 0) & ~escaped)
        if self.has_wights:
            ended = raw_ended
            lowered = state.decay_form > 0
            natural_hp = self._natural_stats(state)[:, HP]
            if self.has_copies:
                natural_hp = jnp.where(state.copied,state.copy_stats[:,HP],natural_hp)
            natural_hp = jnp.where(state.fenrir, 275., natural_hp)
            current_hp = self.progression.base_stats[state.decay_form, HP]
            restored = jnp.rint(hp / jnp.maximum(current_hp, 1) * natural_hp).astype(jnp.int32)
            restored = jnp.where(hp > 0, jnp.maximum(restored, 1), 0)
            hp = jnp.where(ended & lowered, restored, hp)
            wards_used = jnp.where(ended & lowered, state.imp_wards, wards_used)
            state = state.replace(decay_form=jnp.where(ended, 0, state.decay_form),
                                  decay_shreds=jnp.where(ended, 0, state.decay_shreds))
        if self.has_witches:
            ended = raw_ended
            wards_used = jnp.where(ended & state.imp, state.imp_wards, wards_used)
            state = state.replace(imp=state.imp & ~ended,
                                  activation_done=jnp.where(ended, False, state.activation_done))
        if self.has_poisoners:
            ended = victory | withdrawal | lost
            state = state.replace(poison_mode=jnp.where(ended, 0, state.poison_mode),
                poison_damage=jnp.where(ended, 0, state.poison_damage),
                poison_source=jnp.where(ended, -1, state.poison_source))
        if self.has_water:
            ended = victory | withdrawal | lost
            state = state.replace(water_mode=jnp.where(ended,0,state.water_mode),
                                  water_damage=jnp.where(ended,0,state.water_damage),
                                  water_source=jnp.where(ended,-1,state.water_source))
        if self.has_fire:
            ended = victory | withdrawal | lost
            state = state.replace(burn_mode=jnp.where(ended,0,state.burn_mode),
                                  burn_damage=jnp.where(ended,0,state.burn_damage),
                                  burn_source=jnp.where(ended,-1,state.burn_source))
        if self.has_shatterers:
            state = state.replace(armor_shreds=jnp.where(victory | withdrawal | lost, 0, state.armor_shreds))
        if self.has_wolf_lord:
            # Restore every natural form, including dead/escaped units, before XP.
            ended = raw_ended
            natural = (self._natural_stats(state)[:, HP]
                       if self.progression_enabled else self.stats_table[state.enemy, :, HP])
            if self.has_copies:
                natural = jnp.where(state.copied,state.copy_stats[:,HP],natural)
            restored = jnp.rint(hp / 275. * natural).astype(jnp.int32)
            restored = jnp.where(hp > 0, jnp.maximum(restored, 1), 0)
            hp = jnp.where(ended & state.fenrir, restored, hp)
            state = state.replace(fenrir=state.fenrir & ~ended)
        reward_hp = hp
        progression_reward = jnp.float32(0.)
        if self.progression_enabled:
            if self.has_copies or self.has_summons or self.has_item_summons:
                state,hp = self._restore_roster(state.replace(escaped=escaped),hp,raw_ended)
                escaped = state.escaped
            ids, levels, xp, enemies, hp, gains, promoted = self.progression.finish(
                state, hp, escaped, victory, lost, withdrawal, self._party_experience_bonus(state))
            if self.spell_casting_enabled:
                # The summon's growth is discarded with it; slot layers belong to the party.
                promoted = jnp.where(summon_battle, promoted & jnp.uint32(0xFC0), promoted)
            if not self.reference_rewards:
                progression_reward = self.progression.training_reward(state, gains, promoted, victory)
            if self.has_potion_buffs:
                intrinsic = self._potion_intrinsic(ids,levels)
                changed = ((promoted >> self.slots.astype(jnp.uint32)) & 1) != 0
                doses = jnp.where(changed[:,None],state.potion_doses,0)
                rebuilt = self.potion_rules.rebuild(intrinsic,doses)
                state = state.replace(potion_bonus=jnp.where(changed[:,None],rebuilt,state.potion_bonus))
                temporary=self.potion_rules.temporary_bonus(intrinsic+state.potion_bonus,state.potion_active)
                state=state.replace(potion_temporary=jnp.where(changed[:,None],temporary,state.potion_temporary))
                hp = jnp.where(changed,jnp.minimum(intrinsic[:,HP]+state.potion_bonus[:,HP],MAX_HP).astype(jnp.int32),hp)
            movement_gain = jnp.maximum(0, self._hero_level_movement_bonus(ids, levels)
                - self._hero_level_movement_bonus(state.unit_ids, state.unit_levels))
            progress = dict(unit_ids=ids, unit_levels=levels, unit_xp=xp,
                            movement_points=state.movement_points+movement_gain,
                            enemy_progress=enemies,
                            last_xp=gains, last_promoted=promoted)
        if self.capital_enabled:
            # Count actual damage independently of healing/leech and HP rescaling.
            periodic = state.last_poison_damage[6:]+state.last_burn_damage[6:]+state.last_water_damage[6:]
            paid_damage = jnp.minimum(state.enemy_initial_hp,state.enemy_damage_credit+damage_credit+periodic)
            paid_kills = state.enemy_killed_credit | ((state.hp[6:] > 0) & (hp[6:] == 0) & (state.enemy_initial_hp > 0))
            credit = jnp.array([jnp.sum(paid_damage),jnp.sum(paid_kills)],jnp.int32)
            progress["recovery_balance"] = state.recovery_balance+jnp.where(victory & ~summon_battle,credit,0)
            progress["enemy_damage_credit"] = paid_damage
            progress["enemy_killed_credit"] = paid_kills
        recovered_hp = self.restored_hp
        if self.basic_combat:
            # Both survivors and fallen units retain their HP; revival is paid.
            occupied_party = ((self.slots < 6) & (progress['unit_ids'] != 0)
                              if self.progression_enabled else self.restored_hp > 0)
            recovered_hp = jnp.where(occupied_party, hp, 0)
        next_state = state.replace(
            battle_key=key, hp=jnp.minimum(jnp.where(back, recovered_hp, hp), MAX_HP), **progress,
            battle_fort_armor=jnp.where(back,0,state.battle_fort_armor),
            battle_hero_fort_armor=jnp.where(back,0,state.battle_hero_fort_armor),
            battle_fort_defender=jnp.where(back | lost, FORT_NONE, state.battle_fort_defender),
            post_victory=post_team,pending_healers=pending,
            second_strike=another_strike & ~(victory | withdrawal | lost),
            wards_used=jnp.where(back, jnp.uint32(0), wards_used),
            last_immune=immune_slots, last_ward=ward_slots,
            in_battle=~back, position=jnp.where(withdrawal, state.origin, state.position),
            alive=alive, number=state.number + victory.astype(jnp.int32), won=won, lost=lost,
            done=won | lost, priority=jnp.where(back, 0., priority),
            turn_phase=jnp.where(back, 0, phases), defended=defended & ~back,
            retreating=retreating & ~back, escaped=escaped & ~back,
            actor=jnp.where(back, 0, next_actor), round=jnp.where(back, 0, round_number),
            last_event=event.astype(jnp.int32), last_actor=actor,
            last_target=jnp.where(item_using, item_target, jnp.where(attack & ~area_attack, target, -1)),
            last_damage=jnp.where(item_using, item_applied, applied),
            battle_steps=state.battle_steps + 1,
            player_turns=state.player_turns + ((actor < 6) & ~escaping).astype(jnp.int32),
            enemy_turns=state.enemy_turns + ((actor >= 6) & ~escaping).astype(jnp.int32),
        )
        if self.spell_casting_enabled:
            next_state = self.casting.finish_battle(state, next_state, hp, victory, back)
        if self.bot.enabled:
            next_state = self.bot.after_battle(state, next_state, victory, victory | withdrawal | lost)
        if self.territory_enabled:
            next_state = self._capture_cities(next_state,victory & ~summon_battle)
            final_win = self._campaign_won(next_state)
            next_state = next_state.replace(won=final_win,done=next_state.done|final_win)
            won = final_win
        # Ordinary summon victories have no base bonus. Configured target rewards
        # replace the hero's +1 and pay every method at the public step boundary.
        ordinary_victory = victory & ~summon_battle & ~self.target_rewards.mask[jnp.maximum(state.enemy, 0)]
        reward = (ordinary_victory.astype(jnp.float32)
                  + self._objective_reward(won, victory & ~summon_battle) - lost.astype(jnp.float32))
        reward -= jnp.where(round_limit_reached, self.round_limit_penalty, 0.)
        if self.reference_rewards:
            next_state, reward = reference_rewards.battle(self, state, next_state, reward_hp,
                victory, lost, withdrawal, item_using, base_reward=reward)
        reward += progression_reward + capital_training_bonus
        if self.army_victory_level_reward:
            enemy = jnp.clip(state.enemy, 0, self.num_opponents-1)
            first = (victory & ~summon_battle & (state.enemy >= 0)
                     & self.army_victory_level_targets[enemy] & ~state.army_victory_rewarded[enemy])
            reward += first*self.army_victory_level_reward*state.army_battle_level_strength
            next_state = next_state.replace(
                army_victory_rewarded=set_rows(state.army_victory_rewarded, enemy,
                    state.army_victory_rewarded[enemy] | first),
                army_battle_level_strength=jnp.where(victory | withdrawal | lost, 0,
                                                     state.army_battle_level_strength))
        if self.ruins_enabled:
            next_state, ruin_reward = self.ruins.grant(next_state,victory)
            reward += jnp.where(summon_battle, 0., ruin_reward*(4 if self.reference_rewards else 1))
        if self.items_enabled:
            next_state=jax.lax.cond(back,self._refresh_equipment,lambda s:s,next_state)
        if self.battle_items_enabled:
            # The real party is restored before this boundary (including summon battles).
            # A fallen leader keeps the slots during combat and post-battle healing.
            next_state=jax.lax.cond(back & ~summon_battle,
                lambda s:self.battle_item_rules.finish_battle(self,s),lambda s:s,next_state)
        if self.has_witches or self.has_poisoners or self.has_water or self.has_fire or self.has_wights or self.has_hermits:
            next_state = self._start_activation(next_state, random_values[36],
                                                next_state.in_battle & ~next_state.done & (next_state.post_victory == 0))
        if self.has_paralysis:
            next_state = self._prepare_disable(next_state,random_values[67],
                next_state.in_battle & ~next_state.done & (next_state.post_victory == 0) & ~another_strike)
        if with_metrics:
            ended = victory | withdrawal | lost
            upgrades = jax.lax.population_count(next_state.last_promoted & jnp.uint32(63))
            return next_state, reward, jnp.array([victory, ended & party_defeated,
                withdrawal & ~party_defeated, jnp.where(victory, upgrades, 0)], jnp.int32)
        return next_state, reward

    def step(self, state, action, env_params=None):
        """Execute one player decision and automatic responses, then report rewards and episode status."""
        if not self.battle_mode:
            return super().step(state, action, env_params)
        # Validate only the selected move/target here. The full action mask is
        # generated once for the next observation, not twice per transition.
        destination = state.position + self.directions[jnp.clip(action, 0, 7)]
        world_valid = ((action >= 0) & (action < 8)
                       & jnp.all((destination > 0) & (destination < self.size - 1)))
        if self.territory_enabled:
            world_valid &= ~self.territory.blocked(destination)
        elif self.sites_enabled:
            world_valid &= ~self.site_rules.blocked(destination)
        if self.ruins_enabled:
            if not self.territory_enabled:
                world_valid &= ~self.ruins.blocked(destination)
            world_valid |= (action>=0)&(action<8)&self.ruins.attackable(state,destination)
        if self.basic_combat:
            building = jnp.clip(action - BUILD_START, 0, BUILD_SLOTS-1)
            world_valid &= state.movement_points > 0
            world_valid |= ((action >= BUILD_START) & (action < REST)
                            & self.construction.available(state, building)) | (action == REST)
        if self.capital_enabled:
            slot = jnp.clip(jnp.where(action < REVIVE_START, action-HEAL_START, action-REVIVE_START), 0, 5)
            heal, revive = self.capital.available(state, self.max_hp(state), slot)
            world_valid |= ((action >= HEAL_START) & (action < REVIVE_START) & heal
                            | (action >= REVIVE_START) & (action < CAPITAL_ACTIONS) & revive)
        if self.potions_enabled and self.potion_rules.count:
            index = jnp.clip(action-POTION_START, 0, self.potion_rules.action_count-1)
            available = self.potion_rules.available(state, self.max_hp(state), index // 6, index % 6)
            world_valid |= (action >= POTION_START) & (action < POTION_START+self.potion_rules.action_count) & available
        if self.sites_enabled and self.site_rules.action_count:
            index = jnp.clip(action-self.site_rules.buy_start, 0, self.site_rules.action_count-1)
            world_valid |= (action >= self.site_rules.buy_start) & (action < self.site_rules.end) & self.site_rules.available(state)[index]
        if self.recruitment_enabled:
            index = jnp.clip(action-self.recruitment.start, 0, self.recruitment.action_count-1)
            world_valid |= (action >= self.recruitment.start) & (action < self.recruitment.end) & self.recruitment.available(state)[index]
        if self.territory_enabled and self.territory.count:
            index = jnp.clip(action-self.territory.start,0,self.territory.count-1)
            world_valid |= (action>=self.territory.start)&(action<self.territory.end)&self.territory.available(state)[index]
        if self.spell_research_enabled:
            index = jnp.clip(action-self.spell_research.start, 0, self.spell_research.count-1)
            world_valid |= ((action >= self.spell_research.start) & (action < self.spell_research.end)
                            & self.spell_research.available(state, index))
        spell_targets = None
        if self.spell_casting_enabled:
            # One nearest-stack search serves the validity check and the cast itself.
            spell_targets = self.casting.targets(state)
            world_valid |= self.casting.valid(state, action, self.max_hp(state), self.movement_cap(state), spell_targets)
        item_rules = self.battle_item_rules if self.battle_items_enabled and self.battle_item_rules.count else None
        if item_rules is not None:
            index = jnp.clip(action-item_rules.equip_start, 0, item_rules.count-1)
            world_valid |= ((action >= item_rules.equip_start) & (action < item_rules.use_start)
                            & item_rules.equip_available(self, state)[index])
        if self.unit_swaps_enabled:
            world_valid |= self.unit_swaps.valid(state, action)
        controlled = ((state.actor < 6) & (state.hp[state.actor] > 0) & ~state.escaped[state.actor]
                      & ~state.retreating[state.actor] & ~state.paralyzed[state.actor] & ~state.long_paralyzed[state.actor])
        target_slot = jnp.clip(action - SHOOT, 0, 5)
        own_target = self._actor_is_healer(state) if self.basic_combat and self.has_healers else jnp.bool_(False)
        target = target_slot + jnp.where(own_target, 0, 6)
        attack_valid = ((action >= SHOOT) & (action < DEFEND)
                        & (state.hp[target] > 0) & ~state.escaped[target])
        if self.basic_combat:
            attack_valid &= ~self._actor_is_melee(state) | self._melee_targets(state)[target_slot]
        elif self.warrior_slot >= 0:
            attack_valid &= ((state.actor != self.warrior_slot)
                             | self._melee_targets(state)[target_slot])
        if self.has_patriarchs:
            attack_valid = jnp.where(self._actor_is_patriarch(state),(action >= SHOOT) & (action < DEFEND) & self._patriarch_targets(state)[target],attack_valid)
        if self.has_alchemists:
            attack_valid = jnp.where(self._alchemist_units(state)[state.actor],(action >= SHOOT) & (action < DEFEND) & self._alchemist_targets(state)[target],attack_valid)
        attack_valid &= (self._actor_power_factor(state) == 0) | (target != state.actor) | self._actor_power_cures(state)
        unit_valid = attack_valid | (~state.second_strike & ((action == DEFEND) | (action == RETREAT) | ((action == WAIT) & (state.turn_phase[state.actor] == 0) & ~state.waited[state.actor])))
        unit_valid |= (action == FENRIR) & ~state.second_strike & self._actor_is_wolf_lord(state)
        if item_rules is not None:
            index = jnp.clip(action-item_rules.use_start, 0, battle_items.ACTIONS-1)
            unit_valid |= ((action >= item_rules.use_start) & (action < item_rules.end)
                           & item_rules.use_available(self, state)[index])
        if self.has_copies:
            copying = self._actor_copies(state)
            ally = (action >= COPY_ALLY) & (action < COPY_ALLY+COPY_ACTIONS)
            copy_slot = jnp.clip(jnp.where(ally,action-COPY_ALLY,action-SHOOT),0,5)
            copy_valid = ((ally | ((action >= SHOOT) & (action < DEFEND)))
                          & self._copy_targets(state)[copy_slot+jnp.where(ally,0,6)])
            unit_valid = jnp.where(copying & ((action >= SHOOT) & (action < DEFEND) | ally),copy_valid,unit_valid)
            unit_valid = jnp.where(state.round == 0,(copying & copy_valid) | (action == DEFEND),unit_valid)
        if self.has_summons:
            summoning = self._actor_summons(state) > 0
            unit_valid = jnp.where(summoning & (action >= SHOOT) & (action < DEFEND),
                                  self._summon_targets(state)[target_slot],unit_valid)
        battle_valid = jnp.where(controlled,unit_valid,action == CONTINUE)
        if self.progression_enabled and self.has_healers:
            post_valid = jnp.where(state.actor < 6,
                ((action >= SHOOT) & (action < DEFEND) & (state.hp[target_slot] > 0) & ~state.escaped[target_slot]) | (action == WAIT),
                action == CONTINUE)
            if self.has_patriarchs:
                post_valid = jnp.where((state.actor < 6) & self._actor_is_patriarch(state),attack_valid | (action == WAIT),post_valid)
            battle_valid = jnp.where(state.post_victory > 0,post_valid,battle_valid)
        battle_valid = jnp.where(self.round_limit_retreat(state),
                                 action == jnp.where(controlled, RETREAT, CONTINUE), battle_valid)
        battle_valid &= (action != RETREAT) | self._can_retreat(state, state.actor)
        if self.scenario.enabled or self.bot.enabled or self.script_events.enabled:
            world_valid = jnp.where(jnp.any(state.pending_encounters & state.alive), action == CONTINUE, world_valid)
        valid = jnp.where(state.in_battle, battle_valid, world_valid) & ~state.done
        # Map actions can start a battle whose first activation belongs to RED.
        # Keep map logic outside the loop: it runs only once per player decision.
        key, random_key = jax.random.split(state.battle_key)
        random_values = jax.random.uniform(random_key, (self.random_size,))
        next_state, reward = jax.lax.cond(~state.in_battle & valid,
            lambda s: self._world_step(s, action, key, random_values, spell_targets),
            lambda s: (s, jnp.float32(0)), state)
        world_state = next_state

        def pending(carry):
            current, _, first, _ = carry
            return first | (valid & self.automatic_battle_turn(current))

        def advance(carry):
            current, total_reward, first, counts = carry
            automatic = self.automatic_battle_turn(current)
            chosen = jnp.where(first, action, self._automatic_battle_action(current))
            next_key, draw_key = jax.random.split(current.battle_key)
            draws = jax.random.uniform(draw_key, (self.random_size,))
            following, gain, events = self._battle_step(current, chosen, next_key, draws, with_metrics=True)
            counts += jnp.concatenate((jnp.array([1, ~automatic, current.actor >= 6, automatic], jnp.int32), events))
            return following, total_reward+gain, jnp.bool_(False), counts

        next_state, reward, _, counts = jax.lax.while_loop(pending, advance,
            (next_state, reward, valid & state.in_battle, jnp.zeros(8, jnp.int32)))
        decision = ~state.done & ~self.automatic_battle_turn(state)
        steps = state.step_count + decision.astype(jnp.int32)
        next_state = next_state.replace(step_count=steps, done=next_state.done | (steps >= self.max_steps))
        fallback = state.replace(step_count=steps,
                                 done=state.done | (steps >= self.max_steps))
        next_state = jax.tree.map(lambda new, old: jnp.where(valid, new, old), next_state, fallback)
        rested = valid & ~state.in_battle & (action == REST)
        # REST replaces the ordinary step cost with the unused-movement penalty.
        reward = jnp.where(valid, reward, 0.) - jnp.where(decision & ~rested, jnp.float32(self.step_cost), 0.)
        if self.capital_training.enabled:
            next_state, acquisition_reward = self.capital_training.acquire(state, next_state)
            reward += acquisition_reward
        if self.territory_enabled:
            bumped = (~valid & ~state.done & ~state.in_battle & (action>=0) & (action<8)
                & jnp.all((destination>0)&(destination<self.size-1)) & self.territory.blocked(destination))
            reward -= .25*bumped
            next_state = next_state.replace(last_event=jnp.where(bumped,OBSTACLE,next_state.last_event))
        if self.recruitment_enabled:
            reward -= jnp.where(decision, .05*self.recruitment.reward_vacancies(next_state), 0.)
        if self.reference_rewards:
            next_state, completion_reward = reference_rewards.finish(self, state, next_state)
            reward += completion_reward
        if self.script_events.enabled:
            next_state = self.script_events.outcome(next_state)
        timed_out = ~state.done & next_state.done & ~next_state.won & ~next_state.lost
        reward -= self.episode_timeout_penalty * timed_out
        if self.target_rewards.enabled:
            # Exactly once per policy transition, after automatic turns and terminal checks.
            next_state, target_reward = self.target_rewards.finish(self, state, next_state)
            reward += target_reward
        ts = self._timestep(next_state, jnp.where(state.done, 0., reward))
        ts = ts.replace(extras={**ts.extras, 'battle_transition': counts[0],
                                'player_battle_transition': counts[1],
                                'enemy_battle_transition': counts[2],
                                'automatic_battle_transition': counts[3],
                                'battle_victory': counts[4] > 0,
                                'building_constructed': valid & ~state.in_battle & (action >= BUILD_START) & (action < REST),
                                'turn_ended': rested,
                                'campaign_events': campaign_events(self, state, world_state, next_state,
                                    action, valid, counts[4:]),
                                'rest_penalty': (jnp.where(rested, self.rest_penalty(state), 0.)
                                                 if self.basic_combat else jnp.float32(0))})
        if self.episode_reports:
            ts = ts.replace(extras={**ts.extras, 'episode_event': episode_event(
                self, state, world_state, next_state, action, valid, counts[4:])})
        return next_state, ts

    def action_space(self, env_params=None):
        return DiscreteSpace(self.num_actions if self.battle_mode else 8, dtype=jnp.int32)

    def state_space(self, env_params=None):
        if not self.battle_mode:
            return super().state_space(env_params)
        state, _ = self.reset(jax.random.PRNGKey(0))
        return DictSpace({f.name: ArraySpace(getattr(state, f.name).shape, getattr(state, f.name).dtype)
                          for f in dataclasses.fields(state)})


def wrap_wall_action_mask(env):
    return AddActionMaskWrapper(env) if env.mask_walls else env
