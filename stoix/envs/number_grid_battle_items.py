"""Hero battle slots: orbs, talismans and battle potions. See docs/ITEMS.md.

Effects: Python reference campaign_env_inventory._battle_item_effect_definitions
and battle_env hero-item actions. Item IDs and prices: installed GItem.dbf.
"""
import jax.numpy as jnp

from stoix.envs.number_grid_combat import HP, DAMAGE, ARMOR, INITIATIVE, UNITS, accuracy_hits, can_revive
from stoix.envs.number_grid_effects import source_protection, vampiric_heal, disable_attack_accuracy
from stoix.envs.number_grid_updates import set_rows

SLOTS, TALISMAN_USES, TARGETS = 2, 5, 6
ACTIONS = 2*SLOTS*TARGETS  # own slots 1..6 for both items, then enemy slots
BATTLE_ITEM, ITEM_EQUIPPED = 52, 53
(HEAL, REVIVE, SUMMON, STRIKE, DRAIN, DOT, BUFF, EXTRA_TURN, PARALYSIS, FEAR,
 TRANSFORM, LYCANTHROPY, WEAKEN) = range(13)
KIND_NAMES = ('heal', 'revive', 'summon', 'damage', 'drain', 'dot', 'damage_buff', 'extra_turn',
              'paralysis', 'fear', 'transform', 'lycanthropy', 'debuff_damage')
POISON, BURN, WATER = range(3)  # catalogue identifiers; tick order is poison, water, burn
# One-based JAX attack sources (number_grid_combat.ATTACK_TYPES).
EARTH, FIRE, WATER_SOURCE, DEATH, MIND, AIR = 2, 3, 4, 6, 7, 9
SOURCE_LABELS = {EARTH: 'Земля', FIRE: 'Огонь', WATER_SOURCE: 'Вода', DEATH: 'Смерть', MIND: 'Разум', AIR: 'Воздух'}
DOT_FIELDS = ('poison', 'burn', 'water')
# The four reference battle potions share the map potion inventory.
POTION_EFFECTS = {'healing': dict(kind=HEAL, amount=50), 'restoration': dict(kind=HEAL, amount=100),
                  'ointment': dict(kind=HEAL, amount=200), 'life': dict(kind=REVIVE, amount=1)}
# Reference summon templates (BATTLE_TEMPLATE_BY_NAME), by GUnits ID.
SUMMONS = (('squire', 'Squire', 'g000uu0001', 'Сквайр', 200),
           ('angel', 'Angel', 'g000uu0014', 'Ангел', 800),
           ('venerable_warrior', 'Venerable Warrior', 'g000uu0040', 'Старый ветеран', 800),
           ('imp', 'Imp', 'g000uu5034', 'Бес', 200),
           ('infernal_knight', 'Infernal Knight', 'g000uu0166', 'Адский рыцарь', 800),
           ('incubus', 'Incubus', 'g000uu0065', 'Инкуб', 800),
           ('zombie', 'Zombie', 'g000uu0087', 'Зомби', 400),
           ('skeleton_champion', 'Skeleton Champion', 'g000uu0089', 'Скелет-рыцарь', 800),
           ('vampire', 'Vampire', 'g000uu0083', 'Вампир', 800),
           ('lich', 'Lich', 'g000uu0084', 'Лич', 800),
           ('elder_vampire', 'Elder Vampire', 'g000uu0175', 'Высший вампир', 1000),
           ('elf_lord', 'Elf Lord', 'g000uu8009', 'Лесной лорд', 800),
           ('goblin', 'Goblin', 'g000uu5017', 'Гоблин', 200),
           ('orc', 'Orc', 'g000uu5013', 'Орк', 600),
           ('lizard_man', 'Lizard Man', 'g000uu5024', 'Человек-ящер', 600))
WEREWOLF = 'g000uu0092'


def _effect(kind, label, enemy=False, party=False, **values):
    return dict(kind=kind, label=label, enemy=enemy, party=party, **values)


def _damage(amount, source, party=True):
    who = 'вражескому отряду' if party else 'врагу'
    return _effect(STRIKE, f'Урон {who}: {amount}, {SOURCE_LABELS[source].lower()}, точность 80%',
                   True, party, amount=amount, accuracy=80, source=source)


def _drain(amount, share=False):
    extra = '; излишек лечит отряд' if share else ''
    return _effect(DRAIN, f'Похищение жизни у вражеского отряда: {amount}, точность 80%; '
                   f'герой лечится на половину урона{extra}', True, True,
                   amount=amount, accuracy=80, source=DEATH, share=share)


def _dot(dot, amount, party):
    name = ('Яд', 'Ожог', 'Обморожение')[dot]
    who = 'вражескому отряду' if party else 'врагу'
    source = (5, FIRE, WATER_SOURCE)[dot]
    return _effect(DOT, f'{name} {who}: {amount} HP за срабатывание; растущий шанс снятия', True, party,
                   amount=amount, dot=dot, source=source)


# Orb/talisman effects by base name; IDs are G000IG90xx (orb) and 91xx (talisman).
EFFECTS = (
    ('healing', 'Healing', 16, 400, _effect(HEAL, 'Лечение союзника: +100 HP', amount=100)),
    ('restoration', 'Restoration', 17, 400, _effect(HEAL, 'Лечение отряда: +30 HP', party=True, amount=30)),
    ('regeneration', 'Regeneration', 18, 600, _effect(HEAL, 'Лечение отряда: +75 HP', party=True, amount=75)),
    ('life', 'Life', 19, 800, _effect(REVIVE, 'Воскрешение союзника с 50% HP', amount=1, percent=50)),
    ('vigor', 'Vigor', 20, 400, _effect(BUFF, 'Отряду: +25% урона до конца боя',
                                        party=True, multiplier=1.25)),
    ('strength', 'Strength', 21, 600, _effect(BUFF, 'Отряду: +50% урона до конца боя',
                                              party=True, multiplier=1.5)),
    ('fire', 'Fire', 22, 400, _damage(25, FIRE)),
    ('inferno', 'Inferno', 23, 600, _damage(75, FIRE)),
    ('nosferat', 'Nosferat', 24, 600, _drain(25)),
    ('vampire', 'Vampire', 25, 800, _drain(75)),
    ('elder_vampires', 'Elder Vampires', 26, 1000, _drain(75, True)),
    ('thunder', 'Thunder', 27, 400, _damage(100, AIR, party=False)),
    ('lightning', 'Lightning', 28, 600, _damage(50, AIR)),
    ('earth', 'Earth', 29, 400, _damage(25, EARTH)),
    ('stone_rain', 'Stone Rain', 30, 600, _damage(75, EARTH)),
    ('water', 'Water', 31, 400, _damage(25, WATER_SOURCE)),
    ('icefall', 'Icefall', 32, 600, _damage(75, WATER_SOURCE)),
    ('poison', 'Poison', 33, 400, _dot(POISON, 20, False)),
    ('venom', 'Venom', 34, 600, _dot(POISON, 30, False)),
    ('bane', 'Bane', 35, 400, _dot(POISON, 10, True)),
    ('thanatos', 'Thanatos', 36, 800, _dot(POISON, 30, True)),
    ('frost', 'Frost', 37, 400, _dot(WATER, 10, True)),
    ('freezing', 'Freezing', 38, 600, _dot(WATER, 30, True)),
    ('fear', 'Fear', 39, 600, _effect(PARALYSIS, 'Длительный паралич врага, Разум; растущий шанс снятия', True, source=MIND)),
    ('witches', 'Witches', 40, 800, _effect(TRANSFORM, 'Превращение врага в беса, Разум', True, source=MIND)),
)
ORB_ONLY = (
    ('flame', 'Orb of Flame', 'G000IG9041', 1000, _dot(BURN, 15, True)),
    ('weakening', 'Orb of Weakening', 'G000IG9042', 1000,
     _effect(WEAKEN, 'Урон врага ×0,67 до конца боя, Разум', True, source=MIND, multiplier=.67)),
    ('rage', 'Orb of Rage', 'G000IG9043', 1000, _effect(EXTRA_TURN, 'Союзник ходит ещё раз в этом раунде')),
    # Reference-only: no GItem.dbf row in the installed Rise of the Elves.
    ('lycanthropy', 'Orb of Lycanthropy', None, 0,
     _effect(LYCANTHROPY, 'Превращение врага в оборотня, Разум', True, source=MIND, form=WEREWOLF)),
)


def _catalog():
    orbs, talismans = [], []
    for key, name, gid, label, price in SUMMONS:
        index = 1+[s[0] for s in SUMMONS].index(key)
        effect = _effect(SUMMON, f'Призыв в бой: {label}', summon=gid)
        orbs.append(dict(key=key+'_orb', name=name+' Orb', game_id=f'G000IG90{index:02d}', price=price, **effect))
        talismans.append(dict(key=key+'_talisman', name=name+' Talisman', game_id=f'G000IG91{index:02d}',
                              price=price, **effect))
    for key, name, index, price, effect in EFFECTS:
        orbs.append(dict(key='orb_of_'+key, name='Orb of '+name, game_id=f'G000IG90{index:02d}',
                         price=price, **effect))
        talismans.append(dict(key='talisman_of_'+key, name='Talisman of '+name,
                              game_id=f'G000IG91{index:02d}', price=price, **effect))
    for key, name, gid, price, effect in ORB_ONLY:
        orbs.append(dict(key='orb_of_'+key, name=name, game_id=gid, price=price, **effect))
    talismans.append(dict(key='talisman_of_nightmare', name='Talisman of Nightmare', game_id='G000IG9141',
                          price=600, **_effect(FEAR, 'Страх: враг бежит из боя, Разум', True, source=MIND)))
    for item in orbs:
        item.update(category='orb', uses=1, label='Сфера: '+item['label'])
    for item in talismans:
        item.update(category='talisman', uses=TALISMAN_USES,
                    label=f'Талисман, {TALISMAN_USES} зарядов, раз за бой: '+item['label'])
    return tuple(orbs), tuple(talismans)


ORBS, TALISMANS = _catalog()
MAGIC_ITEMS = ORBS+TALISMANS
MAGIC_BY_KEY = {item['key']: item for item in MAGIC_ITEMS}
GAME_IDS = {value['game_id']: key for key, value in UNITS.items()}


def _scenario_magic_keys(game_map):
    from stoix.envs.number_grid_potions import merchants
    keys = set()
    sources = [game_map.get('initial_items', {})]+[c.get('items', {}) for c in game_map.get('chests', [])
                                                     + game_map.get('ruins', [])+game_map.get('cities', [])
                                                     + game_map.get('day_items', [])+merchants(game_map)
                                                     if isinstance(c, dict)]
    for contents in sources:
        keys.update(key for key, count in contents.items() if count and key in MAGIC_BY_KEY)
    return keys


def battle_items_enabled(game_map):
    return bool(game_map.get('battle_items', False))


def scenario_battle_items(game_map):
    """Reference order: battle potions first, then the scenario's orbs/talismans."""
    if not battle_items_enabled(game_map):
        return ()
    from stoix.envs.number_grid_potions import POTION_BY_KEY, scenario_potions
    present = {p['key'] for p in scenario_potions(game_map)}
    potions = tuple(dict(POTION_BY_KEY[key], **effect, category='potion', uses=1, enemy=False, party=False)
                    for key, effect in POTION_EFFECTS.items() if key in present)
    magic = _scenario_magic_keys(game_map)
    return potions+tuple(item for item in MAGIC_ITEMS if item['key'] in magic)


def battle_item_unit_keys(game_map):
    """Summoned and polymorph forms enter the compiled catalogue features."""
    forms = {item.get('summon') or item.get('form') for item in scenario_battle_items(game_map)}
    return tuple(GAME_IDS[gid] for gid in sorted(forms - {None}))


def battle_item_action_names(game_map):
    items = scenario_battle_items(game_map)
    if not items:
        return ()
    return tuple('equip_'+item['key'] for item in items)+tuple(
        f'item{slot+1}_{side}_{target}' for side in ('ally', 'enemy')
        for slot in range(SLOTS) for target in range(TARGETS))


class BattleItemRules:
    def __init__(self, game_map, potions, items, progression, construction):
        self.items = scenario_battle_items(game_map)
        self.count = len(self.items)
        self.keys = tuple(item['key'] for item in self.items)
        self.potions, self.equipment, self.progression = potions, items, progression
        self.kinds = {item['kind'] for item in self.items}
        self.dot_kinds = {item['dot'] for item in self.items if item['kind'] == DOT}
        self.mage_lord = construction.lord['id'] == 'mage'
        if self.count and (any(item['category'] != 'potion' for item in self.items) and items is None):
            raise ValueError('Orbs and talismans require the equipment inventory')
        missing = [gid for gid in (item.get('summon') or item.get('form') for item in self.items)
                   if gid and GAME_IDS.get(gid) not in progression.ids]
        if missing:
            raise ValueError('Battle item units are missing from the catalogue: '+', '.join(missing))

        def table(name, default, dtype):
            return jnp.array([item.get(name, default) for item in self.items] or [default], dtype)
        self.kind = table('kind', -1, jnp.int32)
        self.amount = table('amount', 0, jnp.int32)
        self.accuracy = table('accuracy', 0, jnp.float32)
        self.source = table('source', 0, jnp.uint32)
        self.enemy = table('enemy', False, bool)
        self.party = table('party', False, bool)
        self.percent = table('percent', 0, jnp.float32)
        self.dot = table('dot', 0, jnp.int32)
        self.multiplier = table('multiplier', 1., jnp.float32)
        self.share = table('share', False, bool)
        self.uses = table('uses', 1, jnp.int32)
        self.talisman = jnp.array([item['category'] == 'talisman' for item in self.items] or [False])
        self.orb = jnp.array([item['category'] == 'orb' for item in self.items] or [False])
        ids = lambda gid: progression.ids[GAME_IDS[gid]] if gid else 0
        self.summon = jnp.array([ids(item.get('summon') or item.get('form')) for item in self.items] or [0], jnp.int32)
        self.potion_index = jnp.array([potions.keys.index(item['key']) if item['category'] == 'potion' else -1
                                       for item in self.items] or [-1], jnp.int32)
        self.item_index = jnp.array([items.keys.index(item['key']) if item['category'] != 'potion' else -1
                                     for item in self.items] or [-1], jnp.int32)
        self.observation_size = SLOTS*(self.count+1)+SLOTS
        self.place(0)

    def place(self, start):
        self.equip_start = start
        self.use_start = start+self.count
        self.end = self.use_start+(ACTIONS if self.count else 0)

    # ------------------------------ map: equip ------------------------------
    def stock(self, state):
        """Owned copies by kind; equipping only reserves them."""
        if state.potions.shape[0]:
            potions = state.potions[jnp.maximum(self.potion_index, 0)]
        else:
            # Maps without potion kinds (Mirror match) have no potion items: every
            # potion_index is -1. Eager calls (human-mode quotes) cannot index an
            # empty array, so supply the unused values directly.
            potions = jnp.zeros(self.potion_index.shape, state.potions.dtype)
        if self.equipment is None or not self.equipment.capacity:
            return jnp.where(self.potion_index >= 0, potions, 0)
        owned = jnp.sum(state.item_inventory[None, :] == self.item_index[:, None], axis=1)
        return jnp.where(self.potion_index >= 0, potions, owned)

    def access(self, env, state):
        """Potions always; orbs/talismans: sorcery lore (level 10), a mage lord or a book."""
        hero = env._equipment_hero(state)
        lore = jnp.bool_(self.mage_lord) | ((hero >= 0) & (state.unit_levels[jnp.maximum(hero, 0)] >= 10))
        allowed = ~(self.orb | self.talisman) | lore
        if self.equipment is not None:
            allowed |= self.orb & self.equipment.value(state, 'orbs', 3)
            allowed |= self.talisman & self.equipment.value(state, 'talismans', 3)
        return allowed

    def equip_available(self, env, state):
        reserved = jnp.sum(state.battle_items[:, None] == jnp.arange(self.count)[None, :], axis=0)
        hero = env._equipment_hero(state)
        return (~state.in_battle & ~state.done & ~state.battle_item_equipped_today
                & (hero >= 0) & (state.hp[jnp.maximum(hero, 0)] > 0)
                & jnp.any(state.battle_items < 0) & self.access(env, state)
                & (self.stock(state)[:self.count] > reserved))

    def inventory_positions(self, state):
        """Slots reserve the first copies of each kind, in slot order."""
        kinds = jnp.maximum(state.battle_items, 0)
        matches = state.item_inventory[None, :] == self.item_index[kinds, None]
        ordinal = jnp.array([1, 1+(state.battle_items[0] == state.battle_items[1])], jnp.int32)
        selected = matches & (jnp.cumsum(matches, axis=1) == ordinal[:, None])
        valid = (state.battle_items >= 0) & (self.item_index[kinds] >= 0) & jnp.any(selected, axis=1)
        return jnp.where(valid, jnp.argmax(selected, axis=1), state.item_inventory.size)

    def save_charges(self, state):
        if not state.item_charges_spent.size:
            return state
        positions = self.inventory_positions(state)
        talismans = (state.battle_items >= 0) & self.talisman[jnp.maximum(state.battle_items, 0)]
        positions = jnp.where(talismans, positions, state.item_inventory.size)
        return state.replace(item_charges_spent=set_rows(state.item_charges_spent, positions,
            TALISMAN_USES-state.battle_item_charges))

    def equip(self, env, state, action):
        if not self.count:
            return state
        kind = jnp.clip(action-self.equip_start, 0, self.count-1)
        using = (action >= self.equip_start) & (action < self.use_start) & self.equip_available(env, state)[kind]
        slot = jnp.argmax(state.battle_items < 0)
        selected = using & (jnp.arange(SLOTS) == slot)
        charges = self.uses[kind]
        if state.item_charges_spent.size:
            state = self.save_charges(state)
            matches = state.item_inventory == self.item_index[kind]
            reserved = jnp.sum(state.battle_items == kind)
            copy = jnp.argmax(matches & (jnp.cumsum(matches) == reserved+1))
            charges -= jnp.where(self.talisman[kind], state.item_charges_spent[copy], 0)
        state = state.replace(battle_items=jnp.where(selected, kind, state.battle_items),
            battle_item_charges=jnp.where(selected, charges, state.battle_item_charges),
            battle_item_used=state.battle_item_used & ~selected,
            battle_item_equipped_today=state.battle_item_equipped_today | using,
            last_battle_item=jnp.where(using, kind, -1),
            last_event=jnp.where(using, ITEM_EQUIPPED, state.last_event))
        # Filling slot 0 while slot 1 is occupied changes the reservation order.
        return self.save_charges(state)

    def sync(self, env, state):
        """Return slots to inventory on the map if the hero is dead or access is lost.

        Inventory already owns the reserved copies. Preserve partial talismans;
        death during combat does not unequip them (reference CEffectEndBattle).
        """
        if not self.count:
            return state
        items = state.battle_items
        kind = jnp.maximum(items, 0)
        hero = env._equipment_hero(state)
        fallen = (hero < 0) | (state.hp[jnp.maximum(hero, 0)] <= 0)
        # Map summons temporarily replace the roster before in_battle is set.
        # Their absent hero is still with the stashed party and keeps its equipment.
        held = (items >= 0) & (state.summon_battle
                              | jnp.where(fallen, state.in_battle, self.access(env, state)[kind]))
        first = held[0] & (self.stock(state)[kind[0]] >= 1)
        second = held[1] & (self.stock(state)[kind[1]] >= 1+(first & (items[0] == items[1])).astype(jnp.int32))
        keep = jnp.stack((first, second))
        state = self.save_charges(state)
        return state.replace(battle_items=jnp.where(keep, items, -1),
            battle_item_charges=jnp.where(keep, state.battle_item_charges, 0),
            battle_item_used=state.battle_item_used & keep)

    def finish_battle(self, env, state):
        """Dead leader's reserved items return to the surviving party's inventory."""
        hero = env._equipment_hero(state)
        fallen = (hero < 0) | (state.hp[jnp.maximum(hero, 0)] <= 0)
        state = self.save_charges(state)
        return state.replace(battle_items=jnp.where(fallen, -1, state.battle_items),
            battle_item_charges=jnp.where(fallen, 0, state.battle_item_charges),
            battle_item_used=state.battle_item_used & ~fallen)

    # ------------------------------ battle: use ------------------------------
    def _valid_targets(self, env, state, kind):
        """[12] targets satisfying the item's team, target state and party rules."""
        code = self.kind[kind]
        side = (env.slots >= 6) == self.enemy[kind]
        alive = (state.hp > 0) & ~state.escaped
        valid = alive
        if REVIVE in self.kinds:
            rules = env.progression
            sizes, natural = rules.sizes[env._effective_ids(state)], rules.sizes[state.unit_ids]
            pair = jnp.where(env.slots % 6 < 3, env.slots+3, env.slots-3)
            covered = (env.slots % 6 >= 3) & alive[pair] & (sizes[pair] == 2)
            free = ~covered & ~((natural == 2) & alive[pair])
            dead = (state.unit_ids != 0) & can_revive(state, env.slots) & ~state.escaped & free
            valid = jnp.where(code == REVIVE, dead, valid)
        if SUMMON in self.kinds:
            valid = jnp.where(code == SUMMON, env._summon_targets(state), valid)
        if BUFF in self.kinds:
            original = env._original_primary(state)
            current = env._combat_stats(state, cap=False)[:, DAMAGE]
            boosted = jnp.rint(original*(self.multiplier[kind]-jnp.where(state.weakened,.33,0.)))
            buffable = alive & (original > 0) & (boosted != current)
            party = alive & jnp.any(buffable & side)
            valid = jnp.where(code == BUFF, jnp.where(self.party[kind], party, buffable), valid)
        return valid & side

    def use_available(self, env, state):
        """[24] battle actions; the mask and step share this predicate."""
        if not self.count:
            return jnp.zeros(0, bool)
        actor = state.actor
        acting = (state.in_battle & ~state.done & (actor < 6) & (actor == env._equipment_hero(state))
                  & ~state.retreating[actor] & ~state.paralyzed[actor] & ~state.long_paralyzed[actor]
                  & ~state.second_strike & (state.post_victory == 0))
        masks = []
        for slot in range(SLOTS):
            kind = jnp.maximum(state.battle_items[slot], 0)
            ready = (state.battle_items[slot] >= 0) & ~(self.talisman[kind] & (
                state.battle_item_used[slot] | (state.battle_item_charges[slot] <= 0)))
            masks.append(acting & ready & self._valid_targets(env, state, kind))
        return jnp.concatenate((masks[0][:6], masks[1][:6], masks[0][6:], masks[1][6:]))

    def begin_battle(self, state):
        return state.replace(battle_item_used=jnp.zeros(SLOTS, bool))

    @staticmethod
    def rest(state, resting):
        return state.replace(battle_item_equipped_today=state.battle_item_equipped_today & ~resting)

    def use(self, env, state, hp, used, action, random_values, stats, traits):
        """Apply one validated hero item action. The hero's turn is spent by the caller."""
        slots = env.slots
        index = jnp.clip(action-self.use_start, 0, ACTIONS-1)
        using = (action >= self.use_start) & (action < self.end)
        slot = (index // TARGETS) % SLOTS
        enemy = index >= SLOTS*TARGETS
        target = index % TARGETS+jnp.where(enemy, 6, 0)
        kind = jnp.maximum(state.battle_items[slot], 0)
        code, amount, party = self.kind[kind], self.amount[kind], self.party[kind]
        using &= (code != REVIVE) | can_revive(state, target)
        alive = (hp > 0) & ~state.escaped
        side = (slots >= 6) == enemy
        chosen = jnp.where(party, side & alive, slots == target) & using
        maximum = stats[:, HP].astype(jnp.int32)
        target_slots = jnp.arange(6)+jnp.where(enemy, 6, 0)
        applied = jnp.int32(0)
        immune, warded = jnp.uint32(0), jnp.uint32(0)
        credit = jnp.zeros(6, jnp.int32)
        extra = jnp.zeros(12, bool)
        status = jnp.zeros(6, bool)

        def protect(kind_code, eligible, used):
            hit, used, blocked, ward = source_protection(
                (chosen & eligible & (code == kind_code))[target_slots], self.source[kind], traits, used, target_slots)
            return hit, used, blocked, ward

        if HEAL in self.kinds:
            healed = jnp.where(chosen & alive & (code == HEAL), jnp.minimum(amount, jnp.maximum(maximum-hp, 0)), 0)
            hp = hp+healed
            applied += jnp.sum(healed)
        if REVIVE in self.kinds:
            revived = chosen & (code == REVIVE)
            restored = jnp.where(self.percent[kind] > 0, jnp.rint(maximum*self.percent[kind]/100.), amount)
            restored = jnp.minimum(maximum, jnp.maximum(1, restored.astype(jnp.int32)))
            hp = jnp.where(revived, restored, hp)
            # Death erased the personal award; revival never restores it.
            state = state.replace(battle_earned_xp=jnp.where(revived, 0, state.battle_earned_xp),
                turn_phase=jnp.where(revived, 2, state.turn_phase), priority=jnp.where(revived, 0., state.priority),
                retreating=state.retreating & ~revived, paralyzed=state.paralyzed & ~revived,
                long_paralyzed=state.long_paralyzed & ~revived, feared=state.feared & ~revived)
            state, hp, used = env._cleanse(state, hp, used, revived)
            applied += jnp.sum(jnp.where(revived, hp, 0))
        if SUMMON in self.kinds:
            spawned = chosen & (code == SUMMON)
            uid = self.summon[kind]
            level = env.progression.base_levels[uid]
            fields = env._empty_summon_fields(state)
            updates = {name: jnp.where(spawned.reshape((12,)+(1,)*(value.ndim-1)), value, getattr(state, name))
                       for name, value in fields.items()}
            for dot in DOT_FIELDS:
                source = updates[dot+'_source']
                updates[dot+'_source'] = jnp.where((source >= 0) & spawned[jnp.maximum(source, 0)], -1, source)
            updates.update(env._summon_backups(state, spawned))
            # The summon acts this round at its template initiative and leaves with
            # the battle. Its owner is the hero who used the item: like the original
            # (user check 2026-10-14), it vanishes as soon as the hero dies or flees.
            # The original fighter keeps its saved HP (zero for a corpse, positive
            # for an escapee).
            updates.update(unit_ids=jnp.where(spawned, uid, state.unit_ids),
                unit_levels=jnp.where(spawned, level, state.unit_levels),
                unit_xp=jnp.where(spawned, 0, state.unit_xp),
                summon_owner=jnp.where(spawned, state.actor, state.summon_owner),
                turn_phase=jnp.where(spawned, 0, state.turn_phase),
                activation_done=jnp.where(spawned, False, state.activation_done),
                priority=jnp.where(spawned, env.progression.base_stats[uid, INITIATIVE], state.priority))
            state = state.replace(**updates)
            hp = jnp.where(spawned, env.progression.stats(uid, level)[HP].astype(jnp.int32), hp)
            used = jnp.where(spawned, jnp.uint32(0), used)
            applied += jnp.any(spawned).astype(jnp.int32)
        if STRIKE in self.kinds or DRAIN in self.kinds:
            hits = accuracy_hits(self.accuracy[kind], random_values[12:18], random_values[18:24])
            struck, used, blocked, ward = protect(STRIKE, alive & hits[jnp.arange(12) % 6], used)
            if DRAIN in self.kinds:
                drained, used, drain_blocked, drain_ward = protect(DRAIN, alive & hits[jnp.arange(12) % 6], used)
                struck |= drained
                blocked |= drain_blocked
                ward |= drain_ward
            immune |= blocked
            warded |= ward
            # Same integer armor subtraction as unit attacks, including Defend.
            # Item powers are fixed (at most 100), below the 300 damage limit.
            bonus = jnp.minimum((random_values[24:30]*6).astype(jnp.int32), 5)
            armor = jnp.clip(stats[target_slots, ARMOR], 0, 90).astype(jnp.int32)
            damage = env.progression.damage_rolls[amount, armor, state.defended[target_slots].astype(jnp.int32), bonus]
            removed = jnp.where(struck, jnp.minimum(damage, hp[target_slots]), 0)
            hp = set_rows(hp, target_slots, hp[target_slots]+(-removed))
            credit = jnp.where(enemy, removed, 0)
            applied += jnp.sum(removed)
            if DRAIN in self.kinds:
                pool = jnp.where(code == DRAIN, jnp.sum(removed)//2, 0)
                hp = vampiric_heal(hp, maximum, state.escaped, state.actor, pool, self.share[kind])
        if DOT in self.kinds:
            affected, used, blocked, ward = protect(DOT, alive, used)
            immune |= blocked
            warded |= ward
            duration = 2
            updates = {}
            for dot in sorted(self.dot_kinds):
                name = DOT_FIELDS[dot]
                turns, damage = getattr(state, name+'_mode'), getattr(state, name+'_damage')
                applying = affected & (self.dot[kind] == dot)
                state = env._record_effect(state,target_slots,applying,{'poison':0,'water':1,'burn':2}[name])
                updates[name+'_mode'] = set_rows(turns, target_slots, jnp.where(applying, duration, turns[target_slots]))
                updates[name+'_damage'] = set_rows(damage, target_slots, jnp.where(applying, amount, damage[target_slots]))
                source = getattr(state, name+'_source')
                updates[name+'_source'] = set_rows(source, target_slots, jnp.where(applying, state.actor, source[target_slots]))
                applied += jnp.sum(applying)*amount
            state = state.replace(**updates)
        if BUFF in self.kinds:
            original = env._original_primary(state)
            current = env._combat_stats(state, cap=False)[:, DAMAGE]
            boosted = jnp.rint(original*(self.multiplier[kind]-jnp.where(state.weakened,.33,0.))).astype(jnp.int32)
            buffed = chosen & alive & (code == BUFF) & (original > 0) & (boosted != current)
            state = state.replace(primary_override=jnp.where(buffed, boosted, state.primary_override),
                # Original g000aa9120/9121 have INFINITE=T, for both orb and talisman.
                powerup=state.powerup | buffed, powerup_long=state.powerup_long | buffed, powerup_layered=state.powerup_layered & ~buffed)
            applied += jnp.sum(jnp.where(buffed, boosted-original, 0))
        if EXTRA_TURN in self.kinds:
            extra = chosen & alive & (code == EXTRA_TURN)
            state = state.replace(turn_phase=jnp.where(extra, 0, state.turn_phase),
                priority=jnp.where(extra, stats[:, INITIATIVE], state.priority),
                bonus_turns=state.bonus_turns+extra.astype(jnp.int32))
            status |= extra[:6]  # an ally-only effect
        if PARALYSIS in self.kinds:
            accuracy = disable_attack_accuracy(100., jnp.bool_(True), state.round)
            affected, used, blocked, ward = protect(PARALYSIS, alive & ~state.long_paralyzed & (accuracy > 0), used)
            immune |= blocked
            warded |= ward
            status |= affected
            state = env._record_disable(state,target_slots,affected)
            # g000aa9139 is long paralysis, unlike the two finite equipment procs.
            state = state.replace(long_paralyzed=set_rows(state.long_paralyzed, target_slots, state.long_paralyzed[target_slots] | affected))
        if FEAR in self.kinds:
            affected, used, blocked, ward = protect(FEAR, alive & ~state.retreating, used)
            immune |= blocked
            warded |= ward
            protected = env._fear_protected(state, target_slots)
            state = env._record_disable(state,target_slots,affected & protected)
            status |= affected & ~(protected & state.paralyzed[target_slots])
            state = state.replace(
                paralyzed=set_rows(state.paralyzed, target_slots, state.paralyzed[target_slots] | (affected & protected)),
                retreating=set_rows(state.retreating, target_slots, state.retreating[target_slots] | (affected & ~protected)),
                feared=set_rows(state.feared, target_slots, state.feared[target_slots] | (affected & ~protected)))
        if TRANSFORM in self.kinds:
            from stoix.envs.number_grid_item_combat import transform_imp
            # Like witches and Hag's Ring, reject guardians before spending wards.
            guards = env.progression.capital_guards[state.unit_ids]
            affected, used, blocked, ward = protect(TRANSFORM, alive & ~guards, used)
            immune |= blocked
            warded |= ward
            selected = affected & ~state.imp[target_slots] & (state.decay_form[target_slots] == 0)
            status |= selected
            state, used = transform_imp(env, state, used, target_slots, selected)
        if LYCANTHROPY in self.kinds:
            from stoix.envs.number_grid_item_combat import transform_form
            affected, used, blocked, ward = protect(LYCANTHROPY, alive, used)
            immune |= blocked
            warded |= ward
            selected = affected & ~state.imp[target_slots] & (state.decay_form[target_slots] == 0) & ~state.fenrir[target_slots]
            state, hp, used, selected = transform_form(env, state, hp, used, target_slots, selected,
                                                       jnp.full(6, self.summon[kind]))
            status |= selected
            state = env._record_effect(state,target_slots,selected,4)
            state = state.replace(transform_long=set_rows(state.transform_long, target_slots, state.transform_long[target_slots] | selected),lycanthrope=set_rows(state.lycanthrope, target_slots, state.lycanthrope[target_slots] | selected))
        if WEAKEN in self.kinds:
            affected, used, blocked, ward = protect(WEAKEN, alive, used)
            immune |= blocked
            warded |= ward
            newly = affected & ~state.weakened[target_slots]
            status |= newly
            if env.has_powerups:
                raw = env._combat_stats(state, cap=False)[target_slots, DAMAGE].astype(jnp.int32)
                state = state.replace(
                    primary_override=set_rows(state.primary_override, target_slots, jnp.where(
                        newly, jnp.rint(raw-env._original_primary(state)[target_slots]*(1-self.multiplier[kind])).astype(jnp.int32), state.primary_override[target_slots])),
                    preweak_damage=set_rows(state.preweak_damage, target_slots, jnp.where(newly, raw, state.preweak_damage[target_slots])),
                    powerup_layered=set_rows(state.powerup_layered, target_slots, jnp.where(
                        newly, state.powerup[target_slots], state.powerup_layered[target_slots])))
            state = state.replace(weakened=set_rows(state.weakened, target_slots, state.weakened[target_slots] | newly),
                item_weakened=set_rows(state.item_weakened, target_slots, state.item_weakened[target_slots] | newly))
        applied += jnp.sum(status).astype(jnp.int32)
        state = self.consume(state, slot, kind, using)
        return state.replace(last_battle_item=jnp.where(using, kind, -1)), hp, used, dict(
            using=using, target=jnp.where(party, -1, target), applied=applied, immune=immune,
            ward=warded, credit=credit, extra=extra)

    def consume(self, state, slot, kind, using):
        """Orbs and potions are spent; a talisman spends one of five charges, once per battle."""
        from stoix.envs.number_grid_items import remove_items

        talisman = self.talisman[kind]
        charges = state.battle_item_charges[slot]-1
        spent = using & (~talisman | (charges <= 0))
        mask = jnp.arange(SLOTS) == slot
        remove = None
        if self.equipment is not None and self.equipment.capacity:
            # Select before clearing the slot, including the second identical copy.
            remove = spent & (jnp.arange(state.item_inventory.size) == self.inventory_positions(state)[slot])
        state = state.replace(battle_item_charges=jnp.where(mask & using, charges, state.battle_item_charges))
        state = self.save_charges(state)
        state = state.replace(
            battle_items=jnp.where(mask & spent, -1, state.battle_items),
            battle_item_charges=jnp.where(mask & using, jnp.where(spent, 0, charges), state.battle_item_charges),
            battle_item_used=state.battle_item_used | (mask & using & talisman & ~spent))
        potion = self.potion_index[kind]
        if self.potions.count:
            index = jnp.maximum(potion, 0)
            state = state.replace(potions=set_rows(state.potions, index, state.potions[index]+(
                -(spent & (potion >= 0)).astype(jnp.int32))))
        if self.equipment is not None and self.equipment.capacity:
            state = remove_items(state, remove)
        return state

    def observation(self, state):
        kinds = state.battle_items[:, None] == jnp.arange(-1, self.count)[None, :]
        return jnp.concatenate((kinds.reshape(-1).astype(jnp.float32),
                                state.battle_item_charges/float(TALISMAN_USES)))

    def quotes(self, env, state):
        return dict(equip=self.equip_available(env, state), use=self.use_available(env, state),
                    stock=self.stock(state)[:self.count], access=self.access(env, state)[:self.count])

    @property
    def metadata(self):
        rows = [dict(key=item['key'], name=item['name'], label=item['label'], category=item['category'],
                     kind=KIND_NAMES[item['kind']], game_id=item.get('game_id'), price=item.get('price'),
                     uses=item['uses'], enemy=item['enemy'], party=item['party'],
                     equip_action=self.equip_start+i) for i, item in enumerate(self.items)]
        return dict(items=rows, slots=SLOTS, talisman_uses=TALISMAN_USES, use_start=self.use_start,
                    targets=TARGETS, mage_lord=self.mage_lord)
