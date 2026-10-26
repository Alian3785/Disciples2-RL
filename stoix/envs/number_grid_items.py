"""Hero equipment catalogue, fixed inventory and automatic equipment selection.

Data: installed Rise of the Elves GItem.dbf and Python reference
campaign_env_data.py / campaign_env_inventory.py. See docs/HERO_ITEMS.md.
"""
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_battle_items import MAGIC_ITEMS, TALISMAN_USES
from stoix.envs.number_grid_hero_skills import lore_levels
from stoix.envs.number_grid_spell_items import SPELL_ITEMS
from stoix.envs.number_grid_updates import set_column, set_rows

ARTIFACT, BANNER, BOOK, BOOTS, VALUABLE, USABLE, SCROLL, STAFF = range(8)
SLOT_NAMES = ('Артефакт 1', 'Артефакт 2', 'Знамя', 'Книга', 'Сапоги')


def _item(key, name, gid, category, price, label, **effects):
    return dict(key=key, name=name, game_id='G000IG'+gid, category=category,
                price=price, label=label, **effects)


ITEMS = (
    # S101 trophies: invisibility is not implemented; the zero-value quest totem
    # must survive automatic sales until its scripted exchange.
    _item('sanctuera_scroll', 'Sanctuera scroll', '5051', VALUABLE, 600,
          'Свиток невидимости: эффект пока недоступен, продажа за 120 золота', sell_price=120),
    _item('agshlisga_totem', 'Agshlisga Totem', '7101', VALUABLE, 0,
          'Тотем Агшлисга: сценарный предмет для обмена'),
    # GItem G000IG5092. Its invisibility spell is unavailable in both campaign
    # catalogues; preserve the Charmed Child trophy and its 20% resale value.
    _item('illudere_terra_scroll', 'Illudere terra scroll', '5092', VALUABLE, 2000,
          'Свиток невидимости: трофей, продажа за 400 золота', sell_price=400),
    _item('runestone','Runestone','2001',ARTIFACT,500,'Герою: +10 брони',armor=10),
    _item('holy_chalice','Holy Chalice','2002',ARTIFACT,1000,'Герою: +15 брони',armor=15),
    _item('skull_bracers','Skull Bracers','2003',ARTIFACT,1500,'Герою: +20 брони',armor=20),
    _item('horn_of_awareness','Horn of Awareness','2004',ARTIFACT,2000,'Герою: +25 брони',armor=25),
    _item('etched_circlet','Etched Circlet','2005',ARTIFACT,2500,'Герою: +35 брони',armor=35),
    _item('rusted_shackles','Rusted Shackles','2007',ARTIFACT,1500,'Герой отступает немедленно',instant_retreat=True),
    _item('dwarven_bracer','Dwarven Bracer','3001',ARTIFACT,500,'Герою: +10% урона',damage=1.10),
    _item('unholy_chalice','Unholy Chalice','3002',ARTIFACT,1000,'Герою: +15% урона',damage=1.15),
    _item('ring_of_strength','Ring of Strength','3003',ARTIFACT,1500,'Герою: +20% урона',damage=1.20),
    _item('runic_blade','Runic Blade','3004',ARTIFACT,2000,'Герою: +25% урона',damage=1.25),
    _item('mjolnirs_crown',"Mjolnir's Crown",'3005',ARTIFACT,2500,'Герою: +35% урона',damage=1.35),
    _item('ring_of_ages','Ring of the Ages','3006',ARTIFACT,3750,'Герою: +40% урона, +25% инициативы',damage=1.40,initiative=1.25),
    _item('bethrezens_claw',"Bethrezen's Claw",'3007',ARTIFACT,5000,'Герою: +40% урона, +50% инициативы',damage=1.40,initiative=1.50),
    _item('soul_crystal','Soul Crystal','3015',ARTIFACT,2000,'После попадания: паралич, 80%, Разум',status='paralysis',chance=.8,source=7),
    _item('horn_of_incubus','Horn of Incubus','3016',ARTIFACT,2000,'После попадания: окаменение, 80%, Земля',status='paralysis',chance=.8,source=2),
    _item('unholy_dagger','Unholy Dagger','3017',ARTIFACT,1500,'Герою: лечение на 25% нанесённого урона',drain=True),
    _item('wight_blade','Wight Blade','3018',ARTIFACT,5000,'После попадания: понижение формы, 75%, Смерть',status='decay',chance=.75,source=6),
    _item('thanatos_blade','Thanatos Blade','3019',ARTIFACT,1500,'После попадания: яд 20 HP, 80%, 1–6 ходов',status='poison',chance=.8,source=6,poison=20),
    _item('skull_of_thanatos','Skull of Thanatos','3020',ARTIFACT,3750,'После попадания: яд 35 HP, 80%, 1–6 ходов',status='poison',chance=.8,source=6,poison=35),
    _item('hags_ring',"Hag's Ring",'3021',ARTIFACT,2500,'После попадания: превращение, 70%, Разум',status='imp',chance=.7,source=7),
    _item('lute_of_charming','Lute of Charming','3022',ARTIFACT,1500,'Скидка 10% у торговцев',discount=10),
    _item('banner_protection','Banner of Protection','1001',BANNER,1000,'Отряду: +10 брони',armor=10,priority=3),
    _item('banner_resistance','Banner of Resistance','1002',BANNER,3000,'Отряду: +15 брони',armor=15,priority=7),
    _item('banner_striking','Banner of Striking','1003',BANNER,1000,'Отряду: +10% точности',accuracy_percent=10,priority=2),
    _item('banner_battle','Banner of Battle','1004',BANNER,3000,'Отряду: +15% точности',accuracy_percent=15,priority=6),
    _item('banner_speed','Banner of Speed','1005',BANNER,1000,'Отряду: +10% инициативы',initiative=1.10,priority=1),
    _item('banner_celerity','Banner of Celerity','1006',BANNER,3000,'Отряду: +15% инициативы',initiative=1.15,priority=5),
    _item('banner_strength','Banner of Strength','1007',BANNER,1000,'Отряду: +10% урона',damage=1.10,priority=4),
    _item('banner_might','Banner of Might','1008',BANNER,3000,'Отряду: +15% урона',damage=1.15,priority=8),
    _item('banner_fortitude','Banner of Fortitude','1015',BANNER,5000,'Отряду: +20 брони',armor=20,priority=10),
    _item('banner_war','Banner of War','1016',BANNER,5000,'Отряду: +20% урона',damage=1.20,priority=11),
    _item('banner_health','Banner of Health','1017',BANNER,5000,'Отряду: +10% регенерации при отдыхе',regeneration=1,priority=9),
    _item('tome_air','Tome of Air','4001',BOOK,400,'Герою: защита от первой атаки Воздуха',ward=256),
    _item('tome_water','Tome of Water','4002',BOOK,400,'Герою: защита от первой атаки Воды',ward=8),
    _item('tome_earth','Tome of Earth','4003',BOOK,400,'Герою: защита от первой атаки Земли',ward=2),
    _item('tome_fire','Tome of Fire','4004',BOOK,400,'Герою: защита от первой атаки Огня',ward=4),
    _item('tome_war','Tome of War','4005',BOOK,600,'Отряду: +25% опыта за бой',experience=True),
    _item('tome_arcanum','Tome of Arcanum','4006',BOOK,900,'Разрешает сферы в боевых слотах героя',orbs=True),
    _item('tome_sorcery','Tome of Sorcery','4007',BOOK,1200,'Разрешает талисманы в боевых слотах героя',talismans=True),
    _item('tome_mind','Tome of Thought','4008',BOOK,900,'Герою: защита от первой атаки Разума',ward=64),
    _item('elven_boots','Elven Boots','1010',BOOTS,1200,'Движение по лесу: 2 очка',forest=True),
    _item('elemental_boots','Boots of the Elements','1011',BOOTS,1600,'Движение по воде: 2 очка',water=True),
    _item('boots_speed','Boots of Speed','8003',BOOTS,400,'Отряду: +20% очков движения',movement_percent=20),
    _item('boots_traveling','Boots of Traveling','8004',BOOTS,1200,'Отряду: +40% очков движения',movement_percent=40),
    _item('boots_seven_leagues','Boots of Seven Leagues','8005',BOOTS,2000,'Отряду: +60% очков движения',movement_percent=60),
    *(_item(key,name,str(7001+i),VALUABLE,price,
            f'Цена продажи: {price//5} золота',sell_price=price//5)
      for i,(key,name,price) in enumerate((
          ('bronze_ring','Bronze Ring',250),('silver_ring','Silver Ring',500),
          ('emerald','Emerald',750),('gold_ring','Gold Ring',1000),('ruby','Ruby',1250),
          ('sapphire','Sapphire',1500),('diamond','Diamond',1750),
          ('ancient_relic','Ancient Relic',2000),('royal_scepter','Royal Scepter',2500),
          ('imperial_crown','Imperial Crown',5000)))),
    # Orbs and talismans: hero battle slots, number_grid_battle_items.
    *(dict(key=p['key'],name=p['name'],game_id=p['game_id'],category=USABLE,price=p['price'],label=p['label'])
      for p in MAGIC_ITEMS),
    # Scrolls and staves: map casts, number_grid_casting / number_grid_spell_items.
    *(dict(key=p['key'],name=p['name'],game_id=p['game_id'],category=SCROLL if p['category']=='scroll' else STAFF,
           price=p['price'],label=p['label']) for p in SPELL_ITEMS),
)
ITEM_BY_KEY = {p['key']: p for p in ITEMS}


def item_counts(value, label):
    if not isinstance(value, dict) or any(k not in ITEM_BY_KEY for k in value):
        raise ValueError(label+' must map known item keys to counts')
    if any(type(n) is not int or not 0 <= n <= 1000 for n in value.values()):
        raise ValueError(label+' must contain nonnegative counts no greater than 1000')
    return value


def remove_items(state, remove):
    """Compact inventory copies and their charge histories together."""
    inventory = state.item_inventory
    keep = (inventory >= 0) & ~remove
    destination = jnp.where(keep, jnp.cumsum(keep)-1, inventory.size)
    changes = dict(item_inventory=set_rows(jnp.full_like(inventory, -1), destination, inventory))
    if state.item_charges_spent.size:
        changes['item_charges_spent'] = set_rows(jnp.zeros_like(state.item_charges_spent), destination,
            state.item_charges_spent)
    return state.replace(**changes)


class ItemRules:
    def __init__(self, game_map):
        inventories = [item_counts(game_map.get('initial_items',{}),'initial_items')]
        inventories += [item_counts(c.get('items',{}),'Chest items') for c in game_map.get('chests',[])]
        chest_count = len(inventories)-1
        inventories += [item_counts(r.get('items',{}),'Ruin items') for r in game_map.get('ruins',[])]
        ruin_end = len(inventories)
        # City vault items go to the hero's inventory on capture (docs/TERRITORY_CITIES.md).
        inventories += [item_counts(c.get('items',{}),'City items') for c in game_map.get('cities',[])]
        city_end = len(inventories)
        # Scenario "on day N" item events (docs/HERO_ITEMS.md); day 1 would need a reset grant.
        days = game_map.get('day_items', [])
        if not isinstance(days, list) or any(not isinstance(d, dict) or type(d.get('day')) is not int
                                             or d['day'] < 2 for d in days):
            raise ValueError('day_items must list objects with an integer day of at least 2')
        inventories += [item_counts(d.get('items',{}),'Day items') for d in days]
        # Every merchant copy may end up in the inventory; validated by number_grid_sites.
        from stoix.envs.number_grid_potions import merchants
        stock = [item_counts(m.get('items',{}),'Merchant items') for m in merchants(game_map)]
        stock += [item_counts(game_map.get('event_items', {}), 'Event items')]
        used = {key for contents in inventories+stock for key,count in contents.items() if count}
        self.items = tuple(p for p in ITEMS if p['key'] in used)
        self.keys = tuple(p['key'] for p in self.items)
        self.count = len(self.items)
        self.capacity = sum(sum(c.values()) for c in inventories+stock)
        talismans = {p['key'] for p in MAGIC_ITEMS if p['category'] == 'talisman'}
        self.charge_capacity = self.capacity if game_map.get('battle_items', False) and used & talismans else 0
        if self.capacity > 1000:
            raise ValueError('A scenario may contain at most 1000 equipment/valuable items')
        self.policy = game_map.get('equipment_auto_equip','price')
        if self.policy not in ('price','reference'):
            raise ValueError('equipment_auto_equip must be price or reference')
        self.require_skills = bool(game_map.get('equipment_requires_skills',False))
        # Banner 7, boots 8, artifacts 9, book 11; the lord's own lore from the start.
        self.skill_levels = lore_levels(game_map)
        # Table row 0 is the empty slot; state item IDs are zero-based, -1 empty.
        rows = ({},)+self.items
        self.tables = {key:jnp.array([p.get(key,default) for p in rows],dtype)
            for key,default,dtype in (
                ('category',-1,jnp.int32),('price',0,jnp.int32),('priority',0,jnp.int32),
                ('movement_percent',0,jnp.int32),('ward',0,jnp.uint32),('regeneration',0,jnp.int32),
                ('damage',1.,jnp.float32),('initiative',1.,jnp.float32),('armor',0,jnp.float32),
                ('accuracy_percent',0,jnp.int32),('experience',False,bool),('instant_retreat',False,bool),
                ('drain',False,bool),('discount',0,jnp.int32),('orbs',False,bool),
                ('talismans',False,bool),('forest',False,bool),('water',False,bool))}
        self.status_items = tuple((i,p) for i,p in enumerate(self.items) if 'status' in p)
        with jax.enable_x64(True):
            for key in ('damage','initiative'):
                self.tables[key]=jnp.array([p.get(key,1.) for p in rows],jnp.float64)
        self.has_drain = any(p.get('drain') for p in self.items)
        initial = [self.keys.index(key) for key,count in inventories[0].items() for _ in range(count)]
        self.initial_inventory = jnp.array(initial+[-1]*(self.capacity-len(initial)),jnp.int32)
        self.chest_loot = jnp.array([[c.get(key,0) for key in self.keys] for c in inventories[1:chest_count+1]],jnp.int32).reshape(chest_count,self.count)
        # Chest and ruin loot features skip merchant- and city-only kinds: their columns are always zero.
        looted = {key for contents in inventories[:ruin_end] for key,count in contents.items() if count}
        self.loot_kinds = [i for i,key in enumerate(self.keys) if key in looted]
        self.observation_size = self.capacity+5+chest_count*len(self.loot_kinds)+self.charge_capacity
        self.loot_chests, self.loot_items, _ = self._tokens(inventories[1:chest_count+1])
        self.loot_ruins, self.ruin_items, self.ruin_loot = self._tokens(inventories[chest_count+1:ruin_end])
        self.loot_cities, self.city_items, self.city_loot = self._tokens(inventories[ruin_end:city_end])
        self.item_days = jnp.array([d['day'] for d in days], jnp.int32)
        self.loot_days, self.day_items, self.day_loot = self._tokens(inventories[city_end:])

    def _tokens(self, sources):
        """One token per item copy: its source index and item id; plus per-source counts."""
        tokens = [(s,self.keys.index(key)) for s,c in enumerate(sources) for key,count in c.items() for _ in range(count)]
        contents = jnp.array([[c.get(key,0) for key in self.keys] for c in sources],jnp.int32).reshape(len(sources),self.count)
        return jnp.array([s for s,_ in tokens],jnp.int32), jnp.array([i for _,i in tokens],jnp.int32), contents

    def value(self, state, key, slot=None):
        return self.tables[key][state.equipped+1 if slot is None else state.equipped[slot]+1]

    def movement_cap(self, state, base_points):
        """Boot percentage of the base allowance; truncate fractional points.

        Use the base, never the previously boosted cap or remaining movement.
        """
        percent=self.value(state,'movement_percent',4)
        return base_points+base_points*percent//100

    def collect(self, state, collected):
        return self._collect(state,collected,self.loot_chests,self.loot_items,self.chest_loot)

    def collect_ruins(self, state, collected):
        return self._collect(state,collected,self.loot_ruins,self.ruin_items,self.ruin_loot)

    def collect_cities(self, state, captured):
        return self._collect(state,captured,self.loot_cities,self.city_items,self.city_loot)

    def collect_days(self, state, resting):
        """The rest that begins day N grants its items; days only grow by one, so once."""
        return self._collect(state,resting & (self.item_days == state.day),self.loot_days,self.day_items,self.day_loot)

    def _collect(self, state, collected, sources, tokens, contents):
        if not len(tokens):
            return state
        gained = collected[sources]
        offset = jnp.sum(state.item_inventory >= 0)
        destinations = jnp.where(gained,offset+jnp.cumsum(gained)-1,self.capacity)
        inventory = set_rows(state.item_inventory,destinations,tokens)
        loot = jnp.sum(jnp.where(collected[:,None],contents,0),axis=0)
        # Added, not replaced: a city capture and a chest may pay out on the same move.
        return state.replace(item_inventory=inventory,last_item_loot=state.last_item_loot+loot)

    def choose(self, state, hero_alive, hero_level):
        if not self.capacity:
            return jnp.full(5,-1,jnp.int32)
        ids = state.item_inventory
        categories = self.tables['category'][ids+1]
        prices = self.tables['price'][ids+1]
        score = prices*16+jnp.where(categories==BANNER,self.tables['priority'][ids+1],0)
        if self.policy == 'reference':
            score += jnp.where(categories==BOOTS,self.tables['movement_percent'][ids+1]*100000,0)
        selected=[]
        artifact_slot=jnp.int32(-1)
        for n,category in enumerate((ARTIFACT,ARTIFACT,BANNER,BOOK,BOOTS)):
            valid=(ids>=0)&(categories==category)
            if n==1:
                valid &= jnp.arange(self.capacity)!=artifact_slot
            best=jnp.argmax(jnp.where(valid,score,-1))
            chosen=jnp.where(jnp.any(valid),ids[best],-1)
            if n==0:
                artifact_slot=best
            if category==BOOK and self.policy=='reference':
                # Reference books retain the first selection, unlike other gear.
                first=jnp.argmax(valid)
                retained=(state.equipped[3]>=0)&jnp.any(ids==state.equipped[3])
                chosen=jnp.where(retained,state.equipped[3],jnp.where(jnp.any(valid),ids[first],-1))
            selected.append(chosen)
        allowed=jnp.full(5,hero_alive)
        if self.require_skills:
            allowed &= hero_level>=jnp.array(self.skill_levels)
        return jnp.where(allowed,jnp.stack(selected),-1)

    def numeric_bonus(self, state, intrinsic, hero):
        """Permanent/temporary potions -> artifacts -> banner; round each layer."""
        values = intrinsic+state.potion_bonus+state.potion_temporary
        slots=jnp.arange(12)
        with jax.enable_x64(True):
            # Products must reproduce Python float/round at e.g. 50*1.1*1.15.
            artifact_damage=jnp.prod(self.value(state,'damage')[:2].astype(jnp.float64))
            artifact_initiative=jnp.prod(self.value(state,'initiative')[:2].astype(jnp.float64))
            factors=jnp.array([1.,artifact_damage,1.,1.,artifact_initiative,1.,1.],jnp.float64)
            artifacts=jnp.rint(values.astype(jnp.float64)*factors).astype(jnp.float32)
        artifacts=set_column(artifacts,3,artifacts[:,3]+jnp.sum(self.value(state,'armor')[:2]).astype(artifacts.dtype))
        values=jnp.where((slots==hero)[:,None],artifacts,values)
        with jax.enable_x64(True):
            factors=jnp.array([1.,self.value(state,'damage',2),1.,1.,self.value(state,'initiative',2),1.,1.],jnp.float64)
            banner=jnp.rint(values.astype(jnp.float64)*factors).astype(jnp.float32)
        # GmodifL POWER is a relative percentage, with integer truncation.
        accuracy=values[:,2].astype(jnp.int32)
        accuracy=accuracy+accuracy*self.value(state,'accuracy_percent',2)//100
        banner=set_column(banner,2,jnp.minimum(100,accuracy))
        banner=set_column(banner,3,banner[:,3]+jnp.asarray(self.value(state,'armor',2)).astype(banner.dtype))
        values=jnp.where(((slots<6)&(state.unit_ids!=0))[:,None],banner,values)
        return values-intrinsic-state.potion_bonus-state.potion_temporary

    def counts(self,state):
        return jnp.sum(state.item_inventory[:,None]==jnp.arange(self.count)[None,:],axis=0)

    def observation(self,state):
        return jnp.concatenate(((state.item_inventory+1)/max(1,self.count),
                                (state.equipped+1)/max(1,self.count),self.chest_loot[:,self.loot_kinds].reshape(-1),
                                state.item_charges_spent/float(TALISMAN_USES)))

    @property
    def metadata(self):
        return dict(items=list(self.items), slots=list(SLOT_NAMES),
                    policy=self.policy, requires_skills=self.require_skills,
                    skill_levels=list(self.skill_levels))
