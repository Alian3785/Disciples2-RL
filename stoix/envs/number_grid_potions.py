"""All original potions; scenario-sized actions. Sources: docs/MAP_POTIONS.md."""
import jax
import jax.numpy as jnp
from stoix.envs.number_grid_limits import MAX_HP

from stoix.envs.number_grid_combat import can_revive
from stoix.envs.number_grid_updates import set_column

POTION_START, POTION_TARGETS = 63, 6
POTION_HEAL, POTION_REVIVE, POTION_BUFF = 19, 20, 37
HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE = range(5)
# Secondary damage/accuracy columns of the 7-column potion layers. User request
# 2026-10-19: potions act only on the primary attack, like banners and artifacts.
# Installed Gmodif/GmodifL give all of them the same L_ATTACK class (QTY_DAMAGE,
# POWER percentages); D2ModdingToolset notes DOT damage ignores attack modifiers.
# These columns stay zero; the Python reference also boosted them.
SECONDARY_COLUMNS = (5, 6)


def _p(key, name, gid, effect, amount, duration='temporary', price=800):
    stat = dict(health=HP, damage=DAMAGE, accuracy=ACCURACY, armor=ARMOR,
                initiative=INITIATIVE).get(effect, -1)
    elements={'Fire':'Огонь','Water':'Вода','Earth':'Земля','Air':'Воздух'}
    suffix=dict(health='% к HP',damage='% к урону',accuracy='% к точности',
                armor=' брони',initiative='% к инициативе')
    label = ('+'+str(amount)+' HP' if effect == 'heal' else 'Воскрешение с 1 HP'
             if effect == 'revive' else 'Защита от первой атаки: '+elements[effect[5:]]
             if effect.startswith('ward_') else '+'+str(amount)+suffix[effect])
    return dict(key=key, name=name, game_id='G000IG'+gid, effect=effect,
                amount=amount, duration=duration, stat=stat, price=price, label=label)


POTIONS = (
    _p('healing','Банка исцеления','0005','heal',50,'instant',150),
    _p('restoration','Бутыль лечения','0006','heal',100,'instant',300),
    _p('ointment','Целебная мазь','0018','heal',200,'instant',600),
    _p('life','Зелье воскрешения','0001','revive',1,'instant',400),
    _p('protection','Зелье защиты','0002','armor',15,price=200),
    _p('bark','Зелье коры дерева','0003','armor',30,price=450),
    _p('invulnerability','Эликсир неуязвимости','0017','armor',50,price=700),
    _p('striking','Зелье меткости','0008','accuracy',15,price=200),
    _p('accuracy','Зелье точности','0009','accuracy',30,price=450),
    _p('swiftness','Зелье проворства','0011','initiative',15,price=200),
    _p('speed','Зелье скорости','0012','initiative',30,price=450),
    _p('celerity','Эликсир быстроты','0019','initiative',60,price=700),
    _p('vigor','Зелье бодрости','0014','damage',15,price=200),
    _p('strength','Зелье силы','0015','damage',30,price=450),
    _p('might','Эликсир энергии','0020','damage',50,price=700),
    _p('iron_skin','Зелье железной кожи','0004','armor',10,'permanent'),
    _p('highfather','Эликсир Всевышнего','0007','health',15,'permanent'),
    _p('fortune','Эликсир удачи','0010','accuracy',10,'permanent'),
    _p('quicksilver','Эликсир инициативы','0013','initiative',10,'permanent'),
    _p('titan','Эликсир Силы титана','0016','damage',10,'permanent'),
    _p('fire_ward','Эликсир защиты от Огня','0024','ward_Fire',0,price=400),
    _p('water_ward','Эликсир защиты от Воды','0022','ward_Water',0,price=400),
    _p('earth_ward','Эликсир защиты от Земли','0023','ward_Earth',0,price=400),
    _p('air_ward','Эликсир защиты от Воздуха','0021','ward_Air',0,price=400),
)
POTION_BY_KEY = {p['key']: p for p in POTIONS}


def potion_counts(value, label):
    """Named inventories decouple map files from dense scenario indices."""
    if not isinstance(value, dict) or any(k not in POTION_BY_KEY for k in value):
        raise ValueError(label+' must map known potion keys to counts')
    if any(type(n) is not int or not 0 <= n <= 2**31-1 for n in value.values()):
        raise ValueError(label+' must contain nonnegative int32 counts')
    return value


def merchants(game_map):
    from stoix.envs.number_grid_sites import site_entries
    return [value for kind, value in site_entries(game_map) if kind == 'merchant']


def scenario_potions(game_map):
    initial = potion_counts(game_map.get('initial_potions', {}), 'initial_potions')
    chests = game_map.get('chests', [])
    if not isinstance(chests, list):
        raise ValueError('chests must be a list')
    ruins = game_map.get('ruins', [])
    if not isinstance(ruins,list):
        raise ValueError('ruins must be a list')
    totals = dict(initial)
    for chest in chests+ruins:
        if not isinstance(chest, dict):
            raise ValueError('Each chest must contain position and potions')
        contents = potion_counts(chest.get('potions',{}), 'Chest potions')
        from stoix.envs.number_grid_items import item_counts
        items=item_counts(chest.get('items',{}),'Chest items')
        if not any(contents.values()) and not any(items.values()) and not (chest in ruins and chest.get('gold',0)>0):
            raise ValueError('Chest potions must contain some loot')
        for key, count in contents.items():
            totals[key] = totals.get(key, 0)+count
            if totals[key] > 2**31-1:
                raise ValueError('Chest loot plus initial inventory must fit int32')
    # City vault potions go to the party on capture; a city may hold none.
    for city in game_map.get('cities', []):
        for key, count in potion_counts(city.get('potions', {}), 'City potions').items():
            totals[key] = totals.get(key, 0)+count
            if totals[key] > 2**31-1:
                raise ValueError('City loot plus initial inventory must fit int32')
    for merchant in merchants(game_map):
        if not isinstance(merchant, dict):
            raise ValueError('merchant must be an object')
        for key, count in potion_counts(merchant.get('potions', {}), 'Merchant potions').items():
            if count <= 0:
                raise ValueError('Merchant offers must have positive stock')
            totals[key] = totals.get(key, 0)+count
            if totals[key] > 2**31-1:
                raise ValueError('All obtainable potions must fit int32')
    return tuple(p for p in POTIONS if totals.get(p['key'], 0) > 0)


class PotionRules:
    def __init__(self, game_map):
        self.items = scenario_potions(game_map)
        self.count = len(self.items)
        self.action_count = self.count*POTION_TARGETS
        self.keys = tuple(p['key'] for p in self.items)
        self.has_buffs = any(p['duration'] != 'instant' for p in self.items)
        self.initial_counts = jnp.array([game_map.get('initial_potions',{}).get(k,0) for k in self.keys],jnp.int32)
        cities = game_map.get('cities', [])
        self.city_loot = jnp.array([[c.get('potions',{}).get(k,0) for k in self.keys] for c in cities],
                                   jnp.int32).reshape(len(cities),self.count)
        self.amounts = jnp.array([p['amount'] for p in self.items],jnp.int32)
        self.stats = jnp.array([max(0,p['stat']) for p in self.items],jnp.int32)
        self.healing = jnp.array([p['effect'] == 'heal' for p in self.items],bool)
        self.reviving = jnp.array([p['effect'] == 'revive' for p in self.items],bool)
        self.temporary = jnp.array([p['duration'] == 'temporary' for p in self.items],bool)
        self.permanent = jnp.array([p['duration'] == 'permanent' for p in self.items],bool)
        self.bits = jnp.array([1 << POTIONS.index(p) for p in self.items],jnp.uint32)
        source_bits = {'ward_Fire':4,'ward_Water':8,'ward_Earth':2,'ward_Air':256}
        self.wards = jnp.array([source_bits.get(p['effect'],0) for p in self.items],jnp.uint32)
        # Python reference multiplies all distinct temporary doses before rounding.
        # Precompute those tiny combinations in double precision, preserving ties.
        # Columns 5-6 (secondary damage/accuracy) stay x1: see SECONDARY_COLUMNS.
        modifiers=[]
        for bits in range(2048):
            factors=[1.,1.,1.,0.,1.,1.,1.]
            for i,p in enumerate(POTIONS[4:15]):
                if bits & (1 << i):
                    stat=p['stat']
                    if stat==ARMOR:
                        factors[stat]+=p['amount']
                    else:
                        factors[stat]*=1+p['amount']/100.
            modifiers.append(factors)
        with jax.enable_x64(True):
            self.modifiers=jnp.array(modifiers,jnp.float64)

    def temporary_bonus(self,values,active):
        with jax.enable_x64(True):
            factors=self.modifiers[(active >> 4) & 2047]
            grown=jnp.rint(values.astype(jnp.float64)*factors).astype(jnp.float32)
        grown=set_column(grown,ARMOR,values[...,ARMOR]+factors[...,ARMOR].astype(jnp.float32))
        grown=set_column(grown,ACCURACY,jnp.minimum(grown[...,ACCURACY],100))
        grown=set_column(grown,6,jnp.minimum(grown[...,6],100))
        return grown-values

    def available(self, state, max_hp, potion=None, slot=None):
        if potion is None:
            potion = jnp.arange(self.count)[:,None]
            slot = jnp.arange(POTION_TARGETS)[None,:]
        hp, maximum = state.hp[slot], max_hp[slot]
        target = jnp.where(self.reviving[potion], can_revive(state, slot),
            (hp > 0) & (~self.healing[potion] | (hp < maximum)))
        unused = ~self.temporary[potion] | ((state.potion_active[slot] & self.bits[potion]) == 0)
        # Original-game check by the user, 2026-10-17: each permanent kind once
        # per unit per episode. The catalogue has exactly one permanent kind per stat.
        unused &= ~self.permanent[potion] | (state.potion_doses[slot, self.stats[potion]] == 0)
        return (~state.done & ~state.in_battle & (state.potions[potion] > 0)
                & (maximum > 0) & target & unused)

    def apply(self, state, action, max_hp, intrinsic):
        if not self.count:
            return state
        using = (action >= POTION_START) & (action < POTION_START+self.action_count)
        index = jnp.clip(action-POTION_START,0,self.action_count-1)
        potion, slot = index//6,index%6
        using &= self.available(state,max_hp,potion,slot)
        revive, heal = self.reviving[potion],self.healing[potion]
        permanent, temporary = using & self.permanent[potion],using & self.temporary[potion]
        stat, amount = self.stats[potion],self.amounts[potion]
        hp = jnp.where(revive,1,jnp.minimum(max_hp[slot],state.hp[slot]+amount))
        health = permanent & (stat == HP)
        with jax.enable_x64(True):
            health_hp=jnp.rint(state.hp[slot].astype(jnp.float64)*1.15).astype(jnp.int32)
        hp = jnp.minimum(jnp.where(health,health_hp,hp), MAX_HP)
        hp = jnp.where((using & (revive | heal)) | health,hp,state.hp[slot])
        # One dose per permanent kind; rebuild from new intrinsic stats after growth.
        current = intrinsic+state.potion_bonus[slot]
        affected = jnp.arange(7) == stat  # primary attack only, never SECONDARY_COLUMNS
        with jax.enable_x64(True):
            factor=jnp.where(stat==HP,jnp.float64(1.15),jnp.float64(1.1))
            grown=jnp.rint(current.astype(jnp.float64)*factor).astype(jnp.float32)
        grown = jnp.where(stat == ARMOR,current+amount,grown)
        grown = grown.at[HP].set(jnp.maximum(current[HP]+1,grown[HP]))
        grown = grown.at[ACCURACY].min(100).at[6].min(100)
        delta = jnp.where(permanent & affected,grown-current,0)
        active=state.potion_active[slot] | jnp.where(temporary,self.bits[potion],jnp.uint32(0))
        temporary_bonus=self.temporary_bonus(current+delta,active)
        return state.replace(
            hp=state.hp.at[slot].set(hp),
            potions=state.potions.at[potion].add(-using.astype(jnp.int32)),
            potion_doses=state.potion_doses.at[slot,stat].add(permanent.astype(jnp.int32)),
            potion_bonus=state.potion_bonus.at[slot].add(delta),
            potion_temporary=state.potion_temporary.at[slot].set(jnp.where(using,temporary_bonus,state.potion_temporary[slot])),
            potion_active=state.potion_active.at[slot].set(active),
            potion_wards=state.potion_wards.at[slot].set(state.potion_wards[slot] | jnp.where(temporary,self.wards[potion],jnp.uint32(0))),
            last_potion=jnp.where(using,potion,-1),
            last_event=jnp.where(using,jnp.where(revive,POTION_REVIVE,jnp.where(heal,POTION_HEAL,POTION_BUFF)),state.last_event),
            last_target=jnp.where(using,slot,state.last_target),
            last_damage=jnp.where(using,hp-state.hp[slot],state.last_damage))

    @staticmethod
    def expire(state, resting):
        return state.replace(potion_active=jnp.where(resting,jnp.uint32(0),state.potion_active),
            potion_temporary=jnp.where(resting,0.,state.potion_temporary),
            potion_wards=jnp.where(resting,jnp.uint32(0),state.potion_wards))

    def rebuild(self, intrinsic, doses):
        """Reapply each used permanent kind once after evolution/level growth."""
        used = jnp.concatenate((doses,jnp.zeros_like(doses[:,:len(SECONDARY_COLUMNS)])),axis=1) > 0
        with jax.enable_x64(True):
            factors=jnp.array([1.15,1.1,1.1,1.,1.1,1.1,1.1],jnp.float64)
            grown=jnp.rint(intrinsic.astype(jnp.float64)*factors).astype(jnp.float32)
        grown = set_column(grown,HP,jnp.maximum(intrinsic[:,HP]+1,grown[:,HP]))
        grown = set_column(grown,ARMOR,intrinsic[:,ARMOR]+10)
        grown = set_column(grown,ACCURACY,jnp.minimum(grown[:,ACCURACY],100))
        grown = set_column(grown,6,jnp.minimum(grown[:,6],100))
        return jnp.where(used,grown-intrinsic,0.)

    def observation(self, state):
        stock = state.potions/jnp.maximum(self.initial_counts,1)
        if not self.has_buffs:
            return stock
        return jnp.concatenate((stock,state.potion_doses[:6].reshape(-1)/10.,
            state.potion_active[:6].astype(jnp.float32)/(2**24-1)))

    def quotes(self, state, max_hp):
        hp = state.hp[None,:6]
        recovery = jnp.minimum(jnp.maximum(max_hp[None,:6]-hp,0),self.amounts[:,None])
        return jnp.where(self.available(state,max_hp),
            jnp.where(self.reviving[:,None],1,jnp.where(self.healing[:,None],recovery,self.amounts[:,None])),0)

    def metadata(self):
        counts = self.initial_counts.tolist()
        return [dict(p,initial_count=counts[i],action_start=POTION_START+i*6) for i,p in enumerate(self.items)]
