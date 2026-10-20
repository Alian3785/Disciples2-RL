"""Original 3x3 sites, merchant inventory, training and spell shops. See docs/MAP_SITES.md."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from stoix.envs.number_grid_limits import add_resource

from stoix.envs.number_grid_battle_items import MAGIC_BY_KEY
from stoix.envs.number_grid_items import (
    ARTIFACT, BANNER, BOOK, BOOTS, ITEM_BY_KEY, SCROLL, STAFF, USABLE, VALUABLE, item_counts, remove_items,
)
from stoix.envs.number_grid_spells import shop_spell_ids, spell_book

BOUGHT, TRAINED, SPELL_BOUGHT = 38, 40, 51
SELL_PERCENT = 20  # Installed Globals/GVars.DBF: SELL_RATIO.
# [row, column], relative to the upper-left of the 3x3 object.
APPROACH_OFFSETS = ((1, 3), (2, 3), (3, 1), (3, 2), (3, 3))
TRAINING_PRICES = json.loads((Path(__file__).parent/'data/training_prices.json').read_text())
# Singular key (one site) or plural key (a list); this order is the site order.
SITE_KINDS = (('merchant', 'merchants', 'Торговец'), ('trainer', 'trainers', 'Тренер'),
              ('mercenary', 'mercenaries', 'Лагерь наёмников'),
              ('spell_shop', 'spell_shops', 'Лавка заклинаний'))
# Merchants may sell any priced item but valuables, which they only buy.
SHOP_GROUPS = {ARTIFACT: 'Артефакты', BANNER: 'Знамёна', BOOK: 'Книги', BOOTS: 'Сапоги',
               SCROLL: 'Свитки', STAFF: 'Посохи'}


def site_entries(game_map):
    """(kind, definition) of every 3x3 site, grouped by kind in SITE_KINDS order."""
    entries = []
    for kind, plural, _ in SITE_KINDS:
        single, many = game_map.get(kind), game_map.get(plural)
        if single is not None and many is not None:
            raise ValueError(f'Use either {kind} or {plural}')
        if many is not None and not isinstance(many, list):
            raise ValueError(plural+' must be a list')
        entries += [(kind, value) for value in ([single] if single is not None else many or [])]
    return entries


def sites_present(game_map):
    return bool(site_entries(game_map))


def site_layout(game_map):
    """Validate footprints and derive entrance tiles; never accept arbitrary ranges."""
    from stoix.envs.number_grid_ruins import ruin_geometry
    ruin_cells = {p for ruin in ruin_geometry(game_map) for p in ruin['footprint']}
    names = {kind: name for kind, _, name in SITE_KINDS}
    sites = []
    occupied = {tuple(p) for p in game_map['opponent_positions']}
    occupied.add(tuple(game_map['agent_position']))
    occupied.add(tuple(game_map.get('capital_position', game_map['agent_position'])))
    occupied.update(ruin_cells)
    occupied.update(tuple(c['position']) for c in game_map.get('chests', []))
    occupied.update(tuple(c['position']) for c in game_map.get('cities', [])+game_map.get('mines', []))
    occupied.update(tuple(p) for p in game_map.get('obstacles', []))
    for kind, value in site_entries(game_map):
        anchor = value.get('position') if isinstance(value, dict) else None
        if (not isinstance(anchor, (list, tuple)) or len(anchor) != 2
                or any(type(n) is not int or not 0 < n < game_map['size']-4 for n in anchor)):
            raise ValueError(kind+' needs an interior 3x3 position and space for its entrance')
        r, c = anchor
        cells = [(r+i, c+j) for i in range(3) for j in range(3)]
        if occupied.intersection(cells):
            raise ValueError('Site footprints must not overlap another object or unit')
        occupied.update(cells)
        sites.append(dict(kind=kind, name=value.get('name', names[kind]), position=list(anchor),
                          size=[3, 3], footprint=cells,
                          interaction_tiles=[(r+i, c+j) for i, j in APPROACH_OFFSETS]))
    blocked = ruin_cells | {p for site in sites for p in site['footprint']} | {tuple(p) for p in game_map.get('obstacles', [])}
    for site in sites:
        site['interaction_tiles'] = [p for p in site['interaction_tiles'] if p not in blocked]
        if not site['interaction_tiles']:
            raise ValueError('Site entrance is blocked')
    return sites


def shop_group(product):
    """Merchant panel heading: potions, then the item categories."""
    if 'category' not in product:
        return 'Зелья'
    if product['category'] == USABLE:
        return 'Сферы' if MAGIC_BY_KEY[product['key']]['category'] == 'orb' else 'Талисманы'
    return SHOP_GROUPS[product['category']]


def merchant_items(merchant):
    """A merchant's item stock: known priced item keys except valuables, positive counts."""
    stock = item_counts(merchant.get('items', {}), 'Merchant items')
    if any(n < 1 for n in stock.values()):
        raise ValueError('Merchant stock must be a positive integer')
    if any(ITEM_BY_KEY[k]['category'] not in (*SHOP_GROUPS, USABLE) or ITEM_BY_KEY[k]['price'] <= 0
           for k in stock):
        raise ValueError('Merchants sell priced items other than valuables')
    return stock


def _merchant_offers(merchant):
    # Potions first, as before, then items; one action per offer.
    return [*merchant.get('potions', {}), *merchant_items(merchant)]


def _merchant_prefixes(merchants):
    # One selling merchant keeps the short historical names buy_<potion>.
    selling = [m for m in merchants if _merchant_offers(m)]
    return ['buy_' if len(selling) <= 1 else f'buy_{i}_' for i in range(len(merchants))]


def trade_action_names(game_map):
    entries = site_entries(game_map)
    merchants = [v for k, v in entries if k == 'merchant']
    shops = [v for k, v in entries if k == 'spell_shop']
    if not merchants and not shops and not any(k == 'trainer' for k, _ in entries):
        return ()
    names = []
    for prefix, merchant in zip(_merchant_prefixes(merchants), merchants):
        names += [prefix+key for key in _merchant_offers(merchant)]
    if any(k == 'trainer' for k, _ in entries):
        names += ['train_'+str(slot) for slot in range(6)]
    names += ['buy_spell_'+offer['spell'] for shop in shops for offer in shop.get('spells', [])]
    return tuple(names)


class SiteRules:
    def __init__(self, game_map, potions, items, progression, action_start):
        self.sites = site_layout(game_map)
        definitions = [value for _, value in site_entries(game_map)]
        kinds = [s['kind'] for s in self.sites]
        self.items, self.potions, self.progression = items, potions, progression
        self.names = trade_action_names(game_map)
        self.has_merchant = 'merchant' in kinds
        self.has_trainer = 'trainer' in kinds
        self.has_spell_shop = 'spell_shop' in kinds
        offers = [(i, key, count, False) for i, (kind, d) in enumerate(zip(kinds, definitions)) if kind == 'merchant'
                  for key, count in d.get('potions', {}).items()]
        if any(type(n) is not int or n < 1 for _, _, n, _ in offers):
            raise ValueError('Merchant stock must be a positive integer')
        # Reference grant "inventory": a bought item joins the hero inventory (_grant_merchant_item).
        offers += [(i, key, count, True) for i, (kind, d) in enumerate(zip(kinds, definitions)) if kind == 'merchant'
                   for key, count in merchant_items(d).items()]
        offers.sort(key=lambda offer: offer[0])  # per merchant: potions, then items
        self.buy_keys = tuple(key for _, key, _, _ in offers)
        if self.has_merchant and not self.buy_keys:
            raise ValueError('A merchant must have stock')
        spell_names = [name for name in self.names if name.startswith('buy_spell_')]
        if len(set(spell_names)) != len(spell_names):
            raise ValueError('Each spell may be offered by only one spell shop')
        if len(set(self.names)) != len(self.names):
            raise ValueError('Merchant offers need distinct action names')
        self.item_offers = any(item for *_, item in offers)
        if self.item_offers and not items:
            raise ValueError('Merchant items require the equipment inventory')
        catalogue = [items.items[items.keys.index(k)] if item else potions.items[potions.keys.index(k)]
                     for _, k, _, item in offers]
        self.buy_sites = jnp.array([i for i, *_ in offers], jnp.int32)
        self.buy_items = jnp.array([item for *_, item in offers], bool)
        self.buy_ids = jnp.array([(items.keys if item else potions.keys).index(k) for _, k, _, item in offers], jnp.int32)
        self.initial_stock = jnp.array([n for _, _, n, _ in offers], jnp.int32)
        prices = [definitions[i].get('prices', {}).get(key, p['price'])
                  for (i, key, _, _), p in zip(offers, catalogue)]
        if any(type(p) is not int or p <= 0 for p in prices):
            raise ValueError('Merchant prices must be positive integer gold amounts')
        self.buy_base = jnp.array(prices, jnp.int32)
        self.buy_groups = [shop_group(p) for p in catalogue]
        spell_offers = [(i, o) for i, (kind, d) in enumerate(zip(kinds, definitions)) if kind == 'spell_shop'
                        for o in d.get('spells', [])]
        if self.has_spell_shop and not spell_offers:
            raise ValueError('A spell shop must sell spells')
        if any(type(o.get('price')) is not int or o['price'] < 0 or type(o.get('stock')) is not int or o['stock'] < 1
               for _, o in spell_offers):
            raise ValueError('Spell offers need an integer price and positive stock')
        shop_spell_ids(game_map)  # validates the identifiers
        own, extras = spell_book(game_map)
        book = [row['id'] for row in own+extras]
        self.spell_sites = jnp.array([i for i, _ in spell_offers], jnp.int32)
        self.spell_bits = jnp.array([1 << book.index(o['spell']) for _, o in spell_offers], jnp.uint32)
        self.spell_initial_stock = jnp.array([o['stock'] for _, o in spell_offers], jnp.int32)
        self.spell_prices = jnp.array([o['price'] for _, o in spell_offers], jnp.int32)
        self.spell_offers = [dict(spell=o['spell'], price=o['price'], stock=o['stock'], site=i,
                                  name=next(r['name'] for r in own+extras if r['id'] == o['spell']))
                             for i, o in spell_offers]
        self.buy_start = action_start
        self.train_start = self.buy_start+len(self.buy_keys)
        self.spell_start = self.train_start+(6 if self.has_trainer else 0)
        self.end = self.spell_start+len(spell_offers)
        self.action_count = self.end-action_start
        self.anchors = jnp.array([s['position'] for s in self.sites], jnp.int32)
        self.footprints = jnp.array([p for s in self.sites for p in s['footprint']], jnp.int32)
        # Each site has at most five approach tiles; (-1,-1) pads blocked ones.
        self.approach_tiles = jnp.array([s['interaction_tiles']+[(-1, -1)]*(5-len(s['interaction_tiles']))
                                         for s in self.sites], jnp.int32).reshape(-1, 5, 2)
        self.kind_sites = {kind: jnp.array([i for i, k in enumerate(kinds) if k == kind], jnp.int32)
                           for kind in set(kinds)}
        self.approaches = {kind: jnp.array(next(s for s in self.sites if s['kind'] == kind)['interaction_tiles'], jnp.int32)
                           for kind in set(kinds)}
        self.hero_ids = jnp.array([bool(r.get('hero')) for r in progression.rows])
        prices = [[0, 0, 0, 0, 0]]
        for row in progression.rows[1:]:
            prices.append(TRAINING_PRICES[row['game_id'].lower()])
        self.training = jnp.array(prices, jnp.int32)
        self.observation_size = (self.anchors.size+2*len(self.buy_keys)+(6 if self.has_trainer else 0)
                                 + 2*len(spell_offers))
        self.map_scale = game_map['size']-1

    def blocked(self, position):
        return jnp.any(jnp.all(position[..., None, :] == self.footprints, axis=-1), axis=-1)

    def at_sites(self, state):
        """One flag per site: the party stands on one of its approach tiles."""
        return jnp.any(jnp.all(state.position == self.approach_tiles, axis=-1), axis=-1)

    def at(self, state, kind):
        if kind not in self.kind_sites:
            return jnp.bool_(False)
        if self.kind_sites[kind].size == 1:
            return jnp.any(jnp.all(state.position == self.approaches[kind], axis=-1))
        return jnp.any(self.at_sites(state)[self.kind_sites[kind]])

    def leader_alive(self, state):
        return jnp.any(self.hero_ids[state.unit_ids[:6]] & (state.hp[:6] > 0))

    def ready(self, state):
        return ~state.in_battle & ~state.done & self.leader_alive(state)

    def common(self, state, kind):
        return self.at(state, kind) & self.ready(state)

    def buy_prices(self, state):
        # Reference: two copies of Lute of Charming do not stack.
        discount = jnp.max(self.items.value(state, 'discount')[:2]) if self.items else 0
        return self.buy_base*(100-discount)//100

    def training_quotes(self, state):
        ids, levels = state.unit_ids[:6], state.unit_levels[:6]
        native, threshold, base, early, late = self.training[ids].T
        first = jnp.maximum(0, jnp.minimum(levels, threshold)-native)
        last = jnp.maximum(0, levels-jnp.maximum(native, threshold))
        price = base+first*early+last*late
        cap = jnp.maximum(0, self.progression.required_xp(ids, levels)-1)
        missing = jnp.maximum(0, cap-state.unit_xp[:6])
        eligible = (ids != 0) & (state.hp[:6] > 0) & (price > 0)
        eligible &= self.progression.level_up_allowed(ids, levels)
        amount = jnp.where(eligible, jnp.minimum(missing, state.gold//jnp.maximum(price, 1)), 0)
        # Price/XP, cap, missing XP, affordable XP, total cost (even when away).
        return jnp.stack((price, cap, missing, amount, amount*price), axis=-1)

    def merchant_access(self, state):
        if self.kind_sites['merchant'].size == 1:
            return self.common(state, 'merchant')
        return (self.at_sites(state) & self.ready(state))[self.buy_sites]

    def spell_available(self, state):
        # Reference _step_buy_spell_shop_spell: one copy, never a spell already known.
        return ((self.at_sites(state) & self.ready(state))[self.spell_sites] & (state.spell_stock > 0)
                & (state.gold >= self.spell_prices) & ((state.learned_spells & self.spell_bits) == 0))

    def available(self, state):
        parts = []
        if self.has_merchant:
            parts.append(self.merchant_access(state) & (state.merchant_stock > 0) & (state.gold >= self.buy_prices(state)))
        if self.has_trainer:
            parts.append(self.common(state, 'trainer') & (self.training_quotes(state)[:, 3] > 0))
        if self.has_spell_shop:
            parts.append(self.spell_available(state))
        return jnp.concatenate(parts) if parts else jnp.zeros(0, bool)

    def autosell(self, state):
        if not self.has_merchant or not self.items or not self.items.capacity:
            return state
        ids = state.item_inventory
        remove = ((ids >= 0) & (self.items.tables['category'][ids+1] == VALUABLE)
                  & (self.items.tables['price'][ids+1] > 0)
                  & self.at(state, 'merchant') & ~state.in_battle & ~state.done)
        gold = jnp.sum(jnp.where(remove, self.items.tables['price'][ids+1]*SELL_PERCENT//100, 0))
        credited = add_resource(state.gold, gold)
        return remove_items(state, remove).replace(gold=credited,
                             last_sale_gold=state.last_sale_gold+credited-state.gold,
                             last_sold_count=state.last_sold_count+jnp.sum(remove))

    def apply(self, state, action):
        if not self.action_count:
            return state
        index = jnp.clip(action-self.buy_start, 0, self.action_count-1)
        legal = (action >= self.buy_start) & (action < self.end) & self.available(state)[index]
        if self.has_merchant:
            def buy(s):
                i = jnp.clip(action-self.buy_start, 0, len(self.buy_keys)-1)
                cost = self.buy_prices(s)[i]
                item = self.buy_items[i]
                if self.item_offers:
                    # The inventory stays compact; its capacity counts every merchant copy.
                    free = jnp.sum(s.item_inventory >= 0)
                    s = s.replace(item_inventory=s.item_inventory.at[jnp.where(item, free, s.item_inventory.size)].set(
                        self.buy_ids[i], mode='drop'))
                potion = jnp.where(item, s.potions.size, self.buy_ids[i])
                return s.replace(gold=s.gold-cost, potions=s.potions.at[potion].add(1, mode='drop'),
                    merchant_stock=s.merchant_stock.at[i].add(-1), last_event=jnp.int32(BOUGHT),
                    last_service_cost=cost, last_trade_item=i)
            state = jax.lax.cond(legal & (action < self.train_start), buy, lambda s: s, state)

        if self.has_trainer:
            def train(s):
                slot = jnp.clip(action-self.train_start, 0, 5)
                quote = self.training_quotes(s)[slot]
                return s.replace(gold=s.gold-quote[4], unit_xp=s.unit_xp.at[slot].add(quote[3]),
                    last_xp=s.last_xp.at[slot].set(quote[3]), last_service_cost=quote[4],
                    last_event=jnp.int32(TRAINED), last_target=slot)
            training = legal & (action >= self.train_start)
            if self.has_spell_shop:
                training &= action < self.spell_start
            state = jax.lax.cond(training, train, lambda s: s, state)
        if self.has_spell_shop:
            # A purchase sets the spell's learned bit; casting then needs no research.
            i = jnp.clip(action-self.spell_start, 0, self.spell_prices.size-1)
            bought = legal & (action >= self.spell_start)
            cost = jnp.where(bought, self.spell_prices[i], 0)
            state = state.replace(gold=state.gold-cost,
                spell_stock=state.spell_stock.at[i].add(-bought.astype(jnp.int32)),
                learned_spells=state.learned_spells | jnp.where(bought, self.spell_bits[i], jnp.uint32(0)),
                last_service_cost=jnp.where(bought, cost, state.last_service_cost),
                last_trade_item=jnp.where(bought, i, state.last_trade_item),
                last_event=jnp.where(bought, SPELL_BOUGHT, state.last_event))
        return state

    def observation(self, state):
        parts = [self.anchors.reshape(-1)/self.map_scale,
                 state.merchant_stock/jnp.maximum(self.initial_stock, 1), self.buy_prices(state)/1000.]
        if self.has_trainer:
            parts.append(self.training_quotes(state)[:, 0]/10.)
        if self.has_spell_shop:
            parts += [state.spell_stock/self.spell_initial_stock, self.spell_prices/1000.]
        return jnp.concatenate(parts)

    def quotes(self, state):
        return dict(buy=self.buy_prices(state), train=self.training_quotes(state),
                    spells=self.spell_available(state) if self.has_spell_shop else jnp.zeros(0, bool))

    @property
    def metadata(self):
        bought = [dict((self.items if item else self.potions).items[int(self.buy_ids[i])], action=self.buy_start+i,
                       site=int(self.buy_sites[i]), group=self.buy_groups[i])
                  for i, item in enumerate(self.buy_items.tolist())]
        spells = [dict(offer, action=self.spell_start+i) for i, offer in enumerate(self.spell_offers)]
        return dict(sites=self.sites, buy=bought,
                    train_start=self.train_start if self.has_trainer else None,
                    spell_start=self.spell_start if self.has_spell_shop else None, spells=spells,
                    auto_sale='valuables', sell_percent=SELL_PERCENT)
