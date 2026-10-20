"""Faction recruitment and leadership from the Python reference; see RECRUITMENT.md."""
import jax
import jax.numpy as jnp

HIRED, DISMISSED = 41, 42


def mercenary_camps(game_map):
    """Camp definitions in site order: one 'mercenary' or a 'mercenaries' list."""
    from stoix.envs.number_grid_sites import site_entries
    return [value for kind, value in site_entries(game_map) if kind == 'mercenary']


def mercenary_offers(game_map):
    """(camp, roster entry, action suffix); a unit sold by two camps gets the camp number."""
    offers = [(i, r) for i, camp in enumerate(mercenary_camps(game_map)) for r in camp.get('roster', [])]
    units = [r['unit'] for _, r in offers]
    return [(i, r, r['unit'] if units.count(r['unit']) == 1 else f"{i}_{r['unit']}") for i, r in offers]


def recruit_action_names(game_map):
    if not game_map.get('faction_recruitment'):
        return ()
    return (tuple('hire_'+str(i) for i in range(5))
            + tuple('hire_mercenary_'+suffix for _, _, suffix in mercenary_offers(game_map))
            + tuple('dismiss_'+str(i) for i in range(6)))
# Order and row placement: reference hire.py / campaign_env_data.py.
OFFERS = {
    'empire': ('squire', 'apprentice', 'archer', 'acolyte', 'titan'),
    'mountain_clans': ('reference_g000uu0036', 'travnitsa', 'g000uu0026', 'g000uu0043', 'reference_g000uu0029'),
    'undead_hordes': ('g000uu0086', 'reference_g000uu0080', 'ghost', 'g000uu0092', 'reference_g000uu0093'),
    'legions': ('possessed', 'cultist', 'g000uu0055', 'g000uu0069', 'reference_g000uu0057'),
    'elves': ('reference_g000uu8014', 'reference_g000uu8025', 'reference_g000uu8018', 'medium', 'g000uu8029'),
}


class RecruitmentRules:
    def __init__(self, game_map, progression, construction, start, sites=None):
        self.territory = None
        self.progression = progression
        self.sites = sites
        self.faction_only_rewards = game_map.get('recruitment_reward_faction_only', False)
        if type(self.faction_only_rewards) is not bool:
            raise ValueError('recruitment_reward_faction_only must be boolean')
        camps = mercenary_camps(game_map)
        offers = mercenary_offers(game_map)
        roster = [r for _, r, _ in offers]
        if camps and (any(not camp.get('roster') for camp in camps) or sites is None):
            raise ValueError('Mercenary camp requires a roster and site rules')
        if any(type(r.get('stock')) is not int or r['stock'] < 1 for r in roster):
            raise ValueError('Mercenary stock must be a positive integer')
        if any('unlimited' in r for r in roster):
            raise ValueError('Unlimited mercenary offers are no longer supported')
        self.initial_stock = jnp.array([r['stock'] for r in roster], jnp.int32)
        self.camp_count = len(camps)
        # Site index of each offer's camp; camps follow merchants/trainers in site order.
        self.offer_sites = (sites.kind_sites['mercenary'][jnp.array([i for i, _, _ in offers], jnp.int32)]
                            if camps else jnp.zeros(0, jnp.int32))
        self.keys = OFFERS[construction.faction]+tuple(r['unit'] for r in roster)
        self.offer_count = len(self.keys)
        self.start, self.dismiss_start, self.end = start, start+self.offer_count, start+self.offer_count+6
        self.action_count = self.end-start
        self.mercenary = jnp.arange(self.offer_count) >= 5
        self.ids = jnp.array([progression.ids[k] for k in self.keys], jnp.int32)
        rows = [progression.rows[progression.ids[k]] for k in self.keys]
        costs = [int(r['game_data']['ENROLL_C'].split(':')[0][1:]) for r in rows]
        for i, offer in enumerate(roster, start=5):
            price = offer.get('price', costs[i])
            if type(price) is not int or price < 1:
                raise ValueError('Mercenary prices must be positive integer gold amounts')
            costs[i] = price
        self.costs = jnp.array(costs, jnp.int32)
        self.sizes = progression.sizes[self.ids]
        self.front = jnp.array([r['role'] == 'melee' for r in rows])
        if any(r.get('hero') or r.get('size', 1) != 1 for r in rows[5:]):
            raise ValueError('Reference mercenary camps only hire small non-hero units')
        buildings = {r['original_id'].lower(): i for i, r in enumerate(construction.rows)}
        required = [buildings.get(r['game_data']['ENROLL_B'].lower(), -1) if i < 5 else -1 for i, r in enumerate(rows)]
        self.required = jnp.array(required, jnp.int32)
        self.heroes = jnp.array([bool(r.get('hero')) for r in progression.rows])
        self.base_capacity = game_map.get('hero_leadership_capacity', 3)
        if type(self.base_capacity) is not int or not 1 <= self.base_capacity <= 5:
            raise ValueError('Hero leadership capacity must be an integer from 1 to 5')
        positions = game_map.get('recruitment_positions', [game_map.get('capital_position', game_map['agent_position'])])
        if not positions or any(len(p) != 2 or any(type(v) is not int or not 0 < v < game_map['size']-1 for v in p) for p in positions):
            raise ValueError('Recruitment positions must be interior map cells')
        self.positions = jnp.array(positions, jnp.int32)
        self.observation_size = 4+3*self.offer_count+2*len(positions)+(len(roster)+self.camp_count if roster else 0)
        self.observation_size += 6 * self.faction_only_rewards
        self.scale = game_map['size']-1
        ladder = [(level, min(5, self.base_capacity+i)) for i, level in enumerate((1, 3, 6))
                  if progression.max_hero_level is None or level <= progression.max_hero_level]
        self.metadata = dict(positions=positions, dismiss_start=self.dismiss_start,
            leadership_levels=[level for level, _ in ladder], leadership_capacity=[capacity for _, capacity in ladder],
            vacancy_penalty=.05, hire_reward=3.,
            mercenary_hire_reward=0. if self.faction_only_rewards else 3.,
            mercenaries_reduce_vacancy_penalty=not self.faction_only_rewards, offers=[dict(
                key=k, name=r['name'], unit_id=progression.ids[k], cost=int(self.costs[i]),
                size=r.get('size', 1), action=start+i, mercenary=i >= 5,
                stock_index=i-5 if i >= 5 else None, camp=int(self.offer_sites[i-5]) if i >= 5 else None,
                row='front' if r['role'] == 'melee' else 'back',
                building=construction.rows[required[i]]['name'] if required[i] >= 0 else None,
                building_action=18+required[i] if required[i] >= 0 else None)
                for i, (k, r) in enumerate(zip(self.keys, rows))])

    def composition(self, state):
        # Native roster prevents temporary copies/summons from changing leadership.
        ids = jnp.where(state.in_battle, state.native_ids, state.unit_ids)[:6]
        levels = jnp.where(state.in_battle, state.native_levels, state.unit_levels)[:6]
        heroes = self.heroes[ids]
        level = jnp.max(jnp.where(heroes, levels, 0))
        capacity = jnp.where(jnp.any(heroes), jnp.minimum(5,
            self.base_capacity+(level >= 3).astype(jnp.int32)+(level >= 6)), 0)
        used = jnp.sum(jnp.where(heroes, 0, self.progression.sizes[ids]))
        return jnp.array([capacity, used, jnp.maximum(0, capacity-used), state.hire_rewarded_capacity])

    def reward_occupancy(self, state):
        """Reward-only occupancy; actual leadership still counts every permanent unit."""
        if not self.faction_only_rewards:
            return self.composition(state)[1]
        ids = jnp.where(state.in_battle, state.native_ids, state.unit_ids)[:6]
        counted = ~self.heroes[ids] & ~state.mercenary_units[:6]
        return jnp.sum(jnp.where(counted, self.progression.sizes[ids], 0))

    def reward_vacancies(self, state):
        if not self.faction_only_rewards:
            return self.composition(state)[2]
        return jnp.maximum(0, self.composition(state)[0] - self.reward_occupancy(state))

    def targets(self, state):
        ids = state.unit_ids[:6]
        big_front = self.progression.sizes[ids[:3]] == 2
        empty = (ids == 0) & ~jnp.concatenate((jnp.zeros(3, bool), big_front))
        slots = jnp.arange(6)
        small = jnp.where(self.front[:, None], (slots == 0) | (slots == 2), slots >= 3) & empty
        big = jnp.concatenate((empty[:3] & empty[3:], jnp.zeros(3, bool)))
        valid = jnp.where((self.sizes == 2)[:, None], big, small)
        return jnp.where(jnp.any(valid, axis=1), jnp.argmax(valid, axis=1), -1)

    def dismiss_targets(self, state):
        slots = jnp.arange(6)
        front = jnp.maximum(0, slots-3)
        return jnp.where((slots >= 3) & (self.progression.sizes[state.unit_ids[front]] == 2), front, slots)

    def prices(self, state):
        if not self.initial_stock.size:
            return self.costs
        discount = jnp.max(self.sites.items.value(state, 'discount')[:2]) if self.sites.items else 0
        return jnp.where(self.mercenary, self.costs*(100-discount)//100, self.costs)

    def available(self, state):
        common = ~state.in_battle & ~state.done
        at = (self.territory.service_at(state) if self.territory is not None
              else jnp.any(jnp.all(state.position == self.positions, axis=1)))
        heroes = self.heroes[state.unit_ids[:6]]
        built = (self.required < 0) | ((state.buildings & jnp.left_shift(jnp.uint32(1), jnp.maximum(self.required, 0).astype(jnp.uint32))) != 0)
        access = jnp.full(self.offer_count, at & jnp.any(heroes), bool)
        if self.initial_stock.size:
            camp = (self.sites.common(state, 'mercenary') if self.camp_count == 1
                    else (self.sites.at_sites(state) & self.sites.ready(state))[self.offer_sites])
            access = access.at[5:].set(camp & (state.mercenary_stock > 0))
        hire = common & access & built & (self.composition(state)[2] >= self.sizes) & (state.gold >= self.prices(state)) & (self.targets(state) >= 0)
        targets = self.dismiss_targets(state)
        dismiss = common & jnp.any(heroes & (state.hp[:6] > 0)) & (state.unit_ids[targets] != 0) & ~heroes[targets]
        return jnp.concatenate((hire, dismiss))

    def apply(self, state, action, clear_slots):
        valid = (action >= self.start) & (action < self.end) & self.available(state)[jnp.clip(action-self.start, 0, self.action_count-1)]
        def change(s):
            hiring = action < self.dismiss_start
            offer = jnp.clip(action-self.start, 0, self.offer_count-1)
            target = jnp.where(hiring, self.targets(s)[offer], self.dismiss_targets(s)[jnp.clip(action-self.dismiss_start, 0, 5)])
            size = jnp.where(hiring, self.sizes[offer], self.progression.sizes[s.unit_ids[target]])
            slots = jnp.arange(12)
            affected = (slots == target) | ((size == 2) & (slots == target+3))
            s = clear_slots(s, affected)
            uid = jnp.where(hiring, self.ids[offer], 0)
            level = self.progression.base_levels[uid]
            hp = self.progression.stats(uid, level)[0].astype(jnp.int32)
            cost = jnp.where(hiring, self.prices(s)[offer], 0)
            if self.initial_stock.size:
                index = jnp.clip(offer-5, 0, self.initial_stock.size-1)
                s = s.replace(mercenary_stock=s.mercenary_stock.at[index].add(-((hiring & (offer >= 5)).astype(jnp.int32))))
            s = s.replace(unit_ids=s.unit_ids.at[target].set(uid), unit_levels=s.unit_levels.at[target].set(level),
                native_ids=s.native_ids.at[target].set(uid), native_levels=s.native_levels.at[target].set(level),
                hp=s.hp.at[target].set(hp), gold=s.gold-cost,
                mercenary_units=s.mercenary_units.at[target].set(hiring & self.mercenary[offer]),
                last_event=jnp.where(hiring, HIRED, DISMISSED).astype(jnp.int32), last_target=target,
                last_service_cost=cost)
            used = self.reward_occupancy(s)
            rewarded_hire = hiring & ~(self.faction_only_rewards & self.mercenary[offer])
            reward = jnp.where(rewarded_hire & (used > s.hire_rewarded_capacity), 3., 0.)
            return s.replace(hire_rewarded_capacity=jnp.where(rewarded_hire,
                jnp.maximum(s.hire_rewarded_capacity, used), s.hire_rewarded_capacity)), reward
        return jax.lax.cond(valid, change, lambda s: (s, jnp.float32(0)), state)

    def observation(self, state):
        parts = [self.composition(state)/5., self.ids/len(self.progression.rows),
            self.prices(state)/1000., (self.targets(state)+1)/6., self.positions.reshape(-1)/self.scale]
        if self.initial_stock.size:
            camps = (jnp.atleast_1d(self.sites.at(state, 'mercenary')) if self.camp_count == 1
                     else self.sites.at_sites(state)[self.sites.kind_sites['mercenary']])
            parts.extend((state.mercenary_stock/self.initial_stock, camps))
        if self.faction_only_rewards:
            parts.append(state.mercenary_units[:6].astype(jnp.float32))
        return jnp.concatenate(parts)

    def quotes(self, state):
        return dict(leadership=self.composition(state), targets=self.targets(state), prices=self.prices(state),
                    reward_vacancies=self.reward_vacancies(state))
