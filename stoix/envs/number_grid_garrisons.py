"""City and capital garrisons: hired reserves that defend their settlement under agent control.

Port of the Python reference Big_map/campaign_env_garrison.py (CampaignGarrisonMixin) to every
JAX campaign map, extended to the capital by user request (2026-11-08):

- An owned city keeps a garrison of up to its level in cells (1..5); a large unit takes a whole
  column, two cells. The capital keeps its guardian in the rear middle cell (user rule) and
  up to five more cells. The guardian is the map's capital_defender, else the faction's own
  guardian at level 1: Мизраэль, Видар, Ашкаэль, Ашган or Иллюмиэлль.
- Five shared actions hire the faction's five recruits into the selected garrison remotely:
  gold, the capital building that party recruitment needs, room in the garrison and a free cell
  in the unit's row (warriors in front, the others behind; a large unit needs a free column).
  The first such cell is taken; the reference draws one at random. No reward.
- Living garrison units regenerate on every REST before armies move: the unit's REGEN, the
  lord's bonus and the city level bonus (35 in the capital), as Discipl2.exe getFortRegen;
  terrain and the hero's items do not count. Owned settlements keep their dead in their cells
  for paid revival (capital: 2026-11-09, cities: 2026-11-10); losing a city clears its garrison.
- One selection action per city and capital chooses the garrison for remote hiring, recovery and
  formation swaps. Selection starts at the capital, persists independently of party movement,
  and falls back to the capital if the selected city is lost. Only owned sites can be selected.
- The same six revival and six healing actions serve that selected garrison. Both need the
  capital's Temple and gold at the unit's
  REVIVE_C / HEAL_C rates. Revival gives 1 HP; healing buys whole HP and keeps change.
  Neither gives a reward. A dead guardian may be revived too.
- With the party at an owned city or capital entrance, the same twelve actions move units
  between the party and that garrison: alive or dead, with level, experience, HP, permanent
  potions and the mercenary mark. Leaders and the guardian stay. A unit takes its own cell when
  it is free, else the first free cell of its row, then of the other row; a large unit needs a
  free column. The city's level limits stored units; the party's leadership limits units taken
  in. Temporary potion effects end. Transfer action names stay the same.
- An enemy army that ends its move on an owned city or capital entrance while the hero's party
  is not there attacks the garrison in a battle the agent plays: the party is stashed, the
  garrison stands in its cells with its own permanent potions, the fort's armour applies and
  nobody may retreat. The hero's items, spells and buffs stay with the party. Survivors keep
  their HP, experience and promotions. With the party inside, the party defends first, in an
  ordinary encounter; a garrison and a visiting stack are separate unit groups.
- A garrison that loses, or a city entered with no living defender, gives the city to its
  attacker until the hero takes it back. Losing the capital, or a city marked defended (the
  city_defence_train objective), loses the episode.
"""
import jax
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid_combat import HP
from stoix.envs.number_grid_limits import MAX_HP
from stoix.envs.number_grid_territory import CAPITAL_ARMOR, CAPITAL_REGENERATION

GUARDIANS = dict(empire='g000uu3001', mountain_clans='g000uu3002', legions='g000uu3003',
                 undead_hordes='g000uu3004', elves='g000uu8040')
GUARDIAN_SLOT = 4  # rear middle cell, user rule 2026-11-08
OFFERS = 5
GARRISON_HIRED, CITY_LOST, GARRISON_REVIVED, UNIT_STORED, UNIT_TAKEN = 58, 59, 60, 61, 62
GARRISON_SWAPPED, GARRISON_HEALED, GARRISON_SELECTED = 67, 68, 69
DOSES = 5  # permanent potion kinds, one column of potion_doses each
# Every garrison has its own daily swap limit and used pairs. Keep the original ten pairs
# first, then add the five pairs involving cell 4 for cities; the capital guardian stays.
SWAP_LIMIT, SWAP_PENALTY = 3, .004
SWAP_CELLS = tuple(c for c in range(6) if c != GUARDIAN_SLOT)
SWAP_PAIRS = (tuple((a, b) for i, a in enumerate(SWAP_CELLS) for b in SWAP_CELLS[i+1:])
              + tuple((a, b) for a in range(6) for b in range(a+1, 6) if GUARDIAN_SLOT in (a, b)))


def garrisons_enabled(game_map):
    return bool(game_map.get('battle_mode') and game_map.get('unit_progression') and game_map.get('territory'))


def guardian_spec(game_map, faction):
    """The capital's guardian: the map's capital_defender, else the faction's own at level 1."""
    spec = dict(game_map.get('capital_defender') or {})
    if set(spec) - {'unit', 'level', 'combat_stats'}:
        raise ValueError('capital_defender takes unit, level and combat_stats only')
    return dict(unit=spec.get('unit', GUARDIANS[faction]), level=spec.get('level', 1),
                combat_stats=spec.get('combat_stats', {}))


def garrison_action_names(game_map):
    """Hires, shared recovery and transfers, shared swaps, then one selection per settlement."""
    if not garrisons_enabled(game_map) or not game_map.get('faction_recruitment'):
        return ()
    sites = [f'city_{i}' for i in range(len(game_map.get('cities', [])))] + ['capital']
    names = [f'garrison_hire_{k}' for k in range(OFFERS)]
    if game_map.get('capital_services'):
        names += [f'garrison_capital_revive_{c}' for c in range(6)]
        names += [f'garrison_capital_heal_{c}' for c in range(6)]
    names += [f'party_to_capital_{s}' for s in range(6)] + [f'capital_to_party_{c}' for c in range(6)]
    names += [f'capital_swap_{a}_{b}' for a, b in SWAP_PAIRS]
    names += [f'select_{site}' for site in sites]
    return tuple(names)


def end_battle(state, mask):
    """Close a defeated party/city battle, preserving the campaign's terminal flags."""
    def none(values, empty):
        return jnp.where(mask, jnp.full_like(values, empty), values)
    return state.replace(in_battle=state.in_battle & ~mask,
        priority=none(state.priority, 0), turn_phase=none(state.turn_phase, 0),
        defended=none(state.defended, False), retreating=none(state.retreating, False),
        escaped=none(state.escaped, False), actor=none(state.actor, 0), round=none(state.round, 0),
        post_victory=none(state.post_victory, 0), pending_healers=none(state.pending_healers, False),
        second_strike=none(state.second_strike, False), wards_used=none(state.wards_used, 0),
        battle_fort_armor=none(state.battle_fort_armor, 0), battle_hero_fort_armor=none(state.battle_hero_fort_armor, 0),
        battle_fort_defender=none(state.battle_fort_defender, 0))


def place(free, big, same):
    """[K] cells for K units over [6] free cells, -1 when none fits.

    A unit takes its own cell `same` when it fits, else the first fitting cell of its own
    row, then of the other row. A large unit stands in a front cell over a free column.
    """
    cells = jnp.arange(6)
    column = jnp.concatenate((free[:3] & free[3:], jnp.zeros(3, bool)))
    fits = jnp.where(big[:, None], column[None], free[None])
    rank = jnp.where(cells == same[:, None], 0, jnp.where((cells < 3) == (same[:, None] < 3), 1, 2))*6+cells
    rank = jnp.where(fits, rank, 99)
    return jnp.where(jnp.any(fits, axis=-1), jnp.argmin(rank, axis=-1), -1).astype(jnp.int32)


class GarrisonRules:
    def __init__(self, game_map, progression, territory, recruitment, construction, start, bot_enemy=None,
                 capital=None, potion_rules=None, intrinsic=None):
        self.enabled = garrisons_enabled(game_map) and progression is not None and territory is not None
        self.count = self.observation_size = self.action_count = 0
        self.start = self.end = start
        self.potion_rules, self.intrinsic = potion_rules, intrinsic
        if not self.enabled:
            return
        self.progression, self.territory, self.recruitment = progression, territory, recruitment
        self.cities = territory.count
        self.count = self.cities+1
        self.capital_index = self.cities
        self.positions = jnp.concatenate((territory.positions, jnp.array([territory.capital], jnp.int32)))
        unit = progression.capital_defender_id
        if not bool(progression.capital_guards[unit]) or int(progression.sizes[unit]) != 1:
            raise ValueError('capital_defender must be a small capital guardian')
        spec = guardian_spec(game_map, construction.faction)
        if type(spec['level']) is not int or spec['level'] < 1:
            raise ValueError('capital_defender level must be a positive integer')
        self.guardian = unit
        cells = (jnp.arange(self.count)[:, None] == self.capital_index) & (jnp.arange(6) == GUARDIAN_SLOT)
        self.initial_ids = jnp.where(cells, unit, 0).astype(jnp.int32)
        self.initial_levels = jnp.where(cells, spec['level'], 0).astype(jnp.int32)
        self.initial_hp = self.max_hp(self.initial_ids, self.initial_levels)
        self.guardian_max_hp = int(self.initial_hp[self.capital_index, GUARDIAN_SLOT])
        defended = [i for i, c in enumerate(game_map.get('cities', [])) if c.get('defended', False)]
        if len(defended) > 1 or any(type(c.get('defended', False)) is not bool for c in game_map.get('cities', [])):
            raise ValueError('At most one city may be defended')
        self.defended = defended[0] if defended else -1
        attackers = jnp.ones(len(game_map['opponent_positions']), bool)
        if bot_enemy is not None:
            # Mirror match's bot stops on the player's capital tile while walking; as in the
            # reference, where the capital is outside the garrison system, it does not attack it.
            attackers = attackers.at[bot_enemy].set(False)
        self.attackers = attackers
        self.capital = capital
        names = [c['name'] for c in game_map.get('cities', [])]+['Столица']
        self.metadata = dict(start=start, hire_count=0, sites=names, guardian_slot=GUARDIAN_SLOT, offers=[],
                             guardian=progression.rows[unit]['name'], revive_start=None, store_start=None,
                             take_start=None, select_start=None, dead_stay='owned')
        if recruitment is not None:
            self.offer_ids = recruitment.ids[:OFFERS]
            self.offer_costs = recruitment.costs[:OFFERS]
            self.offer_sizes = recruitment.sizes[:OFFERS]
            self.offer_front = recruitment.front[:OFFERS]
            self.offer_required = recruitment.required[:OFFERS]
            self.hire_end = start+OFFERS
            self.revive_count = 6 if capital is not None else 0
            self.heal_start = self.hire_end+self.revive_count
            self.store_start = self.heal_start+self.revive_count
            self.take_start = self.store_start+6
            self.swap_start = self.take_start+6
            self.select_start = self.swap_start+len(SWAP_PAIRS)
            self.end = self.select_start+self.count
            self.action_count = self.end-start
            pairs = np.array(SWAP_PAIRS)
            self.swap_a, self.swap_b = jnp.array(pairs[:, 0]), jnp.array(pairs[:, 1])
            small, big = np.tile(np.arange(6), (len(pairs), 1)), np.tile(np.arange(6), (len(pairs), 1))
            for i, (a, b) in enumerate(SWAP_PAIRS):
                small[i, [a, b]] = b, a
                # A big unit moves its whole column; the capital's guardian is masked below.
                ca, cb = a % 3, b % 3
                big[i, [ca, ca+3, cb, cb+3]] = cb, cb+3, ca, ca+3
            self.swap_small, self.swap_big = jnp.array(small), jnp.array(big)
            self.swap_bits = jnp.left_shift(jnp.uint32(1), jnp.arange(len(pairs), dtype=jnp.uint32))
            self.metadata.update(hire_count=OFFERS, offers=[dict(
                key=recruitment.keys[i], name=progression.rows[int(recruitment.ids[i])]['name'],
                cost=int(recruitment.costs[i]), size=int(recruitment.sizes[i]),
                row='front' if bool(recruitment.front[i]) else 'back') for i in range(OFFERS)],
                revive_start=self.hire_end if capital is not None else None,
                heal_start=self.heal_start if capital is not None else None,
                store_start=self.store_start, take_start=self.take_start, swap_start=self.swap_start,
                select_start=self.select_start,
                swap_pairs=[list(p) for p in SWAP_PAIRS], swap_limit=SWAP_LIMIT, swap_penalty=SWAP_PENALTY)
        self.observation_size = self.count*(5+6*4)+(self.count*(2+len(SWAP_PAIRS)) if self.action_count else 0)

    # Tables ------------------------------------------------------------
    def potion_bonus(self, ids, levels, doses):
        """Permanent potion deltas [..., 7] of units with these doses, as the party's potion_bonus."""
        shape = ids.shape
        intrinsic = self.intrinsic(ids.reshape(-1), levels.reshape(-1))
        bonus = self.potion_rules.rebuild(intrinsic, doses.reshape(-1, DOSES))
        return jnp.where((ids != 0)[..., None], bonus.reshape(shape+(bonus.shape[-1],)), 0.)

    def max_hp(self, ids, levels, doses=None):
        values = self.progression.stats(ids.reshape(-1), levels.reshape(-1))[:, HP].reshape(ids.shape)
        if doses is not None and self.potion_rules is not None:
            values = values+self.potion_bonus(ids, levels, doses)[..., HP]
        return jnp.where(ids != 0, jnp.minimum(values, MAX_HP).astype(jnp.int32), 0)

    def garrison_max_hp(self, state):
        return self.max_hp(state.garrison_ids, state.garrison_levels, state.garrison_doses)

    def owned(self, state):
        return jnp.concatenate((state.city_owned, jnp.ones(1, bool)))

    def site(self, positions):
        """Garrison index of each entrance position, -1 elsewhere."""
        match = jnp.all(positions[..., None, :] == self.positions, axis=-1)
        return jnp.where(jnp.any(match, axis=-1), jnp.argmax(match, axis=-1), -1).astype(jnp.int32)

    def capacity(self, state):
        return jnp.concatenate((jnp.clip(state.city_levels, 1, 5), jnp.array([6], jnp.int32)))

    def occupied(self, ids):
        """[..., 6] used cells: a unit's own cell and the rear cell behind a large front unit."""
        front_big = self.progression.sizes[ids[..., :3]] == 2
        return (ids != 0) | jnp.concatenate((jnp.zeros_like(front_big), front_big), axis=-1)

    def living(self, state):
        return jnp.any((state.garrison_ids != 0) & (state.garrison_hp > 0), axis=1)

    @staticmethod
    def has_party(state):
        return jnp.any(state.unit_ids[:6] != 0)

    def pending(self, state):
        """A garrison battle waits: the next decision must be CONTINUE."""
        return ~state.in_battle & jnp.any(state.garrison_pending & state.alive)

    def ready(self, state):
        return ~state.in_battle & ~state.done & ~jnp.any(state.garrison_pending & state.alive)

    def transfer_site(self, state):
        """The owned settlement at the party's entrance, or -1 away from an owned settlement."""
        site = self.site(state.position)
        owned = self.owned(state)[jnp.maximum(site, 0)]
        return jnp.where((site >= 0) & owned & self.has_party(state), site, -1)

    def selected_site(self, state):
        """The explicitly selected owned garrison, independently of the party's position."""
        site = jnp.clip(state.selected_garrison, 0, self.count-1)
        return jnp.where(self.owned(state)[site], site, self.capital_index)

    # Hiring --------------------------------------------------------------
    def targets(self, state):
        """[G, 5] cell each offer would take in each garrison, -1 when it does not fit."""
        free = ~self.occupied(state.garrison_ids)
        cells = jnp.arange(6)
        rows = jnp.where(self.offer_front[:, None], cells < 3, cells >= 3)
        small = free[:, None, :] & rows[None]
        column = jnp.concatenate((free[:, :3] & free[:, 3:], jnp.zeros((self.count, 3), bool)), axis=1)
        fits = jnp.where((self.offer_sizes == 2)[None, :, None], column[:, None, :], small)
        room = jnp.sum(~free, axis=1)[:, None]+self.offer_sizes[None] <= self.capacity(state)[:, None]
        fits &= room[..., None]
        return jnp.where(jnp.any(fits, axis=-1), jnp.argmax(fits, axis=-1), -1).astype(jnp.int32)

    # Transfers -------------------------------------------------------------
    def store_targets(self, state):
        """[6] local garrison cell each party unit would take, respecting settlement capacity."""
        site = self.transfer_site(state)
        safe = jnp.maximum(site, 0)
        ids = state.unit_ids[:6]
        free = ~self.occupied(state.garrison_ids[safe])
        sizes = self.progression.sizes[ids]
        room = jnp.sum(~free)+sizes <= self.capacity(state)[safe]
        cells = place(free, sizes == 2, jnp.arange(6))
        return jnp.where((site >= 0) & room, cells, -1)

    def take_targets(self, state):
        """[6] party slot each local garrison unit would take, -1 when it does not fit."""
        site = self.transfer_site(state)
        ids = state.garrison_ids[jnp.maximum(site, 0)]
        free = ~self.occupied(state.unit_ids[:6])
        cells = place(free, self.progression.sizes[ids] == 2, jnp.arange(6))
        return jnp.where(site >= 0, cells, -1)

    def revive_price(self, state):
        return self.capital.prices[state.garrison_ids[self.selected_site(state)], 1]

    def heal_price(self, state):
        """Gold per HP, as the party's Temple healing (HEAL_C)."""
        return self.capital.prices[state.garrison_ids[self.selected_site(state)], 0]

    def available(self, state):
        if not self.action_count:
            return jnp.zeros(0, bool)
        ready = self.ready(state)
        bits = jnp.left_shift(jnp.uint32(1), jnp.maximum(self.offer_required, 0).astype(jnp.uint32))
        built = (self.offer_required < 0) | ((state.buildings & bits) != 0)
        selected = self.selected_site(state)
        hire = (ready & self.owned(state)[selected] & built & (state.gold >= self.offer_costs)
                & (self.targets(state)[selected] >= 0))
        parts = [hire]
        if self.revive_count:
            recovery = self.selected_site(state)
            gids = state.garrison_ids[recovery]
            dead = (gids != 0) & (state.garrison_hp[recovery] <= 0)
            temple = ready & self.capital.temple_built(state)
            parts.append(temple & dead & (state.gold >= self.revive_price(state)))
            # Healing: a living wounded unit and gold for at least one HP; masked otherwise.
            hp, maximum = state.garrison_hp[recovery], self.garrison_max_hp(state)[recovery]
            parts.append(temple & (gids != 0) & (hp > 0) & (hp < maximum) & (state.gold >= self.heal_price(state)))
        site = self.transfer_site(state)
        here = ready & (site >= 0)
        gids = state.garrison_ids[jnp.maximum(site, 0)]
        ids = state.unit_ids[:6]
        heroes = self.progression.heroes
        parts.append(here & (ids != 0) & ~heroes[ids] & (state.summon_owner[:6] < 0) & (self.store_targets(state) >= 0))
        leader = jnp.any(heroes[ids])
        room = self.recruitment.composition(state)[2] >= self.progression.sizes[gids]
        parts.append(here & leader & (gids != 0) & ~self.progression.capital_guards[gids] & room
                     & (self.take_targets(state) >= 0))
        possible, _ = self.swap_pairs(state)
        selected = self.selected_site(state)
        unused = (state.garrison_swaps_used[selected] & self.swap_bits) == 0
        parts.append(ready & possible & unused & (state.garrison_swaps_today[selected] < SWAP_LIMIT))
        parts.append(ready & self.owned(state) & (jnp.arange(self.count) != selected))
        return jnp.concatenate(parts)

    def swap_pairs(self, state):
        """Legal pairs of the selected garrison, protecting the capital's guardian and its column."""
        site = self.selected_site(state)
        ids = state.garrison_ids[site]
        cells = jnp.arange(6)
        column = cells % 3
        real = ids != 0  # a dead unit keeps its cell and can be moved
        # An empty rear cell behind a big front unit is that unit's endpoint.
        owner = jnp.where(real, cells, jnp.where(self.progression.sizes[ids[column]] == 2, column, -1))
        big = (owner >= 0) & (self.progression.sizes[ids[jnp.maximum(owner, 0)]] == 2)
        columns = big[self.swap_a] | big[self.swap_b]
        ca, cb, fixed = column[self.swap_a], column[self.swap_b], GUARDIAN_SLOT % 3
        possible = jnp.where(columns, ca != cb, real[self.swap_a] | real[self.swap_b])
        guardian_safe = jnp.where(columns, (ca != fixed) & (cb != fixed),
                                  (self.swap_a != GUARDIAN_SLOT) & (self.swap_b != GUARDIAN_SLOT))
        possible &= (site != self.capital_index) | guardian_safe
        return possible, columns

    def apply(self, env, state, action, resting):
        """One garrison action of a validated world step, including explicit settlement selection.

        Returns the state and the swap penalty; REST clears the day's swap counters.
        """
        if not self.action_count:
            return state, jnp.float32(0)
        index = jnp.clip(action-self.start, 0, self.action_count-1)
        valid = (action >= self.start) & (action < self.end) & self.available(state)[index]
        state = self._hire(state, action, valid & (action < self.hire_end))
        if self.revive_count:
            state = self._revive(state, action, valid & (action >= self.hire_end) & (action < self.heal_start))
            state = self._heal(state, action, valid & (action >= self.heal_start) & (action < self.store_start))
        state = self._store(env, state, action, valid & (action >= self.store_start) & (action < self.take_start))
        state = self._take(env, state, action, valid & (action >= self.take_start) & (action < self.swap_start))
        state, penalty = self._swap(state, action,
            valid & (action >= self.swap_start) & (action < self.select_start), resting)
        selecting = valid & (action >= self.select_start)
        return state.replace(
            selected_garrison=jnp.where(selecting, jnp.clip(action-self.select_start, 0, self.count-1),
                                         state.selected_garrison),
            last_event=jnp.where(selecting, GARRISON_SELECTED, state.last_event)), penalty

    def _swap(self, state, action, valid, resting):
        """Exchange two selected garrison cells; counters belong to each settlement."""
        site = self.selected_site(state)
        pair = jnp.clip(action-self.swap_start, 0, len(SWAP_PAIRS)-1)
        _, columns = self.swap_pairs(state)
        order = jnp.where(valid, jnp.where(columns[pair], self.swap_big[pair], self.swap_small[pair]), jnp.arange(6))

        def moved(values):
            return values.at[site].set(values[site][order])
        counts = state.garrison_swaps_today.at[site].add(valid.astype(jnp.int32))
        used = state.garrison_swaps_used.at[site].set(state.garrison_swaps_used[site]
            | jnp.where(valid, self.swap_bits[pair], jnp.uint32(0)))
        state = state.replace(garrison_ids=moved(state.garrison_ids), garrison_levels=moved(state.garrison_levels),
            garrison_xp=moved(state.garrison_xp), garrison_hp=moved(state.garrison_hp),
            garrison_doses=moved(state.garrison_doses), garrison_mercenary=moved(state.garrison_mercenary),
            garrison_swaps_today=jnp.where(resting, 0, counts),
            garrison_swaps_used=jnp.where(resting, jnp.uint32(0), used),
            last_target=jnp.where(valid, pair, state.last_target),
            last_event=jnp.where(valid, GARRISON_SWAPPED, state.last_event))
        return state, jnp.where(valid, -SWAP_PENALTY, 0.).astype(jnp.float32)

    def _hire(self, state, action, valid):
        offer = jnp.clip(action-self.start, 0, OFFERS-1)
        site = self.selected_site(state)
        cell = self.targets(state)[site, offer]
        uid = self.offer_ids[offer]
        level = self.progression.base_levels[uid]
        hp = self.progression.stats(uid, level)[HP].astype(jnp.int32)
        put = valid & (jnp.arange(self.count)[:, None] == site) & (jnp.arange(6) == cell)
        cost = jnp.where(valid, self.offer_costs[offer], 0)
        return state.replace(garrison_ids=jnp.where(put, uid, state.garrison_ids),
            garrison_levels=jnp.where(put, level, state.garrison_levels),
            garrison_xp=jnp.where(put, 0, state.garrison_xp), garrison_hp=jnp.where(put, hp, state.garrison_hp),
            garrison_doses=jnp.where(put[..., None], 0, state.garrison_doses),
            garrison_mercenary=state.garrison_mercenary & ~put,
            gold=state.gold-cost, last_service_cost=jnp.where(valid, cost, state.last_service_cost),
            last_target=jnp.where(valid, cell, state.last_target),
            last_event=jnp.where(valid, GARRISON_HIRED, state.last_event))

    def _revive(self, state, action, valid):
        """The Temple raises a dead defender in the selected settlement at 1 HP."""
        site = self.selected_site(state)
        cell = jnp.clip(action-self.hire_end, 0, 5)
        put = valid & (jnp.arange(self.count)[:, None] == site) & (jnp.arange(6) == cell)
        cost = jnp.where(valid, self.revive_price(state)[cell], 0)
        return state.replace(garrison_hp=jnp.where(put, 1, state.garrison_hp), gold=state.gold-cost,
            last_service_cost=jnp.where(valid, cost, state.last_service_cost),
            last_target=jnp.where(valid, cell, state.last_target), last_damage=jnp.where(valid, 1, state.last_damage),
            last_event=jnp.where(valid, GARRISON_REVIVED, state.last_event))

    def _heal(self, state, action, valid):
        """The Temple heals a selected settlement's wounded defender: whole HP the gold
        pays for, up to the maximum; the change is kept. No reward."""
        site = self.selected_site(state)
        cell = jnp.clip(action-self.heal_start, 0, 5)
        price = jnp.maximum(self.heal_price(state)[cell], 1)
        missing = jnp.maximum(self.garrison_max_hp(state)[site, cell]-state.garrison_hp[site, cell], 0)
        healed = jnp.where(valid, jnp.minimum(missing, state.gold // price), 0)
        put = valid & (jnp.arange(self.count)[:, None] == site) & (jnp.arange(6) == cell)
        cost = healed*price
        return state.replace(garrison_hp=jnp.where(put, state.garrison_hp+healed, state.garrison_hp), gold=state.gold-cost,
            last_service_cost=jnp.where(valid, cost, state.last_service_cost),
            last_target=jnp.where(valid, cell, state.last_target), last_damage=jnp.where(valid, healed, state.last_damage),
            last_event=jnp.where(valid, GARRISON_HEALED, state.last_event))

    def _store(self, env, state, action, valid):
        """Party slot -> local garrison cell; the slot is cleared like a dismissal."""
        site = self.transfer_site(state)
        slot = jnp.clip(action-self.store_start, 0, 5)
        cell = self.store_targets(state)[slot]
        uid = state.unit_ids[slot]
        put = valid & (jnp.arange(self.count)[:, None] == site) & (jnp.arange(6) == cell)
        moved = state.replace(garrison_ids=jnp.where(put, uid, state.garrison_ids),
            garrison_levels=jnp.where(put, state.unit_levels[slot], state.garrison_levels),
            garrison_xp=jnp.where(put, state.unit_xp[slot], state.garrison_xp),
            garrison_hp=jnp.where(put, jnp.maximum(state.hp[slot], 0), state.garrison_hp),
            garrison_doses=jnp.where(put[..., None], state.potion_doses[slot], state.garrison_doses),
            garrison_mercenary=jnp.where(put, state.mercenary_units[slot], state.garrison_mercenary),
            last_target=jnp.where(valid, slot, state.last_target),
            last_event=jnp.where(valid, UNIT_STORED, state.last_event))
        slots = jnp.arange(12)
        affected = valid & ((slots == slot) | ((self.progression.sizes[uid] == 2) & (slots == slot+3)))
        return env._clear_roster_slots(moved, affected)

    def _take(self, env, state, action, valid):
        """Local garrison cell -> party slot, with permanent potions and mercenary mark."""
        site = jnp.maximum(self.transfer_site(state), 0)
        cell = jnp.clip(action-self.take_start, 0, 5)
        slot = self.take_targets(state)[cell]
        uid, level = state.garrison_ids[site, cell], state.garrison_levels[site, cell]
        xp, hp = state.garrison_xp[site, cell], state.garrison_hp[site, cell]
        doses, mercenary = state.garrison_doses[site, cell], state.garrison_mercenary[site, cell]
        slots = jnp.arange(12)
        affected = valid & ((slots == slot) | ((self.progression.sizes[uid] == 2) & (slots == slot+3)))
        state = env._clear_roster_slots(state, affected)
        put = valid & (slots == slot)

        def at(values, value):
            mask = put.reshape((12,)+(1,)*(values.ndim-1))
            return jnp.where(mask, jnp.asarray(value).astype(values.dtype), values)
        fields = dict(unit_ids=at(state.unit_ids, uid), unit_levels=at(state.unit_levels, level),
                      unit_xp=at(state.unit_xp, xp), native_ids=at(state.native_ids, uid),
                      native_levels=at(state.native_levels, level), native_xp=at(state.native_xp, xp),
                      hp=at(state.hp, hp), potion_doses=at(state.potion_doses, doses),
                      mercenary_units=at(state.mercenary_units, mercenary))
        if self.potion_rules is not None:
            fields['potion_bonus'] = at(state.potion_bonus, self.potion_bonus(uid[None], level[None], doses[None])[0])
        clear = valid & (jnp.arange(self.count)[:, None] == site) & (jnp.arange(6) == cell)
        return state.replace(**fields, garrison_ids=jnp.where(clear, 0, state.garrison_ids),
            garrison_levels=jnp.where(clear, 0, state.garrison_levels), garrison_xp=jnp.where(clear, 0, state.garrison_xp),
            garrison_hp=jnp.where(clear, 0, state.garrison_hp),
            garrison_doses=jnp.where(clear[..., None], 0, state.garrison_doses),
            garrison_mercenary=state.garrison_mercenary & ~clear,
            last_target=jnp.where(valid, slot, state.last_target),
            last_event=jnp.where(valid, UNIT_TAKEN, state.last_event))

    # Days and arrivals -----------------------------------------------------
    def regenerate(self, state, resting):
        """Getfortregen: REGEN + lord + city level (capital 35), at most 100%, no revival."""
        ids, hp = state.garrison_ids, state.garrison_hp
        maximum = self.garrison_max_hp(state)
        place_bonus = jnp.concatenate((self.territory.fort_bonus[state.city_levels], jnp.array([CAPITAL_REGENERATION])))
        percent = jnp.clip(self.territory.regen[ids]+self.territory.lord_regen+place_bonus[:, None], 0, 100)
        heal = jnp.maximum(1, jnp.rint(maximum*percent/100.).astype(jnp.int32))
        healing = resting & self.owned(state)[:, None] & (ids != 0) & (hp > 0) & (hp < maximum)
        return state.replace(garrison_hp=jnp.where(healing, jnp.minimum(maximum, hp+heal), hp))

    def lose(self, state, sites):
        """Cities in sites pass to their attackers; the capital or a defended city loses the episode."""
        cities = sites[:self.cities]
        defeat = sites[self.capital_index]
        if self.defended >= 0:
            defeat |= cities[self.defended]
        clear = jnp.concatenate((cities, jnp.zeros(1, bool)))[:, None]
        return state.replace(city_owned=state.city_owned & ~cities,
            selected_garrison=jnp.where(jnp.any(sites & (jnp.arange(self.count) == state.selected_garrison)),
                                         self.capital_index, state.selected_garrison),
            garrison_ids=jnp.where(clear, 0, state.garrison_ids), garrison_levels=jnp.where(clear, 0, state.garrison_levels),
            garrison_xp=jnp.where(clear, 0, state.garrison_xp), garrison_hp=jnp.where(clear, 0, state.garrison_hp),
            garrison_doses=jnp.where(clear[..., None], 0, state.garrison_doses),
            garrison_mercenary=state.garrison_mercenary & ~clear,
            last_event=jnp.where(jnp.any(cities), CITY_LOST, state.last_event),
            lost=state.lost | defeat, done=state.done | defeat)

    def advance(self, state, active=True):
        """After the enemy turn: armies on an owned entrance without the party attack its garrison."""
        site = self.site(state.enemy_positions)
        safe = jnp.maximum(site, 0)
        owned = (site >= 0) & self.owned(state)[safe]
        party_there = jnp.all(state.position == self.positions[safe], axis=-1) & self.has_party(state)
        arrived = (active & ~state.done & state.alive & self.attackers & owned & ~party_there
                   & ~state.pending_encounters & ~state.garrison_pending)
        defended = self.living(state)[safe]
        fallen = jnp.any((arrived & ~defended)[:, None] & (site[:, None] == jnp.arange(self.count)), axis=0)
        # Telemetry: the first army that walks into a capital with no living defender.
        entering = arrived & ~defended & (site == self.capital_index)
        state = state.replace(capital_captor=jnp.where(jnp.any(entering), jnp.argmax(entering), state.capital_captor))
        return self.lose(state.replace(garrison_pending=state.garrison_pending | (arrived & defended)), fallen)

    # Battles ---------------------------------------------------------------
    def engage(self, state, allowed):
        """Stash the party and set the garrison against the first waiting army.

        The garrison brings its own permanent potions. The party's potion layers wait in
        garrison_party_layers; its cached equipment bonus is refreshed when the battle ends.
        """
        site = self.site(state.enemy_positions)
        safe = jnp.maximum(site, 0)
        waiting = (state.garrison_pending & state.alive & (site >= 0) & self.owned(state)[safe]
                   & self.living(state)[safe])
        start = allowed & jnp.any(waiting)
        enemy = jnp.argmax(waiting).astype(jnp.int32)
        g = safe[enemy]
        party = jnp.stack((state.unit_ids[:6], state.unit_levels[:6], state.unit_xp[:6], state.hp[:6]))

        def put(values, row):
            return values.at[:6].set(jnp.where(start, jnp.asarray(row).astype(values.dtype), values[:6]))
        layers = jnp.concatenate((state.potion_doses[:6].astype(jnp.uint32), state.potion_active[:6, None],
                                  state.potion_wards[:6, None]), axis=1)
        doses = state.garrison_doses[g]
        fields = dict(potion_doses=put(state.potion_doses, doses),
                      potion_temporary=put(state.potion_temporary, 0.), potion_active=put(state.potion_active, 0),
                      potion_wards=put(state.potion_wards, 0), equipment_bonus=put(state.equipment_bonus, 0.))
        if self.potion_rules is not None:
            fields['potion_bonus'] = put(state.potion_bonus, self.potion_bonus(
                state.garrison_ids[g], state.garrison_levels[g], doses))
        return state.replace(**fields, garrison_party=jnp.where(start, party, state.garrison_party),
            garrison_party_layers=jnp.where(start, layers, state.garrison_party_layers),
            unit_ids=put(state.unit_ids, state.garrison_ids[g]), unit_levels=put(state.unit_levels, state.garrison_levels[g]),
            unit_xp=put(state.unit_xp, state.garrison_xp[g]), hp=put(state.hp, state.garrison_hp[g]),
            garrison_battle=start, garrison_site=jnp.where(start, g, state.garrison_site),
            enemy=jnp.where(start, enemy, state.enemy), origin=jnp.where(start, state.position, state.origin),
            # Stale entries (the city already fell) are dropped.
            garrison_pending=waiting.at[enemy].set(waiting[enemy] & ~start)), start

    def fortify(self, state, started):
        from stoix.envs.number_grid import FORT_BLUE
        g = jnp.maximum(state.garrison_site, 0)
        city = self.territory.fort_bonus[state.city_levels[jnp.minimum(g, max(self.cities-1, 0))]] if self.cities else 0
        armor = jnp.where(g == self.capital_index, CAPITAL_ARMOR, city)
        return state.replace(
            battle_hero_fort_armor=jnp.where(started, armor, state.battle_hero_fort_armor),
            battle_fort_armor=jnp.where(started, 0, state.battle_fort_armor),
            battle_fort_defender=jnp.where(started, FORT_BLUE, state.battle_fort_defender))

    def city_fall(self, state, lost):
        """The garrison battle just lost a city (not the capital)."""
        return lost & state.garrison_battle & (state.garrison_site != self.capital_index)

    def finish(self, before, after, ended, lost):
        """Save the garrison, return the party, and give a fallen city away.

        Dead defenders keep their cells for the Temple while the settlement stays owned.
        """
        closing = before.garrison_battle & ended
        g = jnp.maximum(before.garrison_site, 0)
        ids, hp = after.unit_ids[:6], jnp.maximum(after.hp[:6], 0)
        keep = ids != 0
        row = closing & (jnp.arange(self.count) == g)[:, None]

        def save(values, current):
            mask = row.reshape(row.shape+(1,)*(values.ndim-2))
            kept = keep.reshape(keep.shape+(1,)*(values.ndim-2))
            return jnp.where(mask, jnp.where(kept, current, jnp.zeros_like(current))[None], values).astype(values.dtype)
        party, layers = before.garrison_party, before.garrison_party_layers

        def back(values, current):
            return values.at[:6].set(jnp.where(closing, jnp.asarray(current).astype(values.dtype), values[:6]))
        doses = layers[:, :DOSES].astype(jnp.int32)
        restored = dict(unit_ids=back(after.unit_ids, party[0]), unit_levels=back(after.unit_levels, party[1]),
                        unit_xp=back(after.unit_xp, party[2]), hp=back(after.hp, party[3]),
                        last_xp=back(after.last_xp, jnp.zeros(6, jnp.int32)),
                        last_promoted=jnp.where(closing, after.last_promoted & jnp.uint32(0xFC0), after.last_promoted),
                        potion_doses=back(after.potion_doses, doses),
                        potion_active=back(after.potion_active, layers[:, DOSES]),
                        potion_wards=back(after.potion_wards, layers[:, DOSES+1]))
        if self.potion_rules is not None:
            intrinsic = self.intrinsic(party[0], party[1])
            bonus = jnp.where((party[0] != 0)[:, None], self.potion_rules.rebuild(intrinsic, doses), 0.)
            restored['potion_bonus'] = back(after.potion_bonus, bonus)
            restored['potion_temporary'] = back(after.potion_temporary, jnp.where(
                (party[0] != 0)[:, None], self.potion_rules.temporary_bonus(intrinsic+bonus, layers[:, DOSES]), 0.))
        for name, current in (('native_ids', party[0]), ('native_levels', party[1]), ('native_xp', party[2])):
            restored[name] = back(getattr(after, name), current)
        state = after.replace(garrison_ids=save(after.garrison_ids, ids),
            garrison_levels=save(after.garrison_levels, after.unit_levels[:6]),
            garrison_xp=save(after.garrison_xp, after.unit_xp[:6]), garrison_hp=save(after.garrison_hp, hp),
            garrison_doses=save(after.garrison_doses, after.potion_doses[:6]),
            garrison_mercenary=save(after.garrison_mercenary, before.garrison_mercenary[g]),
            garrison_battle=after.garrison_battle & ~closing,
            garrison_site=jnp.where(closing, -1, after.garrison_site), **restored)
        fallen = self.city_fall(before, lost) & closing
        # Telemetry: the army that beat the capital's garrison.
        taken = closing & lost & (before.garrison_site == self.capital_index)
        state = state.replace(capital_captor=jnp.where(taken, before.enemy, state.capital_captor))
        # A lost city is not the end of the episode: undo the battle's defeat, then give it away.
        state = end_battle(state.replace(lost=state.lost & ~fallen, done=state.done & ~fallen), fallen)
        return self.lose(state, fallen & (jnp.arange(self.count) == g))

    # Observation -------------------------------------------------------------
    def observation(self, state):
        ids, hp = state.garrison_ids, state.garrison_hp
        maximum = self.garrison_max_hp(state)
        cells = jnp.stack((ids/len(self.progression.rows), hp/jnp.maximum(maximum, 1),
                           (self.progression.sizes[ids] == 2).astype(jnp.float32), state.garrison_levels/10.), axis=-1)
        attacked = jnp.any((state.garrison_pending & state.alive)[:, None]
                           & (self.site(state.enemy_positions)[:, None] == jnp.arange(self.count)), axis=0)
        head = jnp.stack((self.owned(state).astype(jnp.float32), self.capacity(state)/6.,
                          jnp.sum(self.occupied(ids), axis=1)/6.,
                          (state.garrison_battle & (state.garrison_site == jnp.arange(self.count))).astype(jnp.float32),
                          attacked.astype(jnp.float32)), axis=-1)
        sites = jnp.concatenate((head, cells.reshape(self.count, -1)), axis=1).reshape(-1)
        if not self.action_count:
            return sites
        swaps = jnp.concatenate((state.garrison_swaps_today[:, None]/SWAP_LIMIT,
            ((state.garrison_swaps_used[:, None] & self.swap_bits) != 0).astype(jnp.float32)), axis=1).reshape(-1)
        selected = (jnp.arange(self.count) == self.selected_site(state)).astype(jnp.float32)
        return jnp.concatenate((sites, swaps, selected))

    def snapshot(self, state):
        """UI: local settlement transfers and the selected garrison's paid recovery offers."""
        maximum = self.garrison_max_hp(state)
        sites = [dict(name=self.metadata['sites'][g], owned=bool(self.owned(state)[g]),
                      capacity=int(self.capacity(state)[g]),
                      units=[dict(cell=c, name=self.progression.rows[int(state.garrison_ids[g, c])]['name'],
                                  hp=int(state.garrison_hp[g, c]), max_hp=int(maximum[g, c]),
                                  level=int(state.garrison_levels[g, c]),
                                  mercenary=bool(state.garrison_mercenary[g, c]))
                             for c in range(6) if int(state.garrison_ids[g, c])])
                 for g in range(self.count)]
        if not self.action_count:
            return sites
        mask = jax.device_get(self.available(state))
        transfer_site = int(self.transfer_site(state))
        recovery_site = int(self.selected_site(state))
        prices = jax.device_get(self.capital.prices[state.garrison_ids]) if self.revive_count else None
        store = jax.device_get(self.store_targets(state)).tolist()
        take = jax.device_get(self.take_targets(state)).tolist()
        for g, site in enumerate(sites):
            here = g == transfer_site
            site['at_party'] = here
            site['selected'] = site['recovery'] = g == recovery_site
            site['select'] = bool(mask[self.select_start-self.start+g])
            site['store'] = [here and bool(mask[self.store_start-self.start+s]) for s in range(6)]
            site['store_cells'] = store if here else [-1]*6
            for unit in site['units']:
                c = unit['cell']
                unit['take'] = here and bool(mask[self.take_start-self.start+c])
                unit['take_slot'] = int(take[c]) if here else -1
                if self.revive_count:
                    unit['revive'] = site['recovery'] and bool(mask[self.hire_end-self.start+c])
                    unit['revive_cost'] = int(prices[g, c, 1])
                    unit['heal'] = site['recovery'] and bool(mask[self.heal_start-self.start+c])
                    unit['heal_price'] = int(prices[g, c, 0])
            site['swaps'] = [site['selected'] and bool(mask[self.swap_start-self.start+k]) for k in range(len(SWAP_PAIRS))]
            site['swaps_today'] = int(state.garrison_swaps_today[g])
        return sites
