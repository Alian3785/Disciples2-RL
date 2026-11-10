"""Defeat only by losing the capital; a destroyed party is replaced by hiring a new leader.

User rule for S143 «Дикие горы» (08.11.2026), as in Disciples II: the party's destruction does
not end the episode. The stack disappears with its items, potions and battle buffs; gold, mana,
spells, buildings, cities and the capital stay. The agent waits at the capital entrance and
may hire the scenario's leader there at level 1 for his ENROLL_C price; ordinary recruitment
follows. Losing the capital's garrison battle is the only defeat. Each destruction of the party,
in battle or by an event's spell, is counted in `party_losses` (telemetry). It has no reward of its
own (user rule 09.11.2026, after -5 and then -1). Party level records and achieved episode
reward multipliers in number_grid_defense_rewards.py survive the party's destruction.

With defeat='party_or_capital' (S143 with unpaid capital attackers, 10.11.2026),
the same party-loss detection instead ends the episode in defeat. The leader's unit
still identifies his party for development rewards; re-hiring is not an action.
"""
import jax.numpy as jnp

from stoix.envs.number_grid_combat import HP
from stoix.envs.number_grid_garrisons import end_battle

LEADER_HIRED = 57  # last_event: a new leader stands in the capital


def leader_hire_names(game_map):
    return (('hire_leader',) if game_map.get('leader_hire')
            and game_map.get('defeat') != 'party_or_capital' else ())


class LeaderHireRules:
    def __init__(self, game_map, progression, garrisons, action):
        spec = game_map.get('leader_hire')
        self.enabled = bool(spec)
        defeat = game_map.get('defeat', 'party')
        self.terminal_party_loss = defeat == 'party_or_capital'
        self.can_hire = self.enabled and not self.terminal_party_loss
        if defeat not in ('party', 'capital', 'party_or_capital'):
            raise ValueError("defeat must be 'party', 'capital' or 'party_or_capital'")
        if (defeat in ('capital', 'party_or_capital')) != self.enabled:
            raise ValueError("defeat 'capital'/'party_or_capital' and leader_hire's unit go together")
        if not self.enabled:
            return
        if progression is None or not garrisons.enabled:
            raise ValueError('leader_hire requires progression and garrisons (a territory campaign)')
        self.garrisons = garrisons
        if set(spec) != {'unit'}:
            raise ValueError('leader_hire takes only unit')
        roster = game_map['hero_roster']
        if spec.get('unit') not in roster:
            raise ValueError('leader_hire unit must be the starting hero')
        self.slot = roster.index(spec['unit'])
        self.unit = progression.initial_ids[self.slot]
        if not bool(progression.heroes[self.unit]):
            raise ValueError('leader_hire unit must be a hero')
        row = progression.rows[int(self.unit)]
        self.level = jnp.int32(row['level'])
        self.price = int(row['game_data']['ENROLL_C'].split(':')[0][1:])
        self.hp = progression.stats(self.unit, self.level)[HP].astype(jnp.int32)
        self.action = action
        self.capital = jnp.array(game_map['capital_position'], jnp.int32)
        self.metadata = dict(action=action if self.can_hire else None, name=row['name'],
                             level=int(row['level']), cost=self.price, slot=self.slot,
                             terminal_party_loss=self.terminal_party_loss)

    @staticmethod
    def has_party(state):
        return jnp.any(state.unit_ids[:6] != 0)

    def available(self, state):
        if not self.can_hire:
            return jnp.bool_(False)
        return (~self.has_party(state) & jnp.all(state.position == self.capital) & (state.gold >= self.price)
                & ~state.in_battle & ~state.done & ~jnp.any(state.garrison_pending & state.alive))

    def apply(self, env, state, action):
        """hire_leader: the leader at his starting cell, level and full HP; full movement today."""
        hired = (action == self.action) & self.available(state)
        cells = jnp.arange(12) == self.slot
        put = lambda values, value: jnp.where(cells & hired, jnp.asarray(value).astype(values.dtype), values)
        hired_state = state.replace(unit_ids=put(state.unit_ids, self.unit), unit_levels=put(state.unit_levels, self.level),
            unit_xp=put(state.unit_xp, 0), native_ids=put(state.native_ids, self.unit),
            native_levels=put(state.native_levels, self.level), native_xp=put(state.native_xp, 0),
            hp=put(state.hp, self.hp), gold=state.gold-jnp.where(hired, self.price, 0),
            last_service_cost=jnp.where(hired, self.price, state.last_service_cost),
            last_event=jnp.where(hired, LEADER_HIRED, state.last_event))
        return hired_state.replace(movement_points=jnp.where(hired, env.movement_cap(hired_state),
                                                             hired_state.movement_points))

    def wipe(self, env, after, wiped):
        """End the episode, or remove the party and wait at the capital, according to its defeat rule."""
        if self.terminal_party_loss:
            return after.replace(lost=after.lost | wiped, done=after.done | wiped,
                                 party_losses=after.party_losses+wiped.astype(jnp.int32))
        state = env._clear_roster_slots(after, (env.slots < 6) & wiped)
        none = lambda values, empty: jnp.where(wiped, jnp.full_like(values, empty), values)
        return state.replace(
            lost=after.lost & ~wiped, done=after.done & ~wiped, party_losses=after.party_losses+wiped.astype(jnp.int32),
            hp=none(after.hp, 0),  # in a battle, the attacker's HP live on as its wounds
            position=jnp.where(wiped, self.capital, after.position), origin=jnp.where(wiped, self.capital, after.origin),
            movement_points=none(after.movement_points, 0), potions=none(after.potions, 0),
            item_inventory=none(after.item_inventory, -1), item_charges_spent=none(after.item_charges_spent, 0),
            equipped=none(after.equipped, -1), battle_items=none(after.battle_items, -1),
            battle_item_charges=none(after.battle_item_charges, 0), battle_item_used=none(after.battle_item_used, False),
            spell_effects=none(after.spell_effects, 0))

    def destroy(self, env, after, wiped):
        """Resolve a party defeat outside a garrison battle using the campaign's defeat rule.

        An attacker standing on an owned city or capital entrance then attacks its garrison,
        in a separate battle.
        """
        state = self.wipe(env, after, wiped)
        if self.terminal_party_loss:
            return end_battle(state, wiped)
        enemy = jnp.maximum(after.enemy, 0)
        site = self.garrisons.site(after.enemy_positions[enemy])
        at_site = wiped & after.alive[enemy] & (site >= 0) & self.garrisons.owned(after)[jnp.maximum(site, 0)]
        state = state.replace(garrison_pending=after.garrison_pending.at[enemy].set(after.garrison_pending[enemy] | at_site))
        return end_battle(state, wiped)
