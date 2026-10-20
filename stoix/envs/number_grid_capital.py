"""Paid capital recovery, ported from campaign_env_economy.py in the reference.

Prices come from Gunits.HEAL_C/REVIVE_C through unit_service_cost_data.py and
unit_revive_costs.py. Current profiles have no level growth in healing prices.
The user's integer-HP rule rounds affordable healing down and retains change.
"""
import jax.numpy as jnp

from stoix.envs.number_grid_combat import can_revive

HEAL_START, REVIVE_START, CAPITAL_ACTIONS = 44, 50, 56
# User-requested RL shaping: 100x below the reference healing bonus.
HEAL_REWARD, REVIVE_REWARD = .0001, .5
CAPITAL_HEAL, CAPITAL_REVIVE = 17, 18


class CapitalRules:
    def __init__(self, progression, construction, position):
        self.territory = None
        self.position = jnp.asarray(position, jnp.int32)
        self.temple_slot = next(i for i, row in enumerate(construction.rows) if row['name'] == 'Храм')
        self.temple_bit = jnp.uint32(1 << self.temple_slot)
        rows = progression.rows[1:]
        for row in rows:
            if any(type(row.get(k)) is not int or row[k] <= 0
                   for k in ('heal_gold_per_hp', 'revive_gold')):
                raise ValueError('Capital service prices must be positive integers')
        self.prices = jnp.array([[0, 0]] + [[r['heal_gold_per_hp'], r['revive_gold']]
                                          for r in rows], jnp.int32)

    def at_capital(self, state):
        return self.territory.service_at(state) if self.territory is not None else jnp.all(state.position == self.position)

    def temple_built(self, state):
        return (state.buildings & self.temple_bit) != 0

    def available(self, state, max_hp, slot=None):
        ids, hp = state.unit_ids[:6], state.hp[:6]
        maximum = max_hp[:6]
        if slot is not None:
            ids, hp, maximum = ids[slot], hp[slot], maximum[slot]
        prices = self.prices[ids]
        common = (self.at_capital(state) & self.temple_built(state)
                  & ~state.in_battle & ~state.done & (ids != 0))
        heal = common & (hp > 0) & (hp < maximum) & (state.gold >= prices[..., 0])
        revive = common & can_revive(state, jnp.arange(6) if slot is None else slot) & (state.gold >= prices[..., 1])
        return heal, revive

    def apply(self, state, action, max_hp):
        """Called from a validated world step; other actions leave recovery intact."""
        healing = (action >= HEAL_START) & (action < REVIVE_START)
        reviving = (action >= REVIVE_START) & (action < CAPITAL_ACTIONS)
        slot = jnp.clip(jnp.where(healing, action-HEAL_START, action-REVIVE_START), 0, 5)
        reviving &= can_revive(state, slot)
        service = healing | reviving
        prices = self.prices[state.unit_ids[slot]]
        amount = jnp.minimum(jnp.maximum(max_hp[slot]-state.hp[slot], 0),
                             state.gold // jnp.maximum(prices[0], 1))
        healed = jnp.where(healing, amount, 0)
        spent = healed * prices[0] + jnp.where(reviving, prices[1], 0)
        # Reference balances may be negative: paid recovery beyond the earned
        # allowance creates debt which subsequent victories must repay first.
        bonus = (jnp.minimum(healed, jnp.maximum(state.recovery_balance[0], 0)) * HEAL_REWARD
                 + jnp.where(reviving & (state.recovery_balance[1] >= 1), REVIVE_REWARD, 0.))
        hp = jnp.where(reviving, 1, state.hp[slot] + healed)
        return state.replace(
            hp=state.hp.at[slot].set(hp), gold=state.gold-spent,
            recovery_balance=state.recovery_balance-jnp.array([healed, reviving], jnp.int32),
            last_event=jnp.where(service, jnp.where(healing, CAPITAL_HEAL, CAPITAL_REVIVE), state.last_event),
            last_target=jnp.where(service, slot, state.last_target),
            last_damage=jnp.where(service, jnp.where(healing, healed, 1), state.last_damage),
            last_service_cost=jnp.where(service,spent,state.last_service_cost),
        ), bonus

    def observation(self, state, size):
        return jnp.concatenate((self.position / (size-1),
            jnp.asarray([self.at_capital(state)], jnp.float32),
            state.recovery_balance / jnp.array([1000., 10.]),
            (self.prices[state.unit_ids[:6]] / jnp.array([3., 1000.])).reshape(-1)))

    def quotes(self, state, max_hp):
        """Display-only rows: rate, missing HP, affordable HP, cost, revive cost,
        heal enabled, revive enabled. Costs and legality stay in the JAX engine.
        """
        prices = self.prices[state.unit_ids[:6]]
        missing = jnp.maximum(max_hp[:6]-state.hp[:6], 0)
        amount = jnp.where(state.hp[:6] > 0,
                           jnp.minimum(missing, state.gold // jnp.maximum(prices[:, 0], 1)), 0)
        heal, revive = self.available(state, max_hp)
        return jnp.stack((prices[:, 0], missing, amount, amount*prices[:, 0],
                          prices[:, 1], heal, revive), axis=1).astype(jnp.int32)

    def metadata(self):
        return dict(position=self.position.tolist(), temple_action=18+self.temple_slot,
                    heal_start=HEAL_START, revive_start=REVIVE_START, revive_hp=1,
                    heal_reward_per_hp=HEAL_REWARD, revive_reward=REVIVE_REWARD,
                    healing_rounding='floor', guardian=False)
