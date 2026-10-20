"""User-requested Capital training rewards (2026-10-16), not original game rules.

Defensive bottles include temporary/permanent armor, permanent HP and elemental
wards. Sticky peaks prevent retreat/re-entry or slot swaps from farming rewards.
"""
import math

import jax.numpy as jnp

from stoix.envs.number_grid_potions import ARMOR, HP


class CapitalTrainingRules:
    def __init__(self, env, game_map):
        fields = ('capital_defense_acquisition_reward', 'capital_defense_effect_reward',
                  'capital_guardian_round_reward', 'capital_entry_level_reward')
        values = [game_map.get(k, 0.) for k in fields]
        self.acquisition_reward, self.effect_reward, self.round_reward, self.entry_level_reward = values
        for key, value in zip(fields, values):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(key + ' must be finite and nonnegative')
        self.enabled = any(values)
        self.count = 0
        self.observation_size = 0
        if not self.enabled:
            return
        if (game_map.get('campaign_objective') != 'capital_guardian' or env.objective_enemy is None
                or not env.progression_enabled or not env.potions_enabled or env.reference_rewards):
            raise ValueError('Capital training rewards require a current-profile capital_guardian map with potions')
        self.enemy = env.objective_enemy
        self.guard_slots = env.progression.capital_guards[env.progression.enemy_ids[self.enemy]]
        if not bool(jnp.any(self.guard_slots)):
            raise ValueError('Capital training target must contain a capital guardian')
        potions = env.potion_rules
        self.indices = jnp.array([i for i, p in enumerate(potions.items)
            if p['effect'] in ('armor', 'health') or p['effect'].startswith('ward_')], jnp.int32)
        self.keys = tuple(potions.keys[int(i)] for i in self.indices.tolist())
        self.count = len(self.keys)
        self.bits = potions.bits[self.indices]
        self.wards = potions.wards[self.indices]
        self.permanent = potions.permanent[self.indices]
        self.stats = potions.stats[self.indices]
        self.health = self.permanent & (self.stats == HP)
        self.observation_size = self.count + 1 + bool(self.entry_level_reward)
        self.max_rounds = env.max_rounds

    def enter(self, state, engaged, strength):
        """Pay for the first hero encounter, before automatic attacks, even if it is lost.

        A zero-strength entry also consumes the encounter. Retreating, upgrading
        or recruiting later never unlocks another payment in the same episode.
        """
        first = (engaged & state.in_battle & ~state.done & ~state.summon_battle & ~state.bot_autobattle
                 & (state.enemy == self.enemy) & (state.capital_entry_strength < 0)
                 & jnp.any(self.guard_slots & (state.hp[6:] > 0))
                 & jnp.any((state.unit_ids[:6] != 0) & (state.hp[:6] > 0)))
        return state.replace(capital_entry_strength=jnp.where(first, strength, state.capital_entry_strength)), (
            self.entry_level_reward * jnp.where(first, strength, 0))

    def acquire(self, before, after):
        acquired = jnp.sum(jnp.maximum(after.potions[self.indices] - before.potions[self.indices], 0))
        acquired = jnp.where(before.done, 0, acquired)
        return after.replace(capital_defense_acquired=after.capital_defense_acquired + acquired), (
            self.acquisition_reward * acquired)

    def active_counts(self, state):
        living = ((state.hp[:6] > 0) & ~state.escaped[:6] & (state.summon_owner[:6] < 0)
                  & ~state.copied[:6] & ~state.imp[:6] & ~state.fenrir[:6] & (state.decay_form[:6] == 0))
        temporary = (state.potion_active[:6, None] & self.bits) != 0
        permanent = ((state.potion_doses[:6, self.stats] > 0)
                     & (state.potion_bonus[:6, self.stats] > 0))
        ward = ((state.potion_wards[:6, None] & ~state.wards_used[:6, None] & self.wards) != 0)
        armor = state.potion_temporary[:6, ARMOR, None] > 0
        active = jnp.where(self.permanent, permanent,
                           temporary & jnp.where(self.wards != 0, ward, armor))
        return jnp.sum(active & living[:, None], axis=0).astype(jnp.int32)

    def battle(self, state, hp, escaped, round_number):
        """Read effects before the activation, including a fatal first enemy hit.

        Round 0 is preparation; round 1 is entry and earns no survival bonus.
        Only genuinely started rounds 2..max_rounds pay, once per episode.
        """
        eligible = (state.in_battle & ~state.done & ~state.summon_battle & ~state.bot_autobattle
                    & (state.enemy == self.enemy) & (state.round > 0) & (state.post_victory == 0)
                    & jnp.any(self.guard_slots & (state.hp[6:] > 0)))
        counts = jnp.where(eligible, self.active_counts(state), 0)
        peak = jnp.maximum(state.capital_defense_peak, counts)
        effect_gain = jnp.sum(peak - state.capital_defense_peak)
        survivors = jnp.any((hp[:6] > 0) & ~escaped[:6] & (state.summon_owner[:6] < 0))
        new_round = (eligible & survivors & jnp.any(self.guard_slots & (hp[6:] > 0))
                     & (round_number > state.round) & (round_number <= self.max_rounds))
        reached = jnp.where(new_round, jnp.maximum(round_number - 1, 0), 0)
        rounds = jnp.maximum(state.capital_rounds_rewarded, reached)
        reward = (self.effect_reward * effect_gain
                  + self.round_reward * (rounds - state.capital_rounds_rewarded))
        return state.replace(capital_defense_peak=peak, capital_rounds_rewarded=rounds), reward

    def observation(self, state):
        values = jnp.concatenate((state.capital_defense_peak / 6.,
            jnp.asarray([state.capital_rounds_rewarded / max(self.max_rounds - 1, 1)], jnp.float32)))
        if self.entry_level_reward:
            # -0.01 means unseen; zero means encountered with only level-1 fighters.
            values = jnp.concatenate((values, jnp.asarray([state.capital_entry_strength / 100.], jnp.float32)))
        return values
