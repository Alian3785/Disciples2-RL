"""User-requested Capital training rewards (2026-10-16, 2026-10-26), not original game rules.

Defensive bottles include temporary/permanent armor, permanent HP and elemental
wards; acquisition also counts accuracy and initiative bottles (2026-10-26).
Sticky peaks prevent retreat/re-entry or slot swaps from farming rewards.
The guardian battle bonus (2026-10-26) pays once per episode for a hero-army
battle with the guardian that starts after enough other armies were defeated.
The Incubus bonus (2026-10-26) pays for every party fighter that evolves into an
Incubus, and once for building the hall that allows it; the most living Incubi
held at once is kept for the charts.
"""
import math

import jax.numpy as jnp

from stoix.envs.number_grid_potions import ARMOR, HP


class CapitalTrainingRules:
    def __init__(self, env, game_map):
        fields = ('capital_defense_acquisition_reward', 'capital_defense_effect_reward',
                  'capital_guardian_round_reward', 'capital_entry_level_reward',
                  'capital_guardian_battle_reward', 'capital_incubus_reward', 'capital_incubus_building_reward',
                  'capital_guardian_rounds_bonus')
        values = [game_map.get(k, 0.) for k in fields]
        (self.acquisition_reward, self.effect_reward, self.round_reward, self.entry_level_reward,
         self.battle_reward, self.incubus_reward, self.incubus_building_reward, self.rounds_bonus) = values
        for key, value in zip(fields, values):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(key + ' must be finite and nonnegative')
        self.min_defeated = game_map.get('capital_guardian_battle_min_defeated', 0)
        if type(self.min_defeated) is not int or self.min_defeated < 0:
            raise ValueError('capital_guardian_battle_min_defeated must be a nonnegative integer')
        self.bonus_rounds = game_map.get('capital_guardian_rounds_bonus_rounds', 10)
        if type(self.bonus_rounds) is not int or self.bonus_rounds < 1:
            raise ValueError('capital_guardian_rounds_bonus_rounds must be a positive integer')
        self.enabled = any(values)
        # Defense peaks and rounds are observed only while they can still pay.
        self.tracks_effects = bool(self.effect_reward or self.round_reward)
        # Entry snapshots also feed the first-contact metrics.
        self.tracks_entry = bool(self.entry_level_reward or self.battle_reward)
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
        # Acquisition also pays for accuracy and initiative bottles (user request 2026-10-26);
        # the effect peaks above stay defensive.
        self.acquired = jnp.array([i for i, p in enumerate(potions.items)
            if p['effect'] in ('armor', 'health', 'accuracy', 'initiative') or p['effect'].startswith('ward_')],
            jnp.int32)
        self.keys = tuple(potions.keys[int(i)] for i in self.indices.tolist())
        self.count = len(self.keys)
        self.bits = potions.bits[self.indices]
        self.wards = potions.wards[self.indices]
        self.permanent = potions.permanent[self.indices]
        self.stats = potions.stats[self.indices]
        self.health = self.permanent & (self.stats == HP)
        self.observation_size = ((self.count + 1) * self.tracks_effects + bool(self.entry_level_reward)
                                 + 2 * bool(self.battle_reward))
        self.max_rounds = env.max_rounds
        self.others = jnp.arange(env.num_opponents) != self.enemy
        # Scripted Empire parties may attack repeatedly: each victory over them counts instead.
        self.bot_victories = env.bot_attack.enabled
        if self.bot_victories:
            self.others &= ~env.bot_attack.mask
        self.incubus = env.progression.ids['incubus']
        # The capital building that evolves a Demonologist into an Incubus (the Legions' Hall of Deception).
        demonologist = env.progression.ids['demonologist']
        self.incubus_building = jnp.sum(jnp.where(env.progression.targets[demonologist] == self.incubus,
                                                  env.progression.building_bits[demonologist], 0)).astype(jnp.uint32)
        if self.incubus_building_reward and not int(self.incubus_building):
            raise ValueError('capital_incubus_building_reward needs a faction building that evolves Incubi')

    def incubus_hall(self, before, after):
        """Pay once per episode when the Incubus building is completed; buildings are never lost."""
        bit = self.incubus_building
        built = ((after.buildings & bit) == bit) & ((before.buildings & bit) != bit) & ~before.done
        return self.incubus_building_reward * built.astype(jnp.float32)

    def incubi(self, state):
        """Living Incubi of the hero's own party; summons and copies do not count."""
        return jnp.sum((state.unit_ids[:6] == self.incubus) & (state.hp[:6] > 0)
                       & (state.summon_owner[:6] < 0) & ~state.copied[:6]).astype(jnp.int32)

    def track_incubi(self, state):
        """Keep the most living Incubi held at once this episode."""
        return state.replace(capital_incubi_peak=jnp.maximum(state.capital_incubi_peak, self.incubi(state)))

    def new_incubi(self, state, ids, victory):
        """Pay for every party fighter evolving into an Incubus in a won battle (also in a raid)."""
        evolved = jnp.sum((ids[:6] == self.incubus) & (state.unit_ids[:6] != self.incubus))
        return self.incubus_reward * jnp.where(victory & ~state.summon_battle & ~state.bot_autobattle, evolved, 0)

    def defeated(self, state):
        """Armies other than the guardian defeated this episode, by battle or map spell."""
        defeated = jnp.sum(state.scenario_spawned & ~state.alive & self.others).astype(jnp.int32)
        return defeated+state.bot_hero_victories if self.bot_victories else defeated

    def enter(self, state, engaged, strength):
        """Pay for hero encounters with the guardian, before automatic attacks, even if lost.

        The entry-level bonus pays on the first encounter only; a zero-strength entry also
        consumes it, and retreating, upgrading or recruiting later never unlocks another
        payment. The battle bonus pays on the first encounter that starts with at least
        `min_defeated` other armies defeated; earlier encounters leave it available.
        """
        contact = (engaged & state.in_battle & ~state.done & ~state.summon_battle & ~state.bot_autobattle
                   & (state.enemy == self.enemy)
                   & jnp.any(self.guard_slots & (state.hp[6:] > 0))
                   & jnp.any((state.unit_ids[:6] != 0) & (state.hp[:6] > 0)))
        first = contact & (state.capital_entry_strength < 0)
        battle = (contact & ~state.capital_guardian_bonus_paid & bool(self.battle_reward)
                  & (self.defeated(state) >= self.min_defeated))
        return state.replace(capital_entry_strength=jnp.where(first, strength, state.capital_entry_strength),
                             capital_guardian_bonus_paid=state.capital_guardian_bonus_paid | battle), (
            self.entry_level_reward * jnp.where(first, strength, 0) + self.battle_reward * battle)

    def acquire(self, before, after):
        acquired = jnp.sum(jnp.maximum(after.potions[self.acquired] - before.potions[self.acquired], 0))
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
        Only genuinely started rounds 2..max_rounds pay, once per episode. Reaching
        `bonus_rounds` survived rounds also pays `rounds_bonus` once per episode.
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
        # One-off bonus (2026-10-27) when the episode's survived guardian rounds first reach the mark.
        milestone = (state.capital_rounds_rewarded < self.bonus_rounds) & (rounds >= self.bonus_rounds)
        reward = (self.effect_reward * effect_gain
                  + self.round_reward * (rounds - state.capital_rounds_rewarded)
                  + self.rounds_bonus * milestone)
        return state.replace(capital_defense_peak=peak, capital_rounds_rewarded=rounds), reward

    def observation(self, state):
        values = jnp.zeros(0, jnp.float32)
        if self.tracks_effects:
            values = jnp.concatenate((state.capital_defense_peak / 6.,
                jnp.asarray([state.capital_rounds_rewarded / max(self.max_rounds - 1, 1)], jnp.float32)))
        if self.entry_level_reward:
            # -0.01 means unseen; zero means encountered with only level-1 fighters.
            values = jnp.concatenate((values, jnp.asarray([state.capital_entry_strength / 100.], jnp.float32)))
        if self.battle_reward:
            # Progress towards the defeated-army threshold, then whether the bonus was paid.
            progress = jnp.minimum(self.defeated(state) / max(self.min_defeated, 1), 1.)
            values = jnp.concatenate((values, jnp.asarray(
                [progress, state.capital_guardian_bonus_paid], jnp.float32)))
        return values
