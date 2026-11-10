"""Capital defense and hero party rewards (user request for S143, 09.11.2026), map key `defense_rewards`.

attacker_xp: an army whose current order leads to the capital entrance pays, when it is destroyed
in any way (garrison or hero battle, a spell), attacker_xp times the experience its units give for
a kill (Gunits XP_KILLED at their levels). Stopping the White Dragon is then worth many scouts;
the separate battle-victory reward also uses the weak-repeat discount described below.

party_level_record: direct reward per point by which the sum of living party units' levels
exceeds its episode record. The initial sum is the baseline; losing, reviving or replacing the
party never resets the record. The garrison and summons do not count: party achievements are
updated on the map, after permanent units have been restored at the end of a battle.

level2_multiplier / level3_multiplier: the best episode-wide multiplier achieved when every
occupied party slot is alive and at least level 2 / 3, including the scenario's leader. Empty
slots do not count. The final transition adds (multiplier - 1) * the raw episode return, so the
sum of rewards, including costs, defeats and the terminal reward, is multiplied exactly once.

mana_guard: a separate one-off reward per distinct enemy guarding one of the rod's mana sites.

party_victory / party_guard_victory: a one-off bonus for a personal victory of the living
scenario leader's party over a permanent map army outside the capital. The guard amount
replaces the ordinary amount for mana guards; the separate mana_guard reward still applies.

level2_bonus / level3_bonus: immediate, one-off payouts when the whole living party reaches
each level milestone. Both are paid if level 3 is reached before level 2 was ever recorded.

repeat_wave_multiplier / repeat_wave_max_xp: after the first destruction, a recurring event
template worth at most the XP threshold pays the reduced fraction of its battle-victory
reward and capital-attacker XP reward. Strong armies and first victories pay in full.

capital_attacker_rewards=false excludes every army ever ordered to the capital from
combat, offensive-cast, recovery-credit and terminal-clearing rewards, including before
its order fires. Gameplay XP and the hero party's development remain unchanged.
party_day_rewards checks cumulative actual hero-party promotions once at each specified
day crossing. A missed deadline is never paid later; garrisons and summons do not count.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

KEYS = {'attacker_xp', 'party_level_record', 'mana_guard', 'level2_multiplier', 'level3_multiplier',
        'party_victory', 'party_guard_victory', 'level2_bonus', 'level3_bonus',
        'repeat_wave_multiplier', 'repeat_wave_max_xp'}


def _spec(game_map):
    spec = game_map.get('defense_rewards')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != KEYS:
        raise ValueError('defense_rewards needs exactly: '+', '.join(sorted(KEYS)))
    if any(type(spec[k]) not in (int, float) or not math.isfinite(spec[k]) or spec[k] < 0 for k in KEYS):
        raise ValueError('defense_rewards values must be finite and nonnegative')
    if not 1 <= spec['level2_multiplier'] <= spec['level3_multiplier']:
        raise ValueError('defense_rewards multipliers must be increasing and at least 1')
    if spec['repeat_wave_multiplier'] > 1 or spec['repeat_wave_max_xp'] <= 0:
        raise ValueError('defense_rewards need repeat_wave_multiplier in [0, 1] and positive repeat_wave_max_xp')
    if type(game_map.get('capital_attacker_rewards', True)) is not bool:
        raise ValueError('capital_attacker_rewards must be a boolean')
    milestones = game_map.get('party_day_rewards', [])
    if not isinstance(milestones, list):
        raise ValueError('party_day_rewards must be a list')
    previous_day = previous_upgrades = 0
    for milestone in milestones:
        if (not isinstance(milestone, dict) or set(milestone) != {'day', 'upgrades', 'reward'}
                or type(milestone['day']) is not int or milestone['day'] <= max(1, previous_day)
                or type(milestone['upgrades']) is not int or milestone['upgrades'] <= previous_upgrades
                or type(milestone['reward']) not in (int, float)
                or not math.isfinite(milestone['reward']) or milestone['reward'] < 0):
            raise ValueError('party_day_rewards needs increasing days/upgrades and finite nonnegative rewards')
        previous_day, previous_upgrades = milestone['day'], milestone['upgrades']
    return spec


class DefenseRewards:
    def __init__(self, game_map, env):
        spec = _spec(game_map)
        self.enabled = spec is not None
        self.observation_size = self.guard_count = self.enemy_count = 0
        self.day_count = 0
        self.suppress_attacker_rewards = False
        if not self.enabled:
            return
        if not env.campaign_events.enabled or not env.progression_enabled or not env.leader_hire.enabled:
            raise ValueError('defense_rewards need campaign_events, unit progression and a scenario leader')
        for name in KEYS:
            setattr(self, name, float(spec[name]))
        self.leader = env.leader_hire.unit
        self.enemy_count = env.num_opponents
        rules = env.campaign_events
        capital = list(game_map['capital_position'])
        targets = np.asarray(rules.targets).tolist()[:rules.target_count]
        # A campaign variant may retain party rewards while removing all capital attacks.
        self.capital_target = targets.index(capital) if capital in targets else None
        self.suppress_attacker_rewards = not game_map.get('capital_attacker_rewards', True)
        # Classify from the entire script, not just the current order: an early
        # interception of a future capital attacker must not earn a field-kill bonus.
        attackers = {army['enemy'] for army in game_map['campaign_events']['armies']
                     if army.get('target') == capital}
        attackers.update(effect['enemy'] for event in game_map['campaign_events']['events']
                         for effect in event['effects'] if effect['kind'] in ('CREATE_STACK', 'CHANGE_ORDER')
                         and effect.get('target') == capital)
        attackers.update(event['enemy'] for event in game_map.get('scheduled_enemies', [])
                         if event.get('target') == capital)
        self.rewardable = jnp.array([not self.suppress_attacker_rewards or i not in attackers
                                    for i in range(self.enemy_count)])
        self.movers = rules.movers
        rods = game_map.get('rod_planter', {})
        guards = sorted({guard for source, guard in zip(rods.get('sources', []), rods.get('guards', []))
                         if guard >= 0 and game_map['mines'][source]['kind'] != 'gold'})
        self.guards = jnp.array(guards, jnp.int32)
        self.guard_count = len(guards)
        self.guard_mask = jnp.zeros(self.enemy_count, bool).at[self.guards].set(True)
        created = {effect['enemy'] for event in game_map['campaign_events']['events']
                   for effect in event['effects'] if effect['kind'] == 'CREATE_STACK'}
        repeating = sorted({effect['enemy'] for event in game_map['campaign_events']['events'] if not event['once']
                            for effect in event['effects'] if effect['kind'] == 'CREATE_STACK'})
        self.permanent = jnp.array([enemy not in created for enemy in range(self.enemy_count)])
        self.repeat_indices = jnp.array(repeating, jnp.int32)
        self.repeating = jnp.zeros(self.enemy_count, bool).at[self.repeat_indices].set(True)
        self.capital = jnp.array(capital, jnp.int32)
        day_rewards = game_map.get('party_day_rewards', [])
        self.day_count = len(day_rewards)
        self.days = jnp.array([m['day'] for m in day_rewards], jnp.int32)
        self.upgrades = jnp.array([m['upgrades'] for m in day_rewards], jnp.int32)
        self.day_rewards = jnp.array([m['reward'] for m in day_rewards], jnp.float32)
        self.upgrade_scale = float(day_rewards[-1]['upgrades']) if day_rewards else 1.
        self.observation_size = 5+len(repeating)+(1+self.day_count if self.day_count else 0)
        self.metadata = {name: float(spec[name]) for name in KEYS}
        self.metadata['mana_guards'] = guards
        self.metadata['repeating_armies'] = repeating
        self.metadata['capital_attacker_rewards'] = not self.suppress_attacker_rewards
        self.metadata['excluded_attackers'] = sorted(attackers) if self.suppress_attacker_rewards else []
        self.metadata['party_day_rewards'] = day_rewards

    def promotion_count(self, state, promoted):
        """Count actual growth/form changes once at the progression boundary, before restoration."""
        party = jnp.any(state.unit_ids[:6] == self.leader)
        eligible = party & ~state.garrison_battle & ~state.summon_battle & ~state.bot_autobattle
        return jnp.where(eligible, jax.lax.population_count(promoted & jnp.uint32(63)), 0).astype(jnp.int32)

    def capital_bound(self, env, state):
        """[enemies] campaign armies whose current order is the capital entrance."""
        if self.capital_target is None:
            return jnp.zeros(env.num_opponents, bool)
        heading = state.campaign_targets == self.capital_target
        return jnp.zeros(env.num_opponents, bool).at[self.movers].set(heading)

    def army_xp(self, env, state, enemy=None):
        """Kill experience of an army (or all armies), at current permanent levels."""
        ids = env.casting.current_enemy_ids(state) if env.spell_casting_enabled else env.progression.enemy_ids
        levels = state.enemy_progress[..., 0]
        if enemy is not None:
            ids, levels = ids[enemy], levels[enemy]
        worth = env.progression.experience(ids, levels, jnp.zeros_like(levels))[..., 1]
        return jnp.sum(jnp.where(ids != 0, worth, 0), axis=-1).astype(jnp.float32)

    def repeat_factor(self, env, state, enemy=None, worth=None):
        """A weak recurring template is paid in full only until its first destruction."""
        worth = self.army_xp(env, state, enemy) if worth is None else worth
        repeated, defeated = self.repeating, state.defense_defeated
        if enemy is not None:
            repeated, defeated = repeated[enemy], defeated[enemy]
        reduced = repeated & defeated & (worth <= self.repeat_wave_max_xp)
        return jnp.where(reduced, self.repeat_wave_multiplier, 1.).astype(jnp.float32)

    def battle(self, before, after, victory):
        """Record each real destruction; award eligible personal victories inside the battle loop."""
        enemy = jnp.maximum(before.enemy, 0)
        leader = jnp.any((after.unit_ids[:6] == self.leader) & (after.hp[:6] > 0))
        outside = jnp.any(before.position != self.capital) & jnp.any(before.enemy_positions[enemy] != self.capital)
        personal = (victory & ~before.summon_battle & ~before.garrison_battle & ~before.bot_autobattle
                    & leader & outside & self.permanent[enemy] & ~before.defense_defeated[enemy]
                    & self.rewardable[enemy])
        bonus = jnp.where(self.guard_mask[enemy], self.party_guard_victory, self.party_victory)
        return after.replace(
            defense_defeated=after.defense_defeated.at[enemy].set(after.defense_defeated[enemy] | victory),
            party_field_victories=after.party_field_victories+personal.astype(jnp.int32)), personal*bonus

    @staticmethod
    def level_sum(state):
        """Sum of living permanent party levels, read on the map after battle restoration."""
        living = (state.unit_ids[:6] != 0) & (state.hp[:6] > 0)
        return jnp.sum(jnp.where(living, state.unit_levels[:6], 0)).astype(jnp.int32)

    def milestones(self, state):
        present = state.unit_ids[:6] != 0
        living = state.hp[:6] > 0
        leader = jnp.any((state.unit_ids[:6] == self.leader) & living)
        all2 = leader & jnp.all(~present | (living & (state.unit_levels[:6] >= 2)))
        all3 = leader & jnp.all(~present | (living & (state.unit_levels[:6] >= 3)))
        return jnp.array([all2, all3])

    def multiplier(self, state):
        all2, all3 = self.milestones(state)
        return jnp.where(all3, self.level3_multiplier, jnp.where(all2, self.level2_multiplier, 1.)).astype(jnp.float32)

    def reset(self, state):
        return state.replace(party_level_record=self.level_sum(state),
                             party_reward_multiplier=self.multiplier(state),
                             party_level_milestones=self.milestones(state))

    def apply(self, env, before, after):
        """Once per decision: capital attackers, mana guards and new party level records."""
        destroyed = before.alive & ~after.alive
        worth = self.army_xp(env, before)
        factor = self.repeat_factor(env, before, worth=worth)
        reward = self.attacker_xp*jnp.sum(jnp.where(
            destroyed & self.capital_bound(env, before) & self.rewardable, worth*factor, 0.))
        guards = before.alive[self.guards] & ~after.alive[self.guards] & ~before.mana_guards_rewarded
        reward += self.mana_guard*jnp.sum(guards)
        record = jnp.where(after.in_battle, before.party_level_record,
                           jnp.maximum(before.party_level_record, self.level_sum(after)))
        reward += self.party_level_record*(record-before.party_level_record)
        multiplier = jnp.where(after.in_battle, before.party_reward_multiplier,
                               jnp.maximum(before.party_reward_multiplier, self.multiplier(after)))
        milestones = self.milestones(after) & ~after.in_battle & ~before.party_level_milestones
        reward += self.level2_bonus*milestones[0]+self.level3_bonus*milestones[1]
        day_milestones = before.party_day_milestones
        if self.day_count:
            crossed = (before.day < self.days) & (after.day >= self.days)
            earned = (crossed & ~before.party_day_milestones & ~after.lost
                      & (after.party_upgrade_count >= self.upgrades))
            reward += jnp.sum(self.day_rewards*earned)
            day_milestones = day_milestones | earned
        return after.replace(party_level_record=record, party_reward_multiplier=multiplier,
                             mana_guards_rewarded=before.mana_guards_rewarded | guards,
                             party_level_milestones=before.party_level_milestones | milestones,
                             defense_defeated=after.defense_defeated | destroyed,
                             party_day_milestones=day_milestones), reward

    @staticmethod
    def finish(before, after, reward):
        """Multiply the entire episode return, after all reward components, on its last step."""
        raw_return = before.defense_raw_return+jnp.where(before.done, 0., reward)
        terminal = ~before.done & after.done
        adjustment = jnp.where(terminal, (after.party_reward_multiplier-1.)*raw_return, 0.)
        return after.replace(defense_raw_return=raw_return), reward+adjustment

    def observation(self, state):
        parts = (jnp.array([state.party_level_record/100., state.party_reward_multiplier-1.,
                                           state.defense_raw_return/1000.], jnp.float32),
                 state.party_level_milestones, state.defense_defeated[self.repeat_indices])
        if self.day_count:
            parts += (jnp.array([state.party_upgrade_count/self.upgrade_scale], jnp.float32),
                      state.party_day_milestones)
        return jnp.concatenate(parts)
