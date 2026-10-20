"""Method-independent target victories and discounted health-progress shaping.

Training objective requested on 2026-10-18, not an original Disciples rule.
Potential shaping follows Ng, Harada and Russell (ICML 1999). It is applied
once at the public decision boundary, with zero potential at a true terminal.
Time-limit truncations retain the potential because PPO bootstraps their value.
"""
import math

import jax.numpy as jnp

from stoix.envs.number_grid_combat import HP


class TargetRewardRules:
    def __init__(self, env, game_map):
        config = game_map.get('target_rewards')
        self.enabled = config is not None
        self.count = 0
        self.mask = jnp.zeros(env.num_opponents, bool)
        if not self.enabled:
            return
        if not isinstance(config, dict):
            raise ValueError('target_rewards must be an object')
        if (not env.progression_enabled or not env.spell_casting_enabled
                or env.reference_rewards or env.bot.enabled or env.scenario.enabled
                or env.script_events.enabled):
            raise ValueError('target_rewards require a static, non-reference campaign with spell casting')
        indices = config.get('enemies')
        if (not isinstance(indices, list) or not indices
                or any(type(i) is not int or not 0 <= i < env.num_opponents for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('target_rewards.enemies must list distinct valid enemy indices')
        if any(i in game_map.get('optional_enemies', []) for i in indices):
            raise ValueError('Reward targets must be required campaign goals')
        for name in ('victory', 'completion', 'progress_scale', 'gamma'):
            value = config.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'target_rewards.{name} must be finite and nonnegative')
            setattr(self, name, float(value))
        if not 0 < self.gamma <= 1:
            raise ValueError('target_rewards.gamma must be in (0, 1]')
        self.indices = jnp.array(indices, jnp.int32)
        if env.goal_enemies is not None and not bool(jnp.all(env.goal_enemies[self.indices])):
            raise ValueError('Reward targets must be required campaign goals')
        self.count = len(indices)
        self.mask = self.mask.at[self.indices].set(True)
        self.ids = env.progression.enemy_ids[self.indices]
        self.occupied = self.ids != 0
        levels = env.progression.initial_enemies[self.indices, :, 0]
        maximum = env.progression.stats(self.ids, levels)[..., HP]
        self.weights = jnp.where(self.occupied, maximum, 0.)
        self.totals = jnp.maximum(1., jnp.sum(self.weights, axis=1))

    def potential(self, env, state):
        """Original slots only; summons cannot add progress or replace a living escapee.

        Fixed initial-HP weights make the bosses count more than their escorts.
        Current-form HP fractions preserve progress through temporary copying,
        transformation and level changes. Map wounds cover inactive armies.
        """
        levels = state.enemy_progress[self.indices, :, 0]
        maximum = env.progression.stats(self.ids, levels)[..., HP]
        fractions = 1. - state.enemy_wounds[self.indices] / jnp.maximum(maximum, 1.)
        enemy = jnp.maximum(state.enemy, 0)
        native_maximum = env.progression.stats(
            env.progression.enemy_ids[enemy], state.enemy_progress[enemy, :, 0])[:, HP]
        live_fraction = state.hp[6:] / jnp.maximum(env.max_hp(state)[6:], 1.)
        live_fraction = jnp.where(state.summon_owner[6:] >= 0,
            state.summon_saved_hp[6:] / jnp.maximum(native_maximum, 1.), live_fraction)
        active = state.in_battle & (self.indices == state.enemy)
        fractions = jnp.where(active[:, None], live_fraction, fractions)
        fractions = jnp.clip(fractions, 0., 1.)
        # Sum lost HP fractions directly: subtracting a rounded near-one ratio
        # produces a spurious positive reward for completely healthy targets.
        progress = jnp.sum(self.weights * (1.-fractions), axis=1) / self.totals
        progress = jnp.where(state.alive[self.indices], progress, 1.)
        return jnp.where(state.won | state.lost, 0., self.progress_scale * jnp.sum(progress))

    def finish(self, env, before, after):
        """Pay each destruction once, then telescope shaping over agent decisions."""
        fresh = (before.alive[self.indices] & ~after.alive[self.indices]
                 & ~before.target_rewarded)
        paid = before.target_rewarded | fresh
        complete = jnp.all(paid) & ~jnp.all(before.target_rewarded)
        potential = self.potential(env, after)
        reward = (self.victory * jnp.sum(fresh) + self.completion * complete
                  + self.gamma * potential - before.target_potential)
        return after.replace(target_rewarded=paid, target_potential=potential), reward
