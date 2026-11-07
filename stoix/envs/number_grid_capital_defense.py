"""Persistent home-capital guardian, resolved with the ordinary JAX combat engine.

Only guardian HP, attacking armies and the PRNG stream leave the private battle.
The player's party, progression, inventory and battle rewards are not copied back.
"""
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_bot import ScriptedBotRules
from stoix.envs.number_grid_combat import HP
from stoix.envs.number_grid_lords import lord_settings
from stoix.envs.number_grid_territory import CAPITAL_ARMOR, CAPITAL_REGENERATION


class CapitalDefenseRules:
    def __init__(self, game_map, progression):
        spec = game_map.get('capital_defender')
        self.enabled = bool(spec)
        self.observation_size = int(self.enabled)
        self.maximum = jnp.int32(0)
        if not self.enabled:
            return
        unit = progression.ids[spec['unit']]
        if not bool(progression.capital_guards[unit]):
            raise ValueError('capital_defender must be a capital guardian')
        self.ids = jnp.array([0, unit, 0, 0, 0, 0], jnp.int32)
        self.levels = jnp.array([0, spec['level'], 0, 0, 0, 0], jnp.int32)
        self.maximum = progression.stats(self.ids, self.levels)[1, HP].astype(jnp.int32)
        row = progression.rows[unit]
        self.regeneration = min(100, CAPITAL_REGENERATION + lord_settings(game_map)['regeneration']
            + int(row.get('regeneration', row.get('game_data', {}).get('REGEN', 0)) or 0))
        self.attackers = jnp.array(spec['attackers'], jnp.int32)
        if (len(set(spec['attackers'])) != len(spec['attackers']) or not spec['attackers']
                or any(type(i) is not int or not 0 <= i < len(game_map['opponent_positions'])
                       for i in spec['attackers'])):
            raise ValueError('capital_defender requires distinct valid attackers')
        self.capital = jnp.array(game_map['capital_position'], jnp.int32)
        self.battle_decisions = 300

    def observation(self, state):
        return jnp.atleast_1d(state.capital_defender_hp/self.maximum)

    def regenerate(self, state):
        hp = state.capital_defender_hp
        restored = jnp.minimum(self.maximum, hp+(self.maximum*self.regeneration+99)//100)
        return state.replace(capital_defender_hp=jnp.where(hp > 0, restored, hp))

    def _battle(self, env, state, enemy, active):
        from stoix.envs.number_grid import FORT_BLUE
        scratch = state.replace(bot_autobattle=jnp.bool_(True),
            position=self.capital, origin=self.capital, enemy=enemy,
            unit_ids=state.unit_ids.at[:6].set(self.ids),
            unit_levels=state.unit_levels.at[:6].set(self.levels),
            unit_xp=state.unit_xp.at[:6].set(0),
            hp=jnp.zeros_like(state.hp).at[1].set(state.capital_defender_hp),
            done=jnp.bool_(False), won=jnp.bool_(False), lost=jnp.bool_(False),
            summon_battle=jnp.bool_(False), spell_effects=jnp.zeros_like(state.spell_effects),
            potion_bonus=jnp.zeros_like(state.potion_bonus), potion_doses=jnp.zeros_like(state.potion_doses),
            potion_temporary=jnp.zeros_like(state.potion_temporary), potion_active=jnp.zeros_like(state.potion_active),
            potion_wards=jnp.zeros_like(state.potion_wards),
            equipped=jnp.full_like(state.equipped, -1), equipment_bonus=jnp.zeros_like(state.equipment_bonus),
            item_inventory=jnp.full_like(state.item_inventory, -1),
            item_charges_spent=jnp.zeros_like(state.item_charges_spent),
            battle_items=jnp.full_like(state.battle_items, -1))
        scratch = env._begin_battle(scratch).replace(battle_hero_fort_armor=jnp.int32(CAPITAL_ARMOR),
            battle_fort_armor=jnp.int32(0), battle_fort_defender=jnp.int32(FORT_BLUE))

        def activation(carry):
            decisions, current = carry
            automatic = env.automatic_battle_turn(current)
            action = jnp.where(automatic, env._automatic_battle_action(current),
                               ScriptedBotRules.blue_action(env, current))
            key, draw = jax.random.split(current.battle_key)
            following, _ = env._battle_step(current, action, key,
                                            jax.random.uniform(draw, (env.random_size,)))
            return decisions+(~automatic).astype(jnp.int32), following

        _, result = jax.lax.while_loop(
            lambda c: active & c[1].in_battle & ~c[1].done
                & ((c[0] < self.battle_decisions) | env.automatic_battle_turn(c[1])),
            activation, (jnp.int32(0), scratch))
        hp = result.hp[1]
        lost = hp <= 0
        defeated = ~result.alive[enemy]
        maximum = env.progression.stats(result.unit_ids[6:], result.unit_levels[6:])[:, HP].astype(jnp.int32)
        wounds = jnp.where(result.in_battle, jnp.maximum(0, maximum-result.hp[6:]), result.enemy_wounds[enemy])
        return state.replace(capital_defender_hp=hp, battle_key=result.battle_key,
            alive=state.alive.at[enemy].set(state.alive[enemy] & ~defeated),
            enemy_wounds=state.enemy_wounds.at[enemy].set(jnp.where(defeated, 0, wounds)),
            enemy_progress=state.enemy_progress.at[enemy].set(result.enemy_progress[enemy]),
            enemy_spell_effects=state.enemy_spell_effects.at[enemy].set(
                jnp.where(defeated, jnp.uint32(0), state.enemy_spell_effects[enemy])),
            pending_encounters=state.pending_encounters.at[enemy].set(state.pending_encounters[enemy] & ~defeated),
            lost=state.lost | lost, done=state.done | lost)

    def advance(self, env, state, active=True):
        """Resolve each arrival once per enemy turn; a present hero defends manually."""
        def defend(index, current):
            enemy = self.attackers[index]
            fight = (active & ~current.done & current.alive[enemy]
                     & ~current.pending_encounters[enemy]
                     & jnp.all(current.enemy_positions[enemy] == self.capital)
                     & jnp.any(current.position != self.capital))
            # Explicit active flags keep vmapped inactive lanes out of the loop.
            return jax.lax.cond(fight, lambda s: self._battle(env, s, enemy, fight), lambda s: s, current)
        return jax.lax.fori_loop(0, len(self.attackers), defend, state)
