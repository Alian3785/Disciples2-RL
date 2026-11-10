"""Thief poisoning of the army that attacks the capital next turn (user rule for S143, 09.11.2026).

`thief_poison` sends the thief, at most twice per hired thief per day (user rule 09.11.2026:
a successful first try grants one free repeat). A failed thief dies; a living thief who used
both tries must also be replaced to try again that day. Replacements cost 200 gold and require
strictly more than that amount left; the day's first hire needs at least 200. Hiring resets
the new thief's tries, so there is no shared daily limit while replacements are affordable.
It is allowed only while an army already on the map is due to stand on the capital entrance
after the next enemy turn: the env replays
that turn's scheduled and scripted army moves on a copy of the state (armies that an event would
only create that day are not foreseen). The thief poisons the first such army with probability
`chance` percent (50). A living thief starts the next day with both tries restored.
Discipl2.exe, CNobleActionResultPoisonStack (0x41d372): every unit of the stack loses
current HP x POISON_S / 100, truncated, to no less than 1 HP; capital guardians are skipped.
POISON_S is 20 in GVars.dbf, read as a percent (0x5964ae). The wounds stay with the army and
heal by its usual regeneration on REST. No reward.
"""
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_combat import HP

THIEF_POISONED, THIEF_FAILED = 65, 66  # last_event: the army is poisoned / the thief died
MAX_TRIES = 2  # per hired thief per day: the second only after a successful first


def _spec(game_map):
    spec = game_map.get('thief_poison')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'cost', 'chance', 'percent'}:
        raise ValueError('thief_poison takes cost, chance and percent')
    if (any(type(spec[k]) is not int for k in spec) or spec['cost'] <= 0
            or not 0 < spec['chance'] <= 100 or not 0 < spec['percent'] <= 100):
        raise ValueError('thief_poison needs a positive cost, chance and percent up to 100')
    return spec


def thief_action_names(game_map):
    return ('thief_poison',) if _spec(game_map) is not None else ()


def capital_attackers(env, state, capital):
    """Armies on the map now that stand on the capital entrance after the next enemy turn.

    Shared by the thief and the spells at the attacker (number_grid_attacker_spells.py).
    """
    future = state.replace(day=state.day+1)
    if env.scenario.enabled:
        future = env.scenario.advance(future)
    if env.script_events.enabled:
        future = env.script_events.advance(future)
    if env.campaign_events.enabled:
        future = env.campaign_events.advance(future)
    return state.alive & future.alive & jnp.all(future.enemy_positions == capital, axis=-1)


class ThiefRules:
    def __init__(self, game_map, start, territory=None, casting=None, progression=None):
        spec = _spec(game_map)
        self.enabled = spec is not None
        self.start = self.end = self.action = start
        self.observation_size = 0
        if not self.enabled:
            return
        if territory is None or casting is None or progression is None:
            raise ValueError('thief_poison requires territory, spell casting (enemy wounds) and progression')
        self.capital = jnp.asarray(territory.capital, jnp.int32)
        self.casting, self.progression = casting, progression
        self.cost, self.chance, self.percent = spec['cost'], spec['chance'], spec['percent']
        self.end = start+1
        self.observation_size = 3
        self.metadata = dict(action=start, cost=self.cost, chance=self.chance, percent=self.percent)

    def attackers(self, env, state):
        return capital_attackers(env, state, self.capital)

    def available(self, env, state, attackers=None):
        """Reuse a living thief with tries left, or hire another while gold permits."""
        attackers = self.attackers(env, state) if attackers is None else attackers
        tries = state.thief_attempts_today
        free = state.thief_alive & (tries < MAX_TRIES)
        # The user's "more than 200" applies to replacements after an attempt that day.
        hire = (state.gold >= self.cost) & ((tries == 0) | (state.gold > self.cost))
        return ~state.in_battle & ~state.done & (free | hire) & jnp.any(attackers)

    def apply(self, env, state, action, key, resting):
        attackers = self.attackers(env, state)
        using = (action == self.action) & self.available(env, state, attackers)
        hiring = using & (~state.thief_alive | (state.thief_attempts_today >= MAX_TRIES))
        target = jnp.argmax(attackers).astype(jnp.int32)
        # The world key only advances with battles: fold in the decision so every try draws anew.
        draw = jax.random.fold_in(jax.random.fold_in(key, 0x7415), state.step_count)
        success = using & (jax.random.uniform(draw) < self.chance/100.)
        ids = self.casting.current_enemy_ids(state)[target]
        maximum = self.progression.stats(ids, state.enemy_progress[target, :, 0])[:, HP].astype(jnp.int32)
        wounds = state.enemy_wounds[target]
        current = jnp.where(ids != 0, jnp.maximum(maximum-wounds, 0), 0)
        poisoned = success & (current > 0) & ~self.progression.capital_guards[ids]
        remaining = jnp.where(poisoned, jnp.maximum(1, current-current*self.percent//100), current)
        return state.replace(
            enemy_wounds=state.enemy_wounds.at[target].set(jnp.where(poisoned, maximum-remaining, wounds)),
            gold=state.gold-jnp.where(hiring, self.cost, 0),
            thief_alive=jnp.where(using, success, state.thief_alive),  # a failed thief is executed
            thief_attempts_today=jnp.where(resting | hiring, 0, state.thief_attempts_today)+using.astype(jnp.int32),
            last_service_cost=jnp.where(using, jnp.where(hiring, self.cost, 0), state.last_service_cost),
            last_target=jnp.where(using, target, state.last_target),
            last_damage=jnp.where(using, jnp.sum(current-remaining), state.last_damage),
            last_event=jnp.where(using, jnp.where(success, THIEF_POISONED, THIEF_FAILED), state.last_event))

    def observation(self, env, state, attackers=None):
        attackers = self.attackers(env, state) if attackers is None else attackers
        return jnp.stack((state.thief_attempts_today/MAX_TRIES, state.thief_alive, jnp.any(attackers))).astype(jnp.float32)
