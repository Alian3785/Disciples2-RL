"""Incremental archer/mage battles. Every simulation operation runs on JAX/GPU.

A step is a map move or ONE unit action. Enemy turns and completing a retreat
use the sole legal CONTINUE action. Saved maps can still use numeric rules.
"""
import dataclasses
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import struct
from stoa import AddActionMaskWrapper, ArraySpace, DictSpace, DiscreteSpace
from stoix.envs.number_grid_legacy import (
    DIRECTIONS, NumberGrid as NumericNumberGrid, NumberGridState,
)

MAP = json.loads((Path(__file__).resolve().parents[2] / 'number_grid_map.json').read_text())
SHOOT, DEFEND, WAIT, RETREAT, CONTINUE, ACTIONS = 8, 14, 15, 16, 17, 18
ACTION_NAMES = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW') + tuple(
    f'shoot_{i}' for i in range(6)) + ('defend', 'wait', 'retreat', 'continue')
MOVE, ENGAGE, HIT, MISS, GUARD, DELAY, FLEE, ESCAPE, VICTORY, DEFEAT, WITHDRAW, LIMIT = range(12)


@struct.dataclass
class BattleState(NumberGridState):
    battle_key: jax.Array
    in_battle: jax.Array
    enemy: jax.Array
    origin: jax.Array
    hp: jax.Array
    priority: jax.Array
    turn_phase: jax.Array  # 0: normal turn, 1: waiting, 2: finished
    defended: jax.Array
    retreating: jax.Array
    escaped: jax.Array
    actor: jax.Array
    round: jax.Array
    last_event: jax.Array
    last_actor: jax.Array
    last_target: jax.Array  # -1 for area attacks or non-attacks
    last_damage: jax.Array  # total HP removed, summed over area targets
    battle_steps: jax.Array
    player_turns: jax.Array
    enemy_turns: jax.Array


class NumberGrid(NumericNumberGrid):
    def __init__(self, max_steps=None, map_config=None):
        game_map = MAP if map_config is None else map_config
        super().__init__(max_steps=max_steps, map_config=game_map)
        self.battle_mode = bool(game_map.get('battle_mode', False))
        if not self.battle_mode:
            return
        self.mask_walls = True
        self.hero_count = int(game_map.get('hero_units', 6))
        self.hero_hp = int(game_map.get('hero_hp', 45))
        self.damage = int(game_map.get('archer_damage', 25))
        self.accuracy = float(game_map.get('archer_accuracy', .8))
        # Missing role fields preserve the rules of saved all-archer maps.
        self.mage_slot = game_map.get('hero_mage_slot', -1)
        self.mage_damage = game_map.get('mage_damage', 20)
        if (type(self.mage_slot) is not int
                or not (-1 <= self.mage_slot < self.hero_count)
                or type(self.mage_damage) is not int or self.mage_damage <= 0):
            raise ValueError('Invalid mage battle configuration')
        self.max_rounds = int(game_map.get('battle_max_rounds', 75))
        counts, health = game_map['enemy_units'], game_map['enemy_hp']
        if not (1 <= self.hero_count <= 6 and self.hero_hp > 0 and self.damage > 0
                and 0 <= self.accuracy <= 1 and self.max_rounds > 0):
            raise ValueError('Invalid archer battle configuration')
        if len(counts) != self.num_opponents or len(health) != self.num_opponents:
            raise ValueError('Every enemy needs troop count and hit points')
        for count, hp in zip(counts, health):
            if not (isinstance(count, int) and isinstance(hp, int)
                    and 1 <= count <= self.hero_count and 0 < hp <= self.hero_hp
                    and (count < self.hero_count or hp < self.hero_hp)):
                raise ValueError('Every enemy must be strictly weaker in count or HP')
        self.enemy_counts = jnp.asarray(counts, jnp.int32)
        self.enemy_health = jnp.asarray(health, jnp.int32)
        self.slots = jnp.arange(12)
        self.hero_full = jnp.where(jnp.arange(6) < self.hero_count, self.hero_hp, 0)
        self.restored_hp = jnp.concatenate((self.hero_full, jnp.zeros(6, jnp.int32)))
        self.observation_version = int(game_map.get('battle_observation_version', 1))
        self.observation_size = (4 + 5 * self.num_opponents + 48 + 6 if self.observation_version == 2
                                 else self.observation_size + 2 * self.num_opponents + 12 * 8 + 6)

    def reset(self, rng_key, env_params=None):
        if not self.battle_mode:
            return super().reset(rng_key, env_params)
        visited = jnp.zeros(self.visited_words, jnp.uint32)
        if self.exploration_bonus:
            r, c = self.map_config['agent_position']
            cell = r * self.size + c
            visited = visited.at[cell // 32].set(jnp.uint32(1 << (cell % 32)))
        zero = jnp.int32(0)
        state = BattleState(
            position=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            number=jnp.int32(self.initial_number), alive=jnp.ones(self.num_opponents, bool),
            step_count=zero, done=jnp.bool_(False), won=jnp.bool_(False), lost=jnp.bool_(False),
            visited=visited, battle_key=rng_key, in_battle=jnp.bool_(False), enemy=jnp.int32(-1),
            origin=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            hp=self.restored_hp, priority=jnp.zeros(12), turn_phase=jnp.zeros(12, jnp.int32),
            defended=jnp.zeros(12, bool), retreating=jnp.zeros(12, bool), escaped=jnp.zeros(12, bool),
            actor=zero, round=zero, last_event=jnp.int32(MOVE), last_actor=jnp.int32(-1),
            last_target=jnp.int32(-1), last_damage=zero, battle_steps=zero,
            player_turns=zero, enemy_turns=zero,
        )
        return state, self._timestep(state, jnp.float32(0), first=True)

    def max_hp(self, state):
        enemy = jnp.maximum(state.enemy, 0)
        opponents = jnp.where(jnp.arange(6) < self.enemy_counts[enemy], self.enemy_health[enemy], 0)
        return jnp.concatenate((self.hero_full, jnp.where(state.in_battle, opponents, 0)))

    def observation(self, state):
        if not self.battle_mode:
            return super().observation(state)
        context = jnp.asarray([
            state.in_battle, (state.enemy + 1) / self.num_opponents,
            state.round / self.max_rounds, state.actor / 11,
            state.origin[0] / (self.size - 1), state.origin[1] / (self.size - 1)
        ], jnp.float32)
        if self.observation_version == 2:
            # No duplicate max-HP arrays or obsolete numeric battle strengths.
            # Signed queue priority contains both order and waiting/acted status.
            world = jnp.stack((self.opponent_positions[:,0] / (self.size-1),
                               self.opponent_positions[:,1] / (self.size-1), state.alive,
                               self.enemy_counts / 6, self.enemy_health / self.hero_hp),axis=1)
            active = (state.hp > 0) & ~state.escaped
            queue = jnp.where(active & (state.turn_phase == 0), state.priority,
                             jnp.where(active & (state.turn_phase == 1), -state.priority, 0.))
            units = jnp.stack((state.hp / self.hero_hp, queue, state.defended,
                               jnp.where(state.escaped, 1., state.retreating * .5)),axis=1)
            return jnp.concatenate((state.position / (self.size-1),
                                    jnp.asarray([state.number / self.number_scale,
                                                 state.step_count / self.max_steps],jnp.float32),
                                    world.reshape(-1), units.reshape(-1), context))
        base = super().observation(state)
        units = jnp.stack((state.hp / self.hero_hp, self.max_hp(state) / self.hero_hp,
                           state.priority, state.turn_phase / 2, state.defended,
                           state.retreating, state.escaped,
                           (self.slots == state.actor) & state.in_battle), axis=1)
        return jnp.concatenate((base, self.enemy_counts / 6, self.enemy_health / self.hero_hp,
                                units.reshape(-1), context))

    def action_mask(self, state):
        if not self.battle_mode:
            return super().action_mask(state)
        destination = state.position[None] + self.directions
        occupied = jnp.any(state.alive[None, :] & jnp.all(
            destination[:, None, :] == self.opponent_positions[None, :, :], axis=-1), axis=1)
        movement = super().action_mask(state) & ~occupied & ~state.in_battle
        controlled = state.in_battle & (state.actor < 6) & ~state.retreating[state.actor]
        targets = (state.hp[6:] > 0) & ~state.escaped[6:] & controlled
        mask = jnp.concatenate((movement, targets, jnp.asarray([
            controlled, controlled & (state.turn_phase[state.actor] == 0),
            controlled, state.in_battle & ~controlled])))
        return jnp.where(state.done, jnp.arange(ACTIONS) == CONTINUE, mask)

    def _timestep(self, state, reward, first=False):
        ts = super()._timestep(state, reward, first)
        if self.battle_mode:
            ts = ts.replace(extras={**ts.extras, 'battle_transition': jnp.bool_(False),
                                    'player_battle_transition': jnp.bool_(False),
                                    'enemy_battle_transition': jnp.bool_(False),
                                    'battle_victory': jnp.bool_(False)})
        return ts

    def _begin_battle(self, state, key=None, random_values=None):
        if key is None:
            key, random_key = jax.random.split(state.battle_key)
            random_values = jax.random.uniform(random_key, (13,))
        priority = random_values[1:] + 1
        enemy_hp = jnp.where(jnp.arange(6) < self.enemy_counts[state.enemy],
                             self.enemy_health[state.enemy], 0)
        hp = jnp.concatenate((self.hero_full, enemy_hp))
        actor = jnp.argmax(jnp.where(hp > 0, priority, -100)).astype(jnp.int32)
        return state.replace(
            battle_key=key, in_battle=jnp.bool_(True), hp=hp, priority=priority,
            turn_phase=jnp.zeros(12, jnp.int32), defended=jnp.zeros(12, bool),
            retreating=jnp.zeros(12, bool), escaped=jnp.zeros(12, bool),
            actor=actor, round=jnp.int32(1), last_event=jnp.int32(ENGAGE),
            last_actor=jnp.int32(-1), last_target=jnp.int32(-1), last_damage=jnp.int32(0),
        )

    def _world_step(self, state, action, key, random_values):
        destination = state.position + self.directions[jnp.clip(action, 0, 7)]
        moved = action < 8  # step() checks the complete legality mask.
        position = jnp.where(moved, destination, state.position)
        nearby = state.alive & (jnp.max(jnp.abs(self.opponent_positions - position), axis=1) == 1)
        engage = moved & jnp.any(nearby)
        enemy = jnp.argmax(nearby).astype(jnp.int32)
        next_state = state.replace(position=position, origin=state.position,
                                   enemy=jnp.where(engage, enemy, state.enemy),
                                   last_event=jnp.int32(MOVE), last_actor=jnp.int32(-1),
                                   last_target=jnp.int32(-1), last_damage=jnp.int32(0))
        bonus = jnp.float32(0)
        if self.exploration_bonus:
            cell = position[0] * self.size + position[1]
            word, bit = cell // 32, jnp.left_shift(jnp.uint32(1), (cell % 32).astype(jnp.uint32))
            first = (state.visited[word] & bit) == 0
            bonus = jnp.where(first & moved, jnp.float32(self.exploration_bonus), 0.)
            next_state = next_state.replace(visited=state.visited.at[word].set(state.visited[word] | bit))
        next_state = jax.lax.cond(engage, lambda s: self._begin_battle(s, key, random_values), lambda s: s, next_state)
        return next_state, bonus

    def _enemy_action(self, state, random_values):
        valid = (state.hp[:6] > 0) & ~state.escaped[:6]
        damage = jnp.where(state.defended[:6], (self.damage + 1) // 2, self.damage)
        kill = valid & (state.hp[:6] <= damage)
        candidates = valid & jnp.where(jnp.any(kill), kill, True)
        # HP dominates the random tie breaker, including among killable targets.
        score = jnp.where(candidates, state.hp[:6] + random_values[:6] * .5, 1e9)
        return SHOOT + jnp.argmin(score).astype(jnp.int32)

    def _battle_step(self, state, action, key, random_values):
        actor = state.actor
        escaping = state.retreating[actor]
        action = jnp.where(actor >= 6, self._enemy_action(state, random_values[1:]), action)
        action = jnp.where(escaping, CONTINUE, action)
        attack = (action >= SHOOT) & (action < DEFEND)
        target = jnp.clip(action - SHOOT, 0, 5) + jnp.where(actor < 6, 6, 0)
        hit = attack & (random_values[0] < self.accuracy)
        area_attack = jnp.bool_(False)
        if self.mage_slot >= 0:
            # A fixed hero role is inferable from the actor/HP slots already in
            # observation v2. One hit roll covers the entire spell, without
            # extra RNG, loops, or host operations in the vmapped learner.
            area_attack = actor == self.mage_slot
            targets = jnp.where(area_attack, self.slots >= 6, self.slots == target)
            targets &= (state.hp > 0) & ~state.escaped
            base_damage = jnp.where(area_attack, self.mage_damage, self.damage)
            damage = jnp.where(state.defended, (base_damage + 1) // 2, base_damage)
            removed = jnp.where(hit & targets, jnp.minimum(damage, state.hp), 0)
            hp = state.hp - removed
            applied = jnp.sum(removed)
        else:
            damage = jnp.where(state.defended[target], (self.damage + 1) // 2, self.damage)
            applied = jnp.where(hit, jnp.minimum(damage, state.hp[target]), 0)
            hp = state.hp.at[target].add(-applied)
        phases = state.turn_phase.at[actor].set(jnp.where(action == WAIT, 1, 2))
        defended = state.defended.at[actor].set(action == DEFEND)
        retreating = state.retreating.at[actor].set(action == RETREAT)
        escaped = state.escaped.at[actor].set(escaping)
        active = (hp > 0) & ~escaped
        normal, waiting = active & (phases == 0), active & (phases == 1)
        new_round = ~jnp.any(normal | waiting)
        phases = jnp.where(new_round, jnp.zeros(12, jnp.int32), phases)
        priority = jnp.where(new_round, random_values[1:] + 1, state.priority)
        round_number = state.round + new_round.astype(jnp.int32)
        scores = jnp.where(active & (phases == 0), priority,
                            jnp.where(active & (phases == 1), -priority, -100))
        next_actor = jnp.argmax(scores).astype(jnp.int32)
        defended = defended.at[next_actor].set(False)
        lost, victory = ~jnp.any(hp[:6] > 0), ~jnp.any(hp[6:] > 0)
        withdrawal = ~jnp.any(active[:6]) & ~lost
        timeout = new_round & (round_number > self.max_rounds) & ~victory & ~lost & ~withdrawal
        alive = state.alive.at[state.enemy].set(~victory)
        won = ~jnp.any(alive) & ~lost
        back = victory | withdrawal
        event = jnp.where(attack, jnp.where(hit, HIT, MISS),
                            jnp.where(action == DEFEND, GUARD,
                            jnp.where(action == WAIT, DELAY,
                            jnp.where(action == RETREAT, FLEE, ESCAPE))))
        event = jnp.where(victory, VICTORY, jnp.where(lost, DEFEAT,
                            jnp.where(withdrawal, WITHDRAW, jnp.where(timeout, LIMIT, event))))
        next_state = state.replace(
            battle_key=key, hp=jnp.where(back, self.restored_hp, hp),
            in_battle=~back, position=jnp.where(withdrawal, state.origin, state.position),
            alive=alive, number=state.number + victory.astype(jnp.int32), won=won, lost=lost,
            done=won | lost | timeout, priority=jnp.where(back, 0., priority),
            turn_phase=jnp.where(back, 0, phases), defended=defended & ~back,
            retreating=retreating & ~back, escaped=escaped & ~back,
            actor=jnp.where(back, 0, next_actor), round=jnp.where(back, 0, round_number),
            last_event=event.astype(jnp.int32), last_actor=actor,
            last_target=jnp.where(attack & ~area_attack, target, -1), last_damage=applied,
            battle_steps=state.battle_steps + 1,
            player_turns=state.player_turns + ((actor < 6) & ~escaping).astype(jnp.int32),
            enemy_turns=state.enemy_turns + ((actor >= 6) & ~escaping).astype(jnp.int32),
        )
        reward = victory.astype(jnp.float32) + 3 * won - lost.astype(jnp.float32)
        return next_state, reward

    def step(self, state, action, env_params=None):
        if not self.battle_mode:
            return super().step(state, action, env_params)
        # Validate only the selected move/target here. The full action mask is
        # generated once for the next observation, not twice per transition.
        destination = state.position + self.directions[jnp.clip(action, 0, 7)]
        world_valid = ((action >= 0) & (action < 8)
                       & jnp.all((destination > 0) & (destination < self.size - 1))
                       & ~jnp.any(state.alive & jnp.all(self.opponent_positions == destination, axis=1)))
        controlled = (state.actor < 6) & ~state.retreating[state.actor]
        target = jnp.clip(action - SHOOT, 0, 5) + 6
        attack_valid = ((action >= SHOOT) & (action < DEFEND)
                        & (state.hp[target] > 0) & ~state.escaped[target])
        unit_valid = attack_valid | (action == DEFEND) | (action == RETREAT) | ((action == WAIT) & (state.turn_phase[state.actor] == 0))
        battle_valid = jnp.where(controlled, unit_valid, action == CONTINUE)
        valid = jnp.where(state.in_battle, battle_valid, world_valid) & ~state.done
        # Share one random draw between the vmapped map/battle branches.
        key, random_key = jax.random.split(state.battle_key)
        random_values = jax.random.uniform(random_key, (13,))
        next_state, reward = jax.lax.cond(state.in_battle, self._battle_step, self._world_step,
                                         state, action, key, random_values)
        steps = state.step_count + 1
        next_state = next_state.replace(step_count=steps, done=next_state.done | (steps >= self.max_steps))
        fallback = state.replace(step_count=jnp.where(state.done, state.step_count, steps),
                                 done=state.done | (steps >= self.max_steps))
        next_state = jax.tree.map(lambda new, old: jnp.where(valid, new, old), next_state, fallback)
        reward = jnp.where(valid, reward, 0.) - jnp.float32(self.step_cost)
        ts = self._timestep(next_state, jnp.where(state.done, 0., reward))
        combat = state.in_battle & valid & ~state.done
        ts = ts.replace(extras={**ts.extras, 'battle_transition': combat,
                                'player_battle_transition': combat & (state.actor < 6) & ~state.retreating[state.actor],
                                'enemy_battle_transition': combat & (state.actor >= 6),
                                'battle_victory': combat & (next_state.last_event == VICTORY)})
        return next_state, ts

    def action_space(self, env_params=None):
        return DiscreteSpace(ACTIONS if self.battle_mode else 8, dtype=jnp.int32)

    def state_space(self, env_params=None):
        if not self.battle_mode:
            return super().state_space(env_params)
        state, _ = self.reset(jax.random.PRNGKey(0))
        return DictSpace({f.name: ArraySpace(getattr(state, f.name).shape, getattr(state, f.name).dtype)
                          for f in dataclasses.fields(state)})


def wrap_wall_action_mask(env):
    return AddActionMaskWrapper(env) if env.mask_walls else env
