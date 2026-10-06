"""Deterministic NumberGrid with simultaneous, loss-first adjacent contacts."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct
from stoa import AddActionMaskWrapper, ArraySpace, DictSpace, DiscreteSpace, Environment, StepType, TimeStep
from numbergrid_progression import validate_progression

MAP = json.loads((Path(__file__).resolve().parents[2] / 'maps/number_grid-24x24-v5.json').read_text())
# Row, column; clockwise from north. Walls/occupied destinations consume a step.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
ACTION_NAMES = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')


@struct.dataclass
class NumberGridState:
    position: jax.Array
    number: jax.Array
    alive: jax.Array
    step_count: jax.Array
    done: jax.Array
    won: jax.Array
    lost: jax.Array
    visited: jax.Array


class NumberGrid(Environment):
    def __init__(self, max_steps=None, map_config=None):
        self.map_config = MAP if map_config is None else map_config
        if 'minimum_reachable_number' in self.map_config:
            validate_progression(self.map_config['opponent_numbers'],
                                 self.map_config['agent_number'],
                                 self.map_config['minimum_reachable_number'])
        max_steps = self.map_config['max_steps'] if max_steps is None else max_steps
        if max_steps <= 0:
            raise ValueError('max_steps must be positive')
        self.size = int(self.map_config['size'])
        self.mask_walls = bool(self.map_config.get('mask_walls', False))
        self.max_steps = int(max_steps)
        self.step_cost = float(self.map_config.get('step_cost', 0.0))
        if not math.isfinite(self.step_cost) or self.step_cost < 0:
            raise ValueError('step_cost must be finite and nonnegative')
        self.exploration_bonus = float(self.map_config.get('exploration_bonus', 0.0))
        if not math.isfinite(self.exploration_bonus) or self.exploration_bonus < 0:
            raise ValueError('exploration_bonus must be finite and nonnegative')
        # One bit per cell: 72 bytes for 24x24, retained entirely on the device.
        self.visited_words = (self.size * self.size + 31) // 32 if self.exploration_bonus else 0
        self.initial_number = int(self.map_config['agent_number'])
        self.num_opponents = len(self.map_config['opponent_numbers'])
        self.number_scale = max(self.initial_number + self.num_opponents,
                                max(self.map_config['opponent_numbers']))
        self.observation_size = 4 + 4 * self.num_opponents
        self.max_return = self.num_opponents + 3 + self.exploration_bonus * ((self.size - 2)**2 - 1)
        self.opponent_positions = jnp.asarray(self.map_config['opponent_positions'], jnp.int32)
        self.opponent_numbers = jnp.asarray(self.map_config['opponent_numbers'], jnp.int32)
        self.directions = jnp.asarray(DIRECTIONS, jnp.int32)

    def reset(self, rng_key, env_params=None):
        del rng_key, env_params
        visited = jnp.zeros(self.visited_words, jnp.uint32)
        if self.exploration_bonus:
            row, col = self.map_config['agent_position']
            cell = row * self.size + col
            visited = visited.at[cell // 32].set(jnp.uint32(1 << (cell % 32)))
        state = NumberGridState(
            position=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            number=jnp.int32(self.initial_number), alive=jnp.ones(self.num_opponents, dtype=bool),
            step_count=jnp.int32(0), done=jnp.bool_(False),
            won=jnp.bool_(False), lost=jnp.bool_(False),
            visited=visited,
        )
        return state, self._timestep(state, jnp.float32(0), first=True)

    def observation(self, state):
        # Physical state: agent and opponents, elapsed fraction. The episodic
        # exploration memory is used only for reward, not appended to policy input.
        opponents = jnp.concatenate([
            self.opponent_positions.astype(jnp.float32) / (self.size - 1),
            self.opponent_numbers[:, None].astype(jnp.float32) / self.number_scale,
            state.alive[:, None].astype(jnp.float32),
        ], axis=1).reshape(-1)
        return jnp.concatenate([
            state.position.astype(jnp.float32) / (self.size - 1),
            jnp.asarray([state.number / self.number_scale], jnp.float32), opponents,
            jnp.asarray([state.step_count / self.max_steps], jnp.float32),
        ])

    def action_mask(self, state):
        """Only perimeter walls are excluded; occupied and dangerous cells stay legal."""
        # Four boundary checks, shared across the eight directions; avoid a
        # broadcasted (8,2) destination array and per-direction reductions.
        row, col = state.position[0], state.position[1]
        north, south = row > 1, row < self.size - 2
        west, east = col > 1, col < self.size - 2
        return jnp.stack((north, north & east, east, south & east,
                          south, south & west, west, north & west))

    def _timestep(self, state, reward, first=False):
        terminated = state.won | state.lost
        step_type = jnp.where(terminated, StepType.TERMINATED,
                             jnp.where(state.done, StepType.TRUNCATED, StepType.MID))
        if first:
            step_type = StepType.FIRST
        extras = {'solved_episode': state.won}
        if self.mask_walls:
            extras['action_mask'] = self.action_mask(state)
        return TimeStep(
            step_type=step_type, reward=reward,
            discount=jnp.where(terminated, 0.0, 1.0).astype(jnp.float32),
            observation=self.observation(state),
            extras=extras,
        )

    def step(self, state, action, env_params=None):
        del env_params
        # JAX gathers clamp out of range; explicitly treat invalid actions as no-op.
        valid_action = (action >= 0) & (action < 8)
        destination = state.position + self.directions[jnp.clip(action, 0, 7)]
        inside = jnp.all((destination >= 1) & (destination < self.size - 1))
        occupied = jnp.any(state.alive & jnp.all(self.opponent_positions == destination, axis=1))
        position = jnp.where(valid_action & inside & ~occupied, destination, state.position)
        distance = jnp.max(jnp.abs(self.opponent_positions - position), axis=1)
        adjacent = state.alive & (distance == 1)
        lost = jnp.any(adjacent & (self.opponent_numbers >= state.number))
        captured = adjacent & (self.opponent_numbers < state.number) & ~lost
        count = jnp.sum(captured, dtype=jnp.int32)
        alive = state.alive & ~captured
        won = ~jnp.any(alive) & ~lost
        steps = state.step_count + 1
        visited = state.visited
        bonus = jnp.float32(0)
        if self.exploration_bonus:
            cell = position[0] * self.size + position[1]
            word, bit = cell // 32, jnp.left_shift(jnp.uint32(1), (cell % 32).astype(jnp.uint32))
            first_visit = (visited[word] & bit) == 0
            moved = jnp.any(position != state.position)
            bonus = jnp.where(first_visit & moved & ~lost, jnp.float32(self.exploration_bonus), 0.0)
            visited = visited.at[word].set(visited[word] | bit)
        next_state = state.replace(
            position=position, number=state.number + count, alive=alive,
            step_count=steps, done=lost | won | (steps >= self.max_steps),
            won=won, lost=lost, visited=visited,
        )
        reward = jnp.where(lost, -1.0, count.astype(jnp.float32) + 3.0 * won)
        reward = (reward + bonus) - jnp.float32(self.step_cost)
        # Raw environments are absorbing; training adds Stoa's auto-reset wrapper.
        next_state = jax.tree.map(lambda old, new: jnp.where(state.done, old, new), state, next_state)
        reward = jnp.where(state.done, 0.0, reward).astype(jnp.float32)
        return next_state, self._timestep(next_state, reward)

    def observation_space(self, env_params=None):
        return ArraySpace(shape=(self.observation_size,), dtype=jnp.float32, name='full_state')

    def action_space(self, env_params=None):
        return DiscreteSpace(num_values=8, dtype=jnp.int32)

    def state_space(self, env_params=None):
        return DictSpace({
            'position': ArraySpace((2,), jnp.int32),
            'number': ArraySpace((), jnp.int32),
            'alive': ArraySpace((self.num_opponents,), bool),
            'step_count': ArraySpace((), jnp.int32),
            'done': ArraySpace((), bool), 'won': ArraySpace((), bool),
            'lost': ArraySpace((), bool),
            'visited': ArraySpace((self.visited_words,), jnp.uint32),
        })


def wrap_wall_action_mask(env):
    """Apply before auto-reset so reset and terminal observations keep their own mask."""
    return AddActionMaskWrapper(env) if env.mask_walls else env
