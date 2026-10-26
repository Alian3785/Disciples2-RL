"""Default Capital's minimal scripted Empire attack (user request 2026-10-26), not the original AI.

In the user's save SaveGame/6789.sg (turn 21) the Empire hero Edwin, a level-2
Pegasus Knight with two Knights and a Marksman, had won 12 battles. The armies he
destroyed start absent (`absent_enemies`); one of their slots holds his party,
which appears on the hero's tile when day `day` begins and attacks at once,
without any movement. The player cannot withdraw (`NumberGrid._can_retreat`): a
loss ends the episode like any lost battle, a victory is an ordinary army victory.
Sources: docs/DEFAULT_MAP.md (Скриптовый бот Империи).
"""
import jax.numpy as jnp


class BotAttackRules:
    def __init__(self, game_map):
        spec = game_map.get('bot_attack')
        self.enabled = bool(spec)
        self.observation_size = 0
        if not self.enabled:
            return
        self.enemy, self.day = spec.get('enemy'), spec.get('day')
        if (type(self.enemy) is not int or not 0 <= self.enemy < len(game_map['opponent_positions'])
                or type(self.day) is not int or self.day < 2):
            raise ValueError('bot_attack needs a valid army index and an attack day of at least 2')
        if self.enemy not in game_map.get('absent_enemies', []) or self.enemy == game_map.get('objective_enemy'):
            raise ValueError('The bot_attack army must be absent until its attack and not be the objective')
        # Days left until the attack, so the policy can prepare for it.
        self.observation_size = 1

    def spawn(self, state, resting):
        """On the REST that begins the attack day, put the party on the hero and queue its attack."""
        due = resting & (state.day == self.day) & ~state.done
        enemy = self.enemy
        return state.replace(
            alive=state.alive.at[enemy].set(state.alive[enemy] | due),
            scenario_spawned=state.scenario_spawned.at[enemy].set(state.scenario_spawned[enemy] | due),
            enemy_positions=state.enemy_positions.at[enemy].set(
                jnp.where(due, state.position, state.enemy_positions[enemy])),
            pending_encounters=state.pending_encounters.at[enemy].set(state.pending_encounters[enemy] | due))

    def observation(self, state):
        return jnp.asarray([jnp.maximum(self.day-state.day, 0)/self.day], jnp.float32)
