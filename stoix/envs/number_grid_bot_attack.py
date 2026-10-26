"""Default Capital's minimal scripted Empire attacks (user requests 2026-10-26), not the original AI.

In the user's save SaveGame/6789.sg (turn 21) the Empire hero Edwin, a level-2
Pegasus Knight with two Knights and a Marksman, had won 12 battles. The armies he
destroyed start absent (`absent_enemies`); one of their slots holds his party,
which appears on the hero's tile when day `day` begins and attacks at once,
without any movement. From day `raids.first_day` on, every `raids.interval` days,
one of the `raids.enemies` parties (the user's two Empire screenshots), drawn
anew each time, attacks the same way at full strength. The player cannot withdraw
(`NumberGrid._can_retreat`): a loss ends the episode like any lost battle. Every
victory counts in `bot_hero_victories`; Edwin's pays like an ordinary army victory,
a raid's pays nothing (no +1, no +2 per promotion) but keeps its XP, so raids
cannot be farmed.
Sources: docs/DEFAULT_MAP.md (Скриптовый бот Империи).
"""
import jax
import jax.numpy as jnp


class BotAttackRules:
    def __init__(self, game_map, initial_progress=None):
        spec = game_map.get('bot_attack')
        self.enabled = bool(spec)
        self.observation_size = 0
        if not self.enabled:
            return
        count = len(game_map['opponent_positions'])
        absent = game_map.get('absent_enemies', [])
        self.enemy, self.day = spec.get('enemy'), spec.get('day')
        if (type(self.enemy) is not int or not 0 <= self.enemy < count
                or type(self.day) is not int or self.day < 2):
            raise ValueError('bot_attack needs a valid army index and an attack day of at least 2')
        raids = spec.get('raids', {})
        self.raid_enemies = raids.get('enemies', [])
        self.first_day, self.interval = raids.get('first_day', 0), raids.get('interval', 1)
        if self.raid_enemies and (
                not isinstance(self.raid_enemies, list) or len(set(self.raid_enemies)) != len(self.raid_enemies)
                or any(type(i) is not int or not 0 <= i < count for i in self.raid_enemies)
                or self.enemy in self.raid_enemies or type(self.first_day) is not int or self.first_day < 2
                or type(self.interval) is not int or self.interval < 1
                or (self.day >= self.first_day and (self.day-self.first_day) % self.interval == 0)):
            raise ValueError('bot_attack raids need distinct armies, a first day of at least 2, a positive '
                             'interval and no day shared with the first attack')
        self.slots = [self.enemy, *self.raid_enemies]
        if any(i not in absent or i == game_map.get('objective_enemy') for i in self.slots):
            raise ValueError('bot_attack armies must be absent until their attacks and not be the objective')
        self.mask = jnp.zeros(count, bool).at[jnp.array(self.slots)].set(True)
        self.raids = jnp.array(self.raid_enemies, jnp.int32)
        self.raid_mask = jnp.zeros(count, bool).at[self.raids].set(True)
        self.initial_progress = initial_progress
        # Days left until the first attack and until the next raid, so the policy can prepare.
        self.observation_size = 1+bool(self.raid_enemies)

    def _arrive(self, state, enemy, due):
        """Put a fresh party on the hero's tile and queue its attack."""
        def row(values, fresh):
            return values.at[enemy].set(jnp.where(due, fresh, values[enemy]))
        return state.replace(
            alive=row(state.alive, True), scenario_spawned=row(state.scenario_spawned, True),
            enemy_positions=row(state.enemy_positions, state.position),
            pending_encounters=row(state.pending_encounters, True),
            enemy_progress=row(state.enemy_progress, self.initial_progress[enemy]),
            enemy_wounds=row(state.enemy_wounds, 0),
            enemy_spell_effects=row(state.enemy_spell_effects, jnp.uint32(0)))

    def spawn(self, state, resting, key):
        """On the REST that begins an attack day, bring Edwin's party or a randomly drawn raid."""
        state = self._arrive(state, self.enemy, resting & (state.day == self.day) & ~state.done)
        if self.raid_enemies:
            due = (resting & (state.day >= self.first_day) & ~state.done
                   & ((state.day-self.first_day) % self.interval == 0))
            chosen = self.raids[jax.random.randint(jax.random.fold_in(key, 23), (), 0, len(self.raid_enemies))]
            state = self._arrive(state, chosen, due)
        return state

    def raid_battle(self, state):
        """Battle against a raid party: its victory pays no +1 and its promotions no +2."""
        return (state.enemy >= 0) & self.raid_mask[jnp.maximum(state.enemy, 0)] & ~state.summon_battle

    def after_battle(self, before, after, victory):
        """Count every hero-army victory over a scripted party, for the threshold and charts."""
        won = victory & self.mask[jnp.maximum(before.enemy, 0)] & (before.enemy >= 0) & ~before.summon_battle
        return after.replace(bot_hero_victories=after.bot_hero_victories+won.astype(jnp.int32))

    def observation(self, state):
        values = [jnp.maximum(self.day-state.day, 0)/self.day]
        if self.raid_enemies:
            left = jnp.where(state.day <= self.first_day, self.first_day-state.day,
                             (self.first_day-state.day) % self.interval)
            values.append(left/self.first_day)
        return jnp.asarray(values, jnp.float32)
