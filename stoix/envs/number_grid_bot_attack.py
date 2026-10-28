"""Default Capital's scripted Empire parties (user requests 2026-10-26/27), not the original AI.

In the user's save SaveGame/6789.sg (turn 21) the Empire hero Edwin, a level-2
Pegasus Knight with two Knights and a Marksman, had won 12 battles. The armies he
destroyed start absent (`absent_enemies`); one of their slots holds his party,
which appears on the hero's tile when day `day` begins and attacks at once,
without any movement or withdrawal. From day `raids.first_day`, one randomly
drawn `raids.enemies` party waits at `raids.position`. It never initiates combat;
after destruction a fresh party returns `raids.respawn_days` days later. Only one
can be alive, and an occupied spawn tile delays arrival until it is free. Every
hero victory counts in `bot_hero_victories`; Edwin's pays like an ordinary army
victory. Later parties retain their XP but pay no +1 victory or +2 promotion
rewards. The separate Incubus reward is unchanged.
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
        self.first_day = raids.get('first_day', 0)
        self.respawn_days = raids.get('respawn_days', 7)
        position = raids.get('position', [])
        if not isinstance(self.raid_enemies, list) or any(
                type(i) is not int or not 0 <= i < count for i in self.raid_enemies):
            raise ValueError('bot_attack raids need a list of valid army indices')
        if self.raid_enemies and (
                len(set(self.raid_enemies)) != len(self.raid_enemies)
                or self.enemy in self.raid_enemies or type(self.first_day) is not int or self.first_day < 2
                or type(self.respawn_days) is not int or self.respawn_days < 1):
            raise ValueError('bot_attack raids need distinct armies, a first day of at least 2 '
                             'and positive respawn_days')
        if self.raid_enemies and (
                not isinstance(position, list) or len(position) != 2
                or any(type(v) is not int or not 0 < v < game_map['size']-1 for v in position)
                or position in game_map.get('obstacles', [])):
            raise ValueError('bot_attack raids need a passable interior position')
        self.position = jnp.array(position, jnp.int32)
        self.slots = [self.enemy, *self.raid_enemies]
        if any(i not in absent or i == game_map.get('objective_enemy') for i in self.slots):
            raise ValueError('bot_attack armies must start absent and not be the objective')
        self.mask = jnp.zeros(count, bool).at[jnp.array(self.slots)].set(True)
        self.raids = jnp.array(self.raid_enemies, jnp.int32)
        self.raid_mask = jnp.zeros(count, bool).at[self.raids].set(True)
        self.initial_progress = initial_progress
        # Days left until Edwin and until the next stationary party.
        self.observation_size = 1+bool(self.raid_enemies)

    def _arrive(self, state, enemy, due, position, attack):
        """Restore a fresh party, optionally queuing Edwin's forced encounter."""
        def row(values, fresh):
            return values.at[enemy].set(jnp.where(due, fresh, values[enemy]))
        return state.replace(
            alive=row(state.alive, True), scenario_spawned=row(state.scenario_spawned, True),
            enemy_positions=row(state.enemy_positions, position),
            pending_encounters=row(state.pending_encounters, attack),
            enemy_progress=row(state.enemy_progress, self.initial_progress[enemy]),
            enemy_wounds=row(state.enemy_wounds, 0),
            summon_marks=row(state.summon_marks, False),
            enemy_spell_effects=row(state.enemy_spell_effects, jnp.uint32(0)))

    def spawn(self, state, resting, key):
        """Edwin attacks on REST; reinforcements wait at their fixed location once due."""
        state = self._arrive(state, self.enemy, resting & (state.day == self.day) & ~state.done,
                             state.position, True)
        if self.raid_enemies:
            occupied = (jnp.all(state.position == self.position)
                        | jnp.any(state.alive & jnp.all(state.enemy_positions == self.position, axis=1)))
            due = ((state.bot_respawn_day > 0) & (state.day >= state.bot_respawn_day)
                   & ~state.done & ~jnp.any(state.alive[self.raids]) & ~occupied)
            chosen = self.raids[jax.random.randint(jax.random.fold_in(key, 23), (), 0, len(self.raid_enemies))]
            state = self._arrive(state, chosen, due, self.position, False)
            state = state.replace(bot_respawn_day=jnp.where(due, 0, state.bot_respawn_day))
        return state

    def schedule_respawn(self, before, after):
        """Start the seven-day clock only on destruction, including map magic and summons."""
        if not self.raid_enemies:
            return after
        killed = jnp.any(before.alive[self.raids] & ~after.alive[self.raids])
        return after.replace(bot_respawn_day=jnp.where(
            killed, after.day+self.respawn_days, after.bot_respawn_day))

    def raid_battle(self, state):
        """Battle against a raid party: its victory pays no +1 and its promotions no +2."""
        return (state.enemy >= 0) & self.raid_mask[jnp.maximum(state.enemy, 0)] & ~state.summon_battle

    def after_battle(self, before, after, victory):
        """Count every hero-army victory over a scripted party, for the threshold and charts."""
        after = self.schedule_respawn(before, after)
        won = victory & self.mask[jnp.maximum(before.enemy, 0)] & (before.enemy >= 0) & ~before.summon_battle
        return after.replace(bot_hero_victories=after.bot_hero_victories+won.astype(jnp.int32))

    def observation(self, state):
        values = [jnp.maximum(self.day-state.day, 0)/self.day]
        if self.raid_enemies:
            left = jnp.maximum(state.bot_respawn_day-state.day, 0)
            values.append(left/self.first_day)
        return jnp.asarray(values, jnp.float32)
