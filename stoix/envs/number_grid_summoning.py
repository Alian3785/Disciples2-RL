"""Copy preparation and fixed-slot summons. Sources: docs/UNIT_CATALOG.md."""
import jax
import jax.numpy as jnp
from stoix.envs.number_grid_limits import MAX_HP, cap_stats

from stoix.envs.number_grid_effects import effect_breaks, SHORT
from stoix.envs.number_grid_combat import HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE, MELEE

COPY_ALLY, COPY_ACTIONS = 57, 6
COPIED, SUMMONED = 33, 34
SHOOT, DEFEND, CONTINUE = 8, 14, 17


def linked_summon_hp(hp, escaped, summon_owner):
    """Follow every owner chain with four unrolled ancestor-doubling passes."""
    parent = jnp.where(summon_owner >= 0, summon_owner, jnp.arange(12))
    blocked = (hp <= 0) | escaped
    for _ in range(4):
        blocked |= blocked[parent]
        parent = parent[parent]
    # A retreating root keeps its own HP; only its descendants disappear.
    return jnp.where((summon_owner >= 0) & blocked[jnp.maximum(summon_owner, 0)], 0, hp)


def advance_linked_queue(hp, escaped, phases, priority, round_number, activated,
                         turns, damage, source, immunities, extra_priority,
                         next_actor, hold, max_rounds, immunity_masks, fallback_damage,
                         applied_rounds, removal_rolls, summon_owner):
    """Bounded device queue: a DOT-killed caster removes dependents immediately.

    Process the first candidate inline and continue only after a death. Keep
    the 24-candidate bound; stopped lanes remain inert under batched execution.
    """
    slots = jnp.arange(12)
    def alive(health):
        active = (health > 0) & ~escaped
        return jnp.any(active[:6]) & jnp.any(active[6:])

    def advance(_, carry):
        health, phase, order, round_no, done, left, amounts, owners, actor, losses, visited, stopped = carry
        active = (health > 0) & ~escaped
        wrap = ~jnp.any(active & (phase < 2)) & alive(health) & ~stopped
        round_no += wrap.astype(jnp.int32)
        phase = jnp.where(wrap,0,phase)
        order = jnp.where(wrap,extra_priority,order)
        done = jnp.where(wrap,False,done)
        enabled = ~stopped & alive(health) & (round_no <= max_rounds)
        score = jnp.where(active & (phase < 2),order,-1e9)
        candidate = jnp.argmax(score).astype(jnp.int32)
        actor = jnp.where(enabled,candidate,actor)
        first = enabled & ~done[candidate]
        immune = (immunities[candidate] & immunity_masks) != 0
        tick = first & (left[candidate] > 0) & ~immune
        raw = jnp.where(tick,jnp.where(amounts[candidate] == 0,fallback_damage,amounts[candidate]),0)
        before = jnp.maximum(health[candidate]-(jnp.cumsum(raw)-raw),0)
        tick &= before > 0
        loss = jnp.minimum(raw,before)
        health = health.at[candidate].add(-jnp.sum(loss))
        losses = losses.at[candidate].add(loss)
        expires = tick & ((left[candidate] == SHORT) | effect_breaks(round_no,applied_rounds[candidate],
            removal_rolls[jnp.minimum(round_no-round_number,1),candidate]))
        remaining = jnp.where((first & immune & (before > 0)) | expires,0,left[candidate])
        left = left.at[candidate].set(remaining)
        health = linked_summon_hp(health, escaped, summon_owner)
        dead = health <= 0
        phase = jnp.where(dead,2,phase)
        done |= dead
        live_dot = (left > 0) & ((health > 0) & ~escaped)[:,None]
        left = jnp.where(live_dot,left,0)
        amounts = jnp.where(live_dot,amounts,0)
        owners = jnp.where(live_dot,owners,-1)
        visited |= enabled & (slots == candidate)
        stopped |= ~enabled | ~alive(health) | (health[candidate] > 0)
        return health,phase,order,round_no,done,left,amounts,owners,actor,losses,visited,stopped

    initial = advance(0, (hp,phases,priority,round_number,activated,
        turns,damage,source,next_actor,jnp.zeros_like(damage),jnp.zeros(12,bool),hold | ~alive(hp)))
    def pending(carry):
        count, current = carry
        return (count < 24) & ~current[-1]

    def following(carry):
        count, current = carry
        return count + 1, advance(count, current)

    _, result = jax.lax.while_loop(pending, following, (jnp.int32(1), initial))
    return result[:-1]


class Summoning:
    @property
    def has_linked_summons(self):
        """Battle summons that vanish with their owner: summoner units' creatures and
        orb/talisman summons, whose owner is the hero who used the item."""
        return self.has_summons or self.has_item_summons

    def _copy_targets(self, state):
        ids = self._effective_ids(state)
        return ((state.hp > 0) & ~state.escaped & (self.slots != state.actor)
                & (self.unit_sizes(state) == 1)
                & ~self.progression.copy_forbidden[ids]
                & ~self.progression.doppelgangers[state.unit_ids])

    def _actor_copies(self, state):
        if not self.has_copies:
            return jnp.bool_(False)
        return (self.progression.doppelgangers[self._effective_ids(state)[state.actor]]
                & ~state.imp[state.actor] & jnp.any(self._copy_targets(state)))

    def _summon_targets(self, state):
        same = (self.slots < 6) == (state.actor < 6)
        # A living large fighter occupies both cells of its formation column.
        partner = self.slots + jnp.where(self.slots % 6 < 3, 3, -3)
        blocked = (state.hp[partner] > 0) & ~state.escaped[partner] & (self.unit_sizes(state)[partner] == 2)
        return same & ((state.hp <= 0) | state.escaped) & ~blocked

    def _actor_summons(self, state):
        if not self.has_summons:
            return jnp.int32(0)
        return jnp.where(state.imp[state.actor], 0,
            self.progression.summon_modes[self._effective_ids(state)[state.actor]])

    def _special_enemy_action(self, state, action, random_values):
        if self.has_copies:
            valid = self._copy_targets(state)
            # The reference prefers the largest HP pool, including allies.
            scores = jnp.where(valid, -state.hp+ jnp.tile(random_values, 2)*.5, 1e9)
            target = jnp.argmin(scores)
            copy_action = jnp.where(target < 6, SHOOT+target, COPY_ALLY+target-6)
            action = jnp.where(self._actor_copies(state), copy_action, action)
        if self.has_summons:
            valid = self._summon_targets(state)[6:]
            front = valid & (jnp.arange(6) < 3)
            valid &= jnp.where(jnp.any(front), jnp.arange(6) < 3, True)
            choice = SHOOT+jnp.argmin(jnp.where(valid, random_values, 2.))
            action = jnp.where(self._actor_summons(state) > 0,
                               jnp.where(jnp.any(valid), choice, DEFEND), action)
        return action

    def _copy_unit(self, state, target, enabled):
        actor = state.actor
        form = self._effective_ids(state)[target]
        level = self._effective_levels(state)[target]
        # Copies inherit permanent statistics of the current form, not buffs,
        # shatters or slow. Witch/Fenrir overlays are represented by actual forms.
        form = jnp.where(state.imp[target], self.progression.imp_copy_id,
                         jnp.where(state.fenrir[target], self.progression.fenrir_copy_id, form))
        level = jnp.where(state.imp[target] | state.fenrir[target], self.progression.base_levels[form], level)
        # Only two slots are needed. Avoid recomputing all twelve units' stats
        # and action-dependent traits twice in each vmapped learner step.
        stats = self._copy_form_stats(state,target)
        native_form = self._effective_ids(state)[target]
        traits = self.progression.traits[native_form]
        traits = jnp.where(state.copied[target] & (state.decay_form[target] == 0),state.copy_traits[target],traits)
        traits = traits.at[0].set(jnp.where(state.fenrir[target],MELEE,traits[0]))
        traits = traits.at[1].set(jnp.where(state.fenrir[target],1,traits[1]))
        traits = jnp.where(state.imp[target],jnp.array([MELEE,1,0,0],jnp.uint32),traits)
        maximum = self._copy_form_stats(state,actor)[HP]
        hp = jnp.rint(state.hp[target] * state.hp[actor] / jnp.maximum(maximum, 1))
        hp = jnp.minimum(stats[HP], jnp.maximum(1, hp)).astype(jnp.int32)
        updates = dict(unit_ids=form, unit_levels=level, hp=hp,
                       copied=jnp.bool_(True), copy_stats=stats, copy_traits=traits,
                       copy_secondary=self._secondary_stats(state,target),
                       armor_shreds=jnp.int32(0), weakened=jnp.bool_(False), item_weakened=jnp.bool_(False),
                       primary_override=jnp.int32(-1), preweak_damage=jnp.int32(-1),
                       powerup=jnp.bool_(False), powerup_long=jnp.bool_(False), powerup_layered=jnp.bool_(False),
                       initiative_override=jnp.int32(-1), slow_original=jnp.int32(-1),
                       healer_wards=state.healer_wards[target], ward_native_used=jnp.uint32(0),
                       wards_used=jnp.uint32(0))
        # Snapshot the actual identity when copying, including manually configured squads.
        for name, value in (('native_ids', state.unit_ids[actor]),
                            ('native_levels', state.unit_levels[actor]), ('native_xp', state.unit_xp[actor])):
            updates[name] = jnp.where(state.copied[actor], getattr(state,name)[actor], value)
        return state.replace(**{name: getattr(state,name).at[actor].set(
            jnp.where(enabled, value, getattr(state,name)[actor])) for name,value in updates.items()})

    def _copy_form_stats(self,state,slot):
        """Permanent current-form values, excluding temporary buffs/debuffs."""
        uid, level = state.unit_ids[slot],state.unit_levels[slot]
        raw = self.progression.raw_stats(uid,level)
        values = jnp.minimum(raw,self.progression.stat_caps[uid])
        if self.has_weakening or self.has_powerups or self.has_potion_buffs:
            values = values.at[DAMAGE].set(raw[DAMAGE])
        values = self._potion_stats(state,values,slot=slot,temporary=False)
        values = jnp.where(state.copied[slot],state.copy_stats[slot],values)
        values = jnp.where(state.decay_form[slot] > 0,self.progression.base_stats[state.decay_form[slot]],values)
        values = values.at[HP].set(jnp.where(state.fenrir[slot],275.,values[HP]))
        values = values.at[DAMAGE].set(jnp.where(state.fenrir[slot],90.,values[DAMAGE]))
        values = values.at[INITIATIVE].set(jnp.where(state.fenrir[slot],65.,values[INITIATIVE]))
        big = self.progression.sizes[self._effective_ids(state)[slot]] == 2
        values = values.at[DAMAGE].set(jnp.where(state.imp[slot],jnp.where(big,30.,20.),values[DAMAGE]))
        values = values.at[ACCURACY].set(jnp.where(state.imp[slot],jnp.where(big,70.,80.),values[ACCURACY]))
        values = values.at[ARMOR].set(jnp.where(state.imp[slot],0.,values[ARMOR]))
        return cap_stats(values.at[INITIATIVE].set(jnp.where(state.imp[slot],jnp.where(big,50.,30.),values[INITIATIVE])))

    def _spawn_units(self, state, target, enabled, rolls):
        rules, actor = self.progression, state.actor
        mode = self._actor_summons(state)
        uid = self._effective_ids(state)[actor]
        free = self._summon_targets(state)
        front = self.slots % 6 < 3
        pair = self.slots+jnp.where(front, 3, -3)
        full = free & free[pair]
        large = full & front & (mode == 2)
        selected = free & jnp.where(mode == 1, self.slots == target, ~full | front) & enabled
        pool = jnp.where(large[:,None], rules.summon_large[uid], rules.summon_small[uid])
        count = jnp.sum(pool > 0, axis=1)
        index = jnp.minimum((rolls*count).astype(jnp.int32), jnp.maximum(count-1,0))
        ids = jnp.take_along_axis(pool,index[:,None],axis=1)[:,0]
        selected &= ids != 0
        levels = rules.base_levels[ids]
        immediate = selected & large & rules.immediate_dragons[uid]
        # Every replacement is a new fighter: clear all local combat layers.
        empty = self._empty_summon_fields(state)
        updates = {name: jnp.where(selected.reshape((12,)+(1,)*(value.ndim-1)), value, getattr(state,name))
                   for name,value in empty.items()}
        # A new unit in the same cell must not inherit the former caster's
        # one-target DOT lock. The victim keeps the already applied damage.
        for kind in ('poison','burn','water'):
            source = updates[kind+'_source']
            updates[kind+'_source'] = jnp.where((source >= 0) & selected[jnp.maximum(source,0)],-1,source)
        updates.update(unit_ids=jnp.where(selected,ids,state.unit_ids),
            unit_levels=jnp.where(selected,levels,state.unit_levels),
            unit_xp=jnp.where(selected,0,state.unit_xp),
            hp=jnp.where(selected,rules.stats(ids,levels)[:,HP].astype(jnp.int32),state.hp),
            summon_owner=jnp.where(selected,actor,state.summon_owner),
            turn_phase=jnp.where(selected,jnp.where(immediate,0,2),state.turn_phase),
            activation_done=jnp.where(selected,~immediate,state.activation_done),
            priority=jnp.where(selected,jnp.where(immediate,rules.base_stats[ids,INITIATIVE],0.),state.priority))
        updates.update(self._summon_backups(state, selected))
        return state.replace(**updates), selected

    def _summon_backups(self, state, selected):
        """Keep the permanent fighter once, even across repeated summons.

        Reference _mark_unit_escaped stores a separate snapshot and clears the
        cell. Save its HP in the native form using the current battle-end rules.
        """
        first = selected & (state.summon_owner < 0)
        save_identity = first & ~state.copied
        updates = {backup: jnp.where(save_identity, getattr(state, current), getattr(state, backup))
                   for backup, current in (('native_ids','unit_ids'), ('native_levels','unit_levels'),
                                            ('native_xp','unit_xp'))}
        hp = state.hp
        if self.has_wights or self.has_wolf_lord:
            natural = self._natural_stats(state)[:, HP]
            if self.has_copies:
                natural = jnp.where(state.copied, state.copy_stats[:,HP], natural)
            if self.has_wights:
                maximum = jnp.where(state.fenrir, 275., natural)
                previous = self.progression.base_stats[state.decay_form, HP]
                restored = jnp.rint(hp / jnp.maximum(previous, 1) * maximum).astype(jnp.int32)
                restored = jnp.where(hp > 0, jnp.maximum(1, restored), 0)
                hp = jnp.where(state.decay_form > 0, restored, hp)
            if self.has_wolf_lord:
                restored = jnp.rint(hp / 275. * natural).astype(jnp.int32)
                restored = jnp.where(hp > 0, jnp.maximum(1, restored), 0)
                hp = jnp.where(state.fenrir, restored, hp)
        if self.has_copies:
            own_stats = self.progression.stats(state.native_ids,state.native_levels)
            own_hp = jnp.minimum(self._potion_stats(state,own_stats,temporary=False)[:,HP], MAX_HP)
            restored = jnp.rint(hp*own_hp/jnp.maximum(state.copy_stats[:,HP],1)).astype(jnp.int32)
            restored = jnp.where(hp > 0, jnp.maximum(1, restored), 0)
            hp = jnp.where(state.copied, restored, hp)
        hp = jnp.minimum(hp, MAX_HP)
        updates['summon_saved_hp'] = jnp.where(first, jnp.where(state.escaped, hp, 0), state.summon_saved_hp)
        return updates

    def _empty_summon_fields(self, state):
        zero = ('effect_rounds','transform_long','powerup_long','saved_powerup_long','defended','retreating','escaped','copied','fenrir','imp','imp_wards','decay_form',
            'decay_shreds','armor_shreds','feared','weakened','saved_weakened','item_weakened',
            'saved_item_weakened','lycanthrope','paralyzed','long_paralyzed',
            'waited','bonus_turns','battle_revived','revival_xp_cutoff','powerup','powerup_layered',
            'saved_powerup','saved_powerup_layered','healer_wards','saved_healer_wards',
            'ward_native_used','saved_ward_native_used','wards_used')
        negative = ('disable_source','initiative_override','saved_initiative_override','slow_original','primary_override',
            'preweak_damage','saved_primary_override','saved_preweak_damage')
        result = {name: jnp.zeros_like(getattr(state,name)) for name in zero}
        result.update({name: jnp.full_like(getattr(state,name),-1) for name in negative})
        for kind in ('poison','burn','water'):
            result[kind+'_mode'] = jnp.zeros(12,jnp.int32)
            result[kind+'_damage'] = jnp.zeros(12,jnp.int32)
            result[kind+'_source'] = jnp.full(12,-1,jnp.int32)
        return result

    def _special_action(self, state, action, random_values):
        selected = (action >= SHOOT) & (action < DEFEND)
        own = action >= COPY_ALLY
        slot = jnp.clip(jnp.where(own,action-COPY_ALLY,action-SHOOT),0,5)
        target = slot+jnp.where((state.actor < 6) ^ own,6,0)
        copy = self._actor_copies(state) & (selected | own) & self._copy_targets(state)[target]
        summon = (self._actor_summons(state) > 0) & selected
        if self.has_copies:
            state = self._copy_unit(state,target,copy)
        if self.has_summons:
            target = slot+jnp.where(state.actor < 6,0,6)
            state, spawned = self._spawn_units(state,target,summon,random_values[68:80])
            summon &= jnp.any(spawned)
        return state,jnp.where(copy | summon,DEFEND,action),jnp.where(copy,COPIED,jnp.where(summon,SUMMONED,-1))

    def _prepare_copies(self, state, action, key, random_values):
        actor = state.actor
        action = jnp.where(actor >= 6,self._special_enemy_action(state,DEFEND,random_values[30:36]),action)
        prepared, _, event = self._special_action(state,action,random_values)
        pending = state.preparation.at[actor].set(False)
        remaining = jnp.any(pending & (prepared.hp > 0))
        priority = self._round_priority(random_values,state.enemy,prepared)
        next_actor = jnp.argmax(jnp.where(jnp.where(remaining,pending,prepared.hp > 0),priority,-1e9))
        # Copy preparation is a unit activation; only the outer decision spends max_steps.
        return prepared.replace(battle_key=key,preparation=pending,round=jnp.where(remaining,0,1),
            actor=next_actor.astype(jnp.int32),priority=priority,
            activation_done=jnp.zeros(12,bool).at[next_actor].set(~remaining),
            last_event=jnp.where(event >= 0,event,4).astype(jnp.int32),last_actor=actor,
            last_target=jnp.int32(-1),last_damage=jnp.int32(0),
            player_turns=state.player_turns+(actor < 6).astype(jnp.int32),
            enemy_turns=state.enemy_turns+(actor >= 6).astype(jnp.int32),
            battle_steps=state.battle_steps+1),jnp.float32(0)

    def _linked_summon_hp(self, state, hp):
        if not self.has_linked_summons:
            return hp
        return linked_summon_hp(hp, state.escaped, state.summon_owner)

    def _restore_roster(self, state, hp, ended):
        own_stats = self.progression.stats(state.native_ids,state.native_levels)
        own_hp = jnp.minimum(self._potion_stats(state,own_stats,temporary=False)[:,HP], MAX_HP)
        restored = jnp.rint(hp*own_hp/jnp.maximum(state.copy_stats[:,HP],1)).astype(jnp.int32)
        restored = jnp.where(hp > 0,jnp.maximum(1,restored),0)
        hp = jnp.where(ended & state.copied,restored,hp)
        summoned = ended & (state.summon_owner >= 0)
        hp = jnp.where(summoned,state.summon_saved_hp,hp)
        restore = ended & (state.copied | (state.summon_owner >= 0))
        state = state.replace(unit_ids=jnp.where(restore,state.native_ids,state.unit_ids),
            unit_levels=jnp.where(restore,state.native_levels,state.unit_levels),
            unit_xp=jnp.where(restore,state.native_xp,state.unit_xp),
            escaped=jnp.where(summoned,state.summon_saved_hp > 0,state.escaped),
            summon_saved_hp=jnp.where(ended,0,state.summon_saved_hp),
            copied=state.copied & ~ended,
            summon_owner=jnp.where(ended,-1,state.summon_owner))
        return state,jnp.minimum(hp, MAX_HP)
