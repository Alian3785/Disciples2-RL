"""Target choice of the original game's battle AI for the scripted opponent.

Ported from the installed Steam Discipl2.exe by static analysis (2026-11-14): main routine
0x5cec33, class dispatch 0x5d0ee5 and the target finders it calls. The AI never waits. Retreat,
item use and the Wolf Lord's transformation are not ported. Rules and addresses:
docs/COMBAT.md#enemy-ai.

Every rule is a list of preference groups over the six cells of one side, a value inside the
group and one of two tie rules. The acting unit's rule is assembled with elementwise selects
and resolved by one reduction pass (`select`); rules for second attacks no unit of the map
has are not compiled. This keeps the battle step close to the cost of a plain lowest-HP rule.
"""
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid_combat import HP, DAMAGE, ARMOR, INITIATIVE, MELEE, AREA, HEALER, armor_with_defend

# LattC.dbf attack classes.
(L_DAMAGE, L_DRAIN, L_PARALYZE, L_HEAL, L_FEAR, L_BOOST, L_PETRIFY, L_LOWER_DAMAGE, L_LOWER_INITIATIVE, L_POISON,
 L_FROSTBITE, L_REVIVE, L_DRAIN_OVERFLOW, L_CURE, L_SUMMON, L_DRAIN_LEVEL, L_GIVE_ATTACK, L_DOPPELGANGER,
 L_TRANSFORM_SELF, L_TRANSFORM_OTHER, L_BLISTER, L_BESTOW_WARDS, L_SHATTER) = (
    1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
ADJACENT = 3  # LAttR.dbf: 1 all, 2 any, 3 adjacent
HITTING = (L_DAMAGE, L_DRAIN, L_DRAIN_OVERFLOW)
# 0x5d024d: attacks that count as dealing damage for "can this side still hurt the other".
HARMFUL = HITTING+(L_SUMMON, L_POISON, L_BLISTER, L_FROSTBITE)
# 0x5cf96b: how much more a shooter values a rear-line specialist, by either of its attacks.
WEIGHTS = ((L_HEAL, 2), (L_PARALYZE, 8), (L_PETRIFY, 8), (L_SUMMON, 10), (L_TRANSFORM_OTHER, 9), (L_GIVE_ATTACK, 3))
# Formation cell of each slot in the original numbering (even cells are the front row). The AI
# walks its sorted target list in this order, which decides ties.
CELL = (0, 2, 4, 1, 3, 5)
WARD_FACTOR = .7  # 0x7b29d0: an unused ward makes a target less attractive
GROUPS, SPAN = 8, 1 << 22  # preference groups 0..8 (0 first) and the value range inside a group
# Packed attributes of a unit type: attack class, second class, then flags and the shooter's weight.
CLOSE, PLAIN, MASS_CURE, HITS, HARMS, WEIGHT = 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 15
IMP = L_DAMAGE | CLOSE | PLAIN | HITS | HARMS | 1 << WEIGHT  # a transformed unit is a small melee fighter


def select(eligible, group, value, flips, rank, coins):
    """Winner of the best preference group: the highest value, then the tie rule.

    Strict comparisons in the original keep the earliest cell among equals. Where it flips a
    coin (0x5d057e), each later equal candidate replaces the current one on random(2) == 1: the
    last one whose coin came up wins and the first stays when none did, so k equal candidates
    get 1/2^(k-1), 1/2^(k-1), ..., 1/4, 1/2. Returns (index, found).
    """
    value = jnp.clip(value.astype(jnp.int32), 1-SPAN//2, SPAN//2-1)
    key = jnp.where(eligible, (GROUPS-group)*SPAN+value+SPAN//2, -1)
    best = jnp.max(key)
    equal = eligible & (key == best)
    lucky = equal & flips & coins
    index = jnp.where(jnp.any(lucky), jnp.argmax(jnp.where(lucky, rank, -1)),
                      jnp.argmin(jnp.where(equal, rank, 99)))
    return index.astype(jnp.int32), best >= 0


class EnemyAI:
    def __init__(self, progression, shoot, defend):
        rows = progression.rows

        def number(row, attack, name):
            value = ((row.get('game_data') or {}).get(attack) or {}).get(name)
            return int(value) if str(value or '').strip() else 0
        primary = [number(r, 'primary_attack', 'CLASS') for r in rows]
        secondary = [number(r, 'secondary_attack', 'CLASS') for r in rows]
        close = [number(r, 'primary_attack', 'REACH') == ADJACENT for r in rows]
        big = np.asarray(progression.sizes) != 1
        cures = np.asarray(progression.mass_cures)
        packed = []
        for i, (first, second) in enumerate(zip(primary, secondary)):
            weight = next((w for attack_class, w in WEIGHTS if attack_class in (first, second)), 1)
            # 0x5cf96b: large units, melee fighters and damage boosters are judged by their HP.
            plain = bool(big[i]) or close[i] or first == L_BOOST
            packed.append(first | second << 5 | CLOSE*close[i] | PLAIN*plain | MASS_CURE*bool(cures[i])
                          | HITS*(first in HITTING) | HARMS*(first in HARMFUL) | weight << WEIGHT)
        self.form = jnp.array(packed, jnp.int32)  # one lookup per step for all of the above
        self.class_immune = jnp.array([sum(1 << int(c) for c in r.get('attack_class_immunities', []))
                                       for r in rows], jnp.uint32)
        # XP_KILLED of the unit type itself, for the Doppelganger's choice (0x5d1c9e).
        self.base_xp = progression.kill_levels[progression.base_levels, jnp.arange(len(rows))]
        # Second attacks that can occur in this map's battles: its units and the forms a Wight
        # lowers them to. Rules for the others are left out of the compiled step.
        present, lower = set(progression.used_ids), np.asarray(progression.lower_forms)
        pending = list(present)
        while pending:
            form = int(lower[pending.pop()])
            if form and form not in present:
                present.add(form)
                pending.append(form)
        seconds = {secondary[i] for i in present if primary[i] in HITTING}
        self.has_periodic = bool(seconds & {L_POISON, L_FROSTBITE, L_BLISTER})
        self.has_stun = (bool(seconds & {L_PARALYZE, L_PETRIFY})
                         or any(primary[i] in (L_PARALYZE, L_PETRIFY) for i in present))
        self.has_sap = L_DRAIN_LEVEL in seconds
        self.has_crack = L_SHATTER in seconds
        self.has_scare = L_FEAR in seconds
        self.cell = jnp.array(CELL, jnp.int32)
        self.shoot, self.defend = shoot, defend

    def _forms(self, env, state):
        """Unit types and packed attributes of every slot's current form."""
        ids = env._effective_ids(state)
        return ids, jnp.where(state.imp, IMP, self.form[ids])

    def action(self, env, state, random_values):
        """The acting enemy's command: SHOOT+target or DEFEND."""
        shoot, defend, cell = self.shoot, self.defend, self.cell
        progression, actor = env.progression, state.actor
        ids, form = self._forms(env, state)
        primary, hitters = form & 31, (form & HITS) != 0
        levels = env._effective_levels(state)
        stats, traits = env._combat_stats(state), env._combat_traits(state)
        hp = state.hp
        alive = (hp > 0) & ~state.escaped
        value = progression.kill_levels[levels, ids]  # XP_KILLED at the unit's level
        maximum = stats[:, HP].astype(jnp.int32)
        armor = armor_with_defend(stats[:, ARMOR], state.defended).astype(jnp.int32)
        effective = hp+hp*armor//100  # 0x5cf8fe
        leaving, disabled = state.retreating, state.paralyzed | state.long_paralyzed
        changed = state.imp | state.lycanthrope
        summoned = state.summon_owner >= 0
        if env.spell_casting_enabled:
            summoned |= state.summon_battle & (env.slots < 6)
        coins = random_values[:6] < .5
        blue, red = slice(0, 6), slice(6, 12)
        tiers = env.all_melee_tiers
        stunners = env.has_paralysis or self.has_stun
        scarers = env.has_fear or self.has_scare

        def able(own, other):
            """0x5cfe77: alive, staying, and with a target its primary attack is not barred from."""
            mine, theirs, roles = alive[own], alive[other], traits[own, 0]
            bits = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[own, 1], 1)-1)
            nearest = jnp.min(jnp.where(theirs[None, :], tiers, 4), axis=1)
            blocked = (jnp.arange(6) >= 3) & jnp.any(mine[:3])
            near = theirs[None, :] & (tiers == nearest[:, None]) & ~blocked[:, None]
            targets = jnp.where((roles == MELEE)[:, None], near, theirs[None, :])
            hurts = jnp.any(targets & ((traits[other, 2][None, :] & bits[:, None]) == 0), axis=1)
            return mine & ~leaving[own] & (hurts | (roles == HEALER))
        foe_able = able(blue, red) if stunners or scarers else None
        mate_able = able(red, blue) if stunners or env.has_alchemists else None
        # 0x5cfc17: paralysis is pointless when nobody on the own side can deal damage.
        side_harms = jnp.any(mate_able & ((form[red] & HARMS) != 0)) if stunners else None

        role = traits[actor, 0]
        source = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[actor, 1], 1)-1)
        damage = stats[actor, DAMAGE].astype(jnp.int32)
        own = form[actor]
        own_class, own_second, hitter = own & 31, (own >> 5) & 31, (own & HITS) != 0
        foes, foe_hp, foe_max, foe_form = alive[blue], hp[blue], maximum[blue], form[blue]
        foe_value, foe_effective = value[blue], effective[blue]
        foe_leaving, foe_disabled, foe_summoned = leaving[blue], disabled[blue], summoned[blue]
        immune = (traits[blue, 2] & source) != 0
        warded = (traits[blue, 3] & ~state.wards_used[blue] & source) != 0

        # 0x5cf96b: rear-line specialists by XP and role, everyone else by how close to death.
        score = jnp.where((foe_form & PLAIN) != 0,
                          jnp.where(foe_effective <= damage, 10000+foe_effective, 10000-foe_effective),
                          10000+foe_value*((foe_form >> WEIGHT) & 15))
        score = jnp.where(warded, (score*jnp.float32(WARD_FACTOR)).astype(jnp.int32), score)

        melee, single = role == MELEE, role != AREA
        reachable = jnp.where(melee, env._melee_targets(state, enemy_side=True), foes)
        exposed = reachable & ~immune
        witch = env._actor_is_witch(state) & single if env.has_witches else jnp.bool_(False)
        ghost = (env._actor_ghost_mode(state) > 0) & single if env.has_paralysis else jnp.bool_(False)
        scarer = env._actor_is_fear_caster(state) & single if env.has_fear else jnp.bool_(False)

        # A unit that only deals damage: 0x5d03ab takes the lowest effective HP in melee, summoned
        # creatures as a last resort; 0x5d059c takes the highest score for a shooter.
        ok = exposed
        group = jnp.where(melee & foe_summoned, 5, 4)
        worth = jnp.where(melee, -foe_effective, score)
        flips = melee & foe_summoned
        # 0x5cf317/0x5cf569: with a special second attack the target follows that attack first.
        if self.has_periodic:  # 0x5d2a42: to a target that does not carry the effect yet
            carried = jnp.where(own_second == L_POISON, state.poison_mode[blue] > 0, jnp.where(
                own_second == L_FROSTBITE, state.water_mode[blue] > 0, state.burn_mode[blue] > 0))
            periodic = (own_second == L_POISON) | (own_second == L_FROSTBITE) | (own_second == L_BLISTER)
            spread = hitter & periodic & exposed & ~carried
            group, worth, flips = jnp.where(spread, 1, group), jnp.where(spread, score, worth), flips & ~spread
        if self.has_sap:  # 0x5d27a9: the highest level above the first
            ranked = reachable & (levels[blue] > 1)
            sap = (hitter & (own_second == L_DRAIN_LEVEL) & ranked
                   & (levels[blue] == jnp.max(jnp.where(ranked, levels[blue], 0))))
            ok |= sap
            group = jnp.where(sap, jnp.where(foe_leaving, 2, 1), group)
            worth, flips = jnp.where(sap, foe_value, worth), flips | sap
        if self.has_crack:  # 0x5d15bb: the thickest armour
            crack = hitter & (own_second == L_SHATTER) & reachable & ~foe_leaving & ~foe_disabled
            ok |= crack
            group = jnp.where(crack, jnp.where(foe_hp*4 >= foe_max, 1, 2), group)
            worth, flips = jnp.where(crack, stats[blue, ARMOR].astype(jnp.int32), worth), flips | crack
        if stunners:  # 0x5d1aca, also petrify: a unit that can act, then the others, then the leaving
            using = ghost | (hitter & ((own_second == L_PARALYZE) | (own_second == L_PETRIFY)))
            stun = using & jnp.where(ghost, foes, reachable) & ~foe_disabled & side_harms
            ok = jnp.where(ghost, stun, ok | stun)
            group = jnp.where(stun, jnp.where(foe_leaving, 3, jnp.where(foe_able, 1, 2)), group)
            worth = jnp.where(stun, jnp.where(foe_leaving, cell, foe_value), worth)
            flips = jnp.where(stun, ~foe_leaving & ~foe_able, flips)
        if scarers:  # 0x5d290e
            using = scarer | (hitter & (own_second == L_FEAR))
            scare = using & jnp.where(scarer, foes, reachable) & ~foe_leaving
            ok = jnp.where(scarer, scare, ok | scare)
            group = jnp.where(scare, jnp.where(foe_able, 1, 2), group)
            worth, flips = jnp.where(scare, foe_value, worth), flips | scare
        if env.has_witches:  # 0x5d261e; capital guardians cannot be transformed
            turn = witch & foes & ~changed[blue] & ~progression.capital_guards[state.unit_ids[blue]]
            ok = jnp.where(witch, turn, ok)
            group = jnp.where(turn, jnp.where((foe_hp*5 > foe_max) & ~foe_leaving, 1, 2), group)
            worth, flips = jnp.where(turn, foe_value, worth), flips | turn
        # 0x5cf743: a target this blow kills outright comes before everything else.
        kill = (exposed & ~foe_summoned & (foe_hp <= damage) & (foe_effective <= damage) & (damage > 0)
                & ~(witch | ghost | scarer))
        group, worth, flips = jnp.where(kill, 0, group), jnp.where(kill, score, worth), flips & ~kill
        # An attack on everyone needs no choice, but is pointless against a wholly immune party.
        class_bit = jnp.left_shift(jnp.uint32(1), own_class.astype(jnp.uint32))
        barred = immune | ((self.class_immune[ids[blue]] & class_bit) != 0)
        everyone = jnp.any(foes & ~barred)
        if stunners:
            everyone &= ~((own_class == L_PARALYZE) | (own_class == L_PETRIFY)) | side_harms
        ok = jnp.where(single, ok, foes & everyone)
        group, worth, flips = jnp.where(single, group, 4), jnp.where(single, worth, -cell), flips & single

        party, idle = jnp.bool_(False), jnp.bool_(False)  # a spell on the whole own side takes no target
        if env.has_healers or env.has_patriarchs or env.has_alchemists:
            mates, mate_hp, mate_max, mate_value = alive[red], hp[red], maximum[red], value[red]
            aside = leaving[red] | summoned[red]
            steady = mates & ~leaving[red] & ~disabled[red]
            healthy = mate_hp*4 >= mate_max
            wounded = mates & (mate_hp < mate_max)
            # 0x5d0bb6: the lowest share of HP, compared exactly; the leaving and the summoned last.
            lower = jnp.sum(wounded[None, :] & (mate_hp[None, :]*mate_max[:, None]
                                                < mate_hp[:, None]*mate_max[None, :]), axis=1)
            support = jnp.bool_(False)
            help_ok, help_group, help_worth = wounded, jnp.where(aside, 2, 1), -lower
            if env.has_healers:
                support = healer = role == HEALER
                if env.has_healer_wards:  # 0x5d1448: the most valuable ally that is not nearly dead
                    warder = env._effect_value(state, 'healer_ward_elements')[actor] > 0
                    help_ok = jnp.where(warder, steady, help_ok)
                    help_group = jnp.where(warder, jnp.where(healthy, 1, 2), help_group)
                    help_worth = jnp.where(warder, mate_value, help_worth)
                else:
                    warder = jnp.bool_(False)
                # 0x5d1960: what a cure removes first.
                ailment = jnp.where(changed[red], 1, jnp.where(disabled[red], 2, jnp.where(
                    state.poison_mode[red] > 0, 3, jnp.where(state.water_mode[red] > 0, 4, jnp.where(
                        state.burn_mode[red] > 0, 5, 9)))))
                ailing = mates & (ailment < 9)
                work = jnp.any(wounded) | warder | (((own & MASS_CURE) != 0) & ~state.imp[actor] & jnp.any(ailing))
                mass = healer & env._actor_mass_healer(state)
                if env.has_powerups:  # 0x5d127e: an ally that deals damage; 0x5d1762: else a cure
                    power = env._actor_power_factor(state) > 0
                    boost = steady & hitters[red] & (env.slots[red] != actor)
                    sick = ailing & env._actor_power_cures(state) & ~boost
                    help_ok = jnp.where(power, boost | sick, help_ok)
                    help_group = jnp.where(power, jnp.where(boost, jnp.where(healthy, 1, 2),
                                                            jnp.where(aside, 4, 3)), help_group)
                    help_worth = jnp.where(power, jnp.where(
                        boost, mate_value, (5-ailment)*20000+jnp.minimum(mate_value, 19999)), help_worth)
                    support, mass = support | power, mass & ~power
                party, idle = mass & work, mass & ~work
            if env.has_patriarchs:  # a fallen ally first, then the heal rule
                elder = env._actor_is_patriarch(state)
                valid = env._patriarch_targets(state)[red]
                fallen = valid & (mate_hp <= 0)
                help_ok = jnp.where(elder, fallen | (valid & wounded), help_ok)
                help_group = jnp.where(elder, jnp.where(fallen, 0, jnp.where(aside, 2, 1)), help_group)
                help_worth = jnp.where(elder, jnp.where(fallen, 0, -lower), help_worth)
                support, party, idle = support | elder, party & ~elder, idle & ~elder
            if env.has_alchemists:
                # 0x5d2418: a unit that can strike again, hitters first. The environment's own
                # conditions for a useful extra turn stay.
                raw = env._combat_stats(state, cap=False)[red]
                current = jnp.where(state.turn_phase[red] == 0, state.priority[red], 0.)
                brewer = env._alchemist_units(state)[actor]
                valid = (env._alchemist_targets(state)[red] & (current < raw[:, INITIATIVE])
                         & (raw[:, DAMAGE] > 0) & ~disabled[red] & mate_able)
                help_ok = jnp.where(brewer, valid, help_ok)
                help_group = jnp.where(brewer, jnp.where(hitters[red], 1, 2), help_group)
                help_worth = jnp.where(brewer, mate_value, help_worth)
                support, party, idle = support | brewer, party & ~brewer, idle & ~brewer
            ok, group = jnp.where(support, help_ok, ok), jnp.where(support, help_group, group)
            worth, flips = jnp.where(support, help_worth, worth), flips | support

        target, found = select(ok, group, worth, flips, cell, coins)
        return jnp.where(party, shoot, jnp.where(found & ~idle, shoot+target, defend))

    def copy_target(self, env, state, valid, random_values):
        """0x5d1c9e: the Doppelganger copies a unit that can hurt the other side, a melee one from
        the front row and a shooter or mage from the rear; the most valuable type wins."""
        ids, form = self._forms(env, state)
        traits = env._combat_traits(state)
        actor = state.actor
        own = (env.slots < 6) == (actor < 6)
        foes = (state.hp > 0) & ~state.escaped & ~own
        bits = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[:, 1], 1)-1)
        hurts = jnp.any(foes[None, :] & ((traits[None, :, 2] & bits[:, None]) == 0), axis=1)
        useful = ((form & HARMS) != 0) & (hurts | ((form & 31) == L_SUMMON))
        liked = useful & (((form & CLOSE) != 0) == (actor % 6 < 3))
        # The sorted list holds the own side from its last cell to the first, then the other side.
        cells = self.cell[env.slots % 6]
        rank = jnp.where(own, 5-cells, 6+cells)
        coins = jnp.tile(random_values[:6] < .5, 2)
        group = jnp.where(liked, 1, jnp.where(useful, 2, 3))
        return select(valid, group, self.base_xp[ids], jnp.ones(12, bool), rank, coins)[0]
