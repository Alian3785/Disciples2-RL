"""Training reward profile from Big_map/train_campaign.py REWARD_CONFIG.

Campaign mechanics remain JAX mechanics. The reference's internal battle-step
reward is summed over automatic activations inside one JAX player decision.
See docs/MAPS.md for source functions and map-specific rewards.
"""
import math
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_capital import CAPITAL_HEAL, CAPITAL_REVIVE
from stoix.envs.number_grid_combat import HP
from stoix.envs.number_grid_spells import newly_learned
from stoix.envs.number_grid_unit_swaps import UNITS_SWAPPED
from stoix.envs.number_grid_garrisons import GARRISON_SWAPPED, SWAP_PENALTY


def validate_overrides(game_map):
    """Reject malformed focused-map rewards instead of silently training with them."""
    keys = ('battle_item_equip_reward', 'battle_item_use_reward', 'leadership_step_penalty',
            'turn_penalty', 'spell_cast_reward', 'new_cell_reward', 'rest_with_moves_penalty',
            'timeout_reward', 'magic_enemy_defeat_reward', 'campaign_victory_reward', 'defeat_reward')
    for key in keys:
        if key in game_map:
            value = game_map[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (key not in ('timeout_reward', 'defeat_reward') and value < 0)
                    or (key == 'defeat_reward' and value > 0)):
                raise ValueError(f'{key} must be a finite reward coefficient')
    pickups = game_map.get('item_pickup_rewards', {})
    if not isinstance(pickups, dict) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                                          for v in pickups.values()):
        raise ValueError('item_pickup_rewards must map item keys to finite nonnegative rewards')
    loot_keys = {k for kind in ('chests', 'ruins') for source in game_map.get(kind, [])
                 for k, count in source.get('items', {}).items() if count > 0}
    if set(pickups) - loot_keys:
        raise ValueError('item_pickup_rewards must name chest or ruin items')


def _wave_bonus(env, before, after):
    multiplier = env.siege_rewards['victory_multiplier'] if env.siege_rewards else 1.
    return multiplier*env.scenario.wave_bonus(before, after)


def _siege_kills(env, state, killed):
    """Pay for original enemy slots once, including across retreats/revivals."""
    bits = jnp.left_shift(jnp.uint32(1), jnp.arange(6, dtype=jnp.uint32))
    mask = jnp.bitwise_or.reduce(jnp.where(killed, bits, jnp.uint32(0)))
    fresh = mask & ~state.siege_rewarded_kills
    reward = env.siege_rewards['unit_kill']*jax.lax.population_count(fresh)
    return state.replace(siege_rewarded_kills=state.siege_rewarded_kills | mask), reward


def world(env, before, after, action, rest):
    """Calculate strategic shaping, including the one-time living-level bonus when a siege spawns."""
    moved = jnp.any(before.position != after.position)
    # Being attacked in the home capital is not an engagement of the hero's party.
    engage = after.in_battle & ~after.summon_battle & ~after.garrison_battle
    if env.defense_rewards.suppress_attacker_rewards:
        engage &= env.defense_rewards.rewardable[jnp.maximum(after.enemy, 0)]
    directional = (action >= 0) & (action < 8)
    cell = after.position[0]*env.size+after.position[1]
    visits = before.reference_visits[cell]
    fresh = visits == 0
    nonempty = env.territory.service_at(after) | jnp.any(jnp.all(
        env.territory.mine_positions == after.position, axis=1))
    if env.chests_enabled:
        positions = jnp.array([c['position'] for c in env.map_config['chests']], jnp.int32).reshape(-1, 2)
        nonempty |= jnp.any(before.chest_alive & jnp.all(positions == after.position, axis=1))
    if env.sites_enabled:
        nonempty |= jnp.any(jnp.all(env.site_rules.approach_tiles == after.position, axis=-1))
    reward = -.002*directional + moved*(.01+(env.map_config.get('new_cell_reward', .08)+.001)*fresh
                                       +nonempty*(.05+.10*fresh))
    penalty = jnp.minimum(.08, .015*visits)+.03*jnp.all(after.position == before.reference_previous)
    reward -= moved*penalty
    reward -= rest*env.rest_penalty(before)
    target_battle = (after.enemy == env.objective_enemy
                     if env.reference_preserve_target_reward else jnp.bool_(False))
    reward += .25*engage*~target_battle
    if env.scenario.enabled:
        reward += _wave_bonus(env, before, after)
    if env.siege_rewards:
        target = env.objective_enemy
        spawned = after.scenario_spawned[target] & ~before.scenario_spawned[target]
        # REST cannot promote/revive the permanent party. Capture preparation
        # before battle setup or automatic attacks on the spawning turn, and pay
        # immediately even if contact happens later or the hero loses this turn.
        living = (before.hp[:6] > 0) & (before.unit_ids[:6] != 0)
        strength = jnp.sum(jnp.where(living, jnp.maximum(before.unit_levels[:6]-1, 0), 0))
        reward += spawned*env.siege_rewards['spawn_level']*strength
        if env.spell_casting_enabled:
            maximum = env.progression.stats(env.casting.enemy_ids[target],
                                            after.enemy_progress[target, :, 0])[:, HP].astype(jnp.int32)
            dead = env.casting.occupied[target] & (after.enemy_wounds[target] >= maximum)
            dead &= before.alive[target] & before.scenario_spawned[target]
            after, bonus = _siege_kills(env, after, dead)
            reward += bonus
    if env.spell_casting_enabled:
        cast_rewarded = after.last_cast_spell >= 0
        destroyed = before.alive & ~after.alive
        if env.defense_rewards.suppress_attacker_rewards:
            target = after.last_spell_target
            cast_rewarded &= (target < 0) | env.defense_rewards.rewardable[jnp.maximum(target, 0)]
            destroyed &= env.defense_rewards.rewardable
        reward += env.map_config.get('spell_cast_reward', .005)*cast_rewarded
        reward += env.map_config.get('magic_enemy_defeat_reward', 0.)*jnp.sum(destroyed)
        reward += .05*engage*before.summon_marks[jnp.maximum(after.enemy, 0)]
    reward += newly_learned(before.learned_spells, after.learned_spells).astype(jnp.float32)
    reward += 12.*(after.hire_rewarded_capacity > before.hire_rewarded_capacity)
    healing = jnp.maximum(0, jnp.sum(after.hp[:6]-before.hp[:6]))
    reward += jnp.where(after.last_event == CAPITAL_HEAL,
        .01*jnp.minimum(healing, jnp.maximum(0, before.recovery_balance[0])), 0.)
    reward += .5*((after.last_event == CAPITAL_REVIVE) & (before.recovery_balance[1] > 0))
    reward += .05*after.last_sold_count
    reward -= .004*(after.last_event == UNITS_SWAPPED)
    reward -= SWAP_PENALTY*(after.last_event == GARRISON_SWAPPED)  # capital garrison swaps, the same cost
    if env.rods.enabled:
        # User request 2026-11-09: a small one-off reward per planted rod, at most once per source.
        reward += env.rods.reward*jnp.sum(after.mine_rods & ~before.mine_rods)
    if env.battle_items_enabled:
        reward += env.map_config.get('battle_item_equip_reward', .02)*(after.battle_item_equipped_today
                                                                     & ~before.battle_item_equipped_today)
    if env.map_config.get('item_pickup_rewards'):
        pickups = env.map_config['item_pickup_rewards']
        rewards = jnp.array([sum(pickups.get(k, 0.)*n for k, n in chest.get('items', {}).items())
                             for chest in env.map_config['chests']], jnp.float32)
        reward += jnp.sum(rewards*(before.chest_alive & ~after.chest_alive))
    temporary = jnp.array([p['duration'] == 'temporary' for p in env.potion_rules.items], bool)
    used = jnp.any((after.potions < before.potions) & temporary)
    # Only chest pickups grant this bonus; merchant purchases/ruin rewards do not.
    reward += 2.*used + 4.*jnp.sum(after.last_loot*temporary)*moved
    pending = before.reference_potion_pending | used
    reward += .05*pending*engage
    reached = jnp.zeros_like(before.reference_city_reached)
    if env.territory.count:
        reached = engage & env.territory.goal_cities & (
            jnp.arange(env.territory.count) == env.territory.enemy_city[jnp.maximum(after.enemy, 0)])
        reward += 6.*jnp.sum(reached & ~before.reference_city_reached)
    after = after.replace(
        reference_visits=before.reference_visits.at[cell].set(jnp.where(moved, jnp.minimum(visits+1, 7), visits)),
        reference_previous=jnp.where(moved, before.position, before.reference_previous),
        reference_city_reached=before.reference_city_reached | reached,
        reference_potion_pending=pending & ~engage)
    return after, reward.astype(jnp.float32)


def battle(env, before, after, hp, victory, lost, withdrawal, item_using, base_reward=None):
    summon = before.summon_battle
    ended = victory | lost | withdrawal
    multiplier = (jnp.where(before.enemy == env.objective_enemy,
                           env.siege_rewards['victory_multiplier'], 1.) if env.siege_rewards else 1.)
    repeat = (env.defense_rewards.repeat_factor(env, before, before.enemy)
              if env.defense_rewards.enabled else jnp.float32(1.))
    rewarded = (env.defense_rewards.rewardable[jnp.maximum(before.enemy, 0)]
                if env.defense_rewards.suppress_attacker_rewards else jnp.bool_(True))
    reward = jnp.where(ended, jnp.where(victory, jnp.where(summon, 0., .5*multiplier*repeat)*rewarded, -.25),
                       .0025*rewarded)
    reward += env.map_config.get('battle_item_use_reward', .05)*item_using*rewarded
    reward += env.map_config.get('magic_enemy_defeat_reward', 0.)*(victory & summon)*rewarded
    real_victory = victory & ~summon
    ids = before.unit_ids[:6]
    present = ids != 0
    denominator = jnp.maximum(1, jnp.sum(present))
    maximum = env.max_hp(before)[:6]
    survival = (.4*jnp.sum(present & (hp[:6] > 0)) +
                .4*jnp.sum(jnp.where(present, jnp.clip(hp[:6]/jnp.maximum(1, maximum), 0., 1.), 0.)))/denominator
    experience = after.battle_xp[1].astype(jnp.float32)
    reward += real_victory*(4.*multiplier+.2*experience/(experience+100.)+survival)*repeat*rewarded
    hero = env.territory.heroes[after.unit_ids[:6]]
    changed = (after.last_promoted & jnp.left_shift(jnp.uint32(1), jnp.arange(6, dtype=jnp.uint32))) != 0
    levels = after.unit_levels[:6]
    # Ordinary numerical growth after the last form has no promotion reward.
    promoted = changed & jnp.where(hero, (levels >= 2) & (levels <= 4), ids != after.unit_ids[:6])
    reward += (~summon)*jnp.sum(jnp.where(promoted, 3.+.5*levels, 0.))
    if env.scenario.enabled:
        reward += _wave_bonus(env, before, after)
    if env.siege_rewards:
        # reward HP precedes victory cleanup and the restoration of a summoned
        # attacking party. Fleeing units retain HP and are not kills.
        killed = (hp[6:] <= 0) & (before.enemy_initial_hp > 0)
        killed &= before.in_battle & (before.enemy == env.objective_enemy)
        after, bonus = _siege_kills(env, after, killed)
        reward += bonus
    defeated = before.reference_defeated.at[before.enemy].set(
        before.reference_defeated[before.enemy] | (real_victory & rewarded))
    if env.reference_preserve_target_reward:
        # Paired Capital experiment: preserve the complete tactical target reward,
        # including +1 for victory and +3 for the objective. Level shaping is
        # applied by NumberGrid afterwards, identically in both reward profiles.
        if base_reward is None:
            raise ValueError('Preserving target rewards requires the base battle reward')
        reward = jnp.where(before.enemy == env.objective_enemy, base_reward, reward)
    return after.replace(reference_defeated=defeated), reward.astype(jnp.float32)


def finish(env, before, after):
    """Once per decision: city/episode completion and permanent HP milestones."""
    terminal = after.done & ~before.done
    reward = (env.map_config.get('defeat_reward', -6.)*(after.lost & ~before.lost)
              + env.map_config.get('timeout_reward', -8.)*(terminal & ~after.won & ~after.lost))
    defeated = after.reference_defeated
    if env.defense_rewards.suppress_attacker_rewards:
        defeated &= env.defense_rewards.rewardable
    reward += terminal*5.*jnp.sum(defeated)/max(1, env.reference_enemy_count)
    # Ruins are collected at the end of battle. Pay for
    # the one-shot source transition, never for inventory changes or purchases.
    pickups = env.map_config.get('item_pickup_rewards', {})
    if pickups and env.ruins_enabled:
        amounts = jnp.array([sum(pickups.get(k, 0.)*n for k, n in source.get('items', {}).items())
                             for source in env.map_config['ruins']], jnp.float32)
        reward += jnp.sum(amounts*(after.ruin_looted & ~before.ruin_looted))
    if env.territory.objective_cities:
        old = jnp.sum(before.city_owned & env.territory.goal_cities)
        new = jnp.sum(after.city_owned & env.territory.goal_cities)
        reward += 40.*(new-old)-32.*((old == 0) & (new > 0))
    elif env.position_objective:
        reward += env.objective_position_reward*(after.won & ~before.won)
    elif not env.reference_preserve_target_reward:
        multiplier = env.siege_rewards['victory_multiplier'] if env.siege_rewards else 1.
        # Reference +8 unless the map overrides it (Mirror match: +16 for beating the bot).
        victory = float(env.map_config.get('campaign_victory_reward', 8.))
        reward += victory*multiplier*(after.won & ~before.won)
    milestones = jnp.max(env.max_hp(after)[:6]) > jnp.array([200, 250])
    # Temporary effects cannot permanently unlock milestone rewards.
    milestones &= ~after.in_battle
    reward += 3.*jnp.sum(milestones & ~before.reference_hp_milestones)
    return after.replace(reference_hp_milestones=before.reference_hp_milestones | milestones), reward
