"""GPU event counters for the reference CampaignMetricsCallback (no reward shaping)."""
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_capital import HEAL_START, REVIVE_START
from stoix.envs.number_grid_casting import SUMMON
from stoix.envs.number_grid_potions import POTION_START
from stoix.envs.number_grid_battle_items import SLOTS, TARGETS, ACTIONS as ITEM_ACTIONS

# Fixed layout keeps telemetry a single small array in the rollout/state tree.
CAMPAIGN_FIELDS = (
    'victories', 'defeats', 'timeouts',
    'battle_victories', 'battle_defeats', 'battle_retreats',
    'unit_upgrades', 'castle_heal_uses', 'castle_healed_hp',
    'chests_collected', 'buildings_built', 'ruins_cleared',
    'sold_items', 'merchant_sale_gold', 'spells_learned', 'spell_casts',
    'summoned_units', 'hired_units', 'battle_items_equipped', 'battle_items_used',
    'unit_swaps', 'enemy_squads_defeated', 'cities_captured',
    'potions_used', 'dragon_battles', 'mercenaries_hired', 'trainer_uses',
    'capital_defense_acquired', 'capital_defense_effects', 'capital_health_effects',
    'capital_guardian_rounds', 'capital_guardian_battles',
    'capital_first_contacts', 'capital_entry_strength',
    'merchant_purchases', 'waves_completed',
)
EPISODE_WINDOW = 100


def empty_campaign_counts():
    return jnp.zeros(len(CAMPAIGN_FIELDS), jnp.float32)


def campaign_events(env, before, world, after, action, valid, battle_counts):
    """Measure executed events, retaining the map result before automatic combat.

    Map summons count as one summon, even when their entire battle finishes
    inside the same decision. Native battle summons are not map spell summons.
    """
    world_action = valid & ~before.in_battle
    terminal = ~before.done & after.done
    values = dict(zip(CAMPAIGN_FIELDS, [jnp.float32(0)] * len(CAMPAIGN_FIELDS)))
    values.update(
        victories=terminal & after.won,
        defeats=terminal & after.lost & ~after.won,
        timeouts=terminal & ~after.won & ~after.lost,
        battle_victories=battle_counts[0], battle_defeats=battle_counts[1],
        battle_retreats=battle_counts[2], unit_upgrades=battle_counts[3],
        enemy_squads_defeated=jnp.sum(before.alive & ~after.alive),
    )
    if env.scenario.enabled and env.map_config.get('campaign_objective') == 'waves':
        rules = env.scenario
        # Match ScenarioRules.wave_bonus: future or partly defeated waves do
        # not count. Measure before auto-reset, including the terminal victory.
        defeated_before = before.scenario_spawned[rules.indices] & ~before.alive[rules.indices]
        defeated_after = after.scenario_spawned[rules.indices] & ~after.alive[rules.indices]
        cleared_before = jnp.all(defeated_before | ~rules.wave_groups, axis=1)
        cleared_after = jnp.all(defeated_after | ~rules.wave_groups, axis=1)
        values['waves_completed'] = jnp.sum(
            cleared_after & ~cleared_before & (rules.wave_numbers > 0))
    if env.bot.enabled:
        from stoix.envs.number_grid import REST
        # REST's private bot combats do not count as the agent's victories.
        values['enemy_squads_defeated'] = (jnp.sum(world.alive & ~after.alive)
            + jnp.where(world_action & (action != REST), jnp.sum(before.alive & ~world.alive), 0))
    if env.basic_combat:
        values['buildings_built'] = jax.lax.population_count(world.buildings & ~before.buildings)
    if env.capital_training.enabled:
        rules = env.capital_training
        effects = after.capital_defense_peak - before.capital_defense_peak
        values['capital_defense_acquired'] = after.capital_defense_acquired - before.capital_defense_acquired
        values['capital_defense_effects'] = jnp.sum(effects)
        values['capital_health_effects'] = jnp.sum(jnp.where(rules.health, effects, 0))
        values['capital_guardian_rounds'] = after.capital_rounds_rewarded - before.capital_rounds_rewarded
        values['capital_guardian_battles'] = (world_action & world.in_battle & ~world.summon_battle
                                              & (world.enemy == rules.enemy))
        first_contact = (before.capital_entry_strength < 0) & (after.capital_entry_strength >= 0)
        values['capital_first_contacts'] = first_contact
        values['capital_entry_strength'] = jnp.where(first_contact, after.capital_entry_strength, 0)
    if env.capital_enabled:
        healing = world_action & (action >= HEAL_START) & (action < REVIVE_START)
        healed = jnp.sum(jnp.maximum(world.hp[:6] - before.hp[:6], 0))
        values['castle_heal_uses'] = healing & (healed > 0)
        values['castle_healed_hp'] = jnp.where(healing, healed, 0)
    if env.potions_enabled:
        values['potions_used'] = (world_action & (action >= POTION_START)
                                  & (action < POTION_START + env.potion_rules.action_count)).astype(jnp.int32)
    if env.map_config.get('campaign_objective') in ('dragon', 'blue_dragon'):
        # Use the map result *before* automatic combat: arrival still counts when
        # the dragon kills the party on its first automatic activation.
        values['dragon_battles'] = (world_action & world.in_battle & ~world.summon_battle
                                    & (world.enemy == env.objective_enemy))
    if env.chests_enabled:
        values['chests_collected'] = jnp.sum(before.chest_alive & ~world.chest_alive)
    if env.ruins_enabled:
        values['ruins_cleared'] = jnp.sum(~before.ruin_looted & after.ruin_looted)
    if env.sites_enabled:
        values['sold_items'] = jnp.where(world_action, world.last_sold_count, 0)
        values['merchant_sale_gold'] = jnp.where(world_action, world.last_sale_gold, 0)
        if env.site_rules.has_merchant:
            # Each successful purchase of a potion or item takes one unit of
            # merchant stock; nothing else changes that stock.
            values['merchant_purchases'] = jnp.sum(before.merchant_stock - world.merchant_stock)
        if env.site_rules.has_trainer:
            start = env.site_rules.train_start
            # One successful purchase for one fighter, regardless of XP bought.
            values['trainer_uses'] = world_action & (action >= start) & (action < start + 6)
    if env.spell_research_enabled:
        values['spells_learned'] = jax.lax.population_count(world.learned_spells & ~before.learned_spells)
    if env.spell_casting_enabled:
        cast = world_action & (world.last_cast_spell >= 0)
        values['spell_casts'] = cast
        values['summoned_units'] = cast & (env.casting.kinds[jnp.maximum(world.last_cast_spell, 0)] == SUMMON)
    if env.recruitment_enabled:
        rules = env.recruitment
        hired = world_action & (action >= rules.start) & (action < rules.dismiss_start)
        offer = jnp.clip(action - rules.start, 0, rules.offer_count - 1)
        values['hired_units'] = hired
        values['mercenaries_hired'] = hired & rules.mercenary[offer]
    if env.battle_items_enabled and env.battle_item_rules.count:
        rules = env.battle_item_rules
        values['battle_items_equipped'] = world_action & (action >= rules.equip_start) & (action < rules.use_start)
        values['battle_items_used'] = valid & before.in_battle & (action >= rules.use_start) & (action < rules.end)
        index = jnp.clip(action - rules.use_start, 0, ITEM_ACTIONS - 1)
        kind = before.battle_items[(index // TARGETS) % SLOTS]
        potion = (kind >= 0) & (rules.potion_index[jnp.maximum(kind, 0)] >= 0)
        # Count the consumed dose, not an inventory delta: victory loot in the
        # same step can replenish that potion; subsequent turns clear last_*.
        values['potions_used'] += (values['battle_items_used'] & potion).astype(jnp.int32)
    if env.unit_swaps_enabled:
        values['unit_swaps'] = world_action & (action >= env.unit_swaps.start) & (action < env.unit_swaps.end)
    if env.territory_enabled:
        values['cities_captured'] = jnp.sum(~before.city_owned & after.city_owned)
    return jnp.stack([jnp.asarray(values[name], jnp.float32) for name in CAMPAIGN_FIELDS])


def summarize_campaign(episodes):
    """Reduce on GPU; transfer at most 100 completed records per learner chunk.

    Totals and participation counts include *every* event/episode. Only the
    rolling-window records are bounded. Single-device rollout order is update,
    time, environment; simultaneous completions use the environment index.
    """
    ended = episodes['is_terminal_step']
    events = episodes['campaign_events']
    counts = episodes['campaign_episode']
    mask = ended[..., None]
    axes = tuple(range(ended.ndim))
    terminal = ended.reshape(-1)
    k = min(EPISODE_WINDOW, terminal.size)
    ranks = jnp.where(terminal, jnp.arange(terminal.size), -1)
    _, indices = jax.lax.top_k(ranks, k)
    records = jnp.concatenate((counts, episodes['episode_return'][..., None],
                               episodes['episode_length'][..., None]), axis=-1)
    recent = records.reshape(-1, len(CAMPAIGN_FIELDS) + 2)[indices[::-1]]
    return {
        'events': jnp.sum(events, axis=axes),
        'completed': jnp.sum(jnp.where(mask, counts, 0), axis=axes),
        'participation': jnp.sum(mask & (counts > 0), axis=axes),
        'count': jnp.sum(ended),
        'recent': recent,
    }
