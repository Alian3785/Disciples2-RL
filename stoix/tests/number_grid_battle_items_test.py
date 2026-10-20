"""GPU contracts: hero battle slots (orbs, talismans, battle potions) on the Default map.

Reference: Disciples2-RL campaign_env_inventory battle items and battle_env hero-item actions.
"""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.envs.number_grid import CONTINUE, DEFEND, REST, SHOOT, WAIT, action_names, load_map
from stoix.envs.number_grid_battle_items import (BATTLE_ITEM, ITEM_EQUIPPED, MAGIC_BY_KEY, MAGIC_ITEMS,
                                                 ORBS, TALISMANS, BattleItemRules)
from stoix.envs.number_grid_items import ITEMS, USABLE, ItemRules
from stoix.envs.number_grid_potions import POTION_START
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS
from stoix.tests.number_grid_fixtures import compiled_method, enemy_roster_state



DEFAULT = load_map('default')
NAMES = action_names(DEFAULT)
NE = 1  # [28, 7] -> enemy 11 at [27, 8]


def owned(env, state, *keys):
    ids = [env.item_rules.keys.index(key) for key in keys]
    return state.replace(item_inventory=jnp.array(ids+[-1]*(env.item_rules.capacity-len(ids)), jnp.int32))


def slots(env, state, *keys):
    rules = env.battle_item_rules
    kinds = [rules.keys.index(key) if key else -1 for key in keys]
    return state.replace(battle_items=jnp.array(kinds, jnp.int32),
                         battle_item_charges=jnp.array([int(rules.uses[k]) if k >= 0 else 0 for k in kinds], jnp.int32),
                         battle_item_used=jnp.zeros(2, bool))


@pytest.mark.core
@pytest.mark.parametrize('armor,defended,expected', [(50, False, 38), (0, True, 38), (90, True, 8)])
def test_item_damage_subtracts_integer_absorbed_part(effects, armor, defended, expected):
    """Damage items subtract truncated armor absorption and cap Defend at 90."""
    from stoix.envs.number_grid_combat import ARMOR
    env, base, *_, apply = effects
    state = enemy_roster_state(env, base, ['goblin']*6)
    # orb_of_vampire drains 75 HP; vampire_orb summons a Vampire instead.
    state = slots(env, owned(env, state, 'orb_of_vampire'), 'orb_of_vampire', None)
    state = state.replace(actor=jnp.int32(1), unit_levels=state.unit_levels.at[1].set(10),
        hp=state.hp.at[6:].set(1000), potion_bonus=state.potion_bonus.at[6:, ARMOR].set(armor),
        defended=state.defended.at[6:].set(defended))
    action = jnp.int32(env.battle_item_rules.use_start+12)
    valid, after, hp, info = apply(jax.tree.map(lambda x: x[None], state), action[None])
    assert valid[0]
    assert jnp.all(hp[0, 6:] == 1000-expected) and jnp.sum(info['credit'][0]) == expected*6
    assert after.battle_items[0, 0] == -1


def allowed(mask, start, end):
    return [NAMES[i] for i in range(start, end) if mask[i]]


@pytest.mark.core
def test_catalog_and_default_actions(default_game):
    env, state, _, _ = default_game
    rules = env.battle_item_rules
    assert (len(ORBS), len(TALISMANS)) == (44, 41)
    assert [p['game_id'] for p in ORBS[:43]] == [f'G000IG90{i:02d}' for i in range(1, 44)]
    assert [p['game_id'] for p in TALISMANS] == [f'G000IG91{i:02d}' for i in range(1, 42)]
    assert ORBS[43]['key'] == 'orb_of_lycanthropy' and ORBS[43]['game_id'] is None  # reference-only
    assert [MAGIC_BY_KEY[k]['price'] for k in ('vampire_orb', 'zombie_talisman', 'orb_of_flame',
                                               'talisman_of_nightmare')] == [800, 400, 1000, 600]
    assert {p['key'] for p in ITEMS if p['category'] == USABLE} == set(MAGIC_BY_KEY)
    # Reference order: battle potions, then the scenario's orbs and talismans.
    assert rules.keys == ('healing', 'restoration', 'ointment', 'life',
                          'vampire_orb', 'lich_orb', 'orb_of_life', 'zombie_talisman')
    equip, use = NAMES[rules.equip_start:rules.use_start], NAMES[rules.use_start:rules.end]
    assert equip == tuple('equip_'+k for k in rules.keys) and len(use) == 24
    assert use[:7] == tuple(f'item1_ally_{t}' for t in range(6))+('item2_ally_0',)
    assert use[12] == 'item1_enemy_0' and use[-1] == 'item2_enemy_5'
    assert rules.observation_size == 20 and state.battle_items.tolist() == [-1, -1]


@pytest.mark.core
def test_equip_needs_access_and_happens_once_per_day(default_game):
    env, initial, step, mask = default_game
    rules = env.battle_item_rules
    equip = {k: rules.equip_start+i for i, k in enumerate(rules.keys)}
    healing = env.potion_rules.keys.index('healing')
    tome = env.item_rules.keys.index('tome_arcanum')
    stocked = owned(env, initial, 'vampire_orb', 'zombie_talisman', 'tome_arcanum')
    stocked = stocked.replace(potions=initial.potions.at[healing].set(1))
    # Potions need no lore; Tome of Arcanum allows orbs, level 10 (sorcery lore) everything.
    assert allowed(mask(stocked), rules.equip_start, rules.use_start) == ['equip_healing']
    book = stocked.replace(equipped=stocked.equipped.at[3].set(tome))
    assert allowed(mask(book), rules.equip_start, rules.use_start) == ['equip_healing', 'equip_vampire_orb']
    lore = stocked.replace(unit_levels=stocked.unit_levels.at[1].set(10))
    assert allowed(mask(lore), rules.equip_start, rules.use_start) == [
        'equip_healing', 'equip_vampire_orb', 'equip_zombie_talisman']
    # Default needs book lore (level 11) to read a book: rest unequips it and the orb's access.
    unread, _ = step(step(book, jnp.int32(equip['vampire_orb']))[0], jnp.int32(REST))
    assert unread.equipped[3] == -1 and unread.battle_items.tolist() == [-1, -1]
    first, ts = step(lore, jnp.int32(equip['vampire_orb']))
    assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('potions_used')] == 0
    assert first.battle_items.tolist() == [rules.keys.index('vampire_orb'), -1]
    assert first.last_event == ITEM_EQUIPPED and first.battle_item_equipped_today
    assert not jnp.any(mask(first)[rules.equip_start:rules.use_start])  # once per day
    rested, _ = step(first, jnp.int32(REST))
    assert allowed(mask(rested), rules.equip_start, rules.use_start) == [
        'equip_healing', 'equip_zombie_talisman']  # the one orb is in a slot
    full, _ = step(rested, jnp.int32(equip['healing']))
    assert full.battle_items.tolist() == [rules.keys.index('vampire_orb'), rules.keys.index('healing')]
    assert full.battle_item_charges.tolist() == [1, 1]
    rested, _ = step(full, jnp.int32(REST))
    assert not jnp.any(mask(rested)[rules.equip_start:rules.use_start])  # both slots taken
    # Reference sync: losing the orb or the last potion empties its slot.
    lost, _ = step(owned(env, rested, 'zombie_talisman'), jnp.int32(REST))
    assert lost.battle_items.tolist() == [-1, rules.keys.index('healing')]
    drunk, _ = step(rested.replace(potions=rested.potions.at[healing].set(0)), jnp.int32(REST))
    assert drunk.battle_items.tolist() == [rules.keys.index('vampire_orb'), -1]


@pytest.fixture(scope='module')
def item_battle(default_game):
    env, initial, step, _ = default_game
    battle, _ = step(initial.replace(position=jnp.array([28, 7], jnp.int32)), jnp.int32(NE))
    assert battle.in_battle and battle.enemy == 11
    battle = owned(env, battle, 'orb_of_life', 'zombie_talisman')
    return slots(env, battle, 'orb_of_life', 'zombie_talisman').replace(
        actor=jnp.int32(1), hp=battle.hp.at[0].set(0))  # the hero acts; Squire 0 has fallen


@pytest.mark.core
def test_hero_revives_and_summons_on_its_turn(default_game, item_battle):
    env, _, step, mask = default_game
    rules, ids = env.battle_item_rules, env.progression.ids
    battle = item_battle
    assert allowed(mask(battle), rules.use_start, rules.end) == [
        'item1_ally_0', 'item2_ally_0', 'item2_ally_3', 'item2_ally_5']
    assert not jnp.any(mask(battle.replace(actor=jnp.int32(0)))[rules.use_start:rules.end])  # hero only
    # Orb of Life: 50% HP, the orb leaves the inventory, the turn is spent.
    revived, ts = step(battle, jnp.int32(NAMES.index('item1_ally_0')))
    assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('potions_used')] == 0  # an orb, not Life Potion
    assert revived.hp[0] == 50 and revived.last_event == BATTLE_ITEM and revived.actor != 1
    assert revived.battle_items.tolist() == [-1, rules.keys.index('zombie_talisman')]
    assert env.item_rules.keys.index('orb_of_life') not in revived.item_inventory.tolist()
    # Zombie Talisman: a summon owned by the hero (slot 1); one of five charges, once per battle.
    summoned, _ = step(battle, jnp.int32(NAMES.index('item2_ally_3')))
    assert summoned.unit_ids[3] == ids['g000uu0087'] and summoned.summon_owner[3] == 1 and summoned.hp[3] == 170
    assert summoned.battle_item_charges.tolist() == [1, 4] and summoned.battle_item_used.tolist() == [False, True]
    again = summoned.replace(actor=jnp.int32(1), turn_phase=summoned.turn_phase.at[1].set(0))
    assert allowed(mask(again), rules.use_start, rules.end) == ['item1_ally_0']
    # The mask and step share one predicate: a masked item action changes nothing.
    rejected, _ = step(battle, jnp.int32(NAMES.index('item1_enemy_0')))
    assert rejected.actor == 1 and rejected.battle_items.tolist() == battle.battle_items.tolist()
    # Victory: the summon leaves, the fallen Squire stays dead, the talisman keeps 4 charges.
    won, _ = step(again.replace(hp=again.hp.at[6:].set(0)), jnp.int32(DEFEND))
    for _ in range(6):
        if won.in_battle:
            won, _ = step(won, jnp.int32(WAIT))  # post-battle healing turns
    assert not won.in_battle
    assert won.unit_ids[3] == 0 and won.hp[0] == 0 and not jnp.any(won.summon_owner >= 0)
    assert won.battle_items.tolist() == battle.battle_items.tolist() and won.battle_item_charges.tolist() == [1, 4]


@pytest.mark.core
def test_battle_potion_heals_and_is_drunk(default_game, item_battle):
    env, _, step, mask = default_game
    healing = env.potion_rules.keys.index('healing')
    battle = slots(env, item_battle, 'healing', None)
    battle = battle.replace(potions=battle.potions.at[healing].set(2), hp=battle.hp.at[0].set(40))
    assert allowed(mask(battle), env.battle_item_rules.use_start, env.battle_item_rules.end) == [
        f'item1_ally_{t}' for t in (0, 1, 2, 4)]
    healed, ts = step(battle, jnp.int32(NAMES.index('item1_ally_0')))
    assert ts.extras['campaign_events'][CAMPAIGN_FIELDS.index('potions_used')] == 1
    assert healed.hp[0] == 90 and healed.potions[healing] == 1 and healed.battle_items.tolist() == [-1, -1]


@pytest.mark.parametrize('item', ['orb_of_life', 'life'])
def test_battle_items_cannot_revive_summons(default_game, item_battle, item):
    env, _, step, mask = default_game
    rules = env.battle_item_rules
    # Create the actual item summon (owned by the hero) that previously came back with 85 HP.
    summoned, _ = step(item_battle, jnp.int32(NAMES.index('item2_ally_3')))
    assert summoned.summon_owner[3] == 1
    life = env.potion_rules.keys.index('life')
    dead = slots(env, summoned, item, None).replace(actor=jnp.int32(1),
        hp=summoned.hp.at[3].set(0), potions=summoned.potions.at[life].set(1),
        turn_phase=summoned.turn_phase.at[1].set(0))
    action = jnp.int32(rules.use_start+3)
    assert not mask(dead)[action]
    rejected, _ = step(dead, action)
    for field in ('hp', 'actor', 'battle_key', 'battle_items', 'battle_item_charges',
                  'battle_item_used', 'item_inventory', 'potions'):
        np.testing.assert_array_equal(getattr(rejected, field), getattr(dead, field))
    # The same Zombie as a permanent party member remains a valid target.
    permanent = dead.replace(summon_owner=dead.summon_owner.at[3].set(-1))
    assert mask(permanent)[action]
    # Living summons can still be healed.
    wounded = slots(env, dead, 'healing', None).replace(hp=dead.hp.at[3].set(1),
        potions=dead.potions.at[env.potion_rules.keys.index('healing')].set(1))
    assert mask(wounded)[action]


@pytest.mark.parametrize('owner,map_summon', [(3, False), (4, False), (-1, True)])
def test_patriarch_cannot_revive_any_summon(default_game, item_battle, owner, map_summon):
    env, _, step, mask = default_game
    ids = env.progression.ids
    patriarch, zombie = ids['patriarch'], ids['g000uu0087']
    state = item_battle.replace(actor=jnp.int32(4),
        unit_ids=item_battle.unit_ids.at[4].set(patriarch).at[3].set(zombie),
        unit_levels=item_battle.unit_levels.at[4].set(env.progression.base_levels[patriarch])
            .at[3].set(env.progression.base_levels[zombie]),
        hp=item_battle.hp.at[4].set(125).at[3].set(0),
        summon_owner=item_battle.summon_owner.at[3].set(owner), summon_battle=jnp.bool_(map_summon))
    action = jnp.int32(SHOOT+3)
    for before in (state, state.replace(post_victory=jnp.int32(1),
                                       pending_healers=jnp.zeros(12, bool).at[4].set(True))):
        assert not mask(before)[action]
        rejected, _ = step(before, action)
        for field in ('hp', 'actor', 'battle_key', 'battle_revived', 'revival_xp_cutoff'):
            np.testing.assert_array_equal(getattr(rejected, field), getattr(before, field))
    permanent = state.replace(summon_owner=state.summon_owner.at[3].set(-1), summon_battle=jnp.bool_(False))
    assert mask(permanent)[action]
    assert mask(state.replace(hp=state.hp.at[3].set(1)))[action]
    # Enemy AI uses the same target predicate; a map summon affects only BLUE.
    enemy = state.replace(actor=jnp.int32(10),
        unit_ids=state.unit_ids.at[10].set(patriarch).at[9].set(zombie),
        hp=state.hp.at[10].set(125).at[9].set(0),
        summon_owner=state.summon_owner.at[9].set(-1 if map_summon else owner+6))
    assert bool(compiled_method(env, '_patriarch_targets')(enemy)[9]) == map_summon


@pytest.mark.core
@pytest.mark.parametrize('victory', [True, False], ids=['victory', 'only_escaped_survivor'])
def test_item_summon_preserves_escaped_fighter_at_battle_end(default_game, item_battle, victory):
    env, _, step, mask = default_game
    before = item_battle.replace(hp=item_battle.hp.at[0].set(37),
        escaped=item_battle.escaped.at[0].set(True), unit_xp=item_battle.unit_xp.at[0].set(17),
        unit_levels=item_battle.unit_levels.at[0].set(2))
    action = jnp.int32(NAMES.index('item2_ally_0'))
    assert mask(before)[action]
    summoned, _ = step(before, action)
    assert summoned.summon_saved_hp[0] == 37 and summoned.summon_owner[0] == 1
    assert not summoned.escaped[0] and summoned.hp[0] > 0
    if victory:
        ending = summoned.replace(actor=jnp.int32(1), hp=summoned.hp.at[6:].set(0),
                                   battle_xp=jnp.array([0, 100], jnp.int32))
        result = finish(step, ending)
    else:
        # Every visible ally is dead, but the escaped Squire is still alive.
        ending = summoned.replace(actor=jnp.int32(1), hp=summoned.hp.at[:6].set(0))
        result = finish(step, ending, CONTINUE)
        assert not result.lost and not result.done
        np.testing.assert_array_equal(result.position, before.origin)
    assert result.hp[0] == 37 and result.unit_ids[0] == before.unit_ids[0]
    assert result.unit_levels[0] == 2 and result.unit_xp[0] == 17 and result.last_xp[0] == 0
    assert not jnp.any(result.summon_saved_hp) and jnp.all(result.summon_owner == -1)


@pytest.mark.parametrize('leaves', ['dies', 'flees'])
def test_item_summon_vanishes_when_the_hero_leaves(default_game, item_battle, leaves):
    """Original rule (user in-game check 2026-10-14): an orb/talisman summon is owned
    by the hero who used it and disappears as soon as the hero dies or flees."""
    env, _, step, _ = default_game
    summoned, _ = step(item_battle, jnp.int32(NAMES.index('item2_ally_3')))
    assert summoned.summon_owner[3] == 1 and summoned.hp[3] > 0
    gone = (summoned.replace(hp=summoned.hp.at[1].set(0)) if leaves == 'dies'
            else summoned.replace(escaped=summoned.escaped.at[1].set(True)))
    hp = compiled_method(env, '_linked_summon_hp')(gone, gone.hp)
    assert hp[3] == 0
    assert hp.at[3].set(gone.hp[3]).tolist() == gone.hp.tolist()  # nobody else is affected


def test_hero_death_ends_a_battle_the_summon_would_carry_on():
    """Super Last Stand: once the lone hero falls, the Angel from the orb leaves too
    and the battle is lost in that same step; the Angel cannot win it any more."""
    from stoix.envs.number_grid import NumberGrid
    env = NumberGrid(map_config=load_map('super_last_stand'))
    rules = env.battle_item_rules
    battle_step = compiled_method(env, '_battle_step')
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(0))
    hero = int(jnp.argmax(env.territory.heroes[state.unit_ids[:6]] & (state.unit_ids[:6] != 0)))
    empty = next(i for i in range(6) if int(state.unit_ids[i]) == 0)
    enemy = next(i for i in range(env.num_opponents) if bool(state.alive[i]))
    state = state.replace(battle_items=state.battle_items.at[0].set(rules.keys.index('angel_orb')),
                          enemy=jnp.int32(enemy))
    battle = compiled_method(env, '_begin_battle')(state, jax.random.PRNGKey(1), jnp.full(env.random_size, .5))
    battle = battle.replace(actor=jnp.int32(hero), activation_done=jnp.zeros(12, bool),
                            turn_phase=jnp.zeros(12, jnp.int32), round=jnp.int32(1))
    summoned, _ = battle_step(battle, jnp.int32(rules.use_start+empty), jax.random.PRNGKey(2),
                              jnp.zeros(env.random_size))
    assert int(summoned.summon_owner[empty]) == hero and int(summoned.hp[empty]) > 0
    # The hero has fallen; whatever acts next, the Angel is removed in this step.
    fallen =summoned.replace(hp=summoned.hp.at[hero].set(0), actor=jnp.int32(empty),
                              activation_done=jnp.zeros(12, bool), turn_phase=jnp.zeros(12, jnp.int32))
    after, reward = battle_step(fallen, jnp.int32(DEFEND), jax.random.PRNGKey(3), jnp.zeros(env.random_size))
    assert after.lost and after.done and not after.won
    assert after.hp[:6].tolist() == [0]*6 and float(reward) < 0


def finish(step, state, action=DEFEND):
    state, _ = step(state, jnp.int32(action))
    for _ in range(6):
        if not state.in_battle:
            break
        assert state.post_victory > 0
        state, _ = step(state, jnp.int32(WAIT))
    assert not state.in_battle
    return state


@pytest.mark.core
@pytest.mark.parametrize('victory', [True, False], ids=['victory', 'summon_defeat'])
@pytest.mark.parametrize('pair', [('healing', 'vampire_orb'), ('zombie_talisman', 'zombie_talisman')])
def test_map_summon_preserves_hero_battle_items(default_game, victory, pair):
    """Reference _build_summoned_blue_team keeps the travelling party's equipment."""
    env, initial, step, mask = default_game
    rules = env.battle_item_rules
    spell = next(row['action'] for row in env.casting.metadata['spells']
                 if row['kind'] == 'summon_battle' and row['action'] is not None)
    state = slots(env, owned(env, initial, *(key for key in pair if key != 'healing')), *pair)
    state = state.replace(position=jnp.array([28, 7], jnp.int32),
        unit_levels=state.unit_levels.at[1].set(10), mana=jnp.full(5, 5000, jnp.int32),
        learned_spells=jnp.uint32(1 << (spell-env.casting.start)),
        potions=state.potions.at[env.potion_rules.keys.index('healing')].set(2),
        battle_item_equipped_today=jnp.bool_(True))
    if pair[0] == 'zombie_talisman':
        state = state.replace(battle_item_charges=jnp.array([2, 4], jnp.int32),
                              item_charges_spent=state.item_charges_spent.at[:2].set(jnp.array([3, 1], jnp.int32)))
    assert mask(state)[spell]
    summoned, _ = step(state, jnp.int32(spell))
    assert summoned.in_battle and summoned.summon_battle
    assert not jnp.any(mask(summoned)[rules.use_start:rules.end])  # items stay with the hero
    for field in ('battle_items', 'battle_item_charges', 'battle_item_equipped_today',
                  'item_inventory', 'item_charges_spent', 'potions'):
        np.testing.assert_array_equal(getattr(summoned, field), getattr(state, field))
    # Resolve either outcome through the real battle boundary, with the hero still stashed.
    ended_hp = summoned.hp.at[6:].set(0) if victory else summoned.hp.at[:6].set(0)
    out = finish(step, summoned.replace(hp=ended_hp), DEFEND if victory else CONTINUE)
    assert not out.summon_battle and not out.lost and not out.done
    assert bool(out.alive[int(summoned.enemy)]) != victory
    for field in ('unit_ids', 'unit_levels', 'unit_xp', 'hp'):
        np.testing.assert_array_equal(getattr(out, field)[:6], getattr(state, field)[:6])
    # Also exercise the next ordinary map synchronization after the party returns.
    rested, _ = step(out, jnp.int32(REST))
    for field in ('battle_items', 'battle_item_charges', 'item_inventory', 'item_charges_spent', 'potions'):
        np.testing.assert_array_equal(getattr(out, field), getattr(state, field))
        np.testing.assert_array_equal(getattr(rested, field), getattr(state, field))


@pytest.mark.core
@pytest.mark.parametrize('withdrawal', [False, True], ids=['victory', 'retreat'])
@pytest.mark.parametrize('pair', [('healing', 'orb_of_life'), ('zombie_talisman', 'zombie_talisman')])
def test_dead_leader_returns_both_slots_without_losing_items_or_charges(default_game, item_battle, withdrawal, pair):
    """Manual Aftermath; reference test_dead_hero_equipment: no loss or recharge."""
    env, _, step, mask = default_game
    rules = env.battle_item_rules
    battle = slots(env, owned(env, item_battle, *(k for k in pair if k != 'healing')), *pair)
    healing = env.potion_rules.keys.index('healing')
    battle = battle.replace(actor=jnp.int32(0), hp=battle.hp.at[1].set(0).at[0].set(100),
                            potions=battle.potions.at[healing].set(2))
    if pair[0] == 'zombie_talisman':
        battle = battle.replace(battle_item_charges=jnp.array([2, 4], jnp.int32),
                                battle_item_used=jnp.ones(2, bool))
    if withdrawal:
        battle = battle.replace(hp=battle.hp.at[:6].set(0).at[0].set(100),
                                retreating=battle.retreating.at[0].set(True))
    else:
        battle = battle.replace(hp=battle.hp.at[6:].set(0))
    out = finish(step, battle, CONTINUE if withdrawal else DEFEND)
    assert out.hp[1] == 0 and jnp.any(out.hp[:6] > 0) and not out.lost
    assert out.battle_items.tolist() == [-1, -1] and out.battle_item_charges.tolist() == [0, 0]
    assert not jnp.any(out.battle_item_used)
    np.testing.assert_array_equal(out.item_inventory, battle.item_inventory)
    np.testing.assert_array_equal(out.potions, battle.potions)
    assert not jnp.any(mask(out)[rules.equip_start:rules.use_start])  # dead leaders cannot equip
    repeated, _ = step(out, jnp.int32(REST))
    np.testing.assert_array_equal(repeated.item_inventory, out.item_inventory)
    np.testing.assert_array_equal(repeated.item_charges_spent, out.item_charges_spent)
    if pair[0] == 'zombie_talisman':
        assert out.item_charges_spent[:2].tolist() == [3, 1]
        # Map revival does not equip automatically. Both partial copies can be re-equipped.
        life = env.potion_rules.keys.index('life')
        revival = repeated.replace(potions=repeated.potions.at[life].set(1),
                                    unit_levels=repeated.unit_levels.at[1].set(10))
        revival, _ = step(revival, jnp.int32(POTION_START+6*life+1))
        assert revival.hp[1] == 1 and revival.battle_items.tolist() == [-1, -1]
        equip = jnp.int32(rules.equip_start+rules.keys.index('zombie_talisman'))
        first, _ = step(revival, equip)
        rested, _ = step(first, jnp.int32(REST))
        second, _ = step(rested, equip)
        assert second.battle_item_charges.tolist() == [2, 4]
        np.testing.assert_array_equal(second.item_inventory, out.item_inventory)


@pytest.mark.core
def test_leader_death_keeps_slots_until_end_and_same_battle_revival_keeps_them(default_game, item_battle):
    env, _, step, _ = default_game
    battle = item_battle.replace(actor=jnp.int32(0), hp=item_battle.hp.at[0].set(100).at[1].set(0),
                                battle_item_charges=jnp.array([1, 3], jnp.int32))
    ongoing, _ = step(battle, jnp.int32(DEFEND))
    assert ongoing.in_battle and ongoing.hp[1] == 0
    np.testing.assert_array_equal(ongoing.battle_items, battle.battle_items)
    np.testing.assert_array_equal(ongoing.battle_item_charges, battle.battle_item_charges)
    # A resurrection before the final transition must preserve the original equipment.
    revived = ongoing.replace(actor=jnp.int32(0), hp=ongoing.hp.at[1].set(1).at[6:].set(0),
                               unit_levels=ongoing.unit_levels.at[1].set(10))
    out = finish(step, revived)
    assert out.hp[1] > 0
    np.testing.assert_array_equal(out.battle_items, battle.battle_items)
    np.testing.assert_array_equal(out.battle_item_charges, battle.battle_item_charges)
    # With a dead leader the post-victory healing phase still belongs to this battle.
    healing, _ = step(battle.replace(hp=battle.hp.at[6:].set(0)), jnp.int32(DEFEND))
    assert healing.in_battle and healing.post_victory == 1
    np.testing.assert_array_equal(healing.battle_items, battle.battle_items)
    ended = finish(step, healing, WAIT)
    assert ended.battle_items.tolist() == [-1, -1]


@pytest.mark.core
def test_partial_talisman_charges_follow_inventory_compaction(default_game):
    env, initial, _, _ = default_game
    # An orb is consumed before the talisman in acquisition order.
    base = slots(env, owned(env, initial, 'orb_of_life', 'zombie_talisman', 'zombie_talisman'),
                 'orb_of_life', 'zombie_talisman')
    base = base.replace(battle_item_charges=jnp.array([1, 2], jnp.int32),
                        item_charges_spent=base.item_charges_spent.at[2].set(1))
    rules = env.battle_item_rules

    @jax.jit
    def roundtrip(state):
        orb = rules.keys.index('orb_of_life')
        talisman = rules.keys.index('zombie_talisman')
        state = rules.consume(state, jnp.int32(0), jnp.int32(orb), jnp.bool_(True))
        # Slot 1 still reserves the first talisman; filling slot 0 must take the other copy.
        state = state.replace(unit_levels=state.unit_levels.at[1].set(10))
        return rules.equip(env, state, jnp.int32(rules.equip_start+talisman))

    equipped = roundtrip(base)
    assert equipped.battle_item_charges.tolist() == [4, 2]
    assert equipped.item_charges_spent[:3].tolist() == [1, 3, 0]
    # Spending the second identical copy must leave the first copy's history intact.
    consumed = jax.jit(rules.consume)(equipped.replace(battle_item_charges=jnp.array([4, 1], jnp.int32)),
        jnp.int32(1), jnp.int32(rules.keys.index('zombie_talisman')), jnp.bool_(True))
    assert consumed.item_charges_spent[:2].tolist() == [1, 0]
    assert consumed.battle_item_charges.tolist() == [4, 0]
    assert int(jnp.sum(consumed.item_inventory >= 0)) == 1


@pytest.mark.core
def test_returned_charges_survive_sales_scrolls_and_new_loot(default_game):
    env, initial, step, mask = default_game
    state = owned(env, initial, 'bronze_ring', 'mind_ward_scroll', 'zombie_talisman')
    state = state.replace(item_charges_spent=state.item_charges_spent.at[2].set(3),
                          position=jnp.array([36, 25], jnp.int32), scroll_magic=jnp.bool_(True),
                          unit_levels=state.unit_levels.at[1].set(10))
    sold, _ = step(state, jnp.int32(REST))  # Alar: sell the valuable preceding both usable items.
    assert sold.last_sold_count == 1 and sold.item_charges_spent[:3].tolist() == [0, 3, 0]
    cast = jnp.int32(NAMES.index('cast_mind_ward_scroll'))
    assert mask(sold)[cast]
    casted, _ = step(sold, cast)
    assert casted.item_inventory[0] == env.item_rules.keys.index('zombie_talisman')
    assert casted.item_charges_spent[:3].tolist() == [3, 0, 0]
    # Picking up another copy leaves its charges fresh and the returned copy partial.
    at_chest = casted.replace(position=jnp.array([24, 37], jnp.int32))
    picked, _ = step(at_chest, jnp.int32(0))
    assert picked.item_inventory[:2].tolist() == [env.item_rules.keys.index('zombie_talisman')]*2
    assert picked.item_charges_spent[:3].tolist() == [3, 0, 0]
    observed = jax.jit(env.item_rules.observation)(picked)
    np.testing.assert_allclose(observed[-env.item_rules.charge_capacity:], picked.item_charges_spent/5)


ENEMIES = ['goblin', 'orc', 'goblin', 'g000uu0090', 'g000uu0087', 'goblin_archer']  # Templar wards, Zombie
CASES = (  # key, enemy side, target, (ally slot, hp) before use
    ('orb_of_fire', 1, 0, None), ('talisman_of_thunder', 1, 1, None), ('orb_of_nosferat', 1, 0, (1, 50)),
    ('orb_of_elder_vampires', 1, 0, (0, 10)), ('orb_of_poison', 1, 1, None), ('orb_of_frost', 1, 0, None),
    ('orb_of_flame', 1, 0, None), ('orb_of_vigor', 0, 0, None), ('orb_of_rage', 0, 0, None),
    ('orb_of_healing', 0, 0, (0, 10)), ('orb_of_restoration', 0, 0, (0, 10)), ('orb_of_fear', 1, 1, None),
    ('talisman_of_nightmare', 1, 1, None), ('orb_of_witches', 1, 2, None), ('orb_of_lycanthropy', 1, 1, None),
    ('orb_of_weakening', 1, 1, None), ('elf_lord_orb', 0, 3, None), ('orb_of_life', 0, 0, (0, 0)))


@pytest.fixture(scope='module')
def effects(default_game):
    """Every reference effect through one vmapped kernel on a Default copy."""
    original, initial, _, _ = default_game
    env = copy.copy(original)
    game_map = dict(DEFAULT, initial_items={p['key']: 1 for p in MAGIC_ITEMS})
    env.item_rules = ItemRules(game_map)
    rules = env.battle_item_rules = BattleItemRules(game_map, env.potion_rules, env.item_rules,
                                                    env.progression, env.construction)
    for flag in ('has_fire', 'has_water', 'has_powerups', 'has_paralysis', 'has_witches', 'has_wights',
                 'has_lycanthropy', 'has_weakening', 'has_item_weakening', 'has_item_summons', 'has_cures'):
        setattr(env, flag, True)
    begin = jax.jit(env._begin_battle)(initial.replace(enemy=jnp.int32(11)))
    base = enemy_roster_state(env, begin, ENEMIES).replace(
        actor=jnp.int32(1), item_inventory=jnp.full(env.item_rules.capacity, -1, jnp.int32),
        item_charges_spent=jnp.zeros(env.item_rules.charge_capacity, jnp.int32))
    count = len(CASES)
    kinds = np.array([[rules.keys.index(key), -1] for key, *_ in CASES], np.int32)
    hp = np.repeat(np.asarray(base.hp)[None], count, 0)
    for row, (*_, wound) in enumerate(CASES):
        if wound:
            hp[row, wound[0]] = wound[1]
    batch = jax.tree.map(lambda x: jnp.repeat(x[None], count, 0), base).replace(
        battle_items=jnp.asarray(kinds), hp=jnp.asarray(hp),
        battle_item_charges=jnp.asarray(np.where(kinds >= 0, np.asarray(rules.uses)[kinds], 0), jnp.int32))
    actions = jnp.array([rules.use_start+side*12+target for _, side, target, _ in CASES], jnp.int32)

    @jax.jit
    @jax.vmap
    def apply(state, action):
        valid = rules.use_available(env, state)[action-rules.use_start]
        after, hp, _, info = rules.use(env, state, state.hp, state.wards_used, action,
                                       jnp.zeros(env.random_size), env._combat_stats(state),
                                       env._combat_traits(state))
        return valid, after, hp, info

    valid, after, hp, info = apply(batch, actions)
    keys = [key for key, *_ in CASES]
    return env, base, keys, valid, after, hp, info, apply


def test_reference_item_effects(effects):
    """Check catalogued item effects against their expected combat-state changes."""
    env, base, keys, valid, after, hp, info, _ = effects
    row = {key: i for i, key in enumerate(keys)}
    rules, ids = env.battle_item_rules, env.progression.ids
    assert bool(jnp.all(valid))
    start = base.hp.tolist()

    def lost(key):
        return [a-b for a, b in zip(start, hp[row[key]].tolist())]

    # Party fire 25 at 80% accuracy; the Templar's fire ward absorbs it and is spent.
    templar = 1 << 9
    assert lost('orb_of_fire')[6:] == [25, 25, 25, 0, 25, 25] and int(info['ward'][row['orb_of_fire']]) == templar
    assert lost('talisman_of_thunder')[6:] == [0, 100, 0, 0, 0, 0]
    # Drain: the Zombie is immune to death; the hero heals half of the damage dealt.
    nosferat = lost('orb_of_nosferat')
    assert nosferat[6:] == [25, 25, 25, 25, 0, 25] and int(info['immune'][row['orb_of_nosferat']]) == 1 << 10
    assert hp[row['orb_of_nosferat'], 1] == 50+125//2
    assert hp[row['orb_of_elder_vampires'], 0] == 100  # the hero is unhurt: the party shares the pool
    # DOTs: single and party, long effects with round-age recovery, the Templar ward blocks water/fire.
    for key, field, amount, targets in (('orb_of_poison', 'poison', 20, [7]), ('orb_of_frost', 'water', 10, [6, 7, 8, 10, 11]),
                                        ('orb_of_flame', 'burn', 15, [6, 7, 8, 10, 11])):
        damage = getattr(after, field+'_damage')[row[key]]
        assert [int(i) for i in jnp.flatnonzero(damage == amount)] == targets, key
        assert jnp.all(getattr(after, field+'_mode')[row[key], jnp.array(targets)] == 2)
    original = env._original_primary(base)
    assert after.primary_override[row['orb_of_vigor'], 0] == jnp.rint(original[0]*1.25)
    assert after.powerup_long[row['orb_of_vigor'], 0]
    rage = row['orb_of_rage']
    assert after.turn_phase[rage, 0] == 0 and after.bonus_turns[rage, 0] == 1 and info['extra'][rage, 0]
    assert hp[row['orb_of_healing'], 0] == 100 and hp[row['orb_of_restoration'], 0] == 40
    assert after.long_paralyzed[row['orb_of_fear'], 7] and after.retreating[row['talisman_of_nightmare'], 7]
    assert after.imp[row['orb_of_witches'], 8]
    lycan = row['orb_of_lycanthropy']
    assert after.decay_form[lycan, 7] == ids['g000uu0092'] and after.lycanthrope[lycan, 7]
    weak = row['orb_of_weakening']
    assert after.item_weakened[weak, 7] and after.weakened[weak, 7]
    assert after.primary_override[weak, 7] == jnp.rint(env._combat_stats(base)[7, 1]*.67)
    summon = row['elf_lord_orb']
    assert after.unit_ids[summon, 3] == ids['g000uu8009'] and after.summon_owner[summon, 3] == 1  # the hero
    life = row['orb_of_life']
    assert hp[life, 0] == 50 and after.battle_items[life].tolist() == [-1, -1]
    # Talismans spend one charge; orbs leave the slot.
    assert after.battle_item_charges[row['talisman_of_thunder']].tolist() == [4, 0]
    assert after.battle_items[row['orb_of_fire']].tolist() == [-1, -1]
    assert int(rules.uses[rules.keys.index('talisman_of_thunder')]) == 5


def test_summon_win_with_dead_party_is_defeat():
    """Super Last Stand: an Angel Orb summon wins after the whole party has died.

    User report 2026-10-14: the summon's HP hid the wiped party, so no defeat was
    set, the last hit paid a victory and the dead party kept moving. The summon
    leaves with the battle, so the battle must end in the party's defeat.
    """
    from stoix.envs.number_grid import NumberGrid
    env = NumberGrid(map_config=load_map('super_last_stand'))
    rules = env.battle_item_rules
    battle_step = compiled_method(env, '_battle_step')
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(0))
    hero = int(jnp.argmax(env.territory.heroes[state.unit_ids[:6]] & (state.unit_ids[:6] != 0)))
    empty = next(i for i in range(6) if int(state.unit_ids[i]) == 0)
    enemy = next(i for i in range(env.num_opponents) if bool(state.alive[i]))
    state = state.replace(battle_items=state.battle_items.at[0].set(rules.keys.index('angel_orb')),
                          enemy=jnp.int32(enemy))
    battle = compiled_method(env, '_begin_battle')(state, jax.random.PRNGKey(1), jnp.full(env.random_size, .5))
    battle = battle.replace(actor=jnp.int32(hero), activation_done=jnp.zeros(12, bool),
                            turn_phase=jnp.zeros(12, jnp.int32), round=jnp.int32(1))
    use = rules.use_start+empty  # item slot 1 -> own empty cell
    assert compiled_method(env, 'action_mask')(battle)[use]
    summoned, _ = battle_step(battle, jnp.int32(use), jax.random.PRNGKey(2), jnp.zeros(env.random_size))
    assert int(summoned.summon_owner[empty]) == empty and int(summoned.hp[empty]) > 0
    target = int(jnp.argmax(summoned.hp[6:] > 0))
    # Every permanent fighter has fallen; the Angel finishes the last enemy (1 HP).
    hp = jnp.where(summoned.summon_owner >= 0, summoned.hp, 0).at[6+target].set(1)
    last = summoned.replace(hp=hp, actor=jnp.int32(empty), activation_done=jnp.zeros(12, bool),
                            turn_phase=jnp.zeros(12, jnp.int32))
    after, reward = battle_step(last, jnp.int32(SHOOT+target), jax.random.PRNGKey(3), jnp.zeros(env.random_size))
    assert after.lost and after.done and not after.won  # like any defeat, the episode ends here
    assert after.hp[:6].tolist() == [0]*6 and after.summon_owner[:6].tolist() == [-1]*6
    assert float(reward) < 0  # no victory payment for the summon's last hit
    # With the hero still alive the same last hit is an ordinary victory.
    wounded = last.replace(hp=last.hp.at[hero].set(1))
    alive, reward = battle_step(wounded, jnp.int32(SHOOT+target), jax.random.PRNGKey(3), jnp.zeros(env.random_size))
    assert not alive.lost and not alive.done and not alive.in_battle and float(reward) > 0


@pytest.mark.parametrize('key', ['orb_of_fear', 'talisman_of_fear'])
def test_magic_item_paralysis_misses_after_round_40_and_keeps_wards(effects, key):
    env, base, *_ = effects
    rules = env.battle_item_rules
    target = 7
    action = jnp.int32(rules.use_start + 12 + target - 6)
    before = slots(env, base, key, None).replace(round=jnp.int32(40))
    apply = jax.jit(lambda s: rules.use(env, s, s.hp, s.wards_used, action,
        jnp.zeros(env.random_size), env._combat_stats(s), env._combat_traits(s)))
    assert apply(before)[0].long_paralyzed[target]
    kind = rules.keys.index(key)
    source = jnp.left_shift(jnp.uint32(1), rules.source[kind].astype(jnp.uint32)-1)
    late = before.replace(round=jnp.int32(41), potion_wards=before.potion_wards.at[target].set(source))
    after, hp, used, info = apply(late)
    assert not jnp.any(after.long_paralyzed | after.paralyzed)
    assert not used[target] and not info['ward'] and info['applied'] == 0
    assert jnp.array_equal(hp, late.hp)
    assert after.battle_item_charges[0] < before.battle_item_charges[0]


@pytest.mark.parametrize('key', ['orb_of_witches', 'talisman_of_witches'])
def test_witches_items_exclude_all_capital_guards_before_wards(effects, key):
    """User request 2026-10-19: items must share the native witch's guardian exclusion."""
    env, base, *_ = effects
    rules = env.battle_item_rules
    roster = ['g000uu3001', 'g000uu3002', 'g000uu3003', 'g000uu3004', 'g000uu8040', 'goblin_archer']
    before = slots(env, owned(env, enemy_roster_state(env, base, roster), key), key, None)
    kind = rules.keys.index(key)
    source = jnp.left_shift(jnp.uint32(1), rules.source[kind].astype(jnp.uint32)-1)
    actions = rules.use_start + 12 + jnp.tile(jnp.arange(6), 2)
    warded = jnp.repeat(jnp.array([False, True]), 6)

    @jax.jit
    @jax.vmap
    def apply(action, ward):
        # Isolate the guardian restriction from native source immunities/wards.
        traits = env._combat_traits(before)
        traits = traits.at[:, 2].set(traits[:, 2] & ~source)
        traits = traits.at[:, 3].set(jnp.where(ward, source, jnp.uint32(0)))
        valid = rules.use_available(env, before)[action-rules.use_start]
        after, hp, used, info = rules.use(env, before, before.hp, before.wards_used, action,
            jnp.zeros(env.random_size), env._combat_stats(before), traits)
        return valid, after, hp, used, info

    valid, after, hp, used, info = apply(actions, warded)
    guarded = jnp.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    assert jnp.all(valid)  # As with a native witch, an ineffective attempt is allowed.
    assert not jnp.any(after.imp[guarded] | after.transform_long[guarded])
    assert jnp.all(hp == before.hp)
    assert jnp.all(after.unit_ids == before.unit_ids)
    assert jnp.all(after.priority[guarded] == before.priority)
    assert jnp.all(used[guarded] == before.wards_used)
    assert not jnp.any(info['ward'][guarded] | info['immune'][guarded] | info['applied'][guarded])
    # Ordinary units still transform; their mind ward still absorbs the effect.
    assert after.imp[5, 11] and info['applied'][5] == 1
    assert not jnp.any(after.imp[11]) and info['applied'][11] == 0
    assert info['ward'][11] == 1 << 11 and used[11, 11] & source
    # Both successful and ineffective uses consume the orb or one talisman charge.
    talisman = key == 'talisman_of_witches'
    assert jnp.all(after.battle_item_charges[:, 0] == (4 if talisman else 0))
    assert jnp.all(after.battle_items[:, 0] == (kind if talisman else -1))
    assert jnp.all(after.battle_item_used[:, 0] == talisman)
