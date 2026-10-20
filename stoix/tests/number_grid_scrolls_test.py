"""CUDA contracts: scrolls and staves cast catalogue spells (docs/SCROLLS_AND_STAVES.md).

Python reference campaign_env_magic.py: _can_cast_scroll_spell, _can_cast_staff_spell,
_step_unlock_scroll_magic. Data: GItem.dbf via scripts/import_spell_items.py.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, REST
from stoix.envs.number_grid_buildings import BuildingRules
from stoix.envs.number_grid_casting import SPELL_CAST, SpellCastRules, cast_action_names
from stoix.envs.number_grid_items import ITEM_BY_KEY, SCROLL, STAFF, ItemRules
from stoix.envs.number_grid_spell_items import SCROLL_MAGE_COST, SCROLL_MAGE_HIRED, SPELL_ITEMS
from stoix.envs.number_grid_spells import SPELL_BY_ID, SpellResearchRules
from stoix.tests.number_grid_fixtures import compiled_method


ROOT = Path(__file__).resolve().parents[2]
MANA = jnp.full(5, 5000, jnp.int32)
# A scroll of an own legion spell (its book row) and other factions' spells (item-only rows).
CARRIED = {'ignis_mare_scroll': 2, 'healing_scroll_empire': 1, 'staff_of_thunder': 1, 'staff_of_celerity': 1}


def scroll_rules(env, lord='warrior', **changes):
    config = {**MAP, 'lord_type': lord, 'battle_observation_version': 41, 'initial_items': CARRIED, **changes}
    construction = BuildingRules(config)
    items = ItemRules(config)
    research = SpellResearchRules(config, construction, 195)
    return SpellCastRules(config, research, construction, env.progression, env.territory, env.ruins, 212, items), items


@pytest.mark.core
def test_catalogue_keeps_only_researchable_spells():
    """Keep item-only spells out of research while retaining supported scroll and staff actions."""
    audit = json.loads((ROOT/'docs/validation/spell-item-sources.json').read_text(encoding='utf-8'))
    assert audit['counts'] == dict(scrolls=120, staves=21, supported_scrolls=101, supported_staves=14)
    assert [p['key'] for p in SPELL_ITEMS] == [r['key'] for r in audit['items'] if r['supported']]
    assert all(p['spell'] in SPELL_BY_ID for p in SPELL_ITEMS)
    assert all(r['spell'] not in SPELL_BY_ID for r in audit['items'] if not r['supported'])
    assert [ITEM_BY_KEY[p['key']]['category'] for p in SPELL_ITEMS] == [
        SCROLL if p['category'] == 'scroll' else STAFF for p in SPELL_ITEMS]
    # GItem.dbf: G000IG5011, 600 gold; Staff of Thunder casts Молния for 100 life mana.
    ward, thunder = ITEM_BY_KEY['mind_ward_scroll'], next(p for p in SPELL_ITEMS if p['key'] == 'staff_of_thunder')
    assert (ward['game_id'], ward['price']) == ('G000IG5011', 600)
    assert (thunder['spell'], thunder['cast_mana']) == ('g000ss0004', [0, 100, 0, 0, 0])
    # No supported staff is level V, so the ruler's level-V block never applies to staves.
    assert max(SPELL_BY_ID[p['spell']]['level'] for p in SPELL_ITEMS if p['category'] == 'staff') == 4


def test_hire_scroll_consumption_and_staff_access(current_game):
    env, initial, _, _ = current_game
    rules, items = scroll_rules(env)
    keys = [p['key'] for p in rules.items]
    assert cast_action_names({**MAP, 'initial_items': CARRIED})[rules.book_count:] == (
        'hire_scroll_mage',) + tuple('cast_'+k for k in keys)
    assert rules.book_count == 17 and rules.count == 20 and not rules.scroll_magic_start
    ids = [row['id'] for row in rules.rows]
    action = {k: rules.item_start+i for i, k in enumerate(keys)}
    available, valid, apply = jax.jit(rules.available), jax.jit(rules.valid), jax.jit(rules.apply)
    max_hp = compiled_method(env, 'max_hp')
    movement_cap = compiled_method(env, 'movement_cap')
    quotes = jax.jit(lambda s: rules.quotes(s, env.max_hp(s), env.movement_cap(s)))

    def status(state):
        return dict(zip(keys, quotes(state)['item_status'].tolist()))

    def check(state):
        """The mask and the step's single-action check agree on every cast action."""
        maximum, cap = max_hp(state), movement_cap(state)
        mask = available(state, maximum, cap)
        for a in range(rules.start, rules.end):
            assert bool(valid(state, jnp.int32(a), maximum, cap)) == bool(mask[a-rules.start])
        return {k: bool(mask[action[k]-rules.start]) for k in keys}, bool(mask[rules.hire_action-rules.start])

    def cast(state, a):
        return apply(state, jnp.int32(a), max_hp(state), movement_cap(state))

    state = initial.replace(item_inventory=items.initial_inventory, mana=MANA, gold=jnp.int32(SCROLL_MAGE_COST),
                            spell_casts=jnp.zeros(rules.book_count, jnp.int32))
    # Locked until a reader is hired: once, for 500 gold, while a scroll or staff is owned.
    usable, hire = check(state)
    assert not any(usable.values()) and hire and set(status(state).values()) == {1}
    assert not check(state.replace(gold=jnp.int32(SCROLL_MAGE_COST-1)))[1]
    assert not check(state.replace(item_inventory=jnp.full_like(state.item_inventory, -1)))[1]
    hired, reward, _ = cast(state, rules.hire_action)
    assert hired.scroll_magic and hired.gold == 0 and hired.last_event == SCROLL_MAGE_HIRED and reward == 0
    usable, hire = check(hired)
    assert not hire and usable == dict(ignis_mare_scroll=True, healing_scroll_empire=False,
                                       staff_of_thunder=False, staff_of_celerity=False)
    # Staff of Thunder needs a mage hero, a mage ruler or sorcery lore (hero level 10).
    assert status(hired) == dict(ignis_mare_scroll=0, healing_scroll_empire=4, staff_of_thunder=7, staff_of_celerity=7)
    assert check(hired.replace(unit_levels=hired.unit_levels.at[1].set(10)))[0]['staff_of_thunder']
    mage, _ = scroll_rules(env, 'mage')
    assert bool(jax.jit(mage.staff_access)(hired)) and not bool(jax.jit(rules.staff_access)(hired))

    # A scroll: no mana, no daily limit, the first copy burns; it hits like the book spell.
    once, reward, _ = cast(hired, action['ignis_mare_scroll'])
    assert reward == pytest.approx(rules.reward) and once.last_event == SPELL_CAST
    assert once.last_cast_spell == ids.index('g000ss0043') and once.last_cast_item == keys.index('ignis_mare_scroll')
    assert once.enemy_wounds[0, 0] == 15 and (once.mana == MANA).all() and not once.spell_casts.any()
    assert jnp.sum(once.item_inventory == items.keys.index('ignis_mare_scroll')) == 1
    twice, _, _ = cast(once, action['ignis_mare_scroll'])
    assert twice.enemy_wounds[0, 0] == 30 and check(twice)[0]['ignis_mare_scroll'] is False
    assert status(twice)['ignis_mare_scroll'] == 2 and (twice.item_inventory[:3] >= 0).all()
    assert (twice.item_inventory[3:] == -1).all()  # inventory stays compact, acquisition order kept

    # Healing needs a wounded unit; the healing scroll is an item-only row.
    wounded = twice.replace(hp=twice.hp.at[0].add(-40))
    assert check(wounded)[0]['healing_scroll_empire']
    healed, _, _ = cast(wounded, action['healing_scroll_empire'])
    assert healed.hp[0] == wounded.hp[0]+30 and healed.last_cast_spell == ids.index('g000ss0007')

    # A staff stays, costs its own CASTING mana each time, with no daily limit.
    lore = healed.replace(unit_levels=healed.unit_levels.at[1].set(10))
    struck, _, _ = cast(lore, action['staff_of_thunder'])
    assert (struck.mana == MANA.at[1].add(-100)).all() and (struck.item_inventory == lore.item_inventory).all()
    again, _, _ = cast(struck, action['staff_of_thunder'])
    assert (again.mana == MANA.at[1].add(-200)).all() and again.last_cast_item == keys.index('staff_of_thunder')
    assert status(lore.replace(mana=jnp.zeros(5, jnp.int32)))['staff_of_thunder'] == 3
    # A support effect does not stack: one celerity buff until the next rest.
    buffed, _, _ = cast(again, action['staff_of_celerity'])
    assert buffed.spell_effects == 1 << ids.index('g000ss0002') and not check(buffed)[0]['staff_of_celerity']
    assert check(jax.jit(rules.rest)(buffed, jnp.bool_(True)))[0]['staff_of_celerity']
    assert rules.observation(buffed).shape == (rules.observation_size,)
    started, _ = scroll_rules(env, starting_scroll_magic_unlocked=True)
    assert started.scroll_magic_start


@pytest.mark.core
def test_default_map_hires_reader_and_reads_scroll(default_game):
    """Exercise hiring a scroll reader and consuming a scroll through Default's public step."""
    env, initial, step, mask = default_game
    rules = env.casting
    scroll = env.item_rules.keys.index('mind_ward_scroll')
    row = [r['id'] for r in rules.rows].index('g000ss0011')  # the Empire's own ward: its book row
    state = initial.replace(item_inventory=initial.item_inventory.at[0].set(scroll), gold=jnp.int32(SCROLL_MAGE_COST))
    assert not initial.scroll_magic and mask(state)[rules.hire_action] and not mask(state)[rules.item_start]
    locked, _ = step(state, jnp.int32(rules.item_start))
    assert locked.item_inventory[0] == scroll and locked.spell_effects == 0
    hired, ts = step(state, jnp.int32(rules.hire_action))
    assert hired.scroll_magic and hired.gold == 0 and hired.last_event == SCROLL_MAGE_HIRED
    assert float(ts.reward) == pytest.approx(-env.step_cost)
    assert not mask(hired)[rules.hire_action] and mask(hired)[rules.item_start]
    ward, ts = step(hired, jnp.int32(rules.item_start))
    assert ward.item_inventory[0] == -1 and ward.spell_effects == 1 << row and ward.last_event == SPELL_CAST
    assert ward.last_cast_item == 0 and (ward.mana == hired.mana).all()
    assert float(ts.reward) == pytest.approx(rules.reward-env.step_cost)
    # The same ward from the book does not stack until the next rest.
    learned = ward.replace(learned_spells=jnp.uint32(1 << row), mana=MANA)
    assert not mask(learned)[rules.start+row] and mask(step(learned, jnp.int32(REST))[0])[rules.start+row]
    assert json.dumps(rules.metadata)
