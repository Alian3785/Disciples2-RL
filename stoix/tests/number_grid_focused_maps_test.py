"""Focused scenarios: source contracts, public GPU transitions, and 101 scroll effects."""
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.envs.number_grid import NumberGrid, REST, action_names, load_map
from stoix.envs.number_grid_casting import BUFF, DAMAGE, DEBUFF, HEALTH, WARD
from stoix.envs.number_grid_spell_items import SPELL_ITEMS

NAMES = ('scroll_train', 'magic_train', 'item_train', 'hire_train', 'orc_duel', 'formation_train')


def reachable(game_map):
    """Independent eight-direction flood fill of physical map cells."""
    blocked = {tuple(p) for p in game_map.get('obstacles', [])}
    seen = {tuple(game_map['agent_position'])}
    todo = deque(seen)
    while todo:
        r, c = todo.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                point = (r+dr, c+dc)
                if (0 < point[0] < game_map['size']-1 and 0 < point[1] < game_map['size']-1
                        and point not in blocked and point not in seen):
                    seen.add(point)
                    todo.append(point)
    return seen


@pytest.mark.core
def test_focused_source_contracts():
    expected = {
        'scroll_train': (26, 3, 'legions', 'mage'),
        'magic_train': (50, 2, 'legions', 'mage'),
        'item_train': (26, 3, 'legions', 'mage'),
        'hire_train': (26, 2, 'empire', 'warrior'),
        'orc_duel': (50, 1, 'empire', 'warrior'),
        'formation_train': (50, 1, 'undead_hordes', 'mage'),
    }
    for name, (size, armies, faction, lord) in expected.items():
        m = load_map(name)
        assert (m['size'], len(m['enemy_rosters']), m['faction'], m['lord_type']) == (size, armies, faction, lord)
        assert m['equipment_requires_skills'] is True
        assert all(len(roster) == 6 for roster in m['enemy_rosters'])
        assert all(len(digest) == 64 for digest in m['reference_sha256'].values())
        accessible = reachable(m)
        assert all(tuple(mine['position']) in accessible for mine in m['mines'])
        assert all(tuple(chest['position']) in accessible for chest in m['chests'])
        if name == 'magic_train':
            assert all(tuple(p) not in accessible for p in m['opponent_positions'])
        else:
            assert all(tuple(p) in accessible for p in m['opponent_positions'])
    scrolls = load_map('scroll_train')['initial_items']
    assert scrolls == {s['key']: 1 for s in SPELL_ITEMS if s['category'] == 'scroll'}
    assert len(scrolls) == 101
    assert load_map('hire_train')['objective_enemy'] == 0
    assert load_map('hire_train')['hero_roster'] == [None, 'duke', None, None, None, None]
    assert load_map('formation_train')['leadership_step_penalty'] == .06


@pytest.fixture(scope='module', params=NAMES)
def focused(request):
    env = NumberGrid(map_config=load_map(request.param))
    state, _ = jax.jit(env.reset)(jax.random.PRNGKey(42))
    yield request.param, env, state
    jax.clear_caches()


def test_public_reset_rest_and_legal_move(focused):
    """Use public GPU step; verify masks, finite observations and scalar/vmap agreement."""
    name, env, state = focused
    step = jax.jit(env.step)
    assert len(action_names(env.map_config)) == env.num_actions
    assert env.observation(state).shape == (env.observation_size,)
    assert bool(env.action_mask(state)[REST])
    rested, ts = step(state, jnp.int32(REST))
    assert int(rested.day) == int(state.day)+1
    assert not bool(rested.done)
    assert np.isfinite(np.asarray(ts.reward)).all()
    assert np.isfinite(np.asarray(env.observation(rested))).all()
    income = np.asarray(env.territory.income(rested))
    if name in ('scroll_train', 'item_train'):
        assert np.array_equal(np.asarray(rested.mana), np.zeros(5))
        assert np.array_equal(income[1:], np.zeros(5))
    legal_move = next(i for i in range(8) if bool(env.action_mask(rested)[i]))
    moved, _ = step(rested, jnp.int32(legal_move))
    assert not np.array_equal(np.asarray(moved.position), np.asarray(rested.position))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), rested)
    batched, _ = jax.jit(jax.vmap(env.step))(batch, jnp.full(2, legal_move, jnp.int32))
    for one, many in zip(jax.tree.leaves(moved), jax.tree.leaves(batched)):
        np.testing.assert_allclose(np.asarray(many[0]), np.asarray(one), rtol=1e-6, atol=1e-6)


def test_magic_can_win_from_reset_using_only_legal_spells():
    """Real public transitions: income, research, casting, kills and terminal win."""
    env = NumberGrid(map_config=load_map('magic_train'))
    state, _ = jax.jit(env.reset)(jax.random.PRNGKey(9))
    step, mask = jax.jit(env.step), jax.jit(env.action_mask)
    damage_rows = sorted((i for i in range(env.casting.book_count) if int(env.casting.kinds[i]) == DAMAGE),
                         key=lambda i: int(env.casting.amounts[i]), reverse=True)
    casts, learns = 0, 0
    strongest = damage_rows[0]
    for _ in range(250):
        available = np.asarray(mask(state))
        # Save mana for a spell that exceeds the enemies' daily regeneration.
        # Spending it on the first affordable weak spell can stall indefinitely.
        cast = env.casting.start+strongest if available[env.casting.start+strongest] else None
        learn = env.spell_research.start+strongest if available[env.spell_research.start+strongest] else None
        tower = env.spell_research.metadata['tower_action']
        action = cast if cast is not None else learn if learn is not None else tower if available[tower] else REST
        assert available[action]
        casts += cast is not None
        learns += cast is None and learn is not None
        state, ts = step(state, jnp.int32(action))
        assert np.isfinite(float(ts.reward))
        assert not bool(state.in_battle)
        if bool(state.done):
            break
    assert bool(state.won) and not bool(state.lost), (int(state.day), casts, learns,
                                                     np.asarray(state.mana), np.asarray(state.enemy_wounds))
    assert casts > 0 and learns > 0 and not np.asarray(state.alive).any()
    assert list(np.asarray(state.position)) == env.map_config['agent_position']
    after, ts = step(state, jnp.int32(REST))
    assert bool(after.won) and float(ts.reward) == 0.
    jax.clear_caches()


def test_all_scroll_words_and_daily_reset():
    """High spell rows must be distinct, nonrenewable, and included in battle modifiers."""
    env = NumberGrid(map_config=load_map('scroll_train'))
    state, _ = jax.jit(env.reset)(jax.random.PRNGKey(1))
    rules = env.casting
    assert rules.count == 101 and rules.effect_shape == (4,)
    assert state.enemy_spell_effects.shape == (3, 4)
    assert int(jnp.sum(state.item_inventory >= 0)) == 101
    assert bool(state.scroll_magic)
    step = jax.jit(env.step)
    selected = []
    for low, high in ((0, 32), (32, 64), (64, 96), (96, 101)):
        rows = [i for i in range(low, high) if int(rules.kinds[i]) in (BUFF, HEALTH, WARD)
                and int(rules.item_rows.tolist().index(i)) >= 0]
        index = next(i for i in rows if bool(env.action_mask(state)[rules.item_start+rules.item_rows.tolist().index(i)]))
        item = rules.item_rows.tolist().index(index)
        action = rules.item_start+item
        before = state
        state, ts = step(state, jnp.int32(action))
        assert int(state.last_cast_spell) == index and np.isfinite(float(ts.reward))
        assert int(jnp.sum(state.item_inventory >= 0)) == int(jnp.sum(before.item_inventory >= 0))-1
        assert np.array_equal(np.asarray(before.mana), np.asarray(state.mana))
        selected.append(index)
        assert all(bool(rules.effect_active(state.spell_effects, rules.bits[i])) for i in selected)
        # Restore inventory only: the effect still prevents a second application.
        restored = state.replace(item_inventory=before.item_inventory)
        assert not bool(env.action_mask(restored)[action])
    active = np.asarray(rules.effect_active(state.spell_effects, rules.bits))
    assert set(np.flatnonzero(active)) == set(selected)
    _, hp = rules.begin_battle(state, state.hp)
    assert np.isfinite(np.asarray(hp)).all()
    rested, _ = step(state, jnp.int32(REST))
    assert not np.asarray(rested.spell_effects).any()
    assert not np.asarray(rested.enemy_spell_effects).any()
    # Debuffs in later words belong only to the targeted army.
    index = next(i for i in range(64, rules.count) if int(rules.kinds[i]) == DEBUFF)
    action = rules.item_start+rules.item_rows.tolist().index(index)
    debuffed, _ = step(rested, jnp.int32(action))
    target = int(debuffed.last_spell_target)
    assert target >= 0
    assert bool(rules.effect_active(debuffed.enemy_spell_effects[target], rules.bits[index]))
    assert int(jnp.sum(jnp.any(debuffed.enemy_spell_effects != 0, axis=-1))) == 1
    jax.clear_caches()
