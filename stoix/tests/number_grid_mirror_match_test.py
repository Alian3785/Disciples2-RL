"""Mirror match CUDA contracts. Sources: docs/MIRROR_MATCH.md.

Core cases share one real map and one compiled bot turn, including real combat.
Full public-step integrations, snapshot import and forced timeout are extended.
No planner trains the learner.
"""
import hashlib
import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from scripts.build_mirror_match import build
from stoix.envs.number_grid import NumberGrid, load_map, DEFEND, REST
from stoix.envs.number_grid_bot import HUNTING, RETURNING, RESTING, DEFEATED
from stoix.envs.number_grid_casting import DAMAGE
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS, campaign_events
from stoix.tests.number_grid_fixtures import compiled_method


@pytest.fixture(scope='module')
def mirror():
    """Share the complete Mirror match environment and a reproducible CUDA reset state."""
    assert jax.default_backend() == 'gpu', 'Mirror match tests require CUDA'
    env = NumberGrid(map_config=load_map('mirror_match'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, state


@pytest.fixture(scope='module')
def navigate(mirror):
    """Reuse one JIT navigation program across movement and encounter scenarios."""
    env, _ = mirror
    return jax.jit(lambda state: env.bot.navigate(env, state))


@pytest.fixture(scope='module')
def bot_turn(mirror):
    """Reuse the real compiled daily bot transition, including its tactical engine."""
    env, _ = mirror
    return jax.jit(lambda state: env.bot.advance(env, state))


def only_bot(env, state, position=(12, 12), hero=(12, 30)):
    """Place the hero and bot for an isolated scenario with all neutral armies removed."""
    return state.replace(alive=jnp.arange(env.num_opponents) == env.bot.enemy,
        position=jnp.array(hero, jnp.int32),
        enemy_positions=state.enemy_positions.at[env.bot.enemy].set(jnp.array(position, jnp.int32)))


@pytest.mark.core
def test_map_layout_and_large_units(mirror):
    """Pin mirrored formations, capital coordinates, large-unit slots, resources, and CLI aliases."""
    env, state = mirror
    m = env.map_config
    assert m['size'] == 50 and m['agent_position'] == [6, 6]
    assert m['scripted_bot']['home'] == [43, 43]
    assert m['reference_enemy_ids'] == list(range(101, 120))+list(range(201, 220))+[250, -75]
    assert len(m['opponent_positions']) == 40
    for i in range(19):
        assert m['opponent_positions'][i+19] == [49-v for v in m['opponent_positions'][i]]
        assert m['enemy_rosters'][i+19] == m['enemy_rosters'][i]
    assert m['opponent_positions'][38] == [25, 25]
    assert m['enemy_units'][38] == 1  # Tiamat is one fighter occupying two cells
    for squad in env.progression.enemy_ids.tolist():
        for slot, uid in enumerate(squad):
            if uid and env.progression.rows[uid].get('size', 1) == 2:
                assert slot < 3 and squad[slot+3] == 0
    assert not m['cities'] and not m['chests'] and not m['ruins']
    assert len(m['mines']) == 4
    names = [env.progression.rows[i]['name'] for i in state.bot_ids.tolist() if i]
    assert names == ['Скваер', 'Рыцарь на пегасе', 'Скваер', 'Служка']
    assert load_map('mirrow_match') == load_map('mirror-match') == load_map('mirror_match')


def test_import_matches_reference_snapshot():
    """Require the map importer to reproduce the pinned local snapshot and its source digest."""
    source = Path('C:/Disciples 2 python reference/Big_map/maps/mirror_match/layout.json')
    if not source.is_file():
        source = Path('/mnt/c/Disciples 2 python reference/Big_map/maps/mirror_match/layout.json')
    if not source.is_file():
        pytest.skip('Local Python reference snapshot is not mounted')
    data = source.read_bytes()
    assert build(json.loads(data), hashlib.sha256(data).hexdigest()) == load_map('mirror_match')


@pytest.mark.core
def test_exclusive_growth_ties_income_and_read_only_queries(mirror):
    """Check exclusive land, stable ties, mine income, water barriers, and ownership-based healing."""
    env, state = mirror
    t = env.territory
    grow = jax.jit(t.next_turn)
    state = grow(state)
    assert state.territory_claims.tolist() == [11]
    assert state.opponent_claims.tolist() == [11]
    assert int(jnp.sum(state.territory_owners == 1)) == 11
    assert int(jnp.sum(state.territory_owners == 2)) == 11
    full = state.replace(territory_claims=t.reachable_counts, opponent_claims=t.opponent_counts)
    full = jax.jit(t.repaint)(full)
    d1, d2 = t.distances[0], t.opponent_distances[0]
    assert jnp.all(full.territory_owners[d1 < d2] == 1)
    assert jnp.all(full.territory_owners[d2 < d1] == 2)
    # Equal distances preserve an existing owner and leave unpainted land neutral.
    ties = (d1 == d2) & (d1 < env.size**2)
    assert jnp.any(ties)
    for owner in (0, 1, 2):
        painted = full.replace(territory_owners=jnp.where(ties, owner, full.territory_owners))
        after = grow(painted)
        assert jnp.all(after.territory_owners[ties] == owner)
    quotes = jax.jit(t.quotes)(full)
    assert quotes['mine_owner'].tolist() == [1, 2, 1, 2]
    assert quotes['income'].tolist() == [150, 75, 0, 0, 0, 0]
    before = full.territory_owners
    jax.jit(t.observation)(full)
    jax.jit(t.regeneration_percent)(full)
    chex.assert_trees_all_equal(before, full.territory_owners)
    assert not jnp.any(quotes['territory'] & quotes['opponent_territory'])
    for water in env.map_config['terrain']['water']:
        assert full.territory_owners[water[0]*env.size+water[1]] == 0
    # Only the agent's ownership grants the extra 10% land regeneration.
    position = t.mine_positions[0]
    cell = position[0]*env.size+position[1]
    own = full.replace(position=position, territory_owners=full.territory_owners.at[cell].set(1))
    enemy = own.replace(territory_owners=own.territory_owners.at[cell].set(2))
    regen = jax.jit(t.regeneration_percent)
    chex.assert_trees_all_equal(regen(own)-regen(enemy), jnp.where(full.unit_ids != 0, 10, 0))


@pytest.mark.core
def test_bfs_neutral_priority_budget_and_blockers(mirror, navigate):
    """Check neutral priority, source-ID ties, legal paths, and the bot's movement budget."""
    env, initial = mirror
    state = only_bot(env, initial, hero=(12, 13))
    # Hero is adjacent but an eligible neutral remains, so it retains priority.
    state = state.replace(alive=state.alive.at[0].set(True),
        enemy_positions=state.enemy_positions.at[0].set(jnp.array([12, 27])))
    moved, target, fight = navigate(state)
    assert target == 0 and not fight
    assert moved.bot_moves_spent == 20  # ten steps with the living flying leader
    assert not moved.pending_encounters[env.bot.enemy]
    assert jnp.any(moved.enemy_positions[env.bot.enemy] != state.position)
    assert not env.territory.blocked(moved.enemy_positions[env.bot.enemy])
    # Equidistant neutrals use source enemy ID; neither can be walked through.
    tied = only_bot(env, initial, hero=(5, 30))
    tied = tied.replace(alive=tied.alive.at[0:2].set(True),
        enemy_positions=tied.enemy_positions.at[0].set(jnp.array([12, 14])).at[1].set(jnp.array([14, 12])))
    moved, target, fight = navigate(tied)
    assert target == 0 and fight
    assert jnp.max(jnp.abs(moved.enemy_positions[env.bot.enemy]-tied.enemy_positions[0])) == 1


def test_pursuit_opens_agent_battle_and_home_return_does_not_overlap(mirror, navigate):
    """Exercise REST-to-combat through public step and prevent home-return overlap with the hero."""
    env, initial = mirror
    step = compiled_method(env, 'step')
    state = only_bot(env, initial, position=(12, 12), hero=(12, 13))
    after, _ = step(state, jnp.int32(REST))
    assert after.in_battle and after.enemy == env.bot.enemy
    chex.assert_trees_all_equal(after.position, state.position)
    assert after.enemy_positions[env.bot.enemy].tolist() == [12, 12]
    assert not after.pending_encounters[env.bot.enemy]
    # Return home occupied by the hero also stops adjacent and queues combat.
    home = env.bot.home
    returning = only_bot(env, initial).replace(position=home, bot_mode=jnp.int32(RETURNING),
        enemy_positions=initial.enemy_positions.at[env.bot.enemy].set(home+jnp.array([-1, 0])))
    moved, _, fight = navigate(returning)
    assert not fight and moved.bot_mode == RETURNING
    assert moved.pending_encounters[env.bot.enemy]
    assert jnp.any(moved.enemy_positions[env.bot.enemy] != home)


@pytest.mark.core
def test_respawn_and_home_rest_restore_only_bot(mirror, bot_turn):
    """Check the defeat countdown and home recovery while preserving player state and bot progress."""
    env, initial = mirror
    defeated = env.bot.mark_defeated(initial)
    first = bot_turn(defeated)
    assert first.bot_mode == DEFEATED and first.bot_timer == 1
    repeated = env.bot.mark_defeated(first)
    assert repeated.bot_timer == 1
    second = bot_turn(repeated)
    assert second.bot_mode == HUNTING and second.alive[env.bot.enemy]
    assert second.bot_moves_spent == 0 and not second.in_battle
    blocked = bot_turn(repeated.replace(position=env.bot.home))
    assert blocked.pending_encounters[env.bot.enemy]
    assert not compiled_method(env, '_campaign_won')(defeated)
    chex.assert_trees_all_equal(second.bot_ids, initial.bot_ids)
    chex.assert_trees_all_equal(second.enemy_progress[env.bot.enemy], initial.enemy_progress[env.bot.enemy])
    # A full home rest revives casualties and retains upgrades/XP.
    maximum = env.bot.maximum(initial)
    wounded = initial.replace(bot_mode=jnp.int32(RESTING), bot_timer=jnp.int32(1),
        enemy_wounds=initial.enemy_wounds.at[env.bot.enemy].set(maximum))
    arrived = bot_turn(wounded.replace(bot_mode=jnp.int32(RETURNING), bot_timer=jnp.int32(0)))
    assert arrived.bot_mode == RESTING and arrived.bot_timer == 1
    chex.assert_trees_all_equal(arrived.enemy_wounds, wounded.enemy_wounds)
    restored = bot_turn(arrived)
    assert restored.bot_mode == HUNTING
    assert not jnp.any(restored.enemy_wounds[env.bot.enemy])
    chex.assert_trees_all_equal(restored.hp, initial.hp)
    chex.assert_trees_all_equal(restored.enemy_progress, wounded.enemy_progress)


@pytest.mark.core
def test_neutral_combat_uses_engine_promotes_without_buildings_and_preserves_player(mirror, bot_turn):
    """Resolve a real neutral fight and require bot upgrades without player rewards or state changes."""
    env, initial = mirror
    b = env.bot.enemy
    state = only_bot(env, initial, position=(12, 12), hero=(5, 30))
    maximum = env.casting.enemy_max_hp(state)[0]
    xp = jnp.maximum(0, env.progression.required_xp(state.bot_ids, state.enemy_progress[b, :, 0])-1)
    state = state.replace(alive=state.alive.at[0].set(True),
        enemy_positions=state.enemy_positions.at[0].set(jnp.array([12, 13])),
        enemy_wounds=state.enemy_wounds.at[0].set(jnp.maximum(0, maximum-1)),
        enemy_progress=state.enemy_progress.at[b, :, 1].set(xp),
        buildings=jnp.uint32(0))
    after = bot_turn(state)
    assert not after.alive[0] and after.alive[b] and after.bot_mode == RETURNING
    assert after.bot_defeated_enemies == 1
    assert jnp.any(after.bot_ids != state.bot_ids)
    for name in ('position', 'hp', 'unit_ids', 'unit_levels', 'unit_xp', 'gold', 'mana',
                 'item_inventory', 'buildings', 'reference_defeated', 'battle_steps', 'player_turns',
                 'enemy_turns', 'bot_hero_victories'):
        chex.assert_trees_all_equal(getattr(after, name), getattr(state, name))
    assert not jnp.array_equal(after.battle_key, state.battle_key)
    assert not after.done and not after.won
    metrics = campaign_events(env, state, after, after, jnp.int32(REST),
                              jnp.bool_(True), jnp.zeros(4, jnp.int32))
    assert metrics[CAMPAIGN_FIELDS.index('enemy_squads_defeated')] == 0
    assert metrics[CAMPAIGN_FIELDS.index('battle_victories')] == 0


@pytest.mark.core
def test_bot_autobattle_does_not_inherit_campaign_experience_bonus(mirror):
    """Exclude the campaign hero's experience multiplier from the private bot battle."""
    env, initial = mirror
    high_level = initial.replace(unit_levels=initial.unit_levels.at[:6].set(12))
    bonus = compiled_method(env, '_party_experience_bonus')
    assert bonus(high_level)
    assert not bonus(high_level.replace(bot_autobattle=jnp.bool_(True)))


def test_bot_timeout_preserves_neutral_wounds_and_starts_full_respawn_timer(mirror, monkeypatch):
    """Limit bot decisions without inventing a victory or removing surviving neutral fighters."""
    env, initial = mirror
    state = only_bot(env, initial).replace(alive=initial.alive.at[env.bot.enemy].set(True))
    # The limit expires after initial red automatic activations. No guaranteed
    # winner is fabricated and the neutral's living units must not disappear.
    monkeypatch.setattr(env.bot, 'battle_decisions', 0)
    after = jax.jit(lambda s: env.bot._autobattle(env, s, jnp.int32(38)))(state)
    assert after.bot_mode == DEFEATED and after.bot_timer == 2
    assert after.alive[38] and not after.alive[env.bot.enemy]
    assert jnp.any(env.casting.enemy_max_hp(after)[38] > after.enemy_wounds[38])
    assert not after.done and not after.won


@pytest.mark.parametrize('summon', [False, True])
def test_only_hero_battle_completes_objective(mirror, summon):
    """Require hero victory to terminate the campaign; a summon win only defeats the opponent."""
    env, initial = mirror
    party = jnp.stack((initial.unit_ids[:6], initial.unit_levels[:6], initial.unit_xp[:6], initial.hp[:6]))
    before = initial.replace(enemy=jnp.int32(env.bot.enemy), summon_battle=jnp.bool_(summon), spell_party=party)
    battle = compiled_method(env, '_begin_battle')(before)
    battle = battle.replace(actor=jnp.int32(0), hp=battle.hp.at[6:].set(0))
    after, timestep = compiled_method(env, 'step')(battle, jnp.int32(DEFEND))
    assert after.won == (not summon) and after.done == (not summon)
    assert after.bot_hero_victories == int(not summon)
    assert after.bot_mode == DEFEATED and after.bot_timer == 2
    assert timestep.terminated() == (not summon) and not timestep.truncated()
    assert timestep.last() == (not summon)
    assert float(timestep.discount) == float(summon)  # Only a hero victory stops bootstrapping.


def test_max_steps_truncates_without_victory_and_preserves_bootstrap(mirror):
    """The public step truncates at its decision budget and retains value bootstrap."""
    env, initial = mirror
    step, mask = compiled_method(env, 'step'), compiled_method(env, 'action_mask')
    moves = mask(initial)[:8]
    assert jnp.any(moves)
    action = jnp.argmax(moves).astype(jnp.int32)
    before = initial.replace(step_count=jnp.int32(env.max_steps-2))
    penultimate, ongoing = step(before, action)
    assert penultimate.step_count == env.max_steps-1 and not penultimate.done
    assert not penultimate.in_battle and not ongoing.last()
    assert float(ongoing.discount) == 1.

    back = (action+4) % 8
    assert mask(penultimate)[back]
    after, timestep = step(penultimate, back)
    assert after.step_count == env.max_steps and after.done
    assert not after.won and not after.lost and after.bot_hero_victories == 0
    assert timestep.last() and timestep.truncated() and not timestep.terminated()
    assert float(timestep.discount) == 1.  # Keep the final state's value in the bootstrap target.
    assert timestep.observation.shape == (env.observation_size,)
    assert jnp.all(jnp.isfinite(timestep.observation))
    chex.assert_trees_all_equal(after.position, initial.position)


def test_magic_kill_respawns_without_victory_and_field_target_is_dynamic(mirror):
    """Exercise spell targeting, lethal damage, and two-day respawn without campaign victory."""
    env, initial = mirror
    b, c = env.bot.enemy, env.casting
    state = only_bot(env, initial, position=(12, 13), hero=(12, 12))
    spell = next(i for i in range(c.book_count) if int(c.kinds[i]) == DAMAGE and bool(c.allowed[i]))
    maximum = c.enemy_max_hp(state)[b]
    state = state.replace(enemy_wounds=state.enemy_wounds.at[b].set(jnp.maximum(0, maximum-1)),
        learned_spells=c.book_bits[spell], mana=jnp.full(5, 9999, jnp.int32))
    field = jax.jit(c.targets)(state)[1]
    assert field == b
    tied = state.replace(alive=state.alive.at[0].set(True),
        enemy_positions=state.enemy_positions.at[0].set(jnp.array([13, 12])))
    assert jax.jit(c.targets)(tied)[0] == b  # source ID -75, not appended row index
    home = state.replace(enemy_positions=state.enemy_positions.at[b].set(env.bot.home))
    assert jax.jit(c.targets)(home)[1] == -1
    after, _ = compiled_method(env, 'step')(state, jnp.int32(c.start+spell))
    assert not after.alive[b] and not after.won and not after.done
    assert after.bot_mode == DEFEATED and after.bot_timer == 2
    assert after.bot_hero_victories == 0
    after, _ = compiled_method(env, 'step')(after, jnp.int32(REST))
    after, _ = compiled_method(env, 'step')(after, jnp.int32(REST))
    assert after.alive[b] and after.bot_mode == HUNTING


@pytest.mark.core
def test_reset_shapes_mask_rng_and_no_neutral_credit(mirror):
    """Check reset features and RNG while requiring a hero victory even after neutrals are gone."""
    env, initial = mirror
    reset = compiled_method(env, 'reset')
    again, timestep = reset(jax.random.PRNGKey(17))
    assert not jnp.array_equal(initial.battle_key, again.battle_key)
    assert again.bot_mode == HUNTING and again.bot_hero_victories == 0
    assert timestep.observation.shape == (env.observation_size,)
    mask = timestep.extras['action_mask']
    assert mask.shape == (env.num_actions,) and mask[REST]
    assert not jnp.any(again.pending_encounters)
    chex.assert_trees_all_equal(initial.territory_owners, again.territory_owners)
    # No neutral objectives remain, but the bot still has to fall in a hero battle.
    only = only_bot(env, again, hero=(5, 30))
    assert not compiled_method(env, '_campaign_won')(only)
    assert only.bot_hero_victories == 0
    assert not timestep.last()


@pytest.mark.core
def test_pursuit_and_blocked_home_queue_encounters(mirror, navigate):
    """Queue hero contact during pursuit or blocked home return without sharing a map cell."""
    env, initial = mirror
    pursuing = only_bot(env, initial, position=(12, 12), hero=(12, 13))
    after, _, neutral_fight = navigate(pursuing)
    assert after.pending_encounters[env.bot.enemy] and not neutral_fight
    chex.assert_trees_all_equal(after.enemy_positions, pursuing.enemy_positions)
    returning = pursuing.replace(position=env.bot.home, bot_mode=jnp.int32(RETURNING),
        enemy_positions=initial.enemy_positions.at[env.bot.enemy].set(env.bot.home+jnp.array([-1, 0])))
    after, _, neutral_fight = navigate(returning)
    assert after.pending_encounters[env.bot.enemy] and not neutral_fight
    assert jnp.any(after.enemy_positions[env.bot.enemy] != env.bot.home)


@pytest.mark.core
@pytest.mark.parametrize('summon', [False, True])
def test_objective_requires_recorded_hero_victory(mirror, summon):
    """Distinguish a recorded hero victory from summon victory or mere absence of live armies."""
    env, initial = mirror
    before = initial.replace(enemy=jnp.int32(env.bot.enemy), summon_battle=jnp.bool_(summon),
        unit_ids=initial.unit_ids.at[6:].set(initial.bot_ids))
    recorded = jax.jit(env.bot.after_battle)(before, before, jnp.bool_(True), jnp.bool_(True))
    assert recorded.bot_hero_victories == int(not summon)
    assert compiled_method(env, '_campaign_won')(recorded) == (not summon)
    assert recorded.bot_mode == DEFEATED and recorded.bot_timer == 2
    absent = initial.replace(alive=jnp.zeros_like(initial.alive))
    assert not compiled_method(env, '_campaign_won')(absent)


@pytest.mark.core
def test_spell_target_follows_bot_position_and_reference_id(mirror):
    """Use the bot's current position, source ID, and home protection when selecting spell targets."""
    env, initial = mirror
    state = only_bot(env, initial, position=(12, 13), hero=(12, 12))
    targets = compiled_method(env.casting, 'targets')
    assert targets(state)[1] == env.bot.enemy
    tied = state.replace(alive=state.alive.at[0].set(True),
        enemy_positions=state.enemy_positions.at[0].set(jnp.array([13, 12])))
    assert targets(tied)[0] == env.bot.enemy
    at_home = state.replace(enemy_positions=state.enemy_positions.at[env.bot.enemy].set(env.bot.home))
    assert targets(at_home)[1] == -1
