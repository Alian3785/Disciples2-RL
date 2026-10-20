"""Reference XP rules, battle integration and growing stats on JAX/CUDA."""
from stoix.tests.number_grid_fixtures import SCENARIO_PARTY, compiled_method
import copy
import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, MAP, SHOOT, CONTINUE, DEFEND, REST, VICTORY
from stoix.envs.number_grid_combat import UNITS


@pytest.fixture(scope='module')
def env(current_game):
    return current_game[0]


def battle(env, enemy=0):
    state, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    return compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(enemy)))


def stack(*states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def win(env, states):
    """Reuse the scalar combat oracle for every independent promotion scenario."""
    attack = compiled_method(env,'_battle_step')
    cases = [jax.tree.map(lambda x,i=i:x[i],states) for i in range(states.hp.shape[0])]
    return stack(*(attack(case,jnp.int32(SHOOT),case.battle_key,jnp.zeros(env.random_size))[0]
                   for case in cases))


def test_initial_experience_and_observation_contract(env):
    state, ts = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    chex.assert_trees_all_equal(compiled_method(env,'unit_experience')(state)[:4],
        jnp.array([[1,25,95,0],[1,60,150,0],[1,25,95,0]]+[[1,20,75,0]]))
    enemy = battle(env, 11)
    chex.assert_trees_all_equal(compiled_method(env,'unit_experience')(enemy)[6:],
        jnp.array([[1,20,70,0],[1,20,80,0]]+[[1,20,70,0]]*4))
    assert ts.observation.shape == (1843,) and env.observation_size == 1843
    encoded = ts.observation[160+5*env.num_opponents:220+5*env.num_opponents].reshape(12,5)
    chex.assert_trees_all_close(encoded[:4,2], jnp.array([.025,.06,.025,.02]))
    chex.assert_trees_all_equal(encoded[4:], jnp.zeros((8,5)))


@pytest.mark.core
def test_xp_is_awarded_once_only_to_surviving_non_escaped_winners(env):
    s = battle(env).replace(actor=jnp.int32(0))
    # 20 XP / 2 survivors gives 10; dead and escaped allies get none.
    s = s.replace(hp=s.hp.at[1].set(0).at[6].set(1), escaped=s.escaped.at[2].set(True))
    won = jax.tree.map(lambda x:x[0], win(env, stack(s)))
    assert won.last_event == VICTORY
    chex.assert_trees_all_equal(won.unit_xp[:6], jnp.array([10,0,0,10,0,0]))
    chex.assert_trees_all_equal(won.last_xp[:6], jnp.array([10,0,0,10,0,0]))
    assert won.hp[1] == 0  # dead units neither earn XP nor revive automatically
    rested, _ = compiled_method(env,'step')(won, jnp.int32(REST))
    chex.assert_trees_all_equal(rested.unit_xp, won.unit_xp)
    assert not jnp.any(rested.last_xp)
    # An action that does not end combat must neither award XP nor promote.
    ongoing, _ = compiled_method(env,'_battle_step')(battle(env).replace(actor=jnp.int32(0)),
        jnp.int32(DEFEND), s.battle_key, jnp.zeros(env.random_size))
    assert not jnp.any(ongoing.unit_xp) and not ongoing.last_promoted


@pytest.mark.core
def test_kill_bank_counts_new_deaths_only_and_uses_exact_half_up(env):
    s = battle(env, 11).replace(actor=jnp.int32(3))
    # Two cultists for the independent half-up rounding scenario.
    s = s.replace(unit_ids=s.unit_ids.at[4].set(s.unit_ids[3]), unit_levels=s.unit_levels.at[4].set(1))
    s = s.replace(hp=s.hp.at[:6].set(jnp.array([0,0,0,30,30,0])).at[6:].set(jnp.array([1,1,0,0,0,0])),
                  battle_xp=jnp.array([0,85]))
    won = jax.tree.map(lambda x:x[0], win(env, stack(s)))
    # Two new kills of 20, not four pre-existing corpses: 125 / 2 -> 63.
    assert won.battle_xp[1] == 125
    chex.assert_trees_all_equal(won.unit_xp[3:5], jnp.array([63,63]))
    assert won.last_xp.sum() == 126


@pytest.mark.core
def test_cap_building_requirement_full_heal_and_discarded_excess(env):
    s = battle(env).replace(actor=jnp.int32(0))
    s = s.replace(hp=s.hp.at[:5].set(10).at[6].set(1),
                  unit_xp=s.unit_xp.at[jnp.array([0,2])].set(94))
    ready = s.replace(buildings=jnp.uint32(1))  # Unholy portal -> berserker
    blocked = ready.replace(blocked_buildings=jnp.uint32(1))
    excess = ready.replace(battle_xp=jnp.array([0,100000]))
    out = win(env, stack(s,ready,blocked,excess))
    for row in (0,2):
        chex.assert_trees_all_equal(out.unit_xp[row,jnp.array([0,2])], jnp.full(2,94))
        chex.assert_trees_all_equal(out.hp[row,:3], jnp.full(3,10))
        assert not out.last_promoted[row]
    for row in (1,3):
        chex.assert_trees_all_equal(out.unit_ids[row,jnp.array([0,2])], jnp.full(2,env.progression.ids['berserker']))
        chex.assert_trees_all_equal(out.unit_xp[row,jnp.array([0,2])], jnp.zeros(2,jnp.int32))
        chex.assert_trees_all_equal(out.hp[row,jnp.array([0,2])], jnp.full(2,170))
        assert out.last_promoted[row] & 5 == 5
    upgraded = jax.tree.map(lambda x:x[1],out)
    chex.assert_trees_all_equal(compiled_method(env,'unit_experience')(upgraded)[0], jnp.array([2,70,550,0]))
    assert compiled_method(env,'unit_stats')(upgraded)[0,1] == 50
    # Construction itself does not spend the old cap or grant a heal/promotion.
    capped = jax.tree.map(lambda x:x[0],out).replace(gold=jnp.int32(200))
    built, _ = compiled_method(env,'step')(capped, jnp.int32(18))
    chex.assert_trees_all_equal(built.unit_ids,capped.unit_ids)
    chex.assert_trees_all_equal(built.unit_xp,capped.unit_xp)
    chex.assert_trees_all_equal(built.hp,capped.hp)
    assert built.buildings == 1


def test_supported_promotion_chain_uses_reference_targets_and_stats(env):
    keys = ['possessed','berserker','dark_paladin','cultist','warlock','demonologist','pandemoneus']
    targets = ['berserker','dark_paladin','infernal_knight','warlock','demonologist','pandemoneus','modeus']
    bits = [0,2,3,4,7,10,12]
    s = battle(env).replace(actor=jnp.int32(0), hp=battle(env).hp.at[:6].set(0).at[0].set(1).at[6].set(1))
    cases = []
    for key,bit in zip(keys,bits):
        cases.append(s.replace(unit_ids=s.unit_ids.at[0].set(env.progression.ids[key]),
            unit_levels=s.unit_levels.at[0].set(UNITS[key]['level']),
            unit_xp=s.unit_xp.at[0].set(UNITS[key]['exp_required']-1), buildings=jnp.uint32(1<<bit)))
    result = win(env,stack(*cases))
    stats = compiled_method(env,'unit_stats',batched=True)(result)
    xp = compiled_method(env,'unit_experience',batched=True)(result)
    for i,key in enumerate(targets):
        assert result.unit_ids[i,0] == env.progression.ids[key]
        assert result.hp[i,0] == UNITS[key]['max_hp'] and result.unit_xp[i,0] == 0
        assert xp[i,0,2] == UNITS[key]['exp_required']
        assert stats[i,0,1] == UNITS[key]['damage']
    assert env.unit_traits(jax.tree.map(lambda x:x[-1],result))[0,3] == 4  # Modeus: fire ward


def test_enemy_winners_keep_experience_after_hero_withdrawal(env):
    s = battle(env).replace(actor=jnp.int32(0))
    s = s.replace(hp=s.hp.at[:6].set(0).at[0].set(10),
                  retreating=s.retreating.at[0].set(True), battle_xp=jnp.array([25,0]))
    out, _ = compiled_method(env,'_battle_step')(s,jnp.int32(CONTINUE),s.battle_key,jnp.zeros(env.random_size))
    assert not out.in_battle and not out.done
    assert out.unit_xp[6] == 25 and out.enemy_progress[0,0,1] == 25
    assert not jnp.any(out.unit_xp[:6])
    begun = compiled_method(env,'_begin_battle')(out)
    assert begun.unit_xp[6] == 25 and begun.hp[6] == 100
    assert not jnp.any(begun.battle_xp)


@pytest.mark.core
def test_capital_guards_receive_no_battle_xp_on_either_side(current_game):
    env, initial, _, _ = current_game
    rules = env.progression
    # Installed GUnits.dbf UNIT_CAT=8 identifies the five faction guardians.
    keys = [key for key, row in UNITS.items() if row.get('game_data', {}).get('UNIT_CAT') == '8']
    assert len(keys) == 5
    ids = jnp.tile(jnp.array([rules.ids[key] for key in keys] + [rules.ids['duke']], jnp.int32), 2)
    guards = rules.capital_guards[ids]
    state = initial.replace(enemy=jnp.int32(0), unit_ids=ids, unit_levels=rules.base_levels[ids],
        unit_xp=jnp.where(guards, 7, 0).astype(jnp.int32), hp=jnp.ones(12,jnp.int32))
    finish = compiled_method(rules, 'finish')
    for victory, lost, withdrawal in ((True,False,False), (False,True,False), (False,False,True)):
        out_ids, levels, xp, enemies, hp, gains, promoted = finish(
            state, state.hp, state.escaped, jnp.bool_(victory), jnp.bool_(lost),
            jnp.bool_(withdrawal), jnp.full(2,1000000,jnp.int32))
        for actual, expected in ((out_ids,state.unit_ids), (levels,state.unit_levels),
                                 (xp,state.unit_xp), (hp,state.hp)):
            chex.assert_trees_all_equal(actual[guards], expected[guards])
        assert not jnp.any(gains[guards])
        winner = 5 if victory else 11
        assert gains[winner] == 1000000  # Guardians do not dilute their ally's share.
        assert promoted == jnp.uint32(1 << winner)
        assert levels[winner] == state.unit_levels[winner]+1
        chex.assert_trees_all_equal(enemies[0,:5,0], state.unit_levels[6:11])
        chex.assert_trees_all_equal(enemies[0,:5,1], state.unit_xp[6:11])
    alone = state.replace(hp=state.hp.at[5].set(0).at[11].set(0))
    result = finish(alone, alone.hp, alone.escaped, jnp.bool_(False), jnp.bool_(False),
                    jnp.bool_(True), jnp.full(2,1000000,jnp.int32))
    chex.assert_trees_all_equal(result[1], alone.unit_levels)
    chex.assert_trees_all_equal(result[2], alone.unit_xp)
    assert not jnp.any(result[5]) and result[6] == 0
    # Keep the original XP thresholds (used by trainers) and kill rewards.
    experience = compiled_method(rules, 'experience')(ids, state.unit_levels, state.unit_xp)
    chex.assert_trees_all_equal(experience[:,2], rules.required[ids])
    assert jnp.all(experience[guards,1] > 0)


def test_terminal_form_growth_and_regeneration_use_new_maximum(env):
    s = battle(env).replace(actor=jnp.int32(0))
    key = 'infernal_knight'
    s = s.replace(unit_ids=s.unit_ids.at[0].set(env.progression.ids[key]),
        unit_levels=s.unit_levels.at[0].set(10), unit_xp=s.unit_xp.at[0].set(1724),
        hp=s.hp.at[:6].set(0).at[0].set(1).at[6].set(1))
    out = jax.tree.map(lambda x:x[0],win(env,stack(s)))
    # Infernal Knight: above level 10, HP +15, damage +5, kill XP +11.
    assert out.unit_levels[0] == 11 and out.unit_xp[0] == 0
    max_hp = 270+6*25+15
    assert out.hp[0] == max_hp
    chex.assert_trees_all_equal(compiled_method(env,'unit_stats')(out)[0], jnp.array([435,165,86,0,50]))
    assert compiled_method(env,'unit_experience')(out)[0,1] == 215+6*22+11
    assert compiled_method(env,'unit_experience')(out)[0,2] == 1725  # terminal form keeps its XP threshold
    hurt = out.replace(hp=out.hp.at[0].set(1))
    rest, _ = compiled_method(env,'step')(hurt,jnp.int32(REST))
    assert rest.hp[0] == 1+(max_hp*70+99)//100  # native 20 + warrior 15 + capital 35
    next_battle = compiled_method(env,'_begin_battle')(rest.replace(enemy=jnp.int32(1)))
    assert next_battle.unit_ids[0] == out.unit_ids[0] and next_battle.hp[0] == rest.hp[0]
    fresh, _ = compiled_method(env,'reset')(jax.random.PRNGKey(43))
    assert fresh.unit_levels[0] == 1 and not jnp.any(fresh.unit_xp)


def test_current_progression_rewards_pay_once_and_charge_only_discarded_xp(env):
    """User request 2026-10-16: +2 per level, -.05 per blocked XP recipient."""
    s = battle(env).replace(actor=jnp.int32(0))
    s = s.replace(hp=s.hp.at[:5].set(10).at[6].set(1),
                  unit_xp=s.unit_xp.at[jnp.array([0, 2])].set(94))
    advance = compiled_method(env, '_battle_step')
    capped, penalty = advance(s, jnp.int32(SHOOT), s.battle_key, jnp.zeros(env.random_size))
    assert float(penalty) == pytest.approx(1.-2*.05)
    assert not capped.last_promoted and capped.unit_xp[0] == 94
    ready = s.replace(buildings=jnp.uint32(1))
    won, reward = advance(ready, jnp.int32(SHOOT), s.battle_key, jnp.zeros(env.random_size))
    assert float(reward) == pytest.approx(1.+2*2.)
    assert won.last_promoted & 5 == 5
    rested, ts = compiled_method(env, 'step')(won, jnp.int32(REST))
    assert not rested.last_promoted and float(ts.reward) <= 0.
    # Heroes and final forms also earn +2; enemy-only promotions never do.
    rules = env.progression
    zero = jnp.zeros(12, jnp.int32)
    reward_fn = compiled_method(rules, 'training_reward')
    assert float(reward_fn(s, zero, jnp.uint32(2), jnp.bool_(True))) == 2.
    assert float(reward_fn(s, zero, jnp.uint32(64), jnp.bool_(True))) == 0.
    assert float(reward_fn(s, zero, jnp.uint32(0), jnp.bool_(True))) == 0.
    for changes in ({'summon_battle': jnp.bool_(True)}, {'bot_autobattle': jnp.bool_(True)}):
        assert float(reward_fn(s.replace(**changes), zero, jnp.uint32(1), jnp.bool_(True))) == 0.
    assert float(reward_fn(s, zero, jnp.uint32(1), jnp.bool_(False))) == 0.
    for key, level, xp in (('duke', 1, 149), ('infernal_knight', 10, 1724)):
        growing = s.replace(unit_ids=s.unit_ids.at[0].set(rules.ids[key]),
            unit_levels=s.unit_levels.at[0].set(level), unit_xp=zero.at[0].set(xp),
            hp=s.hp.at[:6].set(0).at[0].set(10))
        grown, amount = advance(growing, jnp.int32(SHOOT), s.battle_key, jnp.zeros(env.random_size))
        assert grown.unit_levels[0] == level+1 and float(amount) == pytest.approx(3.)


def test_magistrate_clears_six_slot_xp_blocking_observation(default_game):
    """Expose the actual Inquisitor cap and clear it immediately on construction."""
    env, initial, step, _ = default_game
    rules = env.progression
    slots = jnp.array([0, 2])
    inquisitor = rules.ids['inquisitor']
    prerequisites = sum(1 << i for i, b in enumerate(env.construction.rows)
                        if b['name'] in ('Темница', 'Пыточная камера'))
    capped = initial.replace(unit_ids=initial.unit_ids.at[slots].set(inquisitor),
        unit_levels=initial.unit_levels.at[slots].set(3),
        unit_xp=initial.unit_xp.at[slots].set(949), gold=jnp.int32(1500),
        buildings=jnp.uint32(prerequisites))
    flags = compiled_method(rules, 'blocked_xp_observation')
    chex.assert_trees_all_equal(flags(capped), jnp.array([1., 0., 1., 0., 0., 0.]))
    almost = capped.replace(unit_xp=capped.unit_xp.at[slots].set(948))
    assert not jnp.any(flags(almost))
    observe = compiled_method(env, 'observation')
    delta = observe(capped)-observe(almost)
    assert observe(capped).shape == (env.observation_size,)
    assert int(jnp.sum(delta == 1.)) == 2  # two new flags, distinct from XP fractions
    magistrate = next(i for i, b in enumerate(env.construction.rows) if b['name'] == 'Магистрат')
    built, ts = step(capped, jnp.int32(18+magistrate))
    assert float(ts.reward) == pytest.approx(.049)
    assert not jnp.any(flags(built))
    chex.assert_trees_all_equal(built.unit_levels, capped.unit_levels)
    chex.assert_trees_all_equal(built.unit_xp, capped.unit_xp)
    # Unfixable scenario bans, dead slots and transient copies are not requests to build.
    banned = capped.replace(blocked_buildings=capped.blocked_buildings | jnp.uint32(1 << magistrate))
    assert not jnp.any(flags(banned))
    assert not jnp.any(flags(capped.replace(hp=capped.hp.at[slots].set(0))))
    copied = capped.replace(copied=capped.copied.at[slots].set(True),
        native_ids=capped.native_ids.at[slots].set(initial.unit_ids[1]),
        native_levels=capped.native_levels.at[slots].set(1),
        native_xp=capped.native_xp.at[slots].set(149))
    assert not jnp.any(flags(copied))


def test_dynamic_armor_damage_retains_python_rounding(env):
    s = battle(env)
    ids = s.unit_ids.at[0].set(env.progression.ids['infernal_knight']).at[6].set(env.progression.ids['defender_of_faith'])
    levels = s.unit_levels.at[0].set(11).at[6].set(11)
    s = s.replace(unit_ids=ids,unit_levels=levels)
    damage = jax.jit(env.progression.damage)(s,jnp.int32(0),jnp.full(12,6),
        jnp.repeat(jnp.array([False,True]),6),jnp.tile(jnp.arange(6),2))
    stats = compiled_method(env,'unit_stats')(s)
    assert stats[0,1] == 165 and stats[6,3] == 35
    expected = [round((165+bonus)*.65*(.5 if guard else 1))
                for guard in (False,True) for bonus in range(6)]
    chex.assert_trees_all_equal(damage,jnp.array(expected))


def test_foreign_buildings_do_not_fake_promotions():
    game_map = copy.deepcopy(MAP)
    game_map['faction'] = 'empire'
    game_map['hero_roster'] = SCENARIO_PARTY  # Legions units in an Empire capital
    env = NumberGrid(map_config=game_map)
    s = battle(env).replace(actor=jnp.int32(0))
    s = s.replace(buildings=jnp.uint32(1), unit_xp=s.unit_xp.at[0].set(94), hp=s.hp.at[6].set(1))
    # The cross-faction rule belongs to finish(); victory integration is tested
    # above with the shared current environment, without another combat graph.
    ids, _, xp, *_ = jax.jit(env.progression.finish)(s,s.hp,s.escaped,
        jnp.bool_(True),jnp.bool_(False),jnp.bool_(False),jnp.array([0,25],jnp.int32))
    assert xp[0] == 94 and ids[0] == s.unit_ids[0]
    copier = NumberGrid(map_config={**MAP,'hero_units':5,'hero_roster':['possessed']*3+['doppelganger','cultist',None]})
    assert copier.has_copies


def test_human_snapshot_reports_current_form_and_xp(env, human_service):
    game = human_service
    created = game.create(42)
    assert created['snapshot']['unit_experience'][0] == [1,25,95,0]
    faction,s,total = game.sessions[created['session']]
    s = compiled_method(env,'_begin_battle')(s.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    s = s.replace(buildings=jnp.uint32(1), unit_xp=s.unit_xp.at[0].set(94),hp=s.hp.at[6].set(1))
    out = jax.tree.map(lambda x:x[0],win(env,stack(s)))
    snap = game.snapshot(game.env,out,total)
    assert snap['unit_experience'][0] == [2,70,550,0]
    assert created['combat']['catalogue'][snap['state']['unit_ids'][0]]['name'] == 'Берсерк'
    assert snap['max_hp'][0] == 170


def test_doppelganger_branch_is_available_after_prerequisites(env):
    s, _ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    s = s.replace(gold=jnp.int32(9999))
    unavailable = {u['name'] for u in UNITS.values() if u.get('upgrade_unavailable_reason')}
    blocked_slots = [i for i, row in enumerate(env.construction.rows) if row['unit'] in unavailable]
    assert not blocked_slots
    assert not env.construction.metadata()['buildings'][6]['unavailable_reason']
    s = s.replace(buildings=jnp.uint32(1 << 4))
    assert compiled_method(env,'action_mask')(s)[18+6]
    following,_ = compiled_method(env,'step')(s,jnp.int32(18+6))
    assert following.buildings & (1 << 6) and following.gold < s.gold


@pytest.mark.parametrize('override',[dict(exp_kill=-1),dict(exp_required=1.5),dict(exp_current=95)])
def test_invalid_experience_is_rejected_before_training(override):
    with pytest.raises(ValueError):
        NumberGrid(map_config={**MAP,'hero_combat_stats':[override]+[{}]*3})


@pytest.fixture(scope='module')
def duke_reference():
    return json.loads((Path(__file__).parent / 'data/duke_levels.json').read_text())['levels']


@pytest.mark.core
def test_duke_levels_through_99_match_python_reference(env, duke_reference):
    levels = jnp.arange(1, 100, dtype=jnp.int32)
    ids = jnp.full(99, env.progression.ids['duke'], jnp.int32)
    stats = jax.jit(env.progression.stats)(ids, levels)
    expected = jnp.array([r['stats'] for r in duke_reference[:99]], jnp.float32)
    expected = expected.at[:,1].set(jnp.minimum(expected[:,1],400))
    chex.assert_trees_all_equal(stats, expected)
    xp = jax.jit(env.progression.experience)(ids, levels, jnp.zeros(99,jnp.int32))
    chex.assert_trees_all_equal(xp, jnp.array([
        [r['level'],r['exp_kill'],r['exp_required'],0] for r in duke_reference[:99]]))


def test_duke_levelups_need_no_building_heal_once_and_raise_threshold(env, duke_reference):
    levels = [1,2,3,4,9,10,12,13,14,15,57,98]
    s = battle(env).replace(actor=jnp.int32(1))
    s = s.replace(hp=s.hp.at[:6].set(0).at[1].set(1).at[6].set(1))
    cases = [s.replace(unit_levels=s.unit_levels.at[1].set(level),
        unit_xp=s.unit_xp.at[1].set(duke_reference[level-1]['exp_required']-1)) for level in levels]
    # Even abundant XP grants only one level, as in the reference.
    cases.append(cases[0].replace(battle_xp=jnp.array([0,100000])))
    out = win(env, stack(*cases))
    for i, level in enumerate(levels+[1]):
        expected = duke_reference[level]
        assert out.unit_levels[i,1] == level+1 and out.unit_xp[i,1] == 0
        assert out.hp[i,1] == expected['hp'] and out.last_promoted[i] == 2
        assert out.unit_ids[i,1] == env.progression.ids['duke']
        assert not out.buildings[i]
    xp = compiled_method(env,'unit_experience',batched=True)(out)
    chex.assert_trees_all_equal(xp[:,1,2], jnp.array([duke_reference[l]['exp_required'] for l in levels+[1]]))
    # A subsequent battle needs 650, not the initial 150 XP.
    raised = jax.tree.map(lambda x:x[0],out)
    begun = compiled_method(env,'_begin_battle')(raised.replace(enemy=jnp.int32(2))).replace(actor=jnp.int32(1))
    begun = begun.replace(unit_xp=begun.unit_xp.at[1].set(150),
        hp=begun.hp.at[:6].set(0).at[1].set(10).at[6:].set(0).at[6].set(1))
    won = jax.tree.map(lambda x:x[0],win(env,stack(begun)))
    assert won.unit_levels[1] == 2 and won.unit_xp[1] == 160 and won.hp[1] == 10
    rested, _ = compiled_method(env,'step')(won,jnp.int32(REST))
    assert rested.hp[1] == 101  # 10 + ceil(165*55%); new maximum, capital and warrior lord


def test_duke_damage_cap_and_passive_armor_affect_real_attacks(env):
    s = battle(env,11)
    # Compare an ordinary terminal fighter at the 300 cap and the Duke at 400;
    # strike a level-15 Duke with 20 armour, including defence and all bonuses.
    ids = s.unit_ids.at[0].set(env.progression.ids['infernal_knight']).at[6].set(env.progression.ids['duke'])
    levels = s.unit_levels.at[:2].set(99).at[6].set(15)
    s = s.replace(unit_ids=ids,unit_levels=levels)
    actors = jnp.repeat(jnp.array([0,1]),12)
    guards = jnp.tile(jnp.repeat(jnp.array([False,True]),6),2)
    bonuses = jnp.tile(jnp.arange(6),4)
    actual = jax.jit(env.progression.damage)(s,actors,jnp.full(24,6),guards,bonuses)
    expected = [round((damage+bonus)*.8*(.5 if guard else 1))
        for damage in (300,400) for guard in (False,True) for bonus in range(6)]
    chex.assert_trees_all_equal(actual,jnp.array(expected))
    before = s.replace(unit_levels=s.unit_levels.at[1].set(12))
    after = s.replace(unit_levels=s.unit_levels.at[1].set(13))
    priority = jax.jit(lambda state: env._round_priority(jnp.zeros(env.random_size),state=state))
    assert priority(before)[1] == 50 and priority(after)[1] == 75


def test_human_snapshot_exposes_duke_progress_and_passive_bonuses(env, human_service):
    game = human_service
    created = game.create(42)
    snap = created['snapshot']
    assert snap['unit_experience'][1] == [1,60,150,0]
    row = created['combat']['catalogue'][snap['state']['unit_ids'][1]]
    assert row['name'] == 'Герцог' and row['hero'] and not row['upgrades']
    assert [b['level'] for b in row['level_bonuses']] == [4,5,13,14,15]
    _, state, total = game.sessions[created['session']]
    state = state.replace(unit_levels=state.unit_levels.at[1].set(15))
    snap = game.snapshot(game.env,state,total)
    assert snap['unit_experience'][1] == [15,230,7150,0]
    assert snap['max_hp'][1] == 374


def test_hero_levelup_grants_only_movement_delta_until_level_ten(env):
    initial = battle(env).replace(actor=jnp.int32(0),movement_points=jnp.int32(3))
    hero = int(env._equipment_hero(initial))
    scenarios = []
    for level in (1,9,10,11):
        levels = initial.unit_levels.at[hero].set(level)
        required = env.progression.required_xp(initial.unit_ids[hero],levels[hero])
        scenarios.append(initial.replace(unit_levels=levels,native_levels=levels,
            unit_xp=initial.unit_xp.at[hero].set(required-1),hp=initial.hp.at[6].set(1)))
    output = win(env,stack(*scenarios))
    chex.assert_trees_all_equal(output.unit_levels[:,hero],jnp.array([2,10,11,12]))
    chex.assert_trees_all_equal(output.movement_points,jnp.array([4,4,3,3]))
    chex.assert_trees_all_equal(compiled_method(env,'movement_cap',batched=True)(output),
                               jnp.array([21,29,29,29]))
    for i in range(4):
        state = jax.tree.map(lambda x,i=i:x[i],output)
        # An invalid map command cannot grant the same level bonus a second time.
        again, _ = compiled_method(env,'step')(state,jnp.int32(DEFEND))
        assert again.movement_points == state.movement_points
