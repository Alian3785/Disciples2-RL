"""Capital recovery reference rules and user integer-HP override, on CUDA only."""
from collections import OrderedDict

from stoix.tests.number_grid_fixtures import compiled_method, enemy_roster_state
import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, REST, SHOOT, CONTINUE, DEFEND, MAP, ROUND_LIMIT_RETREAT
from stoix.envs.number_grid_capital import CapitalRules, HEAL_START, REVIVE_START
from stoix.envs.number_grid_buildings import BuildingRules, FACTIONS


@pytest.fixture(scope='module')
def game(current_game):
    env, state, advance, battle_step = current_game
    state = state.replace(buildings=env.capital.temple_bit, gold=jnp.int32(1000),
                          recovery_balance=jnp.array([100, 2], jnp.int32))
    return env, state, advance, battle_step


def test_reference_prices_and_faction_temple_mapping(game):
    env, state, _, _ = game
    expected = dict(possessed=[1,50], cultist=[1,50], duke=[1,200], berserker=[2,200],
                    dark_paladin=[2,400], infernal_knight=[3,600], titan=[2,200],
                    acolyte=[1,50], orc=[2,200], goblin=[1,50], goblin_archer=[1,50])
    for name, prices in expected.items():
        assert env.capital.prices[env.progression.ids[name]].tolist() == prices
    for faction, slot in zip(FACTIONS, (24,23,24,24,22)):
        construction = BuildingRules({**MAP, 'faction': faction})
        capital = CapitalRules(env.progression, construction, MAP['agent_position'])
        assert capital.temple_slot == slot and construction.rows[slot]['gold'] == 300
        assert capital.metadata()['temple_action'] == 18+slot
    high_level = state.replace(unit_levels=state.unit_levels.at[1].set(15))
    assert env.capital.quotes(high_level, compiled_method(env,'max_hp')(high_level))[1, [0,4]].tolist() == [1,200]


@pytest.mark.core
def test_services_require_exact_capital_temple_money_and_correct_target(game):
    env, state, advance, _ = game
    wounded = state.replace(hp=state.hp.at[0].set(100).at[1].set(0))
    mask = compiled_method(env,'action_mask')(wounded)
    assert mask[HEAL_START] and mask[REVIVE_START+1]
    invalid = [(wounded.replace(position=state.position+jnp.array([0,1])), HEAL_START),
               (wounded.replace(buildings=jnp.uint32(0)), HEAL_START),
               (wounded.replace(gold=jnp.int32(0)), HEAL_START),
               (wounded.replace(gold=jnp.int32(199)), REVIVE_START+1),
               (wounded, HEAL_START+1), (wounded, REVIVE_START),
               (wounded, HEAL_START+2), (wounded, REVIVE_START+5),
               (wounded.replace(summon_owner=wounded.summon_owner.at[1].set(1)), REVIVE_START+1),
               (wounded.replace(summon_battle=jnp.bool_(True)), REVIVE_START+1),
               (wounded.replace(done=jnp.bool_(True)), REVIVE_START+1),
               (compiled_method(env,'_begin_battle')(wounded.replace(enemy=jnp.int32(0))), HEAL_START)]
    for before, action in invalid:
        assert not compiled_method(env,'action_mask')(before)[action]
        after, _ = advance(before, jnp.int32(action))
        for field in ('hp','gold','buildings','movement_points','day','recovery_balance','battle_key'):
            chex.assert_trees_all_equal(getattr(after,field), getattr(before,field))
    assert compiled_method(env,'observation')(state).shape == (1844,) and env.num_actions == 244
    # Capital follows the actual scenario spawn, not a hard-coded (2, 2).
    other = CapitalRules(env.progression, env.construction, [4,4])
    assert not other.at_capital(state) and other.at_capital(state.replace(position=jnp.array([4,4])))


@pytest.mark.core
def test_integer_partial_healing_change_cap_and_no_turn_cost(game):
    env, state, advance, _ = game
    unit_id = env.progression.ids['berserker']
    wounded = state.replace(unit_ids=state.unit_ids.at[0].set(unit_id),
        unit_levels=state.unit_levels.at[0].set(2), hp=state.hp.at[0].set(150), gold=jnp.int32(5),
        movement_points=jnp.int32(0), built_today=jnp.bool_(True))
    healed, ts = advance(wounded, jnp.int32(HEAL_START))
    assert healed.hp[0] == 152 and healed.gold == 1 and healed.last_service_cost == 4
    assert float(ts.reward) == pytest.approx(.0002-env.step_cost)
    assert not compiled_method(env,'action_mask')(healed)[HEAL_START]  # change cannot buy another HP
    for field in ('movement_points','day','built_today','buildings','map_steps','battle_key'):
        chex.assert_trees_all_equal(getattr(healed,field), getattr(wounded,field))
    full, _ = advance(wounded.replace(hp=wounded.hp.at[0].set(169), gold=jnp.int32(1000)), jnp.int32(HEAL_START))
    assert full.hp[0] == 170 and full.gold == 998
    assert not compiled_method(env,'action_mask')(full)[HEAL_START]


@pytest.mark.core
def test_paid_revive_returns_one_hp_and_keeps_level_experience(game):
    env, state, advance, _ = game
    for slot, cost in ((0,50),(1,200)):
        dead = state.replace(hp=state.hp.at[slot].set(0),
                             unit_xp=state.unit_xp.at[slot].set(17))
        revived, ts = advance(dead, jnp.int32(REVIVE_START+slot))
        assert revived.hp[slot] == 1 and revived.gold == 1000-cost
        assert revived.last_service_cost == cost and revived.recovery_balance.tolist() == [100,1]
        assert float(ts.reward) == pytest.approx(.5-env.step_cost)
        chex.assert_trees_all_equal(revived.unit_xp, dead.unit_xp)
        chex.assert_trees_all_equal(revived.unit_levels, dead.unit_levels)
        assert compiled_method(env,'action_mask')(revived)[HEAL_START+slot]
        rested, _ = advance(dead, jnp.int32(REST))
        assert rested.hp[slot] == 0


def test_reward_allowances_are_separate_and_keep_reference_debt(game):
    env, state, advance, _ = game
    wounded = state.replace(hp=state.hp.at[0].set(110).at[1].set(0),
                              recovery_balance=jnp.array([3,0]))
    healed, ts = advance(wounded, jnp.int32(HEAL_START))
    assert healed.recovery_balance.tolist() == [-7,0]
    assert float(ts.reward) == pytest.approx(.0003-env.step_cost)
    revived, ts = advance(healed, jnp.int32(REVIVE_START+1))
    assert revived.recovery_balance.tolist() == [-7,-1] and revived.hp[1] == 1
    assert float(ts.reward) == pytest.approx(-env.step_cost)
    again, ts = advance(revived.replace(hp=revived.hp.at[0].set(115)), jnp.int32(HEAL_START))
    assert again.recovery_balance.tolist() == [-12,-1]
    assert float(ts.reward) == pytest.approx(-env.step_cost)
    reset, _ = compiled_method(env,'reset')(jax.random.PRNGKey(43))
    assert reset.recovery_balance.tolist() == [0,0]


@pytest.mark.core
def test_only_victory_funds_recovery_once_and_keeps_casualties(game):
    env, state, advance, fight = game
    state = state.replace(recovery_balance=jnp.array([-7,-1]))
    # The Titan counts once and contributes only its initial 250 HP even if
    # a healer in another encounter repeatedly restores damaged enemies.
    battle = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(1))).replace(actor=jnp.int32(1))
    battle = battle.replace(hp=battle.hp.at[0].set(0).at[7].set(1),
        enemy_damage_credit=battle.enemy_damage_credit.at[1].set(249))
    won, _ = fight(battle, jnp.int32(SHOOT+1), battle.battle_key, jnp.zeros(env.random_size))
    assert won.recovery_balance.tolist() == [243,0] and won.hp[0] == 0
    rested, _ = advance(won, jnp.int32(REST))
    chex.assert_trees_all_equal(rested.recovery_balance, won.recovery_balance)
    assert rested.hp[0] == 0
    fleeing = battle.replace(escaped=battle.escaped.at[2:6].set(True),
                              retreating=battle.retreating.at[1].set(True))
    withdrawn, _ = fight(fleeing, jnp.int32(CONTINUE), battle.battle_key, jnp.zeros(env.random_size))
    assert not withdrawn.in_battle and withdrawn.hp[0] == 0
    chex.assert_trees_all_equal(withdrawn.recovery_balance, state.recovery_balance)
    ongoing, _ = fight(battle, jnp.int32(DEFEND), battle.battle_key, jnp.zeros(env.random_size))
    chex.assert_trees_all_equal(ongoing.recovery_balance, state.recovery_balance)

    losing = battle.replace(hp=battle.hp.at[:6].set(0).at[1].set(1), actor=jnp.int32(7))
    lost, _ = fight(losing, jnp.int32(CONTINUE), battle.battle_key, jnp.zeros(env.random_size))
    assert lost.lost and lost.done
    chex.assert_trees_all_equal(lost.recovery_balance, state.recovery_balance)
    limited = battle.replace(turn_phase=jnp.full(12,2).at[1].set(0), round=jnp.int32(env.max_rounds))
    forced, _ = fight(limited, jnp.int32(DEFEND), battle.battle_key, jnp.zeros(env.random_size))
    assert not forced.done and forced.in_battle and forced.last_event == ROUND_LIMIT_RETREAT
    chex.assert_trees_all_equal(forced.recovery_balance, state.recovery_balance)
    limited = limited.replace(step_count=jnp.int32(env.max_steps-1))
    timed, _ = advance(limited, jnp.int32(DEFEND))
    assert timed.done and not timed.lost
    chex.assert_trees_all_equal(timed.recovery_balance, state.recovery_balance)
    healed_enemy = compiled_method(env,'_begin_battle')(state.replace(enemy=jnp.int32(5)))
    healed_enemy = enemy_roster_state(env,healed_enemy,['squire','titan','squire','archer',None,'acolyte'])
    initial_max = compiled_method(env,'max_hp')(healed_enemy)[6:]
    healed_enemy = healed_enemy.replace(enemy_initial_hp=initial_max)
    for _ in range(3):
        healed_enemy = healed_enemy.replace(actor=jnp.int32(11), hp=healed_enemy.hp.at[7].set(200))
        healed_enemy, _ = fight(healed_enemy, jnp.int32(CONTINUE), healed_enemy.battle_key, jnp.zeros(env.random_size))
        assert healed_enemy.hp[7] == 220
        chex.assert_trees_all_equal(healed_enemy.recovery_balance, state.recovery_balance)
    # Seed damage from earlier attacks before the final one-HP sweep.
    healed_enemy = healed_enemy.replace(actor=jnp.int32(3), hp=healed_enemy.hp.at[6:].set((initial_max>0).astype(jnp.int32)),
        enemy_damage_credit=jnp.maximum(initial_max-1,0))
    finished, _ = fight(healed_enemy, jnp.int32(SHOOT+1), healed_enemy.battle_key, jnp.zeros(env.random_size))
    assert not finished.in_battle
    chex.assert_trees_all_equal(finished.recovery_balance,
        state.recovery_balance+jnp.array([initial_max.sum(), (initial_max>0).sum()]))


def test_human_service_uses_engine_quotes_and_payment(game):
    from serve_number_grid import GameService
    env, state, advance, _ = game
    # Reuse the GPU-compiled environment; avoid compiling a second identical server.
    service = GameService.__new__(GameService)
    import threading
    service.lock = threading.Lock()
    key = (env.construction.faction, env.construction.lord['id'])
    service.environments = {key:(env, compiled_method(env,'reset'), advance)}
    token = 'capital-test'
    state = state.replace(hp=state.hp.at[0].set(115).at[1].set(0))
    service.sessions = OrderedDict({token:(key,state,0.)})
    snap = service.snapshot(env,state,0.)
    assert snap['capital_quotes'][0] == [1,5,5,5,50,1,0]
    assert snap['capital_quotes'][1][4:] == [200,0,1]
    result = service.act(token,HEAL_START)
    assert result['snapshot']['state']['hp'][0] == 120
    assert result['snapshot']['state']['gold'] == 995
