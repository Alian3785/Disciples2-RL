"""Trade Train scenario and scenario-priced recruitment on the shared CUDA engine.

Sources: Big_map/maps/trade_train.py, its layout.json, and
campaign_env_economy.py:_step_hire_mercenary_unit in the Python reference.
Run explicitly; this map-specific compilation is outside the core suite.
"""
import copy
from collections import deque

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, NumberGrid, action_names, load_map
from stoix.envs.number_grid_recruitment import RecruitmentRules
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


@pytest.fixture(scope='module')
def trade_game():
    assert jax.default_backend() == 'gpu', 'Run through scripts/run_gpu.sh on CUDA'
    env = NumberGrid(map_config=load_map('trade_train'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, state, compiled_method(env, 'step'), compiled_method(env, 'action_mask')


def test_reference_layout_and_objective(trade_game):
    env, state, _, _ = trade_game
    config = load_map('trade_train')
    assert env.size == 26 and state.position.tolist() == [14, 3]
    assert int(state.gold) == 9000
    assert [env.progression.rows[i]['key'] if i else None for i in state.unit_ids[:6].tolist()] == [
        None, 'duke', None, None, None, None]
    assert state.hp[:6].tolist() == [0, 150, 0, 0, 0, 0]
    assert config['faction'] == 'legions' and config['lord_type'] == 'warrior'
    assert config['campaign_objective'] == 'all_enemies'
    assert config['reward_profile'] == 'python_reference'
    assert config['reference_enemy_ids'] == [1, 2, 3, 4, 5, 6]
    assert state.enemy_positions.tolist() == [[4, 23], [10, 23], [16, 23], [22, 23], [4, 20], [22, 20]]
    assert sum(config['enemy_units']) == 19
    assert config['enemy_initial_levels'] == [
        [3, 1, 0, 4, 3, 0], [3, 2, 0, 4, 3, 0], [4, 3, 0, 5, 4, 0],
        [5, 4, 0, 4, 3, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0]]
    assert state.enemy_progress[:, :, 0].tolist() == config['enemy_initial_levels']
    assert [s['position'] for s in env.site_rules.sites] == [[6, 18], [18, 18], [12, 18]]
    assert not env.scenario.enabled
    for i in range(6):
        only_one = jnp.zeros_like(state.alive).at[i].set(True)
        assert bool(env._goals_left(only_one, state.scenario_spawned))
    assert not bool(env._goals_left(jnp.zeros_like(state.alive), state.scenario_spawned))
    assert env.num_actions == len(action_names(config)) == 195
    observation = compiled_method(env, 'observation')(state)
    assert observation.shape == (1314,) and bool(jnp.all(jnp.isfinite(observation)))


def test_every_site_approach_is_reachable_without_a_forced_battle(trade_game):
    env, state, _, _ = trade_game
    blocked = set(map(tuple, load_map('trade_train')['obstacles']))
    blocked.update(map(tuple, state.enemy_positions.tolist()))
    for site in env.site_rules.sites:
        blocked.update(site['footprint'])
    visited = {tuple(state.position.tolist())}
    queue = deque(visited)
    while queue:
        r, c = queue.popleft()
        for dr, dc in env.directions.tolist():
            p = (r+dr, c+dc)
            if 0 < p[0] < env.size-1 and 0 < p[1] < env.size-1 and p not in blocked and p not in visited:
                visited.add(p)
                queue.append(p)
    for site in env.site_rules.sites:
        assert len(site['footprint']) == 9 and len(site['interaction_tiles']) == 5
        assert all(tuple(p) in visited for p in site['interaction_tiles'])
    for r, c in state.enemy_positions.tolist():
        assert any((r+dr, c+dc) in visited for dr, dc in env.directions.tolist())


def test_all_elite_offers_use_scenario_prices_and_native_levels(trade_game):
    env, initial, step, mask = trade_game
    rules = env.recruitment
    assert rules.keys[5:] == ('defender_of_faith', 'white_mage', 'g000uu0039', 'archdruid',
                             'g000uu0089', 'g000uu0176', 'centaur_savage', 'watcher')
    assert rules.costs[5:].tolist() == [2500, 2000, 2500, 1800, 2000, 2500, 1800, 1800]
    assert rules.initial_stock.tolist() == [1]*8
    assert [o['cost'] for o in rules.metadata['offers'][5:]] == rules.costs[5:].tolist()
    state = initial.replace(position=env.site_rules.approaches['mercenary'][0], movement_points=jnp.int32(0))
    for offer in range(5, 13):
        action = rules.start+offer
        assert bool(mask(state)[action])
        bought, ts = step(state, jnp.int32(action))
        slot = int(bought.last_target)
        uid = int(rules.ids[offer])
        assert int(bought.unit_ids[slot]) == uid
        assert int(bought.unit_levels[slot]) == int(env.progression.base_levels[uid])
        assert int(bought.hp[slot]) == int(env.progression.stats(uid, bought.unit_levels[slot])[0])
        assert int(bought.gold) == 9000-int(rules.costs[offer])
        assert int(bought.mercenary_stock[offer-5]) == 0
        assert int(bought.day) == int(state.day) and int(bought.movement_points) == 0
        assert bool(jnp.isfinite(ts.reward))
        assert not bool(mask(bought)[action])
        duplicate, _ = step(bought, jnp.int32(action))
        assert duplicate.gold == bought.gold
        assert bool(jnp.all(duplicate.unit_ids == bought.unit_ids))
        assert bool(jnp.all(duplicate.mercenary_stock == bought.mercenary_stock))


@pytest.mark.parametrize('restriction', ['away', 'poor', 'dead_leader', 'empty_stock'])
def test_hiring_mask_and_step_reject_invalid_purchase(trade_game, restriction):
    env, initial, step, mask = trade_game
    state = initial.replace(position=env.site_rules.approaches['mercenary'][0])
    if restriction == 'away':
        state = state.replace(position=initial.position)
    elif restriction == 'poor':
        state = state.replace(gold=jnp.int32(2499))
    elif restriction == 'dead_leader':
        state = state.replace(hp=state.hp.at[1].set(0))
    else:
        state = state.replace(mercenary_stock=state.mercenary_stock.at[0].set(0))
    action = env.recruitment.start+5
    assert not bool(mask(state)[action])
    after, _ = step(state, jnp.int32(action))
    assert int(after.gold) == int(state.gold)
    assert bool(jnp.all(after.unit_ids == state.unit_ids))
    assert bool(jnp.all(after.mercenary_stock == state.mercenary_stock))


def test_leadership_limits_the_elite_army(trade_game):
    env, initial, step, mask = trade_game
    names = action_names(load_map('trade_train'))
    state = initial.replace(position=env.site_rules.approaches['mercenary'][0])
    for key in ('defender_of_faith', 'white_mage', 'g000uu0176'):
        state, _ = step(state, jnp.int32(names.index('hire_mercenary_'+key)))
    assert env.recruitment.composition(state).tolist()[:3] == [3, 3, 0]
    # Enough money and a free front slot still do not bypass leadership.
    action = names.index('hire_mercenary_centaur_savage')
    assert int(state.gold) == 2000 and int(state.unit_ids[2]) == 0
    assert not bool(mask(state)[action])
    after, _ = step(state, jnp.int32(action))
    assert int(after.unit_ids[2]) == 0 and int(after.gold) == 2000


def test_merchant_prices_stock_and_delivery(trade_game):
    env, initial, step, mask = trade_game
    rules = env.site_rules
    prices = dict(highfather=800, invulnerability=700, celerity=600, bethrezens_claw=5000,
                  ring_of_ages=3750, banner_war=5000, boots_seven_leagues=2000, tome_sorcery=1200,
                  angel_orb=2500, elder_vampire_talisman=4000, elder_vampire_orb=3000,
                  venerable_warrior_orb=2500)
    assert dict(zip(rules.buy_keys, rules.buy_base.tolist())) == prices
    assert int(jnp.sum(initial.merchant_stock)) == 14
    state = initial.replace(position=rules.approaches['merchant'][0])
    for i, key in enumerate(rules.buy_keys):
        bought = state
        for _ in range(int(initial.merchant_stock[i])):
            assert bool(mask(bought)[rules.buy_start+i])
            before = bought
            bought, _ = step(bought, jnp.int32(rules.buy_start+i))
            assert int(before.gold-bought.gold) == prices[key]
            if bool(rules.buy_items[i]):
                assert int(jnp.sum(bought.item_inventory >= 0)) == int(jnp.sum(before.item_inventory >= 0))+1
            else:
                assert int(bought.potions[int(rules.buy_ids[i])]-before.potions[int(rules.buy_ids[i])]) == 1
        assert int(bought.merchant_stock[i]) == 0
        assert not bool(mask(bought)[rules.buy_start+i])
        duplicate, _ = step(bought, jnp.int32(rules.buy_start+i))
        assert int(duplicate.gold) == int(bought.gold)
    for bad in (state.replace(gold=jnp.int32(0)), initial, state.replace(hp=state.hp.at[1].set(0))):
        assert not bool(mask(bad)[rules.buy_start])
        after, _ = step(bad, jnp.int32(rules.buy_start))
        assert int(after.gold) == int(bad.gold)
        assert bool(jnp.all(after.merchant_stock == bad.merchant_stock))


def test_trainer_caps_xp_before_level_up_and_charges_exactly(trade_game):
    env, initial, step, mask = trade_game
    state = initial.replace(position=env.site_rules.approaches['trainer'][0])
    quotes = compiled_method(env.site_rules, 'training_quotes')
    quote = quotes(state)[1]
    action = env.site_rules.train_start+1
    assert bool(mask(state)[action])
    after, _ = step(state, jnp.int32(action))
    assert int(after.unit_xp[1]) == int(quote[3]) == int(quote[1])
    assert int(after.gold) == int(state.gold-quote[4])
    assert int(after.unit_levels[1]) == int(state.unit_levels[1])
    assert int(after.unit_xp[1]) == int(env.progression.required_xp(after.unit_ids[1], after.unit_levels[1]))-1
    assert not bool(mask(after)[action])
    for bad in (initial, state.replace(gold=jnp.int32(0)), state.replace(hp=state.hp.at[1].set(0))):
        assert not bool(mask(bad)[action])
        after, _ = step(bad, jnp.int32(action))
        assert int(after.gold) == int(bad.gold)
        assert bool(jnp.all(after.unit_xp == bad.unit_xp))


def test_scenario_price_validation_and_catalogue_fallback(trade_game):
    env, initial, _, _ = trade_game
    config = copy.deepcopy(load_map('trade_train'))
    roster = config['mercenaries'][0]['roster']
    for offer in roster:
        del offer['price']
    rules = RecruitmentRules(config, env.progression, env.construction, env.recruitment.start, env.site_rules)
    expected = [int(env.progression.rows[int(uid)]['game_data']['ENROLL_C'].split(':')[0][1:])
                for uid in rules.ids]
    assert rules.costs.tolist() == expected
    for price in (0, -1, True, 1800.5, '1800'):
        roster[0]['price'] = price
        with pytest.raises(ValueError, match='prices must be positive integer'):
            RecruitmentRules(config, env.progression, env.construction, env.recruitment.start, env.site_rules)
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(43))
    assert reset.mercenary_stock.tolist() == [1]*8
    assert bool(jnp.all(reset.merchant_stock == initial.merchant_stock))
    assert not bool(jnp.all(reset.battle_key == initial.battle_key))


def finish_trade_battle(env, battle, outcome='victory'):
    """End an otherwise prepared battle through the actual combat transition."""
    if outcome == 'victory':
        battle = battle.replace(hp=battle.hp.at[6:].set(0))
    elif outcome == 'retreat':
        battle = battle.replace(escaped=battle.escaped.at[:6].set(True))
    else:
        battle = battle.replace(hp=battle.hp.at[:6].set(0))
    # Complete post-battle healing; no eligible healer may postpone resolution.
    battle = battle.replace(actor=jnp.int32(1), round=jnp.int32(1),
                            post_victory=jnp.int32(1 if outcome == 'victory' else 2),
                            pending_healers=jnp.zeros(12, bool))
    return compiled_method(env, '_battle_step')(battle, jnp.int32(DEFEND),
        jax.random.PRNGKey(19), jnp.full(env.random_size, .5))


def test_level_bonus_paid_only_on_victory_once_per_enemy(trade_game):
    """User request 2026-10-14: retain entry strength, pay only after winning."""
    env, initial, _, _ = trade_game
    advance = compiled_method(env, '_world_step')
    base = hero_roster_state(env, initial, ['defender_of_faith', 'duke', None, 'white_mage', None, None])
    base = base.replace(hp=base.hp.at[0].set(1).at[3].set(0),
                        unit_levels=base.unit_levels.at[2].set(99),
                        potion_bonus=base.potion_bonus.at[0, 0].add(100).at[3, 0].add(500))
    flags = initial.army_victory_rewarded
    key, draws = jax.random.PRNGKey(19), jnp.full(env.random_size, .5)
    assert flags.tolist() == [False]*6 and env.army_victory_level_reward == 5.
    for enemy in range(6):
        before = base.replace(position=initial.enemy_positions[enemy]-jnp.array([0, 1]),
                              army_victory_rewarded=flags)
        arrived, entry_reward = advance(before, jnp.int32(2), key, draws)
        claimed, claimed_entry = advance(before.replace(army_victory_rewarded=flags.at[enemy].set(True)),
                                          jnp.int32(2), key, draws)
        assert arrived.in_battle and int(arrived.enemy) == enemy
        assert float(entry_reward) == pytest.approx(float(claimed_entry))
        assert arrived.army_victory_rewarded.tolist() == flags.tolist()
        # Wounded Defender of Faith (5) + Duke (1); dead/empty slots contribute zero.
        assert int(arrived.army_battle_level_strength) == 4
        won, reward = finish_trade_battle(env, arrived)
        _, plain = finish_trade_battle(env, claimed)
        assert not bool(won.alive[enemy])
        assert float(reward-plain) == pytest.approx(20.)
        flags = flags.at[enemy].set(True)
        assert won.army_victory_rewarded.tolist() == flags.tolist()
        assert int(won.army_battle_level_strength) == 0
        # Neither altered HP nor temporary combat levels change the entry snapshot.
        changed = arrived.replace(potion_bonus=arrived.potion_bonus.at[0, 0].add(200),
                                  unit_levels=arrived.unit_levels.at[0].add(3))
        _, boosted = finish_trade_battle(env, changed)
        _, boosted_plain = finish_trade_battle(env, changed.replace(army_victory_rewarded=flags))
        assert float(boosted-boosted_plain) == pytest.approx(20.)
        # Sticky flags also block payment if an already claimed enemy resolves again.
        _, repeated = finish_trade_battle(env, claimed.replace(army_battle_level_strength=jnp.int32(100)))
        assert float(repeated) == pytest.approx(float(plain))
    assert compiled_method(env, 'observation')(won)[-7:].tolist() == [1.]*6+[0.]
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(44))
    assert reset.army_victory_rewarded.tolist() == [False]*6
    assert int(reset.army_battle_level_strength) == 0


@pytest.mark.parametrize('levels,expected', [([1, 1, 1, 1], 0.), ([2, 2, 3, 1], 20.),
                                          ([4, 4, 4, 4], 60.)])
def test_army_victory_bonus_uses_entry_levels_above_one(trade_game, levels, expected):
    env, initial, _, _ = trade_game
    state = hero_roster_state(env, initial, ['possessed', 'duke', 'possessed', 'cultist', None, None])
    state = state.replace(position=initial.enemy_positions[5]-jnp.array([0, 1]),
                          unit_levels=state.unit_levels.at[:4].set(jnp.array(levels)))
    arrived, _ = compiled_method(env, '_world_step')(state, jnp.int32(2),
        jax.random.PRNGKey(23), jnp.full(env.random_size, .5))
    assert not bool(arrived.army_victory_rewarded[5])
    observation = compiled_method(env, 'observation')(arrived)
    assert float(observation[-1]) == pytest.approx(expected/5./100.)
    won, reward = finish_trade_battle(env, arrived)
    _, plain = finish_trade_battle(env, arrived.replace(
        army_victory_rewarded=arrived.army_victory_rewarded.at[5].set(True)))
    assert float(reward-plain) == pytest.approx(expected)
    assert bool(won.army_victory_rewarded[5])  # a zero-level bonus still completes this enemy


@pytest.mark.parametrize('outcome', ['retreat', 'defeat'])
def test_unsuccessful_battle_discards_bonus_without_claiming_enemy(trade_game, outcome):
    env, initial, _, _ = trade_game
    base = hero_roster_state(env, initial, ['defender_of_faith', 'duke', None, None, None, None])
    base = base.replace(position=initial.enemy_positions[5]-jnp.array([0, 1]))
    advance = compiled_method(env, '_world_step')
    key, draws = jax.random.PRNGKey(25), jnp.full(env.random_size, .5)
    arrived, _ = advance(base, jnp.int32(2), key, draws)
    after, reward = finish_trade_battle(env, arrived, outcome)
    _, plain = finish_trade_battle(env, arrived.replace(army_battle_level_strength=jnp.int32(0)), outcome)
    assert float(reward) == pytest.approx(float(plain))
    assert not bool(jnp.any(after.army_victory_rewarded))
    assert int(after.army_battle_level_strength) == 0
    if outcome == 'retreat':
        # Re-entry earns nothing immediately and snapshots the now stronger army.
        after = after.replace(movement_points=jnp.int32(100),
                              unit_levels=after.unit_levels.at[1].add(2))
        retried, _ = advance(after, jnp.int32(2), key, draws)
        assert retried.in_battle and int(retried.army_battle_level_strength) == 6
        won, reward = finish_trade_battle(env, retried)
        _, plain = finish_trade_battle(env, retried.replace(
            army_victory_rewarded=retried.army_victory_rewarded.at[5].set(True)))
        assert float(reward-plain) == pytest.approx(30.)
        assert bool(won.army_victory_rewarded[5])


def test_summons_and_invalid_moves_do_not_claim_victory_level_bonus(trade_game):
    env, initial, step, _ = trade_game
    near = initial.replace(position=initial.enemy_positions[0]-jnp.array([0, 1]))
    spell = next(i for i, row in enumerate(env.casting.rows) if row['id'] == 'g000ss0058')
    prepared = near.replace(learned_spells=jnp.uint32(1 << spell), mana=jnp.full(5, 5000, jnp.int32))
    battle, _ = compiled_method(env, '_world_step')(prepared, jnp.int32(env.casting.start+spell),
        jax.random.PRNGKey(22), jnp.full(env.random_size, .5))
    assert battle.summon_battle and int(battle.army_battle_level_strength) == 0
    # Even a stale snapshot must not reward a summoned army's victory.
    won, reward = finish_trade_battle(env, battle.replace(army_battle_level_strength=jnp.int32(50)))
    _, plain = finish_trade_battle(env, battle)
    assert float(reward) == pytest.approx(float(plain))
    assert not bool(jnp.any(won.army_victory_rewarded))
    invalid, _ = step(near.replace(movement_points=jnp.int32(0)), jnp.int32(2))
    assert not invalid.in_battle and not bool(jnp.any(invalid.army_victory_rewarded))
    assert int(invalid.army_battle_level_strength) == 0


@pytest.mark.parametrize('value', [-5., True, float('nan'), float('inf'), '5'])
def test_invalid_army_victory_level_coefficient_is_rejected(value):
    with pytest.raises(ValueError, match='army_victory_level_reward'):
        NumberGrid(map_config={**load_map('trade_train'), 'army_victory_level_reward': value})


@pytest.mark.parametrize('old_key', ['army_engage_hp_reward', 'army_engage_level_reward',
                                     'objective_engage_hp_reward'])
def test_obsolete_engagement_rewards_are_not_silently_reinterpreted(old_key):
    config = load_map('trade_train')
    config.pop('army_victory_level_reward')
    config[old_key] = 5.
    with pytest.raises(ValueError, match=f'{old_key} was replaced'):
        NumberGrid(map_config=config)
