"""W5: city ownership requires the living leader of the travelling party.

Prepared map positions isolate capture; movement, potion use and battle
finalization run through CampaignEnv.step. No training or scripted bot runs.
"""
from copy import deepcopy

import pytest

from campaign_env import CampaignEnv
from campaign_env_data import placeholder_unit


@pytest.fixture(params=['green_dragon_minimal', 'default'])
def env(request):
    e = CampaignEnv(map_name=request.param, Realcapital=2,
                    observation_version='local5', scripted_capital_bot_enabled=False,
                    use_boss_starting_roster=False, log_enabled=False)
    e.reset(seed=42)
    yield e
    e.close()


def city_data(e):
    city = ('Форпост Пепла' if e._map.name == 'green_dragon_minimal'
            else next(iter(e.FINAL_OBJECTIVE_CITIES)))
    return (city, tuple(e.legions_settlement_source_tile_by_name[city]),
            tuple(e.legions_settlement_territory_source_by_name[city]['required_enemy_ids']))


def clear_city(e):
    city, tile, ids = city_data(e)
    for eid in ids:
        e.grid_env.mark_enemy_defeated(eid)
    # Isolate entry from an unrelated adjacent field army (default enemy 20).
    for eid, pos in e.grid_env.enemy_positions.items():
        if eid not in ids and max(abs(pos[0] - tile[0]), abs(pos[1] - tile[1])) <= 1:
            e.grid_env.mark_enemy_defeated(eid)
    return city, tile, ids


def leader(e):
    return e._resolve_travel_hero(units=e.blue_team_state)


def enter(e, tile):
    # Find a legal adjacent fixture origin and enter with the real movement step.
    for action in range(8):
        dx, dy = e.grid_env._action_to_delta(action)
        origin = (tile[0] - dx, tile[1] - dy)
        if (e.grid_env._is_within_grid(origin)
                and origin not in e.grid_env.obstacle_positions
                and origin not in e.grid_env.dynamic_blocked_positions):
            e.grid_env.agent_pos = origin
            e.moves = 100
            if e.compute_action_mask()[action]:
                result = e.step(action)
                assert tuple(e.grid_env.agent_pos) == tile
                assert result[-1]['movement_applied']
                return result
    pytest.fail('No legal entry into the fixture city')


def assert_not_captured(e, city, result):
    info = result[-1]
    assert not e._garrison_player_owns(city)
    assert city not in e.captured_objective_cities
    assert city not in e.legions_active_settlement_territory_capture_turn_by_name
    assert city not in e.legions_settlement_growth_history_by_name
    assert not info.get('captured_objective_cities')
    assert not info.get('legions_settlement_territories_activated')
    assert info.get('final_objective_reward', 0) == 0
    assert info.get('campaign_result') != 'victory'
    assert not any(result[2:4])


def assert_captured(e, city, result):
    info = result[-1]
    assert e._garrison_player_owns(city)
    assert info['legions_settlement_territories_activated'] == [city]
    if city in e.FINAL_OBJECTIVE_CITIES:
        assert city in e.captured_objective_cities
        assert info['captured_objective_cities'] == [city]
        assert info['final_objective_reward'] == e._compute_final_objective_reward(1)
    else:
        assert not info.get('captured_objective_cities')
        assert info.get('final_objective_reward', 0) == 0


@pytest.mark.parametrize('state', ['dead', 'stale_hp', 'missing', 'empty', 'summoned', 'none'])
def test_entry_without_living_party_leader_never_captures(env, state):
    city, tile, _ = clear_city(env)
    hero = leader(env)
    assert hero['name'] == 'Герцог'
    if state in ('dead', 'stale_hp'):
        hero.update(health=0, hp=150 if state == 'stale_hp' else 0)
    elif state == 'missing':
        env.blue_team_state.remove(hero)
    elif state == 'empty':
        env.blue_team_state[env.blue_team_state.index(hero)] = placeholder_unit('blue', hero['position'])
    elif state == 'summoned':
        hero['Summoned'] = True
    else:
        # The predicate itself must not lazily recreate a missing roster.
        env.blue_team_state = None
        env.grid_env.agent_pos = tile
        assert not env._hero_is_on_city_tile(city_data(env)[2][0])
        assert env.blue_team_state is None
        return
    territory = set(env.legions_territory_tile_set)
    result = enter(env, tile)
    assert_not_captured(env, city, result)
    assert env.legions_territory_tile_set == territory
    assert env._garrison_info()['city_owners'][city] == 'neutral'


def test_living_leader_captures_and_dead_reentry_does_not_revoke_ownership(env):
    city, tile, _ = clear_city(env)
    shape, actions = env.observation_space.shape, env.action_space.n
    result = enter(env, tile)
    assert_captured(env, city, result)
    assert env.observation_space.contains(result[0])
    captured_turn = env.legions_active_settlement_territory_capture_turn_by_name[city]
    leader(env).update(health=0, hp=0)
    result = enter(env, tile)
    assert env._garrison_player_owns(city)
    assert env.legions_active_settlement_territory_capture_turn_by_name[city] == captured_turn
    assert not result[-1].get('legions_settlement_territories_activated')
    assert result[-1].get('final_objective_reward', 0) == 0
    assert env.observation_space.shape == shape and env.action_space.n == actions


def test_revival_requires_new_entry_and_rewards_only_once(env):
    city, tile, _ = clear_city(env)
    hero = leader(env)
    hero.update(health=0, hp=0)
    assert_not_captured(env, city, enter(env, tile))
    action = env.GRID_REVIVE_ACTION_START + env.REVIVE_BOTTLE_POSITIONS.index(hero['position'])
    assert env.compute_action_mask()[action]
    result = env.step(action)
    assert result[-1]['revived']
    assert_not_captured(env, city, result)
    # REST is not a city-entry action, even after the leader has revived.
    assert_not_captured(env, city, env.step(8))
    result = enter(env, tile)
    assert_captured(env, city, result)
    result = enter(env, tile)
    assert result[-1].get('final_objective_reward', 0) == 0
    assert not result[-1].get('legions_settlement_territories_activated')


@pytest.mark.parametrize('alive', [False, True])
def test_capture_resolves_leader_after_real_formation_swap(env, alive):
    city, tile, _ = clear_city(env)
    pair = next(pair for pair in env.GRID_SWAP_UNIT_PAIRS if set(pair) == {7, 8})
    action = env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_PAIRS.index(pair)
    assert env.step(action)[-1]['unit_swap_applied']
    hero = leader(env)
    assert hero['position'] == 7
    hero.update(health=1 if alive else 0, hp=1 if alive else 0)
    env.blue_team_state.reverse()
    result = enter(env, tile)
    (assert_captured if alive else assert_not_captured)(env, city, result)


@pytest.mark.parametrize('alive', [False, True])
def test_enemy_owned_city_requires_living_leader(env, alive):
    city, tile, _ = clear_city(env)
    env._transfer_city_to_bot(city)
    hero = leader(env)
    hero.update(health=1 if alive else 0, hp=1 if alive else 0)
    result = enter(env, tile)
    (assert_captured if alive else assert_not_captured)(env, city, result)
    assert (city in env.bot_city_capture_turns) is not alive
    assert env._garrison_info()['city_owners'][city] == ('player' if alive else 'scripted_bot')


def start_city_battle(e, *, kind='hero', on_city=True):
    city, tile, ids = city_data(e)
    for eid in ids[:-1]:
        e.grid_env.mark_enemy_defeated(eid)
    e.grid_env.agent_pos = tile if on_city else tuple(e.CASTLE_POS)
    e.current_enemy_id = ids[-1]
    e.battle_origin_pos = tuple(e.grid_env.agent_pos)
    e.current_battle_context = {'kind': kind}
    e.mode = e.MODE_BATTLE
    e._init_battle(ids[-1])
    return city, tile, ids[-1]


@pytest.mark.parametrize('alive', [False, True])
@pytest.mark.parametrize('transformed', [False, True])
def test_winning_battle_uses_surviving_leader_after_state_save(env, alive, transformed):
    city, _, eid = start_city_battle(env)
    battle = env.battle_env
    hero = next(u for u in battle.combined if u['team'] == 'blue' and env._is_hero_unit(u))
    if transformed:
        witch = env._build_unit_from_data(env._find_unit_data_by_name('Ведьма'), 'red', 1)
        battle._apply_witch_effect(witch, hero)
        assert hero['transformed']
    hero.update(health=1 if alive else 0, hp=1 if alive else 0)
    assert any(u['team'] == 'blue' and not env._is_hero_unit(u) and u['health'] > 0
               for u in battle.combined)
    battle.winner = 'blue'
    result = env.step(0)
    assert result[-1]['battle_result'] == 'victory'
    assert result[-1]['enemy_defeat_reward'] > 0
    assert not env.grid_env.enemies_alive[eid]
    assert env._is_travel_unit_alive(leader(env)) is alive
    assert not leader(env).get('transformed')
    (assert_captured if alive else assert_not_captured)(env, city, result)


def test_live_guard_blocks_capture_before_battle_is_won(env):
    city, tile, ids = city_data(env)
    env.grid_env.agent_pos = tile
    for eid in ids[:-1]:
        env.grid_env.mark_enemy_defeated(eid)
    info = {}
    assert env._capture_cities_on_hero_entry(0, info) == 0
    assert_not_captured(env, city, (None, 0, False, False, info))


def test_remote_hero_victory_clears_guard_without_remote_capture(env):
    city, _, _ = start_city_battle(env, on_city=False)
    env.battle_env.winner = 'blue'
    assert_not_captured(env, city, env.step(0))


def test_summon_victory_does_not_borrow_main_hero_for_capture(env):
    city, _, eid = start_city_battle(env, kind='summon_spell')
    hero_before = deepcopy(env.blue_team_state)
    env.battle_env.winner = 'blue'
    result = env.step(0)
    assert result[-1]['battle_context_kind'] == 'summon_spell'
    assert result[-1]['battle_result'] == 'victory'
    assert not env.grid_env.enemies_alive[eid]
    assert env.blue_team_state == hero_before
    assert_not_captured(env, city, result)


def test_garrison_victory_does_not_capture_another_city_under_remote_hero(env):
    city, tile, _ = clear_city(env)
    other = next(name for name in env.garrison_city_names if name != city)
    for eid in env.legions_settlement_territory_source_by_name[other]['required_enemy_ids']:
        env.grid_env.mark_enemy_defeated(eid)
    env.grid_env.agent_pos = env.legions_settlement_source_tile_by_name[other]
    env._capture_cities_on_hero_entry(0, {})
    env.city_garrisons[other] = [env._build_unit_from_data(
        env._find_unit_data_by_name('Одержимый'), 'blue', 7)]
    env.scripted_capital_bot_team_state = env._build_battle_team_with_placeholders(
        'blue', [env._build_unit_from_data(env._find_unit_data_by_name('Гном'), 'blue', 8)])
    env.grid_env.agent_pos = tile
    hero_before = deepcopy(env.blue_team_state)
    env._start_city_defence(other)
    env.battle_env.winner = 'blue'
    result = env.step(0)
    assert result[-1]['battle_context_kind'] == 'city_garrison'
    assert env.blue_team_state == hero_before
    assert_not_captured(env, city, result)


def test_last_objective_cannot_win_campaign_until_living_leader_reenters():
    e = CampaignEnv(Realcapital=2, observation_version='local5',
                    scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
                    log_enabled=False)
    e.reset(seed=42)
    try:
        city, tile, _ = clear_city(e)
        for other, ids in e.FINAL_OBJECTIVE_CITIES.items():
            if other != city:
                for eid in ids:
                    e.grid_env.mark_enemy_defeated(eid)
                e.grid_env.agent_pos = e.legions_settlement_source_tile_by_name[other]
                e._capture_cities_on_hero_entry(0, {})
        hero = leader(e)
        hero.update(health=0, hp=0)
        assert_not_captured(e, city, enter(e, tile))
        action = e.GRID_REVIVE_ACTION_START + e.REVIVE_BOTTLE_POSITIONS.index(hero['position'])
        assert e.step(action)[-1]['revived']
        result = enter(e, tile)
        assert result[2] and not result[3]
        assert result[-1]['campaign_result'] == 'victory'
        assert result[-1]['campaign_victory_reason'] == 'objective_cities_cleared'
        assert result[-1]['final_objective_reward'] == e._compute_final_objective_reward(1, captured_before=1)
    finally:
        e.close()
