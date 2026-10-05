"""Revival must not overlap living footprints or spend a rejected item."""
from copy import deepcopy
from unittest.mock import patch
import random
import pytest
from battle_env import BattleEnv, DEFEND_ACTION_INDEX, FIRST_HERO_ITEM_ACTION_START, SECOND_HERO_ITEM_ACTION_START
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit
from formation_occupancy import formation_footprint, formation_footprint_is_free

CATALOG = {row['кто']: row for row in DATA}

def native(name, team, position):
    return map_unit_to_battle(CATALOG[name], team, position)

class FixedRng(random.Random):
    def random(self): return 0.0
    def randint(self, low, high): return low
    def shuffle(self, values): pass
    def choice(self, values): return values[0]

@pytest.fixture
def env():
    e = CampaignEnv(map_name='default', Realcapital=2, typeoflord=2,
                    observation_version='local5', scripted_capital_bot_enabled=False,
                    use_boss_starting_roster=False, log_enabled=False)
    e.reset(seed=1)
    e.gold = 100000.0
    yield e
    e.close()

def roster(env, *, big=True, position=7, blocker_big=False):
    target = native('Чёрт' if big else 'Одержимый', 'blue', position)
    target.update(health=0, hp=0, initiative=0)
    partner = position + 3 if position < 10 else position - 3
    blocker = native('Чёрт' if blocker_big else 'Лич', 'blue', partner)
    blocker['hp'] = blocker['health']
    hero_pos = next(p for p in (8, 9, 7) if p not in (position, partner))
    occupied = {position, partner, hero_pos}
    env.blue_team_state = [target, blocker, native('Советник', 'blue', hero_pos)] + [
        placeholder_unit('blue', pos) for pos in range(7, 13) if pos not in occupied]
    return target, blocker

def potion_action(env, position):
    item = next(name for name in env.scenario_potion_item_names
                if env._potion_item_definition(name).get('effect', {}).get('kind') == 'revive')
    assert env._potion_available_count(item) > 0
    action = (env.GRID_POTION_USE_ACTION_START
              + len(env.GRID_POTION_USE_POSITIONS) * env.scenario_potion_item_names.index(item)
              + env.GRID_POTION_USE_POSITIONS.index(position))
    return action, item

@pytest.mark.parametrize('position', range(1, 13))
@pytest.mark.parametrize('big', [False, True])
def test_shared_footprint_geometry_and_identity(position, big):
    unit = dict(position=position, big=big)
    pair = position + 3 if position in (1, 2, 3, 7, 8, 9) else position - 3
    assert formation_footprint(unit) == ({position, pair} if big else {position})
    assert formation_footprint_is_free(unit, [unit])
    assert not formation_footprint_is_free(unit, [dict(unit)])
    assert formation_footprint_is_free(unit, [dict(position=pair, big=False)]) is (not big)
    assert not formation_footprint_is_free(unit, [dict(position=pair, big=True)])
    opposite = position + 6 if position < 7 else position - 6
    assert formation_footprint_is_free(unit, [dict(position=opposite, big=True)])

@pytest.mark.parametrize('path', ['potion', 'gold'])
@pytest.mark.parametrize('position', range(7, 13))
@pytest.mark.parametrize('cleanup', ['dies', 'disappears'])
def test_public_grid_revival_blocks_overlap_without_cost_then_allows_cleanup(env, path, position, cleanup):
    target, blocker = roster(env, position=position)
    env.grid_env.agent_pos = tuple(next(iter(env.castle_heal_tiles)))
    if path == 'potion':
        action, item = potion_action(env, position)
    else:
        action = env.GRID_CASTLE_REVIVE_ACTION_START + env.CASTLE_REVIVE_POSITIONS.index(position)
        item = None
    gold = env.gold
    bottles = env._potion_available_count(item) if item else None
    budget = env.revive_reward_kill_balance
    with patch.object(env, '_has_temple_built', return_value=True):
        env._get_blue_state()  # Existing masks synchronize derived hero flags.
        before = deepcopy(env.blue_team_state)
        assert not env.action_masks()[action]
        assert env.blue_team_state == before
        info = env.step(action)[4]
        assert not info['revived']
        assert target['health'] == target['hp'] == 0
        assert env.gold == gold
        assert env.revive_reward_kill_balance == budget
        if item:
            assert env._potion_available_count(item) == bottles
            assert not info['potion_consumed']
        else:
            assert info['gold_spent'] == 0
        assert env._revive_unit_at_position(position) == (False, None)
        assert env._revive_unit_with_gold_at_position(position) == (False, None, 0.0)
        if cleanup == 'dies':
            blocker['hp'] = blocker['health'] = 0
        else:
            env.blue_team_state.remove(blocker)
        assert env.action_masks()[action]
        info = env.step(action)[4]
        assert info['revived']
        assert target['health'] == target['hp'] == 1
        if item:
            assert env._potion_available_count(item) == bottles - 1
        else:
            assert env.gold < gold

@pytest.mark.parametrize('path', ['potion', 'gold'])
@pytest.mark.parametrize('big,blocker_big,allowed', [(False, False, True), (False, True, False), (True, False, False)])
def test_grid_small_and_covering_large_controls(env, path, big, blocker_big, allowed):
    target, _ = roster(env, big=big, position=10, blocker_big=blocker_big)
    if path == 'potion':
        assert env._can_revive_position(10) is allowed
        assert env._revive_unit_at_position(10)[0] is allowed
    else:
        assert env._can_castle_revive_position(10) is allowed
        assert env._revive_unit_with_gold_at_position(10)[0] is allowed
    assert (target['health'] > 0) is allowed

@pytest.mark.parametrize('team,front', [('blue', 7), ('red', 1)])
@pytest.mark.parametrize('reverse', [False, True])
def test_scripted_rest_checks_each_newly_revived_footprint_without_running_bot(env, team, front, reverse):
    large = native('Чёрт', team, front)
    small = native('Лич', team, front + 3)
    for unit in (large, small):
        unit['hp'] = unit['health'] = 0
    env.scripted_capital_bot_team_state = [small, large] if reverse else [large, small]
    assert env._fully_restore_scripted_capital_bot_team() == (1, 0)
    first, second = env.scripted_capital_bot_team_state
    assert first['health'] > 0 and second['health'] == 0
    assert env._fully_restore_scripted_capital_bot_team() == (0, 0)
    first['hp'] = first['health'] = 0
    env.scripted_capital_bot_team_state.remove(first)
    assert env._fully_restore_scripted_capital_bot_team() == (1, 0)
    assert second['health'] > 0

def advance_to_hero(env):
    for _ in range(30):
        if env.battle_env.current_blue_attacker_pos == 8:
            return
        env.step(DEFEND_ACTION_INDEX)
    raise AssertionError('Hero did not receive a turn')

@pytest.mark.parametrize('big', [False, True])
def test_campaign_real_equipment_summon_and_orb_revival(env, big):
    target = native('Чёрт' if big else 'Одержимый', 'blue', 7)
    target['health'] = 0
    hero = native('Советник', 'blue', 8)
    progression = BattleEnv(log_enabled=False)
    while hero['Level'] < 10:
        progression._apply_exp_award_to_unit(hero, int(hero['exp_required']))
    env.blue_team_state = [target, hero, native('Одержимый', 'blue', 9),
                          placeholder_unit('blue', 10), native('Сектант', 'blue', 11),
                          native('Сектант', 'blue', 12)]
    for name in ['Lich Orb', 'Orb of Life']:
        env._add_hero_item(name)
    for index, name in enumerate(['Lich Orb', 'Orb of Life']):
        if index:
            env.step(8)
        action = env.GRID_EQUIP_BATTLE_ITEM1_ACTION_START + env.BATTLE_EQUIPPABLE_ITEM_NAMES.index(name)
        assert env.action_masks()[action]
        assert env.step(action)[4]['equip_battle_item_applied']
    enemy_id = next(iter(env._enemy_configs))
    env.enemy_team_states[enemy_id] = [native('Патриарх', 'red', 4)] + [
        placeholder_unit('red', pos) for pos in range(1, 7) if pos != 4]
    env.current_enemy_id = enemy_id
    env.current_battle_context = {'kind': 'hero'}
    env.battle_origin_pos = tuple(env.grid_env.agent_pos)
    env.mode = env.MODE_BATTLE
    env._init_battle(enemy_id)
    battle = env.battle_env
    battle.rng = FixedRng(0)
    assert battle.current_blue_attacker_pos == 8
    summon = FIRST_HERO_ITEM_ACTION_START + 3
    revive = SECOND_HERO_ITEM_ACTION_START
    assert env.action_masks()[summon]
    assert env.step(summon)[4]['battle_hero_item_applied']
    assert env._count_hero_item('Lich Orb') == 0
    advance_to_hero(env)
    assert bool(env.action_masks()[revive]) is (not big)
    before = env._count_hero_item('Orb of Life')
    info = env.step(revive)[4]
    assert info['battle_hero_item_applied'] is (not big)
    assert info['battle_hero_item_consumed'] is (not big)
    assert env._count_hero_item('Orb of Life') == before - int(not big)
    assert (battle._unit_by_position(7)['health'] > 0) is (not big)
    if big:
        battle._subtract_health(battle._unit_by_position(10), 100000)
        advance_to_hero(env)
        assert env.action_masks()[revive]
        assert env.step(revive)[4]['battle_hero_item_applied']
        assert env._count_hero_item('Orb of Life') == 0
        assert battle._unit_by_position(7)['health'] == 1
    cells = [cell for unit in battle.combined if unit['team'] == 'blue' and battle._alive(unit)
             for cell in formation_footprint(unit)]
    assert len(cells) == len(set(cells)) <= 6
