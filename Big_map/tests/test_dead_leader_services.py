"""A party needs a living leader to use mercenaries, trainers and magic towers."""

from copy import deepcopy

import pytest

from campaign_env import CampaignEnv

SERVICES = ("mercenary", "trainer", "spell_shop")


@pytest.fixture
def make_party():
    opened = []

    def create(service, hero_position=8):
        env = CampaignEnv(
            map_name="default" if service == "spell_shop" else "trade_train", observation_version="local5",
            scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
            log_enabled=False, detailed_step_info=True,
        )
        opened.append(env)
        env.reset(seed=42)
        hero = deepcopy(env._resolve_travel_hero())
        hero.update(position=hero_position, stand="ahead" if hero_position < 10 else "behind")
        companion_position = 7 if hero_position != 7 else 8
        companion = env._build_unit_from_data(env._find_unit_data_by_name("Сектант"), "blue", companion_position)
        companion["exp_current"] = 0
        env.blue_team_state = env._build_battle_team_with_placeholders("blue", [hero, companion])
        env._mark_equipment_dirty()
        env._sync_hero_progression_flags()
        env.gold = 10000.0
        if service == "mercenary":
            env.grid_env.agent_pos = env.mercenary_interaction_tiles[0]
            index = next(i for i, option in enumerate(env.active_mercenary_hire_options)
                         if option["unit_name"] == "Белый маг")
            action = env.GRID_MERCENARY_HIRE_ACTION_START + index
        elif service == "trainer":
            env.grid_env.agent_pos = env.trainer_interaction_tiles[0]
            action = env.GRID_TRAINER_ACTION_START + list(env.TRAINER_POSITIONS).index(companion_position)
        else:
            env.grid_env.agent_pos = env.spell_shop_interaction_tiles[0]
            action = env.GRID_SPELL_SHOP_BUY_ACTION_START
        assert env.compute_action_mask()[action], (service, env.grid_env.agent_pos)
        return env, action

    yield create
    for env in opened:
        env.close()


def resources(env):
    return deepcopy((env.gold, env._current_mana_totals(), env.blue_team_state,
                     env.mercenary_site_rosters, env.spell_shop_stocks,
                     env.spell_shop_purchased_spell_ids, env.active_spells))


def assert_blocked(env, action, service):
    assert not env.compute_action_mask()[action]
    if service == "trainer":
        position = env.TRAINER_POSITIONS[action - env.GRID_TRAINER_ACTION_START]
        assert not env._can_train_unit_position(position)
        assert not env._trainer_preview_for_position(position)["trainable"]
    before = resources(env)
    info = env.step(action)[-1]  # Deliberately bypass the mask.
    success = {"mercenary": "hired", "trainer": "trainer_trained", "spell_shop": "spell_shop_buy_purchased"}
    assert not info[success[service]]
    assert resources(env) == before


@pytest.mark.parametrize("service", SERVICES)
@pytest.mark.parametrize("hero_position", range(7, 13))
def test_dead_leader_blocks_masks_and_forced_services_in_every_slot(make_party, service, hero_position):
    env, action = make_party(service, hero_position)
    env._resolve_travel_hero().update(health=0.0, hp=0.0)
    assert any(not env._is_hero_unit(unit) and unit.get("health", 0) > 0 for unit in env.blue_team_state)
    assert_blocked(env, action, service)


@pytest.mark.parametrize("service", SERVICES)
def test_missing_leader_also_blocks_services(make_party, service):
    env, action = make_party(service)
    env.blue_team_state = [unit for unit in env.blue_team_state if not env._is_hero_unit(unit)]
    assert_blocked(env, action, service)


@pytest.mark.parametrize("service", SERVICES)
def test_stale_hp_alias_does_not_make_dead_leader_alive(make_party, service):
    env, action = make_party(service)
    env._resolve_travel_hero().update(health=0.0, hp=100.0)
    assert_blocked(env, action, service)


@pytest.mark.parametrize("service", SERVICES)
def test_reviving_leader_restores_service_without_reset(make_party, service):
    env, action = make_party(service)
    hero = env._resolve_travel_hero()
    position = hero["position"]
    hero.update(health=0.0, hp=0.0)
    assert_blocked(env, action, service)
    revive = env.GRID_REVIVE_ACTION_START + list(env.GRID_BOTTLE_POSITIONS).index(position)
    assert env.compute_action_mask()[revive]
    assert env.step(revive)[-1]["revived"]
    assert env._resolve_travel_hero(alive_only=True) is not None
    assert env.compute_action_mask()[action]
    before = resources(env)
    info = env.step(action)[-1]
    success = {"mercenary": "hired", "trainer": "trainer_trained", "spell_shop": "spell_shop_buy_purchased"}
    assert info[success[service]]
    assert env.gold < before[0]
    if service == "trainer":
        assert info["trainer_xp_gained"] > 0
    elif service == "mercenary":
        assert info["hired_unit_name"] == "Белый маг"
    else:
        assert info["spell_shop_buy_spell_id"] in env.spell_shop_purchased_spell_ids


def test_dead_leader_can_still_shop_at_merchant(make_party):
    env, _ = make_party("mercenary")
    env._resolve_travel_hero().update(health=0.0, hp=0.0)
    env.grid_env.agent_pos = env.merchant_interaction_tiles[0]
    mask = env.compute_action_mask()
    index = next(i for i, name in enumerate(env.scenario_merchant_item_names)
                 if mask[env.GRID_MERCHANT_POTION_BUY_ACTION_START + i])
    action = env.GRID_MERCHANT_POTION_BUY_ACTION_START + index
    assert env.compute_action_mask()[action]
    gold = env.gold
    env.step(action)
    assert env.gold < gold
