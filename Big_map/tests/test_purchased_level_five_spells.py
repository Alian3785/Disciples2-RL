"""Purchased level-5 book spells are available to every ruler.

Public purchases/casts use real action routing and effects. Fixtures provide a
shop position/resources; the support case substitutes one shop catalog entry
without adding a production acquisition route. No spell effect is mocked.
"""
from copy import deepcopy
import pickle

import pytest

from campaign_env import CampaignEnv

SHOP_SPELLS = ("emp_d2_s022", "lod_d2_s021")


@pytest.fixture
def make_env():
    opened = []

    def create(capital=1, lord=1):
        env = CampaignEnv(
            map_name="default", Realcapital=capital, typeoflord=lord,
            observation_version="local5", scripted_capital_bot_enabled=False,
            use_boss_starting_roster=False, log_enabled=False,
        )
        env.reset(seed=42)
        assert (env.Realcapital, env.typeoflord) == (capital, lord)
        env.gold = 5000.0
        for attr in env.MANA_ATTR_BY_KIND.values():
            setattr(env, attr, 10000.0)
        env.grid_env.agent_pos = env.spell_shop_interaction_tiles[0]
        opened.append(env)
        return env

    yield create
    for env in opened:
        env.close()


def buy_action(env, key):
    index = next(i for i, entry in enumerate(env.SPELL_SHOP_BUY_SPELLS)
                 if entry["spell_id"] == key)
    return env.GRID_SPELL_SHOP_BUY_ACTION_START + index


def cast_action(env, key):
    for start, specs, field in (
        (env.grid_legion_damage_spell_action_start,
         env._current_map_offensive_spell_action_specs(), "id"),
        (env.grid_map_support_spell_action_start,
         env._current_map_support_spell_action_specs(), "id"),
        (env.grid_spell_shop_cast_action_start,
         env._spell_shop_cast_spell_entries(), "spell_id"),
    ):
        for index, entry in enumerate(specs):
            if entry[field] == key:
                return start + index
    raise AssertionError(f"No cast action for {key}")


def step(env, action):
    obs, _, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert not truncated
    assert not terminated
    return info


def buy(env, key):
    action = buy_action(env, key)
    assert env.compute_action_mask()[action]
    gold, mana = env.gold, env._current_mana_totals()
    info = step(env, action)
    assert info["spell_shop_buy_purchased"]
    assert env.gold == pytest.approx(gold - info["spell_shop_buy_price"])
    assert env._current_mana_totals() == mana
    assert info["spell_shop_buy_stock_after"] == info["spell_shop_buy_stock_before"] - 1
    assert key in env.spell_shop_purchased_spell_ids
    assert env._spell_shop_spell_owned(key)
    if key in env.active_spells:
        assert env._is_spell_learned(key)
        assert info["spell_shop_buy_added_to_spellbook"]
    return info


def resource_state(env):
    return deepcopy((env.gold, env._current_mana_totals(), env.active_spells,
                     env.spell_shop_stocks, env.spell_shop_purchased_spell_ids,
                     env.spell_cast_counts_by_id_this_turn,
                     env.spell_learning_locked))


def build_tower_fixture(env):
    tower = next(entry for entry in env.active_buildings.values()
                 if isinstance(entry, dict) and entry.get("name") == env.MAGIC_TOWER_BUILDING_NAME)
    tower["built"] = tower["Build"] = 1
    assert env._has_magic_tower_built()


@pytest.mark.parametrize("capital", (1, 2, 3))
@pytest.mark.parametrize("lord", (1, 2, 3))
@pytest.mark.parametrize("key", SHOP_SPELLS)
def test_purchase_cast_matrix_has_real_effects_and_exact_costs(make_env, capital, lord, key):
    env = make_env(capital, lord)
    shape, count = env.observation_space.shape, env.action_space.n
    action = cast_action(env, key)
    assert not env.compute_action_mask()[action]
    before = resource_state(env)
    assert not step(env, action)["spell_cast_executed"]
    assert resource_state(env) == before
    buy(env, key)
    assert env.compute_action_mask()[action]
    own = key in env.active_spells
    assert env._spell_shop_spell_has_regular_action(key) is own
    if own:
        assert not env._can_cast_spell_shop_spell(key)
    mana = env._current_mana_totals()
    costs = env._get_spell_use_costs(env._spell_entry_by_id(key))
    if key == "emp_d2_s022":
        target = env._get_map_offensive_spell_target(key)["enemy_id"]
        hp_before = sum(float(unit.get("health", 0)) for unit in env._get_enemy_team_state(target))
    info = step(env, action)
    assert info["spell_cast_executed"]
    assert not info["blocked_by_typeoflord"]
    for kind, amount in mana.items():
        assert env._current_mana_totals()[kind] == pytest.approx(amount - costs[kind])
    assert env._spell_cast_count_this_turn(key) == 1
    if key == "emp_d2_s022":
        assert info["spell_kind"] == "damage"
        assert info["spell_damage_total"] > 0
        hp_after = sum(float(unit.get("health", 0)) for unit in env._get_enemy_team_state(target))
        assert hp_before - hp_after == pytest.approx(info["spell_damage_total"])
    else:
        assert info["spell_kind"] == "summon_battle"
        assert info["battle_triggered"]
        assert env.mode == env.MODE_BATTLE
        assert env.current_battle_context["kind"] == "summon_spell"
        assert env.current_battle_context["spell_key"] == key
        assert env.current_battle_context["summoned_unit_name"] == "Мститель"
        living_blue = [u for u in env.battle_env.combined
                       if u.get("team") == "blue" and float(u.get("health", 0)) > 0]
        assert len(living_blue) == 1
        assert living_blue[0]["name"] == "Мститель"
    assert env.observation_space.shape == shape
    assert env.action_space.n == count


@pytest.mark.parametrize("lord", (1, 2, 3))
@pytest.mark.parametrize("capital,key", ((1, "emp_d2_s022"), (2, "lod_d2_s021")))
def test_level_five_research_remains_mage_only_before_and_after_purchase(make_env, lord, capital, key):
    env = make_env(capital, lord)
    build_tower_fixture(env)
    action = env.grid_spell_action_start + env.spell_keys.index(key)
    assert bool(env.compute_action_mask()[action]) is (lord == 2)
    if lord != 2:
        before = resource_state(env)
        info = step(env, action)
        assert info["blocked_by_typeoflord"]
        assert not info["spell_learned"]
        assert resource_state(env) == before
    buy(env, key)
    for learned in (1, 0):
        # Inconsistent old flags cannot make provenance a research permission.
        env.active_spells[key]["learned"] = learned
        assert bool(env.compute_action_mask()[action]) is (lord == 2 and learned == 0)
        before = resource_state(env)
        info = step(env, action)
        assert info["blocked_by_typeoflord"] is (lord != 2)
        assert info["spell_learned"] is (lord == 2 and learned == 0)
        if lord != 2 or learned:
            assert resource_state(env) == before


@pytest.mark.parametrize("lord", (1, 2, 3))
@pytest.mark.parametrize("capital,key", ((1, "emp_d2_s022"), (2, "lod_d2_s021")))
def test_learned_flag_alone_does_not_claim_purchase_provenance(make_env, lord, capital, key):
    env = make_env(capital, lord)
    env.active_spells[key]["learned"] = 1
    action = cast_action(env, key)
    assert bool(env.compute_action_mask()[action]) is (lord == 2)
    assert key not in env.spell_shop_purchased_spell_ids
    before = resource_state(env)
    info = step(env, action)
    assert info["blocked_by_typeoflord"] is (lord != 2)
    assert info["spell_cast_executed"] is (lord == 2)
    if lord != 2:
        assert resource_state(env) == before


@pytest.mark.parametrize("lord", (1, 2, 3))
@pytest.mark.parametrize("key", SHOP_SPELLS)
def test_purchase_requires_gold_and_rejects_duplicates_without_cost(make_env, lord, key):
    env = make_env(1, lord)
    action = buy_action(env, key)
    entry = next(entry for entry in env.SPELL_SHOP_BUY_SPELLS if entry["spell_id"] == key)
    price = env._shop_buy_price(entry["price"])
    env.gold = price - 1
    before = resource_state(env)
    assert not env.compute_action_mask()[action]
    info = step(env, action)
    assert not info["spell_shop_buy_purchased"]
    assert info["spell_shop_buy_insufficient_gold"]
    assert resource_state(env) == before
    env.gold = price
    buy(env, key)
    assert env.gold == 0
    before = resource_state(env)
    assert not env.compute_action_mask()[action]
    info = step(env, action)
    assert info["spell_shop_buy_already_owned"]
    assert not info["spell_shop_buy_purchased"]
    assert resource_state(env) == before


@pytest.mark.parametrize("capital", (1, 2))
@pytest.mark.parametrize("lord", (1, 2, 3))
@pytest.mark.parametrize("blocker", ("mana", "target", "daily_limit"))
def test_purchased_spell_does_not_bypass_cast_requirements(make_env, capital, lord, blocker):
    env = make_env(capital, lord)
    key = "emp_d2_s022"
    buy(env, key)
    action = cast_action(env, key)
    assert env.compute_action_mask()[action]
    if blocker == "mana":
        setattr(env, env.MANA_ATTR_BY_KIND["life"], 0.0)
    elif blocker == "target":
        for enemy_id in env.grid_env.enemies_alive:
            env.grid_env.enemies_alive[enemy_id] = False
    else:
        env.spell_cast_counts_by_id_this_turn[key] = env._same_spell_cast_limit_per_turn()
    before = resource_state(env)
    assert not env.compute_action_mask()[action]
    assert not step(env, action)["spell_cast_executed"]
    assert resource_state(env) == before


@pytest.mark.parametrize("capital", (1, 2))
@pytest.mark.parametrize("lord", (1, 2, 3))
def test_cast_daily_limit_and_next_turn_preserve_purchase(make_env, capital, lord):
    env = make_env(capital, lord)
    key = "emp_d2_s022"
    buy(env, key)
    action = cast_action(env, key)
    for _ in range(2 if lord == 2 and capital == 1 else 1):
        assert env.compute_action_mask()[action]
        assert step(env, action)["spell_cast_executed"]
    assert not env.compute_action_mask()[action]
    mana = env._current_mana_totals()
    assert not step(env, action)["spell_cast_executed"]
    assert env._current_mana_totals() == mana
    step(env, 8)
    assert key in env.spell_shop_purchased_spell_ids
    assert env._spell_cast_count_this_turn(key) == 0
    assert env.compute_action_mask()[action]
    assert step(env, action)["spell_cast_executed"]


@pytest.mark.parametrize("capital", (1, 2))
@pytest.mark.parametrize("lord", (1, 3))
@pytest.mark.parametrize("storage", ("deepcopy", "spell_state_pickle"))
def test_purchase_survives_environment_roundtrip_and_reset_clears_it(make_env, capital, lord, storage):
    env = make_env(capital, lord)
    key = "emp_d2_s022"
    buy(env, key)
    if storage == "deepcopy":
        copied = deepcopy(env)
    else:
        # There is no campaign save/load API; serialize the existing spell
        # state only. Full-env pickle has an unrelated pre-existing grid bug.
        copied = make_env(capital, lord)
        fields = ("active_spells", "spell_shop_purchased_spell_ids",
                  "spell_shop_stocks", "spell_cast_counts_by_id_this_turn")
        saved = pickle.loads(pickle.dumps({field: getattr(env, field) for field in fields}))
        for field, value in saved.items():
            setattr(copied, field, value)
    try:
        action = cast_action(copied, key)
        assert copied.compute_action_mask()[action]
        assert step(copied, action)["spell_cast_executed"]
        copied.reset(seed=42)
        assert copied.spell_shop_purchased_spell_ids == set()
        assert not copied._is_spell_learned(key)
        assert not copied.compute_action_mask()[action]
        assert not step(copied, action)["spell_cast_executed"]
        assert key in env.spell_shop_purchased_spell_ids
        assert env._spell_cast_count_this_turn(key) == 0
    finally:
        copied.close()


@pytest.mark.parametrize("lord", (1, 2, 3))
def test_support_spell_purchase_uses_same_rule_without_expanding_real_shop(make_env, lord):
    env = make_env(3, lord)
    key = "mcl_d2_s023"  # Real level-5 buff; test-only shop stock.
    shape, count = env.observation_space.shape, env.action_space.n
    env.SPELL_SHOP_BUY_SPELLS = (
        {"name": "Test level-5 support", "spell_id": key, "price": 1000.0, "stock": 1},
        env.SPELL_SHOP_BUY_SPELLS[1],
    )
    env.spell_shop_stocks = env._create_initial_spell_shop_stocks()
    action = cast_action(env, key)
    assert not env.compute_action_mask()[action]
    buy(env, key)
    assert env.compute_action_mask()[action]
    costs = env._get_spell_use_costs(env.active_spells[key])
    mana = env._current_mana_totals()
    info = step(env, action)
    assert info["spell_cast_executed"]
    assert not info["blocked_by_typeoflord"]
    assert info["spell_units_affected"] > 0
    assert info["spell_kind"] == "buff"
    assert env._blue_stack_spell_effect_summary()["damage_multiplier"] == pytest.approx(1.5)
    for kind, amount in mana.items():
        assert env._current_mana_totals()[kind] == pytest.approx(amount - costs[kind])
    assert env.observation_space.shape == shape
    assert env.action_space.n == count


@pytest.mark.parametrize("lord", (1, 2, 3))
def test_ordinary_researched_spells_still_cast_without_purchase(make_env, lord):
    env = make_env(1, lord)
    key = "emp_d2_s001"
    build_tower_fixture(env)
    research = env.grid_spell_action_start + env.spell_keys.index(key)
    assert env.compute_action_mask()[research]
    assert step(env, research)["spell_learned"]
    assert key not in env.spell_shop_purchased_spell_ids
    action = cast_action(env, key)
    assert env.compute_action_mask()[action]
    assert step(env, action)["spell_cast_executed"]


@pytest.mark.parametrize("lord", (1, 2, 3))
@pytest.mark.parametrize("blocker", ("site", "stock"))
def test_purchase_requires_shop_and_stock(make_env, lord, blocker):
    env = make_env(1, lord)
    key = "emp_d2_s022"
    if blocker == "site":
        env.grid_env.agent_pos = (0, 0)
        assert not env._spell_shop_sites_at_position(env.grid_env.agent_pos)
    else:
        for stock in env.spell_shop_stocks.values():
            stock[env.SPELL_SHOP_BUY_SPELLS[0]["name"]] = 0
    action = buy_action(env, key)
    before = resource_state(env)
    assert not env.compute_action_mask()[action]
    assert not step(env, action)["spell_shop_buy_purchased"]
    assert resource_state(env) == before


@pytest.mark.parametrize("lord", (1, 3))
def test_purchase_does_not_enable_disabled_illusion(make_env, lord):
    env = make_env(2, lord)
    key = "lod_d2_s002"
    # Legacy/invalid acquisition data must not revive explicitly disabled magic.
    env.active_spells[key]["learned"] = 1
    env.spell_shop_purchased_spell_ids.add(key)
    assert not env._can_cast_legion_damage_spell(key)
    assert not env._can_cast_support_spell(key)
    assert not env._can_cast_spell_shop_spell(key)
    action = env.grid_spell_action_start + env.spell_keys.index(key)
    before = resource_state(env)
    assert not env.compute_action_mask()[action]
    assert not step(env, action)["spell_learned"]
    assert resource_state(env) == before


@pytest.mark.parametrize("lord", (1, 3))
def test_purchase_provenance_does_not_make_unsupported_research_available(make_env, lord):
    env = make_env(2, lord)
    key = "lod_d2_s022"
    build_tower_fixture(env)
    env.spell_shop_purchased_spell_ids.add(key)
    action = env.grid_spell_action_start + env.spell_keys.index(key)
    before = resource_state(env)
    assert not env.compute_action_mask()[action]
    info = step(env, action)
    assert not info["spell_research_supported"]
    assert not info["spell_learned"]
    assert resource_state(env) == before
