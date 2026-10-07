"""Campaign service prices follow the original unit, not heuristic power tiers."""
from copy import deepcopy

import pytest

from battle_env import BattleEnv
from campaign_env_economy import CampaignEconomyMixin
from data_dicts_compact_lines import DATA, map_unit_to_battle
from permanent_unit_stats import add_permanent_effect
from unit_service_cost_data import UNIT_SERVICE_COST_PROFILES
from unit_service_costs import known_unit_service_gold_costs


CATALOG = {entry["кто"]: entry for entry in DATA}
# Independent literal expectations from Gunits.HEAL_C / TRAINING_C.
ORIGINAL_PRICES = [
    ("Йети", 2, 4),
    ("Грифон", 2, 4),
    ("Баронесса", 1, 4),
    ("Королевский страж", 1, 4),
    ("Носферату", 1, 4),
    ("Элементалист", 2, 5),
    ("Имперский ассасин", 2, 5),
    ("Скваер", 1, 4),
    ("Нейтральный грифон", 2, 5),
    ("Медведь", 2, 5),
    ("Медуза", 3, 5),
    ("Гоблин старейшина", 2, 6),
]


def fighter(name):
    return map_unit_to_battle(CATALOG[name], "blue", 7)


@pytest.mark.parametrize("name,heal,training", ORIGINAL_PRICES)
@pytest.mark.parametrize("level", [1, 4, 10, 11, 20])
def test_original_prices_at_native_and_dynamic_levels(name, heal, training, level):
    unit = fighter(name)
    unit["Level"] = max(unit["Level"], level)
    assert CampaignEconomyMixin._castle_heal_gold_per_hp(unit) == heal
    assert CampaignEconomyMixin._trainer_gold_per_xp(unit) == training


def test_entire_runtime_catalog_has_original_prices():
    assert len(UNIT_SERVICE_COST_PROFILES) == 356
    missing = []
    for entry in DATA:
        unit = map_unit_to_battle(entry, "blue", 7)
        if unit["max_health"] > 0 and known_unit_service_gold_costs(unit) is None:
            missing.append((unit["name"], unit["unit_id"]))
    assert not missing


@pytest.mark.parametrize("id_key", ["unit_id", "game_unit_id"])
def test_permanent_id_wins_over_name_category_hp_and_temporary_form(id_key):
    unit = {id_key: " G000UU5004 ", "name": "Грифон", "Level": 20,
            "hero": 1, "capital": 1, "is_neutral_unit": False,
            "maxhp": 9999, "max_health": 9999, "_attack_form_unit_id": "g000uu0153"}
    assert known_unit_service_gold_costs(unit) == (2, 5)  # Neutral Griffin.


@pytest.mark.parametrize("name_key", ["name", "кто"])
def test_legacy_units_without_id_resolve_reviewed_name_bindings(name_key):
    assert known_unit_service_gold_costs({name_key: "Баронесса", "Level": 4}) == (1, 4)


def test_unknown_custom_profile_does_not_borrow_a_named_original_id():
    assert known_unit_service_gold_costs({"unit_id": "custom", "name": "Грифон"}) is None


@pytest.mark.parametrize("level,expected", [
    (4, (2, 5)), (5, (3, 7)), (10, (8, 17)), (11, (11, 21)), (12, (14, 25)),
])
def test_nonzero_dynamic_prices_use_native_level_and_both_growth_ranges(monkeypatch, level, expected):
    # Original prices currently have zero increments. Exercise both ranges so
    # a re-export from modified DBFs also applies GDynUpgr correctly.
    monkeypatch.setitem(UNIT_SERVICE_COST_PROFILES, "custom", (4, 10, (2, 5), (1, 2), (3, 4)))
    assert known_unit_service_gold_costs({"unit_id": "custom", "Level": level}) == expected


@pytest.mark.parametrize("level", [None, "bad", float("inf"), float("nan"), -1])
def test_invalid_or_subnative_level_keeps_native_prices(monkeypatch, level):
    monkeypatch.setitem(UNIT_SERVICE_COST_PROFILES, "custom", (4, 10, (2, 5), (1, 2), (3, 4)))
    assert known_unit_service_gold_costs({"unit_id": "custom", "Level": level}) == (2, 5)


def test_real_hero_levelups_and_health_elixirs_do_not_inflate_prices():
    unit = fighter("Баронесса")
    battle = BattleEnv(log_enabled=False)
    for level in range(2, 12):
        battle._apply_exp_award_to_unit(unit, unit["exp_required"] - unit["exp_current"])
        assert unit["Level"] == level
        assert known_unit_service_gold_costs(unit) == (1, 4)
    bear = fighter("Медведь")
    add_permanent_effect(bear, {"kind": "health", "multiplier": 3.0})
    assert bear["max_health"] > 500
    assert known_unit_service_gold_costs(bear) == (2, 5)


def service_party(campaign_factory, name):
    env = campaign_factory(map_name="trade_train", detailed_step_info=True)
    hero = deepcopy(env._resolve_travel_hero())
    hero.update(position=8, stand="ahead")
    target = env._build_unit_from_data(CATALOG[name], "blue", 7)
    target["exp_current"] = 0
    env.blue_team_state = env._build_battle_team_with_placeholders("blue", [hero, target])
    env._mark_equipment_dirty()
    env._sync_hero_progression_flags()
    target = next(u for u in env._get_blue_state() if u["position"] == 7)
    return env, target


@pytest.mark.parametrize("name,rate,_training", ORIGINAL_PRICES[:7])
@pytest.mark.parametrize("full", [False, True])
def test_public_healing_mask_and_payment_use_table_prices(campaign_factory, monkeypatch, name, rate, _training, full):
    env, target = service_party(campaign_factory, name)
    if name == "Баронесса":
        target["Level"] = 4
    max_hp = target["max_health"]
    target.update(hp=max_hp - 10, health=max_hp - 10, maxhp=max_hp)
    env.grid_env.agent_pos = next(iter(env.castle_heal_tiles))
    monkeypatch.setattr(env, "_has_temple_built", lambda: True)
    env.gold = 10 * rate + 1 if full else 3 * rate
    action = env.GRID_CASTLE_HEAL_ACTION_START + list(env.CASTLE_HEAL_POSITIONS).index(7)
    assert env.compute_action_mask()[action]
    info = env.step(action)[4]
    healed = 10 if full else 3
    assert info["healed_amount"] == healed
    assert info["gold_spent"] == healed * rate
    saved = next(u for u in env.blue_team_state if u["position"] == 7)
    assert saved["hp"] == saved["health"] == max_hp - 10 + healed
    assert env.gold == (1 if full else 0)
    assert not env.compute_action_mask()[action]


@pytest.mark.parametrize("name,_heal,rate", ORIGINAL_PRICES[:7])
@pytest.mark.parametrize("budget", ["three_xp", "insufficient", "near_cap"])
def test_public_training_preview_mask_and_payment_agree(campaign_factory, name, _heal, rate, budget):
    env, target = service_party(campaign_factory, name)
    if name == "Баронесса":
        target["Level"] = 4
    env.grid_env.agent_pos = env.trainer_interaction_tiles[0]
    cap = env._trainer_exp_cap(target)
    before = cap - 1 if budget == "near_cap" else 0
    target["exp_current"] = before
    level = target["Level"]
    gold = rate - 1 if budget == "insufficient" else 3 * rate
    env.gold = gold
    expected_xp = {"three_xp": 3, "insufficient": 0, "near_cap": 1}[budget]
    preview = env._trainer_preview_for_position(7)
    assert preview["gold_per_xp"] == rate
    assert preview["gold_needed_full"] == (cap - before) * rate
    assert preview["affordable_xp"] == expected_xp
    action = env.GRID_TRAINER_ACTION_START + list(env.TRAINER_POSITIONS).index(7)
    assert bool(env.compute_action_mask()[action]) == (expected_xp > 0)
    info = env.step(action)[4]  # Also exercise forced execution with too little gold.
    assert info["trainer_trained"] == (expected_xp > 0)
    assert info["trainer_gold_per_xp"] == rate
    assert info["trainer_xp_gained"] == expected_xp
    assert info["gold_spent"] == info["trainer_gold_spent"] == expected_xp * rate
    assert env.gold == gold - expected_xp * rate
    saved = next(u for u in env.blue_team_state if u["position"] == 7)
    assert saved["exp_current"] == before + expected_xp
    assert saved["Level"] == level
    assert not env.compute_action_mask()[action]


def test_evolution_picks_the_new_original_price_profile(campaign_factory):
    env = campaign_factory(map_name="trade_train", Realcapital=2)
    possessed = env._build_unit_from_data(CATALOG["Одержимый"], "blue", 7)
    assert known_unit_service_gold_costs(possessed) == (1, 4)
    building = next(entry for entry in env.active_buildings.values()
                    if isinstance(entry, dict) and entry.get("unit") == "Берсерк")
    building["Build"] = 1
    evolved = env._build_built_unit_promotion(possessed)
    assert evolved["name"] == "Берсерк"
    assert evolved["unit_id"] != possessed["unit_id"]
    assert known_unit_service_gold_costs(evolved) == (2, 5)
