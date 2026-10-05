"""Permanent growth commutes with temporary potions across the whole roster.

The independent control grows without temporary effects, then receives the same
real effect definitions. The subject receives them first. Both the active battle
stats and the pre-potion snapshots must agree; saving alone can hide lost buffs.
These are deterministic arithmetic/lifecycle tests, not training or replays.
"""
from copy import deepcopy

import pytest

from battle_env import (
    BattleEnv, _apply_dynamic_unit_levelup, _apply_hero_levelup_bonuses,
    _is_travel_hero_unit,
)
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from hero_level_abilities import HERO_LEVEL_STAT_ABILITIES, apply_hero_level_stat_abilities
from unit_dynamic_stats import DYNAMIC_STAT_PROFILES, apply_dynamic_stat_growth


TEMPORARY = tuple(p for p in CampaignEnv.POTION_ITEM_DEFINITIONS
                  if p.get("duration") == "temporary")
PERMANENT = tuple(p for p in CampaignEnv.POTION_ITEM_DEFINITIONS
                  if p.get("duration") == "permanent")
TEMPLATES = tuple(map_unit_to_battle(source, "blue", 7) for source in DATA)
BY_NAME = {unit["name"]: unit for unit in TEMPLATES}
HEROES = tuple(unit for unit in TEMPLATES if _is_travel_hero_unit(unit))
STATS = ("damage", "damage_secondary", "accuracy", "accuracy_secondary",
         "initiative_base", "armor", "max_health", "health")
MILESTONES = (3, 4, 5, 6, 7, 8, 9, 10) + tuple(row[0] for row in HERO_LEVEL_STAT_ABILITIES)


class AllPotionScenario(CampaignEnv):
    def _scenario_potion_item_names(self):
        return tuple(p["name"] for p in self.POTION_ITEM_DEFINITIONS)


@pytest.fixture
def env():
    value = AllPotionScenario(
        map_name="trade_train", observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    value.reset(seed=42)
    yield value
    value.close()


def fresh(template="Герцог", *, team="blue", next_level=5):
    unit = deepcopy(BY_NAME[template] if isinstance(template, str) else template)
    unit.update(team=team, position=7 if team == "blue" else 1,
                Level=next_level - 1, turns_into=[])
    if _is_travel_hero_unit(unit):
        apply_hero_level_stat_abilities(unit)
    unit["hp"] = unit["health"]
    unit["maxhp"] = unit["max_health"]
    unit["original_damage"] = unit["damage"]
    unit["base_armor"] = unit["armor"]
    return unit


def public_stats(unit):
    return {key: unit.get(key, 0) for key in STATS}


def permanent_stats(unit):
    """Observe the actual persistence snapshots, without production rebuilds."""
    return {
        key: unit.get("campaign_potion_base_" + (
            "initiative" if key == "initiative_base" else key), unit.get(key, 0))
        for key in STATS
    }


def apply_doses(unit, doses):
    harness = object.__new__(CampaignEnv)
    for _ in range(doses):
        for potion in PERMANENT:
            harness._apply_permanent_potion_bonus_to_unit(unit, potion)


def apply_potions(unit, potions=TEMPORARY):
    """Use production potion arithmetic, including stacking, rounding and wards.

    The position registry is campaign plumbing; the side-neutral battle-stat
    helper is intentionally also tested on RED runtime fighters.
    """
    harness = object.__new__(CampaignEnv)
    harness._campaign_logs = []
    harness.log_enabled = False
    for potion in potions:
        harness._active_potion_positions(potion["name"]).add(unit["position"])
    harness._apply_active_blue_potion_effects([unit])
    # BattleEnv initialization records the damage visible at battle entry.
    unit["original_damage"] = unit["damage"]
    return unit


def assert_same_layers(actual, permanent, buffed, context):
    assert permanent_stats(actual) == public_stats(permanent), context
    assert public_stats(actual) == public_stats(buffed), context
    assert set(actual.get("resistance", [])) == set(buffed.get("resistance", [])), context
    assert 0 <= actual["health"] <= actual["max_health"], context
    assert actual["hp"] == actual["health"], context
    assert actual["maxhp"] == actual["max_health"], context
    assert actual["original_damage"] == actual["damage"], context
    for potion in PERMANENT:
        counter = potion["permanent_counter_key"]
        assert actual.get(counter, 0) == permanent.get(counter, 0), context


def award_level(battle, unit):
    before = unit["Level"]
    unit["exp_current"] = unit["exp_required"] - 1
    battle._apply_exp_award_to_unit(unit, 1)
    assert unit["Level"] == before + 1
    assert unit["exp_current"] == 0


def put_in_campaign(env, unit):
    env.blue_team_state = env._build_battle_team_with_placeholders("blue", [unit])
    env._mark_equipment_dirty()


def save_unit(env, unit):
    battle = BattleEnv(log_enabled=False)
    battle.combined = env._build_battle_team_with_placeholders("blue", [deepcopy(unit)])
    battle.winner = "blue"
    env.battle_env = battle
    env._save_blue_state()
    return next(u for u in env.blue_team_state if u["position"] == unit["position"])


def activate_real_potion(env, unit, potion):
    env._add_hero_item(potion["name"])
    action = (env._potion_action_start_for_item(potion["name"])
              + env.GRID_POTION_USE_POSITIONS.index(unit["position"]))
    assert env.compute_action_mask()[action]
    count = env._potion_available_count(potion["name"])
    info = env.step(action)[-1]
    assert info["potion_applied"] and info["potion_consumed"]
    assert env._potion_available_count(potion["name"]) == count - 1
    assert unit["position"] in env._active_potion_positions(potion["name"])


def test_real_vigor_potion_xp_strength_save_and_expiry(env):
    """B13: 50 + 10 level growth, then strength x1.25 = 75 permanently."""
    unit = fresh(next_level=5)
    unit.update(damage=50, original_damage=50)
    assert "campaign_stat_sources" not in unit
    put_in_campaign(env, unit)
    potion = next(p for p in TEMPORARY if p["effect"] == {
        "kind": "damage", "multiplier": 1.15,
    })
    activate_real_potion(env, unit, potion)
    grown = deepcopy(unit)
    env._apply_active_blue_potion_effects([grown])
    assert grown["damage"] == round(50 * 1.15)
    grown["original_damage"] = grown["damage"]
    award_level(BattleEnv(log_enabled=False), grown)
    assert grown["campaign_potion_base_damage"] == 75
    assert grown["damage"] == round(75 * 1.15)
    saved = save_unit(env, grown)
    assert saved["Level"] == 5
    assert saved["damage"] == 75
    assert not any(k.startswith("campaign_potion_base_") for k in saved)
    # The same still-active potion must be re-applied from the grown baseline.
    for _ in range(2):
        rebuilt = deepcopy(saved)
        env._apply_active_blue_potion_effects([rebuilt])
        assert rebuilt["damage"] == round(75 * 1.15)
        saved = save_unit(env, rebuilt)
        assert saved["damage"] == 75
    env._clear_expired_combat_potion_effects()
    expired = deepcopy(saved)
    env._apply_active_blue_potion_effects([expired])
    assert expired["damage"] == 75
    assert not any(k.startswith("campaign_potion_base_") for k in expired)


@pytest.mark.parametrize("potion", TEMPORARY, ids=lambda p: p["name"])
@pytest.mark.parametrize("name,next_level", [("Герцог", 5), ("Сатир", 10)])
def test_each_real_temporary_potion_preserves_growth_and_expires(env, potion, name, next_level):
    unit = fresh(name, next_level=next_level)
    control = deepcopy(unit)
    battle = BattleEnv(log_enabled=False)
    award_level(battle, control)
    expected_live = apply_potions(deepcopy(control), [potion])
    put_in_campaign(env, unit)
    activate_real_potion(env, unit, potion)
    actual = deepcopy(unit)
    env._apply_active_blue_potion_effects([actual])
    actual["original_damage"] = actual["damage"]
    award_level(battle, actual)
    assert_same_layers(actual, control, expected_live, (name, potion["name"]))
    saved = save_unit(env, actual)
    assert public_stats(saved) == public_stats(control)
    assert set(saved.get("resistance", [])) == set(control.get("resistance", []))
    env._clear_expired_combat_potion_effects()
    expired = deepcopy(saved)
    env._apply_active_blue_potion_effects([expired])
    assert public_stats(expired) == public_stats(control)


@pytest.mark.parametrize("next_level", [10, 11])
@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("doses", [0, 2])
def test_all_252_profiles_keep_exact_growth_with_every_potion(next_level, team, doses):
    assert len(TEMPLATES) == 252
    assert len({u["unit_id"] for u in TEMPLATES}) == 252
    assert all(u["unit_id"] in DYNAMIC_STAT_PROFILES for u in TEMPLATES)
    for template in TEMPLATES:
        actual = fresh(template, team=team, next_level=next_level)
        permanent = deepcopy(actual)
        # Elixirs remain a permanent layer above intrinsic growth.
        apply_doses(actual, doses)
        apply_potions(actual)
        assert apply_dynamic_stat_growth(permanent, next_level)
        apply_doses(permanent, doses)
        buffed = apply_potions(deepcopy(permanent))
        assert apply_dynamic_stat_growth(actual, next_level)
        assert_same_layers(actual, permanent, buffed, (template["name"], next_level, team, doses))


@pytest.mark.parametrize("next_level", MILESTONES)
@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("doses", [0, 2])
def test_all_20_heroes_milestones_ignore_temporary_stats(next_level, team, doses):
    assert len(HEROES) == 20
    battle = BattleEnv(log_enabled=False)
    for template in HEROES:
        actual = fresh(template, team=team, next_level=next_level)
        permanent = deepcopy(actual)
        apply_doses(actual, doses)
        apply_potions(actual)
        award_level(battle, permanent)
        apply_doses(permanent, doses)
        buffed = apply_potions(deepcopy(permanent))
        award_level(battle, actual)
        assert_same_layers(actual, permanent, buffed, (template["name"], next_level, team, doses))
        assert actual.get("hero_level_stat_abilities") == permanent.get("hero_level_stat_abilities")
        assert actual["needaunit"] == permanent["needaunit"]
        assert actual["exp_required"] == permanent["exp_required"]


@pytest.mark.parametrize("skill", HERO_LEVEL_STAT_ABILITIES, ids=lambda row: row[1])
@pytest.mark.parametrize("team", ["blue", "red"])
def test_direct_catalog_skill_grants_are_permanent_and_idempotent(skill, team):
    required, key, _, _, _, _ = skill
    for template in HEROES:
        actual = fresh(template, team=team, next_level=required)
        actual["Level"] = required
        permanent = deepcopy(actual)
        apply_potions(actual)
        apply_hero_level_stat_abilities(permanent)
        expected_live = apply_potions(deepcopy(permanent))
        apply_hero_level_stat_abilities(actual)
        assert key in actual["hero_level_stat_abilities"]
        assert_same_layers(actual, permanent, expected_live, (template["name"], key, team))
        once = deepcopy(actual)
        apply_hero_level_stat_abilities(actual)
        assert actual == once


@pytest.mark.parametrize("name,next_level", [("Герцог", 4), ("Герцог", 5), ("Демон", 11)])
@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("health_state", ["wounded", "one_hp", "dead"])
@pytest.mark.parametrize("doses", [0, 2])
def test_wounded_and_dead_hp_accuracy_caps_survive_growth(name, next_level, team, health_state, doses):
    unit = fresh(name, team=team, next_level=next_level)
    unit.update(accuracy=99, accuracy_secondary=99)
    apply_doses(unit, doses)
    hp = {"wounded": unit["max_health"] // 3, "one_hp": 1, "dead": 0}[health_state]
    unit["health"] = unit["hp"] = hp
    actual, permanent = deepcopy(unit), deepcopy(unit)
    # Potions are applied while alive; the fighter may then be killed in battle.
    if not hp:
        actual["health"] = actual["hp"] = 1
    apply_potions(actual)
    actual["health"] = actual["hp"] = hp
    def mutate(u):
        _apply_dynamic_unit_levelup(u, u["exp_required"])
    if _is_travel_hero_unit(unit):
        def mutate(u):
            u["Level"] += 1
            _apply_hero_levelup_bonuses(u)
    mutate(permanent)
    expected_live = deepcopy(permanent)
    if not hp:
        expected_live["health"] = expected_live["hp"] = 1
    apply_potions(expected_live)
    if not hp:
        expected_live["health"] = expected_live["hp"] = 0
    mutate(actual)
    assert_same_layers(actual, permanent, expected_live, (name, next_level, team, health_state, doses))
    assert actual["accuracy"] <= 100 and actual["accuracy_secondary"] <= 100
    assert actual["health"] == 0 if not hp else actual["health"] > 0


@pytest.mark.parametrize("name,next_level", [
    ("Герцог", 4), ("Герцог", 5), ("Герцог", 13),
    ("Герцог", 14), ("Герцог", 15), ("Сатир", 10), ("Сильфида", 11),
])
@pytest.mark.parametrize("doses", [0, 2])
def test_stacked_potions_spells_equipment_save_and_removal(env, monkeypatch, name, next_level, doses):
    # The spell summary isolates composition arithmetic from random map layout.
    # All potion, artifact and banner applications use their production methods.
    env.typeoflord = 0
    monkeypatch.setattr(env, "_blue_stack_spell_effect_summary", lambda: {
        "active_spell_ids": ["growth-composition"], "damage_multiplier": 1.25,
        "initiative_multiplier": 1.2, "accuracy_multiplier": 1.15,
        "armor_delta": 17, "health_delta": 50, "resistance_types": ["Mind"],
    })
    target = fresh(name, next_level=next_level)
    permanent = deepcopy(target)
    battle = BattleEnv(log_enabled=False)
    award_level(battle, permanent)
    apply_doses(permanent, doses)
    apply_doses(target, doses)
    carrier = target if _is_travel_hero_unit(target) else fresh(next_level=5)
    carrier["hero_abilities"] = ["banner_bearer", "artifact_knowledge"]
    if carrier is not target:
        target["position"] = permanent["position"] = 8
    party = [carrier] if carrier is target else [carrier, target]
    env.blue_team_state = env._build_battle_team_with_placeholders("blue", party)
    equipment = [env.RING_OF_AGES_ITEM_NAME, env.RUNESTONE_ARTIFACT_ITEM_NAME,
                 env.BANNER_OF_STRENGTH_ITEM_NAME]
    for item in equipment:
        env._add_hero_item(item)
    for potion in TEMPORARY:
        assert env._apply_temporary_potion_to_position(potion["name"], target["position"])[0]

    def prepare(team):
        team = deepcopy(team)
        env._clear_equipped_banner_effects(team)
        env._clear_equipped_artifact_effects(team)
        env._apply_active_blue_potion_effects(team)
        env._apply_active_blue_support_spell_effects(team)
        env._apply_equipped_artifact_effects(team, log=False)
        env._apply_equipped_banner_effects(team, log=False)
        for unit in team:
            unit["original_damage"] = unit["damage"]
        return team

    actual_party = prepare(env.blue_team_state)
    actual = next(u for u in actual_party if u["position"] == target["position"])
    permanent["hero_abilities"] = list(target.get("hero_abilities", []))
    control_party = [permanent] if carrier is target else [deepcopy(carrier), permanent]
    expected_live = next(u for u in prepare(control_party)
                         if u["position"] == target["position"])
    award_level(battle, actual)
    assert public_stats(actual) == public_stats(expected_live)
    assert actual["original_damage"] == expected_live["original_damage"]
    assert actual["campaign_potion_base_damage"] == permanent["damage"]
    assert actual["campaign_map_spell_base_max_health"] == permanent["max_health"]
    assert set(actual["resistance"]) == set(expected_live["resistance"])
    assert actual.get("campaign_active_banner") == env.BANNER_OF_STRENGTH_ITEM_NAME
    if carrier is target:
        assert set(actual["campaign_active_artifacts"]) == set(equipment[:2])
    battle.combined = actual_party
    battle.winner = "blue"
    env.battle_env = battle
    env._save_blue_state()
    # A campaign save re-equips gear but removes the temporary battle layers.
    env._clear_expired_combat_potion_effects()
    for item in reversed(equipment):
        assert env._consume_hero_item(item)
    env._mark_equipment_dirty()
    env._refresh_campaign_equipment_effects()
    saved = next(u for u in env.blue_team_state if u["position"] == target["position"])
    assert public_stats(saved) == public_stats(permanent)
    assert set(saved.get("resistance", [])) == set(permanent.get("resistance", []))
    assert not any(k.startswith(("campaign_potion_base_", "campaign_map_spell_base_")) for k in saved)
    for _ in range(3):
        env._mark_equipment_dirty()
        env._refresh_campaign_equipment_effects()
        assert public_stats(saved) == public_stats(permanent)
    for potion in PERMANENT:
        assert saved.get(potion["permanent_counter_key"], 0) == doses


@pytest.mark.parametrize("lord_type,next_level", [(1, 9), (2, 10), (3, 8)])
@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("doses", [0, 2])
def test_lord_might_ignores_potions_and_applies_once(lord_type, next_level, team, doses):
    env = object.__new__(CampaignEnv)
    env.typeoflord = lord_type
    actual = fresh(team=team, next_level=next_level)
    actual["Level"] = next_level
    permanent = deepcopy(actual)
    apply_doses(actual, doses)
    apply_potions(actual)
    assert env._apply_lord_replacement_might_bonus(permanent)
    apply_doses(permanent, doses)
    expected_live = apply_potions(deepcopy(permanent))
    assert env._apply_lord_replacement_might_bonus(actual)
    assert_same_layers(actual, permanent, expected_live, (lord_type, team, doses))
    once = deepcopy(actual)
    assert not env._apply_lord_replacement_might_bonus(actual)
    assert actual == once


@pytest.mark.parametrize("hero", [False, True])
@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("doses", [0, 2])
def test_unknown_profile_ten_percent_fallback_ignores_potions(hero, team, doses):
    actual = fresh(team=team, next_level=5)
    actual.update(name="Unknown growth profile", unit_id="unknown-growth-profile",
                  hero=hero, damage=51, original_damage=51)
    permanent = deepcopy(actual)
    apply_doses(actual, doses)
    apply_potions(actual)
    def mutate(u):
        _apply_dynamic_unit_levelup(u, u["exp_required"])
    if hero:
        def mutate(u):
            u["Level"] += 1
            _apply_hero_levelup_bonuses(u)
    mutate(permanent)
    apply_doses(permanent, doses)
    expected_live = apply_potions(deepcopy(permanent))
    mutate(actual)
    assert_same_layers(actual, permanent, expected_live, (hero, team, doses))
