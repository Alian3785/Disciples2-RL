"""Permanent elixirs survive progression and temporary battle-stat lifecycles."""

from collections import Counter
from copy import deepcopy

import pytest

from battle_env import BattleEnv, _apply_dynamic_unit_levelup, _apply_hero_levelup_bonuses
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


UNITS = {unit["кто"]: unit for unit in DATA}
STATS = (
    "damage", "damage_secondary", "accuracy", "accuracy_secondary",
    "initiative_base", "armor", "max_health",
)


@pytest.fixture
def env():
    value = CampaignEnv(
        map_name="scroll_train", observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    value.reset(seed=42)
    yield value
    value.close()


def unit_named(name="Демон", position=8):
    unit = map_unit_to_battle(UNITS[name], "blue", position)
    unit["hp"] = unit["health"]
    unit["maxhp"] = unit["max_health"]
    return unit


def definitions(env):
    return [d for d in env.POTION_ITEM_DEFINITIONS if d.get("duration") == "permanent"]


def apply_doses(env, unit, doses=1):
    for _ in range(doses):
        for definition in definitions(env):
            env._apply_permanent_potion_bonus_to_unit(unit, definition)


def stats(unit):
    return {key: unit.get(key, 0) for key in STATS}


def place(env, unit):
    env.blue_team_state = [
        unit if u["position"] == unit["position"] else u
        for u in env.blue_team_state
    ]
    env._mark_equipment_dirty()
    return unit


def save_prepared(env, prepared):
    battle = BattleEnv(log_enabled=False)
    battle.combined = deepcopy(prepared)
    env.battle_env = battle
    env._save_blue_state()
    return next(u for u in env.blue_team_state if u["position"] == 8)


@pytest.mark.parametrize("dead", [False, True])
def test_all_permanent_stats_survive_temporary_layers_and_repeated_battle_save(env, dead):
    unit = place(env, unit_named())
    unit.update(damage_secondary=17, accuracy_secondary=61)
    apply_doses(env, unit, doses=2)
    expected = stats(unit)
    unit["health"] = unit["hp"] = 0 if dead else 91
    expected_hp = unit["health"]
    for definition in env.POTION_ITEM_DEFINITIONS:
        if definition.get("duration") == "temporary":
            env._active_potion_positions(definition["name"]).add(8)
    env._blue_stack_spell_effect_summary = lambda: {
        "active_spell_ids": ["test"], "damage_multiplier": 1.25,
        "initiative_multiplier": 1.2, "accuracy_multiplier": 1.15,
        "armor_delta": 17, "health_delta": 50,
        "resistance_types": ["Mind"],
    }
    for _ in range(3):
        prepared = deepcopy(env.blue_team_state)
        env._apply_active_blue_potion_effects(prepared)
        env._apply_active_blue_support_spell_effects(prepared)
        restored = save_prepared(env, prepared)
        assert stats(restored) == expected
        assert restored["health"] == restored["hp"] == expected_hp
        assert all(restored[d["permanent_counter_key"]] == 2 for d in definitions(env))
        assert not any(k.startswith("campaign_potion_base_") for k in restored)
        assert not any(k.startswith("campaign_map_spell_base_") for k in restored)


@pytest.mark.parametrize("name", ["Демон", "Дева рощи", "Солнечная танцовщица", "Сильфида"])
@pytest.mark.parametrize("next_level", [3, 10, 11])
def test_growth_rebuilds_permanent_layer_from_grown_intrinsic_stats(env, name, next_level):
    plain = unit_named(name)
    plain["Level"] = next_level - 1
    actual = deepcopy(plain)
    apply_doses(env, actual, doses=2)
    _apply_dynamic_unit_levelup(actual, actual["exp_required"])
    _apply_dynamic_unit_levelup(plain, plain["exp_required"])
    apply_doses(env, plain, doses=2)
    assert actual["Level"] == next_level
    assert stats(actual) == stats(plain)
    assert actual["health"] == actual["max_health"]
    for definition in definitions(env):
        assert actual[definition["permanent_counter_key"]] == 2


@pytest.mark.parametrize("next_level", [4, 5, 13, 14, 15])
def test_hero_milestones_rebuild_elixirs_after_intrinsic_growth(env, next_level):
    plain = deepcopy(env._resolve_travel_hero())
    plain["Level"] = next_level
    actual = deepcopy(plain)
    apply_doses(env, actual, doses=2)
    _apply_hero_levelup_bonuses(actual)
    _apply_hero_levelup_bonuses(plain)
    apply_doses(env, plain, doses=2)
    assert stats(actual) == stats(plain)
    assert actual["health"] == actual["max_health"]
    assert actual.get("hero_level_stat_abilities") == plain.get("hero_level_stat_abilities")


def test_lord_might_updates_intrinsic_source_once(env):
    env.typeoflord = 1
    plain = deepcopy(env._resolve_travel_hero())
    plain["Level"] = env.HERO_ARTIFACT_KNOWLEDGE_LEVEL
    actual = deepcopy(plain)
    apply_doses(env, actual, doses=2)
    assert env._apply_lord_replacement_might_bonus(actual)
    assert not env._apply_lord_replacement_might_bonus(actual)
    assert env._apply_lord_replacement_might_bonus(plain)
    apply_doses(env, plain, doses=2)
    assert stats(actual) == stats(plain)


@pytest.mark.parametrize("doses", [1, 3])
def test_promotion_rebases_potions_to_new_type_once(env, doses):
    source = unit_named("Демон")
    apply_doses(env, source, doses=doses)
    promoted = unit_named("Молох")
    expected = deepcopy(promoted)
    apply_doses(env, expected, doses=doses)
    env._reapply_persistent_elixir_bonuses_to_promoted_unit(source, promoted)
    assert stats(promoted) == stats(expected)
    assert Counter(promoted["campaign_permanent_potions"]) == Counter(expected["campaign_permanent_potions"])
    for definition in definitions(env):
        assert promoted[definition["permanent_counter_key"]] == doses


def test_wight_devolution_restore_retains_original_permanent_layer(env):
    source = place(env, unit_named("Демон"))
    apply_doses(env, source, doses=2)
    expected = stats(source)
    battle = BattleEnv(log_enabled=False)
    prepared = deepcopy(env.blue_team_state)
    victim = next(u for u in prepared if u["position"] == 8)
    attacker = {"name": "wight", "team": "red", "position": 1}
    assert battle._apply_wight_decay_effect(attacker, victim)
    assert victim["transformed"]
    battle.combined = prepared
    env.battle_env = battle
    env._save_blue_state()
    restored = next(u for u in env.blue_team_state if u["position"] == 8)
    assert stats(restored) == expected
    assert restored["health"] == restored["max_health"]
    assert restored["transformed"] == 0
    assert not restored.get("wight_form_name")
    for definition in definitions(env):
        assert restored[definition["permanent_counter_key"]] == 2


def test_legacy_counter_without_new_source_does_not_replay_old_effect(env):
    source = place(env, unit_named())
    definition = next(d for d in definitions(env) if d["effect"]["kind"] == "damage")
    # An older save already baked two doses into damage. They must not be
    # applied again simply because a new source model sees those counters.
    source["damage"] = 97
    source["original_damage"] = 97
    source[definition["permanent_counter_key"]] = 2
    source["campaign_permanent_potions"] = [definition["name"]] * 2
    env._refresh_campaign_equipment_effects()
    assert source["damage"] == 97
    env._apply_permanent_potion_bonus_to_unit(source, definition)
    expected = env._normalize_damage_value(97 * definition["effect"]["multiplier"])
    assert source["damage"] == expected
    assert source[definition["permanent_counter_key"]] == 3
    for _ in range(3):
        env._mark_equipment_dirty()
        env._refresh_campaign_equipment_effects()
        assert source["damage"] == expected


@pytest.mark.parametrize("name", ["Демон", "Сильфида"])
def test_growth_while_temporary_layers_active_saves_only_grown_permanent_values(env, name):
    unit = place(env, unit_named(name))
    unit["Level"] = 2
    expected_unit = deepcopy(unit)
    apply_doses(env, unit, doses=2)
    _apply_dynamic_unit_levelup(expected_unit, expected_unit["exp_required"])
    apply_doses(env, expected_unit, doses=2)
    for definition in env.POTION_ITEM_DEFINITIONS:
        if definition.get("duration") == "temporary":
            env._active_potion_positions(definition["name"]).add(8)
    env._blue_stack_spell_effect_summary = lambda: {
        "active_spell_ids": ["test"], "damage_multiplier": 1.25,
        "initiative_multiplier": 1.2, "accuracy_multiplier": 1.15,
        "armor_delta": 17, "health_delta": 50,
    }
    prepared = deepcopy(env.blue_team_state)
    env._apply_active_blue_potion_effects(prepared)
    env._apply_active_blue_support_spell_effects(prepared)
    target = next(u for u in prepared if u["position"] == 8)
    _apply_dynamic_unit_levelup(target, target["exp_required"])
    restored = save_prepared(env, prepared)
    assert stats(restored) == stats(expected_unit)
    assert restored["health"] == restored["max_health"]
    # Re-enter with the same still-active buffs and save repeatedly: the
    # temporary multipliers must never become part of the permanent source.
    for _ in range(3):
        prepared = deepcopy(env.blue_team_state)
        env._apply_active_blue_potion_effects(prepared)
        env._apply_active_blue_support_spell_effects(prepared)
        restored = save_prepared(env, prepared)
        assert stats(restored) == stats(expected_unit)
        assert restored["health"] == restored["max_health"]


@pytest.mark.parametrize("copy_form,dead", [(False, False), (True, False), (True, True)])
def test_doppelganger_round_reset_retains_own_elixirs(env, copy_form, dead):
    unit = unit_named("Двойник")
    apply_doses(env, unit, doses=2)
    expected = stats(unit)
    battle = BattleEnv(log_enabled=False)
    battle.combined = [unit]
    if copy_form:
        # The copied target has unrelated combat stats; the next round returns
        # to the Doppelganger's own permanent source, retaining the wound ratio.
        target = unit_named("Демон", position=7)
        unit["health"] = unit["hp"] = unit["max_health"] // 2
        assert battle._apply_doppelganger_copy(unit, target)
    if dead:
        unit["health"] = unit["hp"] = 0
    before_ratio = unit["health"] / unit["max_health"]
    battle._restore_default_doppelgangers()
    if dead:
        assert unit["health"] == unit["hp"] == 0
    else:
        assert stats(unit) == expected
        assert unit["health"] == round(expected["max_health"] * before_ratio)
        assert unit["hp"] == unit["health"]
        for _ in range(3):
            battle._restore_default_doppelgangers()
            assert stats(unit) == expected
    for definition in definitions(env):
        assert unit[definition["permanent_counter_key"]] == 2


def test_fractional_wounds_do_not_round_during_noop_syncs(env):
    hero = env._resolve_travel_hero()
    apply_doses(env, hero)
    hero["health"] = hero["hp"] = 31.125
    before = stats(hero)
    for _ in range(5):
        env._sync_hero_progression_flags()
        env._mark_equipment_dirty()
        env._refresh_campaign_equipment_effects()
        assert hero["health"] == hero["hp"] == 31.125
        assert stats(hero) == before


@pytest.mark.parametrize("persistent_hp", [91.125, 0.0])
def test_wounded_growth_excludes_temporary_flat_health_from_injury_ratio(env, persistent_hp):
    unit = place(env, unit_named())
    unit["Level"] = 2
    expected = deepcopy(unit)
    _apply_dynamic_unit_levelup(expected, expected["exp_required"])
    apply_doses(env, expected, doses=2)
    apply_doses(env, unit, doses=2)
    old_persistent_max = unit["max_health"]
    env._blue_stack_spell_effect_summary = lambda: {
        "active_spell_ids": ["test"], "health_delta": 50,
    }
    prepared = deepcopy(env.blue_team_state)
    env._apply_active_blue_support_spell_effects(prepared)
    target = next(u for u in prepared if u["position"] == 8)
    target["health"] = target["hp"] = persistent_hp + 50 if persistent_hp > 0 else 0
    expected_hp = int(persistent_hp * expected["max_health"] / old_persistent_max + 0.5)
    _apply_dynamic_unit_levelup(target, target["exp_required"])
    expected_runtime_hp = expected_hp + 50 if persistent_hp > 0 else 0
    assert target["health"] == expected_runtime_hp
    restored = save_prepared(env, prepared)
    assert stats(restored) == stats(expected)
    assert restored["health"] == restored["hp"] == expected_hp
