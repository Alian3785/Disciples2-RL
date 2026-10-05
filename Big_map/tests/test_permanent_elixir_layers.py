"""EM3: permanent potion stats belong to a unit, never an equipment snapshot.

Exercise normal potion actions and masks against independently calculated,
gearless-first controls. No training, scripted bot, or boss roster is started.
"""
from copy import deepcopy
import json
import pickle
from types import SimpleNamespace

import pytest

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


class AllPotionScenario(CampaignEnv):
    """Expose existing potion definitions without changing production scenarios."""

    def _scenario_potion_item_names(self):
        return tuple(spec["name"] for spec in self.POTION_ITEM_DEFINITIONS)


POTIONS = tuple(spec for spec in CampaignEnv.POTION_ITEM_DEFINITIONS
                if spec.get("duration") == "permanent")
POTION_IDS = [spec["effect"]["kind"] for spec in POTIONS]
STAT_KEYS = ("damage", "damage_secondary", "armor", "accuracy",
             "accuracy_secondary", "initiative_base", "max_health", "health")
TARGETS = ("hero", "ordinary", "healer", "secondary")
GEAR = ("none", "artifact", "banner", "both")
UNIT_NAMES = {"ordinary": "Одержимый", "healer": "Дева рощи", "secondary": "Сатир"}


def stats(unit, *, health=True):
    return {key: unit[key] for key in STAT_KEYS if health or key != "health"}


def expected_permanent(base, spec, doses):
    """Independent public-stat oracle, retaining per-dose integer rounding."""
    result = deepcopy(base)
    effect = spec["effect"]
    kind = effect["kind"]
    keys = {"damage": ("damage", "damage_secondary"),
            "accuracy": ("accuracy", "accuracy_secondary"),
            "initiative": ("initiative_base",),
            "health": ("max_health", "health")}.get(kind, ())
    for _ in range(doses):
        if kind == "armor":
            result["armor"] += effect["armor_bonus"]
        else:
            for key in keys:
                value = max(0, int(round(result[key] * effect["multiplier"])))
                if kind == "accuracy":
                    value = min(100, value)
                if key == "max_health":
                    value = max(result[key] + 1, value)
                result[key] = value
    return result


@pytest.fixture
def make_env():
    environments = []

    def make(target="hero", *, cls=AllPotionScenario, map_name="default"):
        env = cls(map_name=map_name, Realcapital=2, observation_version="local5",
                  scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
                  log_enabled=False, detailed_step_info=True)
        environments.append(env)
        env.reset(seed=42)
        hero = env._resolve_travel_hero()
        hero["hero_abilities"] = list(set(hero.get("hero_abilities", [])) | {"banner_bearer"})
        if target == "hero":
            unit = hero
        else:
            position = next(u["position"] for u in env.blue_team_state
                            if not env._is_hero_unit(u) and u["max_health"] > 0)
            unit = map_unit_to_battle(next(d for d in DATA if d["кто"] == UNIT_NAMES[target]),
                                      "blue", position)
            env.blue_team_state = [unit if u["position"] == position else u
                                   for u in env.blue_team_state]
            env._mark_equipment_dirty()
        env._refresh_campaign_equipment_effects()
        return env, unit

    yield make
    for env in environments:
        env.close()


def gear_names(env, spec, gear):
    names = []
    if gear in ("artifact", "both"):
        names += [env.RING_OF_AGES_ITEM_NAME, env.RUNESTONE_ARTIFACT_ITEM_NAME]
    if gear in ("banner", "both"):
        names.append({"damage": env.BANNER_OF_STRENGTH_ITEM_NAME,
                      "initiative": env.BANNER_OF_SPEED_ITEM_NAME,
                      "accuracy": env.BANNER_OF_STRIKING_ITEM_NAME,
                      "armor": env.BANNER_OF_PROTECTION_ITEM_NAME,
                      "health": env.BANNER_OF_HEALTH_ITEM_NAME}[spec["effect"]["kind"]])
    return names


def equip(env, names):
    for name in names:
        env._add_hero_item(name)


def remove(env, names):
    for name in reversed(names):
        assert env._consume_hero_item(name)


def potion_action(env, spec, unit):
    return (env._potion_action_start_for_item(spec["name"])
            + env.GRID_POTION_USE_POSITIONS.index(unit["position"]))


def drink(env, unit, spec, doses=1):
    for _ in range(doses):
        before_uses = unit.get(spec["permanent_counter_key"], 0)
        env._add_hero_item(spec["name"])
        count = env._potion_available_count(spec["name"])
        action = potion_action(env, spec, unit)
        assert env.compute_action_mask()[action]
        assert env._can_use_potion_on_position(spec["name"], unit["position"])
        info = env.step(action)[-1]
        assert info["potion_applied"] and info["potion_consumed"]
        assert info["persistent_elixir_applied"] and info["persistent_elixir_consumed"]
        assert env._potion_available_count(spec["name"]) == count - 1
        assert unit[spec["permanent_counter_key"]] == before_uses + 1
        assert unit["health"] == unit.get("hp", unit["health"])
        assert unit["max_health"] == unit.get("maxhp", unit["max_health"])


def force_refresh(env):
    env._mark_equipment_dirty()
    env._refresh_campaign_equipment_effects()


def unit_at(env, position):
    return next(u for u in env.blue_team_state if u["position"] == position)


def save_battle_copy(env):
    """Actual persistence boundary without random combat damage or AI turns."""
    env.battle_env = SimpleNamespace(combined=deepcopy(env.blue_team_state))
    env._save_blue_state()
    force_refresh(env)


def test_covers_exactly_five_permanent_effects():
    assert len(POTIONS) == 5
    assert set(POTION_IDS) == {"damage", "health", "armor", "accuracy", "initiative"}
    assert len({spec["permanent_counter_key"] for spec in POTIONS}) == 5


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("gear", GEAR)
@pytest.mark.parametrize("doses", [1, 3])
@pytest.mark.parametrize("order", ["potion_first", "gear_first"])
def test_public_doses_are_order_independent_and_survive_refresh_and_removal(
    make_env, spec, target, gear, doses, order,
):
    control, reference = make_env(target)
    env, unit = make_env(target)
    initial = stats(reference)
    expected = expected_permanent(initial, spec, doses)
    drink(control, reference, spec, doses)
    assert stats(reference) == expected
    names = gear_names(env, spec, gear)
    equip(control, names)
    geared_expected = stats(reference)
    if order == "gear_first":
        equip(env, names)
        drink(env, unit, spec, doses)
    else:
        drink(env, unit, spec, doses)
        equip(env, names)
    assert stats(unit) == geared_expected
    for _ in range(4):
        force_refresh(env)
        assert stats(unit) == geared_expected
    # An unrelated inventory change caused the original destructive refresh.
    env._add_hero_item(env.TOME_OF_AIR_ITEM_NAME)
    assert stats(unit) == geared_expected
    remove(env, names)
    assert stats(unit) == expected
    assert unit[spec["permanent_counter_key"]] == doses
    assert unit["campaign_permanent_potions"].count(spec["name"]) == doses
    for _ in range(3):
        force_refresh(env)
        assert stats(unit) == expected


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("codec", ["deepcopy", "json", "pickle"])
def test_unit_serialization_and_repeated_battle_save_preserve_permanent_layer(
    make_env, spec, target, codec,
):
    control, reference = make_env(target)
    env, unit = make_env(target)
    drink(control, reference, spec, 2)
    expected = stats(reference)
    names = gear_names(env, spec, "both")
    equip(env, names)
    drink(env, unit, spec, 2)
    position = unit["position"]
    encode = {"deepcopy": deepcopy,
              "json": lambda value: json.loads(json.dumps(value, ensure_ascii=False)),
              "pickle": lambda value: pickle.loads(pickle.dumps(value))}[codec]
    # Persist unit dictionaries, the existing application's durable state shape.
    env.blue_team_state = encode(env.blue_team_state)
    for _ in range(3):
        save_battle_copy(env)
        assert unit_at(env, position)[spec["permanent_counter_key"]] == 2
    remove(env, names)
    assert stats(unit_at(env, position)) == expected


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
def test_dead_targets_and_empty_inventory_cannot_apply_or_increment(make_env, spec):
    env, unit = make_env("ordinary")
    while env._potion_available_count(spec["name"]):
        assert env._consume_potion_item(spec["name"])
    action = potion_action(env, spec, unit)
    before = deepcopy(unit)
    assert not env.compute_action_mask()[action]
    info = env.step(action)[-1]
    assert not info["potion_applied"] and not info["potion_consumed"]
    assert unit == before
    env._add_hero_item(spec["name"])
    unit.update(health=0, hp=0)
    before = deepcopy(unit)
    assert not env.compute_action_mask()[action]
    info = env.step(action)[-1]
    assert not info["potion_applied"] and not info["potion_consumed"]
    assert unit == before
    assert env._potion_available_count(spec["name"]) == 1


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
def test_public_formation_swap_and_revival_keep_bonus_with_unit(make_env, spec):
    env, unit = make_env("ordinary")
    expected = expected_permanent(stats(unit), spec, 2)
    names = gear_names(env, spec, "both")
    equip(env, names)
    drink(env, unit, spec, 2)
    old_position = unit["position"]
    offset, pair = next((i, pair) for i, pair in enumerate(env.GRID_SWAP_UNIT_PAIRS)
                        if old_position in pair and env._can_swap_blue_unit_positions(i))
    action = env.GRID_SWAP_UNIT_ACTION_START + offset
    assert env.compute_action_mask()[action]
    assert env.step(action)[-1]["unit_swap_applied"]
    new_position = next(position for position in pair if position != old_position)
    moved = unit_at(env, new_position)
    assert moved["name"] == unit["name"]
    assert moved[spec["permanent_counter_key"]] == 2
    moved.update(health=0, hp=0)
    env._add_hero_item(env.BONUS_REVIVE_ITEM_NAME)
    revive_spec = env._potion_item_definition(env.BONUS_REVIVE_ITEM_NAME)
    revive_action = potion_action(env, revive_spec, moved)
    assert env.compute_action_mask()[revive_action]
    revived = env.step(revive_action)[-1]
    assert revived["potion_applied"] and revived["potion_consumed"]
    assert moved["health"] > 0 and moved["health"] == moved["hp"]
    remove(env, names)
    assert stats(moved, health=False) == {k: v for k, v in expected.items() if k != "health"}
    assert moved[spec["permanent_counter_key"]] == 2


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
@pytest.mark.parametrize("doses", [1, 3])
def test_promotion_rebuilds_all_permanent_doses_on_new_unit(make_env, spec, doses):
    env, unit = make_env("ordinary")
    expected = env._build_unit_from_data(env._find_unit_data_by_name("Берсерк"), "blue", unit["position"])
    expected_stats = expected_permanent(stats(expected), spec, doses)
    names = gear_names(env, spec, "both")
    equip(env, names)
    drink(env, unit, spec, doses)
    for building in env._get_buildings_for_capital(2).values():
        if isinstance(building, dict) and building.get("unit"):
            building["Build"] = int(building["unit"] == "Берсерк")
            building["built"] = building["Build"]
    env.battle_env = SimpleNamespace(combined=deepcopy(env.blue_team_state),
                                    last_levelups=[unit["name"]])
    source = next(u for u in env.battle_env.combined if u["position"] == unit["position"])
    env.battle_env.last_levelup_units = [source]
    assert env._log_turns_into_levelups() == 1
    assert source["name"] == "Берсерк"
    env._save_blue_state()
    force_refresh(env)
    remove(env, names)
    saved = unit_at(env, unit["position"])
    assert stats(saved) == expected_stats
    assert saved[spec["permanent_counter_key"]] == doses


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
@pytest.mark.parametrize("target", ["hero", "healer", "secondary"])
@pytest.mark.parametrize("start_level", [1, 9, 12, 13])
def test_levelup_with_equipment_matches_gearless_potion_control(
    make_env, spec, target, start_level,
):
    control, reference = make_env(target)
    env, unit = make_env(target)
    for candidate in (reference, unit):
        candidate["Level"] = start_level
        candidate["turns_into"] = []
    # Intrinsic growth precedes the persistent potion layer. The independent
    # control levels naked, then drinks; the subject drinks while geared first.
    battle = BattleEnv(log_enabled=False)
    battle._apply_exp_award_to_unit(reference, reference["exp_required"])
    assert reference["Level"] == start_level + 1
    drink(control, reference, spec, 2)
    save_battle_copy(control)
    names = gear_names(env, spec, "both")
    equip(env, names)
    drink(env, unit, spec, 2)
    battle._apply_exp_award_to_unit(unit, unit["exp_required"])
    assert unit["Level"] == start_level + 1
    save_battle_copy(env)
    remove(env, names)
    actual = unit_at(env, unit["position"])
    expected = unit_at(control, reference["position"])
    assert stats(actual) == stats(expected)
    assert actual[spec["permanent_counter_key"]] == 2


def test_native_scenario_initiative_elixir_survives_unrelated_book_refresh(make_env):
    env, hero = make_env(cls=CampaignEnv, map_name="green_dragon_minimal")
    spec = next(p for p in POTIONS if p["effect"]["kind"] == "initiative")
    expected = expected_permanent(stats(hero), spec, 1)
    equip(env, [env.RING_OF_AGES_ITEM_NAME])
    drink(env, hero, spec)
    env._add_hero_item(env.TOME_OF_AIR_ITEM_NAME)
    remove(env, [env.RING_OF_AGES_ITEM_NAME])
    assert stats(hero) == expected


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
@pytest.mark.parametrize("target", TARGETS)
def test_real_battle_prepare_and_save_keep_same_permanent_stats(make_env, spec, target):
    control, reference = make_env(target, map_name="scroll_train")
    env, unit = make_env(target, map_name="scroll_train")
    drink(control, reference, spec, 2)
    bare_expected = stats(reference, health=False)
    names = gear_names(env, spec, "both")
    equip(control, names)
    equip(env, names)
    drink(env, unit, spec, 2)
    position = unit["position"]
    for _ in range(2):
        for campaign in (control, env):
            campaign._init_battle(1)
        actual = next(u for u in env.battle_env.combined
                      if u["team"] == "blue" and u["position"] == position)
        expected = next(u for u in control.battle_env.combined
                        if u["team"] == "blue" and u["position"] == position)
        # Initial automatic RED turns can wound fighters; intrinsic maxima and
        # attack stats must still be identical under real battle preparation.
        assert stats(actual, health=False) == stats(expected, health=False)
        for campaign in (control, env):
            campaign._save_blue_state()
            force_refresh(campaign)
    remove(env, names)
    saved = unit_at(env, position)
    assert stats(saved, health=False) == bare_expected
    assert saved[spec["permanent_counter_key"]] == 2


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("reverse", [False, True])
def test_all_five_permanent_potions_coexist_without_replacing_each_other(
    make_env, target, reverse,
):
    control, reference = make_env(target)
    env, unit = make_env(target)
    sequence = POTIONS[::-1] if reverse else POTIONS
    expected = stats(reference)
    for spec in sequence:
        expected = expected_permanent(expected, spec, 2)
        drink(control, reference, spec, 2)
    assert stats(reference) == expected
    damage_spec = next(spec for spec in POTIONS if spec["effect"]["kind"] == "damage")
    names = gear_names(env, damage_spec, "both")
    equip(control, names)
    equip(env, names)
    for spec in sequence:
        drink(env, unit, spec, 2)
    assert stats(unit) == stats(reference)
    for _ in range(3):
        force_refresh(env)
        assert stats(unit) == stats(reference)
    remove(env, names)
    assert stats(unit) == expected
    for spec in POTIONS:
        assert unit[spec["permanent_counter_key"]] == 2
        assert unit["campaign_permanent_potions"].count(spec["name"]) == 2


@pytest.mark.parametrize("order", ["potion_first", "gear_first"])
def test_healer_titan_potion_changes_actual_healing_after_equipment_removal(make_env, order):
    env, healer = make_env("healer")
    spec = next(spec for spec in POTIONS if spec["effect"]["kind"] == "damage")
    expected = expected_permanent(stats(healer), spec, 3)["damage"]
    names = gear_names(env, spec, "both")
    if order == "gear_first":
        equip(env, names)
        drink(env, healer, spec, 3)
    else:
        drink(env, healer, spec, 3)
        equip(env, names)
    env._add_hero_item(env.TOME_OF_AIR_ITEM_NAME)
    remove(env, names)
    battle = BattleEnv(log_enabled=False)
    healer = deepcopy(healer)
    healer["position"] = 10
    recipient = map_unit_to_battle(next(d for d in DATA if d["кто"] == "Скваер"), "blue", 7)
    recipient.update(health=100, max_health=1000)
    battle.combined = [healer, recipient]
    assert battle._attack(healer, 1)[0]
    assert recipient["health"] == 100 + expected


@pytest.mark.parametrize("spec", POTIONS, ids=POTION_IDS)
@pytest.mark.parametrize("case", ["zero", "rounding_ties", "accuracy_caps"])
def test_zero_stats_integer_ties_and_caps_have_exact_per_dose_results(make_env, spec, case):
    control, reference = make_env()
    env, unit = make_env()
    values = {
        "zero": dict(damage=0, damage_secondary=0, accuracy=0, accuracy_secondary=0,
                     initiative_base=0, armor=0, max_health=1, health=1),
        "rounding_ties": dict(damage=15, damage_secondary=25, accuracy=65,
                              accuracy_secondary=25, initiative_base=15, armor=15,
                              max_health=15, health=15),
        "accuracy_caps": dict(damage=145, damage_secondary=0, accuracy=99,
                              accuracy_secondary=100, initiative_base=95, armor=99,
                              max_health=1000, health=1000),
    }[case]
    # Both fresh, never-dosed units receive the same synthetic source values.
    # These values make zero-secondary behavior, .5 banker's rounding and the
    # accuracy100 cap observable instead of relying only on ordinary rosters.
    for candidate in (reference, unit):
        candidate.update(values)
        candidate.update(initiative=values["initiative_base"], hp=values["health"],
                         maxhp=values["max_health"], base_armor=values["armor"],
                         original_damage=values["damage"])
    expected = expected_permanent(values, spec, 3)
    drink(control, reference, spec, 3)
    assert stats(reference) == expected
    names = gear_names(env, spec, "both")
    equip(control, names)
    equip(env, names)
    drink(env, unit, spec, 3)
    assert stats(unit) == stats(reference)
    for _ in range(3):
        force_refresh(env)
        assert stats(unit) == stats(reference)
    remove(env, names)
    assert stats(unit) == expected
