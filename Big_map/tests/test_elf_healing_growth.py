"""Elven support growth must reach the field used by real healing actions."""

from copy import deepcopy

import pytest

from battle_env import BattleEnv, _apply_dynamic_unit_levelup
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from unit_dynamic_stats import DYNAMIC_STAT_PROFILES, apply_dynamic_stat_growth


UNITS = {unit["кто"]: unit for unit in DATA}
# Name, base healing, early growth (through level 10), late growth.
HEALERS = [
    ("Дева рощи", 50, 5, 3),
    ("Солнечная танцовщица", 50, 6, 1),
    ("Сильфида", 55, 7, 3),
]
HEALER_TYPES = {"Deva roshi", "Sundancer", "Sylfid"}


def make_healer(name, team="blue"):
    return map_unit_to_battle(UNITS[name], team, 10 if team == "blue" else 4)


def assert_healing_action(battle, healer, amount):
    team = healer["team"]
    recipient = map_unit_to_battle(
        UNITS["Скваер"], team, 7 if team == "blue" else 1,
    )
    recipient.update(health=100, max_health=1000)
    battle.combined = [healer, recipient]
    # Target selection for point healers mirrors an opposing formation cell;
    # mass healers use the same action entry point and heal all living allies.
    target_position = 1 if team == "blue" else 7
    assert battle._attack(healer, target_position)[0]
    assert recipient["health"] == 100 + amount


@pytest.mark.parametrize("name,base,early,late", HEALERS)
@pytest.mark.parametrize("team", ["red", "blue"])
def test_xp_levelup_increases_real_healing(name, base, early, late, team):
    battle = BattleEnv(log_enabled=False)
    healer = make_healer(name, team)
    initial_level = healer["Level"]
    assert_healing_action(battle, healer, base)
    # Exercise dynamic growth rather than choosing an evolution for Deva roshi.
    healer["turns_into"] = []
    battle._apply_exp_award_to_unit(healer, healer["exp_required"])
    assert healer["Level"] == initial_level + 1
    assert_healing_action(battle, healer, base + early)
    assert healer["damage_secondary"] == 0
    assert healer["original_damage"] == base + early


@pytest.mark.parametrize("name,base,early,late", HEALERS)
@pytest.mark.parametrize("next_level", [10, 11])
def test_growth_switches_at_original_threshold(name, base, early, late, next_level):
    healer = make_healer(name)
    healer["Level"] = next_level - 1
    before = deepcopy(healer)
    increment = early if next_level == 10 else late
    _apply_dynamic_unit_levelup(healer, healer["exp_required"])
    assert healer["Level"] == next_level
    assert healer["damage"] == base + increment
    assert healer["damage_secondary"] == before["damage_secondary"]
    assert_healing_action(BattleEnv(log_enabled=False), healer, base + increment)


@pytest.mark.parametrize("name,base,early,late", HEALERS)
def test_repeated_levels_accumulate_once_across_threshold(name, base, early, late):
    battle = BattleEnv(log_enabled=False)
    healer = make_healer(name)
    expected = base
    for next_level in range(healer["Level"] + 1, 14):
        _apply_dynamic_unit_levelup(healer, healer["exp_required"])
        expected += early if next_level <= 10 else late
        assert healer["damage"] == expected
        assert healer["original_damage"] == expected
        assert healer["damage_secondary"] == 0
        assert_healing_action(battle, healer, expected)


@pytest.mark.parametrize("name,base,early,late", HEALERS)
@pytest.mark.parametrize("next_level", [10, 11])
def test_healing_snapshots_grow_and_secondary_snapshots_do_not(
    name, base, early, late, next_level,
):
    healer = make_healer(name)
    healer["original_damage"] = base
    healer["lower_damage_original_damage"] = base
    for prefix in ("potion", "map_spell", "banner", "artifact"):
        healer[f"campaign_{prefix}_base_damage"] = base
        healer[f"campaign_{prefix}_base_damage_secondary"] = 17
    healer["damage_secondary"] = 17
    before = deepcopy(healer)
    assert apply_dynamic_stat_growth(healer, next_level)
    increment = early if next_level == 10 else late
    for key in before:
        if key == "damage" or key.endswith("_damage"):
            assert healer[key] == before[key] + increment
        elif key == "damage_secondary" or key.endswith("_damage_secondary"):
            assert healer[key] == before[key]
    threshold, first, last, _, _, secondary_power = DYNAMIC_STAT_PROFILES[healer["unit_id"]]
    hp, _, _, armor, power, initiative = first if next_level <= threshold else last
    for key, increment in (
        ("max_health", hp), ("health", hp), ("armor", armor),
        ("accuracy", power), ("accuracy_secondary", power if secondary_power else 0),
        ("initiative", initiative),
    ):
        expected = before[key] + increment
        if key.startswith("accuracy"):
            expected = min(100, expected)
        assert healer[key] == expected


@pytest.mark.parametrize("name,base,early,late", HEALERS)
@pytest.mark.parametrize("with_map_buff", [False, True])
def test_campaign_save_and_battle_rebuild_preserve_growth(
    name, base, early, late, with_map_buff,
):
    env = CampaignEnv(
        map_name="scroll_train", observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    try:
        env.reset(seed=42)
        team = env._get_blue_state()
        healer = make_healer(name)
        healer["position"] = 8
        healer["stand"] = "ahead"
        env.blue_team_state = [healer if u["position"] == 8 else u for u in team]
        # Keep the initial automatic RED turns out of this persistence check.
        for unit in env.blue_team_state:
            unit.update(initiative=100, initiative_base=100)
        multiplier = 1.0
        if with_map_buff:
            spell_id, spec = next(
                (key, spec) for key, spec in env._map_support_spell_specs_by_id().items()
                if spec.get("kind") == "buff" and spec.get("damage_multiplier", 1) > 1
            )
            multiplier = spec["damage_multiplier"]
            env._apply_support_spell_to_blue_stack(spell_id)
        env._init_battle(1)
        grown = next(u for u in env.battle_env.combined if u["position"] == 8)
        initial_level = grown["Level"]
        _apply_dynamic_unit_levelup(grown, grown["exp_required"])
        assert grown["damage"] == round(base * multiplier) + early
        assert grown["damage_secondary"] == 0
        env._save_blue_state()
        saved = next(u for u in env.blue_team_state if u["position"] == 8)
        assert saved["Level"] == initial_level + 1
        assert saved["damage"] == base + early
        assert saved["damage_secondary"] == 0
        # Repeated rebuild/save must neither lose growth nor add it again.
        for _ in range(2):
            env._init_battle(1)
            rebuilt = next(u for u in env.battle_env.combined if u["position"] == 8)
            assert rebuilt["damage"] == round((base + early) * multiplier)
            assert rebuilt["damage_secondary"] == 0
            env._save_blue_state()
        env._clear_all_blue_map_spell_effects()
        env._init_battle(1)
        rebuilt = next(u for u in env.battle_env.combined if u["position"] == 8)
        assert rebuilt["damage"] == base + early
        assert_healing_action(env.battle_env, rebuilt, base + early)
    finally:
        env.close()


@pytest.mark.parametrize("next_level", [10, 11])
def test_all_other_roster_units_keep_original_growth_routing(next_level):
    for source in DATA:
        unit = map_unit_to_battle(source, "blue", 7)
        if unit["unit_type"] in HEALER_TYPES:
            continue
        profile = DYNAMIC_STAT_PROFILES.get(unit["unit_id"])
        if profile is None:
            continue
        before = deepcopy(unit)
        threshold, early, late, primary, secondary, secondary_power = profile
        hp, damage, heal, armor, power, initiative = early if next_level <= threshold else late
        assert apply_dynamic_stat_growth(unit, next_level)
        expected = {
            "damage": damage if primary == 1 else heal if primary == 2 else 0,
            "damage_secondary": damage if secondary == 1 else heal if secondary == 2 else 0,
            "armor": armor,
            "accuracy": power,
            "accuracy_secondary": power if secondary_power else 0,
            "initiative": initiative,
            "max_health": hp,
            "health": hp,
        }
        for key, increment in expected.items():
            value = before[key] + increment
            if key.startswith("accuracy"):
                value = min(100, value)
            assert unit[key] == value, (unit["name"], key, next_level)


def test_unknown_profile_is_unchanged():
    unit = make_healer("Дева рощи")
    unit["unit_id"] = "unknown"
    before = deepcopy(unit)
    assert not apply_dynamic_stat_growth(unit, 4)
    assert unit == before
