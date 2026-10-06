"""Scenario levels must initialize real stats, XP thresholds and persistent bases."""
from copy import deepcopy
from dataclasses import replace

import pytest

from battle_env import BattleEnv
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from permanent_unit_stats import rebuild_stat_layers


@pytest.fixture
def siege():
    env = CampaignEnv(map_name="siege_train", observation_version="local5",
                      scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
                      log_enabled=False)
    env.reset(seed=42)
    yield env
    env.close()


def unit_named(roster, name):
    return next(u for u in roster if u["name"] == name)


@pytest.mark.parametrize("name,level,hp,damage,accuracy,required", [
    ("Рыцарь на пегасе", 3, 180, 70, 82, 1150),
    ("Рейнджер людей", 2, 100, 45, 81, 595),
])
def test_siege_override_matches_real_level_ups_and_survives_battle_save(
    siege, name, level, hp, damage, accuracy, required,
):
    eid = siege._map.scheduled_enemy_id
    original = deepcopy(unit_named(siege._enemy_configs[eid], name))
    control = deepcopy(original)
    leveling = BattleEnv(log_enabled=False)
    while control["Level"] < level:
        leveling._apply_exp_award_to_unit(control, int(control["exp_required"]))
    actual = unit_named(siege.enemy_team_states[eid], name)
    assert (actual["Level"], actual["health"], actual["max_health"], actual["damage"],
            actual["accuracy"], actual["exp_required"]) == (level, hp, hp, damage, accuracy, required)
    assert actual["original_damage"] == damage
    for key in ("exp_kill", "exp_current", "campaign_stat_sources", "next_level_exp"):
        assert actual[key] == control[key], key
    for key in ("name", "unit_id", "position", "stand", "big"):
        assert actual[key] == original[key], key
    # Initialization must not mutate the cached map templates.
    assert unit_named(siege._enemy_configs[eid], name) == original
    assert original["Level"] == 1

    # Enter/save a real prepared battle without allowing autonomous attacks.
    siege._init_battle(eid, blue_team=[])
    runtime = unit_named(siege.battle_env.combined, name)
    assert (runtime["max_health"], runtime["damage"]) == (hp, damage)
    siege._save_enemy_state_from_battle(eid)
    saved = unit_named(siege.enemy_team_states[eid], name)
    rebuild_stat_layers(saved)
    assert (saved["Level"], saved["max_health"], saved["damage"], saved["exp_required"]) == (
        level, hp, damage, required,
    )
    for _ in range(2):
        siege.reset(seed=42)
        reset = unit_named(siege.enemy_team_states[eid], name)
        assert (reset["Level"], reset["health"], reset["damage"], reset["exp_required"]) == (
            level, hp, damage, required,
        )


def test_other_units_and_army_composition_are_preserved(siege):
    overrides = siege._map.enemy_unit_level_overrides
    for eid, source in siege._enemy_configs.items():
        actual = siege.enemy_team_states[eid]
        assert [(u["name"], u["position"], u["big"]) for u in actual] == [
            (u["name"], u["position"], u["big"]) for u in source
        ]
        for before, after in zip(source, actual):
            if before["name"] not in overrides.get(eid, {}):
                assert after == before


@pytest.mark.parametrize("name", ["Скваер", "Ангел", "Теург"])
def test_non_hero_overrides_raise_stats_without_changing_form(siege, name):
    source = map_unit_to_battle(next(row for row in DATA if row["кто"] == name), "red", 1)
    initial = deepcopy(source)
    target_level = source["Level"] + 2
    siege._enemy_configs = {1: [source]}
    siege._map = replace(siege._map, enemy_unit_level_overrides={1: {name: target_level}})
    grown = siege._create_initial_enemy_team_states()[1][0]
    assert grown["Level"] == target_level
    assert grown["max_health"] > initial["max_health"]
    assert grown["damage"] > initial["damage"]
    assert grown["name"] == initial["name"]
    assert grown["unit_id"] == initial["unit_id"]
    assert grown["damage"] == grown["original_damage"]
    assert source == initial
    before = deepcopy(grown)
    rebuild_stat_layers(grown)
    for key in ("max_health", "damage", "accuracy", "armor", "exp_kill", "Level"):
        assert grown[key] == before[key]


def test_same_level_is_a_noop_and_lower_level_is_rejected(siege):
    name = "Имперский рыцарь"
    source = unit_named(siege._enemy_configs[siege._map.scheduled_enemy_id], name)
    siege._enemy_configs = {1: [source]}
    siege._map = replace(siege._map, enemy_unit_level_overrides={1: {name: source["Level"]}})
    assert siege._create_initial_enemy_team_states()[1][0] == source
    siege._map = replace(siege._map, enemy_unit_level_overrides={1: {name: source["Level"] - 1}})
    with pytest.raises(ValueError, match="below its template level"):
        siege._create_initial_enemy_team_states()
