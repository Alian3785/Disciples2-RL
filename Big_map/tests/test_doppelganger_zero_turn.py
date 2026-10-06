"""A preparation round must not become an unlimited source of actions or buffs."""
from copy import deepcopy

import pytest

from battle_env import (
    BattleEnv, DEFEND_ACTION_INDEX, WAIT_ACTION_INDEX, RUN_AWAY_ACTION_INDEX,
    FIRST_HERO_ITEM_ACTION_START, TARGET_POSITIONS,
)
from permanent_unit_stats import add_permanent_effect, ensure_stat_sources, persistent_values
from data_dicts_compact_lines import placeholder_unit
from test_doppelganger_copy_survival import initialize, pair, unit


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("effects", ["buff", "debuff", "both", "slow", "shatter"])
@pytest.mark.parametrize("permanent", [False, True])
def test_copies_permanent_stats_without_temporary_combat_layers(team, effects, permanent):
    battle = BattleEnv(log_enabled=False)
    copier, angel = pair(team, "Ангел")
    angel.update(original_damage=125, armor=30, base_armor=30)
    if permanent:
        add_permanent_effect(angel, {"kind": "damage", "multiplier": 1.2})
    baseline = dict(angel)
    ensure_stat_sources(baseline)
    expected = persistent_values(baseline)
    druid = unit("Архидруид", angel["team"], 5)
    if effects in ("buff", "both"):
        assert battle._apply_travnitsa_buff(druid, angel)
    if effects in ("debuff", "both"):
        assert battle._apply_hero_item_damage_debuff(
            source_unit=None, target_unit=angel, effect={"damage_multiplier": 0.5}
        )
    if effects == "slow":
        battle._apply_hermit_initiative_slow({"attack_type_secondary": ""}, angel)
    if effects == "shatter":
        battle._apply_teurg_armor_shred({"attack_type_secondary": ""}, angel)
    before = deepcopy(angel)
    copier["initiative"] = 0
    assert battle._apply_doppelganger_copy(copier, angel)
    assert angel == before
    for stat, value in expected.items():
        assert copier["initiative_base" if stat == "initiative" else stat] == value
    assert copier["initiative"] == 0
    assert copier["damage"] == copier["original_damage"] == expected["damage"]
    assert not copier["powerup"]
    battle._reset_powerup(angel)
    assert copier["damage"] == expected["damage"]
    assert battle._apply_travnitsa_buff(druid, copier)
    assert copier["damage"] == 2 * expected["damage"]
    battle._reset_powerup(copier)
    assert copier["damage"] == expected["damage"]


def test_copy_of_buffed_copy_uses_form_not_original_doppelganger_stats():
    battle = BattleEnv(log_enabled=False)
    first, angel = pair("blue", "Ангел")
    second = unit("Двойник", "blue", 8)
    assert battle._apply_doppelganger_copy(first, angel)
    assert battle._apply_travnitsa_buff(unit("Архидруид", "blue", 9), first)
    assert first["damage"] == 250
    assert battle._apply_doppelganger_copy(second, first)
    assert second["damage"] == second["original_damage"] == 125


def test_copy_of_transformed_target_uses_current_form_stats():
    battle = BattleEnv(log_enabled=False)
    copier, target = pair("blue", "Ангел")
    ensure_stat_sources(target)
    assert battle._apply_hero_item_lycanthropy(
        source_unit=None, target_unit=target,
        effect={"damage_type": "", "transform_unit_name": "Оборотень"},
    )
    expected = target["damage"]
    assert battle._apply_travnitsa_buff(unit("Архидруид", "red", 2), target)
    assert battle._apply_doppelganger_copy(copier, target)
    assert copier["damage"] == copier["original_damage"] == expected
    assert copier["unit_type"] == target["unit_type"]


@pytest.mark.parametrize("team", ["blue", "red"])
def test_zero_round_precedes_faster_units_and_does_not_deal_damage(team):
    copier, target = pair(team, "Скваер")
    target["initiative_base"] = 150
    if team == "red":
        # A BLUE copier holds round zero open after RED prepares its form.
        observer = unit("Двойник", "blue", 8)
        observer["initiative_base"] = 1
        battle = initialize(copier, target, observer)
        assert battle._unit_by_position(1)["doppel_copied"]
        assert battle.current_blue_attacker_pos == 8
        assert battle._unit_by_position(7)["health"] == target["health"]
    else:
        battle = initialize(copier, target)
        assert battle.current_blue_attacker_pos == 7
        assert battle._unit_by_position(7)["health"] == copier["health"]
    assert battle.round_no == 0
    mask = battle.compute_action_mask()
    assert mask[DEFEND_ACTION_INDEX]
    assert not mask[WAIT_ACTION_INDEX]
    assert not mask[RUN_AWAY_ACTION_INDEX]
    assert not mask[FIRST_HERO_ITEM_ACTION_START:].any()


def test_zero_copy_preserves_normal_turn_and_runs_only_once():
    copier, target = pair("blue", "Скваер")
    ally = unit("Лучник", "blue", 8)
    ally["initiative_base"] = 100
    battle = initialize(copier, target, ally)
    _, _, done, truncated, info = battle.step(TARGET_POSITIONS.index(8))
    assert not done and not truncated and info["doppelganger_copied"]
    assert battle.round_no == 1
    assert battle.current_blue_attacker_pos == 7
    assert battle._unit_by_position(1)["health"] == target["health"]
    _, _, _, _, info = battle.step(TARGET_POSITIONS.index(1))
    assert not info.get("doppelganger_zero_turn")
    assert battle.current_blue_attacker_pos == 8
    assert battle._unit_by_position(7)["initiative"] == 0
    assert battle._unit_by_position(1)["health"] < target["health"]


def test_copy_of_double_striker_gets_exactly_two_normal_attacks():
    copier, target = pair("blue", "Скваер")
    target["health"] = target["max_health"] = 500
    ally = unit("Бандит", "blue", 8)
    ally["initiative_base"] = 100
    battle = initialize(copier, target, ally)
    battle.step(TARGET_POSITIONS.index(8))
    assert battle.current_blue_attacker_pos == 7 and battle.blue_attacks_left == 2
    battle.step(TARGET_POSITIONS.index(1))
    assert battle.current_blue_attacker_pos == 7 and battle.blue_attacks_left == 1
    battle.step(TARGET_POSITIONS.index(1))
    assert battle.current_blue_attacker_pos == 8
    assert battle._unit_by_position(1)["health"] == 440


def test_uncopied_doppelganger_is_masked_even_when_forced_as_target():
    battle = initialize(unit("Скваер", "red", 1),
                        unit("Двойник", "blue", 7), unit("Двойник", "blue", 8))
    assert not battle.compute_action_mask()[TARGET_POSITIONS.index(8)]
    _, _, _, _, info = battle.step(TARGET_POSITIONS.index(8))
    assert info["doppelganger_invalid_action"]
    assert battle.round_no == 0 and battle.current_blue_attacker_pos == 8
    assert not battle._unit_by_position(7)["doppel_copied"]


@pytest.mark.parametrize("action", [DEFEND_ACTION_INDEX, WAIT_ACTION_INDEX,
                                    RUN_AWAY_ACTION_INDEX, FIRST_HERO_ITEM_ACTION_START,
                                    -1, 999, TARGET_POSITIONS.index(7)])
def test_skip_or_forced_invalid_action_cannot_repeat_preparation(action):
    copier, target = pair("blue", "Скваер")
    battle = initialize(copier, target)
    _, _, _, _, info = battle.step(action)
    copier = battle._unit_by_position(7)
    assert battle.round_no == 1
    assert battle.current_blue_attacker_pos == 7
    assert copier["unit_type"] == "Doppelganger"
    assert not copier["defense"] and not copier.get("running_away")
    assert info["doppelganger_invalid_action"] == (action != DEFEND_ACTION_INDEX)
    # A normal-round transformation spends the action even when the target has
    # not acted yet. A slower ally lets us inspect before the next round.
    ally = unit("Лучник", "blue", 8)
    ally["initiative"] = ally["initiative_base"] = 70
    battle.combined = [u for u in battle.combined if u["position"] != 8] + [ally]
    battle.step(TARGET_POSITIONS.index(1))
    assert battle.round_no == 1
    assert battle.current_blue_attacker_pos == 8
    assert copier["initiative"] == 0


def test_uncopied_doppelgangers_are_not_targets_and_empty_zero_round_is_skipped():
    blue, red = unit("Двойник", "blue", 7), unit("Двойник", "red", 1)
    target = unit("Лучник", "blue", 8)
    battle = initialize(blue, red, target)
    assert battle.round_no == 0
    # RED already became an archer and is now a valid form.
    assert battle.compute_action_mask()[TARGET_POSITIONS.index(1)]
    battle = initialize(blue, red)
    assert battle.round_no == 1
    assert not battle._has_transform_targets()
    assert battle._unit_by_position(7)["unit_type"] == "Warrior"


def test_normal_battle_has_no_zero_round_and_custom_reset_restores_it():
    battle = initialize(unit("Лучник", "blue", 7), unit("Скваер", "red", 1))
    assert battle.round_no == 1
    for _ in range(2):
        battle._init_with_custom_teams(
            [unit("Скваер", "red", 1)] + [placeholder_unit("red", p) for p in range(2, 7)],
            [unit("Двойник", "blue", 7)] + [placeholder_unit("blue", p) for p in range(8, 13)],
        )
        assert battle.round_no == 0
        battle.step(DEFEND_ACTION_INDEX)
        assert battle.round_no == 1


def test_standard_reset_also_starts_zero_round(monkeypatch):
    import battle_env
    monkeypatch.setattr(battle_env, "UNITS_RED", [unit("Скваер", "red", 1)] + [
        placeholder_unit("red", p) for p in range(2, 7)
    ])
    monkeypatch.setattr(battle_env, "UNITS_BLUE", [unit("Двойник", "blue", 7)] + [
        placeholder_unit("blue", p) for p in range(8, 13)
    ])
    battle = BattleEnv(log_enabled=False)
    for _ in range(2):
        battle.reset(seed=42)
        assert battle.round_no == 0 and battle.current_blue_attacker_pos == 7
        battle.step(DEFEND_ACTION_INDEX)
        assert battle.round_no == 1
