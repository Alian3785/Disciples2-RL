"""Copying a live form cannot kill a Doppelganger through HP rounding."""

from copy import deepcopy
import random

import pytest

from battle_env import BattleEnv, TARGET_POSITIONS
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


CATALOG = {entry["кто"]: entry for entry in DATA}


class FixedRandom(random.Random):
    def random(self):
        return 0.0

    def randint(self, low, high):
        return low

    def shuffle(self, values):
        pass

    def choice(self, values):
        return values[0]


def unit(name, team, position):
    return map_unit_to_battle(CATALOG[name], team, position)


def pair(team, name="Лучник", allied=False):
    position = 7 if team == "blue" else 1
    target_team = team if allied else ("red" if team == "blue" else "blue")
    target_position = position + 1 if allied else (1 if team == "blue" else 7)
    return unit("Двойник", team, position), unit(name, target_team, target_position)


def initialize(*units):
    battle = BattleEnv(log_enabled=True)
    battle.rng = FixedRandom(0)
    occupied = {fighter["position"] for fighter in units}
    roster = list(units) + [
        placeholder_unit("red" if pos < 7 else "blue", pos)
        for pos in range(1, 13) if pos not in occupied
    ]
    battle._init_with_custom_teams(
        [fighter for fighter in roster if fighter["team"] == "red"],
        [fighter for fighter in roster if fighter["team"] == "blue"],
    )
    return battle


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("allied", [False, True])
@pytest.mark.parametrize("copier_hp", [1, 60, 120, 240])
@pytest.mark.parametrize("target_name,target_hp", [
    ("Лучник", 45), ("Лучник", 1),
    ("Скваер", 100), ("Скваер", 50), ("Скваер", 1),
])
def test_copy_retains_wounds_and_ratio_cap_without_rounding_to_death(
    team, allied, copier_hp, target_name, target_hp,
):
    battle = BattleEnv(log_enabled=False)
    copier, target = pair(team, target_name, allied)
    copier["health"] = copier_hp
    target["health"] = target_hp
    original = deepcopy(copier)
    target_before = deepcopy(target)
    battle.combined = [copier, target]

    assert battle._apply_doppelganger_copy(copier, target)

    expected = max(1, round(target_hp * min(1.0, copier_hp / 120)))
    assert copier["health"] == expected
    assert 1 <= copier["health"] <= target_hp <= copier["max_health"]
    assert battle._alive(copier)
    assert copier["max_health"] == target["max_health"]
    assert copier["unit_type"] == target["unit_type"]
    assert copier["damage"] == target["damage"]
    assert copier["doppel_copied"] == 1
    for key in ("name", "unit_id", "team", "position", "exp_kill"):
        assert copier[key] == original[key]
    assert target == target_before
    assert battle._battle_exp_event_count == 0
    assert battle.recovery_killed_positions == set()


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("target_hp", [None, 0, -1])
def test_missing_or_dead_copy_target_is_rejected_without_mutation(team, target_hp):
    battle = BattleEnv(log_enabled=False)
    copier, target = pair(team)
    copier["health"] = 1
    if target_hp is None:
        target = None
    else:
        target["health"] = target_hp
    before = deepcopy(copier)

    assert not battle._apply_doppelganger_copy(copier, target)
    assert copier == before


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("copier_hp", [0, -1])
def test_floor_does_not_revive_dead_copier(team, copier_hp):
    battle = BattleEnv(log_enabled=False)
    copier, target = pair(team)
    copier["health"] = copier_hp
    battle._apply_doppelganger_copy(copier, target)
    assert copier["health"] == 0
    assert not battle._alive(copier)
    battle.combined = [copier, target]
    battle._restore_default_doppelgangers()
    assert copier["health"] == 0


@pytest.mark.parametrize("team", ["blue", "red"])
def test_copied_health_is_capped_by_form_maximum(team):
    battle = BattleEnv(log_enabled=False)
    copier, target = pair(team, "Скваер")
    # Defensive bound even if the incoming target has stale over-max HP.
    target["health"] = 200
    assert battle._apply_doppelganger_copy(copier, target)
    assert copier["health"] == copier["max_health"] == 100


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("target_hp,expected", [
    (100, [50, 50, 50, 50]),
    (50, [25, 12, 6, 3]),
    (1, [1, 1, 1, 1]),
])
def test_repeated_copies_keep_existing_wound_semantics(team, target_hp, expected):
    battle = BattleEnv(log_enabled=False)
    copier, target = pair(team, "Скваер")
    copier["health"] = 60
    target["health"] = target_hp
    observed = []
    for _ in expected:
        assert battle._apply_doppelganger_copy(copier, target)
        observed.append(copier["health"])
    assert observed == expected
    assert target["health"] == target_hp


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("copier_hp,target_name,target_hp", [
    (1, "Лучник", 45), (60, "Скваер", 1), (120, "Владыка рун", 1),
])
def test_repeated_copy_and_default_form_restore_stays_alive(
    team, copier_hp, target_name, target_hp,
):
    battle = BattleEnv(log_enabled=False)
    copier, target = pair(team, target_name)
    copier["health"] = copier_hp
    target["health"] = target_hp
    battle.combined = [copier, target]
    copied_hp, restored_hp = [], []
    for _ in range(4):
        assert battle._apply_doppelganger_copy(copier, target)
        copied_hp.append(copier["health"])
        assert battle._alive(copier)
        battle._restore_default_doppelgangers()
        restored_hp.append(copier["health"])
        assert battle._alive(copier)
        assert copier["max_health"] == 120
        assert copier["unit_type"] == "Doppelganger"
        assert copier["doppel_copied"] == 0
    assert copied_hp == [1] * 4
    # A healthy 45-HP form rounds 1/45 back to 3/120, then stays stable.
    assert restored_hp == ([3] * 4 if target_name == "Лучник" else [1] * 4)


@pytest.mark.parametrize("copier_hp,target_name,target_hp", [
    (1, "Лучник", 45), (60, "Скваер", 1),
])
def test_blue_public_copy_does_not_create_false_defeat(
    monkeypatch, copier_hp, target_name, target_hp,
):
    copier, target = pair("blue", target_name)
    copier["health"] = copier_hp
    target["health"] = target_hp
    battle = initialize(copier, target)
    copier = battle._unit_by_position(7)
    assert battle.current_blue_attacker_pos == 7
    action = TARGET_POSITIONS.index(1)
    assert battle.compute_action_mask()[action]
    # Inspect immediately after the real action/victory check, before a legal
    # enemy counterattack could kill this intentionally wounded copier.
    monkeypatch.setattr(battle, "_advance_until_blue_turn", lambda: None)

    _, _, terminated, truncated, _ = battle.step(action)

    assert copier["health"] == 1
    assert battle._alive(copier)
    assert battle.winner is None
    assert not terminated and not truncated
    assert battle._battle_exp_event_count == 0
    assert battle.recovery_killed_positions == set()
    assert any("Doppelganger выбор" in event for event in battle.pop_pretty_events())


@pytest.mark.parametrize("copier_hp,target_name,target_hp", [
    (1, "Лучник", 45), (60, "Скваер", 1),
])
def test_red_automatic_copy_stays_alive_before_blue_turn(
    copier_hp, target_name, target_hp,
):
    copier, target = pair("red", target_name)
    copier["health"] = copier_hp
    target["health"] = target_hp
    # Native initiative 70 makes this unit act immediately after the 80-INI
    # copier, ahead of the copied form's 50/60 initiative. At 1 HP it cannot
    # outrank the intended copy target; FixedRandom resolves the 1-HP tie.
    observer = unit("Травница", "blue", 10)
    observer["health"] = 1
    battle = initialize(copier, target, observer)
    copier = battle._unit_by_position(1)
    assert copier["health"] == 1
    assert copier["doppel_copied"] == 1
    assert battle._unit_by_position(7)["health"] == target_hp
    assert battle.current_blue_attacker_pos == 10
    assert battle.winner is None
    assert battle._battle_exp_event_count == 0
    assert battle.recovery_killed_positions == set()


def test_blue_action_mask_keeps_copy_target_legality():
    battle = BattleEnv(log_enabled=False)
    copier = unit("Двойник", "blue", 7)
    healthy = unit("Лучник", "red", 1)
    wounded = unit("Скваер", "blue", 8)
    wounded["health"] = 1
    dead = unit("Рыцарь", "red", 2)
    dead["health"] = 0
    large = unit("Демон", "red", 3)
    forbidden = unit("Мизраэль", "red", 4)
    battle.combined = [copier, healthy, wounded, dead, large, forbidden]
    battle.current_blue_attacker_pos = 7
    mask = battle.compute_action_mask()
    for pos in (1, 8):
        assert mask[TARGET_POSITIONS.index(pos)]
    for pos in (2, 3, 4, 5, 7):
        assert not mask[TARGET_POSITIONS.index(pos)]


@pytest.mark.parametrize("allied", [False, True])
def test_red_picker_keeps_wounded_targets_and_rejects_illegal_ones(allied):
    battle = BattleEnv(log_enabled=False)
    copier, wounded = pair("red", "Скваер", allied)
    wounded["health"] = 1
    dead = unit("Рыцарь", "blue", 9)
    dead["health"] = 0
    large = unit("Демон", "blue", 10)
    forbidden = unit("Мизраэль", "blue", 11)
    battle.combined = [copier, wounded, dead, large, forbidden]
    assert battle._pick_highest_hp_non_big(exclude_unit=copier) == wounded["position"]
    wounded["health"] = 0
    assert battle._pick_highest_hp_non_big(exclude_unit=copier) is None
