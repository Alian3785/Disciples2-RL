"""B06: slow removal restores the stat, never an already-spent activation.

Public-step cases use actual catalog units. Direct state cases intentionally isolate
pending/spent eligibility; they do not assert vanilla initiative-queue parity.
"""
from copy import deepcopy
import random

import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX, TARGET_POSITIONS, WAIT_ACTION_INDEX
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

CATALOG = {entry["кто"]: entry for entry in DATA}


class FixedRng(random.Random):
    def randint(self, low, high):
        return low

    def random(self):
        return 0.99  # No incidental natural recovery.

    def shuffle(self, values):
        pass


class RecoverRng(FixedRng):
    def random(self):
        return 0.0


class RecordedBattle(BattleEnv):
    def __init__(self):
        super().__init__()
        self.attack_events = []
        self._patriach_revived_recipients = set()

    def _attack(self, attacker, target_pos):
        self.attack_events.append((self.round_no, attacker["team"], attacker["position"]))
        return super()._attack(attacker, target_pos)


def native(name, team, position, initiative=None):
    unit = map_unit_to_battle(CATALOG[name], team, position)
    unit.update(health=10000, max_health=10000, accuracy=100,
                accuracy_secondary=100, immunity=[], resistance=[])
    if initiative is not None:
        unit.update(initiative_base=initiative, initiative=initiative)
    return unit


def battle_with(*units):
    battle = RecordedBattle()
    battle.rng = FixedRng(0)
    positions = {unit["position"] for unit in units}
    roster = list(units) + [placeholder_unit("red" if p < 7 else "blue", p)
                            for p in range(1, 13) if p not in positions]
    battle._init_with_custom_teams(
        [u for u in roster if u["team"] == "red"],
        [u for u in roster if u["team"] == "blue"],
    )
    return battle


def slow_case(team="blue", initiative=50, current=0):
    battle = RecordedBattle()
    battle.rng = FixedRng(0)
    unit = native("Рыцарь", team, 7 if team == "blue" else 1, initiative)
    hermit = native("Отшельник", "red" if team == "blue" else "blue",
                    1 if team == "blue" else 7)
    battle.combined = [unit, hermit]
    unit["initiative"] = current
    assert battle._apply_hermit_initiative_slow(hermit, unit)
    return battle, unit, hermit


def test_native_knight_hermit_abbess_does_not_repeat_round_one():
    battle = battle_with(native("Рыцарь", "blue", 7),
                         native("Аббатиса", "blue", 10),
                         native("Отшельник", "red", 1))
    assert battle.current_blue_attacker_pos == 7
    battle.step(DEFEND_ACTION_INDEX)
    knight = battle._unit_by_position(7)
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 10)
    assert (knight["initiative_base"], knight["initiative"]) == (25, 0)
    battle.step(TARGET_POSITIONS.index(1))
    # A new action is legal only after the round boundary.
    assert (battle.round_no, battle.current_blue_attacker_pos) == (2, 7)
    assert (knight["initiative_base"], knight["initiative"]) == (50, 50)


@pytest.mark.parametrize("cleanser", ["Аббатиса", "Прорицательница", "Друид", "Архидруид"])
def test_public_cleanse_keeps_spent_knight_zero_before_round_boundary(cleanser):
    battle = battle_with(native("Рыцарь", "blue", 7),
                         native(cleanser, "blue", 10, 10),
                         native("Крестьянин", "blue", 8, 4),
                         native("Отшельник", "red", 1))
    battle.step(DEFEND_ACTION_INDEX)
    assert battle.current_blue_attacker_pos == 10
    knight = battle._unit_by_position(7)
    battle.step(TARGET_POSITIONS.index(1))
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 8)
    assert (knight["initiative_base"], knight["initiative"], knight["hermited"]) == (50, 0, 0)
    assert knight not in battle._candidates()


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("base", [0, 1, 35, 50, 65, 101])
@pytest.mark.parametrize("removal", ["cleanse", "natural", "helper"])
def test_spent_initiative_stays_spent_for_every_removal(team, base, removal):
    battle, unit, _ = slow_case(team, base)
    if removal == "cleanse":
        assert battle._cleanse_negative_effects(unit)
    elif removal == "natural":
        battle.rng = RecoverRng(0)
        battle._apply_start_of_turn_effects(unit)
    else:
        assert battle._restore_hermit_initiative(unit)
    assert unit["initiative"] == 0
    assert unit["initiative_base"] == base
    assert not unit["hermited"]
    assert "hermit_original_initiative_base" not in unit
    restored = deepcopy(unit)
    assert not battle._restore_hermit_initiative(unit)
    assert unit == restored
    assert unit not in battle._candidates()
    battle.rng = FixedRng(0)
    battle._end_round_restore()
    assert unit["initiative"] == base


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("base", [35, 50, 65, 101])
@pytest.mark.parametrize("current", [1, 9, 25, 60, 120])
def test_pending_priority_keeps_existing_max_semantics(team, base, current):
    battle, unit, _ = slow_case(team, base, current=base)
    # Includes wait-like low priority, normal priority and retained high jitter.
    unit["initiative"] = current
    rng_state = battle.rng.getstate()
    assert battle._cleanse_negative_effects(unit)
    assert (unit["initiative_base"], unit["initiative"]) == (base, max(current, base))
    assert battle.rng.getstate() == rng_state  # Removal must not reroll initiative.
    assert unit in battle._candidates()


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("grant", ["alchemist", "item"])
@pytest.mark.parametrize("before_cleanse", [False, True])
def test_authorized_extra_activation_survives_cleanse(team, grant, before_cleanse):
    battle, unit, _ = slow_case(team)
    alchemist = native("Алхимик", team, 10 if team == "blue" else 4)

    def give_turn():
        if grant == "alchemist":
            assert battle._apply_alchemist_support(alchemist, unit)
        else:
            assert battle._apply_hero_item_extra_turn(source_unit=alchemist, target_unit=unit)

    if before_cleanse:
        give_turn()
        assert unit["initiative"] == 25
    assert battle._cleanse_negative_effects(unit)
    if not before_cleanse:
        assert unit["initiative"] == 0
        give_turn()
    assert (unit["initiative"], unit["initiative_base"], unit["bonusturn"]) == (50, 50, 1)
    assert unit in battle._candidates()
    unit["initiative"] = 0  # Consume that legitimate activation.
    assert not battle._cleanse_negative_effects(unit)
    assert unit not in battle._candidates()


@pytest.mark.parametrize("cleanse_first", [False, True])
def test_public_alchemist_can_reactivate_after_spent_turn(cleanse_first):
    battle = battle_with(native("Рыцарь", "blue", 7, 60),
                         native("Алхимик", "blue", 10, 40),
                         native("Крестьянин", "blue", 8, 1),
                         native("Рыцарь", "red", 1, 2))
    knight = battle._unit_by_position(7)
    battle.step(DEFEND_ACTION_INDEX)
    assert battle.current_blue_attacker_pos == 10
    hermit = native("Отшельник", "red", 4)
    assert battle._apply_hermit_initiative_slow(hermit, knight)
    if cleanse_first:
        assert battle._cleanse_negative_effects(knight)
        assert knight["initiative"] == 0
    battle.step(TARGET_POSITIONS.index(1))  # Native Alchemist action.
    if not cleanse_first:
        assert battle._cleanse_negative_effects(knight)
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 7)
    assert knight["bonusturn"] == 1
    battle.step(DEFEND_ACTION_INDEX)
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 8)
    assert knight["initiative"] == 0


def test_public_waited_activation_is_not_erased():
    battle = battle_with(native("Рыцарь", "blue", 7, 60),
                         native("Друид", "blue", 10, 20),
                         native("Рыцарь", "red", 1, 1))
    knight = battle._unit_by_position(7)
    assert battle._apply_hermit_initiative_slow(native("Отшельник", "red", 4), knight)
    battle.step(WAIT_ACTION_INDEX)
    assert (knight["initiative"], knight["waited"]) == (3, 1)
    assert battle.current_blue_attacker_pos == 10
    battle.step(TARGET_POSITIONS.index(1))
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 7)
    assert (knight["initiative"], knight["waited"]) == (60, 1)


@pytest.mark.parametrize("name", ["Мастер клинка", "Бандит"])
def test_cleanse_between_double_strikes_preserves_only_remaining_strike(name):
    battle = battle_with(native(name, "blue", 7, 60),
                         native("Крестьянин", "blue", 8, 1),
                         native("Рыцарь", "red", 1, 2))
    unit = battle._unit_by_position(7)
    assert battle._apply_hermit_initiative_slow(native("Отшельник", "red", 4), unit)
    battle.step(TARGET_POSITIONS.index(1))
    assert (unit["initiative"], battle.blue_attacks_left) == (0, 1)
    assert battle._cleanse_negative_effects(unit)
    assert (unit["initiative"], battle.blue_attacks_left) == (0, 1)
    battle.step(TARGET_POSITIONS.index(1))
    assert battle.attack_events.count((1, "blue", 7)) == 2
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 8)


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("reviver", ["patriarch", "item"])
def test_revival_cleanses_slow_without_overriding_zero_initiative(team, reviver):
    battle, unit, _ = slow_case(team)
    unit["health"] = 0
    healer = native("Патриарх", team, 10 if team == "blue" else 4)
    if reviver == "patriarch":
        assert battle._apply_patriach_support(healer, unit) == "revive_success"
    else:
        assert battle._apply_hero_item_revive(unit, 100) == 100
    assert unit["health"] > 0
    assert (unit["initiative_base"], unit["initiative"], unit["hermited"]) == (50, 0, 0)


@pytest.mark.parametrize("team", ["blue", "red"])
def test_dead_restore_and_escaped_slot_never_become_candidates(team):
    battle, unit, _ = slow_case(team)
    unit["health"] = 0
    assert not battle._cleanse_negative_effects(unit)
    assert battle._restore_hermit_initiative(unit)
    assert (unit["initiative_base"], unit["initiative"]) == (50, 0)
    assert unit not in battle._candidates()
    unit["health"] = 100
    assert battle._apply_hermit_initiative_slow(native("Отшельник", "red", 4), unit)
    battle._mark_unit_escaped(unit)
    assert unit["name"] == "пусто"
    assert not battle._restore_hermit_initiative(unit)
    assert unit not in battle._candidates()
    escaped = battle.escaped_units[-1]
    assert battle._restore_hermit_initiative(escaped)
    assert (escaped["initiative_base"], escaped["initiative"]) == (50, 0)


@pytest.mark.parametrize("current", [0, 1, 70])
def test_legacy_slow_state_without_saved_base(current):
    unit = {"hermited": 1, "initiative_base": 25, "initiative": current}
    assert BattleEnv._restore_hermit_initiative(unit)
    assert unit == {"hermited": 0, "initiative_base": 50,
                    "initiative": 0 if current == 0 else max(current, 50)}
