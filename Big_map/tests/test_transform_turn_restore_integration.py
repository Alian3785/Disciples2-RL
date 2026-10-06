"""Transformation cleanup preserves live activation state through real scheduling.

Mid-round fixtures use native transformations. BLUE consumes actions with step;
RED uses the ordinary automatic controller and its common support executor.
No scheduler or action implementation is replaced in these tests.
"""

import random

import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX, TARGET_POSITIONS
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


CATALOG = {entry["кто"]: entry for entry in DATA}
FORMS = ("witch", "wight", "lycanthropy")


class FixedRng(random.Random):
    def randint(self, low, high):
        return low

    def random(self):
        return 0.99  # Recovery happens only in the explicit expiry cases.

    def choice(self, values):
        return values[0]

    def shuffle(self, values):
        pass


class RecoverRng(FixedRng):
    def random(self):
        return 0.0


class RecordedBattle(BattleEnv):
    def __init__(self):
        super().__init__(log_enabled=False)
        self.attack_events = []

    def _attack(self, attacker, target_pos):
        self.attack_events.append(
            (self.round_no, attacker["team"], attacker["position"], target_pos)
        )
        return super()._attack(attacker, target_pos)


def native(name, team, position, initiative=50):
    unit = map_unit_to_battle(CATALOG[name], team, position)
    unit.update(
        health=10000, max_health=10000, accuracy=100,
        accuracy_secondary=100, immunity=[], resistance=[],
        initiative_base=initiative, initiative=initiative,
    )
    return unit


def case(team="blue", support="Прорицательница", target_name="Рыцарь"):
    own = 6 if team == "blue" else 0
    opposite = "red" if team == "blue" else "blue"
    units = {
        "target": native(target_name, team, own + 1),
        "support": native(support, team, own + 4, 20),
        "enemy": native("Рыцарь", opposite, 1 if team == "blue" else 7, 1),
        "gate": native("Крестьянин", "blue", 12, 1000),
    }
    occupied = {unit["position"] for unit in units.values()}
    roster = list(units.values()) + [
        placeholder_unit("red" if position < 7 else "blue", position)
        for position in range(1, 13) if position not in occupied
    ]
    battle = RecordedBattle()
    battle.rng = FixedRng(0)
    battle._init_with_custom_teams(
        [unit for unit in roster if unit["team"] == "red"],
        [unit for unit in roster if unit["team"] == "blue"],
    )
    units = {key: battle._unit_by_position(unit["position"])
             for key, unit in units.items()}
    # The high-priority BLUE gate prevents RED from acting during initialization.
    assert battle.current_blue_attacker_pos == units["gate"]["position"]
    for unit in battle.combined:
        unit.update(initiative=0, round_effects_done=1)
    units["gate"]["initiative_base"] = 1
    battle.current_blue_attacker_pos = None
    battle.blue_attacks_left = 0
    battle.attack_events.clear()
    return battle, units


def transform(battle, units, kind):
    target, source = units["target"], units["enemy"]
    if kind == "witch":
        battle._apply_witch_effect(source, target)
    elif kind == "wight":
        assert battle._apply_wight_decay_effect(source, target)
    else:
        assert battle._apply_hero_item_lycanthropy(
            source_unit=source, target_unit=target,
            effect={"damage_type": "Mind", "transform_unit_name": "Оборотень"},
        ) == 1
    assert target["transformed"] == 1


def attacks_by(battle, target, round_no=1):
    return [event for event in battle.attack_events
            if event[:3] == (round_no, target["team"], target["position"])]


def spend_target_action(battle, units):
    target = units["target"]
    units["gate"]["initiative"] = 1
    assert battle._advance_until_blue_turn()
    if target["team"] == "blue":
        assert battle.current_blue_attacker_pos == target["position"]
        _, _, terminated, truncated, _ = battle.step(DEFEND_ACTION_INDEX)
        assert not terminated and not truncated
    else:
        assert len(attacks_by(battle, target)) == 1
    assert battle.current_blue_attacker_pos == units["gate"]["position"]
    assert (battle.round_no, target["initiative"]) == (1, 0)


def support_action(battle, units):
    target, support = units["target"], units["support"]
    support["initiative"] = 100
    units["gate"]["initiative"] = 90
    battle.current_blue_attacker_pos = None
    battle.blue_attacks_left = 0
    assert battle._advance_until_blue_turn()
    if support["team"] == "blue":
        assert battle.current_blue_attacker_pos == support["position"]
        selector = target["position"] - 6
        _, _, terminated, truncated, _ = battle.step(TARGET_POSITIONS.index(selector))
        assert not terminated and not truncated
    assert battle.current_blue_attacker_pos == units["gate"]["position"]
    assert battle.round_no == 1
    assert len(attacks_by(battle, support)) == 1


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("kind", FORMS)
@pytest.mark.parametrize("cleanser", ("Прорицательница", "Друид"))
def test_real_cleanse_does_not_reactivate_a_spent_transformed_fighter(team, kind, cleanser):
    battle, units = case(team, support=cleanser)
    target = units["target"]
    target["initiative"] = 50
    transform(battle, units, kind)
    spend_target_action(battle, units)
    target["health"] = max(1, target["max_health"] // 2)
    before = len(attacks_by(battle, target))
    support_action(battle, units)
    assert (target["transformed"], target["initiative_base"], target["initiative"]) == (0, 50, 0)
    assert target not in battle._candidates()
    assert len(attacks_by(battle, target)) == before
    # Let the real scheduler run again: the next opportunity is in round two.
    battle.step(DEFEND_ACTION_INDEX)
    assert battle.round_no == 2
    assert len(attacks_by(battle, target)) == before


@pytest.mark.parametrize("kind", ("wight", "lycanthropy"))
@pytest.mark.parametrize("team", ("blue", "red"))
def test_natural_expiry_after_round_boundary_retains_a_real_new_action(team, kind):
    battle, units = case(team)
    target = units["target"]
    # The original fighter already acted before being transformed in round one.
    transform(battle, units, kind)
    assert target["basestats"][target["position"]][0]["initiative"] == 0
    for unit in battle.combined:
        if unit is not target and unit["max_health"] > 0:
            unit["initiative_base"] = 1
    battle.rng = RecoverRng(0)
    assert battle._advance_until_blue_turn()  # Real round boundary and recovery.
    assert battle.round_no == 2
    assert target["transformed"] == 0
    assert target["initiative_base"] == 50
    if team == "blue":
        assert battle.current_blue_attacker_pos == target["position"]
        assert target["initiative"] > 0
        battle.step(TARGET_POSITIONS.index(units["enemy"]["position"]))
    assert len(attacks_by(battle, target, round_no=2)) == 1
    assert (battle.round_no, target["initiative"]) == (2, 0)


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("kind", FORMS)
@pytest.mark.parametrize("cleanse_first", (False, True))
def test_real_alchemist_extra_action_survives_cleanup_on_either_side(team, kind, cleanse_first):
    battle, units = case(team, support="Алхимик")
    target = units["target"]
    target["initiative"] = 50
    spend_target_action(battle, units)
    # Snapshot zero is normal when a spent fighter is transformed later.
    transform(battle, units, kind)
    if cleanse_first:
        assert battle._cleanse_negative_effects(target)
        assert target["initiative"] == 0
    before = len(attacks_by(battle, target))
    support_action(battle, units)
    assert target["bonusturn"] == 1
    if not cleanse_first:
        assert battle._cleanse_negative_effects(target)
    assert target["initiative"] > 0
    assert target in battle._candidates()
    units["gate"]["initiative"] = 1
    battle.current_blue_attacker_pos = None
    battle.blue_attacks_left = 0
    assert battle._advance_until_blue_turn()
    if team == "blue":
        assert battle.current_blue_attacker_pos == target["position"]
        battle.step(TARGET_POSITIONS.index(units["enemy"]["position"]))
    assert len(attacks_by(battle, target)) == before + 1
    assert (battle.round_no, target["initiative"], target["bonusturn"]) == (1, 0, 1)
    assert target not in battle._candidates()


@pytest.mark.parametrize("name", ("Мастер клинка", "Бандит"))
def test_cleanup_between_double_strikes_does_not_reset_strike_budget(name):
    battle, units = case(target_name=name)
    target = units["target"]
    target["initiative"] = 50
    units["gate"]["initiative"] = 1
    assert battle._advance_until_blue_turn()
    assert battle.current_blue_attacker_pos == target["position"]
    battle.step(TARGET_POSITIONS.index(units["enemy"]["position"]))
    assert (target["initiative"], battle.blue_attacks_left) == (0, 1)
    # Inject the native effect between the two public strikes, then remove it.
    # Witch stores a positive old priority even though the action was spent.
    transform(battle, units, "witch")
    assert battle._cleanse_negative_effects(target)
    assert (target["initiative"], battle.blue_attacks_left) == (0, 1)
    battle.step(TARGET_POSITIONS.index(units["enemy"]["position"]))
    assert len(attacks_by(battle, target)) == 2
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, units["gate"]["position"])
    assert target not in battle._candidates()
