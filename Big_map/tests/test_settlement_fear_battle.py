"""Settlement occupants lose one activation to Fear instead of fleeing.

Battle context belongs to the defending team, not to a particular catalog unit,
temporary form, saved stat snapshot, or the player retreat-action setting.
"""

from copy import deepcopy
import pickle
import random

import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX, TARGET_POSITIONS
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


CATALOG = {entry["кто"]: entry for entry in DATA}
CONTROL_FLAGS = ("paralyzed", "long_paralyzed", "running_away", "feared")


class FixedRandom(random.Random):
    def __init__(self, value=0.0):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value

    def randint(self, low, high):
        return low

    def shuffle(self, values):
        pass

    def choice(self, values):
        return values[0]


class RecordedBattle(BattleEnv):
    def __init__(self, **kwargs):
        super().__init__(log_enabled=True, **kwargs)
        self.attack_events = []

    def _attack(self, attacker, target_pos):
        self.attack_events.append(
            (self.round_no, attacker["team"], attacker["position"])
        )
        return super()._attack(attacker, target_pos)


def other_team(team):
    return "red" if team == "blue" else "blue"


def front_position(team):
    return 7 if team == "blue" else 1


def fighter(name, team, position, initiative=None):
    unit = map_unit_to_battle(CATALOG[name], team, position)
    if initiative is not None:
        unit.update(initiative_base=initiative, initiative=initiative)
    return unit


def fear_pair(team, *, attacker_name="Баронесса", victim_name="Рыцарь",
              protected=True, **kwargs):
    if protected:
        kwargs.setdefault("fear_paralysis_teams", (team,))
    battle = RecordedBattle(**kwargs)
    battle.rng = FixedRandom()
    attacker = fighter(attacker_name, other_team(team), front_position(other_team(team)))
    victim = fighter(victim_name, team, front_position(team))
    battle.combined = [attacker, victim]
    return battle, attacker, victim


def assert_short_paralysis(victim):
    assert tuple(victim.get(key, 0) for key in CONTROL_FLAGS) == (1, 0, 0, 0)


def assert_no_control(victim):
    assert not any(victim.get(key, 0) for key in CONTROL_FLAGS)


def initialize(battle, *units):
    positions = {unit["position"] for unit in units}
    roster = list(units) + [
        placeholder_unit("red" if pos < 7 else "blue", pos)
        for pos in range(1, 13) if pos not in positions
    ]
    battle._init_with_custom_teams(
        [unit for unit in roster if unit["team"] == "red"],
        [unit for unit in roster if unit["team"] == "blue"],
    )


@pytest.mark.parametrize("team", ("blue", "red"))
def test_success_changes_only_existing_short_paralysis_flag(team):
    battle, attacker, victim = fear_pair(team)
    before = deepcopy(victim)

    assert battle._apply_fear_effect(attacker, victim)

    assert_short_paralysis(victim)
    expected = deepcopy(before)
    expected["paralyzed"] = 1
    assert victim == expected  # No settlement metadata or duration on the unit.
    assert not battle.escaped_units


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("roll,hit", ((0.0, True), (0.99, False)))
def test_native_baroness_preserves_accuracy_before_settlement_effect(team, roll, hit):
    battle, attacker, victim = fear_pair(team)
    battle.rng = FixedRandom(roll)
    before_health = victim["health"]
    assert attacker["unit_type"] == "Baroness"
    assert attacker["accuracy"] == 80

    success, reason = battle._attack(attacker, victim["position"])

    assert success
    assert reason == ("ok" if hit else "miss")
    assert victim["health"] == before_health  # Catalog Fear does no damage.
    if hit:
        assert_short_paralysis(victim)
    else:
        assert_no_control(victim)
    assert not battle.escaped_units


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("immunity", ("Mind", "mind", "MIND"))
def test_mind_immunity_precedes_conversion_and_does_not_consume_ward(team, immunity):
    battle, attacker, victim = fear_pair(team)
    victim.update(immunity=[immunity], resistance=["Mind"])
    before = deepcopy(victim)

    assert not battle._apply_fear_effect(attacker, victim)
    assert battle._attack(attacker, victim["position"]) == (True, "immune_damage")

    assert victim == before
    assert not victim.get("resilience_used_types")
    assert_no_control(victim)


@pytest.mark.parametrize("team", ("blue", "red"))
def test_native_mind_immune_unit_remains_immune_inside_settlement(team):
    battle, attacker, victim = fear_pair(team, victim_name="Охотник на ведьм")
    assert "mind" in {str(tag).lower() for tag in victim["immunity"]}
    before = deepcopy(victim)
    assert battle._attack(attacker, victim["position"]) == (True, "immune_damage")
    assert victim == before


@pytest.mark.parametrize("team", ("blue", "red"))
def test_native_baroness_mind_ward_blocks_once_then_short_paralysis(team):
    battle, attacker, victim = fear_pair(team)
    victim["resistance"] = ["Mind"]

    assert battle._attack(attacker, victim["position"]) == (True, "resilience_block")
    assert victim["resilience_used_types"] == ["Mind"]
    assert_no_control(victim)

    assert battle._attack(attacker, victim["position"])[0]
    assert_short_paralysis(victim)
    assert victim["resilience_used_types"] == ["Mind"]


@pytest.mark.parametrize("team", ("blue", "red"))
def test_secondary_fear_keeps_primary_and_mind_ward_checks_separate(team):
    battle, attacker, victim = fear_pair(team, attacker_name="Шаманка")
    victim.update(health=10000, max_health=10000, resistance=["Earth", "Mind"])
    assert attacker["attack_type_primary"].lower() == "earth"
    assert attacker["attack_type_secondary"].lower() == "mind"

    assert battle._attack(attacker, victim["position"])[0]
    assert victim["resilience_used_types"] == ["Earth"]
    assert_no_control(victim)
    assert battle._attack(attacker, victim["position"])[0]
    assert victim["resilience_used_types"] == ["Earth", "Mind"]
    assert_no_control(victim)
    assert battle._attack(attacker, victim["position"])[0]
    assert_short_paralysis(victim)


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("immune_type,applies", (("Earth", True), ("Mind", False)))
def test_secondary_fear_gate_uses_its_own_source(team, immune_type, applies):
    battle, attacker, victim = fear_pair(team, attacker_name="Шаманка")
    victim["immunity"] = [immune_type]

    assert battle._apply_fear_effect(attacker, victim) is applies
    if applies:
        assert_short_paralysis(victim)
    else:
        assert_no_control(victim)


@pytest.mark.parametrize("team", ("blue", "red"))
def test_item_fear_uses_the_same_short_paralysis_and_no_stacking(team):
    battle, attacker, victim = fear_pair(team)
    kwargs = dict(source_unit=attacker, target_unit=victim, effect={"damage_type": "Mind"})
    assert battle._apply_hero_item_fear(**kwargs) == 1.0
    before = deepcopy(victim)
    assert battle._apply_hero_item_fear(**kwargs) == 0.0
    assert victim == before
    assert_short_paralysis(victim)


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("cleanser_name", ("Аббатиса", "Прорицательница", "Друид", "Архидруид"))
def test_native_cleanse_removes_settlement_fear_without_a_residual_escape(team, cleanser_name):
    battle, attacker, victim = fear_pair(team)
    cleanser = fighter(cleanser_name, team, front_position(team) + 3)
    battle.combined.append(cleanser)
    assert battle._attack(attacker, victim["position"])[0]
    assert_short_paralysis(victim)
    initiative = victim["initiative"]

    selector = battle._support_selector_for_recipient(cleanser, victim["position"])
    assert battle._attack(cleanser, selector)[0]

    assert_no_control(victim)
    assert victim["initiative"] == initiative
    assert battle._apply_start_of_turn_effects(victim)
    assert battle._alive(victim)
    assert not battle.escaped_units


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("victim_name,strikes", (("Рыцарь", 1), ("Мастер клинка", 2), ("Бандит", 2)))
def test_one_whole_activation_is_skipped_then_native_attacks_resume(team, victim_name, strikes):
    battle = RecordedBattle(fear_paralysis_teams=(team,))
    battle.rng = FixedRandom()
    victim_pos = front_position(team)
    attacker_pos = front_position(other_team(team))
    units = [
        fighter("Крестьянин", "blue", 9, 100),  # First player action.
        fighter("Крестьянин", "blue", 8, 10),   # Stop before the round boundary.
        fighter(victim_name, team, victim_pos, 60),
        fighter("Баронесса", other_team(team), attacker_pos, 5),
    ]
    for unit in units:
        unit.update(health=10000, max_health=10000, accuracy=100)
    initialize(battle, *units)
    victim = battle._unit_by_position(victim_pos)
    attacker = battle._unit_by_position(attacker_pos)
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 9)

    # Use the actual catalog attack to apply the effect before the next activation.
    assert battle._attack(attacker, victim_pos)[0]
    assert_short_paralysis(victim)
    before = deepcopy(victim)
    for _ in range(3):
        assert not battle._apply_fear_effect(attacker, victim)
        assert victim == before
    attacker["accuracy"] = 0  # Later Baroness turns cannot reapply Fear.

    battle.step(DEFEND_ACTION_INDEX)
    assert (battle.round_no, battle.current_blue_attacker_pos) == (1, 8)
    assert victim["initiative"] == 0
    assert_no_control(victim)
    assert (1, team, victim_pos) not in battle.attack_events
    assert not battle.escaped_units
    assert battle.winner is None
    assert sum("пропускает ход" in event for event in battle._pretty_events) == 1

    # Finish the known remaining one or two BLUE turns, without synthetic resets.
    battle.step(DEFEND_ACTION_INDEX)
    if team == "red":
        assert (battle.round_no, battle.current_blue_attacker_pos) == (1, attacker_pos)
        battle.step(DEFEND_ACTION_INDEX)
    assert (battle.round_no, battle.current_blue_attacker_pos) == (2, 9)
    battle.step(DEFEND_ACTION_INDEX)

    if team == "blue":
        assert battle.current_blue_attacker_pos == victim_pos
        assert battle.blue_attacks_left == strikes
        for _ in range(strikes):
            battle.step(TARGET_POSITIONS.index(attacker_pos))
    assert battle.attack_events.count((2, team, victim_pos)) == strikes
    assert (battle.round_no, battle.current_blue_attacker_pos) == (2, 8)
    assert not battle.escaped_units
    assert_no_control(victim)
    assert sum("пропускает ход" in event for event in battle._pretty_events) == 1


@pytest.mark.parametrize("team", ("blue", "red"))
def test_fear_can_apply_again_after_the_short_effect_is_cleansed(team):
    battle, attacker, victim = fear_pair(team)
    assert battle._apply_fear_effect(attacker, victim)
    assert not battle._apply_fear_effect(attacker, victim)
    assert battle._cleanse_negative_effects(victim)
    assert battle._apply_fear_effect(attacker, victim)
    assert_short_paralysis(victim)


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("setting", ("default", "retreat_disabled", "only_other_team"))
def test_field_fear_still_flees_and_retreat_setting_is_independent(team, setting):
    kwargs = {}
    if setting == "retreat_disabled":
        kwargs["retreat_enabled"] = False
    elif setting == "only_other_team":
        kwargs["fear_paralysis_teams"] = (other_team(team),)
    battle, attacker, victim = fear_pair(team, protected=False, **kwargs)
    name, health, position = victim["name"], victim["health"], victim["position"]

    assert battle._attack(attacker, position)[0]
    assert tuple(victim.get(key, 0) for key in CONTROL_FLAGS) == (0, 0, 1, 1)
    assert not battle._apply_start_of_turn_effects(victim)

    assert len(battle.escaped_units) == 1
    escaped = battle.escaped_units[0]
    assert (escaped["name"], escaped["health"], escaped["position"]) == (name, health, position)
    assert victim["health"] == 0
    assert battle.winner == other_team(team)


@pytest.mark.parametrize("team", ("blue", "red"))
def test_existing_voluntary_retreat_is_not_converted_or_cured(team):
    battle, attacker, victim = fear_pair(team)
    victim.update(running_away=1, feared=0)
    before = deepcopy(victim)
    assert not battle._apply_fear_effect(attacker, victim)
    assert victim == before
    assert not battle._cleanse_negative_effects(victim)
    assert victim == before
    assert not battle._apply_start_of_turn_effects(victim)
    assert len(battle.escaped_units) == 1


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("form", ("witch", "doppelganger"))
@pytest.mark.parametrize("copy_mode", ("live", "deepcopy", "pickle"))
def test_battle_context_survives_form_changes_and_battle_copies(team, form, copy_mode):
    battle, attacker, victim = fear_pair(
        team, victim_name="Двойник" if form == "doppelganger" else "Рыцарь",
    )
    position = victim["position"]
    if form == "witch":
        witch = fighter("Ведьма", attacker["team"], attacker["position"] + 1)
        battle.combined.append(witch)
        assert battle._attack(witch, position)[0]
        assert victim["transformed"] == 1
    else:
        target = fighter("Рыцарь", attacker["team"], attacker["position"] + 1)
        battle.combined.append(target)
        assert battle._apply_doppelganger_copy(victim, target)
        assert victim["doppel_copied"] == 1
    before_snapshot = deepcopy(victim.get("basestats"))

    if copy_mode == "deepcopy":
        battle = deepcopy(battle)
    elif copy_mode == "pickle":
        battle = pickle.loads(pickle.dumps(battle))
    victim = battle._unit_by_position(position)
    attacker = battle._unit_by_position(front_position(other_team(team)))

    assert battle._apply_fear_effect(attacker, victim)
    assert_short_paralysis(victim)
    assert victim.get("basestats") == before_snapshot
    if form == "witch":
        assert battle._restore_transformed_unit(victim)
        assert_short_paralysis(victim)
    else:
        battle._restore_default_doppelgangers()
        assert victim["unit_type"] == "Doppelganger"
    assert battle._cleanse_negative_effects(victim)
    assert battle._apply_fear_effect(attacker, victim)
    assert_short_paralysis(victim)
    assert not battle.escaped_units


@pytest.mark.parametrize("team", ("blue", "red"))
def test_copied_unit_does_not_carry_settlement_protection_into_another_battle(team):
    battle, attacker, victim = fear_pair(team)
    assert battle._apply_fear_effect(attacker, victim)
    assert battle._cleanse_negative_effects(victim)

    field = BattleEnv()
    copied_attacker, copied_victim = deepcopy((attacker, victim))
    field.combined = [copied_attacker, copied_victim]
    assert field._apply_fear_effect(copied_attacker, copied_victim)
    assert tuple(copied_victim.get(key, 0) for key in CONTROL_FLAGS) == (0, 0, 1, 1)
    assert_no_control(victim)


@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("death_route", ("attack", "poison"))
def test_frightened_occupant_dies_normally_and_is_never_an_escape(team, death_route):
    battle, attacker, victim = fear_pair(team)
    assert battle._attack(attacker, victim["position"])[0]
    assert_short_paralysis(victim)
    victim["health"] = 1

    if death_route == "attack":
        killer = fighter("Рыцарь", attacker["team"], attacker["position"] + 1)
        battle.combined.append(killer)
        assert battle._attack(killer, victim["position"])[0]
        battle._check_victory_after_hit()
    else:
        victim.update(poison_turns_left=1, poison_damage_per_tick=10)
        assert not battle._apply_start_of_turn_effects(victim)

    assert victim["health"] == 0
    assert victim["name"] == "Рыцарь"
    assert not victim.get("running_away", 0)
    assert not victim.get("feared", 0)
    assert not battle.escaped_units
    assert battle.winner == other_team(team)
    assert victim not in battle._candidates()
