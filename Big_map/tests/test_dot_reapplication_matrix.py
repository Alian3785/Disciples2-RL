"""B15: native poison/fire/water reapplication across both battle sides.

Use catalogue fighters and their real attack paths. Helpers only make hit rolls
predictable and give targets enough HP. In AOE cases the prospective second
target starts dead and is revived after the first application: otherwise the
first native AOE correctly affects it too and obscures reapplication behavior.
The single-target cached families also get separate already-alive-target tests.
No assertion depends on the internal representation of the source caches.
"""

import random

import pytest

from battle_env import (
    AOE_TYPES,
    BattleEnv,
    DEFEND_ACTION_INDEX,
    DOUBLE_STRIKE_TYPES,
    TARGET_POSITIONS,
)
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


UNITS = {entry["кто"]: entry for entry in DATA}
DOT_BY_TYPE = {
    "Spider": "poison",
    "Dregazul": "poison",
    "Lord": "burn",
    "Ismir son": "uran",
    "Death": "poison",
    "Sentry": "uran",
    "Watcher": "burn",
    "Dead dragon": "poison",
    "Gumtic": "burn",
    "Drulliaan": "uran",
}
CACHED_TYPES = {"Spider", "Dregazul", "Lord", "Ismir son"}
NATIVE_DOT_NAMES = tuple(
    entry["кто"]
    for entry in DATA
    if entry["тип"] in DOT_BY_TYPE or entry["кто"] == "Ниддог"
)
CACHED_DOT_NAMES = tuple(
    name
    for name in NATIVE_DOT_NAMES
    if UNITS[name]["тип"] in CACHED_TYPES or name == "Ниддог"
)
EXPECTED_NATIVE_NAMES = {
    "Имперский ассасин", "Владыка", "Сатир", "Сын Имира", "Смерть",
    "Змей ужаса", "Танатос", "Стингер", "Смотритель", "Часовой", "Головорез",
    "Мантикора", "Гигантский чёрный паук", "Ниддог", "Граф Фламель Кроули",
    "Драллиаан", "Дрега Зул", "Зверь Галлеана", "Гамтик кровавый",
}


class PredictableBattleRandom(random.Random):
    """Native hits always succeed and native effect durations are maximal."""

    def random(self):
        return 0.0

    def randint(self, lower, upper):
        return upper


def native_unit(name, team, position):
    return map_unit_to_battle(UNITS[name], team, position)


def make_battle(name, side, *, second_alive=False):
    battle = BattleEnv(log_enabled=False)
    battle.rng = PredictableBattleRandom(42)
    enemy = "red" if side == "blue" else "blue"
    own_start = 7 if side == "blue" else 1
    enemy_start = 1 if enemy == "red" else 7
    source = native_unit(name, side, own_start)
    source.update(
        accuracy=100, accuracy_secondary=100, health=10000, max_health=10000,
    )
    ally = native_unit("Рыцарь", side, own_start + 1)
    ally.update(accuracy=100, damage=10000)
    first = native_unit("Рыцарь", enemy, enemy_start)
    first.update(
        health=10000, max_health=10000, armor=0, immunity=[], resistance=[],
    )
    second = native_unit("Рыцарь", enemy, enemy_start + 1)
    second.update(
        health=10000 if second_alive else 0, max_health=10000,
        initiative=0, armor=0, immunity=[], resistance=[],
    )
    # A durable third fighter prevents first-target death from ending battle.
    guard = native_unit("Рыцарь", enemy, enemy_start + 2)
    guard.update(
        health=100000, max_health=100000, armor=0, immunity=[], resistance=[],
    )
    combatants = [source, ally, first, second, guard]
    occupied = {unit["position"] for unit in combatants}
    battle.combined = combatants + [
        placeholder_unit("red" if position < 7 else "blue", position)
        for position in range(1, 13)
        if position not in occupied
    ]
    battle.winner = None
    battle._patriach_revived_recipients = set()
    dot = "poison" if name == "Ниддог" else DOT_BY_TYPE[source["unit_type"]]
    return battle, source, ally, first, second, dot


def effect_keys(dot):
    return dot + "_turns_left", dot + "_damage_per_tick"


def revive(battle, unit):
    healer = native_unit(
        "Патриарх", unit["team"], 12 if unit["team"] == "blue" else 6,
    )
    assert battle._apply_patriach_support(healer, unit) == "revive_success"
    assert battle._alive(unit)


def apply_first(battle, source, first, second, dot):
    turns, damage = effect_keys(dot)
    assert battle._attack(source, first["position"])[0]
    assert first[turns] > 0
    assert first[damage] > 0
    assert first["health"] > 0
    assert second.get(turns, 0) == 0


def activate(battle, unit):
    # Simulate the next round for direct helper tests only. Public-step tests
    # below leave this gate and all turn scheduling to BattleEnv.
    unit["round_effects_done"] = 0
    return battle._apply_start_of_turn_effects(unit)


def finish_first_effect(battle, source, ally, first, dot, transition):
    turns, damage = effect_keys(dot)
    if transition in {"allied_kill", "target_revive", "source_revive"}:
        assert battle._attack(ally, first["position"])[0]
        assert first["health"] == 0
    elif transition == "source_kill":
        first["health"] = 1
        assert battle._attack(source, first["position"])[0]
        assert first["health"] == 0
    elif transition in {"tick_death", "last_tick_death"}:
        first["health"] = first[damage]
        if transition == "last_tick_death":
            first[turns] = 1
        else:
            assert first[turns] > 1
        assert not activate(battle, first)
        assert first["health"] == 0
    elif transition == "expiry":
        duration = first[turns]
        for _ in range(duration):
            assert activate(battle, first)
        assert first[turns] == first[damage] == 0
    elif transition == "cleanse":
        assert battle._cleanse_negative_effects(first)
        assert first[turns] == first[damage] == 0
    elif transition == "immunity_removal":
        first["immunity"] = [{"poison": "Poison", "burn": "Fire", "uran": "Water"}[dot]]
        assert activate(battle, first)
        assert first[turns] == first[damage] == 0
    else:
        raise AssertionError(transition)
    if transition == "source_revive":
        source.update(health=0, initiative=0)
        revive(battle, source)


def test_native_dot_catalogue_has_all_19_units_and_10_cached_sources():
    assert set(NATIVE_DOT_NAMES) == EXPECTED_NATIVE_NAMES
    assert len(NATIVE_DOT_NAMES) == 19
    assert len(CACHED_DOT_NAMES) == 10
    assert all(UNITS[name]["тип"] not in AOE_TYPES for name in CACHED_DOT_NAMES)


@pytest.mark.parametrize("name", NATIVE_DOT_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("transition", [
    "allied_kill", "source_kill", "tick_death", "last_tick_death", "expiry",
    "cleanse", "target_revive", "source_revive", "immunity_removal",
])
def test_native_dot_reapplies_after_first_effect_ends(name, side, transition):
    battle, source, ally, first, second, dot = make_battle(name, side)
    apply_first(battle, source, first, second, dot)
    finish_first_effect(battle, source, ally, first, dot, transition)
    if transition == "target_revive":
        revive(battle, first)
        second = first
    else:
        revive(battle, second)
    turns, damage = effect_keys(dot)
    assert second.get(turns, 0) == 0
    assert battle._attack(source, second["position"])[0]
    assert second[turns] > 0, (name, side, transition)
    assert second[damage] == source["damage_secondary"]


@pytest.mark.parametrize("name", CACHED_DOT_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("transition", [
    "allied_kill", "source_kill", "tick_death", "last_tick_death",
    "expiry", "cleanse", "source_revive", "immunity_removal",
])
def test_cached_dot_reapplies_to_already_alive_target(name, side, transition):
    battle, source, ally, first, second, dot = make_battle(
        name, side, second_alive=True,
    )
    apply_first(battle, source, first, second, dot)
    finish_first_effect(battle, source, ally, first, dot, transition)
    assert battle._attack(source, second["position"])[0]
    turns, damage = effect_keys(dot)
    assert second[turns] > 0, (name, side, transition)
    assert second[damage] == source["damage_secondary"]


@pytest.mark.parametrize("name", NATIVE_DOT_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
def test_only_cached_sources_wait_while_first_target_effect_is_active(name, side):
    battle, source, _, first, second, dot = make_battle(name, side)
    apply_first(battle, source, first, second, dot)
    revive(battle, second)
    assert battle._attack(source, second["position"])[0]
    turns, _ = effect_keys(dot)
    assert (second.get(turns, 0) > 0) == (name not in CACHED_DOT_NAMES)
    assert first["health"] > 0 and first[turns] > 0


@pytest.mark.parametrize("name", NATIVE_DOT_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("same_source", [True, False], ids=["same_source", "new_source"])
def test_native_dot_does_not_overwrite_active_target(name, side, same_source):
    battle, source, ally, first, second, dot = make_battle(name, side)
    apply_first(battle, source, first, second, dot)
    turns, damage = effect_keys(dot)
    # One normal tick makes duration refresh observable. Different secondary
    # damage makes an overwrite observable even for a one-turn native effect.
    assert activate(battle, first)
    assert first[turns] > 0
    original_effect = first[turns], first[damage]
    if not same_source:
        replacement = native_unit(name, side, ally["position"])
        replacement.update(
            accuracy=100, accuracy_secondary=100, health=10000, max_health=10000,
        )
        battle.combined[battle.combined.index(ally)] = replacement
        source = replacement
    source["damage_secondary"] = original_effect[1] + 123
    assert battle._attack(source, first["position"])[0]
    assert (first[turns], first[damage]) == original_effect


def public_reapplication(name, side, death, *, second_alive=False):
    """Run actual BLUE actions / automated RED through the native scheduler."""
    battle, source, ally, first, second, dot = make_battle(
        name, side, second_alive=second_alive,
    )
    source.update(initiative_base=100)
    strikes = 2 if source["unit_type"] in DOUBLE_STRIKE_TYPES else 1
    # Direct native strikes leave HP for the target's next real DoT tick.
    first.update(
        initiative_base=30,
        health=strikes * (source["damage"] + 5) + min(source["damage_secondary"], 10),
        max_health=10000, damage=0,
    )
    second.update(initiative_base=0, damage=0)
    ally.update(initiative_base=90)
    if death != "allied_kill":
        ally.update(health=0, initiative=0)
    guard = battle._unit_by_position(first["position"] + 2)
    guard.update(initiative_base=10, damage=0)
    turns, damage = effect_keys(dot)
    source_position = source["position"]
    first_position, second_position = first["position"], second["position"]
    native_attack = battle._attack
    source_events = []

    def record_native_attack(attacker, target_position):
        result = native_attack(attacker, target_position)
        if attacker["position"] == source_position and attacker["team"] == side:
            current_first = battle._unit_by_position(first_position)
            current_second = battle._unit_by_position(second_position)
            if death == "last_tick_death" and current_first.get(turns, 0) > 0:
                current_first[turns] = 1
            source_events.append({
                "first_hp": current_first["health"],
                "first_turns": current_first.get(turns, 0),
                "second_hp": current_second["health"],
                "second_turns": current_second.get(turns, 0),
                "second_damage": current_second.get(damage, 0),
            })
        return result

    battle._attack = record_native_attack
    battle._init_with_custom_teams(
        [unit for unit in battle.combined if unit["team"] == "red"],
        [unit for unit in battle.combined if unit["team"] == "blue"],
    )
    if side == "blue":
        for _ in range(strikes):
            assert battle.current_blue_attacker_pos == source_position
            battle.step(TARGET_POSITIONS.index(first_position))
        if death == "allied_kill":
            assert battle.current_blue_attacker_pos == ally["position"]
            battle.step(TARGET_POSITIONS.index(first_position))
    first = battle._unit_by_position(first_position)
    second = battle._unit_by_position(second_position)
    assert source_events and source_events[0]["first_turns"] > 0
    assert first["health"] == 0
    assert second.get(turns, 0) == 0
    if not second_alive:
        revive(battle, second)
    hp_before = second["health"]
    previous_events = len(source_events)
    if side == "blue":
        for _ in range(5):
            if battle.current_blue_attacker_pos == source_position:
                break
            battle.step(DEFEND_ACTION_INDEX)
        assert battle.current_blue_attacker_pos == source_position
        battle.step(TARGET_POSITIONS.index(second_position))
    else:
        for _ in range(5):
            if len(source_events) > previous_events:
                break
            battle.step(DEFEND_ACTION_INDEX)
    assert len(source_events) > previous_events
    next_attack = source_events[previous_events]
    assert 0 < next_attack["second_hp"] < hp_before
    assert next_attack["second_turns"] > 0, (name, side, death, source_events)
    assert next_attack["second_damage"] == source["damage_secondary"]


@pytest.mark.parametrize("name", NATIVE_DOT_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("death", ["allied_kill", "tick_death", "last_tick_death"])
def test_public_step_reapplies_native_dot_on_both_sides(name, side, death):
    public_reapplication(name, side, death)


@pytest.mark.parametrize("name", CACHED_DOT_NAMES)
@pytest.mark.parametrize("side", ["blue", "red"])
@pytest.mark.parametrize("death", ["allied_kill", "tick_death", "last_tick_death"])
def test_public_step_reapplies_to_already_alive_target(name, side, death):
    public_reapplication(name, side, death, second_alive=True)
