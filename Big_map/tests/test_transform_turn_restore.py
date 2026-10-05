"""Form restoration changes stats, never the live entitlement to an activation."""
from copy import deepcopy
import json
import pickle
import random

import pytest

from battle_env import BattleEnv, HAGS_RING_ARTIFACT_ITEM_NAME
from data_dicts_compact_lines import DATA, map_unit_to_battle

CATALOG = {row["кто"]: row for row in DATA}
CASTERS = ("Ведьма", "Колдунья", "Суккуб", "Сущий")
ROUTES = CASTERS + ("item_transform", "lycanthropy", "artifact")
TURN_KEYS = ("waited", "bonusturn", "round_effects_done", "running_away", "feared")


class FixedRng(random.Random):
    def random(self):
        return 0.0

    def randint(self, low, high):
        return low

    def shuffle(self, values):
        pass


def fighter(name, team, position):
    return map_unit_to_battle(CATALOG[name], team, position)


def scenario(team, route, *, initial=57, name="Рыцарь"):
    other = "red" if team == "blue" else "blue"
    target = fighter(name, team, 7 if team == "blue" else 1)
    caster = fighter(route if route in CASTERS else "Герцог", other,
                     1 if team == "blue" else 7)
    battle = BattleEnv(log_enabled=False)
    battle.rng = FixedRng(0)
    battle.combined = [target, caster]
    battle._patriach_revived_recipients = set()
    target["initiative"] = initial
    target.update(immunity=[], resistance=[])
    original = deepcopy(target)
    if route in CASTERS:
        assert battle._attack(caster, target["position"])[0]
    elif route == "lycanthropy":
        assert battle._apply_hero_item_lycanthropy(
            source_unit=caster, target_unit=target,
            effect={"transform_unit_name": "Оборотень"},
        ) == 1
    elif route == "item_transform":
        assert battle._apply_hero_item_transform(
            source_unit=caster, target_unit=target, effect={},
        ) == 1
    else:
        assert battle._apply_hero_artifact_transform(caster, target, HAGS_RING_ARTIFACT_ITEM_NAME)
    assert target["transformed"] == 1
    return battle, target, original


def restore(battle, target, route):
    if route == "cleanse":
        assert battle._cleanse_negative_effects(target)
    elif route == "expiry":
        target["round_effects_done"] = 0
        target["transform_recover_chance"] = 1.0
        assert battle._apply_start_of_turn_effects(target)
    else:
        assert battle._restore_transformed_unit(target)
    assert target["transformed"] == 0


def test_catalog_covers_all_hostile_transform_casters():
    actual = {row["кто"] for row in DATA
              if map_unit_to_battle(row, "blue", 7)["unit_type"]
              in {"Witch", "Succub", "Wight"}}
    assert actual == set(CASTERS)


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("cleanup", ["cleanse", "expiry", "direct"])
@pytest.mark.parametrize("initial", [0, 57])
def test_spent_activation_stays_spent_for_every_transform_route(team, route, cleanup, initial):
    battle, target, original = scenario(team, route, initial=initial)
    target.update(initiative=0, bonusturn=2, waited=0)
    battle.current_blue_attacker_pos = 10
    battle.blue_attacks_left = 1
    restore(battle, target, cleanup)
    assert target["initiative"] == 0
    assert target["initiative_base"] == original["initiative_base"]
    assert target["bonusturn"] == 2
    assert battle.current_blue_attacker_pos == 10
    assert battle.blue_attacks_left == 1
    assert not battle._restore_transformed_unit(target)
    assert target["initiative"] == 0


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("initial", [0, 57])
def test_pending_activation_and_wait_priority_survive_old_snapshot(team, route, initial):
    battle, target, original = scenario(team, route, initial=initial)
    target.update(initiative=3, waited=1, bonusturn=2, round_effects_done=1,
                  running_away=0, feared=0)
    before = {key: target[key] for key in TURN_KEYS}
    restore(battle, target, "cleanse")
    assert target["initiative"] == 3
    assert target["initiative_base"] == original["initiative_base"]
    assert {key: target[key] for key in TURN_KEYS} == before


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("cleanup", ["cleanse", "expiry"])
def test_old_zero_snapshot_does_not_cancel_a_later_live_activation(team, route, cleanup):
    battle, target, original = scenario(team, route, initial=0)
    battle._end_round_restore()
    assert target["initiative"] > 0
    restore(battle, target, cleanup)
    assert target["initiative"] == original["initiative_base"] > 0


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("route", ["Ведьма", "Колдунья", "Суккуб", "item_transform", "artifact"])
def test_big_demon_form_cannot_regrant_turn(team, route):
    battle, target, original = scenario(team, route, name="Титан")
    assert target["big"] and target["initiative_base"] == 50
    target["initiative"] = 0
    restore(battle, target, "cleanse")
    assert target["initiative"] == 0
    assert target["initiative_base"] == original["initiative_base"]
    assert target["damage"] == original["damage"]


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("storage", ["deepcopy", "pickle", "legacy_list", "json_list", "partial"])
@pytest.mark.parametrize("live", [0, 44])
def test_saved_and_legacy_snapshots_use_current_activation_state(team, storage, live):
    battle, target, original = scenario(team, "lycanthropy", initial=0)
    if storage in {"legacy_list", "json_list", "partial"}:
        target["basestats"] = target["basestats"][target["position"]]
    if storage == "partial":
        target["basestats"][0].pop("initiative", None)
    elif storage == "json_list":
        target = json.loads(json.dumps(target))
        battle.combined[0] = target
    elif storage == "deepcopy":
        battle = deepcopy(battle)
        target = battle.combined[0]
    elif storage == "pickle":
        battle = pickle.loads(pickle.dumps(battle))
        target = battle.combined[0]
    target["initiative"] = live
    restore(battle, target, "cleanse")
    assert target["initiative"] == (original["initiative_base"] if live else 0)
    assert target["unit_id"] == original["unit_id"]


@pytest.mark.parametrize("live", [0, 17])
def test_missing_snapshot_never_changes_turn_state(live):
    battle, target, _ = scenario("blue", "lycanthropy")
    target.update(initiative=live, basestats={})
    assert not battle._restore_transformed_unit(target)
    assert target["initiative"] == live


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("route", ["Ведьма", "Сущий", "lycanthropy"])
def test_repeated_forms_keep_original_stats_without_regranting_turn(team, route):
    battle, target, original = scenario(team, route)
    battle._apply_witch_effect({}, target)
    target.update(initiative=0, round_effects_done=1, bonusturn=1)
    restore(battle, target, "cleanse")
    assert target["initiative"] == 0
    assert target["initiative_base"] == original["initiative_base"]
    assert target["damage"] == original["damage"]
    assert target["round_effects_done"] == target["bonusturn"] == 1


@pytest.mark.parametrize("team", ["blue", "red"])
def test_voluntary_retreat_flag_is_not_rewound_by_form_restore(team):
    battle, target, _ = scenario(team, "Ведьма")
    target.update(initiative=0, running_away=1, feared=0)
    restore(battle, target, "cleanse")
    assert target["running_away"] == 1
    assert target["initiative"] == 0


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("route", ["Ведьма", "Сущий", "lycanthropy"])
@pytest.mark.parametrize("reviver", ["item", "patriarch", "dead_cleanup"])
def test_revive_or_dead_cleanup_does_not_grant_an_unrequested_activation(team, route, reviver):
    battle, target, original = scenario(team, route)
    target.update(health=0, initiative=0, bonusturn=0)
    if reviver == "item":
        assert battle._apply_hero_item_revive(target, 10) > 0
    elif reviver == "patriarch":
        healer = fighter("Патриарх", team, 10 if team == "blue" else 4)
        assert battle._apply_patriach_support(healer, target) == "revive_success"
    else:
        assert battle._restore_transformed_unit(target)
        assert target["health"] == 0
    assert not target["transformed"]
    assert target["initiative"] == target["bonusturn"] == 0
    assert target["initiative_base"] == original["initiative_base"]


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("slow_first", [False, True])
def test_slow_and_transform_cleanup_cannot_restore_spent_activation(team, slow_first):
    target = fighter("Рыцарь", team, 7 if team == "blue" else 1)
    battle = BattleEnv(log_enabled=False)
    battle.combined = [target]
    caster = {"attack_type_secondary": ""}
    if slow_first:
        assert battle._apply_hermit_initiative_slow(caster, target)
    battle._apply_witch_effect(caster, target)
    if not slow_first:
        assert battle._apply_hermit_initiative_slow(caster, target)
    target["initiative"] = 0
    assert battle._cleanse_negative_effects(target)
    assert target["initiative"] == 0
    assert not target["hermited"] and not target["transformed"]
