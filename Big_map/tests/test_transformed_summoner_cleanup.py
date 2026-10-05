"""Dependent summons retain their actual owner across form and life changes.

The public reproduction keeps native catalog stats; narrower lifecycle cases use
real creation/replacement paths and explicit damage to isolate ownership.
"""
from copy import deepcopy
import json
import pickle
import random

import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX, TARGET_POSITIONS
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit

CATALOG = {entry["кто"]: entry for entry in DATA}
NATIVE_SUMMONERS = ["Оккультист", "Мастер оккультист", "Тёмный эльф Лиф", "Тёмный Лаклаан"]


class FixedRng(random.Random):
    def random(self):
        return 0.0

    def randint(self, low, high):
        return low

    def shuffle(self, values):
        pass


def native(name, team, position):
    return map_unit_to_battle(CATALOG[name], team, position)


def battle_with(*units, initialize=False):
    battle = BattleEnv(log_enabled=True)
    battle.rng = FixedRng(0)
    positions = {unit["position"] for unit in units}
    roster = list(units) + [placeholder_unit("red" if p < 7 else "blue", p)
                            for p in range(1, 13) if p not in positions]
    if initialize:
        battle._init_with_custom_teams([u for u in roster if u["team"] == "red"],
                                       [u for u in roster if u["team"] == "blue"])
    else:
        battle.combined = roster
        battle._patriach_revived_recipients = set()
    return battle


def step_to(battle, position):
    action = TARGET_POSITIONS.index(position)
    assert battle.compute_action_mask()[action]
    return battle.step(action)


def point_case(team):
    own, enemy = (7, 1) if team == "blue" else (1, 7)
    other = "red" if team == "blue" else "blue"
    owner = native("Оккультист", team, own)
    killer = native("Кентавр", other, enemy)
    witch = native("Ведьма", other, enemy + 1)
    battle = battle_with(owner, killer, witch)
    assert battle._attack(owner, enemy + 1) == (True, "ok")
    summon = battle._unit_by_position(own + 1)
    assert summon["Summoned"] == own and summon["health"] == 170
    return battle, owner, summon, killer, witch


def kill(battle, target):
    assert battle._apply_hero_item_damage(source_unit=None, target_unit=target,
                                         effect={"amount": 100000}) > 0
    assert target["health"] == 0


def item_summon(battle, source, position, name="Зомби"):
    target = battle._unit_by_position(position)
    assert battle._apply_hero_item_summon(source_unit=source, target_team=source["team"],
                                         target_pos=position, target_unit=target,
                                         effect={"summon_unit_name": name}) == 1
    return target


@pytest.mark.parametrize("transformed", [False, True])
def test_public_centaur_witch_occultist_battle_ends_without_orphan_attack(transformed):
    battle = battle_with(native("Оккультист", "red", 1), native("Кентавр", "blue", 7),
                         native("Ведьма", "blue", 10), initialize=True)
    assert battle.current_blue_attacker_pos == 7
    step_to(battle, 1)
    owner, summon = battle._unit_by_position(1), battle._unit_by_position(2)
    assert (owner["health"], summon["health"], battle.current_blue_attacker_pos) == (35, 170, 10)
    if transformed:
        step_to(battle, 1)
        assert owner["unit_type"] == "Warrior" and owner["transformed"] == 1
    else:
        battle.step(DEFEND_ACTION_INDEX)
        assert owner["unit_type"] == "Summoner"
    assert (battle.round_no, battle.current_blue_attacker_pos) == (2, 7)
    result = step_to(battle, 1)
    assert (owner["health"], summon["health"], battle._unit_by_position(7)["health"]) == (0, 0, 140)
    assert battle.winner == "blue" and result[2] and not result[3]


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("form", ["native", "witch", "lycanthropy", "restored"])
@pytest.mark.parametrize("death", ["direct", "aoe", "item", "poison", "burn", "uran"])
def test_owner_death_cleans_summon_for_all_damage_paths_and_forms(team, form, death):
    battle, owner, summon, killer, witch = point_case(team)
    if form in ("witch", "restored"):
        assert battle._attack(witch, owner["position"]) == (True, "ok")
        if form == "restored":
            assert battle._restore_transformed_unit(owner)
    elif form == "lycanthropy":
        assert battle._apply_hero_item_lycanthropy(source_unit=witch, target_unit=owner,
                                                  effect={"transform_unit_name": "Скваер"}) == 1
    summon["hp"] = summon["health"]  # Both HP projections must be cleared.
    summon["initiative"] = 50
    if death in ("direct", "aoe"):
        owner["health"] = 1
        if death == "aoe":
            killer["unit_type"] = "Mage"
            killer["damage"] = 1  # Cannot kill the healthy summon by collateral damage.
        assert battle._attack(killer, owner["position"])[0]
    elif death == "item":
        kill(battle, owner)
    else:
        owner[f"{death}_turns_left"] = 1
        owner[f"{death}_damage_per_tick"] = 100000
        assert not battle._apply_start_of_turn_effects(owner)
    assert owner["health"] == 0
    assert (summon["health"], summon["hp"], summon["initiative"]) == (0, 0, 0)


@pytest.mark.parametrize("team", ["red", "blue"])
@pytest.mark.parametrize("name", NATIVE_SUMMONERS)
def test_all_native_summon_creation_paths_survive_caster_form_change(team, name):
    own, enemy = (7, 1) if team == "blue" else (1, 7)
    owner = native(name, team, own)
    battle = battle_with(owner)
    assert battle._attack(owner, enemy + 1)[0]
    summons = [u for u in battle.combined if u.get("Summoned") == own]
    assert summons and all(battle._alive(u) for u in summons)
    # Isolate ownership, including Mind-immune bosses, from status eligibility.
    battle._apply_witch_effect(owner, owner)
    kill(battle, owner)
    assert all(u["health"] == 0 and u["initiative"] == 0 for u in summons)


@pytest.mark.parametrize("team", ["red", "blue"])
def test_only_actual_owner_summons_die_among_multiple_casters(team):
    battle, owner, first, killer, witch = point_case(team)
    own, enemy = owner["position"], killer["position"]
    second_owner = battle._unit_by_position(own + 2)
    second_owner.clear()
    second_owner.update(native("Оккультист", team, own + 2))
    assert battle._attack(second_owner, enemy + 3)[0]
    second = battle._unit_by_position(own + 3)
    enemy_owner = witch
    enemy_owner.clear()
    enemy_owner.update(native("Оккультист", killer["team"], enemy + 1))
    assert battle._attack(enemy_owner, own + 2)[0]
    enemy_summon = battle._unit_by_position(enemy + 2)
    battle._apply_witch_effect(killer, owner)
    kill(battle, owner)
    assert first["health"] == 0
    assert battle._alive(second_owner) and battle._alive(second) and battle._alive(enemy_summon)
    # Repeated cleanup cannot record the same dependent defeat twice.
    events = battle._battle_exp_event_count
    battle._kill_linked_summons(owner)
    assert battle._battle_exp_event_count == events


@pytest.mark.parametrize("revival", ["item", "patriarch"])
def test_resurrected_caster_keeps_ownership_of_revived_and_new_summons(revival):
    battle, owner, summon, killer, witch = point_case("blue")
    battle._apply_witch_effect(witch, owner)
    kill(battle, owner)
    assert summon["health"] == 0
    if revival == "item":
        assert battle._apply_hero_item_revive(owner, 50) > 0
    else:
        healer = native("Патриарх", "blue", 10)
        assert battle._apply_patriach_support(healer, owner) == "revive_success"
    assert owner["unit_type"] == "Summoner"
    assert battle._apply_hero_item_revive(summon, 100) > 0
    assert battle._attack(owner, 3)[0]
    fresh = battle._unit_by_position(9)
    kill(battle, owner)
    assert summon["health"] == fresh["health"] == 0


@pytest.mark.parametrize("serializer", ["deepcopy", "pickle"])
def test_copied_battle_ownership_points_to_copied_units(serializer):
    battle, owner, summon, killer, witch = point_case("blue")
    battle._apply_witch_effect(witch, owner)
    copied = deepcopy(battle) if serializer == "deepcopy" else pickle.loads(pickle.dumps(battle))
    kill(copied, copied._unit_by_position(7))
    assert copied._unit_by_position(8)["health"] == 0
    assert battle._alive(owner) and battle._alive(summon)
    # Internal ownership must not introduce cycles or non-JSON state into units.
    json.dumps(battle.combined)


@pytest.mark.parametrize("name", ["Кентавр", *NATIVE_SUMMONERS])
@pytest.mark.parametrize("transformed", [False, True])
def test_item_summon_preserves_creation_time_lifespan(name, transformed):
    owner = native(name, "blue", 7)
    battle = battle_with(owner)
    summon = item_summon(battle, owner, 8)
    if transformed:
        battle._apply_witch_effect(owner, owner)
    kill(battle, owner)
    assert (summon["health"] == 0) is (name in NATIVE_SUMMONERS)


def test_item_summon_from_non_summoner_stays_independent_after_becoming_summoner():
    owner = native("Кентавр", "blue", 7)
    battle = battle_with(owner)
    independent = item_summon(battle, owner, 8)
    assert battle._apply_hero_item_lycanthropy(source_unit=None, target_unit=owner,
                                              effect={"transform_unit_name": "Оккультист"}) == 1
    assert battle._attack(owner, 3)[0]
    dependent = battle._unit_by_position(9)
    kill(battle, owner)
    assert battle._alive(independent) and dependent["health"] == 0


@pytest.mark.parametrize("replacement", ["point", "row", "item"])
@pytest.mark.parametrize("victim", ["replacement", "new_owner"])
def test_reused_caster_dictionary_drops_old_links_and_keeps_new_links(replacement, victim):
    battle, owner, old_summon, killer, witch = point_case("blue")
    kill(battle, owner)
    assert old_summon["health"] == 0
    assert battle._apply_hero_item_revive(old_summon, 100) > 0
    source = battle._unit_by_position(9)
    source.clear()
    source.update(native("Тёмный эльф Лиф" if replacement == "row" else "Оккультист", "blue", 9))
    if replacement == "item":
        new_unit = item_summon(battle, source, 7)
    else:
        assert battle._attack(source, 1)[0]
        new_unit = battle._unit_by_position(7)
    assert new_unit is owner and battle._alive(new_unit), "exercise actual dictionary reuse"
    if victim == "replacement":
        # A replacement that later becomes a summoner still has no ownership of
        # its predecessor's revived dependent, even in the identical slot.
        assert battle._apply_hero_item_lycanthropy(source_unit=None, target_unit=new_unit,
                                                  effect={"transform_unit_name": "Оккультист"}) == 1
        kill(battle, new_unit)
    else:
        kill(battle, source)
        assert new_unit["health"] == 0, "replacement's new incoming owner link was lost"
    assert battle._alive(old_summon)


@pytest.mark.parametrize("team", ["red", "blue"])
def test_escaped_caster_slot_reuse_cannot_transfer_ownership(team):
    battle, owner, old_summon, killer, witch = point_case(team)
    own, enemy = owner["position"], killer["position"]
    battle._mark_unit_escaped(owner)
    assert battle._alive(old_summon), "escape does not change native summon lifespan"
    source = battle._unit_by_position(own + 2)
    source.clear()
    source.update(native("Оккультист", team, own + 2))
    assert battle._attack(source, enemy)[0]
    replacement = battle._unit_by_position(own)
    assert replacement is owner
    assert battle._apply_hero_item_lycanthropy(source_unit=None, target_unit=replacement,
                                              effect={"transform_unit_name": "Оккультист"}) == 1
    assert battle._attack(replacement, enemy + 3)[0]
    fresh = battle._unit_by_position(own + 3)
    kill(battle, replacement)
    assert fresh["health"] == 0 and battle._alive(old_summon)
    assert battle._alive(battle.escaped_units[0])


@pytest.mark.parametrize("creation", ["point", "item"])
def test_summoned_transformed_caster_death_cascades_through_dependents(creation):
    battle, owner, child, killer, witch = point_case("blue")
    assert battle._apply_hero_item_lycanthropy(source_unit=None, target_unit=child,
                                              effect={"transform_unit_name": "Оккультист"}) == 1
    if creation == "point":
        assert battle._attack(child, 3)[0]
        grandchild = battle._unit_by_position(9)
    else:
        grandchild = item_summon(battle, child, 9)
    # Reversion restores Warrior, while the child still owns its dependent.
    assert battle._restore_transformed_unit(child)
    assert child["unit_type"] == "Warrior"
    kill(battle, owner)
    assert child["health"] == grandchild["health"] == 0
    events = battle._battle_exp_event_count
    battle._kill_linked_summons(owner)
    assert battle._battle_exp_event_count == events


@pytest.mark.parametrize("replacement", ["point", "row", "item"])
def test_reused_summon_dictionary_does_not_keep_its_old_owner(replacement):
    battle, owner, old_summon, killer, witch = point_case("blue")
    kill(battle, old_summon)
    source = battle._unit_by_position(9)
    source.clear()
    source.update(native("Тёмный эльф Лиф" if replacement == "row" else "Оккультист", "blue", 9))
    if replacement == "item":
        new_unit = item_summon(battle, source, 8)
    else:
        assert battle._attack(source, 2)[0]
        new_unit = battle._unit_by_position(8)
    assert new_unit is old_summon and battle._alive(new_unit)
    kill(battle, owner)
    assert battle._alive(new_unit)
    kill(battle, source)
    assert new_unit["health"] == 0


@pytest.mark.parametrize("reset_path", ["default", "custom"])
def test_battle_reset_discards_old_links_and_new_summons_register(reset_path, monkeypatch):
    battle, old_owner, old_summon, killer, witch = point_case("blue")
    red = [native("Кентавр", "red", 1)]
    blue = [native("Оккультист", "blue", 7)]
    if reset_path == "default":
        monkeypatch.setattr("battle_env.UNITS_RED", red)
        monkeypatch.setattr("battle_env.UNITS_BLUE", blue)
        battle._reset_state()
    else:
        monkeypatch.setattr(battle, "_advance_until_blue_turn", lambda: True)
        battle._init_with_custom_teams(red, blue)
    owner = battle._unit_by_position(7)
    assert battle._attack(owner, 2)[0]
    fresh = battle._unit_by_position(8)
    battle._kill_linked_summons(old_owner)
    assert battle._alive(fresh) and battle._alive(old_summon)
    kill(battle, owner)
    assert fresh["health"] == 0 and battle._alive(old_summon)
    json.dumps(battle.combined)
