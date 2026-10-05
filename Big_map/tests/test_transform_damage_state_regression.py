"""Damage-status ownership follows forms without reviving expired buff layers.

Helper-level composition tests complement the public-step lifecycle suite.
Native catalogue fighters, real production effect helpers and exact independent
damage expectations are used; no effect, growth or restoration code is mocked.
"""
from copy import deepcopy
import json
import pickle

import pytest

from attack_damage_limits import attack_damage_cap, effective_attack_damage
from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from permanent_unit_stats import persistent_values, rebuild_stat_layers


CATALOG = {row["кто"]: row for row in DATA}
FORMS = ("witch", "wight", "lycanthropy")
DAMAGE_STATE = (
    "damage", "original_damage", "powerup", "teamated",
    "lower_damage_original_damage", "lower_damage_original_unit_type",
    "_battle_damage_factors", "_battle_lower_damage_factors",
)


def scenario(team="blue", name="Рыцарь"):
    target = map_unit_to_battle(CATALOG[name], team, 7 if team == "blue" else 1)
    # Mirror campaign battle initialization, without scheduling an activation.
    target.update(original_damage=target["damage"], base_armor=target["armor"],
                  powerup=0, teamated=0, initiative=0, immunity=[], resistance=[])
    battle = BattleEnv(log_enabled=False)
    battle.combined = [target]
    return battle, target


def buff(battle, target, factor=2.0):
    assert battle._apply_hero_item_damage_buff(
        target_unit=target, effect={"damage_multiplier": factor}) != 0


def lower(battle, target, factor=0.68):
    assert battle._apply_hero_item_damage_debuff(
        source_unit=None, target_unit=target,
        effect={"damage_multiplier": factor}) >= 0
    assert target["teamated"] == 1


def transform(battle, target, form):
    if form == "witch":
        battle._apply_witch_effect({}, target)
    elif form == "wight":
        caster_team = "red" if target["team"] == "blue" else "blue"
        caster = map_unit_to_battle(CATALOG["Сущий"], caster_team,
                                   1 if caster_team == "red" else 7)
        assert battle._apply_wight_decay_effect(caster, target)
    else:
        assert battle._apply_hero_item_lycanthropy(
            source_unit=None, target_unit=target,
            effect={"transform_unit_name": "Оборотень"}) == 1
    assert target["transformed"] == 1


def snapshot(target):
    states = target["basestats"]
    return (states[target["position"]] if isinstance(states, dict) else states)[0]


def damage_state(target):
    return {key: deepcopy(target[key]) for key in DAMAGE_STATE if key in target}


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("order", [("buff", "lower"), ("lower", "buff")])
@pytest.mark.parametrize("expire", [False, True])
@pytest.mark.parametrize("restore", ["direct", "cleanse"])
def test_original_lower_damage_and_buff_lifecycle(team, form, order, expire, restore):
    battle, target = scenario(team)
    for effect in order:
        (buff if effect == "buff" else lower)(battle, target)
    # Buff replaces an earlier Lower Damage layer; Lower Damage after a buff
    # multiplies the uncapped current characteristic.
    expected_active = 68 if order[0] == "buff" else 100
    assert target["damage"] == expected_active
    transform(battle, target, form)
    assert target["damage"] == target["original_damage"]
    assert not target["powerup"] and not target["teamated"]
    if expire:
        battle._reset_powerup(target)
        expected_active = 34 if order[0] == "buff" else 50
        assert snapshot(target)["damage"] == expected_active
        assert snapshot(target)["lower_damage_original_damage"] == 50
        assert snapshot(target)["_battle_lower_damage_factors"] == []
    if restore == "direct":
        assert battle._restore_transformed_unit(target)
        assert target["damage"] == expected_active
        assert target["teamated"] == 1
    else:
        assert battle._cleanse_negative_effects(target)
        assert target["damage"] == (50 if expire or order[0] == "lower" else 100)
        assert target["teamated"] == 0
    assert target["original_damage"] == 50
    assert target["powerup"] == int(not expire)
    assert target["initiative"] == 0
    # Repeated cleanup cannot bring the old buff back through either snapshot.
    battle._reset_powerup(target)
    battle._cleanse_negative_effects(target)
    assert target["damage"] == 50
    assert not target["powerup"] and not target["teamated"]
    assert target["_battle_damage_factors"] == []


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("expire", [False, True])
def test_original_and_temporary_form_have_independent_lower_damage_state(team, form, expire):
    battle, target = scenario(team)
    buff(battle, target)
    lower(battle, target, 0.5)
    transform(battle, target, form)
    native_form_damage = target["damage"]
    buff(battle, target, 1.75)
    lower(battle, target)
    assert snapshot(target)["damage"] == 50
    assert snapshot(target)["lower_damage_original_damage"] == 100
    assert target["lower_damage_original_damage"] == round(native_form_damage * 1.75)
    if expire:
        battle._reset_powerup(target)
        assert target["damage"] == round(native_form_damage * 0.68)
        assert target["lower_damage_original_damage"] == native_form_damage
        assert snapshot(target)["damage"] == 25
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == (25 if expire else 50)
    assert target["lower_damage_original_damage"] == (50 if expire else 100)
    assert target["_battle_damage_factors"] == ([0.5] if expire else [2.0, 0.5])
    assert target["_battle_lower_damage_factors"] == ([] if expire else [2.0])
    assert battle._cleanse_negative_effects(target)
    assert target["damage"] == (50 if expire else 100)


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("form", FORMS)
def test_temporary_lower_damage_cannot_replace_original_damage_on_cure(team, form):
    battle, target = scenario(team)
    transform(battle, target, form)
    # Knight and Imp share Warrior: broad unit_type is not form identity.
    caster_team = "red" if team == "blue" else "blue"
    caster = map_unit_to_battle(CATALOG["Тиамат"], caster_team,
                               1 if caster_team == "red" else 7)
    caster["attack_type_secondary"] = ""
    assert battle._apply_tiamat_damage_debuff(caster, target)
    assert battle._cleanse_negative_effects(target)
    assert target["damage"] == target["original_damage"] == 50
    assert not target["teamated"]
    assert "lower_damage_original_damage" not in target
    assert "_battle_lower_damage_factors" not in target


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("form", ["witch", "wight"])
@pytest.mark.parametrize("expire", [False, True])
def test_repeated_transform_never_backfills_later_damage_status(team, form, expire):
    battle, target = scenario(team, "Старейшина")
    buff(battle, target, 1.25)
    transform(battle, target, form)
    original = damage_state(snapshot(target))
    for _ in range(2):
        buff(battle, target, 2.0)
        lower(battle, target)
        transform(battle, target, form)
        assert damage_state(snapshot(target)) == original
        assert target["damage"] == target["original_damage"]
        assert not target["powerup"] and not target["teamated"]
    if expire:
        battle._reset_powerup(target)
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == (80 if expire else 100)
    assert target["original_damage"] == 80
    assert target["powerup"] == int(not expire)
    assert target["initiative"] == 0


@pytest.mark.parametrize("storage", ["list", "json_list", "pickle", "deepcopy"])
@pytest.mark.parametrize("expire", [False, True])
def test_serialized_and_legacy_list_snapshots_preserve_damage_lifetime(storage, expire):
    battle, target = scenario()
    buff(battle, target)
    lower(battle, target)
    transform(battle, target, "witch")
    if storage in {"list", "json_list"}:
        target["basestats"] = target["basestats"][target["position"]]
    if storage == "json_list":
        target = json.loads(json.dumps(target))
    elif storage == "pickle":
        target = pickle.loads(pickle.dumps(target))
    else:
        target = deepcopy(target)
    battle.combined = [target]
    buff(battle, target, 1.5)
    lower(battle, target, 0.5)
    if expire:
        battle._reset_powerup(target)
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == (34 if expire else 68)
    assert target["powerup"] == int(not expire)
    assert target["lower_damage_original_damage"] == (50 if expire else 100)
    assert battle._cleanse_negative_effects(target)
    assert target["damage"] == (50 if expire else 100)


def test_partial_legacy_snapshot_does_not_adopt_form_only_damage_metadata():
    battle, target = scenario()
    transform(battle, target, "witch")
    original = snapshot(target)
    for key in DAMAGE_STATE[2:]:
        original.pop(key, None)
    buff(battle, target)
    lower(battle, target)
    transform(battle, target, "witch")
    assert not any(key in original for key in DAMAGE_STATE[2:])
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == target["original_damage"] == 50
    assert not any(key in target for key in DAMAGE_STATE[2:])


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("storage", ["list", "dict"])
def test_legacy_snapshot_without_baseline_uses_saved_original_form_damage(repeat, storage):
    battle, target = scenario()
    transform(battle, target, "witch")
    original = snapshot(target)
    # Default-mode snapshots before this fix had no original_damage, powerup
    # or teamated. Their saved ordinary damage is still an exact baseline.
    for key in ("original_damage", "powerup", "teamated"):
        original.pop(key, None)
    if storage == "list":
        target["basestats"] = [original]
    buff(battle, target)
    if repeat:
        transform(battle, target, "witch")
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == target["original_damage"] == 50
    buff(battle, target)
    assert target["damage"] == 100
    battle._reset_powerup(target)
    assert target["damage"] == 50


@pytest.mark.parametrize("form", ["witch", "lycanthropy"])
def test_default_battle_without_original_damage_keeps_permanent_buff_baseline(form):
    battle = BattleEnv(log_enabled=False)
    battle.reset(seed=42)
    target = next(unit for unit in battle.combined
                  if unit["name"] == "Скелет рыцарь")
    target.pop("original_damage", None)
    assert target["damage"] == 100
    transform(battle, target, form)
    transform(battle, target, "witch")
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == target["original_damage"] == 100
    buff(battle, target)
    assert target["damage"] == 200
    battle._reset_powerup(target)
    assert target["damage"] == 100


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("expire", [False, True])
def test_native_support_sets_missing_baseline_before_snapshot(team, expire):
    battle, target = scenario(team)
    target.pop("original_damage")
    healer = map_unit_to_battle(CATALOG["Травница"], team,
                               10 if team == "blue" else 4)
    assert battle._apply_travnitsa_buff(healer, target)
    assert target["damage"] == 62
    assert target["original_damage"] == 50
    transform(battle, target, "witch")
    if expire:
        battle._reset_powerup(target)
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == (50 if expire else 62)
    assert target["original_damage"] == 50
    battle._reset_powerup(target)
    assert target["damage"] == 50


@pytest.mark.parametrize("expire", [False, True])
@pytest.mark.parametrize("lowered", [False, True])
def test_growth_after_transformation_replays_only_unexpired_original_layers(expire, lowered):
    battle, target = scenario(name="Герцог")
    target.update(Level=4, exp_current=target["exp_required"] - 1)
    control = deepcopy(target)
    battle._apply_exp_award_to_unit(control, 1)
    buff(battle, target)
    if lowered:
        lower(battle, target)
    transform(battle, target, "witch")
    buff(battle, target, 1.75)
    if expire:
        battle._reset_powerup(target)
    battle._apply_exp_award_to_unit(target, 1)
    expected = control["damage"]
    if not expire:
        expected = round(expected * 2.0)
    if lowered:
        expected = round(expected * 0.68)
    assert target["damage"] == expected
    assert target["original_damage"] == control["original_damage"]
    assert target["powerup"] == int(not expire)
    assert persistent_values(target) == persistent_values(control)
    assert target["initiative"] == 0
    restored = json.loads(json.dumps(target))
    rebuild_stat_layers(restored)
    assert restored["damage"] == expected


@pytest.mark.parametrize("expire", [False, True])
def test_transformed_buff_expiry_keeps_uncapped_raw_damage_and_form_caps(expire):
    battle, target = scenario(name="Герцог")
    target.update(damage=450, original_damage=450)
    buff(battle, target)
    assert target["damage"] == 900 and effective_attack_damage(target) == 400
    transform(battle, target, "witch")
    assert attack_damage_cap(target) == 300
    buff(battle, target, 25.0)
    assert target["damage"] == 500 and effective_attack_damage(target) == 300
    if expire:
        battle._reset_powerup(target)
        assert target["damage"] == target["original_damage"] == 20
    assert battle._restore_transformed_unit(target)
    assert target["damage"] == (450 if expire else 900)
    assert target["original_damage"] == 450
    assert attack_damage_cap(target) == effective_attack_damage(target) == 400
