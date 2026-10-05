"""Healer ward provenance follows transformations, copies and battle endings."""

import pytest
from battle_env import BattleEnv, DEFEND_ACTION_INDEX, TARGET_POSITIONS
from data_dicts_compact_lines import DATA, map_unit_to_battle

DATA_BY_NAME = {u["кто"]: u for u in DATA}


def unit(name, team="blue", position=7):
    return map_unit_to_battle(DATA_BY_NAME[name], team, position)


def setup(native=False, used=False):
    b = BattleEnv(log_enabled=False)
    healer = unit("Солнечная танцовщица", position=10)
    target = unit("Скваер")
    target.update(
        resistance=["Fire"] if native else [],
        resilience_used_types=["Fire"] if used else [],
    )
    enemy = unit("Ведьма", "red", 1)
    b.combined = [healer, target, enemy]
    return b, healer, target, enemy


def complete(b):
    occupied = {u["position"] for u in b.combined}
    b.combined.extend(
        b._empty_battle_slot("red" if p < 7 else "blue", p)
        for p in range(1, 13)
        if p not in occupied
    )


def block(b, target):
    return b._resilience_blocks({"attack_type_primary": "Fire"}, target)


def expire(b, healer):
    b._apply_start_of_turn_effects(healer)


def clean(target):
    assert not target.get("_temporary_healer_wards")
    assert not target.get("Firedefence")


@pytest.mark.parametrize("native,used", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("hit", [False, True])
def test_original_ward_expires_in_witch_snapshot(native, used, hit):
    b, h, t, e = setup(native, used)
    b._apply_cliric_heal(h, t)
    if hit:
        assert block(b, t)
    b._apply_witch_effect(e, t)
    expire(b, h)
    assert b._restore_transformed_unit(t)
    assert t["resistance"] == (["Fire"] if native else [])
    assert block(b, t) == (native and not (used or hit))
    clean(t)


@pytest.mark.parametrize("twice", [False, True])
@pytest.mark.parametrize("expire_before_restore", [False, True])
def test_later_form_ward_never_backfills_original(twice, expire_before_restore):
    b, h, t, e = setup()
    b._apply_witch_effect(e, t)
    b._apply_cliric_heal(h, t)
    if twice:
        b._apply_witch_effect(e, t)
    if expire_before_restore:
        expire(b, h)
    assert b._restore_transformed_unit(t)
    assert t["resistance"] == []
    assert not block(b, t)
    clean(t)


def test_wards_expire_when_caster_has_changed_form():
    b, h, t, e = setup()
    b._apply_cliric_heal(h, t)
    b._apply_witch_effect(e, h)
    expire(b, h)
    assert t["resistance"] == []
    clean(t)


def test_manual_status_consumption_sticks_across_recast_and_expiry():
    b, h, t, e = setup(native=True)
    b._apply_cliric_heal(h, t)
    e["attack_type_primary"] = "Fire"
    assert not b._apply_paralysis_effect(e, t)
    assert t["resilience_used_types"] == ["Fire"]
    b._apply_cliric_heal(h, t)
    expire(b, h)
    assert not block(b, t)
    assert t["resistance"] == ["Fire"]


@pytest.mark.parametrize("native", [False, True])
def test_doppel_copied_ward_keeps_cross_team_source(native):
    b, h, t, e = setup(native)
    d = unit("Двойник", "red", 2)
    b.combined.append(d)
    b._apply_cliric_heal(h, t)
    assert b._apply_doppelganger_copy(d, t)
    assert block(b, d)
    expire(b, h)
    assert d["resistance"] == (["Fire"] if native else [])
    assert not block(b, d)
    clean(d)


@pytest.mark.parametrize("native", [False, True])
def test_wolf_actual_transform_does_not_restore_expired_ward(native):
    b, h, t, e = setup(native)
    wolf = unit("Повелитель волков")
    wolf.update(resistance=["Fire"] if native else [], resilience_used_types=[])
    b.combined = [wolf, h, e]
    b._apply_cliric_heal(h, wolf)
    b.current_blue_attacker_pos = 7
    b.blue_attacks_left = 1
    b._advance_until_blue_turn = lambda: None
    complete(b)
    b.step(TARGET_POSITIONS.index(7))
    assert wolf["name"] == "Дух Фенрира"
    assert block(b, wolf)
    expire(b, h)
    b._revert_fenrir_survivors()
    assert wolf["name"] == "Повелитель волков"
    assert wolf["resistance"] == (["Fire"] if native else [])
    assert not block(b, wolf)
    clean(wolf)


def test_actual_timeout_branch_cleans_live_and_snapshot_wards():
    b, h, t, e = setup()
    b._apply_cliric_heal(h, t)
    b._apply_witch_effect(e, t)
    b.current_blue_attacker_pos = 7
    b.blue_attacks_left = 1
    b.step_count = 999
    b._advance_until_blue_turn = lambda: None
    complete(b)
    _, _, terminated, truncated, _ = b.step(DEFEND_ACTION_INDEX)
    assert not terminated and truncated
    b._restore_transformed_unit(t)
    assert t["resistance"] == []
    clean(t)


@pytest.mark.parametrize("team", ["blue", "red"])
def test_real_post_victory_heal_cleans_newly_granted_wards(team):
    b = BattleEnv(log_enabled=False)
    h = unit("Дева рощи", team, 10 if team == "blue" else 4)
    t = unit("Скваер", team, 7 if team == "blue" else 1)
    t.update(health=20, resistance=["Fire"])
    e = unit("Скваер", "red" if team == "blue" else "blue", 1 if team == "blue" else 7)
    e["health"] = 0
    b.combined = [h, t, e]
    b._begin_post_victory_healing(team)
    complete(b)
    if team == "blue":
        b.step(0)
    assert b.winner == team
    assert t["health"] > 20
    assert t["resistance"] == ["Fire"]
    clean(t)
