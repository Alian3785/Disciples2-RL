"""Attack-form identity must unwind without leaking across campaign battles."""
from copy import deepcopy

import pytest

import battle_env as battle_module
from attack_damage_limits import ATTACK_FORM_ID_KEY, attack_damage_cap
from battle_env import BattleEnv, _build_default_battle_unit
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from unit_dynamic_xp import dynamic_unit_id


CATALOG = {entry["кто"]: entry for entry in DATA}
HEAVY_IDS = {
    "g000uu0019", "g000uu0020", "g000uu0044", "g000uu0045", "g000uu0047",
    "g000uu0070", "g000uu0071", "g000uu0096", "g000uu8009", "g000uu8011",
}
HEAVY_NAMES = tuple(entry["кто"] for entry in DATA
                    if dynamic_unit_id(entry) in HEAVY_IDS)


def fighter(name, team="blue", position=None):
    if position is None:
        position = 7 if team == "blue" else 1
    unit = map_unit_to_battle(CATALOG[name], team, position)
    unit.update(original_damage=unit["damage"], hp=unit["health"],
                maxhp=unit["max_health"])
    return unit


def transform(battle, target, kind):
    if kind == "witch":
        battle._apply_witch_effect({}, target)
    else:
        assert battle._apply_hero_item_lycanthropy(
            source_unit=None, target_unit=target,
            effect={"transform_unit_name": "Оборотень"},
        ) == 1.0


@pytest.mark.parametrize("team", ["blue", "red"])
@pytest.mark.parametrize("target_name,expected_cap", [("Герцог", 400), ("Лучник", 300)])
@pytest.mark.parametrize("outer", ["witch", "lycanthropy"])
def test_nested_copy_restores_inner_form_then_native_identity(team, target_name, expected_cap, outer):
    battle = BattleEnv(log_enabled=False)
    copier = fighter("Двойник", team)
    target = fighter(target_name, "red" if team == "blue" else "blue")
    native_id = copier["unit_id"]
    assert battle._apply_doppelganger_copy(copier, target)
    assert attack_damage_cap(copier) == expected_cap
    assert copier["unit_id"] == native_id

    transform(battle, copier, outer)
    assert attack_damage_cap(copier) == 300
    # Reapplying a transformation must not backfill the outer form's identity.
    battle._apply_witch_effect({}, copier)
    assert battle._restore_transformed_unit(copier)
    assert attack_damage_cap(copier) == expected_cap
    battle._restore_default_doppelgangers([copier])
    assert attack_damage_cap(copier) == 300
    assert copier["unit_id"] == native_id
    assert ATTACK_FORM_ID_KEY not in copier


@pytest.mark.parametrize("outer", ["witch", "lycanthropy"])
@pytest.mark.parametrize("dies", [False, True])
def test_native_heavy_identity_returns_after_transformation_and_revive(outer, dies):
    battle = BattleEnv(log_enabled=False)
    hero = fighter("Герцог")
    native_id = hero["unit_id"]
    transform(battle, hero, outer)
    assert attack_damage_cap(hero) == 300
    if dies:
        hero["health"] = hero["hp"] = 0
        assert battle._apply_hero_item_revive(hero, 10) == 10
    else:
        assert battle._restore_transformed_unit(hero)
    assert hero["unit_id"] == native_id
    assert attack_damage_cap(hero) == 400
    assert not hero.get("transformed")


@pytest.mark.parametrize("name", HEAVY_NAMES)
def test_natural_template_does_not_install_active_form_override(name):
    unit = _build_default_battle_unit(name, "blue", 7)
    assert ATTACK_FORM_ID_KEY not in unit
    assert attack_damage_cap(unit) == 400
    # Import overrides are authoritative; a template marker cannot shadow them.
    unit["unit_id"] = "g000uu0001"
    assert attack_damage_cap(unit) == 300


@pytest.mark.parametrize("route", ["default_reset", "custom_teams"])
def test_fresh_battle_discards_expired_copy_marker(monkeypatch, route):
    battle = BattleEnv(log_enabled=False)
    copier, hero = fighter("Двойник"), fighter("Герцог", "red")
    assert battle._apply_doppelganger_copy(copier, hero)
    assert attack_damage_cap(copier) == 400
    if route == "default_reset":
        monkeypatch.setattr(battle_module, "UNITS_BLUE", [copier])
        monkeypatch.setattr(battle_module, "UNITS_RED", [hero])
        battle._reset_state()
    else:
        battle._init_with_custom_teams([hero], [copier])
    reset = battle._unit_by_position(7)
    assert not reset["doppel_copied"]
    assert ATTACK_FORM_ID_KEY not in reset
    assert attack_damage_cap(reset) == 300
    assert attack_damage_cap(battle._unit_by_position(1)) == 400


@pytest.fixture
def campaign():
    env = CampaignEnv(
        map_name="trade_train", observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    env.reset(seed=42)
    yield env
    env.close()


def test_non_victory_campaign_save_drops_copied_heavy_cap(campaign):
    copier, hero = fighter("Двойник"), fighter("Герцог", "red")
    campaign.blue_team_state = campaign._build_battle_team_with_placeholders("blue", [copier])
    campaign._mark_equipment_dirty()
    battle = BattleEnv(log_enabled=False)
    battle.combined = [deepcopy(copier), hero]
    campaign.battle_env = battle
    assert battle._apply_doppelganger_copy(battle.combined[0], hero)
    # The timeout/unfinished save path has not run victory's form restoration.
    assert battle.winner is None
    campaign._save_blue_state()
    saved = next(unit for unit in campaign.blue_team_state if unit["position"] == 7)
    assert ATTACK_FORM_ID_KEY not in saved
    assert attack_damage_cap(saved) == 300
    assert saved["damage"] == copier["damage"]
    battle._init_with_custom_teams([hero], campaign.blue_team_state)
    assert attack_damage_cap(battle._unit_by_position(7)) == 300


@pytest.mark.parametrize("name,expected_cap,stale_id", [
    ("Двойник", 300, "g000uu0070"), ("Герцог", 400, "g000uu0001"),
])
def test_enemy_save_discards_legacy_marker_and_restores_native_cap(campaign, name, expected_cap, stale_id):
    original = fighter(name, "red")
    original[ATTACK_FORM_ID_KEY] = stale_id
    enemy_id = 7701
    campaign.enemy_team_states[enemy_id] = [deepcopy(original)]
    battle = BattleEnv(log_enabled=False)
    battle._init_with_custom_teams([original], [])
    runtime = battle._unit_by_position(1)
    transform(battle, runtime, "witch")
    campaign.battle_env = battle
    campaign._save_enemy_state_from_battle(enemy_id)
    saved = next(unit for unit in campaign.enemy_team_states[enemy_id]
                 if unit["position"] == 1)
    assert saved["unit_id"] == original["unit_id"]
    assert ATTACK_FORM_ID_KEY not in saved
    assert attack_damage_cap(saved) == expected_cap
    battle._init_with_custom_teams(campaign.enemy_team_states[enemy_id], [])
    assert attack_damage_cap(battle._unit_by_position(1)) == expected_cap


@pytest.mark.parametrize("dead", [False, True])
def test_bot_and_garrison_save_normalizer_discards_form_marker(campaign, dead):
    battle = BattleEnv(log_enabled=False)
    copier, hero = fighter("Двойник"), fighter("Герцог", "red")
    assert battle._apply_doppelganger_copy(copier, hero)
    if dead:
        copier["health"] = copier["hp"] = 0
    saved = campaign._normalize_scripted_bot_saved_unit(copier)
    assert ATTACK_FORM_ID_KEY not in saved
    assert attack_damage_cap(saved) == 300
    assert saved["damage"] == CATALOG["Двойник"]["урон"]
    assert (saved["health"] == 0) == dead


def test_dead_copy_default_cleanup_discards_heavy_marker():
    battle = BattleEnv(log_enabled=False)
    copier = fighter("Двойник")
    assert battle._apply_doppelganger_copy(copier, fighter("Герцог", "red"))
    copier["health"] = copier["hp"] = 0
    battle._restore_default_doppelgangers([copier])
    assert ATTACK_FORM_ID_KEY not in copier
    assert not copier["doppel_copied"]
    assert attack_damage_cap(copier) == 300
    assert copier["health"] == 0
