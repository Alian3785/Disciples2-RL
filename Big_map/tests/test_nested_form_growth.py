"""Outer battle transforms must unwind before copying/Fenrir and natural XP.

These deterministic lifecycle fixtures do not run training, evaluation or an
autonomous opponent. Escapes use the real saved-recipient XP path.
"""
from copy import deepcopy

import pytest

from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from permanent_unit_stats import add_permanent_effect


TEMPLATES = {unit["name"]: unit for entry in DATA
             for unit in [map_unit_to_battle(entry, "blue", 7)]}
PERSISTENT_FIELDS = (
    "name", "unit_type", "Level", "exp_current", "exp_required", "exp_kill",
    "damage", "damage_secondary", "accuracy", "accuracy_secondary", "armor",
    "initiative_base", "max_health", "health", "maxhp", "hp",
    "attack_type_primary", "attack_type_secondary", "campaign_stat_sources",
)


def fighter(name, elixirs=False):
    unit = deepcopy(TEMPLATES[name])
    unit["original_damage"] = unit["damage"]
    unit["base_armor"] = unit["armor"]
    unit["hp"] = unit["health"]
    unit["maxhp"] = unit["max_health"]
    if elixirs:
        for kind in ("damage", "health", "initiative", "accuracy"):
            add_permanent_effect(unit, {"kind": kind, "multiplier": 1.1})
    unit["exp_current"] = unit["exp_required"] - 1
    return unit


def enter_inner_form(battle, unit, target):
    if unit["name"] == "Двойник":
        assert battle._apply_doppelganger_copy(unit, target)
        return
    # Wolf Lord's transformation lives inline in step(). This is the exact
    # form and original-form snapshot it creates, without advancing combat.
    keys = (
        "name", "unit_type", "damage", "damage_secondary", "initiative",
        "initiative_base", "attack_type_primary", "attack_type_secondary",
        "max_health", "armor", "accuracy", "accuracy_secondary", "immunity",
        "resistance", "big",
    )
    unit["wolflord_base"] = {key: deepcopy(unit[key]) for key in keys}
    unit.update(name="Дух Фенрира", unit_type="Warrior", damage=90,
                initiative=0, initiative_base=65, attack_type_primary="Weapon",
                max_health=275, health=275)


@pytest.mark.parametrize("name", ["Двойник", "Повелитель волков"])
@pytest.mark.parametrize("outer", ["witch", "lycanthropy"])
@pytest.mark.parametrize("route", ["direct_xp", "victory"])
@pytest.mark.parametrize("elixirs", [False, True])
def test_nested_forms_restore_natural_growth(name, outer, route, elixirs):
    expected = fighter(name, elixirs)
    BattleEnv(log_enabled=False)._apply_exp_award_to_unit(expected, 1)

    actual = fighter(name, elixirs)
    target = fighter("Герцог")
    target.update(team="red", position=1)
    battle = BattleEnv(log_enabled=False)
    battle.combined = [actual, target]
    enter_inner_form(battle, actual, target)
    if outer == "witch":
        battle._apply_witch_effect(target, actual)
    else:
        assert battle._apply_hero_item_lycanthropy(
            source_unit=target, target_unit=actual,
            effect={"transform_unit_name": "Оборотень"},
        ) == 1.0
    assert actual["transformed"] == 1

    if route == "direct_xp":
        battle._apply_exp_award_to_unit(actual, 1)
    else:
        battle._battle_exp_tracking_initialized = True
        battle._battle_defeated_exp = {"red": 1.0, "blue": 0.0}
        battle._battle_exp_kills = {"red": [(0, 1.0)], "blue": []}
        target["health"] = target["hp"] = 0
        battle._begin_post_victory_healing("blue")
        assert battle.winner == "blue"

    for field in PERSISTENT_FIELDS:
        assert actual.get(field) == expected.get(field), field
    assert not actual.get("transformed")
    assert not actual.get("doppel_copied")
    assert "wolflord_base" not in actual
    assert not actual.get("basestats")


@pytest.mark.parametrize("effect", ["buff", "debuff"])
def test_repeated_transform_never_backfills_form_only_damage_factors(effect):
    battle = BattleEnv(log_enabled=False)
    actual = fighter("Герцог")
    expected = deepcopy(actual)
    battle._apply_exp_award_to_unit(expected, 1)
    battle._apply_witch_effect({}, actual)
    if effect == "buff":
        battle._apply_hero_item_damage_buff(
            target_unit=actual, effect={"damage_multiplier": 1.75})
    else:
        battle._apply_hero_item_damage_debuff(
            source_unit=None, target_unit=actual,
            effect={"damage_multiplier": 0.5})
    # A second Witch/Hag's Ring hit must keep the first natural-form snapshot.
    battle._apply_witch_effect({}, actual)
    battle._apply_exp_award_to_unit(actual, 1)
    for field in PERSISTENT_FIELDS:
        assert actual.get(field) == expected.get(field), field
    assert not actual.get("_battle_damage_factors")
    assert not actual.get("_battle_lower_damage_factors")
