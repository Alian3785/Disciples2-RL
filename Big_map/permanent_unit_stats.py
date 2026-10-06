"""Permanent growth and elixirs, independent of temporary battle snapshots.

Every actual level/skill mutation enters this gateway, with or without elixirs.
Legacy units keep recoverable bare stats: lost history is never guessed.
Order: intrinsic (type/level/skills), each permanent dose, then the existing
battle layers (tile, temporary potion, map spell, artifacts, banner).
"""
from copy import deepcopy
from functools import wraps

SOURCE_KEY = "campaign_stat_sources"
STATS = ("damage", "damage_secondary", "accuracy", "accuracy_secondary",
         "initiative", "armor", "max_health")
ALIASES = {"initiative": "initiative_base"}
MUTATING_KEY = "_intrinsic_stat_mutation_active"
PROGRESSION_KEYS = ("Level", "exp_current", "exp_required", "exp_kill", "needaunit",
                    "hero_level_stat_abilities", "campaign_lord_might_bonus_levels")


def _normal(stat, value):
    value = max(0, int(round(float(value or 0))))
    return min(100, value) if stat.startswith("accuracy") else value


def has_stat_sources(unit):
    source = unit.get(SOURCE_KEY)
    return isinstance(source, dict) and source.get("version") == 1


def _bare_values(unit):
    result = {}
    for stat in STATS:
        value = unit.get(ALIASES.get(stat, stat), unit.get(stat, 0))
        if stat == "damage":
            value = unit.get("original_damage", unit.get("lower_damage_original_damage", value))
        elif stat == "armor":
            value = unit.get("shatter_original_armor", unit.get("base_armor", value))
        elif stat == "max_health":
            value = unit.get("max_health", unit.get("maxhp", 0))
        elif stat == "initiative":
            value = unit.get("hermit_original_initiative_base", value)
        # Earliest saved layer wins. Never divide a rounded/capped value.
        for layer in ("heal_tile", "potion", "map_spell", "artifact", "banner", "settlement"):
            key = "campaign_" + layer + "_base_" + stat
            if key in unit:
                value = unit[key]
                break
        if stat == "initiative" and "campaign_map_spell_base_initiative_base" in unit:
            value = unit["campaign_map_spell_base_initiative_base"]
        if stat == "armor" and "garrison_base_armor" in unit:
            value = unit["garrison_base_armor"]
        result[stat] = _normal(stat, value)
    return result


def ensure_stat_sources(unit):
    if not has_stat_sources(unit):
        unit[SOURCE_KEY] = {
            "version": 1,
            "base": _bare_values(unit),
            "elixirs": [],
            "legacy_baked": bool(unit.get("campaign_permanent_potions")) or any(
                key.startswith("campaign_") and key.endswith("_uses") and value
                for key, value in unit.items()
            ),
        }
    return unit[SOURCE_KEY]


def persistent_values(unit):
    source = unit[SOURCE_KEY]
    values = {stat: _normal(stat, source["base"].get(stat, 0)) for stat in STATS}
    for effect in source["elixirs"]:
        kind = effect["kind"]
        if kind == "armor":
            values["armor"] += max(0, int(effect.get("armor_bonus", 0) or 0))
            continue
        targets = {
            "damage": ("damage", "damage_secondary"),
            "accuracy": ("accuracy", "accuracy_secondary"),
            "initiative": ("initiative",), "health": ("max_health",),
        }.get(kind, ())
        multiplier = max(0.0, float(effect.get("multiplier", 1.0) or 1.0))
        for stat in targets:
            old = values[stat]
            values[stat] = _normal(stat, old * multiplier)
            if stat == "max_health":
                values[stat] = max(old + 1, values[stat])
    return values


def rebuild_stat_layers(unit, *, health_mode="keep", layers=True):
    """Rebuild derived fields; HP is state, never a source for stat growth.

    Existing layer markers are also the compatibility boundary with battle
    cleanup. Refresh keeps exact current HP; real growth rescales injuries.
    """
    if not has_stat_sources(unit):
        return
    old_max = float(unit.get("max_health", unit.get("maxhp", 0)) or 0)
    old_hp = float(unit.get("health", unit.get("hp", 0)) or 0)
    old_initiative = int(unit.get("initiative", unit.get("initiative_base", 0)) or 0)
    old_initiative_base = int(unit.get("initiative_base", old_initiative) or 0)
    values = persistent_values(unit)

    def stage(layer, stat, multiplier=1.0, bonus=0):
        key = "campaign_" + layer + "_base_" + stat
        if layer == "map_spell" and stat == "initiative" and key not in unit:
            key = "campaign_map_spell_base_initiative_base"
        if key not in unit:
            return
        unit[key] = values[stat]
        values[stat] = _normal(stat, values[stat] * multiplier + bonus)

    if layers:
        stage("heal_tile", "armor", bonus=unit.get("campaign_heal_tile_armor_bonus", 0))
        for stat in ("damage", "damage_secondary", "initiative", "accuracy", "accuracy_secondary"):
            kind = stat.replace("_secondary", "")
            stage("potion", stat, unit.get("campaign_" + kind + "_potion_multiplier", 1.0))
        stage("potion", "armor", bonus=unit.get("campaign_potion_armor_bonus", 0))
        for stat in ("damage", "damage_secondary", "initiative", "accuracy", "accuracy_secondary"):
            kind = stat.replace("_secondary", "")
            stage("map_spell", stat, unit.get("campaign_map_spell_" + kind + "_multiplier", 1.0))
        stage("map_spell", "armor", bonus=unit.get("campaign_map_spell_armor_bonus", 0))
        stage("map_spell", "max_health", bonus=unit.get("campaign_map_spell_health_bonus", 0))
        for layer in ("artifact", "banner"):
            for stat in ("damage", "initiative"):
                stage(layer, stat, unit.get("campaign_" + layer + "_" + stat + "_multiplier", 1.0))
            stage(layer, "armor", bonus=unit.get("campaign_" + layer + "_armor_bonus", 0))
        stage("banner", "accuracy", bonus=unit.get("campaign_banner_accuracy_bonus", 0))
        stage("settlement", "armor", bonus=unit.get("settlement_armor_bonus", 0))
    if layers and "garrison_base_armor" in unit:
        bonus = unit.get("garrison_armor_bonus", 0)
        unit["garrison_base_armor"] = values["armor"]
        values["armor"] = min(90, _normal("armor", values["armor"] + bonus))
    # Battle snapshots describe the derived pre-status values, not intrinsic
    # sources. Refresh them so cleansing cannot erase permanent growth.
    original_damage, original_armor = values["damage"], values["armor"]
    if layers:
        if "lower_damage_original_damage" in unit:
            unit["lower_damage_original_damage"] = original_damage
        if "shatter_original_armor" in unit:
            unit["shatter_original_armor"] = original_armor
        for factor in unit.get("_battle_damage_factors", ()):
            values["damage"] = _normal("damage", values["damage"] * factor)
        if unit.get("teamated"):
            saved = original_damage
            for factor in unit.get("_battle_lower_damage_factors", ()):
                saved = _normal("damage", saved * factor)
            unit["lower_damage_original_damage"] = saved
        if unit.get("shattered_armor", 0):
            unit["shatter_original_armor"] = values["armor"]
            values["armor"] = max(0, values["armor"] - int(unit["shattered_armor"]))
        if unit.get("hermited"):
            unit["hermit_original_initiative_base"] = values["initiative"]
            values["initiative"] = _normal("initiative", values["initiative"] * 0.5)
    for stat, value in values.items():
        unit[stat] = value
    unit["initiative_base"] = values["initiative"]
    unit["original_damage"] = original_damage
    unit["base_armor"] = original_armor
    # Initiative is also activation state. Keep spent turns and existing jitter
    # rather than issuing a new turn whenever a source is recalculated.
    if layers:
        unit["initiative"] = (old_initiative if old_initiative <= 0 else
                              max(0, old_initiative + values["initiative"] - old_initiative_base))
    unit["maxhp"] = values["max_health"]
    if health_mode == "ratio" and old_max > 0 and values["max_health"] != old_max:
        # Flat temporary HP is not an injury/level-growth source. Remove it
        # before scaling, then restore it for the still-active battle layer.
        health_bonus = float(unit.get("campaign_map_spell_health_bonus", 0) or 0) if (
            layers and "campaign_map_spell_base_max_health" in unit
        ) else 0.0
        intrinsic_max = max(0.0, old_max - health_bonus)
        intrinsic_hp = min(intrinsic_max, max(0.0, old_hp - health_bonus))
        if old_hp <= 0:
            old_hp = 0
        elif intrinsic_max > 0:
            grown_max = max(0.0, values["max_health"] - health_bonus)
            old_hp = max(0, int(intrinsic_hp * grown_max / intrinsic_max + 0.5) + health_bonus)
    unit["health"] = min(values["max_health"], max(0, old_hp))
    unit["hp"] = unit["health"]


def add_permanent_effect(unit, effect):
    """Append only the newly consumed dose; counters are historical metadata."""
    source = ensure_stat_sources(unit)
    old_hp = float(unit.get("health", unit.get("hp", 0)) or 0)
    source["elixirs"].append(deepcopy(effect))
    if effect.get("kind") == "health":
        # Preserve the existing potion rule, including per-dose banker's rounding.
        unit["health"] = unit["hp"] = _normal(
            "max_health", old_hp * max(0.0, float(effect.get("multiplier", 1.0) or 1.0))
        )
    rebuild_stat_layers(unit)


def mutate_intrinsic(unit, mutation):
    """Run a level/skill mutation on intrinsic stats, then replay all layers.

    The clean view has no source record, so nested decorated growth helpers run
    once. The battle's wounds and temporary status never become intrinsic stats.
    """
    if unit.get(MUTATING_KEY):
        return mutation(unit)
    clean = deepcopy(unit)
    source = deepcopy(ensure_stat_sources(clean))
    clean.pop(SOURCE_KEY, None)
    clean[MUTATING_KEY] = True
    for key in tuple(clean):
        if key.startswith("campaign_") and "_base_" in key:
            clean.pop(key)
    clean.update(source["base"])
    clean["initiative_base"] = clean["initiative"]
    clean["original_damage"] = clean["damage"]
    clean["base_armor"] = clean["armor"]
    clean["health"] = clean["hp"] = clean["maxhp"] = clean["max_health"]
    for key in ("lower_damage_original_damage", "shatter_original_armor",
                "hermit_original_initiative_base", "garrison_base_armor"):
        clean.pop(key, None)
    result = mutation(clean)
    new_base = _bare_values(clean)
    changed = new_base != source["base"] or any(
        clean.get(key) != unit.get(key) for key in PROGRESSION_KEYS
    )
    if not changed:
        return result
    source["base"] = new_base
    unit[SOURCE_KEY] = source
    for key in PROGRESSION_KEYS:
        if key in clean:
            unit[key] = deepcopy(clean[key])
    rebuild_stat_layers(unit, health_mode="ratio")
    return result


def intrinsic_stat_mutation(function):
    """Decorator for standalone growth functions whose first argument is a unit."""
    @wraps(function)
    def wrapped(unit, *args, **kwargs):
        return mutate_intrinsic(unit, lambda clean: function(clean, *args, **kwargs))
    return wrapped
