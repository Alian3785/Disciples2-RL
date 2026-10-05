"""Source-driven permanent elixir stats, independent of equipment snapshots.

Only elixir recipients opt in. Legacy recipients keep their recoverable bare
stats as a baked baseline: rounded/capped or already-lost history is not guessed.
Order: intrinsic (type/level/skills), each permanent dose, then the existing
battle layers (tile, temporary potion, map spell, artifacts, banner).
"""
from copy import deepcopy
from functools import wraps

SOURCE_KEY = "campaign_stat_sources"
STATS = ("damage", "damage_secondary", "accuracy", "accuracy_secondary",
         "initiative", "armor", "max_health")
ALIASES = {"initiative": "initiative_base", "max_health": "maxhp"}
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
        # Earliest saved layer wins. Never divide a rounded/capped value.
        for layer in ("heal_tile", "potion", "map_spell", "artifact", "banner"):
            key = "campaign_" + layer + "_base_" + stat
            if key in unit:
                value = unit[key]
                break
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
    values = persistent_values(unit)

    def stage(layer, stat, multiplier=1.0, bonus=0):
        key = "campaign_" + layer + "_base_" + stat
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
    for stat, value in values.items():
        unit[stat] = value
    unit["initiative_base"] = values["initiative"]
    unit["original_damage"] = values["damage"]
    unit["base_armor"] = values["armor"]
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
    if not has_stat_sources(unit):
        return mutation(unit)
    clean = deepcopy(unit)
    source = clean.pop(SOURCE_KEY)
    for key in tuple(clean):
        if key.startswith("campaign_") and "_base_" in key:
            clean.pop(key)
    clean.update(source["base"])
    clean["initiative_base"] = clean["initiative"]
    clean["original_damage"] = clean["damage"]
    clean["base_armor"] = clean["armor"]
    clean["health"] = clean["hp"] = clean["maxhp"] = clean["max_health"]
    for key in ("lower_damage_original_damage", "shatter_original_armor",
                "hermit_original_initiative_base"):
        clean.pop(key, None)
    result = mutation(clean)
    unit[SOURCE_KEY]["base"] = _bare_values(clean)
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
