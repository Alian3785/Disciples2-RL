"""Gold per healed HP / purchased XP from Gunits and GDynUpgr.

Use the permanent unit ID, as growth and evolution do. Temporary forms and
HP modifiers must not select another creature's campaign service prices.
"""
from unit_dynamic_xp import dynamic_unit_id
from unit_service_cost_data import UNIT_SERVICE_COST_PROFILES


def known_unit_service_gold_costs(unit):
    """Return (HEAL_C, TRAINING_C), or None for an unknown custom creature.

    Gunits prices apply at the native level. Each subsequent level adds the
    corresponding GDynUpgr values; DYN_UPG2 starts after DYN_UPG_LV.
    """
    if not isinstance(unit, dict):
        return None
    profile = UNIT_SERVICE_COST_PROFILES.get(dynamic_unit_id(unit))
    if profile is None:
        return None
    native_level, last_early_level, base, early, late = profile
    try:
        level = max(native_level, int(round(float(
            unit.get("Level", unit.get("level", unit.get("уровень", native_level)))
        ))))
    except (TypeError, ValueError, OverflowError):
        level = native_level
    early_levels = max(0, min(level, last_early_level) - native_level)
    late_levels = max(0, level - max(native_level, last_early_level))
    return tuple(price + early_levels * first + late_levels * second
                 for price, first, second in zip(base, early, late))
