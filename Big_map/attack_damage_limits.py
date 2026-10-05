"""Rise of the Elves attack-characteristic limits, evaluated after modifiers.

Do not overwrite ``damage`` or permanent/snapshot stats with the result: they
are reversible arithmetic inputs, and ``damage`` also stores healing amounts.
A separate current-form identity survives copying and temporary transforms
without changing the permanent fighter's canonical unit ID.
"""
from unit_dynamic_xp import dynamic_unit_id

DEFAULT_ATTACK_DAMAGE_CAP = 300
HEAVY_STRIKE_ATTACK_DAMAGE_CAP = 400
HEAVY_STRIKE_UNIT_IDS = frozenset({
    "g000uu0019",  # Pegasus Knight
    "g000uu0020",  # Human Ranger
    "g000uu0044",  # Royal Guard
    "g000uu0045",  # Engineer
    "g000uu0047",  # Dwarf Elder (rod planter)
    "g000uu0070",  # Duke
    "g000uu0071",  # Counsellor
    "g000uu0096",  # Death Knight
    "g000uu8009",  # Forest Liege (not ordinary Elf Lord g000uu5008)
    "g000uu8011",  # Forest Guardian
})
ATTACK_FORM_ID_KEY = "_attack_form_unit_id"
# These primary actions heal, buff, summon, copy, or apply status. Their amount
# is not offensive damage. Wolf Lord is deliberately absent: this engine gives
# its normal form an AoE attack even though its native growth profile says 0.
NON_DAMAGE_PRIMARY_TYPES = frozenset({
    "Cliric", "Profit", "Patriach", "Deva roshi", "Sundancer", "Sylfid",
    "Ghost", "Witch", "Shadow", "Incub", "Succub", "Summoner",
    "Occultmaster", "Laclaan", "Lyf", "Baroness", "Travnitsa", "Novice",
    "Alchemist", "Dwarfdruid", "Arhidruid", "Doppelganger",
})


def current_attack_unit_id(unit):
    """Current attack form; explicit canonical IDs beat mutable display names."""
    if ATTACK_FORM_ID_KEY in unit:
        return str(unit.get(ATTACK_FORM_ID_KEY) or "").strip().lower()
    # Old external in-progress snapshots do not identify their copied form.
    # Fail closed to 300 rather than inherit the original hero's exception.
    if unit.get("transformed") or unit.get("doppel_copied"):
        return ""
    return dynamic_unit_id(unit)


def attack_damage_cap(unit):
    return (HEAVY_STRIKE_ATTACK_DAMAGE_CAP
            if current_attack_unit_id(unit) in HEAVY_STRIKE_UNIT_IDS
            else DEFAULT_ATTACK_DAMAGE_CAP)


def effective_attack_damage(unit, value=None):
    """Cap one offensive attack before random bonus, critical, and mitigation."""
    raw = unit.get("damage", 0) if value is None else value
    return min(max(0.0, float(raw or 0)), attack_damage_cap(unit))


def effective_primary_amount(unit):
    """Observation/display value, retaining healing and support semantics."""
    raw = max(0.0, float(unit.get("damage", 0) or 0))
    if unit.get("unit_type") in NON_DAMAGE_PRIMARY_TYPES:
        return raw
    return effective_attack_damage(unit, raw)
