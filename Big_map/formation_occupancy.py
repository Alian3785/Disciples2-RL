"""Shared tactical footprint checks for reviving a fighter in place."""


def formation_footprint(unit):
    """Return one cell, or the front/back pair for a large fighter on either side."""
    position = int(unit.get("position", 0) or 0)
    if position not in range(1, 13):
        return frozenset()
    cells = {position}
    if unit.get("big", False):
        cells.add(position + 3 if position in (1, 2, 3, 7, 8, 9) else position - 3)
    return frozenset(cells)


def formation_footprint_is_free(unit, living_units):
    """Corpses do not reserve cells; every other living footprint must be clear.

    Callers supply living units using their native HP representation. Identity
    excludes only the same fighter, never a different fighter with equal data.
    """
    cells = formation_footprint(unit)
    return bool(cells) and all(
        other is unit or cells.isdisjoint(formation_footprint(other))
        for other in living_units
    )
