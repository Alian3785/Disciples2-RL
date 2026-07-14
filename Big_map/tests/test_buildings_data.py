import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Buildings import (  # noqa: E402
    elves_buildings_d2,
    empire_buildings_d2,
    legions_buildings_d2,
    mountain_clans_buildings_d2,
    undead_hordes_buildings_d2,
)
from data_dicts_compact_lines import DATA  # noqa: E402


BUILDING_TABLES = {
    "empire": empire_buildings_d2,
    "mountain_clans": mountain_clans_buildings_d2,
    "undead_hordes": undead_hordes_buildings_d2,
    "legions": legions_buildings_d2,
    "elves": elves_buildings_d2,
}

BUILDING_TABLES_BY_CAPITAL = {
    1: empire_buildings_d2,
    2: legions_buildings_d2,
    3: mountain_clans_buildings_d2,
    4: undead_hordes_buildings_d2,
    5: elves_buildings_d2,
}


def _building_entries(table: dict):
    return {
        key: entry
        for key, entry in table.items()
        if isinstance(entry, dict)
    }


def test_building_requires_and_blocks_reference_existing_buildings():
    for faction_name, table in BUILDING_TABLES.items():
        entries = _building_entries(table)
        names = {str(entry.get("name", "") or "").strip() for entry in entries.values()}

        for build_key, entry in entries.items():
            assert isinstance(entry.get("requires"), tuple), (faction_name, build_key)
            assert isinstance(entry.get("blocks"), tuple), (faction_name, build_key)
            for requirement_name in entry.get("requires") or ():
                assert requirement_name in names, (
                    faction_name,
                    build_key,
                    "requires",
                    requirement_name,
                )
            for blocked_name in entry.get("blocks") or ():
                assert blocked_name in names, (
                    faction_name,
                    build_key,
                    "blocks",
                    blocked_name,
                )


def test_building_units_exist_in_unit_data():
    unit_names = {
        str(entry.get("кто", "") or "").strip()
        for entry in DATA
        if isinstance(entry, dict)
    }

    for faction_name, table in BUILDING_TABLES.items():
        for build_key, entry in _building_entries(table).items():
            unit_name = str(entry.get("unit", "") or "").strip()
            if not unit_name:
                continue
            assert unit_name in unit_names, (faction_name, build_key, unit_name)


def test_legions_high_temple_requires_sorceress_building():
    assert legions_buildings_d2["lod_d2_b012"]["name"] == "Высокий храм"
    assert legions_buildings_d2["lod_d2_b012"]["unit"] == "Суккуб"
    assert legions_buildings_d2["lod_d2_b012"]["requires"] == ("Дворец Греха",)


def test_progression_targets_follow_their_faction_building_tree():
    for source in DATA:
        capital = int(source.get("столица", 0) or 0)
        table = BUILDING_TABLES_BY_CAPITAL.get(capital)
        if table is None:
            continue

        entries_by_unit = {
            str(entry.get("unit", "") or "").strip(): entry
            for entry in _building_entries(table).values()
            if str(entry.get("unit", "") or "").strip()
        }
        source_name = str(source.get("кто", "") or "").strip()
        source_building = entries_by_unit.get(source_name)

        for target_name in source.get("превращается в", ()) or ():
            target_building = entries_by_unit.get(target_name)
            assert target_building is not None, (
                source_name,
                target_name,
                "missing target building",
            )
            if source_building is None:
                continue
            assert source_building["name"] in target_building.get("requires", ()), (
                source_name,
                target_name,
                source_building["name"],
                target_building.get("requires", ()),
            )


def test_alternative_progression_buildings_block_each_other_symmetrically():
    for source in DATA:
        targets = tuple(source.get("превращается в", ()) or ())
        if len(targets) < 2:
            continue

        capital = int(source.get("столица", 0) or 0)
        table = BUILDING_TABLES_BY_CAPITAL[capital]
        entries_by_unit = {
            str(entry.get("unit", "") or "").strip(): entry
            for entry in _building_entries(table).values()
            if str(entry.get("unit", "") or "").strip()
        }

        for target_name in targets:
            target_building = entries_by_unit[target_name]
            alternative_names = {
                entries_by_unit[alternative]["name"]
                for alternative in targets
                if alternative != target_name
            }
            assert alternative_names <= set(target_building.get("blocks", ())), (
                source.get("кто"),
                target_name,
                alternative_names,
                target_building.get("blocks", ()),
            )
