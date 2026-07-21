import json
from pathlib import Path

import pytest

from tools.d2_campaign_converter import EVENT_TABLES
from tools.d2_campaign_converter import UNIT_TYPE_TO_DATA_NAME
from tools.d2_campaign_converter import _assert_output_is_safe
from tools.d2_campaign_converter import convert_campaign


DATA_DIR = Path(__file__).resolve().parents[1] / "maps" / "data"
SNAPSHOT_PATH = DATA_DIR / "wotans_retribution.json"
REPORT_PATH = DATA_DIR / "wotans_retribution.report.json"
GAME_ROOT = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Disciples II Rise of the Elves"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_generated_snapshot_has_exact_s117_static_counts_and_no_events():
    snapshot = _load(SNAPSHOT_PATH)
    report = _load(REPORT_PATH)

    assert snapshot["scenario"]["source_id"] == "S117SC0000"
    assert snapshot["scenario"]["name"] == "The Charmed Child"
    assert snapshot["scenario"]["map_size"] == 48
    assert report["counts"] == {
        "map_cells": 2304,
        "map_blocks": 72,
        "roads": 194,
        "mountain_entries": 196,
        "capitals": 2,
        "settlements": 6,
        "source_stacks": 63,
        "runtime_enemies": 71,
        "chests": 24,
        "resources": 11,
        "ruins": 4,
        "merchants": 2,
        "trainers": 1,
        "water_tiles": 181,
        "forest_tiles": 441,
    }
    assert report["event_tables_read"] == []
    assert snapshot["scenario"]["excluded_event_tables"] == list(EVENT_TABLES)
    assert not ({"events", "triggers", "event_templates"} & set(snapshot))


def test_snapshot_preserves_source_ids_and_all_type_mappings_are_strict():
    snapshot = _load(SNAPSHOT_PATH)
    assert {row["source_id"] for row in snapshot["settlements"]} == {
        f"S117FT{index:04x}" for index in range(8)
    }
    assert len({row["source_id"] for row in snapshot["stacks"]}) == 63
    assert len({row["source_id"] for row in snapshot["chests"]}) == 24
    assert len({row["source_id"] for row in snapshot["resources"]}) == 11
    assert len({row["source_id"] for row in snapshot["ruins"]}) == 4

    source_unit_types = {
        unit["type_id"]
        for stack in snapshot["stacks"]
        for unit in stack["group"]["units"]
    }
    source_unit_types.update(
        unit["type_id"]
        for settlement in snapshot["settlements"]
        for unit in settlement["garrison"]["units"]
    )
    source_unit_types.update(
        unit["type_id"]
        for ruin in snapshot["ruins"]
        for unit in ruin["garrison"]["units"]
    )
    assert source_unit_types <= set(UNIT_TYPE_TO_DATA_NAME)


@pytest.mark.parametrize("protected_name", ("Scens", "Campaign", "Globals", "Exports"))
def test_converter_refuses_to_write_into_game_directories(tmp_path, protected_name):
    game_root = tmp_path / "game"
    output = game_root / protected_name / "generated.json"

    with pytest.raises(ValueError, match="Refusing to write"):
        _assert_output_is_safe(game_root, output)


@pytest.mark.skipif(not GAME_ROOT.is_dir(), reason="local Disciples II install is absent")
def test_local_read_only_conversion_reproduces_committed_snapshot(tmp_path):
    output = tmp_path / "snapshot.json"
    report_output = tmp_path / "report.json"
    snapshot, report = convert_campaign(
        GAME_ROOT,
        "S117SC0000",
        output,
        report_output,
    )

    assert snapshot == _load(SNAPSHOT_PATH)
    assert report == _load(REPORT_PATH)
