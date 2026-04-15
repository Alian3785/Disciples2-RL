import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _unit_entry(name: str) -> dict:
    return next(
        entry
        for entry in DATA
        if isinstance(entry, dict) and str(entry.get("кто", "") or "") == name
    )


def _neutral_entries() -> list[dict]:
    entries: list[dict] = []
    for entry in DATA:
        if not isinstance(entry, dict):
            continue
        try:
            capital_v = int(round(float(entry.get("столица", entry.get("capital", 0)) or 0)))
        except (TypeError, ValueError):
            capital_v = 0
        if capital_v == 0:
            entries.append(entry)
    return entries


def test_all_neutral_unit_entries_have_level_one_or_higher():
    neutral_entries = _neutral_entries()

    assert neutral_entries
    for entry in neutral_entries:
        level_v = int(round(float(entry.get("уровень", entry.get("Level", 0)) or 0)))
        assert level_v >= 1, entry.get("кто", "")


@pytest.mark.parametrize(
    ("unit_name", "expected_cost_per_hp"),
    [
        ("Гоблин", 1.0),
        ("Варвар", 2.0),
        ("Тролль", 3.0),
        ("Ниддог", 4.0),
    ],
)
def test_neutral_unit_heal_cost_uses_max_health_buckets(unit_name, expected_cost_per_hp):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    unit = map_unit_to_battle(_unit_entry(unit_name), team="blue", position=7)

    assert unit["is_neutral_unit"] is True
    assert int(unit["capital"]) == 0
    assert int(unit["Level"]) == 1
    assert env._castle_heal_gold_per_hp(unit) == pytest.approx(expected_cost_per_hp)


def test_faction_unit_heal_cost_still_uses_level_and_not_hp_bucket():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    unit = map_unit_to_battle(_unit_entry("Скваер"), team="blue", position=7)
    unit["Level"] = 4
    unit["max_health"] = 90

    assert unit["is_neutral_unit"] is False
    assert int(unit["capital"]) == 1
    assert env._castle_heal_gold_per_hp(unit) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("max_health", "expected_revive_cost"),
    [
        (100.0, 50.0),
        (150.0, 200.0),
        (250.0, 400.0),
        (350.0, 600.0),
        (500.0, 800.0),
        (2000.0, 1000.0),
        (2001.0, 1500.0),
    ],
)
def test_neutral_unit_revive_cost_uses_max_health_buckets(max_health, expected_revive_cost):
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    unit = {
        "is_neutral_unit": True,
        "capital": 0,
        "Level": 1,
        "max_health": float(max_health),
    }

    assert env._castle_revive_gold_cost(unit) == pytest.approx(expected_revive_cost)


def test_faction_unit_revive_cost_still_uses_level_and_not_hp_bucket():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    unit = map_unit_to_battle(_unit_entry("Скваер"), team="blue", position=7)
    unit["Level"] = 4
    unit["max_health"] = 90

    assert unit["is_neutral_unit"] is False
    assert int(unit["capital"]) == 1
    assert env._castle_revive_gold_cost(unit) == pytest.approx(600.0)
