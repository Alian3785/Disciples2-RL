import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit
from grid import BLUE_TEAM_TEMPLATE


def _unit_entry(name: str) -> dict:
    return next(entry for entry in DATA if entry.get("кто") == name)


def test_data_mapping_and_grid_materialize_explicit_hero_flag():
    duke_entry = _unit_entry("Герцог")
    berserker_entry = _unit_entry("Одержимый")

    assert duke_entry["hero"] is True
    assert berserker_entry["hero"] is False

    duke = map_unit_to_battle(duke_entry, team="blue", position=8)
    berserker = map_unit_to_battle(berserker_entry, team="blue", position=7)

    assert duke["hero"] is True
    assert berserker["hero"] is False
    assert placeholder_unit("blue", 10)["hero"] is False

    duke_template = next(
        unit for unit in BLUE_TEAM_TEMPLATE if int(unit.get("position", -1)) == 8
    )
    assert duke_template["hero"] is True
    assert all(
        bool(unit.get("hero", False)) is False
        for unit in BLUE_TEAM_TEMPLATE
        if int(unit.get("position", -1)) != 8
    )


def test_campaign_hero_detection_uses_explicit_hero_flag_before_exp_threshold():
    assert CampaignEnv._is_hero_unit({"hero": True, "next_level_exp": 0}) is True
    assert CampaignEnv._is_hero_unit({"hero": False, "next_level_exp": 500}) is False
    assert CampaignEnv._is_hero_unit({"next_level_exp": 500}) is True
    assert CampaignEnv._is_hero_unit({"next_level_exp": 0}) is False


def test_resolve_travel_hero_uses_first_hero_in_roster_without_position_tiebreak():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    first_hero = {"name": "First hero", "hero": True, "position": 11}
    second_hero = {"name": "Second hero", "hero": True, "position": 8}

    resolved = env._resolve_travel_hero(
        units=[
            {"name": "Regular", "hero": False, "position": 7},
            first_hero,
            second_hero,
        ]
    )

    assert resolved is first_hero
