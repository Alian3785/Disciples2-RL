from tools.sim_bughunt_battle import (
    _mirror_team,
    balanced_controlled_team,
    build_team,
)


def test_balanced_controller_schedule_is_even_for_both_original_teams():
    controlled_teams = [balanced_controlled_team(index) for index in range(10)]

    assert controlled_teams.count("blue") == 5
    assert controlled_teams.count("red") == 5


def test_mirror_team_preserves_units_and_maps_red_positions_to_blue():
    original = build_team("red", "1:Астерот,5:Инкуб", range(1, 7))

    mirrored = _mirror_team(original, "blue")

    by_name = {unit.get("name"): unit for unit in mirrored if unit.get("name") != "пусто"}
    assert by_name["Астерот"]["team"] == "blue"
    assert by_name["Астерот"]["position"] == 7
    assert by_name["Астерот"]["stand"] == "ahead"
    assert by_name["Инкуб"]["team"] == "blue"
    assert by_name["Инкуб"]["position"] == 11
    assert by_name["Инкуб"]["stand"] == "behind"
