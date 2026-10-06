"""A source unit occupying two scenario slots must remain one combatant."""

from collections import Counter
from copy import deepcopy

import pytest

from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle
from maps.wotans_retribution import SOURCE_SNAPSHOT

ENEMIES = SOURCE_SNAPSHOT["enemies"]
UNIT_DATA = {unit["кто"]: unit for unit in DATA}


def source_units(enemy):
    category, group_key = {
        "stack": ("stacks", "group"),
        "settlement_garrison": ("settlements", "garrison"),
        "ruin_guardian": ("ruins", "garrison"),
    }[enemy["kind"]]
    source = next(row for row in SOURCE_SNAPSHOT[category]
                  if row["source_id"] == enemy["source_id"])
    return {unit["source_unit_id"]: unit for unit in source[group_key]["units"]}


@pytest.fixture(scope="module")
def env():
    env = CampaignEnv(
        map_name="wotans_retribution", observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    env.reset(seed=42)
    yield env
    env.close()


@pytest.mark.parametrize("enemy", ENEMIES, ids=lambda row: f"enemy{row['enemy_id']}")
def test_source_army_survives_import_and_battle_initialization(env, enemy):
    unique = source_units(enemy)
    expected = [map_unit_to_battle(UNIT_DATA[u["unit_name"]], "red", 1)
                for u in unique.values()]
    enemy_id = enemy["enemy_id"]
    # Reach an actual blue turn before the enemy can attack, summon or transform.
    hero = deepcopy(env._resolve_travel_hero())
    hero.update(initiative=10000, initiative_base=10000)
    env._init_battle(enemy_id, blue_team=[hero])
    assert env.battle_env.current_blue_attacker_pos == hero["position"]

    for team in (env._enemy_configs[enemy_id], env.enemy_team_states[enemy_id],
                 env.battle_env.combined):
        fighters = [u for u in team if u["team"] == "red" and u["name"] != "пусто"]
        assert Counter(u["name"] for u in fighters) == Counter(u["name"] for u in expected)
        assert sum(u["max_health"] for u in fighters) == sum(u["max_health"] for u in expected)
        assert sum(u["exp_kill"] for u in fighters) == sum(u["exp_kill"] for u in expected)
        occupied = set()
        for unit in fighters:
            position = unit["position"]
            cells = {position}
            if unit["big"]:
                assert position in (1, 2, 3)
                cells.add(position + 3)
            assert not (occupied & cells), (enemy_id, unit["name"], position)
            occupied.update(cells)
        assert len(occupied) <= 6

    expected_names = Counter(unit["unit_name"] for unit in unique.values())
    description_names = env._enemy_descriptions[enemy_id].split(": ", 1)[1].split(", ")
    assert Counter(description_names) == expected_names


@pytest.mark.parametrize("enemy_id,name,count", [
    (11, "Холмовой гигант", 1),
    (33, "Дикий гигант", 2),  # Distinct IDs, each repeated in two source cells.
    (40, "Холмовой гигант", 2),
    (6, "Сектант", 2),  # Two small units must survive deduplication of the Beast.
    (1, "Варвар", 2),
])
def test_distinct_same_named_units_are_preserved(env, enemy_id, name, count):
    team = env.enemy_team_states[enemy_id]
    assert sum(unit["name"] == name for unit in team) == count
