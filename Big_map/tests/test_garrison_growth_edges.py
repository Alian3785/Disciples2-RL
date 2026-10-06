"""Building-gated reserve promotions use battle recipients, never the hero roster.

Most cases deterministically record a real BattleEnv defeat and award its XP,
then cross the public CampaignEnv.step boundary.  The final combat case also
uses actual actions, so a missing campaign victory hook cannot be hidden by a
direct call to the promotion helper.
"""

from copy import deepcopy

import pytest

from campaign_env import CampaignEnv


CANONICAL_FIELDS = (
    "name", "Level", "max_health", "maxhp", "damage", "original_damage",
    "damage_secondary", "accuracy", "accuracy_secondary", "initiative_base",
    "armor", "unit_type", "exp_required", "turns_into",
)


@pytest.fixture
def env():
    value = CampaignEnv(
        Realcapital=2, observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )
    value.reset(seed=42)
    city = value.garrison_city_names[0]
    for enemy_id in value.legions_settlement_territory_source_by_name[city]["required_enemy_ids"]:
        value.grid_env.mark_enemy_defeated(enemy_id)
    value.grid_env.agent_pos = value.legions_settlement_source_tile_by_name[city]
    value._capture_cities_on_hero_entry(0, {})
    value.legions_settlement_level_by_name[city] = 5
    value.grid_env.agent_pos = tuple(value.CASTLE_POS)
    yield value
    value.close()


def unit(env, name="Одержимый", position=7, *, ready=True):
    result = env._build_unit_from_data(env._find_unit_data_by_name(name), "blue", position)
    if ready:
        result["exp_current"] = result["exp_required"] - 1
    return result


def build_for(env, target, *, built=True):
    capital = env._find_unit_data_by_name(target)["столица"]
    buildings = env._get_buildings_for_capital(capital)
    entry = next(row for row in buildings.values()
                 if isinstance(row, dict) and row.get("unit") == target)
    entry["built"] = int(built)
    return entry


def start(env, roster, *, exp=12):
    city = env.garrison_city_names[0]
    env.city_garrisons[city] = deepcopy(roster)
    attacker = unit(env, "Скваер", 8, ready=False)
    attacker.update(health=1, hp=1, damage=0, original_damage=0,
                    initiative=1, initiative_base=1, exp_kill=exp)
    # A focused defence fixture, with no autonomous campaign bot enabled.
    env.scripted_capital_bot_team_state = [attacker]
    env._start_city_defence(city)
    return city, env.battle_env


def award_victory(battle):
    for target in battle.combined:
        if target.get("team") == "red" and target.get("health", 0) > 0:
            battle._subtract_health(target, target["health"])
    battle._finalize_victory("blue")
    assert battle.last_battle_exp > 0


def finish(env, *, victory=True):
    if victory:
        award_victory(env.battle_env)
    result = env.step(0)
    assert env.mode == env.MODE_GRID
    assert not any(result[2:4])
    assert env.observation_space.contains(result[0])
    return result


def assert_canonical(env, actual, target, *, exp=0):
    expected = unit(env, target, actual["position"], ready=False)
    for field in CANONICAL_FIELDS:
        if field == "original_damage":
            # Battle initialization materializes this otherwise optional key.
            assert actual.get(field, actual["damage"]) == expected.get(field, expected["damage"])
            continue
        assert actual.get(field) == expected.get(field), field
    assert actual["health"] == actual["hp"] == expected["max_health"]
    assert actual["exp_current"] == exp
    assert actual["base_armor"] == expected["armor"]
    assert "garrison_base_armor" not in actual
    assert "_battle_exp_earned" not in actual


@pytest.mark.parametrize("source,target,position", [
    ("Одержимый", "Берсерк", 7),
    ("Сектант", "Колдун", 10),
    ("Сектант", "Ведьма", 11),
    ("Демон", "Молох", 7),
    ("Скваер", "Охотник на ведьм", 8),
    ("Служка", "Клирик", 10),
    ("Гном", "Гном воин", 9),
    ("Воин", "Тамплиер", 8),
    ("Кентавр копейщик", "Кентавр латник", 7),
])
def test_building_backed_profiles_promote_to_exact_canonical_form(env, source, target, position):
    build_for(env, target)
    defender = unit(env, source, position)
    defender["health"] = defender["hp"] = 17
    hero = deepcopy(env.blue_team_state)
    city, battle = start(env, [defender])
    award_victory(battle)
    assert battle.last_levelups == [source]
    assert battle.last_levelup_units[0] is next(
        u for u in battle.combined if u["team"] == "blue" and u["position"] == position)
    result = finish(env, victory=False)
    assert result[4]["battle_result"] == "victory"
    assert len(env.city_garrisons[city]) == 1
    assert_canonical(env, env.city_garrisons[city][0], target)
    assert env.blue_team_state == hero


@pytest.mark.parametrize("gate", ["absent", "unbuilt", "wrong_branch"])
def test_missing_or_wrong_building_keeps_xp_cap_and_wounded_old_form(env, gate):
    if gate == "unbuilt":
        build_for(env, "Берсерк", built=False)
    elif gate == "wrong_branch":
        build_for(env, "Колдун")
    defender = unit(env)
    defender["health"] = defender["hp"] = 29
    city, battle = start(env, [defender])
    finish(env)
    saved = env.city_garrisons[city][0]
    assert saved["name"] == "Одержимый"
    assert saved["exp_current"] == 94
    assert saved["exp_required"] == 95
    assert saved["health"] == saved["hp"] == 29
    assert saved["max_health"] == 120 and saved["damage"] == 25
    assert saved["armor"] == 0
    assert battle.last_levelups == ["Одержимый"]


@pytest.mark.parametrize("source,target", [("Одержимый", "Берсерк"), ("Демон", "Молох")])
@pytest.mark.parametrize("both_ready", [False, True])
def test_same_named_fighters_keep_distinct_identities_and_large_cells(env, source, target, both_ready):
    build_for(env, target)
    first = unit(env, source, 7)
    second = unit(env, source, 8, ready=both_ready)
    for index, fighter in enumerate((first, second)):
        fighter["source_unit_id"] = f"GARRISON_UNIT_{index}"
        fighter["source_group_id"] = "GARRISON_GROUP"
        fighter["source_pos_cells"] = [index, index + 3] if source == "Демон" else [index]
    city, battle = start(env, [first, second])
    runtime = [u for u in battle.combined
               if u["team"] == "blue" and not env._is_empty_blue_unit(u)]
    assert len(runtime) == 2
    assert {u["source_unit_id"] for u in runtime} == {"GARRISON_UNIT_0", "GARRISON_UNIT_1"}
    assert {u["position"] for u in runtime} == {7, 8}
    assert sum(u["health"] for u in runtime) == first["health"] + second["health"]
    assert sum(u["exp_current"] for u in runtime) == first["exp_current"] + second["exp_current"]
    finish(env)
    saved = {u["position"]: u for u in env.city_garrisons[city]}
    assert set(saved) == {7, 8}
    assert saved[7]["name"] == target
    assert saved[8]["name"] == (target if both_ready else source)
    assert saved[7]["exp_current"] == 0
    assert saved[8]["exp_current"] == (0 if both_ready else 6)
    assert_canonical(env, saved[7], target)
    assert_canonical(env, saved[8], target if both_ready else source, exp=0 if both_ready else 6)
    expected_hp = sum(unit(env, name, ready=False)["max_health"]
                      for name in (target, target if both_ready else source))
    assert sum(u["health"] for u in saved.values()) == expected_hp
    assert sum(u["exp_current"] for u in saved.values()) == (0 if both_ready else 6)
    for original in (first, second):
        actual = saved[original["position"]]
        for key in ("source_unit_id", "source_group_id", "source_pos_cells"):
            assert actual[key] == original[key]
    expected_cells = {7, 8, 10, 11} if source == "Демон" else {7, 8}
    assert env._garrison_occupied_positions(city) == expected_cells


@pytest.mark.parametrize("initial_xp,earned_xp,expected_xp", [(0, 12, 12), (82, 12, 94), (94, 0, 94)])
def test_building_alone_cannot_promote_without_an_earned_levelup(env, initial_xp, earned_xp, expected_xp):
    build_for(env, "Берсерк")
    defender = unit(env)
    defender["exp_current"] = initial_xp
    defender["health"] = defender["hp"] = 31
    city, battle = start(env, [defender], exp=earned_xp)
    for target in battle.combined:
        if target.get("team") == "red" and target.get("health", 0) > 0:
            battle._subtract_health(target, target["health"])
    battle._finalize_victory("blue")
    assert battle.last_battle_exp == earned_xp
    assert battle.last_levelups == []
    finish(env, victory=False)
    saved = env.city_garrisons[city][0]
    assert saved["name"] == "Одержимый" and saved["Level"] == 1
    assert saved["exp_current"] == expected_xp
    assert saved["exp_required"] == 95
    assert saved["health"] == saved["hp"] == 31


def test_dead_and_summoned_recipients_cannot_reappear_as_promoted_reserves(env):
    build_for(env, "Берсерк")
    defenders = [unit(env, position=pos) for pos in (7, 8, 9)]
    city, battle = start(env, defenders)
    by_position = {u["position"]: u for u in battle.combined if u["team"] == "blue"}
    battle._subtract_health(by_position[8], by_position[8]["health"])
    by_position[9]["Summoned"] = True
    finish(env)
    assert [(u["position"], u["name"]) for u in env.city_garrisons[city]] == [(7, "Берсерк")]
    assert by_position[8]["health"] == 0 and by_position[8]["name"] == "Одержимый"
    assert by_position[9]["Summoned"] and by_position[9]["name"] == "Одержимый"


def test_duplicate_and_foreign_events_do_not_promote_other_units_or_twice(env):
    build_for(env, "Берсерк")
    build_for(env, "Темный паладин")
    city, battle = start(env, [unit(env, position=7), unit(env, position=8, ready=False)])
    award_victory(battle)
    recipient = battle.last_levelup_units[0]
    foreign_copy = deepcopy(recipient)
    foreign_copy["position"] = 8
    red = next(u for u in battle.combined if u["team"] == "red" and u["position"] == 2)
    battle.last_levelups.extend([recipient["name"], foreign_copy["name"], red["name"]])
    battle.last_levelup_units.extend([recipient, foreign_copy, red])
    finish(env, victory=False)
    saved = {u["position"]: u for u in env.city_garrisons[city]}
    assert saved[7]["name"] == "Берсерк" and saved[7]["exp_current"] == 0
    assert saved[8]["name"] == "Одержимый" and saved[8]["exp_current"] == 6
    assert foreign_copy["name"] == "Одержимый"
    assert red["name"] == "Скваер"


def test_elixirs_rebase_once_and_persist_through_second_defence(env):
    build_for(env, "Кентавр латник")
    defender = unit(env, "Кентавр копейщик")
    expected = unit(env, "Кентавр латник", ready=False)
    potions = [d for d in env.POTION_ITEM_DEFINITIONS if d.get("duration") == "permanent"]
    assert potions
    for definition in potions:
        for _ in range(2):
            env._apply_permanent_potion_bonus_to_unit(defender, definition)
            env._apply_permanent_potion_bonus_to_unit(expected, definition)
    defender["source_unit_id"] = "GARRISON_ELIXIR_UNIT"
    defender["health"] = defender["hp"] = 13
    city, _ = start(env, [defender])
    finish(env)
    saved = env.city_garrisons[city][0]
    for field in CANONICAL_FIELDS:
        assert saved.get(field) == expected.get(field), field
    assert saved["health"] == saved["hp"] == expected["max_health"]
    assert saved["exp_current"] == 0
    expected_potions = deepcopy(saved["campaign_permanent_potions"])
    city, battle = start(env, [saved])
    runtime = next(u for u in battle.combined if u["team"] == "blue" and u["position"] == 7)
    assert runtime["armor"] > expected["armor"]
    finish(env)
    saved = env.city_garrisons[city][0]
    for field in CANONICAL_FIELDS:
        assert saved.get(field) == expected.get(field), field
    assert saved["source_unit_id"] == "GARRISON_ELIXIR_UNIT"
    assert saved["exp_current"] == 12
    assert saved["campaign_permanent_potions"] == expected_potions
    for definition in potions:
        assert saved[definition["permanent_counter_key"]] == 2


@pytest.mark.parametrize("outcome", ["timeout", "defeat"])
def test_nonvictory_does_not_consume_stale_promotion_events(env, outcome):
    build_for(env, "Берсерк")
    city, battle = start(env, [unit(env)])
    defender = next(u for u in battle.combined if u["team"] == "blue" and u["position"] == 7)
    battle.last_levelups = [defender["name"]]
    battle.last_levelup_units = [defender]
    if outcome == "defeat":
        battle.winner = "red"
    else:
        battle.step = lambda action: (battle._obs(), 0.0, False, True, {})
    result = finish(env, victory=False)
    assert result[4]["battle_result"] == outcome
    assert defender["name"] == "Одержимый" and defender["exp_current"] == 94
    if outcome == "timeout":
        assert env.city_garrisons[city][0]["name"] == "Одержимый"


def test_actual_combat_preserves_hero_resources_and_adds_no_upgrade_reward(env):
    build_for(env, "Берсерк")
    defender = unit(env)
    defender["accuracy"] = 100
    hero = deepcopy(env.blue_team_state)
    hero_position = tuple(env.grid_env.agent_pos)
    resources = (env.gold, deepcopy(env._current_mana_totals()), env.moves, env.turns)
    env._reset_last_upgrade_reward_tracking()
    env._last_upgrade_reward = 123.0
    env._last_upgrade_tier_sum = 77
    city, battle = start(env, [defender])
    battle.seed(42)
    for _ in range(30):
        result = env.step(battle.scripted_action_for_current_blue())
        assert env.observation_space.contains(result[0])
        if env.mode == env.MODE_GRID:
            break
    assert env.mode == env.MODE_GRID
    assert result[4]["battle_result"] == "victory"
    assert battle.last_battle_exp == 12
    assert_canonical(env, env.city_garrisons[city][0], "Берсерк")
    assert env.blue_team_state == hero
    assert tuple(env.grid_env.agent_pos) == hero_position
    assert (env.gold, env._current_mana_totals(), env.moves, env.turns) == resources
    assert env._last_upgrade_reward == 123.0 and env._last_upgrade_tier_sum == 77
    assert result[4].get("unit_upgrade_reward", 0) == 0
    assert result[1] == pytest.approx(env.battle_reward_win * env.battle_reward_scale)
