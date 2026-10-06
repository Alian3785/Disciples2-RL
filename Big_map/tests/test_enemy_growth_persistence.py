"""Enemy and capital-bot saves retain growth while expiring combat layers.

Focused deterministic fixtures only: no autonomous bot, training or evaluation.
"""
from copy import deepcopy
import json

import pytest

from battle_env import BattleEnv
from campaign_env import CampaignEnv


STATS = ("damage", "damage_secondary", "accuracy", "accuracy_secondary",
         "initiative_base", "armor", "max_health", "maxhp", "health", "hp")
PROGRESSION = ("Level", "exp_current", "exp_required", "exp_kill", "needaunit",
               "hero_level_stat_abilities", "campaign_lord_might_bonus_levels")


@pytest.fixture
def env():
    value = CampaignEnv(map_name="scroll_train", observation_version="local5",
                        scripted_capital_bot_enabled=False,
                        use_boss_starting_roster=False, log_enabled=False)
    value.reset(seed=42)
    yield value
    value.close()


def unit(env, name, team="red", position=1):
    return env._build_unit_from_data(env._find_unit_data_by_name(name), team, position)


def potions(env, target):
    for spec in env.POTION_ITEM_DEFINITIONS:
        if spec.get("duration") == "permanent":
            env._apply_permanent_potion_bonus_to_unit(target, spec)


def grown_control(target):
    control = deepcopy(target)
    battle = BattleEnv(log_enabled=False)
    battle._apply_exp_award_to_unit(control, 1)
    return control


@pytest.mark.parametrize("name,level", [("Герцог", 12), ("Герцог", 13),
                                       ("Герцог", 14), ("Теург", 9), ("Теург", 10)])
@pytest.mark.parametrize("elixirs", [False, True])
@pytest.mark.parametrize("route", ["player", "scripted_bot"])
def test_enemy_growth_survives_both_battle_save_routes(env, name, level, elixirs, route):
    original = unit(env, name)
    original["Level"] = level
    original["exp_current"] = original["exp_required"] - 1
    original["source_unit_id"] = "enemy-grown-identity"
    if elixirs:
        potions(env, original)
    original["hp"] = original["health"] = 37
    expected = grown_control(original)
    enemy_id = 7701
    env.enemy_team_states[enemy_id] = [deepcopy(original)]
    battle = BattleEnv(log_enabled=False)
    battle._init_with_custom_teams([original], [])
    runtime = next(u for u in battle.combined if u.get("name") == name)
    battle._apply_exp_award_to_unit(runtime, 1)
    runtime["initiative"] = 0
    if route == "player":
        env.battle_env = battle
        env._save_enemy_state_from_battle(enemy_id)
    else:
        previous = env.battle_env
        env._save_enemy_state_from_battle_env(enemy_id, battle)
        assert env.battle_env is previous
    saved = next(u for u in env.enemy_team_states[enemy_id] if u.get("name") == name)
    for key in (*STATS, *PROGRESSION, "campaign_stat_sources"):
        assert saved.get(key) == expected.get(key), key
    assert saved["source_unit_id"] == "enemy-grown-identity"
    assert saved["initiative"] == saved["initiative_base"]
    # The next encounter consumes persisted JSON, not a stale original roster.
    env.enemy_team_states[enemy_id] = json.loads(json.dumps(env.enemy_team_states[enemy_id]))
    next_battle = BattleEnv(log_enabled=False)
    next_battle._init_with_custom_teams(env.enemy_team_states[enemy_id], [])
    again = next(u for u in next_battle.combined if u.get("name") == name)
    for key in (*STATS, *PROGRESSION, "campaign_stat_sources"):
        assert again.get(key) == expected.get(key), key


@pytest.mark.parametrize("elixirs", [False, True])
def test_enemy_map_debuff_and_settlement_armor_rebase_after_growth(env, elixirs):
    original = unit(env, "Герцог")
    original.update(Level=14, exp_current=original["exp_required"] - 1)
    if elixirs:
        potions(env, original)
    original["health"] = original["hp"] = 43
    expected = grown_control(original)
    enemy_id = 7702
    env.enemy_team_states[enemy_id] = [deepcopy(original)]
    env.enemy_map_spell_effects[enemy_id] = {"test-debuff"}
    summary = {"active_spell_ids": ["test-debuff"], "debuff_types": [],
               "armor_delta": -7, "damage_multiplier": 0.5,
               "initiative_multiplier": 0.5, "accuracy_multiplier": 0.8}
    env._enemy_stack_spell_effect_summary = lambda enemy: summary
    env._recompute_enemy_stack_spell_effects(enemy_id)
    prepared = deepcopy(env.enemy_team_states[enemy_id])
    prepared[0]["armor"] += 20
    prepared[0]["settlement_armor_bonus"] = 20
    battle = BattleEnv(log_enabled=False)
    battle._init_with_custom_teams(prepared, [])
    runtime = next(u for u in battle.combined if u.get("name") == "Герцог")
    battle._apply_exp_award_to_unit(runtime, 1)
    env.battle_env = battle
    env._save_enemy_state_from_battle(enemy_id)
    saved = next(u for u in env.enemy_team_states[enemy_id] if u.get("name") == "Герцог")
    assert saved["damage"] == int(round(expected["damage"] * 0.5))
    assert saved["initiative_base"] == int(round(expected["initiative_base"] * 0.5))
    assert saved["armor"] == max(0, expected["armor"] - 7)
    assert saved["accuracy"] == int(round(expected["accuracy"] * 0.8))
    assert saved["health"] == expected["health"]
    assert saved["campaign_stat_sources"] == expected["campaign_stat_sources"]
    env.enemy_map_spell_effects.pop(enemy_id)
    env._recompute_enemy_stack_spell_effects(enemy_id)
    for key in STATS:
        assert saved.get(key) == expected.get(key), key


def test_scripted_bot_evolution_retains_elixirs_and_provenance(env):
    source = unit(env, "Одержимый", "blue", 7)
    potions(env, source)
    source["source_unit_id"] = "bot-fighter"
    source["health"] = source["hp"] = 13
    expected = unit(env, "Берсерк", "blue", 7)
    potions(env, expected)
    battle = BattleEnv(log_enabled=False)
    battle.combined = [source]
    battle.last_levelups = [source["name"]]
    battle.last_levelup_units = [source]
    assert env._apply_scripted_capital_bot_levelups(battle) == 1
    assert source["name"] == "Берсерк"
    assert source["source_unit_id"] == "bot-fighter"
    for key in (*STATS, "campaign_stat_sources", "campaign_permanent_potions"):
        assert source.get(key) == expected.get(key), key
    assert env._apply_scripted_capital_bot_levelups(battle) == 0
    env._save_scripted_capital_bot_state_from_battle(battle)
    saved = env.scripted_capital_bot_team_state[0]
    for key in (*STATS, "campaign_stat_sources", "campaign_permanent_potions"):
        assert saved.get(key) == expected.get(key), key


@pytest.mark.parametrize("elixirs", [False, True])
def test_bot_save_keeps_grown_stats_and_removes_active_combat_effects(env, elixirs):
    source = unit(env, "Герцог", "blue", 7)
    source.update(Level=14, exp_current=source["exp_required"] - 1)
    if elixirs:
        potions(env, source)
    source["health"] = source["hp"] = 31
    expected = grown_control(source)
    runtime = deepcopy(expected)
    battle = BattleEnv(log_enabled=False)
    battle.combined = [runtime]
    caster = {"name": "effect", "position": 1, "team": "red",
              "attack_type_secondary": "", "unit_type": "Travnitsa"}
    battle._apply_travnitsa_buff(caster, runtime)
    battle._apply_tiamat_damage_debuff(caster, runtime)
    battle._apply_hermit_initiative_slow(caster, runtime)
    battle._apply_teurg_armor_shred(caster, runtime)
    env._save_scripted_capital_bot_state_from_battle(battle)
    saved = env.scripted_capital_bot_team_state[0]
    for key in (*STATS, *PROGRESSION, "campaign_stat_sources"):
        assert saved.get(key) == expected.get(key), key
    assert not saved.get("powerup") and not saved.get("teamated")
    assert not saved.get("hermited") and not saved.get("shattered_armor")
    assert saved["initiative"] == saved["initiative_base"]


@pytest.mark.parametrize("elixirs", [False, True])
def test_enemy_settlement_only_armor_never_becomes_permanent(env, elixirs):
    original = unit(env, "Герцог")
    original.update(Level=14, exp_current=original["exp_required"] - 1)
    if elixirs:
        potions(env, original)
    expected = grown_control(original)
    enemy_id = 7703
    env.enemy_team_states[enemy_id] = [deepcopy(original)]
    prepared = [deepcopy(original)]
    env._resolve_settlement_defender_armor_bonus = lambda enemy: (20, "settlement_level_2")
    env._apply_settlement_defender_armor_bonus(enemy_id, prepared)
    battle = BattleEnv(log_enabled=False)
    battle._init_with_custom_teams(prepared, [])
    runtime = next(u for u in battle.combined if u.get("name") == "Герцог")
    battle._apply_exp_award_to_unit(runtime, 1)
    assert runtime["armor"] == expected["armor"] + 20
    env.battle_env = battle
    env._save_enemy_state_from_battle(enemy_id)
    saved = next(u for u in env.enemy_team_states[enemy_id] if u.get("name") == "Герцог")
    for key in (*STATS, "campaign_stat_sources"):
        assert saved.get(key) == expected.get(key), key


@pytest.mark.parametrize("route", ["bot", "garrison"])
def test_saved_temporary_layers_cannot_reactivate_on_next_growth(env, route):
    source = unit(env, "Герцог", "blue", 7)
    source.update(Level=13, exp_current=source["exp_required"] - 1)
    potions(env, source)
    source["health"] = source["hp"] = 39
    original = deepcopy(source)
    for spec in env.POTION_ITEM_DEFINITIONS:
        if spec.get("duration") == "temporary":
            env._active_potion_positions(spec["name"]).add(7)
    env._apply_active_blue_potion_effects([source])
    env._blue_stack_spell_effect_summary = lambda: {
        "active_spell_ids": ["test"], "damage_multiplier": 1.25,
        "initiative_multiplier": 1.2, "accuracy_multiplier": 1.15,
        "armor_delta": 17, "health_delta": 50, "resistance_types": ["Mind"],
    }
    env._apply_active_blue_support_spell_effects([source])
    battle = BattleEnv(log_enabled=False)
    battle.combined = [source]
    env.battle_env = battle
    if route == "bot":
        env._save_scripted_capital_bot_state_from_battle(battle)
        saved = env.scripted_capital_bot_team_state[0]
    else:
        saved = env._saved_city_battle_team("blue")[0]
    for key in STATS:
        assert saved[key] == original[key], key
    assert saved["resistance"] == original["resistance"]
    assert not any(key.startswith(("campaign_potion_", "campaign_map_spell_"))
                   for key in saved)
    expected = grown_control(original)
    next_battle = BattleEnv(log_enabled=False)
    next_battle._init_with_custom_teams([], [saved])
    again = next(u for u in next_battle.combined if u.get("name") == "Герцог")
    next_battle._apply_exp_award_to_unit(again, 1)
    for key in (*STATS, *PROGRESSION, "campaign_stat_sources"):
        assert again.get(key) == expected.get(key), key
