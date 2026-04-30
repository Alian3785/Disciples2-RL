from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv
from enemy_configs import ENEMY_CONFIGS


def _make_legions_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (10, 10),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[20] = deepcopy(ENEMY_CONFIGS[20])
    return env


def _make_undead_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=4)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (10, 10),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[20] = deepcopy(ENEMY_CONFIGS[20])
    return env


def _make_empire_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=1)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (10, 10),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[20] = deepcopy(ENEMY_CONFIGS[20])
    return env


def _make_mountain_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=3)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (10, 10),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[20] = deepcopy(ENEMY_CONFIGS[20])
    return env


def _make_elves_env() -> CampaignEnv:
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=5)
    env.reset(seed=123)
    env.grid_env.agent_pos = (3, 3)
    env.grid_env.enemy_positions = {
        10: (4, 3),
        20: (10, 10),
    }
    env.grid_env.enemies_alive = {10: True, 20: True}
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[20] = deepcopy(ENEMY_CONFIGS[20])
    return env


def _enable_typeoflord_two(env: CampaignEnv) -> CampaignEnv:
    env.typeoflord = 2
    env.Typeoflord = 2
    return env


def _spell_action_index(env: CampaignEnv, spell_key: str) -> int:
    for idx, candidate_spec in enumerate(env._current_map_offensive_spell_action_specs()):
        if str(candidate_spec.get("id", "") or "") == str(spell_key):
            return env.grid_legion_damage_spell_action_start + idx
    raise AssertionError(f"spell action not found: {spell_key}")


def _support_spell_action_index(env: CampaignEnv, spell_key: str) -> int:
    for idx, candidate_spec in enumerate(env._current_map_support_spell_action_specs()):
        if str(candidate_spec.get("id", "") or "") == str(spell_key):
            return env.grid_map_support_spell_action_start + idx
    raise AssertionError(f"support spell action not found: {spell_key}")


def _scroll_spell_action_index(env: CampaignEnv, spell_key: str) -> int:
    for idx, slot_entry in enumerate(env.scroll_cast_slot_entries()):
        if str(slot_entry.get("spell_id", "") or "") == str(spell_key):
            return env.grid_scroll_cast_action_start + idx
    raise AssertionError(f"scroll spell action not found: {spell_key}")


def _battle_target_action(position: int) -> int:
    return int(position) - 1


def _set_spell_use_mana(
    env: CampaignEnv,
    spell_key: str,
    *,
    multiplier: float = 1.0,
) -> dict[str, float]:
    for mana_kind in env.MANA_KIND_ORDER:
        attr_name = env.MANA_ATTR_BY_KIND[mana_kind]
        setattr(env, attr_name, 0.0)

    spell = env.active_spells[spell_key]
    costs = env._get_spell_use_costs(spell)
    for mana_kind, amount in costs.items():
        attr_name = env.MANA_ATTR_BY_KIND[mana_kind]
        setattr(env, attr_name, float(amount) * float(multiplier))
    return costs


def _grant_spell_use_mana(
    env: CampaignEnv,
    spell_keys: list[str],
    *,
    multiplier: float = 1.0,
) -> dict[str, float]:
    totals = {mana_kind: 0.0 for mana_kind in env.MANA_KIND_ORDER}
    for mana_kind in env.MANA_KIND_ORDER:
        attr_name = env.MANA_ATTR_BY_KIND[mana_kind]
        setattr(env, attr_name, 0.0)

    for spell_key in spell_keys:
        spell = env.active_spells[spell_key]
        for mana_kind, amount in env._get_spell_use_costs(spell).items():
            totals[mana_kind] = float(totals.get(mana_kind, 0.0)) + float(amount)

    for mana_kind, amount in totals.items():
        attr_name = env.MANA_ATTR_BY_KIND[mana_kind]
        setattr(env, attr_name, float(amount) * float(multiplier))
    return totals


def _enemy_unit(
    *,
    position: int,
    hp: float,
    max_hp: float,
    name: str,
    unit_type: str = "Warrior",
    immunity: list[str] | None = None,
    resistance: list[str] | None = None,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    unit = {
        "name": name,
        "team": "red",
        "position": int(position),
        "stand": "ahead",
        "health": float(hp),
        "hp": float(hp),
        "max_health": float(max_hp),
        "maxhp": float(max_hp),
        "unit_type": unit_type,
        "immunity": list(immunity or []),
        "resistance": list(resistance or []),
    }
    if extra_fields:
        unit.update(dict(extra_fields))
    return unit


def _enemy_hp_feature(env: CampaignEnv, enemy_index: int, unit_slot: int) -> float:
    enemy_obs = env._build_enemy_grid_obs()
    slot_size = env.grid_enemy_unit_feature_size
    enemy_block_size = 2 + env.grid_enemy_unit_slots * slot_size
    offset = enemy_index * enemy_block_size + 2 + unit_slot * slot_size + slot_size - 1
    return float(enemy_obs[offset])


def test_legion_damage_spell_actions_require_legions_learned_spell_and_mana():
    env = _make_legions_env()
    spell_key = "lod_d2_s003"
    action = _spell_action_index(env, spell_key)

    assert bool(env.compute_action_mask()[action]) is False

    env.active_spells[spell_key]["learned"] = 1
    assert bool(env.compute_action_mask()[action]) is False

    _set_spell_use_mana(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is True

    non_legions_env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=1)
    non_legions_env.reset(seed=123)
    non_legions_env.grid_env.agent_pos = (3, 3)
    non_legions_env.grid_env.enemy_positions = {10: (4, 3)}
    non_legions_env.grid_env.enemies_alive = {10: True}
    non_legions_env.grid_env.obstacle_positions = set()

    assert bool(non_legions_env.compute_action_mask()[action]) is False


def test_area_damage_spell_hits_nearest_enemy_stack_and_updates_enemy_obs():
    env = _make_legions_env()
    spell_key = "lod_d2_s019"
    env.active_spells[spell_key]["learned"] = 1
    costs = _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=80.0, max_hp=80.0, name="Target A"),
        _enemy_unit(position=2, hp=70.0, max_hp=70.0, name="Target B", unit_type="Mage"),
    ]

    action = _spell_action_index(env, spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_applied"] is True
    assert info["spell_cast_executed"] is True
    assert info["spell_cast_reward"] == pytest.approx(env.reward_spell_cast)
    assert info["target_enemy_id"] == 10
    assert info["spell_damage_units_hit"] == 2
    assert info["spell_damage_units_defeated"] == 0
    assert info["spell_damage_units_blocked"] == 0
    assert info["blue_exp_reward"] == pytest.approx(0.0)
    assert info["blue_exp_raw"] == pytest.approx(0.0)
    assert info["spell_damage_total"] == pytest.approx(120.0)
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(20.0)
    assert env.enemy_team_states[10][1]["health"] == pytest.approx(10.0)
    assert env.infernal_mana == pytest.approx(0.0)
    assert env.life_mana == pytest.approx(0.0)
    assert env.death_mana == pytest.approx(0.0)
    assert costs["infernal"] == pytest.approx(400.0)
    assert costs["life"] == pytest.approx(200.0)
    assert costs["death"] == pytest.approx(200.0)
    assert _enemy_hp_feature(env, enemy_index=0, unit_slot=0) == pytest.approx(20.0 / 80.0)
    assert _enemy_hp_feature(env, enemy_index=0, unit_slot=1) == pytest.approx(10.0 / 70.0)


def test_legion_damage_spell_is_locked_per_turn_and_unlocks_next_turn():
    env = _make_legions_env()
    spell_key = "lod_d2_s003"
    env.active_spells[spell_key]["learned"] = 1
    costs = _set_spell_use_mana(env, spell_key, multiplier=2.0)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Durable target"),
    ]

    action = _spell_action_index(env, spell_key)
    _, _, _, _, first_info = env.step(action)
    mana_after_first_cast = env.infernal_mana
    hp_after_first_cast = env.enemy_team_states[10][0]["health"]

    assert first_info["spell_cast_applied"] is True
    assert first_info["spell_cast_executed"] is True
    assert first_info["spell_cast_reward"] == pytest.approx(env.reward_spell_cast)
    assert first_info["spell_cast_used_this_turn"] is False
    assert mana_after_first_cast == pytest.approx(float(costs["infernal"]))
    assert hp_after_first_cast == pytest.approx(85.0)
    assert bool(env.compute_action_mask()[action]) is False

    _, _, _, _, second_info = env.step(action)

    assert second_info["spell_cast_applied"] is False
    assert second_info["spell_cast_executed"] is False
    assert second_info["spell_cast_reward"] == pytest.approx(0.0)
    assert second_info["spell_cast_used_this_turn"] is True
    assert env.infernal_mana == pytest.approx(mana_after_first_cast)
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(hp_after_first_cast)

    env._advance_turns(1)

    assert bool(env.compute_action_mask()[action]) is True


def test_typeoflord_two_allows_same_legion_damage_spell_twice_per_turn():
    env = _enable_typeoflord_two(_make_legions_env())
    spell_key = "lod_d2_s003"
    env.active_spells[spell_key]["learned"] = 1
    costs = _set_spell_use_mana(env, spell_key, multiplier=3.0)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Durable target"),
    ]

    action = _spell_action_index(env, spell_key)

    _, _, _, _, first_info = env.step(action)
    hp_after_first_cast = env.enemy_team_states[10][0]["health"]
    mana_after_first_cast = env.infernal_mana

    assert first_info["spell_cast_executed"] is True
    assert first_info["spell_cast_used_this_turn"] is False
    assert hp_after_first_cast == pytest.approx(85.0)
    assert mana_after_first_cast == pytest.approx(float(costs["infernal"]) * 2.0)
    assert bool(env.compute_action_mask()[action]) is True

    _, _, _, _, second_info = env.step(action)
    hp_after_second_cast = env.enemy_team_states[10][0]["health"]
    mana_after_second_cast = env.infernal_mana

    assert second_info["spell_cast_executed"] is True
    assert second_info["spell_cast_used_this_turn"] is False
    assert hp_after_second_cast == pytest.approx(70.0)
    assert mana_after_second_cast == pytest.approx(float(costs["infernal"]))
    assert bool(env.compute_action_mask()[action]) is False

    _, _, _, _, third_info = env.step(action)

    assert third_info["spell_cast_executed"] is False
    assert third_info["spell_cast_used_this_turn"] is True
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(hp_after_second_cast)
    assert env.infernal_mana == pytest.approx(mana_after_second_cast)


def test_legion_damage_spell_can_defeat_enemy_stack_without_battle():
    env = _make_legions_env()
    spell_key = "lod_d2_s003"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=10.0, max_hp=10.0, name="Weak target"),
    ]

    action = _spell_action_index(env, spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["battle_triggered"] is False
    assert info["spell_enemy_defeated"] is True
    assert info["spell_cast_executed"] is True
    assert env.grid_env.enemies_alive[10] is False
    assert reward == pytest.approx(env.reward_spell_cast + env._compute_enemy_defeat_reward(10))
    assert info["blue_exp_reward"] == pytest.approx(0.0)
    assert info["blue_exp_raw"] == pytest.approx(0.0)


def test_legion_damage_spell_respects_immunity_and_resistance():
    env = _make_legions_env()
    spell_key = "lod_d2_s008"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Immune", immunity=["Fire"]),
        _enemy_unit(position=2, hp=100.0, max_hp=100.0, name="Resistant", resistance=["Fire"]),
        _enemy_unit(position=3, hp=100.0, max_hp=100.0, name="Vulnerable"),
    ]

    action = _spell_action_index(env, spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_damage_type"] == "Fire"
    assert info["spell_cast_executed"] is True
    assert info["spell_cast_reward"] == pytest.approx(env.reward_spell_cast)
    assert info["spell_damage_units_hit"] == 1
    assert info["spell_damage_units_blocked"] == 2
    assert info["spell_damage_units_defeated"] == 0
    assert info["spell_damage_total"] == pytest.approx(30.0)
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[10][1]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[10][2]["health"] == pytest.approx(70.0)


def test_enemy_units_regenerate_five_percent_on_player_territory_and_fifteen_percent_elsewhere():
    env = _make_legions_env()
    blue_territory_tile = tuple(env.legions_territory_order[1])
    ordinary_tile = (40, 40)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=50.0, max_hp=100.0, name="Wounded"),
        _enemy_unit(position=2, hp=96.0, max_hp=100.0, name="Almost full"),
        _enemy_unit(position=3, hp=0.0, max_hp=100.0, name="Dead"),
    ]
    env.enemy_team_states[20] = [
        _enemy_unit(position=1, hp=30.0, max_hp=100.0, name="Ordinary land wounded"),
    ]
    env.enemy_team_states[30] = [
        _enemy_unit(position=1, hp=40.0, max_hp=100.0, name="Defeated stack wounded"),
    ]
    env.grid_env.enemy_positions = {
        10: blue_territory_tile,
        20: ordinary_tile,
        30: (41, 40),
    }
    env.grid_env.enemies_alive = {10: True, 20: True, 30: False}

    env._advance_turns(1)

    assert blue_territory_tile in env.legions_territory_tile_set
    assert ordinary_tile not in env.legions_territory_tile_set
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(55.0)
    assert env.enemy_team_states[10][1]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[10][2]["health"] == pytest.approx(0.0)
    assert env.enemy_team_states[20][0]["health"] == pytest.approx(45.0)
    assert env.enemy_team_states[30][0]["health"] == pytest.approx(40.0)


def test_legion_debuff_spell_actions_require_legions_learned_spell_and_mana():
    env = _make_legions_env()
    spell_key = "lod_d2_s015"
    action = _spell_action_index(env, spell_key)

    assert bool(env.compute_action_mask()[action]) is False

    env.active_spells[spell_key]["learned"] = 1
    assert bool(env.compute_action_mask()[action]) is False

    _set_spell_use_mana(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is True


def test_legion_debuff_spells_stack_until_next_turn_and_then_expire():
    env = _make_legions_env()
    spell_keys = [
        "lod_d2_s005",
        "lod_d2_s009",
        "lod_d2_s010",
        "lod_d2_s015",
        "lod_d2_s016",
        "lod_d2_s020",
    ]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=120.0,
            max_hp=120.0,
            name="Debuffed target",
            extra_fields={
                "armor": 65,
                "damage": 100,
                "damage_secondary": 20,
                "initiative": 60,
                "initiative_base": 60,
                "accuracy": 80,
                "accuracy_secondary": 40,
            },
        ),
    ]

    for spell_key in spell_keys:
        _, reward, terminated, truncated, info = env.step(_spell_action_index(env, spell_key))
        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(env.reward_spell_cast)
        assert info["spell_cast_executed"] is True
        assert info["spell_cast_action"] is True

    target = env.enemy_team_states[10][0]
    assert target["armor"] == 15
    assert target["damage"] == 85
    assert target["damage_secondary"] == 17
    assert target["initiative_base"] == 34
    assert target["initiative"] == 34
    assert target["accuracy"] == 40
    assert target["accuracy_secondary"] == 20

    env._advance_turns(1)

    target = env.enemy_team_states[10][0]
    assert target["armor"] == 65
    assert target["damage"] == 100
    assert target["damage_secondary"] == 20
    assert target["initiative_base"] == 60
    assert target["initiative"] == 60
    assert target["accuracy"] == 80
    assert target["accuracy_secondary"] == 40


def test_legion_debuff_spell_carries_into_battle_for_nearest_enemy_stack():
    env = _make_legions_env()
    spell_key = "lod_d2_s015"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=100.0,
            max_hp=100.0,
            name="Acc target",
            extra_fields={
                "armor": 10,
                "damage": 50,
                "damage_secondary": 0,
                "initiative": 40,
                "initiative_base": 40,
                "accuracy": 80,
                "accuracy_secondary": 60,
            },
        ),
    ]

    _, _, _, _, info = env.step(_spell_action_index(env, spell_key))

    assert info["spell_cast_executed"] is True
    assert info["spell_debuff_type"] == "accuracy"
    assert info["spell_units_affected"] == 1
    assert info["spell_effect_accuracy_multiplier"] == pytest.approx(0.75)

    env._init_battle(10)
    red_units = [
        u
        for u in env.battle_env.combined
        if u.get("team") == "red" and float(u.get("max_health", 0) or 0) > 0
    ]

    assert len(red_units) == 1
    assert red_units[0]["accuracy"] == 60
    assert red_units[0]["accuracy_secondary"] == 45


def test_legion_map_spell_skips_ruins_and_city_garrisons_when_choosing_target():
    env = _make_legions_env()
    env.grid_env.enemy_positions = {
        70: (4, 3),  # ruins
        22: (5, 3),  # city stack
        10: (6, 3),  # valid field target
    }
    env.grid_env.enemies_alive = {70: True, 22: True, 10: True}
    env.grid_env.obstacle_positions = set()

    spell_key = "lod_d2_s003"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Field target"),
    ]
    env.enemy_team_states[22] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="City target"),
    ]
    env.enemy_team_states[70] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Ruin target"),
    ]

    action = _spell_action_index(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is True

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_executed"] is True
    assert info["target_enemy_id"] == 10
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(85.0)
    assert env.enemy_team_states[22][0]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[70][0]["health"] == pytest.approx(100.0)


def test_legion_summon_spell_can_target_ruins_and_city_garrisons():
    env = _make_legions_env()
    env.grid_env.enemy_positions = {
        70: (4, 3),  # ruins
        22: (5, 3),  # city stack
    }
    env.grid_env.enemies_alive = {70: True, 22: True}
    env.grid_env.obstacle_positions = set()

    summon_spell_key = "lod_d2_s001"
    damage_spell_key = "lod_d2_s003"
    env.active_spells[summon_spell_key]["learned"] = 1
    env.active_spells[damage_spell_key]["learned"] = 1
    _grant_spell_use_mana(env, [summon_spell_key, damage_spell_key])
    env.enemy_team_states[70] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Ruin target"),
    ]
    env.enemy_team_states[22] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="City target"),
    ]

    summon_action = _spell_action_index(env, summon_spell_key)
    damage_action = _spell_action_index(env, damage_spell_key)
    assert bool(env.compute_action_mask()[summon_action]) is True
    assert bool(env.compute_action_mask()[damage_action]) is False

    _, reward, terminated, truncated, info = env.step(summon_action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["battle_triggered"] is True
    assert info["mode"] == "battle"
    assert info["target_enemy_id"] == 70
    assert info["battle_context_kind"] == "summon_spell"
    assert info["spell_summon_unit_name"] == "Адская гончая"
    assert any(
        unit.get("team") == "blue" and unit.get("name") == "Адская гончая"
        for unit in env.battle_env.combined
    )


def test_legion_summon_spell_victory_defeats_enemy_without_blue_exp():
    env = _make_legions_env()
    spell_key = "lod_d2_s006"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    initial_blue_state = deepcopy(env.blue_team_state)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=10.0,
            max_hp=10.0,
            name="Weak target",
            extra_fields={
                "damage": 1,
                "accuracy": 0,
                "initiative": 1,
                "initiative_base": 1,
                "exp_current": 0,
                "exp_required": 200,
                "exp_kill": 40,
            },
        ),
    ]

    _, _, _, _, cast_info = env.step(_spell_action_index(env, spell_key))

    assert cast_info["battle_triggered"] is True
    summoned_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and unit.get("name") == "Белиарх"
    )
    summoned_unit["damage"] = 999
    summoned_unit["accuracy"] = 100
    summoned_unit["initiative"] = 999
    summoned_unit["initiative_base"] = 999

    _, _, terminated, truncated, info = env.step(_battle_target_action(1))

    assert env.mode == env.MODE_GRID
    assert terminated is False
    assert truncated is False
    assert info["battle_result"] == "victory"
    assert info["battle_context_kind"] == "summon_spell"
    assert info["blue_exp_reward"] == pytest.approx(0.0)
    assert info["blue_exp_raw"] == pytest.approx(0.0)
    assert info["enemy_defeat_reward"] == pytest.approx(env.reward_defeat_enemy)
    assert env.grid_env.enemies_alive[10] is False
    assert all(
        float(after.get("health", 0) or 0) == pytest.approx(float(before.get("health", 0) or 0))
        for before, after in zip(initial_blue_state, env.blue_team_state)
    )


def test_summon_victory_over_city_captures_city_immediately():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    city_enemy_id = 68
    already_defeated_city_enemy_id = 34
    other_enemy_id = 1
    city_tile = tuple(env.grid_env.enemy_positions[city_enemy_id])
    env.grid_env.agent_pos = (city_tile[0] - 1, city_tile[1])
    env.grid_env.enemy_positions = {
        already_defeated_city_enemy_id: city_tile,
        city_enemy_id: city_tile,
        other_enemy_id: (0, 0),
    }
    env.grid_env.enemies_alive = {
        already_defeated_city_enemy_id: False,
        city_enemy_id: True,
        other_enemy_id: True,
    }
    env.grid_env.obstacle_positions = set()
    env.enemy_team_states[city_enemy_id] = [
        _enemy_unit(
            position=1,
            hp=10.0,
            max_hp=10.0,
            name="Weak city defender",
            extra_fields={
                "damage": 1,
                "accuracy": 0,
                "initiative": 1,
                "initiative_base": 1,
            },
        ),
    ]

    spell_key = "lod_d2_s006"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)

    _, _, _, _, cast_info = env.step(_spell_action_index(env, spell_key))
    assert cast_info["battle_triggered"] is True
    assert cast_info["target_enemy_id"] == city_enemy_id

    summoned_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and float(unit.get("max_health", 0) or 0) > 0
    )
    summoned_unit["damage"] = 999
    summoned_unit["accuracy"] = 100
    summoned_unit["initiative"] = 999
    summoned_unit["initiative_base"] = 999

    _, _, terminated, truncated, info = env.step(_battle_target_action(1))

    city_name = env._objective_city_for_enemy(city_enemy_id)
    settlement_name = env.legions_settlement_territory_source_name_by_enemy_id[
        city_enemy_id
    ]

    assert env.mode == env.MODE_GRID
    assert terminated is False
    assert truncated is False
    assert env.grid_env.enemies_alive[city_enemy_id] is False
    assert city_name in env.captured_objective_cities
    assert settlement_name in env.legions_active_settlement_territory_capture_turn_by_name
    assert info["captured_objective_cities"] == [city_name]
    assert info["legions_settlement_territories_activated"] == [settlement_name]
    assert info["final_objective_reward"] == pytest.approx(env.reward_all_enemies)
    assert "summon_city_capture_deferred" not in info
    assert "pending_summon_city_capture_count" not in info

    upgrade_action = (
        env.grid_settlement_upgrade_action_start
        + env.settlement_upgrade_names.index(settlement_name)
    )
    env.gold = 999.0
    assert bool(env.compute_action_mask()[upgrade_action]) is True


def test_legion_summon_spell_defeat_persists_enemy_damage_and_enemy_exp():
    env = _make_legions_env()
    spell_key = "lod_d2_s001"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=100.0,
            max_hp=100.0,
            name="Strong target",
            extra_fields={
                "damage": 999,
                "accuracy": 100,
                "initiative": 10,
                "initiative_base": 10,
                "exp_current": 0,
                "exp_required": 200,
                "exp_kill": 50,
            },
        ),
    ]

    _, _, _, _, cast_info = env.step(_spell_action_index(env, spell_key))

    assert cast_info["battle_triggered"] is True
    summoned_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and unit.get("name") == "Адская гончая"
    )
    summoned_unit["damage"] = 20
    summoned_unit["accuracy"] = 100
    summoned_unit["initiative"] = 999
    summoned_unit["initiative_base"] = 999
    summoned_unit["health"] = 1
    summoned_unit["hp"] = 1
    summoned_unit["max_health"] = 1
    summoned_unit["maxhp"] = 1

    _, _, terminated, truncated, info = env.step(_battle_target_action(1))

    assert env.mode == env.MODE_GRID
    assert terminated is False
    assert truncated is False
    assert info["battle_result"] == "defeat"
    assert info["summon_enemy_state_saved"] is True
    assert env.grid_env.enemies_alive[10] is True
    assert 0.0 < float(env.enemy_team_states[10][0]["health"]) < 100.0
    assert env.enemy_team_states[10][0]["exp_current"] == pytest.approx(90.0)


def test_summon_sets_same_turn_hero_battle_bonus_until_main_stack_enters_battle():
    env = _make_legions_env()
    spell_key = "lod_d2_s001"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=100.0,
            max_hp=100.0,
            name="Guard",
            extra_fields={
                "damage": 100,
                "accuracy": 100,
                "initiative": 50,
                "initiative_base": 50,
                "exp_current": 0,
                "exp_required": 200,
                "exp_kill": 40,
            },
        ),
    ]

    _, _, _, _, cast_info = env.step(_spell_action_index(env, spell_key))
    assert cast_info["battle_triggered"] is True

    summoned_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and float(unit.get("max_health", 0) or 0) > 0
    )
    summoned_unit["damage"] = 20
    summoned_unit["accuracy"] = 100
    summoned_unit["initiative"] = 999
    summoned_unit["initiative_base"] = 999
    summoned_unit["health"] = 1
    summoned_unit["hp"] = 1
    summoned_unit["max_health"] = 1
    summoned_unit["maxhp"] = 1

    _, _, _, _, summon_battle_info = env.step(_battle_target_action(1))

    assert summon_battle_info["battle_result"] == "defeat"
    assert env.mode == env.MODE_GRID
    assert env.summon_hero_battle_bonus_pending is True

    _, _, terminated, truncated, hero_battle_info = env.step(3)

    assert terminated is False
    assert truncated is False
    assert hero_battle_info["battle_triggered"] is True
    assert hero_battle_info["mode"] == "battle"
    assert hero_battle_info["summon_hero_battle_bonus_applied"] is True
    assert hero_battle_info["summon_hero_battle_bonus"] == pytest.approx(
        env.reward_summon_hero_battle_engage
    )
    assert hero_battle_info["summon_hero_battle_bonus_pending"] is False
    assert env.summon_hero_battle_bonus_pending is False


def test_same_turn_hero_battle_bonus_requires_same_enemy_that_fought_the_summon():
    env = _make_legions_env()
    spell_key = "lod_d2_s001"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.grid_env.enemy_positions[20] = (3, 4)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=100.0,
            max_hp=100.0,
            name="Summon target",
            extra_fields={
                "damage": 100,
                "accuracy": 100,
                "initiative": 50,
                "initiative_base": 50,
            },
        ),
    ]
    env.enemy_team_states[20] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Other target"),
    ]

    _, _, _, _, cast_info = env.step(_spell_action_index(env, spell_key))
    assert cast_info["battle_triggered"] is True
    assert cast_info["target_enemy_id"] == 10

    summoned_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and float(unit.get("max_health", 0) or 0) > 0
    )
    summoned_unit["health"] = 1
    summoned_unit["hp"] = 1
    summoned_unit["max_health"] = 1
    summoned_unit["maxhp"] = 1
    summoned_unit["initiative"] = 999
    summoned_unit["initiative_base"] = 999
    summoned_unit["accuracy"] = 100

    _, _, _, _, summon_battle_info = env.step(_battle_target_action(1))

    assert summon_battle_info["battle_result"] == "defeat"
    assert env.summon_hero_battle_bonus_pending is True

    _, _, terminated, truncated, other_enemy_battle_info = env.step(1)

    assert terminated is False
    assert truncated is False
    assert other_enemy_battle_info["battle_triggered"] is True
    assert other_enemy_battle_info["enemy_id"] == 20
    assert other_enemy_battle_info["summon_hero_battle_bonus_applied"] is False
    assert other_enemy_battle_info["summon_hero_battle_bonus"] == pytest.approx(0.0)
    assert other_enemy_battle_info["summon_hero_battle_bonus_pending"] is True
    assert env.summon_hero_battle_bonus_pending is True


@pytest.mark.parametrize(
    ("make_env", "spell_key", "summon_unit_name", "requires_typeoflord_two"),
    [
        (_make_empire_env, "emp_d2_s008", "Оживший доспех", False),
        (_make_empire_env, "emp_d2_s016", "Голем", False),
        (_make_mountain_env, "mcl_d2_s005", "Рух", False),
        (_make_mountain_env, "mcl_d2_s011", "Валькирия", False),
        (_make_mountain_env, "mcl_d2_s021", "Каменный предок", True),
        (_make_undead_env, "und_d2_s001", "Скелет", False),
        (_make_undead_env, "und_d2_s006", "Темный энт", False),
        (_make_undead_env, "und_d2_s011", "Кошмар", False),
        (_make_undead_env, "und_d2_s021", "Танатос", True),
        (_make_elves_env, "elf_d2_s002", "Малый энт", False),
        (_make_elves_env, "elf_d2_s007", "Энт", False),
        (_make_elves_env, "elf_d2_s012", "Великий энт", False),
        (_make_elves_env, "elf_d2_s021", "Буйный энт", True),
    ],
)
def test_non_legion_summon_spells_start_battle_with_only_summoned_unit(
    make_env,
    spell_key: str,
    summon_unit_name: str,
    requires_typeoflord_two: bool,
):
    env = make_env()
    if requires_typeoflord_two:
        _enable_typeoflord_two(env)

    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=50.0, max_hp=50.0, name="Target"),
    ]

    action = _spell_action_index(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is True

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert env.mode == env.MODE_BATTLE
    assert info["spell_cast_executed"] is True
    assert info["battle_triggered"] is True
    assert info["battle_context_kind"] == "summon_spell"
    assert info["spell_summon_unit_name"] == summon_unit_name
    battle_blue_units = [
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and float(unit.get("max_health", 0) or 0) > 0
    ]
    assert len(battle_blue_units) == 1
    assert battle_blue_units[0]["name"] == summon_unit_name


def test_undead_damage_spell_actions_require_undead_learned_spell_and_mana():
    env = _make_undead_env()
    spell_key = "und_d2_s002"
    action = _spell_action_index(env, spell_key)

    assert bool(env.compute_action_mask()[action]) is False

    env.active_spells[spell_key]["learned"] = 1
    assert bool(env.compute_action_mask()[action]) is False

    _set_spell_use_mana(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is True

    non_undead_env = _make_legions_env()
    assert non_undead_env._can_cast_legion_damage_spell(spell_key) is False


def test_undead_area_damage_spell_hits_nearest_enemy_stack_only():
    env = _make_undead_env()
    spell_key = "und_d2_s015"
    env.active_spells[spell_key]["learned"] = 1
    costs = _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=80.0, max_hp=80.0, name="Target A"),
        _enemy_unit(position=2, hp=70.0, max_hp=70.0, name="Target B", unit_type="Mage"),
    ]
    env.enemy_team_states[20] = [
        _enemy_unit(position=1, hp=90.0, max_hp=90.0, name="Far target"),
    ]

    action = _spell_action_index(env, spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_applied"] is True
    assert info["spell_cast_executed"] is True
    assert info["target_enemy_id"] == 10
    assert info["spell_damage_units_hit"] == 2
    assert info["spell_damage_units_defeated"] == 0
    assert info["spell_damage_units_blocked"] == 0
    assert info["spell_damage_total"] == pytest.approx(80.0)
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(40.0)
    assert env.enemy_team_states[10][1]["health"] == pytest.approx(30.0)
    assert env.enemy_team_states[20][0]["health"] == pytest.approx(90.0)
    assert costs["infernal"] == pytest.approx(225.0)
    assert costs["death"] == pytest.approx(225.0)


def test_undead_water_damage_spell_respects_immunity_and_resistance():
    env = _make_undead_env()
    spell_key = "und_d2_s003"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Immune", immunity=["Water"]),
        _enemy_unit(position=2, hp=100.0, max_hp=100.0, name="Resistant", resistance=["Water"]),
        _enemy_unit(position=3, hp=100.0, max_hp=100.0, name="Vulnerable"),
    ]

    action = _spell_action_index(env, spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_damage_type"] == "Water"
    assert info["spell_cast_executed"] is True
    assert info["spell_damage_units_hit"] == 1
    assert info["spell_damage_units_blocked"] == 2
    assert info["spell_damage_total"] == pytest.approx(15.0)
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[10][1]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[10][2]["health"] == pytest.approx(85.0)


def test_undead_debuff_spells_stack_until_next_turn_and_then_expire():
    env = _make_undead_env()
    spell_keys = [
        "und_d2_s004",
        "und_d2_s005",
        "und_d2_s009",
        "und_d2_s013",
        "und_d2_s016",
        "und_d2_s017",
    ]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=120.0,
            max_hp=120.0,
            name="Debuffed target",
            extra_fields={
                "armor": 60,
                "damage": 100,
                "damage_secondary": 20,
                "initiative": 60,
                "initiative_base": 60,
                "accuracy": 80,
                "accuracy_secondary": 40,
            },
        ),
    ]

    for spell_key in spell_keys:
        _, reward, terminated, truncated, info = env.step(_spell_action_index(env, spell_key))
        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(env.reward_spell_cast)
        assert info["spell_cast_executed"] is True
        assert info["spell_cast_action"] is True

    target = env.enemy_team_states[10][0]
    assert target["armor"] == 10
    assert target["damage"] == 57
    assert target["damage_secondary"] == 11
    assert target["initiative_base"] == 40
    assert target["initiative"] == 40
    assert target["accuracy"] == 58
    assert target["accuracy_secondary"] == 29

    env._advance_turns(1)

    target = env.enemy_team_states[10][0]
    assert target["armor"] == 60
    assert target["damage"] == 100
    assert target["damage_secondary"] == 20
    assert target["initiative_base"] == 60
    assert target["initiative"] == 60
    assert target["accuracy"] == 80
    assert target["accuracy_secondary"] == 40


def test_empire_damage_spell_actions_require_empire_learned_spell_and_mana():
    env = _make_empire_env()
    spell_key = "emp_d2_s004"
    action = _spell_action_index(env, spell_key)

    assert bool(env.compute_action_mask()[action]) is False

    env.active_spells[spell_key]["learned"] = 1
    assert bool(env.compute_action_mask()[action]) is False

    _set_spell_use_mana(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is True


def test_empire_air_damage_spell_respects_immunity_and_resistance():
    env = _make_empire_env()
    spell_key = "emp_d2_s014"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Immune", immunity=["Air"]),
        _enemy_unit(position=2, hp=100.0, max_hp=100.0, name="Resistant", resistance=["Air"]),
        _enemy_unit(position=3, hp=100.0, max_hp=100.0, name="Vulnerable"),
    ]

    _, reward, terminated, truncated, info = env.step(_spell_action_index(env, spell_key))

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_damage_type"] == "Air"
    assert info["spell_damage_units_hit"] == 1
    assert info["spell_damage_units_blocked"] == 2
    assert info["spell_damage_total"] == pytest.approx(50.0)
    assert env.enemy_team_states[10][2]["health"] == pytest.approx(50.0)


def test_mountain_water_aoe_spell_hits_nearest_stack_and_respects_immunity():
    env = _make_mountain_env()
    spell_key = "mcl_d2_s019"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=100.0, max_hp=100.0, name="Immune", immunity=["Water"]),
        _enemy_unit(position=2, hp=90.0, max_hp=90.0, name="Target A"),
        _enemy_unit(position=3, hp=80.0, max_hp=80.0, name="Target B"),
    ]
    env.enemy_team_states[20] = [
        _enemy_unit(position=1, hp=120.0, max_hp=120.0, name="Far target"),
    ]

    _, reward, terminated, truncated, info = env.step(_spell_action_index(env, spell_key))

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["target_enemy_id"] == 10
    assert info["spell_damage_type"] == "Water"
    assert info["spell_damage_units_hit"] == 2
    assert info["spell_damage_units_blocked"] == 1
    assert info["spell_damage_total"] == pytest.approx(120.0)
    assert env.enemy_team_states[10][0]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[10][1]["health"] == pytest.approx(30.0)
    assert env.enemy_team_states[10][2]["health"] == pytest.approx(20.0)
    assert env.enemy_team_states[20][0]["health"] == pytest.approx(120.0)


def test_elves_debuff_spells_stack_until_next_turn_and_then_expire():
    env = _make_elves_env()
    spell_keys = [
        "elf_d2_s010",
        "elf_d2_s015",
        "elf_d2_s018",
        "elf_d2_s019",
    ]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)
    env.enemy_team_states[10] = [
        _enemy_unit(
            position=1,
            hp=120.0,
            max_hp=120.0,
            name="Debuffed target",
            extra_fields={
                "armor": 90,
                "damage": 100,
                "damage_secondary": 20,
                "initiative": 60,
                "initiative_base": 60,
                "accuracy": 80,
                "accuracy_secondary": 40,
            },
        ),
    ]

    for spell_key in spell_keys:
        _, reward, terminated, truncated, info = env.step(_spell_action_index(env, spell_key))
        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(env.reward_spell_cast)
        assert info["spell_cast_executed"] is True

    target = env.enemy_team_states[10][0]
    assert target["armor"] == 10
    assert target["damage"] == 67
    assert target["damage_secondary"] == 13
    assert target["initiative_base"] == 51
    assert target["initiative"] == 51
    assert target["accuracy"] == 64
    assert target["accuracy_secondary"] == 32

    env._advance_turns(1)

    target = env.enemy_team_states[10][0]
    assert target["armor"] == 90
    assert target["damage"] == 100
    assert target["damage_secondary"] == 20
    assert target["initiative_base"] == 60
    assert target["initiative"] == 60
    assert target["accuracy"] == 80
    assert target["accuracy_secondary"] == 40


def test_elves_high_level_damage_spell_is_available_for_elves_only():
    env = _make_elves_env()
    spell_key = "elf_d2_s022"
    action = _spell_action_index(env, spell_key)

    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is False

    _enable_typeoflord_two(env)
    assert bool(env.compute_action_mask()[action]) is True

    non_elves_env = _make_legions_env()
    assert non_elves_env._can_cast_legion_damage_spell(spell_key) is False


def test_empire_support_spell_actions_require_empire_learned_spell_and_mana():
    env = _make_empire_env()
    spell_key = "emp_d2_s002"
    action = _support_spell_action_index(env, spell_key)

    assert bool(env.compute_action_mask()[action]) is False

    env.active_spells[spell_key]["learned"] = 1
    assert bool(env.compute_action_mask()[action]) is False

    _set_spell_use_mana(env, spell_key, multiplier=2.0)
    assert bool(env.compute_action_mask()[action]) is True

    env.step(action)
    assert bool(env.compute_action_mask()[action]) is False

    env._advance_turns(1)
    assert bool(env.compute_action_mask()[action]) is True

    non_empire_env = _make_legions_env()
    assert bool(non_empire_env.compute_action_mask()[action]) is False


def test_empire_support_spells_apply_to_hero_stack_reapply_same_turn_and_expire_next_turn():
    env = _make_empire_env()
    spell_keys = [
        "emp_d2_s001",
        "emp_d2_s002",
        "emp_d2_s003",
        "emp_d2_s005",
        "emp_d2_s012",
        "emp_d2_s013",
        "emp_d2_s017",
        "emp_d2_s018",
    ]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)

    first_unit = env._get_blue_state()[0]
    first_unit["armor"] = 10
    first_unit["damage"] = 100
    first_unit["damage_secondary"] = 20
    first_unit["initiative"] = 50
    first_unit["initiative_base"] = 50
    first_unit["accuracy"] = 80
    first_unit["accuracy_secondary"] = 40
    first_unit["resistance"] = []

    for spell_key in spell_keys:
        _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, spell_key))
        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(env.reward_spell_cast)
        assert info["spell_cast_executed"] is True

    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 30
    assert battle_first_unit["damage"] == 146
    assert battle_first_unit["damage_secondary"] == 29
    assert battle_first_unit["initiative_base"] == 55
    assert battle_first_unit["accuracy"] == 96
    assert battle_first_unit["accuracy_secondary"] == 48
    assert {"Air", "Water", "Fire"}.issubset(set(battle_first_unit.get("resistance") or []))

    env._save_blue_state()
    saved_first_unit = next(
        unit
        for unit in env.blue_team_state
        if int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert saved_first_unit["armor"] == 10
    assert saved_first_unit["damage"] == 100
    assert saved_first_unit["damage_secondary"] == 20
    assert saved_first_unit["initiative_base"] == 50
    assert saved_first_unit["initiative"] == 50
    assert saved_first_unit["accuracy"] == 80
    assert saved_first_unit["accuracy_secondary"] == 40
    assert "Air" not in set(saved_first_unit.get("resistance") or [])
    assert "Water" not in set(saved_first_unit.get("resistance") or [])
    assert "Fire" not in set(saved_first_unit.get("resistance") or [])

    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 30
    assert {"Air", "Water", "Fire"}.issubset(set(battle_first_unit.get("resistance") or []))

    env._advance_turns(1)
    assert env.blue_map_spell_effects == set()

    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 10
    assert battle_first_unit["damage"] == 100
    assert battle_first_unit["damage_secondary"] == 20
    assert battle_first_unit["initiative_base"] == 50
    assert battle_first_unit["accuracy"] == 80
    assert "Air" not in set(battle_first_unit.get("resistance") or [])
    assert "Water" not in set(battle_first_unit.get("resistance") or [])
    assert "Fire" not in set(battle_first_unit.get("resistance") or [])


def test_empire_heal_support_spells_restore_hero_stack_hp_and_persist():
    env = _make_empire_env()
    spell_keys = ["emp_d2_s007", "emp_d2_s019"]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)

    first_unit = env._get_blue_state()[0]
    second_unit = env._get_blue_state()[1]
    first_unit["max_health"] = 100.0
    first_unit["maxhp"] = 100.0
    first_unit["health"] = 40.0
    first_unit["hp"] = 40.0
    second_unit["max_health"] = 120.0
    second_unit["maxhp"] = 120.0
    second_unit["health"] = 90.0
    second_unit["hp"] = 90.0

    _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, "emp_d2_s007"))
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_executed"] is True
    assert info["spell_heal_total"] == pytest.approx(60.0)
    assert first_unit["health"] == pytest.approx(70.0)
    assert second_unit["health"] == pytest.approx(120.0)

    _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, "emp_d2_s019"))
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_executed"] is True
    assert info["spell_heal_total"] == pytest.approx(30.0)
    assert first_unit["health"] == pytest.approx(100.0)
    assert second_unit["health"] == pytest.approx(120.0)

    env._advance_turns(1)
    assert first_unit["health"] == pytest.approx(100.0)
    assert second_unit["health"] == pytest.approx(120.0)


def test_empire_movement_spell_restores_half_of_grid_moves_and_is_locked_per_turn():
    env = _make_empire_env()
    spell_key = "emp_d2_s006"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key, multiplier=2.0)
    env.moves = 6

    action = _support_spell_action_index(env, spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_executed"] is True
    assert info["spell_moves_restore_fraction"] == pytest.approx(0.5)
    assert info["spell_moves_restored"] == pytest.approx(10.0)
    assert info["moves_before_cast"] == pytest.approx(6.0)
    assert info["moves_after_cast"] == pytest.approx(16.0)
    assert env.moves == 16
    assert bool(env.compute_action_mask()[action]) is False

    _, _, _, _, info = env.step(action)
    assert info["spell_cast_executed"] is False
    assert env.moves == 16

    env._advance_turns(1)
    assert bool(env.compute_action_mask()[action]) is False
    env.moves = 6
    assert bool(env.compute_action_mask()[action]) is True


def test_support_heal_spell_at_full_hp_is_not_rewarded_or_charged():
    env = _make_empire_env()
    spell_key = "emp_d2_s007"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    mana_before = env._current_mana_totals()

    action = _support_spell_action_index(env, spell_key)
    assert bool(env.compute_action_mask()[action]) is False

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info["spell_cast_applied"] is False
    assert info["spell_cast_executed"] is False
    assert info["spell_cast_reward"] == pytest.approx(0.0)
    assert env._current_mana_totals() == pytest.approx(mana_before)
    assert env._spell_cast_limit_reached_this_turn(spell_key) is False


def test_heal_scroll_at_full_hp_is_not_rewarded_or_consumed_until_it_heals():
    env = _make_empire_env()
    spell_key = "g000ss0029"
    item_name = str(env._scroll_spell_definition_by_id(spell_key)["item_name"])
    env.scroll_magic_unlocked = True
    env.heroitems = [item_name]
    env._invalidate_inventory_cache()

    action = _scroll_spell_action_index(env, spell_key)
    assert env.count_scroll_item(item_name) == 1
    assert bool(env.compute_action_mask()[action]) is False

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info["spell_cast_applied"] is False
    assert info["spell_cast_executed"] is False
    assert info["spell_cast_reward"] == pytest.approx(0.0)
    assert info["scroll_item_consumed"] is False
    assert info["scroll_copies_before"] == 1
    assert info["scroll_copies_after"] == 1
    assert env.count_scroll_item(item_name) == 1

    first_unit = env._get_blue_state()[0]
    first_unit["health"] = 70.0
    first_unit["hp"] = 70.0

    assert bool(env.compute_action_mask()[action]) is True
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_applied"] is True
    assert info["spell_cast_executed"] is True
    assert info["spell_heal_total"] == pytest.approx(30.0)
    assert info["scroll_item_consumed"] is True
    assert info["scroll_copies_before"] == 1
    assert info["scroll_copies_after"] == 0
    assert env.count_scroll_item(item_name) == 0


def test_movement_scroll_at_move_cap_is_not_rewarded_or_consumed_until_it_restores_moves():
    env = _make_empire_env()
    spell_key = "g000ss0006"
    item_name = str(env._scroll_spell_definition_by_id(spell_key)["item_name"])
    env.scroll_magic_unlocked = True
    env.heroitems = [item_name]
    env.moves = env.moves_per_turn
    env._invalidate_inventory_cache()

    action = _scroll_spell_action_index(env, spell_key)
    assert env.count_scroll_item(item_name) == 1
    assert bool(env.compute_action_mask()[action]) is False

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert info["spell_cast_applied"] is False
    assert info["spell_cast_executed"] is False
    assert info["scroll_item_consumed"] is False
    assert info["scroll_copies_after"] == 1
    assert env.count_scroll_item(item_name) == 1

    env.moves = 0
    assert bool(env.compute_action_mask()[action]) is True
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_applied"] is True
    assert info["spell_cast_executed"] is True
    assert info["spell_moves_restored"] == pytest.approx(10.0)
    assert info["scroll_item_consumed"] is True
    assert info["scroll_copies_after"] == 0
    assert env.count_scroll_item(item_name) == 0


def test_typeoflord_two_allows_same_support_spell_twice_per_turn():
    env = _enable_typeoflord_two(_make_empire_env())
    spell_key = "emp_d2_s006"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key, multiplier=3.0)
    env.moves = 0

    action = _support_spell_action_index(env, spell_key)

    _, reward, terminated, truncated, first_info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert first_info["spell_cast_executed"] is True
    assert first_info["spell_cast_used_this_turn"] is False
    assert first_info["spell_moves_restored"] == pytest.approx(10.0)
    assert first_info["moves_after_cast"] == pytest.approx(10.0)
    assert env.moves == 10
    assert bool(env.compute_action_mask()[action]) is True

    _, reward, terminated, truncated, second_info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert second_info["spell_cast_executed"] is True
    assert second_info["spell_cast_used_this_turn"] is False
    assert second_info["spell_moves_restored"] == pytest.approx(10.0)
    assert second_info["moves_after_cast"] == pytest.approx(20.0)
    assert env.moves == 20
    assert bool(env.compute_action_mask()[action]) is False

    _, reward, terminated, truncated, third_info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.0)
    assert third_info["spell_cast_executed"] is False
    assert third_info["spell_cast_used_this_turn"] is True
    assert env.moves == 20


def test_mountain_support_spells_apply_to_hero_stack_and_expire_next_turn():
    env = _make_mountain_env()
    spell_keys = [
        "mcl_d2_s001",
        "mcl_d2_s003",
        "mcl_d2_s006",
        "mcl_d2_s009",
        "mcl_d2_s014",
        "mcl_d2_s015",
        "mcl_d2_s016",
        "mcl_d2_s017",
    ]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)

    first_unit = env._get_blue_state()[0]
    second_unit = env._get_blue_state()[1]
    first_unit["armor"] = 10
    first_unit["damage"] = 100
    first_unit["damage_secondary"] = 25
    first_unit["initiative"] = 60
    first_unit["initiative_base"] = 60
    first_unit["accuracy"] = 60
    first_unit["accuracy_secondary"] = 30
    first_unit["max_health"] = 100.0
    first_unit["maxhp"] = 100.0
    first_unit["health"] = 40.0
    first_unit["hp"] = 40.0
    second_unit["max_health"] = 120.0
    second_unit["maxhp"] = 120.0
    second_unit["health"] = 100.0
    second_unit["hp"] = 100.0

    for spell_key in spell_keys:
        _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, spell_key))
        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(env.reward_spell_cast)
        assert info["spell_cast_executed"] is True

    assert first_unit["health"] == pytest.approx(70.0)
    assert second_unit["health"] == pytest.approx(120.0)

    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 53
    assert battle_first_unit["damage"] == 132
    assert battle_first_unit["damage_secondary"] == 33
    assert battle_first_unit["initiative_base"] == 69
    assert battle_first_unit["accuracy"] == 100
    assert battle_first_unit["accuracy_secondary"] == 50
    assert battle_first_unit["health"] == pytest.approx(70.0)

    env._advance_turns(1)
    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 10
    assert battle_first_unit["damage"] == 100
    assert battle_first_unit["damage_secondary"] == 25
    assert battle_first_unit["initiative_base"] == 60
    assert battle_first_unit["accuracy"] == 60
    assert battle_first_unit["accuracy_secondary"] == 30
    assert battle_first_unit["health"] == pytest.approx(70.0)


def test_mountain_movement_spells_restore_up_to_move_cap():
    env = _make_mountain_env()
    spell_keys = ["mcl_d2_s012", "mcl_d2_s020"]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)

    env.moves = 3
    _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, "mcl_d2_s012"))
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_moves_restored"] == pytest.approx(float(env.moves_per_turn - 3))
    assert env.moves == env.moves_per_turn

    env.moves = 8
    _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, "mcl_d2_s020"))
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_moves_restore_fraction"] == pytest.approx(1.0)
    assert info["spell_moves_restored"] == pytest.approx(float(env.moves_per_turn - 8))
    assert env.moves == env.moves_per_turn


def test_elves_support_spells_apply_heal_and_temporary_health_bonus():
    env = _make_elves_env()
    spell_keys = [
        "elf_d2_s005",
        "elf_d2_s006",
        "elf_d2_s014",
        "elf_d2_s020",
    ]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)

    first_unit = env._get_blue_state()[0]
    second_unit = env._get_blue_state()[1]
    first_unit["armor"] = 10
    first_unit["max_health"] = 100.0
    first_unit["maxhp"] = 100.0
    first_unit["health"] = 40.0
    first_unit["hp"] = 40.0
    second_unit["max_health"] = 120.0
    second_unit["maxhp"] = 120.0
    second_unit["health"] = 70.0
    second_unit["hp"] = 70.0

    _, _, _, _, info = env.step(_support_spell_action_index(env, "elf_d2_s006"))
    assert info["spell_heal_total"] == pytest.approx(60.0)
    _, _, _, _, info = env.step(_support_spell_action_index(env, "elf_d2_s014"))
    assert info["spell_heal_total"] == pytest.approx(50.0)
    _, _, _, _, info = env.step(_support_spell_action_index(env, "elf_d2_s005"))
    assert info["spell_cast_executed"] is True
    _, _, _, _, info = env.step(_support_spell_action_index(env, "elf_d2_s020"))
    assert info["spell_cast_executed"] is True
    assert info["spell_effect_health_delta"] == 50

    assert first_unit["health"] == pytest.approx(100.0)
    assert second_unit["health"] == pytest.approx(120.0)

    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 20
    assert battle_first_unit["max_health"] == pytest.approx(150.0)
    assert battle_first_unit["maxhp"] == pytest.approx(150.0)
    assert 0.0 < float(battle_first_unit["health"]) <= 150.0
    assert 0.0 < float(battle_first_unit["hp"]) <= 150.0

    env._save_blue_state()
    saved_first_unit = next(
        unit
        for unit in env.blue_team_state
        if int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert saved_first_unit["armor"] == 10
    assert saved_first_unit["max_health"] == pytest.approx(100.0)
    assert saved_first_unit["maxhp"] == pytest.approx(100.0)
    assert 0.0 < float(saved_first_unit["health"]) <= 100.0
    assert 0.0 < float(saved_first_unit["hp"]) <= 100.0

    env._advance_turns(1)
    assert env.blue_map_spell_effects == set()
    saved_first_unit = next(
        unit
        for unit in env.blue_team_state
        if int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert saved_first_unit["armor"] == 10
    assert saved_first_unit["max_health"] == pytest.approx(100.0)
    assert saved_first_unit["maxhp"] == pytest.approx(100.0)
    assert 0.0 < float(saved_first_unit["health"]) <= 100.0
    assert 0.0 < float(saved_first_unit["hp"]) <= 100.0


def test_elves_movement_spell_restores_thirty_percent_of_grid_moves():
    env = _make_elves_env()
    spell_key = "elf_d2_s004"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)
    env.moves = 5

    _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, spell_key))

    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_executed"] is True
    assert info["spell_moves_restore_fraction"] == pytest.approx(0.3)
    assert info["spell_moves_restored"] == pytest.approx(6.0)
    assert env.moves == 11


def test_level_five_spell_usage_is_masked_until_typeoflord_two():
    cases = [
        (_make_empire_env, "emp_d2_s022", "offensive"),
        (_make_empire_env, "emp_d2_s021", "support"),
        (_make_legions_env, "lod_d2_s023", "offensive"),
        (_make_legions_env, "lod_d2_s021", "offensive"),
        (_make_mountain_env, "mcl_d2_s023", "support"),
        (_make_mountain_env, "mcl_d2_s021", "offensive"),
        (_make_undead_env, "und_d2_s023", "offensive"),
        (_make_undead_env, "und_d2_s022", "support"),
        (_make_undead_env, "und_d2_s021", "offensive"),
        (_make_elves_env, "elf_d2_s022", "offensive"),
        (_make_elves_env, "elf_d2_s021", "offensive"),
    ]

    for make_env, spell_key, kind in cases:
        env = make_env()
        env.active_spells[spell_key]["learned"] = 1
        _set_spell_use_mana(env, spell_key, multiplier=2.0)
        action = (
            _spell_action_index(env, spell_key)
            if kind == "offensive"
            else _support_spell_action_index(env, spell_key)
        )

        assert bool(env.compute_action_mask()[action]) is False

        _, reward, terminated, truncated, info = env.step(action)
        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(0.0)
        assert info["spell_cast_executed"] is False
        assert info["blocked_by_typeoflord"] is True

        _enable_typeoflord_two(env)
        assert bool(env.compute_action_mask()[action]) is True


def test_empire_level_five_support_spells_apply_death_ward_and_heal_hero_stack():
    env = _enable_typeoflord_two(_make_empire_env())
    spell_keys = ["emp_d2_s021", "emp_d2_s023", "emp_d2_s024"]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)

    first_unit = env._get_blue_state()[0]
    second_unit = env._get_blue_state()[1]
    first_unit["max_health"] = 100.0
    first_unit["maxhp"] = 100.0
    first_unit["health"] = 10.0
    first_unit["hp"] = 10.0
    first_unit["resistance"] = []
    second_unit["max_health"] = 120.0
    second_unit["maxhp"] = 120.0
    second_unit["health"] = 30.0
    second_unit["hp"] = 30.0
    second_unit["resistance"] = []

    _, _, _, _, info = env.step(_support_spell_action_index(env, "emp_d2_s021"))
    assert info["spell_cast_executed"] is True
    _, _, _, _, info = env.step(_support_spell_action_index(env, "emp_d2_s023"))
    assert info["spell_heal_total"] == pytest.approx(180.0)
    _, _, _, _, info = env.step(_support_spell_action_index(env, "emp_d2_s024"))
    assert info["spell_heal_total"] == pytest.approx(0.0)

    assert first_unit["health"] == pytest.approx(100.0)
    assert second_unit["health"] == pytest.approx(120.0)

    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert "Death" in set(battle_first_unit.get("resistance") or [])

    env._advance_turns(1)
    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert "Death" not in set(battle_first_unit.get("resistance") or [])
    assert first_unit["health"] == pytest.approx(100.0)


def test_new_level_five_offensive_spells_apply_expected_damage():
    cases = [
        (_make_empire_env, "emp_d2_s022", 125.0),
        (_make_legions_env, "lod_d2_s023", 125.0),
        (_make_undead_env, "und_d2_s023", 125.0),
        (_make_undead_env, "und_d2_s024", 80.0),
        (_make_elves_env, "elf_d2_s022", 125.0),
    ]

    for make_env, spell_key, expected_damage in cases:
        env = _enable_typeoflord_two(make_env())
        env.active_spells[spell_key]["learned"] = 1
        _set_spell_use_mana(env, spell_key)
        env.enemy_team_states[10] = [
            _enemy_unit(position=1, hp=200.0, max_hp=200.0, name="Target"),
        ]

        _, reward, terminated, truncated, info = env.step(_spell_action_index(env, spell_key))

        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(env.reward_spell_cast)
        assert info["spell_cast_executed"] is True
        assert info["spell_damage_total"] == pytest.approx(expected_damage)
        assert env.enemy_team_states[10][0]["health"] == pytest.approx(200.0 - expected_damage)


def test_mountain_level_five_support_spells_apply_and_expire_next_turn():
    env = _enable_typeoflord_two(_make_mountain_env())
    spell_keys = ["mcl_d2_s023", "mcl_d2_s024"]
    for spell_key in spell_keys:
        env.active_spells[spell_key]["learned"] = 1
    _grant_spell_use_mana(env, spell_keys)

    first_unit = env._get_blue_state()[0]
    first_unit["armor"] = 10
    first_unit["damage"] = 100
    first_unit["damage_secondary"] = 20

    for spell_key in spell_keys:
        _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, spell_key))
        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(env.reward_spell_cast)
        assert info["spell_cast_executed"] is True

    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 43
    assert battle_first_unit["damage"] == 150
    assert battle_first_unit["damage_secondary"] == 30

    env._advance_turns(1)
    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert battle_first_unit["armor"] == 10
    assert battle_first_unit["damage"] == 100
    assert battle_first_unit["damage_secondary"] == 20


def test_undead_level_five_weapon_ward_is_applied_and_expires():
    env = _enable_typeoflord_two(_make_undead_env())
    spell_key = "und_d2_s022"
    env.active_spells[spell_key]["learned"] = 1
    _set_spell_use_mana(env, spell_key)

    _, reward, terminated, truncated, info = env.step(_support_spell_action_index(env, spell_key))
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(env.reward_spell_cast)
    assert info["spell_cast_executed"] is True
    assert info["spell_resistance_type"] == "Weapon"

    first_unit = env._get_blue_state()[0]
    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert "Weapon" in set(battle_first_unit.get("resistance") or [])

    env._advance_turns(1)
    env._init_battle(10)
    battle_first_unit = next(
        unit
        for unit in env.battle_env.combined
        if unit.get("team") == "blue" and int(unit.get("position", -1) or -1) == int(first_unit["position"])
    )
    assert "Weapon" not in set(battle_first_unit.get("resistance") or [])
