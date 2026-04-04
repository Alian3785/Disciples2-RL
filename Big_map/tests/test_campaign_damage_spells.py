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


def _spell_action_index(env: CampaignEnv, spell_key: str) -> int:
    for idx, (candidate_key, _damage) in enumerate(env.LEGION_DAMAGE_SPELL_ACTION_SPECS):
        if str(candidate_key) == str(spell_key):
            return env.grid_legion_damage_spell_action_start + idx
    raise AssertionError(f"spell action not found: {spell_key}")


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
        "position": int(position),
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


def test_wounded_enemy_units_regenerate_ten_percent_per_turn_but_dead_units_do_not():
    env = _make_legions_env()
    env.enemy_team_states[10] = [
        _enemy_unit(position=1, hp=50.0, max_hp=100.0, name="Wounded"),
        _enemy_unit(position=2, hp=96.0, max_hp=100.0, name="Almost full"),
        _enemy_unit(position=3, hp=0.0, max_hp=100.0, name="Dead"),
    ]
    env.enemy_team_states[20] = [
        _enemy_unit(position=1, hp=40.0, max_hp=100.0, name="Defeated stack wounded"),
    ]
    env.grid_env.enemies_alive[20] = False

    env._advance_turns(1)

    assert env.enemy_team_states[10][0]["health"] == pytest.approx(60.0)
    assert env.enemy_team_states[10][1]["health"] == pytest.approx(100.0)
    assert env.enemy_team_states[10][2]["health"] == pytest.approx(0.0)
    assert env.enemy_team_states[20][0]["health"] == pytest.approx(40.0)
