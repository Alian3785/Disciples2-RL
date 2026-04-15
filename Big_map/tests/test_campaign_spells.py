from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv


def _mark_magic_tower_built(env: CampaignEnv) -> None:
    for build_key in env.building_keys:
        building = env.active_buildings.get(build_key)
        if not isinstance(building, dict):
            continue
        if str(building.get("name", "") or "").strip() == env.MAGIC_TOWER_BUILDING_NAME:
            building["built"] = 1
            return
    raise AssertionError("magic tower building not found")


def _spell_action_index(env: CampaignEnv, spell_key: str) -> int:
    return env.grid_spell_action_start + env.spell_keys.index(spell_key)


def _spell_mask(env: CampaignEnv):
    return env.compute_action_mask()[
        env.grid_spell_action_start : env.grid_settlement_upgrade_action_start
    ]


def test_spell_actions_require_magic_tower_and_mana():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)

    assert len(env.spell_keys) == 24
    spell_mask = _spell_mask(env)
    assert not spell_mask.any()

    first_spell_key = env.spell_keys[0]
    first_spell = env.active_spells[first_spell_key]
    env.infernal_mana = float(first_spell["learn_hell"])
    spell_mask = _spell_mask(env)
    assert not spell_mask.any()

    _mark_magic_tower_built(env)
    spell_mask = _spell_mask(env)
    assert spell_mask[0]


def test_spell_learning_spends_mana_marks_spell_and_locks_until_next_turn():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    _mark_magic_tower_built(env)

    first_spell_key = env.spell_keys[0]
    second_spell_key = env.spell_keys[1]
    first_spell = env.active_spells[first_spell_key]
    second_spell = env.active_spells[second_spell_key]

    env.infernal_mana = float(first_spell["learn_hell"])
    action = _spell_action_index(env, first_spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert reward == env.reward_spell_learn
    assert terminated is False
    assert truncated is False
    assert info["spell_learned"] is True
    assert info["spell_key"] == first_spell_key
    assert env.active_spells[first_spell_key]["learned"] == 1
    assert env.infernal_mana == 0.0
    assert env.spell_learning_locked is True

    spell_mask = _spell_mask(env)
    assert not spell_mask.any()

    env._advance_turns(1)
    env.infernal_mana = max(float(second_spell["learn_hell"]), 200.0)
    spell_mask = _spell_mask(env)

    assert env.spell_learning_locked is False
    assert bool(spell_mask[env.spell_keys.index(first_spell_key)]) is False
    assert bool(spell_mask[env.spell_keys.index(second_spell_key)]) is True


def test_typeoflord_two_halves_learning_costs_but_not_use_costs():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    _mark_magic_tower_built(env)
    env.typeoflord = 2
    env.Typeoflord = 2

    spell_key = env.spell_keys[0]
    spell = env.active_spells[spell_key]
    base_learning_costs = env._spell_costs_from_entry(spell, prefix="learn")
    adjusted_learning_costs = env._get_spell_learning_costs(spell)
    base_use_costs = env._spell_costs_from_entry(spell, prefix="use")
    adjusted_use_costs = env._get_spell_use_costs(spell)

    for mana_kind, base_cost in base_learning_costs.items():
        assert adjusted_learning_costs[mana_kind] == pytest.approx(float(base_cost) * 0.5)
    assert adjusted_use_costs == base_use_costs

    for mana_kind in env.MANA_KIND_ORDER:
        setattr(env, env.MANA_ATTR_BY_KIND[mana_kind], 0.0)
    for mana_kind, adjusted_cost in adjusted_learning_costs.items():
        setattr(env, env.MANA_ATTR_BY_KIND[mana_kind], float(adjusted_cost))

    action = _spell_action_index(env, spell_key)
    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == env.reward_spell_learn
    assert info["spell_learned"] is True
    assert env.active_spells[spell_key]["learned"] == 1
    for mana_kind, adjusted_cost in adjusted_learning_costs.items():
        if adjusted_cost > 0.0:
            assert getattr(env, env.MANA_ATTR_BY_KIND[mana_kind]) == pytest.approx(0.0)


def test_render_includes_learned_spell_descriptions():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)
    env.reset(seed=123)
    _mark_magic_tower_built(env)

    first_spell_key = env.spell_keys[0]
    first_spell = env.active_spells[first_spell_key]
    env.infernal_mana = float(first_spell["learn_hell"])
    env.step(_spell_action_index(env, first_spell_key))

    rendered = env.render(mode="ansi")
    assert rendered is not None
    assert "Изученные заклинания" in rendered
    assert str(first_spell["description"]) in rendered


def test_campaign_env_exposes_default_typeoflord():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=2)

    _, info = env.reset(seed=123)

    assert env.typeoflord == 1
    assert env.Typeoflord == 1
    assert info["typeoflord"] == 1
    assert env.get_state_summary()["typeoflord"] == 1

    _, _, _, _, step_info = env.step(8)
    assert step_info["typeoflord"] == 1


def test_level_five_spells_are_masked_unless_typeoflord_is_two_for_all_races():
    for realcapital in (1, 2, 3, 4, 5):
        env = CampaignEnv(log_enabled=False, persist_blue_hp=True, realcapital=realcapital)
        env.reset(seed=123)
        _mark_magic_tower_built(env)

        for mana_attr in set(env.MANA_ATTR_BY_KIND.values()):
            setattr(env, mana_attr, 10000.0)

        spell_key = env.spell_keys[-1]
        spell = env.active_spells[spell_key]
        assert int(spell["level"]) == 5

        spell_mask = _spell_mask(env)
        assert bool(spell_mask[env.spell_keys.index(spell_key)]) is False

        _, reward, terminated, truncated, info = env.step(_spell_action_index(env, spell_key))
        assert terminated is False
        assert truncated is False
        assert reward == 0.0
        assert info["spell_key"] == spell_key
        assert info["spell_level"] == 5
        assert info["blocked_by_typeoflord"] is True
        assert info["spell_learned"] is False
        assert env.active_spells[spell_key]["learned"] == 0

        env.typeoflord = 2
        env.Typeoflord = 2
        spell_mask = _spell_mask(env)
        assert bool(spell_mask[env.spell_keys.index(spell_key)]) is True
