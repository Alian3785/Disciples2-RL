"""Scroll casts must change combat state, not only report an applied flag.

These tests use real campaign actions, inventory consumption, battle preparation,
post-battle persistence, and the campaign day boundary. No training is launched.
"""
from collections import Counter
from copy import deepcopy

import pytest

from campaign_env import CampaignEnv
from scroll_spell_data import SCROLL_ITEM_DEFINITIONS

EFFECT_KINDS = {"buff", "debuff", "ward", "health_bonus"}
COMBAT_SCROLLS = [
    spec for spec in SCROLL_ITEM_DEFINITIONS
    if spec.get("supported") and spec.get("spell_kind") in EFFECT_KINDS
    and not spec.get("ignore_terrain_penalty")
]
TERRAIN_SCROLLS = [
    spec for spec in SCROLL_ITEM_DEFINITIONS if spec.get("ignore_terrain_penalty")
]
WARD_SCROLLS = [spec for spec in COMBAT_SCROLLS if spec["spell_kind"] == "ward"]
STAT_KEYS = (
    "armor", "damage", "damage_secondary", "initiative_base", "accuracy",
    "accuracy_secondary", "max_health", "health", "resistance",
)


def living(team):
    return [unit for unit in team if unit.get("health", 0) > 0]


def stats(unit):
    return {key: deepcopy(unit[key]) for key in STAT_KEYS}


@pytest.fixture
def env():
    campaign = CampaignEnv(
        map_name="scroll_train", observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False, detailed_step_info=True,
    )
    campaign.reset(seed=42)
    # Make each magnitude observable, avoiding armor floors/accuracy caps and
    # pre-existing racial wards. Runtime fixture changes do not change game data.
    teams = [campaign._get_blue_state(), *campaign.enemy_team_states.values()]
    for team in teams:
        for unit in living(team):
            unit.update(
                damage=100, damage_secondary=50, initiative=50,
                initiative_base=50, accuracy=50, accuracy_secondary=40,
                armor=90, health=500, hp=500, max_health=500, maxhp=500,
                immunity=[], resistance=[], unit_type="Warrior",
            )
            if unit["team"] == "blue":
                # Battle setup auto-plays RED turns until BLUE can act.
                unit.update(initiative=100, initiative_base=100)
    yield campaign
    campaign.close()


def battle_teams(env, enemy_id=1):
    env._init_battle(enemy_id)
    return {
        side: {unit["position"]: stats(unit)
               for unit in living(env.battle_env.combined)
               if unit["team"] == side}
        for side in ("blue", "red")
    }


def scroll_action(env, spec):
    return env.grid_scroll_cast_action_start + next(
        i for i, entry in enumerate(env.scroll_cast_slot_entries())
        if entry["spell_id"] == spec["spell_id"]
    )


def cast_scroll(env, spec):
    action = scroll_action(env, spec)
    assert env.compute_action_mask()[action]
    before = env.count_scroll_item(spec["item_name"])
    mana = env._current_mana_totals()
    info = env.step(action)[-1]
    assert info["spell_cast_executed"] and info["spell_cast_applied"]
    assert info["scroll_item_consumed"]
    assert env.count_scroll_item(spec["item_name"]) == before - 1
    assert env._current_mana_totals() == mana
    assert env.mode == env.MODE_GRID
    return info


def expected_stats(base, spec):
    expected = deepcopy(base)
    kind = spec.get("spell_kind", spec.get("kind"))
    if kind in {"buff", "debuff"}:
        expected["armor"] = max(0, int(round(base["armor"] + spec.get("armor_delta", 0))))
        for key, multiplier in (
            ("damage", "damage_multiplier"),
            ("damage_secondary", "damage_multiplier"),
            ("initiative_base", "initiative_multiplier"),
            ("accuracy", "accuracy_multiplier"),
            ("accuracy_secondary", "accuracy_multiplier"),
        ):
            value = max(0, int(round(base[key] * spec.get(multiplier, 1.0))))
            expected[key] = min(100, value) if key.startswith("accuracy") else value
    elif kind == "ward":
        expected["resistance"] = sorted(set(base["resistance"] + [spec["resistance_type"]]))
    elif kind == "health_bonus":
        for key in ("health", "max_health"):
            expected[key] += spec["health_delta"]
    return expected


def assert_effect(before, after, spec, side):
    assert set(after[side]) == set(before[side])
    assert after[side] != before[side], "applied flags must correspond to real combat effects"
    for position, unit in after[side].items():
        assert unit == expected_stats(before[side][position], spec)
    other = "red" if side == "blue" else "blue"
    assert after[other] == before[other]


def test_coverage_and_normalization_preserve_source_definitions():
    assert Counter(spec["spell_kind"] for spec in COMBAT_SCROLLS) == {
        "buff": 14, "debuff": 17, "ward": 7, "health_bonus": 1,
    }
    assert len(TERRAIN_SCROLLS) == 2
    original = deepcopy(SCROLL_ITEM_DEFINITIONS)
    combined = dict(CampaignEnv._combined_map_support_spell_specs_by_id())
    combined.update(CampaignEnv._combined_map_offensive_spell_specs_by_id())
    for spec in SCROLL_ITEM_DEFINITIONS:
        if spec.get("supported"):
            normalized = combined[spec["spell_id"]]
            assert normalized == dict(spec, kind=spec["spell_kind"])
            assert normalized is not spec
    assert SCROLL_ITEM_DEFINITIONS == original


@pytest.mark.parametrize("spec", COMBAT_SCROLLS, ids=lambda s: s["spell_id"])
def test_all_39_scrolls_change_battle_stats_consume_once_and_expire(env, spec):
    before = battle_teams(env)
    shape, actions = env.observation_space.shape, env.action_space.n
    # One copy is supplied by scroll_train; keep a second to test failed/recasts.
    env._append_hero_item(spec["item_name"])
    info = cast_scroll(env, spec)
    side = "red" if spec["spell_kind"] == "debuff" else "blue"
    if side == "red":
        assert info["target_enemy_id"] == 1
        map_state = deepcopy(env._get_enemy_team_state(1))
        env._recompute_enemy_stack_spell_effects(1)
        assert env._get_enemy_team_state(1) == map_state  # No compounding.
    else:
        action = scroll_action(env, spec)
        assert not env.compute_action_mask()[action]
        failed = env.step(action)[-1]
        assert not failed["spell_cast_executed"]
        assert not failed["scroll_item_consumed"]
        assert env.count_scroll_item(spec["item_name"]) == 1
    after = battle_teams(env)
    assert_effect(before, after, spec, side)
    if side == "blue":
        env._save_blue_state()
        for unit in living(env._get_blue_state()):
            assert stats(unit) == before["blue"][unit["position"]]
        # The next battle in the same day gets exactly one application.
        assert battle_teams(env) == after
    env._advance_turns(1)
    assert not env.blue_map_spell_effects
    assert not env.enemy_map_spell_effects
    assert battle_teams(env) == before
    # After expiry, the remaining scroll is usable and gets the same result.
    cast_scroll(env, spec)
    assert env.count_scroll_item(spec["item_name"]) == 0
    assert battle_teams(env) == after
    assert env.observation_space.shape == shape
    assert env.action_space.n == actions


@pytest.mark.parametrize("spec", WARD_SCROLLS, ids=lambda s: s["spell_id"])
def test_every_ward_blocks_first_matching_hit_only_and_refreshes_next_battle(env, spec):
    cast_scroll(env, spec)
    for _ in range(2):
        env._init_battle(1)
        battle = env.battle_env
        victim = next(unit for unit in living(battle.combined) if unit["team"] == "blue")
        attacker = next(unit for unit in living(battle.combined) if unit["team"] == "red")
        attacker.update(
            unit_type="Warrior", attack_type_primary=spec["resistance_type"],
            attack_type_secondary="", accuracy=100, damage=20, damage_secondary=0,
        )
        victim["armor"] = 0
        hp = victim["health"]
        assert battle._attack(attacker, victim["position"])[0]
        assert victim["health"] == hp
        assert victim["resilience_used_types"] == [spec["resistance_type"]]
        assert battle._attack(attacker, victim["position"])[0]
        assert victim["health"] < hp
    env._advance_turns(1)
    env._init_battle(1)
    assert all(spec["resistance_type"] not in unit["resistance"]
               for unit in living(env.battle_env.combined) if unit["team"] == "blue")


@pytest.mark.parametrize("damage", [0, 25, 600])
def test_health_bonus_preserves_real_damage_and_does_not_revive_dead_units(env, damage):
    spec = next(s for s in COMBAT_SCROLLS if s["spell_kind"] == "health_bonus")
    blue = living(env._get_blue_state())
    blue[0].update(health=300, hp=300)
    dead = blue[-1]
    dead.update(health=0, hp=0, initiative=0)
    cast_scroll(env, spec)
    env._init_battle(1)
    victim = next(u for u in env.battle_env.combined if u["position"] == blue[0]["position"])
    assert victim["health"] == 350 and victim["max_health"] == 550
    assert victim["hp"] == 350 and victim["maxhp"] == 550
    assert next(u for u in env.battle_env.combined if u["position"] == dead["position"])["health"] == 0
    env.battle_env._subtract_health(victim, damage)
    env._save_blue_state()
    saved = next(u for u in env._get_blue_state() if u["position"] == victim["position"])
    assert saved["health"] == max(0, 300 - damage)
    assert saved["max_health"] == 500
    env._advance_turns(1)
    env._init_battle(1)
    restored = next(u for u in env.battle_env.combined if u["position"] == victim["position"])
    assert restored["health"] == max(0, 300 - damage)
    assert restored["max_health"] == 500


@pytest.mark.parametrize("spec", TERRAIN_SCROLLS, ids=lambda s: s["spell_id"])
def test_terrain_scrolls_keep_existing_effect_and_expiry(env, spec):
    before = battle_teams(env)
    terrain = spec["ignore_terrain_penalty"]
    assert not env._blue_stack_ignores_terrain_penalty(terrain)
    cast_scroll(env, spec)
    assert env._blue_stack_ignores_terrain_penalty(terrain)
    assert battle_teams(env) == before
    env._advance_turns(1)
    assert not env._blue_stack_ignores_terrain_penalty(terrain)


BOOK_SPECS = {
    **CampaignEnv._map_offensive_spell_specs_by_id(),
    **CampaignEnv._map_support_spell_specs_by_id(),
}
BOOK_EFFECTS = [(key, spec) for key, spec in BOOK_SPECS.items()
                if spec.get("kind") in EFFECT_KINDS and not spec.get("ignore_terrain_penalty")]


@pytest.mark.parametrize("key,spec", BOOK_EFFECTS, ids=[key for key, _ in BOOK_EFFECTS])
def test_standard_book_effects_keep_numeric_behavior_and_expiry(env, key, spec):
    before = battle_teams(env)
    inventory = deepcopy(env.heroitems)
    kind = spec["kind"]
    target = {"enemy_id": 1, "position": env.grid_env.enemy_positions[1], "reachable": True}
    result = env._cast_from_spell_spec(
        source="learned_offensive_spell" if kind == "debuff" else "learned_support_spell",
        spell_key=key, spell_description=key, spell_spec=spec,
        spend_mana=False, enforce_turn_limit=False,
        nearest_targetable_enemy=target, nearest_any_enemy=target,
    )
    assert result["spell_cast_applied"]
    assert env.heroitems == inventory
    assert_effect(before, battle_teams(env), spec, "red" if kind == "debuff" else "blue")
    env._advance_turns(1)
    assert battle_teams(env) == before


def test_scroll_and_book_damage_buffs_stack_once_without_mutating_book_specs(env):
    scroll = next(s for s in COMBAT_SCROLLS if s.get("damage_multiplier") == 1.1)
    key, book = next((key, spec) for key, spec in BOOK_EFFECTS
                     if spec.get("kind") == "buff" and spec.get("damage_multiplier"))
    original_book = deepcopy(book)
    before = battle_teams(env)
    env._apply_support_spell_to_blue_stack(key)
    cast_scroll(env, scroll)
    combined = dict(scroll, damage_multiplier=scroll["damage_multiplier"] * book["damage_multiplier"])
    assert_effect(before, battle_teams(env), combined, "blue")
    assert BOOK_SPECS[key] == original_book


@pytest.mark.parametrize(
    "spec", [spec for spec in COMBAT_SCROLLS if spec["spell_kind"] == "debuff"],
    ids=lambda s: s["spell_id"],
)
def test_debuffs_survive_battle_save_without_compounding_or_losing_damage(env, spec):
    before = battle_teams(env)
    cast_scroll(env, spec)
    after = battle_teams(env)
    target = next(u for u in living(env.battle_env.combined) if u["team"] == "red")
    env.battle_env._subtract_health(target, 25)
    env._save_enemy_state_from_battle(1)
    expected = deepcopy(after)
    expected["red"][target["position"]]["health"] -= 25
    assert battle_teams(env) == expected
    env._clear_all_enemy_map_spell_effects()
    expected = deepcopy(before)
    expected["red"][target["position"]]["health"] -= 25
    assert battle_teams(env) == expected


@pytest.mark.parametrize("kind", ["buff", "ward", "health_bonus", "debuff"])
def test_missing_inventory_cannot_apply_or_consume_scroll(env, kind):
    spec = next(s for s in COMBAT_SCROLLS if s["spell_kind"] == kind)
    assert env._consume_hero_item(spec["item_name"])
    before = battle_teams(env)
    action = scroll_action(env, spec)
    assert not env.compute_action_mask()[action]
    info = env.step(action)[-1]
    assert not info["spell_cast_executed"]
    assert not info["scroll_item_consumed"]
    assert env.count_scroll_item(spec["item_name"]) == 0
    assert battle_teams(env) == before
    assert not env.blue_map_spell_effects and not env.enemy_map_spell_effects
