"""Map spell masks and effects must select the same enemy across obstacles."""

from types import SimpleNamespace

import pytest

from campaign_env import CampaignEnv
from campaign_env_masks import CampaignMaskMixin


@pytest.fixture
def selector():
    env = CampaignMaskMixin()
    env.grid_env = SimpleNamespace(
        agent_pos=(1, 1), enemy_positions={1: (3, 1), 2: (1, 5)},
        enemies_alive={1: True, 2: True}, obstacle_positions={(2, y) for y in range(7)},
    )
    env._enemy_descriptions = {}
    env._enemy_team_has_living_units = lambda enemy_id: True
    env._is_enemy_stack_spell_targetable = lambda enemy_id: enemy_id != 1

    def no_path_search(**kwargs):
        pytest.fail("Spell target selection must not search paths")

    env._build_territory_path_distances = no_path_search
    return env


@pytest.mark.parametrize("method", ("get_nearest_enemy_stack", "_get_nearest_enemy_stack_for_mask"))
@pytest.mark.parametrize("case,expected", (
    ("wall", 1), ("tie", 1), ("chebyshev", 1), ("origin", 2),
    ("dead", 2), ("empty", 2), ("protected", 2),
    ("include_dead", 1), ("none_alive", None), ("no_enemies", None),
))
def test_nearest_selection_rules(selector, method, case, expected):
    kwargs = {}
    if case == "tie":
        # Insertion order cannot break ties: the smaller enemy ID wins.
        selector.grid_env.enemy_positions = {2: (1, 3), 1: (3, 1)}
    elif case == "chebyshev":
        # Preserve the grid's eight-direction metric, not Euclidean distance.
        selector.grid_env.enemy_positions = {1: (4, 4), 2: (5, 1)}
    elif case == "origin":
        kwargs["from_pos"] = (1, 6)
    elif case == "dead":
        selector.grid_env.enemies_alive[1] = False
    elif case == "empty":
        selector._enemy_team_has_living_units = lambda enemy_id: enemy_id == 2
    elif case == "protected":
        kwargs["spell_targetable_only"] = True
    elif case == "include_dead":
        selector.grid_env.enemies_alive[1] = False
        selector._enemy_team_has_living_units = lambda enemy_id: enemy_id == 2
        kwargs["only_alive"] = False
    elif case == "none_alive":
        selector.grid_env.enemies_alive = {1: False, 2: False}
    elif case == "no_enemies":
        selector.grid_env.enemy_positions.clear()
    target = getattr(selector, method)(**kwargs)
    assert (target["enemy_id"] if target else None) == expected


@pytest.fixture
def make_env(monkeypatch):
    opened = []

    def create(source):
        staff = None
        if source == "staff":
            specs = CampaignEnv._combined_map_offensive_spell_specs_by_id()
            staff = next(entry for entry in CampaignEnv.STAFF_SPELL_ITEM_DEFINITIONS
                         if specs.get(entry["spell_id"], {}).get("kind") == "damage")
            # Give this test scenario one real staff slot before action-space construction.
            monkeypatch.setattr(CampaignEnv, "_scenario_staff_spell_item_names",
                                lambda self: (staff["item_name"],))
        env = CampaignEnv(
            map_name="scroll_train" if source == "scroll" else "magic_train",
            observation_version="local5", scripted_capital_bot_enabled=False,
            use_boss_starting_roster=False, log_enabled=False, detailed_step_info=True,
        )
        opened.append(env)
        env.reset(seed=42)
        env.grid_env.agent_pos = (1, 1)
        env.grid_env.enemy_positions = {1: (3, 1), 2: (1, 5)}
        env.grid_env.enemies_alive = {1: True, 2: True}
        for attr in env.MANA_ATTR_BY_KIND.values():
            setattr(env, attr, 10000.0)
        if staff:
            env._add_hero_item(staff["item_name"])
        for enemy_id in (1, 2):
            for unit in env._get_enemy_team_state(enemy_id):
                if unit.get("health", 0) > 0:
                    unit.update(health=1000, max_health=1000, immunity=[], resistance=[])
        return env

    yield create
    for env in opened:
        env.close()


def cast_action(env, source):
    if source == "learned":
        key = "lod_d2_s003"
        env.active_spells[key]["learned"] = 1
        entries = env._current_map_offensive_spell_action_specs()
        index = next(i for i, entry in enumerate(entries) if entry["id"] == key)
        return env.grid_legion_damage_spell_action_start + index, key
    if source == "purchased":
        key = "emp_d2_s022"
        env.spell_shop_purchased_spell_ids.add(key)
        entries = env._spell_shop_cast_spell_entries()
        index = next(i for i, entry in enumerate(entries) if entry["spell_id"] == key)
        return env.grid_spell_shop_cast_action_start + index, key
    entries = (env.scroll_cast_slot_entries() if source == "scroll"
               else env.staff_spell_action_entries())
    index = next(i for i, entry in enumerate(entries) if entry["spell_kind"] == "damage")
    start = env.grid_scroll_cast_action_start if source == "scroll" else env.grid_staff_spell_action_start
    return start + index, entries[index]["spell_id"]


@pytest.mark.parametrize("source", ("learned", "purchased", "scroll", "staff"))
@pytest.mark.parametrize("sealed_wall", (False, True))
def test_mask_and_public_cast_damage_the_same_enemy_across_wall(make_env, source, sealed_wall):
    env = make_env(source)
    env.grid_env.obstacle_positions = {(2, y) for y in range(env.grid_size if sealed_wall else 7)}
    env._territory_path_distances_cache.clear()
    action, key = cast_action(env, source)
    mask_target = env._get_map_offensive_spell_target(key, for_mask=True)
    assert mask_target["enemy_id"] == 1
    assert env.compute_action_mask()[action]

    def hp(enemy_id):
        return sum(float(unit.get("health", 0)) for unit in env._get_enemy_team_state(enemy_id))

    before = {enemy_id: hp(enemy_id) for enemy_id in (1, 2)}
    obs, _, _, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert not truncated
    assert info["spell_cast_executed"]
    assert info["target_enemy_id"] == mask_target["enemy_id"]
    assert info["target_enemy_position"] == (3, 1)
    assert hp(1) < before[1]
    assert hp(2) == before[2]
    assert before[1] - hp(1) == pytest.approx(info["spell_damage_total"])


@pytest.mark.parametrize("kind,allow_protected,expected", (
    ("damage", False, 2), ("debuff", False, 2), ("summon_battle", True, 1),
))
def test_resolver_fallback_preserves_spell_target_restrictions(make_env, kind, allow_protected, expected):
    env = make_env("learned")
    env.grid_env.obstacle_positions = {(2, y) for y in range(7)}
    protected = next(iter(env._spell_untargetable_enemy_ids()))
    env.grid_env.enemy_positions[protected] = env.grid_env.enemy_positions.pop(1)
    env.grid_env.enemies_alive[protected] = env.grid_env.enemies_alive.pop(1)
    env.enemy_team_states[protected] = env.enemy_team_states[1]
    target = env._resolve_spell_target_enemy("test", {"kind": kind})
    mask_target = env._get_nearest_enemy_stack_for_mask(spell_targetable_only=not allow_protected)
    assert target == mask_target
    assert target["enemy_id"] == (protected if expected == 1 else 2)
