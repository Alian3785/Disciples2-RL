from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from campaign_env import CampaignEnv as _CampaignEnv


class CampaignEnv(_CampaignEnv):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_boss_starting_roster", False)
        super().__init__(*args, **kwargs)


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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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


def test_active_spells_are_isolated_between_campaign_envs():
    first_env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    second_env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    first_env.reset(seed=123)
    second_env.reset(seed=456)

    spell_key = first_env.spell_keys[0]
    assert first_env.active_spells is not second_env.active_spells
    assert first_env.active_spells[spell_key] is not second_env.active_spells[spell_key]

    first_env.active_spells[spell_key]["learned"] = 1
    assert second_env.active_spells[spell_key]["learned"] == 0

    second_env.reset(seed=789)
    assert first_env.active_spells[spell_key]["learned"] == 1


def test_typeoflord_two_halves_learning_costs_but_not_use_costs():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
    env.reset(seed=123)
    _mark_magic_tower_built(env)
    env.typeoflord = 2

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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)
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
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)

    _, info = env.reset(seed=123)

    assert env.typeoflord == 1
    assert info["typeoflord"] == 1
    assert env.get_state_summary()["typeoflord"] == 1

    _, _, _, _, step_info = env.step(8)
    assert step_info["typeoflord"] == 1


def _expected_blue_names(entries):
    expected = {position: "\u043f\u0443\u0441\u0442\u043e" for position in range(7, 13)}
    expected.update(entries)
    return expected


def test_Realcapital_and_typeoflord_select_starting_blue_roster():
    squire = "\u0421\u043a\u0432\u0430\u0435\u0440"
    pegasus_knight = "\u0420\u044b\u0446\u0430\u0440\u044c \u043d\u0430 \u043f\u0435\u0433\u0430\u0441\u0435"
    empire_ranger = "\u0420\u0435\u0439\u043d\u0434\u0436\u0435\u0440 \u043b\u044e\u0434\u0435\u0439"
    archmage = "\u0410\u0440\u0445\u0438\u043c\u0430\u0433"
    acolyte = "\u0421\u043b\u0443\u0436\u043a\u0430"
    possessed = "\u041e\u0434\u0435\u0440\u0436\u0438\u043c\u044b\u0439"
    duke = "\u0413\u0435\u0440\u0446\u043e\u0433"
    counselor = "\u0421\u043e\u0432\u0435\u0442\u043d\u0438\u043a"
    archdevil = "\u0410\u0440\u0445\u0438\u0434\u044c\u044f\u0432\u043e\u043b"
    cultist = "\u0421\u0435\u043a\u0442\u0430\u043d\u0442"
    dwarf = "\u0413\u043d\u043e\u043c"
    royal_guard = "\u041a\u043e\u0440\u043e\u043b\u0435\u0432\u0441\u043a\u0438\u0439 \u0441\u0442\u0440\u0430\u0436"
    engineer = "\u0418\u043d\u0436\u0435\u043d\u0435\u0440"
    lorekeeper = "\u0425\u0440\u0430\u043d\u0438\u0442\u0435\u043b\u044c \u0437\u043d\u0430\u043d\u0438\u0439"
    herbalist = "\u0422\u0440\u0430\u0432\u043d\u0438\u0446\u0430"
    warrior = "\u0412\u043e\u0438\u043d"
    death_knight = "\u0420\u044b\u0446\u0430\u0440\u044c \u0441\u043c\u0435\u0440\u0442\u0438"
    nosferatu = "\u041d\u043e\u0441\u0444\u0435\u0440\u0430\u0442\u0443"
    lich_queen = "\u041a\u043e\u0440\u043e\u043b\u0435\u0432\u0430 \u043b\u0438\u0447"
    adept = "\u0410\u0434\u0435\u043f\u0442"
    centaur = "\u041a\u0435\u043d\u0442\u0430\u0432\u0440 \u043a\u043e\u043f\u0435\u0439\u0449\u0438\u043a"
    forest_lord = "\u041b\u0435\u0441\u043d\u043e\u0439 \u043b\u043e\u0440\u0434"
    forest_guardian = "\u0425\u0440\u0430\u043d\u0438\u0442\u0435\u043b\u044c \u043b\u0435\u0441\u0430"
    dryad = "\u0414\u0440\u0438\u0430\u0434\u0430"
    medium = "\u041c\u0435\u0434\u0438\u0443\u043c"
    cases = [
        (
            1,
            1,
            _expected_blue_names({7: squire, 8: pegasus_knight, 9: squire, 11: acolyte}),
            8,
            pegasus_knight,
        ),
        (
            1,
            2,
            _expected_blue_names({7: squire, 9: squire, 10: archmage, 12: acolyte}),
            10,
            archmage,
        ),
        (
            1,
            3,
            _expected_blue_names({7: squire, 9: squire, 10: empire_ranger, 12: acolyte}),
            10,
            empire_ranger,
        ),
        (
            2,
            1,
            _expected_blue_names({7: possessed, 8: duke, 9: possessed, 11: cultist}),
            8,
            duke,
        ),
        (
            2,
            2,
            _expected_blue_names({7: possessed, 9: possessed, 10: archdevil, 12: cultist}),
            10,
            archdevil,
        ),
        (
            2,
            3,
            _expected_blue_names({7: possessed, 9: possessed, 10: counselor, 12: cultist}),
            10,
            counselor,
        ),
        (
            3,
            1,
            _expected_blue_names({7: dwarf, 8: royal_guard, 9: dwarf, 11: herbalist}),
            8,
            royal_guard,
        ),
        (
            3,
            2,
            _expected_blue_names({7: dwarf, 9: dwarf, 10: lorekeeper, 12: herbalist}),
            10,
            lorekeeper,
        ),
        (
            3,
            3,
            _expected_blue_names({7: dwarf, 9: dwarf, 10: engineer, 12: herbalist}),
            10,
            engineer,
        ),
        (
            4,
            1,
            _expected_blue_names({7: warrior, 8: death_knight, 9: warrior, 11: adept}),
            8,
            death_knight,
        ),
        (
            4,
            2,
            _expected_blue_names({7: warrior, 9: warrior, 10: lich_queen, 12: adept}),
            10,
            lich_queen,
        ),
        (
            4,
            3,
            _expected_blue_names({7: warrior, 9: warrior, 10: nosferatu, 12: adept}),
            10,
            nosferatu,
        ),
        (
            5,
            1,
            _expected_blue_names({7: centaur, 8: forest_lord, 9: centaur, 11: medium}),
            8,
            forest_lord,
        ),
        (
            5,
            2,
            _expected_blue_names({7: centaur, 9: centaur, 10: dryad, 12: medium}),
            10,
            dryad,
        ),
        (
            5,
            3,
            _expected_blue_names({7: centaur, 9: centaur, 10: forest_guardian, 12: medium}),
            10,
            forest_guardian,
        ),
    ]

    for Realcapital, lord_type, expected_names, hero_position, hero_name in cases:
        env = CampaignEnv(
            log_enabled=False,
            persist_blue_hp=True,
            Realcapital=Realcapital,
            typeoflord=lord_type,
        )
        _, info = env.reset(seed=123)

        names_by_position = {
            int(unit["position"]): str(unit["name"])
            for unit in env.blue_team_state
        }
        heroes_by_position = {
            int(unit["position"]): bool(unit.get("hero"))
            for unit in env.blue_team_state
        }

        assert env.Realcapital == Realcapital
        assert info["typeoflord"] == lord_type
        assert names_by_position == expected_names
        assert heroes_by_position[hero_position] is True
        assert env._resolve_travel_hero()["name"] == hero_name
        assert names_by_position[10] != "\u0411\u0430\u0440\u043e\u043d\u0435\u0441\u0441\u0430"


def test_reset_options_can_change_typeoflord_starting_roster():
    env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=2)

    env.reset(seed=123, options={"Realcapital": 5, "typeoflord": 3})

    assert env.Realcapital == 5
    assert env.typeoflord == 3
    assert env._resolve_travel_hero()["name"] == "\u0425\u0440\u0430\u043d\u0438\u0442\u0435\u043b\u044c \u043b\u0435\u0441\u0430"

    env.Realcapital = 3
    env.reset(seed=123, options={"typeoflord": 2})

    assert env.Realcapital == 3
    assert env._resolve_travel_hero()["name"] == "\u0425\u0440\u0430\u043d\u0438\u0442\u0435\u043b\u044c \u0437\u043d\u0430\u043d\u0438\u0439"


def test_freeze_dynamic_action_layout_skips_reset_refresh():
    env = CampaignEnv(
        log_enabled=False,
        persist_blue_hp=True,
        Realcapital=2,
        freeze_dynamic_action_layout=True,
    )
    action_count = env.action_space.n
    spell_action_start = env.grid_spell_action_start
    spell_keys = tuple(env.spell_keys)
    swap_pairs = tuple(env.GRID_SWAP_UNIT_PAIRS)

    def fail_refresh():
        raise AssertionError("dynamic action layout was refreshed during reset")

    env._refresh_dynamic_action_layout = fail_refresh
    env.reset(seed=123)
    action = env.GRID_SWAP_UNIT_ACTION_START + env.GRID_SWAP_UNIT_PAIRS.index((7, 10))
    env.step(action)
    assert env.grid_unit_swaps_total == 1

    for unit in env.blue_team_state:
        if isinstance(unit, dict) and int(unit.get("position", -1) or -1) == 7:
            unit["position"] = 10
            unit["stand"] = "behind"
            break

    env.reset(seed=123)

    assert env.action_space.n == action_count
    assert env.grid_spell_action_start == spell_action_start
    assert tuple(env.spell_keys) == spell_keys
    assert tuple(env.GRID_SWAP_UNIT_PAIRS) == swap_pairs
    assert env.grid_unit_swaps_total == 0
    assert env.grid_unit_swap_actions_used_this_turn == set()

    mask = env.compute_action_mask()
    assert len(mask) == action_count
    _obs, _reward, terminated, truncated, info = env.step(action)
    assert terminated is False
    assert truncated is False
    assert info["unit_swap_action"] is True

    with pytest.raises(ValueError, match="freeze_dynamic_action_layout"):
        env.reset(seed=123, options={"Realcapital": 5})


def test_level_five_spells_are_masked_unless_typeoflord_is_two_for_all_races():
    for Realcapital in (1, 2, 3, 4, 5):
        env = CampaignEnv(log_enabled=False, persist_blue_hp=True, Realcapital=Realcapital)
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
        spell_mask = _spell_mask(env)
        assert bool(spell_mask[env.spell_keys.index(spell_key)]) is True
