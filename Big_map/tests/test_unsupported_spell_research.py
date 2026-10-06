"""Research must have an implemented cast action, without changing action IDs."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from campaign_env import CampaignEnv
from maps import available_maps, get_map


UNSUPPORTED_BY_CAPITAL = {
    1: {"emp_d2_s009", "emp_d2_s020"},
    2: {
        "lod_d2_s002", "lod_d2_s007", "lod_d2_s011", "lod_d2_s012",
        "lod_d2_s013", "lod_d2_s022", "lod_d2_s024",
    },
    3: {"mcl_d2_s010", "mcl_d2_s022"},
    4: {"und_d2_s008", "und_d2_s014", "und_d2_s019", "und_d2_s020"},
    5: {"elf_d2_s003", "elf_d2_s011", "elf_d2_s023", "elf_d2_s024"},
}


def make_research_env(map_name="default", capital=1):
    # Override fixed-faction scenario settings before observation layout is built.
    # Do not mutate the shared map config or change any other scenario rules.
    config = replace(get_map(map_name), starting_capital_id=None, starting_lord_type=None)
    with patch("campaign_env.get_map", return_value=config):
        env = CampaignEnv(
            map_name=map_name,
            Realcapital=capital,
            typeoflord=2,
            observation_version="local5",
            scripted_capital_bot_enabled=False,
            use_boss_starting_roster=False,
            log_enabled=False,
        )
    env.reset(seed=42, options={"Realcapital": capital, "typeoflord": 2})
    assert env.Realcapital == capital
    env._has_magic_tower_built = lambda: True
    env.spell_learning_locked = False
    for attr in env.MANA_ATTR_BY_KIND.values():
        setattr(env, attr, 100_000.0)
    for spell_key in env.spell_keys:
        env.active_spells[spell_key]["learned"] = 0
    return env


def research_state(env):
    return {
        "mana": env._current_mana_totals(),
        "spells": deepcopy(env.active_spells),
        "learning_locked": env.spell_learning_locked,
        "cast_counts": dict(env.spell_cast_counts_by_id_this_turn),
        "turns": env.turns,
        "moves": env.moves,
        "first_use_counts": dict(env.first_use_counts),
    }


@pytest.mark.parametrize("map_name", available_maps())
@pytest.mark.parametrize("capital", range(1, 6))
def test_research_mask_and_execution_for_every_spell_map_and_faction(map_name, capital):
    env = make_research_env(map_name, capital)
    try:
        unsupported = UNSUPPORTED_BY_CAPITAL[capital]
        shape, count = env.observation_space.shape, env.action_space.n
        keys = tuple(env.spell_keys)
        start = env.grid_spell_action_start
        assert len(keys) == 24
        assert unsupported <= set(keys)
        mask = env.compute_action_mask()
        assert mask.shape == (count,)
        for index, key in enumerate(keys):
            supported = key not in unsupported
            assert env._is_spell_research_supported(key) is supported
            assert bool(mask[start + index]) is supported

            env.spell_learning_locked = False
            before = research_state(env)
            costs = env._get_spell_learning_costs(env.active_spells[key])
            result = env._step_learn_spell(start + index)
            info = result[-1]
            assert info["spell_research_supported"] is supported
            assert info["spell_learned"] is supported
            if supported:
                assert env.active_spells[key]["learned"] == 1
                assert env.spell_learning_locked
                assert info["spell_learning_reward"] == pytest.approx(env.reward_spell_learn)
                for kind, amount in before["mana"].items():
                    assert env._current_mana_totals()[kind] == pytest.approx(amount - costs[kind])
            else:
                assert research_state(env) == before
                assert info["spell_learning_reward"] == 0
                assert result[1] == 0
        assert env.observation_space.shape == shape
        assert env.action_space.n == count
        assert tuple(env.spell_keys) == keys
        assert env.grid_spell_action_start == start
    finally:
        env.close()


@pytest.mark.parametrize("capital", range(1, 6))
@pytest.mark.parametrize("already_learned,locked", [(0, False), (1, False), (0, True), (1, True)])
def test_forced_unsupported_step_preserves_research_state(capital, already_learned, locked):
    env = make_research_env(capital=capital)
    try:
        for key in sorted(UNSUPPORTED_BY_CAPITAL[capital]):
            env.active_spells[key]["learned"] = already_learned
            env.spell_learning_locked = locked
            action = env.grid_spell_action_start + env.spell_keys.index(key)
            before = research_state(env)
            assert not env.compute_action_mask()[action]
            result = env.step(action)
            assert not result[-1]["spell_learned"]
            assert not result[-1]["spell_research_supported"]
            assert result[-1]["spell_learning_reward"] == 0
            assert research_state(env) == before
    finally:
        env.close()


@pytest.mark.parametrize("capital", range(1, 6))
@pytest.mark.parametrize("blocker", ["tower", "daily_limit", "mana", "learned", "lord_level"])
def test_supported_research_keeps_existing_requirements(capital, blocker):
    env = make_research_env(capital=capital)
    try:
        key = next(
            key for key in env.spell_keys
            if env._is_spell_research_supported(key)
            and int(env.active_spells[key]["level"]) == 5
        )
        action = env.grid_spell_action_start + env.spell_keys.index(key)
        assert env.compute_action_mask()[action]
        if blocker == "tower":
            env._has_magic_tower_built = lambda: False
        elif blocker == "daily_limit":
            env.spell_learning_locked = True
        elif blocker == "mana":
            for attr in env.MANA_ATTR_BY_KIND.values():
                setattr(env, attr, 0.0)
        elif blocker == "learned":
            env.active_spells[key]["learned"] = 1
        else:
            env.typeoflord = 1
        before = research_state(env)
        assert not env.compute_action_mask()[action]
        info = env._step_learn_spell(action)[-1]
        assert info["spell_research_supported"]
        assert not info["spell_learned"]
        assert info["spell_learning_reward"] == 0
        assert research_state(env) == before
    finally:
        env.close()


def test_support_predicate_uses_current_faction_catalog_and_disabled_override(monkeypatch):
    env = make_research_env(capital=1)
    try:
        assert not env._is_spell_research_supported("")
        assert not env._is_spell_research_supported(None)
        assert not env._is_spell_research_supported("missing_spell")
        assert not env._is_spell_research_supported("lod_d2_s001")
        # A future catalog addition becomes researchable without another denylist.
        monkeypatch.setattr(
            env, "_map_castable_spell_ids_for_capital",
            lambda capital: ("future_spell", "lod_d2_s002"),
        )
        assert env._is_spell_research_supported("future_spell")
        assert not env._is_spell_research_supported("lod_d2_s002")
    finally:
        env.close()


def test_catalog_totals_are_101_supported_and_19_unsupported():
    total = supported = 0
    for capital in range(1, 6):
        env = make_research_env(capital=capital)
        try:
            total += len(env.spell_keys)
            supported += sum(env._is_spell_research_supported(key) for key in env.spell_keys)
        finally:
            env.close()
    assert total == 120
    assert supported == 101
    assert sum(map(len, UNSUPPORTED_BY_CAPITAL.values())) == 19
