"""B01: combat XP must promote the recipient, never a same-name neighbour."""
from copy import deepcopy

import pytest

from battle_env import TARGET_POSITIONS, WAIT_ACTION_INDEX, BattleEnv
from campaign_env import CampaignEnv


def make_env(map_name="default", capital=2):
    return CampaignEnv(
        map_name=map_name, Realcapital=capital, observation_version="local5",
        scripted_capital_bot_enabled=False, use_boss_starting_roster=False,
        log_enabled=False,
    )


def promotion_cases():
    """Cover every building-backed evolution, including branches and big units."""
    env = make_env()
    try:
        env.reset(seed=42)
        cases = []
        for source, data in env._unit_data_by_name().items():
            unit = env._build_unit_from_data(data, "blue", 7)
            if not unit.get("exp_required", 0) or env._is_hero_unit(unit):
                continue
            capital = data.get("столица")
            buildings = env._get_buildings_for_capital(capital)
            for target in unit.get("turns_into", []):
                if env._find_unit_data_by_name(target) is None:
                    continue
                if not any(isinstance(b, dict) and b.get("unit") == target
                           for b in buildings.values()):
                    continue
                for variant in ("dead_first", "alive_first", "first_earns",
                                "both_earn", "no_building"):
                    cases.append(pytest.param(
                        source, target, capital, variant, "default",
                        id=f"{source}->{target}:{variant}",
                    ))
        cases.append(pytest.param(
            "Гном", "Гном воин", 3, "dead_first", "wotans_retribution",
            id="wotan_dead_first",
        ))
        return cases
    finally:
        env.close()


@pytest.mark.parametrize("source,target,capital,variant,map_name", promotion_cases())
def test_promotion_targets_xp_recipient(source, target, capital, variant, map_name):
    env = make_env(map_name, capital)
    try:
        env.reset(seed=42)
        data = env._find_unit_data_by_name(source)
        first = env._build_unit_from_data(data, "blue", 7)
        second = env._build_unit_from_data(data, "blue", 9)
        first["exp_current"] = 0
        second["exp_current"] = second["exp_required"] - 1
        if variant == "dead_first":
            first.update(health=0, hp=0)
        elif variant == "first_earns":
            first["exp_current"], second["exp_current"] = second["exp_current"], 0
        elif variant == "both_earn":
            first["exp_current"] = second["exp_current"]

        # A controlled one-hit fight, with real combat XP and CampaignEnv.step.
        hero = deepcopy(env._resolve_travel_hero())
        hero.update(position=8, initiative=10000, initiative_base=10000,
                    damage=10000, original_damage=10000, accuracy=100,
                    exp_current=0, exp_required=1000000)
        env.blue_team_state = env._build_battle_team_with_placeholders(
            "blue", [first, hero, second])
        for building in env._get_buildings_for_capital(capital).values():
            if isinstance(building, dict) and building.get("unit"):
                building["Build"] = int(
                    building["unit"] == target and variant != "no_building")
                building["built"] = building["Build"]

        enemy_id = next(iter(env._enemy_configs))
        red = env._build_unit_from_data(
            env._find_unit_data_by_name("Крестьянин"), "red", 2)
        red.update(health=1, hp=1, max_health=1, maxhp=1, damage=0,
                   initiative=0, initiative_base=0, armor=0, exp_kill=30)
        env.enemy_team_states[enemy_id] = env._build_battle_team_with_placeholders(
            "red", [red])
        env.current_enemy_id = enemy_id
        env.current_battle_context = {"kind": "hero"}
        env.battle_origin_pos = tuple(env.grid_env.agent_pos)
        env.mode = env.MODE_BATTLE
        env._init_battle(enemy_id)
        battle = env.battle_env
        battle.rng.seed(42)
        assert battle.winner is None
        info = {}
        for _ in range(20):
            action = (WAIT_ACTION_INDEX if battle._post_victory_team == "blue"
                      else TARGET_POSITIONS.index(2))
            _, _, _, _, info = env.step(action)
            if env.mode != env.MODE_BATTLE:
                break
        assert info.get("battle_result") == "victory"
        assert battle._battle_exp_event_count > 0

        promoted = {9}
        if variant == "first_earns":
            promoted = {7}
        elif variant == "both_earn":
            promoted = {7, 9}
        elif variant == "no_building":
            promoted = set()
        for team in (battle.combined, env.blue_team_state):
            by_position = {u["position"]: u for u in team if u["team"] == "blue"}
            for position in (7, 9):
                unit = by_position[position]
                assert unit["name"] == (target if position in promoted else source)
                if position in promoted:
                    assert unit["health"] == unit["max_health"]
                    assert unit["exp_current"] == 0
            if variant == "dead_first":
                assert by_position[7]["health"] == 0
                assert by_position[7]["exp_current"] == 0
        assert info["unit_upgrades"] == len(promoted)
        expected_tier = env._find_unit_data_by_name(target)["уровень"]
        assert info["unit_upgrade_tier_sum"] == len(promoted) * int(expected_tier)
        assert info["unit_upgrade_reward"] == pytest.approx(
            len(promoted) * env._unit_upgrade_reward_for_tier(int(expected_tier)))
    finally:
        env.close()


@pytest.mark.parametrize("first_dead", [True, False])
def test_scripted_promotion_uses_recipient_without_running_bot(first_dead):
    env = make_env()
    try:
        env.reset(seed=42)
        data = env._find_unit_data_by_name("Одержимый")
        first = env._build_unit_from_data(data, "blue", 7)
        second = env._build_unit_from_data(data, "blue", 9)
        first["exp_current"] = 0
        second["exp_current"] = second["exp_required"] - 1
        if first_dead:
            first.update(health=0, hp=0)
        before = deepcopy(first)
        battle = BattleEnv()
        battle.combined = [first, second]
        battle._apply_exp_award_to_unit(second, 10)
        assert env._apply_scripted_capital_bot_levelups(battle) == 1
        assert first == before
        assert second["name"] == "Берсерк"
        # Re-reading the same event must not chain-evolve the unit again.
        assert env._apply_scripted_capital_bot_levelups(battle) == 0
    finally:
        env.close()


def test_red_levelup_does_not_promote_same_named_blue_unit():
    env = make_env()
    try:
        env.reset(seed=42)
        data = env._find_unit_data_by_name("Одержимый")
        blue = env._build_unit_from_data(data, "blue", 7)
        blue.update(health=0, hp=0)
        red = env._build_unit_from_data(data, "red", 1)
        red["exp_current"] = red["exp_required"] - 1
        for building in env.active_buildings.values():
            if isinstance(building, dict) and building.get("unit") == "Берсерк":
                building["Build"] = 1
        env.battle_env = BattleEnv()
        env.battle_env.combined = [blue, red]
        env.battle_env._apply_exp_award_to_unit(red, 10)
        before = deepcopy(blue)
        assert env._log_turns_into_levelups() == 0
        assert blue == before
        assert env._last_upgrade_reward == 0
    finally:
        env.close()


@pytest.mark.parametrize("custom_teams", [False, True])
def test_new_battle_clears_levelup_recipients(custom_teams):
    battle = BattleEnv()
    battle.reset()
    old_blue = [deepcopy(u) for u in battle.combined if u["team"] == "blue"]
    old_red = [deepcopy(u) for u in battle.combined if u["team"] == "red"]
    recipient = next(u for u in battle.combined if u["team"] == "blue"
                     and u.get("exp_required", 0) and battle._alive(u))
    recipient["exp_current"] = recipient["exp_required"] - 1
    battle._apply_exp_award_to_unit(recipient, 10)
    assert battle.last_levelup_units[0] is recipient
    if custom_teams:
        battle._init_with_custom_teams(old_red, old_blue)
    else:
        battle.reset()
    assert battle.last_levelups == []
    assert battle.last_levelup_units == []
