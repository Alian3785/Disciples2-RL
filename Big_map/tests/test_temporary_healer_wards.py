"""Temporary healer wards preserve native wards and never become campaign stats."""

from copy import deepcopy

import pytest

from battle_env import BattleEnv, DEFEND_ACTION_INDEX
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle

UNITS = {entry["кто"]: entry for entry in DATA}
HEALERS = (
    ("Солнечная танцовщица", ("Fire",)),
    ("Сильфида", ("Air",)),
    ("Дева рощи", ("Fire", "Air", "Water", "Earth")),
)


def unit(name, team="blue", position=7):
    return map_unit_to_battle(UNITS[name], team, position)


def fixture(healer_name, team="blue", native=(), used=()):
    battle = BattleEnv()
    healer = unit(healer_name, team, 10 if team == "blue" else 4)
    target = unit("Модеус", team, 7 if team == "blue" else 1)
    target["resistance"] = list(native)
    target["resilience_used_types"] = list(used)
    battle.combined = [healer, target]
    return battle, healer, target


def blocked(battle, target, element):
    return battle._resilience_blocks({"attack_type_primary": element}, target)


def assert_clean(target):
    assert not target.get("_temporary_healer_wards")
    assert all(
        not target.get(element + "defence", 0)
        for element in ("Fire", "Air", "Water", "Earth")
    )


@pytest.mark.parametrize("healer_name,elements", HEALERS)
@pytest.mark.parametrize("team", ("blue", "red"))
@pytest.mark.parametrize("native_used", (False, True))
@pytest.mark.parametrize("hit", (False, True))
def test_expiry_preserves_native_membership_and_consumption(
    healer_name, elements, team, native_used, hit
):
    native = (*elements, "Mind")
    before_used = elements if native_used else ()
    battle, healer, target = fixture(healer_name, team, native, before_used)
    # Protection applies even when healing is unnecessary.
    assert not battle._apply_cliric_heal(healer, target)
    if hit:
        for element in elements:
            assert blocked(battle, target, element)
            assert not blocked(battle, target, element)
    battle._apply_start_of_turn_effects(healer)
    assert target["resistance"] == list(native)
    for element in elements:
        assert blocked(battle, target, element) is not (native_used or hit)
    assert blocked(battle, target, "Mind")
    assert_clean(target)


@pytest.mark.parametrize("healer_name,elements", HEALERS)
@pytest.mark.parametrize("team", ("blue", "red"))
def test_temporary_only_blocks_one_matching_hit_and_expires(
    healer_name, elements, team
):
    battle, healer, target = fixture(healer_name, team, ("Death",))
    battle._apply_cliric_heal(healer, target)
    for element in elements:
        assert blocked(battle, target, element)
        assert not blocked(battle, target, element)
    battle._apply_start_of_turn_effects(healer)
    assert target["resistance"] == ["Death"]
    assert target["resilience_used_types"] == []
    assert_clean(target)


@pytest.mark.parametrize("native", (False, True))
def test_recast_refreshes_once_without_forgetting_native_consumption(native):
    battle, healer, target = fixture(
        "Солнечная танцовщица", native=("Fire",) if native else ()
    )
    for _ in range(3):
        battle._apply_cliric_heal(healer, target)
        assert blocked(battle, target, "Fire")
        assert not blocked(battle, target, "Fire")
    battle._apply_cliric_heal(healer, target)
    battle._apply_start_of_turn_effects(healer)
    assert target["resistance"] == (["Fire"] if native else [])
    assert not blocked(battle, target, "Fire")
    assert_clean(target)


@pytest.mark.parametrize("same_type", (False, True))
@pytest.mark.parametrize("hit", (False, True))
def test_overlapping_casters_expire_only_their_own_grant(same_type, hit):
    battle, first, target = fixture("Солнечная танцовщица")
    second = unit("Солнечная танцовщица" if same_type else "Дева рощи", position=11)
    battle.combined.append(second)
    battle._apply_cliric_heal(first, target)
    battle._apply_cliric_heal(second, target)
    if hit:
        assert blocked(battle, target, "Fire")
    battle._apply_start_of_turn_effects(first)
    assert "Fire" in target["resistance"]
    assert blocked(battle, target, "Fire") is not hit
    assert not blocked(battle, target, "Fire")
    battle._apply_start_of_turn_effects(second)
    assert target["resistance"] == []
    assert_clean(target)


def test_case_insensitive_native_ward_is_not_duplicated_or_rearmed():
    battle, healer, target = fixture(
        "Солнечная танцовщица", native=("fIrE", "Mind"), used=("FIRE",)
    )
    battle._apply_cliric_heal(healer, target)
    assert target["resistance"] == ["fIrE", "Mind"]
    assert blocked(battle, target, "fire")
    battle._apply_cliric_heal(healer, target)
    battle._apply_start_of_turn_effects(healer)
    assert target["resistance"] == ["fIrE", "Mind"]
    assert not blocked(battle, target, "FIRE")


@pytest.fixture
def campaign():
    env = CampaignEnv(
        map_name="trade_train",
        use_boss_starting_roster=False,
        scripted_capital_bot_enabled=False,
        observation_version="local5",
        log_enabled=False,
    )
    env.reset(seed=42)
    yield env
    env.close()


@pytest.mark.parametrize("healer_name,elements", HEALERS)
@pytest.mark.parametrize("outcome", ("blue", "red", "timeout", "escaped"))
def test_campaign_save_and_next_battle_drop_only_temporary_wards(
    campaign, healer_name, elements, outcome
):
    env = campaign
    enemy_id = next(iter(env._enemy_configs))
    blue = [unit("Герцог"), unit(healer_name, position=10)]
    red = [unit("Модеус", "red", 1), unit(healer_name, "red", 4)]
    env.blue_team_state = env._build_battle_team_with_placeholders("blue", blue)
    env.enemy_team_states[enemy_id] = env._build_battle_team_with_placeholders(
        "red", red
    )
    expected = {u["team"]: deepcopy(u["resistance"]) for u in (blue[0], red[0])}
    battle = BattleEnv()
    battle.combined = deepcopy(env.blue_team_state + env.enemy_team_states[enemy_id])
    env.battle_env = battle
    for target_pos, healer_pos in ((7, 10), (1, 4)):
        target = battle._unit_by_position(target_pos)
        battle._apply_cliric_heal(battle._unit_by_position(healer_pos), target)
    if outcome == "escaped":
        battle._mark_unit_escaped(battle._unit_by_position(7))
        battle.winner = "red"
    else:
        battle.winner = None if outcome == "timeout" else outcome
    env._save_blue_state()
    env._save_enemy_state_from_battle(enemy_id)
    for team, saved, pos in (
        ("blue", env.blue_team_state, 7),
        ("red", env.enemy_team_states[enemy_id], 1),
    ):
        target = next(u for u in saved if u["position"] == pos)
        assert target["resistance"] == expected[team]
        assert_clean(target)
    rebuilt = BattleEnv()
    rebuilt._init_with_custom_teams(
        env.enemy_team_states[enemy_id], env.blue_team_state
    )
    for team, pos in (("blue", 7), ("red", 1)):
        target = rebuilt._unit_by_position(pos)
        assert target["resistance"] == expected[team]
        assert_clean(target)


@pytest.mark.parametrize("team", ("blue", "red"))
def test_battle_victory_cleans_active_and_escaped_wards(team):
    battle, healer, target = fixture("Дева рощи", team, ("Fire",))
    battle._apply_cliric_heal(healer, target)
    escaped = deepcopy(target)
    escaped["position"] += 1
    battle.escaped_units.append(escaped)
    battle._finalize_victory(team)
    for saved in (target, escaped):
        assert saved["resistance"] == ["Fire"]
        assert_clean(saved)


@pytest.mark.parametrize("healer_name,elements", HEALERS)
def test_real_attack_path_blocks_once_then_damage_lands(healer_name, elements):
    battle, healer, target = fixture(healer_name)
    target.update(health=1000, max_health=1000, immunity=[])
    attacker = unit("Рыцарь", "red", 1)
    attacker.update(accuracy=100, damage=20)
    battle.combined.append(attacker)
    # The healing action maps the opponent's formation cell to its ally.
    assert battle._attack(healer, 1)[0]
    for element in elements:
        attacker["attack_type_primary"] = element
        before = target["health"]
        battle._attack(attacker, target["position"])
        assert target["health"] == before
        battle._attack(attacker, target["position"])
        assert target["health"] < before


@pytest.mark.parametrize("outcome", ("victory", "timeout"))
def test_real_campaign_battle_boundary_clears_both_teams(campaign, outcome):
    env = campaign
    enemy_id = next(iter(env._enemy_configs))
    env.blue_team_state = env._build_battle_team_with_placeholders(
        "blue", [unit("Герцог"), unit("Дева рощи", position=10)]
    )
    env.enemy_team_states[enemy_id] = env._build_battle_team_with_placeholders(
        "red", [unit("Модеус", "red", 1), unit("Дева рощи", "red", 4)]
    )
    env.current_enemy_id = enemy_id
    env.battle_origin_pos = tuple(env.grid_env.agent_pos)
    env.current_battle_context = {"kind": "hero"}
    env.mode = env.MODE_BATTLE
    env._init_battle(enemy_id)
    battle = env.battle_env
    hero, enemy = battle._unit_by_position(7), battle._unit_by_position(1)
    baseline = {7: deepcopy(hero["resistance"]), 1: deepcopy(enemy["resistance"])}
    for target_pos, healer_pos in ((7, 10), (1, 4)):
        battle._apply_cliric_heal(
            battle._unit_by_position(healer_pos), battle._unit_by_position(target_pos)
        )
    if outcome == "victory":
        for combatant in battle.combined:
            if combatant["team"] == "red":
                combatant["health"] = 0
        battle._finalize_victory("blue")
    else:
        # Reach the actual step-limit branch before either caster activates again.
        for combatant in battle.combined:
            combatant["initiative"] = 0
            combatant["initiative_base"] = 1
            combatant["damage"] = 0
        hero["initiative"] = hero["initiative_base"] = 100
        battle.current_blue_attacker_pos = 7
        battle.blue_attacks_left = 1
        battle.step_count = 999
    info = env.step(DEFEND_ACTION_INDEX)[-1]
    assert env.battle_env is None
    assert info["battle_result"] == ("timeout" if outcome == "timeout" else "victory")
    saved = next(u for u in env.blue_team_state if u["position"] == 7)
    assert saved["resistance"] == baseline[7]
    assert_clean(saved)
    if outcome == "timeout":
        saved = next(u for u in env.enemy_team_states[enemy_id] if u["position"] == 1)
        assert saved["resistance"] == baseline[1]
        assert_clean(saved)
    for combatant in battle.combined:
        assert_clean(combatant)


def test_new_battle_drops_active_grants_from_input_without_mutating_input():
    battle, healer, target = fixture("Дева рощи", native=("Fire",))
    battle._apply_cliric_heal(healer, target)
    original = deepcopy(target)
    rebuilt = BattleEnv()
    rebuilt._init_with_custom_teams([unit("Рыцарь", "red", 1)], [healer, target])
    saved = rebuilt._unit_by_position(target["position"])
    assert saved["resistance"] == ["Fire"]
    assert_clean(saved)
    assert target == original
