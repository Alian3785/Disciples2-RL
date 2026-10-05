"""Only this battle's deaths fund XP, including across campaign retreats."""

from copy import deepcopy
import pickle

import pytest

import battle_env as battle_module
from battle_env import BattleEnv, RUN_AWAY_ACTION_INDEX
from campaign_env import CampaignEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle, placeholder_unit


DATA_BY_NAME = {unit["кто"]: unit for unit in DATA}


def make_unit(team, offset=0, *, name="Рыцарь", dead=False, xp=60):
    unit = map_unit_to_battle(DATA_BY_NAME[name], team, (7 if team == "blue" else 1) + offset)
    unit.update(exp_current=0, exp_required=100000, exp_kill=xp,
                initiative_base=1000 if team == "blue" else 0, initiative=0,
                immunity=[], resistance=[])
    if dead:
        unit.update(health=0, hp=0, initiative_base=0)
    return unit


def start(units, *, initializer="custom", monkeypatch=None, battle=None):
    battle = battle or BattleEnv(log_enabled=False)
    battle.rng.seed(42)
    occupied = {unit["position"] for unit in units}
    units = units + [placeholder_unit("red" if pos < 7 else "blue", pos)
                     for pos in range(1, 13) if pos not in occupied]
    red = [unit for unit in units if unit["team"] == "red"]
    blue = [unit for unit in units if unit["team"] == "blue"]
    if initializer == "default":
        monkeypatch.setattr(battle_module, "UNITS_RED", red)
        monkeypatch.setattr(battle_module, "UNITS_BLUE", blue)
        battle._reset_state()
    else:
        battle._init_with_custom_teams(red, blue)
    return battle


def get(battle, team, offset=0):
    return battle._unit_by_position((7 if team == "blue" else 1) + offset)


def kill(battle, unit):
    assert unit["health"] > 0
    battle._subtract_health(unit, unit["health"])
    unit["initiative"] = 0
    assert unit["health"] == 0


def assert_no_xp_state_leak(battle):
    for unit in [*battle.combined, *battle.escaped_units]:
        assert "_battle_exp_earned" not in unit


@pytest.mark.parametrize("losing_team", ["blue", "red"])
@pytest.mark.parametrize("initializer", ["custom", "default"])
@pytest.mark.parametrize("retreat", [False, True], ids=["corpses_only", "retreat"])
@pytest.mark.parametrize("ledger", ["zero", "empty", "absent"])
def test_initialized_zero_death_battle_never_rewards_old_corpses(
    losing_team, initializer, retreat, ledger, monkeypatch,
):
    winning_team = "red" if losing_team == "blue" else "blue"
    units = [make_unit(winning_team), make_unit(losing_team, dead=True)]
    if retreat:
        units.append(make_unit(losing_team, 1, name="Герцог"))
    battle = start(units, initializer=initializer, monkeypatch=monkeypatch)
    winner = get(battle, winning_team)
    winner["exp_current"] = 17
    if ledger == "empty":
        battle._battle_defeated_exp.clear()
    elif ledger == "absent":
        del battle._battle_defeated_exp
    if retreat:
        battle._mark_unit_escaped(get(battle, losing_team, 1))
    battle._check_victory_after_hit()
    assert battle.winner == winning_team
    assert battle._battle_exp_event_count == 0
    assert battle.last_battle_exp == 0
    assert winner["exp_current"] == 17
    assert battle.last_levelups == []
    assert_no_xp_state_leak(battle)


@pytest.mark.parametrize("winning_team", ["blue", "red"])
def test_real_kill_ignores_existing_corpse_and_resets_next_battle(winning_team):
    losing_team = "red" if winning_team == "blue" else "blue"
    battle = start([make_unit(winning_team), make_unit(losing_team),
                    make_unit(losing_team, 1, dead=True, xp=500)])
    battle.exp_multiplier = 1.5
    winner = get(battle, winning_team)
    kill(battle, get(battle, losing_team))
    battle._check_victory_after_hit()
    expected = 90 if winning_team == "blue" else 60
    assert winner["exp_current"] == expected
    assert battle.last_battle_exp == expected
    assert battle._battle_exp_event_count == 1
    assert_no_xp_state_leak(battle)

    # Carry the actual aftermath into a new fight, then retreat without a kill.
    units = [deepcopy(unit) for unit in battle.combined if unit["name"] != "пусто"]
    units.append(make_unit(losing_team, 2, name="Герцог"))
    start(units, battle=battle)
    battle._mark_unit_escaped(get(battle, losing_team, 2))
    battle._check_victory_after_hit()
    assert battle._battle_exp_event_count == 0
    assert battle.last_battle_exp == 0
    assert get(battle, winning_team)["exp_current"] == expected


@pytest.mark.parametrize("winning_team", ["blue", "red"])
@pytest.mark.parametrize("reviver", ["item", "patriarch"])
def test_old_corpse_revived_and_killed_again_still_awards_each_new_death(
    winning_team, reviver,
):
    losing_team = "red" if winning_team == "blue" else "blue"
    battle = start([make_unit(winning_team), make_unit(losing_team, dead=True),
                    make_unit(losing_team, 1, name="Герцог")])
    victim = get(battle, losing_team)
    if reviver == "patriarch":
        healer = make_unit(losing_team, 2, name="Патриарх")
        assert battle._apply_patriach_support(healer, victim) == "revive_success"
    else:
        assert battle._apply_hero_item_revive(victim, 10) == 10
    kill(battle, victim)
    assert battle._apply_hero_item_revive(victim, 10) == 10
    kill(battle, victim)
    battle._mark_unit_escaped(get(battle, losing_team, 1))
    battle._check_victory_after_hit()
    assert battle._battle_exp_event_count == 2
    assert battle.last_battle_exp == 120
    assert get(battle, winning_team)["exp_current"] == 120
    assert_no_xp_state_leak(battle)


@pytest.mark.parametrize("winning_team", ["blue", "red"])
def test_escape_and_recipient_revival_keep_existing_death_timing(winning_team):
    losing_team = "red" if winning_team == "blue" else "blue"
    battle = start([make_unit(winning_team), make_unit(winning_team, 1),
                    make_unit(losing_team, xp=60), make_unit(losing_team, 1, xp=90),
                    make_unit(losing_team, 2, xp=120),
                    make_unit(losing_team, 3, dead=True, xp=500)])
    survivor = get(battle, winning_team, 1)
    kill(battle, get(battle, losing_team))  # 30 each.
    battle._mark_unit_escaped(get(battle, winning_team))
    kill(battle, get(battle, losing_team, 1))  # 90 for the remaining winner.
    kill(battle, survivor)  # Loses the 120 earned before death.
    assert battle._apply_hero_item_revive(survivor, 10) == 10
    kill(battle, get(battle, losing_team, 2))  # 120 after revival.
    battle._check_victory_after_hit()
    assert battle.last_battle_exp == 270
    assert survivor["exp_current"] == 120
    assert battle.escaped_units[0]["exp_current"] == 30
    assert_no_xp_state_leak(battle)


@pytest.mark.parametrize("winning_team", ["blue", "red"])
@pytest.mark.parametrize("absent_ledger", [False, True])
def test_constructor_only_direct_callers_keep_legacy_xp(winning_team, absent_ledger):
    losing_team = "red" if winning_team == "blue" else "blue"
    battle = BattleEnv(log_enabled=False)
    winner = make_unit(winning_team)
    battle.combined = [winner, make_unit(losing_team, dead=True)]
    if absent_ledger:
        if hasattr(battle, "_battle_exp_tracking_initialized"):
            del battle._battle_exp_tracking_initialized
        del battle._battle_exp_event_count
        del battle._battle_defeated_exp
    battle._apply_battle_exp(losing_team)
    assert battle.last_battle_exp == 60
    assert winner["exp_current"] == 60


@pytest.mark.parametrize("winning_team", ["blue", "red"])
@pytest.mark.parametrize("corpse_state", ["fresh_kill", "old_corpse", "revived_old"])
def test_summon_overwrite_awards_only_fresh_deaths(
    winning_team, corpse_state,
):
    losing_team = "red" if winning_team == "blue" else "blue"
    battle = start([make_unit(winning_team),
                    make_unit(losing_team, dead=corpse_state != "fresh_kill"),
                    make_unit(losing_team, 1, name="Герцог")])
    corpse = get(battle, losing_team)
    summoner = get(battle, losing_team, 1)
    if corpse_state == "revived_old":
        assert battle._apply_hero_item_revive(corpse, 10) == 10
    if corpse_state != "old_corpse":
        kill(battle, corpse)
    assert battle._apply_hero_item_summon(
        source_unit=summoner, target_team=losing_team,
        target_pos=corpse["position"], target_unit=corpse,
        effect={"summon_unit_name": "Рыцарь"},
    ) == 1
    assert corpse["exp_kill"] == 0
    kill(battle, corpse)
    battle._mark_unit_escaped(summoner)

    expected = 0 if corpse_state == "old_corpse" else 60
    battle._check_victory_after_hit()
    assert battle.last_battle_exp == expected
    assert get(battle, winning_team)["exp_current"] == expected
    assert_no_xp_state_leak(battle)


@pytest.mark.parametrize("copy_state", [deepcopy, lambda state: pickle.loads(pickle.dumps(state))],
                         ids=["deepcopy", "pickle"])
def test_initialized_empty_battle_keeps_authoritative_zero_after_roundtrip(copy_state):
    original = start([make_unit("red"), make_unit("blue", dead=True),
                      make_unit("blue", 1, name="Герцог")])
    battle = copy_state(original)
    battle._mark_unit_escaped(get(battle, "blue", 1))
    battle._check_victory_after_hit()
    assert battle._battle_exp_event_count == 0
    assert battle.last_battle_exp == 0
    assert get(battle, "red")["exp_current"] == 0


def test_native_campaign_repeated_retreat_cannot_farm_knight_corpse_xp():
    env = CampaignEnv(map_name="trade_train", observation_version="local5",
                      scripted_capital_bot_enabled=False,
                      use_boss_starting_roster=False, log_enabled=False)
    try:
        env.reset(seed=42)
        def unit(name, team, position):
            return env._build_unit_from_data(env._find_unit_data_by_name(name), team, position)
        corpse = unit("Рыцарь", "blue", 8)
        corpse.update(health=0, hp=0, initiative=0)
        hero = unit("Герцог", "blue", 7)
        enemy = unit("Колдун", "red", 1)
        env.blue_team_state = env._build_battle_team_with_placeholders("blue", [hero, corpse])
        enemy_id = next(iter(env._enemy_configs))
        env.enemy_team_states[enemy_id] = env._build_battle_team_with_placeholders("red", [enemy])
        origin = next((x, y) for x in range(env.grid_size) for y in range(env.grid_size)
                      if (x, y) not in env.castle_heal_tiles)
        env.grid_env.agent_pos = origin
        for _ in range(3):
            env.current_enemy_id = enemy_id
            env.current_battle_context = {"kind": "hero"}
            env.battle_origin_pos = origin
            env.mode = env.MODE_BATTLE
            env._init_battle(enemy_id)
            battle = env.battle_env
            battle.rng.seed(42)
            before = get(battle, "red")["exp_current"]
            for _ in range(8):
                _, _, _, _, info = env.step(RUN_AWAY_ACTION_INDEX)
                if env.mode == env.MODE_GRID:
                    break
            assert env.mode == env.MODE_GRID, info
            assert info.get("battle_result") == "defeat"
            assert [u["name"] for u in battle.escaped_units] == ["Герцог"]
            assert battle.escaped_units[0]["health"] > 0
            assert battle._battle_exp_event_count == 0
            assert battle.last_battle_exp == 0
            persisted_enemy = next(u for u in env.enemy_team_states[enemy_id] if u["position"] == 1)
            assert persisted_enemy["exp_current"] == before == enemy["exp_current"]
            assert persisted_enemy["Level"] == enemy["Level"]
            assert next(u for u in env.blue_team_state if u["position"] == 8)["health"] == 0
    finally:
        env.close()
