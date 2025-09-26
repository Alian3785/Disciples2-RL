import math
from typing import List, Dict

import numpy as np
import pytest

from battle_env import BattleEnv, UNITS_RED, UNITS_BLUE, POISON_TURNS, BURN_TURNS


@pytest.fixture
def env() -> BattleEnv:
    env = BattleEnv(log_enabled=False)
    obs, info = env.reset()
    # deterministic RNG for predictable rolls in tests
    env.rng.seed(42)
    return env


def _unit_by_name(env: BattleEnv, name: str) -> Dict:
    return next(u for u in env.combined if u["Name"] == name)


def _prepare_victim(env: BattleEnv, position: int, extra: Dict = None) -> Dict:
    unit = env._unit_by_position(position)
    unit["Health"] = 100
    unit["poison_turns_left"] = 0
    unit["burn_turns_left"] = 0
    unit["poison_damage_per_tick"] = 0
    unit["burn_damage_per_tick"] = 0
    unit["Immunity"] = list(unit.get("Immunity", []))
    unit["Resilience"] = list(unit.get("Resilience", []))
    unit["Armor"] = unit.get("Armor", 0)
    if extra:
        unit.update(extra)
    return unit


def _step_until(env: BattleEnv, target_name: str) -> Dict:
    """Advance auto turns until unit with given name has the move."""
    while True:
        nxt = env._pop_next()
        if nxt is None:
            env._end_round_restore()
            continue
        if nxt["Name"] == target_name:
            return nxt
        env._apply_start_of_turn_effects(nxt)
        nxt["Initiative"] = 0


def test_poison_tick_damage(env: BattleEnv):
    victim = _prepare_victim(env, 8)
    victim["poison_turns_left"] = 2
    victim["poison_damage_per_tick"] = 15

    env._apply_start_of_turn_effects(victim)
    assert victim["Health"] == 85
    assert victim["poison_turns_left"] == 1

    env._apply_start_of_turn_effects(victim)
    assert victim["Health"] == 70
    assert victim["poison_turns_left"] == 0
    assert victim["poison_damage_per_tick"] == 0


def test_burn_tick_damage(env: BattleEnv):
    victim = _prepare_victim(env, 8)
    victim["burn_turns_left"] = 2
    victim["burn_damage_per_tick"] = 12

    env._apply_start_of_turn_effects(victim)
    assert victim["Health"] == 88
    assert victim["burn_turns_left"] == 1

    env._apply_start_of_turn_effects(victim)
    assert victim["Health"] == 76
    assert victim["burn_turns_left"] == 0
    assert victim["burn_damage_per_tick"] == 0


def test_poison_and_burn_stack(env: BattleEnv):
    victim = _prepare_victim(env, 8)
    victim["poison_turns_left"] = 1
    victim["poison_damage_per_tick"] = 10
    victim["burn_turns_left"] = 1
    victim["burn_damage_per_tick"] = 20

    env._apply_start_of_turn_effects(victim)
    assert victim["Health"] == 70
    assert victim["poison_turns_left"] == 0
    assert victim["burn_turns_left"] == 0


def test_poison_immune(env: BattleEnv):
    victim = _prepare_victim(env, 9, {"Immunity": ["poison"]})
    victim["poison_turns_left"] = 2
    victim["poison_damage_per_tick"] = 25

    env._apply_start_of_turn_effects(victim)
    assert victim["Health"] == 100
    assert victim["poison_turns_left"] == 0
    assert victim["poison_damage_per_tick"] == 0


def test_burn_immune(env: BattleEnv):
    victim = _prepare_victim(env, 9, {"Immunity": ["Fire"]})
    victim["burn_turns_left"] = 2
    victim["burn_damage_per_tick"] = 30

    env._apply_start_of_turn_effects(victim)
    assert victim["Health"] == 100
    assert victim["burn_turns_left"] == 0
    assert victim["burn_damage_per_tick"] == 0


def test_resilience_blocks_first_hit(env: BattleEnv):
    victim = _prepare_victim(env, 8, {"Resilience": ["Weapon"]})
    attacker = _unit_by_name(env, "Герцог")
    attacker["Damage"] = 50
    attacker["AttackType1"] = "Weapon"
    victim["Health"] = 60

    hit, reason = env._attack(attacker, victim["position"])
    assert hit is True
    assert reason == "resilience_block"
    assert victim["Health"] == 60
    assert "Weapon" in victim["resilience_used_types"]

    hit, reason = env._attack(attacker, victim["position"])
    assert victim["Health"] == 10


def test_armor_reduces_damage(env: BattleEnv):
    victim = _prepare_victim(env, 8, {"Armor": 50})
    attacker = _unit_by_name(env, "Герцог")
    attacker["Damage"] = 100

    env._attack(attacker, victim["position"])
    assert victim["Health"] == 50


def test_warrior_cant_hit_backline_with_front_guard(env: BattleEnv):
    # ensure front line enemy alive and blocking
    front_victim = env._unit_by_position(1)
    back_victim = env._unit_by_position(4)
    front_victim["Health"] = 100
    back_victim["Health"] = 100

    attacker = _unit_by_name(env, "Герцог")
    options = env._warrior_allowed_targets(attacker)
    assert 4 not in options

    front_victim["Health"] = 0
    env._unit_by_position(2)["Health"] = 0
    env._unit_by_position(3)["Health"] = 0
    options = env._warrior_allowed_targets(attacker)
    assert 4 in options


def test_demon_double_attack(env: BattleEnv):
    demon = _unit_by_name(env, "Астерот")
    demon["Damage"] = 40
    demon["Initiative"] = 100
    env.current_blue_attacker_pos = demon["position"]
    env.blue_attacks_left = 2

    target = _prepare_victim(env, 1)
    target["Health"] = 300

    env.step(0)
    assert env.blue_attacks_left == 1
    assert target["Health"] == 260
    assert env.current_blue_attacker_pos == demon["position"]

    env.step(0)
    assert target["Health"] == 220
    assert env.current_blue_attacker_pos != demon["position"]


def test_mass_poison_from_dead_dragon(env: BattleEnv):
    dragon = _unit_by_name(env, "Мертвый дракон")
    dragon["Type"] = "Dead dragon"
    dragon["Damage"] = 10
    dragon["Accuracy"] = 100
    dragon["Damage2"] = 5
    dragon["AttackType2"] = "poison"
    dragon["Accuracy2"] = 100

    victims = [u for u in env.combined if u["team"] == "blue"]
    for v in victims:
        v["Health"] = 100
        v["Immunity"] = []
        v["Resilience"] = []

    env._attack(dragon, None)
    for v in victims:
        if env._alive(v):
            assert v["poison_turns_left"] == POISON_TURNS
            assert v["poison_damage_per_tick"] == 5
