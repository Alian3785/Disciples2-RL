from battle_env import (
    BattleEnv,
    POST_VICTORY_HEALER_NAMES,
    RUN_AWAY_ACTION_INDEX,
    TARGET_POSITIONS,
    WAIT_ACTION_INDEX,
)
from data_dicts_compact_lines import DATA as UNIT_DATA, placeholder_unit


EXPECTED_POST_VICTORY_HEALERS = frozenset(
    {
        "Служка",
        "Жрец",
        "Священник",
        "Архангел",
        "Медиум",
        "Оракул",
        "Дева рощи",
        "Патриарх",
        "Клирик",
        "Аббатиса",
        "Прорицательница",
        "Нейтральный эльфийский оракул",
        "Солнечная танцовщица",
        "Сильфида",
    }
)


def _unit(
    name: str,
    team: str,
    position: int,
    unit_type: str,
    *,
    health: int = 100,
    max_health: int = 100,
    damage: int = 20,
) -> dict:
    return {
        "name": name,
        "team": team,
        "position": position,
        "stand": "ahead" if position in (1, 2, 3, 7, 8, 9) else "behind",
        "unit_type": unit_type,
        "health": health,
        "max_health": max_health,
        "damage": damage,
        "damage_secondary": 0,
        "initiative": 0 if health <= 0 else 10,
        "initiative_base": 10,
        "accuracy": 100,
        "accuracy_secondary": 0,
        "attack_type_primary": "Life" if unit_type in {
            "Cliric",
            "Profit",
            "Patriach",
        } else "Weapon",
        "attack_type_secondary": "",
        "immunity": [],
        "resistance": [],
        "armor": 0,
        "big": False,
        "exp_kill": 0,
        "exp_required": "max",
        "exp_current": 0,
    }


def _start_decided_battle(red: list[dict], blue: list[dict]) -> BattleEnv:
    red_positions = {int(unit["position"]) for unit in red}
    blue_positions = {int(unit["position"]) for unit in blue}
    red = red + [
        placeholder_unit("red", position)
        for position in range(1, 7)
        if position not in red_positions
    ]
    blue = blue + [
        placeholder_unit("blue", position)
        for position in range(7, 13)
        if position not in blue_positions
    ]
    env = BattleEnv(log_enabled=False)
    env._init_with_custom_teams(red, blue)
    for unit in env.combined:
        if float(unit.get("health", 0) or 0) <= 0:
            unit["initiative"] = 0
    env._check_victory_after_hit()
    return env


def test_post_victory_healer_allowlist_matches_requested_units_only():
    assert POST_VICTORY_HEALER_NAMES == EXPECTED_POST_VICTORY_HEALERS
    data_names = {str(unit.get("кто", "") or "") for unit in UNIT_DATA}
    assert EXPECTED_POST_VICTORY_HEALERS <= data_names
    assert "Вампир" not in POST_VICTORY_HEALER_NAMES
    assert "Высший вампир" not in POST_VICTORY_HEALER_NAMES


def test_blue_point_healer_gets_one_turn_before_victory_is_final():
    env = _start_decided_battle(
        [_unit("Enemy", "red", 1, "Warrior", health=0)],
        [
            _unit("Служка", "blue", 7, "Cliric", damage=20),
            _unit("Скваер", "blue", 8, "Warrior", health=30),
        ],
    )
    ally = env._unit_by_position(8)

    assert env.winner is None
    assert env._post_victory_team == "blue"
    assert env.current_blue_attacker_pos == 7

    heal_action = TARGET_POSITIONS.index(2)  # red pos2 mirrors blue pos8
    mask = env.compute_action_mask()
    assert bool(mask[heal_action]) is True
    assert bool(mask[WAIT_ACTION_INDEX]) is True
    assert bool(mask[RUN_AWAY_ACTION_INDEX]) is False

    _, reward, terminated, truncated, info = env.step(heal_action)

    assert ally["health"] == 50
    assert env.winner == "blue"
    assert reward == env.reward_win
    assert terminated is True
    assert truncated is False
    assert info["post_victory_healer_name"] == "Служка"
    assert info["post_victory_healed_hp"] == 20.0


def test_killing_blow_returns_nonterminal_step_until_healer_acts():
    env = BattleEnv(log_enabled=False)
    red = [_unit("Enemy", "red", 1, "Warrior", health=10)] + [
        placeholder_unit("red", position) for position in range(2, 7)
    ]
    blue = [
        _unit("Скваер", "blue", 7, "Warrior", damage=100),
        _unit("Жрец", "blue", 8, "Cliric", damage=40),
        _unit("Ally", "blue", 9, "Warrior", health=20),
    ] + [placeholder_unit("blue", position) for position in range(10, 13)]
    env._init_with_custom_teams(red, blue)
    env.current_blue_attacker_pos = 7
    env.blue_attacks_left = 1
    env._unit_by_position(7)["initiative"] = 10

    attack_action = TARGET_POSITIONS.index(1)
    _, _, terminated, truncated, info = env.step(attack_action)

    assert terminated is False
    assert truncated is False
    assert env.winner is None
    assert env.current_blue_attacker_pos == 8
    assert info["post_victory_heal_started"] is True

    heal_action = TARGET_POSITIONS.index(3)  # mirrors blue pos9
    _, _, terminated, _, _ = env.step(heal_action)
    assert terminated is True
    assert env.winner == "blue"
    assert env._unit_by_position(9)["health"] == 60


def test_each_blue_healer_gets_exactly_one_sequential_turn():
    env = _start_decided_battle(
        [_unit("Enemy", "red", 1, "Warrior", health=0)],
        [
            _unit("Клирик", "blue", 7, "Profit", damage=20),
            _unit("Жрец", "blue", 8, "Cliric", damage=40),
            _unit("Скваер", "blue", 9, "Warrior", health=10),
        ],
    )
    wounded = env._unit_by_position(9)

    mass_heal_action = TARGET_POSITIONS.index(1)
    _, _, terminated, _, first_info = env.step(mass_heal_action)
    assert terminated is False
    assert first_info["post_victory_healer_name"] == "Клирик"
    assert wounded["health"] == 30
    assert env.current_blue_attacker_pos == 8

    point_heal_action = TARGET_POSITIONS.index(3)  # mirrors blue pos9
    _, _, terminated, _, second_info = env.step(point_heal_action)
    assert terminated is True
    assert second_info["post_victory_healer_name"] == "Жрец"
    assert wounded["health"] == 70
    assert env.winner == "blue"


def test_patriach_can_use_post_victory_turn_to_revive():
    env = _start_decided_battle(
        [_unit("Enemy", "red", 1, "Warrior", health=0)],
        [
            _unit("Патриарх", "blue", 7, "Patriach", damage=120),
            _unit("Скваер", "blue", 8, "Warrior", health=0, max_health=100),
        ],
    )
    fallen = env._unit_by_position(8)

    revive_action = TARGET_POSITIONS.index(2)  # mirrors blue pos8
    assert bool(env.compute_action_mask()[revive_action]) is True

    _, _, terminated, _, info = env.step(revive_action)

    assert terminated is True
    assert fallen["health"] == 50
    assert info["post_victory_healed_hp"] == 50.0


def test_red_post_victory_healing_is_resolved_automatically():
    env = _start_decided_battle(
        [
            _unit("Служка", "red", 1, "Cliric", damage=20),
            _unit("Squire", "red", 2, "Warrior", health=30),
        ],
        [_unit("Enemy", "blue", 7, "Warrior", health=0)],
    )
    wounded = env._unit_by_position(2)

    assert env.winner == "red"
    assert env._post_victory_team is None
    assert wounded["health"] == 50


def test_vampires_and_copied_healer_types_do_not_delay_victory():
    env = _start_decided_battle(
        [_unit("Enemy", "red", 1, "Warrior", health=0)],
        [
            _unit("Вампир", "blue", 7, "Vampire"),
            _unit("Двойник", "blue", 8, "Cliric"),
        ],
    )

    assert env.winner == "blue"
    assert env._post_victory_team is None
    assert env.current_blue_attacker_pos is None
