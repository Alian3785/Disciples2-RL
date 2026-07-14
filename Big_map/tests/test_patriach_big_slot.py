from battle_env import BattleEnv, TARGET_POSITIONS


def _unit(
    name: str,
    team: str,
    position: int,
    unit_type: str,
    *,
    health: int = 100,
    big: bool = False,
) -> dict:
    return {
        "name": name,
        "team": team,
        "position": position,
        "stand": "ahead" if position in (1, 2, 3, 7, 8, 9) else "behind",
        "unit_type": unit_type,
        "health": health,
        "max_health": 100,
        "damage": 10,
        "damage_secondary": 0,
        "initiative": 0 if health <= 0 else 10,
        "initiative_base": 10,
        "accuracy": 100,
        "accuracy_secondary": 0,
        "attack_type_primary": "Weapon",
        "attack_type_secondary": "",
        "immunity": [],
        "resistance": [],
        "armor": 0,
        "big": big,
    }


def test_patriach_cannot_revive_unit_behind_living_big_ally():
    env = BattleEnv(log_enabled=False)
    env._init_with_custom_teams(
        [_unit("Enemy", "red", 1, "Warrior")],
        [
            _unit("Чёрный дракон", "blue", 8, "Mage", big=True),
            _unit("Патриарх", "blue", 9, "Patriach"),
            _unit("Теург", "blue", 11, "Teurg", health=0),
        ],
    )
    dragon = env._unit_by_position(8)
    patriarch = env._unit_by_position(9)
    teurg = env._unit_by_position(11)
    dragon["health"] = dragon["max_health"]
    patriarch["health"] = patriarch["max_health"]
    teurg["initiative"] = 0
    env.current_blue_attacker_pos = patriarch["position"]
    env.blue_attacks_left = 1

    assert env._is_position_behind_big(teurg["position"]) is True
    assert teurg not in env._patriach_targetable_allies(patriarch)
    assert env._patriach_auto_target(patriarch) is None
    assert env._apply_patriach_support(patriarch, teurg) == "invalid"

    target_action = TARGET_POSITIONS.index(5)  # red pos5 mirrors blue pos11
    assert bool(env.compute_action_mask()[target_action]) is False

    dragon["health"] = 0

    assert env._is_position_behind_big(teurg["position"]) is False
    assert teurg in env._patriach_targetable_allies(patriarch)
    assert env._patriach_auto_target(patriarch) == teurg["position"]
    assert bool(env.compute_action_mask()[target_action]) is True
    assert env._apply_patriach_support(patriarch, teurg) == "revive_success"
    assert teurg["health"] > 0


def test_patriach_can_revive_each_ally_only_once_per_battle():
    env = BattleEnv(log_enabled=False)
    env._init_with_custom_teams(
        [_unit("Enemy", "red", 1, "Warrior")],
        [
            _unit("Патриарх", "blue", 9, "Patriach"),
            _unit("Теург", "blue", 10, "Teurg", health=0),
        ],
    )
    patriarch = env._unit_by_position(9)
    teurg = env._unit_by_position(10)
    patriarch["health"] = patriarch["max_health"]
    teurg["initiative"] = 0
    target_action = TARGET_POSITIONS.index(4)  # red pos4 mirrors blue pos10
    env.current_blue_attacker_pos = patriarch["position"]
    env.blue_attacks_left = 1

    assert bool(env.compute_action_mask()[target_action]) is True
    assert env._apply_patriach_support(patriarch, teurg) == "revive_success"

    teurg["health"] = 0
    teurg["initiative"] = 0

    assert teurg not in env._patriach_targetable_allies(patriarch)
    assert env._patriach_auto_target(patriarch) is None
    assert bool(env.compute_action_mask()[target_action]) is False
    assert env._apply_patriach_support(patriarch, teurg) == "invalid"


def test_patriach_revive_limit_does_not_block_healing_or_other_allies():
    env = BattleEnv(log_enabled=False)
    env._init_with_custom_teams(
        [_unit("Enemy", "red", 1, "Warrior")],
        [
            _unit("Патриарх", "blue", 9, "Patriach"),
            _unit("Теург", "blue", 10, "Teurg", health=0),
            _unit("Жрец", "blue", 11, "Cliric", health=0),
        ],
    )
    patriarch = env._unit_by_position(9)
    teurg = env._unit_by_position(10)
    priest = env._unit_by_position(11)
    teurg["initiative"] = 0
    priest["initiative"] = 0

    assert env._apply_patriach_support(patriarch, teurg) == "revive_success"

    # Лимит касается только повторного воскрешения: живого юнита лечить можно.
    teurg["health"] = 60
    assert env._apply_patriach_support(patriarch, teurg) == "heal_success"

    teurg["health"] = 0
    teurg["initiative"] = 0
    assert env._apply_patriach_support(patriarch, teurg) == "invalid"
    assert env._apply_patriach_support(patriarch, priest) == "revive_success"


def test_patriach_revive_limit_resets_for_a_new_battle():
    env = BattleEnv(log_enabled=False)
    red = [_unit("Enemy", "red", 1, "Warrior")]
    blue = [
        _unit("Патриарх", "blue", 9, "Patriach"),
        _unit("Теург", "blue", 10, "Teurg", health=0),
    ]
    env._init_with_custom_teams(red, blue)
    patriarch = env._unit_by_position(9)
    teurg = env._unit_by_position(10)
    teurg["initiative"] = 0

    assert env._apply_patriach_support(patriarch, teurg) == "revive_success"
    teurg["health"] = 0
    teurg["initiative"] = 0
    assert env._apply_patriach_support(patriarch, teurg) == "invalid"

    env._init_with_custom_teams(red, blue)
    patriarch = env._unit_by_position(9)
    teurg = env._unit_by_position(10)
    teurg["initiative"] = 0
    assert env._apply_patriach_support(patriarch, teurg) == "revive_success"


def test_patriach_revive_limit_is_shared_but_other_revive_methods_still_work():
    env = BattleEnv(log_enabled=False)
    env._init_with_custom_teams(
        [_unit("Enemy", "red", 1, "Warrior")],
        [
            _unit("Первый Патриарх", "blue", 8, "Patriach"),
            _unit("Второй Патриарх", "blue", 9, "Patriach"),
            _unit("Теург", "blue", 10, "Teurg", health=0),
        ],
    )
    first_patriarch = env._unit_by_position(8)
    second_patriarch = env._unit_by_position(9)
    teurg = env._unit_by_position(10)
    teurg["initiative"] = 0

    assert env._apply_patriach_support(first_patriarch, teurg) == "revive_success"

    teurg["health"] = 0
    teurg["initiative"] = 0
    assert env._apply_patriach_support(second_patriarch, teurg) == "invalid"

    # Предметы и эликсиры не используют и не сбрасывают лимит Патриарха.
    assert env._apply_hero_item_revive(teurg, 50) == 50
    assert teurg["health"] == 50

    teurg["health"] = 0
    teurg["initiative"] = 0
    assert env._apply_patriach_support(second_patriarch, teurg) == "invalid"
