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
