import random
from typing import Any, Dict, List, Sequence, Tuple


def handle_occultmaster_action(
    battle: Any,
    attacker: Dict[str, Any],
    red_front_positions: Sequence[int],
    red_back_positions: Sequence[int],
    blue_front_positions: Sequence[int],
    blue_back_positions: Sequence[int],
) -> Tuple[bool, str]:
    occult_team = attacker.get("team")
    the_position = attacker.get("position")
    occult_positions: List[int] = []
    spawned_any = False

    for unit in battle.combined:
        if unit.get("team") != occult_team:
            continue

        name_empty = not (unit.get("name") or "").strip()

        health_value = unit.get("health")
        try:
            health_zero = int(health_value) <= 0
        except (TypeError, ValueError):
            health_zero = False

        if not (name_empty or health_zero):
            continue

        pos_value = unit.get("position")
        try:
            pos_int = int(pos_value)
        except (TypeError, ValueError):
            continue

        occult_positions.append(pos_int)

    occult_positions.sort()
    battle._log(" ".join(str(pos) for pos in occult_positions))

    if occult_team == "red":
        row_pairs = list(zip(red_front_positions, red_back_positions))
    else:
        row_pairs = list(zip(blue_front_positions, blue_back_positions))

    def _row_slot_is_empty_or_dead(position: int) -> bool:
        unit = battle._unit_by_position(position)
        if unit is None:
            return True
        return not battle._alive(unit)

    def _spawn_occultmaster_unit(template: Dict[str, Any], position: int, stand: str) -> None:
        nonlocal spawned_any
        summoned = {
            **template,
            "team": occult_team,
            "position": position,
            "stand": stand,
            "paralyzed": 0,
            "long_paralyzed": 0,
            "running_away": 0,
            "transformed": 0,
            "basestats": [],
        }

        summoned.setdefault("defense", 0)
        summoned.setdefault("poison_damage_per_tick", 0)
        summoned.setdefault("burn_turns_left", 0)
        summoned.setdefault("burn_damage_per_tick", 0)
        summoned.setdefault("burn_source_lord_pos", None)
        summoned.setdefault("uran_turns_left", 0)
        summoned.setdefault("uran_damage_per_tick", 0)
        summoned.setdefault("resilience_used_types", [])
        summoned.setdefault("bonusturn", 0)
        summoned.setdefault("original_damage", summoned.get("damage", 0))

        slot = battle._unit_by_position(position)
        if slot is None:
            battle.combined.append(summoned)
        else:
            slot.clear()
            slot.update(summoned)

        battle._log(f"OCCULTMASTER: призван {summoned['name']} на pos{position} ({summoned['stand']}).")
        spawned_any = True

    dragon_template = {
        "name": "Дракон смерти",
        "initiative": 0,
        "initiative_base": 35,
        "unit_type": "Mage",
        "damage": 55,
        "damage_secondary": 0,
        "health": 375,
        "max_health": 375,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": ["death"],
        "resistance": [],
        "attack_type_primary": "death",
        "attack_type_secondary": "",
        "big": True,
        "Summoned": the_position,
    }

    dracolich_template = {
        "name": "Драколич",
        "initiative": 0,
        "initiative_base": 35,
        "unit_type": "Mage",
        "damage": 75,
        "damage_secondary": 0,
        "health": 525,
        "max_health": 525,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": ["death"],
        "resistance": [],
        "attack_type_primary": "death",
        "attack_type_secondary": "",
        "big": True,
        "Summoned": the_position,
    }

    vampire_template = {
        "name": "Вампир",
        "initiative": 0,
        "initiative_base": 40,
        "unit_type": "Vampire",
        "damage": 50,
        "damage_secondary": 0,
        "health": 185,
        "max_health": 185,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": ["death"],
        "resistance": [],
        "attack_type_primary": "death",
        "attack_type_secondary": "",
        "big": False,
        "Summoned": the_position,
    }

    spirit_template = {
        "name": "Дух",
        "initiative": 0,
        "initiative_base": 60,
        "unit_type": "Archer",
        "damage": 60,
        "damage_secondary": 0,
        "health": 150,
        "max_health": 150,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": ["Weapon", "death"],
        "resistance": [],
        "attack_type_primary": "death",
        "attack_type_secondary": "",
        "big": False,
        "Summoned": the_position,
    }

    lich_template = {
        "name": "Лич",
        "initiative": 0,
        "initiative_base": 40,
        "unit_type": "Mage",
        "damage": 70,
        "damage_secondary": 0,
        "health": 140,
        "max_health": 140,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 0,
        "immunity": ["death"],
        "resistance": [],
        "attack_type_primary": "death",
        "attack_type_secondary": "",
        "big": False,
        "Summoned": the_position,
    }

    shadow_template = {
        "name": "Тень",
        "initiative": 0,
        "initiative_base": 20,
        "unit_type": "Shadow",
        "damage": 0,
        "damage_secondary": 0,
        "health": 135,
        "max_health": 135,
        "armor": 0,
        "accuracy": 50,
        "accuracy_secondary": 0,
        "immunity": ["death"],
        "resistance": [],
        "attack_type_primary": "Mind",
        "attack_type_secondary": "",
        "big": False,
        "Summoned": the_position,
    }

    wight_template = {
        "name": "Сущий",
        "initiative": 0,
        "initiative_base": 50,
        "unit_type": "Wight",
        "damage": 75,
        "damage_secondary": 0,
        "health": 105,
        "max_health": 105,
        "armor": 0,
        "accuracy": 80,
        "accuracy_secondary": 80,
        "immunity": ["death", "Weapon"],
        "resistance": [],
        "attack_type_primary": "death",
        "attack_type_secondary": "death",
        "big": False,
        "Summoned": the_position,
    }

    filler_variants = [
        vampire_template,
        spirit_template,
        lich_template,
        shadow_template,
        wight_template,
    ]

    for ahead_pos, behind_pos in row_pairs:
        pair_label = f"pos{ahead_pos}/pos{behind_pos}"
        ahead_empty = _row_slot_is_empty_or_dead(ahead_pos)
        behind_empty = _row_slot_is_empty_or_dead(behind_pos)

        battle._log(
            f"OCCULTMASTER ROW CHECK {pair_label}: "
            f"ahead_empty={ahead_empty}, behind_empty={behind_empty}, "
            f"ahead_unit={battle._unit_by_position(ahead_pos)}, "
            f"behind_unit={battle._unit_by_position(behind_pos)}"
        )

        if ahead_empty and behind_empty:
            battle._log(f"Пустой ряд {pair_label}")
            dragon_choice = dragon_template if random.random() < 0.5 else dracolich_template
            _spawn_occultmaster_unit(dragon_choice, ahead_pos, "ahead")
        else:
            battle._log(f"Не пустой ряд {pair_label}")
            row_contains_big = False
            for pos in (ahead_pos, behind_pos):
                unit = battle._unit_by_position(pos)
                if unit is not None and bool(unit.get("big", False)) and battle._alive(unit):
                    row_contains_big = True
                    break

            if row_contains_big:
                continue

            empty_slots: List[Tuple[int, str]] = []
            if ahead_empty:
                empty_slots.append((ahead_pos, "ahead"))
            if behind_empty:
                empty_slots.append((behind_pos, "behind"))

            for spawn_pos, spawn_stand in empty_slots:
                spawn_template = random.choice(filler_variants)
                _spawn_occultmaster_unit(spawn_template, spawn_pos, spawn_stand)

    return spawned_any, ("ok" if spawned_any else "no_slots")
