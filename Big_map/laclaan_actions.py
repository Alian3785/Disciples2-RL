import random
from typing import Any, Dict, List, Sequence, Tuple


def handle_laclaan_action(
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

        battle._log(f"?? LACLAAN: призван {summoned['name']} на pos{position} ({summoned['stand']}).")
        spawned_any = True

    black_dragon_template = {
        "name": "Чёрный дракон",
        "initiative": 40,
        "initiative_base": 40,
        "team": "blue",
        "position": 7,
        "stand": "ahead",
        "unit_type": "Mage",
        "damage": 125,
        "damage_secondary": 0,
        "health": 800,
        "max_health": 800,
        "armor": 0,
        "accuracy": 75,
        "accuracy_secondary": 0,
        "immunity": ["death"],
        "resistance": [],
        "attack_type_primary": "death",
        "attack_type_secondary": "",
        "big": True,
        "Summoned": the_position,
    }

    high_vampire_template = {
        "name": "Высший вампир",
        "initiative": 0,
        "initiative_base": 40,
        "team": "blue",
        "position": 10,
        "stand": "behind",
        "unit_type": "Highvampire",
        "damage": 60,
        "damage_secondary": 0,
        "health": 210,
        "max_health": 210,
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

    for ahead_pos, behind_pos in row_pairs:
        pair_label = f"pos{ahead_pos}/pos{behind_pos}"
        ahead_empty = _row_slot_is_empty_or_dead(ahead_pos)
        behind_empty = _row_slot_is_empty_or_dead(behind_pos)

        battle._log(
            f"LACLAAN ROW CHECK {pair_label}: "
            f"ahead_empty={ahead_empty}, behind_empty={behind_empty}, "
            f"ahead_unit={battle._unit_by_position(ahead_pos)}, "
            f"behind_unit={battle._unit_by_position(behind_pos)}"
        )

        if ahead_empty and behind_empty:
            battle._log(f"Пустой ряд {pair_label}")
            _spawn_occultmaster_unit(black_dragon_template, ahead_pos, "ahead")
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
                _spawn_occultmaster_unit(high_vampire_template, spawn_pos, spawn_stand)

    return spawned_any, ("ok" if spawned_any else "no_slots")

