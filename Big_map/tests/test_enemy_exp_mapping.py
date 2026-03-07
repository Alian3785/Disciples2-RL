import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_dicts_compact_lines import DATA
from enemy_configs import ENEMY_CONFIGS


def _build_name_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entry in DATA:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("кто", "") or "").strip()
        if name and name not in index:
            index[name] = entry
    return index


def _is_empty_enemy_slot(unit: dict) -> bool:
    name = str(unit.get("name", "") or "").strip().lower()
    return name == "пусто"


def test_enemy_units_keep_exp_kill_from_data_mapping():
    name_index = _build_name_index()

    for team in ENEMY_CONFIGS.values():
        for unit in team:
            if _is_empty_enemy_slot(unit):
                continue

            unit_name = str(unit.get("name", "") or "").strip()
            assert unit_name in name_index, f'Unit "{unit_name}" must exist in DATA'

            src = name_index[unit_name]
            expected_exp_kill = int(round(float(src.get("опыт убийства", 0) or 0)))
            actual_exp_kill = int(round(float(unit.get("exp_kill", 0) or 0)))
            assert actual_exp_kill == expected_exp_kill


def test_enemy_configs_have_positive_total_exp_reward_pool():
    totals = []
    for team in ENEMY_CONFIGS.values():
        total = 0.0
        for unit in team:
            if _is_empty_enemy_slot(unit):
                continue
            total += float(unit.get("exp_kill", 0) or 0)
        totals.append(total)

    # Regression guard: all-zero exp pools means no post-battle progression.
    assert any(total > 0.0 for total in totals)

