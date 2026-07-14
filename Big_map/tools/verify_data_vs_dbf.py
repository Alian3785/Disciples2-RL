"""Массовая сверка юнитов DATA с оригинальными таблицами Disciples 2 (Globals/*.dbf).

Сравнивает по имени юнита: HP, броню, уровень, опыт за убийство, опыт до уровня,
размер, урон/точность/инициативу обеих атак, тип атаки, двойной удар, иммунитеты
и защиты. Полный отчёт пишется в outputs/dbf_verify_report.md, в stdout — сводка.

Запуск:
    python tools/verify_data_vs_dbf.py
    python tools/verify_data_vs_dbf.py --globals "D:\\Disciples II Восстание Эльфов\\Globals"
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from console_encoding import setup_utf8_console
from data_dicts_compact_lines import DATA
from inspect_sg_map import parse_dbf_rows

setup_utf8_console()

DEFAULT_GLOBALS = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Disciples II Rise of the Elves\Globals"
)
NO_ID = "g000000000"

# LattS: источники атак → имена типов, как в DATA
SOURCE_NAMES = {
    "0": "weapon",
    "1": "mind",
    "2": "life",
    "3": "death",
    "4": "fire",
    "5": "water",
    "6": "earth",
    "7": "air",
}


def norm_name(value: str) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


def to_int(value, default: int = 0) -> int:
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return default


def load_dbf_units(globals_dir: Path) -> Dict[str, List[dict]]:
    gunits = parse_dbf_rows(globals_dir / "Gunits.dbf")
    texts = parse_dbf_rows(globals_dir / "Tglobal.dbf")
    attacks = {r["ATT_ID"].lower(): r for r in parse_dbf_rows(globals_dir / "Gattacks.dbf")}
    immu = parse_dbf_rows(globals_dir / "Gimmu.dbf")
    immu_class = parse_dbf_rows(globals_dir / "GimmuC.dbf")
    latt_c = {r["ID"]: r["TEXT"] for r in parse_dbf_rows(globals_dir / "LattC.dbf")}

    text_by_id = {r["TXT_ID"].lower(): r["TEXT"] for r in texts if r.get("TXT_ID")}

    # Иммунитеты к источникам атак: cat 3 = всегда (иммунитет), cat 2 = однократно (защита)
    immune_always: Dict[str, set] = defaultdict(set)
    immune_once: Dict[str, set] = defaultdict(set)
    for r in immu:
        uid = r.get("UNIT_ID", "").lower()
        src = SOURCE_NAMES.get(r.get("IMMUNITY", ""), f"src{r.get('IMMUNITY')}")
        cat = r.get("IMMUNECAT", "")
        (immune_always if cat == "3" else immune_once)[uid].add(src)
    # Иммунитеты к классам атак (яд, паралич и т.п.)
    for r in immu_class:
        uid = r.get("UNIT_ID", "").lower()
        cls_name = latt_c.get(r.get("IMMUNITY", ""), f"class{r.get('IMMUNITY')}")
        cls_name = cls_name.replace("L_", "").lower()
        cat = r.get("IMMUNECAT", "")
        (immune_always if cat == "3" else immune_once)[uid].add(cls_name)

    def attack_info(att_id: str) -> Optional[dict]:
        row = attacks.get(str(att_id or "").lower())
        if row is None or str(att_id or "").lower() == NO_ID:
            return None
        dam = to_int(row.get("QTY_DAM"), 0)
        heal = to_int(row.get("QTY_HEAL"), 0)
        return {
            "initiative": to_int(row.get("INITIATIVE")),
            "source": SOURCE_NAMES.get(row.get("SOURCE", ""), f"src{row.get('SOURCE')}"),
            "power": to_int(row.get("POWER")),
            "damage": dam if dam > 0 else heal,  # у лекарей сила атаки лежит в QTY_HEAL
            "class": latt_c.get(row.get("CLASS", ""), row.get("CLASS", "")),
        }

    by_name: Dict[str, List[dict]] = defaultdict(list)
    for row in gunits:
        uid = row.get("UNIT_ID", "").lower()
        name = text_by_id.get(row.get("NAME_TXT", "").lower(), "")
        if not name:
            continue
        by_name[norm_name(name)].append(
            {
                "unit_id": row.get("UNIT_ID", ""),
                "name": name,
                "level": to_int(row.get("LEVEL")),
                "hp": to_int(row.get("HIT_POINT")),
                "armor": to_int(row.get("ARMOR")),
                "xp_killed": to_int(row.get("XP_KILLED")),
                "xp_next": to_int(row.get("XP_NEXT")),
                "big": row.get("SIZE_SMALL", "T").upper() != "T",
                "atck_twice": row.get("ATCK_TWICE", "").upper() == "T",
                "attack1": attack_info(row.get("ATTACK_ID")),
                "attack2": attack_info(row.get("ATTACK2_ID")),
                "immune_always": immune_always.get(uid, set()),
                "immune_once": immune_once.get(uid, set()),
            }
        )
    return by_name


DOUBLE_STRIKE_DATA_TYPES = {"Demon", "Elfarcher"}


def compare_unit(du: dict, dbf: dict) -> List[str]:
    diffs: List[str] = []

    def num(field: str, ours, theirs):
        if to_int(ours, -10**9) != to_int(theirs, -10**9):
            diffs.append(f"{field}: у нас {ours}, в DBF {theirs}")

    num("здоровье", du.get("здоровье"), dbf["hp"])
    num("броня", round(float(du.get("броня", 0) or 0) * 100), dbf["armor"])
    num("уровень", du.get("уровень"), dbf["level"])
    num("опыт убийства", du.get("опыт убийства"), dbf["xp_killed"])
    if str(du.get("нужный опыт")) != "max":
        num("нужный опыт", du.get("нужный опыт"), dbf["xp_next"])
    if bool(to_int(du.get("размер", 0))) != dbf["big"]:
        diffs.append(f"размер: у нас {du.get('размер')}, в DBF big={dbf['big']}")

    a1 = dbf["attack1"]
    if a1 is None:
        diffs.append("в DBF нет первичной атаки")
    else:
        num("инит", du.get("инит"), a1["initiative"])
        num("точн", round(float(du.get("точн", 0) or 0) * 100), a1["power"])
        num("урон", du.get("урон"), a1["damage"])
        ours_type = norm_name(du.get("тип атаки1"))
        if ours_type != a1["source"]:
            diffs.append(f"тип атаки1: у нас {ours_type}, в DBF {a1['source']}")

    a2 = dbf["attack2"]
    ours_t2 = norm_name(du.get("тип атаки2"))
    if a2 is None:
        if ours_t2:
            diffs.append(f"атака2: у нас '{ours_t2}', в DBF отсутствует")
    else:
        if not ours_t2:
            diffs.append(f"атака2: в DBF есть ({a2['source']}, урон {a2['damage']}), у нас нет")
        else:
            num("точн2", round(float(du.get("точн2", 0) or 0) * 100), a2["power"])
            num("урон2", du.get("урон2"), a2["damage"])
            if ours_t2 != a2["source"]:
                diffs.append(f"тип атаки2: у нас {ours_t2}, в DBF {a2['source']}")

    ours_twice = str(du.get("тип", "")) in DOUBLE_STRIKE_DATA_TYPES
    if ours_twice != dbf["atck_twice"]:
        diffs.append(
            f"двойной удар: у нас {'да' if ours_twice else 'нет'} (тип {du.get('тип')}), "
            f"в DBF {'да' if dbf['atck_twice'] else 'нет'}"
        )

    ours_imm = {norm_name(x) for x in (du.get("иммунитет") or [])}
    ours_ward = {norm_name(x) for x in (du.get("защита") or [])}
    if ours_imm != dbf["immune_always"]:
        diffs.append(
            f"иммунитет: у нас {sorted(ours_imm) or '—'}, в DBF {sorted(dbf['immune_always']) or '—'}"
        )
    if ours_ward != dbf["immune_once"]:
        diffs.append(
            f"защита: у нас {sorted(ours_ward) or '—'}, в DBF {sorted(dbf['immune_once']) or '—'}"
        )
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globals", default=str(DEFAULT_GLOBALS), help="Папка Globals оригинала.")
    args = parser.parse_args()

    by_name = load_dbf_units(Path(args.globals))
    by_id = {
        str(candidate.get("unit_id", "") or "").lower(): candidate
        for candidates in by_name.values()
        for candidate in candidates
        if candidate.get("unit_id")
    }

    matched_ok: List[str] = []
    with_diffs: List[Tuple[str, str, List[str]]] = []
    not_found: List[str] = []

    for du in DATA:
        if not isinstance(du, dict):
            continue
        name = str(du.get("кто", "")).strip()
        unit_id = str(du.get("unit_id", "") or "").strip().lower()
        id_candidate = by_id.get(unit_id)
        candidates = [id_candidate] if id_candidate is not None else by_name.get(norm_name(name), [])
        if not candidates:
            not_found.append(name)
            continue
        best = min(candidates, key=lambda c: len(compare_unit(du, c)))
        diffs = compare_unit(du, best)
        if diffs:
            with_diffs.append((name, best["unit_id"], diffs))
        else:
            matched_ok.append(name)

    report: List[str] = ["# Сверка DATA с Gunits.dbf оригинала", ""]
    report.append(
        f"Юнитов в DATA: {len([u for u in DATA if isinstance(u, dict)])} | "
        f"полное совпадение: {len(matched_ok)} | с расхождениями: {len(with_diffs)} | "
        f"нет в оригинале: {len(not_found)}"
    )
    report.append("")
    report.append("## Юниты с расхождениями")
    for name, uid, diffs in with_diffs:
        report.append(f"\n### {name} ({uid}, расхождений: {len(diffs)})")
        for d in diffs:
            report.append(f"- {d}")
    report.append("\n## Нет в оригинале (кастомные или переименованные)")
    for name in not_found:
        report.append(f"- {name}")
    report.append("\n## Полное совпадение")
    for name in matched_ok:
        report.append(f"- {name}")

    out_path = PROJECT_ROOT / "outputs" / "dbf_verify_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report), encoding="utf-8")

    print(report[2])
    print(f"Полный отчёт: {out_path}")
    diff_counts = sorted(with_diffs, key=lambda x: -len(x[2]))
    print("\nТоп-15 по числу расхождений:")
    for name, uid, diffs in diff_counts[:15]:
        print(f"  {len(diffs):2d}  {name}")


if __name__ == "__main__":
    main()
