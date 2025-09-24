#!/usr/bin/env python3
"""
Проверка, что Death юниты полностью удалены
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lesssimpleenv import UNITS_RED, UNITS_BLUE, TYPE2ID, ATK2ID

def check_no_death():
    print("=== ПРОВЕРКА ОТСУТСТВИЯ DEATH ЮНИТОВ ===")

    all_units = UNITS_RED + UNITS_BLUE
    print(f"Всего юнитов: {len(all_units)}")

    # Проверим, что нет Death юнитов
    death_units = [u for u in all_units if u.get("Type") == "Death"]
    print(f"Найдено Death юнитов: {len(death_units)}")

    if len(death_units) == 0:
        print("✅ Death юниты успешно удалены!")
    else:
        print("❌ Найдены Death юниты:")
        for unit in death_units:
            print(f"  - {unit['team'].upper()} {unit['Name']}#{unit['position']}")

    # Проверим словари
    print("
=== ПРОВЕРКА СЛОВАРЕЙ ===")
    print(f"TYPE2ID содержит 'Death': {'Death' in TYPE2ID}")
    print(f"ATK2ID содержит 'Death': {'Death' in ATK2ID}")
    print(f"ATK2ID содержит '20': {'20' in ATK2ID}")

    # Проверим типы юнитов
    print("
=== ТЕКУЩИЕ ТИПЫ ЮНИТОВ ===")
    types = set(u.get("Type") for u in all_units)
    print(f"Типы юнитов: {sorted(types)}")

    if "Death" not in types:
        print("✅ Тип 'Death' успешно удален!")
    else:
        print("❌ Тип 'Death' все еще присутствует!")

    print("
✅ Проверка завершена!"
if __name__ == "__main__":
    check_no_death()
