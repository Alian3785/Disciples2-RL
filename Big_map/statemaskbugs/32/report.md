# №32. Наблюдение после предельных состояний остаётся конечным, нормализованным и правильного размера.

- Раздел: Сохранение состояния и маски действий
- Результат: **ОШИБКА/ПРОБЕЛ**
- Прогонов: 100
- Успешно: 50
- Неуспешно: 50
- Время: 1.48 с

Оракул: целевой генеративный инвариант `finite_observation`.

## Найденное

- 50/100: ValueError: cannot convert float NaN to integer
- 50/100: shape=(9864,), min=0.0, max=1.0, finite=True.

## Примеры seed

- seed `1050001`: ValueError: cannot convert float NaN to integer
- seed `1050003`: ValueError: cannot convert float NaN to integer
- seed `1050005`: ValueError: cannot convert float NaN to integer
- seed `1050007`: ValueError: cannot convert float NaN to integer
- seed `1050009`: ValueError: cannot convert float NaN to integer
