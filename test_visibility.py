#!/usr/bin/env python3
import sys
sys.path.append('.')
from lesssimpleenv import BattleEnv
import numpy as np

# Создаем среду
env = BattleEnv(log_enabled=False, reward_illegal=-0.1)
obs, info = env.reset()

print('=== ТЕСТИРОВАНИЕ ВИДИМОСТИ ЮНИТОВ ===')

# Получаем observation
observation = env._obs()
print(f'Размер observation: {len(observation)} (ожидается {12 * 26})')

# Разбираем observation по позициям
FEATURES_PER_POS = 26  # теперь 26 из-за добавленного флага видимости
for i, pos in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    start_idx = i * FEATURES_PER_POS
    end_idx = start_idx + FEATURES_PER_POS
    pos_features = observation[start_idx:end_idx]

    u = env._unit_by_position(pos)
    visible = env._is_unit_visible(u)

    print(f'Позиция {pos}:')
    print(f'  Юнит: {u.get("Name", "None") if u else "None"}')
    print(f'  Команда: {u.get("team", "None") if u else "None"}')
    print(f'  Позиция: {u.get("stand", "None") if u else "None"}')
    print(f'  Видим: {visible}')
    print(f'  Флаг видимости в obs: {pos_features[25]}')  # индекс 25 - это флаг видимости
    print()

print('✅ Тестирование видимости завершено!')

