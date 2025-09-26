#!/usr/bin/env python3
import sys
sys.path.append('.')
from lesssimpleenv import BattleEnv

# Создаем среду
env = BattleEnv(log_enabled=True, reward_illegal=-0.1)
obs, info = env.reset()

print('=== ТЕСТИРОВАНИЕ ДВОЙНОГО ХОДА ASTEROT ===')

# Находим агента Asterot'а
asterot_agent = None
for u in env.combined:
    if u.get('Type') == 'Asterot' and u.get('team') == 'blue' and u.get('Twohits', 0) == 1:
        asterot_agent = u
        break

if asterot_agent:
    asterot_pos = asterot_agent["position"]
    print(f'✅ Найден агент Asterot: {asterot_agent["Name"]}#{asterot_pos}')
    print(f'   Twohits: {asterot_agent["Twohits"]}')
    print(f'   Команда: {asterot_agent["team"]}')
    print(f'   Позиция: {asterot_agent["stand"]}')

    # Имитируем первый ход Asterot'а
    print('\n🔄 ПЕРВЫЙ ХОД ASTEROT:')
    print('   Выбираем недоступную цель (должен получить штраф, но сохранить Twohits=1)')

    # Ищем недоступную цель для Asterot'а
    allowed_targets = env._warrior_allowed_targets(asterot_agent)
    print(f'   Доступные цели: {allowed_targets}')

    # Выбираем недоступную цель (например, позицию 4, если она недоступна)
    invalid_target = 4 if 4 not in allowed_targets else (5 if 5 not in allowed_targets else 6)
    print(f'   Выбираем недоступную цель: {invalid_target}')

    # Делаем шаг с недоступной целью
    obs, reward, terminated, truncated, info = env.step(invalid_target - 1)  # action = позиция - 1
    print(f'   Награда за ход: {reward}')

    # Получаем актуальное состояние Asterot'а по позиции
    asterot_agent = env._unit_by_position(asterot_pos)
    print(f'   Twohits после первого хода: {asterot_agent["Twohits"] if asterot_agent else "N/A"}')

    if 'asterot_needs_second_turn' in info:
        print('✅ Asterot готов к второму ходу (флаг asterot_needs_second_turn=True)')
    else:
        print('❌ Asterot НЕ готов к второму ходу')

    print('\n🔄 ВТОРОЙ ХОД ASTEROT:')
    print('   Делаем второй ход (даже после неудачного первого)')

    # Проверяем состояние Asterot'а перед вторым ходом
    asterot_before = env._unit_by_position(asterot_pos)
    print(f'   Twohits перед вторым ходом: {asterot_before["Twohits"] if asterot_before else "N/A"}')

    # Делаем второй ход
    obs, reward, terminated, truncated, info = env.step(0)  # выбираем первую позицию
    print(f'   Награда за второй ход: {reward}')

    # Получаем актуальное состояние Asterot'а после второго хода
    asterot_after = env._unit_by_position(asterot_pos)
    print(f'   Twohits после второго хода: {asterot_after["Twohits"] if asterot_after else "N/A"}')

    if asterot_before and asterot_after:
        print(f'   Twohits изменился с {asterot_before["Twohits"]} на {asterot_after["Twohits"]}')
    else:
        print('   ❌ Не удалось получить состояние Asterotа')

    print('\n✅ Тест двойного хода Asterot завершен!')
else:
    print('❌ Агент Asterot не найден')
