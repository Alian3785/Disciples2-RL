#!/usr/bin/env python3
import sys
sys.path.append('.')
from lesssimpleenv import BattleEnv

# Создаем среду
env = BattleEnv(log_enabled=True, reward_illegal=-0.1)
obs, info = env.reset()

print('=== ТЕСТИРОВАНИЕ ТИПА LORD ===')

# Находим Lord'а
lord = None
target = None
for unit in env.combined:
    if unit.get('Type') == 'Lord':
        lord = unit
    elif unit.get('Type') == 'Mage' and unit.get('team') == 'blue':
        target = unit
        break

print(f'Lord найден: {lord["Name"]}#{lord["position"]} (DMG: {lord["Damage"]}, DMG2: {lord["Damage2"]})')
print(f'Цель найдена: {target["Name"]}#{target["position"]} (HP: {target["Health"]})')

# Тестируем атаку Lord'а
if lord and target:
    print('\n=== ТЕСТИРОВАНИЕ АТАКИ LORD ===')
    print(f'Lord атакует цель с Accuracy2: {lord["Accuracy2"]}%')
    print(f'AttackType2: {lord["AttackType2"]}')
    print(f'Цель имеет иммунитет: {target.get("Immunity", [])}')
    print(f'Цель имеет сопротивляемость: {target.get("Resilience", [])}')

    # Имитируем успешную атаку - сначала устанавливаем цель живой
    target["Health"] = 100  # Делаем цель живой для теста
    print(f'До атаки HP цели: {target["Health"]}')
    print(f'Burn цели до атаки: {target.get("Burn", 0)}')

    env._attack_single_target(lord, target["position"])

    print(f'После атаки HP цели: {target["Health"]}')
    print(f'Burn цели: {target.get("Burn", 0)}')

print('\n✅ Lord и атака протестированы!')
