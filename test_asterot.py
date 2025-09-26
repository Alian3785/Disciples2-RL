#!/usr/bin/env python3
import sys
sys.path.append('.')
from lesssimpleenv import BattleEnv

# Создаем среду
env = BattleEnv(log_enabled=False, reward_illegal=-0.1)
obs, info = env.reset()

print('✅ Среда инициализируется корректно!')
print(f'Размер observation: {len(obs)}')
print('🔍 Проверяем Asterot в команде...')

found_asterot = False
for u in env.combined:
    if u.get('Type') == 'Asterot':
        print(f'  Астерот: {u["Name"]}#{u["position"]} (команда {u["team"]}, Twohits: {u.get("Twohits", 0)})')
        found_asterot = True

if not found_asterot:
    print('  Астерот не найден')
else:
    print(f'  Найдено Asterot юнитов: {found_asterot}')

print('\n✅ Тест завершен!')
