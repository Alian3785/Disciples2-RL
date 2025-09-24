#!/usr/bin/env python3
"""
Консольная визуализация для GridWorldCombatEnv и SB3 PPO моделей.
Использует только стандартную библиотеку Python.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Константы для визуализации
GRID_SIZE = 10
CELL_WIDTH = 3  # Ширина клетки в символах

# Символы для отображения
SYMBOLS = {
    'empty': ' . ',
    'agent': ' A ',
    'agent_wounded': ' W ',
    'enemy1': ' 1 ',
    'enemy2': ' 2 ',
    'enemy3': ' 3 ',
    'heal': ' H ',
    'wall': ' ■ ',
    'border': '┼'
}

# Цвета для терминала (если поддерживается)
COLORS = {
    'reset': '\033[0m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'purple': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'bold': '\033[1m',
    'dim': '\033[2m'
}


class ConsoleVisualizer:
    """Консольная визуализация GridWorld среды"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.vec_env = None
        self.game_speed = 1.0
        self.paused = False
        self.running = True
        self.current_step = 0
        self.total_reward = 0.0
        self.success = False
        self.agent_level = 1
        self.wounded = False
        self.next_target = 0

        # Координаты игровых объектов
        self.agent_pos = (1, 1)
        self.agent_pos_history = deque(maxlen=10)
        self.enemy_positions = [(0, 0), (0, 0), (0, 0)]
        self.enemy_levels = [1, 2, 3]
        self.enemies_alive = [True, True, True]
        self.heal_cell = (0, 0)

        # Статистика
        self.stats = {
            'steps': 0,
            'total_reward': 0.0,
            'moves': 0,
            'combats': 0,
            'heals': 0,
            'deaths': 0
        }

        # Загружаем модель и среду
        self._load_model()

    def _load_model(self):
        """Загрузить модель из чекпоинта"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Чекпоинт не найден: {self.checkpoint_path}")

        # Создаем среду
        from walkrandomwalk import GridWorldCombatEnv
        self.env = GridWorldCombatEnv()

        # Загружаем нормализацию
        log_dir = "./logs/ppo_grid_wandb_diverse_spawn_goalx2"
        vec_normalize_path = os.path.join(log_dir, "vecnormalize.pkl")

        if os.path.exists(vec_normalize_path):
            vec_env = make_vec_env(lambda: self.env, n_envs=1)
            self.vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = True
            print(f"VecNormalize загружен из: {vec_normalize_path}")
        else:
            vec_env = make_vec_env(lambda: self.env, n_envs=1)
            self.vec_env = vec_normalize_path
            print("VecNormalize не найден, используется базовая среда")

        self.model = PPO.load(self.checkpoint_path, device="auto")
        print(f"Модель загружена из: {self.checkpoint_path}")

    def _clear_screen(self):
        """Очистить экран консоли"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_colored(self, text: str, color: str = 'reset'):
        """Вывести текст с цветом"""
        if self._supports_color():
            print(f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}", end='')
        else:
            print(text, end='')

    def _supports_color(self) -> bool:
        """Проверить поддержку цвета в терминале"""
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            return False

    def _get_cell_symbol(self, x: int, y: int) -> str:
        """Получить символ для клетки"""
        # Проверяем, есть ли объект в этой клетке
        if (x, y) == self.agent_pos:
            return SYMBOLS['agent_wounded'] if self.wounded else SYMBOLS['agent']
        elif (x, y) == self.heal_cell:
            return SYMBOLS['heal']
        else:
            # Проверяем врагов
            for i, pos in enumerate(self.enemy_positions):
                if (x, y) == pos and self.enemies_alive[i]:
                    return SYMBOLS[f'enemy{i+1}']

        return SYMBOLS['empty']

    def _draw_grid(self):
        """Отрисовать игровое поле"""
        print("\n" + "="*60)
        print("🎯 GRIDWORLD COMBAT ENVIRONMENT - PPO Model Visualizer")
        print("="*60)

        # Заголовок сетки
        header = "   " + "".join(f"{i:3}" for i in range(GRID_SIZE))
        print(header)
        print("  ┌" + "───┬" * (GRID_SIZE-1) + "───┐")

        # Рисуем сетку
        for y in range(GRID_SIZE):
            row = f"{y:2}│"
            for x in range(GRID_SIZE):
                symbol = self._get_cell_symbol(x, y)

                # Цветовая кодировка
                if (x, y) == self.agent_pos:
                    if self.wounded:
                        self._print_colored(symbol, 'red')
                    else:
                        self._print_colored(symbol, 'green')
                elif (x, y) == self.heal_cell:
                    self._print_colored(symbol, 'yellow')
                elif any((x, y) == pos for pos in self.enemy_positions):
                    for i, pos in enumerate(self.enemy_positions):
                        if (x, y) == pos and self.enemies_alive[i]:
                            color = ['red', 'yellow', 'purple'][i]
                            self._print_colored(symbol, color)
                            break
                else:
                    print(symbol, end='')

                if x < GRID_SIZE - 1:
                    print("│", end='')
                else:
                    print("│", end='')

            print()  # Новая строка

            # Разделители между строками
            if y < GRID_SIZE - 1:
                print("  ├" + "───┼" * (GRID_SIZE-1) + "───┤")

        # Нижняя граница
        print("  └" + "───┴" * (GRID_SIZE-1) + "───┘")

    def _draw_ui(self):
        """Отрисовать пользовательский интерфейс"""
        print("\n" + "="*60)
        print("📊 СТАТУС ИГРЫ:")
        print("-"*60)

        # Информация об агенте
        agent_status = [
            f"🎮 Агент: Уровень {self.agent_level}, Позиция {self.agent_pos}",
            f"💔 Wounded: {'Да' if self.wounded else 'Нет'}",
            f"🎯 Следующая цель: {self.next_target + 1 if self.next_target < 3 else 'Нет'}",
            f"✅ Успех: {'Да' if self.success else 'Нет'}"
        ]

        for status in agent_status:
            print(f"  {status}")

        # Информация о врагах
        print("\n👹 Враги:")
        for i in range(3):
            alive = self.enemies_alive[i]
            pos = self.enemy_positions[i]
            level = self.enemy_levels[i]
            status = "жив" if alive else "мертв"
            print(f"  Враг {i+1}: Позиция {pos}, Уровень {level}, Статус: {status}")

        # Статистика
        print("\n📈 СТАТИСТИКА:")
        print(f"  Шаги: {self.current_step}")
        print(f"  Награда: {self.total_reward:.3f}")
        print(f"  Движений: {self.stats['moves']}")
        print(f"  Боев: {self.stats['combats']}")
        print(f"  Лечений: {self.stats['heals']}")
        print(f"  Смертей: {self.stats['deaths']}")

        # Управление
        print("\n🎮 УПРАВЛЕНИЕ:")
        print("  SPACE - Пауза/Продолжить")
        print("  R - Сброс симуляции")
        print("  + / - - Изменить скорость")
        print("  Q / ESC - Выход")

        # Скорость
        print(f"\n⚡ Скорость: {self.game_speed:.1f}x")

        # Отладочная информация
        print("\n🔍 ОТЛАДКА:")
        if len(self.agent_pos_history) >= 5:
            recent_moves = list(self.agent_pos_history)[-5:]
            print(f"  Последние 5 позиций: {recent_moves}")

    def _reset_simulation(self):
        """Сбросить симуляцию"""
        obs = self.vec_env.reset()
        self.current_step = 0
        self.total_reward = 0.0
        self.success = False
        self.agent_level = 1
        self.wounded = False
        self.next_target = 0

        # Обновляем состояние из базовой среды
        base_env = self.vec_env.venv.envs[0].env
        self.agent_pos = tuple(base_env.pos)
        self.agent_pos_history.clear()
        self.enemy_positions = [tuple(pos) for pos in base_env.enemy_positions]
        self.enemies_alive = base_env.enemies_alive.copy()

        # Сбрасываем статистику
        self.stats = {
            'steps': 0,
            'total_reward': 0.0,
            'moves': 0,
            'combats': 0,
            'heals': 0,
            'deaths': 0
        }

        print("🔄 Симуляция сброшена")

    def _step_simulation(self):
        """Выполнить один шаг симуляции"""
        if self.success:
            return

        # Получить текущее наблюдение из среды
        if self.current_step == 0:
            obs = self.vec_env.reset()
            obs = obs[0]  # Берем первое наблюдение из батча
        else:
            # Получить наблюдение из текущего состояния среды
            base_env = self.vec_env.venv.envs[0].env
            obs = [
                float(base_env.pos[0]),
                float(base_env.pos[1]),
                float(base_env.agent_level),
                1.0 if base_env.wounded else 0.0,
                float(base_env.enemy_positions[0][0]),
                float(base_env.enemy_positions[0][1]),
                float(base_env.enemy_levels[0]),
                1.0 if base_env.enemies_alive[0] else 0.0,
                float(base_env.enemy_positions[1][0]),
                float(base_env.enemy_positions[1][1]),
                float(base_env.enemy_levels[1]),
                1.0 if base_env.enemies_alive[1] else 0.0,
                float(base_env.enemy_positions[2][0]),
                float(base_env.enemy_positions[2][1]),
                float(base_env.enemy_levels[2]),
                1.0 if base_env.enemies_alive[2] else 0.0,
            ]

        # Получить действие от модели
        import numpy as np
        action, _ = self.model.predict(np.array(obs).reshape(1, -1), deterministic=True)
        action = int(action[0])

        # Выполнить действие в среде
        obs, rewards, dones, infos = self.vec_env.step([action])
        obs = obs[0]
        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0]

        self.current_step += 1
        self.total_reward += reward

        # Обновить состояние из базовой среды
        base_env = self.vec_env.venv.envs[0].env
        old_pos = self.agent_pos
        self.agent_pos = tuple(base_env.pos)
        self.agent_level = base_env.agent_level
        self.wounded = base_env.wounded
        self.success = base_env.next_required_idx == 3
        self.next_target = base_env.next_required_idx
        self.enemies_alive = base_env.enemies_alive.copy()
        self.enemy_positions = [tuple(pos) for pos in base_env.enemy_positions]

        # Отслеживание позиции агента
        self.agent_pos_history.append(self.agent_pos)

        # Обновление статистики
        if old_pos != self.agent_pos:
            self.stats['moves'] += 1
        if reward > 0:
            if 'heal' in str(action).lower():
                self.stats['heals'] += 1
            else:
                self.stats['combats'] += 1
        if reward < 0 and reward != -0.005:  # Не считаем обычный штраф за шаг
            self.stats['deaths'] += 1

        # Отладочная информация
        if self.current_step % 10 == 0:  # Каждые 10 шагов
            print(f"\n📍 Шаг {self.current_step}: Агент на {self.agent_pos}, "
                  f"Уровень {self.agent_level}, Награда: {reward:.3f}")

        # Проверка на зацикливание
        if len(self.agent_pos_history) >= 8:
            recent_positions = list(self.agent_pos_history)[-8:]
            unique_positions = set(recent_positions)
            if len(unique_positions) == 1:  # Агент не двигается 8 шагов подряд
                print(f"\n⚠️  Агент зациклился на позиции {self.agent_pos}")
                print("Попытка сброса окружения...")
                obs = self.vec_env.reset()
                return

        # Проверка на завершение эпизода
        if done:
            print(f"\n🎉 Эпизод завершен на шаге {self.current_step}!")
            print(f"Финальная позиция: {self.agent_pos}, Уровень: {self.agent_level}")
            print(f"Успех: {self.success}, Wounded: {self.wounded}")
            return

        # Проверка на превышение лимита шагов
        if self.current_step > 300:
            print(f"\n⏰ Превышен лимит шагов ({self.current_step}), симуляция может зациклиться")
            return

        # Пауза для лучшей видимости
        time.sleep(0.5 / self.game_speed)

    def run(self):
        """Основной игровой цикл"""
        print("🎮 Запуск консольной визуализации GridWorld...")
        print("📋 Загрузка завершена. Нажмите любую клавишу для начала...")

        try:
            input()  # Ждем нажатия клавиши
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Остановка визуализации...")
            return

        while self.running:
            self._clear_screen()

            # Отрисовка
            self._draw_grid()
            self._draw_ui()

            # Обработка ввода
            try:
                import select
                import sys

                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == ' ':
                        self.paused = not self.paused
                        print(f"\n{'⏸️  Пауза' if self.paused else '▶️  Продолжение'}")
                    elif key == 'r':
                        self._reset_simulation()
                    elif key in ['+', '=']:
                        self.game_speed = min(3.0, self.game_speed + 0.1)
                        print(f"\n⚡ Скорость увеличена: {self.game_speed:.1f}x")
                    elif key == '-':
                        self.game_speed = max(0.1, self.game_speed - 0.1)
                        print(f"\n🐌 Скорость уменьшена: {self.game_speed:.1f}x")
                    elif key in ['q', '\x1b']:  # ESC
                        self.running = False
                        print("\n❌ Выход из визуализации...")
                        break
            except (ImportError, OSError):
                # Fallback для систем без select
                print("\n⚠️  Интерактивное управление недоступно в этой среде")
                print("Запуск автоматической симуляции...")
                time.sleep(2)

            if not self.paused and not self.success:
                self._step_simulation()

        print("🎮 Визуализация завершена")


def main():
    parser = argparse.ArgumentParser(description="Консольная визуализация для GridWorld Combat")
    parser.add_argument("--checkpoint", default="./checkpoints2/ppo_model_800k_steps.zip",
                       help="Путь к чекпоинту модели")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Начальная скорость симуляции")

    args = parser.parse_args()

    try:
        visualizer = ConsoleVisualizer(args.checkpoint)
        visualizer.game_speed = args.speed
        visualizer.run()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
