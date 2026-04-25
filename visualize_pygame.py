"""
Pygame-ce визуализация для GridWorldCombatEnv и SB3 PPO моделей.
Загружает чекпоинт и визуализирует перемещение по gridworld в реальном времени.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple
import pygame
import numpy as np

# Инициализация pygame-ce
pygame.init()

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

# Константы для визуализации
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 900
FPS = 60

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 20, 60)
BLUE = (30, 144, 255)
GREEN = (50, 205, 50)
YELLOW = (255, 215, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
LIGHT_GRAY = (200, 200, 200)

# Цвета для элементов gridworld
AGENT_COLOR = (50, 205, 50)      # Зеленый для агента
ENEMY_COLORS = [(220, 20, 60), (255, 165, 0), (128, 0, 128)]  # Красный, оранжевый, фиолетовый
HEAL_COLOR = (255, 215, 0)       # Желтый для клетки лечения
DEAD_COLOR = (100, 100, 100)     # Серый для мертвых врагов
ACTIVE_COLOR = (255, 215, 0, 200)  # Золотой для активного состояния

# Размеры игрового поля
GRID_SIZE = 10
FIELD_MARGIN = 80
CELL_SIZE = 50
FIELD_WIDTH = GRID_SIZE * CELL_SIZE
FIELD_HEIGHT = GRID_SIZE * CELL_SIZE
FIELD_X = (WINDOW_WIDTH - FIELD_WIDTH) // 2
FIELD_Y = FIELD_MARGIN + 50


class GridWorldVisualizer:
    """Класс для визуализации gridworld среды"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.env = None
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
        self.agent_pos_history = []  # Для отслеживания перемещений
        self.enemy_positions = [(0, 0), (0, 0), (0, 0)]
        self.enemy_levels = [1, 2, 3]
        self.enemies_alive = [True, True, True]
        self.heal_cell = (0, 0)

        # Анимации
        self.damage_animation = None  # (x, y, damage)
        self.heal_animation = False
        self.move_animation = None  # (old_x, old_y, new_x, new_y, progress)

        # Инициализация pygame
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("GridWorld Combat Visualizer - PPO Model")
        self.clock = pygame.time.Clock()

        # Шрифты
        self.font_large = pygame.font.SysFont(None, 36)
        self.font_medium = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 18)

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
            # Создаем VecNormalize для корректной работы с моделью
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.vec_env import VecNormalize

            vec_env = make_vec_env(lambda: self.env, n_envs=1)
            self.vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = True
            print(f"VecNormalize загружен из: {vec_normalize_path}")
        else:
            # Создаем простую обертку
            from stable_baselines3.common.env_util import make_vec_env
            self.vec_env = make_vec_env(lambda: self.env, n_envs=1)
            print("VecNormalize не найден, используется базовая среда")

        self.model = PPO.load(self.checkpoint_path, device="auto")
        print(f"Модель загружена из: {self.checkpoint_path}")

    def _base_env(self):
        """Вернуть исходную среду из VecNormalize или обычного DummyVecEnv."""
        vec_env = self.vec_env.venv if hasattr(self.vec_env, "venv") else self.vec_env
        return vec_env.envs[0].env

    def _get_cell_center(self, x: int, y: int) -> Tuple[int, int]:
        """Получить центр клетки"""
        return (FIELD_X + x * CELL_SIZE + CELL_SIZE // 2,
                FIELD_Y + y * CELL_SIZE + CELL_SIZE // 2)

    def _update_state_from_env(self):
        """Обновить состояние из среды"""
        # Это будет заполняться при симуляции
        pass

    def _draw_grid(self):
        """Отрисовать сетку"""
        # Фон поля
        field_rect = pygame.Rect(FIELD_X, FIELD_Y, FIELD_WIDTH, FIELD_HEIGHT)
        pygame.draw.rect(self.screen, DARK_GRAY, field_rect, 2)

        # Рисуем клетки сетки
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                cell_rect = pygame.Rect(FIELD_X + x * CELL_SIZE,
                                      FIELD_Y + y * CELL_SIZE,
                                      CELL_SIZE, CELL_SIZE)

                # Клетка лечения
                if (x, y) == self.heal_cell:
                    pygame.draw.rect(self.screen, HEAL_COLOR, cell_rect, 3)
                    heal_font = pygame.font.SysFont(None, 24)
                    heal_text = heal_font.render("H", True, BLACK)
                    heal_center = self._get_cell_center(x, y)
                    self.screen.blit(heal_text, (heal_center[0] - heal_text.get_width() // 2,
                                                heal_center[1] - heal_text.get_height() // 2))
                else:
                    # Обычная клетка
                    pygame.draw.rect(self.screen, LIGHT_GRAY, cell_rect, 1)

    def _draw_agent(self):
        """Отрисовать агента"""
        agent_center = self._get_cell_center(*self.agent_pos)

        # Основной круг агента
        radius = CELL_SIZE // 3
        pygame.draw.circle(self.screen, AGENT_COLOR, agent_center, radius)

        # Граница
        pygame.draw.circle(self.screen, BLACK, agent_center, radius, 2)

        # Уровень агента
        level_font = pygame.font.SysFont(None, 14)
        level_text = level_font.render(str(self.agent_level), True, WHITE)
        self.screen.blit(level_text, (agent_center[0] - level_text.get_width() // 2,
                                     agent_center[1] - level_text.get_height() // 2))

        # Wounded статус
        if self.wounded:
            wounded_font = pygame.font.SysFont(None, 10)
            wounded_text = wounded_font.render("W", True, RED)
            self.screen.blit(wounded_text, (agent_center[0] - radius - 5,
                                          agent_center[1] - radius - 5))

    def _draw_enemies(self):
        """Отрисовать врагов"""
        for i, (pos, level, alive) in enumerate(zip(self.enemy_positions, self.enemy_levels, self.enemies_alive)):
            if not alive:
                continue

            enemy_center = self._get_cell_center(*pos)
            color = ENEMY_COLORS[i]

            # Круг врага
            radius = CELL_SIZE // 3
            pygame.draw.circle(self.screen, color, enemy_center, radius)

            # Граница
            pygame.draw.circle(self.screen, BLACK, enemy_center, radius, 2)

            # Уровень врага
            level_font = pygame.font.SysFont(None, 14)
            level_text = level_font.render(str(level), True, WHITE)
            self.screen.blit(level_text, (enemy_center[0] - level_text.get_width() // 2,
                                         enemy_center[1] - level_text.get_height() // 2))

            # Номер врага
            num_font = pygame.font.SysFont(None, 10)
            num_text = num_font.render(str(i + 1), True, WHITE)
            self.screen.blit(num_text, (enemy_center[0] - num_text.get_width() // 2,
                                       enemy_center[1] + radius - 5))

    def _draw_animations(self):
        """Отрисовать анимации"""
        # Анимация урона
        if self.damage_animation:
            x, y, damage = self.damage_animation
            center = self._get_cell_center(x, y)
            damage_font = pygame.font.SysFont(None, 24)
            damage_text = damage_font.render(f"-{damage}", True, RED)
            self.screen.blit(damage_text, (center[0] - damage_text.get_width() // 2,
                                          center[1] - damage_text.get_height() // 2))
            self.damage_animation = None  # Одноразовая анимация

        # Анимация лечения
        if self.heal_animation:
            heal_center = self._get_cell_center(*self.agent_pos)
            heal_font = pygame.font.SysFont(None, 20)
            heal_text = heal_font.render("HEAL!", True, YELLOW)
            self.screen.blit(heal_text, (heal_center[0] - heal_text.get_width() // 2,
                                        heal_center[1] - heal_text.get_height() // 2 - 30))
            self.heal_animation = False

    def _draw_ui(self):
        """Отрисовать пользовательский интерфейс"""
        # Фон UI
        ui_rect = pygame.Rect(0, WINDOW_HEIGHT - 180, WINDOW_WIDTH, 180)
        pygame.draw.rect(self.screen, DARK_GRAY, ui_rect)

        # Разделитель
        pygame.draw.line(self.screen, WHITE, (0, WINDOW_HEIGHT - 180),
                        (WINDOW_WIDTH, WINDOW_HEIGHT - 180), 2)

        # Информация об агенте
        agent_info = [
            f"Уровень агента: {self.agent_level}",
            f"Wounded: {'Да' if self.wounded else 'Нет'}",
            f"Следующая цель: {self.next_target + 1 if self.next_target < 3 else '-'}",
            f"Успех: {'Да' if self.success else 'Нет'}"
        ]

        for i, info in enumerate(agent_info):
            info_text = self.font_small.render(info, True, WHITE)
            self.screen.blit(info_text, (10, WINDOW_HEIGHT - 170 + i * 20))

        # Информация о шаге
        step_text = self.font_medium.render(f"Шаг: {self.current_step}", True, WHITE)
        self.screen.blit(step_text, (300, WINDOW_HEIGHT - 170))

        # Награда
        reward_text = self.font_medium.render(f"Награда: {self.total_reward:.2f}", True, WHITE)
        self.screen.blit(reward_text, (300, WINDOW_HEIGHT - 140))

        # Победитель/Статус
        status_y = WINDOW_HEIGHT - 80
        if self.success:
            status_text = self.font_large.render("МИССИЯ ВЫПОЛНЕНА!", True, GREEN)
        else:
            status_text = self.font_medium.render("В процессе...", True, YELLOW)

        status_x = WINDOW_WIDTH // 2 - status_text.get_width() // 2
        self.screen.blit(status_text, (status_x, status_y))

        # Управление
        controls = [
            "SPACE - Пауза/Продолжить",
            "R - Сброс",
            "+/- - Скорость",
            "ESC - Выход"
        ]

        for i, control in enumerate(controls):
            control_text = self.font_small.render(control, True, LIGHT_GRAY)
            self.screen.blit(control_text, (WINDOW_WIDTH - 200, WINDOW_HEIGHT - 170 + i * 20))

        # Скорость
        speed_text = self.font_medium.render(f"Скорость: {self.game_speed:.1f}x", True, WHITE)
        self.screen.blit(speed_text, (WINDOW_WIDTH - 200, WINDOW_HEIGHT - 40))

        # Отладочная информация на экране
        debug_info = [
            f"Шаг: {self.current_step}",
            f"Позиции врагов:",
            f"  1: {self.enemy_positions[0]} ({'жив' if self.enemies_alive[0] else 'мертв'})",
            f"  2: {self.enemy_positions[1]} ({'жив' if self.enemies_alive[1] else 'мертв'})",
            f"  3: {self.enemy_positions[2]} ({'жив' if self.enemies_alive[2] else 'мертв'})",
            f"История поз. (посл. 5): {self.agent_pos_history[-5:]}"
        ]

        debug_x = FIELD_X + FIELD_WIDTH + 20
        debug_y = FIELD_Y + 200

        debug_title = self.font_small.render("Отладка:", True, YELLOW)
        self.screen.blit(debug_title, (debug_x, debug_y))

        for i, info in enumerate(debug_info):
            info_text = self.font_small.render(info, True, WHITE)
            self.screen.blit(info_text, (debug_x, debug_y + 20 + i * 18))

        # Легенда
        legend_items = [
            ("A - Агент", AGENT_COLOR),
            ("1 - Враг 1", ENEMY_COLORS[0]),
            ("2 - Враг 2", ENEMY_COLORS[1]),
            ("3 - Враг 3", ENEMY_COLORS[2]),
            ("H - Лечение", HEAL_COLOR)
        ]

        legend_x = FIELD_X + FIELD_WIDTH + 20
        legend_y = FIELD_Y + 350

        legend_title = self.font_small.render("Легенда:", True, WHITE)
        self.screen.blit(legend_title, (legend_x, legend_y))

        for i, (text, color) in enumerate(legend_items):
            # Цветной квадрат
            square_size = 15
            square_rect = pygame.Rect(legend_x, legend_y + 25 + i * 25, square_size, square_size)
            pygame.draw.rect(self.screen, color, square_rect)

            # Текст
            item_text = self.font_small.render(text, True, WHITE)
            self.screen.blit(item_text, (legend_x + square_size + 10, legend_y + 25 + i * 25))

    def _handle_events(self):
        """Обработать события pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_simulation()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.game_speed = min(3.0, self.game_speed + 0.1)
                elif event.key == pygame.K_MINUS:
                    self.game_speed = max(0.1, self.game_speed - 0.1)

    def _reset_simulation(self):
        """Сбросить симуляцию"""
        obs = self.vec_env.reset()
        self.current_step = 0
        self.total_reward = 0.0
        self.success = False
        self.agent_level = 1
        self.wounded = False
        self.next_target = 0

        # Обновляем состояние агента из базовой среды
        base_env = self._base_env()
        self.agent_pos = tuple(base_env.pos)
        self.agent_pos_history.clear()  # Очищаем историю позиций
        self.enemy_positions = [tuple(pos) for pos in base_env.enemy_positions]
        self.enemies_alive = base_env.enemies_alive.copy()

        print("Симуляция сброшена")

    def _step_simulation(self):
        """Выполнить один шаг симуляции"""
        if self.success:
            return

        # Получить текущее наблюдение из среды
        if self.current_step == 0:
            # Первый шаг - получить наблюдение после сброса
            obs = self.vec_env.reset()
            obs = obs[0]  # Берем первое наблюдение из батча
        else:
            # Получить наблюдение из текущего состояния среды
            base_env = self._base_env()
            obs = np.array([
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
            ], dtype=np.float32)

        # Получить действие от модели
        action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
        action = int(action[0]) if isinstance(action, np.ndarray) else int(action)

        # Выполнить действие в среде
        obs, rewards, dones, infos = self.vec_env.step([action])
        obs = obs[0]  # Берем первое наблюдение из батча
        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0]

        self.current_step += 1
        self.total_reward += reward

        # Обновить состояние из базовой среды
        base_env = self._base_env()
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
        if len(self.agent_pos_history) > 10:
            self.agent_pos_history.pop(0)

        # Отладочная информация
        if self.current_step % 5 == 0:  # Каждые 5 шагов
            print(f"Шаг {self.current_step}: Позиция агента {self.agent_pos}, Уровень {self.agent_level}, "
                  f"Wounded: {self.wounded}, Цель: {self.next_target + 1}, Награда: {reward:.3f}")

        # Проверка на зацикливание - если агент не двигается последние 8 шагов
        if len(self.agent_pos_history) >= 8:
            recent_positions = self.agent_pos_history[-8:]
            unique_positions = set(recent_positions)
            if len(unique_positions) == 1:  # Агент не двигается 8 шагов подряд
                print(f"Агент зациклился на позиции {self.agent_pos} в течение 8 шагов")
                print(f"Позиции за последние 8 шагов: {recent_positions}")
                print("Попытка сброса окружения...")
                obs = self.vec_env.reset()
                return

        # Проверка на завершение эпизода
        if done:
            print(f"Эпизод завершен на шаге {self.current_step}")
            print(f"Финальная позиция: {self.agent_pos}, Уровень: {self.agent_level}")
            print(f"Успех: {self.success}, Wounded: {self.wounded}")
            return

        # Проверка на превышение лимита шагов
        if self.current_step > 200:
            print("Превышен лимит шагов (200), симуляция может зациклиться")
            return

        # Небольшая пауза для лучшей видимости
        time.sleep(0.3 / self.game_speed)

    def run(self):
        """Основной игровой цикл"""
        print("Запуск визуализации... (SPACE - пауза, R - сброс, +/- - скорость)")

        while self.running:
            self._handle_events()

            if not self.paused and not self.success:
                self._step_simulation()

            # Отрисовка
            self.screen.fill(BLACK)

            # Игровое поле
            self._draw_grid()

            # Анимации
            self._draw_animations()

            # Игровые объекты
            self._draw_enemies()
            self._draw_agent()

            # UI
            self._draw_ui()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        print("Визуализация завершена")


def main():
    parser = argparse.ArgumentParser(description="Pygame визуализация для GridWorld Combat")
    parser.add_argument("--checkpoint", default="./checkpoints2/ppo_model_400k_steps.zip",
                       help="Путь к чекпоинту модели")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Начальная скорость симуляции")

    args = parser.parse_args()

    try:
        visualizer = GridWorldVisualizer(args.checkpoint)
        visualizer.game_speed = args.speed
        visualizer.run()
    except Exception as e:
        print(f"Ошибка: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
