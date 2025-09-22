import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple


# ====== СРЕДА 10x10: Бои по порядку, WOUNDED и лечение ДЕЙСТВИЕМ в (0,0) ======
class GridWorldCombatEnv(gym.Env):
    """
    - Поле 10x10. Агент стартует в (1,1), уровень = 1.
    - Враги (строгий порядок): (0,5) lvl1 -> (4,7) lvl2 -> (9,9) lvl3.
    - Действия:
        0..7 — движение (8 направлений),
        8 — heal: работает ТОЛЬКО в (0,0), снимает wounded, позиция не меняется.
    - Бой при входе на клетку врага:
        * Если wounded=True -> немедленный проигрыш (fail_penalty).
        * Если враг не тот по порядку -> проигрыш (fail_penalty).
        * Иначе, если agent_level >= enemy_level -> победа:
            enemy погибает, agent_level += 1, агент становится wounded=True,
            выдаётся win_reward; на 3-й победе ДОПОЛНИТЕЛЬНО goal_reward и эпизод завершается.
        * Иначе -> проигрыш (fail_penalty) и wounded=True.
    - Лечение: НЕТ автолечения. Только действие heal в (0,0) снимает ранения.
    - Награды: step_cost за шаг, win_reward за каждую победу,
      goal_reward плюс на 3-й победе, fail_penalty за поражение.
    - Таймаут: max_steps (по умолчанию 1000).
    - Наблюдение (float32, 16 признаков):
      (ax, ay, agent_level, wounded,
       e1x, e1y, e1_level, e1_alive,
       e2x, e2y, e2_level, e2_alive,
       e3x, e3y, e3_level, e3_alive)
    """
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 step_cost: float = -0.01,
                 goal_reward: float = 1.0,
                 win_reward: float = 0.5,
                 fail_penalty: float = -0.5,
                 max_steps: int = 1000):
        super().__init__()
        self.size = 10
        self.start = np.array([1, 1], dtype=np.int32)
        self.heal_cell = np.array([0, 0], dtype=np.int32)

        # Враги по порядку
        self.enemy_positions = [
            np.array([0, 5], dtype=np.int32),  # idx0 lvl1
            np.array([4, 7], dtype=np.int32),  # idx1 lvl2
            np.array([9, 9], dtype=np.int32),  # idx2 lvl3
        ]
        self.enemy_levels = [1, 2, 3]

        # Действия: движение (8) + heal (1) = 9
        self.action_space = spaces.Discrete(9)
        # 0:Up, 1:Down, 2:Right, 3:Left, 4:Up-Right, 5:Up-Left, 6:Down-Right, 7:Down-Left, 8:Heal
        self._action_to_delta = {
            0: np.array([ 0, -1], dtype=np.int32),
            1: np.array([ 0,  1], dtype=np.int32),
            2: np.array([ 1,  0], dtype=np.int32),
            3: np.array([-1,  0], dtype=np.int32),
            4: np.array([ 1, -1], dtype=np.int32),
            5: np.array([-1, -1], dtype=np.int32),
            6: np.array([ 1,  1], dtype=np.int32),
            7: np.array([-1,  1], dtype=np.int32),
            # 8: heal (без перемещения)
        }

        # Наблюдение
        low  = np.array([0, 0, 0, 0,   0, 0, 1, 0,   0, 0, 1, 0,   0, 0, 1, 0], dtype=np.float32)
        high = np.array([9, 9, 6, 1,   9, 9, 3, 1,   9, 9, 3, 1,   9, 9, 3, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32, shape=(16,))

        # Параметры наград/эпизода
        self.render_mode = render_mode
        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.win_reward = float(win_reward)
        self.fail_penalty = float(fail_penalty)
        self.max_steps = int(max_steps)

        # Состояние
        self.pos = self.start.copy()
        self.steps = 0
        self.agent_level = 1
        self.wounded = False
        self.enemies_alive = [True, True, True]
        self.next_required_idx = 0  # кого нужно бить следующим (0..2)

    @property
    def action_meanings(self):
        return ["Up", "Down", "Right", "Left", "Up-Right", "Up-Left", "Down-Right", "Down-Left", "Heal"]

    # ---------- Вспомогательные ----------
    def _get_obs(self) -> np.ndarray:
        e = []
        for i in range(3):
            x, y = self.enemy_positions[i]
            e.extend([float(x), float(y), float(self.enemy_levels[i]), 1.0 if self.enemies_alive[i] else 0.0])
        return np.array([
            float(self.pos[0]),
            float(self.pos[1]),
            float(self.agent_level),
            1.0 if self.wounded else 0.0,
            *e
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "agent_level": int(self.agent_level),
            "wounded": bool(self.wounded),
            "heal_cell": (int(self.heal_cell[0]), int(self.heal_cell[1])),
            "enemies": [
                {"pos": (int(self.enemy_positions[i][0]), int(self.enemy_positions[i][1])),
                 "level": int(self.enemy_levels[i]),
                 "alive": bool(self.enemies_alive[i])}
                for i in range(3)
            ],
            "next_required_idx": int(self.next_required_idx),
            "steps": self.steps,
            "is_success": bool(self.next_required_idx == 3),
        }

    def _enemy_at_pos(self) -> Optional[int]:
        for i in range(3):
            if self.enemies_alive[i] and np.array_equal(self.pos, self.enemy_positions[i]):
                return i
        return None

    # ---------- API Gymnasium ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.pos = self.start.copy()
        self.steps = 0
        self.agent_level = 1
        self.wounded = False
        self.enemies_alive = [True, True, True]
        self.next_required_idx = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), "Недопустимое действие"

        reward = self.step_cost
        terminated = False

        if action == 8:
            # HEAL — доступен только в (0,0); позиция не меняется
            if np.array_equal(self.pos, self.heal_cell):
                self.wounded = False
            # если не в (0,0), действие ничего не делает кроме траты шага
        else:
            # Движение
            delta = self._action_to_delta[int(action)]
            self.pos = self.pos + delta
            np.clip(self.pos, 0, self.size - 1, out=self.pos)

            # Бой, если на клетке враг
            enemy_idx = self._enemy_at_pos()
            if enemy_idx is not None:
                if self.wounded:
                    reward = self.fail_penalty
                    terminated = True
                else:
                    if enemy_idx != self.next_required_idx:
                        reward = self.fail_penalty
                        terminated = True
                    else:
                        enemy_lvl = self.enemy_levels[enemy_idx]
                        if self.agent_level >= enemy_lvl:
                            # Победа
                            self.enemies_alive[enemy_idx] = False
                            self.agent_level += 1
                            self.next_required_idx += 1
                            self.wounded = True  # после боя ранимся

                            # Награда за победу; на последней добавляем финальную
                            reward = self.step_cost + self.win_reward
                            if self.next_required_idx == 3:
                                reward += self.goal_reward
                                terminated = True
                        else:
                            reward = self.fail_penalty
                            terminated = True
                            self.wounded = True

        self.steps += 1
        truncated = bool(self.steps >= self.max_steps)
        info = self._get_info()
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for i, pos in enumerate(self.enemy_positions):
            x, y = map(int, pos)
            grid[y][x] = str(i + 1) if self.enemies_alive[i] else "·"
        # Клетка лечения
        hx, hy = map(int, self.heal_cell)
        grid[hy][hx] = "H"
        ax, ay = map(int, self.pos)
        grid[ay][ax] = "✔" if self.next_required_idx == 3 else "A"
        status = (f"[Agent Lvl: {self.agent_level}] Wounded: {self.wounded} | "
                  f"Heal@{tuple(self.heal_cell)} | Next target: "
                  f"{self.next_required_idx + 1 if self.next_required_idx < 3 else '-'} | "
                  f"Steps: {self.steps}")
        out = "\n".join(" ".join(row) for row in grid)
        if self.render_mode == "human":
            print(status)
            print(out)
        return status + "\n" + out

    def close(self):
        pass


# ====== ОБУЧЕНИЕ PPO c VecNormalize (без VecMonitor) ======
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy

    log_dir = "./logs/ppo_grid_combat_heal_action_at_origin"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(render: bool = False):
        return GridWorldCombatEnv(render_mode="human" if render else None, max_steps=1000)

    # === TRAIN: DummyVecEnv (внутри уже Monitor) -> VecNormalize ===
    n_envs = 16
    train_env = make_vec_env(make_env, n_envs=n_envs, monitor_dir=log_dir)  # Monitor внутри
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # === EVAL для колбэка: тот же стек, training=False ===
    eval_env = make_vec_env(make_env, n_envs=1)  # тоже с Monitor (без сохранения)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # Пресет PPO для исследования
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.02,
        ent_coef=0.02,
        vf_coef=0.5,
        verbose=1,
        # tensorboard_log=log_dir,
    )

    # Обучение 1e6 шагов (без прогресс-бара)
    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

    # Сохранение модели и статистик нормализации
    model.save(os.path.join(log_dir, "ppo_grid_final"))
    train_env.save(os.path.join(log_dir, "vecnormalize.pkl"))

    # ====== РУЧНАЯ ОЦЕНКА ======
    eval_env_raw = make_vec_env(make_env, n_envs=1)
    eval_env_loaded = VecNormalize.load(os.path.join(log_dir, "vecnormalize.pkl"), eval_env_raw)
    eval_env_loaded.training = False
    eval_env_loaded.norm_reward = True  # False, если нужны "сырые" награды

    mean_reward, std_reward = evaluate_policy(model, eval_env_loaded, n_eval_episodes=50, deterministic=True)
    print(f"[Eval] mean_reward={mean_reward:.3f} ± {std_reward:.3f}")

    # ====== ДЕМОНСТРАЦИЯ (визуализация + лог действий/наблюдений) ======

    # Помощник для разворачивания всех обёрток до исходного env
    def unwrap_env(e):
        env = e
        # разматываем .env, пока можно
        while hasattr(env, "env"):
            env = env.env
        return env

    # 1) Env с render_mode='human'
    demo_env_raw = make_vec_env(lambda: GridWorldCombatEnv(render_mode="human", max_steps=1000), n_envs=1)

    # 2) Грузим те же статы нормализации на демо-окружение
    from stable_baselines3.common.vec_env import VecNormalize as VN
    demo_env = VN.load(os.path.join(log_dir, "vecnormalize.pkl"), demo_env_raw)
    demo_env.training = False
    demo_env.norm_reward = True  # можно False, если нужны «сырые» награды

    # 3) Базовый env (для печати render и сырого obs)
    wrapped = demo_env.venv.envs[0]     # это Monitor
    base_env = unwrap_env(wrapped)      # это уже GridWorldCombatEnv

    # Список действий
    meanings = base_env.action_meanings
    actions_help = ", ".join([f"{i}:{m}" for i, m in enumerate(meanings)])

    obs = demo_env.reset()
    done = False
    truncated = False
    total_r = 0.0
    steps = 0

    print("\nДемонстрация после обучения:")
    print("Доступные действия:", actions_help)
    # стартовая визуализация и obs
    base_env.render()
    print(f"Obs(model, normalized) = {obs[0].tolist()}")
    print(f"Obs(raw)              = {base_env._get_obs().tolist()}")

    while not (done or truncated):
        # Выбор действия
        action, _ = model.predict(obs, deterministic=True)
        a = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
        print(f"\n[Step {steps+1}] Chosen action: {a} ({meanings[a]})")

        # Применяем действие
        obs, rewards, dones, infos = demo_env.step(action)
        done = bool(dones[0])
        truncated = bool(infos[0].get("TimeLimit.truncated", False))
        total_r += float(rewards[0])

        # Визуализация и текущее наблюдение
        base_env.render()
        print(f"Obs(model, normalized) = {obs[0].tolist()}")
        print(f"Obs(raw)              = {base_env._get_obs().tolist()}")
        print(f"Reward: {float(rewards[0]):.3f} | Done: {done} | Truncated: {truncated}")

        steps += 1

    last_info = infos[0] if infos else {}
    print(f"\nУспех: {last_info.get('is_success', False)} | "
          f"Уровень агента: {last_info.get('agent_level')} | "
          f"Wounded: {last_info.get('wounded')} | "
          f"Следующий враг: {last_info.get('next_required_idx')} | "
          f"Шагов: {steps} | Суммарная награда: {total_r:.3f}")