import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple

# ====== СРЕДА 10x10: Бои по порядку, WOUNDED и лечение ДЕЙСТВИЕМ в (0,0), враги рандомно (с упорядочением дистанций) ======
class GridWorldCombatEnv(gym.Env):
    """
    - Поле 10x10. Агент стартует в (1,1), уровень = 1.
    - ТРИ врага с уровнями 1, 2, 3 появляются на СЛУЧАЙНЫХ координатах при каждом reset(),
      НО:
        * не на старте (1,1)
        * не в клетке лечения (0,0)
        * и с упорядочением: dist(start,e1) < dist(start,e2) < dist(start,e3) (манхэттен).
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
    - Награды:
        step_cost (мягкий отрицательный), win_reward за победу 1 и 2, goal_reward добавляется на 3-й победе,
        fail_penalty смягчён.
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
                 step_cost: float = -0.005,   # идея (7): мягче плата за шаг
                 goal_reward: float = 2.0,    # увеличенная финальная награда
                 win_reward: float = 0.5,     # промежуточные победы как раньше
                 fail_penalty: float = -0.35, # идея (7): мягче штраф за провал
                 max_steps: int = 1000):
        super().__init__()
        self.size = 10
        self.start = np.array([1, 1], dtype=np.int32)
        self.heal_cell = np.array([0, 0], dtype=np.int32)

        # Уровни врагов фиксированы, координаты — рандомно на reset()
        self.enemy_levels = [1, 2, 3]
        self.enemy_positions = [np.zeros(2, dtype=np.int32) for _ in range(3)]

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

    def _manhattan(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))

    def _sample_unique_enemy_positions(self):
        """
        Идея (4): упорядоченный спавн по расстоянию от старта:
        dist(e1) < dist(e2) < dist(e3).
        """
        size = self.size
        banned = {
            self.start[1] * size + self.start[0],          # (1,1)
            self.heal_cell[1] * size + self.heal_cell[0],  # (0,0)
        }

        # Все допустимые клетки
        coords = [np.array([idx % size, idx // size], dtype=np.int32)
                  for idx in range(size * size) if idx not in banned]

        # Считаем дистанции до старта и сортируем
        dists = [(p, self._manhattan(p, self.start)) for p in coords]
        dists.sort(key=lambda t: t[1])

        # Выбираем e1 среди минимальной дистанции (рандом среди равных)
        min_d = dists[0][1]
        cand_e1 = [p for (p, d) in dists if d == min_d]
        e1 = cand_e1[self.np_random.integers(len(cand_e1))]

        # Выбираем e2: дистанция строго > dist(e1), берём минимальную из таких; рандом среди равных
        d_gt_e1 = [ (p,d) for (p,d) in dists if d > self._manhattan(e1, self.start) ]
        # На всякий случай (должно хватать на 10x10)
        assert len(d_gt_e1) > 0, "Не удалось найти позицию для e2 с бОльшей дистанцией"
        min_d2 = d_gt_e1[0][1]
        cand_e2 = [p for (p,d) in d_gt_e1 if d == min_d2 and not np.array_equal(p, e1)]
        e2 = cand_e2[self.np_random.integers(len(cand_e2))]

        # Выбираем e3: дистанция строго > dist(e2)
        d_gt_e2 = [ (p,d) for (p,d) in dists if d > self._manhattan(e2, self.start) and not np.array_equal(p, e1) ]
        assert len(d_gt_e2) > 0, "Не удалось найти позицию для e3 с бОльшей дистанцией"
        min_d3 = d_gt_e2[0][1]
        cand_e3 = [p for (p,d) in d_gt_e2 if d == min_d3 and not np.array_equal(p, e1) and not np.array_equal(p, e2)]
        e3 = cand_e3[self.np_random.integers(len(cand_e3))]

        self.enemy_positions = [e1, e2, e3]

    # ---------- API Gymnasium ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.pos = self.start.copy()
        self.steps = 0
        self.agent_level = 1
        self.wounded = False
        self.enemies_alive = [True, True, True]
        self.next_required_idx = 0

        # Рандомизируем координаты врагов (3 разные клетки, упорядоченные по дистанции)
        self._sample_unique_enemy_positions()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), "Недопустимое действие"

        reward = self.step_cost
        terminated = False

        if action == 8:
            # HEAL — доступен только в (0,0); позиция не меняется
            if np.array_equal(self.pos, self.heal_cell):
                self.wounded = False
            # иначе — просто потратили шаг
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

                            # Награда за победу; на последней добавляем увеличенную финальную
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
        status = ("[Agent Lvl: {}] Wounded: {} | "
                  "Heal@{} | Next target: "
                  "{} | "
                  "Steps: {}".format(self.agent_level, self.wounded,
                  tuple(self.heal_cell),
                  self.next_required_idx + 1 if self.next_required_idx < 3 else '-',
                  self.steps))
        out = "\n".join(" ".join(row) for row in grid)
        if self.render_mode == "human":
            print(status)
            print(out)
        return status + "\n" + out

    def close(self):
        pass


# ====== ОБУЧЕНИЕ PPO c VecNormalize + W&B (с логами средней награды/шагов и аннилом энтропии) ======
if __name__ == "__main__":
    import torch.nn as nn
    from collections import deque

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
    from stable_baselines3.common.evaluation import evaluate_policy

    # ---------- W&B (опционально) ----------
    wandb = None
    wandb_run = None
    try:
        import wandb as _wandb
        wandb = _wandb
        run_name = "ppo-gridworld-{}".format(int(time.time()))
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gridworld-ppo"),
            name=run_name,
            config={"algo": "PPO", "env": "GridWorldCombatEnv"},
            reinit=True,
            anonymous="allow",
        )
        print("[W&B] логирование включено.")
    except ImportError:
        print("[W&B] библиотека wandb не установлена. Продолжаем без логирования.")
    except Exception as e:
        print("[W&B] ошибка подключения: {}. Продолжаем без логирования.".format(str(e)))
        wandb = None
        wandb_run = None

    # ---------- Кастомный колбэк: лог скользящей средней награды/длины эпизода ----------
    class WBLoggerCallback(BaseCallback):
        def __init__(self, wandb_run, window: int = 100, verbose: int = 0):
            super().__init__(verbose)
            self.wandb_run = wandb_run
            self.rews = deque(maxlen=window)
            self.lens = deque(maxlen=window)

        def _on_step(self) -> bool:
            if self.wandb_run is None:
                return True
            infos = self.locals.get("infos", [])
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    self.rews.append(ep["r"])
                    self.lens.append(ep["l"])
            if self.rews:
                data = {
                    "train/ep_rew_mean_100": float(np.mean(self.rews)),
                    "train/ep_len_mean_100": float(np.mean(self.lens)),
                    "time/total_timesteps": int(self.num_timesteps),
                }
                try:
                    wandb.log(data, step=self.num_timesteps)
                except Exception:
                    pass
            return True

    # ---------- EvalCallback c логом метрик в W&B ----------
    class EvalWBCallback(EvalCallback):
        def __init__(self, *args, wandb_run=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.wandb_run = wandb_run

        def _on_step(self) -> bool:
            result = super()._on_step()
            if self.wandb_run is not None and (self.n_calls % self.eval_freq == 0):
                try:
                    wandb.log(
                        {
                            "eval/mean_reward": float(self.last_mean_reward),
                            "time/total_timesteps": int(self.model.num_timesteps),
                        },
                        step=self.model.num_timesteps,
                    )
                except Exception:
                    pass
            return result

    # ---------- Энтропия по расписанию через колбэк ----------
    class EntropyAnnealCallback(BaseCallback):
        def __init__(self, start: float, end: float, wandb_run=None, verbose: int = 0):
            super().__init__(verbose)
            self.start = float(start)
            self.end = float(end)
            self.wandb_run = wandb_run

        def _on_step(self) -> bool:
            total = max(1, int(getattr(self.model, "_total_timesteps", 1)))
            progress_remaining = max(0.0, 1.0 - (self.num_timesteps / float(total)))
            new_coef = self.end + (self.start - self.end) * progress_remaining
            self.model.ent_coef = float(new_coef)
            if self.wandb_run is not None:
                try:
                    wandb.log(
                        {
                            "train/ent_coef": float(new_coef),
                            "time/total_timesteps": int(self.num_timesteps),
                        },
                        step=self.num_timesteps,
                    )
                except Exception:
                    pass
            return True

    log_dir = "./logs/ppo_grid_wandb_ordspawn_softpen"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(render: bool = False):
        return GridWorldCombatEnv(render_mode="human" if render else None, max_steps=1000)

    # === TRAIN: DummyVecEnv (внутри уже Monitor) -> VecNormalize ===
    n_envs = 16
    train_env = make_vec_env(make_env, n_envs=n_envs, monitor_dir=log_dir)  # Monitor внутри
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # === EVAL для колбэка: тот же стек, training=False ===
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_callback_cls = EvalWBCallback if wandb_run is not None else EvalCallback
    eval_callback = eval_callback_cls(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        **({"wandb_run": wandb_run} if wandb_run is not None else {}),
    )

    # === Линейные шедулеры для lr/clip ===
    def linear_schedule(initial, final):
        initial = float(initial)
        final = float(final)
        def f(progress_remaining: float):
            # progress_remaining идёт от 1.0 -> 0.0 по мере обучения
            return final + (initial - final) * progress_remaining
        return f

    # === Усиленная сеть ===
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        ortho_init=True,
    )

    # === PPO ===
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(3e-4, 1e-4),
        n_steps=2048,
        batch_size=8192,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.3, 0.2),
        clip_range_vf=0.3,
        max_grad_norm=0.5,
        target_kl=0.02,
        ent_coef=0.05,   # стартовое значение, аннилим колбэком
        vf_coef=0.5,
        verbose=1,
        # tensorboard_log=log_dir,
    )

    # Собираем колбэки
    callbacks = [eval_callback]
    if wandb_run is not None:
        callbacks.append(WBLoggerCallback(wandb_run, window=100))
        callbacks.append(EntropyAnnealCallback(start=0.05, end=0.01, wandb_run=wandb_run))
    else:
        callbacks.append(EntropyAnnealCallback(start=0.05, end=0.01))
    callback = CallbackList(callbacks)

    # Обучение 1e6 шагов
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    # Сохранения
    model.save(os.path.join(log_dir, "ppo_grid_final"))
    train_env.save(os.path.join(log_dir, "vecnormalize.pkl"))

    # Завершение W&B
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as e:
            print("[W&B] ошибка при завершении: {}".format(e))

    # ====== РУЧНАЯ ОЦЕНКА ======
    eval_env_raw = make_vec_env(make_env, n_envs=1)
    from stable_baselines3.common.vec_env import VecNormalize as VN
    eval_env_loaded = VN.load(os.path.join(log_dir, "vecnormalize.pkl"), eval_env_raw)
    eval_env_loaded.training = False
    eval_env_loaded.norm_reward = True

    mean_reward, std_reward = evaluate_policy(model, eval_env_loaded, n_eval_episodes=50, deterministic=True)
    print("[Eval] mean_reward={:.3f} ± {:.3f}".format(mean_reward, std_reward))

    # ====== ДЕМОНСТРАЦИЯ (визуализация + лог действий/наблюдений) ======

    # Помощник для развёртывания всех обёрток до исходного env
    def unwrap_env(e):
        env = e
        while hasattr(env, "env"):
            env = env.env
        return env

    # 1) Env с render_mode='human'
    demo_env_raw = make_vec_env(lambda: GridWorldCombatEnv(render_mode="human", max_steps=1000), n_envs=1)

    # 2) Грузим те же статы нормализации на демо-окружение
    demo_env = VN.load(os.path.join(log_dir, "vecnormalize.pkl"), demo_env_raw)
    demo_env.training = False
    demo_env.norm_reward = True  # можно False, если нужны «сырые» награды

    # 3) Базовый env (для печати render и сырого obs)
    wrapped = demo_env.venv.envs[0]     # это Monitor
    base_env = unwrap_env(wrapped)      # это уже GridWorldCombatEnv

    # Список действий
    meanings = base_env.action_meanings
    actions_help = ", ".join(["{}:{}".format(i, m) for i, m in enumerate(meanings)])

    obs = demo_env.reset()
    done = False
    truncated = False
    total_r = 0.0
    steps = 0

    print("\nДемонстрация после обучения:")
    print("Доступные действия:", actions_help)
    # стартовая визуализация и obs
    base_env.render()
    print("Obs(model, normalized) = {}".format(obs[0].tolist()))
    print("Obs(raw)              = {}".format(base_env._get_obs().tolist()))

    while not (done or truncated):
        # Выбор действия
        action, _ = model.predict(obs, deterministic=True)
        a = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
        print("\n[Step {}] Chosen action: {} ({})".format(steps+1, a, meanings[a]))

        # Применяем действие
        obs, rewards, dones, infos = demo_env.step(action)
        done = bool(dones[0])
        truncated = bool(infos[0].get("TimeLimit.truncated", False))
        total_r += float(rewards[0])

        # Визуализация и текущее наблюдение
        base_env.render()
        print("Obs(model, normalized) = {}".format(obs[0].tolist()))
        print("Obs(raw)              = {}".format(base_env._get_obs().tolist()))
        print("Reward: {:.3f} | Done: {} | Truncated: {}".format(float(rewards[0]), done, truncated))

        steps += 1

    last_info = infos[0] if infos else {}
    print("\nУспех: {} | "
          "Уровень агента: {} | "
          "Wounded: {} | "
          "Следующий враг: {} | "
          "Шагов: {} | Суммарная награда: {:.3f}".format(
          last_info.get('is_success', False),
          last_info.get('agent_level'),
          last_info.get('wounded'),
          last_info.get('next_required_idx'),
          steps, total_r))