"""Трассировка боёв с драконом-целью за эпизод: сколько заходов, чем кончаются,
до какого HP продавлен дракон, и продолжает ли агент зачищать карту после выхода
на дракона. Модель 7.5M (blue_dragon), стохастика (чаще доходит до дракона).

  python tools/trace_dragon_episode.py [episodes]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from campaign_env import CampaignEnv
from grid import DEFAULT_GRID_SIZE

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODEL = "campaign_checkpoints/offline-1782207920/campaign_ppo_step_7500000.zip"
VECNORM = "campaign_checkpoints/offline-1782207920/vecnormalize_step_7500000.pkl"
STEP_CAP = 5000


def mask_fn(env):
    return env.compute_action_mask()


def make_env():
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE, persist_blue_hp=True, log_enabled=False,
        freeze_dynamic_action_layout=True, empire_territory_enabled=True,
        scripted_capital_bot_enabled=False, campaign_objective="blue_dragon",
        max_grid_steps=3000, Realcapital=2,
    )
    return Monitor(ActionMasker(base, mask_fn))


def enemies_defeated(base):
    ea = getattr(base.grid_env, "enemies_alive", {}) or {}
    return sum(1 for a in ea.values() if not bool(a)), len(ea)


def dragon_red_hp(base):
    be = getattr(base, "battle_env", None)
    if be is None:
        return None
    return sum(max(0.0, float(u.get("health", 0) or 0))
               for u in getattr(be, "combined", []) if u.get("team") == "red")


def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VECNORM, env)
    env.training = False
    env.norm_reward = False
    base = env.venv.envs[0].unwrapped
    model = MaskablePPO.load(MODEL, env=env)
    DRAGON_ID = int(base.GREEN_DRAGON_OBJECTIVE_ENEMY_ID)
    DRAGON_MAX_HP = 700
    print(f"model: {MODEL} | DRAGON_ID={DRAGON_ID} | episodes: {EPISODES} | STOCHASTIC\n")

    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        done = np.array([False])
        steps = 0
        cur_enemy = None
        dragon_engagements = dragon_retreats = dragon_wipes = dragon_wins = 0
        dragon_min_hp = None
        first_dragon_step = None
        defeated_at_first_dragon = None
        rest_after_dragon = 0
        non_dragon_battles_after = 0
        result = ""
        while not done[0] and steps < STEP_CAP:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=False, action_masks=masks)
            obs, _, done, infos = env.step(action)
            steps += 1
            info = infos[0]
            # начало боя
            if info.get("enemy_id") is not None and info.get("mode") == "battle":
                cur_enemy = int(info.get("enemy_id"))
                if cur_enemy == DRAGON_ID and first_dragon_step is None:
                    first_dragon_step = steps
                    defeated_at_first_dragon = enemies_defeated(base)[0]
            # сэмплируем HP дракона во время боя с ним
            if cur_enemy == DRAGON_ID:
                hp = dragon_red_hp(base)
                if hp is not None:
                    dragon_min_hp = hp if dragon_min_hp is None else min(dragon_min_hp, hp)
            # после первого выхода на дракона — считаем активность
            if first_dragon_step is not None and info.get("rest_action"):
                rest_after_dragon += 1
            # разрешение боя
            if info.get("battle_result") is not None:
                if cur_enemy == DRAGON_ID:
                    dragon_engagements += 1
                    if info.get("battle_retreat"):
                        dragon_retreats += 1
                    elif info.get("battle_result") == "victory":
                        dragon_wins += 1
                    else:
                        dragon_wipes += 1
                elif first_dragon_step is not None:
                    non_dragon_battles_after += 1
                cur_enemy = None
            if done[0]:
                result = str(info.get("campaign_result", "") or "")

        defeated_end, total = enemies_defeated(base)
        dmin = f"{dragon_min_hp:.0f}/{DRAGON_MAX_HP} ({100*dragon_min_hp/DRAGON_MAX_HP:.0f}%)" if dragon_min_hp is not None else "—"
        if first_dragon_step is None:
            print(f"Эп {ep}: до дракона НЕ дошёл | побеждено {defeated_end}/{total} | "
                  f"{steps} шаг | конец: {result or 'лимит'}")
        else:
            post = defeated_end - (defeated_at_first_dragon or 0)
            print(f"Эп {ep}: {steps} шаг, конец: {result or 'лимит'} | побеждено {defeated_end}/{total}")
            print(f"   дракон: первый заход на шаге {first_dragon_step} (тогда было побеждено {defeated_at_first_dragon}/{total})")
            print(f"   заходов в бой с драконом: {dragon_engagements}  (отступлений {dragon_retreats}, вайпов {dragon_wipes}, побед {dragon_wins})")
            print(f"   мин. HP дракона за бои: {dmin}")
            print(f"   ПОСЛЕ первого дракона: отрядов добито ещё {post}, обычных боёв {non_dragon_battles_after}, отдыхов {rest_after_dragon}")


if __name__ == "__main__":
    main()
