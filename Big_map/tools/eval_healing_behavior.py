"""Инструментовка: лечится ли агент (отдых/замок), пользуется ли зельями, и с каким HP
доходит до дракона. N детерминированных эпизодов последней 10M-модели (blue_dragon).

  python tools/eval_healing_behavior.py [episodes]
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

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MODEL_PATH = "campaign_models/offline-1782142870/final_model.zip"
VECNORM_PATH = "campaign_models/offline-1782142870/vecnormalize.pkl"
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


def hp_fraction(base):
    units = [u for u in base._get_blue_state()
             if str(u.get("name", "")).strip() and str(u.get("name", "")).strip() != "пусто"]
    cur = sum(max(0.0, float(u.get("health", 0) or 0)) for u in units)
    mx = sum(max(1.0, float(u.get("max_health", u.get("health", 1)) or 1)) for u in units)
    return cur / mx if mx > 0 else 0.0


def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False
    env.norm_reward = False
    base = env.venv.envs[0].unwrapped
    model = MaskablePPO.load(MODEL_PATH, env=env)

    print(f"model: {MODEL_PATH} | episodes: {EPISODES} | deterministic\n")
    keys = ["rest", "rest_units", "castle", "castle_hp", "item_used", "item_equip", "sold", "sale_gold"]
    agg = {k: 0.0 for k in keys}
    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        done = np.array([False])
        steps = 0
        c = {k: 0.0 for k in keys}
        hp_at_dragon = None
        min_hp = 1.0
        while not done[0] and steps < STEP_CAP:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, done, infos = env.step(action)
            steps += 1
            info = infos[0]
            if info.get("rest_action"):
                c["rest"] += 1
                c["rest_units"] += int(info.get("healed_units", 0) or 0)
            if info.get("castle_heal_action") and float(info.get("healed_amount", 0) or 0) > 0:
                c["castle"] += 1
                c["castle_hp"] += float(info.get("healed_amount", 0) or 0)
            if info.get("battle_hero_item_action") and info.get("battle_hero_item_applied"):
                c["item_used"] += 1
            if info.get("equip_battle_item_action") and info.get("equip_battle_item_applied"):
                c["item_equip"] += 1
            sold = int(info.get("merchant_sold_items_count", 0) or 0)
            if sold > 0:
                c["sold"] += sold
                c["sale_gold"] += float(info.get("merchant_sale_gold", 0) or 0)
            frac = hp_fraction(base)
            min_hp = min(min_hp, frac)
            if hp_at_dragon is None and info.get("green_dragon_reached"):
                hp_at_dragon = frac
        for k in keys:
            agg[k] += c[k]
        dragon_hp_str = f"{hp_at_dragon*100:.0f}%" if hp_at_dragon is not None else "не дошёл"
        print(f"Эп {ep}: отдых(rest)={int(c['rest'])} (вылечил юнитов суммарно {int(c['rest_units'])}), "
              f"замок/храм={int(c['castle'])} ({c['castle_hp']:.0f} HP) | "
              f"предметы: исп.={int(c['item_used'])} экип.={int(c['item_equip'])} | "
              f"продано {int(c['sold'])} (+{c['sale_gold']:.0f} зол.) | "
              f"HP армии у дракона: {dragon_hp_str} | мин.HP за эп: {min_hp*100:.0f}% | {steps} шаг")

    n = EPISODES
    print(f"\n=== СРЕДНЕЕ за {n} эп. ===")
    print(f"  Отдых (rest) на эпизод:        {agg['rest']/n:.2f}  (вылечено юнитов {agg['rest_units']/n:.2f})")
    print(f"  Лечение в замке/храме:         {agg['castle']/n:.2f}  ({agg['castle_hp']/n:.0f} HP)")
    print(f"  Боевые предметы использовано:  {agg['item_used']/n:.2f}")
    print(f"  Боевые предметы экипировано:   {agg['item_equip']/n:.2f}")
    print(f"  Продано предметов:             {agg['sold']/n:.2f}  (+{agg['sale_gold']/n:.0f} зол.)")


if __name__ == "__main__":
    main()
