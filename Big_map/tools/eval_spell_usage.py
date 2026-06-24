"""Прогоняет модель в логируемом окружении blue_dragon и печатает только строки
применения заклинаний — для подсчёта, какие заклинания агент кастует. Также считает призывы."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import io
import contextlib
import re
from collections import Counter

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from campaign_env import CampaignEnv
from grid import DEFAULT_GRID_SIZE

MODEL = sys.argv[1]
VEC = sys.argv[2]
EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 3


def mask_fn(env):
    return env.compute_action_mask()


def make_env():
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE, persist_blue_hp=True, log_enabled=True,
        freeze_dynamic_action_layout=True, empire_territory_enabled=True,
        scripted_capital_bot_enabled=False, campaign_objective="blue_dragon",
        max_grid_steps=2500, Realcapital=2,
    )
    return Monitor(ActionMasker(base, mask_fn))


def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC, env)
    env.training = False
    env.norm_reward = False
    model = MaskablePPO.load(MODEL, env=env)

    spell_counter = Counter()
    summon_counter = Counter()
    spell_re = re.compile(r'Применено заклинание "([^"]+)"')
    summon_re = re.compile(r'[Пп]ризыв[^\n]*?"([^"]+)"')
    episodes_done = 0

    for _ in range(EPISODES):
        obs = env.reset()
        done = np.array([False])
        while not done[0]:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                obs, _, done, _ = env.step(action)
            for line in buf.getvalue().splitlines():
                m = spell_re.search(line)
                if m:
                    spell_counter[m.group(1)] += 1
                ms = summon_re.search(line)
                if ms and "заклинание" not in line:
                    summon_counter[ms.group(1)] += 1
        episodes_done += 1

    total = sum(spell_counter.values())
    print(f"эпизодов: {episodes_done} | всего кастов заклинаний: {total} "
          f"({total/max(1,episodes_done):.1f}/эп)")
    print("ТОП заклинаний (имя: всего | на эпизод):")
    for name, cnt in spell_counter.most_common(15):
        print(f"  {name}: {cnt} | {cnt/max(1,episodes_done):.1f}/эп")
    if summon_counter:
        print("Призывы:")
        for name, cnt in summon_counter.most_common(10):
            print(f"  {name}: {cnt} | {cnt/max(1,episodes_done):.1f}/эп")


if __name__ == "__main__":
    main()
