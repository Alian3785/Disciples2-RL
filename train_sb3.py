"""
Training script for PPO on BattleEnv.
This file focuses ONLY on training and checkpointing; no visualization here.
"""

from __future__ import annotations

import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env

from battle_env import BattleEnv, TARGET_POSITIONS, TYPE_LIST, ATTACK_TYPES

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False


TOTAL_STEPS     = int(os.environ.get("TOTAL_STEPS", "100000"))
N_ENVS          = int(os.environ.get("N_ENVS", "8"))
MODEL_SAVE_FREQ = int(os.environ.get("MODEL_SAVE_FREQ", "1000"))
EVAL_FREQ       = int(os.environ.get("EVAL_FREQ", "10000"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "100000"))


def main() -> None:
    check_env(BattleEnv(log_enabled=False), warn=True)

    USE_WANDB = bool(WANDB_AVAILABLE) and os.environ.get("WANDB_DISABLED", "false").lower() not in ("1","true","yes")
    run = None
    if USE_WANDB:
        try:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "red-blue-battle"),
                name=os.environ.get("RUN_NAME", f"ppo-{TOTAL_STEPS//1000}k"),
                config={
                    "algo": "PPO",
                    "total_timesteps": TOTAL_STEPS,
                    "n_envs": N_ENVS,
                    "n_steps": 1024,
                    "batch_size": 2048,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "learning_rate": 3e-4,
                    "clip_range": 0.2,
                    "model_save_freq": MODEL_SAVE_FREQ,
                    "eval_freq": EVAL_FREQ,
                    "obs_dim": 12*45,
                    "types": TYPE_LIST,
                    "attack_types": ATTACK_TYPES,
                    "action_space": len(TARGET_POSITIONS),
                    "checkpoint_every": CHECKPOINT_EVERY,
                },
                sync_tensorboard=True,
                save_code=True,
            )
            run_id = run.id
        except Exception as e:
            print(f"[W&B] Disabled: {e}")
            USE_WANDB = False
            run = None
            run_id = time.strftime("%Y%m%d-%H%M%S")
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")

    checkpoint_dir = f"./checkpoints/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env():
        return Monitor(BattleEnv(reward_win=1.0, reward_loss=-1.0, reward_step=0.0, log_enabled=False))

    vec_env  = make_vec_env(make_env, n_envs=N_ENVS)
    eval_env = Monitor(BattleEnv(log_enabled=False))

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=2048,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        tensorboard_log=f"./tb_logs/{run_id}",
    )

    class MillionStepCheckpointCallback(BaseCallback):
        def __init__(self, save_dir: str, milestone_steps: int = 1_000_000, verbose: int = 1):
            super().__init__(verbose)
            self.save_dir = save_dir
            self.milestone_steps = int(milestone_steps)
            self.next_milestone = self.milestone_steps
            os.makedirs(self.save_dir, exist_ok=True)

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_milestone:
                path = os.path.join(self.save_dir, f"ppo_step_{self.next_milestone}.zip")
                self.model.save(path)
                if self.verbose:
                    print(f"[Checkpoint] Saved at {self.next_milestone} steps → {path}")
                if USE_WANDB:
                    wandb.log({"checkpoint_saved_at": self.next_milestone})
                self.next_milestone += self.milestone_steps
            return True

    checkpoint_cb = MillionStepCheckpointCallback(save_dir=checkpoint_dir, milestone_steps=CHECKPOINT_EVERY, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{run_id}/best",
        log_path=f"./eval/{run_id}",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = [eval_cb, checkpoint_cb]
    if USE_WANDB:
        from wandb.integration.sb3 import WandbCallback
        callbacks.insert(0, WandbCallback(gradient_save_freq=10000, model_save_path=f"./models/{run_id}", model_save_freq=MODEL_SAVE_FREQ, verbose=2))

    model.learn(total_timesteps=TOTAL_STEPS, callback=CallbackList(callbacks))

    final_model_path = f"./models/{run_id}/final_model"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    if USE_WANDB and run is not None:
        run.finish()


if __name__ == "__main__":
    main()


