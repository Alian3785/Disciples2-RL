"""SubprocVecEnv performance benchmarks using the actual campaign trainer.

Training FPS includes all PPO optimizer updates, including the final update.
Startup, saving, and evaluation are reported separately from model.learn().
"""

import argparse
import json
import os
import random
import runpy
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def profile_training(args):
    # Set thread limits before importing numerical libraries, including in spawn workers.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TORCH_DISABLE_DYNAMO"] = "1"
    from sb3_contrib.ppo_mask import MaskablePPO

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = {"n_envs": args.envs, "threads": args.threads, "seed": args.seed,
               "requested_steps": args.steps, "rollout_size": args.rollout_size,
               "batch_size": args.batch_size, "epochs": 5,
               "resume_model": args.resume_model,
               "resume_vecnormalize": args.resume_vecnormalize,
               "phases": [], "status": "running"}
    original_collect = MaskablePPO.collect_rollouts
    original_train = MaskablePPO.train
    original_learn = MaskablePPO.learn
    original_load = MaskablePPO.load

    def configured_load(cls, *positional, **kwargs):
        # load() otherwise restores the old rollout length from the checkpoint,
        # silently ignoring the trainer's CLI PPO settings when resuming.
        kwargs.update(n_steps=args.rollout_size // args.envs,
                      batch_size=args.batch_size, n_epochs=5)
        return original_load(*positional, **kwargs)

    def measured_collect(model, *positional, **kwargs):
        before = model.num_timesteps
        start = time.perf_counter()
        result = original_collect(model, *positional, **kwargs)
        profile["phases"].append({"phase": "rollout", "seconds": time.perf_counter() - start,
                                  "steps": model.num_timesteps - before,
                                  "total_steps": model.num_timesteps})
        return result

    def measured_train(model, *positional, **kwargs):
        start = time.perf_counter()
        result = original_train(model, *positional, **kwargs)
        profile["phases"].append({"phase": "update", "seconds": time.perf_counter() - start,
                                  "total_steps": model.num_timesteps})
        save_json(output / "profile.json", profile)
        return result

    def measured_learn(model, *positional, **kwargs):
        before_steps = model.num_timesteps
        before_updates = model._n_updates
        profile.update(start_timesteps=before_steps, effective_n_envs=model.n_envs,
                       effective_n_steps=model.n_steps, effective_batch_size=model.batch_size,
                       torch_threads=__import__("torch").get_num_threads())
        assert model.n_envs == args.envs
        assert model.n_steps * model.n_envs == args.rollout_size
        save_json(output / "profile.json", profile)
        start = time.perf_counter()
        result = original_learn(model, *positional, **kwargs)
        seconds = time.perf_counter() - start
        actual_steps = model.num_timesteps - before_steps
        profile.update(learn_seconds=seconds, actual_steps=actual_steps,
                       final_timesteps=model.num_timesteps,
                       fps=actual_steps / seconds, updates=model._n_updates - before_updates,
                       torch_threads=__import__("torch").get_num_threads())
        return result

    MaskablePPO.collect_rollouts = measured_collect
    MaskablePPO.train = measured_train
    MaskablePPO.learn = measured_learn
    MaskablePPO.load = classmethod(configured_load)
    sys.argv = [str(ROOT / "train_campaign.py"),
                "--map", "super_last_stand", "--total-steps", str(args.steps),
                "--n-envs", str(args.envs), "--vec-env", "subproc",
                "--vec-start-method", "spawn", "--torch-num-threads", str(args.threads),
                "--ppo-n-steps", str(args.rollout_size // args.envs),
                "--ppo-batch-size", str(args.batch_size), "--ppo-n-epochs", "5",
                "--seed", str(args.seed), "--no-comet", "--eval-freq", "0",
                "--checkpoint-freq", "1000000000", "--run-id", output.name,
                "--output-root", str(output), "--comet-offline-dir", str(output / "comet_offline"),
                "--behavior-report-path", str(output / "behavior.jsonl")]
    if args.resume_model:
        sys.argv += ["--resume-model", args.resume_model,
                     "--resume-vecnormalize", args.resume_vecnormalize]
    profile["command"] = sys.argv.copy()
    start = time.perf_counter()
    try:
        runpy.run_path(str(ROOT / "train_campaign.py"), run_name="__main__")
        profile["status"] = "completed"
    except BaseException as exc:
        profile.update(status="failed", error=repr(exc))
        raise
    finally:
        profile["script_seconds"] = time.perf_counter() - start
        save_json(output / "profile.json", profile)
    print("BENCHMARK_RESULT " + json.dumps({k: v for k, v in profile.items()
                                           if k not in ("command", "phases")}))


def run_sweep(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = [(n, t) for n in (1, 2, 4, 6, 8, 12) for t in (1, 2, 4)]
    configs += [(8, 8), (12, 8)]
    random.Random(2026).shuffle(configs)
    save_json(output / "sweep_plan.json", {"configurations": configs, "seed": args.seed,
                                           "steps": args.steps, "rollout_size": args.rollout_size})
    rows = []
    for index, (envs, threads) in enumerate(configs, 1):
        run_dir = output / f"e{envs}_t{threads}_s{args.seed}"
        run_dir.mkdir(exist_ok=True)
        profile_path = run_dir / "profile.json"
        if profile_path.exists() and json.loads(profile_path.read_text()).get("status") == "completed":
            profile = json.loads(profile_path.read_text())
        else:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "train",
                       "--output", str(run_dir), "--envs", str(envs), "--threads", str(threads),
                       "--steps", str(args.steps), "--rollout-size", str(args.rollout_size),
                       "--batch-size", str(args.batch_size), "--seed", str(args.seed)]
            print(f"START {index}/{len(configs)} envs={envs} threads={threads}", flush=True)
            start = time.perf_counter()
            with (run_dir / "train.log").open("w") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            wall_seconds = time.perf_counter() - start
            if process.returncode:
                print(f"FAILED {run_dir}: exit={process.returncode}", flush=True)
                raise SystemExit(process.returncode)
            profile = json.loads(profile_path.read_text())
            profile["process_seconds"] = wall_seconds
            save_json(profile_path, profile)
        row = {k: v for k, v in profile.items() if k not in ("command", "phases")}
        row["run_dir"] = str(run_dir)
        row["rollout_seconds"] = sum(p["seconds"] for p in profile["phases"] if p["phase"] == "rollout")
        row["update_seconds"] = sum(p["seconds"] for p in profile["phases"] if p["phase"] == "update")
        rows.append(row)
        save_json(output / "sweep_results.json", rows)
        print(f"DONE {index}/{len(configs)} envs={envs} threads={threads} fps={row['fps']:.1f} "
              f"rollout={row['rollout_seconds']:.2f}s update={row['update_seconds']:.2f}s", flush=True)


def evaluate(args):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from functools import partial

    import numpy as np
    import torch
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from battle_env import BattleEnv
    from maps.super_last_stand import SCHEDULED_WAVES
    from train_campaign import MaskablePPO, make_env

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # BattleEnv owns random.Random() independently of Gym's seed. Seed that
    # generator explicitly in evaluation only, before custom teams are initialized.
    # The training benchmark retains the original environment unchanged.
    battle_seeds = random.Random(0)
    original_battle_init = BattleEnv.__init__

    def seeded_battle_init(battle, *positional, **kwargs):
        original_battle_init(battle, *positional, **kwargs)
        battle.seed(battle_seeds.getrandbits(64))

    BattleEnv.__init__ = seeded_battle_init
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.eval_output).resolve() if args.eval_output else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path) if args.model_path else run_dir / "models" / "final_model.zip"
    vec_path = Path(args.vecnormalize_path) if args.vecnormalize_path else run_dir / "models" / "vecnormalize.pkl"
    wave_by_enemy = {int(enemy): int(wave["wave_index"])
                     for wave in SCHEDULED_WAVES for enemy in wave["enemy_ids"]}
    map_name = getattr(args, "map_name", "super_last_stand")
    objective = getattr(args, "objective", None)
    if map_name != "super_last_stand":
        wave_by_enemy = {}
    env = DummyVecEnv([partial(make_env, map_name=map_name, campaign_objective=objective,
                               use_boss_starting_roster=False if getattr(args, "no_boss_roster", False) else None,
                               eval_max_episode_steps=5000)])
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False
    model = MaskablePPO.load(str(model_path), env=env, device="cpu")
    base = env.envs[0].unwrapped
    results = {"model_path": str(model_path), "model_steps": int(model.num_timesteps),
               "map": map_name, "objective": base.campaign_objective,
               "use_boss_starting_roster": base.use_boss_starting_roster}
    start = time.perf_counter()
    for deterministic in (True, False):
        mode = "deterministic" if deterministic else "stochastic"
        episodes = []
        for index in range(args.episodes):
            episode_seed = args.seed + index
            set_random_seed(episode_seed)
            battle_seeds.seed(episode_seed)
            env.seed(episode_seed)
            obs = env.reset()
            previous_position = tuple(base.grid_env.agent_pos)
            capital_position = tuple(base.CASTLE_POS)
            reward_sum = 0.0
            length = 0
            enemies = set()
            waves = 0
            wave_stacks = 0
            reached_wave = 0
            activity = Counter()
            named_actions = {key: Counter() for key in ("buildings", "hires", "learned_spells", "cast_spells")}
            positions = set()
            encounters = set()
            max_tier = 0
            while True:
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=get_action_masks(env))
                obs, reward, done, infos = env.step(action)
                info = infos[0]
                reward_sum += float(reward[0])
                length += 1
                if info.get("battle_result") == "victory" or info.get("spell_enemy_defeated"):
                    enemy_id = info.get("enemy_id", info.get("target_enemy_id"))
                    if enemy_id is not None:
                        enemies.add(int(enemy_id))
                waves = max(waves, int(info.get("waves_defeated_count", 0)))
                wave_stacks = max(wave_stacks, int(info.get("wave_stacks_defeated_count", 0)))
                if info.get("battle_triggered") or info.get("battle_result") or info.get("spell_enemy_defeated"):
                    enemy_id = info.get("enemy_id", info.get("target_enemy_id"))
                    if enemy_id is not None:
                        encounters.add(int(enemy_id))
                        reached_wave = max(reached_wave, wave_by_enemy.get(int(enemy_id), 0))
                if info.get("terminal_defeat_enemy_id") is not None:
                    reached_wave = max(reached_wave, wave_by_enemy.get(int(info["terminal_defeat_enemy_id"]), 0))
                if info.get("agent_pos") is not None:
                    position = tuple(info["agent_pos"])
                    positions.add(position)
                    activity["capital_returns"] += int(position == capital_position and previous_position != capital_position)
                    previous_position = position
                activity["castle_heals"] += int(bool(info.get("castle_heal_action") and info.get("healed_amount", 0) > 0))
                activity["castle_revives"] += int(bool(info.get("castle_revive_action") and info.get("revived")))
                activity["buildings"] += int(bool(info.get("build_action") and info.get("built")))
                activity["hires"] += int(info.get("hired_units_count", 0) or bool(info.get("hired")))
                activity["spell_casts"] += int(bool(info.get("spell_cast_action") and info.get("spell_cast_executed")))
                activity["spells_learned"] += int(bool(info.get("spell_learned")))
                activity["summons"] += (int(info.get("spell_units_affected", 1) or 1)
                                        if info.get("spell_kind") == "summon_battle" and info.get("spell_cast_executed") else 0)
                activity["unit_swaps"] += int(bool(info.get("unit_swap_action") and info.get("unit_swap_applied")))
                activity["unit_upgrades"] += (int(info.get("unit_upgrades", 0) or 0)
                                              if info.get("battle_result") == "victory" else 0)
                activity["ruins"] += int(bool(info.get("ruin_reward_applied")))
                collected = info.get("collected_chests")
                activity["chests"] += (len(collected) if isinstance(collected, (list, tuple))
                                       else int(info.get("collected_chests_count", 0) or 0))
                activity["items_used"] = max(activity["items_used"], int(info.get("battle_items_used_total", 0)))
                max_tier = max(max_tier, int(info.get("unit_upgrade_max_tier", 0) or 0))
                if info.get("build_action") and info.get("built"):
                    named_actions["buildings"][str(info.get("build_name") or info.get("build_key"))] += 1
                if info.get("hired"):
                    named_actions["hires"][str(info.get("hired_unit_name") or info.get("hire_unit_name"))] += 1
                if info.get("spell_learned"):
                    named_actions["learned_spells"][str(info.get("spell_description") or info.get("spell_key"))] += 1
                if info.get("spell_cast_action") and info.get("spell_cast_executed"):
                    named_actions["cast_spells"][str(info.get("spell_key"))] += 1
                if done[0]:
                    episodes.append({"seed": episode_seed, "reward": reward_sum, "length": length,
                                     "result": info.get("campaign_result", "other"),
                                     "waves": waves, "wave_stacks": wave_stacks,
                                     "reached_wave": reached_wave, "activity": dict(activity),
                                     "named_actions": {key: dict(value) for key, value in named_actions.items()},
                                     "unique_positions": len(positions), "max_unit_tier": max_tier,
                                     "encounter_ids": sorted(encounters),
                                     "enemies": len(enemies), "enemy_ids": sorted(enemies),
                                     "terminal_enemy": info.get("terminal_defeat_enemy_id"),
                                     "terminal_enemy_description": info.get("terminal_defeat_enemy_description", "")})
                    break
        results[mode] = {"episodes": episodes,
                         "wins": sum(e["result"] == "victory" for e in episodes),
                         "timeouts": sum(e["result"] in ("timeout", "eval_timeout") for e in episodes),
                         "dragon_encounters": sum(31 in e["encounter_ids"] for e in episodes),
                         "dragon_victories": sum(31 in e["enemy_ids"] for e in episodes),
                         "max_reached_wave": max(e["reached_wave"] for e in episodes),
                         "max_waves_defeated": max(e["waves"] for e in episodes),
                         "reached_wave_counts": dict(Counter(e["reached_wave"] for e in episodes)),
                         "terminal_enemies": dict(Counter(str(e["terminal_enemy"]) for e in episodes)),
                         "common_actions": {key: dict(sum((Counter(e["named_actions"][key]) for e in episodes), Counter()).most_common(8))
                                            for key in ("buildings", "hires", "learned_spells", "cast_spells")},
                         "mean_activity": {key: float(np.mean([e["activity"].get(key, 0) for e in episodes]))
                                           for key in sorted({k for e in episodes for k in e["activity"]})},
                         **{f"mean_{key}": float(np.mean([e[key] for e in episodes]))
                            for key in ("reward", "length", "waves", "wave_stacks", "enemies", "reached_wave", "unique_positions", "max_unit_tier")}}
        save_json(output_dir / "evaluation.json", results)
        print(mode, json.dumps({k: v for k, v in results[mode].items() if k != "episodes"}), flush=True)
    env.close()
    results["evaluation_seconds"] = time.perf_counter() - start
    save_json(output_dir / "evaluation.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("train", "sweep"):
        task = sub.add_parser(name)
        task.add_argument("--output", required=True)
        task.add_argument("--steps", type=int, default=12000)
        task.add_argument("--rollout-size", type=int, default=12000)
        task.add_argument("--batch-size", type=int, default=500)
        task.add_argument("--seed", type=int, default=42)
        if name == "train":
            task.add_argument("--envs", type=int, required=True)
            task.add_argument("--threads", type=int, required=True)
            task.add_argument("--resume-model")
            task.add_argument("--resume-vecnormalize")
    task = sub.add_parser("eval")
    task.add_argument("--run-dir", required=True)
    task.add_argument("--model-path")
    task.add_argument("--vecnormalize-path")
    task.add_argument("--eval-output")
    task.add_argument("--map", dest="map_name", default="super_last_stand")
    task.add_argument("--objective", choices=("cities", "dragon", "blue_dragon", "full_party"))
    task.add_argument("--no-boss-roster", action="store_true",
                      help="Evaluate with the ordinary starting party")
    task.add_argument("--episodes", type=int, default=20)
    task.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if args.mode in ("train", "sweep"):
        if args.steps % args.rollout_size or args.rollout_size % args.batch_size:
            parser.error("steps must be divisible by rollout size, and rollout by batch size")
        divisors = (args.envs,) if args.mode == "train" else (1, 2, 4, 6, 8, 12)
        if any(args.rollout_size % n for n in divisors):
            parser.error("rollout size must be divisible by every environment count")
        if args.mode == "train" and bool(args.resume_model) != bool(args.resume_vecnormalize):
            parser.error("resuming requires both --resume-model and --resume-vecnormalize")
    {"train": profile_training, "sweep": run_sweep, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
