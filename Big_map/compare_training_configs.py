"""Sequential longer training and held-out evaluation of benchmark finalists."""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from benchmark_training import ROOT, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--configs", required=True, help="Comma-separated envs:threads pairs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--steps", type=int, default=192000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--rollout-size", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configurations = [tuple(int(v) for v in config.split(":")) for config in args.configs.split(",")]
    trials = [(n, t, s) for n, t in configurations for s in args.seeds]
    random.Random(2026).shuffle(trials)
    save_json(output / "comparison_plan.json", {**vars(args), "trials": trials,
                                               "evaluation_seed": 10000})
    rows = []
    for index, (envs, threads, seed) in enumerate(trials, 1):
        run_dir = output / f"e{envs}_t{threads}_s{seed}"
        run_dir.mkdir(exist_ok=True)
        profile_path = run_dir / "profile.json"
        complete = profile_path.exists() and json.loads(profile_path.read_text()).get("status") == "completed"
        if not complete:
            print(f"TRAIN {index}/{len(trials)} envs={envs} threads={threads} seed={seed}", flush=True)
            command = [sys.executable, "-u", str(ROOT / "benchmark_training.py"), "train",
                       "--output", str(run_dir), "--envs", str(envs), "--threads", str(threads),
                       "--seed", str(seed), "--steps", str(args.steps),
                       "--rollout-size", str(args.rollout_size), "--batch-size", str(args.batch_size)]
            start = time.perf_counter()
            with (run_dir / "train.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            profile = json.loads(profile_path.read_text())
            profile["process_seconds"] = time.perf_counter() - start
            save_json(profile_path, profile)
        profile = json.loads(profile_path.read_text())
        print(f"TRAINED envs={envs} threads={threads} seed={seed} fps={profile['fps']:.1f}", flush=True)
        evaluation_path = run_dir / "evaluation.json"
        complete_eval = evaluation_path.exists() and "evaluation_seconds" in json.loads(evaluation_path.read_text())
        if not complete_eval:
            print(f"EVAL {index}/{len(trials)} {args.episodes} deterministic + {args.episodes} stochastic episodes", flush=True)
            command = [sys.executable, "-u", str(ROOT / "benchmark_training.py"), "eval",
                       "--run-dir", str(run_dir), "--episodes", str(args.episodes), "--seed", "10000"]
            with (run_dir / "eval.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        evaluation = json.loads(evaluation_path.read_text())
        row = {k: v for k, v in profile.items() if k not in ("phases", "command")}
        row["run_dir"] = str(run_dir)
        row["evaluation"] = {mode: {k: v for k, v in evaluation[mode].items() if k != "episodes"}
                             for mode in ("deterministic", "stochastic")}
        rows.append(row)
        save_json(output / "comparison_results.json", rows)
        print(f"DONE {index}/{len(trials)} " + json.dumps(row["evaluation"]), flush=True)


if __name__ == "__main__":
    main()
