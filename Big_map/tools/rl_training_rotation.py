#!/usr/bin/env python3
"""Prepare and finalize the scheduled GitHub Actions training rotation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAP_ROTATION = (
    "default",
    "small",
    "green_dragon_minimal",
    "super_last_stand",
    "orc_duel",
    "builder",
    "formation_train",
    "hire_train",
    "item_train",
    "magic_train",
    "scroll_train",
    "siege_train",
    "trade_train",
    "wotans_retribution",
)
# One seed is held constant until every supported map/objective pair has run.
# The order deliberately keeps modes of the same physical map adjacent.
MODE_ROTATION = (
    ("default", "cities"),
    ("default", "green_dragon"),
    ("default", "blue_dragon"),
    ("default", "full_party"),
    ("small", "cities"),
    ("small", "green_dragon"),
    ("small", "blue_dragon"),
    ("green_dragon_minimal", "green_dragon"),
    ("green_dragon_minimal", "blue_dragon"),
    ("super_last_stand", "waves"),
    ("orc_duel", "orc"),
    ("builder", "build_all"),
    ("formation_train", "all_enemies"),
    ("hire_train", "orc"),
    ("item_train", "all_enemies"),
    ("magic_train", "all_enemies"),
    ("scroll_train", "all_enemies"),
    ("siege_train", "target_enemy"),
    ("trade_train", "all_enemies"),
    ("wotans_retribution", "cities"),
)
NEXT_MODE = {
    mode: MODE_ROTATION[(index + 1) % len(MODE_ROTATION)]
    for index, mode in enumerate(MODE_ROTATION)
}
# Default objective for a manually selected smoke map. Scheduled runs use the
# exact pair stored in next_map + next_objective instead.
MAP_OBJECTIVES = {
    "default": "full_party",
    "small": "green_dragon",
    "green_dragon_minimal": "green_dragon",
    "super_last_stand": "waves",
    "orc_duel": "orc",
    "builder": "build_all",
    "formation_train": "all_enemies",
    "hire_train": "orc",
    "item_train": "all_enemies",
    "magic_train": "all_enemies",
    "scroll_train": "all_enemies",
    "siege_train": "target_enemy",
    "trade_train": "all_enemies",
    "wotans_retribution": "cities",
}
OBJECTIVE_FLAGS = {
    "green_dragon": "--dragon",
    "blue_dragon": "--blue-dragon",
    "full_party": "--full-party",
}
DEFAULT_STATE = {
    "active": True,
    "seed": 101,
    "next_map": "default",
    "next_objective": "cities",
    "completed_runs": 0,
    "last_success": None,
}


def load_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("active"), bool):
        raise ValueError("state.active must be a boolean")
    if not isinstance(data.get("seed"), int) or data["seed"] < 0:
        raise ValueError("state.seed must be a non-negative integer")
    if "next_map" not in data and data.get("next_objective") is not None:
        # Migrate the previous three-objective small-map rotation at the next
        # successful run. A completed legacy cycle starts the all-map rotation
        # from its first map without discarding the accumulated seed/counters.
        data["next_map"] = "default"
        data["next_objective"] = "cities"
    if data.get("next_map") not in MAP_ROTATION:
        raise ValueError(f"state.next_map must be one of {MAP_ROTATION}")
    if "next_objective" not in data:
        # Migrate the previous physical-map rotation to the first mode of the
        # pending map. This keeps its seed and completion counter intact.
        data["next_objective"] = next(
            objective
            for map_name, objective in MODE_ROTATION
            if map_name == data["next_map"]
        )
    next_mode = (str(data["next_map"]), str(data["next_objective"]))
    if next_mode not in MODE_ROTATION:
        raise ValueError(f"state next mode must be one of {MODE_ROTATION}, got {next_mode}")
    if not isinstance(data.get("completed_runs"), int) or data["completed_runs"] < 0:
        raise ValueError("state.completed_runs must be a non-negative integer")
    if "last_success" not in data:
        raise ValueError("state.last_success is required")
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not token:
        raise ValueError("run token must contain at least one safe character")
    return token


def prepare_run(
    *,
    state: dict[str, Any],
    output_base: Path,
    event_name: str,
    manual_map: str,
    run_token: str,
) -> dict[str, str]:
    is_manual = event_name == "workflow_dispatch"
    map_name = manual_map if is_manual else str(state["next_map"])
    if map_name not in MAP_ROTATION:
        raise ValueError(f"unknown map: {map_name}")
    objective = (
        MAP_OBJECTIVES[map_name]
        if is_manual
        else str(state["next_objective"])
    )
    mode = (map_name, objective)
    if mode not in MODE_ROTATION:
        raise ValueError(f"unknown map/objective mode: {mode}")
    seed = int(state["seed"])
    token = _safe_token(run_token)
    prefix = (
        f"smoke-{map_name}-{objective}"
        if is_manual
        else f"seed{seed}-{map_name}-{objective}"
    )
    run_id = f"{prefix}-{token}"
    if is_manual:
        run_dir = output_base / run_id
        total_steps = 24_576
        checkpoint_freq = 12_288
        eval_freq = 2_048
    else:
        run_dir = output_base / str(seed) / map_name / objective / run_id
        total_steps = 1_000_000
        checkpoint_freq = 500_000
        eval_freq = 83_334

    return {
        "should_run": "true",
        "reason": "manual-smoke" if is_manual else "scheduled-training",
        "is_manual": str(is_manual).lower(),
        "advance_state": str(not is_manual).lower(),
        "seed": str(seed),
        "campaign_map": map_name,
        "objective": objective,
        "objective_flag": OBJECTIVE_FLAGS.get(objective, ""),
        "cycle_complete": str(mode == MODE_ROTATION[-1]).lower(),
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "total_steps": str(total_steps),
        "checkpoint_freq": str(checkpoint_freq),
        "eval_freq": str(eval_freq),
    }


def _require_files(directory: Path, expected_names: set[str], label: str) -> None:
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    missing = expected_names - actual_names
    if missing:
        raise FileNotFoundError(f"missing {label}: {', '.join(sorted(missing))}")


def finalize_success(
    *,
    state_file: Path,
    run_dir: Path,
    seed: int,
    map_name: str,
    objective: str,
    run_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    state = load_state(state_file)
    if map_name not in MAP_ROTATION:
        raise ValueError(f"unknown map: {map_name}")
    mode = (map_name, objective)
    if mode not in MODE_ROTATION:
        raise ValueError(f"unknown map/objective mode: {mode}")
    expected_mode = (str(state["next_map"]), str(state["next_objective"]))
    if seed != state["seed"] or mode != expected_mode:
        raise ValueError("run no longer matches the current rotation state")

    checkpoint_dir = run_dir / "checkpoints"
    model_best_dir = run_dir / "models" / "best"
    eval_dir = run_dir / "eval"
    _require_files(
        checkpoint_dir,
        {
            "campaign_ppo_step_500000.zip",
            "vecnormalize_step_500000.pkl",
            "campaign_ppo_step_1000000.zip",
            "vecnormalize_step_1000000.pkl",
        },
        "scheduled checkpoints",
    )
    _require_files(model_best_dir, {"best_model.zip", "best_vecnormalize.pkl"}, "best model")
    _require_files(eval_dir, {"evaluations.npz"}, "evaluation output")

    eval_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best_model.zip", "best_vecnormalize.pkl"):
        shutil.move(str(model_best_dir / name), str(eval_dir / name))

    completed_at = datetime.now(timezone.utc).isoformat()
    next_map, next_objective = NEXT_MODE[mode]
    cycle_complete = mode == MODE_ROTATION[-1]
    next_seed = seed + 1 if cycle_complete else seed
    manifest = {
        "status": "success",
        "run_id": run_id,
        "seed": seed,
        "map": map_name,
        "objective": objective,
        "commit_sha": commit_sha,
        "completed_at": completed_at,
        "parameters": {
            "total_steps": 1_000_000,
            "map": map_name,
            "objective": objective,
            "n_envs": 12,
            "vec_env": "subproc",
            "vec_start_method": "forkserver",
            "torch_num_threads": 4,
            "checkpoint_freq": 500_000,
            "eval_freq": 83_334,
            "scripted_bot": False,
            "comet": False,
            "vec_check_nan": False,
            "diagnostics": False,
        },
        "artifacts": {
            "persistent": [
                "eval/evaluations.npz",
                "eval/best_model.zip",
                "eval/best_vecnormalize.pkl",
                "logs/",
                "tensorboard/",
                "plots/",
            ],
            "checkpoint_artifact_retention_days": 1,
        },
    }
    write_json_atomic(run_dir / "manifest.json", manifest)

    state.update(
        {
            "active": True,
            "seed": next_seed,
            "next_map": next_map,
            "next_objective": next_objective,
            "completed_runs": int(state["completed_runs"]) + 1,
            "last_success": {
                "run_id": run_id,
                "seed": seed,
                "map": map_name,
                "objective": objective,
                "completed_at": completed_at,
                "commit_sha": commit_sha,
            },
        }
    )
    write_json_atomic(state_file, state)
    return manifest


def write_github_outputs(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Select the next run without changing state")
    prepare.add_argument("--state-file", type=Path, required=True)
    prepare.add_argument("--output-base", type=Path, required=True)
    prepare.add_argument(
        "--event-name",
        choices=("schedule", "workflow_dispatch"),
        required=True,
    )
    prepare.add_argument("--manual-map", choices=MAP_ROTATION, default="default")
    prepare.add_argument("--run-token", required=True)
    prepare.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"]) if os.getenv("GITHUB_OUTPUT") else None,
    )

    complete = subparsers.add_parser("complete", help="Validate outputs and advance state")
    complete.add_argument("--state-file", type=Path, required=True)
    complete.add_argument("--run-dir", type=Path, required=True)
    complete.add_argument("--seed", type=int, required=True)
    complete.add_argument("--map-name", choices=MAP_ROTATION, required=True)
    complete.add_argument("--objective", required=True)
    complete.add_argument("--run-id", required=True)
    complete.add_argument("--commit-sha", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        values = prepare_run(
            state=load_state(args.state_file),
            output_base=args.output_base,
            event_name=args.event_name,
            manual_map=args.manual_map,
            run_token=args.run_token,
        )
        if args.github_output is not None:
            write_github_outputs(args.github_output, values)
        print(json.dumps(values, indent=2))
        return 0

    manifest = finalize_success(
        state_file=args.state_file,
        run_dir=args.run_dir,
        seed=args.seed,
        map_name=args.map_name,
        objective=args.objective,
        run_id=args.run_id,
        commit_sha=args.commit_sha,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
