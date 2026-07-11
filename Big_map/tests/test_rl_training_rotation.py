import json
from pathlib import Path

import pytest

from tools.rl_training_rotation import DEFAULT_STATE
from tools.rl_training_rotation import finalize_success
from tools.rl_training_rotation import prepare_run


def _write_state(path: Path, state: dict | None = None) -> None:
    path.write_text(json.dumps(state or DEFAULT_STATE), encoding="utf-8")


def _create_success_outputs(run_dir: Path) -> None:
    checkpoints = run_dir / "checkpoints"
    best = run_dir / "models" / "best"
    eval_dir = run_dir / "eval"
    checkpoints.mkdir(parents=True)
    best.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    for name in (
        "campaign_ppo_step_500000.zip",
        "vecnormalize_step_500000.pkl",
        "campaign_ppo_step_1000000.zip",
        "vecnormalize_step_1000000.pkl",
    ):
        (checkpoints / name).write_bytes(b"checkpoint")
    (best / "best_model.zip").write_bytes(b"model")
    (best / "best_vecnormalize.pkl").write_bytes(b"vecnormalize")
    (eval_dir / "evaluations.npz").write_bytes(b"eval")


def test_inactive_rotation_waits_until_activation_slot(tmp_path):
    waiting = prepare_run(
        state=dict(DEFAULT_STATE),
        output_base=tmp_path,
        event_name="schedule",
        activation_slot=False,
        manual_objective="cities",
        run_token="12-1",
    )
    ready = prepare_run(
        state=dict(DEFAULT_STATE),
        output_base=tmp_path,
        event_name="schedule",
        activation_slot=True,
        manual_objective="cities",
        run_token="12-1",
    )

    assert waiting["should_run"] == "false"
    assert ready["should_run"] == "true"
    assert ready["objective"] == "cities"
    assert ready["seed"] == "101"
    assert ready["total_steps"] == "1000000"


def test_manual_smoke_does_not_advance_state(tmp_path):
    state = dict(DEFAULT_STATE)
    prepared = prepare_run(
        state=state,
        output_base=tmp_path,
        event_name="workflow_dispatch",
        activation_slot=False,
        manual_objective="blue_dragon",
        run_token="99-2",
    )

    assert prepared["should_run"] == "true"
    assert prepared["advance_state"] == "false"
    assert prepared["objective_flag"] == "--blue-dragon"
    assert prepared["total_steps"] == "24576"
    assert prepared["eval_freq"] == "2048"
    assert state == DEFAULT_STATE


def test_successful_triple_advances_objectives_then_seed(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(state_file)

    for index, (seed, objective, next_seed, next_objective) in enumerate(
        (
            (101, "cities", 101, "green_dragon"),
            (101, "green_dragon", 101, "blue_dragon"),
            (101, "blue_dragon", 102, "cities"),
        ),
        start=1,
    ):
        run_dir = tmp_path / f"run-{index}"
        _create_success_outputs(run_dir)
        finalize_success(
            state_file=state_file,
            run_dir=run_dir,
            seed=seed,
            objective=objective,
            run_id=f"run-{index}",
            commit_sha="abc123",
        )

        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["seed"] == next_seed
        assert state["next_objective"] == next_objective
        assert state["completed_runs"] == index
        assert (run_dir / "eval" / "best_model.zip").is_file()
        assert (run_dir / "eval" / "best_vecnormalize.pkl").is_file()
        assert (run_dir / "manifest.json").is_file()


def test_missing_output_does_not_change_state(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(state_file, {**DEFAULT_STATE, "active": True})
    original = state_file.read_text(encoding="utf-8")
    run_dir = tmp_path / "failed-run"
    _create_success_outputs(run_dir)
    (run_dir / "eval" / "evaluations.npz").unlink()

    with pytest.raises(FileNotFoundError, match="evaluations.npz"):
        finalize_success(
            state_file=state_file,
            run_dir=run_dir,
            seed=101,
            objective="cities",
            run_id="failed-run",
            commit_sha="abc123",
        )

    assert state_file.read_text(encoding="utf-8") == original
