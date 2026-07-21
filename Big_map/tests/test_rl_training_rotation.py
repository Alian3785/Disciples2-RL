import json
from pathlib import Path

import pytest

from maps import available_maps
from tools.rl_training_rotation import DEFAULT_STATE
from tools.rl_training_rotation import MAP_OBJECTIVES
from tools.rl_training_rotation import MAP_ROTATION
from tools.rl_training_rotation import finalize_success
from tools.rl_training_rotation import load_state
from tools.rl_training_rotation import prepare_run


def test_workflow_rotates_all_maps_every_three_hours():
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "rl-training-rotation.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert 'cron: "17 */3 * * *"' in workflow
    assert "CYCLE_BRANCH: rl-training-all-maps-cycle" in workflow
    assert "CAMPAIGN_MAP: ${{ steps.prepare.outputs.campaign_map }}" in workflow
    assert '--map "$CAMPAIGN_MAP"' in workflow
    assert 'if [[ "$CYCLE_COMPLETE" != "true" ]]' in workflow
    for map_name in MAP_ROTATION:
        assert f"          - {map_name}" in workflow


def test_rotation_contains_every_registered_map_once():
    assert len(MAP_ROTATION) == 12
    assert len(set(MAP_ROTATION)) == len(MAP_ROTATION)
    assert set(MAP_ROTATION) == set(available_maps())


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


def test_scheduled_run_selects_next_map_for_one_million_steps(tmp_path):
    prepared = prepare_run(
        state=dict(DEFAULT_STATE),
        output_base=tmp_path,
        event_name="schedule",
        manual_map="trade_train",
        run_token="12-1",
    )

    assert prepared["should_run"] == "true"
    assert prepared["campaign_map"] == "default"
    assert prepared["objective"] == "green_dragon"
    assert prepared["objective_flag"] == "--dragon"
    assert prepared["seed"] == "101"
    assert prepared["total_steps"] == "1000000"
    assert prepared["cycle_complete"] == "false"


def test_manual_smoke_uses_selected_map_without_advancing_state(tmp_path):
    state = dict(DEFAULT_STATE)
    prepared = prepare_run(
        state=state,
        output_base=tmp_path,
        event_name="workflow_dispatch",
        manual_map="small",
        run_token="99-2",
    )

    assert prepared["should_run"] == "true"
    assert prepared["advance_state"] == "false"
    assert prepared["campaign_map"] == "small"
    assert prepared["objective"] == "green_dragon"
    assert prepared["objective_flag"] == "--dragon"
    assert prepared["total_steps"] == "24576"
    assert prepared["eval_freq"] == "2048"
    assert state == DEFAULT_STATE


def test_only_default_and_small_override_their_objective_to_green_dragon(tmp_path):
    for map_name in MAP_ROTATION:
        state = {**DEFAULT_STATE, "next_map": map_name}
        prepared = prepare_run(
            state=state,
            output_base=tmp_path,
            event_name="schedule",
            manual_map="default",
            run_token=f"run-{map_name}",
        )

        assert prepared["campaign_map"] == map_name
        assert prepared["objective"] == MAP_OBJECTIVES[map_name]
        assert prepared["objective_flag"] == (
            "--dragon" if map_name in {"default", "small"} else ""
        )


def test_successful_cycle_advances_all_twelve_maps_then_seed(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(state_file)

    for index, map_name in enumerate(MAP_ROTATION):
        run_dir = tmp_path / f"run-{index + 1}"
        _create_success_outputs(run_dir)
        finalize_success(
            state_file=state_file,
            run_dir=run_dir,
            seed=101,
            map_name=map_name,
            objective=MAP_OBJECTIVES[map_name],
            run_id=f"run-{index + 1}",
            commit_sha="abc123",
        )

        state = json.loads(state_file.read_text(encoding="utf-8"))
        expected_next_map = MAP_ROTATION[(index + 1) % len(MAP_ROTATION)]
        expected_seed = 102 if index == len(MAP_ROTATION) - 1 else 101
        assert state["seed"] == expected_seed
        assert state["next_map"] == expected_next_map
        assert state["completed_runs"] == index + 1
        assert state["last_success"]["map"] == map_name
        assert (run_dir / "eval" / "best_model.zip").is_file()
        assert (run_dir / "eval" / "best_vecnormalize.pkl").is_file()
        assert (run_dir / "manifest.json").is_file()


def test_legacy_objective_state_migrates_to_first_map(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {
            "active": True,
            "seed": 118,
            "next_objective": "cities",
            "completed_runs": 51,
            "last_success": None,
        },
    )

    state = load_state(state_file)

    assert state["next_map"] == "default"
    assert "next_objective" not in state
    assert state["seed"] == 118
    assert state["completed_runs"] == 51


def test_missing_output_does_not_change_state(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(state_file)
    original = state_file.read_text(encoding="utf-8")
    run_dir = tmp_path / "failed-run"
    _create_success_outputs(run_dir)
    (run_dir / "eval" / "evaluations.npz").unlink()

    with pytest.raises(FileNotFoundError, match="evaluations.npz"):
        finalize_success(
            state_file=state_file,
            run_dir=run_dir,
            seed=101,
            map_name="default",
            objective="green_dragon",
            run_id="failed-run",
            commit_sha="abc123",
        )

    assert state_file.read_text(encoding="utf-8") == original
