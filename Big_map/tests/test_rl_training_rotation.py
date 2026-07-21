import json
from pathlib import Path

import pytest

from maps import available_maps
from tools.rl_training_rotation import DEFAULT_STATE
from tools.rl_training_rotation import MAP_ROTATION
from tools.rl_training_rotation import MODE_ROTATION
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
    assert "CYCLE_BRANCH: rl-training-20-modes-cycle" in workflow
    assert "CAMPAIGN_MAP: ${{ steps.prepare.outputs.campaign_map }}" in workflow
    assert '--map "$CAMPAIGN_MAP"' in workflow
    assert 'if [[ "$CYCLE_COMPLETE" != "true" ]]' in workflow
    for map_name in MAP_ROTATION:
        assert f"          - {map_name}" in workflow


def test_rotation_contains_every_registered_map_once():
    assert len(MAP_ROTATION) == 14
    assert len(set(MAP_ROTATION)) == len(MAP_ROTATION)
    assert set(MAP_ROTATION) == set(available_maps())


def test_mode_rotation_contains_all_twenty_map_objective_pairs_once():
    assert len(MODE_ROTATION) == 20
    assert len(set(MODE_ROTATION)) == len(MODE_ROTATION)
    assert {map_name for map_name, _objective in MODE_ROTATION} == set(MAP_ROTATION)
    assert MODE_ROTATION == (
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
    assert prepared["objective"] == "cities"
    assert prepared["objective_flag"] == ""
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


def test_scheduled_rotation_uses_all_twenty_modes_and_objective_flags(tmp_path):
    for map_name, objective in MODE_ROTATION:
        state = {
            **DEFAULT_STATE,
            "next_map": map_name,
            "next_objective": objective,
        }
        prepared = prepare_run(
            state=state,
            output_base=tmp_path,
            event_name="schedule",
            manual_map="default",
            run_token=f"run-{map_name}",
        )

        assert prepared["campaign_map"] == map_name
        assert prepared["objective"] == objective
        expected_flag = {
            "green_dragon": "--dragon",
            "blue_dragon": "--blue-dragon",
            "full_party": "--full-party",
        }.get(objective, "")
        assert prepared["objective_flag"] == expected_flag
        assert prepared["seed"] == "101"
        assert prepared["cycle_complete"] == str(
            (map_name, objective) == MODE_ROTATION[-1]
        ).lower()


def test_successful_cycle_advances_all_twenty_modes_then_seed(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(state_file)

    for index, (map_name, objective) in enumerate(MODE_ROTATION):
        run_dir = tmp_path / f"run-{index + 1}"
        _create_success_outputs(run_dir)
        finalize_success(
            state_file=state_file,
            run_dir=run_dir,
            seed=101,
            map_name=map_name,
            objective=objective,
            run_id=f"run-{index + 1}",
            commit_sha="abc123",
        )

        state = json.loads(state_file.read_text(encoding="utf-8"))
        expected_next_map, expected_next_objective = MODE_ROTATION[
            (index + 1) % len(MODE_ROTATION)
        ]
        expected_seed = 102 if index == len(MODE_ROTATION) - 1 else 101
        assert state["seed"] == expected_seed
        assert state["next_map"] == expected_next_map
        assert state["next_objective"] == expected_next_objective
        assert state["completed_runs"] == index + 1
        assert state["last_success"]["map"] == map_name
        assert state["last_success"]["objective"] == objective
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
    assert state["next_objective"] == "cities"
    assert state["seed"] == 118
    assert state["completed_runs"] == 51


def test_previous_physical_map_state_migrates_to_its_first_mode(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {
            "active": True,
            "seed": 119,
            "next_map": "small",
            "completed_runs": 54,
            "last_success": None,
        },
    )

    state = load_state(state_file)

    assert state["seed"] == 119
    assert state["next_map"] == "small"
    assert state["next_objective"] == "cities"


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
            objective="cities",
            run_id="failed-run",
            commit_sha="abc123",
        )

    assert state_file.read_text(encoding="utf-8") == original
