import json
import re
from pathlib import Path

import pytest

from maps import available_maps
from tools.rl_training_rotation import DEFAULT_STATE
from tools.rl_training_rotation import MAP_ROTATION
from tools.rl_training_rotation import MODE_ROTATION
from tools.rl_training_rotation import finalize_success
from tools.rl_training_rotation import load_state
from tools.rl_training_rotation import prepare_run


def test_workflow_rotates_selected_maps_every_three_hours():
    workflow_path = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "rl-training-rotation.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert 'cron: "17 */3 * * *"' in workflow
    assert "CYCLE_BRANCH: rl-training-7-maps-1500k-cycle" in workflow
    assert "CAMPAIGN_MAP: ${{ steps.prepare.outputs.campaign_map }}" in workflow
    assert '--map "$CAMPAIGN_MAP"' in workflow
    assert 'if [[ "$CYCLE_COMPLETE" != "true" ]]' in workflow
    assert tuple(re.findall(r"^          - (\w+)$", workflow, re.MULTILINE)) == MAP_ROTATION
    assert "default: orc_duel" in workflow
    assert "inputs.map_name || 'orc_duel'" in workflow
    assert "1,500,000-step" in workflow


def test_rotation_contains_only_selected_registered_maps_once():
    assert len(MAP_ROTATION) == 7
    assert len(set(MAP_ROTATION)) == len(MAP_ROTATION)
    assert set(MAP_ROTATION) <= set(available_maps())


def test_mode_rotation_contains_selected_seven_map_objective_pairs_once():
    assert len(MODE_ROTATION) == 7
    assert len(set(MODE_ROTATION)) == len(MODE_ROTATION)
    assert {map_name for map_name, _objective in MODE_ROTATION} == set(MAP_ROTATION)
    assert MODE_ROTATION == (
        ("orc_duel", "orc"),
        ("builder", "build_all"),
        ("formation_train", "all_enemies"),
        ("hire_train", "orc"),
        ("item_train", "all_enemies"),
        ("magic_train", "all_enemies"),
        ("scroll_train", "all_enemies"),
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
        "campaign_ppo_step_1500000.zip",
        "vecnormalize_step_1500000.pkl",
    ):
        (checkpoints / name).write_bytes(b"checkpoint")
    (best / "best_model.zip").write_bytes(b"model")
    (best / "best_vecnormalize.pkl").write_bytes(b"vecnormalize")
    (eval_dir / "evaluations.npz").write_bytes(b"eval")


def test_scheduled_run_selects_next_map_for_one_and_a_half_million_steps(tmp_path):
    prepared = prepare_run(
        state=dict(DEFAULT_STATE),
        output_base=tmp_path,
        event_name="schedule",
        manual_map="builder",
        run_token="12-1",
    )

    assert prepared["should_run"] == "true"
    assert prepared["campaign_map"] == "orc_duel"
    assert prepared["objective"] == "orc"
    assert prepared["objective_flag"] == ""
    assert prepared["seed"] == "101"
    assert prepared["total_steps"] == "1500000"
    assert prepared["cycle_complete"] == "false"


def test_manual_smoke_uses_selected_map_without_advancing_state(tmp_path):
    state = dict(DEFAULT_STATE)
    prepared = prepare_run(
        state=state,
        output_base=tmp_path,
        event_name="workflow_dispatch",
        manual_map="magic_train",
        run_token="99-2",
    )

    assert prepared["should_run"] == "true"
    assert prepared["advance_state"] == "false"
    assert prepared["campaign_map"] == "magic_train"
    assert prepared["objective"] == "all_enemies"
    assert prepared["objective_flag"] == ""
    assert prepared["total_steps"] == "24576"
    assert prepared["eval_freq"] == "2048"
    assert state == DEFAULT_STATE


def test_scheduled_rotation_uses_all_seven_modes(tmp_path):
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
            manual_map="orc_duel",
            run_token=f"run-{map_name}",
        )

        assert prepared["campaign_map"] == map_name
        assert prepared["objective"] == objective
        assert prepared["objective_flag"] == ""
        assert prepared["total_steps"] == "1500000"
        assert prepared["checkpoint_freq"] == "500000"
        assert prepared["seed"] == "101"
        assert prepared["cycle_complete"] == str(
            (map_name, objective) == MODE_ROTATION[-1]
        ).lower()


def test_successful_cycle_advances_all_seven_modes_then_seed(tmp_path):
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
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["parameters"]["total_steps"] == 1_500_000
        assert manifest["parameters"]["checkpoint_freq"] == 500_000


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

    assert state["next_map"] == "orc_duel"
    assert state["next_objective"] == "orc"
    assert state["seed"] == 118
    assert state["completed_runs"] == 51


def test_previous_physical_map_state_migrates_to_its_first_mode(tmp_path):
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {
            "active": True,
            "seed": 119,
            "next_map": "builder",
            "completed_runs": 54,
            "last_success": None,
        },
    )

    state = load_state(state_file)

    assert state["seed"] == 119
    assert state["next_map"] == "builder"
    assert state["next_objective"] == "build_all"


@pytest.mark.parametrize("map_name", [
    "default", "small", "green_dragon_minimal", "super_last_stand",
    "siege_train", "trade_train", "wotans_retribution",
])
def test_retired_map_state_migrates_without_losing_progress(tmp_path, map_name):
    state_file = tmp_path / "state.json"
    previous = {
        **DEFAULT_STATE,
        "seed": 138,
        "next_map": map_name,
        "next_objective": "cities",
        "completed_runs": 443,
        "last_success": {"run_id": "previous-run"},
    }
    _write_state(state_file, previous)

    state = load_state(state_file)

    assert state == {**previous, "next_map": "orc_duel", "next_objective": "orc"}
    assert json.loads(state_file.read_text(encoding="utf-8")) == previous
    with pytest.raises(ValueError, match="unknown map"):
        prepare_run(
            state=state,
            output_base=tmp_path,
            event_name="workflow_dispatch",
            manual_map=map_name,
            run_token="retired-map",
        )


@pytest.mark.parametrize("invalid", [
    {"next_map": "unknown_map"},
    {"next_map": "builder", "next_objective": "cities"},
])
def test_invalid_rotation_state_is_rejected(tmp_path, invalid):
    state_file = tmp_path / "state.json"
    _write_state(state_file, {**DEFAULT_STATE, **invalid})
    with pytest.raises(ValueError):
        load_state(state_file)


@pytest.mark.parametrize("missing", [
    "eval/evaluations.npz",
    "checkpoints/campaign_ppo_step_1500000.zip",
    "checkpoints/vecnormalize_step_1500000.pkl",
])
def test_missing_output_does_not_change_state(tmp_path, missing):
    state_file = tmp_path / "state.json"
    _write_state(state_file)
    original = state_file.read_text(encoding="utf-8")
    run_dir = tmp_path / "failed-run"
    _create_success_outputs(run_dir)
    (run_dir / missing).unlink()

    with pytest.raises(FileNotFoundError, match=re.escape(Path(missing).name)):
        finalize_success(
            state_file=state_file,
            run_dir=run_dir,
            seed=101,
            map_name="orc_duel",
            objective="orc",
            run_id="failed-run",
            commit_sha="abc123",
        )

    assert state_file.read_text(encoding="utf-8") == original
