"""Regression coverage for the CLI summary; no learning or model creation."""

import ast
import contextlib
import io
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import train_campaign
from train_campaign import CampaignMetricsCallback, EvalStepCapWrapper


def _render_final_summary(metrics):
    """Run the actual CLI summary statements without running the training main."""
    path = Path(train_campaign.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    )
    start = next(
        index for index, node in enumerate(main.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "print"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "ИТОГИ ОБУЧЕНИЯ:"
    )
    end = next(
        index for index in range(start + 1, len(main.body))
        if isinstance(main.body[index], ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "summoned_episode_plot_path"
            for target in main.body[index].targets
        )
    )
    summary = ast.Module(body=main.body[start:end], type_ignores=[])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(summary, str(path), "exec"), {"metrics_cb": metrics})
    return output.getvalue()


def _assert_winrate(metrics, expected):
    summary = _render_final_summary(metrics)
    lines = [line.strip() for line in summary.splitlines() if "Winrate:" in line]
    assert lines == [f"Winrate: {expected:.1f}%"]
    # The console must agree with the pre-existing all-outcome metric.
    payload = metrics._build_log_payload()
    assert f"{100 * payload['campaign/victory_rate']:.1f}" == f"{expected:.1f}"


def _record_results(results):
    metrics = CampaignMetricsCallback()
    for result in results:
        metrics._consume_info({"campaign_result": result, "episode": {"r": 1, "l": 1}})
    assert metrics.result_counter == Counter(results)
    return metrics


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        (["victory", "eval_timeout"], 50.0),
        (["eval_timeout", "victory"], 50.0),
        (["eval_timeout"] * 3, 0.0),
        (["timeout"] * 3, 0.0),
        (["victory", "timeout"], 50.0),
        (["victory", "defeat"], 50.0),
        (["victory", "victory", "defeat", "timeout", "eval_timeout", "eval_timeout"], 33.3),
        (["victory"] * 3, 100.0),
        (["defeat"] * 3, 0.0),
        (["victory", "future_nonwinning_result"], 50.0),
    ],
)
def test_final_winrate_includes_every_recorded_campaign_result(results, expected):
    _assert_winrate(_record_results(results), expected)


def test_no_completed_episodes_keeps_summary_safe_and_omits_winrate():
    metrics = CampaignMetricsCallback()
    assert "Winrate:" not in _render_final_summary(metrics)
    assert metrics._build_log_payload()["campaign/victory_rate"] == 0.0


def test_ongoing_steps_and_nonterminal_battles_do_not_enter_denominator():
    metrics = _record_results(["victory", "eval_timeout"])
    for info in ({}, {"battle_result": "victory"}, {"battle_result": "timeout"}):
        metrics._consume_info(info)
    assert sum(metrics.result_counter.values()) == 2
    _assert_winrate(metrics, 50.0)


class _ControlledEnv(gym.Env):
    """Minimal deterministic steps behind the real cap/Monitor/VecEnv chain."""

    def __init__(self, steps):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self.steps = iter(steps)
        self._map = SimpleNamespace(timeout_reward=None)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        terminated, truncated, result = next(self.steps)
        info = {} if result is None else {"campaign_result": result}
        return np.zeros(1, dtype=np.float32), 1.0, terminated, truncated, info


def _vec_factory(steps, cap):
    return lambda: Monitor(EvalStepCapWrapper(_ControlledEnv(steps), max_steps=cap))


def _consume_vector_step(vec, metrics):
    _, _, dones, infos = vec.step(np.zeros(vec.num_envs, dtype=int))
    metrics.locals = {"infos": infos, "dones": dones}
    assert metrics._on_step()
    return dones, infos


def test_real_step_cap_and_autoreset_count_one_timeout_and_one_victory():
    vec = DummyVecEnv([_vec_factory(
        [(False, False, None), (False, False, None), (True, False, "victory")], 2,
    )])
    metrics = CampaignMetricsCallback()
    try:
        vec.reset()
        dones, _ = _consume_vector_step(vec, metrics)
        assert not dones[0]
        assert not metrics.result_counter
        dones, infos = _consume_vector_step(vec, metrics)
        assert dones[0] and infos[0]["eval_step_cap_hit"]
        assert infos[0]["episode"]["l"] == 2
        assert infos[0]["campaign_result"] == "eval_timeout"
        _consume_vector_step(vec, metrics)
        assert metrics.result_counter == {"eval_timeout": 1, "victory": 1}
        assert len(metrics.episode_lengths) == 2
        _assert_winrate(metrics, 50.0)
    finally:
        vec.close()


@pytest.mark.parametrize(
    ("terminated", "truncated", "result", "expected"),
    [(True, False, "victory", 100.0), (True, True, "victory", 100.0),
     (False, True, "timeout", 0.0), (True, False, "defeat", 0.0)],
)
def test_existing_end_at_cap_boundary_is_counted_once(terminated, truncated, result, expected):
    vec = DummyVecEnv([_vec_factory([(terminated, truncated, result)], 1)])
    metrics = CampaignMetricsCallback()
    try:
        vec.reset()
        dones, infos = _consume_vector_step(vec, metrics)
        assert dones[0]
        assert "eval_step_cap_hit" not in infos[0]
        assert metrics.result_counter == {result: 1}
        assert len(metrics.episode_lengths) == 1
        _assert_winrate(metrics, expected)
    finally:
        vec.close()


def test_parallel_completed_episodes_count_once_and_ongoing_env_is_ignored():
    vec = DummyVecEnv([
        _vec_factory([(True, False, "victory")], 2),
        _vec_factory([(False, False, None)], 1),
        _vec_factory([(False, False, None)], 2),
        _vec_factory([(True, False, "defeat")], 2),
    ])
    metrics = CampaignMetricsCallback()
    try:
        vec.reset()
        dones, _ = _consume_vector_step(vec, metrics)
        assert dones.tolist() == [True, True, False, True]
        assert metrics.result_counter == {"victory": 1, "eval_timeout": 1, "defeat": 1}
        assert len(metrics.episode_lengths) == 3
        _assert_winrate(metrics, 33.3)
        metrics._reset_log_window()
        _assert_winrate(metrics, 33.3)
    finally:
        vec.close()
