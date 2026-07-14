import numpy as np
import pytest

from battle_env import BattleEnv
from data_dicts_compact_lines import DATA, map_unit_to_battle


def _entry(name: str) -> dict:
    return next(entry for entry in DATA if entry.get("кто") == name)


@pytest.mark.parametrize("name", ["Оживший доспех", "Голем", "Мизраэль"])
def test_max_exp_required_is_canonicalized_when_building_battle_unit(name: str):
    unit = map_unit_to_battle(_entry(name), "red", 1)

    assert unit["exp_required"] == 0


def test_observation_accepts_legacy_max_exp_required_value():
    env = BattleEnv(log_enabled=False)
    env.reset(seed=1)
    env.combined[0]["exp_required"] = "max"

    observation = env._obs()

    assert np.isfinite(observation).all()
    assert observation.shape == env.observation_space.shape
