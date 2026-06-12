import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch as th

import train_campaign  # noqa: F401
from sb3_contrib.common.maskable.distributions import MaskableCategorical


def test_maskable_categorical_remasking_ignores_stale_cached_probs():
    distribution = MaskableCategorical(logits=th.zeros((2, 3), dtype=th.float32))
    distribution.probs = th.full((2, 3), 1e-12, dtype=th.float32)

    masks = th.tensor(
        [
            [True, False, False],
            [False, True, True],
        ],
        dtype=th.bool,
    )

    distribution.apply_masking(masks)

    assert th.allclose(distribution.probs.sum(dim=-1), th.ones(2))
    assert distribution.probs[0, 0] == 1.0
    assert distribution.probs[0, 1] == 0.0
    assert distribution.probs[0, 2] == 0.0
    assert distribution.probs[1, 0] == 0.0
