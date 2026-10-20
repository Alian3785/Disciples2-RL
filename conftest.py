"""Require the production CUDA backend before collecting any project tests."""
import os
import json
from pathlib import Path
import platform

import pytest


def pytest_runtest_logreport(report):
    """Persist completed phases even if the wall-time guard kills a later test."""
    output = os.environ.get('NUMBERGRID_TEST_DURATIONS')
    if output:
        with Path(output).open('a', encoding='utf-8') as stream:
            record = dict(nodeid=report.nodeid,phase=report.when,
                          seconds=report.duration,outcome=report.outcome)
            if report.failed:
                record['failure'] = report.longreprtext
            stream.write(json.dumps(record)+'\n')


def pytest_sessionstart(session):
    os.environ['NUMBERGRID_TEST_RUN'] = '1'
    requested = os.environ.get('JAX_PLATFORMS')
    if requested not in (None, 'cuda'):
        raise pytest.UsageError(
            'Project tests require JAX_PLATFORMS=cuda; CPU runs are not supported. '
            'Use bash scripts/run_gpu.sh -m pytest.')
    os.environ['JAX_PLATFORMS'] = 'cuda'

    import jax

    try:
        # Also reject an already-initialized non-GPU default backend.
        if jax.default_backend() != 'gpu' or not jax.devices('gpu'):
            raise RuntimeError('The default JAX backend is not a GPU')
    except RuntimeError as error:
        raise pytest.UsageError(
            'A working CUDA GPU is required for project tests; no CPU fallback. '
            'Use bash scripts/run_gpu.sh -m pytest.') from error
    output = os.environ.get('NUMBERGRID_TEST_DURATIONS')
    if output:
        metadata = dict(python=platform.python_version(),jax=jax.__version__,backend=jax.default_backend(),
                        devices=[d.device_kind for d in jax.devices('gpu')])
        Path(output).with_name('runtime.json').write_text(json.dumps(metadata,indent=2)+'\n')
