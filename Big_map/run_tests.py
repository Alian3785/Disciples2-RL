"""Run the regression suite with two workers; forward all pytest arguments."""
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    # Test runs must not write shared battle dumps or multiply BLAS threads in
    # every worker. The runner only sets these for its own process and children.
    os.environ["SAVE_UNITS_ARRAYS_FLAG"] = "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    import pytest
    return pytest.main([
        "-q", "--rootdir", str(root), "-o", "testpaths=tests",
        "-n", "2", "--dist=worksteal", *sys.argv[1:],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
