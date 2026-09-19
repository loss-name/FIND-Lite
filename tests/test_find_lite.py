import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from find_lite import find_lite


def sphere(x):
    return float(np.dot(x, x))


def test_reproducible_and_budget_exact():
    bounds = [(-5.0, 5.0)] * 3
    a = find_lite(sphere, bounds, 100, seed=7)
    b = find_lite(sphere, bounds, 100, seed=7)
    assert a["evaluations"] == 100
    assert a["best_f"] == b["best_f"]
    assert np.allclose(a["best_x"], b["best_x"])
    assert len(a["history"]) == 100


def test_budget_must_cover_initialization():
    with pytest.raises(ValueError, match="initialization"):
        find_lite(sphere, [(-1.0, 1.0)] * 2, 3)


def test_invalid_bounds_and_nonfinite_objective():
    with pytest.raises(ValueError):
        find_lite(sphere, [(1.0, -1.0)], 10)
    with pytest.raises(ValueError, match="non-finite"):
        find_lite(lambda x: float("nan"), [(-1.0, 1.0)], 10)


def test_result_is_feasible_and_history_is_monotone():
    bounds = [(-2.0, 2.0)] * 4
    result = find_lite(sphere, bounds, 80, seed=9)
    x = np.asarray(result["best_x"])
    assert np.all(x >= -2.0) and np.all(x <= 2.0)
    history = np.asarray(result["history"])[:, 1]
    assert np.all(np.diff(history) <= 0.0)
