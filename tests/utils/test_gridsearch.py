# utils/test_gridsearch.py
"""Test for utils._gridsearch."""

import pytest
import numpy as np

import opinf


gridsearch = opinf.utils.gridsearch


def test_gridsearch1D():
    """Test utils.gridsearch() for 1D problems."""

    def min1D(x):
        """One-dimensional function with minimizer x = 2."""
        xm3 = x - 3
        return 2.25 * xm3**4 + 0.5 * xm3**3 - 3.75 * xm3**2

    candidates = np.linspace(1.5, 4.5, 10)

    # Gridsearch only.
    xhat = gridsearch(min1D, candidates, gridsearch_only=True)
    assert isinstance(xhat, float)
    xloc = np.argmin(np.abs(candidates - xhat))
    assert xhat == candidates[xloc]

    # Gridsearch plus minimization.
    xhat = gridsearch(min1D, candidates, verbose=True)
    assert isinstance(xhat, float)
    assert np.isclose(xhat, 2)

    gridsearch(min1D, candidates.reshape((-1, 1)), verbose=False)

    # Limited candidates
    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        xhat = gridsearch(min1D, np.linspace(2, 4, 10), label="x")
    assert len(wn) == 2
    assert wn[0].message.args[0] == (
        "smallest x candidate won grid search, "
        "consider using smaller candidates"
    )
    assert wn[1].message.args[0] == (
        "x grid search performed better than optimization, "
        "falling back on grid search solution"
    )
    assert np.isclose(xhat, 2)

    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        xhat = gridsearch(
            min1D, np.linspace(0.5, 2, 10), label="y", verbose=True
        )
    assert len(wn) == 2
    assert wn[0].message.args[0] == (
        "largest y candidate won grid search, consider using larger candidates"
    )
    assert wn[1].message.args[0] == (
        "y grid search performed better than optimization, "
        "falling back on grid search solution"
    )
    assert np.isclose(xhat, 2)

    # Failed grid search
    with pytest.raises(RuntimeError) as ex:
        gridsearch(lambda x: np.inf, candidates, verbose=True, label="w")
    assert ex.value.args[0] == "w grid search failed"

    # Failed optimization
    with pytest.warns(opinf.errors.OpInfWarning) as wn:
        gridsearch(min1D, [1e8], label="z")
    assert len(wn) == 1
    assert wn[0].message.args[0] == (
        "z grid search performed better than optimization, "
        "falling back on grid search solution"
    )


def test_gridsearchND(dim=4):
    """Test utils.gridsearch() for greater-than-one-dimensional problems."""
    np.random.seed(dim)

    def rosen(x):
        """The Rosenbrock function, minimizer at x = [1, 1, ..., 1]."""
        return np.sum(
            100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0
        )

    candidates = np.concatenate(
        (10 ** np.random.uniform(-5, 2, size=(20, dim)), [[1.01] * dim])
    )

    # Gridsearch only.
    xhat = gridsearch(rosen, candidates, gridsearch_only=True)
    assert isinstance(xhat, np.ndarray)
    assert xhat.shape == (dim,)
    xloc = np.argmin(np.linalg.norm(candidates - xhat, axis=1))
    assert np.allclose(xhat, candidates[xloc])

    # Gridsearch plus minimization.
    xhat = gridsearch(rosen, candidates, verbose=True)
    assert isinstance(xhat, np.ndarray)
    assert xhat.shape == (dim,)
    assert np.allclose(xhat, 1, atol=3e-2)


if __name__ == "__main__":
    pytest.main([__file__])
