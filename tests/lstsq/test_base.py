# lstsq/_base.py
"""Tests for lstsq._base.py."""

import pytest
import numpy as np
import scipy.linalg as la

import opinf


def test_lstsq_size():
    """Test lstsq.lstsq_size()."""
    m, r = 3, 7

    # Try with bad input combinations.
    with pytest.raises(ValueError) as ex:
        opinf.lstsq.lstsq_size("cAHB", r)
    assert ex.value.args[0] == "argument m > 0 required since 'B' in modelform"

    with pytest.raises(ValueError) as ex:
        opinf.lstsq.lstsq_size("cAH", r, m=10)
    assert ex.value.args[0] == "argument m=10 invalid since 'B' in modelform"

    # Test without inputs.
    assert opinf.lstsq.lstsq_size("c", r) == 1
    assert opinf.lstsq.lstsq_size("A", r) == r
    assert opinf.lstsq.lstsq_size("cA", r) == 1 + r
    assert opinf.lstsq.lstsq_size("cAH", r) == 1 + r + r*(r+1)//2
    assert opinf.lstsq.lstsq_size("cG", r) == 1 + r*(r+1)*(r+2)//6

    # Test with inputs.
    assert opinf.lstsq.lstsq_size("cB", r, m) == 1 + m
    assert opinf.lstsq.lstsq_size("AB", r, m) == r + m
    assert opinf.lstsq.lstsq_size("cAB", r, m) == 1 + r + m
    assert opinf.lstsq.lstsq_size("AHB", r, m) == r + r*(r+1)//2 + m
    assert opinf.lstsq.lstsq_size("GB", r, m) == r*(r+1)*(r+2)//6 + m

    # Test with affines.
    assert opinf.lstsq.lstsq_size("c", r, affines={"c": [0, 0]}) == 2
    assert opinf.lstsq.lstsq_size("A", r, affines={"A": [0, 0]}) == 2*r


class TestBaseSolver:
    """Test lstsq._base._BaseSolver."""

    class Dummy(opinf.lstsq._base._BaseSolver):
        """Instantiable version of _BaseSolver."""
        _LSTSQ_LABEL = "some OpInf problem"

        def predict(*args, **kwargs):
            pass

    def test_properties(self):
        """Test A, B, k, d, r, properties."""
        solver = self.Dummy()
        assert solver.A is None
        assert solver.B is None
        assert solver.k is None
        assert solver.d is None
        assert solver.r is None

        with pytest.raises(AttributeError) as ex:
            solver.A = 1
        assert ex.value.args[0] == "can't set attribute (call fit())"

    def test_fit(self, k=30, d=20, r=5):
        """Test fit()."""
        solver = self.Dummy()

        # Bad dimensions.
        A = np.random.random(k)
        B = np.random.random((k, r))
        with pytest.raises(ValueError) as ex:
            solver.fit(A, B)
        assert ex.value.args[0] == "A must be two-dimensional"

        A = np.random.random((k, d))
        B = np.random.random((k, r, d))
        with pytest.raises(ValueError) as ex:
            solver.fit(A, B)
        assert ex.value.args[0] == "B must be one- or two-dimensional"

        # Mismatched shapes.
        A = np.random.random((k, d))
        B = np.random.random((k - 1, r))
        with pytest.raises(ValueError) as ex:
            solver.fit(A, B)
        assert ex.value.args[0] == "A.shape[0] != B.shape[0]"

        # Correct usage, r > 1.
        A = np.random.random((k, d))
        B = np.random.random((k, r))
        assert solver.fit(A, B) is solver
        assert solver.A is A
        assert solver.B is B
        assert solver.k == k
        assert solver.d == d
        assert solver.r == r

        # Correct usage, r = 1.
        B = np.random.random(k)
        assert solver.fit(A, B) is solver
        assert solver.A is A
        assert solver.B.shape == (k, 1)
        assert solver.k == k
        assert solver.d == d
        assert solver.r == 1
        assert np.all(solver.B[:, 0] == B)

    def test_check_is_trained(self, k=10, d=4, r=3):
        """Test _check_is_trained()"""
        solver = self.Dummy()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver._check_is_trained()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.empty((k, d))
        B = np.empty((k, r))
        solver.fit(A, B)

        # Try after calling fit() but with a missing attribute.
        with pytest.raises(AttributeError) as ex:
            solver._check_is_trained("_nonexistentattribute")
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Correct usage.
        solver._check_is_trained()

    # String representations --------------------------------------------------
    def test_str(self, k=20, d=6, r=3):
        """Test __str__() and __repr__()."""
        # Before fitting.
        solver = self.Dummy()
        assert str(solver) == "Least-squares solver for some OpInf problem"

        rep = repr(solver)
        assert rep.startswith("<Dummy object at ")
        assert len(rep.split('\n')) == 2

        A = np.empty((k, d))
        B = np.empty((k, r))
        solver.fit(A, B)
        strlines = str(solver).split('\n')
        assert len(strlines) == 4
        assert strlines[0] == "Least-squares solver for some OpInf problem"
        assert strlines[1] == f"A: ({k:d}, {d:d})"
        assert strlines[2] == f"X: ({d:d}, {r:d})"
        assert strlines[3] == f"B: ({k:d}, {r:d})"

        replines = repr(solver).split('\n')
        assert len(replines) == 5
        assert replines[1:] == strlines

    # Post-processing ---------------------------------------------------------
    def test_cond(self, k=20, d=11, r=3):
        """Test cond()."""
        solver = self.Dummy()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.cond()
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        # Contrived test 1
        A = np.eye(d)
        B = np.zeros((d, r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), 1)

        # Contrived test 2
        A = np.diag(np.arange(1, d+1))
        B = np.zeros((d, r))
        solver.fit(A, B)
        assert np.isclose(solver.cond(), d)

        # Random test
        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        svals = la.svdvals(A)
        solver.fit(A, B)
        assert np.isclose(solver.cond(), svals[0] / svals[-1])

    def test_misfit(self, k=20, d=10, r=4):
        """Test misfit()."""
        solver = self.Dummy()

        # Try before calling fit().
        with pytest.raises(AttributeError) as ex:
            solver.misfit(0)
        assert ex.value.args[0] == "lstsq solver not trained (call fit())"

        A = np.random.standard_normal((k, d))
        B = np.random.standard_normal((k, r))
        solver.fit(A, B)

        # Try with badly shaped X.
        X = np.random.standard_normal((d+1, r-1))
        with pytest.raises(ValueError) as ex:
            solver.misfit(X)
        assert ex.value.args[0] == \
            f"X.shape = {(d+1, r-1)} != {(d, r)} = (d, r)"

        # Two-dimensional case.
        X = np.random.standard_normal((d, r))
        misfit = solver.misfit(X)
        assert isinstance(misfit, np.ndarray)
        assert misfit.shape == (r,)
        assert np.allclose(misfit, la.norm(A @ X - B, ord=2, axis=0)**2)

        # One-dimensional case.
        b = B[:, 0]
        solver.fit(A, b)
        assert solver.r == 1
        x = np.random.standard_normal(d)
        misfit = solver.misfit(x)
        assert isinstance(misfit, float)
        assert np.isclose(misfit, np.linalg.norm(A @ x - b)**2)


class TestPlainSolver:
    """Test lstsq._base.PlainSolver."""

    def test_predict(self, k=20, d=11, r=3):
        """Test predict()."""
        # Set up and manually solve a least-squares problem.
        A = np.random.standard_normal((k, d))
        B = np.random.random((k, r))
        U, s, Vt = la.svd(A, full_matrices=False)
        Xtrue = Vt.T @ np.diag(1 / s) @ U.T @ B

        # Check the least-squares solution.
        solver = opinf.lstsq.PlainSolver().fit(A, B)
        Xpred = solver.predict()
        assert np.allclose(Xpred, Xtrue)
