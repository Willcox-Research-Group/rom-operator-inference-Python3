# lstsq/test_tsvd.py
"""Tests for lstsq._tsvd.py."""

import pytest
import numpy as np

import opinf


class TestTotalLeastSquaresSolver:
    """Test lstsq._tsvd.TruncatedSVDSolver."""

    Solver = opinf.lstsq.TruncatedSVDSolver

    def test_init(self):
        """Test __init__() and properties."""
        with pytest.raises(TypeError) as ex:
            self.Solver("ten")
        assert ex.value.args[0] == "num_svdmodes must be an integer"

        for n in (10, -3, None):
            solver = self.Solver(n)
            assert solver.data_matrix is None
            if n is None:
                assert solver.num_svdmodes is None
            else:
                assert solver.num_svdmodes == n
            assert solver.max_modes is None

        repr(solver)

    def test_fit(self, k=20, d=10, r=6):
        """Test fit()."""
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        nn = (n := min(D.shape)) - 2
        solver = self.Solver(nn)
        out = solver.fit(D, Z)
        assert out is solver
        assert solver.num_svdmodes == nn
        assert solver.max_modes == n
        repr(solver)
        solver.verify()

        with pytest.raises(ValueError) as ex:
            solver.num_svdmodes = n * 2
        assert ex.value.args[0] == f"only {n} SVD modes available"

        solver = self.Solver(None)
        solver.fit(D, Z)
        assert solver.num_svdmodes == n

        for nn in (0, -3):
            solver = self.Solver(nn)
            solver.fit(D, Z)
            assert solver.num_svdmodes == n + nn
            assert solver.max_modes == n

        assert solver.tcond() < solver.cond()

    def test_predict(self, k=15, d=7, r=5):
        """Test predict()."""
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver(min(D.shape) - 2).fit(D, Z)
        Ohat = solver.predict()
        assert Ohat.shape == (r, d)
