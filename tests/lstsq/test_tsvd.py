# lstsq/test_tsvd.py
"""Tests for lstsq._tsvd.py."""

import pytest
import numpy as np
import scipy.linalg as la

import opinf

try:
    from .test_base import _TestSolverTemplate
except ImportError:
    from test_base import _TestSolverTemplate


class TestTotalLeastSquaresSolver(_TestSolverTemplate):
    """Test lstsq._tsvd.TruncatedSVDSolver."""

    Solver = opinf.lstsq.TruncatedSVDSolver

    def get_solvers(self):
        """Yield solvers to test."""
        yield self.Solver(None)
        yield self.Solver(3)

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

        solver.num_svdmodes = -4
        assert solver.num_svdmodes == n - 4

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

        solver = self.Solver(1).fit(D, Z)
        assert solver.tcond() == 1

        solver = self.Solver(nn := 2 * (k + d))
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            solver.fit(D, Z)
        assert wn[0].message.args[0] == (
            f"only {n} SVD modes available, "
            f"setting num_svdmodes={n} (was {nn})"
        )

        assert np.isclose(solver.tcond(), solver.cond())

    def test_solve(self, k=15, d=7, r=5):
        """Test solve()."""
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver(min(D.shape) - 2).fit(D, Z)

        Ohat_true = la.lstsq(D, Z.T)[0].T
        resid_true = la.norm(solver.residual(Ohat_true))
        Ohat = solver.solve()
        resid = la.norm(solver.residual(Ohat))
        assert resid > resid_true

        solver = self.Solver(None).fit(D, Z)
        Ohat = solver.solve()
        assert np.allclose(Ohat, Ohat_true)


if __name__ == "__main__":
    pytest.main([__file__])
