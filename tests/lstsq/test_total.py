# lstsq/test_total.py
"""Tests for lstsq._total.py."""

import pytest
import numpy as np
import scipy.linalg as la

import opinf

try:
    from .test_base import _TestSolverTemplate
except ImportError:
    from test_base import _TestSolverTemplate


class TestTotalLeastSquaresSolver(_TestSolverTemplate):
    """Test lstsq._total.TotalLeastSquaresSolver."""

    Solver = opinf.lstsq.TotalLeastSquaresSolver

    def get_solvers(self):
        """Yield solvers to test."""
        yield self.Solver()
        yield self.Solver(lapack_driver="gesvd")

    def test_fit_and_str(self, k=20, d=10, r=5):
        D = np.random.random((d, d))
        Z = np.random.random((r, d))
        with pytest.raises(ValueError) as ex:
            self.Solver().fit(D, Z)
        assert ex.value.args[0] == (
            "total least-squares system is underdetermined, "
            f"k > d + r is required (k = {d}, d = {d}, r = {r})"
        )
        return super().test_fit_and_str(k, d, r)

    def test_solve(self, k=15, d=7, r=5):
        """Lightly test solve()."""
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver().fit(D, Z)

        # One-dimensional accuracy test.
        z = Z[0]
        solver.fit(D, z)
        ohat_TLS = solver.solve()
        DtD = D.T @ D
        minsval = la.svdvals(np.column_stack((D, z))).min()
        A = la.solve(DtD - minsval**2 * np.eye(d), DtD, assume_a="sym")
        ohat_OLS = la.lstsq(D, z)[0]
        assert np.allclose(ohat_TLS, A @ ohat_OLS)

    def test_cond(self, k=20, d=11, r=3):
        """Test cond()."""
        for solver in self.get_solvers():
            # Try before calling fit().
            with pytest.raises(AttributeError) as ex:
                solver.cond()
            assert ex.value.args[0] == "solver not trained, call fit()"

            # Random test
            D = np.random.standard_normal((k, d))
            Z = np.random.standard_normal((r, k))
            svals = la.svdvals(D)
            solver.fit(D, Z)
            cond = solver.cond()
            assert np.isclose(cond, svals[0] / svals[-1])

            assert solver.augcond > solver.cond()
            assert solver.error > 0


if __name__ == "__main__":
    pytest.main([__file__])
