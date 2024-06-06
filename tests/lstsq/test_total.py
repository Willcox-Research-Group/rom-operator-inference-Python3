# lstsq/test_total.py
"""Tests for lstsq._total.py."""

import os
import numpy as np
import scipy.linalg as la

import opinf


class TestTotalLeastSquaresSolver:
    """Test lstsq._total.TotalLeastSquaresSolver."""

    Solver = opinf.lstsq.TotalLeastSquaresSolver

    def test_fit(self, k=20, d=10, r=6):
        """Test fit()."""
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver()
        out = solver.fit(D, Z)
        assert out is solver

    def test_predict(self, k=15, d=7, r=5):
        """Test predict()."""
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver().fit(D, Z)
        Ohat = solver.predict()
        assert Ohat.shape == (r, d)

        # One-dimensional accuracy test.
        z = Z[0]
        solver.fit(D, z)
        ohat_TLS = solver.predict()
        DtD = D.T @ D
        minsval = la.svdvals(np.column_stack((D, z))).min()
        A = la.solve(DtD - minsval**2 * np.eye(d), DtD, assume_a="sym")
        ohat_OLS = la.lstsq(D, z)[0]
        assert np.allclose(ohat_TLS, A @ ohat_OLS)

    # Persistence -------------------------------------------------------------
    def test_save(self, k=18, d=10, r=4, outfile="_tlssavetest.h5"):
        """Lightly test save()."""
        if os.path.isfile(outfile):  # pragma: no cover
            os.remove(outfile)

        solver = self.Solver(lapack_driver="gesvd")
        solver.save(outfile)
        assert os.path.isfile(outfile)

        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        solver.fit(D, Z)
        solver.save(outfile, overwrite=True)
        assert os.path.isfile(outfile)

        os.remove(outfile)

    def test_load(self, k=20, d=10, r=8, outfile="_l2solverloadtest.h5"):
        """Test load() and verify it is the inverse of save()."""
        if os.path.isfile(outfile):  # pragma: no cover
            os.remove(outfile)

        solver = self.Solver(lapack_driver="gesvd")
        solver.save(outfile)
        solver2 = self.Solver.load(outfile)
        assert solver2.__class__ is self.Solver
        assert solver2.data_matrix is None
        assert solver2.options["lapack_driver"] == "gesvd"

        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver().fit(D, Z)
        solver.save(outfile, overwrite=True)
        solver2 = self.Solver.load(outfile)
        assert solver2.__class__ is self.Solver
        assert solver2.r == r
        assert solver2.k == k
        assert solver2.d == d
        assert np.all(solver2.data_matrix == D)
        assert np.all(solver2.lhs_matrix == Z)
        assert np.all(solver2.predict() == solver.predict())

        os.remove(outfile)

    def test_copy(self, k=16, d=6, r=3):
        """Test copy()."""
        solver = self.Solver(lapack_driver="gesvd")
        solver2 = solver.copy()
        assert solver2.__class__ is self.Solver
        assert solver2.data_matrix is None
        assert solver2.options["lapack_driver"] == "gesvd"

        D = np.random.random((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver().fit(D, Z)
        solver2 = solver.copy()
        assert solver2.__class__ is self.Solver
        assert solver2.r == r
        assert solver2.k == k
        assert solver2.d == d
        assert np.all(solver2.data_matrix == D)
        assert np.all(solver2.lhs_matrix == Z)
        assert np.all(solver2.predict() == solver.predict())
