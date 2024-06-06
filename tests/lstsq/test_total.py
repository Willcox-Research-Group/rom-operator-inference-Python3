# lstsq/test_total.py
"""Tests for lstsq._total.py."""

import numpy as np

import opinf


class TestTotalLeastSquaresSolver:
    """Test lstsq._total.TotalLeastSquaresSolver."""

    Solver = opinf.lstsq.TotalLeastSquaresSolver

    def test_predict(self, k=15, d=7, r=5):
        """Test predict()."""
        D = np.random.standard_normal((k, d))
        Z = np.random.random((r, k))
        solver = self.Solver().fit(D, Z)
        Ohat = solver.predict()
        assert Ohat.shape == (r, d)
