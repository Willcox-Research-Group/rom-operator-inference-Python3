# lstsq/test_total.py
"""Tests for lstsq._total.py."""

import numpy as np

import opinf


class TestTotalLeastSquaresSolver:
    """Test lstsq._base.TotalLeastSquaresSolver."""

    def test_predict_tls(self, k=15, d=7, r=5):
        """Test predict()."""
        A = np.random.standard_normal((k, d))
        B = np.random.random((k, r))
        # Check the least-squares solution.
        solver = opinf.lstsq.TotalLeastSquaresSolver().fit(A, B)
        Xpred = solver.predict()
        assert Xpred.shape == (d, r)
