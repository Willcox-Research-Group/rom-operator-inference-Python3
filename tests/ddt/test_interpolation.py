# ddt/test_interpolation.py
"""Tests for ddt._interpolation.py"""

import pytest
import numpy as np
import scipy.interpolate as interp

import opinf


_module = opinf.ddt._interpolation


class TestInterpDerivativeEstimator:
    """Test opinf.ddt.InterpDerivativeEstimator."""

    Estimator = _module.InterpDerivativeEstimator

    def test_init(self, k=100):
        """Test __init__() and properties."""
        t = np.linspace(0, 1, k)

        with pytest.raises(ValueError) as ex:
            self.Estimator(t, 100)
        assert ex.value.args[0].startswith("invalid InterpolatorClass")

        with pytest.raises(TypeError) as ex:
            self.Estimator(t, new_time_domain=10)
        assert ex.value.args[0] == (
            "new_time_domain must be a one-dimensional array or None"
        )

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Estimator(t, new_time_domain=(2 * t))
        assert wn[0].message.args[0] == (
            "new_time_domain extrapolates beyond time_domain"
        )

        est = self.Estimator(t, "akima", t / 2, method="makima")
        assert est.InterpolatorClass is interp.Akima1DInterpolator
        assert "axis" in est.options
        assert est.options["axis"] == 1
        assert est.new_time_domain.shape == t.shape
        repr(est)

    def test_estimate(self, r=5, m=3, k=20):
        """Use verify() to test estimate()."""
        t = np.linspace(0, 1, k)
        t2 = (t[1:-2] / 1.5) + (1 / 6)
        Q = np.random.random((r, k))
        U = np.random.random((m, k))

        for name in self.Estimator._interpolators:
            est = self.Estimator(t, InterpolatorClass=name)
            errors = est.verify(plot=False, return_errors=True)
            for label, results in errors.items():
                if label == "dts":
                    continue
                assert (
                    np.min(results) < 5e-7
                ), f"problem with InterpolatorClass '{name}', test '{label}'"

            est = self.Estimator(t, InterpolatorClass=name, new_time_domain=t2)
            Q_, Qdot_, U_ = est.estimate(Q, U)
            assert Q_.shape == (r, t2.size) == Qdot_.shape
            assert U_.shape == (m, t2.size)

        # One-dimensional inputs.
        est.estimate(Q, U[0])


if __name__ == "__main__":
    pytest.main([__file__])
