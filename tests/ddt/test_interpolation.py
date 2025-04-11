# ddt/test_interpolation.py
"""Tests for ddt._interpolation.py"""

import pytest
import numpy as np
import scipy.interpolate as interp

import opinf

try:
    from .test_base import _TestDerivativeEstimatorTemplate
except ImportError:
    from test_base import _TestDerivativeEstimatorTemplate


_module = opinf.ddt._interpolation


class TestInterpDerivativeEstimator(_TestDerivativeEstimatorTemplate):
    """Test opinf.ddt.InterpDerivativeEstimator."""

    Estimator = _module.InterpDerivativeEstimator

    def get_estimators(self):
        t = np.linspace(0, 1, 100)
        t2 = np.linspace(0.1, 0.9, 40)

        for name in self.Estimator._interpolators:
            yield self.Estimator(t, InterpolatorClass=name)
            yield self.Estimator(t, InterpolatorClass=name, new_time_domain=t2)

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

        return super().test_init()


if __name__ == "__main__":
    pytest.main([__file__])
