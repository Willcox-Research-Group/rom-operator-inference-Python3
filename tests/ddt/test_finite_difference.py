# ddt/test_finite_difference.py
"""Tests for ddt._finite_difference.py"""

import pytest
import warnings
import numpy as np

import opinf

try:
    from .test_base import _TestDerivativeEstimatorTemplate
except ImportError:
    from test_base import _TestDerivativeEstimatorTemplate


_module = opinf.ddt._finite_difference


# New API =====================================================================
def test_finite_difference(r=10, k=20, m=3):
    """Test ddt._finite_difference()."""
    Q = np.random.random((r, k))
    U = np.random.random((m, k))

    with pytest.raises(ValueError) as ex:
        _module._finite_difference(Q, [0, 1, 2], mode="best")
    assert ex.value.args[0] == "invalid finite difference mode 'best'"

    Q_, dQdt = _module._finite_difference(Q, [1, 1, 1], "bwd", inputs=None)
    assert Q_.shape == (r, k - 2)
    assert dQdt.shape == (r, k - 2)

    Q_, dQdt, U_ = _module._finite_difference(Q, [1, 1, 2, 3], "fwd", inputs=U)
    assert Q_.shape == (r, k - 3)
    assert dQdt.shape == (r, k - 3)
    assert U_.shape == (m, k - 3)

    U1d = U[0, :]
    _, _, U1d_ = _module._finite_difference(Q, [1, 1, 2, 3], "fwd", inputs=U1d)
    assert U1d_.shape == (k - 3,)


class TestUniformFiniteDifferencer(_TestDerivativeEstimatorTemplate):
    """Test ddt.UniformFiniteDifferencer."""

    Estimator = _module.UniformFiniteDifferencer

    def get_estimators(self):
        t = np.linspace(0, 1, 100)
        for name in self.Estimator._schemes.keys():
            yield self.Estimator(t, scheme=name)

    def test_init(self, k=100):
        """Test __init__(), time_domain, scheme, __str__(), and __repr__()."""
        t = np.linspace(0, 1, k)
        differ = self.Estimator(t)
        assert differ.time_domain is t
        assert differ.dt == (t[1] - t[0])

        # Try with non-uniform spacing.
        t2 = t**2
        with pytest.raises(ValueError) as ex:
            self.Estimator(t2)
        assert ex.value.args[0] == "time domain must be uniformly spaced"

        # Bad scheme.
        with pytest.raises(ValueError) as ex:
            self.Estimator(t, scheme="best")
        assert ex.value.args[0] == "invalid finite difference scheme 'best'"

        # Valid schemes.
        for name, func in self.Estimator._schemes.items():
            assert callable(func)
            differ = self.Estimator(t, scheme=name)
            assert differ.scheme is func

        return super().test_init(k)

    def test_estimate(self):
        super().test_estimate(check_against_time=False)


class TestNonuniformFiniteDifferencer(_TestDerivativeEstimatorTemplate):
    """Test ddt.NonuniformFiniteDifferencer."""

    Estimator = _module.NonuniformFiniteDifferencer

    def get_estimators(self):
        t = np.linspace(0, 1, 100) ** 2
        yield self.Estimator(t)

    def test_init(self, k=100):
        """Test __init__(), time_domain, and order."""
        t = np.linspace(0, 1, k)

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Estimator(t)
        assert wn[0].message.args[0] == (
            "time_domain is uniformly spaced, consider using "
            "UniformFiniteDifferencer"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", opinf.errors.OpInfWarning)
            return super().test_init()


# Old API =====================================================================
def test_ddt_uniform(r=10, k=20):
    """Coverage for ddt.ddt_uniform() errors. Tested more thoroughly in
    TestUniformFiniteDifferencer.test_estimate() with schemes "ord2", "ord4",
    and "ord6".
    """
    dt = 1e-4
    Y = np.random.random((r, k))

    # Try with bad data shape.
    with pytest.raises(opinf.errors.DimensionalityError) as ex:
        _module.ddt_uniform(Y[:, 0], dt, order=2)
    assert ex.value.args[0] == "states must be two-dimensional"

    # Try with bad dt type.
    with pytest.raises(TypeError) as ex:
        _module.ddt_uniform(Y, np.array([dt, 2 * dt]), order=4)
    assert ex.value.args[0] == "time step dt must be a scalar (e.g., float)"

    # Try with bad order.
    with pytest.raises(NotImplementedError) as ex:
        _module.ddt_uniform(Y, dt, order=-1)
    assert ex.value.args[0] == "invalid order '-1'; valid options: {2, 4, 6}"


def test_ddt_nonuniform(r=11, k=21):
    """Coverage for ddt.ddt_nonuniform(). Tested more thoroughly in
    TestNonuniformFiniteDifferencer.test_estimate().
    """
    t = np.sort(np.random.random(k))
    Y = np.random.random((r, k))

    # Try with bad data shape.
    with pytest.raises(opinf.errors.DimensionalityError) as ex:
        _module.ddt_nonuniform(Y[:, 0], t)
    assert ex.value.args[0] == "states must be two-dimensional"

    # Try with bad time shape.
    with pytest.raises(opinf.errors.DimensionalityError) as ex:
        _module.ddt_nonuniform(Y, np.dstack((t, t)))
    assert ex.value.args[0] == "time t must be one-dimensional"

    with pytest.raises(opinf.errors.DimensionalityError) as ex:
        _module.ddt_nonuniform(Y, np.hstack((t, t)))
    assert ex.value.args[0] == "states not aligned with time t"


def test_ddt(r=12, k=22):
    """Coverage for ddt.ddt(). More thorough tests in the test_estimate()
    methods of the Test[...]FiniteDifferencer classes.
    """
    t = np.linspace(0, 1, k)
    dt = t[1] - t[0]
    t2 = t**2
    Y = np.random.random((r, k))

    def _single_test(*args, **kwargs):
        dY_ = _module.ddt(*args, **kwargs)
        assert dY_.shape == Y.shape

    _single_test(Y, dt)
    _single_test(Y, dt=dt)
    for o in [2, 4, 6]:
        _single_test(Y, dt, o)
        _single_test(Y, dt, order=o)
        _single_test(Y, dt=dt, order=o)
        _single_test(Y, order=o, dt=dt)
        _single_test(Y, t)
        _single_test(Y, t=t2)

    # Try with bad arguments.
    with pytest.raises(TypeError) as ex:
        _module.ddt(Y)
    assert ex.value.args[0] == (
        "at least one other argument required (dt or t)"
    )

    with pytest.raises(TypeError) as ex:
        _module.ddt(Y, order=2)
    assert ex.value.args[0] == (
        "keyword argument 'order' requires float argument dt"
    )

    with pytest.raises(TypeError) as ex:
        _module.ddt(Y, other=2)
    assert ex.value.args[0] == "ddt() got unexpected keyword argument 'other'"

    with pytest.raises(TypeError) as ex:
        _module.ddt(Y, 2)
    assert ex.value.args[0] == "invalid argument type '<class 'int'>'"

    with pytest.raises(TypeError) as ex:
        _module.ddt(Y, dt, 4, None)
    assert ex.value.args[0] == (
        "ddt() takes 2 or 3 positional arguments but 4 were given"
    )


if __name__ == "__main__":
    pytest.main([__file__])
