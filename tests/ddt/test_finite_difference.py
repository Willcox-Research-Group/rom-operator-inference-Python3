# ddt/test_finite_difference.py
"""Tests for ddt._finite_difference.py"""

import pytest
import numpy as np

import opinf


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


class TestUniformFiniteDifferencer:
    """Test ddt.UniformFiniteDifferencer."""

    Diff = _module.UniformFiniteDifferencer

    def test_init(self, k=100):
        """Test __init__(), time_domain, and scheme."""
        t = np.linspace(0, 1, k)
        differ = self.Diff(t)
        assert differ.time_domain is t
        assert differ.dt == (t[1] - t[0])

        # Try with non-uniform spacing.
        t2 = t**2
        with pytest.raises(ValueError) as ex:
            self.Diff(t2)
        assert ex.value.args[0] == "time domain must be uniformly spaced"

        # Too many dimensions.
        t3 = np.sort(np.random.random((3, 3, 3)))
        with pytest.raises(ValueError) as ex:
            self.Diff(t3)
        assert ex.value.args[0] == (
            "time_domain should be a one-dimensional array"
        )

        # Multiple time domains (for handling multiple data trajectories).
        ts = [t + i for i in range(3)]
        differ = self.Diff(ts)
        assert differ.time_domain is ts
        assert differ.dt is None

        # Bad scheme.
        with pytest.raises(ValueError) as ex:
            self.Diff(t, scheme="best")
        assert ex.value.args[0] == "invalid finite difference scheme 'best'"

        # Custom scheme.

        def myscheme():
            pass

        differ = self.Diff(t, myscheme)
        assert differ.scheme is myscheme

        # Registered schemes.
        for name, func in self.Diff._schemes.items():
            assert callable(func)
            differ = self.Diff(t, scheme=name)
            assert differ.scheme is func

    def test_estimate(self, r=3, k=100, m=2):
        """Use verify() to validate estimate()
        for all registered difference schemes.
        """
        t = np.linspace(0, 1, k)

        # Coverage for error messages.
        differ = self.Diff([t, t, t])
        with pytest.raises(RuntimeError) as ex:
            differ.estimate(None)
        assert ex.value.args[0] == "dt is None"

        differ = self.Diff(t)
        Q = np.random.random(k)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            differ.estimate(Q)
        assert ex.value.args[0] == "states must be two-dimensional"

        Q = np.random.random((r, k))
        U = np.random.random((m, k))
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            differ.estimate(Q, U[:, :-1])
        assert ex.value.args[0] == "states and inputs not aligned"

        # One-dimensional inputs.
        differ.estimate(Q, U[0])

        # Test all schemes.
        for name, _ in self.Diff._schemes.items():
            differ = self.Diff(t, scheme=name)
            errors = differ.verify(plot=False, return_errors=True)
            for label, results in errors.items():
                if label == "dts":
                    continue
                assert (
                    np.min(results) < 5e-7
                ), f"problem with scheme '{name}', test '{label}'"


class TestNonuniformFiniteDifferencer:
    """Test ddt.NonuniformFiniteDifferencer."""

    Diff = _module.NonuniformFiniteDifferencer

    def test_init(self, k=100):
        """Test __init__(), time_domain, and order."""
        t = np.linspace(0, 1, k)

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Diff(t)
        assert wn[0].message.args[0] == (
            "time_domain is uniformly spaced, consider using "
            "UniformFiniteDifferencer"
        )

        t = t**2
        differ = self.Diff(t)
        assert differ.time_domain is t

        # Too many dimensions.
        t3 = np.sort(np.random.random((3, 3, 3)))
        with pytest.raises(ValueError) as ex:
            self.Diff(t3)
        assert ex.value.args[0] == (
            "time_domain should be a one-dimensional array"
        )

        # Multiple time domains (for handling multiple data trajectories).
        ts = [t + i for i in range(3)]
        differ = self.Diff(ts)
        assert differ.time_domain is ts

    def test_estimate(self, r=3, k=100, m=4):
        """Use verify() to test estimate()."""
        t = np.linspace(0, 1, k) ** 2
        differ = self.Diff(t)

        # Coverage for error messages.
        Q = np.random.random(k)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            differ.estimate(Q)
        assert ex.value.args[0] == "states must be two-dimensional"

        Q = np.random.random((r, k))
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            differ.estimate(Q[:, :-1])
        assert ex.value.args[0] == "states not aligned with time_domain"

        U = np.random.random((m, k))
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            differ.estimate(Q, U[:, :-1])
        assert ex.value.args[0] == "inputs not aligned with time_domain"

        # One-dimensional inputs.
        differ.estimate(Q, U[0])

        # Test the implementation.
        differ.verify()


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
