# test_pre.py
"""Tests for rom_operator_inference.pre.py"""

import pytest
import numpy as np
from scipy import linalg as la
from collections import namedtuple

import rom_operator_inference as roi


# Basis calculation -----------------------------------------------------------
@pytest.fixture
def set_up_basis_data():
    n = 200
    k = 50
    return np.random.random((n,k)) - .5


def test_mean_shift(set_up_basis_data):
    """Test pre.mean_shift()."""
    X = set_up_basis_data
    xbar, Xshifted = roi.pre.mean_shift(X)
    assert xbar.shape == (X.shape[0],)
    assert Xshifted.shape == X.shape
    assert np.allclose(np.mean(Xshifted, axis=1), np.zeros(X.shape[0]))
    assert np.allclose(xbar.reshape((-1,1)) + Xshifted, X)

    # Try using bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.mean_shift(np.random.random((3,3,3)))
    assert exc.value.args[0] == "data X must be two-dimensional"


def test_pod_basis(set_up_basis_data):
    """Test pre.pod_basis() on a small case with the ARPACK solver."""
    X = set_up_basis_data
    n,k = X.shape
    r = k // 10

    # Attempt an invalid mode.
    with pytest.raises(NotImplementedError) as exc:
        roi.pre.pod_basis(X, r, mode="dense")
    assert exc.value.args[0] == "invalid mode 'dense'"

    # Attempt with scipy.linalg.
    Vr = roi.pre.pod_basis(X, r, mode="simple")
    assert Vr.shape == (n,r)

    # Attempt with ARPACK.
    Vr = roi.pre.pod_basis(X, r, mode="arpack")
    assert Vr.shape == (n,r)

    Ur = la.svd(X)[0][:,:r]
    for j in range(r):              # Make sure the columns have the same sign.
        if not np.isclose(Ur[0,j], Vr[0,j]):
            Ur[:,j] = -Ur[:,j]
    assert np.allclose(Vr, Ur)

    # Attempt with randomized_svd().
    Vr = roi.pre.pod_basis(X, r, mode="randomized")
    assert Vr.shape == (n,r)
    # No accuracy test, since that is not guaranteed by randomized SVD.


# Differentiation routines ----------------------------------------------------
DynamicState = namedtuple("DynamicState", ["time", "state", "derivative"])

def _difference_data(t):
    Y = np.row_stack((t,
                      t**2/2,
                      t**3/3,
                      np.sin(t),
                      np.exp(t),
                      1/(t+1),
                      t + t**2/2 + t**3/3 + np.sin(t) - np.exp(t)
                    ))
    dY = np.row_stack((np.ones_like(t),
                       t,
                       t**2,
                       np.cos(t),
                       np.exp(t),
                       -1/(t+1)**2,
                       1 + t + t**2 + np.cos(t) - np.exp(t)
                     ))
    return DynamicState(t, Y, dY)


@pytest.fixture
def set_up_uniform_difference_data():
    t = np.linspace(0, 1, 400)
    return _difference_data(t)


@pytest.fixture
def set_up_nonuniform_difference_data():
    t = np.linspace(0, 1, 400)**2
    return _difference_data(t)


def test_fwd4(set_up_uniform_difference_data):
    """Test pre._fwd4()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 5):
        # One-dimensional test.
        dY0 = roi.pre._fwd4(Y[0,j:j+5], dt)
        assert isinstance(dY0, float)
        assert np.isclose(dY0, dY[0,j])

        # Two-dimensional test.
        dYj = roi.pre._fwd4(Y[:,j:j+5].T, dt)
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])

        # Check agreement.
        assert dY0 == dYj[0]


def test_fwd6(set_up_uniform_difference_data):
    """Test pre._fwd6()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 7):
        # One-dimensional test.
        dY0 = roi.pre._fwd6(Y[0,j:j+7], dt)
        assert isinstance(dY0, float)
        assert np.isclose(dY0, dY[0,j])

        # Two-dimensional test.
        dYj = roi.pre._fwd6(Y[:,j:j+7].T, dt).T
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])

        # Check agreement.
        assert dY0 == dYj[0]


def test_compute_xdot_uniform(set_up_uniform_difference_data):
    """Test pre.compute_xdot_uniform()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for o in [2, 4, 6]:
        dY_ = roi.pre.compute_xdot_uniform(Y, dt, order=o)
        assert dY_.shape == Y.shape
        print("(order, error):", (o, np.linalg.norm(dY - dY_)))
        assert np.allclose(dY, dY_, atol=1e-4)

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.compute_xdot_uniform(Y[:,0], dt, order=2)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad order.
    with pytest.raises(NotImplementedError) as exc:
        roi.pre.compute_xdot_uniform(Y, dt, order=-1)
    assert exc.value.args[0] == "invalid order '-1'"

    # Try with bad dt type.
    with pytest.raises(TypeError) as exc:
        roi.pre.compute_xdot_uniform(Y, np.array([dt, 2*dt]), order=-1)
    assert exc.value.args[0] == "time step dt must be a scalar (e.g., float)"


def test_compute_xdot_nonuniform(set_up_nonuniform_difference_data):
    """Test pre.compute_xdot()."""
    dynamicstate = set_up_nonuniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dY_ = roi.pre.compute_xdot_nonuniform(Y, t)
    assert dY_.shape == Y.shape
    assert np.allclose(dY, dY_, atol=1e-4)

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.compute_xdot_nonuniform(Y[:,0], t)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad time shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.compute_xdot_nonuniform(Y, np.dstack((t,t)))
    assert exc.value.args[0] == "time t must be one-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.pre.compute_xdot_nonuniform(Y, np.hstack((t,t)))
    assert exc.value.args[0] == "data X not aligned with time t"
