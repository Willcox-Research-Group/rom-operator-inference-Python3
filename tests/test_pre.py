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


def test_pod_basis(set_up_basis_data):
    """Test pre.pod_basis() on a small case with the ARPACK solver."""
    X = set_up_basis_data
    n,k = X.shape
    r = k // 10

    # Attempt an invalid mode.
    with pytest.raises(ValueError) as exc:
        roi.pre.pod_basis(X, r, mode="dense")
    assert exc.value.args[0] == "invalid mode 'dense'"

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

@pytest.fixture
def set_up_uniform_difference_data():
    t = np.linspace(0, 1, 400)
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


def test_fwd4(set_up_uniform_difference_data):
    """Test pre._fwd4()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 5):
        dYj = roi.pre._fwd4(Y[:,j:j+5], dt)
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])


def test_fwd6(set_up_uniform_difference_data):
    """Test pre._fwd6()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 7):
        dYj = roi.pre._fwd6(Y[:,j:j+7], dt)
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])


def test_compute_xdot_uniform(set_up_uniform_difference_data):
    """Test pre.compute_xdot_uniform()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for o in [2, 4, 6]:
        dY_ = roi.pre.compute_xdot_uniform(Y, dt, order=o)
        assert dY_.shape == Y.shape
        assert np.allclose(dY, dY_, atol=1e-4)

    with pytest.raises(ValueError) as exc:
        roi.pre.compute_xdot_uniform(Y, dt, order=-1)
    assert exc.value.args[0] == "invalid order '-1'"


@pytest.fixture
def set_up_nonuniform_difference_data():
    t = np.linspace(0, 1, 400)**2
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


def test_compute_xdot(set_up_nonuniform_difference_data):
    """Test pre.compute_xdot()."""
    dynamicstate = set_up_nonuniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dY_ = roi.pre.compute_xdot(Y, t)
    assert dY_.shape == Y.shape
    assert np.allclose(dY, dY_, atol=1e-4)
