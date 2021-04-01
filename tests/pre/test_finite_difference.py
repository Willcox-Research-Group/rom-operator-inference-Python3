# pre/test_finite_difference.py
"""Tests for rom_operator_inference.pre._finite_difference.py"""

import pytest
import numpy as np

import rom_operator_inference as opinf


# Derivative approximation ====================================================
@pytest.mark.usefixtures("set_up_uniform_difference_data")
def test_fwd4(set_up_uniform_difference_data):
    """Test pre._finite_difference._fwd4()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 5):
        # One-dimensional test.
        dY0 = opinf.pre._finite_difference._fwd4(Y[0,j:j+5], dt)
        assert isinstance(dY0, float)
        assert np.isclose(dY0, dY[0,j])

        # Two-dimensional test.
        dYj = opinf.pre._finite_difference._fwd4(Y[:,j:j+5].T, dt)
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])

        # Check agreement.
        assert dY0 == dYj[0]


def test_fwd6(set_up_uniform_difference_data):
    """Test pre._finite_difference._fwd6()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for j in range(Y.shape[1] - 7):
        # One-dimensional test.
        dY0 = opinf.pre._finite_difference._fwd6(Y[0,j:j+7], dt)
        assert isinstance(dY0, float)
        assert np.isclose(dY0, dY[0,j])

        # Two-dimensional test.
        dYj = opinf.pre._finite_difference._fwd6(Y[:,j:j+7].T, dt).T
        assert dYj.shape == Y[:,j].shape
        assert np.allclose(dYj, dY[:,j])

        # Check agreement.
        assert dY0 == dYj[0]


def test_xdot_uniform(set_up_uniform_difference_data):
    """Test pre._finite_difference.xdot_uniform()."""
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]
    for o in [2, 4, 6]:
        dY_ = opinf.pre.xdot_uniform(Y, dt, order=o)
        assert dY_.shape == Y.shape
        assert np.allclose(dY, dY_, atol=1e-4)

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        opinf.pre.xdot_uniform(Y[:,0], dt, order=2)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad order.
    with pytest.raises(NotImplementedError) as exc:
        opinf.pre.xdot_uniform(Y, dt, order=-1)
    assert exc.value.args[0] == "invalid order '-1'; valid options: {2, 4, 6}"

    # Try with bad dt type.
    with pytest.raises(TypeError) as exc:
        opinf.pre.xdot_uniform(Y, np.array([dt, 2*dt]), order=-1)
    assert exc.value.args[0] == "time step dt must be a scalar (e.g., float)"


def test_xdot_nonuniform(set_up_nonuniform_difference_data):
    """Test pre._finite_difference.xdot_nonuniform()."""
    dynamicstate = set_up_nonuniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dY_ = opinf.pre.xdot_nonuniform(Y, t)
    assert dY_.shape == Y.shape
    assert np.allclose(dY, dY_, atol=1e-4)

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        opinf.pre.xdot_nonuniform(Y[:,0], t)
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad time shape.
    with pytest.raises(ValueError) as exc:
        opinf.pre.xdot_nonuniform(Y, np.dstack((t,t)))
    assert exc.value.args[0] == "time t must be one-dimensional"

    with pytest.raises(ValueError) as exc:
        opinf.pre.xdot_nonuniform(Y, np.hstack((t,t)))
    assert exc.value.args[0] == "data X not aligned with time t"


def test_xdot(set_up_uniform_difference_data,
              set_up_nonuniform_difference_data):
    """Test pre._finite_difference.xdot()."""
    # Uniform tests.
    dynamicstate = set_up_uniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative
    dt = t[1] - t[0]

    def _single_test(*args, **kwargs):
        dY_ = opinf.pre.xdot(*args, **kwargs)
        assert dY_.shape == Y.shape
        assert np.allclose(dY, dY_, atol=1e-4)

    _single_test(Y, dt)
    _single_test(Y, dt=dt)
    for o in [2, 4, 6]:
        _single_test(Y, dt, o)
        _single_test(Y, dt, order=o)
        _single_test(Y, dt=dt, order=o)
        _single_test(Y, order=o, dt=dt)
        _single_test(Y, t)

    # Nonuniform tests.
    dynamicstate = set_up_nonuniform_difference_data
    t, Y, dY = dynamicstate.time, dynamicstate.state, dynamicstate.derivative

    _single_test(Y, t)
    _single_test(Y, t=t)

    # Try with bad arguments.
    with pytest.raises(TypeError) as exc:
        opinf.pre.xdot(Y)
    assert exc.value.args[0] == \
        "at least one other argument required (dt or t)"

    with pytest.raises(TypeError) as exc:
        opinf.pre.xdot(Y, order=2)
    assert exc.value.args[0] == \
        "keyword argument 'order' requires float argument dt"

    with pytest.raises(TypeError) as exc:
        opinf.pre.xdot(Y, other=2)
    assert exc.value.args[0] == \
        "xdot() got unexpected keyword argument 'other'"

    with pytest.raises(TypeError) as exc:
        opinf.pre.xdot(Y, 2)
    assert exc.value.args[0] == \
        "invalid argument type '<class 'int'>'"

    with pytest.raises(TypeError) as exc:
        opinf.pre.xdot(Y, dt, 4, None)
    assert exc.value.args[0] == \
        "xdot() takes from 2 to 3 positional arguments but 4 were given"
