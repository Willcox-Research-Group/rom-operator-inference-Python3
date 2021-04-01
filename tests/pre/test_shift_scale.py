# pre/test_shift_scale.py
"""Tests for rom_operator_inference.pre._shift_scale.py."""

import pytest
import numpy as np

import rom_operator_inference as opinf


# Data preprocessing: shifting and MinMax scaling / unscaling =================
def test_shift(set_up_basis_data):
    """Test pre._shift_scale.shift()."""
    X = set_up_basis_data

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        opinf.pre.shift(np.random.random((3,3,3)))
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad shift vector.
    with pytest.raises(ValueError) as exc:
        opinf.pre.shift(X, X)
    assert exc.value.args[0] == "shift_by must be one-dimensional"

    # Correct usage.
    Xshifted, xbar = opinf.pre.shift(X)
    assert xbar.shape == (X.shape[0],)
    assert Xshifted.shape == X.shape
    assert np.allclose(np.mean(Xshifted, axis=1), np.zeros(X.shape[0]))
    for j in range(X.shape[1]):
        assert np.allclose(Xshifted[:,j], X[:,j] - xbar)

    Y = np.random.random(X.shape)
    Yshifted = opinf.pre.shift(Y, xbar)
    for j in range(Y.shape[1]):
        assert np.allclose(Yshifted[:,j], Y[:,j] - xbar)

    # Verify inverse shifting.
    assert np.allclose(X, opinf.pre.shift(Xshifted, -xbar))


def test_scale(set_up_basis_data):
    """Test pre._shift_scale.scale()."""
    X = set_up_basis_data

    # Try with bad scales.
    with pytest.raises(ValueError) as exc:
        opinf.pre.scale(X, (1,2,3), (4,5))
    assert exc.value.args[0] == "scale_to must have exactly 2 elements"

    with pytest.raises(ValueError) as exc:
        opinf.pre.scale(X, (1,2), (3,4,5))
    assert exc.value.args[0] == "scale_from must have exactly 2 elements"

    # Scale X to [-1,1] and then scale Y with the same transformation.
    Xscaled, scaled_to, scaled_from = opinf.pre.scale(X, (-1,1))
    assert Xscaled.shape == X.shape
    assert scaled_to == (-1,1)
    assert isinstance(scaled_from, tuple)
    assert len(scaled_from) == 2
    assert round(scaled_from[0],8) == round(X.min(),8)
    assert round(scaled_from[1],8) == round(X.max(),8)
    assert round(Xscaled.min(),8) == -1
    assert round(Xscaled.max(),8) == 1

    # Verify inverse scaling.
    assert np.allclose(opinf.pre.scale(Xscaled, scaled_from, scaled_to), X)
