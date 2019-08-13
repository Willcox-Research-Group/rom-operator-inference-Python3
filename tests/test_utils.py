# test_utils.py
"""Tests for rom_operator_inference.utils.py."""

import pytest
import numpy as np

import rom_operator_inference as roi


# rom_operator_inference.utils.kron_compact() ---------------------------------
def _test_kron_compact_single_vector(n):
    """Do one vector test of rom_operator_inference.utils.kron_compact()."""
    x = np.random.random(n)
    x2 = roi.utils.kron_compact(x)
    assert x2.ndim == 1
    assert x2.shape[0] == n*(n+1)//2
    for i in range(n):
        assert np.allclose(x2[i*(i+1)//2:(i+1)*(i+2)//2], x[i]*x[:i+1])

def _test_kron_compact_single_matrix(n):
    """Do one matrix test of rom_operator_inference.utils.kron_compact()."""
    X = np.random.random((n,n))
    X2 = roi.utils.kron_compact(X)
    assert X2.ndim == 2
    assert X2.shape[0] == n*(n+1)//2
    assert X2.shape[1] == n
    for i in range(n):
        assert np.allclose(X2[i*(i+1)//2:(i+1)*(i+2)//2], X[i]*X[:i+1])

def test_kron_compact(n_tests=100):
    """Test rom_operator_inference.utils.kron_compact()."""
    for n in np.random.randint(2, 100, n_tests):
        _test_kron_compact_single_vector(n)
        _test_kron_compact_single_matrix(n)


# rom_operator_inference.utils.F2H() ------------------------------------------
def _test_F2H_single(r):
    """Do one test of rom_operator_inference.utils.F2H()."""
    x = np.random.random(r)

    # Do a valid F2H() calculation and check dimensions.
    s = r*(r+1)//2
    F = np.random.random((r,s))
    H = roi.utils.F2H(F)
    assert H.shape == (r,r**2)

    # Check that F(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(F @ roi.utils.kron_compact(x), Hxx)

    # Check properties of the tensor for H.
    Htensor = H.reshape((r,r,r))
    assert np.allclose(Htensor @ x @ x, Hxx)
    for subH in H:
        assert np.allclose(subH, subH.T)

def test_F2H(n_tests=100):
    """Test rom_operator_inference.utils.F2H()."""
    # Try to do F2H() with a bad second dimension.
    r = 5
    sbad = r*(r+3)//2
    F = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        roi.utils.F2H(F)
    assert exc.value.args[0] == \
        f"invalid shape (r,s) = {(r,sbad)} with s != r(r+1)/2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_F2H_single(r)
