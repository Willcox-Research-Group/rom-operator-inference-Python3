# utils/test_kronecker.py
"""Tests for rom_operator_inference.utils._kronecker.py."""

import pytest
import numpy as np

import rom_operator_inference as opinf


# Kronecker (Khatri-Rao) products =============================================
# utils.kron2c() --------------------------------------------------------------
def _test_kron2c_single_vector(n):
    """Do one vector test of utils._kronecker.kron2c()."""
    x = np.random.random(n)
    x2 = opinf.utils.kron2c(x)
    assert x2.ndim == 1
    assert x2.shape[0] == n*(n+1)//2
    for i in range(n):
        assert np.allclose(x2[i*(i+1)//2:(i+1)*(i+2)//2], x[i]*x[:i+1])


def _test_kron2c_single_matrix(n):
    """Do one matrix test of utils._kronecker.kron2c()."""
    X = np.random.random((n,n))
    X2 = opinf.utils.kron2c(X)
    assert X2.ndim == 2
    assert X2.shape[0] == n*(n+1)//2
    assert X2.shape[1] == n
    for i in range(n):
        assert np.allclose(X2[i*(i+1)//2:(i+1)*(i+2)//2], X[i]*X[:i+1])


def test_kron2c(n_tests=100):
    """Test utils._kronecker.kron2c()."""
    # Try with bad input.
    with pytest.raises(ValueError) as exc:
        opinf.utils.kron2c(np.random.random((3,3,3)), checkdim=True)
    assert exc.value.args[0] == "x must be one- or two-dimensional"

    # Correct inputs.
    for n in np.random.randint(2, 100, n_tests):
        _test_kron2c_single_vector(n)
        _test_kron2c_single_matrix(n)


# utils.kron3c() --------------------------------------------------------------
def _test_kron3c_single_vector(n):
    """Do one vector test of utils._kronecker.kron3c()."""
    x = np.random.random(n)
    x3 = opinf.utils.kron3c(x)
    assert x3.ndim == 1
    assert x3.shape[0] == n*(n+1)*(n+2)//6
    for i in range(n):
        assert np.allclose(x3[i*(i+1)*(i+2)//6:(i+1)*(i+2)*(i+3)//6],
                           x[i]*opinf.utils.kron2c(x[:i+1]))


def _test_kron3c_single_matrix(n):
    """Do one matrix test of utils._kronecker.kron3c()."""
    X = np.random.random((n,n))
    X3 = opinf.utils.kron3c(X)
    assert X3.ndim == 2
    assert X3.shape[0] == n*(n+1)*(n+2)//6
    assert X3.shape[1] == n
    for i in range(n):
        assert np.allclose(X3[i*(i+1)*(i+2)//6:(i+1)*(i+2)*(i+3)//6],
                           X[i]*opinf.utils.kron2c(X[:i+1]))


def test_kron3c(n_tests=50):
    """Test utils._kronecker.kron3c()."""
    # Try with bad input.
    with pytest.raises(ValueError) as exc:
        opinf.utils.kron3c(np.random.random((2,4,3)), checkdim=True)
    assert exc.value.args[0] == "x must be one- or two-dimensional"

    # Correct inputs.
    for n in np.random.randint(2, 30, n_tests):
        _test_kron3c_single_vector(n)
        _test_kron3c_single_matrix(n)


# Matricized tensor management ================================================
# utils.expand_H() -----------------------------------------------------------
def _test_expand_H_single(r):
    """Do one test of utils._kronecker.expand_H()."""
    x = np.random.random(r)

    # Do a valid expand_H() calculation and check dimensions.
    s = r*(r+1)//2
    Hc = np.random.random((r,s))
    H = opinf.utils.expand_H(Hc)
    assert H.shape == (r,r**2)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(Hc @ opinf.utils.kron2c(x), Hxx)

    # Check properties of the tensor for H.
    Htensor = H.reshape((r,r,r))
    assert np.allclose(Htensor @ x @ x, Hxx)
    for subH in H:
        assert np.allclose(subH, subH.T)


def test_expand_H(n_tests=100):
    """Test utils._kronecker.expand_H()."""
    # Try to do expand_H() with a bad second dimension.
    r = 5
    sbad = r*(r+3)//2
    Hc = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.expand_H(Hc)
    assert exc.value.args[0] == \
        f"invalid shape (r,s) = {(r,sbad)} with s != r(r+1)/2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_expand_H_single(r)


# utils.compress_H() ----------------------------------------------------------
def _test_compress_H_single(r):
    """Do one test of utils._kronecker.compress_H()."""
    x = np.random.random(r)

    # Do a valid compress_H() calculation and check dimensions.
    H = np.random.random((r,r**2))
    s = r*(r+1)//2
    Hc = opinf.utils.compress_H(H)
    assert Hc.shape == (r,s)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(Hxx, Hc @ opinf.utils.kron2c(x))

    # Check that expand_H() and compress_H() are inverses up to symmetry.
    H2 = opinf.utils.expand_H(Hc)
    Ht = H.reshape((r,r,r))
    Htnew = np.empty_like(Ht)
    for i in range(r):
        Htnew[i] = (Ht[i] + Ht[i].T) / 2
    assert np.allclose(H2, Htnew.reshape(H.shape))


def test_compress_H(n_tests=100):
    """Test utils._kronecker.compress_H()."""
    # Try to do compress_H() with a bad second dimension.
    r = 5
    r2bad = r**2 + 1
    H = np.random.random((r, r2bad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.compress_H(H)
    assert exc.value.args[0] == \
        f"invalid shape (r,a) = {(r,r2bad)} with a != r**2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_compress_H_single(r)


# utils.expand_G() -----------------------------------------------------------
def _test_expand_G_single(r):
    """Do one test of utils._kronecker.expand_G()."""
    x = np.random.random(r)

    # Do a valid expand_H() calculation and check dimensions.
    s = r*(r+1)*(r+2)//6
    Gc = np.random.random((r,s))
    G = opinf.utils.expand_G(Gc)
    assert G.shape == (r,r**3)

    # Check that Gc(x^3) == G(x⊗x⊗x).
    Gxxx = G @ np.kron(x,np.kron(x,x))
    assert np.allclose(Gc @ opinf.utils.kron3c(x), Gxxx)

    # Check properties of the tensor for G.
    Gtensor = G.reshape((r,r,r,r))
    assert np.allclose(Gtensor @ x @ x @ x, Gxxx)
    for subG in G:
        assert np.allclose(subG, subG.T)


def test_expand_G(n_tests=50):
    """Test utils._kronecker.expand_G()."""
    # Try to do expand_G() with a bad second dimension.
    r = 5
    sbad = r*(r+1)*(r+3)//6
    Gc = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.expand_G(Gc)
    assert exc.value.args[0] == \
        f"invalid shape (r,s) = {(r,sbad)} with s != r(r+1)(r+2)/6"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 30, n_tests):
        _test_expand_G_single(r)


# utils.compress_G() ----------------------------------------------------------
def _test_compress_G_single(r):
    """Do one test of utils._kronecker.compress_G()."""
    x = np.random.random(r)

    # Do a valid compress_G() calculation and check dimensions.
    G = np.random.random((r,r**3))
    s = r*(r+1)*(r+2)//6
    Gc = opinf.utils.compress_G(G)
    assert Gc.shape == (r,s)

    # Check that Gc(x^3) == G(x⊗x⊗x).
    Gxxx = G @ np.kron(x,np.kron(x,x))
    assert np.allclose(Gxxx, Gc @ opinf.utils.kron3c(x))

    # Check that expand_G() and compress_G() are "inverses."
    assert np.allclose(Gc, opinf.utils.compress_G(opinf.utils.expand_G(Gc)))


def test_compress_G(n_tests=50):
    """Test utils._kronecker.compress_G()."""
    # Try to do compress_H() with a bad second dimension.
    r = 5
    r3bad = r**3 + 1
    G = np.random.random((r, r3bad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.compress_G(G)
    assert exc.value.args[0] == \
        f"invalid shape (r,a) = {(r,r3bad)} with a != r**3"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 30, n_tests):
        _test_compress_G_single(r)
