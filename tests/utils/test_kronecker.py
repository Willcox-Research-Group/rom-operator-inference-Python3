# utils/test_kronecker.py
"""Tests for rom_operator_inference.utils._kronecker."""

import pytest
import numpy as np

import rom_operator_inference as opinf


# Index generation for fast self-product kronecker evaluation =================
def test_kron2c_indices(n_tests=100):
    """Test utils._kronecker.kron2c_indices()."""
    mask = opinf.utils.kron2c_indices(4)
    assert np.all(mask == np.array([[0, 0],
                                    [1, 0], [1, 1],
                                    [2, 0], [2, 1], [2, 2],
                                    [3, 0], [3, 1], [3, 2], [3, 3]],
                                   dtype=int))
    submask = opinf.utils.kron2c_indices(3)
    assert np.allclose(submask, mask[:6])

    r = 10
    _r2 = r * (r + 1) // 2
    mask = opinf.utils.kron2c_indices(r)
    assert mask.shape == (_r2, 2)
    assert np.all(mask[0] == 0)
    assert np.all(mask[-1] == r - 1)
    assert mask.sum(axis=0)[0] == sum(i*(i+1) for i in range(r))

    # Ensure consistency with utils.kron2c().
    for _ in range(n_tests):
        x = np.random.random(r)
        assert np.allclose(np.prod(x[mask], axis=1), opinf.utils.kron2c(x))


def test_kron3c_indices(n_tests=100):
    """Test utils._kronecker.kron3c_indices()."""
    mask = opinf.utils.kron3c_indices(2)
    assert np.all(mask == np.array([[0, 0, 0],
                                    [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                                   dtype=int))

    r = 10
    mask = opinf.utils.kron3c_indices(r)
    _r3 = r * (r + 1) * (r + 2) // 6
    mask = opinf.utils.kron3c_indices(r)
    assert mask.shape == (_r3, 3)
    assert np.all(mask[0] == 0)
    assert np.all(mask[-1] == r - 1)

    # Ensure consistency with utils.kron3c().
    for _ in range(n_tests):
        x = np.random.random(r)
        assert np.allclose(np.prod(x[mask], axis=1), opinf.utils.kron3c(x))


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
    X = np.random.random((n, n))
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
        opinf.utils.kron2c(np.random.random((3, 3, 3)), checkdim=True)
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
    X = np.random.random((n, n))
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
        opinf.utils.kron3c(np.random.random((2, 4, 3)), checkdim=True)
    assert exc.value.args[0] == "x must be one- or two-dimensional"

    # Correct inputs.
    for n in np.random.randint(2, 30, n_tests):
        _test_kron3c_single_vector(n)
        _test_kron3c_single_matrix(n)


# Matricized tensor management ================================================
# utils.expand_quadratic() ----------------------------------------------------
def _test_expand_quadratic_single(r):
    """Do one test of utils._kronecker.expand_quadratic()."""
    x = np.random.random(r)

    # Do a valid expand_quadratic() calculation and check dimensions.
    s = r*(r+1)//2
    Hc = np.random.random((r, s))
    H = opinf.utils.expand_quadratic(Hc)
    assert H.shape == (r, r**2)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x, x)
    assert np.allclose(Hc @ opinf.utils.kron2c(x), Hxx)

    # Check properties of the tensor for H.
    Htensor = H.reshape((r, r, r))
    assert np.allclose(Htensor @ x @ x, Hxx)
    for subH in H:
        assert np.allclose(subH, subH.T)


def test_expand_quadratic(n_tests=100):
    """Test utils._kronecker.expand_quadratic()."""
    # Try to do expand_quadratic() with a bad second dimension.
    r = 5
    sbad = r*(r+3)//2
    Hc = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.expand_quadratic(Hc)
    assert exc.value.args[0] == \
        f"invalid shape (r, s) = {(r, sbad)} with s != r(r+1)/2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_expand_quadratic_single(r)


# utils.compress_quadratic() --------------------------------------------------
def _test_compress_quadratic_single(r):
    """Do one test of utils._kronecker.compress_quadratic()."""
    x = np.random.random(r)

    # Do a valid compress_quadratic() calculation and check dimensions.
    H = np.random.random((r, r**2))
    s = r*(r+1)//2
    Hc = opinf.utils.compress_quadratic(H)
    assert Hc.shape == (r, s)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x, x)
    assert np.allclose(Hxx, Hc @ opinf.utils.kron2c(x))

    # Check that expand_quadratic() and compress_quadratic()
    # are inverses up to symmetry.
    H2 = opinf.utils.expand_quadratic(Hc)
    Ht = H.reshape((r, r, r))
    Htnew = np.empty_like(Ht)
    for i in range(r):
        Htnew[i] = (Ht[i] + Ht[i].T) / 2
    assert np.allclose(H2, Htnew.reshape(H.shape))


def test_compress_quadratic(n_tests=100):
    """Test utils._kronecker.compress_quadratic()."""
    # Try to do compress_quadratic() with a bad second dimension.
    r = 5
    r2bad = r**2 + 1
    H = np.random.random((r, r2bad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.compress_quadratic(H)
    assert exc.value.args[0] == \
        f"invalid shape (r, a) = {(r, r2bad)} with a != r**2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_compress_quadratic_single(r)


# utils.expand_cubic() --------------------------------------------------------
def _test_expand_cubic_single(r):
    """Do one test of utils._kronecker.expand_cubic()."""
    x = np.random.random(r)

    # Do a valid expand_cubic() calculation and check dimensions.
    s = r*(r+1)*(r+2)//6
    Gc = np.random.random((r, s))
    G = opinf.utils.expand_cubic(Gc)
    assert G.shape == (r, r**3)

    # Check that Gc(x^3) == G(x⊗x⊗x).
    Gxxx = G @ np.kron(x, np.kron(x, x))
    assert np.allclose(Gc @ opinf.utils.kron3c(x), Gxxx)

    # Check properties of the tensor for G.
    Gtensor = G.reshape((r, r, r, r))
    assert np.allclose(Gtensor @ x @ x @ x, Gxxx)
    for subG in G:
        assert np.allclose(subG, subG.T)


def test_expand_cubic(n_tests=50):
    """Test utils._kronecker.expand_cubic()."""
    # Try to do expand_cubic() with a bad second dimension.
    r = 5
    sbad = r*(r+1)*(r+3)//6
    Gc = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.expand_cubic(Gc)
    assert exc.value.args[0] == \
        f"invalid shape (r, s) = {(r, sbad)} with s != r(r+1)(r+2)/6"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 30, n_tests):
        _test_expand_cubic_single(r)


# utils.compress_cubic() ------------------------------------------------------
def _test_compress_cubic_single(r):
    """Do one test of utils._kronecker.compress_cubic()."""
    x = np.random.random(r)

    # Do a valid compress_cubic() calculation and check dimensions.
    G = np.random.random((r, r**3))
    s = r*(r+1)*(r+2)//6
    Gc = opinf.utils.compress_cubic(G)
    assert Gc.shape == (r, s)

    # Check that Gc(x^3) == G(x⊗x⊗x).
    Gxxx = G @ np.kron(x, np.kron(x, x))
    assert np.allclose(Gxxx, Gc @ opinf.utils.kron3c(x))

    # Check that expand_cubic() and compress_cubic() are "inverses."
    G_new = opinf.utils.expand_cubic(Gc)
    assert np.allclose(Gc, opinf.utils.compress_cubic(G_new))


def test_compress_cubic(n_tests=50):
    """Test utils._kronecker.compress_cubic()."""
    # Try to do compress_cubic() with a bad second dimension.
    r = 5
    r3bad = r**3 + 1
    G = np.random.random((r, r3bad))
    with pytest.raises(ValueError) as exc:
        opinf.utils.compress_cubic(G)
    assert exc.value.args[0] == \
        f"invalid shape (r, a) = {(r, r3bad)} with a != r**3"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 30, n_tests):
        _test_compress_cubic_single(r)
