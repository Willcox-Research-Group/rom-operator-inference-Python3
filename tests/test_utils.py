# test_utils.py
"""Tests for rom_operator_inference.utils.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


def test_get_least_squares_size():
    """Test utils.get_least_squares_size()."""
    m, r = 3, 7

    # Try with bad input combinations.
    with pytest.raises(ValueError) as ex:
        roi.utils.get_least_squares_size("cAHB", r)
    assert ex.value.args[0] == "argument m > 0 required since 'B' in modelform"

    with pytest.raises(ValueError) as ex:
        roi.utils.get_least_squares_size("cAH", r, m=10)
    assert ex.value.args[0] == "argument m=10 invalid since 'B' in modelform"

    # Test without inputs.
    assert roi.utils.get_least_squares_size("c", r) == 1
    assert roi.utils.get_least_squares_size("A", r) == r
    assert roi.utils.get_least_squares_size("cA", r) == 1 + r
    assert roi.utils.get_least_squares_size("cAH", r) == 1 + r + r*(r+1)//2
    assert roi.utils.get_least_squares_size("cG", r) == 1 + r*(r+1)*(r+2)//6

    # Test with inputs.
    assert roi.utils.get_least_squares_size("cB", r, m) == 1 + m
    assert roi.utils.get_least_squares_size("AB", r, m) == r + m
    assert roi.utils.get_least_squares_size("cAB", r, m) == 1 + r + m
    assert roi.utils.get_least_squares_size("AHB", r, m) == r + r*(r+1)//2 + m
    assert roi.utils.get_least_squares_size("GB", r, m) == r*(r+1)*(r+2)//6 + m

    # Test with affines.
    assert roi.utils.get_least_squares_size("c", r, affines={"c":[0,0]}) == 2
    assert roi.utils.get_least_squares_size("A", r, affines={"A":[0,0]}) == 2*r


# utils.lstsq_reg() -----------------------------------------------------------
def _test_lstsq_reg_single(k,d,r):
    """Do one test of utils.lstsq_reg()."""
    A = np.random.random((k, d))
    b = np.random.random(k)
    B = np.random.random((k,r))
    I = np.eye(d)

    # VECTOR TEST
    x = la.lstsq(A, b)[0]

    # Ensure that the least squares solution is the usual one when P=0.
    x_ = roi.utils.lstsq_reg(A, b, P=0)[0]
    assert np.allclose(x, x_)

    # Ensure that the least squares solution is the usual one when P=0 (matrix)
    x_ = roi.utils.lstsq_reg(A, b, np.zeros((d,d)))[0]
    assert np.allclose(x, x_)

    # Check that the regularized least squares solution has decreased norm.
    x_ = roi.utils.lstsq_reg(A, b, P=2)[0]
    assert la.norm(x_) <= la.norm(x)

    x_ = roi.utils.lstsq_reg(A, b, P=2*I)[0]
    assert la.norm(x_) <= la.norm(x)

    # MATRIX TEST
    X = la.lstsq(A, B)[0]

    # Ensure that the least squares solution is the usual one when P=0.
    X_ = roi.utils.lstsq_reg(A, B, P=0)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares solution is the usual one when P=0 (matrix)
    X_ = roi.utils.lstsq_reg(A, B, P=0)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares solution is the usual one when P=0 (list)
    X_ = roi.utils.lstsq_reg(A, B, P=[np.zeros((d,d))]*r)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares problem decouples correctly.
    Ps = [l*I for l in range(r)]
    X_ = roi.utils.lstsq_reg(A, B, P=Ps)[0]
    for j in range(r):
        xj_ = roi.utils.lstsq_reg(A, B[:,j], P=Ps[j])[0]
        assert np.allclose(xj_, X_[:,j])

    # Check that the regularized least squares solution has decreased norm.
    X_ = roi.utils.lstsq_reg(A, B, P=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=[2]*r)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=[2*I]*r)[0]
    assert la.norm(X_) <= la.norm(X)

    # Test residuals actually give the Frobenius norm squared of the misfit.
    Acond = np.linalg.cond(A)
    X_, res, rnk, svdvals = roi.utils.lstsq_reg(A, B, P=0)
    assert np.isclose(np.sum(res), np.sum((A @ X_ - B)**2))
    assert np.isclose(abs(svdvals[0]/svdvals[-1]), Acond)

    # Ensure residuals larger, condition numbers smaller with regularization.
    X_, res, rnk, svdvals = roi.utils.lstsq_reg(A, B, P=2)
    assert np.sum(res) > np.sum((A @ X_ - B)**2)
    assert abs(svdvals[0]/svdvals[-1]) < Acond

    X_, res, rnk, svdvals = roi.utils.lstsq_reg(A, B, P=[2*I]*r)
    assert np.sum(res) > np.sum((A @ X_ - B)**2)
    assert abs(svdvals[0]/svdvals[-1]) < Acond


def test_lstsq_reg(n_tests=5):
    """Test utils.lstsq_reg()."""
    A = np.random.random((20,10))
    B = np.random.random((20, 5))

    # Negative regularization parameter not allowed.
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B, -1)
    assert exc.value.args[0] == "regularization parameter must be nonnegative"

    # Try with underdetermined system.
    with pytest.warns(la.LinAlgWarning) as exc:
        roi.utils.lstsq_reg(A[:8,:], B[:8,:])
    assert exc[0].message.args[0] == "least squares system is underdetermined"

    # b must be one- or two-dimensional
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, np.random.random((2,2,2)))
    assert exc.value.args[0] == "`b` must be one- or two-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, np.array(5))
    assert exc.value.args[0] == "`b` must be one- or two-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B[:,0],
                            [np.random.random((10,10)) for i in range(5)])
    assert exc.value.args[0] == "`b` must be two-dimensional with multiple P"

    # Badly shaped regularization matrix.
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B, np.random.random((5,5)))
    assert exc.value.args[0] == \
        "P must be (d,d) with d = number of columns of A"

    # Bad number of regularization matrices (list).
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B,
                            [np.random.random((10,10)) for i in range(3)])
    assert exc.value.args[0] == \
        "multiple P requires exactly r entries with r = number of columns of b"

    # Bad number of regularization matrices (generator).
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B,
                            (np.random.random((10,10)) for i in range(3)))
    assert exc.value.args[0] == \
        "multiple P requires exactly r entries with r = number of columns of b"


    # Do individual tests.
    k, m = 200, 20
    for r in np.random.randint(4, 50, n_tests):
        _test_lstsq_reg_single(k, min(k, 1 + r + r*(r+1)//2 + m), r)


# utils.kron2c() --------------------------------------------------------------
def _test_kron2c_single_vector(n):
    """Do one vector test of utils.kron2c()."""
    x = np.random.random(n)
    x2 = roi.utils.kron2c(x)
    assert x2.ndim == 1
    assert x2.shape[0] == n*(n+1)//2
    for i in range(n):
        assert np.allclose(x2[i*(i+1)//2:(i+1)*(i+2)//2], x[i]*x[:i+1])


def _test_kron2c_single_matrix(n):
    """Do one matrix test of utils.kron2c()."""
    X = np.random.random((n,n))
    X2 = roi.utils.kron2c(X)
    assert X2.ndim == 2
    assert X2.shape[0] == n*(n+1)//2
    assert X2.shape[1] == n
    for i in range(n):
        assert np.allclose(X2[i*(i+1)//2:(i+1)*(i+2)//2], X[i]*X[:i+1])


def test_kron2c(n_tests=100):
    """Test utils.kron2c()."""
    # Try with bad input.
    with pytest.raises(ValueError) as exc:
        roi.utils.kron2c(np.random.random((3,3,3)))
    assert exc.value.args[0] == "x must be one- or two-dimensional"

    # Correct inputs.
    for n in np.random.randint(2, 100, n_tests):
        _test_kron2c_single_vector(n)
        _test_kron2c_single_matrix(n)


# utils.kron3c() --------------------------------------------------------------
def _test_kron3c_single_vector(n):
    """Do one vector test of utils.kron3c()."""
    x = np.random.random(n)
    x3 = roi.utils.kron3c(x)
    assert x3.ndim == 1
    assert x3.shape[0] == n*(n+1)*(n+2)//6
    for i in range(n):
        assert np.allclose(x3[i*(i+1)*(i+2)//6:(i+1)*(i+2)*(i+3)//6],
                            x[i]*roi.utils.kron2c(x[:i+1]))


def _test_kron3c_single_matrix(n):
    """Do one matrix test of utils.kron3c()."""
    X = np.random.random((n,n))
    X3 = roi.utils.kron3c(X)
    assert X3.ndim == 2
    assert X3.shape[0] == n*(n+1)*(n+2)//6
    assert X3.shape[1] == n
    for i in range(n):
        assert np.allclose(X3[i*(i+1)*(i+2)//6:(i+1)*(i+2)*(i+3)//6],
                            X[i]*roi.utils.kron2c(X[:i+1]))


def test_kron3c(n_tests=50):
    """Test utils.kron3c()."""
    # Try with bad input.
    with pytest.raises(ValueError) as exc:
        roi.utils.kron3c(np.random.random((2,4,3)))
    assert exc.value.args[0] == "x must be one- or two-dimensional"

    # Correct inputs.
    for n in np.random.randint(2, 30, n_tests):
        _test_kron3c_single_vector(n)
        _test_kron3c_single_matrix(n)


# utils.expand_Hc() -----------------------------------------------------------
def _test_expand_Hc_single(r):
    """Do one test of utils.expand_Hc()."""
    x = np.random.random(r)

    # Do a valid expand_Hc() calculation and check dimensions.
    s = r*(r+1)//2
    Hc = np.random.random((r,s))
    H = roi.utils.expand_Hc(Hc)
    assert H.shape == (r,r**2)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(Hc @ roi.utils.kron2c(x), Hxx)

    # Check properties of the tensor for H.
    Htensor = H.reshape((r,r,r))
    assert np.allclose(Htensor @ x @ x, Hxx)
    for subH in H:
        assert np.allclose(subH, subH.T)


def test_expand_Hc(n_tests=100):
    """Test utils.expand_Hc()."""
    # Try to do expand_Hc() with a bad second dimension.
    r = 5
    sbad = r*(r+3)//2
    Hc = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        roi.utils.expand_Hc(Hc)
    assert exc.value.args[0] == \
        f"invalid shape (r,s) = {(r,sbad)} with s != r(r+1)/2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_expand_Hc_single(r)


# utils.compress_H() ----------------------------------------------------------
def _test_compress_H_single(r):
    """Do one test of utils.compress_H()."""
    x = np.random.random(r)

    # Do a valid compress_H() calculation and check dimensions.
    H = np.random.random((r,r**2))
    s = r*(r+1)//2
    Hc = roi.utils.compress_H(H)
    assert Hc.shape == (r,s)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(Hxx, Hc @ roi.utils.kron2c(x))

    # Check that expand_Hc() and compress_H() are inverses up to symmetry.
    H2 = roi.utils.expand_Hc(Hc)
    Ht = H.reshape((r,r,r))
    Htnew = np.empty_like(Ht)
    for l in range(r):
        Htnew[l] = (Ht[l] + Ht[l].T) / 2
    assert np.allclose(H2, Htnew.reshape(H.shape))


def test_compress_H(n_tests=100):
    """Test utils.compress_H()."""
    # Try to do compress_H() with a bad second dimension.
    r = 5
    r2bad = r**2 + 1
    H = np.random.random((r, r2bad))
    with pytest.raises(ValueError) as exc:
        roi.utils.compress_H(H)
    assert exc.value.args[0] == \
        f"invalid shape (r,a) = {(r,r2bad)} with a != r**2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_compress_H_single(r)


# utils.expand_Gc() -----------------------------------------------------------
def _test_expand_Gc_single(r):
    """Do one test of utils.expand_Gc()."""
    x = np.random.random(r)

    # Do a valid expand_Hc() calculation and check dimensions.
    s = r*(r+1)*(r+2)//6
    Gc = np.random.random((r,s))
    G = roi.utils.expand_Gc(Gc)
    assert G.shape == (r,r**3)

    # Check that Gc(x^3) == G(x⊗x⊗x).
    Gxxx = G @ np.kron(x,np.kron(x,x))
    assert np.allclose(Gc @ roi.utils.kron3c(x), Gxxx)

    # Check properties of the tensor for G.
    Gtensor = G.reshape((r,r,r,r))
    assert np.allclose(Gtensor @ x @ x @ x, Gxxx)
    for subG in G:
        assert np.allclose(subG, subG.T)


def test_expand_Gc(n_tests=50):
    """Test utils.expand_Gc()."""
    # Try to do expand_Gc() with a bad second dimension.
    r = 5
    sbad = r*(r+1)*(r+3)//6
    Gc = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        roi.utils.expand_Gc(Gc)
    assert exc.value.args[0] == \
        f"invalid shape (r,s) = {(r,sbad)} with s != r(r+1)(r+2)/6"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 30, n_tests):
        _test_expand_Gc_single(r)


# utils.compress_G() ----------------------------------------------------------
def _test_compress_G_single(r):
    """Do one test of utils.compress_G()."""
    x = np.random.random(r)

    # Do a valid compress_G() calculation and check dimensions.
    G = np.random.random((r,r**3))
    s = r*(r+1)*(r+2)//6
    Gc = roi.utils.compress_G(G)
    assert Gc.shape == (r,s)

    # Check that Gc(x^3) == G(x⊗x⊗x).
    Gxxx = G @ np.kron(x,np.kron(x,x))
    assert np.allclose(Gxxx, Gc @ roi.utils.kron3c(x))

    # Check that expand_Gc() and compress_G() are "inverses."
    assert np.allclose(Gc, roi.utils.compress_G(roi.utils.expand_Gc(Gc)))


def test_compress_G(n_tests=50):
    """Test utils.compress_G()."""
    # Try to do compress_H() with a bad second dimension.
    r = 5
    r3bad = r**3 + 1
    G = np.random.random((r, r3bad))
    with pytest.raises(ValueError) as exc:
        roi.utils.compress_G(G)
    assert exc.value.args[0] == \
        f"invalid shape (r,a) = {(r,r3bad)} with a != r**3"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 30, n_tests):
        _test_compress_G_single(r)
