# pre/test_reprojection.py
"""Tests for rom_operator_inference.pre._reprojection.py"""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


# Reprojection schemes ========================================================
def test_reproject_discrete(n=50, m=5, r=3):
    """Test pre._reprojection.reproject_discrete()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
    I = np.eye(n)
    D = np.diag(1 - np.logspace(-1, -2, n))
    W = la.qr(np.random.normal(size=(n,n)))[0]
    A = W.T @ D @ W
    Ht = np.random.random((n,n,n))
    H = (Ht + Ht.T) / 20
    H = H.reshape((n, n**2))
    B = np.random.random((n,m))
    U = np.random.random((m,k))
    B1d = np.random.random(n)
    U1d = np.random.random(k)
    Vr = np.eye(n)[:,:r]
    x0 = np.zeros(n)
    x0[0] = 1

    # Try with bad initial condition shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.reproject_discrete(lambda x:x, Vr, x0[:-1], k)
    assert exc.value.args[0] == "basis Vr and initial condition x0 not aligned"

    # Linear case, no inputs.
    f = lambda x: A @ x
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("A").fit(Vr, X_)
    assert np.allclose(Vr @ X_, model.predict(X_[:,0], k))
    assert np.allclose(model.A_, Vr.T @ A @ Vr)

    # Linear case, 1D inputs.
    f = lambda x, u: A @ x + B1d * u
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k, U1d)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("AB").fit(Vr, X_, U1d)
    assert np.allclose(X_, Vr.T @ model.predict(X_[:,0], k, U1d))
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_.flatten(), Vr.T @ B1d)

    # Linear case, 2D inputs.
    f = lambda x, u: A @ x + B @ u
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k, U)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("AB").fit(Vr, X_, U)
    assert np.allclose(X_, Vr.T @ model.predict(X_[:,0], k, U))
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_, Vr.T @ B)

    # Quadratic case, no inputs.
    f = lambda x: A @ x + H @ np.kron(x,x)
    X_ = roi.pre.reproject_discrete(f, Vr, x0, k)
    assert X_.shape == (r,k)
    model = roi.InferredDiscreteROM("AH").fit(Vr, X_)
    assert np.allclose(X_, Vr.T @ model.predict(X_[:,0], k))
    assert np.allclose(model.A_, Vr.T @ A @ Vr, atol=1e-6, rtol=1e-6)
    H_ = Vr.T @ H @ np.kron(Vr, Vr)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(model.H_ @ roi.utils.kron2c(x_), H_ @ x2_)


def test_reproject_continuous(n=100, m=20, r=10):
    """Test pre._reprojection.reproject_continuous()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
    I = np.eye(n)
    D = np.diag(1 - np.logspace(-1, -2, n))
    W = la.qr(np.random.normal(size=(n,n)))[0]
    A = W.T @ D @ W
    Ht = np.random.random((n,n,n))
    H = (Ht + Ht.T) / 20
    H = H.reshape((n, n**2))
    B = np.random.random((n,m))
    U = np.random.random((m,k))
    B1d = np.random.random(n)
    U1d = np.random.random(k)
    Vr = np.eye(n)[:,:r]
    X = np.random.random((n,k))

    # Try with bad initial condition shape.
    with pytest.raises(ValueError) as exc:
        roi.pre.reproject_continuous(lambda x:x, Vr, X[:-1,:])
    assert exc.value.args[0] == \
        f"X and Vr not aligned, first dimension {n-1} != {n}"

    # Linear case, no inputs.
    f = lambda x: A @ x
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("A").fit(Vr, X_, Xdot_)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)

    # Linear case, 1D inputs.
    f = lambda x, u: A @ x + B1d * u
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X, U1d)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("AB").fit(Vr, X_, Xdot_, U1d)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_.flatten(), Vr.T @ B1d)

    # Linear case, 2D inputs.
    f = lambda x, u: A @ x + B @ u
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X, U)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("AB").fit(Vr, X_, Xdot_, U)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    assert np.allclose(model.B_, Vr.T @ B)

    # Quadratic case, no inputs.
    f = lambda x: A @ x + H @ np.kron(x,x)
    X_, Xdot_ = roi.pre.reproject_continuous(f, Vr, X)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    model = roi.InferredContinuousROM("AH").fit(Vr, X_, Xdot_)
    assert np.allclose(model.A_, Vr.T @ A @ Vr)
    H_ = Vr.T @ H @ np.kron(Vr, Vr)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(model.H_ @ roi.utils.kron2c(x_), H_ @ x2_)
