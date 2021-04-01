# pre/test_reprojection.py
"""Tests for rom_operator_inference.pre._reprojection.py"""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as opinf


# Reprojection schemes ========================================================
def test_reproject_discrete(n=50, m=5, r=3):
    """Test pre._reprojection.reproject_discrete()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
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
        opinf.pre.reproject_discrete(lambda x: x, Vr, x0[:-1], k)
    assert exc.value.args[0] == "basis Vr and initial condition x0 not aligned"

    # Linear case, no inputs.
    def f(x):
        return A @ x
    X_ = opinf.pre.reproject_discrete(f, Vr, x0, k)
    assert X_.shape == (r,k)
    rom = opinf.InferredDiscreteROM("A").fit(Vr, X_)
    assert np.allclose(Vr @ X_, rom.predict(X_[:,0], k))
    assert np.allclose(rom.A_, Vr.T @ A @ Vr)

    # Linear case, 1D inputs.
    def f(x, u):
        return A @ x + B1d * u
    X_ = opinf.pre.reproject_discrete(f, Vr, x0, k, U1d)
    assert X_.shape == (r,k)
    rom = opinf.InferredDiscreteROM("AB").fit(Vr, X_, U1d)
    assert np.allclose(X_, Vr.T @ rom.predict(X_[:,0], k, U1d))
    assert np.allclose(rom.A_, Vr.T @ A @ Vr)
    assert np.allclose(rom.B_.flatten(), Vr.T @ B1d)

    # Linear case, 2D inputs.
    def f(x, u):
        return A @ x + B @ u
    X_ = opinf.pre.reproject_discrete(f, Vr, x0, k, U)
    assert X_.shape == (r,k)
    rom = opinf.InferredDiscreteROM("AB").fit(Vr, X_, U)
    assert np.allclose(X_, Vr.T @ rom.predict(X_[:,0], k, U))
    assert np.allclose(rom.A_, Vr.T @ A @ Vr)
    assert np.allclose(rom.B_, Vr.T @ B)

    # Quadratic case, no inputs.
    def f(x):
        return A @ x + H @ np.kron(x,x)
    X_ = opinf.pre.reproject_discrete(f, Vr, x0, k)
    assert X_.shape == (r,k)
    rom = opinf.InferredDiscreteROM("AH").fit(Vr, X_)
    assert np.allclose(X_, Vr.T @ rom.predict(X_[:,0], k))
    assert np.allclose(rom.A_, Vr.T @ A @ Vr, atol=1e-6, rtol=1e-6)
    H_ = Vr.T @ H @ np.kron(Vr, Vr)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(rom.H_ @ opinf.utils.kron2c(x_), H_ @ x2_)


def test_reproject_continuous(n=100, m=20, r=10):
    """Test pre._reprojection.reproject_continuous()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
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
        opinf.pre.reproject_continuous(lambda x:x, Vr, X[:-1,:])
    assert exc.value.args[0] == \
        f"X and Vr not aligned, first dimension {n-1} != {n}"

    # Linear case, no inputs.
    def f(x):
        return A @ x
    X_, Xdot_ = opinf.pre.reproject_continuous(f, Vr, X)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    rom = opinf.InferredContinuousROM("A").fit(Vr, X_, Xdot_)
    assert np.allclose(rom.A_, Vr.T @ A @ Vr)

    # Linear case, 1D inputs.
    def f(x, u):
        return A @ x + B1d * u
    X_, Xdot_ = opinf.pre.reproject_continuous(f, Vr, X, U1d)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    rom = opinf.InferredContinuousROM("AB").fit(Vr, X_, Xdot_, U1d)
    assert np.allclose(rom.A_, Vr.T @ A @ Vr)
    assert np.allclose(rom.B_.flatten(), Vr.T @ B1d)

    # Linear case, 2D inputs.
    def f(x, u):
        return A @ x + B @ u
    X_, Xdot_ = opinf.pre.reproject_continuous(f, Vr, X, U)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    rom = opinf.InferredContinuousROM("AB").fit(Vr, X_, Xdot_, U)
    assert np.allclose(rom.A_, Vr.T @ A @ Vr)
    assert np.allclose(rom.B_, Vr.T @ B)

    # Quadratic case, no inputs.
    def f(x):
        return A @ x + H @ np.kron(x,x)
    X_, Xdot_ = opinf.pre.reproject_continuous(f, Vr, X)
    assert X_.shape == (r,k)
    assert Xdot_.shape == (r,k)
    rom = opinf.InferredContinuousROM("AH").fit(Vr, X_, Xdot_)
    assert np.allclose(rom.A_, Vr.T @ A @ Vr)
    H_ = Vr.T @ H @ np.kron(Vr, Vr)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(rom.H_ @ opinf.utils.kron2c(x_), H_ @ x2_)
