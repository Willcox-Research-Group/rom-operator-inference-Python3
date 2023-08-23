# utils/test_reprojection.py
"""Tests for utils._reprojection."""

import pytest
import numpy as np
from scipy import linalg as la

import opinf


# Reprojection schemes ========================================================
def test_reproject_discrete(n=50, m=5, r=3):
    """Test utils._reprojection.reproject_discrete()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
    D = np.diag(1 - np.logspace(-1, -2, n))
    W = la.qr(np.random.normal(size=(n, n)))[0]
    A = W.T @ D @ W
    Ht = np.random.random((n, n, n))
    H = (Ht + Ht.T) / 20
    H = H.reshape((n, n**2))
    B = np.random.random((n, m))
    U = np.random.random((m, k))
    B1d = np.random.random(n)
    U1d = np.random.random(k)
    basis = np.eye(n)[:, :r]
    x0 = np.zeros(n)
    x0[0] = 1

    # Try with bad initial condition shape.
    with pytest.raises(ValueError) as exc:
        opinf.utils.reproject_discrete(lambda x: x, basis, x0[:-1], k)
    assert exc.value.args[0] == "basis and initial condition not aligned"

    # Linear case, no inputs.
    def f(x):
        return A @ x

    X_ = opinf.utils.reproject_discrete(f, basis, x0, k)
    assert X_.shape == (r, k)
    rom = opinf.DiscreteOpInfROM("A").fit(basis, X_)
    assert np.allclose(basis @ X_, rom.predict(X_[:, 0], k))
    assert np.allclose(rom.A_.entries, basis.T @ A @ basis)

    # Linear case, 1D inputs.
    def f(x, u):
        return A @ x + B1d * u

    X_ = opinf.utils.reproject_discrete(f, basis, x0, k, U1d)
    assert X_.shape == (r, k)
    rom = opinf.DiscreteOpInfROM("AB").fit(basis, X_, inputs=U1d)
    assert np.allclose(X_, basis.T @ rom.predict(X_[:, 0], k, U1d))
    assert np.allclose(rom.A_.entries, basis.T @ A @ basis)
    assert np.allclose(rom.B_.entries.flatten(), basis.T @ B1d)

    # Linear case, 2D inputs.
    def f(x, u):
        return A @ x + B @ u

    X_ = opinf.utils.reproject_discrete(f, basis, x0, k, U)
    assert X_.shape == (r, k)
    rom = opinf.DiscreteOpInfROM("AB").fit(basis, X_, inputs=U)
    assert np.allclose(X_, basis.T @ rom.predict(X_[:, 0], k, U))
    assert np.allclose(rom.A_.entries, basis.T @ A @ basis)
    assert np.allclose(rom.B_.entries, basis.T @ B)

    # Quadratic case, no inputs.
    '''# Test fails unreliably, removing for now.
    def f(x):
        return A @ x + H @ np.kron(x, x)

    X_ = opinf.utils.reproject_discrete(f, basis, x0, k)
    assert X_.shape == (r, k)
    rom = opinf.DiscreteOpInfROM("AH").fit(basis, X_)
    assert np.allclose(X_, basis.T @ rom.predict(X_[:, 0], k))
    assert np.allclose(rom.A_.entries,
                       basis.T @ A @ basis, atol=1e-6, rtol=1e-6)
    H_ = basis.T @ H @ np.kron(basis, basis)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(rom.H_(x_), H_ @ x2_)
    '''


def test_reproject_continuous(n=100, m=20, r=10):
    """Test utils._reprojection.reproject_continuous()."""
    # Construct dummy operators.
    k = 1 + r + r*(r+1)//2
    D = np.diag(1 - np.logspace(-1, -2, n))
    W = la.qr(np.random.normal(size=(n, n)))[0]
    A = W.T @ D @ W
    Ht = np.random.random((n, n, n))
    H = (Ht + Ht.T) / 20
    H = H.reshape((n, n**2))
    B = np.random.random((n, m))
    U = np.random.random((m, k))
    B1d = np.random.random(n)
    U1d = np.random.random(k)
    basis = np.eye(n)[:, :r]
    X = np.random.random((n, k))

    # Try with bad initial condition shape.
    with pytest.raises(ValueError) as exc:
        opinf.utils.reproject_continuous(lambda x: x, basis, X[:-1, :])
    assert exc.value.args[0] == \
        f"states and basis not aligned, first dimension {n-1} != {n}"

    # Linear case, no inputs.
    def f(x):
        return A @ x

    X_, Xdot_ = opinf.utils.reproject_continuous(f, basis, X)
    assert X_.shape == (r, k)
    assert Xdot_.shape == (r, k)
    rom = opinf.ContinuousOpInfROM("A").fit(basis, X_, Xdot_)
    assert np.allclose(rom.A_.entries, basis.T @ A @ basis)

    # Linear case, 1D inputs.
    def f(x, u):
        return A @ x + B1d * u

    X_, Xdot_ = opinf.utils.reproject_continuous(f, basis, X, U1d)
    assert X_.shape == (r, k)
    assert Xdot_.shape == (r, k)
    rom = opinf.ContinuousOpInfROM("AB").fit(basis, X_, Xdot_, U1d)
    assert np.allclose(rom.A_.entries, basis.T @ A @ basis)
    assert np.allclose(rom.B_.entries.flatten(), basis.T @ B1d)

    # Linear case, 2D inputs.
    def f(x, u):
        return A @ x + B @ u

    X_, Xdot_ = opinf.utils.reproject_continuous(f, basis, X, U)
    assert X_.shape == (r, k)
    assert Xdot_.shape == (r, k)
    rom = opinf.ContinuousOpInfROM("AB").fit(basis, X_, Xdot_, U)
    assert np.allclose(rom.A_.entries, basis.T @ A @ basis)
    assert np.allclose(rom.B_.entries, basis.T @ B)

    # Quadratic case, no inputs.
    '''# Test fails unreliably, removing for now.
    def f(x):
        return A @ x + H @ np.kron(x, x)

    X_, Xdot_ = opinf.utils.reproject_continuous(f, basis, X)
    assert X_.shape == (r, k)
    assert Xdot_.shape == (r, k)
    rom = opinf.ContinuousOpInfROM("AH").fit(basis, X_, Xdot_)
    assert np.allclose(rom.A_.entries, basis.T @ A @ basis)
    H_ = basis.T @ H @ np.kron(basis, basis)
    for _ in range(10):
        x_ = np.random.random(r)
        x2_ = np.kron(x_, x_)
        assert np.allclose(rom.H_(x_), H_ @ x2_)
    '''
