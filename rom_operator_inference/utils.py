# opinf_helper.py
"""Utility functions for the operator inference."""

import numpy as np
from scipy import linalg as la


def kron_compact(x):
    """Calculate the unique terms of the Kronecker product x ⊗ x.

    Parameters
    ----------
    x : (n,) ndarray

    Returns
    -------
    x ⊗ x : (n(n+1)/2,) ndarray
        The "compact" Kronecker product of x with itself.
    """
    return np.concatenate([x[i]*x[:i+1] for i in range(x.shape[0])], axis=0)


def F2H(F):
    """Calculate the matricized quadratic operator that operates on the full
    Kronecker product.

    Parameters
    ----------
    F : (r,s) ndarray
        The matricized quadratic tensor that operates on the COMPACT Kronecker
        product. Here s = r * (r+1) / 2.

    Returns
    -------
    H : (r,r**2) ndarray
        The matricized quadratic tensor that operators on the full Kronecker
        product. This is a symmetric operator in the sense that each layer of
        H.reshape((r,r,r)) is a symmetric (r,r) matrix.
    """
    r,s = F.shape
    if s != r*(r+1)//2:
        raise ValueError(f"invalid shape (r,s) = {(r,s)} with s != r(r+1)/2")

    H = np.zeros((r,r**2))
    fj = 0
    for i in range(r):
        for j in range(i+1):
            if i == j:      # Place column corresonding to unique term.
                H[:,(i*r)+j] = F[:,fj]
            else:           # Distribute columns corresponding to repeated terms.
                fill = F[:,fj] / 2
                H[:,(i*r)+j] = fill
                H[:,(j*r)+i] = fill
            fj += 1

    return H


def normal_equations(D, r, reg, num):
    """Solve the normal equations corresponding to the regularized ordinary
    least squares problem

    minimize ||Do - r|| + k||Fo||

    Parameters
    ----------
    D : (K, n+1+s) ndarray
        Data matrix.
    r : (K,1) ndarray
        X dot data reduced.
    k : float
        Regularization parameter
    num : int
        Index of the OLS problem we are solving (0 ≤ num < r).

    Returns
    -------
    o : (n+1+s,) ndarray
        The least squares solution.
    """
    K,rps = D.shape

    F = np.eye(rps)
    F[num,num] = 0

    pseudo = reg*F
    rhs = np.zeros((rps))

    Dplus = np.vstack((D,pseudo))
    Rplus = np.vstack((r.reshape((-1,1)),rhs.reshape((-1,1))))

    return la.lstsq(Dplus, Rplus)[0]
