# opinf_helper.py
"""Utility functions for the operator inference."""

import numpy as np
from scipy import linalg as la


def lstsq_reg(A, b, reg=0):
    """Solve the l2- (Tikhonov)-regularized ordinary least squares problem

        min_{x} ||Ax - b||_2^2 + reg*||x||_2^2

    by solving the equivalent ordinary least squares problem

                || [   A   ]    _  [ b ] ||^2
        min_{x} || [ reg*I ] x     [ 0 ] ||_2,

    with scipy.linalg.lstsq().
    See https://docs.scipy.org/doc/scipy/reference/linalg.html.

    Parameters
    ----------
    A : (k,d) ndarray
        The "left-hand side" matrix.

    b : (k,) or (k,r) ndarray
        The "right-hand side" vector. If `b` is a two-dimensional array, then r
        independent least squares problems are solved.

    reg : float
        The l2 (Tikhonov) regularization factor.

    Returns
    -------
    x : (d,) or (d,r) ndarray
        The least squares solution. If `b` is a two-dimensional array, then
        each column is a solution to the regularized least squares problem with
        the corresponding column of b.

    residual : float or (r,) ndarray
        The residual of the regularized least squares problem. If `b` is a
        two-dimensional array, then an array of residuals are returned that
        correspond to the columns of b.

    rank : int
        Effective rank of `A`.

    s : (min(k, d),) ndarray or None
        Singular values of `A`.
    """
    if reg < 0:
        raise ValueError("regularization parameter must be nonnegative")
    if b.ndim not in {1,2}:
        raise ValueError("parameter `b` must be one- or two-dimensional")

    # No regularization -> do ordinary least squares.
    if reg == 0 or reg is None:
        return la.lstsq(A, b)

    d = A.shape[1]
    reg_I = np.diag(np.full(d, np.sqrt(reg)))           # reg * identity
    pad = np.zeros(d) if b.ndim == 1 else np.zeros((d,b.shape[1]))
    lhs = np.vstack((A, reg_I))
    rhs = np.concatenate((b, pad))

    return la.lstsq(lhs, rhs)


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
            if i == j:      # Place column for unique term.
                H[:,(i*r)+j] = F[:,fj]
            else:           # Distribute columns for repeated terms.
                fill = F[:,fj] / 2
                H[:,(i*r)+j] = fill
                H[:,(j*r)+i] = fill
            fj += 1

    return H
