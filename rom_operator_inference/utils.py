# opinf_helper.py
"""Utility functions for the operator inference."""

import numpy as _np
from scipy import linalg as _la


def lstsq_reg(A, b, G=0):
    """Solve the l2- (Tikhonov)-regularized ordinary least squares problem

        min_{x} ||Ax - b||_2^2 + ||Gx||_2^2

    by solving the equivalent ordinary least squares problem

                || [ A ]    _  [ b ] ||^2
        min_{x} || [ G ] x     [ 0 ] ||_2,

    with scipy.linalg.lstsq().
    See https://docs.scipy.org/doc/scipy/reference/linalg.html.

    Parameters
    ----------
    A : (k,d) ndarray
        The "left-hand side" matrix.

    b : (k,) or (k,r) ndarray
        The "right-hand side" vector. If a two-dimensional array, then r
        independent least squares problems are solved.

    G : float > 0, (d,d) ndarray, or list of r (d,d) ndarrays
        The Tikhonov regularization matrix or matrices, in one of the
        following formats:
        * float > 0: G * I (a scaled identity matrix) is the regularization
            matrix.
        * (d,d) ndarray: G is the regularization matrix.
        * list of r (d,d) ndarrays: the jth matrix in the list is the regularization matrix for the jth column of b. Only valid if b is two-dimensional.

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
    # Check dimensions.
    if b.ndim not in {1,2}:
        raise ValueError("`b` must be one- or two-dimensional")
    d = A.shape[1]

    # If G is a list of ndarrays, decouple the problem by column.
    if isinstance(G, list) or isinstance(G, tuple):
        if b.ndim != 2:
            raise ValueError("`b` must be two-dimensional with multiple G")
        r = b.shape[1]
        if len(G) != r:
            raise ValueError(
                "list G must have r entries with r = number of columns of b")
        X = _np.empty((d,r))
        residuals = []
        for j in range(r):
            X[:,j], res, rank, s = lstsq_reg(A, b[:,j], G[j])
            residuals.append(res)
        return X, _np.array(residuals), rank, s

    # If G is a scalar, construct the default regularization matrix G*I.
    if _np.isscalar(G):
        if G == 0:
            return _la.lstsq(A, b)
        elif G < 0:
            raise ValueError("regularization parameter must be nonnegative")
        G = _np.diag(_np.full(d, G))                # regularizer * identity
    if G.shape != (d,d):
        raise ValueError("G must be (d,d) with d = number of columns of A")

    pad = _np.zeros(d) if b.ndim == 1 else _np.zeros((d,b.shape[1]))
    lhs = _np.vstack((A, G))
    rhs = _np.concatenate((b, pad))

    return _la.lstsq(lhs, rhs)


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
    return _np.concatenate([x[i]*x[:i+1] for i in range(x.shape[0])], axis=0)


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

    H = _np.zeros((r,r**2))
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
