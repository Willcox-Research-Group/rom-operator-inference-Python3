# opinf_helper.py
"""Utility functions for the operator inference."""

import numpy as _np
import warnings as _warnings
from scipy import linalg as _la


# Least squares solver ========================================================
def get_least_squares_size(modelform, r, m=0, affines=None):
    """Calculate the number of columns in the operator matrix O in the Operator
    Inference least squares problem.

    Parameters
    ---------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    r : int
        The dimension of the reduced order model.

    m : int
        The dimension of the inputs of the model.
        Must be zero unless 'B' is in `modelform`.

    affines : dict(str -> list(callables))
        Functions that define the structures of the affine operators.
        Keys must match the modelform:
        * 'c': Constant term c(µ).
        * 'A': Linear state matrix A(µ).
        * 'H': Quadratic state matrix H(µ).
        * 'B': linear Input matrix B(µ).
        For example, if the constant term has the affine structure
        c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

    Returns
    -------
    ncols : int
        The number of columns in the Operator Inference least squares problem.
    """
    has_inputs = 'B' in modelform
    if has_inputs and m == 0:
        raise ValueError(f"argument m > 0 required since 'B' in modelform")
    if not has_inputs and m != 0:
        raise ValueError(f"argument m={m} invalid since 'B' in modelform")

    if affines is None:
        affines = {}

    qc = len(affines['c']) if 'c' in affines else 1 if 'c' in modelform else 0
    qA = len(affines['A']) if 'A' in affines else 1 if 'A' in modelform else 0
    qH = len(affines['H']) if 'H' in affines else 1 if 'H' in modelform else 0
    qB = len(affines['B']) if 'B' in affines else 1 if 'B' in modelform else 0

    return qc + qA*r + qH*r*(r+1)//2 + qB*m


def lstsq_reg(A, b, P=0):
    """Solve the l2- (Tikhonov)-regularized ordinary least squares problem

        min_{x} ||Ax - b||_2^2 + ||Px||_2^2

    by solving the equivalent ordinary least squares problem

                || [ A ]    _  [ b ] ||^2
        min_{x} || [ P ] x     [ 0 ] ||_2,

    with scipy.linalg.lstsq().
    See https://docs.scipy.org/doc/scipy/reference/linalg.html.

    Parameters
    ----------
    A : (k,d) ndarray
        The "left-hand side" matrix.

    b : (k,) or (k,r) ndarray
        The "right-hand side" vector. If a two-dimensional array, then r
        independent least squares problems are solved.

    P : float >= 0, (d,d) ndarray, or list of r (d,d) ndarrays
        The Tikhonov regularization matrix or matrices, in one of the
        following formats:
        * float > 0: P * I (a scaled identity matrix) is the regularization
            matrix.
        * (d,d) ndarray: P is the regularization matrix.
        * list of r (d,d) ndarrays: the jth matrix in the list is the regularization
          matrix for the jth column of b. Only valid if b is two-dimensional.

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
    k,d = A.shape
    if k < d:
        _warnings.warn("least squares system is underdetermined",
                       _la.LinAlgWarning)

    # If P is a list of ndarrays, decouple the problem by column.
    if isinstance(P, list) or isinstance(P, tuple):
        if b.ndim != 2:
            raise ValueError("`b` must be two-dimensional with multiple P")
        r = b.shape[1]
        if len(P) != r:
            raise ValueError(
                "list P must have r entries with r = number of columns of b")
        X = _np.empty((d,r))
        residuals = _np.empty(r)
        for j in range(r):
            X[:,j], residuals[j], rank, s = lstsq_reg(A, b[:,j], P[j])
        return X, residuals, rank, s

    # If P is a scalar, construct the default regularization matrix P*I.
    if _np.isscalar(P):
        if P == 0:
            # Default case: fall back to default scipy.linalg.lstsq().
            return _la.lstsq(A, b)
        elif P < 0:
            raise ValueError("regularization parameter must be nonnegative")
        P = _np.diag(_np.full(d, P))                # regularizer * identity
    if P.shape != (d,d):
        raise ValueError("P must be (d,d) with d = number of columns of A")

    pad = _np.zeros(d) if b.ndim == 1 else _np.zeros((d,b.shape[1]))
    lhs = _np.vstack((A, P))
    rhs = _np.concatenate((b, pad))

    return _la.lstsq(lhs, rhs)


# Kronecker products ==========================================================
def kron_compact(x):
    """Calculate the unique terms of the Kronecker product x ⊗ x.

    Parameters
    ----------
    x : (n,) or (n,k) ndarray
        If two-dimensional, the product is computed column-wise (Khatri-Rao).

    Returns
    -------
    x ⊗ x : (n(n+1)/2,) or (n(n+1)/2,k) ndarray
        The "compact" Kronecker product of x with itself.
    """
    if x.ndim not in (1,2):
        raise ValueError("x must be one- or two-dimensional")
    return _np.concatenate([x[i]*x[:i+1] for i in range(x.shape[0])], axis=0)


def kron_col(x, y):
    """Calculate the full column-wise Kronecker (Khatri-Rao) product x ⊗ y.

    Parameters
    ----------
    x : (n,) or (n,k) ndarray
        If two-dimensional, the product is computed column-wise (Khatri-Rao).

    y : (m,) or (m,k) ndarray
        Must have the same number of dimensions as x. If two-dimensional, must
        have the same number of columns as x.

    Returns
    -------
    x ⊗ y : (nm,) or (nm,k) ndarray
        The full Kronecker product of x and y, by column if two-dimensional.
    """
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions")
    if x.ndim == 1:
        return _np.kron(x,y)
    elif x.ndim == 2:
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same number of columns")
        return _np.column_stack([_np.kron(xcol,ycol)
                                 for xcol,ycol in zip(x.T, y.T)])
    else:
        raise ValueError("x and y must be one- or two-dimensional")


# Matricized tensor management ================================================
def compress_H(H):
    """Calculate the matricized quadratic operator that operates on the compact
    Kronecker product.

    Parameters
    ----------
    H : (r,r**2) ndarray
        The matricized quadratic tensor that operates on the Kronecker product.
        This should be a symmetric operator in the sense that each layer of
        H.reshape((r,r,r)) is a symmetric (r,r) matrix, but it is not required.

    Returns
    -------
    Hc : (r,s) ndarray
        The matricized quadratic tensor that operates on the COMPACT Kronecker
        product. Here s = r * (r+1) / 2.
    """
    r = H.shape[0]
    r2 = H.shape[1]
    if r2 != r**2:
        raise ValueError(f"invalid shape (r,a) = {(r,r2)} with a != r**2")
    s = r * (r+1) // 2
    Hc = _np.zeros((r, s))

    fj = 0
    for i in range(r):
        for j in range(i+1):
            if i == j:      # Place column for unique term.
                Hc[:,fj] = H[:,(i*r)+j]
            else:           # Combine columns for repeated terms.
                fill = H[:,(i*r)+j] + H[:,(j*r)+i]
                Hc[:,fj] = fill
            fj += 1

    return Hc


def expand_Hc(Hc):
    """Calculate the matricized quadratic operator that operates on the full
    Kronecker product.

    Parameters
    ----------
    Hc : (r,s) ndarray
        The matricized quadratic tensor that operates on the COMPACT Kronecker
        product. Here s = r * (r+1) / 2.

    Returns
    -------
    H : (r,r**2) ndarray
        The matricized quadratic tensor that operates on the full Kronecker
        product. This is a symmetric operator in the sense that each layer of
        H.reshape((r,r,r)) is a symmetric (r,r) matrix.
    """
    r,s = Hc.shape
    if s != r*(r+1)//2:
        raise ValueError(f"invalid shape (r,s) = {(r,s)} with s != r(r+1)/2")

    H = _np.zeros((r,r**2))
    fj = 0
    for i in range(r):
        for j in range(i+1):
            if i == j:      # Place column for unique term.
                H[:,(i*r)+j] = Hc[:,fj]
            else:           # Distribute columns for repeated terms.
                fill = Hc[:,fj] / 2
                H[:,(i*r)+j] = fill
                H[:,(j*r)+i] = fill
            fj += 1

    return H
