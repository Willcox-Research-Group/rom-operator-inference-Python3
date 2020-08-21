# utils/_solver.py
"""Operator Inference least-squares solver."""

__all__ = [
            "get_least_squares_size",
            "lstsq_reg",
          ]

import types
import warnings
import itertools
import numpy as np
import scipy.linalg as la


def get_least_squares_size(modelform, r, m=0, affines=None):
    """Calculate the number of columns in the operator matrix O in the Operator
    Inference least-squares problem.

    Parameters
    ---------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
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
        * 'G': Cubic state matrix G(µ).
        * 'B': linear Input matrix B(µ).
        For example, if the constant term has the affine structure
        c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

    Returns
    -------
    ncols : int
        The number of columns in the Operator Inference least-squares problem.
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
    qG = len(affines['G']) if 'G' in affines else 1 if 'G' in modelform else 0
    qB = len(affines['B']) if 'B' in affines else 1 if 'B' in modelform else 0

    return qc + qA*r + qH*r*(r+1)//2 + qG*r*(r+1)*(r+2)//6 + qB*m


def lstsq_reg(A, b, P=0):
    """Solve the l2-norm Tikhonov-regularized ordinary least-squares problem

        min_{x} ||Ax - b||_2^2 + ||Px||_2^2

    by solving the equivalent ordinary least-squares problem

                || [ A ]    _  [ b ] ||^2
        min_{x} || [ P ] x     [ 0 ] ||_2,

    with scipy.linalg.lstsq() (equivalent to numpy.linalg.lstsq()).
    See https://docs.scipy.org/doc/scipy/reference/linalg.html.

    Parameters
    ----------
    A : (k,d) ndarray
        The "left-hand side" matrix.

    b : (k,) or (k,r) ndarray
        The "right-hand side" vector. If a two-dimensional array, then r
        independent least-squares problems are solved.

    P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
        Tikhonov regularization factor(s). The regularization matrix in the
        least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,d) ndarray: the matrix `P`.
        * list of r floats or (d,d) ndarrays: the jth entry in the list is the
            regularization factor for the jth column of `b`. Only valid if `b`
            is two-dimensional and has r columns.

    Returns
    -------
    x : (d,) or (d,r) ndarray
        The least-squares solution. If `b` is a two-dimensional array, then
        each column is a solution to the regularized least-squares problem with
        the corresponding column of b.

    residual : float or (r,) ndarray
        The residual of the regularized least-squares problem. If `b` is a
        two-dimensional array, then an array of residuals are returned that
        correspond to the columns of b.

    rank : int
        Effective rank of `A`.

    s : (min(k, d),) ndarray or None
        Singular values of `A`.
    """
    # Check dimensions of b.
    if b.ndim not in {1,2}:
        raise ValueError("`b` must be one- or two-dimensional")
    k,d = A.shape

    # If P is a sequence, decouple the problem by column.
    if isinstance(P, (list, tuple, range, types.GeneratorType)):
        # Check that the problem can be properly decoupled.
        if b.ndim != 2:
            raise ValueError("`b` must be two-dimensional with multiple P")
        r = b.shape[1]
        if hasattr(P, "__len__") and len(P) != r:
            raise ValueError("multiple P requires exactly r entries "
                             "with r = number of columns of b")

        # Solve each independent problem (iteratively for now).
        argszip = zip(itertools.repeat(A), b.T, P)
        result = [lstsq_reg(*args) for args in argszip]
        if len(result) != r:
            raise ValueError("multiple P requires exactly r entries "
                             "with r = number of columns of b")

        # Unpack and return the results.
        X = np.empty((d,r))
        residuals = np.empty(r)
        for j,(x, res, rnk, ss) in enumerate(result):
            X[:,j] = x
            residuals[j] = 0 if isinstance(res ,np.ndarray) else res
        rank, s = result[0][-2:]
        # TODO: better treatment of rank, s
        return X, residuals, rank, s

    # If P is a scalar, construct the default regularization matrix P*I.
    if np.isscalar(P):
        # Default case: fall back to default scipy.linalg.lstsq().
        if P == 0:
            if k < d:   # Warn the user if the system is underdetermined.
                warnings.warn("least squares system is underdetermined",
                               la.LinAlgWarning, stacklevel=2)
            return la.lstsq(A, b)
        elif P < 0:
            raise ValueError("regularization parameter must be nonnegative")
        P = np.diag(np.full(d, P))                # regularizer * identity
    if P.shape != (d,d):
        raise ValueError("P must be (d,d) with d = number of columns of A")

    pad = np.zeros(d) if b.ndim == 1 else np.zeros((d,b.shape[1]))
    lhs = np.vstack((A, P))
    rhs = np.concatenate((b, pad))

    return la.lstsq(lhs, rhs)
