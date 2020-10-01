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


# Solver classes ==============================================================
class _BaseLstsqSolver:
    """Base class for least-squares solvers for problems of the form

        min_{x} ||Ax - b||^2 + ||Px||^2.
    """
    def __init__(self, compute_extras=True):
        """Set behavior parameters.

        Parameters
        ----------
        compute_extras : bool
            If True, predict() returns residual / conditioning information in
            addition to the solution; if False, predict() returns the solution.
        """
        self.compute_extras = compute_extras

    def _check_shapes(self, A, b):
        """Verify the shapes of A and b are consistent for ||Ax - b||."""
        # Record dimensions of A (and ensure A is two-dimensional).
        self.k, self.d = A.shape

        # Check dimensions of b.
        if b.ndim not in {1,2}:
            raise ValueError("`b` must be one- or two-dimensional")
        if b.shape[0] != self.k:
            raise ValueError("inputs not aligned: A.shape[0] != b.shape[0]")
        self._bndim = b.ndim


class LstsqSolverL2(_BaseLstsqSolver):
    """Solve the l2-norm ordinary least-squares problem with L2 regularization:

        min_{x} ||Ax - b||_2^2 + ||λx||_2^2,   λ > 0.

    If b is two-dimensional, the problem is solved in the Frobenius norm:

        min_{X} ||AX - B||_F^2 + ||λX||_F^2,   λ > 0.
    """
    def fit(self, A, b):
        """Take the SVD of A in preparation to solve the least-squares problem.

        Parameters
        ----------
        A : (k,d) ndarray
            The "left-hand side" matrix.

        b : (k,) or (k,r) ndarray
            The "right-hand side" vector. If a two-dimensional array, then r
            independent least-squares problems are solved.
        """
        self._check_shapes(A, b)

        # Compute the SVD of A and save what is needed to solve the problem.
        U,s,Vt = la.svd(A, full_matrices=False)
        self._V = Vt.T
        self._s = s
        self._Utb = U.T @ b

        # Save what is needed for extra outputs if desired.
        if self.compute_extras:
            self._A = A
            self._b = b
            self._cond = abs(s[0] / s[-1]) if s[-1] > 0 else np.inf

        return self

    def predict(self, λ):
        """Solve the least-squares problem with regularization parameter λ.

        Parameters
        ----------
        λ : float
            Regularization parameter.

        Returns
        -------
        x : (d,) or (d,r) ndarray
            Least-squares solution. If `b` is a two-dimensional array, then
            each column is a solution to the regularized least-squares problem
            with the corresponding column of b.

        **If compute_extras is True, the following are also returned.**

        misfit: float
            Residual (data misfit) of the non-regularized problem:
            * if `b` is one-dimensional: ||Ax - b||_2^2
            * if `b` is two-dimensional: ||Ax - b||_F^2

        residual : float
            Residual of the regularized problem:
            * if `b` is one-dimensional: ||Ax - b||_2^2 + ||λx||_2^2
            * if `b` is two-dimensional: ||Ax - b||_F^2 + ||λx||_F^2

        datacond : float
            Condition number of A, σ_max(A) / σ_min(A).

        dataregcond : float
            Effective condition number of regularized A, s_max(A) / s_min(A)
            where s(A) = (σ(A)^2 + λ^2) / σ(A).
        """
        # Check that λ is a nonnegative scalar.
        if not np.isscalar(λ):
            raise ValueError("regularization parameter must be a scalar")
        if λ < 0:
            raise ValueError("regularization parameter must be nonnegative")

        # Warn for underdeterminedness.
        if λ == 0 and self.k < self.d:
            warnings.warn("least-squares system is underdetermined "
                          "(will compute minimum-norm solution)",
                          la.LinAlgWarning, stacklevel=2)

        # Invert / filter the singular values and compute the solution.
        if λ == 0:
            Sinv = 1 / self._s                          # σinv = 1/σ
        else:
            Sinv = self._s / (self._s**2 + λ**2)        # σinv = σ/(σ^2 + λ^2)
        if self._bndim == 2:
            Sinv = Sinv.reshape((-1,1))

        x = self._V @ (Sinv * self._Utb)                # x = V Sinv U.T b

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            # Residuals (without, then with regularization)
            misfit = np.sum((self._A @ x - self._b)**2) # ||Ax-b||^2
            residual = misfit + λ**2*np.sum(x**2)       # ||Ax-b||^2 + ||λx||^2

            # Condition numbers (without, then with regularization).
            regcond = abs(Sinv.max() / Sinv.min())

            return x, misfit, residual, self._cond, regcond

        return x

class LstsqSolverTikhonov(_BaseLstsqSolver):
    """Solve the l2-norm ordinary least-squares problem with Tikhonov
    regularization:
                                              || [ A ]    _  [ b ] ||^2
        min_{x} ||Ax - b||_2^2 + ||Px||_2^2 = || [ P ] x     [ 0 ] ||_2.

    If b is two-dimensional, the problem is solved in the Frobenius norm:
                                              || [ A ]    _  [ B ] ||^2
        min_{X} ||AX - B||_F^2 + ||PX||_F^2 = || [ P ] X     [ 0 ] ||_F.
    """
    def fit(self, A, b):
        """Pad b appropriately to prepare to solve the least-squares problem.

        Parameters
        ----------
        A : (k,d) ndarray
            The "left-hand side" matrix.

        b : (k,) or (k,r) ndarray
            The "right-hand side" vector. If a two-dimensional array, then r
            independent least-squares problems are solved.
        """
        self._check_shapes(A, b)

        # Pad b and save what is needed to solve the problem.
        pad = np.zeros(self.d) if self._bndim == 1 else np.zeros((self.d,
                                                                  b.shape[1]))
        self._rhs = np.concatenate((b, pad))
        self._A = A

        # Save what is needed for extra outputs if desired.
        if self.compute_extras:
            self._b = b
            self._cond = np.linalg.cond(A)

        return self

    def predict(self, P):
        """Solve the least-squares problem with regularization parameter λ.

        Parameters
        ----------
        P : (d,d) ndarray
            Regularization matrix.

        Returns
        -------
        x : (d,) or (d,r) ndarray
            Least-squares solution. If `b` is a two-dimensional array, then
            each column is a solution to the regularized least-squares problem
            with the corresponding column of b.

        **If compute_extras is True, the following are also returned.**

        misfit: float
            Residual (data misfit) of the non-regularized problem:
            * if `b` is one-dimensional: ||Ax - b||_2^2
            * if `b` is two-dimensional: ||Ax - b||_F^2

        residual : float
            Residual of the regularized problem:
            * if `b` is one-dimensional: ||Ax - b||_2^2 + ||λx||_2^2
            * if `b` is two-dimensional: ||Ax - b||_F^2 + ||λx||_F^2

        cond : float
            Condition number of A, σ_max(A) / σ_min(A).

        regcond : float
            Condition number of regularized A, σ_max(G) / σ_min(G) where
            G = [A.T | P.T].T is the augmented data matrix.
        """
        # Validate P. # TODO: allow sparse P.
        if not isinstance(P, np.ndarray):
            raise ValueError("regularization matrix must be a NumPy array")
        if P.shape != (self.d,self.d):
            raise ValueError("P.shape != (d,d) where d = A.shape[1]")

        # Construct and solve the augmented problem.
        lhs = np.vstack((self._A, P))
        x, residuals, _, s = la.lstsq(lhs, self._rhs)

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            misfit = np.sum((self._A @ x - self._b)**2) # ||Ax-b||^2
            residual = np.sum(residuals)                # ||Ax-b||^2 + ||Px||^2
            regcond = abs(s[0] / s[-1])                 # cond([A.T | P.T].T)

            return x, misfit, residual, self._cond, regcond

        return x


def lstsq_reg(A, b, P=0):
    """Solve the l2-norm Tikhonov-regularized ordinary least-squares problem

        min_{x} ||Ax - b||_2^2 + ||Px||_2^2.

    Parameters
    ----------
    A : (k,d) ndarray
        The "left-hand side" matrix.

    b : (k,) or (k,r) ndarray
        The "right-hand side" vector. If a two-dimensional array, then r
        independent least-squares problems are solved.

    P : float >= 0, (d,) narray, (d,d) ndarray, or sequence of length r
        Tikhonov regularization parameter(s). The regularization matrix in the
        least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,) ndarray: diag(P), a diagonal matrix.
        * (d,d) ndarray: the matrix `P`.
        * sequence : the jth entry in the sequence is the regularization
            parameter for the jth column of `b`. Only valid if `b` is two-
            dimensional and has r columns.

    Returns
    -------
    x : (d,) or (d,r) ndarray
        Least-squares solution. If `b` is a two-dimensional array, then
        each column is a solution to the regularized least-squares problem
        with the corresponding column of b.

    misfit: float
        Residual (data misfit) of the non-regularized problem:
        * if `b` is one-dimensional: ||Ax - b||_2^2
        * if `b` is two-dimensional: ||Ax - b||_F^2

    residual : float
        Residual of the regularized problem:
        * if `b` is one-dimensional: ||Ax - b||_2^2 + ||λx||_2^2
        * if `b` is two-dimensional: ||Ax - b||_F^2 + ||λx||_F^2

    cond : float
        Condition number of A, σ_max(A) / σ_min(A).

    regcond : float or list of length r
        Effective condition number of regularized A.
    """
    isarray = isinstance(P, np.ndarray)

    # If P is a scalar, solve the corresponding L2 problem.
    if np.isscalar(P) or (isarray and P.shape == (A.shape[1])):
        return LstsqSolverL2(compute_extras=True).fit(A, b).predict(P)

    # If P is a single matrix, solve the corresponding Tikhonov problem.
    elif isarray and P.ndim == 2:
        return LstsqSolverTikhonov(compute_extras=True).fit(A, b).predict(P)

    # If P is a sequence, decouple the problem by column.
    elif isinstance(P, (list, tuple, range, types.GeneratorType)):
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
        X = np.empty((A.shape[1],r))
        misfit, residual = 0, 0
        regconds = []
        for j, (x, mis, res, cond, regcond) in enumerate(result):
            X[:,j] = x
            misfit += mis
            residual += res
            regconds.append(regcond)

        return X, misfit, residual, cond, regconds

    else:
        raise ValueError(f"invalid input P of type '{type(P).__name__}'")
