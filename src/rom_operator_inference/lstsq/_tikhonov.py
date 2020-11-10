# lstsq/_tikhonov.py
"""Operator Inference least-squares solvers with Tikhonov regularization."""

__all__ = [
            "SolverL2",
            "SolverTikhonov",
            "SolverTikhonovDecoupled",
            "solver",
            "solve",
          ]

import types
import warnings
import numpy as np
import scipy.linalg as la


# Solver classes ==============================================================
class _BaseSolver:
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


class SolverL2(_BaseSolver):
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
            * if `b` is two-dimensional: ||AX - B||_F^2

        residual : float
            Residual of the regularized problem:
            * if `b` is one-dimensional: ||Ax - b||_2^2 + ||λx||_2^2
            * if `b` is two-dimensional: ||AX - B||_F^2 + ||λX||_F^2

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


class SolverTikhonov(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem with Tikhonov
    regularization:

        min_{x} ||Ax - b||_2^2 + ||Px||_2^2,    P > 0 (SPD matrix).

    If b is two-dimensional, the problem is solved in the Frobenius norm:

        min_{X} ||AX - B||_F^2 + ||PX||_F^2,    P > 0 (SPD matrix).
    """
    def __init__(self, compute_extras=True, check_regularizer=False):
        """Set behavior parameters.

        Parameters
        ----------
        compute_extras : bool
            If True, predict() returns residual / conditioning information in
            addition to the solution; if False, predict() returns the solution.

        check_regularizer : bool
            If True, ensure that a regularization matrix is full rank (via
            numpy.linalg.matrix_rank()) before attempting to solve the
            corresponding least-squares problem. Expensive for large problems.
        """
        _BaseSolver.__init__(self, compute_extras)
        self.check_regularizer = check_regularizer

    def fit(self, A, b):
        """Prepare to solve the least-squares problem via the normal equations.

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
        self._rhs = A.T @ b
        self._AtA = A.T @ A

        # Save what is needed for extra outputs if desired.
        if self.compute_extras:
            self._A = A
            self._b = b
            self._cond = np.linalg.cond(A)

        return self

    def _process_regularizer(self, P):
        """Validate the type and shape of the regularizer and compute P.T P."""
        # TODO: allow sparse P.
        if not isinstance(P, np.ndarray):
            raise ValueError("regularization matrix must be a NumPy array")

        # One-dimensional input (diagonals of the regularization matrix).
        if P.shape == (self.d,):
            if np.any(P <= 0):
                raise ValueError("diagonal P must be positive definite")
            return np.diag(P**2)

        # Two-dimensional input (the regularization matrix).
        elif P.shape == (self.d,self.d):
            if self.check_regularizer and np.linalg.matrix_rank(P) != self.d:
                raise ValueError("regularizer P is rank deficient")
            return P.T @ P

        # Anything else is invalid.
        else:
            raise ValueError("P.shape != (d,d) or (d,) where d = A.shape[1]")

    def predict(self, P):
        """Solve the least-squares problem with regularization matrix P.

        Parameters
        ----------
        P : (d,d) or (d,) ndarray
            Regularization matrix (or the diagonals of the regularization
            matrix if one-dimensional).

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
        # Construct and solve the augmented problem.
        lhs = self._AtA + self._process_regularizer(P)
        x = la.solve(lhs, self._rhs, assume_a="pos")

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            misfit = np.sum((self._A @ x - self._b)**2) # ||Ax-b||^2
            Px = P * x if P.ndim == 1 else P @ x
            residual = misfit + np.sum(Px**2)           # ||Ax-b||^2 + ||Px||^2
            regcond = np.sqrt(np.linalg.cond(lhs))      # cond([A.T | P.T].T)

            return x, misfit, residual, self._cond, regcond

        return x


class SolverTikhonovDecoupled(SolverTikhonov):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but a different Tikhonov regularizer,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2.
    """
    def fit(self, A, B):
        """Prepare to solve the least-squares problem via the normal equations.

        Parameters
        ----------
        A : (k,d) ndarray
            The "left-hand side" matrix.

        B : (k,r) ndarray
            The "right-hand side" matrix. Each column of B defines a separate
            least-squares problem.
        """
        if B.ndim != 2:
            raise ValueError("`B` must be two-dimensional")
        self.r = B.shape[1]
        return SolverTikhonov.fit(self, A, B)

    def predict(self, Ps):
        """Solve the least-squares problem with regularization matrices Ps.

        Parameters
        ----------
        P : sequence of r (d,d) or (d,) ndarrays
            Regularization matrices (or the diagonals of the regularization
            matrices if one-dimensional), one for each column of B.

        Returns
        -------
        X : (d,r) ndarray
            Least-squares solution; each column is a solution to the
            problem with the corresponding column of B.

        **If compute_extras is True, the following are also returned.**

        misfit: float
            Residual (data misfit) of the raw problem: ||AX - B||_F^2

        residuals : list of r floats
            Residuals of the regularized problems:
            ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2.

        cond : float
            Condition number of A, σ_max(A) / σ_min(A).

        regconds : list of r floats
            Condition numbers of regularized A, σ_max(G) / σ_min(G) where
            G = [A.T | P.T].T is the augmented data matrix.
        """
        if hasattr(Ps, "__len__") and len(Ps) != self.r:
            raise ValueError("len(Ps) != number of columns of B")

        # Allocate space for the solution and initialize extras if desired.
        X = np.empty((self.d,self.r))
        if self.compute_extras:
            misfit = 0
            residuals = []
            regconds = []

        # Solve each independent problem (iteratively for now).
        for j, P, rhs in zip(range(self.r), Ps, self._rhs.T):
            # Construct and solve the augmented problem.
            lhs = self._AtA + self._process_regularizer(P)
            X[:,j] = la.solve(lhs, self._rhs[:,j], assume_a="pos")

            # Record extras if desired.
            if self.compute_extras:
                misfit += np.sum((self._A @ X[:,j] - self._b[:,j])**2)
                Px = P * X[:,j] if P.ndim == 1 else P @ X[:,j]
                residuals.append(misfit + np.sum(Px**2))
                regconds.append(np.sqrt(np.linalg.cond(lhs)))

        # Compute data misfit if desired.
        if self.compute_extras:
            return X, misfit, residuals, self._cond, regconds

        return X


# Convenience functions =======================================================
def solver(A, b, P, **kwargs):
    """Select and initialize an appropriate solver for the ordinary least-
    squares problem with Tikhonov regularization,

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

    **kwargs
        Additional arguments for the solver object.
        * compute_extras : bool
            If True, solver.predict() returns residual / conditioning
            information in addition to the solution; if False,
            solver.predict() returns the solution only.
        * check_regularizer : bool
            If True, ensure that a regularization matrix is full rank (via
            numpy.linalg.matrix_rank()) before attempting to solve the
            corresponding least-squares problem. Expensive for large problems.

    Returns
    -------
    solver
        Least-squares solver object, with a predict() method mapping the
        regularization factor to the least-squares solution.
    """
    isarray = isinstance(P, np.ndarray)

    # If P is a scalar, solve the corresponding L2 problem.
    if np.isscalar(P):
        if "compute_extras" in kwargs:                  # pragma: no cover
            kwargs.pop("compute_extras")
        solver = SolverL2(**kwargs)

    # If P is a single matrix, solve the corresponding Tikhonov problem.
    elif isarray and (P.shape == (A.shape[1],) or P.ndim == 2):
        solver = SolverTikhonov(**kwargs)

    # If P is a sequence, decouple the problem by column.
    elif isinstance(P, (np.ndarray, list, types.GeneratorType)):
        solver = SolverTikhonovDecoupled(**kwargs)

    else:
        raise ValueError(f"invalid input P of type '{type(P).__name__}'")

    return solver.fit(A, b)


def solve(A, b, P=0, **kwargs):
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
            dimensional.

    **kwargs
        Additional arguments for the solver object.
        * compute_extras : bool
            If True, solver.predict() returns residual / conditioning
            information in addition to the solution; if False,
            solver.predict() returns the solution only.
        * check_regularizer : bool
            If True, ensure that a regularization matrix is full rank (via
            numpy.linalg.matrix_rank()) before attempting to solve the
            corresponding least-squares problem. Expensive for large problems.

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
        * if `b` is one-dimensional: ||Ax - b||_2^2 + ||Px||_2^2
        * if `b` is two-dimensional: ||Ax - b||_F^2 + ||Px||_F^2

    cond : float
        Condition number of A, σ_max(A) / σ_min(A).

    regcond : float or list of length r
        Effective condition number of regularized A.
    """
    return solver(A, b, P, **kwargs).predict(P)
