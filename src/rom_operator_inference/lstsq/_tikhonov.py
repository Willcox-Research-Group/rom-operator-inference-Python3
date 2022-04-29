# lstsq/_tikhonov.py
"""Operator Inference least-squares solvers with Tikhonov regularization."""

__all__ = [
            "SolverL2",
            "SolverL2Decoupled",
            "SolverTikhonov",
            "SolverTikhonovDecoupled",
            "solver",
            "solve",
          ]

import warnings
import numpy as np
import scipy.linalg as la

from ._base import _BaseSolver


# Solver classes ==============================================================
class _BaseTikhonovSolver(_BaseSolver):
    """Base solver for regularized linear least-squares problems of the form

        sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2.
    """
    def __init__(self):
        """Initialize attributes."""
        self.__A, self.__B = None, None

    # Properties: matrices ----------------------------------------------------
    @property
    def A(self):
        """(k, d) ndarray: "left-hand side" data matrix."""
        return self.__A

    @A.setter
    def A(self, A):
        raise AttributeError("can't set attribute (call fit())")

    @property
    def B(self):
        """(k, r) ndarray: "right-hand side" matrix B = [ b_1 | ... | b_r ]."""
        return self.__B

    @B.setter
    def B(self, B):
        raise AttributeError("can't set attribute (call fit())")

    # Properties: matrix dimensions -------------------------------------------
    @property
    def k(self):
        """int > 0 : number of equations in the least-squares problem
        (number of rows of A).
        """
        return self.A.shape[0] if self.A is not None else None

    @property
    def d(self):
        """int  > 0 : number of unknowns to learn in each problem
        (number of columns of A).
        """
        return self.A.shape[1] if self.A is not None else None

    @property
    def r(self):
        """int > 0: number of independent least-squares problems
        (number of columns of B).
        """
        return self.B.shape[1] if self.B is not None else None

    # Validation --------------------------------------------------------------
    def _process_fit_arguments(self, A, B):
        """Verify the dimensions of A and B are consistent with the expression
        AX - B, then save A and B.

        Parameters
        ----------
        A : (k, d) ndarray
            The "left-hand side" matrix.
        B : (k, r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        # Extract the dimensions of A (and ensure A is two-dimensional).
        k, d = A.shape
        if k < d:
            warnings.warn("original least-squares system is underdetermined!",
                          la.LinAlgWarning, stacklevel=2)
        self.__A = A

        # Check dimensions of b.
        if B.ndim == 1:
            B = B.reshape((-1, 1))
        if B.ndim != 2:
            raise ValueError("`B` must be one- or two-dimensional")
        if B.shape[0] != k:
            raise ValueError("inputs not aligned: A.shape[0] != B.shape[0]")
        self.__B = B

    def _check_is_trained(self, attr=None):
        """Raise an AttributeError if fit() has not been called."""
        trained = (self.A is not None) and (self.B is not None)
        if attr is not None:
            trained *= hasattr(self, attr)
        if not trained:
            raise AttributeError("lstsq solver not trained (call fit())")

    # Post-processing ---------------------------------------------------------
    def cond(self):
        """Calculate the 2-norm condition number of the data matrix A."""
        self._check_is_trained()
        return np.linalg.cond(self.A)

    def misfit(self, X):
        """Calculate the data misfit (residual) of the non-regularized problem
        for each column of B = [ b_1 | ... | b_r ].

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.

        Returns
        -------
        resids : (r,) ndarray or float (r = 1)
            Data misfits ||Ax_i - b_i||_2^2, i = 1, ..., r.
        """
        self._check_is_trained()
        if self.r == 1 and X.ndim == 1:
            X = X.reshape((-1, 1))
        if X.shape != (self.d, self.r):
            raise ValueError(f"X.shape = {X.shape} != "
                             f"{(self.d, self.r)} = (d, r)")
        resids = np.sum((self.A @ X - self.B)**2, axis=0)
        return resids[0] if self.r == 1 else resids


class SolverL2(_BaseTikhonovSolver):
    """Solve the l2-norm ordinary least-squares problem with L2 regularization:

        sum_{i} min_{x_i} ||Ax_i - b_i||_2^2 + ||λx_i||_2^2,    λ ≥ 0,

    or, written in the Frobenius norm,

        min_{X} ||AX - B||_F^2 + ||λX||_F^2,                    λ ≥ 0.
    """
    # Validation --------------------------------------------------------------
    def _process_regularizer(self, regularizer):
        """Validate the regularization hyperparameter and return
        regularizer^2."""
        if not np.isscalar(regularizer):
            raise TypeError("regularization hyperparameter must be a scalar")
        if regularizer < 0:
            raise ValueError("regularization hyperparameter must be "
                             "non-negative")
        return regularizer**2

    # Helper methods ----------------------------------------------------------
    def _inv_svals(self, regularizer):
        """Compute the regularized inverse singular value matrix,
        Σ^* = Σ (Σ^2 + (λ^2)I)^{-1}. Note Σ^* = Σ^{-1} for λ = 0.
        """
        regularizer2 = self._process_regularizer(regularizer)
        svals = self._svals
        return 1/svals if regularizer2 == 0 else svals/(svals**2+regularizer2)

    # Main methods ------------------------------------------------------------
    def fit(self, A, B):
        """Take the SVD of A in preparation to solve the least-squares problem.

        Parameters
        ----------
        A : (k, d) ndarray
            The "left-hand side" matrix.
        B : (k, r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        self._process_fit_arguments(A, B)

        # Compute the SVD of A and save what is needed to solve the problem.
        U, svals, Vt = la.svd(self.A, full_matrices=False)
        self._V = Vt.T
        self._svals = svals
        self._UtB = U.T @ self.B

        return self

    def predict(self, regularizer):
        """Solve the least-squares problem with the non-negative scalar
        regularization hyperparameter λ.

        Parameters
        ----------
        regularizer : float ≥ 0
            Scalar regularization hyperparameter.

        Returns
        -------
        X : (d, r) or (d,) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
            The result is flattened to a one-dimensional array if r = 1.
        """
        self._check_is_trained("_V")

        svals_inv = self._inv_svals(regularizer).reshape((-1, 1))
        X = self._V @ (svals_inv * self._UtB)        # X = V svals_inv U.T B

        return np.ravel(X) if self.r == 1 else X

    # Post-processing ---------------------------------------------------------
    def cond(self):
        """Calculate the 2-norm condition number of the data matrix A."""
        self._check_is_trained("_svals")
        return abs(self._svals.max() / self._svals.min())

    def regcond(self, regularizer):
        """Compute the 2-norm condition number of the regularized data matrix.

        Parameters
        ----------
        regularizer : float ≥ 0
            Scalar regularization hyperparameter.

        Returns
        -------
        rc : float ≥ 0
            cond([A.T | λI.T].T), computed from filtered singular values of A.
        """
        self._check_is_trained("_svals")
        svals2 = self._svals**2 + self._process_regularizer(regularizer)
        return np.sqrt(svals2.max() / svals2.min())

    def residual(self, X, regularizer):
        """Calculate the residual of the regularized problem for each column of
        B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||λx_i||_2^2.

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
        regularizer : float ≥ 0
            Scalar regularization hyperparameter.

        Returns
        -------
        resids : (r,) ndarray or float (r = 1)
            Residuals ||Ax_i - b_i||_2^2 + ||λx_i||_2^2, i = 1, ..., r.
        """
        self._check_is_trained()
        regularizer2 = self._process_regularizer(regularizer)
        return self.misfit(X) + regularizer2*np.sum(X**2, axis=0)


class SolverL2Decoupled(SolverL2):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but different L2 regularizations,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2,    λ_i > 0.
    """
    # Validation --------------------------------------------------------------
    def _check_regularizers(self, regularizers):
        if len(regularizers) != self.r:
            raise ValueError("len(regularizers) != number of columns of B")

    # Main methods ------------------------------------------------------------
    def predict(self, regularizers):
        """Solve the least-squares problem with regularization hyperparameters
        regularizers.

        Parameters
        ----------
        regularizers : sequence of r floats or (r,) ndarray
            Scalar regularization hyperparameters, one for each column of B.

        Returns
        -------
        X : (d, r) or (d,) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
            The result is flattened to a one-dimensional array if r = 1.
        """
        self._check_is_trained("_V")
        self._check_regularizers(regularizers)

        # Allocate space for the solution.
        X = np.empty((self.d, self.r))

        # Solve each independent regularized lstsq problem (iteratively).
        for j, regularizer in enumerate(regularizers):
            svals_inv = self._inv_svals(regularizer)
            # X = V svals_inv U.T B
            X[:, j] = self._V @ (svals_inv * self._UtB[:, j])

        return np.ravel(X) if self.r == 1 else X

    # Post-processing ---------------------------------------------------------
    def regcond(self, regularizers):
        """Compute the 2-norm condition number of each regularized data matrix.

        Parameters
        ----------
        regularizers : sequence of r floats or (r,) ndarray
            Scalar regularization hyperparameters, one for each column of B.

        Returns
        -------
        rcs : (r,) ndarray
            cond([A.T | (λ_i I).T].T), i = 1, ..., r, computed from filtered
            singular values of the data matrix A.
        """
        self._check_is_trained("_svals")
        self._check_regularizers(regularizers)
        regularizer_2s = np.array([
            self._process_regularizer(lm) for lm in regularizers])
        svals2 = self._svals**2 + regularizer_2s.reshape((-1, 1))
        return np.sqrt(svals2.max(axis=1) / svals2.min(axis=1))

    def residual(self, X, regularizers):
        """Calculate the residual of the regularized problem for each column of
        B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2.

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
        regularizers : sequence of r floats or (r,) ndarray
            Scalar regularization hyperparameters, one for each column of B.

        Returns
        -------
        resids : (r,) ndarray
            Residuals ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2, i = 1, ..., r.
        """
        self._check_is_trained()
        self._check_regularizers(regularizers)
        regularizer_2s = np.array([
            self._process_regularizer(lm) for lm in regularizers])
        return self.misfit(X) + regularizer_2s*np.sum(X**2, axis=0)


class SolverTikhonov(_BaseTikhonovSolver):
    """Solve the l2-norm ordinary least-squares problem with Tikhonov
    regularization:

        sum_{i} min_{x_i} ||Ax_i - b_i||_2^2 + ||Px_i||_2^2,    P > 0 (SPD).

    or, written in the Frobenius norm,

        min_{X} ||AX - B||_F^2 + ||PX||_F^2,                    P > 0 (SPD).
    """
    # Validation --------------------------------------------------------------
    def _process_regularizer(self, P):
        """Validate the type and shape of the regularizer."""
        # TODO: allow sparse P.
        if not isinstance(P, np.ndarray):
            raise TypeError("regularization matrix must be a NumPy array")

        # One-dimensional input (diagonals of the regularization matrix).
        if P.shape == (self.d,):
            if np.any(P < 0):
                raise ValueError("diagonal P must be positive semi-definite")
            return np.diag(P)

        # Two-dimensional input (the regularization matrix).
        elif P.shape != (self.d, self.d):
            raise ValueError("P.shape != (d, d) or (d,) where d = A.shape[1]")

        return P

    # Helper methods ----------------------------------------------------------
    def _lhs(self, P):
        """Expand P if needed and compute A.T A + P.T P, the left-hand side of
        the modified Normal equations for Tikhonov-regularized least squares.
        """
        P = self._process_regularizer(P)
        return P, self._AtA + (P.T @ P)

    # Main methods ------------------------------------------------------------
    def fit(self, A, B):
        """Prepare to solve the least-squares problem via the normal equations.

        Parameters
        ----------
        A : (k, d) ndarray
            The "left-hand side" matrix.
        B : (k, r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        self._process_fit_arguments(A, B)

        # Compute both sides of the Normal equations.
        self._rhs = self.A.T @ self.B
        self._AtA = self.A.T @ self.A

        return self

    def predict(self, P, trynormal=True):
        """Solve the least-squares problem with regularization matrix P.

        Parameters
        ----------
        P : (d, d) or (d,) ndarray
            Regularization matrix (or the diagonals of the regularization
            matrix if one-dimensional).
        trynormal : bool
            If True, attempt to solve the problem via the normal equations,
            falling back on a full least-squares solver if the problem is
            too ill-conditioned. If False, skip the normal equations attempt.

        Returns
        -------
        X : (d, r) or (d,) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
            The result is flattened to a one-dimensional array if r = 1.
        """
        self._check_is_trained("_AtA")

        P, lhs = self._lhs(P)
        if trynormal:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=la.LinAlgWarning)
                    # Attempt to solve the problem via the normal equations.
                    X = la.solve(lhs, self._rhs, assume_a="pos")
            except (la.LinAlgError, la.LinAlgWarning):
                # For ill-conditioned normal equations, use la.lstsq().
                print("normal equations solve failed, switching lstsq solver")
                trynormal = False
        if not trynormal:
            Bpad = np.vstack((self.B, np.zeros((self.d, self.r))))
            X = la.lstsq(np.vstack((self.A, P)), Bpad)[0]

        return np.ravel(X) if self.r == 1 else X

    # Post-processing ---------------------------------------------------------
    def regcond(self, P):
        """Compute the 2-norm condition number of the regularized data matrix.

        Parameters
        ----------
        P : (d, d) or (d,) ndarray
            Regularization matrix (or the diagonals of the regularization
            matrix if one-dimensional).

        Returns
        -------
        rc : float ≥ 0
            cond([A.T | P.T].T), computed as sqrt(cond(A.T A + P.T P)).
        """
        self._check_is_trained("_AtA")
        return np.sqrt(np.linalg.cond(self._lhs(P)[1]))

    def residual(self, X, P):
        """Calculate the residual of the regularized problem for each column of
        B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||Px_i||_2^2.

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
        P : (d, d) or (d,) ndarray
            Regularization matrix (or the diagonals of the regularization
            matrix if one-dimensional).

        Returns
        -------
        resids : (r,) ndarray or float (r = 1)
            Residuals ||Ax_i - b_i||_2^2 + ||Px_i||_2^2, i = 1, ..., r.
        """
        self._check_is_trained()
        P = self._process_regularizer(P)
        return self.misfit(X) + np.sum((P @ X)**2, axis=0)


class SolverTikhonovDecoupled(SolverTikhonov):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but a different Tikhonov regularizer,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2.
    """
    # Validation --------------------------------------------------------------
    def _check_Ps(self, Ps):
        """Validate Ps."""
        if len(Ps) != self.r:
            raise ValueError("len(Ps) != number of columns of B")

    # Main methods ------------------------------------------------------------
    def predict(self, Ps):
        """Solve the least-squares problems with regularization matrices Ps.

        Parameters
        ----------
        Ps : sequence of r (d, d) or (d,) ndarrays
            Regularization matrices (or the diagonals of the regularization
            matrices if one-dimensional), one for each column of B.

        Returns
        -------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
        """
        self._check_is_trained("_AtA")
        self._check_Ps(Ps)

        # Allocate space for the solution.
        X = np.empty((self.d, self.r))

        # Solve each independent problem (iteratively for now).
        Bpad = None
        for j, [P, rhs] in enumerate(zip(Ps, self._rhs.T)):
            P, lhs = self._lhs(P)
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=la.LinAlgWarning)
                try:
                    # Attempt to solve the problem via the normal equations.
                    X[:, j] = la.solve(lhs, self._rhs[:, j], assume_a="pos")
                except (la.LinAlgError, la.LinAlgWarning):
                    # For ill-conditioned normal equations, use la.lstsq().
                    if Bpad is None:
                        Bpad = np.vstack((self.B, np.zeros((self.d, self.r))))
                    X[:, j] = la.lstsq(np.vstack((self.A, P)), Bpad[:, j])[0]

        return X

    # Post-processing ---------------------------------------------------------
    def regcond(self, Ps):
        """Compute the 2-norm condition number of each regularized data matrix.

        Parameters
        ----------
        Ps : sequence of r (d, d) or (d,) ndarrays
            Regularization matrices (or the diagonals of the regularization
            matrices if one-dimensional), one for each column of B.

        Returns
        -------
        rcs : float ≥ 0
            cond([A.T | P_i.T].T), i = 1, ..., r, computed as
            sqrt(cond(A.T A + P_i.T P_i)).
        """
        self._check_is_trained("_AtA")
        self._check_Ps(Ps)
        return np.array([np.sqrt(np.linalg.cond(self._lhs(P)[1])) for P in Ps])

    def residual(self, X, Ps):
        """Calculate the residual of the regularized problem for each column of
        B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2.

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
        Ps : sequence of r (d, d) or (d,) ndarrays
            Regularization matrices (or the diagonals of the regularization
            matrices if one-dimensional), one for each column of B.

        Returns
        -------
        resids : (r,) ndarray
            Residuals ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2, i = 1, ..., r.
        """
        self._check_is_trained()
        self._check_Ps(Ps)
        misfit = self.misfit(X)
        Pxs = np.array([np.sum((P @ X[:, j])**2) for j, P in enumerate(Ps)])
        return misfit + Pxs


# Convenience functions =======================================================
def solver(A, B, P):
    """Select and initialize an appropriate solver for the ordinary least-
    squares problem with Tikhonov regularization,

        sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2.

    Parameters
    ----------
    A : (k, d) ndarray
        The "left-hand side" matrix.
    B : (k, r) ndarray
        The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
    P : float >= 0 or ndarray of shapes (r,), (d,), (d, d), (r, d), (r, d, d)
        Tikhonov regularization hyperparameter(s). The regularization matrix
        in the least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,) ndarray: diag(P), a diagonal matrix.
        * (d, d) ndarray: the matrix `P`.
        * sequence of length r : the jth entry in the sequence is the
            regularization hyperparameter for the jth column of `b`. Only
            valid if `b` is two-dimensional and has exactly r columns.

    Returns
    -------
    solver
        Least-squares solver object, with a predict() method mapping the
        regularization factor to the least-squares solution.
    """
    d = A.shape[1]
    if B.ndim == 1:
        B = B.reshape((-1, 1))

    # P is a scalar: single L2-regularized problem.
    if np.isscalar(P):
        solver = SolverL2()

    # P is a sequence of r scalars: decoupled L2-regularized problems.
    elif np.shape(P) == (B.shape[1],):
        solver = SolverL2Decoupled()

    # P is a dxd matrix (or a 1D array of length d for diagonal P):
    # single Tikhonov-regularized problem.
    elif isinstance(P, np.ndarray) and (P.shape in [(d,), (d, d)]):
        solver = SolverTikhonov()

    # P is a sequence of r matrices: decoupled Tikhonov-regularized problems.
    elif np.shape(P) in [(B.shape[1], d), (B.shape[1], d, d)]:
        solver = SolverTikhonovDecoupled()

    else:
        raise ValueError("invalid or misaligned input P")

    return solver.fit(A, B)


def solve(A, B, P=0):
    """Solve the l2-norm Tikhonov-regularized ordinary least-squares problem

        sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2.

    Parameters
    ----------
    A : (k, d) ndarray
        The "left-hand side" matrix.
    B : (k, r) ndarray
        The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
    P : float >= 0 or ndarray of shapes (r,), (d,), (d, d), (r, d), (r, d, d)
        Tikhonov regularization hyperparameter(s). The regularization matrix
        in the least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,) ndarray: diag(P), a diagonal matrix.
        * (d, d) ndarray: the matrix `P`.
        * sequence of length r : the jth entry in the sequence is the
            regularization hyperparameter for the jth column of `b`. Only
            valid if `b` is two-dimensional and has exactly r columns.

    Returns
    -------
    x : (d,) or (d, r) ndarray
        Least-squares solution. If `b` is a two-dimensional array, then
        each column is a solution to the regularized least-squares problem
        with the corresponding column of b.
    """
    return solver(A, B, P).predict(P)
