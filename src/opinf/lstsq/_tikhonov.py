# lstsq/_tikhonov.py
"""Operator Inference least-squares solvers with Tikhonov regularization."""

__all__ = [
    "L2Solver",
    "L2SolverDecoupled",
    "TikhonovSolver",
    "TikhonovSolverDecoupled",
]

import abc
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

from ._base import _BaseSolver


# Solver classes ==============================================================
class _BaseTikhonovSolver(_BaseSolver):
    """Base solver for regularized linear least-squares problems of the form

        sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2.
    """
    # Properties: regularization ----------------------------------------------
    @abc.abstractmethod
    def regularizer(self):                                  # pragma: no cover
        """Regularization scalar, matrix, or list of these."""
        raise NotImplementedError

    # Main methods ------------------------------------------------------------
    def fit(self, A, B):
        """Verify dimensions and save A and B.

        Parameters
        ----------
        A : (k, d) ndarray
            The "left-hand side" matrix.
        B : (k, r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        _BaseSolver.fit(self, A, B)
        if self.k < self.d:
            warnings.warn(
                "non-regularized least-squares system is underdetermined!",
                la.LinAlgWarning, stacklevel=2
            )
        return self

    # Post-processing ---------------------------------------------------------
    @abc.abstractmethod
    def regcond(self):                                      # pragma: no cover
        """Compute the condition number of the regularized data matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def residual(self, X):                                  # pragma: no cover
        """Calculate the residual of the regularized regression problem."""
        raise NotImplementedError


class L2Solver(_BaseTikhonovSolver):
    """Solve the l2-norm ordinary least-squares problem with L2 regularization:

        sum_{i} min_{x_i} ||Ax_i - b_i||_2^2 + ||λx_i||_2^2,    λ ≥ 0,

    or, written in the Frobenius norm,

        min_{X} ||AX - B||_F^2 + ||λX||_F^2,                    λ ≥ 0.

    The solution is calculated using the singular value decomposition of A:
    If A = U Σ V^T, then X = V Σinv(λ) U^T B, where
    Σinv(λ)[i, i] = Σ[i, i] / (Σ[i, i]^2 + λ^2).
    """
    _LSTSQ_LABEL = r"min_{X} ||AX - B||_F^2 + ||λX||_F^2"

    def __init__(self, regularizer=0):
        """Store the regularizer and initialize attributes.

        Parameters
        ----------
        regularizer : float ≥ 0
            Scalar L2 regularization hyperparameter.
        """
        _BaseTikhonovSolver.__init__(self)
        self.regularizer = regularizer

    # Properties --------------------------------------------------------------
    @property
    def regularizer(self):
        return self.__reg

    @regularizer.setter
    def regularizer(self, reg):
        """Set the regularization hyperparameter."""
        if not np.isscalar(reg):
            raise TypeError("regularization hyperparameter must be a scalar")
        if reg < 0:
            raise ValueError("regularization hyperparameter must be "
                             "non-negative")
        self.__reg = reg

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
        _BaseTikhonovSolver.fit(self, A, B)

        # Compute the SVD of A and save what is needed to solve the problem.
        U, svals, Vt = la.svd(self.A, full_matrices=False)
        self._V = Vt.T
        self._svals = svals
        self._UtB = U.T @ self.B

        return self

    def predict(self):
        """Solve the regularized least-squares problem.

        Returns
        -------
        X : (d, r) or (d,) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
            The result is flattened to a one-dimensional array if r = 1.
        """
        self._check_is_trained("_V")

        # X = V svals_inv U.T B
        svals = self._svals.reshape((-1, 1))
        svals_inv = svals / (svals**2 + self.regularizer**2)
        X = self._V @ (svals_inv * self._UtB)

        return np.ravel(X) if self.r == 1 else X

    # Post-processing ---------------------------------------------------------
    def cond(self):
        """Calculate the 2-norm condition number of the data matrix A."""
        self._check_is_trained("_svals")
        return abs(self._svals.max() / self._svals.min())

    def regcond(self):
        """Compute the 2-norm condition number of the regularized data matrix.

        Returns
        -------
        rc : float ≥ 0
            cond([A.T | λI.T].T), computed from filtered singular values of A.
        """
        self._check_is_trained("_svals")
        svals2 = self._svals**2 + self.regularizer**2
        return np.sqrt(svals2.max() / svals2.min())

    def residual(self, X):
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
        return self.misfit(X) + (self.regularizer**2)*np.sum(X**2, axis=0)


class L2SolverDecoupled(L2Solver):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix A but different L2 regularizations λ_i > 0 for the
    columns of X and B:

        min_{x_i} ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2,    i = 1, ..., r.

    The solution is calculated using the singular value decomposition of A.
    """
    _LSTSQ_LABEL = r"min_{x_i} ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2"

    def __init__(self, regularizer):
        """Store the regularizer and initialize attributes.

        Parameters
        ----------
        regularizer : (r,) ndarray or list(r floats ≥ 0)
            Scalar L2 regularization hyperparameter for each column of X and B.
        """
        return L2Solver.__init__(self, regularizer)

    # Properties --------------------------------------------------------------
    def _check_regularizer_shape(self):
        if self.regularizer.shape != (self.r,):
            raise ValueError("len(regularizer) != number of columns of B")

    @L2Solver.regularizer.setter
    def regularizer(self, regs):
        """(r,) ndarray : Scalar regularization hyperparameters, one for each
        column of X and B.
        """
        self._L2Solver__reg = np.array(regs)
        if self.r is not None:
            self._check_regularizer_shape()

    # Main methods ------------------------------------------------------------
    def fit(self, A, B):
        """Take the SVD of A and store B.

        Parameters
        ----------
        A : (k, d) ndarray
            The "left-hand side" matrix.
        B : (k, r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        L2Solver.fit(self, A, B)
        self._check_regularizer_shape()
        return self

    # Post-processing ---------------------------------------------------------
    def regcond(self):
        """Compute the 2-norm condition number of each regularized data matrix.

        Returns
        -------
        rcs : (r,) ndarray
            cond([A.T | (λ_i I).T].T), i = 1, ..., r, computed from filtered
            singular values of the data matrix A.
        """
        self._check_is_trained("_svals")
        svals2 = self._svals**2 + self.regularizer.reshape((-1, 1))**2
        return np.sqrt(svals2.max(axis=1) / svals2.min(axis=1))

    def residual(self, X):
        """Calculate the residual of the regularized problem for each column of
        B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2.

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.

        Returns
        -------
        resids : (r,) ndarray
            Residuals ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2, i = 1, ..., r.
        """
        return L2Solver.residual(self, X)


class TikhonovSolver(_BaseTikhonovSolver):
    """Solve the l2-norm ordinary least-squares problem with Tikhonov
    regularization:

        sum_{i} min_{x_i} ||Ax_i - b_i||_2^2 + ||Px_i||_2^2,    P > 0 (SPD).

    or, written in the Frobenius norm,

        min_{X} ||AX - B||_F^2 + ||PX||_F^2,                    P > 0 (SPD).

    The problem is solved by taking the singular value decomposition of the
    augmented data matrix [A.T | P.T].T, which is equivalent to solving

        min_{X} || [A]    _  [B] ||^{2}
                || [P] X     [0] ||_{F}

    or the Normal equations

        (A.T A + P.T P) X = A.T B.
    """
    _LSTSQ_LABEL = r"min_{X} ||AX - B||_F^2 + ||PX||_F^2"

    def __init__(self, regularizer, method="svd"):
        """Store the regularizer and initialize attributes.

        Parameters
        ----------
        regularizer : (d, d) or (d,) ndarray
            Symmetric semi-positive-definite regularization matrix P
            or, if P is diagonal, just the diagonal entries.
        method : str
            The strategy for solving the regularized least-squares problem.
            * "svd": take the SVD of the stacked data matrix [A.T | P.T].T.
            * "normal": solve the normal equations (A.T A + P.T P) X = A.T B.
        """
        _BaseTikhonovSolver.__init__(self)
        self.regularizer = regularizer
        self.method = method

    # Properties --------------------------------------------------------------
    def _check_regularizer_shape(self):
        if self.regularizer.shape != (self.d, self.d):
            raise ValueError("regularizer.shape != (d, d) (d = A.shape[1])")

    @property
    def regularizer(self):
        """(d, d) ndarray:
        symmetric semi-positive-definite regularization matrix P.
        """
        return self.__reg

    @regularizer.setter
    def regularizer(self, P):
        """Set the regularization matrix."""
        if sparse.issparse(P):
            P = P.toarray()
        elif not isinstance(P, np.ndarray):
            P = np.array(P)

        if P.ndim == 1:
            if np.any(P < 0):
                raise ValueError("diagonal regularizer must be "
                                 "positive semi-definite")
            P = np.diag(P)

        self.__reg = P
        if self.d is not None:
            self._check_regularizer_shape()

    @property
    def method(self):
        """str : Strategy for solving the regularized least-squares problem.
        * "svd": take the SVD of the stacked data matrix [A.T | P.T].T.
        * "normal": solve the normal equations (A.T A + P.T P) X = A.T B.
        """
        return self.__method

    @method.setter
    def method(self, method):
        """Set the method and precompute stuff as needed."""
        if method not in ("svd", "normal"):
            raise ValueError("method must be 'svd' or 'normal'")
        self.__method = method

    # Main routines -----------------------------------------------------------
    def fit(self, A, B):
        """Store A and B."""
        _BaseTikhonovSolver.fit(self, A, B)
        self._check_regularizer_shape()

        # Pad B for "svd" solve.
        self._Bpad = np.vstack((self.B, np.zeros((self.d, self.r))))

        # Precompute Normal equations terms for "normal" solve.
        self._AtA = self.A.T @ self.A
        self._rhs = self.A.T @ self.B

        return self

    def predict(self):
        """Solve the least-squares problem.

        Returns
        -------
        X : (d, r) or (d,) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
            The result is flattened to a one-dimensional array if r = 1.
        """
        self._check_is_trained("_Bpad")

        if self.method == "svd":
            X = la.lstsq(np.vstack((self.A, self.regularizer)), self._Bpad)[0]
        elif self.method == "normal":
            lhs = self._AtA + (self.regularizer.T @ self.regularizer)
            X = la.solve(lhs, self._rhs, assume_a="pos")

        return np.ravel(X) if self.r == 1 else X

    # Post-processing ---------------------------------------------------------
    def regcond(self):
        """Compute the 2-norm condition number of the regularized data matrix.

        Returns
        -------
        rc : float ≥ 0
            cond([A.T | P.T].T).
        """
        self._check_is_trained()
        return np.linalg.cond(np.vstack((self.A, self.regularizer)))

    def residual(self, X):
        """Calculate the residual of the regularized problem for each column of
        B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||Px_i||_2^2.

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.

        Returns
        -------
        resids : (r,) ndarray or float (r = 1)
            Residuals ||Ax_i - b_i||_2^2 + ||Px_i||_2^2, i = 1, ..., r.
        """
        self._check_is_trained()
        return self.misfit(X) + np.sum((self.regularizer @ X)**2, axis=0)


class TikhonovSolverDecoupled(TikhonovSolver):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but a different Tikhonov regularizer,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2,     i = 1, ..., r.
    """
    _LSTSQ_LABEL = r"sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2"

    def __init__(self, regularizer, method="svd"):
        """Store the regularizer and initialize attributes.

        Parameters
        ----------
        regularizer : (d, d) or (d,) ndarray
            Symmetric semi-positive-definite regularization matrix P
            or, if P is diagonal, just the diagonal entries.
        method : str
            The strategy for solving the regularized least-squares problem.
            * "svd": take the SVD of the stacked data matrix [A.T | P.T].T.
            * "normal": solve the normal equations (A.T A + P.T P) X = A.T B.
        """
        return TikhonovSolver.__init__(self, regularizer, method)

    # Properties --------------------------------------------------------------
    def _check_regularizer_shape(self):
        """Check that the regularizer has the correct shape."""
        if len(self.regularizer) != self.r:
            raise ValueError("len(regularizer) != r")
        for i, P in enumerate(self.regularizer):
            if P.shape != (self.d, self.d):
                raise ValueError(f"regularizer[{i}].shape != (d, d)")

    @property
    def regularizer(self):
        """r (d, d) ndarrays : symmetric semi-positive-definite regularization
        matrices [P_1, ..., P_r], one for each column of X and B.
        """
        return self._TikhonovSolver__reg

    @regularizer.setter
    def regularizer(self, Ps):
        """Set the regularization matrices."""
        regs = []
        for P in Ps:
            if sparse.issparse(P):
                P = P.toarray()
            elif not isinstance(P, np.ndarray):
                P = np.array(P)
            if P.ndim == 1:
                if np.any(P < 0):
                    raise ValueError("diagonal regularizer must be "
                                     "positive semi-definite")
                P = np.diag(P)
            regs.append(P)

        self._TikhonovSolver__reg = regs
        if self.d is not None:
            self._check_regularizer_shape()

    # Main methods ------------------------------------------------------------
    def predict(self):
        """Solve the least-squares problems.

        Returns
        -------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.
        """
        self._check_is_trained()

        # Allocate space for the solution.
        X = np.empty((self.d, self.r))

        # Solve each independent problem (iteratively for now).
        for j, P in enumerate(self.regularizer):
            if self.method == "svd":
                X[:, j] = la.lstsq(np.vstack((self.A, P)), self._Bpad[:, j])[0]
            elif self.method == "normal":
                lhs = self._AtA + P.T @ P
                X[:, j] = la.solve(lhs, self._rhs[:, j], assume_a="pos")

        return X

    # Post-processing ---------------------------------------------------------
    def regcond(self):
        """Compute the 2-norm condition number of each regularized data matrix.

        Returns
        -------
        rcs : float ≥ 0
            cond([A.T | P_i.T].T), i = 1, ..., r, computed as
            sqrt(cond(A.T A + P_i.T P_i)).
        """
        self._check_is_trained()
        return np.array([np.linalg.cond(np.vstack((self.A, P)))
                         for P in self.regularizer])

    def residual(self, X):
        """Calculate the residual of the regularized problem for each column of
        B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2.

        Parameters
        ----------
        X : (d, r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
            solution to the subproblem with the corresponding column of B.

        Returns
        -------
        resids : (r,) ndarray
            Residuals ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2, i = 1, ..., r.
        """
        self._check_is_trained()
        misfit = self.misfit(X)
        Pxs = np.array([np.sum((P @ X[:, j])**2)
                        for j, P in enumerate(self.regularizer)])
        return misfit + Pxs
