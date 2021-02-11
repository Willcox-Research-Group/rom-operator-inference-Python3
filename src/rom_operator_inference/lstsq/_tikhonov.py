# lstsq/_tikhonov.py
"""Operator Inference least-squares solvers with Tikhonov regularization."""
# TODO: update documentation to say extras are now attributes, not returns

__all__ = [
            "SolverL2",
            "SolverL2Decoupled",
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

        sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2.
    """
    def __init__(self, compute_extras=True):
        """Set behavior parameters.

        Parameters
        ----------
        compute_extras : bool
            If True, record residual / conditioning information as attributes.
        """
        self.A, self.B = None, None
        self.compute_extras = compute_extras

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
        A : (k,d) ndarray
            The "left-hand side" matrix.

        B : (k,r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        # Extract the dimensions of A (and ensure A is two-dimensional).
        k, d = A.shape
        if k < d:
            warnings.warn("original least-squares system is underdetermined!",
                          la.LinAlgWarning, stacklevel=2)
        self.A = A

        # Check dimensions of b.
        if B.ndim == 1:
            B = B.reshape((-1,1))
        if B.ndim != 2:
            raise ValueError("`B` must be one- or two-dimensional")
        if B.shape[0] != k:
            raise ValueError("inputs not aligned: A.shape[0] != B.shape[0]")
        self.B = B

    # Main methods ------------------------------------------------------------
    def fit(*args, **kwargs):
        raise NotImplementedError("fit() implemented by child classes")

    def predict(*args, **kwargs):
        raise NotImplementedError("predict() implemented by child classes")


class SolverL2(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem with L2 regularization:

        sum_{i} min_{x_i} ||Ax_i - b_i||_2^2 + ||λx_i||_2^2,    λ ≥ 0,

    or, written in the Frobenius norm,

        min_{X} ||AX - B||_F^2 + ||λX||_F^2,                    λ ≥ 0.

    Attributes
    ----------
    compute_extras : bool
        If True, predict() records the remaining attributes listed below.

    cond_ : float
        Condition number of the matrix A.

    regcond_ : float
        Effective condition number of the regularized A, [A.T|λI.T].T,
        computed from filtered singular values of A.

    misfit_ : float
        Data misfit (without regularization) ||AX - B||_F^2.

    residual_ : float
        Problem residual (with regularization) ||AX - B||_F^2 + ||λX||_F^2.
    """
    def fit(self, A, B):
        """Take the SVD of A in preparation to solve the least-squares problem.

        Parameters
        ----------
        A : (k,d) ndarray
            The "left-hand side" matrix.

        B : (k,r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        self._process_fit_arguments(A, B)

        # Compute the SVD of A and save what is needed to solve the problem.
        U,Σ,Vt = la.svd(self.A, full_matrices=False)
        self._V = Vt.T
        self._Σ = Σ
        self._UtB = U.T @ self.B

        # Save what is needed for extra outputs if desired.
        if self.compute_extras:
            self.cond_ = Σ[0]/Σ[-1] if Σ[-1] > 0 else np.inf

        return self

    def _process_regularizer(self, λ):
        """Validate the regularization hyperparameter and return λ^2."""
        if not np.isscalar(λ):
            raise TypeError("regularization hyperparameter λ must be "
                            "a scalar")
        if λ < 0:
            raise ValueError("regularization hyperparameter λ must be "
                             "non-negative")
        return λ**2

    def predict(self, λ):
        """Solve the least-squares problem with the non-negative scalar
        regularization hyperparameter λ.

        Parameters
        ----------
        λ : float ≥ 0
            Scalar regularization hyperparameter.

        Returns
        -------
        X : (d,r) or (d,) ndarray
            Least-squares solution X = [ x_1 | ... | x_r]; each column is the
            solution to the subproblem with the corresponding column of B.
            The result is flattened to a one-dimensional array if r = 1.
        """
        if not hasattr(self, "_V"):
            raise AttributeError("lstsq solver not trained (call fit())")
        λ2 = self._process_regularizer(λ)

        # Invert / filter the singular values and compute the solution.
        Σ = self._Σ.reshape((-1,1))
        if λ2 == 0:
            Σinv = 1 / Σ                        # (Σinv)_ii = 1/(σ_i)
        else:
            Σinv = Σ / (Σ**2 + λ2)              # (Σinv)_ii = σ_i/(σ_i^2 + λ^2)

        X = self._V @ (Σinv * self._UtB)        # X = V Σinv U.T B

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            # Data misfit (no regularization): ||AX - B||_F^2.
            self.misfit_ = np.sum((self.A @ X - self.B)**2)
            # Problem residual: ||AX - B||_F^2 + ||λX||_F^2.
            self.residual_ = self.misfit_ + λ2*np.sum(X**2)
            # Condition number of regularized problem.
            self.regcond_ = abs(Σinv.max() / Σinv.min())

        return np.ravel(X) if self.r == 1 else X


class SolverL2Decoupled(SolverL2):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but different L2 regularizations,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2,    λ_i > 0.

    Attributes
    ----------
    compute_extras : bool
        If True, predict() records the remaining attributes listed below.

    cond_ : float
        Condition number of the matrix A.

    regcond_ : (r,) ndarray
        Effective condition numbers of the regularized A, [A.T| (λ_i I).T].T,
        computed from filtered singular values of A, i = 1,...,r.

    misfit_ : (r,) ndarray
        Data misfits ||Ax_i - b_i||_2^2, i = 1,...,r.

    residual_ : (r,) ndarray
        Problem residuals ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2, i = 1,...,r.
    """
    def predict(self, λs):
        """Solve the least-squares problems with regularization hyperparameters λs.

        Parameters
        ----------
        λs : sequence of r floats or (r,) ndarray
            Scalar regularization hyperparameters, one for each column of B.

        Returns
        -------
        X : (d,r) ndarray
            Least-squares solution; each column is a solution to the
            problem with the corresponding column of B.
        """
        if not hasattr(self, "_V"):
            raise AttributeError("lstsq solver not trained (call fit())")
        if not hasattr(λs, "__len__") or len(λs) != self.r:
            raise ValueError("len(λs) != number of columns of B")

        # Allocate space for the solution and initialize extras if desired.
        X = np.empty((self.d,self.r))
        if self.compute_extras:
            λ2s = []
            regconds = []

        # Solve each independent regularized lstsq problem (iteratively).
        Σ = self._Σ
        for j, λ in enumerate(λs):
            λ2 = self._process_regularizer(λ)

            if λ2 == 0:
                Σinv = 1 / Σ                    # (Σinv)_ii = 1/(σ_i)
            else:
                Σinv = Σ / (Σ**2 + λ2)          # (Σinv)_ii = σ_i/(σ_i^2 + λ^2)

            X[:,j] = self._V @ (Σinv * self._UtB[:,j])      # X = V Σinv U.T B

            if self.compute_extras:
                λ2s.append(λ2)
                regconds.append(abs(Σinv.max() / Σinv.min()))

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            self.misfit_ = np.sum((self.A @ X - self.B)**2, axis=0)
            self.residual_ = self.misfit_ + np.array(λ2s)*np.sum(X**2, axis=0)
            self.regcond_ = np.array(regconds)

        return np.ravel(X) if self.r == 1 else X


class SolverTikhonov(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem with Tikhonov
    regularization:

        sum_{i} min_{x_i} ||Ax_i - b_i||_2^2 + ||Px_i||_2^2,    P > 0 (SPD).

    or, written in the Frobenius norm,

        min_{X} ||AX - B||_F^2 + ||PX||_F^2,                    P > 0 (SPD).

    Attributes
    ----------
    compute_extras : bool
        If True, predict() records the remaining attributes listed below.

    cond_ : float
        Condition number of the matrix A.

    regcond_ : float
        Condition number of the regularized A, [A.T|P.T].T.

    misfit_ : float
        Data misfit (without regularization) ||AX - B||_F^2.

    residual_ : float
        Problem residual (with regularization) ||AX - B||_F^2 + ||PX||_F^2.
    """
    def fit(self, A, B):
        """Prepare to solve the least-squares problem via the normal equations.

        Parameters
        ----------
        A : (k,d) ndarray
            The "left-hand side" matrix.

        B : (k,r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        self._process_fit_arguments(A, B)

        # Compute both sides of the Normal equations.
        self._rhs = self.A.T @ self.B
        self._AtA = self.A.T @ self.A

        # Save what is needed for extra outputs if desired.
        if self.compute_extras:
            self.cond_ = np.linalg.cond(A)

        return self

    def _process_regularizer(self, P):
        """Validate the type and shape of the regularizer and compute P.T P."""
        # TODO: allow sparse P.
        if not isinstance(P, np.ndarray):
            raise TypeError("regularization matrix must be a NumPy array")

        # One-dimensional input (diagonals of the regularization matrix).
        if P.shape == (self.d,):
            if np.any(P < 0):
                raise ValueError("diagonal P must be positive semi-definite")
            return np.diag(P), np.diag(P**2)

        # Two-dimensional input (the regularization matrix).
        elif P.shape != (self.d,self.d):
            raise ValueError("P.shape != (d,d) or (d,) where d = A.shape[1]")

        return P, P.T @ P

    def predict(self, P):
        """Solve the least-squares problem with regularization matrix P.

        Parameters
        ----------
        P : (d,d) or (d,) ndarray
            Regularization matrix (or the diagonals of the regularization
            matrix if one-dimensional).

        Returns
        -------
        X : (d,r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r]; each column is the
            solution to the subproblem with the corresponding column of B.
        """
        if not hasattr(self, "_AtA"):
            raise AttributeError("lstsq solver not trained (call fit())")
        # Construct and solve the augmented problem.
        P, PtP = self._process_regularizer(P)
        lhs = self._AtA + PtP

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=la.LinAlgWarning)
            try:
                # Attempt to solve the problem via the normal equations.
                X = la.solve(lhs, self._rhs, assume_a="pos")
            except (la.LinAlgError, la.LinAlgWarning) as e:
                # For ill-conditioned normal equations, use la.lstsq().
                print(f"normal equations solve failed, switching lstsq solver")
                Bpad = np.vstack((self.B, np.zeros((self.d, self.r))))
                X = la.lstsq(np.vstack((self.A, P)), Bpad)[0]

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            # Data misfit (no regularization): ||AX - B||_F^2.
            self.misfit_ = np.sum((self.A @ X - self.B)**2)
            # Problem residual: ||AX - B||_F^2 + ||PX||_F^2.
            self.residual_ = self.misfit_ + np.sum((P @ X)**2)
            # Conditioning of regularized problem: cond([A.T | P.T].T).
            self.regcond_ = np.sqrt(np.linalg.cond(lhs))

        return np.ravel(X) if self.r == 1 else X


class SolverTikhonovDecoupled(SolverTikhonov):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but a different Tikhonov regularizer,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2.

    Attributes
    ----------
    compute_extras : bool
        If True, predict() records the remaining attributes listed below.

    cond_ : float
        Condition number of the matrix A.

    regcond_ : (r,) ndarray
        Effective condition numbers of the regularized A, [A.T| (P_i I).T].T,
        for i = 1,...,r.

    misfit_ : (r,) ndarray
        Data misfits ||Ax_i - b_i||_2^2, i = 1,...,r.

    residual_ : (r,) ndarray
        Problem residuals ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2, i = 1,...,r.
    """
    def predict(self, Ps):
        """Solve the least-squares problems with regularization matrices Ps.

        Parameters
        ----------
        Ps : sequence of r (d,d) or (d,) ndarrays
            Regularization matrices (or the diagonals of the regularization
            matrices if one-dimensional), one for each column of B.

        Returns
        -------
        X : (d,r) ndarray
            Least-squares solution X = [ x_1 | ... | x_r]; each column is the
            solution to the subproblem with the corresponding column of B.
        """
        if not hasattr(self, "_AtA"):
            raise AttributeError("lstsq solver not trained (call fit())")
        if not hasattr(Ps, "__len__") or len(Ps) != self.r:
            raise ValueError("len(Ps) != number of columns of B")

        # Allocate space for the solution and initialize extras if desired.
        X = np.empty((self.d,self.r))
        if self.compute_extras:
            Px_norms = []
            regconds = []

        # Solve each independent problem (iteratively for now).
        Bpad = None
        for j, [P, rhs] in enumerate(zip(Ps, self._rhs.T)):
            P, PtP = self._process_regularizer(P)
            lhs = self._AtA + PtP
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=la.LinAlgWarning)
                try:
                    # Attempt to solve the problem via the normal equations.
                    X[:,j] = la.solve(lhs, self._rhs[:,j], assume_a="pos")
                except (la.LinAlgError, la.LinAlgWarning) as e:
                    # For ill-conditioned normal equations, use la.lstsq().
                    if Bpad is None:
                        Bpad = np.vstack((self.B, np.zeros((self.d, self.r))))
                    X[:,j] = la.lstsq(np.vstack((self.A, P)), Bpad[:,j])[0]

            # Compute extras if desired.
            if self.compute_extras:
                Px_norms.append(np.sum((P @ X[:,j])**2))
                regconds.append(np.sqrt(np.linalg.cond(lhs)))

        # Record extras if desired.
        if self.compute_extras:
            self.misfit_ = np.sum((self.A @ X - self.B)**2, axis=0)
            self.residual_ = self.misfit_ + np.array(Px_norms)
            self.regcond_ = np.array(regconds)

        return np.ravel(X) if self.r == 1 else X


# Convenience functions =======================================================
def solver(A, B, P, **kwargs):
    """Select and initialize an appropriate solver for the ordinary least-
    squares problem with Tikhonov regularization,

        sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2.

    Parameters
    ----------
    A : (k,d) ndarray
        The "left-hand side" matrix.

    B : (k,r) ndarray
        The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].

    P : float >= 0 or ndarray of shapes (r,), (d,), (d,d), (r,d), or (r,d,d)
        Tikhonov regularization hyperparameter(s). The regularization matrix in the
        least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,) ndarray: diag(P), a diagonal matrix.
        * (d,d) ndarray: the matrix `P`.
        * sequence of length r : the jth entry in the sequence is the
            regularization hyperparameter for the jth column of `b`. Only valid if
            `b` is two-dimensional and has exactly r columns.

    **kwargs
        Additional arguments for the solver object.
        * compute_extras : bool
            If True, record residual / conditioning information as attributes:
            - cond_: condition number of the matrix A.
            - regcond_: condition number of the regularized matrix [A.T|P.T].T.
            - misfit_: data misfit ||Ax - b||^2.
            - residual_: problem residual ||Ax - b||^2 + ||Px||^2.

    Returns
    -------
    solver
        Least-squares solver object, with a predict() method mapping the
        regularization factor to the least-squares solution.
    """
    d = A.shape[1]
    if B.ndim == 1:
        B = B.reshape((-1,1))

    # P is a scalar: single L2-regularized problem.
    if np.isscalar(P):
        solver = SolverL2(**kwargs)

    # P is a sequence of r scalars: decoupled L2-regularized problems.
    elif np.shape(P) == (B.shape[1],):
        solver = SolverL2Decoupled(**kwargs)

    # P is a dxd matrix (or a 1D array of length d for diagonal P):
    # single Tikhonov-regularized problem.
    elif isinstance(P, np.ndarray) and (P.shape in [(d,), (d,d)]):
        solver = SolverTikhonov(**kwargs)

    # P is a sequence of r matrices: decoupled Tikhonov-regularized problems.
    elif np.shape(P) in [(B.shape[1],d), (B.shape[1],d,d)]:
        solver = SolverTikhonovDecoupled(**kwargs)

    else:
        raise ValueError(f"invalid or misaligned input P")

    return solver.fit(A, B)


def solve(A, B, P=0, **kwargs):
    """Solve the l2-norm Tikhonov-regularized ordinary least-squares problem

        sum_{i} min_{x_i} ||Ax_i - b_i||^2 + ||P_i x_i||^2.

    Parameters
    ----------
    A : (k,d) ndarray
        The "left-hand side" matrix.

    B : (k,r) ndarray
        The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].

    P : float >= 0 or ndarray of shapes (r,), (d,), (d,d), (r,d), or (r,d,d)
        Tikhonov regularization hyperparameter(s). The regularization matrix in the
        least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,) ndarray: diag(P), a diagonal matrix.
        * (d,d) ndarray: the matrix `P`.
        * sequence of length r : the jth entry in the sequence is the
            regularization hyperparameter for the jth column of `b`. Only valid if
            `b` is two-dimensional and has exactly r columns.

    **kwargs
        Additional arguments for the solver object.
        * compute_extras : bool
            If True, record residual / conditioning information as attributes:
            - cond_: condition number of the matrix A.
            - regcond_: condition number of the regularized matrix [A.T|P.T].T.
            - misfit_: data misfit ||Ax - b||^2.
            - residual_: problem residual ||Ax - b||^2 + ||Px||^2.

    Returns
    -------
    x : (d,) or (d,r) ndarray
        Least-squares solution. If `b` is a two-dimensional array, then
        each column is a solution to the regularized least-squares problem
        with the corresponding column of b.
    """
    return solver(A, B, P, **kwargs).predict(P)



# TODO: make A and B properties.
