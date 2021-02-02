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

        min_{x} ||Ax - b||^2 + ||Px||^2.
    """
    def __init__(self, compute_extras=True):
        """Set behavior parameters.

        Parameters
        ----------
        compute_extras : bool
            If True, record residual / conditioning information as attributes.
        """
        self.compute_extras = compute_extras

    def _check_shapes(self, A, b):
        """Verify the shapes of A and b are consistent for ||Ax - b||."""
        # Record dimensions of A (and ensure A is two-dimensional).
        self.k, self.d = A.shape

        # Warn about underdeterminedness (even with regularization).
        if self.k < self.d:
            warnings.warn("original least-squares system is underdetermined!",
                          la.LinAlgWarning, stacklevel=2)

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

    Attributes
    ----------
    k : int
        Number of equations in the least-squares problem (number of rows of A).

    d : int
        Number of unknowns to learn in each problem (number of columns of A).

    compute_extras : bool
        If True, predict() records the remaining attributes listed below.

    cond_ : float
        Condition number of the matrix A.

    regcond_ : float
        Effective condition number of the regularized A, [A.T|λI.T].T,
        computed from filtered singular values of A.

    misfit_ : float
        Data misfit ||Ax - b||_2^2, or ||AX - B||_F^2 if b is two-dimensional.

    residual_ : float
        Problem residual ||Ax - b||_2^2 + ||λx||_2^2,
        or ||AX - B||_F^2 + ||λX||_F^2 if b is two-dimensional.
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
            self.cond_ = abs(s[0] / s[-1]) if s[-1] > 0 else np.inf

        return self

    def _process_regularizer(self, λ):
        """Validate the regularization parameter and return λ^2."""
        if not np.isscalar(λ):
            raise ValueError("regularization parameter must be a scalar")
        if λ < 0:
            raise ValueError("regularization parameter must be nonnegative")
        return λ**2

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
        """
        λ2 = self._process_regularizer(λ)

        # Invert / filter the singular values and compute the solution.
        if λ2 == 0:
            Sinv = 1 / self._s                          # σinv = 1/σ
        else:
            Sinv = self._s / (self._s**2 + λ2)          # σinv = σ/(σ^2 + λ^2)
        if self._bndim == 2:
            Sinv = Sinv.reshape((-1,1))

        x = self._V @ (Sinv * self._Utb)                # x = V Sinv U.T b

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            # Data misfit (no regularization): ||Ax-b||^2.
            self.misfit_ = np.sum((self._A @ x - self._b)**2)
            # Problem residual: ||Ax-b||^2 + ||λx||^2.
            self.residual_ = self.misfit_ + λ2*np.sum(x**2)
            # Condition number of regularized problem.
            self.regcond_ = abs(Sinv.max() / Sinv.min())

        return x


class SolverL2Decoupled(SolverL2):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but different L2 regularizations,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||λ_i x_i||_2^2,    λ_i > 0.

    Attributes
    ----------
    k : int
        Number of equations in the least-squares problem (number of rows of A).

    d : int
        Number of unknowns to learn in each problem (number of columns of A).

    r : int
        Number of independent least-squares problems (number of columns of B).

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
    def fit(self, A, B):
        """Take the SVD of A in preparation to solve the least-squares problem.

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
        return SolverL2.fit(self, A, B)

    def predict(self, λs):
        """Solve the least-squares problems with regularization parameters λs.

        Parameters
        ----------
        λs : sequence of r floats or (r,) ndarray

        Returns
        -------
        X : (d,r) ndarray
            Least-squares solution; each column is a solution to the
            problem with the corresponding column of B.
        """
        if hasattr(λs, "__len__") and len(λs) != self.r:
            raise ValueError("len(λs) != number of columns of B")

        # Allocate space for the solution and initialize extras if desired.
        X = np.empty((self.d,self.r))
        if self.compute_extras:
            λ2s = []
            regconds = []

        # Solve each independent problem (iteratively for now).
        for j, λ in zip(range(self.r), λs):
            # Solve each regularized problem.
            λ2 = self._process_regularizer(λ)
            if λ2 == 0:
                Sinv = 1 / self._s                      # σinv = 1/σ
            else:
                Sinv = self._s / (self._s**2 + λ2)      # σinv = σ/(σ^2 + λ^2)
            X[:,j] = self._V @ (Sinv * self._Utb[:,j])  # xj = V Sinvj U.T bj

            if self.compute_extras:
                λ2s.append(λ2)
                regconds.append(abs(Sinv.max() / Sinv.min()))

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            self.misfit_ = np.sum((self._A @ X - self._b)**2, axis=0)
            self.residual_ = self.misfit_ + np.array(λ2s)*np.sum(X**2, axis=0)
            self.regcond_ = np.array(regconds)

        return X


class SolverTikhonov(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem with Tikhonov
    regularization:

        min_{x} ||Ax - b||_2^2 + ||Px||_2^2,    P > 0 (SPD matrix).

    If b is two-dimensional, the problem is solved in the Frobenius norm:

        min_{X} ||AX - B||_F^2 + ||PX||_F^2,    P > 0 (SPD matrix).

    Attributes
    ----------
    k : int
        Number of equations in the least-squares problem (number of rows of A).

    d : int
        Number of unknowns to learn in each problem (number of columns of A).

    compute_extras : bool
        If True, predict() records the remaining attributes listed below.

    cond_ : float
        Condition number of the matrix A.

    regcond_ : float
        Condition number of the regularized A, [A.T|P.T].T.

    misfit_ : float
        Data misfit ||Ax - b||_2^2, or ||AX - B||_F^2 if b is two-dimensional.

    residual_ : float
        Problem residual ||Ax - b||_2^2 + ||Px||_2^2,
        or ||AX - B||_F^2 + ||PX||_F^2 if b is two-dimensional.
    """
    def __init__(self, compute_extras=False, check_regularizer=False):
        """Set behavior parameters.

        Parameters
        ----------
        compute_extras : bool
            If True, record residual / conditioning information as attributes:
            * cond_: condition number of the matrix A.
            * regcond_: condition number of the regularized matrix [A.T|P.T].T.
            * misfit_: data misfit ||Ax - b||^2.
            * residual_: problem residual ||Ax - b||^2 + ||Px||^2.

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
            self.cond_ = np.linalg.cond(A)

        return self

    def _process_regularizer(self, P):
        """Validate the type and shape of the regularizer and compute P.T P."""
        # TODO: allow sparse P.
        if not isinstance(P, np.ndarray):
            raise ValueError("regularization matrix must be a NumPy array")

        # One-dimensional input (diagonals of the regularization matrix).
        if P.shape == (self.d,):
            if np.any(P < 0):
                raise ValueError("diagonal P must be positive semi-definite")
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
        """
        # Construct and solve the augmented problem.
        lhs = self._AtA + self._process_regularizer(P)
        x = la.solve(lhs, self._rhs, assume_a="pos")

        # Compute residuals and condition numbers if desired.
        if self.compute_extras:
            if P.ndim == 1:
                Px = P.reshape((-1,1)) * x if x.ndim == 2 else P * x
            else:
                Px = P @ x
            # Data misfit (no regularization): ||Ax-b||^2.
            self.misfit_ = np.sum((self._A @ x - self._b)**2)
            # Problem residual: ||Ax-b||^2 + ||Px||^2.
            self.residual_ = self.misfit_ + np.sum(Px**2)
            # Conditioning of regularized problem: cond([A.T | P.T].T).
            self.regcond_ = np.sqrt(np.linalg.cond(lhs))

        return x


class SolverTikhonovDecoupled(SolverTikhonov):
    """Solve r independent l2-norm ordinary least-squares problems, each with
    the same data matrix but a different Tikhonov regularizer,

        min_{x_i} ||Ax_i - b_i||_2^2 + ||P_i x_i||_2^2.

    Attributes
    ----------
    k : int
        Number of equations in the least-squares problem (number of rows of A).

    d : int
        Number of unknowns to learn in each problem (number of columns of A).

    r : int
        Number of independent least-squares problems (number of columns of B).

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
        """Solve the least-squares problems with regularization matrices Ps.

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
        """
        if hasattr(Ps, "__len__") and len(Ps) != self.r:
            raise ValueError("len(Ps) != number of columns of B")

        # Allocate space for the solution and initialize extras if desired.
        X = np.empty((self.d,self.r))
        if self.compute_extras:
            Px_norms = []
            regconds = []

        # Solve each independent problem (iteratively for now).
        for j, P, rhs in zip(range(self.r), Ps, self._rhs.T):
            # Construct and solve the augmented problem.
            lhs = self._AtA + self._process_regularizer(P)
            X[:,j] = la.solve(lhs, self._rhs[:,j], assume_a="pos")

            # Compute extras if desired.
            if self.compute_extras:
                Px = P * X[:,j] if P.ndim == 1 else P @ X[:,j]
                Px_norms.append(np.sum(Px**2))
                regconds.append(np.sqrt(np.linalg.cond(lhs)))

        # Record extras if desired.
        if self.compute_extras:
            self.misfit_ = np.sum((self._A @ X - self._b)**2, axis=0)
            self.residual_ = self.misfit_ + np.array(Px_norms)
            self.regcond_ = np.array(regconds)

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

    P : float >= 0 or ndarray of shapes (r,), (d,), (d,d), (r,d), or (r,d,d)
        Tikhonov regularization parameter(s). The regularization matrix in the
        least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,) ndarray: diag(P), a diagonal matrix.
        * (d,d) ndarray: the matrix `P`.
        * sequence of length r : the jth entry in the sequence is the
            regularization parameter for the jth column of `b`. Only valid if
            `b` is two-dimensional and has exactly r columns.

    **kwargs
        Additional arguments for the solver object.
        * compute_extras : bool
            If True, record residual / conditioning information as attributes:
            - cond_: condition number of the matrix A.
            - regcond_: condition number of the regularized matrix [A.T|P.T].T.
            - misfit_: data misfit ||Ax - b||^2.
            - residual_: problem residual ||Ax - b||^2 + ||Px||^2.
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
    d = A.shape[1]

    # P is a scalar: single L2-regularized problem.
    if np.isscalar(P):
        if "check_regularizer" in kwargs:                   # pragma: no cover
            kwargs.pop("check_regularizer")
        solver = SolverL2(**kwargs)

    # P is a sequence of r scalars: decoupled L2-regularized problems.
    elif b.ndim == 2 and np.shape(P) == (b.shape[1],):
        if "check_regularizer" in kwargs:                   # pragma: no cover
            kwargs.pop("check_regularizer")
        solver = SolverL2Decoupled(**kwargs)

    # P is a dxd matrix (or a 1D array of length d for diagonal P):
    # single Tikhonov-regularized problem.
    elif isinstance(P, np.ndarray) and (P.shape in [(d,), (d,d)]):
        solver = SolverTikhonov(**kwargs)

    # P is a sequence of r matrices: decoupled Tikhonov-regularized problems.
    elif b.ndim == 2 and np.shape(P) in [(b.shape[1],d), (b.shape[1],d,d)]:
        solver = SolverTikhonovDecoupled(**kwargs)

    else:
        raise ValueError(f"invalid or misaligned input P")

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

    P : float >= 0 or ndarray of shapes (r,), (d,), (d,d), (r,d), or (r,d,d)
        Tikhonov regularization parameter(s). The regularization matrix in the
        least-squares problem depends on the format of the argument:
        * float >= 0: `P`*I, a scaled identity matrix.
        * (d,) ndarray: diag(P), a diagonal matrix.
        * (d,d) ndarray: the matrix `P`.
        * sequence of length r : the jth entry in the sequence is the
            regularization parameter for the jth column of `b`. Only valid if
            `b` is two-dimensional and has exactly r columns.

    **kwargs
        Additional arguments for the solver object.
        * compute_extras : bool
            If True, record residual / conditioning information as attributes:
            - cond_: condition number of the matrix A.
            - regcond_: condition number of the regularized matrix [A.T|P.T].T.
            - misfit_: data misfit ||Ax - b||^2.
            - residual_: problem residual ||Ax - b||^2 + ||Px||^2.
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
    """
    return solver(A, b, P, **kwargs).predict(P)
