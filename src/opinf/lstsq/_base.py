# lstsq/_base.py
"""Base class for solvers for the Operator Inference regression problem."""

__all__ = [
    "lstsq_size",
    "PlainSolver",
]

import abc
import numpy as np
import scipy.linalg as la


def lstsq_size(modelform, r, m=0, affines=None):
    """Calculate the number of columns in the operator matrix O in the Operator
    Inference least-squares problem. This is also the number of columns in the
    data matrix D.

    Parameters
    ----------
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
    if 'B' in modelform and m == 0:
        raise ValueError("argument m > 0 required since 'B' in modelform")
    if 'B' not in modelform and m != 0:
        raise ValueError(f"argument m={m} invalid since 'B' in modelform")

    if affines is None:
        affines = {}

    qs = [(len(affines[op]) if (op in affines and op in modelform)
           else 1 if op in modelform else 0) for op in "cAHGB"]
    rs = [1, r, r*(r+1)//2, r*(r+1)*(r+2)//6, m]

    return sum(qq*rr for qq, rr in zip(qs, rs))


class _BaseSolver(abc.ABC):
    """Base class for solvers for the Operator Inference regression problem.
    Child classes should receive and store hyperparameters in the constructor
    (regularization scalars, truncation size, etc.).
    """
    _LSTSQ_LABEL = ""

    def __init__(self):
        self.__A = None
        self.__B = None

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

    def _check_is_trained(self, attr=None):
        """Raise an AttributeError if fit() has not been called."""
        trained = (self.A is not None) and (self.B is not None)
        if attr is not None:
            trained *= hasattr(self, attr)
        if not trained:
            raise AttributeError("lstsq solver not trained (call fit())")

    # String representation ---------------------------------------------------
    def __str__(self):
        """String representation: class name + dimensions."""
        out = [f"Least-squares solver for {self._LSTSQ_LABEL}"]
        if (self.A is not None) and (self.B is not None):
            out.append(f"A: {self.A.shape}")
            out.append(f"X: {self.d, self.r}")
            out.append(f"B: {self.B.shape}")
        return '\n'.join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        uniqueID = f"<{self.__class__.__name__} object at {hex(id(self))}>"
        return f"{uniqueID}\n{str(self)}"

    # Main methods -----------------------------------------------------------
    def fit(self, A, B):
        """Verify dimensions and save A and B.

        Parameters
        ----------
        A : (k, d) ndarray
            The "left-hand side" matrix.
        B : (k, r) ndarray
            The "right-hand side" matrix B = [ b_1 | b_2 | ... | b_r ].
        """
        # Validate and store B.
        if A.ndim != 2:
            raise ValueError("A must be two-dimensional")
        if B.ndim == 1:
            B = B.reshape((-1, 1))
        if B.ndim != 2:
            raise ValueError("B must be one- or two-dimensional")
        if B.shape[0] != A.shape[0]:
            raise ValueError("A.shape[0] != B.shape[0]")

        self.__A = A
        self.__B = B

        return self

    @abc.abstractmethod
    def predict(*args, **kwargs):                           # pragma: no cover
        """Solver the learning problem."""
        raise NotImplementedError

    # Post-processing --------------------------------------------------------
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


class PlainSolver(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem without any
    regularization, i.e.,

        min_{X} ||AX - B||_F^2.

    The solution is calculated using scipy.linalg.lstsq().
    """
    _LSTSQ_LABEL = "||AX - B||"

    def __init__(self, options=None):
        """Store keyword arguments for scipy.linalg.lstsq().

        Parameters
        ----------
        options : dict
            Keyword arguments for scipy.linalg.lstsq().
        """
        self.options = {} if options is None else options

    def predict(self):
        """Solve the least-squares problem."""
        return la.lstsq(self.A, self.B, **self.options)[0]
