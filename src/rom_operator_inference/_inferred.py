# _inferred.py
"""Nonparametric Operator Inference ROM classes.

Classes
-------
* _InferredMixin
* InferredDiscreteROM(_InferredMixin, _NonparametricMixin, _DiscreteROM)
* InferredContinuousROM(_InferredMixin, _NonparametricMixin, _ContinuousROM)
"""

__all__ = [
            "InferredDiscreteROM",
            "InferredContinuousROM",
          ]

import numpy as np

from .utils import lstsq_reg, kron2c, kron3c
from ._base import _DiscreteROM, _ContinuousROM, _NonparametricMixin


class _InferredMixin:
    """Mixin class for reduced model classes that use Operator Inference."""

    @staticmethod
    def _check_training_data_shapes(datasets):
        """Ensure that each data set has the same number of columns."""
        k = datasets[0].shape[1]
        for data in datasets:
            if data.shape[1] != k:
                raise ValueError("data sets not aligned, dimension 1")

    def fit(self, Vr, X, rhs, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and rhs are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or velocity
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_inputargs(U, 'U')

        # Store dimensions and check that number of samples is consistent.
        if Vr is not None:
            self.n, self.r = Vr.shape   # Full dimension, reduced dimension.
        else:
            self.n = None
            self.r = X.shape[0]
        _tocheck = [X, rhs]
        if self.has_inputs:             # Input dimension.
            if U.ndim == 1:
                U = U.reshape((1,-1))
                self.m = 1
            else:
                self.m = U.shape[0]
            _tocheck.append(U)
        else:
            self.m = None
        self._check_training_data_shapes(_tocheck)
        k = X.shape[1]

        # Project states and rhs to the reduced subspace (if not done already).
        self.Vr = Vr
        X_ = self.project(X, 'X')
        rhs_ = self.project(rhs, 'rhs')

        # Construct the "Data matrix" D = [X^T, (X ⊗ X)^T, U^T, 1].
        D_blocks = []
        if self.has_constant:
            D_blocks.append(np.ones((k,1)))

        if self.has_linear:
            D_blocks.append(X_.T)

        if self.has_quadratic:
            X2_ = kron2c(X_)
            D_blocks.append(X2_.T)
            _r2 = X2_.shape[0]  # = r(r+1)/2, size of compact quadratic Kron.

        if self.has_cubic:
            X3_ = kron3c(X_)
            D_blocks.append(X3_.T)
            _r3 = X3_.shape[0]  # = r(r+1)(r+2)/6, size of compact cubic Kron.

        if self.has_inputs:
            D_blocks.append(U.T)
            m = U.shape[0]
            self.m = m

        D = np.hstack(D_blocks)
        R = rhs_.T

        # Solve for the reduced-order model operators via least squares.
        Otrp, res, _, sval = lstsq_reg(D, R, P)

        # Record info about the least squares solution.
        # Condition number of the raw data matrix.
        self.datacond_ = np.linalg.cond(D)
        # Condition number of regularized data matrix.
        self.dataregcond_ = abs(sval[0]/sval[-1]) if sval[-1] > 0 else np.inf
        # Squared Frobenius data misfit (without regularization).
        self.misfit_ = np.sum(((D @ Otrp) - R)**2)
        # Squared Frobenius residual of the regularized least squares problem.
        self.residual_ = np.sum(res) if res.size > 0 else self.misfit_

        # Extract the reduced operators from Otrp.
        i = 0
        if self.has_constant:
            self.c_ = Otrp[i:i+1][0]        # Note that c_ is one-dimensional.
            i += 1
        else:
            self.c_ = None

        if self.has_linear:
            self.A_ = Otrp[i:i+self.r].T
            i += self.r
        else:
            self.A_ = None

        if self.has_quadratic:
            self.Hc_ = Otrp[i:i+_r2].T
            i += _r2
        else:
            self.Hc_ = None

        if self.has_cubic:
            self.Gc_ = Otrp[i:i+_r3].T
            i += _r3
        else:
            self.Gc_ = None

        if self.has_inputs:
            self.B_ = Otrp[i:i+self.m].T
            i += self.m
        else:
            self.B_ = None

        self._construct_f_()
        return self


# Nonparametric Operator Inference models -------------------------------------
class InferredDiscreteROM(_InferredMixin, _NonparametricMixin, _DiscreteROM):
    """Reduced order model for a discrete dynamical system of
    the form

        x_{j+1} = f(x_{j}, u_{j}),              x_{0} = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax.

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x).

    has_inputs : bool
        Whether or not there is a linear input term Bu.

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    def fit(self, Vr, X, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X is assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k-1) or (k-1,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InferredMixin.fit(self, Vr,
                                  X[:,:-1], X[:,1:],    # x_j's and x_{j+1}'s.
                                  U[...,:X.shape[1]-1] if U is not None else U,
                                  P)


class InferredContinuousROM(_InferredMixin, _NonparametricMixin,
                            _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),             x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax(t).
        'H' : Quadratic state term H(x⊗x)(t).
        'G' : Cubic state term G(x⊗x⊗x)(t).
        'B' : Input term Bu(t).
        For example, modelform=="AB" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable(float, (r,) ndarray, func?) -> (r,) ndarray
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(t, x_) if 'B' is not in `modelform` (no inputs) and
        f_(t, x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and possibly an input function) to reduced state. Calculated in fit().

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, X, Xdot, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and Xdot are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        Xdot : (n,k) or (r,k) ndarray
            Column-wise velocity training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InferredMixin.fit(self, Vr, X, Xdot, U, P)
