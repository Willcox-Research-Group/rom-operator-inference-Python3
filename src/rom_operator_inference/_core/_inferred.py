# _core/_inferred.py
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

from ._base import _DiscreteROM, _ContinuousROM, _NonparametricMixin
from ..lstsq import lstsq_reg
from ..utils import kron2c, kron3c


class _InferredMixin:
    """Mixin class for reduced model classes that use Operator Inference."""

    def _check_training_data_shapes(self, datasets, labels):
        """Ensure that each data set has the same number of columns."""
        for data, label in zip(datasets, labels):
            # Ensure each data set is two-dimensional.
            if data.ndim != 2:
                raise ValueError(f"{label} must be two-dimensional")
            # Ensure each data set has the same number of columns.
            if data.shape[1] != datasets[0].shape[1]:
                raise ValueError("training data not aligned "
                                 f"({label}.shape[1] != {labels[0]}.shape[1])")
            # Validate the number of rows.
            if label.startswith("X") and data.shape[0] not in (self.n, self.r):
                raise ValueError(f"invalid training set ({label}.shape[0] "
                                 f"!= n={self.n} or r={self.r})")
            elif label.startswith("U") and data.shape[0] != self.m:
                raise ValueError(f"invalid training input "
                                 f"({label}.shape[0] != m={self.m})")

    def _process_fit_arguments(self, Vr, X, rhs, U):
        """Do sanity checks, extract dimensions, check and fix data sizes, and
        get projected data for the Operator Inference least-squares problem.

        Returns
        -------
        X_ : (r,k) ndarray
            Projected state snapshots.

        rhs_ : (r,k) ndarray
            Projected right-hand-side data.

        U : (m,k) ndarray
            Inputs, potentially reshaped.
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_inputargs(U, 'U')

        # Store basis and dimensions.
        if Vr is not None:
            self.n, self.r = Vr.shape   # Full dimension, reduced dimension.
        else:
            self.n = None               # No full dimension.
            self.r = X.shape[0]         # Reduced dimension.
        self.Vr = Vr

        # Ensure training data sets have consistent sizes.
        if self.has_inputs:
            if U.ndim == 1:             # Reshape one-dimensional inputs.
                U = U.reshape((1,-1))
            self.m = U.shape[0]         # Input dimension.
            self._check_training_data_shapes([X, rhs, U], ["X", "Xdot", "U"])
        else:
            self.m = None               # No input dimension.
            self._check_training_data_shapes([X, rhs], ["X", "Xdot"])

        # Project states and rhs to the reduced subspace (if not done already).
        X_ = self.project(X, 'X')
        rhs_ = self.project(rhs, 'rhs')

        return X_, rhs_, U

    def _construct_data_matrix(self, X_, U):
        """Construct the Operator Inference data matrix D (before any
        regularization) from projected data.

        If modelform="cAHB", this is D = [1 | X_.T | (X_ ⊗ X_).T | U.T].

        Returns
        -------
        D : (k,d(r,m)) ndarray
            Non-regularized Operator Inference data matrix.
        """
        D = []
        if self.has_constant:           # Constant term.
            D.append(np.ones((X_.shape[1],1)))

        if self.has_linear:             # Linear state term.
            D.append(X_.T)

        if self.has_quadratic:          # (compact) quadratic state term.
            D.append(kron2c(X_).T)

        if self.has_cubic:              # (compact) cubic state term.
            D.append(kron3c(X_).T)

        if self.has_inputs:             # Linear input term.
            D.append(U.T)

        return np.hstack(D)

    def _solve_opinf_lstsq(self, D, R, P):
        """Solve the Operator Inference least-squares problem and record
        data about the conditioning and residuals of the problem.

        Returns
        -------
        O : (r,d(r,m)) ndarray
            Solution to the Operator Inference least-squares problem, i.e.,
            the inferred operators in block matrix form.
        """
        Rtrp = R.T

        # Solve for the reduced-order model operators via least squares.
        Otrp, mis, res, cond, regcond = lstsq_reg(D, Rtrp, P)

        # Record info about the least squares solution.
        self.misfit_ = mis          # ||DO.T - R.T||_F^2
        self.residual_ = res        # ||DO.T - R.T||_F^2 + ||PO.T||_F^2
        self.datacond_ = cond       # cond(D)
        self.dataregcond_ = regcond # cond([D.T | P.T].T)

        return Otrp.T

    def _extract_operators(self, O):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squarse problem.
        """
        i = 0
        if self.has_constant:           # Constant term (one-dimensional).
            self.c_ = O[:,i:i+1][:,0]
            i += 1
        else:
            self.c_ = None

        if self.has_linear:             # Linear state matrix.
            self.A_ = O[:,i:i+self.r]
            i += self.r
        else:
            self.A_ = None

        if self.has_quadratic:          # (compact) Qudadratic state matrix.
            _r2 = self.r * (self.r + 1) // 2
            self.Hc_ = O[:,i:i+_r2]
            i += _r2
        else:
            self.Hc_ = None

        if self.has_cubic:              # (compact) Cubic state matrix.
            _r3 = self.r * (self.r + 1) * (self.r + 2) // 6
            self.Gc_ = O[:,i:i+_r3]
            i += _r3
        else:
            self.Gc_ = None

        if self.has_inputs:             # Linear input matrix.
            self.B_ = O[:,i:i+self.m]
            i += self.m
        else:
            self.B_ = None

        return

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
        X_, rhs_, U = self._process_fit_arguments(Vr, X, rhs, U)
        D = self._construct_data_matrix(X_, U)
        O = self._solve_opinf_lstsq(D, rhs_, P)
        self._extract_operators(O)
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
