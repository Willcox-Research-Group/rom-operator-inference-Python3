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
from .. import lstsq
from ..utils import kron2c, kron3c


class _InferredMixin:
    """Mixin class for reduced model classes that use Operator Inference."""

    def _check_training_data_shapes(self, datasets):
        """Ensure that each data set has the same number of columns and a
        valid number or rows (determined by the basis).

        Parameters
        ----------
        datasets: list of (ndarray, str) tuples
            Datasets paired with labels, e.g., [(X, "states"), (dX, "ddts")].
        """
        data0, label0 = datasets[0]
        for data, label in datasets:
            if label == "inputs":
                if self.m != 1:     # inputs.shape = (m,k)
                    if data.ndim != 2:
                        raise ValueError("inputs must be two-dimensional "
                                         "(m > 1)")
                    if data.shape[0] != self.m:
                        raise ValueError(f"inputs.shape[0] = {data.shape[0]} "
                                         f"!= {self.m} = m")
                else:               # inputs.shape = (1,k) or (k,)
                    if data.ndim not in (1,2):
                        raise ValueError("inputs must be one- or "
                                         "two-dimensional (m = 1)")
                    if data.ndim == 2 and data.shape[0] != 1:
                        raise ValueError("inputs.shape != (1,k) (m = 1)")
            else:
                if data.ndim != 2:
                    raise ValueError(f"{label} must be two-dimensional")
                if data.shape[0] not in (self.n, self.r):
                    raise ValueError(f"{label}.shape[0] != n or r "
                                     f"(n={self.n}, r={self.r})")
            if data.shape[-1] != data0.shape[-1]:
                raise ValueError(f"{label}.shape[-1] = {data.shape[-1]} "
                                 f"!= {data0.shape[-1]} = {label0}.shape[-1]")

    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(self, basis, states, rhs, inputs):
        """Do sanity checks, extract dimensions, check and fix data sizes, and
        get projected data for the Operator Inference least-squares problem.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and rhs are assumed to already be projected.
        states : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or time derivative
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).
        inputs : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        Returns
        -------
        states_ : (r,k) ndarray
            Projected state snapshots.
        rhs_ : (r,k) ndarray
            Projected right-hand-side data.
        """
        self._check_inputargs(inputs, 'inputs')
        self._clear()

        # Store basis and reduced dimension.
        self.basis = basis
        if basis is None:
            self.r = states.shape[0]

        # Get input dimension if needed.
        to_check = [(states, "states"), (rhs, self._RHS_LABEL)]
        if self.has_inputs:
            self.m = 1 if inputs.ndim == 1 else inputs.shape[0]
            to_check.append((inputs, "inputs"))

        # Ensure training data sets have consistent sizes.
        self._check_training_data_shapes(to_check)

        # Project states and rhs to the reduced subspace (if needed).
        states_ = self.project(states, "states")
        rhs_ = self.project(rhs, self._RHS_LABEL)

        return states_, rhs_

    def _assemble_data_matrix(self, states_, inputs):
        """Construct the Operator Inference data matrix D from projected data.

        If modelform="cAHB", this is D = [1 | X_.T | (X_ ⊗ X_).T | U.T],

        where X_ = states_ and U = inputs.

        Parameters
        ----------
        states_ : (r,k) ndarray
            Column-wise projected snapshot training data.
        inputs : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input).

        Returns
        -------
        D : (k,d(r,m)) ndarray
            Operator Inference data matrix (no regularization).
        """
        D = []
        if self.has_constant:           # Constant term.
            D.append(np.ones((states_.shape[1],1)))

        if self.has_linear:             # Linear state term.
            D.append(states_.T)

        if self.has_quadratic:          # (compact) Quadratic state term.
            D.append(kron2c(states_).T)

        if self.has_cubic:              # (compact) Cubic state term.
            D.append(kron3c(states_).T)

        if self.has_inputs:             # Linear input term.
            D.append(np.atleast_2d(inputs).T)

        return np.hstack(D)

    def _extract_operators(self, Ohat):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squares problem.

        Parameters
        ----------
        Ohat : (r,d(r,m)) ndarray
            Block matrix of ROM operator coefficients, the transpose of the
            solution to the Operator Inference linear least-squares problem.
        """
        i = 0
        if self.has_constant:           # Constant term (one-dimensional).
            self.c_ = Ohat[:,i:i+1][:,0]
            i += 1

        if self.has_linear:             # Linear state matrix.
            self.A_ = Ohat[:,i:i+self.r]
            i += self.r

        if self.has_quadratic:          # (compact) Qudadratic state matrix.
            _r2 = self.r * (self.r + 1) // 2
            self.H_ = Ohat[:,i:i+_r2]
            i += _r2

        if self.has_cubic:              # (compact) Cubic state matrix.
            _r3 = self.r * (self.r + 1) * (self.r + 2) // 6
            self.G_ = Ohat[:,i:i+_r3]
            i += _r3

        if self.has_inputs:             # Linear input matrix.
            self.B_ = Ohat[:,i:i+self.m]
            i += self.m

        return

    def _construct_solver(self, basis, states, rhs, inputs, regularizer):
        """Construct a solver object mapping the regularizer to solutions
        of the Operator Inference least-squares problem.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and rhs are assumed to already be projected.
        states : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or time derivative
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).
        inputs : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0, (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB". This parameter is used here
            only to determine the correct type of solver.
        """
        states_, rhs_ = self._process_fit_arguments(basis, states, rhs, inputs)
        D = self._assemble_data_matrix(states_, inputs)
        self.solver_ = lstsq.solver(D, rhs_.T, regularizer)

    def _evaluate_solver(self, regularizer):
        """Evaluate the least-squares solver with the given regularizer.

        Parameters
        ----------
        regularizer : float >= 0, (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".
        """
        OhatT = self.solver_.predict(regularizer)
        self._extract_operators(np.atleast_2d(OhatT.T))

    def fit(self, basis, states, rhs, inputs, regularizer):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and rhs are assumed to already be projected.
        states : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or time derivative
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).
        inputs : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0, (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        self._construct_solver(basis, states, rhs, inputs, regularizer)
        self._evaluate_solver(regularizer)
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
    """
    def fit(self, basis, states, nextstates=None, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, states is assumed to already be projected (r,k).
        states : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        nextstates : (n,k) or (r,k) ndarray or None
            Column-wise snapshot training data corresponding to the next
            iteration of the state snapshots, i.e.,
            F(states[:,j]) = nextstates[:,j] where F is the full-order model.
            If None, assume state j+1 is the iteration after state j, i.e.,
            F(states[:,j]) = states[:,j+1].
        inputs : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0, (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        if nextstates is None:
            nextstates = states[:,1:]
            states = states[:,:-1]
        if inputs is not None:
            inputs = inputs[...,:states.shape[1]]
        return _InferredMixin.fit(self, basis,
                                  states, nextstates, inputs, regularizer)


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
    """
    def fit(self, basis, states, ddts, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and ddts are assumed to already be projected (r,k).
        states : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        ddts : (n,k) or (r,k) ndarray
            Column-wise time derivative training data (each column is a
            snapshot), either full order (n rows) or projected to reduced
            order (r rows).
        inputs : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0 or (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InferredMixin.fit(self, basis,
                                  states, ddts, inputs, regularizer)
