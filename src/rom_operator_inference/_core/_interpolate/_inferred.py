# _core/_interpolate/_inferred.py
"""Parametric ROM classes that use interpolation and Operator Inference.

Classes
-------
* _InterpolatedInferredMixin(_InferredMixin, _InterpolatedMixin)
* InterpolatedInferredDiscreteROM(_InterpolatedInferredMixin, _DiscreteROM)
* InterpolatedInferredContinuousROM(_InterpolatedInferredMixin, _ContinuousROM)
"""

__all__ = [
            "InterpolatedInferredDiscreteROM",
            "InterpolatedInferredContinuousROM",
          ]

import numpy as np
import scipy.interpolate as interp

from ._base import _InterpolatedMixin
from .._base import _DiscreteROM, _ContinuousROM
from .._inferred import (_InferredMixin,
                         InferredDiscreteROM,
                         InferredContinuousROM)


# Interpolated inferred mixin (private) =======================================
class _InterpolatedInferredMixin(_InferredMixin, _InterpolatedMixin):
    """Mixin for interpolatory ROM classes that use Operator Inference."""

    def _process_fit_arguments(self, ModelClass,
                               basis, params, states, rhss, inputs):
        """Do sanity checks, extract dimensions, and check data sizes."""
        # TODO: self.p = self._check_params(params)
        # ^extract self.p and check for consistent sizes.^
        self._check_inputargs(inputs, 'inputs')
        self._clear()

        # Check that parameters are one-dimensional.
        params = np.array(params)
        self.p = 1 if params.ndim == 1 else params.shape[1]

        # Check that the number of params matches the number of training sets.
        s = len(params)
        to_check = [(states, "states"), (rhss, self._RHS_LABEL)]
        if self.has_inputs:
            self.m = 1 if inputs[0].ndim == 1 else inputs[0].shape[0]
            to_check.append((inputs, "inputs"))
        else:
            inputs = [None]*s
        self._check_number_of_training_datasets(s, to_check)

        # Store basis and reduced dimension.
        self.basis = basis
        if basis is None:
            self.r = states[0].shape[0]

        return inputs

    def fit(self, ModelClass, basis,
            params, states, rhss, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        ModelClass: class
            ROM class, either _ContinuousROM or _DiscreteROM, to use for the
            newly constructed model.

        basis : (n,r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and rhss are assumed to already be projected (r,k).

        params : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        states : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array states[i] corresponds to the ith parameter params[i].

        rhss : list of s (n,k) or (r,k) ndarrays or None
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array states[i] corresponds to the ith parameter params[i].
            Igored if the model is discrete (according to `ModelClass`).

        inputs : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        regularizer : float >= 0 or (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        inputs = self._process_fit_arguments(ModelClass, basis, params,
                                             states, rhss, inputs)

        # TODO: figure out how to handle regularizer...
        # (scalar, array, list(arrays)).

        # Train one model per parameter sample.
        self.models_ = []
        for µ, X, dX, U in zip(params, states, rhss, inputs):
            model = ModelClass(self.modelform)
            model.fit(basis, X, dX, U, regularizer)
            model.parameter = µ
            self.models_.append(model)

        # Select the interpolator based on the parameter dimension.
        if self.p == 1:
            Interpolator = interp.CubicSpline
        else:
            print("MODELS TRAINED BUT INTERPOLATION NOT IMPLEMENTED FOR p > 1")
            return self

        # Construct interpolators.
        for atr in self.modelform:
            ops = getattr(self, f"{atr}s_")     # ops = self.cs_
            op = Interpolator(params, ops)
            op.shape = ops[0].shape
            setattr(self, f"{atr}_", op)        # self.c_ = op

        return self


# Interpolated inferred models (public) =======================================
class InterpolatedInferredDiscreteROM(_InterpolatedInferredMixin,
                                      _DiscreteROM):
    """Reduced order model for a high-dimensional discrete dynamical system,
    parametrized by a scalar µ, of the form

        x_{j+1}(µ) = f(x_{j}(µ), u_{j}; µ),     x_{0}(µ) = x0(µ).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    ordinary least-squares problems, then interpolating those models with
    respect to the scalar parameter µ.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        Structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.
    """
    def fit(self, basis, params,
            states, nextstates=None, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, states are assumed to already be projected (r,k).
        params : (s,) ndarray
            Parameter values at which the snapshot data is collected.
        states : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array states[i] corresponds to the ith parameter params[i].
        nextstates : list of s (n,k) or (r,k) ndarrays OR None
            Column-wise snapshot training data corresponding to the next
            iteration of the state snapshots, i.e.,
            F(states[i][:,j]) = nextstates[i][:,j] where F is the full-order
            model. If None, assume state j+1 is the iteration after state j,
            i.e., F(states[i][:,j]) = states[i][:,j+1].
        inputs : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0 or (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        if nextstates is None:
            nextstates = [None]*len(states)
        return _InterpolatedInferredMixin.fit(self, InferredDiscreteROM,
                                              basis, params,
                                              states, nextstates, inputs,
                                              regularizer)

    def predict(self, µ, x0, niters, inputs=None):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then step forward this new ROM `niters` steps.

        Parameters
        ----------
        µ : float
            Parameter of interest for the prediction.
        x0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).
        niters : int
            Number of times to step the system forward.
        inputs : (m,niters-1) ndarray
            Inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) ndarray
            Approximate solutions to the full-order system, including the
            given initial condition.
        """
        return self(µ).predict(x0, niters, inputs)


class InterpolatedInferredContinuousROM(_InterpolatedInferredMixin,
                                        _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs, parametrized
    by a scalar µ, of the form

         dx / dt = f(t, x(t;µ), u(t); µ),       x(0;µ) = x0(µ).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    ordinary least-squares problems, then interpolating those models with
    respect to the scalar parameter µ.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        Structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c(µ)
        'A' : Linear state term A(µ)x(t).
        'H' : Quadratic state term H(µ)(x⊗x)(t).
        'H' : Cubic state term G(µ)(x⊗x⊗x)(t).
        'B' : Input term B(µ)u(t).
        For example, modelform=="cA" means f(t, x(t); µ) = c(µ) + A(µ)x(t;µ).
    """
    def fit(self, basis, params, states, ddts, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and ddts are assumed to already be projected (r,k).
        params : (s,) ndarray
            Parameter values at which the snapshot data is collected.
        states : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array states[i] corresponds to the ith parameter params[i].
        ddts : list of s (n,k) or (r,k) ndarrays
            Column-wise time derivative training data (each column is a
            snapshot), either full order (n rows) or projected to reduced
            order (r rows). The ith array ddts[i] corresponds to the ith
            parameter, params[i].
        inputs : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0 or (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see utils.lstsq_reg(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        Returns
        -------
        self
        """
        return _InterpolatedInferredMixin.fit(self, InferredContinuousROM,
                                              basis, params,
                                              states, ddts, inputs,
                                              regularizer)

    def predict(self, µ, x0, t, input_func=None, **options):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then simulate this interpolated ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : float
            Parameter of interest for the prediction.
        x0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).
        t : (nt,) ndarray
            Time domain over which to integrate the reduced-order system.
        input_func : callable or (m,nt) ndarray
            Input as a function of time (preferred) or the input at the
            times `t`. If given as an array, u(t) is approximated by a cubic
            spline interpolating the known data points.

        options
            Arguments for solver.integrate.solve_ivp(), such as the following:
            method : str
                ODE solver for the reduced-order system.
                * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
                * 'RK23': Explicit Runge-Kutta method of order 3(2).
                * 'Radau': Implicit Runge-Kutta method of the Radau IIA family
                    of order 5.
                * 'BDF': Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the
                    derivative.
                * 'LSODA': Adams/BDF method with automatic stiffness detection
                    and switching. This wraps the Fortran solver from ODEPACK.
            max_step : float
                Maximimum allowed integration step size.
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        X_ROM : (n,nt) ndarray
            Reduced-order approximation to the full-order system over `t`.
        """
        model = self(µ)
        out = model.predict(x0, t, input_func, **options)
        self.sol_ = model.sol_
        return out
