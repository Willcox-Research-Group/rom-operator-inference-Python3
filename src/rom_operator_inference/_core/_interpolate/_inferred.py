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

from ._base import _InterpolatedMixin, _Interp2DMulti
from .._base import _DiscreteROM, _ContinuousROM
from .._inferred import (_InferredMixin,
                        InferredDiscreteROM,
                        InferredContinuousROM)


# Interpolated inferred mixin (private) =======================================
class _InterpolatedInferredMixin(_InferredMixin, _InterpolatedMixin):
    """Mixin for interpolatory ROM classes that use Operator Inference."""

    def _process_fit_arguments(self, ModelClass, Vr, µs, Xs, Xdots, Us):
        """Do sanity checks, extract dimensions, check and fix data sizes, and
        get perpare arguments for individual Operator Inference ROMs.

        Returns
        -------
        is_continuous : bool
            Whether or not ModelClass is a subclass of _ContinuousROM.

        Xdots : list of s (r,k) ndarrays (or None)
            Right-hand-side data. rhss_[i] corresponds to µ[i].

        Us : list of s (m,k) ndarrays (or None)
            Inputs, potentially reshaped. Us[i] corresponds to µ[i].
        """
        # Check modelform, affines dictionary, and inputs.
        self._check_modelform(trained=False)
        # TODO: self.p = self._check_params(µs): extract self.p and check for consistent sizes.
        self._check_inputargs(Us, 'Us')
        is_continuous = issubclass(ModelClass, _ContinuousROM)

        # Check that parameters are one-dimensional.
        µs = np.array(µs)
        self.p = µs.ndim

        # Check that the number of params matches the number of training sets.
        s = µs.shape[0]
        if len(Xs) != s:
            raise ValueError("num parameter samples != num state snapshot "
                             f"training sets ({s} != {len(Xs)})")
        if is_continuous and len(Xdots) != s:
            raise ValueError("num parameter samples != num time derivative "
                             f"training sets ({s} != {len(Xdots)})")
        elif not is_continuous:
            Xdots = [None] * s
        if self.has_inputs and len(Us) != s:
            raise ValueError("num parameter samples != num input "
                             f"training sets ({s} != {len(Us)})")

        # Store basis and dimensions.
        if Vr is not None:
            self.n, self.r = Vr.shape   # Full dimension, reduced dimension.
        else:
            self.n = None               # No full dimension.
            self.r = Xs[0].shape[0]     # Reduced dimension.
        self.Vr = Vr

        # Ensure training data sets have consistent sizes.
        if self.has_inputs:
            if not isinstance(Us, list):
                Us = list(Us)
            self.m = 1 if Us[0].ndim == 1 else Us[0].shape[0]
            for i in range(s):
                if Us[i].ndim == 1:     # Reshape one-dimensional inputs.
                    Us[i] = Us[i].reshape((1,-1))
                if is_continuous:
                    self._check_training_data_shapes([Xs[i], Xdots[i], Us[i]],
                                                     [f"Xs[{i}]",
                                                      f"Xdots[{i}]",
                                                      f"Us[{i}]"])
                else:
                    self._check_training_data_shapes([Xs[i], Us[i]],
                                                     [f"Xs[{i}]", f"Us[{i}]"])
        else:
            self.m = None
            Us = [None] * s
            if is_continuous:
                for i in range(s):
                    self._check_training_data_shapes([Xs[i], Xdots[i]],
                                                     [f"Xs[{i}]",
                                                      f"Xdots[{i}]"])
        return is_continuous, µs, Xdots, Us

    def fit(self, ModelClass, Vr, µs, Xs, Xdots, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        ModelClass: class
            ROM class, either _ContinuousROM or _DiscreteROM, to use for the
            newly constructed model.

        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, Xs and rhss are assumed to already be projected (r,k).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) or (r,k) ndarrays or None
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].
            Igored if the model is discrete (according to `ModelClass`).

        Us : list of s (m,k) or (k,) ndarrays or None
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
        continuous, µs, Xdots, Us = self._process_fit_arguments(ModelClass,
                                                               Vr, µs,
                                                               Xs, Xdots, Us)

        # TODO: figure out how to handle P (scalar, array, list(arrays)).

        # Train one model per parameter sample.
        self.Vr = Vr
        self.models_ = []
        for µ, X, Xdot, U in zip(µs, Xs, Xdots, Us):
            model = ModelClass(self.modelform)
            if continuous:
                model.fit(Vr, X, Xdot, U, P)
            else:
                model.fit(Vr, X, U, P)
            model.parameter = µ
            self.models_.append(model)

        # Select the interpolator based on the parameter dimension.
        if self.p == 1:
            Interpolator = interp.CubicSpline
        elif self.p == 2:
            Interpolator = _Interp2DMulti
        else:
            print("MODELS TRAINED BUT INTERPOLATION NOT IMPLEMENTED FOR p > 2")
            return self

        # Construct interpolators.
        for lbl, atr in zip(["constant","linear","quadratic","cubic","inputs"],
                            ["c",       "A",     "Hc",       "Gc",   "B"]):
            if getattr(self, f"has_{lbl}"):         # if self.has_constant
                ops = getattr(self, f"{atr}s_")     # ops = self.cs_
                op = Interpolator(µs, ops)
                op.shape = ops[0].shape
                setattr(self, f"{atr}_", op)        # self.c_ = op
            else:
                setattr(self, f"{atr}_", None)      # self.c_ = None

        return self


# Interpolated inferred models (public) =======================================
class InterpolatedInferredDiscreteROM(_InterpolatedInferredMixin, _DiscreteROM):
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

    s : int
        The number of training parameter samples, hence also the number of
        reduced models computed via inference and used in the interpolation.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    dataconds_ : (s,) ndarray
        Condition numbers of the raw data matrices for each least-squares
        problem.

    dataregconds_ : (s,) ndarray
        Condition numbers of the regularized data matrices for each
        least-squares problem.

    residuals_ : (s,) ndarray
        The squared Frobenius-norm residuals of the regularized least-squares
        problems for computing each set of reduced-order model operators.

    misfits_ : (s,) ndarray
        The squared Frobenius-norm data misfits of the (nonregularized)
        least-squares problems for computing each set of reduced-order model
        operators.

    cs_ : list of s (r,) ndarrays or None
        Learned ROM constant terms, or None if 'c' is not in `modelform`.

    As_ : list of s (r,r) ndarrays or None
        Learned ROM linear state matrices, or None if 'A' not in `modelform`.

    Hcs_ : list of s (r,r(r+1)/2) ndarrays or None
        Learned ROM quadratic state matrices (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    Hs_ : list of s (r,r**2) ndarrays or None
        Learned ROM quadratic state matrices (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hcs_ if desired; not used in
        solving the ROM.

    Gcs_ : list of s (r,r(r+1)(r+2)/6) ndarrays or None
        Learned ROM cubic state matrices (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    Gs_ : list of s (r,r**3) ndarrays or None
        Learned ROM cubic state matrices (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gcs_ if desired; not used in
        solving the ROM.

    Bs_ : list of s (r,m) ndarrays or None
        Learned ROM input matrices, or None if 'B' not in `modelform`.

    fs_ : list of func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operators for each parameter sample, defined by
        cs_, As_, and/or Hcs_.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, µs, Xs, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, Xs are assumed to already be projected (r,k).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Us : list of s (m,k-1) or (k-1,) ndarrays or None
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
        return _InterpolatedInferredMixin.fit(self, InferredDiscreteROM,
                                              Vr, µs, Xs, None, Us, P)

    def predict(self, µ, x0, niters, U=None):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then step forward this new ROM `niters` steps.

        Parameters
        ----------
        µ : float
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) ndarray
            The approximate solutions to the full-order system, including the
            given initial condition.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        model = self(µ)     # See __call__().
        return model.predict(x0, niters, U)


class InterpolatedInferredContinuousROM(_InterpolatedInferredMixin, _ContinuousROM):
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
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c(µ)
        'A' : Linear state term A(µ)x(t).
        'H' : Quadratic state term H(µ)(x⊗x)(t).
        'H' : Cubic state term G(µ)(x⊗x⊗x)(t).
        'B' : Input term B(µ)u(t).
        For example, modelform=="cA" means f(t, x(t); µ) = c(µ) + A(µ)x(t;µ).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c(µ).

    has_linear : bool
        Whether or not there is a linear state term A(µ)x(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(µ)(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(µ)(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term B(µ)u(t).

    n : int
        The dimension of the original model.

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    s : int
        The number of training parameter samples, hence also the number of
        reduced models computed via inference and used in the interpolation.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    dataconds_ : (s,) ndarray
        Condition numbers of the raw data matrices for each least-squares
        problem.

    dataregconds_ : (s,) ndarray
        Condition numbers of the regularized data matrices for each
        least-squares problem.

    residuals_ : (s,) ndarray
        The squared Frobenius-norm residuals of the regularized least-squares
        problems for computing each set of reduced-order model operators.

    misfits_ : (s,) ndarray
        The squared Frobenius-norm data misfits of the (nonregularized)
        least-squares problems for computing each set of reduced-order model
        operators.

    cs_ : list of s (r,) ndarrays or None
        Learned ROM constant terms, or None if 'c' is not in `modelform`.

    As_ : list of s (r,r) ndarrays or None
        Learned ROM linear state matrices, or None if 'A' not in `modelform`.

    Hcs_ : list of s (r,r(r+1)/2) ndarrays or None
        Learned ROM quadratic state matrices (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    Hs_ : list of s (r,r**2) ndarrays or None
        Learned ROM quadratic state matrices (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hcs_ if desired; not used in
        solving the ROM.

    Gcs_ : list of s (r,r(r+1)(r+2)/6) ndarrays or None
        Learned ROM cubic state matrices (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    Gs_ : list of s (r,r**3) ndarrays or None
        Learned ROM cubic state matrices (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gcs_ if desired; not used in
        solving the ROM.

    Bs_ : list of s (r,m) ndarrays or None
        Learned ROM input matrices, or None if 'B' not in `modelform`.

    fs_ : list of func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operators for each parameter sample, defined by
        cs_, As_, and/or Hcs_.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, µs, Xs, Xdots, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, Xs and Xdots are assumed to already be projected (r,k).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) or (r,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) or (r,k) ndarrays
            Column-wise velocity training data (each column is a snapshot),
            either full order (n rows) ro projected to reduced order (r rows).
            The ith array Xdots[i] corresponds to the ith parameter, µs[i].

        Us : list of s (m,k) or (k,) ndarrays or None
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
        return _InterpolatedInferredMixin.fit(self, InferredContinuousROM,
                                              Vr, µs, Xs, Xdots, Us, P)

    def predict(self, µ, x0, t, u=None, **options):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then simulate this interpolated ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : float
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        t : (nt,) ndarray
            The time domain over which to integrate the reduced-order system.

        u : callable or (m,nt) ndarray
            The input as a function of time (preferred) or the input at the
            times `t`. If given as an array, u(t) is approximated by a cubic
            spline interpolating the known data points.

        options
            Arguments for solver.integrate.solve_ivp(), such as the following:
            method : str
                The ODE solver for the reduced-order system.
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
                The maximimum allowed integration step size.
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        X_ROM : (n,nt) ndarray
            The approximate solution to the full-order system over `t`.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        model = self(µ)     # See __call__().
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out
