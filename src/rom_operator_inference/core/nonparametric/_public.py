# core/nonparametric/_public.py
"""Public nonparametric Operator Inference ROM classes."""

__all__ = [
    "SteadyOpInfROM",
    "DiscreteOpInfROM",
    "ContinuousOpInfROM",
]

import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from .._base import _BaseROM
from ._base import _NonparametricOpInfROM


class SteadyOpInfROM(_NonparametricOpInfROM):
    """Mixin for models that solve the steady-state ROM problem,

        g = f(q, u).

    The problem may also be parametric, i.e., q and f may depend on an
    independent parameter µ.
    """
    _LHS_ARGNAME = "forcing"
    _LHS_LABEL = "g"
    _STATE_LABEL = "q"
    _INPUT_LABEL = "u"


class DiscreteOpInfROM(_NonparametricOpInfROM):
    """Mixin for models that solve the discrete-time ROM problem,

        q_{j+1} = f(q_{j}, u_{j}),         q_{0} = q0.

    The problem may also be parametric, i.e., q and f may depend on an
    independent parameter µ.
    """
    _LHS_ARGNAME = "nextstates"
    _LHS_LABEL = r"q_{j+1}"
    _STATE_LABEL = r"q_{j}"
    _INPUT_LABEL = r"u_{j}"
    modelform = property(_BaseROM.modelform.fget,
                         _BaseROM.modelform.fset,
                         _BaseROM.modelform.fdel,
    """Structure of the reduced-order model. Each character
    indicates the presence of a different term in the model:
    'c' : Constant term c
    'A' : Linear state term Aq_{j}.
    'H' : Quadratic state term H[q_{j} ⊗ q_{j}].
    'G' : Cubic state term G[q_{j} ⊗ q_{j} ⊗ q_{j}].
    'B' : Input term Bu_{j}.
    For example, modelform="AB" means f(q, u) = Aq + Bu.
    """)

    def f_(self, q_, u=None):
        """Reduced-order model function for discrete models.

        Parameters
        ----------
        q_ : (r,) ndarray
            Reduced state vector.
        u : (m,) ndarray or float or None
            Input vector corresponding to the state q_.
        """
        q_new = np.zeros_like(self.r, dtype=float)
        if self.has_constant:
            q_new += self.c_()
        if self.has_linear:
            q_new += self.A_(q_)
        if self.has_quadratic:
            q_new += self.H_(q_)
        if self.has_cubic:
            q_new += self.G_(q_)
        if self.has_inputs:
            q_new += self.B_(u)
        return q_new

    def predict(self, q0, niters, inputs=None):
        """Step forward the learned ROM `niters` steps.

        Parameters
        ----------
        q0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected to
            reduced order (r-vector).
        niters : int
            Number of times to step the system forward.
        inputs : (m,niters-1) ndarray
            Inputs for the next niters-1 time steps.

        Returns
        -------
        states : (n,niters) or (r,niters) ndarray
            Approximate solution to the system, including the given
            initial condition. If the basis is None, return solutions in the
            reduced r-dimensional subspace (r,niters). Otherwise, map solutions
            to the full n-dimensional space with the basis (n,niters).
        """
        self._check_is_trained()

        # Process inputs and project initial conditions if needed.
        self._check_inputargs(inputs, "inputs")
        q0_ = self.project(q0, "q0")

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 0:
            raise ValueError("argument 'niters' must be a nonnegative integer")

        # Create the solution array and fill in the initial condition.
        states = np.empty((self.r,niters))
        states[:,0] = q0_.copy()

        # Run the iteration.
        if self.has_inputs:
            if callable(inputs):
                raise TypeError("inputs must be an array, not a callable")
            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(inputs)
            if U.ndim != 2 or U.shape[0] != self.m or U.shape[1] < niters - 1:
                raise ValueError("invalid input shape "
                                 f"({U.shape} != {(self.m,niters-1)}")
            for j in range(niters-1):
                states[:,j+1] = self.f_(states[:,j], U[:,j])    # f(xj,uj)
        else:
            for j in range(niters-1):
                states[:,j+1] = self.f_(states[:,j])            # f(xj)

        # Reconstruct the approximation to the full-order model if possible.
        return self.basis @ states if self.basis is not None else states


class ContinuousOpInfROM(_NonparametricOpInfROM):
    """Base class for models that solve the continuous (ODE) ROM problem,

        dx / dt = f(t, q(t), u(t)),         q(0) = q0.

    The problem may also be parametric, i.e., q and f may depend on an
    independent parameter µ.
    """
    _LHS_ARGNAME = "ddts"
    _LHS_LABEL = "dq / dt"
    _STATE_LABEL = "q(t)"
    _INPUT_LABEL = "u(t)"
    modelform = property(_BaseROM.modelform.fget,
                         _BaseROM.modelform.fset,
                         _BaseROM.modelform.fdel,
    """Structure of the reduced-order model. Each character
    indicates the presence of a different term in the model:
    'c' : Constant term c
    'A' : Linear state term Aq(t).
    'H' : Quadratic state term H[q(t) ⊗ q(t)].
    'G' : Cubic state term G[q(t) ⊗ q(t) ⊗ q(t)].
    'B' : Input term Bu(t).
    For example, modelform="AB" means f(t, q(t), u(t)) = Aq(t) + Bu(t).
    """)

    def f_(self, t, q_, u=None):
        """Reduced-order model function for continuous models.

        Parameters
        ----------
        t : float
            Time, a scalar.
        q_ : (r,) ndarray
            Reduced state vector corresponding to time `t`.
        u : func(float) -> (m,)
            Input function that maps time `t` to an input vector of length m.
        """
        dqdt_ = np.zeros(self.r, dtype=float)
        if self.has_constant:
            dqdt_ += self.c_()
        if self.has_linear:
            dqdt_ += self.A_(q_)
        if self.has_quadratic:
            dqdt_ += self.H_(q_)
        if self.has_cubic:
            dqdt_ += self.G_(q_)
        if self.has_inputs:
            dqdt_ += self.B_(u(t))
        return dqdt_

    def predict(self, q0, t, input_func=None, reconstruct=True, **options):
        """Simulate the learned ROM with scipy.integrate.solve_ivp().

        Parameters
        ----------
        q0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).
        t : (nt,) ndarray
            Time domain over which to integrate the reduced-order system.
        input_func : callable or (m,nt) ndarray
            Input as a function of time (preferred) or the input at the
            times `t`. If given as an array, a cubic spline interpolates the
            known data points as needed.
        reconstruct : bool
            If True and the basis is not None, map the solutions to the full
            n-dimensional space.
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
        states : (n,nt) or (r,nt) ndarray
            Approximate solution to the system over the time domain `t`.
            If the basis is None, return solutions in the reduced
            r-dimensional subspace (r,nt). Otherwise, map the solutions to the
            full n-dimensional space with the basis (n,nt).
        """
        self._check_is_trained()

        # Process inputs and project initial conditions if needed.
        self._check_inputargs(input_func, "input_func")
        q0_ = self.project(q0, "q0")

        # Verify time domain.
        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Interpret control input argument `u`.
        if self.has_inputs:
            if callable(input_func):
                out = input_func(t[0])
                if np.isscalar(out):
                    if self.m == 1:     # u : R -> R, wrap output as array.
                        _u = input_func

                        def input_func(s):
                            """Wrap scalar inputs as a 2D array"""
                            return np.array([_u(s)])

                    else:               # u : R -> R, but m != 1.
                        raise ValueError("input_func() must return ndarray"
                                         f" of shape (m,)={(self.m,)}")
                elif not isinstance(out, np.ndarray):
                    raise ValueError("input_func() must return ndarray"
                                     f" of shape (m,)={(self.m,)}")
                elif out.shape != (self.m,):
                    message = "input_func() must return ndarray" \
                              f" of shape (m,)={(self.m,)}"
                    if self.m == 1:
                        raise ValueError(message + " or scalar")
                    raise ValueError(message)
            else:                   # input_func not callable ((m,nt) array).
                U = np.atleast_2d(input_func)
                if U.shape != (self.m,nt):
                    raise ValueError("invalid input shape "
                                     f"({U.shape} != {(self.m,nt)}")
                input_func = CubicSpline(t, U, axis=1)
            def fun(t, q_):
                return self.f_(t, q_, input_func)
        else:
            fun = self.f_  # (t, q_)

        # Integrate the reduced-order model.
        # TODO: rename self.predict_result_
        out = solve_ivp(fun,                    # Integrate f_(t, q_, u)
                        [t[0], t[-1]],          # over this time interval
                        q0_,                    # with this initial condition
                        t_eval=t,               # evaluated at these points
                        # jac=self._jac,          # with this Jacobian
                        **options)              # and these solver options.

        # Warn if the integration failed.
        if not out.success:                           # pragma: no cover
            warnings.warn(out.message, IntegrationWarning)

        # Return state results.
        self.predict_result_ = out
        if reconstruct and (self.basis is not None):
            return self.reconstruct(out.y)
        return out.y
