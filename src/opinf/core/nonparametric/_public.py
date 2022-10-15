# core/nonparametric/_public.py
"""Public nonparametric Operator Inference ROM classes."""

__all__ = [
    # "SteadyOpInfROM",
    "DiscreteOpInfROM",
    "ContinuousOpInfROM",
]

import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from .._base import _BaseROM
from ._base import _NonparametricOpInfROM


class SteadyOpInfROM(_NonparametricOpInfROM):               # pragma: no cover
    """Reduced-order model for a nonparametric steady state problem:

        g = F(q).

    Here q is the state and g is a forcing term. The structure of F(q) is user
    specified (modelform), and the corresponding low-dimensional operators are
    inferred through a least-squares regression.

    Attributes
    ----------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        Structure of the reduced-order model. Each character indicates one term
        in the low-dimensional function F(q, u):
        'c' : Constant term c
        'A' : Linear state term Aq.
        'H' : Quadratic state term H[q ⊗ q].
        'G' : Cubic state term G[q ⊗ q ⊗ q].
        For example, modelform="AH" means F(q) = Aq + H[q ⊗ q].
    n : int
        Dimension of the high-dimensional state.
    m : int or None
        Dimension of the input, or None if no inputs are present.
    r : int
        Dimension of the low-dimensional (reduced-order) state.
    basis : (n, r) ndarray or None
        Basis matrix defining the relationship between the high- and
        low-dimensional state spaces. If None, arguments of fit() are assumed
        to be in the reduced dimension.
    c_, A_, H_ G_, B_ : Operator objects (see opinf.core.operators) or None
        Low-dimensional operators composing the reduced-order model.
    """
    _LHS_ARGNAME = "forcing"
    _LHS_LABEL = "g"
    _STATE_LABEL = "q"
    _INPUT_LABEL = None
    # TODO: disallow input terms?

    def evaluate(self, state_):
        """Evaluate the right-hand side of the model, i.e., F(q).

        Parameters
        ----------
        state_ : (r,) ndarray
            Low-dimensional state vector q.

        Returns
        -------
        g_: (r,) ndarray
            Evaluation of the model.
        """
        return _BaseROM.evaluate(self, state_, None)

    def fit(self, basis, states, forcing=None,
            regularizer=0, known_operators=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the reduced state space (e.g., POD basis matrix).
            If None, states and forcing are assumed to already be projected.
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        forcing : (n, k) or (r, k) ndarray or None
            Column-wise forcing data corresponding to the training snapshots,
            either full order (n rows) or projected to reduced order (r rows).
        regularizer : float >= 0, (d, d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + r(r+1)/2 when `modelform`="AH".
        known_operators : dict or None
            Dictionary of known full-order operators.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.

        Returns
        -------
        self
        """
        return _NonparametricOpInfROM.fit(self, basis,
                                          states, forcing, None,
                                          regularizer, known_operators)

    def predict(self, forcing, guess=None):
        raise NotImplementedError("TODO")


class DiscreteOpInfROM(_NonparametricOpInfROM):
    r"""Reduced-order model for a nonparametric discrete dynamical system:

        q_{j+1} = F(q_{j}, u_{j}),         q_{0} = q0.

    Here q is the state and u is the (optional) input. The structure of F(q, u)
    is user specified (modelform), and the corresponding low-dimensional
    operators are inferred through a least-squares regression.

    Attributes
    ----------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        Structure of the reduced-order model. Each character indicates one term
        in the low-dimensional function F(q, u):
        'c' : Constant term c
        'A' : Linear state term Aq.
        'H' : Quadratic state term H[q ⊗ q].
        'G' : Cubic state term G[q ⊗ q ⊗ q].
        'B' : Input term Bu.
        For example, modelform="AB" means F(q, u) = Aq + Bu.
    n : int
        Dimension of the high-dimensional state.
    m : int or None
        Dimension of the input, or None if no inputs are present.
    r : int
        Dimension of the low-dimensional (reduced-order) state.
    basis : (n, r) ndarray or None
        Basis matrix defining the relationship between the high- and
        low-dimensional state spaces. If None, arguments of fit() are assumed
        to be in the reduced dimension.
    c_, A_, H_ G_, B_ : Operator objects (see opinf.core.operators) or None
        Low-dimensional operators composing the reduced-order model.
    """
    _LHS_ARGNAME = "nextstates"
    _LHS_LABEL = r"q_{j+1}"
    _STATE_LABEL = r"q_{j}"
    _INPUT_LABEL = r"u_{j}"

    # TODO: convenience method for dealing with multiple trajectories (bursts).
    # @staticmethod
    # def stack_training_bursts(statelist):
    #     """Translate a collection of state trajectories to
    #     (states, nextstates) arrays that are appropriate arguments for fit().
    #
    #     Parameters
    #     ----------
    #     statelist : list of s (n, k_i) ndarrays
    #         Collection of state trajectories from various initial conditions.
    #         Q = statelist[i] is the snapshot matrix for Q.shape[1] iterations
    #         starting at initial condition q0 = Q[i, 0].
    #
    #     Returns
    #     -------
    #     states : (n, sum_i(k_i)-s) ndarray
    #         States
    #     nextstates : (n, sum_i(k_i)-s) ndarray
    #         Nextstates
    #     """
    #     states = np.hstack([Q[:, :-1] for Q in statelist])
    #     nextstates = np.hstack([Q[:, 1:] for Q in statelist])
    #     return states, nextstates

    def evaluate(self, state_, input_=None):
        r"""Evaluate the right-hand side of the model, i.e., the F(q, u) of

            q_{j+1} = F(q_{j}, u_{j}).

        Parameters
        ----------
        state_ : (r,) ndarray
            Low-dimensional state vector q_{j}.
        input_ : (m,) ndarray or None
            Input vector u_{j} corresponding to the state.

        Returns
        -------
        nextstate_: (r,) ndarray
            Evaluation q_{j+1} of the model.
        """
        return _BaseROM.evaluate(self, state_, input_)

    def fit(self, basis, states, nextstates=None, inputs=None,
            regularizer=0, known_operators=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the reduced state space (e.g., POD basis matrix).
            If None, states and nextstates are assumed to already be projected.
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        nextstates : (n, k) or (r, k) ndarray or None
            Column-wise snapshot training data corresponding to the next
            iteration of the state snapshots, i.e.,
            FOM(states[:, j], inputs[:, j]) = nextstates[:, j]
            where FOM is the full-order model.
            If None, assume state j+1 is the iteration after state j, i.e.,
            FOM(states[:, j], inputs[:, j]) = states[:, j+1].
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be
            one-dimensional if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0, (d, d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + r(r+1)/2 when `modelform`="AH".
        known_operators : dict or None
            Dictionary of known full-order operators.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.

        Returns
        -------
        self
        """
        if nextstates is None and states is not None:
            nextstates = states[:, 1:]
            states = states[:, :-1]
        if inputs is not None:
            inputs = inputs[..., :states.shape[1]]
        return _NonparametricOpInfROM.fit(self, basis,
                                          states, nextstates, inputs,
                                          regularizer, known_operators)

    def predict(self, state0, niters, inputs=None, decode=True):
        """Step forward the learned ROM `niters` steps.

        Parameters
        ----------
        state0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected to
            reduced order (r-vector).
        niters : int
            Number of times to step the system forward.
        inputs : (m, niters-1) ndarray
            Inputs for the next niters-1 time steps, i.e.,
            >>> states[:, 0] = states0
            >>> states[:, 1] = F(state0, inputs[:, 0])
            >>> states[:, 2] = F(states[:, 1], inputs[:, 1])
            ...
        decode : bool
            If True and the basis is not None, reconstruct the solutions in
            the original n-dimensional state space.

        Returns
        -------
        states : (n, niters) or (r, niters) ndarray
            Approximate solution to the system, including the given
            initial condition. If the basis exists and decode=True,
            return solutions in the full n-dimensional state space (n rows);
            otherwise, return reduced-order state solution (r rows).
        """
        self._check_is_trained()

        # Process inputs and project initial conditions if needed.
        self._check_inputargs(inputs, "inputs")
        state0_ = self.encode(state0, "state0")

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 1:
            raise ValueError("argument 'niters' must be a positive integer")

        # Create the solution array and fill in the initial condition.
        states_ = np.empty((self.r, niters))
        states_[:, 0] = state0_.copy()

        # Run the iteration.
        if 'B' in self.modelform:
            if callable(inputs):
                raise TypeError("inputs must be NumPy array, not callable")
            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(inputs)
            if U.ndim != 2 or U.shape[0] != self.m or U.shape[1] < niters - 1:
                raise ValueError("inputs.shape = "
                                 f"({U.shape} != {(self.m, niters-1)}")
            for j in range(niters - 1):
                states_[:, j+1] = self.evaluate(states_[:, j], U[:, j])
        else:
            for j in range(niters - 1):
                states_[:, j+1] = self.evaluate(states_[:, j])

        # Return state results.
        if decode and (self.basis is not None):
            return self.decode(states_)
        return states_


class ContinuousOpInfROM(_NonparametricOpInfROM):
    """Reduced-order model for a nonparametric system of ordinary differential
    equations:

        dq / dt = F(t, q(t), u(t)),         q(0) = q0.

    Here q(t) is the state and u(t) is the (optional) input. The structure of
    F(t, q(t), u(t)) is user specified (modelform), and the corresponding
    low-dimensional operators are inferred through a least-squares regression.

    Attributes
    ----------
    modelform : str
        Structure of the reduced-order model. Each character indicates one
        term in the low-dimensional model F(t, q(t), u(t)):
        'c' : Constant term c.
        'A' : Linear state term Aq(t).
        'H' : Quadratic state term H[q(t) ⊗ q(t)].
        'G' : Cubic state term G[q(t) ⊗ q(t) ⊗ q(t)].
        'B' : Input term Bu(t).
        For example, modelform="AB" means F(t, q(t), u(t)) = Aq(t) + Bu(t).
    n : int
        Dimension of the high-dimensional state.
    m : int or None
        Dimension of the input, or None if no inputs are present.
    r : int
        Dimension of the low-dimensional (reduced-order) state.
    basis : (n, r) ndarray or None
        Basis matrix defining the relationship between the high- and
        low-dimensional state spaces. If None, arguments of fit() are assumed
        to be in the reduced dimension.
    c_, A_, H_ G_, B_ : Operator objects (see opinf.core.operators) or None
        Low-dimensional operators composing the reduced-order model.
    """
    _LHS_ARGNAME = "ddts"
    _LHS_LABEL = "dq / dt"
    _STATE_LABEL = "q(t)"
    _INPUT_LABEL = "u(t)"

    def evaluate(self, t, state_, input_func=None):
        """Evaluate the right-hand side of the model, i.e., the F(t, q, u) of

            dq / dt = F(t, q(t), u(t)).

        Parameters
        ----------
        t : float
            Time, a scalar.
        state_ : (r,) ndarray
            Reduced state vector q(t) corresponding to time `t`.
        input_func : callable(float) -> (m,)
            Input function that maps time `t` to an input vector of length m.

        Returns
        -------
        dqdt_: (r,) ndarray
            Evaluation of the model.
        """
        input_ = None if 'B' not in self.modelform else input_func(t)
        return _BaseROM.evaluate(self, state_, input_)

    def fit(self, basis, states, ddts, inputs=None,
            regularizer=0, known_operators=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the reduced state space (e.g., POD basis matrix).
            If None, states and ddts are assumed to already be projected.
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).
        ddts : (n, k) or (r, k) ndarray
            Column-wise time derivative training data (each column is a
            snapshot), either full order (n rows) or projected to reduced
            order (r rows).
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be
            one-dimensional if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0 or (d, d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".
        known_operators : dict or None
            Dictionary of known full-order operators.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.

        Returns
        -------
        self
        """
        return _NonparametricOpInfROM.fit(self, basis,
                                          states, ddts, inputs,
                                          regularizer, known_operators)

    def predict(self, state0, t, input_func=None, decode=True, **options):
        """Simulate the learned ROM with scipy.integrate.solve_ivp().

        Parameters
        ----------
        state0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).
        t : (nt,) ndarray
            Time domain over which to integrate the reduced-order model.
        input_func : callable or (m, nt) ndarray
            Input as a function of time (preferred) or the input at the
            times `t`. If given as an array, cubic spline interpolation
            on the known data points is used as needed.
        decode : bool
            If True and the basis is not None, reconstruct the solutions
            in the original n-dimensional state space.
        options
            Arguments for scipy.integrate.solve_ivp(), such as the following:
            method : str
                ODE solver for the reduced-order model.
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
        states : (n, nt) or (r, nt) ndarray
            Approximate solution to the system over the time domain `t`.
            If the basis exists and decode=True, return solutions in the
            original n-dimensional state space (n rows); otherwise, return
            reduced-order state solutions (r rows).
            A more detailed report on the integration results is stored as
            the attribute `predict_result_`.
        """
        self._check_is_trained()

        # Process inputs and project initial conditions if needed.
        self._check_inputargs(input_func, "input_func")
        state0_ = self.encode(state0, "state0")

        # Verify time domain.
        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Interpret control input argument `input_func`.
        if 'B' in self.modelform:
            if not callable(input_func):
                # input_func must be (m, nt) ndarray. Interploate -> callable.
                U = np.atleast_2d(input_func)
                if U.shape != (self.m, nt):
                    raise ValueError("input_func.shape = "
                                     f"{U.shape} != {(self.m, nt)}")
                input_func = CubicSpline(t, U, axis=1)

            # Check dimension of input_func() outputs.
            _tmp = input_func(t[0])
            _shape = _tmp.shape if isinstance(_tmp, np.ndarray) else None
            if self.m == 1:
                if not (np.isscalar(_tmp) or _shape == (1,)):
                    raise ValueError("input_func() must return ndarray"
                                     " of shape (m,) = (1,) or scalar")
            elif _shape != (self.m,):
                raise ValueError("input_func() must return ndarray"
                                 f" of shape (m,) = {(self.m,)}")

        # Integrate the reduced-order model.
        out = solve_ivp(self.evaluate,          # Integrate this function
                        [t[0], t[-1]],          # over this time interval
                        state0_,                # with this initial condition
                        t_eval=t,               # evaluated at these points
                        # jac=self.jacobian,          # with this Jacobian
                        args=(input_func,),     # with this input function
                        **options)              # and these solver options.

        # Warn if the integration failed.
        if not out.success:                           # pragma: no cover
            warnings.warn(out.message, IntegrationWarning)

        # Return state results.
        self.predict_result_ = out
        if decode and (self.basis is not None):
            return self.decode(out.y)
        return out.y
