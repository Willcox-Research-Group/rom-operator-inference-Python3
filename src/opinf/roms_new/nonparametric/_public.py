# roms/nonparametric/_public.py
"""Public nonparametric Operator Inference ROM classes."""

__all__ = [
    # "SteadyROM",
    "DiscreteROM",
    "ContinuousROM",
]

import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from ._base import _NonparametricROM


class SteadyROM(_NonparametricROM):  # pragma: no cover
    r"""Reduced-order model for a nonparametric steady state problem:

    ..math ::
        \widehat{\mathbf{g}}
        = \widehat{\mathbf{F}}(\qhat).

    Here q is the state and g is a forcing term. The structure of F(q) is user
    specified (modelform), and the corresponding low-dimensional operators are
    inferred through a least-squares regression.

    Attributes
    ----------
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
    c_, A_, H_ G_, B_ : Operator objects (see opinf.operators) or None
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
        return _NonparametricROM.evaluate(self, state_, None)

    def fit(self, states, forcing=None, *, solver=None, regularizer=None):
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
        known_operators : dict or None
            Dictionary of known full-order operators.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
        solver : lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:
            * None: lstsq.PlainSolver(), SVD-based solve without regularization
            * float > 0: lstsq.L2Solver(), SVD-based solve with scalar Tikhonov
                regularization

        Returns
        -------
        self
        """
        if solver is None and regularizer is not None:
            solver = regularizer  # pragma: no cover
        return _NonparametricROM.fit(
            self, states, forcing, inputs=None, solver=solver
        )

    def predict(self, forcing, guess=None):
        raise NotImplementedError("TODO")


class DiscreteROM(_NonparametricROM):
    r"""Reduced-order model for a nonparametric discrete dynamical system.

    .. math::
        \qhat_{j+1}
        = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j}).

    Here, :math:`\qhat\in\RR^{r}` is the reduced state
    and :math:`\u\in\RR^{m}` is the (optional) input.
    The structure of :math:`\widehat{\mathbf{F}}` is specified through the
    ``operators`` attribute.

    Parameters
    ----------
    basis : opinf.basis object or (n, r) ndarray
        Basis for the reduced space (e.g., POD).
    operators : list of opinf.operators objects
        Operators comprising the terms of the reduced-order model.
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
        r"""Evaluate and sum each model operator.

        This is the right-hand side of the model, i.e., the function
        :math:`\widehat{\mathbf{F}}` where the model can be written as
        :math:`\qhat_{j+1}
        = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state_ : (r,) ndarray
            Low-dimensional state vector.
        input_ : (m,) ndarray or None
            Input vector corresponding to the state.

        Returns
        -------
        evaluation : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        return _NonparametricROM.evaluate(self, state_, input_)

    def jacobian(self, state_, input_=None):
        r"""Construct and sum the Jacobian of each model operators.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\frac{
        \partial \widehat{\mathbf{F}}}{\partial \qhat}`
        where the model can be written as
        :math:`\qhat_{j+1}
        = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state_ : (r,) ndarray
            Low-dimensional state vector.
        input_ : (m,) ndarray or None
            Input vector corresponding to the state.

        Returns
        -------
        jac : (r, r) ndarray
            Jacobian of the right-hand side of the model.
        """
        return _NonparametricROM.jacobian(self, state_, input_)

    def fit(
        self,
        basis,
        states,
        nextstates=None,
        inputs=None,
        solver=None,
        *,
        regularizer=None,
    ):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            either full order (`n` rows) or compressed to reduced order
            (`r` rows).
        nextstates : (n, k) or (r, k) ndarray or None
            Column-wise snapshot training data corresponding to the next
            iteration of the state snapshots, i.e.,
            ``nextstates[:, j] = FOM(states[:, j], inputs[:, j])``
            where ``FOM`` is the full-order model.
            Each column is one snapshot, either full order (`n` rows) or
            compressed to reduced order (`r` rows). If ``None``, assume
            ``states[:, j+1]`` is the iteration following ``states[:, j]``.
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots.
            If one-dimensional, assume :math:`m = 1` (scalar input).
        solver : lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            - ``None``: ``lstsq.PlainSolver()``, SVD-based solve without
                regularization.
            - float > 0: ``lstsq.L2Solver()``, SVD-based solve with scalar
                Tikhonov regularization.

        Returns
        -------
        self
        """
        if nextstates is None and states is not None:
            nextstates = states[:, 1:]
            states = states[:, :-1]
        if inputs is not None:
            inputs = inputs[..., : states.shape[1]]
        if solver is None and regularizer is not None:
            solver = regularizer  # pragma: no cover
        return _NonparametricROM.fit(
            self, basis, states, nextstates, inputs=inputs, solver=solver
        )

    def predict(self, state0, niters, inputs=None, decompress=True):
        """Step forward the reduced-order discrete dynamical system
        ``niters`` steps. Essentially, this amounts to the following.

            >>> states_[:, 0] = rom.compress(state0)
            >>> states_[:, 1] = rom.evaluate(states_[:, 0], inputs[:, 0])
            >>> states_[:, 2] = rom.evaluate(states_[:, 1], inputs[:, 1])
            ...                                     # Repeat `niters` times.
            >>> states = rom.decompress(states_)

        Parameters
        ----------
        state0 : (n,) or (r,) ndarray
            Initial state vector, either full order (`n`-vector) or compressed
            to reduced order (`r`-vector).
        niters : int
            Number of times to step the system forward.
        inputs : (m, niters-1) ndarray or None
            Inputs for the next ``niters - 1`` time steps.
        decompress : bool
            If ``True`` and the ``basis`` is not ``None``, reconstruct the
            solutions in the original `n`-dimensional state space.

        Returns
        -------
        states : (n, niters) or (r, niters) ndarray
            Computed solution to the system, including the initial condition
            ``state0``. If the ``basis`` exists and ``decompress=True``, return
            solutions in the full state space (`n` rows); otherwise, return
            reduced solutions in the reduced state space (`r` rows).
        """
        self._check_is_trained()

        # Process inputs and project initial conditions if needed.
        self._check_inputargs(inputs, "inputs")
        state0_ = self.compress(state0, "state0")

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 1:
            raise ValueError("argument 'niters' must be a positive integer")

        # Create the solution array and fill in the initial condition.
        states_ = np.empty((self.r, niters))
        states_[:, 0] = state0_.copy()

        # Run the iteration.
        if self._has_inputs:
            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(inputs)
            if U.ndim != 2 or U.shape[0] != self.m or U.shape[1] < niters - 1:
                raise ValueError(
                    f"inputs.shape = ({U.shape} "
                    f"!= {(self.m, niters-1)} = (m, niters-1)"
                )
            for j in range(niters - 1):
                states_[:, j + 1] = self.evaluate(states_[:, j], U[:, j])
        else:
            for j in range(niters - 1):
                states_[:, j + 1] = self.evaluate(states_[:, j])

        # Return state results.
        if decompress and (self.basis is not None):
            return self.basis.decompress(states_)
        return states_


class ContinuousROM(_NonparametricROM):
    r"""Reduced-order model for a nonparametric system of ordinary differential
    equations.

    .. math::
        \frac{\textup{d}}{\textup{d}t} \qhat(t)
        = \widehat{\mathbf{F}}(\qhat(t), \u(t)).

    Here, :math:`\qhat(t)\in\RR^{r}` is the reduced state
    and :math:`\u(t)\in\RR^{m}` is the (optional) input.
    The structure of :math:`\widehat{\mathbf{F}}` is specified through the
    ``operators`` attribute.

    Parameters
    ----------
    basis : opinf.basis object or (n, r) ndarray
        Basis for the reduced space (e.g., POD).
    operators : list of opinf.operators objects
        Operators comprising the terms of the reduced-order model.
    """
    _LHS_ARGNAME = "ddts"
    _LHS_LABEL = "dq / dt"
    _STATE_LABEL = "q(t)"
    _INPUT_LABEL = "u(t)"

    def evaluate(self, t, state_, input_func=None):
        r"""Evaluate and sum each model operator.

        This is the right-hand side of the model, i.e., the function
        :math:`\widehat{\mathbf{F}}` where the model can be written as
        :math:`\frac{\textup{d}}{\textup{d}t} \qhat(t)
        = \widehat{\mathbf{F}}(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time, a scalar.
        state_ : (r,) ndarray
            Low-dimensional state vector :math:`\qhat(t)`
            corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to an input vector of length
            `m`.

        Returns
        -------
        dqdt_ : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricROM.evaluate(self, state_, input_)

    def jacobian(self, t, state_, input_func=None):
        r"""Construct and sum the Jacobian of each model operators.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\frac{
        \partial \widehat{\mathbf{F}}}{\partial \qhat}`
        where the model can be written as
        :math:`\frac{\textup{d}}{\textup{d}t} \qhat(t)
        = \widehat{\mathbf{F}}(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time, a scalar.
        state_ : (r,) ndarray
            Low-dimensional state vector :math:`\qhat(t)`
            corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to an input vector of length
            `m`.

        Returns
        -------
        jac : (r, r) ndarray
            Jacobian of the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricROM.jacobian(self, state_, input_)

    def fit(self, states, ddts, inputs=None, solver=None, *, regularizer=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            either full order (`n` rows) or compressed to reduced order
            (`r` rows).
        ddts : (n, k) or (r, k) ndarray
            Column-wise time derivative training data. Each column
            ``ddts[:, j]`` corresponds to the snapshot ``states[:, j]``.
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots.
            If one-dimensional, assume :math:`m = 1` (scalar input).
        solver : opinf.lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: ``opinf.lstsq.PlainSolver()``, SVD-based solve without
              regularization.
            * **float > 0:** ``opinf.lstsq.L2Solver()``, SVD-based solve with
              scalar Tikhonov regularization.

        Returns
        -------
        self
        """

        if solver is None and regularizer is not None:
            solver = regularizer  # pragma: no cover
        return _NonparametricROM.fit(
            self, states, ddts, inputs=inputs, solver=solver
        )

    def predict(self, state0, t, input_func=None, decompress=True, **options):
        """Solve the reduced-order system of ordinary differential equations.
        This method wraps ``scipy.integrate.solve_ivp()``.

        Parameters
        ----------
        state0 : (n,) or (r,) ndarray
            Initial state vector, either full order (`n`-vector) or compressed
            to reduced order (`r`-vector).
        t : (nt,) ndarray
            Time domain over which to integrate the reduced-order model.
        input_func : callable or (m, nt) ndarray
            Input as a function of time (preferred) or the input values at the
            times `t`. If given as an array, cubic spline interpolation on the
            known data points is used as needed.
        decompress : bool
            If ``True`` and the ``basis`` is not ``None``, reconstruct the
            solutions in the original `n`-dimensional state space.
        options
            Arguments for ``scipy.integrate.solve_ivp()``,
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.
            Common options:

            * **method : str** ODE solver for the reduced-order model.

              * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
              * 'RK23': Explicit Runge-Kutta method of order 3(2).
              * 'Radau': Implicit Runge-Kutta method of the Radau IIA family
                of order 5.
              * 'BDF': Implicit multi-step variable-order (1 to 5) method
                based on a backward differentiation formula for the
                derivative.
              * 'LSODA': Adams/BDF method with automatic stiffness detection
                and switching. This wraps the Fortran solver from ODEPACK.

            * **max_step : float** Maximimum allowed integration step size.

        Returns
        -------
        states : (n, nt) or (r, nt) ndarray
            Computed solution to the system over the time domain ``t``.
            If the ``basis`` exists and ``decompress=True``, return
            solutions in the full state space (`n` rows); otherwise, return
            reduced solutions in the reduced state space (`r` rows).
            A more detailed report on the integration results is stored as
            the ``predict_result_`` attribute.
        """
        self._check_is_trained()

        # Process inputs and project initial conditions if needed.
        self._check_inputargs(input_func, "input_func")
        state0_ = self.compress(state0, "state0")

        # Verify time domain.
        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Interpret control input argument `input_func`.
        if self._has_inputs:
            if not callable(input_func):
                # input_func must be (m, nt) ndarray. Interploate -> callable.
                U = np.atleast_2d(input_func)
                if U.shape != (self.m, nt):
                    raise ValueError(
                        f"input_func.shape = {U.shape} "
                        f"!= {(self.m, nt)} = (m, len(t))"
                    )
                input_func = CubicSpline(t, U, axis=1)

            # Check dimension of input_func() outputs.
            _tmp = input_func(t[0])
            _shape = _tmp.shape if isinstance(_tmp, np.ndarray) else None
            if self.m == 1:
                if not (np.isscalar(_tmp) or _shape == (1,)):
                    raise ValueError(
                        "input_func() must return ndarray"
                        " of shape (m,) = (1,) or scalar"
                    )
            elif _shape != (self.m,):
                raise ValueError(
                    "input_func() must return ndarray"
                    f" of shape (m,) = ({self.m},)"
                )

        if "method" in options and options["method"] in (
            # These methods require the Jacobian.
            "BDF",
            "Radau",
            "LSODA",
        ):
            options["jac"] = self.jacobian

        # Integrate the reduced-order model.
        out = solve_ivp(
            self.evaluate,  # Integrate this function
            [t[0], t[-1]],  # over this time interval
            state0_,  # from this initial condition
            args=(input_func,),  # with this input function
            t_eval=t,  # evaluated at these points
            **options,
        )  # using these solver options.

        # Warn if the integration failed.
        if not out.success:  # pragma: no cover
            warnings.warn(out.message, IntegrationWarning)

        # Return state results.
        self.predict_result_ = out
        if decompress and (self.basis is not None):
            return self.basis.decompress(out.y)
        return out.y
