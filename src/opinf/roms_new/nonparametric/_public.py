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
from ... import errors


class SteadyROM(_NonparametricROM):  # pragma: no cover
    r"""Reduced-order model for a nonparametric steady state problem:

    .. math:: \widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat).

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the reduced-order model.
    """
    _LHS_ARGNAME = "forcing"
    _LHS_LABEL = "g"
    _STATE_LABEL = "q"
    _INPUT_LABEL = None
    # TODO: disallow input terms?

    def evaluate(self, state):
        r"""Evaluate the right-hand side of the model, i.e.,
        :math:`\widehat{\mathbf{F}}(\qhat)`.

        Parameters
        ----------
        state : (r,) ndarray
            Reduced-order state vector.

        Returns
        -------
        g_: (r,) ndarray
            Evaluation of the model.
        """
        return _NonparametricROM.evaluate(self, state, None)

    def fit(self, states, forcing=None, *, solver=None, regularizer=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        states : (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            compressed to the reduced-order state space.
        forcing : (r, k) ndarray or None
            Column-wise forcing data corresponding to the training snapshots.
        solver : lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * None: :class:`opinf.lstsq.PlainSolver()`, SVD-based solve without
              regularization
            * float > 0: :class:`opinf.lstsq.L2Solver()`, SVD-based solve with
              scalar Tikhonov regularization

        Returns
        -------
        self
        """
        if solver is None and regularizer is not None:
            solver = regularizer  # pragma: no cover
        return _NonparametricROM.fit(
            self, states, forcing, inputs=None, solver=solver
        )

    def jacobian(self, state):
        r"""Construct and sum the state Jacobian each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\ddqhat\widehat{\mathbf{F}}}`
        where the model can be written as
        :math:`\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat)`.

        Parameters
        ----------
        state : (r,) ndarray
            Reduced-order state vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian the right-hand side of the model.
        """
        return _NonparametricROM.jacobian(self, state, input_=None)

    def predict(self, forcing, guess=None):
        """Solve the model with the given forcing and initial guess."""
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
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the reduced-order model.
    """
    _LHS_ARGNAME = "nextstates"
    _LHS_LABEL = r"q_{j+1}"
    _STATE_LABEL = r"q_{j}"
    _INPUT_LABEL = r"u_{j}"

    @staticmethod
    def stack_trajectories(statelist, inputlist=None):
        """Translate a collection of state trajectories and (optionally) inputs
        to arrays that are appropriate arguments for :meth:`fit()`.

        Parameters
        ----------
        statelist : list of s (r, k_i) ndarrays
            Collection of state trajectories.
        inputlist : list of s (m, k_i) ndarrays
            Collection of inputs corresponding to the state trajectories.

        Returns
        -------
        states : (r, sum_i(k_i)) ndarray
            Snapshot matrix with data from all but the final snapshot of each
            trajectory in ``statelist``.
        nextstates : (r, sum_i(k_i)) ndarray
            Snapshot matrix with data from all but the first snapshot of each
            trajectory in ``statelist``.
        inputs : (r, sum_i(k_i)) ndarray
            Input matrix with data from all but the last input for each
            trajectory. Only returned if ``inputlist`` is provided.
        """
        states = np.hstack([Q[:, :-1] for Q in statelist])
        nextstates = np.hstack([Q[:, 1:] for Q in statelist])
        if inputlist is not None:
            inputs = np.hstack(
                [
                    U[..., : (S.shape[1] - 1)]
                    for S, U in zip(statelist, inputlist)
                ]
            )
            return states, nextstates, inputs
        return states, nextstates

    def evaluate(self, state, input_=None):
        r"""Evaluate and sum each model operator.

        This is the right-hand side of the model, i.e., the function
        :math:`\widehat{\mathbf{F}}` where the model can be written as
        :math:`\qhat_{j+1} = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state : (r,) ndarray
            Reduced-order state vector.
        input_ : (m,) ndarray or None
            Input vector corresponding to the state.

        Returns
        -------
        evaluation : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        return _NonparametricROM.evaluate(self, state, input_)

    def jacobian(self, state, input_=None):
        r"""Construct and sum the state Jacobian each model operators.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\ddqhat\widehat{\mathbf{F}}}`
        where the model can be written as
        :math:`\qhat_{j+1} = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state : (r,) ndarray
            Reduced-order state vector.
        input_ : (m,) ndarray or None
            Input vector corresponding to the state.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian the right-hand side of the model.
        """
        return _NonparametricROM.jacobian(self, state, input_)

    def fit(
        self,
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
        states : (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            compressed to the reduced-order state space.
        nextstates : (r, k) ndarray or None
            Column-wise snapshot training data corresponding to the next
            iteration of the compressed state snapshots.
            If ``None``, assume ``states[:, j+1]`` is the iteration following
            ``states[:, j]``.
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
            self, states, nextstates, inputs=inputs, solver=solver
        )

    def predict(self, state0, niters, inputs=None):
        """Step forward the reduced-order discrete dynamical system
        ``niters`` steps. Essentially, this amounts to the following.

        .. code-block:: python

           >>> states[:, 0] = state0
           >>> states[:, 1] = rom.evaluate(states[:, 0], inputs[:, 0])
           >>> states[:, 2] = rom.evaluate(states[:, 1], inputs[:, 1])
           ...                                     # Repeat `niters` times.

        Parameters
        ----------
        state0 : (r,) ndarray
            Initial reduced-order state.
        niters : int
            Number of times to step the system forward.
        inputs : (m, niters-1) ndarray or None
            Inputs for the next ``niters - 1`` time steps.

        Returns
        -------
        states : (r, niters) ndarray
            Solution to the system, including the initial condition ``state0``.
        """
        self._check_is_trained()

        # Check initial condition dimension and process inputs.
        if (_shape := np.shape(state0)) != (self.r,):
            raise errors.DimensionalityError(
                "initial condition not aligned with model "
                f"(state0.shape = {_shape} != ({self.r},) = (r,))"
            )
        self._check_inputargs(inputs, "inputs")

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 1:
            raise ValueError("argument 'niters' must be a positive integer")

        # Create the solution array and fill in the initial condition.
        states = np.empty((self.r, niters))
        states[:, 0] = state0.copy()

        # Run the iteration.
        if self._has_inputs:
            if callable(inputs):
                raise TypeError("inputs must be NumPy array, not callable")

            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(inputs)
            if U.ndim != 2 or U.shape[0] != self.m or U.shape[1] < niters - 1:
                raise ValueError(
                    f"inputs.shape = ({U.shape} "
                    f"!= {(self.m, niters-1)} = (m, niters-1)"
                )
            for j in range(niters - 1):
                states[:, j + 1] = self.evaluate(states[:, j], U[:, j])
        else:
            for j in range(niters - 1):
                states[:, j + 1] = self.evaluate(states[:, j])

        # Return state results.
        return states


class ContinuousROM(_NonparametricROM):
    r"""Reduced-order model for a nonparametric system of ordinary differential
    equations.

    .. math:: \ddt\qhat(t) = \Ophat(\qhat(t), \u(t)).

    Here, :math:`\qhat(t)\in\RR^{r}` is the reduced state
    and :math:`\u(t)\in\RR^{m}` is the (optional) input.
    The structure of :math:`\widehat{\mathbf{F}}` is specified through the
    ``operators`` argument.

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the reduced-order model.
    """
    _LHS_ARGNAME = "ddts"
    _LHS_LABEL = "dq / dt"
    _STATE_LABEL = "q(t)"
    _INPUT_LABEL = "u(t)"

    def evaluate(self, t, state, input_func=None):
        r"""Apply each operator and sum the results.

        This is the right-hand side of the model, i.e., the function
        :math:`\widehat{\mathbf{F}}` where the model can be written as
        :math:`\ddt \qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time, a scalar.
        state : (r,) ndarray
            Reduced-order state vector :math:`\qhat(t)`
            corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to an input vector of length
            ``m``.

        Returns
        -------
        dqdt_ : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricROM.evaluate(self, state, input_)

    def jacobian(self, t, state, input_func=None):
        r"""Construct and sum the state Jacobian each model operators.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function
        :math:`ddqhat\widehat{\mathbf{F}}(\qhat(t), \u(t))`
        where the model can be written as
        :math:`\ddt\qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time, a scalar.
        state : (r,) ndarray
            Reduced-order state vector corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to an vector of length ``m``.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricROM.jacobian(self, state, input_)

    def fit(self, states, ddts, inputs=None, solver=None, *, regularizer=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        states : (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            compressed to the reduced-order state space.
        ddts : (r, k) ndarray
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

    def predict(self, state0, t, input_func=None, **options):
        """Solve the reduced-order system of ordinary differential equations.
        This method wraps ``scipy.integrate.solve_ivp()``.

        Parameters
        ----------
        state0 : (r,) ndarray
            Initial state vector,compressed to reduced order.
        t : (nt,) ndarray
            Time domain over which to integrate the reduced-order model.
        input_func : callable or (m, nt) ndarray
            Input as a function of time (preferred) or the input values at the
            times `t`. If given as an array, cubic spline interpolation on the
            known data points is used as needed.
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
        states : (r, nt) ndarray
            Computed solution to the system over the time domain ``t``.
            A more detailed report on the integration results is stored as
            the ``predict_result_`` attribute.
        """
        self._check_is_trained()

        # Check initial condition dimension and process inputs.
        if (_shape := np.shape(state0)) != (self.r,):
            raise errors.DimensionalityError(
                "initial condition not aligned with model "
                f"(state0.shape = {_shape} != ({self.r},) = (r,))"
            )
        self._check_inputargs(input_func, "input_func")

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
            state0,  # from this initial condition
            args=(input_func,),  # with this input function
            t_eval=t,  # evaluated at these points
            **options,
        )  # using these solver options.

        # Warn if the integration failed.
        if not out.success:  # pragma: no cover
            warnings.warn(out.message, IntegrationWarning)

        # Return state results.
        self.predict_result_ = out
        return out.y
