# models/nonparametric/_public.py
"""Public nonparametric model classes."""

__all__ = [
    # "SteadyModel",
    "DiscreteModel",
    "ContinuousModel",
]

import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from ._base import _NonparametricModel
from ... import errors


class SteadyModel(_NonparametricModel):  # pragma: no cover
    r"""Nonparametric steady state model.

    .. math:: \widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat).

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the model.
    """
    _LHS_ARGNAME = "forcing"
    _LHS_LABEL = "g"
    _STATE_LABEL = "q"
    _INPUT_LABEL = None
    # TODO: disallow input terms?

    def rhs(self, state):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the function :math:`\widehat{\mathbf{F}}(\qhat)`
        where the model can be written as
        :math:`\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat)`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.

        Returns
        -------
        g: (r,) ndarray
            Evaluation of the right-hand-side of the model.
        """
        return _NonparametricModel.rhs(self, state, None)

    def jacobian(self, state):
        r"""Sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function
        :math:`\ddqhat\widehat{\mathbf{F}}}(\qhat)`
        where the model can be written as
        :math:`\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat)`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        return _NonparametricModel.jacobian(self, state, input_=None)

    def fit(self, states, forcing=None, *, solver=None, regularizer=None):
        """Learn the model operators from data.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        forcing : (r, k) ndarray or None
            Forcing training data. Each column ``forcing[:, j]``
            corresponds to the snapshot ``states[:, j]``.
            If ``None``, set ``forcing = 0``.
        solver : lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * None: :class:`opinf.lstsq.PlainSolver()`,
              SVD-based solve without regularization
            * **float > 0**: :class:`opinf.lstsq.L2Solver()`,
              SVD-based solve with scalar Tikhonov regularization

        Returns
        -------
        self
        """
        if solver is None and regularizer is not None:
            solver = regularizer  # pragma: no cover
        return _NonparametricModel.fit(
            self, states, forcing, inputs=None, solver=solver
        )

    def predict(self, forcing, guess=None):
        """Solve the model with the given forcing and initial guess."""
        raise NotImplementedError("TODO")


class DiscreteModel(_NonparametricModel):
    r"""Nonparametric discrete dynamical system model.

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
        Operators comprising the terms of the model.
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

    def rhs(self, state, input_=None):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the function :math:`\widehat{\mathbf{F}}(\qhat, \u)`
        where the model can be written as
        :math:`\qhat_{j+1} = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math:`\u`.

        Returns
        -------
        nextstate : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        return _NonparametricModel.rhs(self, state, input_)

    def jacobian(self, state, input_=None):
        r"""Sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function
        :math:`\ddqhat\widehat{\mathbf{F}}}(\qhat, \u)`
        where the model can be written as
        :math:`\qhat_{j+1} = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math`\u`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        return _NonparametricModel.jacobian(self, state, input_)

    def fit(
        self,
        states,
        nextstates=None,
        inputs=None,
        solver=None,
        *,
        regularizer=None,
    ):
        """Learn the model operators from data.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        nextstates : (r, k) ndarray or None
            Next iteration training data. Each column ``nextstates[:, j]``
            is the iteration following ``states[:, j]``.
            If ``None``, set ``nextstates[:, j] = states[:, j+1]``.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).
        solver : lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: :class:`opinf.lstsq.PlainSolver()`,
              SVD-based solve without regularization.
            * **float > 0**: :class:`opinf.lstsq.L2Solver()`,
              SVD-based solve with scalar Tikhonov regularization.

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
        return _NonparametricModel.fit(
            self, states, nextstates, inputs=inputs, solver=solver
        )

    def predict(self, state0, niters, inputs=None):
        """Step forward the reduced-order discrete dynamical system
        ``niters`` steps. Essentially, this amounts to the following.

        .. code-block:: python

           >>> states[:, 0] = state0
           >>> states[:, 1] = rom.rhs(states[:, 0], inputs[:, 0])
           >>> states[:, 2] = rom.rhs(states[:, 1], inputs[:, 1])
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
        if (_shape := np.shape(state0)) != (self.state_dimension,):
            raise errors.DimensionalityError(
                "initial condition not aligned with model "
                f"(state0.shape = {_shape} != "
                f"({self.state_dimension},) = (r,))"
            )
        self._check_inputargs(inputs, "inputs")

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 1:
            raise ValueError("argument 'niters' must be a positive integer")

        # Create the solution array and fill in the initial condition.
        states = np.empty((self.state_dimension, niters))
        states[:, 0] = state0.copy()

        # Run the iteration.
        if self._has_inputs:
            if callable(inputs):
                raise TypeError("inputs must be NumPy array, not callable")

            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(inputs)
            if (
                U.ndim != 2
                or U.shape[0] != self.input_dimension
                or U.shape[1] < niters - 1
            ):
                raise ValueError(
                    f"inputs.shape = ({U.shape} "
                    f"!= {(self.input_dimension, niters-1)} = (m, niters-1)"
                )
            for j in range(niters - 1):
                states[:, j + 1] = self.rhs(states[:, j], U[:, j])
        else:
            for j in range(niters - 1):
                states[:, j + 1] = self.rhs(states[:, j])

        # Return state results.
        return states


class ContinuousModel(_NonparametricModel):
    r"""Nonparametric system of ordinary differential equations.

    .. math:: \ddt\qhat(t) = \Ophat(\qhat(t), \u(t)).

    Here, :math:`\qhat(t)\in\RR^{r}` is the state
    and :math:`\u(t)\in\RR^{m}` is the (optional) input.
    The structure of :math:`\widehat{\mathbf{F}}` is specified through the
    ``operators`` argument.

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the model.
    """
    _LHS_ARGNAME = "ddts"
    _LHS_LABEL = "dq / dt"
    _STATE_LABEL = "q(t)"
    _INPUT_LABEL = "u(t)"

    def rhs(self, t, state, input_func=None):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the right-hand side of the model, i.e., the function
        :math:`\widehat{\mathbf{F}}(\qhat(t), \u(t))`
        where the model can be written as
        :math:`\ddt \qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time :math:`t`, a scalar.
        state : (r,) ndarray
            State vector :math:`\qhat(t)` corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to the input vector
            :math:`\u(t)`.

        Returns
        -------
        dqdt : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricModel.rhs(self, state, input_)

    def jacobian(self, t, state, input_func=None):
        r"""Sum the state Jacobian each model operators.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function
        :math:`ddqhat\widehat{\mathbf{F}}(\qhat(t), \u(t))`
        where the model can be written as
        :math:`\ddt\qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))`.

        Parameters
        ----------
        t : float
            Time :math:`t`, a scalar.
        state : (r,) ndarray
            State vector :math:`\qhat(t)` corresponding to time ``t``.
        input_func : callable(float) -> (m,), or None
            Input function that maps time ``t`` to an input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        input_ = None if not self._has_inputs else input_func(t)
        return _NonparametricModel.jacobian(self, state, input_)

    def fit(self, states, ddts, inputs=None, solver=None, *, regularizer=None):
        """Learn the model operators from data.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        ddts : (r, k) ndarray
            Snapshot time derivative training data. Each column
            ``ddts[:, j]`` corresponds to the snapshot ``states[:, j]``.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).
        solver : opinf.lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: ``opinf.lstsq.PlainSolver()``,
              SVD-based solve without regularization.
            * **float > 0:** ``opinf.lstsq.L2Solver()``,
              SVD-based solve with scalar Tikhonov regularization.

        Returns
        -------
        self
        """

        if solver is None and regularizer is not None:
            solver = regularizer  # pragma: no cover
        return _NonparametricModel.fit(
            self, states, ddts, inputs=inputs, solver=solver
        )

    def predict(self, state0, t, input_func=None, **options):
        """Solve the reduced-order system of ordinary differential equations.
        This method wraps ``scipy.integrate.solve_ivp()``.

        Parameters
        ----------
        state0 : (r,) ndarray
            Initial state vector.
        t : (nt,) ndarray
            Time domain over which to integrate the model.
        input_func : callable or (m, nt) ndarray
            Input as a function of time (preferred) or the input values at the
            times ``t``. If given as an array, cubic spline interpolation on
            the known data points is used as needed.
        options
            Arguments for ``scipy.integrate.solve_ivp()``,
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.
            Common options:

            * **method : str** ODE solver for the model.

              * `'RK45'` (default): Explicit Runge-Kutta method of order 5(4).
              * `'RK23'`: Explicit Runge-Kutta method of order 3(2).
              * `'Radau'`: Implicit Runge-Kutta method of the Radau IIA family
                of order 5.
              * `'BDF'`: Implicit multi-step variable-order (1 to 5) method
                based on a backward differentiation formula for the
                derivative.
              * `'LSODA'`: Adams/BDF method with automatic stiffness detection
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
        if (_shape := np.shape(state0)) != (self.state_dimension,):
            raise errors.DimensionalityError(
                "initial condition not aligned with model "
                f"(state0.shape = {_shape} != "
                f"({self.state_dimension},) = (r,))"
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
                if U.shape != (self.input_dimension, nt):
                    raise ValueError(
                        f"input_func.shape = {U.shape} "
                        f"!= {(self.input_dimension, nt)} = (m, len(t))"
                    )
                input_func = CubicSpline(t, U, axis=1)

            # Check dimension of input_func() outputs.
            _tmp = input_func(t[0])
            _shape = _tmp.shape if isinstance(_tmp, np.ndarray) else None
            if self.input_dimension == 1:
                if not (np.isscalar(_tmp) or _shape == (1,)):
                    raise ValueError(
                        "input_func() must return ndarray"
                        " of shape (m,) = (1,) or scalar"
                    )
            elif _shape != (self.input_dimension,):
                raise ValueError(
                    "input_func() must return ndarray"
                    f" of shape (m,) = ({self.input_dimension},)"
                )

        if "method" in options and options["method"] in (
            # These methods require the Jacobian.
            "BDF",
            "Radau",
            "LSODA",
        ):
            options["jac"] = self.jacobian

        # Integrate the model.
        out = solve_ivp(
            self.rhs,  # Integrate this function
            [t[0], t[-1]],  # over this time interval
            state0,  # from this initial condition
            args=(input_func,),  # with this input function
            t_eval=t,  # evaluated at these points
            **options,  # using these solver options.
        )

        # Warn if the integration failed.
        if not out.success:  # pragma: no cover
            warnings.warn(out.message, IntegrationWarning)

        # Return state results.
        self.predict_result_ = out
        return out.y
