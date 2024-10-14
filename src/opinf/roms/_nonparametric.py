# roms/_nonparametric.py
"""Nonparametric ROM class."""

__all__ = [
    "ROM",
]

import warnings
import numpy as np

from ..models import _utils as modutils
from ._base import _BaseROM


def _identity(x):
    return x


class ROM(_BaseROM):
    r"""Nonparametric reduced-order model.

    This class connects classes from the various submodules to form a complete
    reduced-order modeling workflow.

    High-dimensional data
    :math:`\to` transformed / preprocessed data
    :math:`\to` compressed data
    :math:`\to` low-dimensional model.

    Parameters
    ----------
    model : :mod:`opinf.models` object
        Nonparametric system model, an instance of one of the following:

        * :class:`opinf.models.ContinuousModel`
        * :class:`opinf.models.DiscreteModel`
    lifter : :mod:`opinf.lift` object or None
        Lifting transformation.
    transformer : :mod:`opinf.pre` object or None
        Preprocesser.
    basis : :mod:`opinf.basis` object or None
        Dimensionality reducer.
    ddt_estimator : :mod:`opinf.ddt` object or None
        Time derivative estimator.
        Ignored if ``model`` is not time continuous.
    """

    def __init__(
        self,
        model,
        *,
        lifter=None,
        transformer=None,
        basis=None,
        ddt_estimator=None,
    ):
        """Store each argument as an attribute."""
        if not modutils.is_nonparametric(model):
            raise TypeError("'model' must be a nonparametric model instance")
        super().__init__(model, lifter, transformer, basis, ddt_estimator)

    # Training and evaluation -------------------------------------------------
    def _fix_single_trajectory(sefl, states, lhs, inputs):
        """If data comes from a single trajectory, ``states``, ``lhs``, and/or
        ``inputs`` may be arrays instead of lists of arrays. This method casts
        each input as a list of arrays if ``states`` is an array.
        """
        if states[0].ndim == 1:
            states = [states]
            if lhs is not None:
                lhs = [lhs]
            if inputs is not None:
                inputs = [inputs]
        return states, lhs, inputs

    def _fit_and_return_training_data(
        self,
        states,
        lhs,
        inputs,
        fit_transformer,
        fit_basis,
    ):
        """Process the training data, fit the model, and return the processed
        training data.
        """
        self._check_fit_args(lhs=lhs, inputs=inputs)
        states, lhs, inputs = self._fix_single_trajectory(states, lhs, inputs)

        states, lhs, inputs = super().fit(
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )

        # Concatentate trajectories.
        if inputs is not None:
            inputs = np.hstack(inputs)
        self.model.fit(np.hstack(states), np.hstack(lhs), inputs)

        return states, inputs

    def fit(
        self,
        states,
        lhs=None,
        inputs=None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
    ):
        """Calibrate the model to training data.

        Parameters
        ----------
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data corresponding to a different trajectory;
            each column ``states[i][:, j]`` is one snapshot.
            If there is only one trajectory of training data (s = 1),
            ``states`` may be an (n, k) ndarray. In this case, it is assumed
            that ``lhs`` and ``inputs`` (if given) are arrays, not a sequence
            of arrays.
        lhs : list of s (n, k_i) ndarrays or None
            Left-hand side regression data. Each array ``lhs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``lhs[i][:, j]`` corresponds to the snapshot ``states[i][:, j]``.

            - If the model is time continuous, these are the time derivatives
              of the state snapshots.
            - If the model is fully discrete, these are the "next states"
              corresponding to the state snapshots.

            If ``None``, these are estimated using :attr:`ddt_estimator`
            (time continuous) or extracted from ``states`` (fully discrete).
        inputs : list of s (m, k_i) ndarrays or None
            Input training data. Each array ``inputs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``inputs[i][:, j]`` corresponds to the snapshot ``states[:, j]``.
            May be a two-dimensional array if :math:`m=1` (scalar input).
            Only required if one or more model operators depend on inputs.
        fit_transformer : bool
            If ``True`` (default), calibrate the preprocessing transformation
            using the ``states``. If ``False``, assume the transformer is
            already calibrated.
        fit_basis : bool
            If ``True``, calibrate the high-to-low dimensional mapping
            using the ``states``.
            If ``False``, assume the basis is already calibrated.

        Returns
        -------
        self
        """
        self._fit_and_return_training_data(
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )
        return self

    def fit_regselect(
        self,
        states: list,
        candidates: list,
        lhs: list = None,
        inputs: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        train_time_domains: np.ndarray = None,
        test_time_length: float = 0,
        num_test_iters: int = 0,
        stability_margin: float = 5.0,
        verbose: bool = False,
        **predict_options: dict,
    ):
        """Calibrate the model to training data, selecting the regularization
        hyperparameter(s) that minimize the training error while maintaining
        stability over the testing regime.

        This method requires the :attr:`model` to have a ``solver`` of one of
        the following types:

        * :class:`opinf.lstsq.L2Solver`
        * :class:`opinf.lstsq.L2DecoupledSolver`
        * :class:`opinf.lstsq.TikhonovSolver`
        * :class:`opinf.lstsq.TikhonovDecoupledSolver`

        The ``solver.regularizer`` is repeatedly adjusted, and the model is
        recalibrated, until a best regularization is selected.

        The following parameters are always required or optional.

        .. list-table::

           * - **Required**
             - ``states``, ``candidates``
           * - **Optional**
             - ``lhs``, ``inputs``, ``fit_transformer``, ``fit_basis``,
               ``regularizer_factory``, ``gridsearch_only``,
               ``stability_margin``, ``verbose``

        Whether the remaining parameters are required, optional, or not allowed
        depends on the type of :attr:`model`

        .. list-table::
           :header-rows: 1

           * -
             - Time-continuous models
             - Time-discrete models
           * - **Required**
             - ``train_time_domains``
             -
           * - **Optional**
             - ``test_time_length``
             - ``num_test_iters``
           * - **Not allowed**
             - ``num_test_iters``
             - ``train_time_domains``, ``test_time_length``

        Parameters
        ----------
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data corresponding to a different trajectory;
            each column ``states[i][:, j]`` is one snapshot.
            If there is only one trajectory of training data (s = 1),
            ``states`` may be an (n, k) ndarray. In this case, it is assumed
            that ``lhs`` and ``inputs`` (if given) are arrays, not a sequence
            of arrays.
        lhs : list of s (n, k_i) ndarrays or None
            Left-hand side regression data. Each array ``lhs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``lhs[i][:, j]`` corresponds to the snapshot ``states[i][:, j]``.

            - If the model is time continuous, these are the time derivatives
              of the state snapshots.
            - If the model is fully discrete, these are the "next states"
              corresponding to the state snapshots.

            If ``None``, these are estimated using :attr:`ddt_estimator`
            (time continuous) or extracted from ``states`` (fully discrete).
        candidates : list of regularization parameters
            Regularization hyperparameters to check. If a single hyperparameter
            is given, use it as the start of an optimization-based search.
        inputs : list of s (m, k_i + num_test_iters) ndarrays or callables None
            Input training data. Only required if the :attr:`model` takes
            external inputs. This argument is different for time-continuous and
            time-discrete models: TODO
            Each array ``inputs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``inputs[i][:, j]`` corresponds to the snapshot ``states[:, j]``.
            May be a two-dimensional array if :math:`m=1` (scalar input).
        fit_transformer : bool
            If ``True`` (default), calibrate the preprocessing transformation
            using the ``states``. If ``False``, assume the transformer is
            already calibrated.
        fit_basis : bool
            If ``True``, calibrate the high-to-low dimensional mapping
            using the ``states``.
            If ``False``, assume the basis is already calibrated.
        train_time_domain : list of s (k_i,) ndarrays or (k,) ndarray or None
            Time domain corresponding to the training states. This argument is
            **required** if the model is time continuous. If a single
            one-dimensional array is provided, assume that all training
            trajectories are measured over the same time domain.
        test_time_domain : list of S (K,) ndarrays or None
            Time domain(s) over which to enforce model stability. This argument
            should only be provided if the model is time continuous.
        test_niters : list of S ints or None
            Number of iterations over which to enforce mode stability. This
            argument should only be provided if the model is fully discrete.
        test_initial_conditions : list of S (K,) ndarrays or None
            Initial condition(s) for stability enforcement checks. If ``None``
            (default), use the initial conditions of the training states.
        test_inputs : list of S callables/(m, test_niters_i) ndarrays or None
            Input function(s) for stability enforcement checks. This argument
            is **required** if the model takes inputs. If the model is time
            continuous this may be a callable, an array, or a list of these; if
            the model is fully discrete, this must be an array or a list of
            arrays. Most importantly, this argument must align with the
            ``test_initial_conditions``.
        predict_method : str or None
            Integration method to use in the model predictions. This argument
            is only used if the model is time continuous.
        candidates : list of regularization parameters
            Regularization hyperparameters to check. If a single hyperparameter
            is given, use it as the start of an optimization-based search.
        regularizer_factory : callable or None
            Function mapping regularization hyperparameters to the full
            regularizer.
        stability_margin : float,
            Factor by which the reduced states may deviate from the range of
            the training data without being flagged as unstable.
        verbose : bool
            If ``True``, print information during the regularization selection.

        Raises
        ------
        ValueError
            If any required arguments are missing.

        Warns
        -----
        OpInfWarning
            If any arguments are provided that are ignored because of the
            type of model.
        """
        # Check required and disallowed arguments.
        mtype = "time continuous" if self._iscontinuous else "fully discrete"

        def _warn_is_ignored(argname):
            warnings.warn(
                f"argument '{argname} is ignored when model is {mtype}"
            )

        def _is_required(argname):
            raise ValueError(
                f"argument '{argname}' is required when model is {mtype}"
            )

        # Validate arguments.
        states, lhs, inputs = self._fix_single_trajectory(states, lhs, inputs)
        if self._iscontinuous:
            if train_time_domains is None:
                _is_required("train_time_domain")
            if num_test_iters != 0:
                _warn_is_ignored("num_test_iters")
            if np.isscalar(train_time_domains[0]):
                train_time_domains = [train_time_domains] * len(states)
            if inputs is not None:
                if not callable(inputs[0]):
                    raise TypeError(
                        "argument 'inputs' must be sequence of callables"
                    )
                input_functions = inputs
                inputs = [
                    u(t) for u, t in zip(input_functions, train_time_domains)
                ]
            else:
                input_functions = None
            if test_time_length < 0:
                raise ValueError(
                    "argument 'test_time_length' must be nonnegative"
                )
        else:
            if train_time_domains is not None:
                _warn_is_ignored("train_time_domain")
            if test_time_length != 0:
                _warn_is_ignored("test_time_length")
            for key in predict_options:
                _warn_is_ignored(key)
            if num_test_iters < 0:
                raise ValueError(
                    "argument 'num_test_iters' must be a nonnegative integer"
                )
        if regularizer_factory is None:
            regularizer_factory = _identity

        # Fit the model for the first time.
        states, inputs = self._fit_and_return_training_data(
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )

        # Do the regularization selection.
        if self._iscontinuous:
            return self._fit_regselect_continuous(
                train_time_domains=train_time_domains,
                parameters=None,
                states=states,
                input_functions=input_functions,
                candidates=candidates,
                regularizer_factory=regularizer_factory,
                gridsearch_only=gridsearch_only,
                test_time_length=test_time_length,
                stability_margin=stability_margin,
                verbose=verbose,
                **predict_options,
            )
        return self._fit_regselect_discrete(
            parameters=None,
            states=states,
            inputs=inputs,
            candidates=candidates,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            num_test_iters=num_test_iters,
            stability_margin=stability_margin,
            verbose=verbose,
        )

    def predict(self, state0, *args, **kwargs):
        """Evaluate the reduced-order model.

        Arguments are the same as the ``predict()`` method of :attr:`model`.

        Parameters
        ----------
        state0 : (n,) ndarray
            Initial state, expressed in the original state space.
        *args : list
            Other positional arguments to the ``predict()`` method of
            :attr:`model`.
        **kwargs : dict
            Keyword arguments to the ``predict()`` method of :attr:`model`.

        Returns
        -------
        states: (n, k) ndarray
            Solution to the model, expressed in the original state space.
        """
        q0_ = self.encode(state0, fit_transformer=False, fit_basis=False)
        states = self.model.predict(q0_, *args, **kwargs)
        return self.decode(states)
