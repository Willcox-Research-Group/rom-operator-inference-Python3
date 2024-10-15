# roms/_nonparametric.py
"""Nonparametric ROM class."""

__all__ = [
    "ROM",
]

import numpy as np

from ..models import _utils as modutils
from ._base import _BaseROM


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

    # Training ----------------------------------------------------------------
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
            parameters=None,
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )
        return self

    def fit_regselect_continuous(
        self,
        candidates: list,
        train_time_domains: np.ndarray,
        states: list,
        ddts: list = None,
        input_functions: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        test_time_length: float = 0,
        stability_margin: float = 5.0,
        test_cases: list = None,
        verbose: bool = False,
        **predict_options: dict,
    ):
        """Calibrate the time-continuous model to training data, selecting the
        regularization hyperparameter(s) that minimize the training error while
        maintaining stability over the testing regime.

        This method requires the :attr:`model` to be time-continuous and to
        have a ``solver`` of one of the following types:

        * :class:`opinf.lstsq.L2Solver`
        * :class:`opinf.lstsq.L2DecoupledSolver`
        * :class:`opinf.lstsq.TikhonovSolver`
        * :class:`opinf.lstsq.TikhonovDecoupledSolver`

        The ``solver.regularizer`` is repeatedly adjusted, and the model is
        recalibrated, until a best regularization is selected.

        Parameters
        ----------
        candidates : list of regularization hyperparameters
            Regularization hyperparameters to check before carrying out a
            derivative-free optimization.
        train_time_domains : list of s (k_i,) ndarrays
            Time domain corresponding to the training states.
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data corresponding to a different trajectory;
            each column ``states[i][:, j]`` is one snapshot.
        ddts : list of s (n, k_i) ndarrays or None
            Snapshot time derivative data. Each array ``ddts[i]`` are the time
            derivatives of ``states[i]``; each column ``ddts[i][:, j]``
            corresponds to the snapshot ``states[i][:, j]``. If ``None``
            (default), these are estimated using :attr:`ddt_estimator`.
        input_functions : list of s callables or None
            Input functions mapping time to input vectors. Only required if the
            :attr:`model` takes external inputs. Each ``input_functions[i]``
            is the function corresponding to ``states[i]``, and
            ``input_functions[i](train_time_domains[i][j])`` is the input
            vector corresponding to the snapshot ``states[i][:, j]``.
        fit_transformer : bool
            If ``True`` (default), calibrate the preprocessing transformation
            using the ``states``.
            If ``False``, assume the transformer is already calibrated.
        fit_basis : bool
            If ``True`` (default), calibrate the high-to-low dimensional
            mapping using the ``states``.
            If ``False``, assume the basis is already calibrated.
        regularizer_factory : callable or None
            Function mapping regularization hyperparameters to the full
            regularizer. Specifically, ``regularizer_factory(candidates[i])``
            will be assigned to ``model.solver.regularizer`` for each ``i``.
            If ``None`` (default), set ``regularizer_factory()`` to the
            identity function.
        gridsearch_only : bool
            If ``True``, stop after checking all regularization ``candidates``
            and do not follow up with optimization.
        test_time_length : float or None
            Amount of time after the training regime in which to require model
            stability.
        stability_margin : float
            Factor by which the predicted reduced states may deviate from the
            range of the training reduced states without the trajectory being
            classified as unstable.
        test_cases : list of ContinuousRegTest objects
            Additional test cases for which the model is required to be stable.
            See :class:`opinf.utils.ContinuousRegTest`.
        verbose : bool
            If ``True``, print information during the regularization selection.
        predict_options : dict or None
            Extra arguments for :meth:`opinf.models.ContinuousModel.predict`.

        Notes
        -----
        If there is only one trajectory of training data (s = 1), ``states``
        may be provided as an (n, k) ndarray. In this case, it is assumed that
        ``ddts`` (if provided) is an (n, k) ndarray and that ``inputs`` (if
        provided) is a single callable.

        The ``train_time_domains`` may be a single one-dimensional array, in
        which case it is assumed that each trajectory ``states[i]`` corresponds
        to the same time domain.
        """
        super().fit_regselect_continuous(
            candidates=candidates,
            train_time_domains=train_time_domains,
            parameters=None,
            states=states,
            ddts=ddts,
            input_functions=input_functions,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            test_time_length=test_time_length,
            stability_margin=stability_margin,
            test_cases=test_cases,
            verbose=verbose,
            **predict_options,
        )

    def fit_regselect_discrete(
        self,
        candidates: list,
        states: list,
        inputs: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        num_test_iters: int = 0,
        stability_margin: float = 5.0,
        test_cases: list = None,
        verbose: bool = False,
    ):
        """Calibrate the fully discrete model to training data, selecting the
        regularization hyperparameter(s) that minimize the training error
        while maintaining stability over the testing regime.

        This method requires the :attr:`model` to be time-continuous and to
        have a ``solver`` of one of the following types:

        * :class:`opinf.lstsq.L2Solver`
        * :class:`opinf.lstsq.L2DecoupledSolver`
        * :class:`opinf.lstsq.TikhonovSolver`
        * :class:`opinf.lstsq.TikhonovDecoupledSolver`

        The ``solver.regularizer`` is repeatedly adjusted, and the model is
        recalibrated, until a best regularization is selected.

        candidates : list of regularization hyperparameters
            Regularization hyperparameters to check. If a single hyperparameter
            is given, use it as the start of an optimization-based search.
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data corresponding to a different trajectory;
            each column ``states[i][:, j]`` is one snapshot. This method
            assumes the snapshots are sequential, i.e., the model maps
            ``states[i][:, j]`` to ``states[i][:, j+1]``.
        states : list of s (r, k_i) ndarrays
            State snapshots in the reduced state space. This method assumes
            the snapshots are sequential, i.e., the model maps
            ``states[i][:, j]`` to ``states[i][:, j+1]``.
        inputs : list of s (m, k_i + num_test_iters) ndarrays
            Inputs corresponding to the training data, together with inputs
            for the testing regime. Only required if the :attr:`model` takes
            external inputs.
        regularizer_factory : callable or None
            Function mapping regularization hyperparameters to the full
            regularizer. Specifically, ``regularizer_factory(candidates[i])``
            will be assigned to ``model.solver.regularizer`` for each ``i``.
        gridsearch_only : bool
            If ``True``, stop after checking all regularization ``candidates``
            and do not follow up with optimization.
        num_test_iters : int
            Number of iterations after the training data in which to require
            model stability.
        stability_margin : float,
            Factor by which the reduced states may deviate from the range of
            the training data without being flagged as unstable.
        test_cases : list of DiscreteRegTest objects
            Additional test cases for which the model is required to be stable.
            See :class:`opinf.utils.DiscreteRegTest`.
        verbose : bool
            If ``True``, print information during the regularization selection.

        Notes
        -----
        If there is only one trajectory of training data (s = 1), ``states``
        may be provided as an (n, k) ndarray. In this case, it is assumed that
        ``inputs`` (if provided) is a single (m, k) ndarray.
        """
        return super().fit_regselect_discrete(
            candidates=candidates,
            parameters=None,
            states=states,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            num_test_iters=num_test_iters,
            stability_margin=stability_margin,
            test_cases=test_cases,
            verbose=verbose,
        )

    # Evaluation --------------------------------------------------------------
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
