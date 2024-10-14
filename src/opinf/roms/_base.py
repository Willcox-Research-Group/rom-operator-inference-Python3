# roms/_base.py
"""Base for ROM classes."""

__all__ = []

import abc
import warnings
import numpy as np

from .. import errors, post, utils
from .. import lift, pre, basis as _basis, ddt
from ..models import _utils as modutils


class _BaseROM(abc.ABC):
    """Reduced-order model.

    This class connects classes from the various submodules to form a complete
    reduced-order modeling workflow.

    High-dimensional data
        -> transformed / preprocessed data
        -> compressed data
        -> low-dimensional model.

    Parameters
    ----------
    model : :mod:`opinf.models` object
        System model.
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

    def __init__(self, model, lifter, transformer, basis, ddt_estimator):
        """Store attributes. Child classes should verify model type."""
        self.__model = model

        # Verify lifter.
        if not (lifter is None or isinstance(lifter, lift.LifterTemplate)):
            warnings.warn(
                "lifter not derived from LifterTemplate, "
                "unexpected behavior may occur",
                errors.OpInfWarning,
            )
        self.__lifter = lifter

        # Verify transformer.
        if not (
            transformer is None
            or isinstance(
                transformer,
                (pre.TransformerTemplate, pre.TransformerMulti),
            )
        ):
            warnings.warn(
                "transformer not derived from TransformerTemplate "
                "or TransformerMulti, unexpected behavior may occur",
                errors.OpInfWarning,
            )
        self.__transformer = transformer

        # Verify basis.
        if not (
            basis is None
            or isinstance(basis, (_basis.BasisTemplate, _basis.BasisMulti))
        ):
            warnings.warn(
                "basis not derived from BasisTemplate or BasisMulti, "
                "unexpected behavior may occur",
                errors.OpInfWarning,
            )
        self.__basis = basis

        # Verify ddt_estimator.
        if (ddt_estimator is not None) and not self._iscontinuous:
            warnings.warn(
                "ddt_estimator ignored for discrete models",
                errors.OpInfWarning,
            )
            ddt_estimator = None
        if not (
            ddt_estimator is None
            or isinstance(ddt_estimator, ddt.DerivativeEstimatorTemplate)
        ):
            warnings.warn(
                "ddt_estimator not derived from DerivativeEstimatorTemplate, "
                "unexpected behavior may occur",
                errors.OpInfWarning,
            )
        self.__ddter = ddt_estimator

    # Properties --------------------------------------------------------------
    @property
    def lifter(self):
        """Lifting transformation."""
        return self.__lifter

    @property
    def transformer(self):
        """Preprocesser."""
        return self.__transformer

    @property
    def basis(self):
        """Dimensionality reducer."""
        return self.__basis

    @property
    def ddt_estimator(self):
        """Time derivative estimator."""
        return self.__ddter

    @property
    def model(self):
        """System model."""
        return self.__model

    @property
    def _iscontinuous(self):
        """``True`` if the model is time continuous (semi-discrete),
        ``False`` if the model if fully discrete.
        """
        return modutils.is_continuous(self.model)

    # Printing ----------------------------------------------------------------
    def __str__(self):
        """String representation."""
        lines = []
        for label, obj in [
            ("lifter", self.lifter),
            ("transformer", self.transformer),
            ("basis", self.basis),
            ("ddt_estimator", self.ddt_estimator),
            ("model", self.model),
        ]:
            if obj is not None:
                lines.append(f"{label}: {str(obj)}")

        body = "\n  ".join("\n".join(lines).split("\n"))
        return f"{self.__class__.__name__}\n  {body}"

    def __repr__(self):
        """Repr: address + string representatation."""
        return utils.str2repr(self)

    # Mappings between original and latent state spaces -----------------------
    def encode(
        self,
        states,
        lhs=None,
        inplace: bool = False,
        *,
        fit_transformer: bool = False,
        fit_basis: bool = False,
    ):
        """Map high-dimensional data to its low-dimensional representation.

        Parameters
        ----------
        states : (n,) or (n, k) ndarray
            State snapshots in the original state space.
        lhs : (n,) or (n, k) ndarray or None
            Left-hand side regression data.

            - If the model is time continuous, these are the time derivatives
              of the state snapshots.
            - If the model is fully discrete, these are the "next states"
              corresponding to the state snapshots.
        inplace : bool
            If ``True``, modify the ``states`` and ``lhs`` in-place in the
            preprocessing transformation (if applicable).

        Returns
        -------
        states_encoded : (r,) or (r, k) ndarray
            Low-dimensional representation of ``states``
            in the latent reduced state space.
        lhs_encoded : (r,) or (r, k) ndarray
            Low-dimensional representation of ``lhs``
            in the latent reduced state space.
            **Only returned** if ``lhs`` is not ``None``.
        """
        # Lifting.
        if self.lifter is not None:
            if lhs is not None:
                if self._iscontinuous:
                    lhs = self.lifter.lift_ddts(states, lhs)
                else:
                    lhs = self.lifter.lift(lhs)
            states = self.lifter.lift(states)

        # Preprocessing.
        if self.transformer is not None:
            if fit_transformer:
                states = self.transformer.fit_transform(
                    states,
                    inplace=inplace,
                )
            else:
                states = self.transformer.transform(states, inplace=inplace)
            if lhs is not None:
                if self._iscontinuous:
                    lhs = self.transformer.transform_ddts(lhs, inplace=inplace)
                else:
                    lhs = self.transformer.transform(lhs, inplace=inplace)

        # Dimensionality reduction.
        if self.basis is not None:
            if fit_basis:
                self.basis.fit(states)
            states = self.basis.compress(states)
            if lhs is not None:
                lhs = self.basis.compress(lhs)

        if lhs is not None:
            return states, lhs
        return states

    def decode(self, states_encoded, locs=None):
        """Map low-dimensional data to the original state space.

        Parameters
        ----------
        states_encoded : (r, ...) ndarray
            Low-dimensional state or states
            in the latent reduced state space.
        locs : slice or (p,) ndarray of integers or None
            If given, return the decoded state at only the p specified
            locations (indices) described by ``locs``.

        Returns
        -------
        states_decoded : (n, ...) ndarray
            Version of ``states_compressed`` in the original state space.
        """
        inplace = False

        # Reverse dimensionality reduction.
        states = states_encoded
        if self.basis is not None:
            inplace = True
            states = self.basis.decompress(states, locs=locs)

        # Reverse preprocessing.
        if self.transformer is not None:
            states = self.transformer.inverse_transform(
                states,
                inplace=inplace,
                locs=locs,
            )

        # Reverse lifting.
        if self.lifter is not None:
            states = self.lifter.unlift(states)

        return states

    def project(self, states):
        """Project a high-dimensional state vector to the subset of the
        high-dimensional space that can be represented by the basis.

        This is done by

        1. expressing the state in low-dimensional latent coordinates, then
        2. reconstructing the high-dimensional state corresponding to those
           coordinates.

        In other words, ``project(Q)`` is equivalent to ``decode(encode(Q))``.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.

        Returns
        -------
        state_projected : (n, ...) ndarray
            Matrix of `n`-dimensional projected state vectors, or a single
            projected state vector.
        """
        return self.decode(self.encode(states))

    # Training ----------------------------------------------------------------
    def _check_fit_args(self, lhs, inputs):
        """Verify required arguments for :meth:`fit()`."""

        # Make sure lhs is given if required.
        if lhs is None and self._iscontinuous and self.ddt_estimator is None:
            raise ValueError(
                "argument 'lhs' required when model is time-continuous"
                " and ddt_estimator=None"
            )

        # Make sure inputs are passed in correctly when requried.
        if inputs is None and self.model._has_inputs:
            raise ValueError(
                "argument 'inputs' required (model depends on external inputs)"
            )

    @abc.abstractmethod
    def fit(
        self,
        states,
        lhs,
        inputs,
        fit_transformer: bool,
        fit_basis: bool,
    ):
        """Calibrate the model to training data.

        Child classes should overwrite this method to include a call to
        the ``fit()`` method of :attr:`model`.

        Parameters
        ----------
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data for a single trajectory; each column
            ``states[i][:, j]`` is one snapshot.
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
            If ``True``, calibrate the preprocessing transformation
            using the ``states``. If ``False``, assume the transformer is
            already calibrated.
        fit_basis : bool
            If ``True``, calibrate the high-to-low dimensional mapping
            using the ``states``.
            If ``False``, assume the basis is already calibrated.

        Returns
        -------
        states : list of s (r, k_i) ndarrays
            State snapshots in the reduced state space.
        lhs : list of s (r, k_i) ndarrays or None
            Left-hand side regression data in the reduced state space.
        inputs : list of s (m, k_i) ndarrays or None
            Processed input training data.

        Notes
        -----
        This abstract method returns the processed training data, but in child
        classes, ``fit()`` should return ``self``.
        """
        # Lifting.
        if self.lifter is not None:
            if lhs is not None:
                if self._iscontinuous:
                    lhs = [
                        self.lifter.lift_ddts(Q, Z)
                        for Q, Z in zip(states, lhs)
                    ]
                else:
                    lhs = [self.lifter.lift(Z) for Z in lhs]
            states = [self.lifter.lift(Q) for Q in states]

        # Preprocessing.
        if self.transformer is not None:
            if fit_transformer:
                self.transformer.fit(np.hstack(states))
            states = [self.transformer.transform(Q) for Q in states]
            if lhs is not None:
                if self._iscontinuous:
                    lhs = [self.transformer.transform_ddts(Z) for Z in lhs]
                else:
                    lhs = [self.transformer.transform(Z) for Z in lhs]

        # Dimensionality reduction.
        if self.basis is not None:
            if fit_basis:
                self.basis.fit(np.hstack(states))
            states = [self.basis.compress(Q) for Q in states]
            if lhs is not None:
                lhs = [self.basis.compress(Z) for Z in lhs]

        # Time derivative estimation / discrete LHS
        if lhs is None:
            if self._iscontinuous:
                if inputs is None:
                    states, lhs = zip(
                        *[self.ddt_estimator.estimate(Q) for Q in states]
                    )
                else:
                    states, lhs, inputs = zip(
                        *[
                            self.ddt_estimator.estimate(Q, U)
                            for Q, U in zip(states, inputs)
                        ]
                    )
            else:
                lhs = [Q[:, 1:] for Q in states]
                states = [Q[:, :-1] for Q in states]
                if inputs is not None:
                    inputs = [
                        U[..., : Q.shape[1]] for Q, U in zip(states, inputs)
                    ]

        return states, lhs, inputs

    def _fit_regselect_continuous(
        self,
        train_time_domains: np.ndarray,
        parameters: list,
        states: list,
        input_functions: list,
        candidates: list,
        regularizer_factory,
        gridsearch_only: bool,
        test_time_length: float,
        stability_margin: float,
        verbose: bool,
        **predict_options: dict,
    ):
        """Regularization selection algorithm for time-continuous models.

        Parameters
        ----------
        train_time_domains : list of s (k_i,) ndarrays
            Time domain corresponding to the training ``states``.
            If a single array is given, assume all training trajectories
            correspond to the same time domain.
        parameters : list of (s scalars or (p,) 1D ndarrays) or None
            Parameter values corresponding to the states.
            If provided, is is assumed that the :attr:`model` is parametric.
        states : list of s (r, k_i) ndarrays
            State snapshots in the reduced state space.
        input_functions : list of s (callables or None)
            Input functions corresponding to the training data, if the
            :attr:`model` takes external inputs.
        candidates : list of regularization parameters
            Regularization hyperparameters to check. If a single hyperparameter
            is given, use it as the start of an optimization-based search.
        regularizer_factory : callable or None
            Function mapping regularization hyperparameters to the full
            regularizer. Specifically, ``regularizer_factory(candidates[i])``
            will be assigned to ``model.solver.regularizer`` for each ``i``.
        gridsearch_only : bool
            If ``True``, stop after checking all regularization ``candidates``
            and do not follow up with optimization.
        test_time_length : float
            Amount of time after the training regime in which to require model
            stability.
        stability_margin : float,
            Factor by which the reduced states may deviate from the range of
            the training data without being flagged as unstable.
        verbose : bool
            If ``True``, print information during the regularization selection.
        predict_options : dict
            Extra arguments for :meth:`opinf.models.ContinuousModel.predict`.
        """
        shifts = [np.mean(Q, axis=1).reshape((-1, 1)) for Q in states]
        limits = [
            stability_margin * np.abs(Q - qbar).max(axis=1)
            for Q, qbar in zip(states, shifts)
        ]
        limmasks = [lims > max(lims) / 50 for lims in limits]

        def unstable(_Q, ell, size):
            """Return ``True`` if the solution is unstable."""
            if _Q.shape[-1] != size:
                return True
            bounds = np.abs(_Q - shifts[ell]).max(axis=1)
            return np.any(bounds[limmasks[ell]] > limits[ell][limmasks[ell]])

        # Extend the training time domains by the testing time length.
        if test_time_length > 0:
            time_domains = []
            for t_train in train_time_domains:
                dt = np.mean(np.diff(t_train))
                t_test = t_train[-1] + np.linspace(
                    dt,
                    dt + test_time_length,
                    int(test_time_length / dt),
                )
                time_domains.append(np.concatenate(((t_train, t_test))))
        else:
            time_domains = train_time_domains

        if input_functions is None:
            input_functions = [None] * len(states)
        loop_collections = [states, input_functions, time_domains]
        if is_parametric := parameters is not None:
            loop_collections.insert(0, parameters)

        def update_model(reg_params):
            """Reset the regularizer and refit the model operators."""
            self.model.solver.regularizer = regularizer_factory(reg_params)
            self.model.refit()

        def training_error(reg_params):
            """Compute the training error for a single regularization
            candidate.
            """
            update_model(reg_params)

            # Solve the model, check for stability, and compute training error.
            error = 0
            for ell, entries in enumerate(zip(*loop_collections)):
                if is_parametric:
                    params, Q, input_func, t = entries
                    predict_args = (params, Q[:, 0], t, input_func)
                else:
                    Q, input_func, t = entries
                    predict_args = (Q[:, 0], t, input_func)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solution = self.model.predict(
                        *predict_args, **predict_options
                    )
                if unstable(solution, ell, t.size):
                    return np.inf
                trainsize = Q.shape[-1]
                solution_train = solution[:, :trainsize]
                error += post.Lp_error(Q, solution_train, t[:trainsize])[1]
            return error / len(states)

        best_regularization = utils.gridsearch(
            training_error,
            candidates,
            gridsearch_only=gridsearch_only,
            label="regularization",
            verbose=verbose,
        )

        update_model(best_regularization)
        return self

    def _fit_regselect_discrete(
        self,
        parameters: list,
        states: list,
        inputs: list,
        candidates: list,
        regularizer_factory,
        gridsearch_only: bool,
        num_test_iters: int,
        stability_margin: float,
        verbose: bool,
    ):
        """Regularization selection algorithm for time-discrete models.

        Parameters
        ----------
        parameters : list of (s scalars or (p,) 1D ndarrays) or None
            Parameter values corresponding to the states.
            If provided, is is assumed that the :attr:`model` is parametric.
        states : list of s (r, k_i) ndarrays
            State snapshots in the reduced state space. This method assumes
            the snapshots are sequential, i.e., the model maps
            ``states[i][:, j]`` to ``states[i][:, j+1]``.
        input_functions : list of s (m, k_i + num_test_iters) ndarray
            Inputs corresponding to the training data, together with inputs
            for the testing regime. Only required if the :attr:`model` takes
            external inputs.
        candidates : list of regularization parameters
            Regularization hyperparameters to check. If a single hyperparameter
            is given, use it as the start of an optimization-based search.
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
        verbose : bool
            If ``True``, print information during the regularization selection.
        """
        shifts = [np.mean(Q, axis=1).reshape((-1, 1)) for Q in states]
        limits = [
            stability_margin * np.abs(Q - qbar).max(axis=1)
            for Q, qbar in zip(states, shifts)
        ]

        def unstable(_Q, ell, size):
            """Return ``True`` if the solution is unstable."""
            if _Q.shape[-1] != size:
                return True
            return np.any(np.abs(_Q - shifts[ell]).max(axis=1) > limits[ell])

        # Extend the training time domains by the testing time length.
        num_iters = [Q.shape[-1] for Q in states]
        if num_test_iters > 0:
            num_iters = [n + num_test_iters for n in num_iters]

        loop_collections = [states, inputs, num_iters]
        if is_parametric := parameters is not None:
            loop_collections.insert(0, parameters)

        def update_model(reg_params):
            """Reset the regularizer and refit the model operators."""
            self.model.solver.regularizer = regularizer_factory(reg_params)
            self.model.refit()

        def training_error(reg_params):
            """Compute the mean training error for a single regularization
            candidate.
            """
            update_model(reg_params)

            # Solve the model, check for stability, and compute training error.
            error = 0
            for ell, entries in enumerate(loop_collections):
                if is_parametric:
                    params, Q, U, niter = entries
                    predict_args = (params, Q[:, 0], niter, U)
                else:
                    Q, U, niter = entries
                    predict_args = (Q[:, 0], niter, U)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    solution = self.model.predict(*predict_args)
                if unstable(solution, ell, niter):
                    print("UNSTABLE")
                    return np.inf
                error += post.frobenius_error(Q, solution[:, : Q.shape[-1]])[1]
            return error / len(states)

        best_regularization = utils.gridsearch(
            training_error,
            candidates,
            gridsearch_only=gridsearch_only,
            label="regularization",
            verbose=verbose,
        )

        update_model(best_regularization)
        return self

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Evaluate the model."""
        raise NotImplementedError  # pragma: no cover

    # Verification ------------------------------------------------------------
    # TODO
