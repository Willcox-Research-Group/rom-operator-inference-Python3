# models/mono/_parametric.py
"""Parametric monolithic dynamical systems models."""

__all__ = [
    "ParametricSteadyModel",
    "ParametricDiscreteModel",
    "ParametricContinuousModel",
]

import copy
import warnings
import numpy as np
import scipy.interpolate as spinterp

from ._base import _MonolithicModel
from ._nonparametric import (
    _FrozenSteadyModel,
    _FrozenDiscreteModel,
    _FrozenContinuousModel,
)
from ... import errors, utils
from ... import operators_new as _operators


# Base class ==================================================================
class _ParametricMonolithicModel(_MonolithicModel):
    r"""Base class for parametric monolithic models.

    Parent class: :class:`opinf.mono._base._MonolithicModel`

    Child classes:

    * ``_InterpolatedMonolithicModel``
    * :class:`opinf.models.ParametricSteadyModel`
    * :class:`opinf.models.ParametricDiscreteModel`
    * :class:`opinf.models.ParametricContinuousModel`
    """

    _ModelClass = NotImplemented  # Must be specified by child classes.

    # ModelClass properties ---------------------------------------------------
    @property
    def _LHS_ARGNAME(self):  # pragma: no cover
        """Name of LHS argument in fit(), e.g., "ddts"."""
        return self._ModelClass._LHS_ARGNAME

    @property
    def _LHS_LABEL(self):  # pragma: no cover
        """String representation of LHS, e.g., "dq / dt"."""
        return self._ModelClass._LHS_LABEL

    @property
    def _STATE_LABEL(self):  # pragma: no cover
        """String representation of state, e.g., "q(t)"."""
        return self._ModelClass._STATE_LABEL

    @property
    def _INPUT_LABEL(self):  # pragma: no cover
        """String representation of input, e.g., "u(t)"."""
        return self._ModelClass._INPUT_LABEL

    @property
    def ModelClass(self):
        """Nonparametric model class that represents this parametric model
        when evaluated at a particular parameter value, a subclass of
        :class:`opinf.mono._base._MonolithicModel`.

        Examples
        --------
        >>> model = MyParametricModel(init_args).fit(fit_args)
        >>> model_evaluated = model.evaluate(parameter_value)
        >>> type(model_evaluated) is MyParametricModel.ModelClass
        True
        """
        return self._ModelClass

    # Properties: operators ---------------------------------------------------
    _operator_abbreviations = dict()

    def _isvalidoperator(self, op):
        """All monolithic operators are allowed."""
        return isinstance(
            op,
            (
                _operators._base._NonparametricOperator,
                _operators._base._ParametricOperator,
            ),
        )

    @staticmethod
    def _check_operator_types_unique(ops):
        """Raise a ValueError if any two operators represent the same kind
        of operation (e.g., two constant operators).
        """
        OpClasses = {
            (op.OperatorClass if _operators.is_parametric(op) else type(op))
            for op in ops
        }
        if len(OpClasses) != len(ops):
            raise ValueError("duplicate type in list of operators to infer")

    def _get_operator_of_type(self, OpClass):
        """Return the first operator in the model corresponding to the
        operator class ``OpClass``.
        """
        for op in self.operators:
            if (
                _operators.is_parametric(op) and op.OperatorClass is OpClass
            ) or (_operators.is_nonparametric(op) and isinstance(op, OpClass)):
                return op

    @property
    def operators(self):
        """Operators comprising the terms of the model."""
        return _MonolithicModel.operators.fget(self)

    @operators.setter
    def operators(self, ops):
        """Set the operators."""
        _MonolithicModel.operators.fset(self, ops)

        # Check at least one operator is parametric.
        parametric_operators = [
            op for op in self.operators if _operators.is_parametric(op)
        ]
        if len(parametric_operators) == 0:
            warnings.warn(
                "no parametric operators detected, "
                "consider using a nonparametric model class",
                UserWarning,
            )

        # Check that not every operator is interpolated.
        if not isinstance(self, _InterpolatedMonolithicModel):
            interpolated_operators = [
                op for op in self.operators if _operators.is_interpolated(op)
            ]
            if len(interpolated_operators) == len(self.operators):
                warnings.warn(
                    "all operators interpolatory, "
                    "consider using an InterpolatedModel class",
                    UserWarning,
                )
        self.__p = self._check_parameter_dimension_consistency(self.operators)

    def _clear(self):
        """Reset the entries of the non-intrusive operators and the
        state, input, and parameter dimensions.
        """
        _MonolithicModel._clear(self)
        self.__p = self._check_parameter_dimension_consistency(self.operators)

    # Properties: dimensions --------------------------------------------------
    @staticmethod
    def _check_parameter_dimension_consistency(ops):
        """Ensure all operators have the same parameter dimension."""
        ps = {
            op.parameter_dimension
            for op in ops
            if _operators.is_parametric(op)
            and op.parameter_dimension is not None
        }
        if len(ps) > 1:
            raise errors.DimensionalityError(
                "operators not aligned "
                "(parameter_dimension must be the same for all operators)"
            )
        return ps.pop() if len(ps) == 1 else None

    @property
    def parameter_dimension(self):
        """Dimension :math:`p` of the parameters."""
        return self.__p

    @parameter_dimension.setter
    def parameter_dimension(self, p):
        """Set the parameter dimension. Not allowed if any
        existing operators have ``parameter_dimension != p``.
        """
        if self.operators is not None:
            for op in self.operators:
                if _operators.is_nonparametric(op):
                    continue
                if (opp := op.parameter_dimension) is not None and opp != p:
                    raise AttributeError(
                        "can't set attribute "
                        f"(existing operators have p = {self.__p})"
                    )
        self.__p = p

    def _set_parameter_dimension_from_data(self, parameters):
        """Extract and save the dimension of the parameter space from a set of
        parameter values.

        Parameters
        ----------
        parameters : (s, p) or (p,) ndarray
            Parameter value(s).
        """
        if (dim := len(shape := np.shape(parameters))) == 1:
            self.parameter_dimension = 1
        elif dim == 2:
            self.parameter_dimension = shape[1]
        else:
            raise ValueError("parameter values must be scalars or 1D arrays")

    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(
        self,
        parameters,
        states,
        lhs,
        inputs,
        solver=None,
    ):
        """Prepare training data for Operator Inference by extracting
        dimensions, validating data sizes, and modifying the left-hand side
        data if there are any known operators.
        """
        # Clear non-intrusive operator data.
        self._clear()

        # Fully intrusive case, no least-squares problem to solve.
        if len(self._indices_of_operators_to_infer) == 0:
            warnings.warn(
                "all operators initialized intrusively, nothing to learn",
                errors.UsageWarning,
            )
            return None, None, None, None, None

        # Validate / process solver.
        solver = self._check_solver(solver)

        # Process parameters.
        parameters = np.array(parameters)
        self._set_parameter_dimension_from_data(parameters)
        n_datasets = len(parameters)

        def _check_valid_dimension0(dataset, label):
            """Dimension 0 must be s (number of training parameters)."""
            if (datalen := len(dataset)) != n_datasets:
                raise errors.DimensionalityError(
                    f"len({label}) = {datalen} "
                    f"!= {n_datasets} = len(parameters)"
                )

        def _check_valid_dimension1(dataset, label):
            """Dimension 1 must be r (state dimensions)."""
            for i, subset in enumerate(dataset):
                if (dim := len(subset)) != (r := self.state_dimension):
                    raise errors.DimensionalityError(
                        f"len({label}[{i}]) = {dim} != {r} = r"
                    )

        # Process states, extract model dimension if needed.
        states = [np.atleast_2d(Q) for Q in states]
        if self.state_dimension is None:
            self.state_dimension = states[0].shape[0]
        _check_valid_dimension0(states, "states")
        _check_valid_dimension1(states, "states")

        def _check_valid_dimension2(dataset, label):
            """Dimension 2 must match across datasets (number of snapshots)."""
            for i, subset in enumerate(dataset):
                if (dim := subset.shape[-1]) != (k := states[i].shape[-1]):
                    raise errors.DimensionalityError(
                        f"{label}[{i}].shape[-1] = {dim} "
                        f"!= {k} = states[{i}].shape[-1]"
                    )

        # Process LHS.
        lhs = [np.atleast_2d(L) for L in lhs]
        _check_valid_dimension0(lhs, self._LHS_ARGNAME)
        _check_valid_dimension1(lhs, self._LHS_ARGNAME)
        _check_valid_dimension2(lhs, self._LHS_ARGNAME)

        # Process inputs, extract input dimension if needed.
        self._check_inputargs(inputs, "inputs")
        if self._has_inputs:
            inputs = [np.atleast_2d(U) for U in inputs]
            if not self.input_dimension:
                self.input_dimension = inputs[0].shape[0]
            _check_valid_dimension0(lhs, self._LHS_ARGNAME)
            for i, subset in enumerate(inputs):
                if (dim := subset.shape[0]) != (m := self.input_dimension):
                    raise errors.DimensionalityError(
                        f"inputs[{i}].shape[0] = {dim} != {m} = m"
                    )
            _check_valid_dimension2(inputs, "inputs")
        elif inputs is None:
            inputs = [None] * n_datasets

        # Subtract known operator evaluations from the LHS.
        for ell in self._indices_of_known_operators:
            for i, lhsi in enumerate(lhs):
                lhs[i] = lhsi - self.operators[ell].apply(
                    parameters[i], states[i], inputs[i]
                )

        return parameters, states, lhs, inputs, solver

    def _assemble_data_matrix(self, states, inputs):
        """Assemble the data matrix for operator inference."""
        raise NotImplementedError("future release")

    def _fit_solver(self, parameters, states, lhs, inputs=None, solver=None):
        """Construct a solver for the operator inference least-squares
        regression."""
        (
            parameters_,
            states_,
            lhs_,
            inputs_,
            solver_,
        ) = self._process_fit_arguments(
            parameters, states, lhs, inputs, solver=solver
        )

        # Fully intrusive case (nothing to learn).
        if states_ is lhs_ is None:
            self.solver_ = None
            return

        # Set up non-intrusive learning.
        D = self._assemble_data_matrix(states_, inputs_)
        self.solver_ = solver_.fit(D, np.hstack(lhs_).T)

    def _extract_operators(self, Ohat):
        """Unpack the operator matrix and populate operator entries."""
        raise NotImplementedError("future release")

    def _evaluate_solver(self):
        """Evaluate the least-squares solver and process the results."""
        # Fully intrusive case (nothing to learn).
        if self.solver_ is None:
            return

        # Execute non-intrusive learning.
        OhatT = self.solver_.predict()
        self._extract_operators(np.atleast_2d(OhatT.T))

    def fit(self, parameters, states, lhs, inputs=None, solver=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ophat}
           \sum_{i=1}^{s}\sum_{j=0}^{k_{i}-1}\left\|
           \Ophat(\qhat_{i,j}, \u_{i,j}; \bfmu_i) - \dot{\qhat}_{i,j}
           \right\|_2^2

        where
        :math:`\zhat = \Ophat(\qhat, \u)` is the model and

        * :math:`\qhat_j\in\RR^r` is a measurement of the state,
        * :math:`\u_j\in\RR^m` is a measurement of the input, and
        * :math:`\zhat_j\in\RR^r` is a measurement of the left-hand side
          of the model.

        The *operator matrix* :math:`\Ohat\in\RR^{r\times d(r,m)}` is such that
        :math:`\Ophat(\q,\u) = \Ohat\d(\qhat,\u)` for some data vector
        :math:`\d(\qhat,\u)\in\RR^{d(r,m)}`; the *data matrix*
        :math:`\D\in\RR^{k\times d(r,m)}` is given by
        :math:`[~\d(\qhat_0,\u_0)~~\cdots~~\d(\qhat_{k-1},\u_{k-1})~]\trp`.
        Finally,
        :math:`\Zhat = [~\zhat_0~~\cdots~~\zhat_{k-1}~]\in\RR^{r\times k}`.
        See the :mod:`opinf.operators_new` module for more explanation.

        The strategy for solving the regression, as well as any additional
        regularization or constraints, are specified by the ``solver``.

        Parameters
        ----------
        parameters : list of s scalars or (p,) 1D ndarrays
            Parameter values for which training data are available.
        states : list of s (r, k) ndarrays
            Snapshot training data. Each array ``states[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``states[i][:, j]`` is a single snapshot.
        ddts : list of s (r, k) ndarrays
            Snapshot time derivative data. Each array ``ddts[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``ddts[i][:, j]`` corresponds to the snapshot ``states[i, :, j]``.
        inputs : list of s (m, k) or (k,) ndarrays, or None
            Input training data. Each array ``inputs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``inputs[i][:, j]`` corresponds to the snapshot ``states[:, j]``.
            May be a two-dimensional array if `m=1` (scalar input).
        solver : :mod:`opinf.lstsq` object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: :class:`opinf.lstsq.PlainSolver`,
              SVD-based solve without regularization.
            * **float > 0:** :class:`opinf.lstsq.L2Solver`,
              SVD-based solve with scalar Tikhonov regularization.

        Returns
        -------
        self
        """
        self._fit_solver(parameters, states, lhs, inputs, solver=solver)
        self._evaluate_solver()

    # Parametric evaluation ---------------------------------------------------
    def evaluate(self, parameter):
        """Construct the nonparametric model for the given parameter value.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value at which to evaluate the model.

        Returns
        -------
        model : _NonparametricMonolithicModel
            Nonparametric model of type ``ModelClass``.
        """
        return self.ModelClass(
            [op.evaluate(parameter) for op in self.operators]
        )

    def rhs(self, parameter, state, input_=None):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the function :math:`\widehat{\mathbf{F}}(\qhat, \u)`
        where the model can be written as one of the following:

        * :math:`\ddt\qhat(t; \bfmu) = \Ophat(\qhat(t), \u(t); \bfmu)`
          (continuous time)
        * :math:`\qhat_{j+1}(\bfmu) = \Ophat(\qhat_j, \u_j; \bfmu)`
          (discrete time)
        * :math:`\widehat{\mathbf{g}}(\bfmu) = \Ophat(\qhat, \u; \bfmu)`
          (steady state)

        Parameters
        ----------
        parameter : (p,) ndarray
            Parameter value :math:`\bfmu`.
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math:`\u`.

        Returns
        -------
        evaluation : (r,) ndarray
            Evaluation of the right-hand side of the model.

        Notes
        -----
        For repeated ``rhs()`` calls with the same parameter value, use
        :meth:`evaluate` to first get the nonparametric model corresponding
        to the parameter value.

           # Instead of this...
           >>> values = [parametric_model.rhs(parameter, q, input_)
           ...           for q in list_of_states]
           # ...it is faster to do this.
           >>> model_at_parameter = parametric_model.evaluate(parameter)
           >>> values = [model_at_parameter.rhs(q, input_)
           ...           for q in list_of_states]
        """
        return self.evaluate(parameter).rhs(state, input_)  # pragma: no cover

    def jacobian(self, parameter, state, input_):
        r"""Construct and sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\ddqhat\Ophat(\qhat, \u)`
        where the model can be written as one of the following:

        * :math:`\ddt\qhat(t) = \Ophat(\qhat(t), \u(t))` (continuous time)
        * :math:`\qhat_{j+1} = \Ophat(\qhat_{j}, \u_{j})` (discrete time)
        * :math:`\widehat{\mathbf{g}} = \Ophat(\qhat, \u)` (steady state)

        Parameters
        ----------
        parameter : (p,) ndarray
            Parameter value :math:`\bfmu`.
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math:`\u`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.

        Notes
        -----
        For repeated ``rhs()`` calls with the same parameter value, use
        :meth:`evaluate` to first get the nonparametric model corresponding
        to the parameter value.

           # Instead of this...
           >>> values = [parametric_model.rhs(parameter, q, input_)
           ...           for q in list_of_states]
           # ...it is faster to do this.
           >>> model_at_parameter = parametric_model.evaluate(parameter)
           >>> values = [model_at_parameter.rhs(q, input_)
           ...           for q in list_of_states]

        """
        return self.evaluate(parameter).jacobian(
            state, input_
        )  # pragma: no cover

    def predict(self, parameter, *args, **kwargs):
        """Solve the model at the given parameter value."""
        return self.evaluate(parameter).predict(*args, **kwargs)


# Public classes ==============================================================
class ParametricSteadyModel(_ParametricMonolithicModel):
    """Parametric steady models."""

    _ModelClass = _FrozenSteadyModel


class ParametricDiscreteModel(_ParametricMonolithicModel):
    """Parametric time-discrete models."""

    _ModelClass = _FrozenDiscreteModel


class ParametricContinuousModel(_ParametricMonolithicModel):
    """Parametric continuous models."""

    _ModelClass = _FrozenContinuousModel


# Special case: fully interpolation-based models ==============================
class _InterpolatedMonolithicModel(_ParametricMonolithicModel):
    """Base class for parametric monolithic models where all operators MUST be
    interpolation-based parametric operators. In this special case, the
    inference problems completely decouple by training parameter.

    Parameters
    ----------
    operators : list of :mod:`opinf.operators` objects
        Operators comprising the terms of the model.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from ``scipy.interpolate``.
        If ``None`` (default), use ``scipy.interpolate.CubicSpline``
        for one-dimensional parameters and
        ``scipy.interpolate.LinearNDInterpolator`` otherwise.
    """

    @property
    def _ModelFitClass(self):
        """Parent of ModelClass that has a callable ``fit()`` method."""
        return self.ModelClass.__bases__[-1]

    def __init__(self, operators, InterpolatorClass=None):
        """Define the model structure and set the interpolator class."""
        _ParametricMonolithicModel.__init__(self, operators)
        self.set_interpolator(InterpolatorClass)

    @classmethod
    def _from_models(cls, parameters, models, InterpolatorClass: type = None):
        """Interpolate a collection of non-parametric models.

        Parameters
        ----------
        parameters : list of s scalars or (p,) 1D ndarrays
            Parameter values for which the operator entries are known.
        models : list of s :mod:`opinf.models` objects
            Nonparametric models with fully populated operator entries.
        InterpolatorClass : type or None
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from ``scipy.interpolate``.
            If ``None`` (default), use ``scipy.interpolate.CubicSpline``
            for one-dimensional parameters and
            ``scipy.interpolate.LinearNDInterpolator`` otherwise.
        """
        # Check for consistency in the models.
        opclasses = [type(op) for op in models[0].operators]
        ModelFitClass = cls._ModelClass.__bases__[-1]
        for mdl in models:
            # Model class.
            if not isinstance(mdl, ModelFitClass):
                raise TypeError(
                    "expected roms of type " f"{ModelFitClass.__name__}"
                )
            # Operator count and type.
            if len(mdl.operators) != len(opclasses):
                raise ValueError(
                    "models not aligned (inconsistent number of operators)"
                )
            for ell, op in enumerate(mdl.operators):
                if not isinstance(op, opclasses[ell]):
                    raise ValueError(
                        "models not aligned (inconsistent operator types)"
                    )
            # Entries are set.
            mdl._check_is_trained()

        # Extract the operators from the individual models.
        return cls(
            operators=[
                _operators._nonparametric_to_interpolated(
                    OpClass
                )._from_operators(
                    training_parameters=parameters,
                    operators=[mdl.operators[ell] for mdl in models],
                    InterpolatorClass=InterpolatorClass,
                )
                for ell, OpClass in enumerate(opclasses)
            ],
            InterpolatorClass=InterpolatorClass,
        )

    def set_interpolator(self, InterpolatorClass):
        """Set the interpolator for the operator entries.

        Parameters
        ----------
        InterpolatorClass : type or None
            Class for the elementwise interpolation. Must obey the syntax

                >>> interpolator = InterpolatorClass(data_points, data_values)
                >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from ``scipy.interpolate``.
            If ``None`` (default), use ``scipy.interpolate.CubicSpline``
            for one-dimensional parameters and
            ``scipy.interpolate.LinearNDInterpolator`` otherwise.
        """
        for op in self.operators:
            op.set_interpolator(InterpolatorClass)
        self.__InterpolatorClass = InterpolatorClass

    # Properties: operators ---------------------------------------------------
    _operator_abbreviations = {
        "c": _operators.InterpolatedConstantOperator,
        "A": _operators.InterpolatedLinearOperator,
        "H": _operators.InterpolatedQuadraticOperator,
        "G": _operators.InterpolatedCubicOperator,
        "B": _operators.InterpolatedInputOperator,
        "N": _operators.InterpolatedStateInputOperator,
    }

    def _isvalidoperator(self, op):
        """Only interpolated parametric operators are allowed."""
        return type(op) in (
            _operators.InterpolatedConstantOperator,
            _operators.InterpolatedLinearOperator,
            _operators.InterpolatedQuadraticOperator,
            _operators.InterpolatedCubicOperator,
            _operators.InterpolatedInputOperator,
            _operators.InterpolatedStateInputOperator,
        )

    # Fitting -----------------------------------------------------------------
    def _assemble_data_matrix(self, *args, **kwargs):  # pragma: no cover
        """Assemble the data matrix for operator inference."""
        raise NotImplementedError(
            "_assemble_data_matrix() not used by this class"
        )

    def _extract_operators(self, *args, **kwargs):  # pragma: no cover
        """Unpack the operator matrix and populate operator entries."""
        raise NotImplementedError(
            "_extract_operators() not used by this class"
        )

    def _fit_solver(self, parameters, states, lhs, inputs=None, solver=None):
        """Construct a solver for the operator inference least-squares
        regression.
        """
        (
            parameters_,
            states_,
            lhs_,
            inputs_,
            solver_,
        ) = self._process_fit_arguments(
            parameters, states, lhs, inputs, solver=solver
        )
        n_datasets = len(parameters)

        # Distribute training data to individual OpInf problems.
        nonparametric_models = []
        for i in range(n_datasets):
            model_i = self._ModelFitClass(
                operators=[
                    op.OperatorClass(
                        op.entries[i] if op.entries is not None else None
                    )
                    for op in self.operators
                ]
            )
            model_i._fit_solver(
                states_[i],
                lhs_[i],
                inputs_[i],
                copy.deepcopy(solver_),
            )
            nonparametric_models.append(model_i)

        self.solvers_ = [mdl.solver_ for mdl in nonparametric_models]
        self.__submodels = nonparametric_models
        self.__training_parameters = parameters_

    def _evaluate_solver(self):
        """Evaluate the least-squares solver and process the results."""
        # Solve each independent subproblem.
        # TODO: parallelize?
        for model_i in self.__submodels:
            model_i.evaluate_solver()

        # Interpolate the resulting operators.
        for op, ell in enumerate(self.operators):
            op._clear()
            op.set_training_parameters(self.__training_parameters)
            op.set_entries([mdl.operators[ell] for mdl in self.__submodels])

        self.__InterpolatorClass = type(self.operators[0].interpolator)

        return self

    # Model persistence -------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Serialize the model, saving it in HDF5 format.
        The model can be recovered with the :meth:`load()` class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension ``.h5`` (HDF5).
        overwrite : bool
            If ``True`` and the specified ``savefile`` already exists,
            overwrite the file. If ``False`` (default) and the specified
            ``savefile`` already exists, raise an error.
        """
        with utils.hdf5_savehandle(savefile, overwrite=overwrite) as hf:
            # Metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_operators"] = len(self.operators)
            meta.attrs["r"] = (
                int(self.state_dimension) if self.state_dimension else 0
            )
            meta.attrs["m"] = (
                int(self.input_dimension) if self.input_dimension else 0
            )

            # Interpolator class.
            suppress_warnings = False
            InterpolatorClassName = self.__InterpolatorClass.__name__
            meta.attrs["InterpolatorClass"] = InterpolatorClassName
            if InterpolatorClassName != "NoneType" and not hasattr(
                spinterp, InterpolatorClassName
            ):
                warnings.warn(
                    "cannot serialize InterpolatorClass "
                    f"'{InterpolatorClassName}', must pass in the class "
                    "when calling load()"
                )
                suppress_warnings = True

            # Operator data.
            hf.create_dataset(
                "indices_infer", data=self._indices_of_operators_to_infer
            )
            hf.create_dataset(
                "indices_known", data=self._indices_of_known_operators
            )
            with warnings.catch_warnings():
                if suppress_warnings:
                    warnings.simplefilter("ignore", UserWarning)
                for i, op in enumerate(self.operators):
                    op.save(hf.create_group(f"operator_{i}"))

    @classmethod
    def load(cls, loadfile: str, InterpolatorClass: type = None):
        """Load a serialized model from an HDF5 file, created previously from
        the :meth:`save()` method.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            Not required if the saved operator utilizes a class from
            ``scipy.interpolate``.

        Returns
        -------
        op : _Operator
            Initialized operator object.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            # Load metadata.
            num_operators = int(hf["meta"].attrs["num_operators"])
            indices_infer = [int(i) for i in hf["indices_infer"][:]]
            indices_known = [int(i) for i in hf["indices_known"][:]]

            # Get the InterpolatorClass.
            SavedClassName = hf["meta"].attrs["InterpolatorClass"]
            if InterpolatorClass is None:
                # Load from scipy.interpolate.
                if hasattr(spinterp, SavedClassName):
                    InterpolatorClass = getattr(spinterp, SavedClassName)
                elif SavedClassName != "NoneType":
                    raise ValueError(
                        f"unknown InterpolatorClass '{SavedClassName}', "
                        f"call load({loadfile}, {SavedClassName})"
                    )
            else:
                # Warn the user if the InterpolatorClass does not match.
                if SavedClassName != (
                    InterpolatorClassName := InterpolatorClass.__name__
                ):
                    warnings.warn(
                        f"InterpolatorClass={InterpolatorClassName} does not "
                        f"match loadfile InterpolatorClass '{SavedClassName}'",
                        UserWarning,
                    )

            # Load operators.
            ops = []
            for i in range(num_operators):
                gp = hf[f"operator_{i}"]
                OpClassName = gp["meta"].attrs["class"]
                ops.append(
                    getattr(_operators, OpClassName).load(
                        gp, InterpolatorClass
                    )
                )

            # Construct the model.
            model = cls(ops)
            model._indices_of_operators_to_infer = indices_infer
            model._indices_of_known_operators = indices_known
            if r := int(hf["meta"].attrs["r"]):
                model.state_dimension = r
            if (m := int(hf["meta"].attrs["m"])) and model._has_inputs:
                model.input_dimension = m

        return model

    def copy(self):
        """Make a copy of the model."""
        return self.__class__(
            operators=[op.copy() for op in self.operators],
            InterpolatorClass=self.__InterpolatorClass,
        )


class InterpolatedDiscreteModel(_InterpolatedMonolithicModel):
    _ModelClass = _FrozenDiscreteModel

    def fit(self, parameters, states, ddts, inputs=None, solver=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ohat^{(1)},\ldots,\Ohat^{(s)}}
           \sum_{i=1}^{s}\sum_{j=0}^{k_{i}-1}\left\|
           \Ophat(\qhat_{i,j}, \u_{i,j}; \bfmu_i) - \dot{\qhat}_{i,j}
           \right\|_2^2
           = \min_{\Ohat^{(1)},\ldots,\Ohat^{(s)}}
           \sum_{i=1}^{s}\left\|
           \D^{(i)}(\Ohat^{(i)})\trp
           - [~\dot{\qhat}_{i,0}~~\cdots~~\dot{\qhat}_{i,k_{i}-1}~]\trp
           \right\|_F^2

        where
        :math:`\ddt\qhat(t) = \Ophat(\qhat(t), \u(t);
        \bfmu) = \Ohat(\bfmu)\d(\qhat(t), \u(t))` is the model and

        * :math:`\qhat_{i,j}\in\RR^r` is the :math:`j`-th measurement of the
          state corresponding to training parameter value :math:`\bfmu_i`,
        * :math:`\u_{i,j}\in\RR^m` is the :math:`j`-th measurement of the
          input corresponding to training parameter value :math:`\bfmu_i`,
        * :math:`\dot{qhat}_{i,j}\in\RR^r` is a measurement of the time
          derivative of the state corresponding to the state-input pair
          :math:`(\qhat_{i,j},\u_{i,j})`,
        * :math:`\D^{(i)}` is the data matrix for data corresponding to
          training parameter value :math:`\bfmu_i`, given by
          :math:`[~\d(\qhat_{i,0},\u_{i,0})~~\cdots~~
          \d(\qhat_{i,k_i-1},\u_{i,k_i-1})~]\trp`.
        * :math:`\Ohat^{(i)} = \Ohat(\bfmu)` is the operator matrix
          evaluated at training parameter value :math:`\bfmu_i`.

        Because all operators in this model are interpolatory, the
        least-squares problem decouples into :math:`s` individual regressions.
        That is, for :math:`i = 1, \ldots, s`, we solve (independently) the
        regressions

        .. math::
           \min_{\Ohat^{(i)}}\left\|
           \D^{(i)}(\Ohat^{(i)})\trp
           - [~\dot{\qhat}_{i,0}~~\cdots~~\dot{\qhat}_{i,k_{i}-1}~]\trp
           \right\|_F^2

        and define the full operator matrix via elementwise interpolation,

        .. math::
           \Ohat(\bfmu) = \textrm{interpolate}(
           (\bfmu_1, \Ohat^{(i)}), \ldots, (\bfmu_s, \Ohat^{(s)}); \bfmu).

        The strategy for solving the regression, as well as any additional
        regularization or constraints, are specified by the ``solver``.

        Parameters
        ----------
        parameters : list of s scalars or (p,) 1D ndarrays
            Parameter values for which the operator entries are known
            or will be inferred from data.
        states : list of s (r, k) ndarrays
            Snapshot training data. Each array ``states[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``states[i][:, j]`` is a single snapshot.
        ddts : list of s (r, k) ndarrays
            Snapshot time derivative data. Each array ``ddts[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``ddts[i][:, j]`` corresponds to the snapshot ``states[i, :, j]``.
        inputs : list of s (m, k) or (k,) ndarrays, or None
            Input training data. Each array ``inputs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``inputs[i][:, j]`` corresponds to the snapshot ``states[:, j]``.
            May be a two-dimensional array if `m=1` (scalar input).
        solver : :mod:`opinf.lstsq` object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: :class:`opinf.lstsq.PlainSolver`,
              SVD-based solve without regularization.
            * **float > 0:** :class:`opinf.lstsq.L2Solver`,
              SVD-based solve with scalar Tikhonov regularization.

        Returns
        -------
        self
        """
        return _InterpolatedMonolithicModel.fit(
            self, parameters, states, ddts, inputs, solver
        )


class InterpolatedContinuousModel(_InterpolatedMonolithicModel):
    _ModelClass = _FrozenContinuousModel

    def fit(self, parameters, states, ddts, inputs=None, solver=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ohat^{(1)},\ldots,\Ohat^{(s)}}
           \sum_{i=1}^{s}\sum_{j=0}^{k_{i}-1}\left\|
           \Ophat(\qhat_{i,j}, \u_{i,j}; \bfmu_i) - \dot{\qhat}_{i,j}
           \right\|_2^2
           = \min_{\Ohat^{(1)},\ldots,\Ohat^{(s)}}
           \sum_{i=1}^{s}\left\|
           \D^{(i)}(\Ohat^{(i)})\trp
           - [~\dot{\qhat}_{i,0}~~\cdots~~\dot{\qhat}_{i,k_{i}-1}~]\trp
           \right\|_F^2

        where
        :math:`\ddt\qhat(t) = \Ophat(\qhat(t), \u(t);
        \bfmu) = \Ohat(\bfmu)\d(\qhat(t), \u(t))` is the model and

        * :math:`\qhat_{i,j}\in\RR^r` is the :math:`j`-th measurement of the
          state corresponding to training parameter value :math:`\bfmu_i`,
        * :math:`\u_{i,j}\in\RR^m` is the :math:`j`-th measurement of the
          input corresponding to training parameter value :math:`\bfmu_i`,
        * :math:`\dot{qhat}_{i,j}\in\RR^r` is a measurement of the time
          derivative of the state corresponding to the state-input pair
          :math:`(\qhat_{i,j},\u_{i,j})`,
        * :math:`\D^{(i)}` is the data matrix for data corresponding to
          training parameter value :math:`\bfmu_i`, given by
          :math:`[~\d(\qhat_{i,0},\u_{i,0})~~\cdots~~
          \d(\qhat_{i,k_i-1},\u_{i,k_i-1})~]\trp`.
        * :math:`\Ohat^{(i)} = \Ohat(\bfmu)` is the operator matrix
          evaluated at training parameter value :math:`\bfmu_i`.

        Because all operators in this model are interpolatory, the
        least-squares problem decouples into :math:`s` individual regressions.
        That is, for :math:`i = 1, \ldots, s`, we solve (independently) the
        regressions

        .. math::
           \min_{\Ohat^{(i)}}\left\|
           \D^{(i)}(\Ohat^{(i)})\trp
           - [~\dot{\qhat}_{i,0}~~\cdots~~\dot{\qhat}_{i,k_{i}-1}~]\trp
           \right\|_F^2

        and define the full operator matrix via elementwise interpolation,

        .. math::
           \Ohat(\bfmu) = \textrm{interpolate}(
           (\bfmu_1, \Ohat^{(i)}), \ldots, (\bfmu_s, \Ohat^{(s)}); \bfmu).

        The strategy for solving the regression, as well as any additional
        regularization or constraints, are specified by the ``solver``.

        Parameters
        ----------
        parameters : list of s scalars or (p,) 1D ndarrays
            Parameter values for which the operator entries are known
            or will be inferred from data.
        states : list of s (r, k) ndarrays
            Snapshot training data. Each array ``states[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``states[i][:, j]`` is a single snapshot.
        ddts : list of s (r, k) ndarrays
            Snapshot time derivative data. Each array ``ddts[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``ddts[i][:, j]`` corresponds to the snapshot ``states[i][:, j]``.
        inputs : list of s (m, k) or (k,) ndarrays, or None
            Input training data. Each array ``inputs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``inputs[i][:, j]`` corresponds to the snapshot
            ``states[i][:, j]``.
            May be a two-dimensional array if `m=1` (scalar input).
        solver : :mod:`opinf.lstsq` object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: :class:`opinf.lstsq.PlainSolver`,
              SVD-based solve without regularization.
            * **float > 0:** :class:`opinf.lstsq.L2Solver`,
              SVD-based solve with scalar Tikhonov regularization.

        Returns
        -------
        self
        """
        return _InterpolatedMonolithicModel.fit(
            self, parameters, states, ddts, inputs, solver
        )
