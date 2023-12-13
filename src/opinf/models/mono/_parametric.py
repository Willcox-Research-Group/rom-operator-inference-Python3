# models/mono/_parametric.py
"""Parametric monolithic dynamical systems models."""

__all__ = [
    "ParametricSteadyModel",
    "ParametricDiscreteModel",
    "ParametricContinuousModel",
]

import warnings
import numpy as np

from ._base import _MonolithicModel
from ._nonparametric import (
    _FrozenSteadyModel,
    _FrozenDiscreteModel,
    _FrozenContinuousModel,
)
from ... import errors
from ... import operators_new as _operators


# Base class ==================================================================
class _ParametricMonolithicModel(_MonolithicModel):
    r"""Base class for parametric monolithic models.

    Parent class: :class:`_MonolithicModel`

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
        :class:`_MonolithicModel`.

        Examples
        --------
        >>> model = MyParametricModel(init_args).fit(fit_args)
        >>> model_evaluated = model.evaluate(parameter_value).
        >>> type(model_evaluated) is MyParametricModel.ModelClass
        True
        """
        return self._ModelClass

    # Constructor -------------------------------------------------------------
    def __init__(self, modelform):
        """Define the model structure.

        Parameters
        ----------
        operators : list of :mod:`opinf.operators` objects
            Operators comprising the terms of the model.
        """
        # Validate the ModelClass.
        if not issubclass(self.ModelClass, _MonolithicModel):
            raise RuntimeError(
                "invalid ModelClass " f"'{self.ModelClass.__name__}'"
            )

        _MonolithicModel.__init__(self, modelform)

    # Properties: operators ---------------------------------------------------
    _operator_abbreviations = dict()

    def _isvalidoperator(self, op):
        """All monolithic operators are allowed."""
        # TODO: allow any monolithic operator.
        raise NotImplementedError

    @staticmethod
    def _check_operator_types_unique(ops):
        """Raise a ValueError if any two operators represent the same kind
        of operation (e.g., two constant operators).
        """
        if len({op.OperatorClass for op in ops}) != len(ops):
            raise ValueError("duplicate type in list of operators to infer")

    def _get_operator_of_type(self, OpClass):
        """Return the first operator in the model corresponding to the
        operator class ``OpClass``.
        """
        for op in self.operators:
            if op.OperatorClass is OpClass:
                return op

    @property
    def operators(self):
        """Operators comprising the terms of the model."""
        return _MonolithicModel.operators.fget(self)

    @operators.setter
    def operators(self, ops):
        """Set the operators."""
        _MonolithicModel.operators.fset(ops)
        self.__p = self._check_parameter_dimension_consistency(ops)

    def _clear(self):
        """Reset the entries of the non-intrusive operators and the
        state, input, and parameter dimensions.
        """
        _MonolithicModel._clear(self)
        # TODO: raise a warning if none of the operators are parametric.
        self.__p = self._check_parameter_dimension_consistency(self.operators)

    # Properties: dimensions --------------------------------------------------
    @staticmethod
    def _check_parameter_dimension_consistency(ops):
        """Ensure all operators have the same parameter dimension."""
        ps = {
            op.parameter_dimension
            for op in ops
            if op.parameter_dimension is not None
        }
        if len(ps) > 1:
            raise errors.DimensionalityError(
                "operators not aligned "
                "(parameter dimension must be the same for all operators)"
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
        if self.__operators is not None:
            for op in self.operators:
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
        self, parameters, states, lhs, inputs, solver=None
    ):
        """Prepare training data for Operator Inference by extracting
        dimensions, projecting known operators, and validating data sizes.
        """
        # Clear non-intrusive operator data.
        self._clear()

        # Fully intrusive case, no least-squares problem to solve.
        if len(self._indices_of_operators_to_infer) == 0:
            warnings.warn(
                "all operators initialized intrusively, nothing to learn",
                UserWarning,
            )
            return None, None, None, None, None

        # Validate / process solver.
        solver = self._check_solver(solver)

        # Check that the number of training sets is consistent.
        n_datasets = len(parameters)
        for data, label in [
            (states, "states"),
            (lhs, self._LHS_ARGNAME),
            (inputs, "inputs"),
        ]:
            if (datalen := len(data)) != n_datasets:
                raise ValueError(
                    f"len({label}) = {datalen} "
                    f"!= {n_datasets} = len(parameters)"
                )

        # Process parameters.
        parameters = np.array(parameters)
        self._set_parameter_dimension_from_data(parameters)

        def _check_valid_dimension1(dataset, label):
            """Dimension 1 must be r (state dimensions)."""
            if (dim := dataset.shape[1]) != self.state_dimension:
                raise errors.DimensionalityError(
                    f"{label}.shape[1] = {dim} != r = {self.state_dimension}"
                )

        # Process states, extract model dimension if needed.
        states = np.array([np.atleast_2d(Q) for Q in states])
        if self.state_dimension is None:
            self.state_dimension = states[0].shape[0]
        _check_valid_dimension1(states, "states")

        def _check_valid_dimension2(dataset, label):
            """Dimension 2 must be k (number of snapshots)."""
            if (dim := dataset.shape[2]) != (k := states.shape[2]):
                raise errors.DimensionalityError(
                    f"{label}.shape[-1] = {dim} != {k} = states.shape[-1]"
                )

        # Process LHS.
        lhs = np.array([np.atleast_2d(L) for L in lhs])
        _check_valid_dimension1(lhs, self._LHS_ARGNAME)
        _check_valid_dimension2(lhs, self._LHS_ARGNAME)

        # Process inputs, extract input dimension if needed.
        self._check_inputargs(inputs, "inputs")
        if self._has_inputs:
            inputs = np.array([np.atleast_2d(U) for U in inputs])
            if not self.input_dimension:
                self.input_dimension = inputs.shape[1]
            if inputs.shape[1] != self.input_dimension:
                raise errors.DimensionalityError(
                    f"inputs.shape[1] = {inputs.shape[0]} "
                    f"!= {self.input_dimension} = m"
                )
            _check_valid_dimension2(inputs, "inputs")

        # Subtract known operator evaluations from the LHS.
        for ell in self._indices_of_known_operators:
            for i, lhsi in enumerate(lhs):
                lhs[i] = lhsi - self.operators[ell].apply(states[i], inputs[i])

        return parameters, states, lhs, inputs, solver

    def _fit_solver(self, states, lhs, inputs=None, solver=None):
        """Construct a solver object mapping the regularizer to solutions
        of the Operator Inference least-squares problem.
        """
        states_, lhs_, inputs_, solver = self._process_fit_arguments(
            states, lhs, inputs, solver=solver
        )

        # Fully intrusive case (nothing to learn).
        if states_ is lhs_ is None:
            self.solver_ = None
            return

        # Set up non-intrusive learning.
        D = self._assemble_data_matrix(states_, inputs_)
        self.solver_ = solver.fit(D, lhs_.T)

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
           \min_{\Ohat}\sum_{j=0}^{k-1}\left\|
           \fhat(\qhat_j, \u_j) - \zhat_j
           \right\|_2^2
           = \min_{\Ohat}\left\|\D\Ohat\trp - \dot{\Qhat}\trp\right\|_F^2

        where
        :math:`\zhat = \fhat(\qhat, \u)` is the model and

        * :math:`\qhat_j\in\RR^r` is a measurement of the state,
        * :math:`\u_j\in\RR^m` is a measurement of the input, and
        * :math:`\zhat_j\in\RR^r` is a measurement of the left-hand side
          of the model.

        The *operator matrix* :math:`\Ohat\in\RR^{r\times d(r,m)}` is such that
        :math:`\fhat(\q,\u) = \Ohat\d(\qhat,\u)` for some data vector
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
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        lhs : (r, k) ndarray
            Left-hand side training data. Each column ``lhs[:, j]``
            corresponds to the snapshot ``states[:, j]``.
            The interpretation of this argument depends on the setting:
            forcing data for steady-state problems, next iteration for
            discrete-time problems, and time derivatives of the state for
            continuous-time problems.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).
        solver : :mod:`opinf.lstsq` object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: :class:`opinf.lstsq.PlainSolver`, SVD-based solve
              without regularization.
            * float > 0: :class:`opinf.lstsq.L2Solver`, SVD-based solve with
              scalar Tikhonov regularization.

        Returns
        -------
        self
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

        raise NotImplementedError("future release")

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


# Special case: fully interpolation-based models ==============================
class _InterpolatedMonolithicModel(_ParametricMonolithicModel):
    """Base class for parametric monolithic models where all operators MUST be
    interpolation-based parametric operators. In this special case, the
    inference problems completely decouple by training parameter.
    """

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

    # def submodels(self):

    def fit(self, parameters, states, lhs, inputs=None, solver=None):
        r"""Learn the model operators from data.

        The operators are inferred by solving the regression problem

        .. math::
           \min_{\Ohat}\sum_{j=0}^{k-1}\left\|
           \fhat(\qhat_j, \u_j) - \zhat_j
           \right\|_2^2
           = \min_{\Ohat}\left\|\D\Ohat\trp - \dot{\Qhat}\trp\right\|_F^2

        where
        :math:`\zhat = \fhat(\qhat, \u)` is the model and

        * :math:`\qhat_j\in\RR^r` is a measurement of the state,
        * :math:`\u_j\in\RR^m` is a measurement of the input, and
        * :math:`\zhat_j\in\RR^r` is a measurement of the left-hand side
          of the model.

        The *operator matrix* :math:`\Ohat\in\RR^{r\times d(r,m)}` is such that
        :math:`\fhat(\q,\u) = \Ohat\d(\qhat,\u)` for some data vector
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
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        lhs : (r, k) ndarray
            Left-hand side training data. Each column ``lhs[:, j]``
            corresponds to the snapshot ``states[:, j]``.
            The interpretation of this argument depends on the setting:
            forcing data for steady-state problems, next iteration for
            discrete-time problems, and time derivatives of the state for
            continuous-time problems.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).
        solver : :mod:`opinf.lstsq` object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: :class:`opinf.lstsq.PlainSolver`, SVD-based solve
              without regularization.
            * float > 0: :class:`opinf.lstsq.L2Solver`, SVD-based solve with
              scalar Tikhonov regularization.

        Returns
        -------
        self
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

        # Distribute training data to individual OpInf problems.
        num_models = len(parameters)
        for op in self.operators:
            if len(op) != num_models:
                raise ValueError(
                    "Interpolatory models require len(operator) == "
                    "len(parameters) for each operator in the model"
                )
        nonparametric_models = [
            self.ModelClass.__bases__[-1](
                operators=[op.entries[i] for op in self.operators]
            ).fit(states_[i], lhs_[i], inputs_[i], solver_)
            for i in range(num_models)
        ]
        for ell in range(len(self.operators)):
            self.operators[ell].set_entries(
                [mdl.operators[ell] for mdl in nonparametric_models]
            )

        return self


class ParametricSteadyModel(_ParametricMonolithicModel):
    """Parametric steady models."""

    _ModelClass = _FrozenSteadyModel


class ParametricDiscreteModel(_ParametricMonolithicModel):
    """Parametric time-discrete models."""

    _ModelClass = _FrozenDiscreteModel


class ParametricContinuousModel(_ParametricMonolithicModel):
    """Parametric continuous models."""

    _ModelClass = _FrozenContinuousModel
