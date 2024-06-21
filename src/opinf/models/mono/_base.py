# models/mono/_base.py
"""Abstract base class for monolithic dynamical systems models."""

__all__ = []

import abc
import warnings
import numpy as np

from ... import errors, lstsq
from ... import operators as _operators


class _Model(abc.ABC):
    """Base class for all monolithic models.

    Child classes:

    * :class:`opinf.models.mono._nonparametric._NonparametricModel`
    * :class:`opinf.models.mono._parametric._ParametricModel`
    """

    def __init__(self, operators, solver=None):
        """Define the model structure.

        Parameters
        ----------
        operators : list of :mod:`opinf.operators` objects
            Operators comprising the terms of the model.
        solver : :mod:`opinf.lstsq` object
            Solver for the Operator Inference regression.
        """
        self.__r = None
        self.__m = None
        self.__operators = None
        self.__solver = None

        self.operators = operators
        self.solver = solver

    # Properties: operators ---------------------------------------------------
    _operator_abbreviations = dict()  # Abbreviations for model operators.

    @staticmethod
    @abc.abstractmethod
    def _isvalidoperator(op):  # pragma: no cover
        """Return True if and only if ``op`` is a valid operator object
        for this class of model.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _check_operator_types_unique(ops):  # pragma: no cover
        """Raise a ValueError if any two operators represent the same kind
        of operation (e.g., two constant operators).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_operator_of_type(self, OpClass):  # pragma: no cover
        """Return the first operator in the model corresponding to the
        operator class ``OpClass``.
        """
        raise NotImplementedError

    @property
    def operators(self):
        """Operators comprising the terms of the model."""
        return self.__operators

    @operators.setter
    def operators(self, ops):
        """Set the operators."""
        if self._isvalidoperator(ops):
            ops = [ops]
        if len(ops) == 0 or ops is None:
            raise ValueError("at least one operator required")

        toinfer = []  # Operators to infer (no entries yet).
        known = []  # Operators whose entries are set.
        self._has_inputs = False  # Whether any operators use inputs.
        ops = list(ops)
        for i in range(len(ops)):
            op = ops[i]
            if isinstance(op, str):
                if op in (self._operator_abbreviations):
                    op = ops[i] = self._operator_abbreviations[op]()
                else:
                    raise TypeError(
                        f"operator abbreviation '{op}' not recognized"
                    )
            if not self._isvalidoperator(op):
                raise TypeError(
                    f"invalid operator of type '{op.__class__.__name__}'"
                )
            if _operators.is_uncalibrated(op):
                toinfer.append(i)
            else:
                known.append(i)
            if _operators.has_inputs(op):
                self._has_inputs = True
        self._check_operator_types_unique([ops[i] for i in toinfer])

        # Store attributes.
        self.__r = self._check_state_dimension_consistency(ops)
        self.__m = self._check_input_dimension_consistency(ops)
        self.__operators = ops
        self._indices_of_operators_to_infer = toinfer
        self._indices_of_known_operators = known
        self._fully_intrusive = len(toinfer) == 0

    def _clear(self):
        """Reset the entries of the non-intrusive operators and the
        state and input dimensions.
        """
        for i in self._indices_of_operators_to_infer:
            self.operators[i]._clear()
        self.__r = self._check_state_dimension_consistency(self.operators)
        self.__m = self._check_input_dimension_consistency(self.operators)

    def __iter__(self):
        """Iterate through the model operators."""
        for op in self.operators:
            yield op

    @property
    def c_(self):
        """:class:`opinf.operators.ConstantOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.ConstantOperator)

    @property
    def A_(self):
        """:class:`opinf.operators.LinearOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.LinearOperator)

    @property
    def H_(self):
        """:class:`opinf.operators.QuadraticOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.QuadraticOperator)

    @property
    def G_(self):
        """:class:`opinf.operators.CubicOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.CubicOperator)

    @property
    def B_(self):
        """:class:`opinf.operators.InputOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.InputOperator)

    @property
    def N_(self):
        """:class:`opinf.operators.StateInputOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.StateInputOperator)

    # Properties: dimensions --------------------------------------------------
    @staticmethod
    def _check_state_dimension_consistency(ops):
        """Ensure all operators have the same state dimension, except for
        inferrable operators whose entries have not been set.
        """
        rs = {
            op.state_dimension
            for op in ops
            if not _operators.is_uncalibrated(op)
        }
        if len(rs) > 1:
            raise errors.DimensionalityError(
                "operators not aligned "
                "(state dimension must be the same for all operators)"
            )
        return rs.pop() if len(rs) == 1 else None

    @property
    def state_dimension(self):
        """Dimension :math:`r` of the state."""
        return self.__r

    @state_dimension.setter
    def state_dimension(self, r):
        """Set the state dimension.
        Not allowed if any existing operators have ``state_dimension != r``.
        """
        if self.__operators is not None:
            for op in self.operators:
                if op.entries is not None and op.state_dimension != r:
                    raise AttributeError(
                        "can't set attribute "
                        f"(existing operators have r = {self.__r})"
                    )
        self.__r = int(r)

    @staticmethod
    def _check_input_dimension_consistency(ops):
        """Ensure all *input* operators with initialized entries have the same
        ``input dimension``.
        """
        inputops = [op for op in ops if _operators.has_inputs(op)]
        if len(inputops) == 0:
            return 0
        ms = {
            op.input_dimension
            for op in inputops
            if not _operators.is_uncalibrated(op)
        }
        if len(ms) > 1:
            raise errors.DimensionalityError(
                "input operators not aligned "
                "(input dimension must be the same for all input operators)"
            )
        return ms.pop() if len(ms) == 1 else None

    @property
    def input_dimension(self):
        """Dimension :math:`m` of the input (zero if there are no inputs)."""
        return self.__m

    @input_dimension.setter
    def input_dimension(self, m):
        """Set the input dimension.
        Only allowed if an input-using operator is present in the model
        and the ``input_dimension`` of every existing input operator agrees.
        """
        if not self._has_inputs and m != 0:
            raise AttributeError("can't set attribute (no input operators)")
        if self.__operators is not None:
            for op in self.operators:
                if (
                    _operators.has_inputs(op)
                    and not _operators.is_uncalibrated(op)
                    and op.input_dimension != m
                ):
                    raise AttributeError(
                        "can't set attribute "
                        f"(existing input operators have m = {self.__m})"
                    )
        self.__m = int(m)

    # Properties: solver ------------------------------------------------------
    @property
    def solver(self):
        """Solver for the least-squares regression, see :mod:`opinf.lstsq`."""
        return self.__solver

    @solver.setter
    def solver(self, solver):
        """Set the solver, including default options."""
        if self._fully_intrusive:
            if solver is not None:
                warnings.warn(
                    "all operators initialized explicity, setting solver=None",
                    errors.OpInfWarning,
                )
            self.__solver = None
            return

        # Defaults and shortcuts.
        if solver is None:
            # No regularization.
            solver = lstsq.PlainSolver()
        elif np.isscalar(solver):
            if solver == 0:
                # Also no regularization.
                solver = lstsq.PlainSolver()
            elif solver > 0:
                # Scalar Tikhonov (L2) regularization.
                solver = lstsq.L2Solver(solver)
            else:
                raise ValueError("if a scalar, solver must be nonnegative")

        # Light validation: must be instance w/ fit(), solve().
        if isinstance(solver, type):
            raise TypeError("solver must be an instance, not a class")
        for mtd in "fit", "solve":
            if not hasattr(solver, mtd) or not callable(getattr(solver, mtd)):
                warnings.warn(
                    f"solver should have a '{mtd}()' method",
                    errors.OpInfWarning,
                )

        self.__solver = solver

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr, Wr=None):
        r"""Construct a reduced-order model by taking the (Petrov-)Galerkin
        projection of each model operator.

        Consider a model :math:`\z = \f(\q, \u)` where

        * :math:`\q\in\RR^n` is the model state,
        * :math:`\u\in\RR^m` is the input, and
        * :math:`\z\in\RR^n` is the model left-hand side.

        Given a *trial basis* :math:`\Vr\in\RR^{n\times r}` and a *test basis*
        :math:`\Wr\in\RR^{n\times r}`, the corresponding
        *intrusive reduced-order model* is the model
        :math:`\zhat = \fhat(\qhat, \u)` where

        .. math::
           \zhat = \Wr\trp\z,
           \qquad
           \fhat(\qhat,\u) = (\Wr\trp\Vr)^{-1}\Wr\trp\f(\Vr\qhat,\u).

        Here,

        * :math:`\qhat\in\RR^r` is the reduced-order state,
        * :math:`\u\in\RR^m` is the input (as before), and
        * :math:`\zhat\in\RR^r` is the reduced-order left-hand side.

        This approach uses the low-dimensional state approximation
        :math:`\q = \Vr\qhat`.
        If :math:`\Wr = \Vr`, the result is called a *Galerkin projection*.
        If :math:`\Wr \neq \Vr`, it is called a *Petrov-Galerkin projection*.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        reduced_model : Model
            Reduced-order model obtained from (Petrov-)Galerkin projection.
        """
        # TODO: preserve the _to_infer indices?
        return self.__class__(
            [
                (
                    old_op.copy()
                    if _operators.is_uncalibrated(old_op)
                    else old_op.galerkin(Vr, Wr)
                )
                for old_op in self.operators
            ]
        )

    # Validation methods ------------------------------------------------------
    def _check_inputargs(self, u, argname):
        """Check that the model structure agrees with input arguments."""
        if self._has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required")

        if not self._has_inputs and u is not None:
            warnings.warn(
                f"argument '{argname}' should be None, "
                "argument will be ignored",
                errors.OpInfWarning,
            )

    def _check_is_trained(self):
        """Ensure that the model is trained and ready for prediction."""
        if self.state_dimension is None:
            raise AttributeError("no state_dimension (call fit())")
        if self._has_inputs and (self.input_dimension is None):
            raise AttributeError("no input_dimension (call fit())")

        for op in self.operators:
            if op.entries is None:
                raise AttributeError("model not trained (call fit())")

    def __eq__(self, other):
        """Two models are equal if they have equivalent operators."""
        if not isinstance(other, self.__class__):
            return False
        if len(self.operators) != len(other.operators):
            return False
        for selfop, otherop in zip(self.operators, other.operators):
            if selfop != otherop:
                return False
        if self.state_dimension != other.state_dimension:
            return False
        if self.input_dimension != other.input_dimension:
            return False
        return True

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Make a copy of the model."""
        return self.__class__([op.copy() for op in self.operators])
