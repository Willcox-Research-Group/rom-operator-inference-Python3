# roms/_base.py
"""Abstract base classes for reduced-order models."""

__all__ = []

import abc
import warnings
import numpy as np

from .. import errors
from .. import operators_new as _operators


_is_inputop = _operators._base._is_input_operator

_OPERATOR_SHORTCUTS = {
    "c": _operators.ConstantOperator,
    "A": _operators.LinearOperator,
    "H": _operators.QuadraticOperator,
    "G": _operators.CubicOperator,
    "B": _operators.InputOperator,
    "N": _operators.StateInputOperator,
}


class _MonolithicROM(abc.ABC):
    """Base class for all monolithic reduced-order model classes."""

    _LHS_ARGNAME = "lhs"  # Name of LHS argument in fit(), e.g., "ddts".
    _LHS_LABEL = None  # String representation of LHS, e.g., "dq / dt".
    _STATE_LABEL = None  # String representation of state, e.g., "q(t)".
    _INPUT_LABEL = None  # String representation of input, e.g., "u(t)".

    def __init__(self, operators):
        """Define the model structure.

        Parameters
        ----------
        operators : list of :mod:`opinf.operators` objects
            Operators comprising the terms of the reduced-order model.
        """
        self.__r = None
        self.__m = None
        self.__operators = None

        self.operators = operators

    # Properties: operators ---------------------------------------------------
    @property
    def operators(self):
        """Operators comprising the terms of the reduced-order model."""
        return self.__operators

    @operators.setter
    def operators(self, ops):
        """Set the operators."""
        if isinstance(ops, _operators._base._NonparametricOperator):
            ops = [ops]
        if len(ops) == 0 or ops is None:
            raise ValueError("at least one operator required")

        toinfer = []  # Operators to infer (no entries yet).
        known = []  # Operators whose entries are set.
        self._has_inputs = False  # Whether any operators use inputs.
        ops = list(ops)
        for i in range(len(ops)):
            op = ops[i]
            if isinstance(op, str) and op in _OPERATOR_SHORTCUTS:
                op = ops[i] = _OPERATOR_SHORTCUTS[op]()
            if not isinstance(op, _operators._base._NonparametricOperator):
                raise TypeError("expected list of nonparametric operators")
            if op.entries is None:
                toinfer.append(i)
            else:
                known.append(i)
            if _operators._base._is_input_operator(op):
                self._has_inputs = True

        # Store attributes.
        self.__r = self._check_state_dimension_consistency(ops)
        self.__m = self._check_input_dimension_consistency(ops)
        self.__operators = ops
        self._indices_of_operators_to_infer = toinfer
        self._indices_of_known_operators = known

    def _clear(self):
        """Reset the entries of the non-intrusive ROM operators and the
        state and input dimensions.
        """
        for i in self._indices_of_operators_to_infer:
            self.operators[i]._clear()
        self.__r = self._check_state_dimension_consistency(self.operators)
        self.__m = self._check_input_dimension_consistency(self.operators)

    def __iter__(self):
        """Iterate through the ROM operators."""
        for op in self.operators:
            yield op

    def _get_operator_of_type(self, OpClass):
        """Return the first operator of type ``OpClass``."""
        for op in self.operators:
            if isinstance(op, OpClass):
                return op

    @property
    def c_(self):
        """ConstantOperator, of shape (r,)."""
        return self._get_operator_of_type(_operators.ConstantOperator)

    @property
    def A_(self):
        """LinearOperator, of shape (r, r)."""
        return self._get_operator_of_type(_operators.LinearOperator)

    @property
    def H_(self):
        """QuadraticOperator, of shape (r, r(r+1)/2)."""
        return self._get_operator_of_type(_operators.QuadraticOperator)

    @property
    def G_(self):
        """CubicOperator, of shape (r, r(r+1)(r+2)/6)."""
        return self._get_operator_of_type(_operators.CubicOperator)

    @property
    def B_(self):
        """InputOperator, of shape (r, m)."""
        return self._get_operator_of_type(_operators.InputOperator)

    @property
    def N_(self):
        """StateInputOperator, of shape (r, rm)."""
        return self._get_operator_of_type(_operators.StateInputOperator)

    # Properties: dimensions --------------------------------------------------
    @staticmethod
    def _check_state_dimension_consistency(ops):
        """Ensure all operators with initialized entries have the same
        state dimension (``shape[0]``).
        """
        rs = {op.shape[0] for op in ops if op.entries is not None}
        if len(rs) > 1:
            raise errors.DimensionalityError(
                "operators not aligned "
                "(state dimension must be the same for all operators)"
            )
        return rs.pop() if len(rs) == 1 else None

    @property
    def r(self):
        """Dimension of the reduced-order state."""
        return self.__r

    @r.setter
    def r(self, r):
        """Set the reduced-order state dimension.
        Not allowed if any existing operators have ``shape[0] != r``.
        """
        if self.__operators is not None:
            for op in self.operators:
                if op.entries is not None and op.shape[0] != r:
                    raise AttributeError(
                        "can't set attribute "
                        f"(existing operators have r = {self.__r})"
                    )
        self.__r = r

    @staticmethod
    def _check_input_dimension_consistency(ops):
        """Ensure all *input* operators with initialized entries have the same
        input dimension (``m``).
        """
        if len(inputops := [op for op in ops if _is_inputop(op)]) == 0:
            return 0
        ms = {op.m for op in inputops if op.entries is not None}
        if len(ms) > 1:
            raise errors.DimensionalityError(
                "input operators not aligned "
                "(input dimension must be the same for all input operators)"
            )
        return ms.pop() if len(ms) == 1 else None

    @property
    def m(self):
        """Dimension of the input term, if present."""
        return self.__m

    @m.setter
    def m(self, m):
        """Set input dimension.
        Only allowed if an input-using operator is present in the model
        and the ``m`` attribute of every existing input operator agrees.
        """
        if not self._has_inputs and m != 0:
            raise AttributeError("can't set attribute (no input operators)")
        if self.__operators is not None:
            for op in self.operators:
                if _is_inputop(op) and op.entries is not None and op.m != m:
                    raise AttributeError(
                        "can't set attribute "
                        f"(existing input operators have m = {self.__m})"
                    )
        self.__m = m

    # String representation ---------------------------------------------------
    def __str__(self):
        """String representation: structure of the model, dimensions, etc."""
        # Build model structure.
        out, terms = [], []
        for op in self.operators:
            terms.append(op._str(self._STATE_LABEL, self._INPUT_LABEL))
        structure = " + ".join(terms)
        out.append(f"Model structure: {self._LHS_LABEL} = {structure}")

        # Report dimensions.
        if self.r:
            out.append(f"State dimension r = {self.r:d}")
        if self.m:
            out.append(f"Input dimension m = {self.m:d}")

        return "\n".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        uniqueID = f"<{self.__class__.__name__} object at {hex(id(self))}>"
        return f"{uniqueID}\n{str(self)}"

    # Validation methods ------------------------------------------------------

    def _check_inputargs(self, u, argname):
        """Check that the model structure agrees with input arguments."""
        if self._has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required")

        if not self._has_inputs and u is not None:
            warnings.warn(
                f"argument '{argname}' should be None, "
                "argument will be ignored",
                UserWarning,
            )

    def _check_is_trained(self):
        """Ensure that the model is trained and ready for prediction."""
        if self.r is None:
            raise AttributeError("no reduced dimension 'r' (call fit())")
        if self._has_inputs and (self.m is None):
            raise AttributeError("no input dimension 'm' (call fit())")

        for op in self.operators:
            if op.entries is None:
                raise AttributeError("model not trained (call fit())")

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr, Wr=None):
        """Construct a new ROM by taking the Galerkin projection of each
        full-order operator.
        """
        return self.__class__(
            [
                old_op.galerkin(Vr, Wr)
                if i in self._indices_of_known_operators
                else old_op.__class__()
                for i, old_op in enumerate(self.operators)
            ]
        )

    # ROM evaluation ----------------------------------------------------------
    def evaluate(self, state_, input_=None):
        r"""Evaluate and sum each model operator.

        This is the right-hand side of the model, i.e., the function
        :math:`\widehat{\mathbf{F}}(\qhat, \u)`
        where the model can be written as one of the following:

        * :math:`\ddt\qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))`
          (continuous time)
        * :math:`\qhat_{j+1} = \widehat{\mathbf{F}}(\qhat_j, \u_j)`
          (discrete time)
        * :math:`\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat, \u)`
            (steady state)

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
        state_ = np.atleast_1d(state_)
        out = np.zeros(state_.shape, dtype=float)
        for op in self.operators:
            out += op.apply(state_, input_)
        return out

    def jacobian(self, state_, input_=None):
        r"""Construct and sum the Jacobian of each model operators.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\frac{
        \partial \widehat{\mathbf{F}}}{\partial \qhat}`
        where the model can be written as one of the following:

        - :math:`\frac{\text{d}}{\text{d}t}\qhat(t)
            = \widehat{\mathbf{F}}(\qhat(t), \u(t))`
            (continuous time)
        - :math:`\qhat_{j+1}
            = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`
            (discrete time)
        - :math:`\widehat{\mathbf{g}}
            = \widehat{\mathbf{F}}(\qhat, \u)`
            (steady state)

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
        out = np.zeros((self.r, self.r), dtype=float)
        for op in self.operators:
            out += op.jacobian(state_, input_)
        return out

    # Abstract public methods (must be implemented by child classes) ----------
    @abc.abstractmethod
    def fit(*args, **kwargs):
        """Train the reduced-order model with the specified data."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def predict(*args, **kwargs):
        """Solve the reduced-order model under specified conditions."""
        raise NotImplementedError  # pragma: no cover

    # Model persistence (not required but suggested) --------------------------
    def save(*args, **kwargs):
        """Save the reduced-order structure / operators in HDF5 format."""
        raise NotImplementedError("use pickle/joblib")

    @classmethod
    def load(*args, **kwargs):
        """Load a previously saved reduced-order model from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")


# TODO: class _ParametricMonolithicROM(_MonolithicROM)?
