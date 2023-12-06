# models/_base.py
"""Abstract base classes for dynamical systems models."""

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


class _MonolithicModel(abc.ABC):
    """Base class for all monolithic model classes."""

    _LHS_ARGNAME = "lhs"  # Name of LHS argument in fit(), e.g., "ddts".
    _LHS_LABEL = None  # String representation of LHS, e.g., "dq / dt".
    _STATE_LABEL = None  # String representation of state, e.g., "q(t)".
    _INPUT_LABEL = None  # String representation of input, e.g., "u(t)".

    def __init__(self, operators):
        """Define the model structure.

        Parameters
        ----------
        operators : list of :mod:`opinf.operators` objects
            Operators comprising the terms of the model.
        """
        self.__r = None
        self.__m = None
        self.__operators = None

        self.operators = operators

    # Properties: operators ---------------------------------------------------
    @property
    def operators(self):
        """Operators comprising the terms of the model."""
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

    def _get_operator_of_type(self, OpClass):
        """Return the first operator of type ``OpClass``."""
        for op in self.operators:
            if isinstance(op, OpClass):
                return op

    @property
    def c_(self):
        """:class:`opinf.operators_new.ConstantOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.ConstantOperator)

    @property
    def A_(self):
        """:class:`opinf.operators_new.LinearOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.LinearOperator)

    @property
    def H_(self):
        """:class:`opinf.operators_new.QuadraticOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.QuadraticOperator)

    @property
    def G_(self):
        """:class:`opinf.operators_new.CubicOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.CubicOperator)

    @property
    def B_(self):
        """:class:`opinf.operators_new.InputOperator` (or ``None``)."""
        return self._get_operator_of_type(_operators.InputOperator)

    @property
    def N_(self):
        """:class:`opinf.operators_new.StateInputOperator` (or ``None``)."""
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
        self.__r = r

    @staticmethod
    def _check_input_dimension_consistency(ops):
        """Ensure all *input* operators with initialized entries have the same
        ``input dimension``.
        """
        if len(inputops := [op for op in ops if _is_inputop(op)]) == 0:
            return 0
        ms = {op.input_dimension for op in inputops if op.entries is not None}
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
                    _is_inputop(op)
                    and op.entries is not None
                    and op.input_dimension != m
                ):
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
        if self.state_dimension:
            out.append(f"State dimension r = {self.state_dimension:d}")
        if self.input_dimension:
            out.append(f"Input dimension m = {self.input_dimension:d}")

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
        if self.state_dimension is None:
            raise AttributeError("no reduced dimension 'r' (call fit())")
        if self._has_inputs and (self.input_dimension is None):
            raise AttributeError("no input dimension 'm' (call fit())")

        for op in self.operators:
            if op.entries is None:
                raise AttributeError("model not trained (call fit())")

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr, Wr=None):
        r"""Construct a reduced-order model by taking the (Petrov-)Galerkin
        projection of each model operator.

        Consider a model :math:`\z = \f(\q,\u)` where

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
           \fhat(\qhat,\u) = \Wr\trp\f(\Vr\qhat,\u).

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
        return self.__class__(
            [
                old_op.galerkin(Vr, Wr)
                if old_op.entries is not None
                else old_op.copy()
                for old_op in self.operators
            ]
        )

    # Model evaluation --------------------------------------------------------
    def rhs(self, state, input_=None):
        r"""Evaluate the right-hand side of the model by applying each operator
        and summing the results.

        This is the function :math:`\widehat{\mathbf{F}}(\qhat, \u)`
        where the model can be written as one of the following:

        * :math:`\ddt\qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))`
          (continuous time)
        * :math:`\qhat_{j+1} = \widehat{\mathbf{F}}(\qhat_j, \u_j)`
          (discrete time)
        * :math:`\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat, \u)`
          (steady state)

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector corresponding to the state.

        Returns
        -------
        evaluation : (r,) ndarray
            Evaluation of the right-hand side of the model.
        """
        state = np.atleast_1d(state)
        out = np.zeros(state.shape, dtype=float)
        for op in self.operators:
            out += op.apply(state, input_)
        return out

    def jacobian(self, state, input_=None):
        r"""Construct and sum the state Jacobian of each model operator.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function
        :math:`\ddqhat\widehat{\mathbf{F}}}(\qhat, \u)`
        where the model can be written as one of the following:

        - :math:`\ddt\qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))`
          (continuous time)
        - :math:`\qhat_{j+1} = \widehat{\mathbf{F}}(\qhat_{j}, \u_{j})`
          (discrete time)
        - :math:`\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\qhat, \u)`
          (steady state)

        Parameters
        ----------
        state : (r,) ndarray
            State vector :math:`\qhat`.
        input_ : (m,) ndarray or None
            Input vector :math:`\u`.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model.
        """
        r = self.state_dimension
        out = np.zeros((r, r), dtype=float)
        for op in self.operators:
            out += op.jacobian(state, input_)
        return out

    # Abstract public methods (must be implemented by child classes) ----------
    @abc.abstractmethod
    def fit(*args, **kwargs):
        """Train the model with the specified data via operator inference."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def predict(*args, **kwargs):
        """Solve the model under specified conditions."""
        raise NotImplementedError  # pragma: no cover

    # Model persistence (not required but suggested) --------------------------
    def save(*args, **kwargs):
        """Save the model structure and operators in HDF5 format."""
        raise NotImplementedError("use pickle/joblib")

    @classmethod
    def load(*args, **kwargs):
        """Load a previously saved model from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")


# TODO: class _ParametricMonolithicModel(_MonolithicModel)?
