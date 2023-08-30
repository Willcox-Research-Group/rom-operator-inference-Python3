# roms/_base.py
"""Abstract base classes for reduced-order models."""

__all__ = []

import abc
import warnings
import numpy as np

from .. import errors
from .. import basis as _basis
from .. import operators_new as _operators


class _BaseROM(abc.ABC):
    """Base class for all opinf reduced model classes."""
    _LHS_ARGNAME = "lhs"    # Name of LHS argument in fit(), e.g., "ddts".
    _LHS_LABEL = None       # String representation of LHS, e.g., "dq / dt".
    _STATE_LABEL = None     # String representation of state, e.g., "q(t)".
    _INPUT_LABEL = None     # String representation of input, e.g., "u(t)".

    def __init__(self, basis, operators):
        """Set the basis and define the model structure."""
        self.__r = None
        self.__m = None
        self.__basis = None
        self.__operators = None

        self.basis = basis
        if isinstance(operators, str):
            operators = [
                OpClass() for key, OpClass in [
                    ('c', _operators.ConstantOperator),
                    ('A', _operators.LinearOperator),
                    ('H', _operators.QuadraticOperator),
                    ('G', _operators.CubicOperator),
                    ('B', _operators.InputOperator),
                    ('N', _operators.StateInputOperator),
                ] if key in operators]
        self.operators = operators

    # Properties: basis -------------------------------------------------------
    @property
    def basis(self):
        """Basis for the reduced space (e.g., POD), of shape `(n, r)`."""
        return self.__basis

    @basis.setter
    def basis(self, basis):
        """Set the ``basis``, thereby fixing the dimensions ``n`` and ``r``.

        Parameters
        ----------
        basis : opinf.basis object or (n, r) ndarray
            An instantiated basis object or the entries of a linear basis, in
            which case entries are wrapped in an ``opinf.basis.LinearBasis``
            object.
        """
        if basis is not None:
            if not isinstance(basis, _basis._base._BaseBasis):
                basis = _basis.LinearBasis().fit(basis)
            if basis.shape[0] < basis.shape[1]:
                raise ValueError("basis must be n x r with n > r")
            self.__r = basis.shape[1]
        self.__basis = basis

    @basis.deleter
    def basis(self):
        """Delete the basis (but not operator / dimension attributes)."""
        self.__basis = None

    # Properties: operators ---------------------------------------------------
    @property
    def operators(self):
        """Operators comprising the terms of the reduced-order model."""
        return self.__operators

    @operators.setter
    def operators(self, ops):
        """Set the operators."""
        if len(ops) == 0:
            raise ValueError("at least one operator required")

        toinfer = []                    # Operators to infer (no entries yet).
        known = []                      # Operators whose entries are set.
        self._has_inputs = False        # Whether any operators use inputs.
        for i, op in enumerate(ops):
            if not isinstance(op, _operators._base._BaseNonparametricOperator):
                raise TypeError("expected list of nonparametric operators")
            if op.entries is None:
                toinfer.append(i)
            else:
                known.append(i)
            if isinstance(op, (_operators.InputOperator,
                               _operators.StateInputOperator)):
                self._has_inputs = True
        if not self._has_inputs:
            self.m = 0

        # Validate shapes of operators with known entries.
        if len(known) > 0:
            if self.basis is None:
                # No basis, so assume all operators must have same shape[0].
                rs = {ops[i].shape[0] for i in known}
                if len(rs) > 1:
                    raise errors.DimensionalityError(
                        "operators not aligned (shape[0] must be the same)")
                self.r = rs.pop()
            else:
                for i in known:
                    if (dim := ops[i].shape[0]) not in (self.r, self.n):
                        raise errors.DimensionalityError(
                            "operators not aligned with basis "
                            f"(operators[{i}].shape[0] = {dim} must be "
                            f"r = {self.r} or n = {self.n})")

        # Store attributes.
        self.__operators = ops
        self._indices_of_operators_to_infer = toinfer
        self._indices_of_known_operators = known

    def _clear(self):
        """Reset the entries of the non-intrusive ROM operators."""
        for i in self._indices_of_operators_to_infer:
            self.operators[i]._clear()

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
    @property
    def n(self):
        """Dimension of the full-order model."""
        return self.basis.shape[0] if self.basis is not None else None

    @n.setter
    def n(self, n):
        """Setting this dimension is not allowed."""
        raise AttributeError("can't set attribute (n = basis.shape[0])")

    @property
    def m(self):
        """Dimension of the input term, if present."""
        return self.__m

    @m.setter
    def m(self, m):
        """Set input dimension; only allowed if an input-using operator is
        present in the model.
        """
        if not self._has_inputs and m != 0:
            raise AttributeError("can't set attribute (no input operators)")
        self.__m = m

    @property
    def r(self):
        """Dimension of the reduced-order model."""
        return self.__r

    @r.setter
    def r(self, r):
        """Set ROM dimension; only allowed if the basis is None."""
        if self.basis is not None:
            raise AttributeError("can't set attribute (r = basis.shape[1])")
        self.__r = r

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
        if self.n:
            out.append(f"Full-order dimension    n = {self.n:d}")
        if self.m:
            out.append(f"Input/control dimension m = {self.m:d}")
        if self.r:
            out.append(f"Reduced-order dimension r = {self.r:d}")

        return '\n'.join(out)

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
            warnings.warn(f"argument '{argname}' should be None, "
                          "argument will be ignored", UserWarning)

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
    def compress(self, state, label="argument"):
        """Map high-dimensional states to low-dimensional latent coordinates.

        Parameters
        ----------
        state : (n, ...) or (r, ...) ndarray
            High- or low-dimensional state vector or a collection of these.
            If ``state.shape[0]`` is r (already low-dimensional), do nothing.
        label : str
            Name for state (used only in error reporting).

        Returns
        -------
        state_ : (r, ...) ndarray
            Low-dimensional encoding of `state`.
        """
        if self.r is None:
            raise AttributeError("reduced dimension 'r' not set")
        if state.shape[0] == self.r:
            return state
        if self.basis is None:
            raise AttributeError("basis not set")
        if state.shape[0] != self.n:
            raise errors.DimensionalityError(f"{label} not aligned with basis")
        return self.basis.compress(state)

    def decompress(self, state_, label="argument"):
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        state_ : (r, ...) ndarray
            Low-dimensional latent coordinate vector or a collection of these.
        label : str
            Name for ``state_`` (used only in error reporting).

        Returns
        -------
        state : (n, ...) ndarray
            High-dimensional decoding of ``state_``.
        """
        if self.basis is None:
            raise AttributeError("basis not set")
        if state_.shape[0] != self.r:
            raise errors.DimensionalityError(f"{label} not aligned with basis")
        return self.basis.decompress(state_)

    # ROM evaluation ----------------------------------------------------------
    def evaluate(self, state_, input_=None):
        r"""Evaluate and sum each model operator.

        This is the right-hand side of the model, i.e., the function
        :math:`\widehat{\mathbf{F}}(\widehat{\mathbf{q}}, \mathbf{u})`
        where the model can be written as one of the following:

        - :math:`\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
            = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}(t), \mathbf{u}(t))`
            (continuous time)
        - :math:`\widehat{\mathbf{q}}_{j+1}
            = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}, \mathbf{u}_{j})`
            (discrete time)
        - :math:`\widehat{\mathbf{g}}
            = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}, \mathbf{u})`
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
            out += op.evaluate(state_, input_)
        return out

    def jacobian(self, state_, input_=None):
        r"""Construct and sum the Jacobian of each model operators.

        This the derivative of the right-hand side of the model with respect
        to the state, i.e., the function :math:`\frac{
        \partial \widehat{\mathbf{F}}}{\partial \widehat{\mathbf{q}}}`
        where the model can be written as one of the following:

        - :math:`\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
            = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}(t), \mathbf{u}(t))`
            (continuous time)
        - :math:`\widehat{\mathbf{q}}_{j+1}
            = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}, \mathbf{u}_{j})`
            (discrete time)
        - :math:`\widehat{\mathbf{g}}
            = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}, \mathbf{u})`
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
        raise NotImplementedError                           # pragma: no cover

    @abc.abstractmethod
    def predict(*args, **kwargs):
        """Solve the reduced-order model under specified conditions."""
        raise NotImplementedError                           # pragma: no cover

    # Model persistence (not required but suggested) --------------------------
    def save(*args, **kwargs):
        """Save the reduced-order structure / operators in HDF5 format."""
        raise NotImplementedError("use pickle/joblib")

    @classmethod
    def load(*args, **kwargs):
        """Load a previously saved reduced-order model from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")
