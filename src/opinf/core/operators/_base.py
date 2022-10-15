# core/operators/_base.py
"""Abstract base classes for operators.

Classes
-------
* _BaseNonparametricOperator: base for operators without parameter dependence.
* _BaseParametricOperator: base for operators with parameter dependence.
"""

__all__ = [
    "is_operator",
    "is_parametric_operator",
]

import abc
import numpy as np


def is_operator(op):
    """Return True if `op` is a valid Operator object."""
    return isinstance(op, (
        _BaseNonparametricOperator,
        _BaseParametricOperator)
    )


def is_parametric_operator(op):
    """Return True if `op` is a valid ParametricOperator object."""
    return isinstance(op, _BaseParametricOperator)


class _BaseNonparametricOperator(abc.ABC):
    """Base class for reduced-order model operators that do not depend on
    external parameters. Call the instantiated object to evaluate the operator
    on an input.

    Attributes
    ----------
    entries : ndarray
        Actual NumPy array representing the operator.
    shape : tuple
        Shape of the operator entries array.
    """
    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        """Discrete representation of the operator."""
        return self.__entries

    @property
    def shape(self):
        """Shape of the operator."""
        return self.entries.shape

    def __getitem__(self, key):
        """Slice into the entries of the operator."""
        return self.entries[key]

    def __eq__(self, other):
        """Return True if two Operator objects are numerically equal."""
        if not isinstance(other, self.__class__):
            return False
        if self.shape != other.shape:
            return False
        return np.all(self.entries == other.entries)

    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def __init__(self, entries):
        """Set operator entries."""
        self.__entries = entries

    @abc.abstractmethod                                     # pragma: no cover
    def evaluate(self, *args, **kwargs):
        """Apply the operator mapping to the given states / inputs."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Apply the operator mapping to the given states / inputs."""
        return self.evaluate(*args, **kwargs)

    # Validation --------------------------------------------------------------
    @staticmethod
    def _validate_entries(entries):
        """Ensure argument is a NumPy array and screen for NaN, Inf entries."""
        if not isinstance(entries, np.ndarray):
            raise TypeError("operator entries must be NumPy array")
        if np.any(np.isnan(entries)):
            raise ValueError("operator entries must not be NaN")
        elif np.any(np.isinf(entries)):
            raise ValueError("operator entries must not be Inf")


class _BaseParametricOperator(abc.ABC):
    """Base class for reduced-order model operators that depend on external
    parameters. Calling the instantiated object with an external parameter
    results in a non-parametric operator:

    >>> parametric_operator = MyParametricOperator(init_args)
    >>> nonparametric_operator = parametric_operator(parameter_value)
    >>> isinstance(nonparametric_operator, _BaseNonparametricOperator)
    True
    """
    # Properties --------------------------------------------------------------
    # Must be specified by child classes.
    _OperatorClass = NotImplemented

    @property
    def OperatorClass(self):
        """Class of nonparametric operator to represent this parametric
        operator at a particular parameter, a subclass of
        core.operators._BaseNonparametricOperator:
        >>> type(MyParametricOperator(init_args)(parameter_value)).
        """
        return self._OperatorClass

    @property
    def p(self):
        """Dimension of the parameter space, i.e., individual parameters are
        ndarrays of shape (p,) (or scalars if p = 1)
        """
        return self.__p

    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def __init__(self):
        """Validate the OperatorClass.
        Child classes must implement this method, which should set and
        validate attributes needed to construct the parametric operator.
        """
        self.__p = None

        # Validate the OperatorClass.
        if not issubclass(self.OperatorClass, _BaseNonparametricOperator):
            raise RuntimeError("invalid OperatorClass "
                               f"'{self._OperatorClass.__name__}'")

    @abc.abstractmethod
    def __call__(self, parameter):                          # pragma: no cover
        """Return the nonparametric operator corresponding to the parameter,
        of type self.OperatorClass.
        """
        raise NotImplementedError

    # Input validation (shape checking) ---------------------------------------
    @staticmethod
    def _check_shape_consistency(iterable, prefix="operator matrix"):
        """Ensure that each array in `iterable` has the same shape."""
        shape = np.shape(iterable[0])
        if any(np.shape(A) != shape for A in iterable):
            raise ValueError(f"{prefix} shapes inconsistent")

    def _set_parameter_dimension(self, parameters):
        """Extract and save the dimension of the parameter space."""
        shape = np.shape(parameters)
        if len(shape) == 1:
            self.__p = 1
        elif len(shape) == 2:
            self.__p = shape[1]
        else:
            raise ValueError("parameter values must be scalars or 1D arrays")

    def _check_parameter_dimension(self, parameter):
        """Ensure a new parameter has the expected shape."""
        if self.p is None:
            raise RuntimeError("parameter dimension p not set")
        if np.atleast_1d(parameter).shape[0] != self.p:
            raise ValueError(f"expected parameter of shape ({self.p:d},)")
