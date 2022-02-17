# core/operators/_base.py

__all__ = []

import abc
import numpy as np


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
    @abc.abstractmethod
    def __init__(self, entries):
        """Set operator entries and save operator name."""
        self.__entries = entries

    @abc.abstractmethod
    def __call__(*args, **kwargs):                          # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _validate_entries(entries):
        """Ensure argument is a NumPy array and screen for NaN, Inf entries."""
        if not isinstance(entries, np.ndarray):
            raise TypeError("operator entries must be NumPy array")
        if np.any(np.isnan(entries)):
            raise ValueError("operator entries must not be NaN")
        elif np.any(np.isinf(entries)):
            raise ValueError("operator entries must not be Inf")

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
        return np.all(self.entries == other.entries)


class _BaseParametricOperator(abc.ABC):
    """Base class for reduced-order model operators that depend on external
    parameters. Calling the instantiated object with an external parameter
    results in a non-parametric operator:

    >>> parametric_operator = MyParametricOperator(init_args)
    >>> nonparametric_operator = parametric_operator(param)
    >>> isinstance(nonparametric_operator, _BaseNonparametricOperator)
    True
    """
    @abc.abstractmethod
    def __init__(self):                                     # pragma: no cover
        """Set operator entries, affine functions, name, etc."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, parameter):                          # pragma: no cover
        """Return the nonparametric operator corresponding to the parameter."""
        raise NotImplementedError
