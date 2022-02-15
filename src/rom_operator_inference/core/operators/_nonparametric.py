# core/operators/nonparametric.py
"""Classes for polynomial operators with no external parameter dependencies."""

__all__ = [
            "ConstantOperator",
            "LinearOperator",
            "QuadraticOperator",
            # CrossQuadraticOperator",
            "CubicOperator",
          ]

import abc
import numpy as np

from ...utils import (kron2c_indices, kron3c_indices,
                      compress_quadratic, compress_cubic)


# Base non-parametric operator ================================================
class _BaseNonparametricOperator(abc.ABC):
    """Base class for operators that are part of reduced-order models.
    Call the instantiated object to evaluate the operator on an input.

    Attributes
    ----------
    entries : ndarray
        Actual NumPy array representing the operator.
    shape : tuple
        Shape of the operator entries array.
    symbol : str
        Mathematical symbol for the operator, e.g., 'A' or 'H'.
        Used in the string representation of the operator and associated ROM.
    """
    @abc.abstractmethod
    def __init__(self, entries, symbol):
        """Set operator entries and save operator name."""
        self.__entries = entries
        self.symbol = symbol

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

    @abc.abstractmethod
    def _str(self):
        raise NotImplementedError                           # pragma: no cover

    @abc.abstractmethod
    def __call__(*args, **kwargs):
        raise NotImplementedError                           # pragma: no cover

    def __getitem__(self, key):
        """Slice into the discrete representation of the operator."""
        return self.entries[key]

    def __eq__(self, other):
        """Test whether two Operator objects are numerically equal."""
        if not isinstance(other, self.__class__):
            return False
        return np.all(self.entries == other.entries)


# Public non-parametric operators =============================================
class ConstantOperator(_BaseNonparametricOperator):
    """Constant terms."""
    def __init__(self, entries, symbol='c'):
        self._validate_entries(entries)

        # Flatten operator if needed or report dimension error.
        if entries.ndim > 1:
            if entries.ndim == 2 and 1 in entries.shape:
                entries = np.ravel(entries)
            else:
                raise ValueError("constant operator must be one-dimensional")

        _BaseNonparametricOperator.__init__(self, entries, symbol)

    def __call__(self, *args):
        return self.entries

    def _str(self, label=None):
        return self.symbol


class LinearOperator(_BaseNonparametricOperator):
    """Linear state or input operator.

    Example
    -------
    >>> import numpy as np
    >>> A = opinf.core.operators.LinearOperator(np.random.random((10, 10)))
    >>> q = np.random.random(10)
    >>> A(q)                        # Evaluate Aq.
    """
    def __init__(self, entries, symbol='A'):
        """Check dimensions and set operator entries."""
        self._validate_entries(entries)

        if entries.ndim == 1:
            entries = entries.reshape((-1, 1))
        if entries.ndim != 2:
            raise ValueError("linear operator must be two-dimensional")

        _BaseNonparametricOperator.__init__(self, entries, symbol)

    def __call__(self, q):
        return self.entries @ np.atleast_1d(q)

    def _str(self, label):
        return f"{self.symbol}{label}"


class QuadraticOperator(_BaseNonparametricOperator):
    """Operator for quadratic interactions of a state/input with itself
    (compact Kronecker).

    Example
    -------
    >>> import numpy as np
    >>> H = opinf.core.operators.QuadraticOperator(np.random.random((10, 100)))
    >>> q = np.random.random(10)
    >>> H(q)                        # Evaluate H[q ⊗ q].
    """
    def __init__(self, entries, symbol='H'):
        """Check dimensions and set operator entries."""
        self._validate_entries(entries)

        # TODO: allow reshaping from three-dimensional tensor?
        if entries.ndim != 2:
            raise ValueError("quadratic operator must be two-dimensional")
        r1, r2 = entries.shape
        # TODO: relax this requirement?
        # If so, need to try to compress if r2 is not a perfect square.
        if r2 == r1**2:
            entries = compress_quadratic(entries)
        elif r2 != r1 * (r1 + 1) // 2:
            raise ValueError("invalid dimensions for quadratic operator")
        self._mask = kron2c_indices(r1)

        _BaseNonparametricOperator.__init__(self, entries, symbol)

    def __call__(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)

    def _str(self, label):
        return f"{self.symbol}[{label} ⊗ {label}]"


# class CrossQuadraticOperator(QuadraticOperator):
#     """Quadratic terms of different states / inputs (full Kronecker)."""
#     def __init__(self, entries, symbol='N'):
#         self._validate_entries(entries)
#
#         _BaseNonparametricOperator.__init__(self, entries, symbol)
#
#     def __call__(self, q1, q2):
#         return self.entries @ np.kron(q1, q2)
#         # TODO: what about two-dimensional inputs?
#         # la.khatri_rao() will do this, but that *requires* that the
#         # inputs q1 and q2 are both two-dimensional.
#         # TODO: and what about scalar inputs? (special case of r=1 or m=1).
#
#     def _str(self, label1, label2):
#         return f"{self.symbol}[{label1} ⊗ {label2}]"


class CubicOperator(_BaseNonparametricOperator):
    """Cubic terms of a state/input with itself (compact Kronecker).

    Example
    -------
    >>> import numpy as np
    >>> G = opinf.core.operators.CubicOperator(np.random.random((10, 1000)))
    >>> q = np.random.random(10)
    >>> G(q)                        # Evaluate G[q ⊗ q ⊗ q].
    """
    def __init__(self, entries, symbol='G'):
        self._validate_entries(entries)

        # TODO: allow reshaping from three-dimensional tensor?
        if entries.ndim != 2:
            raise ValueError("cubic operator must be two-dimensional")
        r1, r2 = entries.shape
        # TODO: relax this requirement?
        # If so, need to try to compress if r2 is not a perfect square.
        if r2 == r1**3:
            entries = compress_cubic(entries)
        elif r2 != r1 * (r1 + 1) * (r1 + 2) // 6:
            raise ValueError("invalid dimensions for cubic operator")
        self._mask = kron3c_indices(r1)

        _BaseNonparametricOperator.__init__(self, entries, symbol)

    def __call__(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)

    def _str(self, label):
        return f"{self.symbol}[{label} ⊗ {label} ⊗ {label}]"
