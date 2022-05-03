# core/operators/_nonparametric.py
"""Classes for polynomial operators with no external parameter dependencies."""

__all__ = [
            "ConstantOperator",
            "LinearOperator",
            "QuadraticOperator",
            # CrossQuadraticOperator",
            "CubicOperator",
            "nonparametric_operators",
          ]

import numpy as np

from ...utils import (kron2c_indices, kron3c_indices,
                      compress_quadratic, compress_cubic)
from ._base import _BaseNonparametricOperator


class ConstantOperator(_BaseNonparametricOperator):
    """Constant terms."""
    def __init__(self, entries):
        self._validate_entries(entries)

        # Flatten operator if needed or report dimension error.
        if entries.ndim > 1:
            if entries.ndim == 2 and 1 in entries.shape:
                entries = np.ravel(entries)
            else:
                raise ValueError("constant operator must be one-dimensional")

        _BaseNonparametricOperator.__init__(self, entries)

    def evaluate(self, *args):
        return self.entries


class LinearOperator(_BaseNonparametricOperator):
    """Linear state or input operator.

    Example
    -------
    >>> import numpy as np
    >>> A = opinf.core.operators.LinearOperator(np.random.random((10, 10)))
    >>> q = np.random.random(10)
    >>> A.evaluate(q)               # Compute Aq.
    """
    def __init__(self, entries):
        """Check dimensions and set operator entries."""
        self._validate_entries(entries)

        if entries.ndim == 1:
            entries = entries.reshape((-1, 1))
        if entries.ndim != 2:
            raise ValueError("linear operator must be two-dimensional")

        _BaseNonparametricOperator.__init__(self, entries)

    def evaluate(self, q):
        return self.entries @ np.atleast_1d(q)


class QuadraticOperator(_BaseNonparametricOperator):
    """Operator for quadratic interactions of a state/input with itself
    (compact Kronecker).

    Example
    -------
    >>> import numpy as np
    >>> H = opinf.core.operators.QuadraticOperator(np.random.random((10, 100)))
    >>> q = np.random.random(10)
    >>> H.evaluate(q)               # Compute H[q ⊗ q].
    """
    def __init__(self, entries):
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

        _BaseNonparametricOperator.__init__(self, entries)

    def evaluate(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)


# class CrossQuadraticOperator(QuadraticOperator):
#     """Quadratic terms of states / inputs (full Kronecker)."""
#     def __init__(self, entries):
#         self._validate_entries(entries)
#
#         _BaseNonparametricOperator.__init__(self, entries, symbol)
#
#     def evaluate(self, q1, q2):
#         return self.entries @ np.kron(q1, q2)
#         # TODO: what about two-dimensional inputs?
#         # la.khatri_rao() will do this, but that *requires* that the
#         # inputs q1 and q2 are both two-dimensional.
#         # TODO: and what about scalar inputs? (special case of r=1 or m=1).


class CubicOperator(_BaseNonparametricOperator):
    """Cubic terms of a state/input with itself (compact Kronecker).

    Example
    -------
    >>> import numpy as np
    >>> G = opinf.core.operators.CubicOperator(np.random.random((10, 1000)))
    >>> q = np.random.random(10)
    >>> G.evaluate(q)               # Compute G[q ⊗ q ⊗ q].
    """
    def __init__(self, entries):
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

        _BaseNonparametricOperator.__init__(self, entries)

    def evaluate(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)


# Dictionary relating modelform keys to operator classes.
nonparametric_operators = {
    "c": ConstantOperator,
    "A": LinearOperator,
    "H": QuadraticOperator,
    "G": CubicOperator,
    "B": LinearOperator,
}
