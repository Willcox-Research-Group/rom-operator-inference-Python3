# operators/_nonparametric.py
"""Classes for polynomial operators with no external parameter dependencies."""

# TODO: Jacobian operations for each operator.

__all__ = [
            "ConstantOperator",
            "LinearOperator",
            "QuadraticOperator",
            # CrossQuadraticOperator",
            "CubicOperator",
            "nonparametric_operators",
          ]

import numpy as np

from .. import utils
from ._base import _BaseNonparametricOperator


class ConstantOperator(_BaseNonparametricOperator):
    r"""Constant operator.

    .. math::
        \widehat{\mathbf{c}} \in \mathbb{R}^{r}

    Examples
    --------
    >>> import numpy as np
    >>> c = opinf.operators.ConstantOperator(np.random.random(10))
    >>> c.evaluate()                # Extract c.
    """
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

    def jacobian(self, *args):
        r = self.entries.size
        return np.zeros((r, r))


class LinearOperator(_BaseNonparametricOperator):
    r"""Linear state or input operator.

    .. math::
        \widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}

        \widehat{\mathbf{u}} \mapsto \widehat{\mathbf{B}}\widehat{\mathbf{u}}

    Examples
    --------
    >>> import numpy as np
    >>> A = opinf.operators.LinearOperator(np.random.random((10, 10)))
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

    def jacobian(self, *args):
        return self.entries


class QuadraticOperator(_BaseNonparametricOperator):
    r"""Operator for quadratic interactions of a state/input with itself.

    .. math::
        \widehat{\mathbf{q}} \mapsto
        \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]


    Examples
    --------
    >>> import numpy as np
    >>> H = opinf.operators.QuadraticOperator(np.random.random((10, 100)))
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
            entries = utils.compress_quadratic(entries)
        elif r2 != r1 * (r1 + 1) // 2:
            raise ValueError("invalid dimensions for quadratic operator")
        self._mask = utils.kron2c_indices(r1)
        Ht = utils.expand_quadratic(entries).reshape([r1]*3)
        self._jac = Ht + Ht.transpose(0, 2, 1)

        _BaseNonparametricOperator.__init__(self, entries)

    def evaluate(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)

    def jacobian(self, q):
        """Compute the Jacobian H[(q ⊗ I) + (I ⊗ q)]."""
        return self._jac @ np.atleast_1d(q)


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
    r"""Operator for cubic interactions of a state/input with itself.

    .. math::
        \widehat{\mathbf{q}} \mapsto
        \widehat{\mathbf{H}}[\widehat{\mathbf{q}}
        \otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]

    Examples
    --------
    >>> import numpy as np
    >>> G = opinf.operators.CubicOperator(np.random.random((10, 1000)))
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
            entries = utils.compress_cubic(entries)
        elif r2 != r1 * (r1 + 1) * (r1 + 2) // 6:
            raise ValueError("invalid dimensions for cubic operator")
        self._mask = utils.kron3c_indices(r1)
        Gt = utils.expand_cubic(entries).reshape([r1]*4)
        self._jac = Gt + Gt.transpose(0, 2, 1, 3) + Gt.transpose(0, 3, 1, 2)

        _BaseNonparametricOperator.__init__(self, entries)

    def evaluate(self, q):
        return self.entries @ np.prod(np.atleast_1d(q)[self._mask], axis=1)

    def jacobian(self, q):
        """Compute the Jacobian G[(I ⊗ q ⊗ q) + (q ⊗ I ⊗ q) + (q ⊗ q ⊗ I)]."""
        q_ = np.atleast_1d(q)
        return (self._jac @ q_) @ q_


# Dictionary relating modelform keys to operator classes.
nonparametric_operators = {
    "c": ConstantOperator,
    "A": LinearOperator,
    "H": QuadraticOperator,
    "G": CubicOperator,
    "B": LinearOperator,
}
