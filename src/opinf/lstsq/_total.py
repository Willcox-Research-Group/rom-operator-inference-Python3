# lstsq/_total.py
"""Operator Inference total least-squares solver."""

__all__ = [
    "TotalLeastSquaresSolver",
]

import numpy as np
import scipy.linalg as la

from ._base import _BaseSolver


class TotalLeastSquaresSolver(_BaseSolver):
    """Solve the total least-squares problem without any
    regularization, i.e.,

        argmin_{X, G, H} ||[G H]||_F such that (A+G)X = B+H

    The solution is calculated using standard computations
    (see Wikipedia for example).
    """
    _LSTSQ_LABEL = r"argmin_{X, G, H} ||[G H]||_F | (A+G)X = B+H"

    def predict(self):
        """Solve the total least-squares problem."""
        # A augmented with B
        Vh = la.svd(np.hstack((self.A, self.B)), full_matrices=False)[2]
        n = self.A.shape[1]
        d = self.B.shape[1]
        V = Vh.T
        # This line could throw a la.LinAlgError exception
        inv_VBB = la.inv(V[n:, -d:])
        return -V[:n, n:] @ inv_VBB
