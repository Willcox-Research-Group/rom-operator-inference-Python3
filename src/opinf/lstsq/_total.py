# lstsq/_total.py
"""Operator Inference total least-squares solver."""

__all__ = [
    "TotalLeastSquaresSolver",
]

import types
import numpy as np
import scipy.linalg as la

from ._base import SolverTemplate


class TotalLeastSquaresSolver(SolverTemplate):
    r"""Solve the total least-squares problem without any
    regularization, i.e.,

    .. math::
       \argmin_{\Ohat, \bfEpsilon_\D, \bfEpsilon_\Z}
       \|[~\bfEpsilon_\D~~\bfEpsilon_{\Z\trp}~]\|_{F}
       \quad\text{such that}\quad
       (\D + \bfEpsilon_\D)\Ohat\trp = \Z\trp + \bfEpsilon_{\Z\trp}

    The solution is computed via the singular value decomposition of the
    augmented
    (see Wikipedia for example).
    """

    def __init__(self, lapack_driver="gesdd"):
        SolverTemplate.__init__(self)
        self.__options = types.MappingProxyType(
            dict(full_matrices=False, lapack_driver=lapack_driver)
        )

    @property
    def options(self):
        """Keyword arguments for ``scipy.linalg.svd()``."""
        return self.__options

    # Main methods ------------------------------------------------------------
    def fit(self, data_matrix, lhs_matrix):
        r"""Verify dimensions and compute the singular value decomposition of
        the data matrix in preparation to solve the least-squares problem.

        Parameters
        ----------
        data_matrix : (k, d) ndarray
            Data matrix :math:`\D`.
        lhs_matrix : (r, k) ndarray
            "Left-hand side" data matrix :math:`\Z` (not its transpose!).
            If one-dimensional, assume :math:`r = 1`.
        """
        SolverTemplate.fit(self, data_matrix, lhs_matrix)

        # Compute the SVD of the concatenation [D Z^T].
        D_ZT = np.hstack((self.data_matrix, self.lhs_matrix.T))
        Psi = la.svd(D_ZT, **self.options)[2].T

        # Extract the relevant blocks of the right singular vector matrix.
        self._Psi_ZZt = Psi[self.d :, -self.r :].T
        self._Psi_DZt = Psi[: self.d, self.d :].T
        return self

    def predict(self):
        r"""Solve the Operator Inference regression.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        return -la.solve(self._Psi_ZZt, self._Psi_DZt)
