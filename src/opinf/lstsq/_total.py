# lstsq/_total.py
"""Operator Inference total least-squares solver."""

__all__ = [
    "TotalLeastSquaresSolver",
]

import types
import numpy as np
import scipy.linalg as la

from .. import utils
from ._base import SolverTemplate, _require_trained


class TotalLeastSquaresSolver(SolverTemplate):
    r"""Solve the total least-squares problem without any
    regularization, i.e.,

    .. math::
       \argmin_{\Ohat}
       \left\|[~\Delta_\D~~\Delta_{\Z\trp}~]\right\|_{F}^{2}
       \quad\text{such that}\quad
       (\D + \Delta_\D)\Ohat\trp = \Z\trp + \Delta_{\Z\trp}.

    The solution is computed via the singular value decomposition of the
    augmented :math:`k \times (d + r)` matrix :math:`[~\D~~\Z\trp~]`: if
    :math:`\bfPhi\bfSigma\bfPsi\trp = [~\D~~\Z\trp~]`, write the
    :math:`(d + r) \times (d + r)` right singular vector matrix
    :math:`\bfPsi` as

    .. math::
       \bfPsi = \left[\begin{array}{cc}
           \bfPsi_{\D\D} & \bfPsi_{\D\Z} \\
           \bfPsi_{\Z\D} & \bfPsi_{\Z\Z}
       \end{array}\right],

    where
    :math:`\bfPsi_{\D\D}` is :math:`d \times d`,
    :math:`\bfPsi_{\D\Z}` is :math:`d \times r`,
    :math:`\bfPsi_{\Z\D}` is :math:`r \times d`, and
    :math:`\bfPsi_{\Z\Z}` is :math:`r \times r`.
    Then the optimal solution is given by
    :math:`\Ohat = -\bfPsi_{\Z\Z}^{-\mathsf{T}}\bfPsi_{\D\Z}\trp`.

    Parameters
    ----------
    lapack_driver : str
        LAPACK routine for computing the SVD. See :func:`scipy.linalg.svd()`.
    """

    def __init__(self, lapack_driver: str = "gesdd"):
        SolverTemplate.__init__(self)
        self.__options = types.MappingProxyType(
            dict(full_matrices=False, lapack_driver=lapack_driver)
        )

    # Properties --------------------------------------------------------------
    @property
    def options(self) -> dict:
        """Keyword arguments for :func:`scipy.linalg.svd()`."""
        return self.__options

    def __str__(self) -> str:
        """String representation: dimensions + solver options."""
        start = SolverTemplate.__str__(self)
        if self.data_matrix is not None:
            lines = start.split("\n")
            lines.insert(
                4,
                f"    Augmented condition number: {self._augcond:.4e}",
            )
            start = "\n".join(lines)
        kwargs = self._print_kwargs(self.options)
        return start + f"\n  SVD solver: scipy.linalg.svd({kwargs})"

    # Main methods ------------------------------------------------------------
    def fit(self, data_matrix: np.ndarray, lhs_matrix: np.ndarray):
        r"""Verify dimensions, compute the singular value decomposition of
        the data matrix, and solve the problem.

        Parameters
        ----------
        data_matrix : (k, d) ndarray
            Data matrix :math:`\D`.
        lhs_matrix : (r, k) or (k,) ndarray
            "Left-hand side" data matrix :math:`\Z` (not its transpose!).
            If one-dimensional, assume :math:`r = 1`.
        """
        SolverTemplate.fit(self, data_matrix, lhs_matrix)

        # Compute the SVD of the concatenation [D Z^T].
        D_Zt = np.hstack((self.data_matrix, self.lhs_matrix.T))
        Phi, _svals, PsiT = la.svd(D_Zt, **self.options)
        self._augcond = _svals.max() / _svals.min()

        # Extract the relevant blocks of the singular vector matrices.
        Psi_ZZt = PsiT[self.d :, self.d :]
        Psi_DZt = PsiT[self.d :, : self.d]
        Phi_Zt_Sig_ZT = Phi[:, self.d :] * _svals[self.d :]

        # Solve the problem and compute the residuals.
        self._Ohat = -la.solve(Psi_ZZt, Psi_DZt)
        Deltas = -Phi_Zt_Sig_ZT @ np.hstack((Psi_ZZt, Psi_DZt))
        self._norm_of_errors = la.norm(Deltas)

        return self

    @_require_trained
    def solve(self) -> np.ndarray:
        r"""Return the total least-squares solution to the Operator Inference
        regression.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        return self._Ohat

    # Post-processing ---------------------------------------------------------
    @property
    @_require_trained
    def augcond(self) -> float:
        r""":math:`2`-norm condition number of the augmented data matrix
        :math:`[~\D~~\Z\trp~]`.
        """
        return self._augcond

    @property
    @_require_trained
    def error(self) -> float:
        r"""Frobenius norm of the error matrices,
        :math:`\|[~\Delta_\D~~\Delta_{\Z\trp}~\|_F`.
        """
        return self._norm_of_errors

    # Persistence -------------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False):
        """Serialize the solver, saving it in HDF5 format.
        The model can be recovered with the :meth:`load()` class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension ``.h5`` (HDF5).
        overwrite : bool
            If ``True`` and the specified ``savefile`` already exists,
            overwrite the file. If ``False`` (default) and the specified
            ``savefile`` already exists, raise an error.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            self._save_dict(hf, "options")
            if self.data_matrix is not None:
                for attr in ("data_matrix", "lhs_matrix", "_Ohat"):
                    hf.create_dataset(attr, data=getattr(self, attr))
                for attr in ("_augcond", "_norm_of_errors"):
                    hf.create_dataset(attr, data=[getattr(self, attr)])

    @classmethod
    def load(cls, loadfile: str):
        """Load a serialized solver from an HDF5 file, created previously from
        the :meth:`save()` method.

        Parameters
        ----------
        loadfile : str
            Path to the file where the model was stored via :meth:`save()`.

        Returns
        -------
        solver : L2Solver
            Loaded solver.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            options = cls._load_dict(hf, "options")
            solver = cls(lapack_driver=options["lapack_driver"])
            if "data_matrix" in hf:
                D = hf["data_matrix"][:]
                Z = hf["lhs_matrix"][:]
                SolverTemplate.fit(solver, D, Z)
                solver._Ohat = hf["_Ohat"][:]
                for attr in ("_augcond", "_norm_of_errors"):
                    setattr(solver, attr, float(hf[attr][0]))
        return solver

    def copy(self):
        """Make a copy of the solver."""
        solver = self.__class__(lapack_driver=self.options["lapack_driver"])
        if self.data_matrix is not None:
            SolverTemplate.fit(solver, self.data_matrix, self.lhs_matrix)
            for attr in ("_Ohat", "_augcond", "_norm_of_errors"):
                setattr(solver, attr, getattr(self, attr))
        return solver
