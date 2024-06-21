# lstsq/_total.py
"""Operator Inference truncated singular value decomposition solver."""

__all__ = [
    "TruncatedSVDSolver",
]

import types
import warnings
import numpy as np
import scipy.linalg as la

from .. import errors, utils
from ._base import SolverTemplate, _require_trained


class TruncatedSVDSolver(SolverTemplate):
    r"""Solve the Frobenius-norm ordinary least-squares problem with a
    low-rank best approximation of the data matrix.

    That is, solve

    .. math::
       \argmin_{\Ohat}\|\tilde{\D}\Ohat\trp - \Z\trp\|_{F}^{2}

    where :math:`\tilde{\D}` is the best rank-:math:`d'` approximation of
    :math:`\D` for a given :math:`d' \le \operatorname{rank}(\D)`, i.e.,

    .. math ::
       \tilde{\D}
       = \argmin_{\D' \in \RR^{k \times d}}\|\D' - \D\|_{F}
       \quad\textrm{such that}\quad
       \operatorname{rank}(\D') = d'.

    If :math:`\D = \bfPhi\bfSigma\bfPsi\trp` is the singular value
    decomposition of :math:\D`, then defining

    .. math::
       \bfPhi' = \bfPhi_{:d', :}
       \quad
       \bfSigma' = \bfSigma_{:d', :d'}
       \quad
       \bfPsi' = \bfPsi_{:d', :},

    the solution to this problem is given by
    :math:`\Ohat\trp = \bfPsi'(\bfSigma')^{-1}(\bfPhi')\trp\Z\trp` (i.e.,
    :math:`\Ohat = \Z\bfPhi'(\bfSigma')^{-1}(\bfPsi')\trp`).

    Parameters
    ----------
    num_svdmodes : int or None
        Number of singular value decomposition modes to retain.
        If ``None``, use all available modes (no truncation).
        If a negative integer, use ``k - abs(num_svdmodes)`` modes where
        ``k`` is the number of available modes.
    lapack_driver : str
        LAPACK routine for computing the singular value decomposition.
        See :func:`scipy.linalg.svd()`.
    """

    def __init__(self, num_svdmodes: int, lapack_driver: str = "gesdd"):
        """Set the number of SVD columns to keep."""
        SolverTemplate.__init__(self)
        self.num_svdmodes = num_svdmodes
        self.__options = types.MappingProxyType(
            dict(full_matrices=False, lapack_driver=lapack_driver)
        )

    # Properties --------------------------------------------------------------
    @property
    def num_svdmodes(self) -> int:
        """Number of singular value decomposition modes to retain."""
        return self.__nmodes

    @num_svdmodes.setter
    def num_svdmodes(self, nmodes: int):
        """Set the number of columns."""
        if nmodes is None:
            self.__nmodes = self.max_modes
            return

        if not isinstance(nmodes, int):
            raise TypeError("num_svdmodes must be an integer")

        if self.data_matrix is not None:
            d = self.max_modes
            if nmodes <= 0:
                nmodes = d + nmodes
            if nmodes > d:
                raise ValueError(f"only {d} SVD modes available")
        self.__nmodes = nmodes

    @property
    def options(self):
        """Keyword arguments for :func:`scipy.linalg.svd()`.
        These cannot be changed after instantiation.
        """
        return self.__options

    @property
    def max_modes(self):
        """Maximum number of singular value decomposition modes available."""
        return None if self.data_matrix is None else self._ZPhi.shape[1]

    def __str__(self):
        """String representation: dimensions + solver options."""
        start = SolverTemplate.__str__(self)
        kwargs = self._print_kwargs(self.options)
        out = [f"SVD solver: scipy.linalg.svd({kwargs})"]
        if self.data_matrix is not None:
            out.append(
                f"using {self.num_svdmodes} of {self.max_modes} SVD modes"
            )
            lines = start.split("\n")
            lines.insert(
                3,
                f"    Truncation condition number: {self.tcond():.4e}",
            )
            start = "\n".join(lines)
        return f"{start}\n  " + "\n  ".join(out)

    # Main methods ------------------------------------------------------------
    def fit(self, data_matrix: np.ndarray, lhs_matrix: np.ndarray):
        """Save data matrices and prepare to solve the regression problem."""
        SolverTemplate.fit(self, data_matrix, lhs_matrix)

        Phi, svals, PsiT = la.svd(self.data_matrix, **self.options)
        self._svals = svals
        self._ZPhi = self.lhs_matrix @ Phi
        self._SinvPsiT = PsiT / svals.reshape((-1, 1))

        # Reset num_svdmodes.
        d = self._ZPhi.shape[1]
        if (n := self.num_svdmodes) is None:
            n = d
        if n <= 0:
            n = d + n
        if n > d:
            warnings.warn(
                f"only {d} SVD modes available, "
                f"setting num_svdmodes={d} (was {n})",
                errors.OpInfWarning,
            )
            n = d
        self.__nmodes = n

        return self

    @_require_trained
    def solve(self) -> np.ndarray:
        r"""Solve the Operator Inference regression.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        n = self.num_svdmodes
        Ohat = self._ZPhi[:, :n] @ self._SinvPsiT[:n, :]
        return Ohat

    # Post-processing ---------------------------------------------------------
    @_require_trained
    def tcond(self) -> float:
        r"""Compute the :math:`2`-norm condition number of :math:`\tilde{\D}`,
        the low-rank approximation to :math:`\D` from the truncated SVD.
        """
        return self._svals[0] / self._svals[self.num_svdmodes - 1]

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
            hf.create_dataset("num_svdmodes", data=[self.num_svdmodes])
            if self.data_matrix is not None:
                hf.create_dataset("data_matrix", data=self.data_matrix)
                hf.create_dataset("lhs_matrix", data=self.lhs_matrix)
                for attr in ("_svals", "_ZPhi", "_SinvPsiT"):
                    hf.create_dataset(attr, data=getattr(self, attr))

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
        solver : TruncatedSVDSolver
            Loaded solver.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            options = cls._load_dict(hf, "options")
            num_svdmodes = int(hf["num_svdmodes"][0])
            solver = cls(num_svdmodes, lapack_driver=options["lapack_driver"])
            if "data_matrix" in hf:
                D = hf["data_matrix"][:]
                Z = hf["lhs_matrix"][:]
                SolverTemplate.fit(solver, D, Z)
                for attr in ("_svals", "_ZPhi", "_SinvPsiT"):
                    setattr(solver, attr, hf[attr][:])
        return solver

    def copy(self):
        """Make a copy of the solver."""
        solver = self.__class__(
            num_svdmodes=self.num_svdmodes,
            lapack_driver=self.options["lapack_driver"],
        )
        if self.data_matrix is not None:
            SolverTemplate.fit(solver, self.data_matrix, self.lhs_matrix)
            for attr in ("_svals", "_ZPhi", "_SinvPsiT"):
                setattr(solver, attr, getattr(self, attr))
        return solver
