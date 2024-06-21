# lstsq/_tikhonov.py
"""Operator Inference least-squares solvers with Tikhonov regularization."""

__all__ = [
    "L2Solver",
    "L2DecoupledSolver",
    "TikhonovSolver",
    "TikhonovDecoupledSolver",
]

import abc
import types
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

from .. import errors, utils
from ._base import SolverTemplate, _require_trained


# Solver classes ==============================================================
class _BaseRegularizedSolver(SolverTemplate):
    r"""Base class for solvers of regularized linear least-squares problems.

    .. math::
       \argmin_{\ohat_i}
       \|\D\ohat_i - \z_i\|_2^2
       + \sum_{i=1}^{r}\|\bfGamma_i\ohat_i\|_2^2,
       \quad i = 1, \ldots, r.

    For each :math:`i`, this is equivalent to the following stacked ordinary
    least-squares problem:

    .. math::
       \argmin{\Ohat}
       \left\|
           \left[\begin{array}{c}\D \\ \bfGamma_i\end{array}\right]\ohat_i
           - \left[\begin{array}{c}\z_i \\ \0\end{array}\right]
        \right\|_2^2.

    The exact solution is described by the normal equations:

    .. math::
        (\D\trp\D + \bfGamma_i\trp\bfGamma_i)\ohat_i = \D\trp\z_i.
    """

    # Properties: regularization ----------------------------------------------
    @abc.abstractmethod
    def regularizer(self):  # pragma: no cover
        """Regularization scalar, matrix, or list of these."""
        raise NotImplementedError

    # Main methods ------------------------------------------------------------
    def fit(self, data_matrix: np.ndarray, lhs_matrix: np.ndarray):
        r"""Verify dimensions and save the data matrices.

        Parameters
        ----------
        data_matrix : (k, d) ndarray
            Data matrix :math:`\D`.
        lhs_matrix : (r, k) or (k,) ndarray
            "Left-hand side" data matrix :math:`\Z` (not its transpose!).
            If one-dimensional, assume :math:`r = 1`.
        """
        SolverTemplate.fit(self, data_matrix, lhs_matrix)
        if self.k < self.d:
            warnings.warn(
                "non-regularized least-squares system is underdetermined",
                errors.OpInfWarning,
            )
        return self

    # Post-processing ---------------------------------------------------------
    @abc.abstractmethod
    def regcond(self) -> float:  # pragma: no cover
        r"""Compute the :math:`2`-norm condition number of the regularized
        data matrix :math:`[~\D\trp~~\bfGamma\trp~]\trp.`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def regresidual(self, Ohat: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Compute the residual of the regularized regression problem."""
        raise NotImplementedError

    # Persistence -------------------------------------------------------------
    def _save(self, savefile, overwrite=False, extras=tuple()):
        """Serialize the solver, saving it in HDF5 format.
        The model can be recovered with the :meth:`_load()` class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension ``.h5`` (HDF5).
        overwrite : bool
            If ``True`` and the specified ``savefile`` already exists,
            overwrite the file. If ``False`` (default) and the specified
            ``savefile`` already exists, raise an error.
        extras : list
            Names of additional attributes to save.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            reg = self.regularizer
            if self.__class__ is L2Solver:
                reg = [reg]
            hf.create_dataset("regularizer", data=reg)

            if isinstance(self, TikhonovSolver):
                meta = hf.create_dataset("meta", shape=(0,))
                meta.attrs["method"] = self.method

            self._save_dict(hf, "options")

            if self.data_matrix is not None:
                hf.create_dataset("data_matrix", data=self.data_matrix)
                hf.create_dataset("lhs_matrix", data=self.lhs_matrix)
                for attr in extras:
                    hf.create_dataset(attr, data=getattr(self, attr))

    @classmethod
    def _load(cls, loadfile: str, extras=tuple()):
        """Load a serialized solver from an HDF5 file, created previously from
        the :meth:`save()` method.

        Parameters
        ----------
        loadfile : str
            Path to the file where the model was stored via :meth:`save()`.
        extras : list
            Names of additional attributes to load.

        Returns
        -------
        solver : _BaseRegularizedSolver
            Loaded solver.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:

            reg = hf["regularizer"][:]
            if cls is L2Solver:
                reg = reg[0]

            options = cls._load_dict(hf, "options")
            kwargs = dict(
                regularizer=reg,
                lapack_driver=options["lapack_driver"],
            )

            if issubclass(cls, TikhonovSolver):
                if "cond" in options:
                    kwargs["cond"] = options["cond"]
                kwargs["method"] = hf["meta"].attrs["method"]

            solver = cls(**kwargs)

            if "data_matrix" in hf:
                D = hf["data_matrix"][:]
                Z = hf["lhs_matrix"][:]
                _BaseRegularizedSolver.fit(solver, D, Z)
                for attr in extras:
                    setattr(solver, attr, hf[attr][:])

        return solver


class L2Solver(_BaseRegularizedSolver):
    r"""Solve the Frobenius-norm ordinary least-squares problem with
    :math:`L_2` regularization.

    That is, solve

    .. math::
        \argmin_{\Ohat}\|\D\Ohat\trp - \Z\trp\|_F^2 + \|\lambda\Ohat\trp\|_F^2

    for some specified :math:`\lambda \ge 0`.

    The exact solution is described by the normal equations:

    .. math::
        (\D\trp\D + \lambda^2\I)\Ohat\trp = \D\trp\Z\trp,

    that is,

    .. math::
        \Ohat = \Z\D(\D\trp\D + \lambda^2\I)^{-\mathsf{T}}.

    Instead of solving these equations directly, the solution is calculated
    using the singular value decomposition of the data matrix :math:`\D`:
    if :math:`\D = \bfPhi\bfSigma\bfPsi\trp`, then
    :math:`\Ohat\trp = \bfPsi\bfSigma^{*}\bfPhi\trp\Z\trp` (i.e.,
    :math:`\Ohat = \Z\bfPhi\bfSigma^{*}\bfPsi\trp`), where
    :math:`\bfSigma^{*}` is a diagonal matrix with :math:`i`-th diagonal entry
    :math:`\Sigma_{i,i}^{*} = \Sigma_{i,i}/(\Sigma_{i,i}^{2} + \lambda^2).`

    Parameters
    ----------
    regularizer : float
        Scalar :math:`L_2` regularization constant.
    lapack_driver : str
        LAPACK routine for computing the singular value decomposition.
        See :func:`scipy.linalg.svd()`.
    """

    def __init__(self, regularizer, lapack_driver: str = "gesdd"):
        """Store the regularizer and initialize attributes."""
        _BaseRegularizedSolver.__init__(self)
        self.regularizer = regularizer
        self.__options = types.MappingProxyType(
            dict(full_matrices=False, lapack_driver=lapack_driver)
        )

    # Properties --------------------------------------------------------------
    @property
    def regularizer(self):
        r"""Scalar :math:`L_2` regularization constant
        :math:`\lambda > 0.`
        """
        return self.__reg

    @regularizer.setter
    def regularizer(self, reg):
        """Set the regularization constant."""
        if not np.isscalar(reg):
            raise TypeError("regularization constant must be a scalar")
        if reg < 0:
            raise ValueError("regularization constant must be nonnegative")
        self.__reg = reg

    @property
    def options(self):
        """Keyword arguments for :func:`scipy.linalg.svd()`.
        These cannot be changed after instantiation.
        """
        return self.__options

    def __str__(self):
        """String representation: dimensions + solver options."""
        start = SolverTemplate.__str__(self)
        kwargs = self._print_kwargs(self.options)
        return start + f"\n  SVD solver: scipy.linalg.svd({kwargs})"

    # Main methods ------------------------------------------------------------
    def fit(self, data_matrix: np.ndarray, lhs_matrix: np.ndarray):
        r"""Verify dimensions and compute the singular value decomposition of
        the data matrix in preparation to solve the least-squares problem.

        Parameters
        ----------
        data_matrix : (k, d) ndarray
            Data matrix :math:`\D`.
        lhs_matrix : (r, k) or (k,) ndarray
            "Left-hand side" data matrix :math:`\Z` (not its transpose!).
            If one-dimensional, assume :math:`r = 1`.
        """
        _BaseRegularizedSolver.fit(self, data_matrix, lhs_matrix)

        Phi, svals, PsiT = la.svd(self.data_matrix, **self.options)
        self._svals = svals
        self._ZPhi = self.lhs_matrix @ Phi
        self._PsiT = PsiT

        return self

    @_require_trained
    def solve(self) -> np.ndarray:
        r"""Solve the Operator Inference regression.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        svals = self._svals.reshape((-1, 1))
        svals_inv = svals / (svals**2 + self.regularizer**2)
        return (self._ZPhi * svals_inv.T) @ self._PsiT

    # Post-processing ---------------------------------------------------------
    @_require_trained
    def cond(self):
        r"""Compute the :math:`2`-norm condition number of the data matrix
        :math:`\D`.
        """
        return self._svals.max() / self._svals.min()

    @_require_trained
    def regcond(self) -> float:
        r"""Compute the :math:`2`-norm condition number of the regularized
        data matrix :math:`[~\D\trp~~\lambda\I~]\trp.`

        Returns
        -------
        cond : float
            Condition number of the regularized data matrix.
        """
        svals2 = self._svals**2 + self.regularizer**2
        return np.sqrt(svals2.max() / svals2.min())

    @_require_trained
    def regresidual(self, Ohat: np.ndarray) -> np.ndarray:
        r"""Compute the residual of the regularized regression objective for
        each row of the given operator matrix.

        Specifically, given a potential :math:`\Ohat`, compute

        .. math::
           \|\D\ohat_i - \z_i\|_2^2 + \|\lambda\ohat_i\|_2^2,
           \quad i = 1, \ldots, r,

        where :math:`\ohat_i` and :math:`\z_i` are the :math:`i`-th rows of
        :math:`\Ohat` and :math:`\Z`, respectively.

        Parameters
        ----------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat`.

        Returns
        -------
        residuals : (r,) ndarray
            :math:`2`-norm residuals for each row of the operator matrix.
        """
        residual = self.residual(Ohat)
        return residual + (self.regularizer**2 * np.sum(Ohat**2, axis=-1))

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
        return self._save(savefile, overwrite, ["_svals", "_ZPhi", "_PsiT"])

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
        return cls._load(loadfile, ["_svals", "_ZPhi", "_PsiT"])

    def copy(self):
        """Make a copy of the solver."""
        solver = self.__class__(
            regularizer=self.regularizer,
            lapack_driver=self.options["lapack_driver"],
        )
        if self.data_matrix is not None:
            SolverTemplate.fit(solver, self.data_matrix, self.lhs_matrix)
            solver._svals = self._svals
            solver._ZPhi = self._ZPhi
            solver._PsiT = self._PsiT
        return solver


class L2DecoupledSolver(L2Solver):
    r"""Solve :math:`r` independent :math:`2`-norm ordinary least-squares
    problems, each with the same data matrix but a different :math:`L_2`
    regularization.

    That is, for :math:`i = 1, \ldots, r`, construct :math:`\Ohat` by solving

    .. math::
        \argmin_{\Ohat}\|\D\ohat_i - \z_i\|_2^2 + \|\lambda_i\Ohat_i\|_2^2

    where :math:`\ohat_i` and :math:`\z_i` are the :math:`i`-th rows of
    :math:`\Ohat` and :math:`\Z`, respectively, with corresponding
    regularization constant :math:`\lambda_i > 0`.

    The exact solution for the :math:`i`-th problem is described by the normal
    equations:

    .. math::
        (\D\trp\D + \lambda_i^2\I)\ohat_i = \D\trp\z_i.

    Instead of solving these equations directly, the solution is calculated
    using the singular value decomposition of the data matrix
    (see :class:`L2Solver`).

    Parameters
    ----------
    regularizer : (r,) ndarray
        Scalar :math:`L_2` regularization constants, one for each row
        of the operator matrix.
    lapack_driver : str
        LAPACK routine for computing the singular value decomposition.
        See :func:`scipy.linalg.svd()`.
    """

    # Properties --------------------------------------------------------------
    def _check_regularizer_shape(self):
        if (shape1 := self.regularizer.shape) != (shape2 := (self.r,)):
            raise errors.DimensionalityError(
                f"regularizer.shape = {shape1} != {shape2} = (r,)"
            )

    @property
    def regularizer(self):
        r"""Scalar :math:`L_2` regularization constants, one for each row
        of the  operator matrix :math:`\Ohat`.
        """
        return self.__regs

    @regularizer.setter
    def regularizer(self, regs):
        """Set the regularization constants."""
        regs = np.array(regs)
        if regs.ndim != 1:
            raise ValueError("regularizer must be one-dimensional")
        if np.any(regs < 0):
            raise ValueError("regularization constants must be nonnegative")
        self.__regs = regs
        if self.r is not None:
            self._check_regularizer_shape()

    # Main methods ------------------------------------------------------------
    def fit(self, data_matrix: np.ndarray, lhs_matrix: np.ndarray):
        r"""Verify dimensions and compute the singular value decomposition of
        the data matrix in preparation to solve the least-squares problem.

        Parameters
        ----------
        data_matrix : (k, d) ndarray
            Data matrix :math:`\D`.
        lhs_matrix : (r, k) or (k,) ndarray
            "Left-hand side" data matrix :math:`\Z` (not its transpose!).
            If one-dimensional, assume :math:`r = 1`.
        """
        L2Solver.fit(self, data_matrix, lhs_matrix)
        self._check_regularizer_shape()
        return self

    # Post-processing ---------------------------------------------------------
    @_require_trained
    def regcond(self) -> float:
        r"""Compute the :math:`2`-norm condition number of each regularized
        data matrix, :math:`[~\D\trp~~\lambda_i\I~]\trp` for
        :math:`i = 1, \ldots, r`.

        Returns
        -------
        conds : (r,) ndarray
            Condition numbers of the regularized data matrices.
        """
        svals2 = self._svals**2 + self.regularizer.reshape((-1, 1)) ** 2
        return np.sqrt(svals2.max(axis=1) / svals2.min(axis=1))

    def regresidual(self, Ohat: np.ndarray) -> np.ndarray:
        r"""Compute the residual of the regularized regression objective for
        each row of the given operator matrix.

        Specifically, given a potential :math:`\Ohat`, compute

        .. math::
           \|\D\ohat_i - \z_i\|_2^2 + \|\lambda_i\ohat_i\|_2^2,
           \quad i = 1, \ldots, r,

        where :math:`\ohat_i` and :math:`\z_i` are the :math:`i`-th rows of
        :math:`\Ohat` and :math:`\Z`, respectively, and :math:`\lambda_i \ge 0`
        is the corresponding regularization constant.

        Parameters
        ----------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat`.

        Returns
        -------
        residuals : (r,) ndarray
            :math:`2`-norm residuals for each row of the operator matrix.
        """
        return L2Solver.regresidual(self, Ohat)


class TikhonovSolver(_BaseRegularizedSolver):
    r"""Solve the Frobenius-norm ordinary least-squares problem with
    Tikhonov regularization.

    That is, solve

    .. math::
       \argmin_{\Ohat}\|\D\Ohat\trp - \Z\trp\|_F^2 + \|\bfGamma\Ohat\trp\|_F^2

    for a specified symmetric-positive-definite regularization matrix
    :math:`\bfGamma \in \RR^{d \times d}`. This is equivalent to solving the
    following stacked least-squares problem:

    .. math::
       \argmin_{\Ohat}
       \left\|
           \left[\begin{array}{c}\D \\ \bfGamma\end{array}\right]\Ohat\trp
           - \left[\begin{array}{c}\Z\trp \\ \0\end{array}\right]
       \right\|_F^2.

    The exact solution is described by the normal equations:

    .. math::
       (\D\trp\D + \bfGamma\trp\bfGamma)\Ohat\trp = \D\trp\Z\trp,

    that is,

    .. math::
       \Ohat = \Z\D(\D\trp\D + \bfGamma\trp\bfGamma)^{-\mathsf{T}}.

    Parameters
    ----------
    regularizer : (d, d) or (d,) ndarray
        Symmetric semi-positive-definite regularization matrix :math:`\bfGamma`
        or, if ``regularizer`` is a one-dimensional array, the diagonal entries
        of :math:`\bfGamma`. Here, ``d`` is the number of columns in the data
        matrix.
    method : str
        Strategy for solving the regularized least-squares problem.
        **Options**:

        * ``"lstsq"``: solve the stacked least-squares problem via
          :func:`scipy.linalg.lstsq()`; by default, this computes and uses the
          singular value decomposition of the stacked data matrix
          :math:`[~D\trp~~\bfGamma\trp~]\trp`.
        * ``"normal"``: directly solve the normal equations
          :math:`(\D\trp\D + \bfGamma\trp\bfGamma) \Ohat\trp = \D\trp\Z\trp`
          via :func:`scipy.linalg.solve()`.
    cond : float or None
        Cutoff for 'small' singular values of the data matrix,
        see :func:`scipy.linalg.lstsq()`. Ignored if ``method = "normal"``.
    lapack_driver : str or None
        Which LAPACK driver is used to solve the least-squares problem,
        see :func:`scipy.linalg.lstsq()`. Ignored if ``method = "normal"``.
    """

    def __init__(
        self,
        regularizer,
        method: str = "lstsq",
        cond: float = None,
        lapack_driver: str = None,
    ):
        """Store the regularizer and initialize attributes."""
        _BaseRegularizedSolver.__init__(self)
        self.regularizer = regularizer
        self.method = method
        self.__options = dict(cond=cond, lapack_driver=lapack_driver)

    # Properties --------------------------------------------------------------
    @property
    def options(self):
        """Keyword arguments for :func:`scipy.linalg.lstsq()`."""
        return self.__options

    def __str__(self):
        """String representation: dimensions + solver options."""
        s = SolverTemplate.__str__(self)
        if self.method == "lstsq":
            kwargs = self._print_kwargs(self.options)
            return s + f"\n  solver ('lstsq'): scipy.linalg.lstsq({kwargs})"
        return s + "\n  solver ('normal'): scipy.linalg.solve(assume_a='pos')"

    def _check_regularizer_shape(self):
        if (shape1 := self.regularizer.shape) != (shape2 := (self.d, self.d)):
            raise errors.DimensionalityError(
                f"regularizer.shape = {shape1} != {shape2} = (d, d)"
            )

    @property
    def regularizer(self):
        r"""Symmetric semi-positive-definite :math:`d \times d` regularization
        matrix :math:`\bfGamma`.
        """
        return self.__reg

    @regularizer.setter
    def regularizer(self, G):
        """Set the regularization matrix."""
        if sparse.issparse(G):
            G = G.toarray()
        elif not isinstance(G, np.ndarray):
            G = np.array(G)

        if G.ndim == 1:
            if np.any(G < 0):
                raise ValueError(
                    "diagonal regularizer must be positive semi-definite"
                )
            G = np.diag(G)

        self.__reg = G

        if self.d is not None:
            self._check_regularizer_shape()

    @property
    def method(self):
        """Strategy for solving the regularized least-squares problem, either
        ``"lstsq"`` (default) or ``"normal"``.
        """
        return self.__method

    @method.setter
    def method(self, method):
        """Set the method and precompute stuff as needed."""
        if method not in ("lstsq", "normal"):
            raise ValueError("method must be 'lstsq' or 'normal'")
        self.__method = method

    # Main routines -----------------------------------------------------------
    def fit(self, data_matrix: np.ndarray, lhs_matrix: np.ndarray):
        r"""Verify dimensions and precompute quantities in preparation to
        solve the least-squares problem.

        Parameters
        ----------
        data_matrix : (k, d) ndarray
            Data matrix :math:`\D`.
        lhs_matrix : (r, k) or (k,) ndarray
            "Left-hand side" data matrix :math:`\Z` (not its transpose!).
        """
        _BaseRegularizedSolver.fit(self, data_matrix, lhs_matrix)
        self._check_regularizer_shape()
        D, Z = self.data_matrix, self.lhs_matrix

        # Pad lhs matrix for "svd" solve.
        self._ZtPad = np.vstack((Z.T, np.zeros((self.d, self.r))))

        # Precompute normal equations terms for "normal" solve.
        self._DtD = D.T @ D
        self._DtZt = D.T @ Z.T

        return self

    @_require_trained
    def solve(self) -> np.ndarray:
        r"""Solve the Operator Inference regression.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        if self.method == "lstsq":
            DPad = np.vstack((self.data_matrix, self.regularizer))
            Ohat = la.lstsq(DPad, self._ZtPad, **self.options)[0].T
        elif self.method == "normal":
            regD = self._DtD + (self.regularizer.T @ self.regularizer)
            Ohat = la.solve(regD, self._DtZt, assume_a="pos").T
        return Ohat

    # Post-processing ---------------------------------------------------------
    @_require_trained
    def regcond(self) -> float:
        r"""Compute the :math:`2`-norm condition number of the regularized
        data matrix :math:`[~\D\trp~~\bfGamma\trp~]\trp.`

        Returns
        -------
        cond : float
            Condition number of the regularized data matrix.
        """
        return np.linalg.cond(np.vstack((self.data_matrix, self.regularizer)))

    @_require_trained
    def regresidual(self, Ohat: np.ndarray) -> np.ndarray:
        r"""Compute the residual of the regularized regression objective for
        each row of the given operator matrix.

        Specifically, given a potential :math:`\Ohat`, compute

        .. math::
           \|\D\ohat_i - \z_i\|_2^2 + \|\bfGamma\ohat_i\|_2^2,
           \quad i = 1, \ldots, r,

        where :math:`\ohat_i` and :math:`\z_i` are the :math:`i`-th rows of
        :math:`\Ohat` and :math:`\Z`, respectively.

        Parameters
        ----------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat`.

        Returns
        -------
        residuals : (r,) ndarray
            :math:`2`-norm residuals for each row of the operator matrix.
        """
        residual = self.residual(Ohat)
        return residual + np.sum((self.regularizer @ Ohat.T) ** 2, axis=0)

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
        return self._save(savefile, overwrite, ["_ZtPad", "_DtD", "_DtZt"])

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
        return cls._load(loadfile, ["_ZtPad", "_DtD", "_DtZt"])

    def copy(self):
        """Make a copy of the solver."""
        solver = self.__class__(
            regularizer=self.regularizer,
            method=self.method,
            cond=self.options["cond"],
            lapack_driver=self.options["lapack_driver"],
        )
        if self.data_matrix is not None:
            SolverTemplate.fit(solver, self.data_matrix, self.lhs_matrix)
            solver._ZtPad = self._ZtPad
            solver._DtD = self._DtD
            solver._DtZt = self._DtZt
        return solver


class TikhonovDecoupledSolver(TikhonovSolver):
    r"""Solve :math:`r` independent :math:`2`-norm ordinary least-squares
    problems, each with the same data matrix but a different Tikhonov
    regularization.

    That is, for :math:`i = 1, \ldots, r`, construct :math:`\Ohat` by solving

    .. math::
        \argmin_\Ohat\|\D\Ohat\trp - \Z\trp\|_F^2 + \|\bfGamma\Ohat\trp\|_F^2

    where :math:`\ohat_i` and :math:`\z_i` are the :math:`i`-th rows of
    :math:`\Ohat` and :math:`\Z`, respectively, with corresponding
    symmetric-positive-definite regularization matrix :math:`\bfGamma_i`.

    This is equivalent to solving the following stacked least-squares problems:

    .. math::
       \argmin{\Ohat}
       \left\|
           \left[\begin{array}{c}\D \\ \bfGamma_i\end{array}\right]\Ohat\trp
           - \left[\begin{array}{c}\Z\trp \\ \0\end{array}\right]
       \right\|_F^2,
       \quad i = 1, \ldots, r.

    The exact solution of the :math:`i`-th problem is described by the normal
    equations:

    .. math::
       (\D\trp\D + \bfGamma_i\trp\bfGamma_i)\ohat_i = \D\trp\z_i.

    Parameters
    ----------
    regularizer : list of r (d, d) or (d,) ndarrays
        Symmetric semi-positive-definite regularization matrices
        :math:`\bfGamma_1,\ldots,\bfGamma_r`.
        If the ``i``th entry of ``regularizer`` is a one-dimensional array,
        it is intepreted as the diagonal entries of :math:`\bfGamma_i`. Here,
        `d` is the number of columns in the data matrix.
    method : str
        Strategy for solving the regularized least-squares problem.
        **Options**:

        * ``"lstsq"``: solve the stacked least-squares problem via
          :func:`scipy.linalg.lstsq()`; by default, this computes and uses the
          singular value decomposition of the stacked data matrix
          :math:`[~D\trp~~\bfGamma\trp~]\trp`.
        * ``"normal"``: directly solve the normal equations
          :math:`(\D\trp\D + \bfGamma\trp\bfGamma) \Ohat\trp = \D\trp\Z\trp`
          via :func:`scipy.linalg.solve()`.
    cond : float or None
        Cutoff for 'small' singular values of the data matrix,
        see :func:`scipy.linalg.lstsq()`. Ignored if ``method = "normal"``.
    lapack_driver : str or None
        Which LAPACK driver is used to solve the least-squares problem,
        see :func:`scipy.linalg.lstsq()`. Ignored if ``method = "normal"``.
    """

    # Properties --------------------------------------------------------------
    def _check_regularizer_shape(self):
        """Check that the regularizer has the correct shape."""
        if len(self.regularizer) != self.r:
            raise ValueError("len(regularizer) != r")
        for i, G in enumerate(self.regularizer):
            if (shape1 := G.shape) != (shape2 := (self.d, self.d)):
                raise ValueError(
                    f"regularizer[{i}].shape = {shape1} != {shape2} = (d, d)"
                )

    @property
    def regularizer(self):
        r"""Symmetric semi-positive-definite regularization matrices
        :math:`\bfGamma_1,\ldots,\bfGamma_r`, one for each row of the
        operator matrix.
        """
        return self.__regs

    @regularizer.setter
    def regularizer(self, Gs):
        """Set the regularization matrices."""
        regs = []
        for G in Gs:
            if sparse.issparse(G):
                G = G.toarray()
            elif not isinstance(G, np.ndarray):
                G = np.array(G)
            if G.ndim == 1:
                if np.any(G < 0):
                    raise ValueError(
                        "diagonal regularizer must be positive semi-definite"
                    )
                G = np.diag(G)
            regs.append(G)

        self.__regs = regs
        if self.d is not None:
            self._check_regularizer_shape()

    # Main methods ------------------------------------------------------------
    @_require_trained
    def solve(self) -> np.ndarray:
        r"""Solve the Operator Inference regression.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        Ohat = np.empty((self.r, self.d))

        # Solve each independent regression problem (sequentially for now).
        for i, Gamma in enumerate(self.regularizer):
            if self.method == "lstsq":
                Dpad = np.vstack((self.data_matrix, Gamma))
                Ohat[i] = la.lstsq(Dpad, self._ZtPad[:, i])[0]
            elif self.method == "normal":
                regD = self._DtD + Gamma.T @ Gamma
                Ohat[i] = la.solve(regD, self._DtZt[:, i], assume_a="pos")
        return Ohat

    # Post-processing ---------------------------------------------------------
    @_require_trained
    def regcond(self) -> float:
        r"""Compute the :math:`2`-norm condition number of each regularized
        data matrix :math:`[~\D\trp~~\bfGamma_i\trp~]\trp,~~i = 1, \ldots, r`.

        Returns
        -------
        conds : (r,) ndarray
            Condition numbers for the regularized data matrices.
        """
        return np.array(
            [
                np.linalg.cond(np.vstack((self.data_matrix, G)))
                for G in self.regularizer
            ]
        )

    @_require_trained
    def regresidual(self, Ohat: np.ndarray) -> np.ndarray:
        r"""Compute the residual of the regularized regression objective for
        each row of the given operator matrix.

        Specifically, given a potential :math:`\Ohat`, compute

        .. math::
           \|\D\ohat_i - \z_i\|_2^2 + \|\bfGamma_i\ohat_i\|_2^2,
           \quad i = 1, \ldots, r,

        where :math:`\ohat_i` and :math:`\z_i` are the :math:`i`-th rows of
        :math:`\Ohat` and :math:`\Z`, respectively, and :math:`\bfGamma_i` is
        the corresponding symmetric-positive-definite regularization matrix.

        Parameters
        ----------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat`.

        Returns
        -------
        residuals : (r,) ndarray
            :math:`2`-norm residuals for each row of the operator matrix.
        """
        residual = self.residual(Ohat)
        rg = [np.sum((G @ oi) ** 2) for G, oi in zip(self.regularizer, Ohat)]
        return residual + np.array(rg)
