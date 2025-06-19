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
from ..operators import _utils as oputils
from ._base import SolverTemplate, _require_trained


def _symmetrize(A):
    return (A.T + A) / 2


def _check_sigmas(sigmas):
    if np.any(sigmas < np.finfo(np.float64).eps):
        raise RuntimeError("zero residual --> posterior is deterministic")
    return sigmas


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
    def regularizer(self):
        """Regularization scalar, matrix, or list of these."""
        raise NotImplementedError  # pragma: no cover

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

    @abc.abstractmethod
    def posterior(self):
        r"""Solve the Bayesian operator inference regression, constructing the
        means and inverse covariances of probability distributions for the
        rows of an operator matrix posterior.

        See :cite:`guo2022bayesopinf` for details.

        Returns
        -------
        means : list of r (d,) ndarrays
            Mean vectors.
        precisions : list of r (d, d) ndarrays
            Inverse covariance matrices.
        """
        raise NotImplementedError  # pragma: no cover

    # Post-processing ---------------------------------------------------------
    @abc.abstractmethod
    def regcond(self) -> float:
        r"""Compute the :math:`2`-norm condition number of the regularized
        data matrix :math:`[~\D\trp~~\bfGamma\trp~]\trp.`
        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def regresidual(self, Ohat: np.ndarray) -> np.ndarray:
        """Compute the residual of the regularized regression problem."""
        raise NotImplementedError  # pragma: no cover

    # Persistence -------------------------------------------------------------
    def reset(self) -> None:
        """Reset the solver by deleting data matrices and the regularizer."""
        super().reset()
        self.regularizer = None

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
            if (reg := self.regularizer) is not None:
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

            reg = None
            if "regularizer" in hf:
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
        Scalar :math:`L_2` regularization constant :math:`\lambda`.
    lapack_driver : str
        LAPACK routine for computing the singular value decomposition.
        See :func:`scipy.linalg.svd()`.
    """

    def __init__(self, regularizer=None, lapack_driver: str = "gesdd"):
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
        if reg is not None:
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
        kwargs = self._print_kwargs(self.options)
        if self.regularizer is not None:
            if np.isscalar(self.regularizer):
                regstr = f"{self.regularizer:.4e}"
            else:
                regstr = f"{self.regularizer.shape}"
        else:
            regstr = "None"
        return "\n  ".join(
            [
                SolverTemplate.__str__(self),
                f"regularizer:     {regstr}",
                f"SVD solver:      scipy.linalg.svd({kwargs})",
            ]
        )

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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
        svals = self._svals.reshape((-1, 1))
        svals_inv = svals / (svals**2 + self.regularizer**2)
        return (self._ZPhi * svals_inv.T) @ self._PsiT

    def posterior(self):
        r"""Solve the Bayesian operator inference regression, constructing the
        means and inverse covariances of probability distributions for the
        rows of an operator matrix posterior.

        In this method, the :attr:`regularizer`, denoted :math:`\lambda`, is
        interpreted as a prior variance for the operator matrix distribution.
        The :math:`i`-th row of the operator matrix follows a multivariate
        normal distribution with the following mean and covariance.

        .. math::
           \bfmu_i &= \argmin_{\bfxi}\left\{
               \|\D\bfxi - \z_i\|_2^2 + \lambda^2\|\bfxi\|_2^2
           \right\},
           \\
           \bfSigma_i &= \sigma_i^2 \left(
               \D\trp\D + \lambda^2\I
           \right)^{-1},
           \\
           \sigma_i^2 &= \frac{1}{k}\left(
               \|\D\bfmu_i - \z_i\|_2^2 + \lambda^2\|\bfmu_i\|_2^2
           \right),

        where :math:`\z_i\in\RR^k` is the :math:`i`-th row of :math:`\Z`.
        See :cite:`guo2022bayesopinf` for details.

        Returns
        -------
        means : list of r (d,) ndarrays
            Mean vectors.
        precisions : list of r (d, d) ndarrays
            Inverse covariance matrices.

        Raises
        ------
        RuntimeError
            If any solver residual :math:`\sigma_i^2` is zero (a perfect
            regression), meaning the resulting covariance should be zero.
            In this case, do a deterministic regression with :meth:`solve()`,
            not a Bayesian regression.
        """
        Ohat = self.solve()
        DTD = _symmetrize(self.data_matrix.T @ self.data_matrix)
        invcov_unscaled = DTD + (self.regularizer**2 * np.eye(self.d))
        sigmas = _check_sigmas(self.regresidual(Ohat)) / self.k
        return Ohat, [invcov_unscaled / sig for sig in sigmas]

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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
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
        \argmin_{\Ohat}\|\D\ohat_i - \z_i\|_2^2 + \|\lambda_i\ohat_i\|_2^2

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
        if regs is not None:
            regs = np.array(regs)
            if regs.ndim != 1:
                raise ValueError("regularizer must be one-dimensional")
            if np.any(regs < 0):
                raise ValueError(
                    "regularization constants must be nonnegative"
                )
        self.__regs = regs
        if self.r is not None and regs is not None:
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
        if self.regularizer is not None:
            self._check_regularizer_shape()
        return self

    def posterior(self):
        r"""Solve the Bayesian operator inference regression, constructing the
        means and inverse covariances of probability distributions for the
        rows of an operator matrix posterior.

        In this method, the :math:`i`-th entry of :attr:`regularizer`, denoted
        :math:`\lambda_i`, is interpreted as a prior variance for the
        :math:`i`-th row of the operator matrix, which follows a multivariate
        normal distribution with the following mean and covariance.

        .. math::
           \bfmu_i &= \argmin_{\bfxi}\left\{
               \|\D\bfxi - \z_i\|_2^2 + \lambda_i^2\|\bfxi\|_2^2
           \right\},
           \\
           \bfSigma_i &= \sigma_i^2 \left(
               \D\trp\D + \lambda_i^2\I
           \right)^{-1},
           \\
           \sigma_i^2 &= \frac{1}{k}\left(
               \|\D\bfmu_i - \z_i\|_2^2 + \lambda_i^2\|\bfmu_i\|_2^2
           \right),

        where :math:`\z_i\in\RR^k` is the :math:`i`-th row of :math:`\Z`.
        See :cite:`guo2022bayesopinf` for details.

        Returns
        -------
        means : list of r (d,) ndarrays
            Mean vectors.
        precisions : list of r (d, d) ndarrays
            Inverse covariance matrices.

        Raises
        ------
        RuntimeError
            If any solver residual :math:`\sigma_i^2` is zero (a perfect
            regression), meaning the resulting covariance should be zero.
            In this case, do a deterministic regression with :meth:`solve()`,
            not a Bayesian regression.
        """
        Ohat = self.solve()
        DTD = _symmetrize(self.data_matrix.T @ self.data_matrix)
        Id = np.eye(self.d)
        sigmas = _check_sigmas(self.regresidual(Ohat)) / self.k
        precisions = [
            (DTD + (reg**2 * Id)) / sig
            for reg, sig in zip(self.regularizer, sigmas)
        ]
        return Ohat, precisions

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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
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
        regularizer=None,
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
        kwargs = self._print_kwargs(self.options)
        if self.regularizer is not None:
            if self.regularizer[0].ndim == 1:
                regstr = f"     {self.regularizer.shape}"
            else:
                regstr = (
                    f"     {len(self.regularizer)} "
                    f"{self.regularizer[0].shape} ndarrays"
                )
        else:
            regstr = "None"
        if self.method == "lstsq":
            kwargs = self._print_kwargs(self.options)
            spstr = f"solver ('lstsq'): scipy.linalg.lstsq({kwargs})"
        else:
            spstr = "solver ('normal'): scipy.linalg.solve(assume_a='pos')"
        return "\n  ".join(
            [SolverTemplate.__str__(self), f"regularizer: {regstr}", spstr]
        )

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
        if G is not None:
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

        if self.d is not None and G is not None:
            self._check_regularizer_shape()

    @classmethod
    def get_operator_regularizer(
        cls,
        operators: list,
        regularization_parameters: list,
        state_dimension: int,
        input_dimension: int = 0,
    ):
        r"""Construct a regularizer so that each operator is regularized
        separately.

        The regularization term for this solver is

        .. math::
           \|\bfGamma\Ohat\trp\|_F^2

        where :math:`\Ohat\in\RR^{r\times d}` is the unknown and
        :math:`\bfGamma\in\RR^{d \times d}` is a given regularization matrix.
        This method constructs :math:`\bfGamma` such that each operator
        represented in :math:`\Ohat` is regularized separately. For example, if
        :math:`\Ohat = [~\chat~~\Ahat~~\Hhat~~\Bhat~]`, then :math:`\bfGamma`
        may be designed so that

        .. math::
           \|\bfGamma\Ohat\trp\|_F^2
           = \gamma_1^2\|\chat\|_F^2
           + \gamma_2^2\|\Ahat\|_F^2
           + \gamma_3^2\|\Hhat\|_F^2
           + \gamma_4^2\|\Bhat\|_F^2.

        Parameters
        ----------
        operators : list of opinf.operators objects
            Collection of operators comprising the operator matrix.
        regularization_parameters : list of floats or ndarrays
            Regularization hyperparameters for each operator, i.e.,
            ``regularization_parameters[i]`` corresponds to ``operators[i]``.
        state_dimension : int
            Dimension of the (reduced) state.
        input_dimension : int
            Dimension of the input.
            If there is no input, this should be 0 (default).

        Returns
        -------
        regularizer : (d,) ndarray
            Diagonals of the regularization matrix so that ``operators[i]`` is
            regularized with constant ``regularization_parameters[i]``.

        Warns
        -----
        OpInfWarning
            If only one operator is provided (use :class:`L2Solver` instead),
            or if the ``input_dimension`` is provided but none of the operators
            act on inputs.

        Raises
        ------
        ValueError
            If a different number of ``operators`` and '
            ``regularization_parameters`` are provided, or if the
            ``input_dimension`` is not provided but at least one of the
            operators acts on inputs.
        TypeError
            If an entry ``operators`` has an unsupported type.
        """
        if (n1 := len(operators)) != (n2 := len(regularization_parameters)):
            raise ValueError(
                f"len(operators) == {n1} != "
                f"{n2} == len(regularization_parameters)"
            )
        if n1 == 1:
            warnings.warn(
                "consider using L2Solver for models with only one operator",
                errors.OpInfWarning,
            )

        # Check if input_dimension is needed or not.
        has_inputs = [oputils.has_inputs(op) for op in operators]
        inputs_required = any(has_inputs)
        if inputs_required and input_dimension == 0:
            idx = np.argmax(has_inputs)
            raise ValueError(
                "argument 'input_dimension' required, "
                f"operators[{idx}] acts on inputs"
            )
        elif not inputs_required and input_dimension > 0:
            warnings.warn(
                "argument 'input_dimension' ignored, "
                "no operators act on inputs",
                errors.OpInfWarning,
            )

        # Get operator dimensions.
        r, m = state_dimension, input_dimension
        dims = []
        for op in operators:
            if oputils.is_nonparametric(op):
                dims.append(op.operator_dimension(r, m))
            elif oputils.is_affine(op):
                dims.append(op.operator_dimension(None, r, m))
            else:
                raise TypeError(
                    f"unsupported operator type '{type(op).__name__}'"
                )

        # Construct the regularizer.
        regularizer = np.zeros(sum(dims))
        index = 0
        for dim, reg in zip(dims, regularization_parameters):
            endex = index + dim
            regularizer[index:endex] = reg
            index = endex

        return regularizer

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
        if self.regularizer is not None:
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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
        if self.method == "lstsq":
            DPad = np.vstack((self.data_matrix, self.regularizer))
            Ohat = la.lstsq(DPad, self._ZtPad, **self.options)[0].T
        elif self.method == "normal":
            regD = self._DtD + (self.regularizer.T @ self.regularizer)
            Ohat = la.solve(regD, self._DtZt, assume_a="pos").T
        return Ohat

    def posterior(self):
        r"""Solve the Bayesian operator inference regression, constructing the
        means and inverse covariances of probability distributions for the
        rows of an operator matrix posterior.

        In this method, the :attr:`regularizer`, denoted :math:`\bfGamma`, is
        interpreted as a prior covariance for the operator matrix distribution.
        The :math:`i`-th row of the operator matrix follows a multivariate
        normal distribution with the following mean and covariance.

        .. math::
           \bfmu_i &= \argmin_{\bfxi}\left\{
               \|\D\bfxi - \z_i\|_2^2 + \|\bfGamma\bfxi\|_2^2
           \right\},
           \\
           \bfSigma_i &= \sigma_i^2 \left(
               \D\trp\D + \bfGamma\trp\bfGamma
           \right)^{-1},
           \\
           \sigma_i^2 &= \frac{1}{k}\left(
               \|\D\bfmu_i - \z_i\|_2^2 + \|\bfGamma\bfmu_i\|_2^2
           \right),

        where :math:`\z_i\in\RR^k` is the :math:`i`-th row of :math:`\Z`.
        See :cite:`guo2022bayesopinf` for details.

        Returns
        -------
        means : list of r (d,) ndarrays
            Mean vectors.
        precisions : list of r (d, d) ndarrays
            Inverse covariance matrices.

        Raises
        ------
        RuntimeError
            If any solver residual :math:`\sigma_i^2` is zero (a perfect
            regression), meaning the resulting covariance should be zero.
            In this case, do a deterministic regression with :meth:`solve()`,
            not a Bayesian regression.
        """
        Ohat = self.solve()
        DTD = _symmetrize(self.data_matrix.T @ self.data_matrix)
        GTG = _symmetrize(self.regularizer.T @ self.regularizer)
        invcov_unscaled = DTD + GTG
        sigmas = _check_sigmas(self.regresidual(Ohat)) / self.k
        return Ohat, [invcov_unscaled / sig for sig in sigmas]

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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
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
        regs = None
        if Gs is not None:
            regs = []
            for G in Gs:
                if sparse.issparse(G):
                    G = G.toarray()
                elif not isinstance(G, np.ndarray):
                    G = np.array(G)
                if G.ndim == 1:
                    if np.any(G < 0):
                        raise ValueError(
                            "diagonal regularizer must be "
                            "positive semi-definite"
                        )
                    G = np.diag(G)
                regs.append(G)

        self.__regs = regs
        if self.d is not None and Gs is not None:
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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
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

    def posterior(self):
        r"""Solve the Bayesian operator inference regression, constructing the
        means and inverse covariances of probability distributions for the
        rows of an operator matrix posterior.

        In this method, the :math:`i`-th entry of :attr:`regularizer`, denoted
        :math:`\bfGamma_i`, is interpreted as a prior covariance for the
        :math:`i`-th row of the operator matrix, which follows a multivariate
        normal distribution with the following mean and covariance.

        .. math::
           \bfmu_i &= \argmin_{\bfxi}\left\{
               \|\D\bfxi - \z_i\|_2^2 + \|\bfGamma_i\bfxi\|_2^2
           \right\},
           \\
           \bfSigma_i &= \sigma_i^2 \left(
               \D\trp\D + \bfGamma_i\trp\bfGamma_i
           \right)^{-1},
           \\
           \sigma_i^2 &= \frac{1}{k}\left(
               \|\D\bfmu_i - \z_i\|_2^2 + \|\bfGamma_i\bfmu_i\|_2^2
           \right),

        where :math:`\z_i\in\RR^k` is the :math:`i`-th row of :math:`\Z`.
        See :cite:`guo2022bayesopinf` for details.

        Returns
        -------
        means : list of r (d,) ndarrays
            Mean vectors.
        precisions : list of r (d, d) ndarrays
            Inverse covariance matrices.

        Raises
        ------
        RuntimeError
            If any solver residual :math:`\sigma_i^2` is zero (a perfect
            regression), meaning the resulting covariance should be zero.
            In this case, do a deterministic regression with :meth:`solve()`,
            not a Bayesian regression.
        """
        Ohat = self.solve()
        DTD = _symmetrize(self.data_matrix.T @ self.data_matrix)
        sigmas = _check_sigmas(self.regresidual(Ohat)) / self.k
        precisions = [
            (DTD + _symmetrize(Gamma.T @ Gamma)) / sig
            for Gamma, sig in zip(self.regularizer, sigmas)
        ]
        return Ohat, precisions

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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
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
        if self.regularizer is None:
            raise AttributeError("solver regularizer not set")
        residual = self.residual(Ohat)
        rg = [np.sum((G @ oi) ** 2) for G, oi in zip(self.regularizer, Ohat)]
        return residual + np.array(rg)
