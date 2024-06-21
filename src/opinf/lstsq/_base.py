# lstsq/_base.py
"""Base class for solvers for the Operator Inference regression problem."""

__all__ = [
    "lstsq_size",
    "SolverTemplate",
    "PlainSolver",
]

import os
import abc
import copy
import warnings
import numpy as np
import scipy.linalg as la

from .. import errors, utils


_require_trained = utils.requires2(
    attr="data_matrix",
    message="solver not trained, call fit()",
)


def lstsq_size(modelform, r, m=0, affines=None) -> int:
    r"""Compute the number of columns in the operator matrix :math:`\Ohat` in
    the Operator Inference least-squares problem. This is also the number of
    columns in the data matrix :math:`\D`.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.
    r : int
        The dimension of the reduced order model.
    m : int
        The dimension of the inputs of the model.
        Must be zero unless 'B' is in `modelform`.
    affines : dict(str -> list(callables))
        Functions that define the structures of the affine operators.
        Keys must match the modelform:
        * 'c': Constant term c(µ).
        * 'A': Linear state matrix A(µ).
        * 'H': Quadratic state matrix H(µ).
        * 'G': Cubic state matrix G(µ).
        * 'B': linear Input matrix B(µ).
        For example, if the constant term has the affine structure
        c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

    Returns
    -------
    ncols : int
        The number of columns in the Operator Inference least-squares problem.
    """
    if "B" in modelform and m == 0:
        raise ValueError("argument m > 0 required since 'B' in modelform")
    if "B" not in modelform and m != 0:
        raise ValueError(f"argument m={m} invalid since 'B' in modelform")

    if affines is None:
        affines = {}

    qs = [
        (
            len(affines[op])
            if (op in affines and op in modelform)
            else 1 if op in modelform else 0
        )
        for op in "cAHGB"
    ]
    rs = [1, r, r * (r + 1) // 2, r * (r + 1) * (r + 2) // 6, m]

    return sum(qq * rr for qq, rr in zip(qs, rs))


class SolverTemplate(abc.ABC):
    r"""Template for solvers for the Operator Inference regression
    :math:`\Z \approx \Ohat\D\trp` (or :math:`\D\Ohat\trp \approx \Z\trp`)
    for the operator matrix :math:`\Ohat`.

    Child classes formulate the regression, which may include regularization
    terms and/or optimization constraints. Hyperparameters should be set in
    the constructor (regularization terms, etc.).
    """

    def __init__(self):
        self.__D = None
        self.__Z = None

    # Properties: matrices ----------------------------------------------------
    @property
    def data_matrix(self) -> np.ndarray:
        r""":math:`k \times d` data matrix :math:`\D`."""
        return self.__D

    @property
    def lhs_matrix(self) -> np.ndarray:
        r""":math:`r \times k` left-hand side data :math:`\Z`."""
        return self.__Z

    # Properties: matrix dimensions -------------------------------------------
    @property
    def k(self) -> int:
        r"""Number of equations in the least-squares problem
        (number of rows of :math:`\D` and number of columns of :math:`\Z`).
        """
        return D.shape[0] if (D := self.data_matrix) is not None else None

    @property
    def d(self) -> int:
        r"""Number of unknowns in each row of the operator matrix
        (number of columns of :math:`\D` and :math:`\Ohat`).
        """
        return D.shape[1] if (D := self.data_matrix) is not None else None

    @property
    def r(self) -> int:
        r"""Number of operator matrix rows to learn
        (number of rows of :math:`\Z` and :math:`\Ohat`)
        """
        return Z.shape[0] if (Z := self.lhs_matrix) is not None else None

    # String representation ---------------------------------------------------
    def __str__(self) -> str:
        """String representation: class name + dimensions."""
        out = [self.__class__.__name__]
        if (self.data_matrix is not None) and (self.lhs_matrix is not None):
            out.append(f"  Data matrix:     {self.data_matrix.shape}")
            out.append(f"    Condition number: {self.cond():.4e}")
            out.append(f"  LHS matrix:      {self.lhs_matrix.shape}")
            out.append(f"  Operator matrix: {self.r, self.d}")
        else:
            out[0] += " (not trained)"
        return "\n".join(out)

    def __repr__(self) -> str:
        """Unique ID + string representation."""
        return utils.str2repr(self)

    @staticmethod
    def _print_kwargs(opts: dict) -> str:
        """Utility for printing routine options in __str__()."""
        return ", ".join([f"{key}={repr(val)}" for key, val in opts.items()])

    # Main methods -----------------------------------------------------------
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
        # Verify dimensions.
        if data_matrix.ndim != 2:
            raise ValueError("data_matrix must be two-dimensional")
        if lhs_matrix.ndim == 1:
            lhs_matrix = lhs_matrix.reshape((1, -1))
        if lhs_matrix.ndim != 2:
            raise ValueError("lhs_matrix must be one- or two-dimensional")
        if (k1 := lhs_matrix.shape[1]) != (k2 := data_matrix.shape[0]):
            raise errors.DimensionalityError(
                "data_matrix and lhs_matrix not aligned "
                f"(lhs_matrix.shape[-1] = {k1} != {k2} = data_matrix.shape[0])"
            )

        self.__D = data_matrix
        self.__Z = lhs_matrix

        return self

    @abc.abstractmethod
    def solve(self) -> np.ndarray:  # pragma: no cover
        r"""Solve the Operator Inference regression.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        raise NotImplementedError

    # Post-processing --------------------------------------------------------
    @_require_trained
    def cond(self) -> float:
        r"""Compute the :math:`2`-norm condition number of the data matrix
        :math:`\D`.

        Returns
        -------
        conditionnumber : float
        """
        return np.linalg.cond(self.data_matrix)

    @_require_trained
    def residual(self, Ohat: np.ndarray) -> np.ndarray:
        r"""Compute the residual of the :math:`2`-norm regression objective for
        each row of the given operator matrix.

        Specifically, given a potential :math:`\Ohat`, compute

        .. math::
           \|\D\ohat_i - \z_i\|_2^2,
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
        if self.r == 1 and Ohat.ndim == 1:
            Ohat = Ohat.reshape((1, -1))
        if Ohat.shape != (shape := (self.r, self.d)):
            raise errors.DimensionalityError(
                f"Ohat.shape = {Ohat.shape} != {shape} = (r, d)"
            )
        return np.sum(
            (self.data_matrix @ Ohat.T - self.lhs_matrix.T) ** 2,
            axis=0,
        )

    # Persistence -------------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False):  # pragma: no cover
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
        raise NotImplementedError

    @classmethod
    def load(cls, loadfile: str):  # pragma: no cover
        """Load a serialized solver from an HDF5 file, created previously from
        the :meth:`save()` method.

        Parameters
        ----------
        loadfile : str
            Path to the file where the model was stored via :meth:`save()`.

        Returns
        -------
        solver : SolverTemplate
            Loaded solver.
        """
        raise NotImplementedError

    def _save_dict(self, hf, attr="options"):
        """Helper method for serializing a dictionary in HDF5 format."""
        options = hf.create_dataset(attr, shape=(0,))
        for key, value in getattr(self, attr).items():
            options.attrs[key] = "NULL" if value is None else value

    @staticmethod
    def _load_dict(hf, attr="options"):
        """Helper method for loading a dictionary from HDF5 format. This
        function is the inverse of :meth:`_save_dict()`."""
        options = dict()
        for key in hf[attr].attrs:
            value = hf[attr].attrs[key]
            options[key] = None if value == "NULL" else value
        return options

    def copy(self):
        """Make a copy of the solver."""
        return copy.deepcopy(self)

    # Verification ------------------------------------------------------------
    def verify(self):
        """Verify the solver.

        If the solver is already trained, check :meth:`solve()`,
        :meth:`copy()`, and (if implemented) :meth:`save()` and :meth:`load()`.
        If the solver is not trained, do nothing.
        """
        if self.data_matrix is None:
            return print("cannot verify solve() before calling fit()")

        tempfile = "_solververification.h5"

        def _make_copy(solver):
            newsolver = solver.copy()
            if (s2cls := newsolver.__class__) is not (scls := self.__class__):
                raise errors.VerificationError(
                    f"{scls.__name__}.copy() returned object "
                    f"of type '{s2cls.__name__}'"
                )
            return newsolver

        def _check_equality(obj1, obj2, Ohat1, operation):
            if obj1.r != obj2.r or obj1.d != obj2.d or obj1.k != obj2.k:
                raise errors.VerificationError(
                    f"{operation} does not preserve problem dimensions"
                )
            if not np.allclose(obj2.solve(), Ohat1):
                raise errors.VerificationError(
                    f"{operation} does not preserve the result of solve()"
                )

        try:
            # Verify solve().
            Ohat = self.solve()
            if (shape1 := Ohat.shape) != (shape2 := (self.r, self.d)):
                raise errors.VerificationError(
                    "solve() did not return array of shape (r, d) "
                    f"(expected {shape2}, got {shape1})"
                )

            # Verify copy().
            self2 = _make_copy(self)
            _check_equality(self, self2, Ohat, "copy()")
            print("solve() is consistent with problem dimensions")

            # Verify save()/load().
            try:
                self2.save(tempfile, overwrite=True)
                self3 = self2.load(tempfile)
                _check_equality(self2, self3, Ohat, "save()/load()")
            except NotImplementedError:
                print("save() and/or load() not implemented")
        finally:
            if os.path.isfile(tempfile):
                os.remove(tempfile)


class PlainSolver(SolverTemplate):
    r"""Solve the :math:`2`-norm ordinary least-squares problem without any
    regularization or constraints.

    That is, solve

    .. math::
        \argmin_{\Ohat} \|\D\Ohat\trp - \Z\trp\|_F^2.

    The solution is calculated using :func:`scipy.linalg.lstsq()`.

    Parameters
    ----------
    cond : float or None
        Cutoff for 'small' singular values of the data matrix.
        See :func:`scipy.linalg.lstsq()`.
    lapack_driver : str or None
        Which LAPACK driver is used to solve the least-squares problem.
        See :func:`scipy.linalg.lstsq()`.
    """

    def __init__(self, cond=None, lapack_driver=None):
        """Store least-squares solver options."""
        SolverTemplate.__init__(self)
        self.__options = dict(cond=cond, lapack_driver=lapack_driver)

    @property
    def options(self):
        """Keyword arguments for :func:`scipy.linalg.lstsq()`."""
        return self.__options

    def __str__(self):
        """String representation: dimensions + solver options."""
        start = SolverTemplate.__str__(self)
        kwargs = self._print_kwargs(self.options)
        return start + f"\n  solver: scipy.linalg.lstsq({kwargs})"

    # Main methods ------------------------------------------------------------
    def fit(self, data_matrix, lhs_matrix):
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
                "least-squares system is underdetermined",
                errors.OpInfWarning,
                stacklevel=2,
            )
        return self

    def solve(self):
        r"""Solve the Operator Inference regression via
        :func:`scipy.linalg.lstsq()`.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix :math:`\Ohat` (not its transpose!).
        """
        results = la.lstsq(self.data_matrix, self.lhs_matrix.T, **self.options)
        return results[0].T

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
                hf.create_dataset("data_matrix", data=self.data_matrix)
                hf.create_dataset("lhs_matrix", data=self.lhs_matrix)

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
        solver : SolverTemplate
            Loaded solver.
        """
        options = dict()
        with utils.hdf5_loadhandle(loadfile) as hf:

            options = cls._load_dict(hf, "options")
            solver = cls(**options)

            if "data_matrix" in hf:
                D = hf["data_matrix"][:]
                Z = hf["lhs_matrix"][:]
                solver.fit(D, Z)

        return solver

    def copy(self):
        """Make a copy of the solver."""
        solver = self.__class__(
            cond=self.options["cond"],
            lapack_driver=self.options["lapack_driver"],
        )
        if self.data_matrix is not None:
            SolverTemplate.fit(solver, self.data_matrix, self.lhs_matrix)
        return solver
