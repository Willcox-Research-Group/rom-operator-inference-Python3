# basis/_pod.py
"""Dimensionality reduction with Proper Orthogonal Decomposition (POD)."""

__all__ = [
    "PODBasis",
    "pod_basis",
    "svdval_decay",
    "cumulative_energy",
    "residual_energy",
]

import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import sklearn.utils.extmath as sklmath
import matplotlib.pyplot as plt

from .. import errors, utils
from ._base import BasisTemplate
from ._linear import LinearBasis


# Helper functions ============================================================
def _Wmult(W, arr):
    """Matrix multiply ``W`` and ``arr``, where ``W`` may be a one-dimensional
    array representing diagonals or a two-dimensional array.
    """
    if W.ndim == 1:
        if arr.ndim == 1:
            return W * arr
        elif arr.ndim == 2:
            return W.reshape((-1, 1)) * arr
        else:
            raise ValueError("expected one- or two-dimensional array")
    return W @ arr


# Main class ==================================================================
class PODBasis(LinearBasis):
    r"""Proper othogonal decomposition basis, consisting of the principal left
    singular vectors of a collection of states.

    .. math::
       \text{svd}(\Q) = \bfPhi\bfSigma\bfPsi\trp
       \qquad\Longrightarrow\qquad
       \text{pod}(\Q, r) = \bfPhi_{:,:r}

    The low-dimensional approximation is linear, see :class:`LinearBasis`.
    Here, :math:`\Q\in\mathbb{R}^{n\times k}` is a collection of states,
    the only argument of :meth:`fit`.

    The POD basis entries :math:`\Vr = \bfPhi_{:,:r}\in\RR^{n\times r}`
    are always orthogonal, i.e., :math:`\Vr\trp\Vr = \I`. If a weight matrix
    :math:`\W` is specified, a weighted SVD is computed so that
    :math:`\Vr\trp\W\Vr = \I`.

    The number of left singular vectors :math:`r` is the dimension of the
    reduced state and is set by specifying exactly one of the constructor
    arguments ``num_modes``, ``svdval_threshold``, ``residual_energy``,
    ``cumulative_energy``, or ``projection_error`. Once the basis entries are
    set by calling :meth:`fit`, the reduced state dimension :math:`r` can be
    updated by calling :meth:`set_dimension`.

    The POD singular values, which are used to select :math:`r`, are the
    diagonals of :math:`\bfSigma` and are denoted
    :math:`\sigma_1,\ldots,\sigma_k`.

    Parameters
    ----------
    num_vectors : int
        Set the reduced state dimension :math:`r` to ``num_vectors``.
    svdval_threshold : float
        Choose :math:`r` as the number of normalized POD singular values that
        are greater than the given threshold, i.e.,
        :math:`\sigma_{i}/\sigma_{1} \ge` ``svdval_threshold`` for
        :math:`i=1,\ldots,r`.
    cumulative_energy : float
        Choose :math:`r` as the smallest integer such that
        :math:`\sum_{i=1}^{r}\sigma_i^2\big/\sum_{j=1}^{k}\sigma_j^2 \ge `
        ``cumulative_energy``.
    residual_energy : float
        Choose :math:`r` as the smallest integer such that
        :math:`\sum_{i=r+1}^k\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2 \le `
        ``residual_energy``.
    projection_error : float
        Choose :math:`r` as the smallest integer such that
        :math:`\|\Q - \Vr\Vr\trp\Q\|_F \big/ \|\Q\|_F \le `
        ``projection_error``.
    max_vectors : int
        Maximum number of POD basis vectors to store. After calling
        :meth:`fit`, the ``reduced_state_dimension`` can be increased up to
        ``max_vectors``. If not given (default), record all :math:`k` left
        singular vectors.
    mode : str
        Strategy for computing the thin SVD of the states.

        **Options:**

        * ``"dense"`` (default): Use ``scipy.linalg.svd()`` to compute the SVD.
          May be inefficient for very large state matrices.
        * ``"randomized"``: Compute an approximate SVD with a randomized
          approach via ``sklearn.utils.extmath.randomized_svd()``.
          May be more efficient but less accurate for very large state
          matrices.
    weights : (n, n) ndarray or (n,) ndarray None
        Weight matrix :math:`\W` or its diagonals.
        When provided, a weighted singular value decomposition of the states
        is used to ensure that the left singular vectors are orthogonal with
        respect to the weight matrix, i.e., :math:`\bfPhi\trp\W\bfPhi = \I`.
        If ``None`` (default), set :math:`\W` to the identity.
    name : str
        Label for the state variable that this basis approximates.
    solver_options : dict
        Options to pass to the SVD solver (``scipy.linalg.svd()`` if
        ``mode="dense"``, ``sklearn.utils.extmath.randomized_svd()`` if
        ``mode=="randomized"``).
    """

    # Valid modes for the SVD engine.
    __MODES = (
        "dense",
        "randomized",
        # "streaming",  # TODO
    )

    # Constructors ------------------------------------------------------------
    def __init__(
        self,
        num_vectors: int = None,
        svdval_threshold: float = None,
        cumulative_energy: float = None,
        residual_energy: float = None,
        projection_error: float = None,
        max_vectors: int = None,
        mode: str = "dense",
        weights: np.ndarray = None,
        name: str = None,
        **solver_options,
    ):
        """Initialize an empty basis."""
        # Superclass constructor.
        LinearBasis.__init__(self, entries=None, weights=weights, name=name)

        # Store dimension selection criteria.
        self._set_dimension_selection_criterion(
            num_vectors=num_vectors,
            svdval_threshold=svdval_threshold,
            cumulative_energy=cumulative_energy,
            residual_energy=residual_energy,
            projection_error=projection_error,
        )
        self.__energy_is_being_estimated = False

        # Initialize hyperparameter properties.
        if max_vectors is not None:
            max_vectors = int(max_vectors)
            if max_vectors <= 0:
                raise ValueError("max_vectors must be a positive integer")
        self.__max_vectors_desired = max_vectors
        self.mode = mode
        self.solver_options = solver_options

        # Initialize entry properties.
        self.__leftvecs = None
        self.__svdvals = None
        self.__rightvecs = None
        self.__residual_energy = None
        self.__cumulative_energy = None

        # Store weights (separate from LinearBasis.__weights)
        if weights is not None:
            if weights.ndim == 1:
                self.__sqrt_weights = np.sqrt(weights)
            else:  # (weights.ndim == 2, checked by LinearBasis)
                self.__sqrt_weights = la.sqrtm(weights)
                self.__sqrt_weights_cho = la.cho_factor(self.__sqrt_weights)

    @classmethod
    def from_svd(
        cls,
        leftvecs: np.ndarray,
        svdvals: np.ndarray,
        rightvecs: np.ndarray = None,
        num_vectors: int = None,
        svdval_threshold: float = None,
        residual_energy: float = None,
        cumulative_energy: float = None,
        projection_error: float = None,
        max_vectors: int = None,
        mode: str = "dense",
        weights: np.ndarray = None,
        name: str = None,
    ):
        r"""Initialize a :class:`PODBasis` from the singular value
        decomposition of an :math:`n\times k` matrix.

        Parameters
        ----------
        leftvecs : (n, r) ndarray
            Left singular vectors.
        svdvals : (r,) ndarray
            Singular values.
        rightvecs : (k, r) ndarray or None
            Right singular vectors. Each *column* is a singular vector.

        See :meth:`__init__` for details on other arguments.

        Returns
        -------
        Initialized :class:`PODBasis` object.
        """
        # Default dimensionality criterion: use all left singular vectors.
        if all(
            arg is None
            for arg in (
                num_vectors,
                svdval_threshold,
                residual_energy,
                cumulative_energy,
                projection_error,
            )
        ):
            num_vectors = leftvecs.shape[1]

        basis = cls(
            num_vectors=num_vectors,
            svdval_threshold=svdval_threshold,
            residual_energy=residual_energy,
            cumulative_energy=cumulative_energy,
            projection_error=projection_error,
            max_vectors=max_vectors,
            mode=mode,
            weights=weights,
            name=name,
        )
        basis._store_svd(leftvecs, svdvals, rightvecs)
        basis._set_dimension_from_criterion()

        return basis

    # Properties: hyperparameters ---------------------------------------------
    @property
    def mode(self) -> str:
        """Strategy for computing the thin SVD of the states, either
        ``'dense'`` or ``'randomized'``.
        """
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m not in self.__MODES:
            raise AttributeError(
                f"invalid mode '{m}', options: "
                + ", ".join([f"{x}" for x in self.__MODES])
            )
        self.__mode = m

    @property
    def solver_options(self) -> dict:
        """Options to pass to the SVD solver."""
        return self.__solver_options

    @solver_options.setter
    def solver_options(self, options):
        if options is None:
            options = dict()
        if not isinstance(options, dict):
            raise TypeError("solver_options must be a dictionary")
        self.__solver_options = options

    # Properties: entries -----------------------------------------------------
    @property
    def leftvecs(self):
        """Leading left singular vectors of the training data."""
        return self.__leftvecs

    @property
    def max_vectors(self) -> int:
        """Number of POD basis vectors stored in memory.
        The ``reduced_state_dimension`` may be increased up to ``max_vectors``.
        """
        return None if self.leftvecs is None else self.leftvecs.shape[1]

    @property
    def svdvals(self):
        """Normalized singular values of the training data."""
        return self.__svdvals

    @property
    def rightvecs(self):
        """Leading *right* singular vectors of the training data."""
        return self.__rightvecs

    @property
    def cumulative_energy(self) -> float:
        r"""Amount of singular value energy captured by the basis,
        :math:`\sum_{i=1}^r\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2`.
        """
        return self.__cumulative_energy

    @property
    def residual_energy(self) -> float:
        r"""Amount of singular value energy *not* captured by the basis,
        :math:`\sum_{i=r+1}^k\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2`.
        """
        return self.__residual_energy

    # Dimension selection -----------------------------------------------------
    def _set_dimension_selection_criterion(
        self,
        num_vectors: int = None,
        svdval_threshold: float = None,
        cumulative_energy: float = None,
        residual_energy: float = None,
        projection_error: float = None,
    ):
        args = [
            ("num_vectors", num_vectors),
            ("svdval_threshold", svdval_threshold),
            ("cumulative_energy", cumulative_energy),
            ("residual_energy", residual_energy),
            ("projection_error", projection_error),
        ]
        provided = [(arg[1] is not None) for arg in args]

        # More than one argument provided.
        if sum(provided) > 1:
            firstarg = args[np.argmax(provided)]
            warnings.warn(
                "received multiple dimension selection criteria, using "
                f"{firstarg[0]}={firstarg[1]}",
                errors.UsageWarning,
            )
            self.__criterion = firstarg
            return

        # Return the one provided argument.
        for name, val in args:
            if val is not None:
                self.__criterion = (name, val)
                return

        # No arguments provided.
        raise ValueError(
            "exactly one dimension selection criterion must be provided"
        )

    @BasisTemplate.reduced_state_dimension.setter
    def reduced_state_dimension(self, r):
        """Set the reduced state dimension :math:`r`. If there is SVD data,
        set the basis entries to the first r left singular vectors.

        If r > max_vectors, set r = max_vectors and raise a warning.
        """
        r = int(r)

        # No basis data yet, but when fit() is called there will be r vectors.
        if self.svdvals is None:
            self._set_dimension_selection_criterion(num_vectors=r)
            BasisTemplate.reduced_state_dimension.fset(self, r)
            return

        # Basis data already exists, change the dimension and update.
        if r > (k := self.max_vectors):
            warnings.warn(
                "selected reduced dimension exceeds number of stored vectors, "
                f"setting reduced_state_dimension = max_vectors = {k:d}",
                errors.UsageWarning,
            )
            r = self.max_vectors
        BasisTemplate.reduced_state_dimension.fset(self, r)

        # Update singular value energies.
        r = self.reduced_state_dimension
        svdvals2 = self.svdvals**2
        self.__cumulative_energy = np.sum(svdvals2[:r]) / np.sum(svdvals2)
        self.__residual_energy = 1 - self.__cumulative_energy

        # Update entries.
        LinearBasis.__init__(
            self,
            entries=self.__leftvecs[:, :r],
            weights=self.weights,
            check_orthogonality=True,
            name=self.name,
        )

    def _set_dimension_from_criterion(self):
        """Set the dimension by interpreting the ``__criterion`` attribute."""
        criterion, value = self.__criterion
        svdvals2 = self.svdvals**2
        nsvdvals = svdvals2.size
        energy = np.cumsum(svdvals2) / np.sum(svdvals2)

        if criterion == "num_vectors":
            r = value
        elif criterion == "svdval_threshold":
            r = np.count_nonzero(self.svdvals >= value)
        elif criterion == "cumulative_energy":
            r = int(np.searchsorted(energy, value)) + 1
            if self.__energy_is_being_estimated:
                warnings.warn(
                    "cumulative energy is being estimated from only "
                    f"{nsvdvals:d} singular values",
                    errors.UsageWarning,
                )
        elif criterion == "residual_energy":
            r = np.count_nonzero(1 - energy >= value) + 1
            if self.__energy_is_being_estimated:
                warnings.warn(
                    "residual energy is being estimated from only "
                    f"{nsvdvals:d} singular values",
                    errors.UsageWarning,
                )
        elif criterion == "projection_error":
            r = np.count_nonzero(np.sqrt(1 - energy) >= value) + 1
            if self.__energy_is_being_estimated:
                warnings.warn(
                    "projection error is being estimated from only "
                    f"{nsvdvals:d} singular values",
                    errors.UsageWarning,
                )

        self.reduced_state_dimension = r

    def set_dimension(
        self,
        num_vectors: int = None,
        svdval_threshold: float = None,
        cumulative_energy: float = None,
        residual_energy: float = None,
        projection_error: float = None,
    ):
        r"""Set the reduced state dimension :math:`r`.

        Parameters
        ----------
        num_vectors : int
            Set the reduced state dimension :math:`r` to ``num_vectors``.
        svdval_threshold : float
            Choose :math:`r` as the number of normalized POD singular values
            that are greater than the given threshold, i.e.,
            :math:`\sigma_{i}/\sigma_{1} \ge` ``svdval_threshold`` for
            :math:`i=1,\ldots,r`.
        cumulative_energy : float
            Choose :math:`r` as the smallest integer such that
            :math:`\sum_{i=1}^{r}\sigma_i^2\big/\sum_{j=1}^{k}\sigma_j^2 \ge `
            ``cumulative_energy``.
        residual_energy : float
            Choose :math:`r` as the smallest integer such that
            :math:`\sum_{i=r+1}^k\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2 \le `
            ``residual_energy``.
        projection_error : float
            Choose :math:`r` as the smallest integer such that
            :math:`\|\Q - \Vr\Vr\trp\Q\|_F \big/ \|\Q\|_F \le `
            ``projection_error``.
        """
        self._set_dimension_selection_criterion(
            num_vectors=num_vectors,
            svdval_threshold=svdval_threshold,
            residual_energy=residual_energy,
            cumulative_energy=cumulative_energy,
            projection_error=projection_error,
        )

        # No basis data yet, do nothing.
        if self.svdvals is None:
            return

        # Basis data exists, set the dimension and update.
        self._set_dimension_from_criterion()

    # Fitting -----------------------------------------------------------------
    def _store_svd(self, left, svdvals, right):
        """Store the singular value decomposition, normalizing the singular
        values.
        """
        self.__leftvecs = left
        svdvals = np.sort(svdvals)[::-1]
        self.__svdvals = svdvals / svdvals[0]
        self.__rightvecs = right

    def fit(self, states):
        """Compute the POD basis entries by taking the SVD of the states.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of :math:`k` :math:`n`-dimensional snapshots.
        """
        if np.ndim(states) == 3:
            states = np.hstack(states)

        # Limit maximum number of vectors if needed.
        rmax = min(states.shape)
        keep = self.__max_vectors_desired
        if keep is None:
            keep = rmax
        elif keep > rmax:
            warnings.warn(
                f"only {rmax:d} singular vectors can be extracted "
                f"from ({states.shape[0]:d} x {states.shape[1]:d}) snapshots, "
                f"setting max_vectors={rmax:d}",
                errors.UsageWarning,
            )
            keep = rmax

        # SVD mode settings.
        self.__energy_is_being_estimated = False
        options = self.solver_options.copy()
        if self.mode == "dense":
            options["full_matrices"] = False
            driver = la.svd
        elif self.mode == "randomized":
            options["n_components"] = keep
            if "random_state" not in options:
                options["random_state"] = None
            driver = sklmath.randomized_svd
            if keep < rmax:
                self.__energy_is_being_estimated = True

        # Weight the states.
        if self.weights is not None:
            if states.shape[0] != (nW := self.__sqrt_weights.shape[0]):
                raise errors.DimensionalityError(
                    f"states not aligned with weights, should have {nW:d} rows"
                )
            states = _Wmult(self.__sqrt_weights, states)

        # Compute the SVD.
        V, svdvals, Wt = driver(states, **options)

        # Unweight the basis.
        if self.weights is not None:
            if self.__sqrt_weights.ndim == 1:
                V = _Wmult(1 / self.__sqrt_weights, V)
            else:
                V = la.cho_solve(self.__sqrt_weights_cho, V)
                # V = la.solve(sqrtW, V)

        # Store the results.
        self._store_svd(
            left=V[:, :keep],
            svdvals=svdvals,
            right=Wt[:keep, :].T,
        )
        self._set_dimension_from_criterion()

        return self

    # Visualization -----------------------------------------------------------
    def _check_svdvals_exist(self):
        """Raise an AttributeError if there are no singular values stored."""
        if self.svdvals is None:
            raise AttributeError("no singular value data, call fit()")

    def _plot_single(self, plotter, threshold, ax, kwargs):
        """Execute a single plotting routine."""
        self._check_svdvals_exist()
        plotter(self.svdvals, threshold=threshold, plot=True, ax=ax, **kwargs)
        return ax if ax is not None else plt.gca()

    def plot_svdval_decay(self, threshold=None, ax=None, **kwargs):
        """Plot the normalized singular value decay.

        Parameters
        ----------
        threshold : float or list[floats] or None
            Cutoff value(s) to mark on the plot.
        ax : plt.Axes or None
            Matplotlib Axes to plot on.
            If ``None`` (default), a new single-axes figure is created.
        kwargs : dict
            Options to pass to ``plt.semilogy()``

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        return self._plot_single(svdval_decay, threshold, ax, kwargs)

    def plot_cumulative_energy(self, threshold=None, ax=None, **kwargs):
        r"""Plot the cumulative singular value energy.

        The cumulative energy of :math:`r` singular values is defined by

        .. math::
           \kappa_r = \sum_{i=1}^r\sigma_i^2 \big/ \sum_{j=1}^k\sigma_j^2.

        This method plots :math:`\kappa_r` as a function of :math:`r`.

        Parameters
        ----------
        threshold : float or list[floats] or None
            Threshold energy value(s) to mark on the plot.
        ax : plt.Axes or None
            Matplotlib Axes to plot on.
            If ``None`` (default), a new single-axes figure is created.
        kwargs : dict
            Options to pass to ``plt.semilogy()``

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        return self._plot_single(cumulative_energy, threshold, ax, kwargs)

    def plot_residual_energy(self, threshold=None, ax=None, **kwargs):
        r"""Plot the residual singular value energy.

        The residual energy of :math:`r` singular values is defined by

        .. math::
           \epsilon_r
           = \sum_{i=r+1}^k\sigma_i^2 \big/ \sum_{j=1}^k\sigma_j^2
           = 1 - \left(
           \sum_{i=1}^r\sigma_i^2 \big/ \sum_{j=1}^k\sigma_j^2
           \right)

        This method plots :math:`\epsilon_r` as a function of :math:`r`.

        Parameters
        ----------
        threshold : float or list[floats] or None
            Cutoff value(s) to mark on the plot.
        ax : plt.Axes or None
            Matplotlib Axes to plot on.
            If ``None`` (default), a new single-axes figure is created.
        kwargs : dict
            Options to pass to ``plt.semilogy()``

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        return self._plot_single(residual_energy, threshold, ax, kwargs)

    def plot_energy(self):
        """Plot the normalized singular values and the cumulative and residual
        energies.
        """
        self._check_svdvals_exist()
        r = self.reduced_state_dimension

        def _rline(ax, ymin):
            ax.axvline(r, color="gray", linewidth=0.5, zorder=1)
            ax.text(
                r + 0.5,
                ymin,
                f"r = {r}",
                color="gray",
                horizontalalignment="left",
                verticalalignment="bottom",
            )

        fig, axes = plt.subplots(1, 2, figsize=(13.44, 4.8))

        ax = self.plot_svdval_decay(ax=axes[0])
        ax.set_title("POD singular values")
        _rline(ax, 1.05 * ax.get_ylim()[0])

        ax = self.plot_cumulative_energy(ax=axes[1], color="C0")
        ax.set_ylabel("Cumulative energy", color="C0")
        ax.tick_params(axis="y", which="both", color="C0", labelcolor="C0")
        _rline(ax, 0.01 + ax.get_ylim()[0])

        ax = self.plot_residual_energy(ax=axes[1].twinx())
        ax.set_ylabel("Residual energy", color="C1")
        ax.tick_params(axis="y", which="both", color="C1", labelcolor="C1")
        ax.set_title("POD singular value energy")

        fig.tight_layout()

    # Persistence -------------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the basis to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            if self.name:
                meta.attrs["name"] = self.name

            meta.attrs["criterion_type"] = self.__criterion[0]
            hf.create_dataset("criterion_value", data=[self.__criterion[1]])

            if self.__max_vectors_desired is not None:
                hf.create_dataset(
                    "max_vectors",
                    data=[self.__max_vectors_desired],
                )

            if (w := self.weights) is not None:
                if isinstance(w, sparse.dia_array):
                    w = w.data[0]
                hf.create_dataset("weights", data=w)

            if self.leftvecs is not None:
                hf.create_dataset("leftvecs", data=self.leftvecs)
                hf.create_dataset("svdvals", data=self.svdvals)
                hf.create_dataset("rightvecs", data=self.rightvecs)

    @classmethod
    def load(cls, loadfile, max_vectors=None):
        """Load a basis from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the basis was stored (via save()).
        max_vectors : int or None
            Maximum number of POD modes to load. If None (default), load all.

        Returns
        -------
        PODBasis object
        """
        kwargs = {}
        with utils.hdf5_loadhandle(loadfile) as hf:
            attrs = hf["meta"].attrs

            if "name" in attrs:
                kwargs["name"] = attrs["name"]

            kwargs[attrs["criterion_type"]] = hf["criterion_value"][0]

            if "max_vectors" in hf:
                kwargs["max_vectors"] = hf["max_vectors"][0]

            if "weights" in hf:
                kwargs["weights"] = hf["weights"][:]

            if "leftvecs" in hf:
                if (r := max_vectors) is not None:
                    r = min(r, hf["leftvecs"].shape[1])
                    if "num_vectors" in kwargs:
                        kwargs["num_vectors"] = r
                kwargs["leftvecs"] = hf["leftvecs"][:, :r]
                kwargs["svdvals"] = hf["svdvals"][:]
                kwargs["rightvecs"] = hf["rightvecs"][:, :r]
                return cls.from_svd(**kwargs)

            return cls(**kwargs)


# Functional API ==============================================================
def pod_basis(
    states,
    num_vectors: int = None,
    svdval_threshold: float = None,
    cumulative_energy: float = None,
    residual_energy: float = None,
    projection_error: float = None,
    mode: str = "dense",
    weights: np.ndarray = None,
    return_rightvecs: bool = False,
    **solver_options,
):
    r"""Compute a POD basis from the given states.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of :math:`k` :math:`n`-dimensional snapshots.
    return_rightvecx : bool
        If ``True``, return the right singular vectors as well.

    See :class:`PODBasis` for details on other arguments.

    Returns
    -------
    basis : (n, r) ndarray
        POD basis matrix, the first :math:`r` left singular vectors of the
        states.
    svdvals : (n,), (k,), or (r,) ndarray
        Normalized singular values in descending order.
        Always returns as many as are calculated:
        :math:`min\{n, k\}` for ``mode="dense"``,
        :math:`r` for ``mode="randomized"``.
    rightvecs : (k, r) ndarray
        First :math:`r` **right** singular vectors, as columns.
        **Only returned if ``return_W=True``.**
    """
    basis = PODBasis(
        num_vectors=num_vectors,
        svdval_threshold=svdval_threshold,
        cumulative_energy=cumulative_energy,
        residual_energy=residual_energy,
        projection_error=projection_error,
        mode=mode,
        weights=weights,
        **solver_options,
    ).fit(states)

    if return_rightvecs:
        r = basis.reduced_state_dimension
        return basis.entries, basis.svdvals, basis.rightvecs[:, :r]

    return basis.entries, basis.svdvals


def svdval_decay(
    singular_values,
    threshold: float = 1e-8,
    plot: bool = True,
    ax=None,
    **kwargs,
):
    """Count the number of **normalized** singular values that are greater than
    a specified threshold.

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot matrix, e.g.,
        ``scipy.linalg.svdvals(states)``.
    threshold : float or list[floats]
        Cutoff value(s) for the singular values.
    plot : bool
        If ``True``, plot the singular values and the cutoff value(s) against
        the singular value index.
    ax : plt.Axes or None
        Axes to plot the results on if ``plot=True``.
        If not given, a new single-axes figure is created.
    kwargs : dict
        Options to pass to ``plt.semilogy()``

    Returns
    -------
    ranks : int or list(int)
        Number of singular values greater than the cutoff value(s).
    """
    singular_values = np.sort(singular_values)[::-1]
    singular_values /= singular_values[0]

    # Calculate the number of singular values above the cutoff value(s).
    if threshold:
        if one_threshold := np.isscalar(threshold):
            threshold = [threshold]
        ranks = [np.count_nonzero(singular_values > ep) for ep in threshold]

    if plot:
        # Plot singular values.
        if ax is None:
            ax = plt.figure().add_subplot(111)

        options = dict(
            marker="*",
            color="k",
            markersize=8,
            markeredgewidth=0,
            zorder=3,
        )
        options.update(kwargs)
        marker = options.pop("marker")

        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, singular_values, marker, **options)
        ax.set_xlim((0, j.size + 1))

        # Draw cutoff value(s).
        if threshold:
            ylim = ax.get_ylim()
            for sigma, r in zip(threshold, ranks):
                ax.axhline(sigma, color="gray", linewidth=0.5)
                ax.text(
                    j[-1] + 0.5,
                    1.05 * sigma,
                    f"{sigma:.2e}",
                    color="gray",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.axvline(r, color="gray", linewidth=0.5)
                ax.text(
                    r + 0.5,
                    1.05 * ylim[0],
                    f"r = {r}",
                    color="gray",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )
            ax.set_ylim(ylim)

        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalized singular values")

    if threshold:
        return ranks[0] if one_threshold else ranks


def cumulative_energy(
    singular_values,
    threshold: float = 0.9999,
    plot: bool = True,
    ax=None,
    **kwargs,
):
    r"""Compute the number of singular values needed to surpass a given
    cumulative energy threshold.

    The cumulative energy of :math:`r` singular values is defined by

    .. math::
       \kappa_r = \sum_{i=1}^r\sigma_i^2 \big/ \sum_{j=1}^k\sigma_j^2.

    This function determines the smalled :math:`r` such that
    :math:`kappa_r` ``\ge thresh``.

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot matrix, e.g.,
        ``scipy.linalg.svdvals(states)``.
    threshold : float or list(float)
        Energy capture threshold(s). Default is 99.99%.
    plot : bool
        If ``True``, plot the cumulative energy and the capture threshold(s)
        against the singular value index (linear scale).
    ax : plt.Axes or None
        Axes to plot the results on if ``plot=True``.
        If not given, a new single-axes figure is created.
    kwargs : dict
        Options to pass to ``plt.plot()``

    Returns
    -------
    ranks : int or list(int)
        Number of singular values required to capture the specified energy.
    """
    # Calculate the cumulative energy.
    svdvals2 = np.sort(singular_values)[::-1] ** 2
    energy = np.cumsum(svdvals2) / np.sum(svdvals2)
    energy = np.concatenate(([0], energy))

    # Determine the points at which the cumulative energy passes the threshold.
    if threshold:
        if one_threshold := np.isscalar(threshold):
            threshold = [threshold]
        ranks = [int(np.searchsorted(energy, xi)) for xi in threshold]

    if plot:
        # Plot cumulative energy and threshold value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)

        options = dict(
            linestyle="-",
            marker="o",
            color="C2",
            markersize=4,
            markeredgewidth=0,
            linewidth=0.5,
            zorder=3,
        )
        options.update(kwargs)

        j = np.arange(singular_values.size + 1)
        ax.plot(j, energy, **options)
        ax.set_xlim(0, j.size)
        ylim = (ax.get_ylim()[0], 1.05)

        if threshold:
            for kappa, r in zip(threshold, ranks):
                ax.axhline(kappa, color="gray", linewidth=0.5)
                ax.text(
                    j[-1] + 0.5,
                    kappa + 0.01,
                    f"{kappa:%}",
                    color="gray",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.axvline(r, color="gray", linewidth=0.5)
                ax.text(
                    r + 0.5,
                    ylim[0] + 0.01,
                    f"r = {r}",
                    color="gray",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )

        ax.set_ylim(ylim)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Cumulative energy")

    if threshold:
        return ranks[0] if one_threshold else ranks


def residual_energy(
    singular_values,
    threshold: float = 1e-6,
    plot: bool = True,
    ax=None,
    **kwargs,
):
    r"""Compute the number of singular values needed such that the residual
    energy drops beneath a given threshold.

    The residual energy of :math:`r` singular values is defined by

    .. math::
        \epsilon_r
        = \sum_{i=r+1}^k\sigma_i^2 \big/ \sum_{j=1}^k\sigma_j^2
        = 1 - \left(
        \sum_{i=1}^r\sigma_i^2 \big/ \sum_{j=1}^k\sigma_j^2
        \right)

    This method plots :math:`\epsilon_r` as a function of :math:`r`.

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot matrix, e.g.,
        ``scipy.linalg.svdvals(states)``.
    threshold : float or list[floats]
        Energy residual threshold(s). Default is 10^-6.
    plot : bool
        If ``True``, plot the residual energy and the threshold(s)
        against the singular value index.
    ax : plt.Axes or None
        Axes to plot the results on if ``plot=True``.
        If not given, a new single-axes figure is created.
    kwargs : dict
        Options to pass to ``plt.semilogy()``

    Returns
    -------
    ranks : int or list[int]
        Number of singular values required to shrink the residual energy below
        the specified threshold.
    """
    # Calculate the residual energy.
    svdvals2 = np.sort(singular_values)[::-1] ** 2
    res_energy = 1 - (np.cumsum(svdvals2) / np.sum(svdvals2))
    res_energy = np.concatenate(([1], res_energy))

    # Determine the points when the residual energy dips under the threshold.
    if threshold:
        if one_threshold := np.isscalar(threshold):
            threshold = [threshold]
        ranks = [np.count_nonzero(res_energy > eps) for eps in threshold]

    if plot:
        # Plot residual energy and threshold value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)

        options = dict(
            linestyle="-",
            marker="s",
            color="C1",
            markersize=4,
            markeredgewidth=0,
            linewidth=0.5,
            zorder=3,
        )
        options.update(kwargs)

        j = np.arange(singular_values.size + 1)
        ax.semilogy(j, res_energy, **options)
        ax.set_xlim(0, j.size)
        ylim = (res_energy[-2] / 5, 1.1)

        if threshold:
            for epsilon, r in zip(threshold, ranks):
                ax.axhline(epsilon, color="gray", linewidth=0.5)
                ax.text(
                    j[-1] - 0.5,
                    1.1 * epsilon,
                    f"{epsilon:.2e}",
                    color="gray",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )
                ax.axvline(r, color="gray", linewidth=0.5)
                ax.text(
                    r + 0.5,
                    1.1 * ylim[0],
                    f"r = {r}",
                    color="gray",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )

        ax.set_ylim(ylim)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Residual energy")

    if threshold:
        return ranks[0] if one_threshold else ranks
