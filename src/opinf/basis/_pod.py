# basis/_pod.py
"""Dimensionality reduction with Proper Orthogonal Decomposition (POD)."""

__all__ = [
    "PODBasis",
    "pod_basis",
    "svdval_decay",
    "cumulative_energy",
    "residual_energy",
    "projection_error",
]

import warnings
import numpy as np
import scipy.linalg as la
import sklearn.utils.extmath as sklmath
import matplotlib.pyplot as plt

from .. import errors, utils
from ._base import BasisTemplate
from ._linear import LinearBasis  # , _Wmult


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
    residual_energy : float
        Choose :math:`r` as the smallest integer such that
        :math:`\sum_{i=r+1}^k\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2 \le `
        ``residual_energy``.
    cumulative_energy : float
        Choose :math:`r` as the smallest integer such that
        :math:`\sum_{i=1}^{r}\sigma_i^2\big/\sum_{j=1}^{k}\sigma_j^2 \ge `
        ``cumulative_energy``.
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
        residual_energy: float = None,
        cumulative_energy: float = None,
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
            residual_energy=residual_energy,
            cumulative_energy=cumulative_energy,
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

    @classmethod
    def from_svd(
        cls,
        left: np.ndarray,
        svdvals: np.ndarray,
        right: np.ndarray = None,
        num_vectors: int = None,
        weights: np.ndarray = None,
    ):
        """Initialize a :class:`PODBasis` from a singular value decomposition.

        Parameters
        ----------
        left : (n, k) ndarray
            Left singular vectors.
        svdvals : (k,) ndarray
            Singular values.
        right : (k, k) ndarray or None
            Right singular vectors (each *column* is a singular vector).
        num_vectors : int
            Number of singular vectors to use in the basis entries.
        weights : (n,) or (n, n) ndarray
            Weight matrix for the left singular vectors, i.e.,
            ``left.T @ weights @ left`` is the identity matrix.

        Returns
        -------
        Initialized :class:`PODBasis` object.
        """
        k = left.shape[1]
        if num_vectors is None:
            num_vectors = k

        basis = cls(num_vectors=num_vectors, weights=weights)
        basis._store_svd(left[:, :k], svdvals, right[:, :k])
        basis._set_dimension_from_criterion()

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
        """Singular values of the training data."""
        return self.__svdvals

    @property
    def rightvecs(self):
        """Leading *right* singular vectors of the training data."""
        return self.__rightvecs

    @property
    def residual_energy(self) -> float:
        r"""Amount of singular value energy *not* captured by the basis,
        :math:`\sum_{i=r+1}^k\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2`.
        """
        return self.__residual_energy

    @property
    def cumulative_energy(self) -> float:
        r"""Amount of singular value energy captured by the basis,
        :math:`\sum_{i=1}^r\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2`.
        """
        return self.__cumulative_energy

    # Dimension selection -----------------------------------------------------
    def _set_dimension_selection_criterion(
        self,
        num_vectors: int = None,
        svdval_threshold: float = None,
        residual_energy: float = None,
        cumulative_energy: float = None,
        projection_error: float = None,
    ):
        args = [
            ("num_vectors", num_vectors),
            ("svdval_threshold", svdval_threshold),
            ("residual_energy", residual_energy),
            ("cumulative_energy", cumulative_energy),
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
        """Set the reduced state dimension :math:`r`.

        If r > max_vectors, set r = max_vectors and raise a warning.
        """
        r = int(r)

        # No basis data yet, but when fit() is called there will be r vectors.
        if self.svdvals is None:
            self.__criterion = ("num_vectors", r)
            BasisTemplate.reduced_state_dimension.fset(self, r)
            return

        # Basis data already exists, change the dimension and update.
        if r > self.max_vectors:
            warnings.warn(
                "selected reduced dimension exceeds number of stored vectors, "
                "setting reduced_state_dimension = max_vectors",
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
            self.__leftvecs[:, :r],
            self.weights,
            orthogonalize=False,
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
        elif criterion == "residual_energy":
            r = np.count_nonzero(1 - energy >= value) + 1
            if self.__energy_is_being_estimated:
                warnings.warn(
                    "residual energy is being estimated from only "
                    f"{nsvdvals:d} singular values",
                    errors.UsageWarning,
                )
        elif criterion == "cumulative_energy":
            r = int(np.searchsorted(energy, value)) + 1
            if self.__energy_is_being_estimated:
                warnings.warn(
                    "cumulative energy is being estimated from only "
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
        residual_energy: float = None,
        cumulative_energy: float = None,
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
        residual_energy : float
            Choose :math:`r` as the smallest integer such that
            :math:`\sum_{i=r+1}^k\sigma_i^2\big/\sum_{j=1}^k\sigma_j^2 \le `
            ``residual_energy``.
        cumulative_energy : float
            Choose :math:`r` as the smallest integer such that
            :math:`\sum_{i=1}^{r}\sigma_i^2\big/\sum_{j=1}^{k}\sigma_j^2 \ge `
            ``cumulative_energy``.
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
    # def _spatial_weighting(self, state, weights):
    #     """Weight a state or a collection of states (spatially)."""
    #     if weights is not None:
    #         if weights.ndim == 1:
    #             if state.ndim == 2:
    #                 weights = weights.reshape(-1, 1)
    #             return weights * state
    #         return weights @ state
    #     return state

    def _store_svd(self, left, svdvals, right):
        """Store the singular value decomposition."""
        self.__leftvecs = left
        self.__svdvals = svdvals
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
            options["n_components"] = self.max_vectors
            if "random_state" not in options:
                options["random_state"] = None
            driver = sklmath.randomized_svd
            if keep < rmax:
                self.__energy_is_being_estimated = True

        # # Weight the states.
        # if spatialweights is not None:
        #     if spatialweights.ndim == 1:
        #         root_weights = np.sqrt(spatialweights)
        #         inv_root_weights = 1 / root_weights
        #     elif spatialweights.ndim == 2:
        #         root_weights = la.sqrtm(spatialweights)
        #         inv_root_weights = la.inv(root_weights)
        #     else:
        #         raise ValueError("1D spatial weights only")
        #     states = self._spatial_weighting(states, root_weights)

        # Compute the SVD.
        V, svdvals, Wt = driver(states, **options)

        # Unweight the basis vectors.
        # if spatialweights is not None:
        #     if spatialweights.ndim == 1:
        #         V *= np.reshape(1 / root_weights, (-1, 1))
        #     elif spatialweights.ndim == 2:
        #         V = inv_root_weights @ V

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

    # TODO: use the functional API for these methods (one implementation).
    def plot_svdval_decay(self, threshold=None, normalize=True, ax=None):
        """Plot the normalized singular value decay.

        Parameters
        ----------
        threshold : float or None
            Cutoff value to mark on the plot.
        normalize : bool
            If True, normalize so that the maximum singular value is 1.
        ax : plt.Axes or None
            Matplotlib Axes to plot on. If None, a new figure is created.

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        self._check_svdvals_exist()
        if ax is None:
            ax = plt.figure().add_subplot(111)

        singular_values = self.svdvals
        if normalize:
            singular_values = singular_values / singular_values[0]
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, singular_values, "k*", ms=10, mew=0, zorder=3)
        ax.set_xlim((0, j.size))

        if threshold is not None:
            rank = np.count_nonzero(singular_values > threshold)
            ax.axhline(threshold, color="gray", linewidth=0.5)
            ax.axvline(rank, color="gray", linewidth=0.5)
            # TODO: label lines with text.

        ax.set_xlabel("Singular value index")
        ax.set_ylabel(("Normalized s" if normalize else "") + "ingular values")

        return ax

    def plot_residual_energy(self, threshold=None, ax=None):
        """Plot the residual singular value energy decay, defined by

            residual_j = sum(svdvals[j+1:]**2) / sum(svdvals**2).

        Parameters
        ----------
        threshold : 0 ≤ float ≤ 1 or None
            Cutoff value to mark on the plot.
        ax : plt.Axes or None
            Matplotlib Axes to plot on. If None, a new figure is created.

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        self._check_svdvals_exist()
        if ax is None:
            ax = plt.figure().add_subplot(111)

        svdvals2 = self.svdvals**2
        res_energy = 1 - (np.cumsum(svdvals2) / np.sum(svdvals2))
        j = np.arange(1, svdvals2.size + 1)
        ax.semilogy(j, res_energy, "C0.-", ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)

        if threshold is not None:
            rank = np.count_nonzero(res_energy > threshold) + 1
            ax.axhline(threshold, color="gray", linewidth=0.5)
            ax.axvline(rank, color="gray", linewidth=0.5)
            # TODO: label lines with text.

        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Residual energy")

        return ax

    def plot_cumulative_energy(self, threshold=None, ax=None):
        """Plot the cumulative singular value energy, defined by

            energy_j = sum(svdvals[:j]**2) / sum(svdvals**2).

        Parameters
        ----------
        threshold : 0 ≤ float ≤ 1 or None
            Cutoff value to mark on the plot.
        ax : plt.Axes or None
            Matplotlib Axes to plot on. If None, a new figure is created.

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        self._check_svdvals_exist()
        if ax is None:
            ax = plt.figure().add_subplot(111)

        svdvals2 = self.svdvals**2
        cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)
        j = np.arange(1, svdvals2.size + 1)
        ax.plot(j, cum_energy, "C0.-", ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)

        if threshold is not None:
            rank = int(np.searchsorted(cum_energy, threshold)) + 1
            ax.axhline(threshold, color="gray", linewidth=0.5)
            ax.axvline(rank, color="gray", linewidth=0.5)
            # TODO: label lines with text.

        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Residual energy")

        return ax

    def plot_energy(self):
        """Plot the normalized singular value and residual energy decay."""
        self._check_svdvals_exist()

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        self.plot_svdval_decay(ax=axes[0])
        self.plot_residual_energy(ax=axes[1])
        fig.tight_layout()

        return fig, axes

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
            meta.attrs["economize"] = int(self.economize)
            if self.reduced_state_dimension is not None:
                meta.attrs["r"] = self.reduced_state_dimension

            if self.entries is not None:
                hf.create_dataset("entries", data=self.__entries)
                hf.create_dataset("svdvals", data=self.__svdvals)
                hf.create_dataset("dual", data=self.__dual)

    @classmethod
    def load(cls, loadfile, rmax=None):
        """Load a basis from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the basis was stored (via save()).
        rmax : int or None
            Maximum number of POD modes to load. If None (default), load all.

        Returns
        -------
        PODBasis object
        """
        entries, svdvals, dualT, r = None, None, None, rmax
        with utils.hdf5_loadhandle(loadfile) as hf:
            economize = bool(hf["meta"].attrs["economize"])
            if "r" in hf["meta"].attrs:
                r = int(hf["meta"].attrs["r"])
                if rmax is not None:
                    r = min(r, rmax)

            if "entries" in hf:
                entries = hf["entries"][:, :rmax]
                svdvals = hf["svdvals"][:rmax]
                dualT = hf["dual"][:, :rmax].T

        out = cls(economize=economize)
        out._store_svd(entries, svdvals, dualT)
        out.r = r
        return out


# Functional API ==============================================================
def pod_basis(
    states,
    r: int = None,
    mode: str = "dense",
    return_W: bool = False,
    **options,
):
    """Compute the POD basis of rank r corresponding to the states.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots. Each column is a single snapshot of dimension n.
    r : int or None
        Number of POD basis vectors and singular values to compute.
        If None (default), compute the full SVD.
    mode : str
        Strategy to use for computing the truncated SVD of the states. Options:

        * "dense" (default): Use scipy.linalg.svd() to compute the SVD.
            May be inefficient for very large matrices.
        * "randomized": Compute an approximate SVD with a randomized approach
            using sklearn.utils.extmath.randomized_svd(). This gives faster
            results at the cost of some accuracy.

    return_W : bool
        If True, also return the first r *right* singular vectors.
    options
        Additional parameters for the SVD solver, which depends on `mode`:

        * "dense": scipy.linalg.svd()
        * "randomized": sklearn.utils.extmath.randomized_svd()

    Returns
    -------
    basis : (n, r) ndarray
        First r POD basis vectors (left singular vectors).
        Each column is a single basis vector of dimension n.
    svdvals : (n,), (k,), or (r,) ndarray
        Singular values in descending order. Always returns as many as are
        calculated: r for mode="randomize", min(n, k) for "dense".
    W : (k, r) ndarray
        First r **right** singular vectors, as columns.
        **Only returned if return_W=True.**
    """
    # Validate the rank.
    rmax = min(states.shape)
    if r is None:
        r = rmax
    if r > rmax or r < 1:
        raise ValueError(f"invalid POD rank r = {r} (need 1 ≤ r ≤ {rmax})")

    if mode == "dense":
        V, svdvals, Wt = la.svd(states, full_matrices=False, **options)
        W = Wt.T

    elif mode == "randomized":
        if "random_state" not in options:
            options["random_state"] = None
        V, svdvals, Wt = sklmath.randomized_svd(states, r, **options)
        W = Wt.T

    else:
        raise NotImplementedError(f"invalid mode '{mode}'")

    if return_W:
        return V[:, :r], svdvals, W[:, :r]
    return V[:, :r], svdvals


def svdval_decay(
    singular_values,
    tol: float = 1e-8,
    normalize: bool = True,
    plot: bool = True,
    ax=None,
):
    """Count the number of normalized singular values that are greater than
    the specified tolerance.

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    tol : float or list(float)
        Cutoff value(s) for the singular values.
    normalize : bool
        If True, normalize so that the maximum singular value is 1.
    plot : bool
        If True, plot the singular values and the cutoff value(s) against the
        singular value index.
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        The number of singular values greater than the cutoff value(s).
    """
    # Calculate the number of singular values above the cutoff value(s).
    one_tol = np.isscalar(tol)
    if one_tol:
        tol = [tol]
    singular_values = np.sort(singular_values)[::-1]
    if normalize:
        singular_values /= singular_values[0]
    ranks = [np.count_nonzero(singular_values > epsilon) for epsilon in tol]

    if plot:
        # Visualize singular values and cutoff value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, singular_values, "C0*", ms=10, mew=0, zorder=3)
        ax.set_xlim((0, j.size))
        ylim = ax.get_ylim()
        for epsilon, r in zip(tol, ranks):
            ax.axhline(epsilon, color="black", linewidth=0.5, alpha=0.75)
            ax.axvline(r, color="black", linewidth=0.5, alpha=0.75)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"Singular value index $j$")
        ax.set_ylabel(r"Singular value $\sigma_j$")

    return ranks[0] if one_tol else ranks


def cumulative_energy(singular_values, thresh=0.9999, plot=True, ax=None):
    """Compute the number of singular values needed to surpass a given
    energy threshold. The energy of j singular values is defined by

        energy_j = sum(singular_values[:j]**2) / sum(singular_values**2).

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    thresh : float or list(float)
        Energy capture threshold(s). Default is 99.99%.
    plot : bool
        If True, plot the singular values and the cumulative energy against
        the singular value index (linear scale).
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        The number of singular values required to capture more than each
        energy capture threshold.
    """
    # Calculate the cumulative energy.
    svdvals2 = np.sort(singular_values)[::-1] ** 2
    cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)

    # Determine the points at which the cumulative energy passes the threshold.
    one_thresh = np.isscalar(thresh)
    if one_thresh:
        thresh = [thresh]
    ranks = [int(np.searchsorted(cum_energy, xi)) + 1 for xi in thresh]

    if plot:
        # Visualize cumulative energy and threshold value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.plot(j, cum_energy, "C2.-", ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)
        for xi, r in zip(thresh, ranks):
            ax.axhline(xi, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(r, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Cumulative energy")

    return ranks[0] if one_thresh else ranks


def residual_energy(singular_values, tol=1e-6, plot=True, ax=None):
    """Compute the number of singular values needed such that the residual
    energy drops beneath the given tolerance. The residual energy of j
    singular values is defined by

        residual_j = 1 - sum(singular_values[:j]**2) / sum(singular_values**2).

    Parameters
    ----------
    singular_values : (n,) ndarray
        Singular values of a snapshot set, e.g., scipy.linalg.svdvals(states).
    tol : float or list(float)
        Energy residual tolerance(s). Default is 10^-6.
    plot : bool
        If True, plot the singular values and the residual energy against
        the singular value index (log scale).
    ax : plt.Axes or None
        Matplotlib Axes to plot the results on if plot = True.
        If not given, a new single-axes figure is created.

    Returns
    -------
    ranks : int or list(int)
        Number of singular values required to for the residual energy to drop
        beneath each tolerance.
    """
    # Calculate the residual energy.
    svdvals2 = np.sort(singular_values)[::-1] ** 2
    res_energy = 1 - (np.cumsum(svdvals2) / np.sum(svdvals2))

    # Determine the points when the residual energy dips under the tolerance.
    one_tol = np.isscalar(tol)
    if one_tol:
        tol = [tol]
    ranks = [np.count_nonzero(res_energy > epsilon) + 1 for epsilon in tol]

    if plot:
        # Visualize residual energy and tolerance value(s).
        if ax is None:
            ax = plt.figure().add_subplot(111)
        j = np.arange(1, singular_values.size + 1)
        ax.semilogy(j, res_energy, "C1.-", ms=10, lw=1, zorder=3)
        ax.set_xlim(0, j.size)
        for epsilon, r in zip(tol, ranks):
            ax.axhline(epsilon, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(r, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"Singular value index")
        ax.set_ylabel(r"Residual energy")

    return ranks[0] if one_tol else ranks


def projection_error(states, basis):
    """Calculate the absolute and relative projection errors induced by
    projecting states to a low dimensional basis, i.e.,

        absolute_error = ||Q - Vr Vr^T Q||_F,
        relative_error = ||Q - Vr Vr^T Q||_F / ||Q||_F

    where Q = states and Vr = basis. Note that Vr Vr^T is the orthogonal
    projector onto subspace of R^n defined by the basis.

    Parameters
    ----------
    states : (n, k) or (k,) ndarray
        Matrix of k snapshots where each column is a single snapshot, or a
        single 1D snapshot. If 2D, use the Frobenius norm; if 1D, the l2 norm.
    Vr : (n, r) ndarray
        Low-dimensional basis of rank r. Each column is one basis vector.

    Returns
    -------
    absolute_error : float
        Absolute projection error ||Q - Vr Vr^T Q||_F.
    relative_error : float
        Relative projection error ||Q - Vr Vr^T Q||_F / ||Q||_F.
    """
    norm_of_states = la.norm(states)
    absolute_error = la.norm(states - basis @ (basis.T @ states))
    return absolute_error, absolute_error / norm_of_states
