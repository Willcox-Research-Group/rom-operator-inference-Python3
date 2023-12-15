# pre/basis/_pod.py
"""Tools for basis computation and reduced-dimension selection."""

__all__ = [
    "PODBasis",
    "PODBasisMulti",
    "pod_basis",
    "svdval_decay",
    "cumulative_energy",
    "residual_energy",
    "projection_error",
]

import h5py
import numpy as np
import scipy.linalg as la
import sklearn.utils.extmath as sklmath
import matplotlib.pyplot as plt

from ..errors import LoadfileFormatError
from ..utils import hdf5_savehandle, hdf5_loadhandle
from ._linear import LinearBasis, LinearBasisMulti


class PODBasis(LinearBasis):
    r"""Proper othogonal decomposition basis, derived from the principal left
    singular vectors of a collection of states :math:`\Q`:

    .. math::
       \text{svd}(\Q) = \V\Sigma\W\trp
       \qquad\Longrightarrow\qquad
       \text{pod}(\Q, r) = \V_{:,:r}

    The low-dimensional approximation is linear:

    .. math::
       \q \approx \Vr\qhat = \sum_{i=1}^r \hat{q}_i \v_i

    Parameters
    ----------
    economize : bool
        If ``True``, throw away basis vectors beyond the first ``r`` whenever
        the ``r`` attribute is changed.
    """

    def __init__(self, economize=False):
        """Initialize an empty basis."""
        self.__r = None
        self.__entries = None
        self.__svdvals = None
        self.__dual = None
        # self.__spatialweights = None
        self.economize = bool(economize)
        LinearBasis.__init__(self)

    # Dimension selection -----------------------------------------------------
    def __shrink_stored_entries_to(self, r):
        if self.entries is not None and r is not None:
            self.__entries = self.__entries[:, :r].copy()
            self.__dual = self.__dual[:, :r].copy()

    @property
    def r(self):
        """Dimension of the basis, i.e., the number of basis vectors."""
        return self.__r

    @r.setter
    def r(self, r):
        """Set the reduced dimension."""
        if r is None:
            self.__r = None
            return
        if self.entries is None:
            raise AttributeError("empty basis (call fit() first)")
        if self.__entries.shape[1] < r:
            raise ValueError(
                f"only {self.__entries.shape[1]:d} " "basis vectors stored"
            )
        self.__r = r

        # Forget higher-order basis vectors.
        if self.economize:
            self.__shrink_stored_entries_to(r)

    @property
    def economize(self):
        """If True, throw away basis vectors beyond the first `r` whenever
        the `r` attribute is changed."""
        return self.__economize

    @economize.setter
    def economize(self, econ):
        """Set the economize flag."""
        self.__economize = bool(econ)
        if self.__economize:
            self.__shrink_stored_entries_to(self.r)

    def set_dimension(
        self, r=None, cumulative_energy=None, residual_energy=None
    ):
        """Set the basis dimension, i.e., the number of basis vectors.

        Parameters
        ----------
        r : int or None
            Number of basis vectors to include in the basis.
        cumulative_energy : float or None
            Cumulative energy threshold. If provided and r=None, choose the
            smallest number of basis vectors so that the cumulative singular
            value energy exceeds the given threshold.
        residual_energy : float or None
            Residual energy threshold. If provided, r=None, and
            cumulative_energy=None, choose the smallest number of basis vectors
            so that the residual singular value energy is less than the given
            threshold.

        Returns
        -------
        r : int
            Selected basis dimension.
        """
        if r is None:
            self._check_svdvals_exist()
            svdvals2 = self.svdvals**2
            cum_energy = np.cumsum(svdvals2) / np.sum(svdvals2)
            if cumulative_energy is not None:
                r = int(np.searchsorted(cum_energy, cumulative_energy)) + 1
            elif residual_energy is not None:
                r = np.count_nonzero(1 - cum_energy > residual_energy) + 1
            else:
                r = self.entries.shape[1]
        self.r = r

    @property
    def rmax(self):
        """Total number of stored basis vectors, i.e., the maximum value of r.
        Always the same as the dimension r if economize=True.
        """
        return None if self.__entries is None else self.__entries.shape[1]

    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        """Entries of the basis."""
        return None if self.__entries is None else self.__entries[:, : self.r]

    @property
    def shape(self):
        """Dimensions of the basis (state_dimension, reduced_dimension)."""
        return None if self.entries is None else (self.n, self.r)

    @property
    def svdvals(self):
        """Singular values of the training data."""
        return self.__svdvals

    @property
    def dual(self):
        """Leading *right* singular vectors."""
        return None if self.__dual is None else self.__dual[:, : self.r]

    # @property
    # def spatialweights(self):
    #     """Symmetric positive definite weighting matrix for the inner
    #     product. """
    #     return self.__spatialweights

    # @spatialweights.setter
    # def spatialweights(self, weights):
    #     """Set the spatial weights, checking dimension."""
    #     if weights is not None and self.n is not None:
    #         if weights.shape[0] != self.n:
    #             raise ValueError(f"{weights.shape} spatialweights "
    #                              f"not aligned with dimension n = {self.n}")
    #     self.__spatialweights = weights

    # def _spatial_weighting(self, state, weights):
    #     """Weight a state or a collection of states (spatially)."""
    #     if weights is not None:
    #         if weights.ndim == 1:
    #             if state.ndim == 2:
    #                 weights = weights.reshape(-1, 1)
    #             return weights * state
    #         return weights @ state
    #     return state

    # Dimension reduction -----------------------------------------------------
    # def compress(self, state):
    #     """Map high-dimensional states to low-dimensional latent coordinates.

    #     Parameters
    #     ----------
    #     state : (n,) or (n, k) ndarray
    #         High-dimensional state vector, or a collection of k such vectors
    #         organized as the columns of a matrix.

    #     Returns
    #     -------
    #     state_ : (r,) or (r, k) ndarray
    #         Low-dimensional latent coordinate vector, or a collection of k
    #         such vectors organized as the columns of a matrix.
    #     """
    #     # if self.spatialweights.ndim == 1 and
    #     # # state.ndim == 2 and state.shape[1] < self.r:
    #     #     return self._spatial_weighting(self.entries,
    #     #                                 self.spatialweights).T @ state
    #     return self.entries.T @ self._spatial_weighting(state,
    #                                                     self.spatialweights)

    # Fitting -----------------------------------------------------------------
    @staticmethod
    def _validate_rank(states, r):
        """Validate the rank `r` (if given)."""
        rmax = min(states.shape)
        if r is not None and (r > rmax or r < 1):
            raise ValueError(f"invalid POD rank r = {r} (need 1 ≤ r ≤ {rmax})")

    def _store_svd(self, V, svals, Wt):
        """Store SVD components as private attributes."""
        self.__entries = V
        self.__svdvals = np.sort(svals)[::-1] if svals is not None else None
        self.__dual = Wt.T if Wt is not None else None

    def _fit(
        self, driver, states, r, cumulative_energy, residual_energy, **options
    ):
        """Compute the POD basis of rank r corresponding to the states using
        `driver()` to compute the SVD.

        Parameters
        ----------
        driver : callable
            Function that computes the SVD of the given states.
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a single snapshot of
            dimension n.
        r : int or None
            Number of vectors to include in the basis.
            If None, use the largest possible basis (r = min{n, k}).
        cumulative_energy : float or None
            Cumulative energy threshold. If provided and r=None, choose the
            smallest number of basis vectors so that the cumulative singular
            value energy exceeds the given threshold.
        residual_energy : float or None
            Residual energy threshold. If provided, r=None, and
            cumulative_energy=None, choose the smallest number of basis vectors
            so that the residual singular value energy is less than the given
            threshold.
        options
            Additional parameters for scipy.linalg.svd().

        Notes
        -----
        This method computes the full singular value decomposition of `states`.
        The method fit_randomized() uses a randomized SVD.
        """
        #  spatialweights, temporalweights, **options):
        if np.ndim(states) == 3:
            # Concatenate list of states.
            states = np.hstack(states)
        self._validate_rank(states, r)

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
        self._store_svd(V, svdvals, Wt)
        self.set_dimension(r, cumulative_energy, residual_energy)
        # self.spatialweights = spatialweights

        return self

    def fit(
        self,
        states,
        r=None,
        cumulative_energy=None,
        residual_energy=None,
        **options,
    ):
        """Compute the POD basis of rank r corresponding to the states
        via the compact/thin singular value decomposition (scipy.linalg.svd()).

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a single snapshot of
            dimension n.
        r : int or None
            Number of vectors to include in the basis.
            If None, use the largest possible basis (r = min{n, k}).
        cumulative_energy : float or None
            Cumulative energy threshold. If provided and r=None, choose the
            smallest number of basis vectors so that the cumulative singular
            value energy exceeds the given threshold.
        residual_energy : float or None
            Residual energy threshold. If provided, r=None, and
            cumulative_energy=None, choose the smallest number of basis vectors
            so that the residual singular value energy is less than the given
            threshold.
        options
            Additional parameters for scipy.linalg.svd().

        Notes
        -----
        This method computes the full singular value decomposition of `states`.
        The method fit_randomized() uses a randomized SVD.
        """
        # spatialweights=None, temporalweights=None, **options):
        options["full_matrices"] = False

        return self._fit(
            la.svd, states, r, cumulative_energy, residual_energy, **options
        )

    def fit_randomized(self, states, r, **options):
        """Compute the POD basis of rank r corresponding to the states
        via the randomized singular value decomposition
        (sklearn.utils.extmath.randomized_svd()).

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a single snapshot of
            dimension n.
        r : int
            Number of vectors to include in the basis.
        options
            Additional parameters for sklearn.utils.extmath.randomized_svd().

        Notes
        -----
        This method uses an iterative method to approximate a partial singular
        value decomposition, which can be useful for very large n.
        The method fit() computes the full singular value decomposition.
        """
        # spatialweights=None, temporalweights=None, **options):
        if "random_state" not in options:
            options["random_state"] = None
        options["n_components"] = r

        return self._fit(
            sklmath.randomized_svd,
            states,
            r,
            None,
            None,
            # spatialweights, temporalweights,
            **options,
        )

    # Visualization -----------------------------------------------------------
    def _check_svdvals_exist(self):
        """Raise an AttributeError if there are no singular values stored."""
        if self.svdvals is None:
            raise AttributeError("no singular value data (call fit() first)")

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
        with hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["economize"] = int(self.economize)
            if self.r is not None:
                meta.attrs["r"] = self.r

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
        with hdf5_loadhandle(loadfile) as hf:
            if "meta" not in hf:
                raise LoadfileFormatError(
                    "invalid save format " "(meta/ not found)"
                )
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


class PODBasisMulti(LinearBasisMulti):
    r"""Block-diagonal proper othogonal decomposition basis, derived from the
    principal left singular vectors of a collection of states grouped into
    blocks.

    .. math::
       \Q = \left[\begin{array}{c}
       \Q_1 \\ \vdots \\ \Q_{n_\text{vars}}
       \end{array}\right]
       \qquad\Longrightarrow\qquad
       \text{pod\_multi}(\Q; r_1,\ldots,r_{n_\text{vars}})
       = \left[\begin{array}{ccc}
       \V_1 & & \\
       & \ddots & \\
       & & \V_{n_\text{vars}}
       \end{array}\right]

    where :math:`\V_i = \text{pod}(\Q_i;r_i), i=1,\ldots,n_\text{vars}`.

    The low-dimensional approximation is linear (see :class:`PODBasis`).

    Parameters
    ----------
    num_variables : int
        Number of variables represented in a single snapshot (number of
        individual bases to learn). The dimension `n` of the snapshots
        must be evenly divisible by num_variables; for example,
        num_variables=3 means the first n entries of a snapshot correspond to
        the first variable, and the next n entries correspond to the second
        variable, and the last n entries correspond to the third variable.
    economize : bool
        If True, throw away basis vectors beyond the first `r` whenever
        the `r` attribute is changed.
    variable_names : list of num_variables strings, optional
        Names for each of the `num_variables` variables.
        Defaults to "variable 1", "variable 2", ....
    """
    _BasisClass = PODBasis

    def __init__(self, num_variables, economize=False, variable_names=None):
        """Initialize an empty basis."""
        # Store dimensions.
        LinearBasisMulti.__init__(
            self, num_variables, variable_names=variable_names
        )
        self.economize = bool(economize)

    # Properties -------------------------------------------------------------
    @property
    def rs(self):
        """Dimensions for each diagonal basis block, i.e., `rs[i]` is the
        number of basis vectors in the representation for state variable `i`.
        """
        rs = [basis.r for basis in self.bases]
        return rs if any(rs) else None

    @rs.setter
    def rs(self, rs):
        """Reset the basis dimensions."""
        if len(rs) != self.num_variables:
            raise ValueError(f"rs must have length {self.num_variables}")

        # This will raise an AttributeError if the entries are not set.
        for basis, r in zip(self.bases, rs):
            basis.r = r  # Economization is also taken care of here.

        self._set_entries()

    @property
    def economize(self):
        """If True, throw away basis vectors beyond the first `r` whenever
        the `r` attribute is changed."""
        return self.__economize

    @economize.setter
    def economize(self, econ):
        """Set the economize flag."""
        econ = bool(econ)
        for basis in self.bases:
            basis.economize = econ
        self.__economize = econ

    # Main routines -----------------------------------------------------------
    def fit(
        self,
        states,
        rs=None,
        cumulative_energy=None,
        residual_energy=None,
        **options,
    ):
        """Fit the basis to the data.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a single snapshot of
            dimension n.
        rs : list(int) or None
            Number of basis vectors for each state variable.
            If None, use the largest possible bases (ri = min{ni, k}).
        cumulative_energy : float or None
            Cumulative energy threshold. If provided and rs=None, choose the
            smallest number of basis vectors so that the cumulative singular
            value energy exceeds the given threshold.
        residual_energy : float or None
            Residual energy threshold. If provided, rs=None, and
            cumulative_energy=None, choose the smallest number of basis vectors
            so that the residual singular value energy is less than the given
            threshold.
        options
            Additional parameters for scipy.linalg.svd().
        """
        # Split the state and compute the basis for each variable.
        if rs is None:
            rs = [None] * self.num_variables
        for basis, r, var in zip(
            self.bases, rs, np.split(states, self.num_variables, axis=0)
        ):
            basis.fit(var, r, cumulative_energy, residual_energy, **options)

        self._set_entries()
        return self

    def fit_randomized(self, states, rs, **options):
        """Compute the POD basis of rank r corresponding to the states
        via the randomized singular value decomposition
        (sklearn.utils.extmath.randomized_svd()).

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a single snapshot of
            dimension n.
        rs : list(int) or None
            Number of basis vectors for each state variable.
        options
            Additional parameters for sklearn.utils.extmath.randomized_svd().

        Notes
        -----
        This method uses an iterative method to approximate a partial singular
        value decomposition, which can be useful for very large n.
        The method fit() computes the full singular value decomposition.
        """
        # Fit the individual bases.
        if not isinstance(rs, list) or len(rs) != self.num_variables:
            raise TypeError(f"rs must be list of length {self.num_variables}")
        for basis, r, var in zip(
            self.bases, rs, np.split(states, self.num_variables, axis=0)
        ):
            basis.fit_randomized(var, r, **options)

        self._set_entries()
        return self

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
        LinearBasisMulti.save(self, savefile, overwrite)
        with h5py.File(savefile, "a") as hf:
            hf["meta"].attrs["economize"] = self.economize

    @classmethod
    def load(cls, loadfile):
        """Load a basis from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the basis was stored (via save()).

        Returns
        -------
        PODBasis object
        """
        # basis = LinearBasisMulti.load(cls, loadfile)
        basis = super(cls, cls).load(loadfile)
        with h5py.File(loadfile, "r") as hf:
            basis.economize = hf["meta"].attrs["economize"]
        return basis


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

    if mode == "dense" or mode == "simple":
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
) -> None:
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
