# basis/_linear.py
"""Linear basis class."""

__all__ = [
    "LinearBasis",
]

import warnings
import numpy as np
import matplotlib.pyplot as plt

from .. import errors, utils
from ._base import BasisTemplate


class LinearBasis(BasisTemplate):
    r"""Linear basis for representing the low-dimensional state approximation

    .. math::
       \q \approx \Vr\qhat = \sum_{i=1}^r \hat{q}_i \v_i,

    where :math:`\q\in\RR^n`,
    :math:`\Vr = [\v_1, \ldots, \v_r]\in \RR^{n \times r}`, and
    :math:`\qhat = [\hat{q}_1,\ldots,\hat{q}_r]\trp\in\RR^r`.

    Parameters
    ----------
    basis : (n, r) ndarray
        Basis entries :math:`\Vr`.
    """

    # orthogonalize : bool
    #     If ``True``, take the SVD of V to orthogonalize the basis.
    # weights : (n, n) ndarray or (n,) ndarray None
    #     Weight matrix :math:`\W` or its diagonals.
    #     If ``None`` (default), set :math:`\W` to the identity.
    #     Raise a warning if :math:`\Vr\trp\W\Vr` is not the identity.

    # TODO: weight matrix
    def __init__(self, basis):
        """Initialize the basis entries."""
        self.entries = basis

    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        r"""Entries of the basis matrix :math:`\Vr\in\RR^{n \times r}`."""
        return self.__entries

    @entries.setter
    def entries(self, V):
        """Set the basis entries."""
        if V is None:
            self.__entries = None
            self.full_state_dimension = None
            self.reduced_state_dimension = None
            return

        n, r = V.shape
        if not np.allclose(V.T @ V, np.eye(r)):
            warnings.warn("basis is not orthogonal", errors.UsageWarning)

        self.__entries = V
        self.full_state_dimension = n
        self.reduced_state_dimension = r

    def __getitem__(self, key):
        """self[:] --> self.entries."""
        return self.entries[key]

    def fit(self, *args, **kwargs):
        """Do nothing, the basis entries are set in the constructor."""
        return self

    def __str__(self):
        """String representation: class and dimensions."""
        out = [self.__class__.__name__]
        out.append(
            f"Full state dimension    n = {self.full_state_dimension:d}"
        )
        out.append(
            f"Reduced state dimension r = {self.reduced_state_dimension:d}"
        )
        return "\n".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Dimension reduction -----------------------------------------------------
    def compress(self, state):
        r"""Map high-dimensional states to low-dimensional latent coordinates.

        .. math:: \q \mapsto \qhat = \Vr\trp\q.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.

        Returns
        -------
        states_compressed : (r, ...) ndarray
            Matrix of `r`-dimensional latent coordinate vectors, or a single
            coordinate vector.
        """
        return self.entries.T @ state

    def decompress(self, states_compressed, locs=None):
        r"""Map low-dimensional latent coordinates to high-dimensional states.

        .. math:: \qhat \mapsto \breve{\q} = \Vr\qhat

        Parameters
        ----------
        states_compressed : (r, ...) ndarray
            Matrix of `r`-dimensional latent coordinate vectors, or a single
            coordinate vector.
        locs : slice or (p,) ndarray of integers or None
            If given, return the decompressed state at only the `p` specified
            locations (indices) described by ``locs``.

        Returns
        -------
        states_decompressed : (n, ...) or (p, ...) ndarray
            Matrix of `n`-dimensional decompressed state vectors, or the `p`
            entries of such at the entries specified by ``locs``.
        """
        Vr = self.entries if locs is None else self.entries[locs]
        return Vr @ states_compressed

    # Visualizations ----------------------------------------------------------
    def plot1D(self, x, num_modes=None, ax=None, **kwargs):
        """Plot the basis vectors over a one-dimensional domain.

        Parameters
        ----------
        x : (n,) ndarray
            One-dimensional spatial domain over which to plot the vectors.
        num_modes : int or None
            Number of basis vectors to plot.
            If ``None`` (default), plot all basis vectors.
        ax : plt.Axes or None
            Matplotlib Axes to plot on.
            If ``None`` (default), a new figure is created.
        kwargs : dict
            Other keyword arguments to pass to ``plt.plot()``.

        Returns
        -------
        ax : plt.Axes
            Matplotlib Axes for the plot.
        """
        if num_modes is None:
            num_modes = self.reduced_state_dimension
        if ax is None:
            ax = plt.figure().add_subplot(111)

        for j in range(num_modes):
            ax.plot(x, self.entries[:, j], **kwargs)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("spatial domain x")
        ax.set_ylabel("basis vectors v(x)")

        return ax

    # Persistence -------------------------------------------------------------
    def __eq__(self, other):
        """Two LinearBasis objects are equal if their type, dimensions, and
        basis entries are the same.
        """
        if not isinstance(other, self.__class__):
            return False
        if self.shape != other.shape:
            return False
        return np.all(self.entries == other.entries)

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
            hf.create_dataset("entries", data=self.entries)

    @classmethod
    def load(cls, loadfile):
        """Load a basis from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the basis was stored (via save()).

        Returns
        -------
        LinearBasis
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            return cls(hf["entries"][:])
