# basis/_linear.py
"""Linear basis class."""

__all__ = [
    "LinearBasis",
]

import warnings
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from .. import errors, utils
from ._base import BasisTemplate


requires_entries = utils.requires2(
    "entries",
    "basis entries not initialized",
)


class LinearBasis(BasisTemplate):
    r"""Linear low-dimensional state approximation.

    This class approximates high-dimensional states :math:`\q\in\RR^n` as a
    linear combination of :math:`r` basis vectors
    :math:`\v_1,\ldots,\v_r\in\RR^n`. The basis matrix
    :math:`\Vr = [~\v_1~~\cdots~~\v_r~]\in \RR^{n \times r}` and an (optional)
    weighting matrix :math:`\W\in\RR^{n \times n}` define the approximation.

    The encoding from the high-dimensional space :math:`\RR^n` to the
    low-dimensional space :math:`\RR^r` is given by

    .. math::
       \q \mapsto \qhat = \Vr\trp\W\q,

    while the decoding from low-dimensional space :math:`\RR^r` to the
    high-dimensional space :math:`\RR^n` is defined as

    .. math::
       \qhat \mapsto \breve{\q} = \Vr\qhat = \sum_{i=1}^r \hat{q}_i \v_i,

    where :math:`\qhat = [\hat{q}_1,\ldots,\hat{q}_r]\trp\in\RR^r`.

    Basis entries :math:`\Vr` and the weights :math:`\W` are specified
    explicitly in the constructor, not learned from state data.

    Parameters
    ----------
    entries : (n, r) ndarray
        Basis entries :math:`\Vr\in\RR^{n\times r}`.
    weights : (n, n) ndarray or (n,) ndarray None
        Weight matrix :math:`\W\in\RR^n` or its diagonal entries.
        If ``None`` (default), set :math:`\W` to the identity.
    check_orthogonality : bool
        If ``True``, raise a warning if the basis is not orthogonal, i.e.,
        if :math:`\Vr\trp\W\Vr` is not the identity.
    name : str or None
        Label for the state variable that this basis approximates.

    Notes
    -----
    Pair with a :class:`opinf.pre.ShiftScaleTransformer` to do centered
    approximations of the form :math:`\q \approx\Vr\qhat + \bar{\q}`.
    """

    def __init__(
        self,
        entries,
        weights=None,
        check_orthogonality: bool = True,
        name: str = None,
    ):
        """Initialize the basis entries."""
        BasisTemplate.__init__(self, name=name)

        # Empty intializer for child classes (POD).
        if entries is None:
            self.__entries = None
            self.__weights = weights
            return

        # Set the entries.
        self.__entries = entries
        BasisTemplate.full_state_dimension.fset(self, entries.shape[0])
        BasisTemplate.reduced_state_dimension.fset(self, entries.shape[1])

        # Set the weights.
        if weights is not None:
            if (dim := np.ndim(weights)) == 1:
                n = np.size(weights)
                weights = sparse.dia_array(([weights], [0]), shape=(n, n))
            elif dim != 2:
                raise ValueError("expected one- or two-dimensional weights")
        self.__weights = weights

        # Verify orthogonality if desired.
        if check_orthogonality:
            V, W = self.entries, self.weights
            Id = (V.T @ V) if W is None else (V.T @ W @ V)
            if not np.allclose(Id, np.eye(self.reduced_state_dimension)):
                warnings.warn("basis not orthogonal", errors.OpInfWarning)

    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        r"""Entries of the basis matrix :math:`\Vr\in\RR^{n \times r}`. Also
        accessible via indexing (``basis[:]``).
        """
        return self.__entries

    @property
    def weights(self) -> np.ndarray:
        r"""Weight matrix :math:`\W \in \RR^{n \times n}`."""
        return self.__weights

    @property
    def full_state_dimension(self):
        r"""Dimension :math:`n` of the full state."""
        return BasisTemplate.full_state_dimension.fget(self)

    @property
    def reduced_state_dimension(self):
        r"""Dimension :math:`r` of the reduced (compressed) state."""
        return BasisTemplate.reduced_state_dimension.fget(self)

    @requires_entries
    def __getitem__(self, key):
        """self[:] --> self.entries."""
        return self.entries[key]

    def fit(self, *args, **kwargs):
        """Do nothing, the basis entries are set in the constructor."""
        return self

    # Dimension reduction -----------------------------------------------------
    @requires_entries
    def compress(self, state: np.ndarray) -> np.ndarray:
        r"""Map high-dimensional states to low-dimensional latent coordinates.

        .. math:: \q \mapsto \qhat = \Vr\trp\q.

        If a weight matrix :math:`\W` is present, the compression is

        .. math:: \q \mapsto \qhat = \Vr\trp\W\q.

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
        if self.weights is not None:
            state = self.weights @ state
        return self.entries.T @ state

    @requires_entries
    def decompress(
        self,
        states_compressed: np.ndarray,
        locs=None,
    ) -> np.ndarray:
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
    @requires_entries
    def plot1D(self, x=None, num_vectors=None, ax=None, **kwargs):
        """Plot the basis vectors over a one-dimensional domain.

        Parameters
        ----------
        x : (n,) ndarray or None
            One-dimensional spatial domain over which to plot the vectors.
            Defaults to [0, 1] with `n` points.
        num_vectors : int or None
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
        if x is None:
            x = np.linspace(0, 1, self.full_state_dimension)
        if num_vectors is None:
            num_vectors = self.reduced_state_dimension
        num_vectors = min(num_vectors, self.reduced_state_dimension)
        if ax is None:
            ax = plt.figure().add_subplot(111)

        for j in range(num_vectors):
            ax.plot(x, self.entries[:, j], **kwargs)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("spatial domain")
        ax.set_ylabel("basis vectors")

        return ax

    # Persistence -------------------------------------------------------------
    def __eq__(self, other) -> bool:
        """Two LinearBasis objects are equal if their type, dimensions, and
        basis entries are the same.
        """
        if not isinstance(other, self.__class__):
            return False
        if self.shape != other.shape:
            return False
        if self.weights is None and other.weights is not None:
            return False
        if (w1 := self.weights) is not None:
            if (w2 := other.weights) is None:
                return False
            if sparse.issparse(w1) and sparse.issparse(w2):
                w1, w2 = w1.data[0], w2.data[0]
            if not np.allclose(w1, w2):
                return False
        return np.all(self.entries == other.entries)

    def save(self, savefile: str, overwrite: bool = False):
        """Save the basis to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis to.
        overwrite : bool
            If ``True``, overwrite the file if it already exists.
            If ``False`` (default), raise a ``FileExistsError`` if the file
            already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            if self.name:
                meta = hf.create_dataset("meta", shape=(0,))
                meta.attrs["name"] = self.name
            hf.create_dataset("entries", data=self.entries)
            if (w := self.weights) is not None:
                if isinstance(w, sparse.dia_array):
                    w = w.data[0]
                hf.create_dataset("weights", data=w)

    @classmethod
    def load(cls, loadfile: str):
        """Load a basis from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the basis was stored via :meth:`save`.

        Returns
        -------
        LinearBasis
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            name = None
            if "meta" in hf:
                name = hf["meta"].attrs["name"]
            entries = hf["entries"][:]
            weights = hf["weights"][:] if "weights" in hf else None
            return cls(entries, weights, name=name)
