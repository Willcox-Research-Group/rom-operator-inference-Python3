# basis/_linear.py
"""Linear basis class."""

__all__ = [
    "LinearBasis",
]

import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from .. import errors, utils
from ._base import BasisTemplate


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


def weighted_svd(Q, W):
    r"""Compute the weighted singular value decomposition of a matrix.

    The weighed SVD is a decomposition :math:`\bfPhi\bfSigma\bfPsi\trp = \Q`
    such that :math:`\bfPhi\trp\W\bfPhi = \I`.

    Parameters
    ----------
    Q : (n, k) ndarray
        Matrix to take the WSVD of.
    W : (n,) or (n, n) ndarray
        Weight matrix.

    Returns
    -------
    Phi : (n, k) ndarray
        Left singular vectors of ``Q``, orthonormal with respect to ``W``.
    """
    if W.ndim not in {1, 2}:
        raise ValueError("expected one- or two-dimensional spatial weights")

    # Weight the matrix.
    root_weights = np.sqrt(W) if W.ndim == 1 else la.sqrtm(W)
    WrootQ = _Wmult(root_weights, Q)

    # Compute the (non-weighted) SVD.
    Phi = la.svd(WrootQ, full_matrices=False)[0]

    # Unweight the singular vectors.
    if W.ndim == 1:
        return _Wmult(1 / root_weights, Phi)
    return la.solve(root_weights, Phi)


class LinearBasis(BasisTemplate):
    r"""Linear basis for representing the low-dimensional state approximation

    .. math::
       \q \approx \Vr\qhat = \sum_{i=1}^r \hat{q}_i \v_i,

    where :math:`\q\in\RR^n`,
    :math:`\Vr = [\v_1, \ldots, \v_r]\in \RR^{n \times r}`, and
    :math:`\qhat = [\hat{q}_1,\ldots,\hat{q}_r]\trp\in\RR^r`.

    Parameters
    ----------
    entries : (n, r) ndarray
        Basis entries :math:`\Vr`.
    weights : (n, n) ndarray or (n,) ndarray None
        Weight matrix :math:`\W` or its diagonals.
        If ``None`` (default), set :math:`\W` to the identity.
        Raise a warning if :math:`\Vr\trp\W\Vr` is not the identity.
    orthogonalize : bool
        If ``True``, take the SVD of :math:`\Vr` to orthogonalize the basis.
    check_orthogonality : bool
        If ``True``, raise a warning if the basis is not orthogonal
        (with respect to the weights, if given).
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
        orthogonalize: bool = False,
        check_orthogonality: bool = True,
        name: str = None,
    ):
        """Initialize the basis entries."""
        BasisTemplate.__init__(self, name=name)

        # Empty intializer for child classes (POD).
        if entries is None:
            self.__entries = None
            self.__weights = None
            return

        # Orthogonalize the basis entries if desired.
        if orthogonalize:
            if weights is None:
                entries = la.svd(entries, full_matrices=False)[0]
            else:
                entries = weighted_svd(entries, weights)

        # Set the entries.
        self.__entries = entries
        BasisTemplate.full_state_dimension.fset(self, entries.shape[0])
        self.reduced_state_dimension = entries.shape[1]

        # Set the weights.
        if weights is not None and np.ndim(weights) == 1:
            n = np.size(weights)
            weights = sparse.dia_array(([weights], [0]), shape=(n, n))
        self.__weights = weights

        # Verify orthogonality if desired.
        if check_orthogonality:
            V, W = self.entries, self.weights
            Id = (V.T @ V) if W is None else (V.T @ W @ V)
            if not np.allclose(Id, np.eye(self.reduced_state_dimension)):
                warnings.warn("basis not orthogonal", errors.UsageWarning)

    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        r"""Entries of the basis matrix :math:`\Vr\in\RR^{n \times r}`."""
        return self.__entries

    @property
    def weights(self) -> np.ndarray:
        r"""Weight matrix :math:`\W \in \RR^{n \times n}`.
        A warning is raised if :math:`\Vr\trp\W\Vr` is not the identity matrix.
        """
        return self.__weights

    @property
    def full_state_dimension(self):
        """Full state dimension."""
        return BasisTemplate.full_state_dimension.fget(self)

    def __getitem__(self, key):
        """self[:] --> self.entries."""
        return self.entries[key]

    def fit(self, *args, **kwargs):
        """Do nothing, the basis entries are set in the constructor."""
        return self

    def __str__(self):
        """String representation: class and dimensions."""
        out = [self.__class__.__name__]
        if (name := self.name) is not None:
            out[0] = f"{out[0]} for variable '{name}'"
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
        if self.entries is None:
            raise AttributeError("basis entries not initialized")
        if self.weights is not None:
            state = self.weights @ state
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
        if self.entries is None:
            raise AttributeError("basis entries not initialized")
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
            if (w := self.weights) is not None:
                if isinstance(w, sparse.dia_array):
                    w = w.data[0]
                hf.create_dataset("weights", data=w)

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
            entries = hf["entries"][:]
            weights = hf["weights"][:] if "weights" in hf else None
            return cls(entries, weights)
