# pre/basis/_base.py
"""Base basis class."""

__all__ = [
    "BasisTemplate",
]

import abc
import numpy as np
import scipy.linalg as la

from .. import errors, utils


class BasisTemplate(abc.ABC):
    """Template class for bases.

    Classes that inherit from this template must implement the methods
    :meth:`fit`, :meth:`compress`, and :meth:`decompress`.

    See :class:`PODBasis` for an example.
    """

    def __init__(self, name: str = None):
        """Initialize attributes."""
        self.__n = None
        self.__r = None
        self.__name = name

    # Properties --------------------------------------------------------------
    @property
    def full_state_dimension(self) -> int:
        r"""Dimension :math:`n` of the full state."""
        return self.__n

    @full_state_dimension.setter
    def full_state_dimension(self, n: int):
        """Set the full state dimension."""
        self.__n = int(n) if n is not None else None

    @property
    def reduced_state_dimension(self) -> int:
        r"""Dimension :math:`r` of the reduced (compressed) state."""
        return self.__r

    @reduced_state_dimension.setter
    def reduced_state_dimension(self, r: int):
        """Set the reduced state dimension."""
        self.__r = int(r) if r is not None else None

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions :math:`(n, r)` of the basis."""
        return (self.full_state_dimension, self.reduced_state_dimension)

    @property
    def name(self) -> str:
        """Label for the state variable that this basis approximates."""
        return self.__name

    @name.setter
    def name(self, label: str):
        """Set the state variable name."""
        self.__name = str(label) if label is not None else None

    def __str__(self):
        """String representation: class and dimensions."""
        out = [self.__class__.__name__]
        if (name := self.name) is not None:
            out[0] = f"{out[0]} for variable '{name}'"
        if (n := self.full_state_dimension) is not None:
            out.append(f"Full state dimension    n = {n:d}")
        if (r := self.reduced_state_dimension) is not None:
            out.append(f"Reduced state dimension r = {r:d}")
        return "\n  ".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        return utils.str2repr(self)

    # Fitting -----------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, states):
        """Construct the basis.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of :math:`k` :math:`n`-dimensional snapshots.

        Returns
        -------
        self
        """
        raise NotImplementedError  # pragma: no cover

    # Dimension reduction -----------------------------------------------------
    @abc.abstractmethod
    def compress(self, states):
        """Map high-dimensional states to low-dimensional latent coordinates.

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
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def decompress(self, states_compressed, locs=None):
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        states_compressed : (r, ...) ndarray
            Matrix of `r`-dimensional latent coordinate vectors, or a single
            coordinate vector.
        locs : slice or (p,) ndarray of integers or None
            If given, return the decompressed state at *only* the
            `p` specified locations (indices) described by ``locs``.

        Returns
        -------
        states_decompressed : (n, ...) or (p, ...) ndarray
            Matrix of `n`-dimensional decompressed state vectors, or the `p`
            entries of such at the entries specified by ``locs``.
        """
        raise NotImplementedError  # pragma: no cover

    # Projection --------------------------------------------------------------
    def project(self, state):
        """Project a high-dimensional state vector to the subset of the
        high-dimensional space that can be represented by the basis.

        This is done by

        1. expressing the state in low-dimensional latent coordinates, then
        2. reconstructing the high-dimensional state corresponding to those
           coordinates.

        In other words, ``project(Q)`` is equivalent to
        ``decompress(compress(Q))``.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.

        Returns
        -------
        state_projected : (n, ...) ndarray
            Matrix of `n`-dimensional projected state vectors, or a single
            projected state vector.
        """
        return self.decompress(self.compress(state))

    def projection_error(self, state, relative=True) -> float:
        r"""Compute the error of the basis representation of a state or states.

        This function computes :math:`\frac{\|\Q - \mathcal{P}(\Q)\|}{\|\Q\|}`,
        where :math:`\Q` is the ``state`` and :math:`\mathcal{P}` is the
        projection defined by :meth:`project()`.
        If ``state`` is one-dimensional then :math:`||\cdot||` is the vector
        2-norm. If ``state`` is two-dimensional then :math:`||\cdot||` is the
        matrix Frobenius norm.

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of `k` such vectors
            organized as the columns of a matrix.
        relative : bool
            If ``True`` (default), return the relative projection error
            ``norm(state - project(state)) / norm(state)``.
            If ``False``, return the absolute projection error
            ``norm(state - project(state))``.

        Returns
        -------
        float
            Relative error of the projection (``relative=True``) or
            absolute error of the projection (``relative=False``).
        """
        diff = la.norm(state - self.project(state))
        if relative:
            diff /= la.norm(state)
        return diff

    # Model persistence -------------------------------------------------------
    def save(self, savefile: str, overwrite: bool = False):
        """Save the transformer to an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    @classmethod
    def load(cls, loadfile: str):
        """Load a transformer from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify(self):
        """Verify that :meth:`compress()` and :meth:`decompress()` are
        consistent in the sense that the range of :meth:`decompress()` is in
        the domain of :meth:`compress()` and that :meth:`project()` defines
        a projection operator, i.e., ``project(project(Q)) = project(Q)``.
        """
        if (n := self.full_state_dimension) is None:
            raise AttributeError("basis not trained, call fit()")
        states = np.random.random((n, 20))
        statevec = states[:, 0]

        # Verify compress().
        states_compressed = self.compress(states)
        if states_compressed.shape[1] != states.shape[1]:
            raise errors.VerificationError(
                "compress(states).shape[1] != states.shape[1]"
            )
        statevec_compressed = self.compress(statevec)
        if np.ndim(statevec_compressed) != 1:
            raise errors.VerificationError(
                "compress(single_state_vector).ndim != 1"
            )

        # Verify decompress().
        states_projected = self.decompress(states_compressed)
        if states_projected.shape != states.shape:
            raise errors.VerificationError(
                "decompress(compress(states)).shape != states.shape"
            )
        statevec_projected = self.decompress(statevec_compressed)
        if np.ndim(statevec_projected) != 1:
            raise errors.VerificationError(
                "decompress(compress(single_state_vector)).ndim != 1"
            )
        self._verify_locs(states_compressed, states_projected)

        # Verify project().
        states_projected2 = self.project(states_projected)
        if not np.allclose(states_projected2, states_projected):
            raise errors.VerificationError(
                "project(project(states)) != project(states)"
            )
        print("compress() and decompress() are consistent")

    def _verify_locs(self, states_compressed, states_projected):
        """Verification of decompress() with locs != None."""
        n = states_projected.shape[0]
        locs = np.sort(np.random.choice(n, size=(n // 3), replace=False))
        states_projected_at_locs = states_projected[locs]
        states_at_locs_projected = self.decompress(
            states_compressed,
            locs=locs,
        )
        if states_at_locs_projected.shape != states_projected_at_locs.shape:
            raise errors.VerificationError(
                "decompress(states_compressed, locs).shape "
                "!= decompress(states_compressed)[locs].shape"
            )
        if not np.allclose(states_at_locs_projected, states_projected_at_locs):
            raise errors.VerificationError(
                "decompress(states_compressed, locs) "
                "!= decompress(states_compressed)[locs]"
            )
