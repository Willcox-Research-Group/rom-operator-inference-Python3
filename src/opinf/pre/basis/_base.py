# pre/basis/_base.py
"""Base basis class."""

import abc
import scipy.linalg as la


class _BaseBasis(abc.ABC):
    """Abstract base class for all basis classes."""

    # Fitting -----------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, *args, **kwargs):                         # pragma: no cover
        """Construct the basis."""
        raise NotImplementedError

    # Dimension reduction -----------------------------------------------------
    @abc.abstractmethod
    def compress(self, state):                              # pragma: no cover
        """Map high-dimensional states to low-dimensional latent coordinates.

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix.

        Returns
        -------
        state_ : (r,) or (r, k) ndarray
            Low-dimensional latent coordinate vector, or a collection of k
            such vectors organized as the columns of a matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decompress(self, state_, locs=None):                # pragma: no cover
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        state_ : (r,) or (r, k) ndarray
            Low-dimensional latent coordinate vector, or a collection of k
            such vectors organized as the columns of a matrix.
        locs : slice or (p,) ndarray of integers or None
            If given, return the reconstructed state at only the specified
            locations (indices).

        Returns
        -------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix. If `locs` is given, only
            the specified coordinates are returned.
        """
        raise NotImplementedError

    # Projection --------------------------------------------------------------
    def project(self, state):
        """Project a high-dimensional state vector to the subset of the
        high-dimensional space that can be represented by the basis by
        1) expressing the state in low-dimensional latent coordinates, then 2)
        reconstructing the high-dimensional state corresponding to those
        coordinates. That is, ``project(Q)`` is equivalent to
        ``decompress(compress(Q))``.

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of `k` such vectors
            organized as the columns of a matrix.

        Returns
        -------
        state_projected : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of `k` such vectors
            organized as the columns of a matrix, projected to the basis range.
        """
        return self.decompress(self.compress(state))

    def projection_error(self, state, relative=True):
        r"""Compute the error of the basis representation of a state or states:
        ``|| state - project(state) || / || state ||``.

        If ``state`` is one-dimensional then :math:`||\cdot||` is the vector
        2-norm. If ``state`` is two-dimensional then :math:`||\cdot||` is the
        Frobenius norm.

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of `k` such vectors
            organized as the columns of a matrix.
        relative : bool
            If True, return the relative error
            ``|| state - project(state) || / || state ||``.
            If False, return the absolute error
            ``|| state - project(state) ||``.

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

    # Persistence -------------------------------------------------------------
    def save(self, *args, **kwargs):
        """Save the basis to an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")      # pragma: no cover

    def load(self, *args, **kwargs):
        """Load a basis from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")      # pragma: no cover
