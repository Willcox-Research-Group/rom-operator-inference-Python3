# pre/basis/_base.py
"""Base basis class."""

import abc
import scipy.linalg as la

from ..transform._base import _check_is_transformer


class _BaseBasis(abc.ABC):
    """Abstract base class for all basis classes."""
    def __init__(self, transformer=None):
        """Set the transformer."""
        self.transformer = transformer

    # Transformer -------------------------------------------------------------
    @property
    def transformer(self):
        """Transformer for pre-processing states before dimension reduction."""
        return self.__transformer

    @transformer.setter
    def transformer(self, tf):
        """Set the transformer, ensuring it has the necessary attributes."""
        if tf is not None:
            _check_is_transformer(tf)
        self.__transformer = tf

    # Fitting -----------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, *args, **kwargs):                         # pragma: no cover
        """Construct the basis."""
        raise NotImplementedError

    # Encoding / Decoding -----------------------------------------------------
    @abc.abstractmethod
    def encode(self, state):                                # pragma: no cover
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
    def decode(self, state_):                               # pragma: no cover
        """Map low-dimensional latent coordinates to high-dimensional states.

        Parameters
        ----------
        state_ : (r,) or (r, k) ndarray
            Low-dimensional latent coordinate vector, or a collection of k
            such vectors organized as the columns of a matrix.

        Returns
        -------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix.
        """
        raise NotImplementedError

    # Projection --------------------------------------------------------------
    def project(self, state):
        """Project a high-dimensional state vector to the subset of the high-
        dimensional space that can be represented by the basis by encoding the
        state in low-dimensional latent coordinates, then decoding those
        coordinates: project(Q) = decode(encode(Q)).

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix.

        Returns
        -------
        state_projected : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix, projected to the basis range.
        """
        return self.decode(self.encode(state))

    def projection_error(self, state, relative=True):
        """Compute the error of the basis representation of a state or states:

            err_absolute = || state - project(state) ||
            err_relative = || state - project(state) || / || state ||

        If `state` is one-dimensional then || . || is the vector 2-norm.
        If `state` is two-dimensional then || . || is the Frobenius norm.
        See scipy.linalg.norm().

        Parameters
        ----------
        state : (n,) or (n, k) ndarray
            High-dimensional state vector, or a collection of k such vectors
            organized as the columns of a matrix.
        relative : bool
            If True, normalize the error by the norm of the original state.

        Returns
        -------
            Relative error of the projection (relative=True) or
            absolute error of the projection (relative=False).
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
