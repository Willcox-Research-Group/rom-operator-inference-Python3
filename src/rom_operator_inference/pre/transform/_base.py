# pre/transform/_base.py
"""Base transformer class."""

import abc


class _BaseTransformer(abc.ABC):
    """Abstract base class for all transformer classes."""
    # Main routines -----------------------------------------------------------
    def fit(self, states):
        """Learn (but do not apply) the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.

        Returns
        -------
        self
        """
        self.fit_transform(states)
        return self

    @abc.abstractmethod
    def transform(self, states, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        raise NotImplementedError                           # pragma: no cover

    @abc.abstractmethod
    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        raise NotImplementedError                           # pragma: no cover

    @abc.abstractmethod
    def inverse_transform(self, states_transformed, inplace=False):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        inplace : bool
            If True, overwrite the input data during inverse transformation.
            If False, create a copy of the data to untransform.

        Returns
        -------
        states: (n, k) ndarray
            Matrix of k untransformed snapshots of dimension n.
        """
        raise NotImplementedError                           # pragma: no cover

    # Model persistence -------------------------------------------------------
    def save(self, *args, **kwargs):
        """Save the transformer to an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a transformer from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")


def _check_is_transformer(obj):
    """Raise a RuntimeError if `obj` cannot be used as a transformer."""
    for mtd in _BaseTransformer.__abstractmethods__:
        if not hasattr(obj, mtd):
            raise TypeError(f"transformer missing required method {mtd}()")
