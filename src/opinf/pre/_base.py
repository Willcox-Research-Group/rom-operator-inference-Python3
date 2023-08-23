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
    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        inplace : bool
            If True, overwrite the input data during inverse transformation.
            If False, create a copy of the data to untransform.
        locs : slice or (p,) ndarray of integers or None
            If given, return the untransformed variable at only the specified
            locations (indices). In this case, `inplace` is ignored.

        Returns
        -------
        states: (n, k) ndarray
            Matrix of k untransformed snapshots of dimension n, or the p
            entries of such at the indices specified by `loc`.
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
