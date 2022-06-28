# pre/transform/_base.py
"""Base transformer class."""

import abc


__transformer_required_methods = (
    "fit_transform",
    "transform",
    "inverse_transform",
)


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
    def transform(self, states):                        # pragma: no cover
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, states):                    # pragma: no cover
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        return NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, states_transformed):    # pragma: no cover
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
        raise NotImplementedError

    # Model persistence -------------------------------------------------------
    @abc.abstractmethod
    def save(self, savefile, overwrite=False):               # pragma: no cover
        """Save the transformer to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the transformer in.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, loadfile):                               # pragma: no cover
        """Load a transformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the transformer was stored (via save()).

        Returns
        -------
        _BaseTransformer
        """
        raise NotImplementedError


def _check_is_transformer(obj):
    """Raise a RuntimeError of `obj` cannot be used as a transformer."""
    if isinstance(obj, _BaseTransformer):
        return
    for mtd in __transformer_required_methods:
        if not hasattr(obj, mtd):
            raise TypeError(f"invalid transformer, missing method {mtd}()")
