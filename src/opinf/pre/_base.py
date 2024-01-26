# pre/transform/_base.py
"""Template class for transformers."""

__all__ = [
    "TransformerTemplate",
]

import abc
import numpy as np
import scipy.linalg as la

from .. import errors, ddt


class TransformerTemplate(abc.ABC):
    """Template class for transformers.

    Classes that inherit from this template must implement the methods
    :meth:`fit_transform()`, :meth:`transform()`, and
    :meth:`inverse_transform()`. The optional :meth:`transform_ddts()` method
    is used by the ROM class when snapshot time derivative data are available
    in the native state variables.

    See :class:`SnapshotTransformer` for an example.

    The default implementation of :meth:`fit()` simply calls
    :meth:`fit_transform()`.
    """

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
        states_transformed : (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        raise NotImplementedError  # pragma: no cover

    def transform_ddts(self, ddts, inplace=False):
        r"""Apply the learned transformation to snapshot time derivatives.

        If the transformation is denoted :math:`\mathcal{T} : q \mapsto q'`,
        this function implements :math:`\mathcal{T}'` such that
        :math:`\mathcal{T}'(\ddt q) = \ddt \mathcal{T}(q)`.

        Parameters
        ----------
        ddts : (n, k) ndarray
            Matrix of k snapshot time derivatives.
        inplace : bool
            If True, overwrite the input data during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        ddts_transforme : (n, k) ndarray
            Matrix of k transformed snapshot time derivatives.
        """
        return NotImplemented  # pragma: no cover

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
        raise NotImplementedError  # pragma: no cover

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
            If given, assume ``states_transformed`` consists of the transformed
            variables at only the specified locations (indices).
            In this case, `inplace` is ignored.

        Returns
        -------
        states: (n, k) or (p, k) ndarray
            Matrix of k untransformed snapshots of dimension n, or the p
            entries of such at the indices specified by `loc`.
        """
        raise NotImplementedError  # pragma: no cover

    # Model persistence -------------------------------------------------------
    def save(self, *args, **kwargs):
        """Save the transformer to an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a transformer from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify(self, states, t=None, tol: float = 1e-4):
        r"""Verify that :meth:`transform()` and :meth:`inverse_transform()`
        are consistent and that :meth:`transform_ddts()`, if implemented,
        is consistent with

        * The :meth:`transform()` / :meth:`inverse_transform()` consistency
          check verifies that
          ``inverse_transform(transform(states)) == states``.
        * The :meth:`transform_ddts()` consistency check uses
          :meth:`opinf.ddt.ddt()` to estimate the time derivatives of the
          states and the transformed states, then verfies that the relative
          difference between
          ``transform_ddts(opinf.ddt.ddt(states, t))`` and
          ``opinf.ddt.ddt(transform(states), t)`` is less than ``tol``.
          If this check fails, consider using a finer time mesh.

        Parameters
        ----------
        states : (n, k)
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        t : (k,) ndarray or None
            Time domain corresponding to the states.
            Only required if :meth:`transform_ddts()` is implemented.
        tol : float > 0
            Tolerance for the finite difference check of
            :meth:`transform_ddts()`.
            Only used if :meth:`transform_ddts()` is implemented.
        """
        # Verify transform().
        states_transformed = self.transform(states)
        if states_transformed.shape != states.shape:
            raise errors.VerificationError(
                "transform(states).shape != states.shape"
            )

        # Verify inverse_transform().
        states_recovered = self.inverse_transform(states_transformed)
        if states_recovered.shape != states_transformed.shape:
            raise errors.VerificationError(
                "inverse_transform(transform(states)).shape "
                "!= transform(states).shape"
            )

        # Check locs argument of inverse_transform().
        n = states.shape[0]
        locs = np.sort(np.random.choice(n, size=n // 3, replace=False))
        states_transformed_at_locs = states_transformed[locs]
        states_recovered_at_locs = self.inverse_transform(
            states_transformed_at_locs,
            locs=locs,
        )
        states_at_locs = states[locs]
        if states_recovered_at_locs.shape != states_at_locs.shape:
            raise errors.VerificationError(
                "inverse_transform(transform(states)[locs], locs).shape "
                "!= states[locs].shape"
            )

        # Verify that transform() and inverse_transform() are inverses.
        if not np.allclose(states_recovered, states):
            raise errors.VerificationError(
                "transform() and inverse_transform() are not inverses"
            )
        if not np.allclose(states_recovered_at_locs, states_at_locs):
            raise errors.VerificationError(
                "transform() and inverse_transform() are not inverses "
                "(locs != None)"
            )
        print("transform() and inverse_transform() are consistent")

        # Finite difference check for transform_ddts().
        if self.transform_ddts(states) is NotImplemented:
            return
        if t is None:
            raise ValueError(
                "time domain 't' required for finite difference check"
            )
        ddts_transformed = self.transform_ddts(ddt.ddt(states, t))
        ddts_est = ddt.ddt(states_transformed, t)
        if (
            diff := la.norm(ddts_transformed - ddts_est) / la.norm(ddts_est)
        ) > tol:
            raise errors.VerificationError(
                "transform_ddts() failed finite difference check,\n\t"
                "|| transform_ddts(d/dt[states]) - d/dt[transform(states)] || "
                f" / || d/dt[transform(states)] || = {diff} > {tol = }"
            )
        print("transform() and transform_ddts() are consistent")


class _MultivarMixin:
    """Private mixin class for transfomers and bases with multivariate states.

    Parameters
    ----------
    num_variables : int
        Number of variables represented in a single snapshot (number of
        individual transformations to learn). The dimension `n` of the
        snapshots must be evenly divisible by num_variables; for example,
        num_variables=3 means the first n entries of a snapshot correspond to
        the first variable, and the next n entries correspond to the second
        variable, and the last n entries correspond to the third variable.
    variable_names : list of num_variables strings, optional
        Names for each of the `num_variables` variables.
        Defaults to "variable 1", "variable 2", ....

    Attributes
    ----------
    n : int
        Total dimension of the snapshots (all variables).
    ni : int
        Dimension of individual variables, i.e., ni = n / num_variables.

    Notes
    -----
    Child classes must set `n` in their fit() methods.
    """

    def __init__(self, num_variables, variable_names=None):
        """Store variable information."""
        if not np.isscalar(num_variables) or num_variables < 1:
            raise ValueError("num_variables must be a positive integer")
        self.__num_variables = num_variables
        self.variable_names = variable_names
        self.__n = None

    # Properties --------------------------------------------------------------
    @property
    def num_variables(self):
        """Number of variables represented in a single snapshot."""
        return self.__num_variables

    @property
    def variable_names(self):
        """Names for each of the `num_variables` variables."""
        return self.__variable_names

    @variable_names.setter
    def variable_names(self, names):
        if names is None:
            names = [f"variable {i+1}" for i in range(self.num_variables)]
        if not isinstance(names, list) or len(names) != self.num_variables:
            raise TypeError(
                "variable_names must be a list of"
                f" length {self.num_variables}"
            )
        self.__variable_names = names

    @property
    def n(self):
        """Total dimension of the snapshots (all variables)."""
        return self.__n

    @n.setter
    def n(self, nn):
        """Set the total and individual variable dimensions."""
        if nn % self.num_variables != 0:
            raise ValueError("n must be evenly divisible by num_variables")
        self.__n = nn

    @property
    def ni(self):
        """Dimension of individual variables, i.e., ni = n / num_variables."""
        return None if self.n is None else self.n // self.num_variables

    # Convenience methods -----------------------------------------------------
    def get_varslice(self, var):
        """Get the indices (as a slice) where the specified variable resides.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.

        Returns
        -------
        s : slice
            Slice object for accessing the specified variable, i.e.,
            variable = state[s] for a single snapshot or
            variable = states[:, s] for a collection of snapshots.
        """
        if var in self.variable_names:
            var = self.variable_names.index(var)
        return slice(var * self.ni, (var + 1) * self.ni)

    def get_var(self, var, states):
        """Extract the ith variable from the states.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.
        states : (n, ...) ndarray

        Returns
        -------
        states_var : ndarray, shape (n, num_states)
        """
        self._check_shape(states)
        return states[..., self.get_varslice(var)]

    def _check_shape(self, Q):
        """Verify the shape of the snapshot set Q."""
        if Q.shape[0] != self.n:
            raise ValueError(
                f"states.shape[0] = {Q.shape[0]:d} "
                f"!= {self.num_variables} * {self.ni} "
                "= num_variables * n_i"
            )
