# pre/_base.py
"""Template class for transformers."""

__all__ = [
    "TransformerTemplate",
]

import abc
import numpy as np
import scipy.linalg as la

from .. import errors, ddt, utils


class TransformerTemplate(abc.ABC):
    """Template class for transformers.

    Classes that inherit from this template must implement the methods
    :meth:`fit_transform()`, :meth:`transform()`, and
    :meth:`inverse_transform()`. The optional :meth:`transform_ddts()` method
    is used by the ROM class when snapshot time derivative data are available
    in the native state variables.

    See :class:`ShiftScaleTransformer` for an example.

    The default implementation of :meth:`fit()` simply calls
    :meth:`fit_transform()`.

    Parameters
    ----------
    name : str
        Label for the state variable that this transformer acts on.
    """

    def __init__(self, name: str = None):
        """Initialize attributes."""
        self.__n = None
        self.__name = name

    # Properties --------------------------------------------------------------
    @property
    def state_dimension(self):
        r"""Dimension :math:`n` of the state."""
        return self.__n

    @state_dimension.setter
    def state_dimension(self, n):
        """Set the state dimension."""
        self.__n = int(n) if n is not None else None

    @property
    def name(self):
        """Label for the state variable that this transformer acts on."""
        return self.__name

    @name.setter
    def name(self, label):
        """Set the state variable name."""
        self.__name = str(label) if label is not None else None

    def __str__(self) -> str:
        """String representation: scaling type + centering bool."""
        out = [self.__class__.__name__]
        if self.state_dimension is not None:
            out.append(f"(state dimension n = {self.state_dimension:d})")
        else:
            out.append("(call fit() or fit_transform() to train)")
        return " ".join(out)

    def __repr__(self) -> str:
        """Unique ID + string representation."""
        return utils.str2repr(self)

    def _check_shape(self, Q):
        """Verify the shape of the snapshot set Q."""
        if (n := self.state_dimension) is not None and (n2 := Q.shape[0]) != n:
            raise ValueError(
                f"states.shape[0] = {n2:d} != {n:d} = state dimension n"
            )

    # Main routines -----------------------------------------------------------
    def fit(self, states):
        """Learn (but do not apply) the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.

        Returns
        -------
        self
        """
        self.fit_transform(states, inplace=False)
        return self

    @abc.abstractmethod
    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.
        inplace : bool
            If ``True``, overwrite ``states`` during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of `k` `n`-dimensional transformed snapshots.
        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def transform(self, states, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional snapshots, or a single snapshot.
        inplace : bool
            If ``True``, overwrite ``states`` during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, ...) ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
        """
        raise NotImplementedError  # pragma: no cover

    def transform_ddts(self, ddts, inplace=False):
        r"""Apply the learned transformation to snapshot time derivatives.

        If the transformation is denoted by :math:`\mathcal{T}(q)`,
        this function implements :math:`\mathcal{T}'` such that
        :math:`\mathcal{T}'(\ddt q) = \ddt \mathcal{T}(q)`.

        Parameters
        ----------
        ddts : (n, ...) ndarray
            Matrix of `n`-dimensional snapshot time derivatives, or a
            single snapshot time derivative.
        inplace : bool
            If True, overwrite ``ddts`` during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        ddts_transformed : (n, ...) ndarray
            Transformed `n`-dimensional snapshot time derivatives.
        """
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, ...) or (p, ...)  ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
        inplace : bool
            If ``True``, overwrite ``states_transformed`` during the inverse
            transformation. If ``False``, create a copy of the data to
            untransform.
        locs : slice or (p,) ndarray of integers or None
            If given, assume ``states_transformed`` contains the transformed
            snapshots at only the `p` indices described by ``locs``.

        Returns
        -------
        states_untransformed: (n, ...) or (p, ...) ndarray
            Matrix of `n`-dimensional untransformed snapshots, or the `p`
            entries of such at the indices specified by ``locs``.
        """
        raise NotImplementedError  # pragma: no cover

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the transformer to an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    @classmethod
    def load(cls, loadfile):
        """Load a transformer from an HDF5 file."""
        raise NotImplementedError("use pickle/joblib")  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify(self, tol: float = 1e-4):
        r"""Verify that :meth:`transform()` and :meth:`inverse_transform()`
        are consistent and that :meth:`transform_ddts()`, if implemented,
        is consistent with :meth:`transform()`.

        * The :meth:`transform()` / :meth:`inverse_transform()` consistency
          check verifies that
          ``inverse_transform(transform(states)) == states``.
        * The :meth:`transform_ddts()` consistency check uses
          :meth:`opinf.ddt.ddt()` to estimate the time derivatives of the
          states and the transformed states, then verfies that the relative
          difference between
          ``transform_ddts(opinf.ddt.ddt(states, t))`` and
          ``opinf.ddt.ddt(transform(states), t)`` is less than ``tol``, where
          ``t = numpy.linspace(0, 0.1, 20)``.

        Parameters
        ----------
        tol : float > 0
            Tolerance for the finite difference check of
            :meth:`transform_ddts()`.
            Only used if :meth:`transform_ddts()` is implemented.
        """
        if (n := self.state_dimension) is None:
            raise AttributeError(
                "transformer not trained (state_dimension not set), "
                "call fit() or fit_transform()"
            )
        states = np.random.random((n, 20))

        # Verify transform().
        states_transformed = self.transform(states, inplace=False)
        if states_transformed.shape != states.shape:
            raise errors.VerificationError(
                "transform(states).shape != states.shape"
            )
        if states_transformed is states:
            raise errors.VerificationError(
                "transform(states, inplace=False) is states"
            )
        states_copy = states.copy()
        states_transformed = self.transform(states_copy, inplace=True)
        if states_transformed is not states_copy:
            raise errors.VerificationError(
                "transform(states, inplace=True) is not states"
            )

        # Verify inverse_transform().
        states_recovered = self.inverse_transform(
            states_transformed,
            inplace=False,
        )
        if states_recovered.shape != states.shape:
            raise errors.VerificationError(
                "inverse_transform(transform(states)).shape != states.shape"
            )
        if states_recovered is states_transformed:
            raise errors.VerificationError(
                "inverse_transform(states_transformed, inplace=False) "
                "is states_transformed"
            )
        states_transformed_copy = states_transformed.copy()
        states_recovered = self.inverse_transform(
            states_transformed_copy,
            inplace=True,
        )
        if states_recovered is not states_transformed_copy:
            raise errors.VerificationError(
                "inverse_transform(states_transformed, inplace=True) "
                "is not states_transformed"
            )
        if not np.allclose(states_recovered, states):
            raise errors.VerificationError(
                "transform() and inverse_transform() are not inverses"
            )
        self._verify_locs(states, states_transformed)
        print("transform() and inverse_transform() are consistent")

        # Finite difference check for transform_ddts().
        if self.transform_ddts(states) is NotImplemented:
            return
        t = np.linspace(0, 0.1, states.shape[1])
        ddts = ddt.ddt(states, t)
        ddts_transformed = self.transform_ddts(ddts, inplace=False)
        if ddts_transformed is ddts:
            raise errors.VerificationError(
                "transform_ddts(ddts, inplace=False) is ddts"
            )
        ddts_est = ddt.ddt(states_transformed, t)
        if (
            diff := la.norm(ddts_transformed - ddts_est) / la.norm(ddts_est)
        ) > tol:
            raise errors.VerificationError(
                "transform_ddts() failed finite difference check,\n\t"
                "|| transform_ddts(d/dt[states]) - d/dt[transform(states)] || "
                f" / || d/dt[transform(states)] || = {diff} > {tol} = tol"
            )
        ddts_transformed = self.transform_ddts(ddts, inplace=True)
        if ddts_transformed is not ddts:
            raise errors.VerificationError(
                "transform_ddts(ddts, inplace=True) is not ddts"
            )
        print("transform() and transform_ddts() are consistent")

    def _verify_locs(self, states, states_transformed):
        """Verification for inverse_transform() with locs != None"""
        n = states.shape[0]
        locs = np.sort(np.random.choice(n, size=(n // 3), replace=False))
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
        if not np.allclose(states_recovered_at_locs, states_at_locs):
            raise errors.VerificationError(
                "transform() and inverse_transform() are not inverses "
                "(locs != None)"
            )
