# lift/_base.py
"""Template class for lifting transformations."""

__all__ = [
    "LifterTemplate",
]

import abc
import numpy as np
import scipy.linalg as la

from .. import errors, ddt, utils


class LifterTemplate(abc.ABC):
    """Template class for lifting transformations.

    Classes that inherit from this template must implement the methods
    :meth:`lift()` and :meth:`unlift()`. The optional :meth:`lift_ddts` method
    is used by the ROM class when snapshot time derivative data are available
    in the native state variables.

    See :class:`QuadraticLifter` for an example.
    """

    def __str__(self):
        """String representation: class name."""
        return self.__class__.__name__

    def __repr__(self):
        """Unique ID + string representation."""
        return utils.str2repr(self)

    @staticmethod
    @abc.abstractmethod
    def lift(states):  # pragma: no cover
        """Lift the native state variables to the learning variables.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.

        Returns
        -------
        lifted_states : (n_new, k) ndarray
            Learning variables.
        """
        raise NotImplementedError

    @staticmethod
    def lift_ddts(states, ddts):  # pragma: no cover
        """Lift the native state time derivatives to the time derivatives
        of the learning variables.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.
        ddts : (n, k) ndarray
            Time derivatives of the native state variables. Each column
            ``ddts[:, j]`` corresponds to the state vector ``states[:, j]``.

        Returns
        -------
        ddts : (n_new, k) ndarray
            Time derivatives of the learning variables.
        """
        return NotImplemented

    @staticmethod
    @abc.abstractmethod
    def unlift(lifted_states):  # pragma: no cover
        """Recover the native state variables from the learning variables.

        Parameters
        ----------
        lifted_states : (n, k) ndarray
            Learning variables.

        Returns
        -------
        states : (n, k) ndarray
            Native state variables.
        """
        raise NotImplementedError

    # Testing -----------------------------------------------------------------
    def verify(self, states, t=None, tol: float = 1e-4):
        r"""Verify that :meth:`lift` and :meth:`unlift` are consistent and that
        :meth:`lift_ddts`, if implemented, gives valid time derivatives.

        * The :meth:`lift` / :meth:`unlift` consistency check verifies that
          ``unlift(lift(states)) == states``.
        * The :meth:`lift_ddts` consistency check uses :meth:`opinf.ddt.ddt`
          to estimate the time derivatives of the states and the lifted
          states, then verfies that the relative difference between
          ``lift_ddts(states, opinf.ddt.ddt(states, t))`` and
          ``opinf.ddt.ddt(lift(states), t)`` is less than ``tol``.
          If this check fails, consider using a finer time mesh.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.
        t : (k,) ndarray or None
            Time domain corresponding to the states.
            Only required if :meth:`lift_ddts` is implemented.
        tol : float > 0
            Tolerance for the finite difference check of :meth:`lift_ddts`.
            Only used if :meth:`lift_ddts` is implemented.
        """
        # Verify lift() and unlift() are inverses.
        lifted_states = self.lift(states)
        if (k1 := lifted_states.shape[1]) != states.shape[1]:
            raise errors.VerificationError(
                f"{k1} = lift(states).shape[1] "
                f"!= states.shape[1] = {states.shape[1]}"
            )

        unlifted_states = self.unlift(lifted_states)
        if (shape := unlifted_states.shape) != states.shape:
            raise errors.VerificationError(
                f"{shape} = unlift(lift(states)).shape "
                f"!= states.shape = {states.shape}"
            )
        if not np.allclose(unlifted_states, states):
            raise errors.VerificationError("unlift(lift(states)) != states")
        print("lift() and unlift() are consistent")

        # Finite difference checks for lift_ddts().
        if self.lift_ddts(states, states) is NotImplemented:
            return
        if t is None:
            raise ValueError(
                "time domain 't' required for finite difference check"
            )
        lifted_ddts = self.lift_ddts(states, ddt.ddt(states, t))
        if (shape := lifted_ddts.shape) != (shape2 := lifted_states.shape):
            raise errors.VerificationError(
                f"{shape} = lift_ddts(states, ddts).shape "
                f"!= lift(states).shape = {shape2}"
            )
        lddts_est = ddt.ddt(lifted_states, t)
        if (
            diff := la.norm(lifted_ddts - lddts_est) / la.norm(lddts_est)
        ) > tol:
            raise errors.VerificationError(
                "lift_ddts() failed finite difference check,\n\t"
                "|| lift_ddts(states, d/dt[states]) - d/dt[lift(states)] || "
                f" / || d/dt[lift(states)] || = {diff} > tol = {tol}"
            )
        print("lift() and lift_ddts() are consistent")
