# lift/_base.py
"""Template class for lifting transformation managers."""

import abc
import numpy as np

from .. import errors, ddt


class LifterTemplate(abc.ABC):
    """Template class for lifting transformation managers.

    Classes that inherit from this template must implement the methods
    :meth:`lift` and :meth:`unlift`, and may also choose to implement
    :meth:`lift_ddts`.

    See :class:`QuadraticLifter` for an example.
    """

    @staticmethod
    @abc.abstractmethod
    def lift(states):
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
    def lift_ddts(states, ddts):
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
    def unlift(lifted_states):
        """Extract the native state variables from the learning variables.

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
    def verify_consistency(self, states, t=None):
        """Verify (1) that :meth:`lift` and :meth:`unlift` are consistent,
        i.e., that ``unlift(lift(states)) == states``, and
        (2) that :meth:`lift_ddts`, if implemented, gives valid derivatives.


        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.
        t : (k,) ndarray or None
            Time domain corresponding to the states.
            Only required if :meth:`lift_ddts` is implemented.
        """
        # Verify lift() and unlift() are inverses.
        lifted_states = self.lift(states)
        if (k1 := lifted_states.shape[1]) != states.shape[1]:
            raise errors.VerificationError(
                f"{k1} = lift(states).shape[1] != {states.shape[1] = }"
            )

        unlifted_states = self.unlift(lifted_states)
        if (shape := unlifted_states.shape) != states.shape:
            raise errors.VerificationError(
                f"{shape} = unlift(lift(states)).shape != {states.shape = }"
            )
        if not np.allclose(unlifted_states, states):
            raise errors.VerificationError("unlift(lift(states)) != states")

        # Finite difference checks for lift_ddts().
        if self.lift_ddts(states, states) is NotImplemented:
            return
        ddts = ddt.ddt_nonuniform(states, t)
        ddts_lifted = self.lift_ddts(states, ddts)
        ddts_lifted2 = ddt.ddt_nonuniform(lifted_states, t)
        if not np.allclose(ddts_lifted, ddts_lifted2):
            raise errors.VerificationError(
                "ddts_lifted() failed finite difference check"
            )
