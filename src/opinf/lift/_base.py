# lift/_base.py
"""Template class for lifting transformation managers."""

import abc


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
