# lift/_polynomial.py
r"""Polynomial lifting maps, e.g., :math:`q \to (q, q^2, ...)`."""

__all__ = [
    "QuadraticLifter",
    "PolynomialLifter",
]

import numbers
import warnings
import numpy as np

from .. import errors
from ._base import LifterTemplate


class QuadraticLifter(LifterTemplate):
    r"""Quadratic lifting map :math:`q \to (q, q^2)`."""

    @staticmethod
    def lift(states):
        r"""Apply the lifting map :math:`q \to (q, q^2)`.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.

        Returns
        -------
        lifted_states : (2n, k) ndarray
            Learning variables.
        """
        return np.concatenate((states, states**2))

    @staticmethod
    def lift_ddts(states, ddts):
        r"""Get the time derivatives of the lifted variables,
        :math:`\frac{\partial}{\partial t}(q, q^2) = (q_t, 2qq_t)`.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.
        ddts : (n, k) ndarray
            Time derivatives of the native state variables. Each column
            ``ddts[:, j]`` corresponds to the state vector ``states[:, j]``.

        Returns
        -------
        ddts : (2n, k) ndarray
            Time derivatives of the learning variables.
        """
        return np.concatenate((ddts, 2 * states * ddts))

    @staticmethod
    def unlift(lifted_states):
        r"""Apply the reverse lifting map :math:`(q, q^2) \to q`.

        Parameters
        ----------
        lifted_states : (2n, k) ndarray
            Learning variables.

        Returns
        -------
        states : (n, k) ndarray
            Native state variables.
        """
        return np.split(lifted_states, 2, axis=0)[0]


class PolynomialLifter(LifterTemplate):
    r"""Polynomial lifting map :math:`q \to (q, q^2, q^3, ...)`.

    Parameters
    ----------
    orders : tuple
        Polynomial orders in the learning variables. For example,
        ``orders=(1, 2, 4)`` means the lifting transformation is given by
        :math:`q \to (q, q^2, q^4)`. The orders need not be positive integers,
        e.g., ``orders=(-1, 0.5)`` indicates :math:`q \to (1/q, \sqrt{q})`.
    """

    def __init__(self, orders: tuple):
        """Set the polynomial orders."""
        self.orders = orders

    # Properties --------------------------------------------------------------
    @property
    def orders(self) -> tuple:
        r"""Polynomial orders in the learning variables. For example,
        ``orders=(1, 2, 4)`` means the lifting transformation is given by
        :math:`q \to (q, q^2, q^4)`. The orders need not be positive integers,
        e.g., ``orders=(-1, 0.5)`` indicates :math:`q \to (1/q, \sqrt{q})`.
        """
        return self.__orders

    @orders.setter
    def orders(self, ps: tuple):
        """Set the polynomial orders."""
        if isinstance(ps, numbers.Number):
            ps = (int(ps),)
        for p in ps:
            if not isinstance(p, numbers.Number):
                raise TypeError("'orders' must be a sequence of numbers")
        if len(ps) == 1 and ps[0] == 0:
            warnings.warn(
                "q -> q^0 = 1 is not invertible",
                errors.OpInfWarning,
            )

        self.__orders = tuple(ps)
        self.__nvars = len(self.__orders)

    @property
    def num_variables(self) -> int:
        """Number of learning variables."""
        return self.__nvars

    def __str__(self) -> str:
        """String representation: lifting map description"""
        variables = ", ".join(
            [(f"q^{p}" if p != 1 else "q") for p in self.__orders]
        )
        if self.num_variables > 1:
            variables = "(" + variables + ")"
        return f"Lifting map q -> {variables}"

    # Lifting map -------------------------------------------------------------
    def lift(self, states):
        r"""Apply the lifting map :math:`q \to (q, q^2, ...)`.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.

        Returns
        -------
        lifted_states : (num_variables * n, k) ndarray
            Learning variables.
        """
        return np.concatenate([states**p for p in self.__orders])

    def lift_ddts(self, states, ddts):
        """Get the time derivatives of the lifted variables,
        :math:`(q_t, 2qq_t, ...)`.

        Parameters
        ----------
        states : (n, k) ndarray
            Native state variables.
        ddts : (n, k) ndarray
            Time derivatives of the native state variables. Each column
            ``ddts[:, j]`` corresponds to the state vector ``states[:, j]``.

        Returns
        -------
        ddts : (num_variables * n, k) ndarray
            Time derivatives of the learning variables.
        """
        return np.concatenate(
            [p * states ** (p - 1) * ddts for p in self.__orders]
        )

    def unlift(self, lifted_states):
        r"""Apply the reverse lifting map :math:`(q, q^2, ...) \to q`.

        Parameters
        ----------
        lifted_states : (num_variables * n, k) ndarray
            Learning variables.

        Returns
        -------
        states : (n, k) ndarray
            Native state variables.
        """
        variables = np.split(lifted_states, self.num_variables, axis=0)
        if 1 in self.__orders:
            return variables[self.__orders.index(1)]
        for var, p in zip(variables, self.__orders):
            if p != 0:
                return var ** (1 / p)
        raise ZeroDivisionError("q -> q^0 = 1 is not invertible")
