# from .. import utils
from ._base import OpInfOperator

import numpy as np

# import scipy.linalg as la
# import scipy.sparse as sparse
from scipy.special import comb

__all__ = ["PolyOperator"]


class PolyOperator(OpInfOperator):

    def __init__(self, polynomial_order, entries=None):
        """Initialize an empty operator."""
        if polynomial_order < 0 or (
            not np.isclose(polynomial_order, int(polynomial_order))
        ):
            raise ValueError(
                "expected non-negative integer polynomial order"
                + f" polynomial_order. Got p={polynomial_order}"
            )

        self.polynomial_order = polynomial_order

        super().__init__(entries=entries)

    @staticmethod
    def operator_dimension(r: int, p: int, m=None) -> int:
        """
        computes the number of non-redundant terms in a vector of length r
        that is taken to the power p with the Kronecker product

        Parameters
        ----------
        r : int
            State dimension.
        p : int
            Polynomial order.
        m : int or None
            Input dimension -- currently not used
        """
        if r < 0 or (not np.isclose(r, int(r))):
            raise ValueError(
                f"expected non-negative integer reduced dimension r. Got r={r}"
            )

        if p < 0 or (not np.isclose(p, int(p))):
            raise ValueError(
                f"expected non-negative integer polynomial order p. Got p={p}"
            )

        return comb(r, p, repetition=True, exact=True)

    def my_operator_dimension(self, r: int) -> int:
        """
        Like PolyOperator.operator_dimension but uses self.polynomial_order
        for p.
        """
        return PolyOperator.operator_dimension(r=r, p=self.polynomial_order)

    def datablock(states: np.ndarray, inputs=None) -> np.ndarray:

        pass

    def apply(self, state: np.ndarray, input_=None) -> np.ndarray:
        pass
