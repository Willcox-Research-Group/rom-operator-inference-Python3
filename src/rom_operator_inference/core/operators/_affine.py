# core/operators/_affine.py
"""Classes for operators that depend affinely on external parameters, i.e.,

    A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.
"""

__all__ = [
    "AffineConstantOperator",
    "AffineLinearOperator",
    "AffineQuadraticOperator",
    # AffineCrossQuadraticOperator",
    "AffineCubicOperator",
]

import numpy as np

from ._base import _BaseParametricOperator
from ._nonparametric import (ConstantOperator,
                             LinearOperator,
                             QuadraticOperator,
                             # CrossQuadraticOperator,
                             CubicOperator)


class _AffineOperator(_BaseParametricOperator):
    """Base class for parametric operators with affine structure, i.e.,

        A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.

    The matrix A(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Scalar-valued coefficient functions in each term of the affine
        expansion (θ_{i}'s).
    matrices : list of `nterms` ndarrays, all of the same shape
        Operator matrices in each term of the affine expansion (A_{i}'s).
    """
    def __init__(self, coefficient_functions, matrices):
        """Save the coefficient functions and operator matrices.

        Parameters
        ----------
        coefficient_functions : list of `nterms` callables
            Scalar-valued coefficient functions in each term of the affine
            expansion (θ_{i}'s).
        matrices : list of `nterms` ndarrays, all of the same shape
            Operator matrices in each term of the affine expansion (A_{i}'s).
        """
        _BaseParametricOperator.__init__(self)

        # Ensure that the coefficient functions are callable.
        if any(not callable(theta) for theta in coefficient_functions):
            raise TypeError("coefficient functions of affine operator "
                            "must be callable")
        self.__coefficient_functions = coefficient_functions

        # Check that the right number of terms are included.
        # if (n_coeffs := len(coeffs) != (n_matrices := len(matrices)):
        n_coeffs, n_matrices = len(coefficient_functions), len(matrices)
        if n_coeffs != n_matrices:
            raise ValueError(f"{n_coeffs} = len(coefficient_functions) "
                             f"!= len(matrices) = {n_matrices}")

        # Check that each matrix in the list has the same shape.
        self._check_shape_consistency(matrices, "operator matrix")
        self.__matrices = matrices

    @property
    def coefficient_functions(self):
        """Coefficient scalar-valued functions in the affine expansion."""
        return self.__coefficient_functions

    @property
    def matrices(self):
        """Component matrices in each term of the affine expansion."""
        return self.__matrices

    @property
    def shape(self):
        """Shape: the shape of the operator matrices."""
        return self.matrices[0].shape

    @staticmethod
    def _validate_coefficient_functions(coefficient_functions, parameter):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        coefficient_functions : list of `nterms` callables
            Scalar-valued coefficient functions in each term of the affine
            expansion (θ_{i}'s).
        parameter : (p,) ndarray or float (p = 1).
            Parameter input to use as a test for the coefficient functions (µ).
        """
        for theta in coefficient_functions:
            if not callable(theta):
                raise TypeError("coefficient functions of affine operator "
                                "must be callable")
            elif not np.isscalar(theta(parameter)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, parameter):
        """Evaluate the affine operator at the given parameter."""
        entries = np.sum([thetai(parameter)*Ai for thetai, Ai in zip(
                          self.coefficient_functions, self.matrices)],
                         axis=0)
        return self.OperatorClass(entries)

    def __len__(self):
        """Length: number of terms in the affine expansion."""
        return len(self.matrices)

    def __eq__(self, other):
        """Test whether the operator matrices of two AffineOperator objects
        are numerically equal. Coefficient functions are *NOT* compared.
        """
        if not isinstance(other, self.__class__):
            return False
        if len(self) != len(other):
            return False
        if self.shape != other.shape:
            return False
        return all(np.all(left == right)
                   for left, right in zip(self.matrices, other.matrices))


class AffineConstantOperator(_AffineOperator):
    """Constant operator with affine parametric structure, i.e.,

        c(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * c_{i}.

    The vector c(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Scalar-valued coefficient functions in each term of the affine
        expansion (θ_{i}'s).
    matrices : list of `nterms` ndarrays, all of the same shape
        Operator matrices in each term of the affine expansion (c_{i}'s).
    """
    _OperatorClass = ConstantOperator


class AffineLinearOperator(_AffineOperator):
    """Linear operator with affine parametric structure, i.e.,

        A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.

    The matrix A(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Scalar-valued coefficient functions in each term of the affine
        expansion (θ_{i}'s).
    matrices : list of `nterms` ndarrays, all of the same shape
        Operator matrices in each term of the affine expansion (A_{i}'s).
    """
    _OperatorClass = LinearOperator


class AffineQuadraticOperator(_AffineOperator):
    """Quadratic operator with affine parametric structure, i.e.,

        H(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * H_{i}.

    The matrix H(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Scalar-valued coefficient functions in each term of the affine
        expansion (θ_{i}'s).
    matrices : list of `nterms` ndarrays, all of the same shape
        Operator matrices in each term of the affine expansion (H_{i}'s).
    """
    _OperatorClass = QuadraticOperator


class AffineCubicOperator(_AffineOperator):
    """Cubic operator with affine parametric structure, i.e.,

        G(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * G_{i}.

    The matrix G(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    coefficient_functions : list of `nterms` callables
        Scalar-valued coefficient functions in each term of the affine
        expansion (θ_{i}'s).
    matrices : list of `nterms` ndarrays, all of the same shape
        Operator matrices in each term of the affine expansion (G_{i}'s).
    """
    _OperatorClass = CubicOperator
