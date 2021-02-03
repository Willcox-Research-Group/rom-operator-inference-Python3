# _core/_affine/_base.py
"""Base for ROMs with affine parametric dependence..

Classes
-------
* AffineOperator
* _AffineMixin(_ParametricMixin)
"""

__all__ = [
            "AffineOperator",
          ]

import numpy as np

from .._base import _ParametricMixin


# Affine operator (public) ====================================================
class AffineOperator:
    """Class for representing a linear operator with affine structure, i.e.,

        A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.

    The matrix A(µ) is constructed by calling the object once the coefficient
    functions and component matrices are set.

    Attributes
    ----------
    coefficient_functions : list of q callables
        Coefficient scalar-valued functions that define the operator.
        Each must take the same sized input and return a scalar.

    matrices : list of q ndarrays, all of the same shape
        Component matrices defining the linear operator.
    """
    def __init__(self, coeffs, matrices):
        """Save the coefficient functions and component matrices.

        Parameters
        ----------
        coeffs : list of q callables
            Coefficient scalar-valued functions that define the operator.
            Each must take the same sized input and return a scalar.

        matrices : list of q ndarrays, all of the same shape
            Component matrices defining the linear operator.
        """
        if any(not callable(θ) for θ in coeffs):
            raise TypeError("coefficients of affine operator must be callable")
        self.__θs = coeffs

        # Check that the right number of terms are included.
        # if (n_coeffs := len(coeffs) != (n_matrices := len(matrices)):
        n_coeffs, n_matrices = len(coeffs), len(matrices)
        if n_coeffs != n_matrices:
            raise ValueError(f"{n_coeffs} = len(coeffs) "
                             f"!= len(matrices) = {n_matrices}")

        # Check that each matrix in the list has the same shape.
        shape = matrices[0].shape
        if any(A.shape != shape for A in matrices):
            raise ValueError("affine component matrix shapes do not match")

        self.__As = matrices

    @property
    def coefficient_functions(self):
        return self.__θs

    @property
    def matrices(self):
        """The component matrices."""
        return self.__As

    @property
    def shape(self):
        """Shape: the shape of the component matrices."""
        return self.matrices[0].shape

    @staticmethod
    def validate_coeffs(θs, µ):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        µ : float or (p,) ndarray
            A test input for the coefficient functions.
        """
        for θ in θs:
            if not callable(θ):
                raise TypeError("coefficient functions of affine operator "
                                 "must be callable")
            elif not np.isscalar(θ(µ)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, µ):
        """Evaluate the affine operator at the given parameter."""
        return np.sum([θi(µ)*Ai for θi,Ai in zip(self.coefficient_functions,
                                                 self.matrices)], axis=0)

    def __len__(self):
        """Length: number of terms in the affine expansion."""
        return len(self.coefficient_functions)

    def __eq__(self, other):
        """Test whether the component matrices of two AffineOperator objects
        are numerically equal. The coefficient functions are *NOT* compared.
        """
        if not isinstance(other, AffineOperator):
            return False
        if len(self) != len(other):
            return False
        return all(np.allclose(left, right)
                   for left, right in zip(self.matrices, other.matrices))


# Affine base mixin (private) =================================================
class _AffineMixin(_ParametricMixin):
    """Mixin class for affinely parametric reduced model classes."""
    # Validation --------------------------------------------------------------
    def _check_affines_keys(self, affines):
        """Check the keys of the affines argument."""
        # Check for unnecessary affine keys.
        surplus = [repr(key) for key in affines if key not in self.modelform]
        if surplus:
            _noun = "key" + ('' if len(surplus) == 1 else 's')
            raise KeyError(f"invalid affine {_noun} {', '.join(surplus)}")
