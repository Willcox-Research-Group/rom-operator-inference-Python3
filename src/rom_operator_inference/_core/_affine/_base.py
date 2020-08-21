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
    nterms : int
        The number of terms in the sum defining the linear operator.

    coefficient_functions : list of `nterms` callables
        The coefficient scalar-valued functions that define the operator.
        Each must take the same sized input and return a scalar.

    matrices : list of `nterms` ndarrays of the same shape
        The component matrices defining the linear operator.
    """
    def __init__(self, coeffs, matrices=None):
        """Save the coefficient functions and component matrices (optional).

        Parameters
        ----------
        coeffs : list of `nterms` callables
            The coefficient scalar-valued functions that define the operator.
            Each must take the same sized input and return a scalar.

        matrices : list of `nterms` ndarrays of the same shape
            The component matrices defining the linear operator.
            Can also be assigned later by setting the `matrices` attribute.
        """
        self.coefficient_functions = coeffs
        self._nterms = len(coeffs)
        if matrices:
            self.matrices = matrices
        else:
            self._ready = False

    @property
    def nterms(self):
        """The number of component matrices."""
        return self._nterms

    @property
    def matrices(self):
        """The component matrices."""
        return self._matrices

    @matrices.setter
    def matrices(self, ms):
        """Set the component matrices, checking that the shapes are equal."""
        if len(ms) != self.nterms:
            _noun = "matrix" if self.nterms == 1 else "matrices"
            raise ValueError(f"expected {self.nterms} {_noun}, got {len(ms)}")

        # Check that each matrix in the list has the same shape.
        shape = ms[0].shape
        for m in ms:
            if m.shape != shape:
                raise ValueError("affine operator matrix shapes do not match "
                                 f"({m.shape} != {shape})")

        # Store matrix list and shape, and mark as ready (for __call__()).
        self._matrices = ms
        self.shape = shape
        self._ready = True

    def validate_coeffs(self, µ):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        µ : float or (p,) ndarray
            A test input for the coefficient functions.
        """
        for θ in self.coefficient_functions:
            if not callable(θ):
                raise ValueError("coefficients of affine operator must be "
                                 "callable functions")
            elif not np.isscalar(θ(µ)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, µ):
        """Evaluate the affine operator at the given parameter."""
        if not self._ready:
            raise RuntimeError("component matrices not initialized!")
        return np.sum([θi(µ)*Ai for θi,Ai in zip(self.coefficient_functions,
                                                 self.matrices)], axis=0)

    def __eq__(self, other):
        """Test whether the component matrices of two AffineOperator objects
        are numerically equal. The coefficient functions are *NOT* compared.
        """
        if not isinstance(other, AffineOperator):
            return False
        if self.nterms != other.nterms:
            return False
        if not (self._ready and other._ready):
            return False
        return all([np.allclose(self.matrices[l], other.matrices[l])
                                            for l in range(self.nterms)])

# Affine base mixin (private) =================================================
class _AffineMixin(_ParametricMixin):
    """Mixin class for affinely parametric reduced model classes."""

    def _check_affines(self, affines, µ=None):
        """Check the keys of the affines argument."""
        # Check for unnecessary affine keys.
        surplus = [repr(key) for key in affines if key not in self.modelform]
        if surplus:
            _noun = "key" + ('' if len(surplus) == 1 else 's')
            raise KeyError(f"invalid affine {_noun} {', '.join(surplus)}")

        if µ is not None:
            for a in affines.values():
                AffineOperator(a).validate_coeffs(µ)
