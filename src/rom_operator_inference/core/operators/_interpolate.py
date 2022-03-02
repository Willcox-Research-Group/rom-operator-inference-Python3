# _interpolate.py
"""Classes for parametric operators where the parametric dependence is handled
with element-wise interpolation, i.e.,

    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).
"""

__all__ = [
    "Spline1dConstantOperator",
    "Spline1dLinearOperator",
    "Spline1dQuadraticOperator",
    # Spline1dCrossQuadraticOperator",
    "Spline1dCubicOperator",
]

import numpy as np
import scipy.interpolate

from ._base import _BaseParametricOperator
from ._nonparametric import (ConstantOperator,
                             LinearOperator,
                             QuadraticOperator,
                             # CrossQuadraticOperator,
                             CubicOperator)


# Base class ==================================================================
class _InterpolatedOperator(_BaseParametricOperator):
    """Base class for parametric operators where the parameter dependence
    is handled with element-wise interpolation, i.e.,

    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).

    Here A1 is the operator matrix corresponding to the parameter µ1, etc.
    The matrix A(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameter_samples : list of `nterms` scalars or ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Component matrices corresponding to the `parameter_samples`.
    OperatorClass : class
        Class of operator to construct, a subclass of
        core.operators._BaseNonparametricOperator.
    InterpolatorClass : scipy.interpolate class
        Type of interpolator to use. Must obey the following syntax.
        >>> interpolator_object = Interpolator(data_points, data_values)
        >>> interpolator_evaluation = interpolator_object(new_data_point)
    """
    # Must be specified by child classes.
    _InterpolatorClass = NotImplemented

    @property
    def InterpolatorClass(self):
        """scipy.interpolate class for the elementwise interpolation."""
        return self._InterpolatorClass

    def __init__(self, parameter_samples, matrices, **kwargs):
        """Construct the interpolator.

        Parameters
        ----------
        parameter_samples : list of `nterms` scalars or ndarrays
            Parameter values at which the operators matrices are known.
        matrices : list of `nterms` ndarray, all of the same shape.
            Operator entries corresponding to the `parameter_samples`.
        kwargs
            Additional arguments for the interpolator.
        """
        _BaseParametricOperator.__init__(self)

        # Ensure there are the same number of parameter samples and matrices.
        n_params, n_matrices = len(parameter_samples), len(matrices)
        if n_params != n_matrices:
            raise ValueError(f"{n_params} = len(parameter_samples) "
                             f"!= len(matrices) = {n_matrices}")

        # Ensure parameter / matrix shapes are consistent.
        self._check_shape_consistency(parameter_samples, "parameter sample")
        self._check_shape_consistency(matrices, "operator matrix")

        # Construct the spline.
        self.__parameter_samples = parameter_samples
        self.__matrices = matrices
        self.__interpolator = self.InterpolatorClass(parameter_samples,
                                                     matrices, **kwargs)

    @property
    def parameter_samples(self):                            # pragma: no cover
        """Parameter values at which the operators matrices are known."""
        return self.__parameter_samples

    @property
    def matrices(self):
        """Operator matrices corresponding to the parameter_samples."""
        return self.__matrices

    @property
    def shape(self):
        """Shape: shape of the operator matrices to interpolate."""
        return self.matrices[0].shape

    @property
    def interpolator(self):
        """Interpolator object for evaluating the operator at a parameter."""
        return self.__interpolator

    def __call__(self, parameter):
        """Return the nonparametric operator corresponding to the parameter."""
        return self.OperatorClass(self.interpolator(parameter))

    def __len__(self):
        """Length: number of data points for the interpolation."""
        return len(self.matrices)

    def __eq__(self, other):
        """Test whether the parameter samples and operator matrices of two
        InterpolatedOperator objects are numerically equal.
        """
        if not isinstance(other, self.__class__):
            return False
        if len(self) != len(other):
            return False
        if self.shape != other.shape:
            return False
        if any(not np.all(left == right)
               for left, right in zip(self.parameter_samples,
                                      other.parameter_samples)):
            return False
        return all(np.all(left == right)
                   for left, right in zip(self.matrices, other.matrices))


# One-dimensional cubic spline operators ======================================
class Spline1dConstantOperator(_InterpolatedOperator):
    """Constant operator with 1D Spline interpolation, i.e.,

        c(µ) = cubic_spline([µ1, µ2, ...], [c1[i,j], c2[i,j], ...])(µ),

    where c1 is the operator vector corresponding to the parameter µ1, etc.
    The vector c(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameter_samples : list of `nterms` scalars or ndarrays
        Parameter values at which the operators vectors are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator vectors corresponding to the `parameter_samples`.
    """
    _OperatorClass = ConstantOperator
    _InterpolatorClass = scipy.interpolate.CubicSpline


class Spline1dLinearOperator(_InterpolatedOperator):
    """Linear operator with elementwise 1D cubic spline interpolation, i.e.,

        A(µ) = cubic_spline([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ),

    where A1 is the operator matrix corresponding to the parameter µ1, etc.
    The matrix A(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameter_samples : list of `nterms` scalars or ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameter_samples`.
    """
    _OperatorClass = LinearOperator
    _InterpolatorClass = scipy.interpolate.CubicSpline


class Spline1dQuadraticOperator(_InterpolatedOperator):
    """Quadratic operator with elementwise 1D cubic spline interpolation, i.e.,

        H(µ) = cubic_spline([µ1, µ2, ...], [H1[i,j], H2[i,j], ...])(µ),

    where H1 is the operator matrix corresponding to the parameter µ1, etc.
    The matrix H(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameter_samples : list of `nterms` scalars or ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameter_samples`.
    """
    _OperatorClass = QuadraticOperator
    _InterpolatorClass = scipy.interpolate.CubicSpline


class Spline1dCubicOperator(_InterpolatedOperator):
    """Cubic operator with elementwise 1D cubic spline interpolation, i.e.,

        G(µ) = cubic_spline([µ1, µ2, ...], [G1[i,j], G2[i,j], ...])(µ),

    where G1 is the operator matrix corresponding to the parameter µ1, etc.
    The matrix G(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameter_samples : list of `nterms` scalars or ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameter_samples`.
    """
    _OperatorClass = CubicOperator
    _InterpolatorClass = scipy.interpolate.CubicSpline


# N-dimensional linear interpolation operators ================================
