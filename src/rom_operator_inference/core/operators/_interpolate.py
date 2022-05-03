# core/operators/_interpolate.py
"""Classes for parametric operators where the parametric dependence is handled
with element-wise interpolation, i.e.,

    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).
"""

__all__ = [
    "InterpolatedConstantOperator",
    "InterpolatedLinearOperator",
    "InterpolatedQuadraticOperator",
    # "InterpolatedCrossQuadraticOperator",
    "InterpolatedCubicOperator",
    "interpolated_operators",
]

import numpy as np

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
    parameters : list of `nterms` scalars or (p,) ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameters`, i.e.,
        `matrices[i]` are the operator entries at the value `parameters[i]`.
    p : int
        Dimension of the parameter space.
    shape : tuple
        Shape of the operator entries.
    OperatorClass : class
        Class of operator to construct, a subclass of
        core.operators._BaseNonparametricOperator.
    interpolator : scipy.interpolate class (or similar)
        Object that constructs the operator at new parameter values, i.e.,
        >>> new_operator_entries = interpolator(new_parameter)
    """
    # Abstract method implementation ------------------------------------------
    def __init__(self, parameters, matrices, InterpolatorClass):
        """Construct the elementwise operator interpolator.

        Parameters
        ----------
        parameters : list of `nterms` scalars or 1D ndarrays
            Parameter values at which the operators matrices are known.
        matrices : list of `nterms` ndarray, all of the same shape.
            Operator entries corresponding to the `parameters`.
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax
            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)
            This is usually a class from scipy.interpolate.
        """
        _BaseParametricOperator.__init__(self)

        # Ensure there are the same number of parameter samples and matrices.
        n_params, n_matrices = len(parameters), len(matrices)
        if n_params != n_matrices:
            raise ValueError(f"{n_params} = len(parameters) "
                             f"!= len(matrices) = {n_matrices}")

        # Preprocess matrices (e.g., nan/inf checking, compression as needed)
        matrices = [self.OperatorClass(A).entries for A in matrices]

        # Ensure parameter / matrix shapes are consistent.
        self._check_shape_consistency(parameters, "parameter sample")
        self._check_shape_consistency(matrices, "operator matrix")
        self._set_parameter_dimension(parameters)

        # Construct the spline.
        self.__parameters = parameters
        self.__matrices = matrices
        self.set_interpolator(InterpolatorClass)

    def __call__(self, parameter):
        """Return the nonparametric operator corresponding to the parameter."""
        self._check_parameter_dimension(parameter)
        return self.OperatorClass(self.interpolator(parameter))

    # Properties --------------------------------------------------------------
    @property
    def parameters(self):
        """Parameter values at which the operators matrices are known."""
        return self.__parameters

    @property
    def matrices(self):
        """Operator matrices corresponding to the parameters."""
        return self.__matrices

    @property
    def shape(self):
        """Shape: shape of the operator matrices to interpolate."""
        return self.matrices[0].shape

    def set_interpolator(self, InterpolatorClass):
        """Construct the interpolator for the operator entries.

        Parameters
        ----------
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax
            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)
            This is usually a class from scipy.interpolate.
        """
        self.__interpolator = InterpolatorClass(self.parameters, self.matrices)

    @property
    def interpolator(self):
        """Interpolator object for evaluating the operator at a parameter."""
        return self.__interpolator

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
        paramshape = np.shape(self.parameters[0])
        if paramshape != np.shape(other.parameters[1]):
            return False
        if any(not np.all(left == right)
               for left, right in zip(self.parameters,
                                      other.parameters)):
            return False
        return all(np.all(left == right)
                   for left, right in zip(self.matrices, other.matrices))


# Public interpolated operator classes ========================================
class InterpolatedConstantOperator(_InterpolatedOperator):
    """Constant operator with elementwise interpolation, i.e.,

        c(µ) = Interpolator([µ1, µ2, ...], [c1[i,j], c2[i,j], ...])(µ),

    where c1 is the operator vector corresponding to the parameter µ1, etc.
    The vector c(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameters : list of `nterms` scalars or (p,) ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameters`, i.e.,
        `matrices[i]` are the operator entries at the value `parameters[i]`.
    p : int
        Dimension of the parameter space.
    shape : tuple
        Shape of the operator entries.
    OperatorClass : class
        Class of operator to construct, a subclass of
        core.operators._BaseNonparametricOperator.
    interpolator : scipy.interpolate class (or similar)
        Object that constructs the operator at new parameter values, i.e.,
        >>> new_operator_entries = interpolator(new_parameter)
    """
    _OperatorClass = ConstantOperator


class InterpolatedLinearOperator(_InterpolatedOperator):
    """Linear operator with elementwise interpolation, i.e.,

        A(µ) = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ),

    where A1 is the operator matrix corresponding to the parameter µ1, etc.
    The matrix A(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameters : list of `nterms` scalars or (p,) ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameters`, i.e.,
        `matrices[i]` are the operator entries at the value `parameters[i]`.
    p : int
        Dimension of the parameter space.
    shape : tuple
        Shape of the operator entries.
    OperatorClass : class
        Class of operator to construct, a subclass of
        core.operators._BaseNonparametricOperator.
    interpolator : scipy.interpolate class (or similar)
        Object that constructs the operator at new parameter values, i.e.,
        >>> new_operator_entries = interpolator(new_parameter)
    """
    _OperatorClass = LinearOperator


class InterpolatedQuadraticOperator(_InterpolatedOperator):
    """Quadratic operator with elementwise interpolation, i.e.,

        H(µ) = Interpolator([µ1, µ2, ...], [H1[i,j], H2[i,j], ...])(µ),

    where H1 is the operator matrix corresponding to the parameter µ1, etc.
    The matrix H(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameters : list of `nterms` scalars or (p,) ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameters`, i.e.,
        `matrices[i]` are the operator entries at the value `parameters[i]`.
    p : int
        Dimension of the parameter space.
    shape : tuple
        Shape of the operator entries.
    OperatorClass : class
        Class of operator to construct, a subclass of
        core.operators._BaseNonparametricOperator.
    interpolator : scipy.interpolate class (or similar)
        Object that constructs the operator at new parameter values, i.e.,
        >>> new_operator_entries = interpolator(new_parameter)
    """
    _OperatorClass = QuadraticOperator


class InterpolatedCubicOperator(_InterpolatedOperator):
    """Cubic operator with elementwise interpolation, i.e.,

        G(µ) = Interpolator([µ1, µ2, ...], [G1[i,j], G2[i,j], ...])(µ),

    where G1 is the operator matrix corresponding to the parameter µ1, etc.
    The matrix G(µ) for a given µ is constructed by calling the object.

    Attributes
    ----------
    parameters : list of `nterms` scalars or (p,) ndarrays
        Parameter values at which the operators matrices are known.
    matrices : list of `nterms` ndarray, all of the same shape.
        Operator matrices corresponding to the `parameters`, i.e.,
        `matrices[i]` are the operator entries at the value `parameters[i]`.
    p : int
        Dimension of the parameter space.
    shape : tuple
        Shape of the operator entries.
    OperatorClass : class
        Class of operator to construct, a subclass of
        core.operators._BaseNonparametricOperator.
    interpolator : scipy.interpolate class (or similar)
        Object that constructs the operator at new parameter values, i.e.,
        >>> new_operator_entries = interpolator(new_parameter)
    """
    _OperatorClass = CubicOperator


# Dictionary relating modelform keys to operator classes.
interpolated_operators = {
    "c": InterpolatedConstantOperator,
    "A": InterpolatedLinearOperator,
    "H": InterpolatedQuadraticOperator,
    "G": InterpolatedCubicOperator,
    "B": InterpolatedLinearOperator,
}
