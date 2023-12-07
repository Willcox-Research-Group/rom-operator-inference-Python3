# operators/_interpolate.py
"""Classes for parametric operators where the parametric dependence is handled
with element-wise interpolation, i.e.,

    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).
"""

__all__ = [
    "InterpolatedConstantOperator",
    "InterpolatedLinearOperator",
    "InterpolatedQuadraticOperator",
    "InterpolatedCubicOperator",
    "InterpolatedInputOperator",
    "InterpolatedStateInputOperator",
]

import numpy as np
import scipy.linalg as la

from ._base import _ParametricOperator, _requires_entries
from ._nonparametric import (
    ConstantOperator,
    LinearOperator,
    QuadraticOperator,
    CubicOperator,
    InputOperator,
    StateInputOperator,
)


# Base class ==================================================================
class _InterpolatedOperator(_ParametricOperator):
    r"""Base class for parametric operators where the parameter dependence
    is handled with element-wise interpolation.

    For a set of training parameter values :math:`\{\bfmu_i\}_{i=1}^{s}`,
    this type of operator is given by
    :math:`\Ophat_\ell(\qhat, \u, \bfmu) = \Ohat_\ell(\bfmu)\d(\qhat, \u)`
    where :math:`\Ohat_{\ell}(\bfmu)` is calculated by interpolating
    operator entries that correspond to each parameter value:

    .. math::
       \Ohat_{\ell}(\bfmu)
        = \textrm{interpolate}(
       (\bfmu_1,\ldots,\bfmu_s),
       (\Ophat_{\ell}^{(1)},\ldots,\Ophat_{\ell}^{(s)});
       \bfmu),

    where :math:`\Ophat_\ell^{(i)} = \Ophat_\ell(\bfmu_i)` for each
    :math:`i=1,\ldots,s`.
    """

    # Initialization ----------------------------------------------------------
    def __init__(self, training_parameters, InterpolatorClass, entries=None):
        """Construct the elementwise operator interpolator.

        Parameters
        ----------
        training_parameters : list of ``nterms`` scalars or 1D ndarrays
            Parameter values for which the operators entries are known
            or will be inferred from data.
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from ``scipy.interpolate``.
        entries : list of ``nterms`` ndarray, or None
            Operator entries corresponding to the ``training_parameters``.
        """
        _ParametricOperator.__init__(self)

        # Ensure parameter shapes are consistent and store parameters.
        self._check_shape_consistency(
            training_parameters,
            "training parameter",
        )
        self._set_parameter_dimension(training_parameters)
        self.__parameters = np.array(training_parameters)
        self.__s = len(training_parameters)
        self.__InterpolatorClass = InterpolatorClass

        # Set the operator entries if provided.
        if entries is not None:
            self.set_entries(entries)
        else:
            self.__entries = None
            self.__interpolator = None

    def _clear(self) -> None:
        """Reset the operator to its post-constructor state without entries."""
        self.__entries = None
        self.__interpolator = None

    def set_entries(self, entries):
        r"""Set the operator entries, the matrices
        :math:`\Ohat_{\ell}^{(1)},\ldots,\Ohat_{\ell}^{(s)}`.

        Parameters
        ----------
        entries : (r, sd) ndarray or list of s (r, d) ndarrays
            Operator entries, either as a list of matrices or the horizontal
            concatenatation of the list of matrices.
        """
        n_params = len(self.training_parameters)
        if np.ndim(entries) == 2:
            entries = np.split(entries, n_params, axis=1)

        self._check_shape_consistency(entries, "operator entries")
        if (n_matrices := len(entries)) != n_params:
            raise ValueError(
                f"{n_params} = len(parameters) "
                f"!= len(entries) = {n_matrices}"
            )

        self.__entries = np.array(
            [self.OperatorClass(A).entries for A in entries]
        )
        self.set_interpolator(self.__InterpolatorClass)

    @_requires_entries
    def set_interpolator(self, InterpolatorClass):
        """Construct the interpolator for the operator entries.

        Parameters
        ----------
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from ``scipy.interpolate``.
        """
        self.__interpolator = InterpolatorClass(
            self.training_parameters,
            self.entries,
        )
        self.__InterpolatorClass = InterpolatorClass

    # Properties --------------------------------------------------------------
    @property
    def state_dimension(self) -> int:
        r"""Dimension of the state :math:`\qhat` that the operator acts on."""
        return None if self.entries is None else self.entries[0].shape[0]

    @property
    def training_parameters(self):
        """Parameter values for which the operators entries are known."""
        return self.__parameters

    @property
    def entries(self):
        """(s, r, d) ndarray: Operator entries corresponding to the training
        parameters values, i.e., ``entries[i]`` are the operator entries
        corresponding to the parameter value ``training_parameters[i]``.
        """
        return self.__entries

    @entries.setter
    def entries(self, entries):
        """Set the operator entries."""
        self.set_entries(entries)

    @entries.deleter
    def entries(self):
        """Reset the ``entries`` attribute."""
        self._clear()

    @property
    def shape(self) -> tuple:
        """Shape: shape of the operator entries to interpolate."""
        return None if self.entries is None else self.entries[0].shape

    @property
    def interpolator(self):
        """Interpolator object for evaluating the operator at specified
        parameter values.
        """
        return self.__interpolator

    # Magic methods -----------------------------------------------------------
    def __len__(self):
        """Length: number of training data points for the interpolation."""
        return self.__s

    def __eq__(self, other):
        """Test whether the training parameters and operator entries of two
        _InterpolatedOperator objects are the same.
        """
        if not isinstance(other, self.__class__):
            return False
        if len(self) != len(other):
            return False
        if self.training_parameters.shape != other.training_parameters.shape:
            return False
        if not np.all(self.training_parameters == other.training_parameters):
            return False
        if (self.entries is None and other.entries is not None) or (
            self.entries is not None and other.entries is None
        ):
            return False
        if self.entries is not None:
            if self.shape != other.shape:
                return False
            return all(
                np.all(left == right)
                for left, right in zip(self.entries, other.entries)
            )
        return True

    # Evaluation --------------------------------------------------------------
    @_requires_entries
    def evaluate(self, parameter):
        """Return the nonparametric operator corresponding to the parameter."""
        self._check_parameter_dimension(parameter)
        return self.OperatorClass(self.interpolator(parameter))

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr, Wr=None):
        r"""TODO"""
        return self.__class__(
            self.training_parameters,
            self.__InterpolatorClass,
            entries=[
                self.OperatorClass(A).galerkin(Vr, Wr).entries
                for A in self.entries
            ],
        )

    # Operator inference ------------------------------------------------------
    def datablock(self, states, inputs=None):
        r"""Return the data matrix block corresponding to the operator.

        For interpolated operators, this is a block diagonal matrix where the
        :math:`i`-th block is the data block for the corresponding
        nonparametric operator class with the state snapshots and inputs
        for training parameter :math:`\bfmu_i`.

        Parameters
        ----------
        states : list of s (r, k) or (k,) ndarrays
            State snapshots for each of the `s` training parameter values.
            If each snapshot matrix is 1D, it is assumed that :math:`r = 1`.
        inputs : list of s (m, k) or (k,) ndarrays
            Inputs corresponding to the state snapshots.
            If each input matrix is 1D, it is assumed that :math:`m = 1`.

        Returns
        -------
        block : (sd, sk) ndarray
            Data block for the interpolated operator. Here, `d` is the number
            of rows in the data block corresponding to a single training
            parameter value.
        """
        return la.block_diag(
            [
                self.OperatorClass.datablock(Q, U)
                for Q, U in zip(states, inputs)
            ]
        )

    def operator_dimension(self, r, m):
        r"""Number of columns `sd` in the concatenated operator matrix
        :math:`[~\Ophat_{\ell}^{(1)}~~\cdots~~\Ophat_{\ell}^{(s)}~]`.
        """
        return len(self) * self.OperatorClass.operator_dimension(r, m)


# Public interpolated operator classes ========================================
class InterpolatedConstantOperator(_InterpolatedOperator):
    """Constant operator with elementwise interpolation, i.e.,

        c(µ) = Interpolator([µ1, µ2, ...], [c1[i,j], c2[i,j], ...])(µ),

    where c1 is the operator vector corresponding to the parameter µ1, etc.
    """

    _OperatorClass = ConstantOperator


class InterpolatedLinearOperator(_InterpolatedOperator):
    """Linear operator with elementwise interpolation, i.e.,

        A(µ) = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ),

    where A1 is the operator matrix corresponding to the parameter µ1, etc.
    """

    _OperatorClass = LinearOperator


class InterpolatedQuadraticOperator(_InterpolatedOperator):
    """Quadratic operator with elementwise interpolation, i.e.,

        H(µ) = Interpolator([µ1, µ2, ...], [H1[i,j], H2[i,j], ...])(µ),

    where H1 is the operator matrix corresponding to the parameter µ1, etc.
    """

    _OperatorClass = QuadraticOperator


class InterpolatedCubicOperator(_InterpolatedOperator):
    """Cubic operator with elementwise interpolation, i.e.,

        G(µ) = Interpolator([µ1, µ2, ...], [G1[i,j], G2[i,j], ...])(µ),

    where G1 is the operator matrix corresponding to the parameter µ1, etc.
    """

    _OperatorClass = CubicOperator


class InterpolatedInputOperator(_InterpolatedOperator):
    """Cubic operator with elementwise interpolation, i.e.,

        G(µ) = Interpolator([µ1, µ2, ...], [G1[i,j], G2[i,j], ...])(µ),

    where G1 is the operator matrix corresponding to the parameter µ1, etc.
    """

    _OperatorClass = InputOperator


class InterpolatedStateInputOperator(_InterpolatedOperator):
    """Cubic operator with elementwise interpolation, i.e.,

        G(µ) = Interpolator([µ1, µ2, ...], [G1[i,j], G2[i,j], ...])(µ),

    where G1 is the operator matrix corresponding to the parameter µ1, etc.
    """

    _OperatorClass = StateInputOperator
