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

import warnings
import numpy as np
import scipy.linalg as la

from ..utils import hdf5_savehandle, hdf5_loadhandle
from ._base import _ParametricOperator, _InputMixin, _requires_entries
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
       (\bfmu_1,\Ohat_{\ell}^{(1)}),\ldots,(\Ohat_{\ell}^{(s)}\bfmu_s);\bfmu),

    where :math:`\Ophat_\ell^{(i)} = \Ophat_\ell(\bfmu_i)` for each
    :math:`i=1,\ldots,s`.

    Parent class: :class:`opinf.operators_new._base._ParametricOperator`

    Child classes:

    * :class:`opinf.operators_new.InterpolatedConstantOperator`
    * :class:`opinf.operators_new.InterpolatedLinearOperator`
    * :class:`opinf.operators_new.InterpolatedQuadraticOperator`
    * :class:`opinf.operators_new.InterpolatedCubicOperator`
    * :class:`opinf.operators_new.InterpolatedInputOperator`
    * :class:`opinf.operators_new.InterpolatedStateInputOperator`
    """

    # Initialization ----------------------------------------------------------
    def __init__(self, training_parameters, InterpolatorClass, entries=None):
        """Construct the elementwise operator interpolator.

        Parameters
        ----------
        training_parameters : list of `s` scalars or 1D ndarrays
            Parameter values for which the operators entries are known
            or will be inferred from data.
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from ``scipy.interpolate``.
        entries : list of `s` ndarray, or None
            Operator entries corresponding to the ``training_parameters``.
        """
        _ParametricOperator.__init__(self)

        # Ensure parameter shapes are consistent and store parameters.
        self._check_shape_consistency(
            training_parameters,
            "training parameter",
        )
        self.__parameters = np.array(training_parameters)
        self._set_parameter_dimension_from_data(self.__parameters)
        self.__s = len(self.__parameters)
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
        # TODO: Handle special cases
        # r = 1: (sd,) ndarray or list of s (d,) ndarrays
        # d = 1: (r, s) ndarray or list of s (r,) ndarrays
        # r = d = 1: (s,) ndarray or list of s floats
        if np.ndim(entries) == 2:
            entries = np.split(entries, n_params, axis=1)

        self._check_shape_consistency(entries, "operator entries")
        if (n_matrices := len(entries)) != n_params:
            raise ValueError(
                f"{n_params} = len(training_parameters) "
                f"!= len(entries) = {n_matrices}"
            )

        self.__entries = np.array(
            [self.OperatorClass(A).entries for A in entries]
        )
        self.set_InterpolatorClass(self.__InterpolatorClass)

    def set_InterpolatorClass(self, InterpolatorClass):
        """Set ``InterpolatorClass`` and, if ``entries`` exists, construct the
        interpolator for the operator entries.

        Parameters
        ----------
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from ``scipy.interpolate``.
        """
        if self.entries is not None:
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
        """Shape of the operator entries matrix when evaluated
        at a parameter value.
        """
        return None if self.entries is None else self.entries[0].shape

    @property
    def InterpolatorClass(self):
        """Class for the elementwise interpolation,
        e.g., a class from ``scipy.interpolate``.
        """
        return self.__InterpolatorClass

    @InterpolatorClass.setter
    def InterpolatorClass(self, IC):
        """Set the InterpolatorClass."""
        self.set_InterpolatorClass(IC)

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
        if self.training_parameters.shape != other.training_parameters.shape:
            return False
        if not np.all(self.training_parameters == other.training_parameters):
            return False
        if self.__InterpolatorClass != other.__InterpolatorClass:
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
        r"""Evaluate the operator at the given parameter value,
        :math:`\Ophat_{\ell}(\cdot,\cdot;\bfmu)`.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value :math:`\bfmu` at which to evalute the operator.

        Returns
        -------
        op : {mod}`opinf.operators` operator of type ``OperatorClass``.
            Nonparametric operator corresponding to the parameter value.
        """
        self._check_parametervalue_dimension(parameter)
        return self.OperatorClass(self.interpolator(parameter))

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr, Wr=None):
        r"""Project this operator to a low-dimensional linear space.

        Consider an interpolatory operator

        .. math::
           \f_\ell(\q,\u;\bfmu)
           = \textrm{interpolate}(
           (\bfmu_1,\f_{\ell}^{(1)}(\q,\u)),\ldots,
           (\bfmu_s,\f_{\ell}^{(s)}(\q,\u)); \bfmu),

        where

        * :math:`\q\in\RR^n` is the full-order state,
        * :math:`\u\in\RR^m` is the input,
        * :math:`\bfmu_1,\ldots,\bfmu_s\in\RR^p`
          are the (fixed) training parameter values,
        * :math:`\f_{\ell}^{(i)}(\q,\u) = \f(\q,\u;\bfmu_i)`
          are the operators evaluated at the training parameter values, and
        * :math:`\bfmu\in\RR^p` is a new parameter value
          at which to evaluate the operator.

        Given a *trial basis* :math:`\Vr\in\RR^{n\times r}` and a *test basis*
        :math:`\Wr\in\RR^{n\times r}`, the corresponding *intrusive projection*
        of :math:`\f` is the interpolatory operator

        .. math::
           \fhat_{\ell}(\qhat,\u;\bfmu)
           = \textrm{interpolate}(
           (\bfmu_1,\Wr\trp\f_{\ell}^{(1)}(\Vr\qhat,\u)),\ldots,
           (\bfmu_s,\Wr\trp\f_{\ell}^{(s)}(\Vr\qhat,\u)); \bfmu),

        Here, :math:`\qhat\in\RR^r` is the reduced-order state, which enables
        the low-dimensional state approximation :math:`\q = \Vr\qhat`.
        If :math:`\Wr = \Vr`, the result is called a *Galerkin projection*.
        If :math:`\Wr \neq \Vr`, it is called a *Petrov-Galerkin projection*.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : operator
            New object of the same class as ``self``.
        """
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
            *[
                self.OperatorClass.datablock(Q, U)
                for Q, U in zip(states, inputs)
            ]
        )

    def operator_dimension(self, r, m):
        r"""Number of columns `sd` in the concatenated operator matrix
        :math:`[~\Ohat_{\ell}^{(1)}~~\cdots~~\Ohat_{\ell}^{(s)}~]`.
        """
        return len(self) * self.OperatorClass.operator_dimension(r, m)

    # Model persistence -------------------------------------------------------
    def copy(self):  # pragma: no cover
        """Return a copy of the operator."""
        return self.__class__(
            training_parameters=self.training_parameters.copy(),
            InterpolatorClass=self.__InterpolatorClass,
            entries=self.entries.copy() if self.entries is not None else None,
        )

    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the operator to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        with hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["class"] = self.__class__.__name__
            meta.attrs["InterpolatorClass"] = self.__InterpolatorClass.__name__
            hf.create_dataset(
                "training_parameters", data=self.training_parameters
            )
            if self.entries is not None:
                hf.create_dataset("entries", data=self.entries)

    @classmethod
    def load(cls, loadfile: str, InterpolatorClass):
        """Load a parametric operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from ``scipy.interpolate``.
            A ``UserWarning`` is thrown if this class does not match the
            metadata in the ``loadfile``.

        Returns
        -------
        op : _Operator
            Initialized operator object.
        """
        with hdf5_loadhandle(loadfile) as hf:
            ClassName = hf["meta"].attrs["class"]
            if ClassName != cls.__name__:
                raise TypeError(
                    f"file '{loadfile}' contains '{ClassName}' "
                    f"object, use '{ClassName}.load()'"
                )

            if (SavedClassName := hf["meta"].attrs["InterpolatorClass"]) != (
                InterpolatorClassName := InterpolatorClass.__name__
            ):
                warnings.warn(
                    f"InterpolatorClass={InterpolatorClassName} does not "
                    f"match loadfile InterpolatorClass '{SavedClassName}'",
                    UserWarning,
                )
            return cls(
                hf["training_parameters"][:],
                InterpolatorClass,
                (hf["entries"][:] if "entries" in hf else None),
            )


# Public interpolated operator classes ========================================
class InterpolatedConstantOperator(_InterpolatedOperator):
    r"""Parametric constant operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \chat(\bfmu) \in \RR^r`
    where the parametric dependence is handled with elementwise interpolation.

    .. math::
       \chat(\bfmu) = \textrm{interpolate}(
       (\bfmu_1,\chat^{(1)}),\ldots,(\bfmu_s,\chat^{(s)}); \bfmu)

    Here,

    * :math:`\bfmu_1,\ldots,\bfmu_s\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\chat^{(i)} = \chat(\bfmu_i) \in \RR^r`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators_new.ConstantOperator`.

    Parameters
    ----------
    training_parameters : list of `s` scalars or 1D ndarrays
        Parameter values :math:`\bfmu_1,\ldots,\bfmu_s` for which
        the operators entries are known or will be inferred from data.
    InterpolatorClass : type
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from ``scipy.interpolate``.
    entries : list of `s` ndarrays, or None
        Operator entries :math:`\chat^{(1)},\ldots,\chat^{(s)}`
        corresponding to the ``training_parameters``.
    """
    _OperatorClass = ConstantOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return 0


class InterpolatedLinearOperator(_InterpolatedOperator):
    r"""Parametric linear operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Ahat(\bfmu)\qhat`
    where :math:`\Ahat(\bfmu) \in \RR^{r \times r}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Ahat(\bfmu) = \textrm{interpolate}(
       (\bfmu_1,\Ahat^{(1)}),\ldots,(\bfmu_s,\Ahat^{(s)}); \bfmu)

    Here,

    * :math:`\bfmu_1,\ldots,\bfmu_s\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Ahat^{(i)} = \Ahat(\bfmu_i) \in \RR^{r \times r}`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators_new.LinearOperator`

    Parameters
    ----------
    training_parameters : list of `s` scalars or 1D ndarrays
        Parameter values :math:`\bfmu_1,\ldots,\bfmu_s` for which
        the operators entries are known or will be inferred from data.
    InterpolatorClass : type
        Class for the elementwise interpolation. Must obey the syntax

        .. code-block:: pycon

           >>> interpolator = InterpolatorClass(data_points, data_values)
           >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from ``scipy.interpolate``.
    entries : list of `s` ndarrays, or None
        Operator entries :math:`\Ahat^{(1)},\ldots,\Ahat^{(s)}`
        corresponding to the ``training_parameters``.
    """
    _OperatorClass = LinearOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return 0


class InterpolatedQuadraticOperator(_InterpolatedOperator):
    r"""Parametric quadratic operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Hhat(\bfmu)[\qhat\otimes\qhat]`
    where :math:`\Ahat(\bfmu) \in \RR^{r \times r^2}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Hhat(\bfmu) = \textrm{interpolate}(
       (\bfmu_1,\Hhat^{(1)}),\ldots,(\bfmu_s,\Hhat^{(s)}); \bfmu)

    Here,

    * :math:`\bfmu_1,\ldots,\bfmu_s\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Hhat^{(i)} = \Hhat(\bfmu_i) \in \RR^{r \times r^2}`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators_new.QuadraticOperator`.

    Parameters
    ----------
    training_parameters : list of `s` scalars or 1D ndarrays
        Parameter values :math:`\bfmu_1,\ldots,\bfmu_s` for which
        the operators entries are known or will be inferred from data.
    InterpolatorClass : type
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from ``scipy.interpolate``.
    entries : list of `s` ndarrays, or None
        Operator entries :math:`\Hhat^{(1)},\ldots,\Hhat^{(s)}`
        corresponding to the ``training_parameters``.
    """
    _OperatorClass = QuadraticOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return 0


class InterpolatedCubicOperator(_InterpolatedOperator):
    r"""Parametric cubic operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Ghat(\bfmu)[\qhat\otimes\qhat\otimes\qhat]`
    where :math:`\Ghat(\bfmu) \in \RR^{r \times r^3}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Ghat(\bfmu) = \textrm{interpolate}(
       (\bfmu_1,\Ghat^{(1)}),\ldots,(\bfmu_s,\Ghat^{(s)}); \bfmu)

    Here,

    * :math:`\bfmu_1,\ldots,\bfmu_s\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Ghat^{(i)} = \Ghat(\bfmu_i) \in \RR^{r \times r^3}`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators_new.CubicOperator`.

    Parameters
    ----------
    training_parameters : list of `s` scalars or 1D ndarrays
        Parameter values :math:`\bfmu_1,\ldots,\bfmu_s` for which
        the operators entries are known or will be inferred from data.
    InterpolatorClass : type
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from ``scipy.interpolate``.
    entries : list of `s` ndarrays, or None
        Operator entries :math:`\Ghat^{(1)},\ldots,\Ghat^{(s)}`
        corresponding to the ``training_parameters``.
    """
    _OperatorClass = CubicOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return 0


class InterpolatedInputOperator(_InterpolatedOperator, _InputMixin):
    r"""Parametric input operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Bhat(\bfmu)\u`
    where :math:`\Bhat(\bfmu) \in \RR^{r \times m}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Bhat(\bfmu) = \textrm{interpolate}(
       (\bfmu_1,\Bhat^{(1)}),\ldots,(\bfmu_s,\Bhat^{(s)}); \bfmu)

    Here,

    * :math:`\bfmu_1,\ldots,\bfmu_s\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Bhat^{(i)} = \Bhat(\bfmu_i) \in \RR^{r \times m}`
      are the operator entries evaluated at the training parameter values.


    See :class:`opinf.operators_new.InputOperator`.

    Parameters
    ----------
    training_parameters : list of `s` scalars or 1D ndarrays
        Parameter values :math:`\bfmu_1,\ldots,\bfmu_s` for which
        the operators entries are known or will be inferred from data.
    InterpolatorClass : type
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from ``scipy.interpolate``.
    entries : list of `s` ndarrays, or None
        Operator entries :math:`\Bhat^{(1)},\ldots,\Bhat^{(s)}`
        corresponding to the ``training_parameters``.
    """
    _OperatorClass = InputOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return None if self.entries is None else self.shape[1]


class InterpolatedStateInputOperator(_InterpolatedOperator, _InputMixin):
    r"""Parametric state-input operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Nhat(\bfmu)[\u\otimes\qhat]`
    where :math:`\Nhat(\bfmu) \in \RR^{r \times rm}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Nhat(\bfmu) = \textrm{interpolate}(
       (\bfmu_1,\Nhat^{(1)}),\ldots,(\bfmu_s,\Nhat^{(s)}); \bfmu)

    Here,

    * :math:`\bfmu_1,\ldots,\bfmu_s\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Nhat^{(i)} = \Nhat(\bfmu_i) \in \RR^{r \times rm}`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators_new.StateInputOperator`.

    Parameters
    ----------
    training_parameters : list of `s` scalars or 1D ndarrays
        Parameter values :math:`\bfmu_1,\ldots,\bfmu_s` for which
        the operators entries are known or will be inferred from data.
    InterpolatorClass : type
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from ``scipy.interpolate``.
    entries : list of `s` ndarrays, or None
        Operator entries :math:`\Nhat^{(1)},\ldots,\Nhat^{(s)}`
        corresponding to the ``training_parameters``.
    """
    _OperatorClass = StateInputOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        if self.entries is None:
            return None
        r, rm = self.shape
        return rm // r
