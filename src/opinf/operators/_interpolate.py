# operators/_interpolate.py
"""Classes for OpInf parametric operators where the parametric dependence is
handled with element-wise interpolation.
"""

__all__ = [
    "InterpConstantOperator",
    "InterpLinearOperator",
    "InterpQuadraticOperator",
    "InterpCubicOperator",
    "InterpInputOperator",
    "InterpStateInputOperator",
    # Deprecations:
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
import scipy.interpolate as spinterp

from .. import errors, utils
from ._base import ParametricOpInfOperator, InputMixin
from ._nonparametric import (
    ConstantOperator,
    LinearOperator,
    QuadraticOperator,
    CubicOperator,
    InputOperator,
    StateInputOperator,
)


# Base class ==================================================================
class _InterpOperator(ParametricOpInfOperator):
    r"""Base class for parametric operators where the parameter dependence
    is handled with element-wise interpolation.

    For a set of training parameter values :math:`\{\bfmu_i\}_{i=0}^{s-1}`,
    this type of operator is given by
    :math:`\Ophat_\ell(\qhat, \u, \bfmu) = \Ohat_\ell(\bfmu)\d(\qhat, \u)`
    where :math:`\Ohat_{\ell}(\bfmu)` is calculated by interpolating the
    operator matrix entries that correspond to each parameter value:

    .. math::
       \Ohat_{\ell}(\bfmu)
        = \textrm{interpolate}(
       (\bfmu_0,\Ohat_{\ell}^{(0)}),
       \ldots,(\Ohat_{\ell}^{(s-1)}\bfmu_{s-1});\bfmu),

    where :math:`\Ohat_\ell^{(i)} = \Ohat_\ell(\bfmu_i)` for each
    :math:`i=0,\ldots,s-1`.

    Parent class: :class:`opinf.operators.ParametricOpInfOperator`

    Parameters
    ----------
    training_parameters : list of s scalars or (p,) 1D ndarrays
        Parameter values for which the operator matrix is known
        or will be inferred from data. If not provided in the constructor,
        use :meth:`set_training_parameters` later.
    entries : list of s ndarrays, or None
        Operator matrices corresponding to the ``training_parameters``.
        If not provided in the constructor, use :meth:`set_entries` later.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from :mod:`scipy.interpolate`.
        If ``None`` (default), use :class:`scipy.interpolate.CubicSpline`
        for one-dimensional parameters and
        :class:`scipy.interpolate.LinearNDInterpolator` otherwise.
        If not provided in the constructor, use :meth:`set_interpolator` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.
    """

    # Initialization ----------------------------------------------------------
    def __init__(
        self,
        training_parameters=None,
        entries=None,
        InterpolatorClass: type = None,
        fromblock=False,
    ):
        """Set attributes and, if training parameters and entries are given,
        construct the elementwise operator matrix interpolator.
        """
        ParametricOpInfOperator.__init__(self)

        self.__parameters = None
        self.__interpolator = None
        self.__InterpolatorClass = InterpolatorClass

        if training_parameters is not None:
            self.set_training_parameters(training_parameters)
        if entries is not None:
            self.set_entries(entries, fromblock=fromblock)

    @classmethod
    def _from_operators(
        cls,
        training_parameters,
        operators,
        InterpolatorClass: type = None,
    ):
        """Interpolate an existing set of nonparametric operators with
        populated entries.

        Parameters
        ----------
        operators : list of nonparametric :mod:`opinf.operators` operators
            Operators to interpolate with ``entries`` already set.
        """
        # Check everything is initialized.
        for op in operators:
            if not isinstance(op, cls._OperatorClass):
                raise TypeError(
                    "can only interpolate operators of type "
                    f"'{cls._OperatorClass.__name__}'"
                )
            if op.entries is None:
                raise ValueError(
                    "operators must have entries set in order to interpolate"
                )

        # Extract the entries.
        return cls(
            training_parameters,
            InterpolatorClass=InterpolatorClass,
            entries=[op.entries for op in operators],
            fromblock=False,
        )

    def _clear(self) -> None:
        """Reset the operator to its post-constructor state without entries."""
        ParametricOpInfOperator._clear(self)
        self.__interpolator = None

    # Properties --------------------------------------------------------------
    @property
    def training_parameters(self):
        """Parameter values where the operator matrix is known
        or will be inferred from data.
        """
        return self.__parameters

    @training_parameters.setter
    def training_parameters(self, params):
        """Set the training parameter values."""
        self.set_training_parameters(params)

    def set_training_parameters(self, training_parameters):
        """Set the training parameter values.

        Parameters
        ----------
        training_parameters : list of s scalars or (p,) 1D ndarrays
            Parameter values for which the operator matrix is known
            or will be inferred from data.
        """
        if self.__interpolator is not None:
            raise AttributeError(
                "can't set attribute (entries already set, call _clear())"
            )

        # Check argument dimensions.
        self._check_shape_consistency(
            training_parameters,
            "training_parameters",
        )

        parameters = np.array(training_parameters)
        if parameters.ndim not in (1, 2):
            raise ValueError("parameter values must be scalars or 1D arrays")
        self._set_parameter_dimension_from_values(parameters)
        if parameters.ndim == 2 and parameters.shape[-1] == 1:
            parameters = parameters.ravel()
        self.__parameters = parameters

    @property
    def entries(self) -> np.ndarray:
        """Operator matrices corresponding to the training parameters values,
        i.e., ``entries[i]`` is the operator matrix corresponding to the
        parameter value ``training_parameters[i]``.
        """
        return ParametricOpInfOperator.entries.fget(self)

    def set_entries(self, entries, fromblock: bool = False) -> None:
        r"""Set the operator matrices at the training parameter values.

        Parameters
        ----------
        entries : list of s (r, d) ndarrays, or (r, sd) ndarray
            Operator matrices, either as a list of arrays
            (``fromblock=False``, default)
            or as a horizontal concatenatation of arrays (``fromblock=True``).
        fromblock : bool
            If ``True``, interpret ``entries`` as a horizontal concatenation
            of arrays; if ``False`` (default), interpret ``entries`` as a list
            of arrays.
        """
        if self.training_parameters is None:
            raise AttributeError(
                "training_parameters have not been set, "
                "call set_training_parameters() first"
            )

        # Extract / verify the entries.
        n_params = len(self.__parameters)
        if fromblock:
            if entries.ndim not in (1, 2):
                raise ValueError(
                    "entries must be a 1- or 2-dimensional ndarray "
                    "when fromblock=True"
                )
            entries = np.split(entries, n_params, axis=-1)
        if np.ndim(entries) > 1:
            self._check_shape_consistency(entries, "entries")
        if (n_arrays := len(entries)) != n_params:
            raise ValueError(
                f"{n_params} = len(training_parameters) "
                f"!= len(entries) = {n_arrays}"
            )

        ParametricOpInfOperator.set_entries(
            self,
            np.array([self._OperatorClass(A).entries for A in entries]),
        )
        self.set_interpolator(self.__InterpolatorClass)

    # Interpolation -----------------------------------------------------------
    @property
    def interpolator(self):
        """Interpolator object for evaluating the operator at specified
        parameter values.
        """
        return self.__interpolator

    def set_interpolator(self, InterpolatorClass):
        """Construct the interpolator for the operator matrix.

        Parameters
        ----------
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            This can be, e.g., a class from :mod:`scipy.interpolate`.
        """
        if self.entries is not None:
            params = self.training_parameters
            entries = self.entries

            # Default interpolator classes.
            if InterpolatorClass is None:
                if (dim := params.ndim) == 1:
                    InterpolatorClass = spinterp.CubicSpline
                    paramsort = np.argsort(params)
                    params = params[paramsort]
                    entries = self.entries[paramsort]
                elif dim == 2:
                    InterpolatorClass = spinterp.LinearNDInterpolator

            # Do the interpolation.
            self.__interpolator = InterpolatorClass(params, entries)

        self.__InterpolatorClass = InterpolatorClass

    # Magic methods -----------------------------------------------------------
    def __eq__(self, other) -> bool:
        """Test whether the training parameters and operator matrices of two
        _InterpOperator objects are the same.
        """
        if not isinstance(other, self.__class__):
            return False
        if (
            self.training_parameters is None
            and other.training_parameters is not None
        ) or (
            self.training_parameters is not None
            and other.training_parameters is None
        ):
            return False
        if self.training_parameters is not None:
            if (
                self.training_parameters.shape
                != other.training_parameters.shape
            ):
                return False
            if not np.all(
                self.training_parameters == other.training_parameters
            ):
                return False
        if self.__InterpolatorClass is not other.__InterpolatorClass:
            return False
        if (self.entries is None and other.entries is not None) or (
            self.entries is not None and other.entries is None
        ):
            return False
        if self.entries is not None:
            if self.shape != other.shape:
                return False
            return np.allclose(self.entries, other.entries)
        return True

    def __str__(self):
        lines = ParametricOpInfOperator.__str__(self).split("\n")

        nparams = "None"
        if (params := self.training_parameters) is not None:
            nparams = len(params)
        lines.insert(-1, f"  training parameters: {nparams}")

        ICname = "None"
        if (IC := self.__InterpolatorClass) is not None:
            ICname = IC.__name__
        lines.insert(-1, f"  type(interpolator):  {ICname}")

        return "\n".join(lines)

    # Evaluation --------------------------------------------------------------
    @utils.requires("entries")
    def evaluate(self, parameter):
        r"""Evaluate the operator at the given parameter value.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value :math:`\bfmu` at which to evalute the operator.

        Returns
        -------
        op : nonparametric :mod:`opinf.operators` operator
            Nonparametric operator corresponding to the parameter value.
        """
        self._check_parametervalue_dimension(parameter)
        if self.parameter_dimension == 1 and not np.isscalar(parameter):
            parameter = parameter[0]
        entries = self.interpolator(parameter)
        if entries.ndim == 3:
            entries = entries[0]
        return self._OperatorClass(entries)

    # Dimensionality reduction ------------------------------------------------
    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Project this operator to a low-dimensional linear space.

        Consider an interpolatory operator

        .. math::
           \f_\ell(\q,\u;\bfmu)
           = \textrm{interpolate}(
           (\bfmu_0,\f_{\ell}^{(0)}(\q,\u)),\ldots,
           (\bfmu_{s-1},\f_{\ell}^{(s-1)}(\q,\u)); \bfmu),

        where

        * :math:`\q\in\RR^n` is the full-order state,
        * :math:`\u\in\RR^m` is the input,
        * :math:`\bfmu_0,\ldots,\bfmu_{s-1}\in\RR^p`
          are the (fixed) training parameter values,
        * :math:`\f_{\ell}^{(i)}(\q,\u) = \f_{\ell}(\q,\u;\bfmu_i)`
          is the operators evaluated at the :math:`i`-th training parameter
          values, :math:`i=1,\ldots,s`, and
        * :math:`\bfmu\in\RR^p` is a new parameter value
          at which to evaluate the operator.

        Given a *trial basis* :math:`\Vr\in\RR^{n\times r}` and a *test basis*
        :math:`\Wr\in\RR^{n\times r}`, the corresponding *intrusive projection*
        of :math:`\f` is the interpolatory operator

        .. math::
           \fhat_{\ell}(\qhat,\u;\bfmu)
           = \textrm{interpolate}(
           (\bfmu_0,\Wr\trp\f_{\ell}^{(0)}(\Vr\qhat,\u)),\ldots,
           (\bfmu_{s-1},\Wr\trp\f_{\ell}^{(s-1)}(\Vr\qhat,\u)); \bfmu),

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
            training_parameters=self.training_parameters,
            entries=[
                self._OperatorClass(A).galerkin(Vr, Wr).entries
                for A in self.entries
            ],
            InterpolatorClass=self.__InterpolatorClass,
            fromblock=False,
        )

    # Operator inference ------------------------------------------------------
    @classmethod
    def datablock(cls, parameters, states, inputs=None) -> np.ndarray:
        r"""Return the data matrix block corresponding to the operator.

        For interpolated operators, this is a block diagonal matrix where the
        :math:`i`-th block is the data block for the corresponding
        nonparametric operator class with the state snapshots and inputs
        for training parameter :math:`\bfmu_i`.

        Parameters
        ----------
        parameters : (s, p) ndarray
            Traning parameter values :math:`\bfmu_{0},\ldots,\bfmu_{s-1}`.
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
        if not issubclass(cls, InputMixin):
            inputs = [None] * len(parameters)
        return la.block_diag(
            *[
                cls._OperatorClass.datablock(Q, U)
                for Q, U in zip(states, inputs)
            ]
        )

    @classmethod
    def operator_dimension(cls, s: int, r: int, m: int) -> int:
        r"""Number of columns in the concatenated operator matrix.

        Parameters
        ----------
        s : int
            Number of training parameter values.
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return s * cls._OperatorClass.operator_dimension(r, m)

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Return a copy of the operator."""
        params = self.training_parameters
        return self.__class__(
            training_parameters=params.copy() if params is not None else None,
            InterpolatorClass=self.__InterpolatorClass,
            entries=self.entries.copy() if self.entries is not None else None,
        )

    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the operator to an HDF5 file.

        If the :attr:`interpolator` is not from :mod:`scipy.interpolate`,
        it must be passed to :meth:`load()` when recovering the operator.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["class"] = self.__class__.__name__

            InterpolatorClassName = (
                "NoneType"
                if self.__InterpolatorClass is None
                else self.__InterpolatorClass.__name__
            )
            meta.attrs["InterpolatorClass"] = InterpolatorClassName
            if InterpolatorClassName != "NoneType" and not hasattr(
                spinterp, InterpolatorClassName
            ):
                warnings.warn(
                    "cannot serialize InterpolatorClass "
                    f"'{InterpolatorClassName}', must pass in the class "
                    "when calling load()",
                    errors.OpInfWarning,
                )

            if self.training_parameters is not None:
                hf.create_dataset(
                    "training_parameters", data=self.training_parameters
                )
            if self.entries is not None:
                hf.create_dataset("entries", data=self.entries)

    @classmethod
    def load(cls, loadfile: str, InterpolatorClass: type = None):
        """Load a parametric operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        InterpolatorClass : type or None
            Class for the elementwise interpolation. Must obey the syntax

               >>> interpolator = InterpolatorClass(data_points, data_values)
               >>> interpolator_evaluation = interpolator(new_data_point)

            Not required if the saved operator utilizes a class from
            :mod:`scipy.interpolate`.

        Returns
        -------
        op : _InterpOperator
            Initialized operator object.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            ClassName = hf["meta"].attrs["class"]
            if ClassName != cls.__name__:
                raise TypeError(
                    f"file '{loadfile}' contains '{ClassName}' "
                    f"object, use '{ClassName}.load()'"
                )

            # Get the InterpolatorClass.
            SavedClassName = hf["meta"].attrs["InterpolatorClass"]
            if InterpolatorClass is None:
                # Load from scipy.interpolate.
                if hasattr(spinterp, SavedClassName):
                    InterpolatorClass = getattr(spinterp, SavedClassName)
                elif SavedClassName != "NoneType":
                    raise ValueError(
                        f"unknown InterpolatorClass '{SavedClassName}', "
                        f"call {ClassName}.load({loadfile}, {SavedClassName})"
                    )
            else:
                # Warn the user if the InterpolatorClass does not match.
                if SavedClassName != (
                    InterpolatorClassName := InterpolatorClass.__name__
                ):
                    warnings.warn(
                        f"InterpolatorClass={InterpolatorClassName} does not "
                        f"match loadfile InterpolatorClass '{SavedClassName}'",
                        errors.OpInfWarning,
                    )

            return cls(
                training_parameters=(
                    hf["training_parameters"][:]
                    if "training_parameters" in hf
                    else None
                ),
                entries=(hf["entries"][:] if "entries" in hf else None),
                InterpolatorClass=InterpolatorClass,
                fromblock=False,
            )


# Public interpolated operator classes ========================================
class InterpConstantOperator(_InterpOperator):
    r"""Parametric constant operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \chat(\bfmu) \in \RR^r`
    where the parametric dependence is handled with elementwise interpolation.

    .. math::
       \chat(\bfmu) = \textrm{interpolate}(
       (\bfmu_0,\chat^{(0)}),\ldots,(\bfmu_{s-1},\chat^{(s-1)}); \bfmu)

    Here,

    * :math:`\bfmu_0,\ldots,\bfmu_{s-1}\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\chat^{(i)} = \chat(\bfmu_i) \in \RR^r`
      is the operator vector evaluated at the training parameter values.

    See :class:`opinf.operators.ConstantOperator`.

    Parameters
    ----------
    training_parameters : list of s scalars or (p,) 1D ndarrays
        Parameter values for which the operator vector is known or
        will be inferred from data. If not provided in the constructor,
        use :meth:`set_training_parameters` later.
    entries : list of s ndarrays, or None
        Operator vectors corresponding to the ``training_parameters``.
        If not provided in the constructor, use :meth:`set_entries` later.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from :mod:`scipy.interpolate`.
        If ``None`` (default), use :class:`scipy.interpolate.CubicSpline`
        for one-dimensional parameters and
        :class:`scipy.interpolate.LinearNDInterpolator` otherwise.
        If not provided in the constructor, use :meth:`set_interpolator` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.
    """

    _OperatorClass = ConstantOperator


class InterpLinearOperator(_InterpOperator):
    r"""Parametric linear operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Ahat(\bfmu)\qhat`
    where :math:`\Ahat(\bfmu) \in \RR^{r \times r}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Ahat(\bfmu) = \textrm{interpolate}(
       (\bfmu_0,\Ahat^{(0)}),\ldots,(\bfmu_{s-1},\Ahat^{(s-1)}); \bfmu)

    Here,

    * :math:`\bfmu_0,\ldots,\bfmu_{s-1}\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Ahat^{(i)} = \Ahat(\bfmu_i) \in \RR^{r \times r}`
      is the operator matrix for training parameter value :math:`\bfmu_i`.

    See :class:`opinf.operators.LinearOperator`

    Parameters
    ----------
    training_parameters : list of s scalars or (p,) 1D ndarrays
        Parameter values for which the operator entries are known
        or will be inferred from data. If not provided in the constructor,
        use :meth:`set_training_parameters` later.
    entries : list of s ndarrays, or None
        Operator entries corresponding to the ``training_parameters``.
        If not provided in the constructor, use :meth:`set_entries` later.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from :mod:`scipy.interpolate`.
        If ``None`` (default), use :class:`scipy.interpolate.CubicSpline`
        for one-dimensional parameters and
        :class:`scipy.interpolate.LinearNDInterpolator` otherwise.
        If not provided in the constructor, use :meth:`set_interpolator` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.
    """

    _OperatorClass = LinearOperator


class InterpQuadraticOperator(_InterpOperator):
    r"""Parametric quadratic operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Hhat(\bfmu)[\qhat\otimes\qhat]`
    where :math:`\Ahat(\bfmu) \in \RR^{r \times r^2}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Hhat(\bfmu) = \textrm{interpolate}(
       (\bfmu_0,\Hhat^{(0)}),\ldots,(\bfmu_{s-1},\Hhat^{(s-1)}); \bfmu)

    Here,

    * :math:`\bfmu_0,\ldots,\bfmu_{s-1}\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Hhat^{(i)} = \Hhat(\bfmu_i) \in \RR^{r \times r^2}`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators.QuadraticOperator`.

    Parameters
    ----------
    training_parameters : list of s scalars or (p,) 1D ndarrays
        Parameter values for which the operator entries are known
        or will be inferred from data. If not provided in the constructor,
        use :meth:`set_training_parameters` later.
    entries : list of s ndarrays, or None
        Operator entries corresponding to the ``training_parameters``.
        If not provided in the constructor, use :meth:`set_entries` later.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from :mod:`scipy.interpolate`.
        If ``None`` (default), use :class:`scipy.interpolate.CubicSpline`
        for one-dimensional parameters and
        :class:`scipy.interpolate.LinearNDInterpolator` otherwise.
        If not provided in the constructor, use :meth:`set_interpolator` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.
    """

    _OperatorClass = QuadraticOperator


class InterpCubicOperator(_InterpOperator):
    r"""Parametric cubic operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Ghat(\bfmu)[\qhat\otimes\qhat\otimes\qhat]`
    where :math:`\Ghat(\bfmu) \in \RR^{r \times r^3}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Ghat(\bfmu) = \textrm{interpolate}(
       (\bfmu_0,\Ghat^{(0)}),\ldots,(\bfmu_{s-1},\Ghat^{(s-1)}); \bfmu)

    Here,

    * :math:`\bfmu_0,\ldots,\bfmu_{s-1}\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Ghat^{(i)} = \Ghat(\bfmu_i) \in \RR^{r \times r^3}`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators.CubicOperator`.

    Parameters
    ----------
    training_parameters : list of s scalars or (p,) 1D ndarrays
        Parameter values for which the operator entries are known
        or will be inferred from data. If not provided in the constructor,
        use :meth:`set_training_parameters` later.
    entries : list of s ndarrays, or None
        Operator entries corresponding to the ``training_parameters``.
        If not provided in the constructor, use :meth:`set_entries` later.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from :mod:`scipy.interpolate`.
        If ``None`` (default), use :class:`scipy.interpolate.CubicSpline`
        for one-dimensional parameters and
        :class:`scipy.interpolate.LinearNDInterpolator` otherwise.
        If not provided in the constructor, use :meth:`set_interpolator` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.
    """

    _OperatorClass = CubicOperator


class InterpInputOperator(_InterpOperator, InputMixin):
    r"""Parametric input operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Bhat(\bfmu)\u`
    where :math:`\Bhat(\bfmu) \in \RR^{r \times m}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Bhat(\bfmu) = \textrm{interpolate}(
       (\bfmu_0,\Bhat^{(0)}),\ldots,(\bfmu_{s-1},\Bhat^{(s-1)}); \bfmu)

    Here,

    * :math:`\bfmu_0,\ldots,\bfmu_{s-1}\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Bhat^{(i)} = \Bhat(\bfmu_i) \in \RR^{r \times m}`
      are the operator entries evaluated at the training parameter values.


    See :class:`opinf.operators.InputOperator`.

    Parameters
    ----------
    training_parameters : list of s scalars or (p,) 1D ndarrays
        Parameter values for which the operator entries are known
        or will be inferred from data. If not provided in the constructor,
        use :meth:`set_training_parameters` later.
    entries : list of s ndarrays, or None
        Operator entries corresponding to the ``training_parameters``.
        If not provided in the constructor, use :meth:`set_entries` later.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from :mod:`scipy.interpolate`.
        If ``None`` (default), use :class:`scipy.interpolate.CubicSpline`
        for one-dimensional parameters and
        :class:`scipy.interpolate.LinearNDInterpolator` otherwise.
        If not provided in the constructor, use :meth:`set_interpolator` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.
    """

    _OperatorClass = InputOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return None if self.entries is None else self.shape[1]


class InterpStateInputOperator(_InterpOperator, InputMixin):
    r"""Parametric state-input operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu) = \Nhat(\bfmu)[\u\otimes\qhat]`
    where :math:`\Nhat(\bfmu) \in \RR^{r \times rm}` and
    the parametric dependence is handled with elementwise interpolation.

    .. math::
       \Nhat(\bfmu) = \textrm{interpolate}(
       (\bfmu_0,\Nhat^{(0)}),\ldots,(\bfmu_{s-1},\Nhat^{(s-1)}); \bfmu)

    Here,

    * :math:`\bfmu_0,\ldots,\bfmu_{s-1}\in\RR^p`
      are the (fixed) training parameter values, and
    * :math:`\Nhat^{(i)} = \Nhat(\bfmu_i) \in \RR^{r \times rm}`
      are the operator entries evaluated at the training parameter values.

    See :class:`opinf.operators.StateInputOperator`.

    Parameters
    ----------
    training_parameters : list of s scalars or (p,) 1D ndarrays
        Parameter values for which the operator entries are known
        or will be inferred from data. If not provided in the constructor,
        use :meth:`set_training_parameters` later.
    entries : list of s ndarrays, or None
        Operator entries corresponding to the ``training_parameters``.
        If not provided in the constructor, use :meth:`set_entries` later.
    InterpolatorClass : type or None
        Class for the elementwise interpolation. Must obey the syntax

            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)

        This can be, e.g., a class from :mod:`scipy.interpolate`.
        If ``None`` (default), use :class:`scipy.interpolate.CubicSpline`
        for one-dimensional parameters and
        :class:`scipy.interpolate.LinearNDInterpolator` otherwise.
        If not provided in the constructor, use :meth:`set_interpolator` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.
    """

    _OperatorClass = StateInputOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        if self.entries is None:
            return None
        r, rm = self.shape
        return rm // r


# Utilities ===================================================================
def is_interpolated(obj) -> bool:
    """Return ``True`` if ``obj`` is a interpolated operator object."""
    return isinstance(obj, _InterpOperator)


def nonparametric_to_interpolated(OpClass: type) -> type:
    """Get the interpolated operator class corresponding to a nonparametric
    operator class.

    """
    for InterpClassName in __all__:
        InterpClass = eval(InterpClassName)
        if not isinstance(InterpClass, type) or not issubclass(
            InterpClass, _InterpOperator
        ):  # pragma: no cover
            continue
        if InterpClass._OperatorClass is OpClass:
            return InterpClass
    raise TypeError(
        f"_InterpOperator for class '{OpClass.__name__}' not found"
    )


# Deprecations ================================================================
class InterpolatedConstantOperator(InterpConstantOperator):
    def __init__(
        self,
        training_parameters=None,
        entries=None,
        InterpolatorClass: type = None,
        fromblock=False,
    ):
        warnings.warn(
            "InterpolatedConstantOperator has been renamed "
            "and will be removed in an upcoming release, use "
            "InterpConstantOperator",
            DeprecationWarning,
        )
        InterpConstantOperator.__init__(
            self,
            training_parameters=training_parameters,
            entries=entries,
            InterpolatorClass=InterpolatorClass,
            fromblock=fromblock,
        )


class InterpolatedLinearOperator(InterpLinearOperator):
    def __init__(
        self,
        training_parameters=None,
        entries=None,
        InterpolatorClass: type = None,
        fromblock=False,
    ):
        warnings.warn(
            "InterpolatedLinearOperator has been renamed "
            "and will be removed in an upcoming release, use "
            "InterpLinearOperator",
            DeprecationWarning,
        )
        InterpLinearOperator.__init__(
            self,
            training_parameters=training_parameters,
            entries=entries,
            InterpolatorClass=InterpolatorClass,
            fromblock=fromblock,
        )


class InterpolatedQuadraticOperator(InterpQuadraticOperator):
    def __init__(
        self,
        training_parameters=None,
        entries=None,
        InterpolatorClass: type = None,
        fromblock=False,
    ):
        warnings.warn(
            "InterpolatedQuadraticOperator has been renamed "
            "and will be removed in an upcoming release, use "
            "InterpQuadraticOperator",
            DeprecationWarning,
        )
        InterpQuadraticOperator.__init__(
            self,
            training_parameters=training_parameters,
            entries=entries,
            InterpolatorClass=InterpolatorClass,
            fromblock=fromblock,
        )


class InterpolatedCubicOperator(InterpCubicOperator):
    def __init__(
        self,
        training_parameters=None,
        entries=None,
        InterpolatorClass: type = None,
        fromblock=False,
    ):
        warnings.warn(
            "InterpolatedCubicOperator has been renamed "
            "and will be removed in an upcoming release, use "
            "InterpCubicOperator",
            DeprecationWarning,
        )
        InterpCubicOperator.__init__(
            self,
            training_parameters=training_parameters,
            entries=entries,
            InterpolatorClass=InterpolatorClass,
            fromblock=fromblock,
        )


class InterpolatedInputOperator(InterpInputOperator):
    def __init__(
        self,
        training_parameters=None,
        entries=None,
        InterpolatorClass: type = None,
        fromblock=False,
    ):
        warnings.warn(
            "InterpolatedInputOperator has been renamed "
            "and will be removed in an upcoming release, use "
            "InterpInputOperator",
            DeprecationWarning,
        )
        InterpInputOperator.__init__(
            self,
            training_parameters=training_parameters,
            entries=entries,
            InterpolatorClass=InterpolatorClass,
            fromblock=fromblock,
        )


class InterpolatedStateInputOperator(InterpStateInputOperator):
    def __init__(
        self,
        training_parameters=None,
        entries=None,
        InterpolatorClass: type = None,
        fromblock=False,
    ):
        warnings.warn(
            "InterpolatedStateInputOperator has been renamed "
            "and will be removed in an upcoming release, use "
            "InterpStateInputOperator",
            DeprecationWarning,
        )
        InterpStateInputOperator.__init__(
            self,
            training_parameters=training_parameters,
            entries=entries,
            InterpolatorClass=InterpolatorClass,
            fromblock=fromblock,
        )
