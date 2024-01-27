# operators/_base.py
"""Abstract base classes for operators."""

__all__ = [
    "is_nonparametric",
    "has_inputs",
    "is_parametric",
]

import abc
import numpy as np
import scipy.sparse as sparse

from .. import errors, utils


# Nonparametric operators =====================================================
class _NonparametricOperator(abc.ABC):
    """Base class for operators that do not depend on external parameters.

    Child classes:

    * :class:`opinf.operators.ConstantOperator`
    * :class:`opinf.operators.LinearOperator`
    * :class:`opinf.operators.QuadraticOperator`
    * :class:`opinf.operators.CubicOperator`
    * :class:`opinf.operators.InputOperator`
    * :class:`opinf.operators.StateInputOperator`
    """

    # Initialization ----------------------------------------------------------
    def __init__(self, entries=None):
        """Initialize an empty operator."""
        self._clear()
        if entries is not None:
            self.set_entries(entries)

    def _clear(self):
        """Delete operator ``entries`` and related attributes."""
        self.__entries = None

    @staticmethod
    def _validate_entries(entries):
        """Ensure argument is a NumPy array and screen for NaN, Inf entries."""
        if not (isinstance(entries, np.ndarray) or sparse.issparse(entries)):
            raise TypeError(
                "operator entries must be NumPy or scipy.sparse array"
            )
        if np.any(np.isnan(entries)):
            raise ValueError("operator entries must not be NaN")
        elif np.any(np.isinf(entries)):
            raise ValueError("operator entries must not be Inf")

    @abc.abstractmethod
    def set_entries(self, entries):
        """Set the ``entries`` attribute."""
        self.__entries = entries

    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        r"""
        Discrete representation of the operator, the matrix :math:`\Ohat`.
        """
        return self.__entries

    @entries.setter
    def entries(self, entries):
        """Set the ``entries`` attribute."""
        self.set_entries(entries)

    @entries.deleter
    def entries(self):
        """Reset the ``entries`` attribute."""
        self._clear()

    @property
    def shape(self):
        """Shape of the operator entries array."""
        return None if self.entries is None else self.entries.shape

    @property
    def state_dimension(self):
        r"""Dimension of the state :math:`\qhat` that the operator acts on."""
        return None if self.entries is None else self.entries.shape[0]

    # Magic methods -----------------------------------------------------------
    def __getitem__(self, key):
        """Slice into the entries of the operator."""
        return None if self.entries is None else self.entries[key]

    def __eq__(self, other):
        """Two operator objects are equal if they are of the same class
        and have the same ``entries`` array.
        """
        if not isinstance(other, self.__class__):
            return False
        if (self.entries is None and other.entries is not None) or (
            self.entries is not None and other.entries is None
        ):
            return False
        if self.entries is not None:
            if self.shape != other.shape:
                return False
            return np.all(self.entries == other.entries)
        return True

    def __add__(self, other):
        """Nonparametric operators are linear in their entries."""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"can't add object of type '{other.__class__.__name__}' "
                f"to object of type '{self.__class__.__name__}'"
            )
        return self.__class__(self.entries + other.entries)

    @abc.abstractmethod
    def _str(self, statestr, inputstr):  # pragma: no cover
        """String representation of the operator, used when printing out the
        structure of a model.

        Parameters
        ----------
        statestr : str
            String representation of the state, e.g., ``"q(t)"`` or ``"q_j"``.
        inputstr : str
            String representation of the input, e.g., ``"u(t)"`` or ``"u_j"``.

        Returns
        -------
        opstr : str
            String representation of the operator acting on the state/input,
            e.g., ``"Aq(t)"`` or ``"Bu(t)"`` or ``"H[q(t) ⊗ q(t)]"``.
        """
        raise NotImplementedError

    # Evaluation --------------------------------------------------------------
    @abc.abstractmethod
    def apply(self, state, input_=None):  # pragma: no cover
        """Apply the operator mapping to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        (r,) ndarray
        """
        raise NotImplementedError

    @utils.requires("entries")
    def jacobian(self, state, input_=None):  # pragma: no cover
        r"""Construct the state Jacobian of the operator.

        If :math:`[\![\q]\!]_{i}` denotes the entry :math:`i` of a vector
        :math:`\q`, then the entries of the state Jacobian are given by

        .. math::
           [\![\ddqhat\Ophat(\qhat,\u)]\!]_{i,j}
           = \frac{\partial}{\partial[\![\qhat]\!]_j}
           [\![\Ophat(\qhat,\u)]\!]_i.

        If a child class does not implement this method, it is assumed that
        the Jacobian is zero (i.e., the operator does not act on the state).

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian.
        """
        return 0

    # Dimensionality reduction ------------------------------------------------
    @abc.abstractmethod
    def galerkin(self, Vr, Wr, func):
        r"""Get the (Petrov-)Galerkin projection of this operator.

        Subclasses may implement this function as follows:

        .. code-block:: python

           @utils.requires("entries")
           def galerkin(self, Vr, Wr=None):
               '''Docstring'''
               return _NonparametricOperator.galerkin(self, Vr, Wr,
                   lambda A, V, W:  # compute Galerkin projection of A.
               )

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.
        func : callable
            Function of the operator entries, Vr, and Wr that returns the
            entries of the Galerkin projection of the operator.

        Returns
        -------
        op : operator
            New object of the same class as ``self``.
        """
        if Wr is None:
            Wr = Vr
        n, _ = Wr.shape
        if self.entries.shape[0] != n:
            raise errors.DimensionalityError("basis and operator not aligned")
        return self.__class__(func(self.entries, Vr, Wr))

    # Operator inference ------------------------------------------------------
    @staticmethod
    @abc.abstractmethod
    def datablock(states, inputs=None):  # pragma: no cover
        r"""Construct the data matrix block corresponding to the operator.

        For a nonparametric operator
        :math:`\Ophat_{\ell}(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat, \u)`,
        the data matrix block corresponding to the data pairs
        :math:`\{(\qhat_j,\u_j)\}_{j=0}^{k-1}` is the matrix

        .. math::
           \D\trp = \left[\begin{array}{c|c|c|c}
           & & & \\
           \d_{\ell}(\qhat_0,\u_0) & \d_{\ell}(\qhat_1,\u_1)
           & \cdots &
           \\d_{\ell}(\qhat_{k-1},\u_{k-1})
           \\ & & &
           \end{array}\right]
           \in \RR^{d \times k}.

        Here, ``states`` is the snapshot matrix
        :math:`[~\qhat_0~~\cdots~~\qhat_{k-1}~]`
        and ``inputs`` is the (optional) input matrix
        :math:`[~\u_0~~\cdots~~\u_{k-1}~]`.

        Child classes should implement this method as a ``@staticmethod``.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors. Each column is a single input vector.
            If one dimensional, it is assumed that :math:`m = 1`.

        Returns
        -------
        block : (d, k) or (d,) ndarray
            Data matrix block. Here, :math:`d` is ``entries.shape[1]``.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def operator_dimension(r: int, m: int = None) -> int:  # pragma: no cover
        r"""Column dimension of the operator entries.

        Child classes should implement this method as a @staticmethod.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.

        Returns
        -------
        Number of columns in the operator entries.
        This is also the number of rows in the data matrix block.
        """
        raise NotImplementedError

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Return a copy of the operator."""
        entries = self.entries.copy() if self.entries is not None else None
        return self.__class__(entries)

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
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["class"] = self.__class__.__name__
            if self.entries is not None:
                hf.create_dataset("entries", data=self.entries)

    @classmethod
    def load(cls, loadfile: str):
        """Load an operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            if (ClassName := hf["meta"].attrs["class"]) != cls.__name__:
                raise TypeError(
                    f"file '{loadfile}' contains '{ClassName}' "
                    f"object, use '{ClassName}.load()"
                )
            return cls(hf["entries"][:] if "entries" in hf else None)


# Mixin for operators acting on inputs ----------------------------------------
class _InputMixin(abc.ABC):
    """Mixin for operator classes whose ``apply()`` method acts on
    the ``input_`` argument.

    Child classes:

    * :class:`opinf.operators.InputOperator`
    * :class:`opinf.operators.StateInputOperator`
    * :class:`opinf.operators.InterpolatedInputOperator`
    * :class:`opinf.operators.InterpolatedStateInputOperator`
    """

    @property
    @abc.abstractmethod
    def input_dimension(self) -> int:  # pragma: no cover
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        raise NotImplementedError


# Parametric operators ========================================================
class _ParametricOperator(abc.ABC):
    r"""Base class for operators that depend on external parameters, i.e.,
    :math:`\Ophat_\ell(\qhat,\u;\bfmu) = \Ohat_\ell(\bfmu)\d_\ell(\qhat,\u)`.

    Evaluating a ``_ParametricOpertor`` at a specific parameter value
    results in a :class:`opinf.operators._base._NonparametricOperator`.

    Examples
    --------
    >>> parametric_operator = MyParametricOperator(init_args)
    >>> nonparametric_operator = parametric_operator.evaluate(parameter_value)
    >>> isinstance(nonparametric_operator, _NonparametricOperator)
    True

    Child classes:

    * :class:`opinf.operators._interpolate._InterpolatedOperator`
    * ``_AffineOperator`` (TODO)
    """

    # Meta properties ---------------------------------------------------------
    # Must be specified by child classes.
    _OperatorClass = NotImplemented

    @property
    def OperatorClass(self):
        """Nonparametric :mod:`opinf.operators` class that represents
        this parametric operator evaluated at a particular parameter value.

        Examples
        --------
        >>> Op = MyParametricOperator(init_args).evaluate(parameter_value)
        >>> isinstance(Op, MyParametricOperator.OperatorClass)
        True
        """
        return self._OperatorClass

    # Initialization ----------------------------------------------------------
    def __init__(self):
        """Initialize the parameter_dimension."""
        self.__p = None

    @abc.abstractmethod
    def _clear(self) -> None:  # pragma: no cover
        """Reset the operator to its post-constructor state."""
        raise NotImplementedError

    def _set_parameter_dimension_from_data(self, parameters):
        """Extract and save the dimension of the parameter space from a set of
        parameter values.

        Parameters
        ----------
        parameters : (s, p) or (s,) ndarray
            Parameter value(s).
        """
        if (dim := len(shape := np.shape(parameters))) == 1:
            self.__p = 1
        elif dim == 2:
            self.__p = shape[1]
        else:
            raise ValueError("parameter values must be scalars or 1D arrays")

    # Verification ------------------------------------------------------------
    @staticmethod
    def _check_shape_consistency(iterable, prefix: str):
        """Ensure that each array in `iterable` has the same shape."""
        shape = np.shape(iterable[0])
        if any(np.shape(A) != shape for A in iterable):
            raise ValueError(f"{prefix} shapes inconsistent")

    # Properties --------------------------------------------------------------
    @property
    @abc.abstractmethod
    def state_dimension(self) -> int:  # pragma: no cover
        r"""Dimension of the state :math:`\qhat` that the operator acts on."""
        raise NotImplementedError

    @property
    def parameter_dimension(self) -> int:
        r"""
        Dimension of the parameters :math:`\bfmu` that the operator acts on.
        """
        return self.__p

    @property
    @abc.abstractmethod
    def shape(self) -> tuple:  # pragma: no cover
        """Shape of the operator entries matrix when evaluated
        at a parameter value.
        """
        raise NotImplementedError

    # Evaluation --------------------------------------------------------------
    def _check_parametervalue_dimension(self, parameter):
        """Ensure a new parameter value has the expected shape."""
        if (pdim := self.parameter_dimension) is None:
            raise RuntimeError("parameter_dimension not set")
        if np.atleast_1d(parameter).shape[0] != pdim:
            raise ValueError(f"expected parameter of shape ({pdim:d},)")

    @abc.abstractmethod
    def evaluate(self, parameter):  # pragma: no cover
        r"""Evaluate the operator at the given parameter value,
        resulting in a nonparametric operator of type ``OperatorClass``.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value :math:`\bfmu` at which to evalute the operator.

        Returns
        -------
        evaluated_operator : {mod}`opinf.operators` nonparametric operator.
            Nonparametric operator corresponding to the parameter value.
        """
        raise NotImplementedError

    def apply(self, parameter, state, input_):
        r"""Apply the operator to the given state and input
        at the specified parameter value,
        :math:`\Ophat_\ell(\qhat,\u;\bfmu)`.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value.
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        (r,) ndarray

        Notes
        -----
        For repeated ``apply()`` calls with the same parameter value, use
        :meth:`evaluate` to first get the nonparametric operator
        corresponding to the parameter value.

        .. code-block::

           # Instead of this...
           >>> values = [parametric_operator.apply(parameter, q, input_)
           ...           for q in list_of_states]
           # ...it is faster to do this.
           >>> operator_at_parameter = parametric_operator.evaluate(parameter)
           >>> values = [operator_at_parameter.apply(q, input_)
           ...           for q in list_of_states]
        """
        return self.evaluate(parameter).apply(state, input_)

    def jacobian(self, parameter, state, input_=None):
        r"""Construct the state Jacobian of the operator,
        :math:`\ddqhat\Ophat_\ell(\qhat,\u;\bfmu)`.

        If :math:`[\![\q]\!]_{i}` denotes the entry :math:`i` of a vector
        :math:`\q`, then the entries of the state Jacobian are given by

        .. math::
           [\![\ddqhat\Ophat_\ell(\qhat,\u;\bfmu)]\!]_{i,j}
           = \frac{\partial}{\partial[\![\qhat]\!]_j}
           [\![\Ophat_\ell(\qhat,\u;\bfmu)]\!]_i.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value.
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian.

        Notes
        -----
        For repeated ``jacobian()`` calls with the same parameter value, use
        :meth:`evaluate` to first get the nonparametric operator
        corresponding to the parameter value.

        .. code-block::

           # Instead of this...
           >>> values = [parametric_operator.jacobian(parameter, q, input_)
           ...           for q in list_of_states]
           # ...it is faster to do this.
           >>> operator_at_parameter = parametric_operator.evaluate(parameter)
           >>> values = [operator_at_parameter.jacobian(q, input_)
           ...           for q in list_of_states]
        """
        return self.evaluate(parameter).jacobian(state, input_)

    # Dimensionality reduction ------------------------------------------------
    @abc.abstractmethod
    def galerkin(self, Vr, Wr=None):  # pragma: no cover
        r"""Get the (Petrov-)Galerkin projection of this operator.

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
        raise NotImplementedError

    # Operator inference ------------------------------------------------------
    @abc.abstractmethod
    def datablock(self, states, inputs=None):  # pragma: no cover
        r"""Return the data matrix block corresponding to the operator.

        Parameters
        ----------
        states : list of s (r, k) ndarrays
            State snapshots for each of the `s` training parameter values.
        inputs : list of s (m, k) ndarrays
            Inputs corresponding to the state snapshots.

        Returns
        -------
        block : ndarray
            Data block for the parametric operator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def operator_dimension(self, r, m):  # pragma: no cover
        """Number of columns in the operator matrix."""
        raise NotImplementedError

    # Model persistence -------------------------------------------------------
    @abc.abstractmethod
    def copy(self):  # pragma: no cover
        """Return a copy of the operator."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(
        self,
        savefile: str,
        overwrite: bool = False,
    ) -> None:  # pragma: no cover
        """Save the operator to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, loadfile: str):  # pragma: no cover
        """Load a parametric operator from an HDF5 file.

        Some classes may require more arguments for operator attributes that
        cannot be serialized. Child classes should implement this method as a
        ``@classmethod``.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.

        Returns
        -------
        op : _Operator
            Initialized operator object.
        """
        raise NotImplementedError


# Utilities ===================================================================
def is_nonparametric(obj) -> bool:
    """Return ``True`` if ``obj`` is a nonparametric operator object."""
    return isinstance(obj, _NonparametricOperator)


def has_inputs(obj) -> bool:
    r"""Return ``True`` if ``obj`` is an operator object whose ``apply()``
    method acts on the ``input_`` argument, i.e.,
    :math:`\Ophat_{\ell}(\qhat,\u)` depends on :math:`\u`.
    """
    return isinstance(obj, _InputMixin)


def is_parametric(obj) -> bool:
    """Return ``True`` if ``obj`` is a parametric operator object."""
    return isinstance(obj, _ParametricOperator)
