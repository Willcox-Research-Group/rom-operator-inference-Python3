# operators/_base.py
"""Abstract base classes for operators.

Classes
-------

* _NonparametricOperator: base for monolithic nonparametric operators.
* _InputMixin: Mix-in for operators that act on the input.
"""

__all__ = []

import abc
import functools
import numpy as np

from .. import errors
from ..utils import hdf5_savehandle, hdf5_loadhandle


def _requires_entries(func):
    """Wrapper for Operator methods that require the ``entries`` attribute
    to be initialized first through ``set_entries()``.
    """

    @functools.wraps(func)
    def _decorator(self, *args, **kwargs):
        if self.entries is None:
            raise RuntimeError(
                "operator entries have not been set, "
                "call set_entries() first"
            )
        return func(self, *args, **kwargs)

    return _decorator


# Nonparametric operators =====================================================
class _NonparametricOperator(abc.ABC):
    """Base class for reduced-order model operators that do not depend on
    external parameters. Call the instantiated object to apply the operator
    to an input.
    """

    def __init__(self, entries=None):
        """Initialize empty operator."""
        self._clear()
        self.evaluate = self.__call__
        if entries is not None:
            self._validate_entries(entries)
            self.set_entries(entries)

    @staticmethod
    def _validate_entries(entries):
        """Ensure argument is a NumPy array and screen for NaN, Inf entries."""
        if not isinstance(entries, np.ndarray):
            raise TypeError("operator entries must be NumPy array")
        if np.any(np.isnan(entries)):
            raise ValueError("operator entries must not be NaN")
        elif np.any(np.isinf(entries)):
            raise ValueError("operator entries must not be Inf")

    def _clear(self):
        """Delete operator ``entries`` and related attributes."""
        self.__entries = None

    # Properties --------------------------------------------------------------
    @property
    def entries(self):
        """Discrete representation of the operator."""
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
        if self.shape != other.shape:
            return False
        return np.all(self.entries == other.entries)

    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def _str(self, statestr, inputstr):  # pragma: no cover
        """String representation of the operator, used when printing out the
        structure of a reduced-order model.

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

    # Initialization ----------------------------------------------------------
    @abc.abstractmethod
    def set_entries(self, entries):
        """Set the ``entries`` attribute."""
        self.__entries = entries

    # Evaluation --------------------------------------------------------------
    @abc.abstractmethod
    def __call__(self, state_, input_=None):  # pragma: no cover
        """Apply the operator mapping to the given state / input.

        This method is also accessible as ``apply()``.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        (r,) ndarray
        """
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, state_, input_):  # pragma: no cover
        '''Mirror of __call__().

        Subclasses should define the following method exactly as follows:

        .. code-block:: python

           @functools.wraps(__call__)
           def apply(self, state_, input_):
               """Mirror of __call__()."""
               return self(state_, input_)
        '''
        return self(state_, input_)

    @_requires_entries
    def jacobian(self, state_, input_=None):  # pragma: no cover
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
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian.
        """
        return 0

    # Dimensionality reduction - - - - - - - - - - - - - - - - - - - - - - - -
    @abc.abstractmethod
    def galerkin(self, Vr, Wr, func):
        r"""Return the projection of the operator.

        For a full-order operator
        :math:`\mathcal{F}:\RR^{n}\times\RR^{m}\to\RR^{n}`,
        the projection is the operator
        :math:`\Ophat:\RR^{r}\times\RR^{m}\to\RR^{r}`
        defined by

        .. math::
           \Ophat(\qhat, \u) = \Wr\trp\mathcal{F}(\Vr\qhat, u)

        where
        :math:`\qhat\in\RR^{r}` is the reduced-order state,
        :math:`\u\in\RR^{m}` is the input, and
        :math:`\q\approx\Vr\qhat_{r}\in\RR^{n}` is the reduced-order
        approximation of the full-order state, with trial basis
        :math:`\Vr\in\RR^{n \times r}` (``Vr``)
        and test basis :math:`\Wr\in\RR^{n \times r}` (``Wr``).
        If :math:`\Wr = \Vr`, the result is called a *Galerkin projection*.
        If :math:`\Wr \neq \Vr`, it is called a *Petrov-Galerkin projection*.

        For example, consider the bilinear full-order operator
        :math:`\mathcal{N}(\q,\u) = \N[\u\otimes\q]` where
        :math:`\N\in\RR^{n \times nm}`.
        The Galerkin projection of this operator is the bilinear operator
        :math:`\widehat{\mathcal{N}}(\qhat,\u) = \Wr\trp\N[\u\otimes\Vr\qhat]`,
        which can also be written as
        :math:`\widehat{\mathcal{N}}(\qhat,\u) = \Nhat[\u\otimes\qhat]`
        where :math:`\Nhat = \Wr\trp\N(\I_m\otimes\Vr) \in \RR^{r\times rm}`.

        Subclasses may implement this function as follows:

        .. code-block:: python

           @_requires_entries
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
            entries of the Galerking projection of the operator.

        Returns
        -------
        op : operator
            New object of the same class as ``self``.
        """
        if Wr is None:
            Wr = Vr
        n, r = Wr.shape
        if self.entries.shape[0] == n:
            return self.__class__(func(self.entries, Vr, Wr))
        raise errors.DimensionalityError("basis and operator not aligned")

    # Data matrix construction - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    @abc.abstractmethod
    def datablock(states_, inputs=None):  # pragma: no cover
        r"""Construct the data matrix block corresponding to the operator.

        For an operator :math:`\Ophat(\qhat,\u)`,
        the data matrix block is the matrix :math:`\D` such that

        .. math::
           \left[\begin{array}{c|c|c|c}
           & & & \\
           \Ophat(\qhat_0,\u_0) & \Ophat(\qhat_1,\u_1)
           & \cdots &
           \Ophat(\qhat_{k-1},\u_{k-1})
           \\ & & &
           \end{array}\right]
           =
           \Ohat\D

        where :math:`\Ohat` is a matrix (the operator entries) that is
        *independent of the data* :math:`\{(\qhat_j,\u_j)\}_{j=0}^{k-1}`.

        Child classes should implement this method as a @staticmethod.

        Parameters
        ----------
        states_ : (r, k) or (k,) ndarray
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
    def column_dimension(r, m=None):  # pragma: no cover
        r"""Column dimension of the operator entries.

        Child classes should implement this method as a @staticmethod.

        Parameters
        ----------
        r : int
            Dimension of the reduced-order state space.
        m : int or None
            Number of inputs.
        """
        raise NotImplementedError

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
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
            if self.entries is not None:
                hf.create_dataset("entries", data=self.entries)

    @classmethod
    def load(cls, loadfile):
        """Load an operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored (via ``save()``).
        """
        with hdf5_loadhandle(loadfile) as hf:
            ClassName = hf["meta"].attrs["class"]
            if ClassName != cls.__name__:
                raise TypeError(
                    f"file '{loadfile}' contains '{ClassName}' "
                    f"object, use '{ClassName}.load()"
                )
            return cls(hf["entries"][:] if "entries" in hf else None)


# Parametric operators ========================================================
# TODO


# Mixin for operators acting on inputs ========================================
class _InputMixin(abc.ABC):
    """Mixin for operator classes whose ``apply()`` method acts on the
    ``input_`` argument.
    """

    @abc.abstractmethod
    def m(self):  # pragma: no cover
        """Input dimension. Subclasses should implement this method as follows:

        .. code-block:: python

           @property
           def m(self):
               '''Input dimension.'''
               if self.entries is None:
                   return None
               return  # calculate m from self.entries.
        """
        raise NotImplementedError


def _is_input_operator(obj):
    """Return ``True`` if ``obj`` is an operator class whose ``apply()``
    method uses the ``input_`` argument (by checking that it is derived from
    ``operators._base._InputMixin``).
    """
    return isinstance(obj, _InputMixin)
