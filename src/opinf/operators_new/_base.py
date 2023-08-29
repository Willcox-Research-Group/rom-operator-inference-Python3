# operators/_base.py
"""Abstract base classes for operators.

Classes
-------
* _BaseNonparametricOperator: base for operators without parameter dependence.
* _BaseParametricOperator: base for operators with parameter dependence.
"""

__all__ = []

import abc
import functools
import numpy as np


def _requires_entries(func):
    """Wrapper for Operator methods that require the ``entries`` attribute
    to be initialized first through ``set_entries()``.
    """
    @functools.wraps(func)
    def _decorator(self, *args, **kwargs):
        if self.entries is None:
            raise RuntimeError("operator entries have not been set, "
                               "call set_entries() first")
        return func(self, *args, **kwargs)

    return _decorator


class _BaseNonparametricOperator(abc.ABC):
    """Base class for reduced-order model operators that do not depend on
    external parameters. Call the instantiated object to evaluate the operator
    on an input.
    """
    def __init__(self, entries=None):
        """Initialize empty operator."""
        self.__entries = None
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

    @property
    def shape(self):
        """Shape of the operator entries array."""
        return self.entries.shape

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
    def _str(self, statestr, inputstr):                     # pragma: no cover
        """String representation of the operator, used when printing out the
        structure of a reduced-order model.

        Parameters
        ----------
        statestr : str
            String representation of the state, e.g., `"q(t)"` or "`q_j`".
        inputstr : str
            String representation of the input, e.g., `"u(t)"` or "`u_j`".

        Returns
        -------
        opstr : str
            String representation of the operator acting on the state/input,
            e.g., Aq(t) or Bu(t) or H[q(t) ⊗ q(t)].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_entries(self, entries):
        """Set the ``entries`` attribute."""
        self.__entries = entries

    @abc.abstractmethod
    def __call__(selfm, state_, input_=None):                # pragma: no cover
        """Apply the operator mapping to the given state / input.

        This method is also accessible as ``evaluate()``.

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
    def evaluate(self, state_, input_):                     # pragma: no cover
        '''Mirror of __call__().

        Subclasses should define the following method exactly as follows:

            @functools.wraps(__call__)
            def evaluate(self, state_, input_):
                """Mirror of __call__()."""
                return self(state_, input_)
        '''
        return self(state_, input_)

    @abc.abstractmethod
    def jacobian(self, state_, input_=None):                # pragma: no cover
        """Construct the Jacobian of the operator at the given state / input.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        (r, r) ndarray
        """
        raise NotImplementedError

    @abc.abstractmethod
    def datablock(state_, input_=None):                     # pragma: no cover
        r"""Return the data matrix block corresponding to the operator.

        Let :math:`\widehat{\mathbf{F}}(\widehat{\mathbf{q}},\mathbf{u})`
        represent the operator acting on a pair of state and input vectors.
        The data matrix block is the matrix
        :math:`\widehat{\mathbf{Z}}` such that the operator inference problem

        .. math::
            \min_{\widehat{\mathbf{F}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}, \mathbf{u}_{j})
            - \widehat{\mathbf{y}}_{j}
            \right\|_{2}^{2},

        can be written equivalently as

        .. math::
            \min_{\widehat{\mathbf{X}}}\left\|
            \widehat{\mathbf{X}}\widehat{\mathbf{Z}} - \widehat{\mathbf{Y}}
            \right\|_{F}^{2}

        where :math:`\widehat{\mathbf{X}}` are the operator entries,
        :math:`\widehat{\mathbf{Z}}` is the data matrix block containing the
        state and input data, and :math:`\widehat{\mathbf{Y}} = [~
        \widehat{\mathbf{y}}_{0}~~\cdots~~\widehat{\mathbf{y}}_{k-1}
        ~]`.

        This method should NOT depend on the ``entries`` attribute.

        Parameters
        ----------
        state_ : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        input_ : (m, k) or (k,) ndarray or None
            Input vectors. Each column is a single input vector.
            If one dimensional, it is assumed that :math:`m = 1`.

        Returns
        -------
        block : (d, k) or (d,) ndarray
            Data matrix block. Here, :math:`d` is ``entries.shape[1]``.
        """
        raise NotImplementedError
