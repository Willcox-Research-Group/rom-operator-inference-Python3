# operators/_nonparametric.py
"""Classes for operators with no external parameter dependencies."""

__all__ = [
            "ConstantOperator",
            "LinearOperator",
            "QuadraticOperator",
            "CubicOperator",
            "InputOperator",
            "StateInputOperator",
          ]

import functools
import numpy as np
import scipy.linalg as la

from . import _kronecker
from ._base import _BaseNonparametricOperator, _requires_entries


class ConstantOperator(_BaseNonparametricOperator):
    r"""Constant 'operator' :math:`\widehat{\mathbf{c}} \in \mathbb{R}^{r}`
    representing the action
    :math:`(\widehat{\mathbf{q}},\mathbf{u}) \mapsto \widehat{\mathbf{c}}`.

    Examples
    --------
    >>> import numpy as np
    >>> c = opinf.operators.ConstantOperator()
    >>> entries = np.random.random(10)          # Operator entries.
    >>> c.set_entries(np.random.random(10))
    >>> c.shape
    (10,)
    >>> out = c.evaluate()                      # "Evaluate" the operator.
    >>> np.allclose(out, entries)
    True
    """
    @staticmethod
    def _str(statestr=None, inputstr=None):
        return "c"

    def set_entries(self, entries):
        """Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r,) ndarray
            Discrete representation of the operator.
        """
        if np.isscalar(entries):
            entries = np.atleast_1d(entries)
        self._validate_entries(entries)

        # Ensure that the operator is one-dimensional.
        if entries.ndim != 1:
            if entries.ndim == 2 and 1 in entries.shape:
                entries = np.ravel(entries)
            else:
                raise ValueError("ConstantOperator entries must be "
                                 "one-dimensional")

        _BaseNonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def __call__(self, state_=None, input_=None):
        r"""Apply the operator mapping to the given state / input.
        Since this is a constant operator, the result is always the same:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto\widehat{\mathbf{c}}`.

        Parameters
        ----------
        state_ : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            The "evaluation" :math:`\widehat{\mathbf{c}}`.
        """
        if self.entries.shape[0] == 1:
            if state_ is None or np.isscalar(state_):       # r = k = 1.
                return self.entries[0]
            return np.full_like(state_, self.entries[0])    # r = 1, k > 1.
        # if state_ is None or np.ndim(state_) == 1:
        #     return self.entries
        if np.ndim(state_) == 2:                            # r, k > 1.
            return np.outer(self.entries, np.ones(state_.shape[-1]))
        return self.entries                                 # r > 1, k = 1.

    @functools.wraps(__call__)
    def evaluate(self, state_=None, input_=None):           # pragma: no cover
        """Mirror of __call__()."""
        return self(state_, input_)

    @_requires_entries
    def jacobian(self, state_=None, input_=None):
        r"""Evaluate the Jacobian of the operator, the zero matrix:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \mathbf{0}\in\mathbb{R}^{r\times r}`.

        Parameters
        ----------
        state_ : (r,) ndarray or None
            State vector (not used).
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            Operator Jacobian, the zero matrix.
        """
        r = self.entries.size
        return np.zeros((r, r))

    @staticmethod
    def datablock(state_, input_=None):
        r"""Return the data matrix block corresponding to the operator,
        a vector of 1's.

        .. math::
            \min_{\widehat{\mathbf{c}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{c}} - \widehat{\mathbf{y}}_{j}
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{c}}}\left\|
            \widehat{\mathbf{c}}\mathbf{1}^{\mathsf{T}} - \widehat{\mathbf{Y}}
            \right\|_{F}^{2},

        where :math:`\mathbf{1} \in \mathbb{R}^{k}` is a vector of 1's
        and :math:`\widehat{\mathbf{Y}} = [~
        \widehat{\mathbf{y}}_{0}~~\cdots~~\widehat{\mathbf{y}}_{k-1}
        ~]\in\mathbb{R}^{r \times k}`.

        Parameters
        ----------
        state_ : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        input_ : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        ones : (k,) ndarray
            Vector of 1's.
        """
        return np.ones(np.atleast_1d(state_).shape[-1])


class LinearOperator(_BaseNonparametricOperator):
    r"""Linear state operator
    :math:`\widehat{\mathbf{A}} \in \mathbb{R}^{r \times r}`
    representing the action
    :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
    \widehat{\mathbf{A}}\widehat{\mathbf{q}}`.

    Examples
    --------
    >>> import numpy as np
    >>> A = opinf.operators.LinearOperator()
    >>> entries = np.random.random((10, 10))    # Operator entries.
    >>> A.set_entries(entries)
    >>> A.shape
    (10, 10)
    >>> q = np.random.random(10)                # State vector.
    >>> out = A.evaluate(q)                     # Compute Aq.
    >>> np.allclose(out, entries @ q)
    True
    """
    @staticmethod
    def _str(statestr, inputstr=None):
        return f"A{statestr}"

    def set_entries(self, entries):
        """Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, r) ndarray
            Discrete representation of the operator.
        """
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        self._validate_entries(entries)

        # Ensure that the operator is two-dimensional and square.
        if entries.ndim != 2:
            raise ValueError("LinearOperator entries must be two-dimensional")
        if entries.shape[0] != entries.shape[1]:
            raise ValueError("LinearOperator entries must be square (r x r)")

        _BaseNonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def __call__(self, state_, input_=None):
        r"""Apply the operator mapping to the given state:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})
        \mapsto\widehat{\mathbf{A}}\widehat{\mathbf{q}}`.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\widehat{\mathbf{A}}\widehat{\mathbf{q}}`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state_              # r = 1.
        return self.entries @ state_                        # r > 1.

    @functools.wraps(__call__)
    def evaluate(self, state_, input_=None):                # pragma: no cover
        """Mirror of __call__()."""
        return self(state_, input_)

    @_requires_entries
    def jacobian(self, state_=None, input_=None):
        r"""Evaluate the Jacobian of the operator, the operator entries:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto\widehat{\mathbf{A}}`.

        Parameters
        ----------
        state_ : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            Operator Jacobian, the operator entries.
        """
        return self.entries

    @staticmethod
    def datablock(state_, input_=None):
        r"""Return the data matrix block corresponding to the operator,
        the ``state_``.

        .. math::
            \min_{\widehat{\mathbf{A}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
            - \widehat{\mathbf{y}}_{j}
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{A}}}\left\|
            \widehat{\mathbf{A}}\widehat{\mathbf{Q}} - \widehat{\mathbf{Y}}
            \right\|_{F}^{2}.

        Here, :math:`\widehat{\mathbf{Q}} = [~
        \widehat{\mathbf{q}}_{0} ~~ \cdots ~~ \widehat{\mathbf{q}}_{k-1}
        ~] \in \mathbb{R}^{r\times k}` is the ``state_``
        and :math:`\widehat{\mathbf{Y}} = [~
        \widehat{\mathbf{y}}_{0}~~\cdots~~\widehat{\mathbf{y}}_{k-1}
        ~]\in\mathbb{R}^{r \times k}`.

        Parameters
        ----------
        state_ : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        input_ : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        state_ : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
        """
        return state_


class QuadraticOperator(_BaseNonparametricOperator):
    r"""Quadratic state operator
    :math:`\widehat{\mathbf{H}} \in \mathbb{R}^{r \times r^{2}}`
    representing the action
    :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
    \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]`.

    Internally, the action of the operator is computed as the product of a
    :math:`r \times r(r+1)/2` matrix and a compressed version of the Kronecker
    product :math:`\widehat{\mathbf{q}} \otimes \widehat{\mathbf{q}}`.

    Examples
    --------
    >>> import numpy as np
    >>> H = opinf.operators.QuadraticOperator()
    >>> entries = np.random.random((10, 100))   # Operator entries.
    >>> H.set_entries(entries)
    >>> H.shape                                 # Compressed shape.
    (10, 55)
    >>> q = np.random.random(10)                # State vector.
    >>> out = H.evaluate(q)                     # Compute H[q ⊗ q].
    >>> np.allclose(out, entries @ np.kron(q, q))
    True
    """
    @staticmethod
    def _str(statestr, inputstr=None):
        return f"H[{statestr} ⊗ {statestr}]"

    def set_entries(self, entries):
        """Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, r**2) or (r, r(r+1)/2) or (r, r, r) ndarray
            Discrete representation of the operator.
        """
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        self._validate_entries(entries)

        # Ensure that the operator has valid dimensions.
        if entries.ndim == 3 and len(set(entries.shape)) == 1:
            # Reshape (r x r x r) tensor.
            entries = entries.reshape((entries.shape[0], -1))
        if entries.ndim != 2:
            raise ValueError("QuadraticOperator entries must be "
                             "two-dimensional")
        r, r2 = entries.shape
        if r2 == r**2:
            entries = _kronecker.compress_quadratic(entries)
        elif r2 != r * (r + 1) // 2:
            raise ValueError("invalid QuadraticOperator entries dimensions")

        # Precompute compressed Kronecker product mask and Jacobian matrix.
        self._mask = _kronecker.kron2c_indices(r)
        Ht = _kronecker.expand_quadratic(entries).reshape((r, r, r))
        self._jac = Ht + Ht.transpose(0, 2, 1)

        _BaseNonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def __call__(self, state_, input_=None):
        r"""Apply the operator mapping to the given state:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]`.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\widehat{\mathbf{H}}
            [\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state_**2           # r = 1
        return self.entries @ np.prod(state_[self._mask], axis=1)

    @functools.wraps(__call__)
    def evaluate(self, state_, input_=None):                # pragma: no cover
        """Mirror of __call__()."""
        return self(state_, input_)

    @_requires_entries
    def jacobian(self, state_, input_=None):
        r"""Evaluate the Jacobian of the operator:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \widehat{\mathbf{H}}[
        (\mathbf{I}\otimes\widehat{\mathbf{q}})
        + (\widehat{\mathbf{q}}\otimes\mathbf{I})^\mathsf{T}]`.

        Parameters
        ----------
        state_ : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            Operator Jacobian.
        """
        return self._jac @ np.atleast_1d(state_)

    @staticmethod
    def datablock(state_, input_=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri-Rao product of the state with itself:
        :math:`\widehat{\mathbf{Q}}\odot\widehat{\mathbf{Q}}` where
        :math:`\widehat{\mathbf{Q}}` is the ``state_``.

        .. math::
            \min_{\widehat{\mathbf{H}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{H}}[
            \widehat{\mathbf{q}}_{j}\otimes\widehat{\mathbf{q}}_{j}]
            - \widehat{\mathbf{y}}_{j}
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{H}}}\left\|
            \widehat{\mathbf{H}}[
            \widehat{\mathbf{Q}} \odot \widehat{\mathbf{Q}}]
            - \widehat{\mathbf{Y}}
            \right\|_{F}^{2}.

        Here, :math:`\widehat{\mathbf{Q}} = [~
        \widehat{\mathbf{q}}_{0} ~~ \cdots ~~ \widehat{\mathbf{q}}_{k-1}
        ~] \in \mathbb{R}^{r\times k}` is the ``state_``
        and :math:`\widehat{\mathbf{Y}} = [~
        \widehat{\mathbf{y}}_{0}~~\cdots~~\widehat{\mathbf{y}}_{k-1}
        ~]\in\mathbb{R}^{r \times k}`.
        The Khatri-Rao product :math:`\odot` is the Kronecker product applied
        columnwise:

        .. math::
            \left[\begin{array}{ccc}
            && \\
            \widehat{\mathbf{q}}_{0} & \cdots & \widehat{\mathbf{q}}_{k-1}
            \\ &&
            \end{array}\right]
            \odot
            \left[\begin{array}{ccc}
            && \\
            \widehat{\mathbf{p}}_{0} & \cdots & \widehat{\mathbf{p}}_{k-1}
            \\ &&
            \end{array}\right]
            =
            \left[\begin{array}{ccc}
            && \\
            \widehat{\mathbf{q}}_{0} \otimes \widehat{\mathbf{p}}_{0}
            & \cdots &
            \widehat{\mathbf{q}}_{k-1} \otimes \widehat{\mathbf{p}}_{k-1}
            \\ &&
            \end{array}\right].

        Internally, a compressed Khatri-Rao product with
        :math:`r(r+1)/2 < r^{2}` degrees of freedom is used for efficiency.

        Parameters
        ----------
        state_ : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        input_ : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        state_ : (r(r+1)/2, k) or (r(r+1)/2) ndarray
            Compressed Khatri-Rao product of the state with itself.
        """
        if state_.ndim == 1:
            return state_**2
        return _kronecker.kron2c(state_)


class CubicOperator(_BaseNonparametricOperator):
    r"""Cubic state operator
    :math:`\widehat{\mathbf{G}} \in \mathbb{R}^{r \times r^{3}}`
    repesenting the action
    :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
    \widehat{\mathbf{G}}[\widehat{\mathbf{q}}
    \otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]`.

    Internally, the action of the operator is computed as the product of a
    :math:`r \times r(r+1)(r+2)/6` matrix and a compressed version of the
    triple Kronecker product :math:`\widehat{\mathbf{q}} \otimes
    \widehat{\mathbf{q}} \otimes \widehat{\mathbf{q}}`.

    Examples
    --------
    >>> import numpy as np
    >>> G = opinf.operators.CubicOperator()
    >>> entries = np.random.random((10, 1000))  # Operator entries.
    >>> G.set_entries(entries)
    >>> G.shape                                 # Compressed shape.
    (10, 220)
    >>> q = np.random.random(10)                # State vector.
    >>> out = G.evaluate(q)                     # Compute G[q ⊗ q ⊗ q].
    >>> np.allclose(out, entries @ np.kron(q, np.kron(q, q)))
    True
    """
    @staticmethod
    def _str(statestr, inputstr=None):
        return f"G[{statestr} ⊗ {statestr} ⊗ {statestr}]"

    def set_entries(self, entries):
        """Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, r**3) or (r, r(r+1)(r+2)/6) or (r, r, r, r) ndarray
            Discrete representation of the operator.
        """
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        self._validate_entries(entries)

        # Ensure that the operator has valid dimensions.
        if entries.ndim == 4 and len(set(entries.shape)) == 1:
            # Reshape (r x r x r x r) tensor.
            entries = entries.reshape((entries.shape[0], -1))
        if entries.ndim != 2:
            raise ValueError("CubicOperator entries must be two-dimensional")
        r, r3 = entries.shape
        if r3 == r**3:
            entries = _kronecker.compress_cubic(entries)
        elif r3 != r * (r + 1) * (r + 2) // 6:
            raise ValueError("invalid CubicOperator entries dimensions")

        # Precompute compressed Kronecker product mask and Jacobian tensor.
        self._mask = _kronecker.kron3c_indices(r)
        Gt = _kronecker.expand_cubic(entries).reshape([r]*4)
        self._jac = Gt + Gt.transpose(0, 2, 1, 3) + Gt.transpose(0, 3, 1, 2)

        _BaseNonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def __call__(self, state_, input_=None):
        r"""Apply the operator mapping to the given state:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \widehat{\mathbf{G}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}
        \otimes\widehat{\mathbf{q}}]`.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\widehat{\mathbf{G}}
            [\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}
            \otimes\widehat{\mathbf{q}}]`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state_**3           # r = 1.
        return self.entries @ np.prod(state_[self._mask], axis=1)

    @functools.wraps(__call__)
    def evaluate(self, state_, input_=None):                # pragma: no cover
        """Mirror of __call__()."""
        return self(state_, input_)

    @_requires_entries
    def jacobian(self, state_, input_=None):
        r"""Evaluate the Jacobian of the operator:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \widehat{\mathbf{G}}[
        (\mathbf{I}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}})
        + (\widehat{\mathbf{q}}\otimes\mathbf{I}\otimes\widehat{\mathbf{q}})
        + (\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}
        \otimes\mathbf{I})^\mathsf{T}]`.

        Parameters
        ----------
        state_ : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            Operator Jacobian.
        """
        q_ = np.atleast_1d(state_)
        return (self._jac @ q_) @ q_

    @staticmethod
    def datablock(state_, input_=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri-Rao product of the state with itself three times:
        :math:`\widehat{\mathbf{Q}}\odot\widehat{\mathbf{Q}}
        \odot\widehat{\mathbf{Q}}`
        where :math:`\widehat{\mathbf{Q}}` is the ``state_``.

        .. math::
            \min_{\widehat{\mathbf{G}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{G}}[
            \widehat{\mathbf{q}}_{j}\otimes\widehat{\mathbf{q}}_{j}]
            - \widehat{\mathbf{y}}_{j}
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{G}}}\left\|
            \widehat{\mathbf{G}}[
            \widehat{\mathbf{Q}}
            \odot \widehat{\mathbf{Q}}
            \odot \widehat{\mathbf{Q}}]
            - \widehat{\mathbf{Y}}
            \right\|_{F}^{2}.

        Here, :math:`\widehat{\mathbf{Q}} = [~
        \widehat{\mathbf{q}}_{0} ~~ \cdots ~~ \widehat{\mathbf{q}}_{k-1}
        ~] \in \mathbb{R}^{r\times k}` is the ``state_``
        and :math:`\widehat{\mathbf{Y}} = [~
        \widehat{\mathbf{y}}_{0}~~\cdots~~\widehat{\mathbf{y}}_{k-1}
        ~]\in\mathbb{R}^{r \times k}`.
        The Khatri-Rao product :math:`\odot` is the Kronecker product applied
        columnwise:

        .. math::
            \left[\begin{array}{ccc}
            && \\
            \widehat{\mathbf{q}}_{0} & \cdots & \widehat{\mathbf{q}}_{k-1}
            \\ &&
            \end{array}\right]
            \odot
            \left[\begin{array}{ccc}
            && \\
            \widehat{\mathbf{p}}_{0} & \cdots & \widehat{\mathbf{p}}_{k-1}
            \\ &&
            \end{array}\right]
            =
            \left[\begin{array}{ccc}
            && \\
            \widehat{\mathbf{q}}_{0} \otimes \widehat{\mathbf{p}}_{0}
            & \cdots &
            \widehat{\mathbf{q}}_{k-1} \otimes \widehat{\mathbf{p}}_{k-1}
            \\ &&
            \end{array}\right].

        Internally, a compressed triple Khatri-Rao product with
        :math:`r(r+1)(r+2)/6<r^{3}` degrees of freedom is used for efficiency.

        Parameters
        ----------
        state_ : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        input_ : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        state_ : (r(r+1)(r+2)/6, k) or (r(r+1)(r+2)/6,) ndarray
            Compressed Khatri-Rao product of the state with itself.
        """
        if state_.ndim == 1:
            return state_**3
        return _kronecker.kron3c(state_)


class InputOperator(_BaseNonparametricOperator):
    r"""Linear input operator
    :math:`\widehat{\mathbf{B}} \in \mathbb{R}^{r \times m}`
    representing the action
    :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
    \widehat{\mathbf{B}}\mathbf{u}`.

    Examples
    --------
    >>> import numpy as np
    >>> B = opinf.operators.LinearOperator()
    >>> entries = np.random.random((10, 3))     # Operator entries.
    >>> B.set_entries(entries)
    >>> B.shape
    (10, 3)
    >>> u = np.random.random(3)                 # Input vector.
    >>> out = B.evaluate(u)                     # Compute Bu.
    >>> np.allclose(out, entries @ u)
    True
    """
    @staticmethod
    def _str(statestr, inputstr):
        return f"B{inputstr}"

    def set_entries(self, entries):
        """Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, m) ndarray
            Discrete representation of the operator.
        """
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        self._validate_entries(entries)

        # Ensure that the operator is two-dimensional.
        if entries.ndim == 1:
            # Assumes r = entries.size, m = 1.
            entries = entries.reshape((-1, 1))
        if entries.ndim != 2:
            raise ValueError("InputOperator entries must be two-dimensional")

        _BaseNonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def __call__(self, state_, input_):
        r"""Apply the operator mapping to the given state:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})
        \mapsto\widehat{\mathbf{B}}\mathbf{u}`.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector (not used).
        input_ : (m,) ndarray
            Input vector.

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\widehat{\mathbf{B}}\mathbf{u}`.
        """
        if self.entries.shape[1] == 1:
            if self.entries.shape[0] == 1:
                return self.entries[0, 0] * input_          # r = m = 1.
            if np.ndim(input_) == 1:                        # r, k > 1, m = 1.
                return np.outer(self.entries[:, 0], input_)
            return self.entries[:, 0] * input_              # r > 1, m = k = 1.
        return self.entries @ input_                        # m > 1.

    @functools.wraps(__call__)
    def evaluate(self, state_, input_):                     # pragma: no cover
        """Mirror of __call__()."""
        return self(state_, input_)

    @_requires_entries
    def jacobian(self, state_=None, input_=None):
        r"""Evaluate the Jacobian of the operator, the zero matrix:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \mathbf{0}\in\mathbb{R}^{r\times r}`.

        Parameters
        ----------
        state_ : (r,) ndarray or None
            State vector (not used).
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            Operator Jacobian, the zero matrix
        """
        r = self.entries.shape[0]
        return np.zeros((r, r))

    @staticmethod
    def datablock(state_, input_):
        r"""Return the data matrix block corresponding to the operator,
        the ``input_``.

        .. math::
            \min_{\widehat{\mathbf{B}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{B}}\mathbf{u}_{j}
            - \widehat{\mathbf{y}}_{j}
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{B}}}\left\|
            \widehat{\mathbf{B}}\mathbf{U} - \widehat{\mathbf{Y}}
            \right\|_{F}^{2}.

        Here, :math:`\mathbf{U} = [~
        \mathbf{u}_{0} ~~ \cdots ~~ \mathbf{u}_{k-1}
        ~] \in \mathbb{R}^{m\times k}` is the ``input_``
        and :math:`\widehat{\mathbf{Y}} = [~
        \widehat{\mathbf{y}}_{0}~~\cdots~~\widehat{\mathbf{y}}_{k-1}
        ~]\in\mathbb{R}^{r \times k}`.

        Parameters
        ----------
        state_ : (r, k) or (k,) ndarray
            State vectors (not used).
        input_ : (m, k) or (k,) ndarray
            Input vectors. Each column is a single input vector.
            If one dimensional, it is assumed that :math:`m = 1`.

        Returns
        -------
        input_ : (m, k) or (k,) ndarray
            Input vectors. Each column is a single input vector.
        """
        return input_


class StateInputOperator(_BaseNonparametricOperator):
    r"""Linear state / input interaction operator
    :math:`\widehat{\mathbf{N}} \in \mathbb{R}^{r \times rm}`
    representing the action
    :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
    \widehat{\mathbf{N}}[\mathbf{u}\otimes\widehat{\mathbf{q}}]`.

    Examples
    --------
    >>> import numpy as np
    >>> N = opinf.operators.StateInputOperator()
    >>> entries = np.random.random((10, 3))
    >>> N.set_entries(entries)
    >>> N.shape
    (10, 3)
    >>> q = np.random.random(10)                # State vector.
    >>> u = np.random.random(3)                 # Input vector.
    >>> out = B.evaluate(u)                     # Compute N[u ⊗ q].
    >>> np.allclose(out, entries @ np.kron(u, q))
    True
    """
    @staticmethod
    def _str(statestr, inputstr):
        return f"N[{inputstr} ⊗ {statestr}]"

    def set_entries(self, entries):
        """Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, rm) ndarray
            Discrete representation of the operator.
        """
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        self._validate_entries(entries)

        # Ensure that the operator has valid dimensions.
        if entries.ndim != 2:
            raise ValueError("StateInputOperator entries must be "
                             "two-dimensional")
        r, rm = entries.shape
        m = rm // r
        if rm != r * m:
            raise ValueError("invalid StateInputOperator entries dimensions")

        _BaseNonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def __call__(self, state_, input_):
        r"""Apply the operator mapping to the given state:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \widehat{\mathbf{N}}[\mathbf{u}\otimes\widehat{\mathbf{q}}]`.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector.

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\widehat{\mathbf{N}}[
            \mathbf{u}\otimes\widehat{\mathbf{q}}]`.
        """
        # Determine if arguments represent one snapshot or several.
        dim = np.ndim(state_)
        # multi = (self.entries.shape[0] == 1 and dim == 1) or dim > 1
        single = dim <= 1 and (self.entries.shape[0] != 1 or dim != 1)

        N_ = self.entries
        if self.entries.shape[0] == 1:
            N_ = N_[0]
            if self.entries.shape[1] == 1 and single:
                return N_[0] * input_ * state_              # r = m = k = 1.
        if single:
            return N_ @ np.kron(input_, state_)             # k = 1.
        Q_ = np.atleast_2d(state_)
        U = np.atleast_2d(input_)
        return N_ @ la.khatri_rao(U, Q_)                    # k > 1.

    @functools.wraps(__call__)
    def evaluate(self, state_, input_):                     # pragma: no cover
        """Mirror of __call__()."""
        return self(state_, input_)

    @_requires_entries
    def jacobian(self, state_, input_):
        r"""Evaluate the Jacobian of the operator:
        :math:`(\widehat{\mathbf{q}},\mathbf{u})\mapsto
        \sum_{i=1}^{m}u_{i}\widehat{\mathbf{N}}_{i}`, where
        :math:`\widehat{\mathbf{N}}
        =[~\widehat{\mathbf{N}}_{1}~~\cdots~~\widehat{\mathbf{N}}_{m}~]`.

        Parameters
        ----------
        state_ : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            Operator Jacobian.
        """
        r, rm = self.entries.shape
        m = rm // r
        u = np.atleast_1d(input_)
        if u.shape[0] != m:
            raise ValueError("invalid input_ shape")
        return np.sum([
            um*Nm for um, Nm in zip(u, np.split(self.entries, m, axis=1))
        ], axis=0)

    @staticmethod
    def datablock(state_, input_):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri-Rao product of the inputs and the states:

        :math:`\mathbf{U}\odot\widehat{\mathbf{Q}}` where
        :math:`\widehat{\mathbf{Q}}` is the ``state_`` and
        :math:`\mathbf{U}` is the ``input_``.

        .. math::
            \min_{\widehat{\mathbf{N}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{N}}[
            \mathbf{u}_{j}\otimes\widehat{\mathbf{q}}_{j}]
            - \widehat{\mathbf{y}}_{j}
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{N}}}\left\|
            \widehat{\mathbf{N}}[
            \mathbf{U} \odot \widehat{\mathbf{Q}}]
            - \widehat{\mathbf{Y}}
            \right\|_{F}^{2}.

        Here, :math:`\widehat{\mathbf{Q}} = [~
        \widehat{\mathbf{q}}_{0} ~~ \cdots ~~ \widehat{\mathbf{q}}_{k-1}
        ~] \in \mathbb{R}^{r\times k}` is the ``state_``,
        :math:`\mathbf{U} = [~
        \mathbf{u}_{0} ~~ \cdots ~~ \mathbf{u}_{k-1}
        ~] \in \mathbb{R}^{m\times k}` is the ``input_``,
        and :math:`\widehat{\mathbf{Y}} = [~
        \widehat{\mathbf{y}}_{0}~~\cdots~~\widehat{\mathbf{y}}_{k-1}
        ~]\in\mathbb{R}^{r \times k}`.
        The Khatri-Rao product :math:`\odot` is the Kronecker product applied
        columnwise:

        .. math::
            \left[\begin{array}{ccc}
            && \\
            \mathbf{u}_{0} & \cdots & \mathbf{u}_{k-1}
            \\ &&
            \end{array}\right]
            \odot
            \left[\begin{array}{ccc}
            && \\
            \widehat{\mathbf{q}}_{0} & \cdots & \widehat{\mathbf{q}}_{k-1}
            \\ &&
            \end{array}\right]
            =
            \left[\begin{array}{ccc}
            && \\
            \mathbf{u}_{0} \otimes \widehat{\mathbf{q}}_{0}
            & \cdots &
            \mathbf{u}_{k-1} \otimes \widehat{\mathbf{q}}_{k-1}
            \\ &&
            \end{array}\right].

        Parameters
        ----------
        state_ : (r, k) or (k,) ndarray
            State vectors (not used).
            If one dimensional, it is assumed that :math:`r = 1`.
        input_ : (m, k) or (k,) ndarray or None
            Input vectors. Each column is a single input vector.
            If one dimensional, it is assumed that :math:`m = 1`.

        Returns
        -------
        input_ : (m, k) or (k,) ndarray or None
            Input vectors. Each column is a single input vector.
        """
        Q_ = np.atleast_2d(state_)
        U = np.atleast_2d(input_)
        if Q_.shape[0] == 1 and U.shape[0] == 1:
            return U[0] * Q_[0]
        return la.khatri_rao(U, Q_)
