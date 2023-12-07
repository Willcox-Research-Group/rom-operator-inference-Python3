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

import numpy as np
import scipy.linalg as la

from .. import utils
from ._base import _requires_entries, _NonparametricOperator, _InputMixin


# No dependence on state or input =============================================
class ConstantOperator(_NonparametricOperator):
    r"""Constant operator
    :math:`\Ophat(\qhat,\u) = \chat \in \RR^{r}`.

    Examples
    --------
    >>> import numpy as np
    >>> c = opinf.operators.ConstantOperator()
    >>> entries = np.random.random(10)          # Operator entries.
    >>> c.set_entries(np.random.random(10))
    >>> c.shape
    (10,)
    >>> out = c.apply()                         # "Apply" the operator.
    >>> np.allclose(out, entries)
    True
    """

    @property
    def input_dimension(self):
        """Input dimension (always zero for this operator)."""
        return 0

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
                raise ValueError(
                    "ConstantOperator entries must be one-dimensional"
                )

        _NonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def apply(self, state=None, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat(\qhat,\u) = \chat`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            :math:`\chat`.
        """
        if self.entries.shape[0] == 1:
            if state is None or np.isscalar(state):  # r = k = 1.
                return self.entries[0]
            return np.full_like(state, self.entries[0])  # r = 1, k > 1.
        # if state is None or np.ndim(state) == 1:
        #     return self.entries
        if np.ndim(state) == 2:  # r, k > 1.
            return np.outer(self.entries, np.ones(state.shape[-1]))
        return self.entries  # r > 1, k = 1.

    @_requires_entries
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\chat = \Wr\trp\c`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : ConstantOperator
            Projected operator.
        """
        return _NonparametricOperator.galerkin(
            self, Vr, Wr, lambda c, V, W: W.T @ c
        )

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator.

        Since

        .. math::
           \sum_{j=0}^{k-1}\left\| \chat - \zhat_{j} \right\|_{2}^{2}
           = \left\| \chat\1\trp
           - [~\zhat_0~~\cdots~~\zhat_{k-1}~] \right\|_{F}^{2},

        the data block is :math:`\1\trp\in\RR^{1\times k}`.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        block : (1, k) ndarray
            Vector of ones.
        """
        return np.ones((1, np.atleast_1d(states).shape[-1]))

    @staticmethod
    def operator_dimension(r=None, m=None):
        r"""Column dimension of the operator entries (always 1).

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return 1


# Dependent on state but not on input =========================================
class LinearOperator(_NonparametricOperator):
    r"""Linear state operator
    :math:`\Ophat(\qhat,\u) = \Ahat\qhat`
    where :math:`\Ahat \in \RR^{r \times r}`.

    Examples
    --------
    >>> import numpy as np
    >>> A = opinf.operators.LinearOperator()
    >>> entries = np.random.random((10, 10))    # Operator entries.
    >>> A.set_entries(entries)
    >>> A.shape
    (10, 10)
    >>> q = np.random.random(10)                # State vector.
    >>> out = A.apply(q)                        # Apply the operator to q.
    >>> np.allclose(out, entries @ q)
    True
    """

    @property
    def input_dimension(self):
        """Input dimension (always zero for this operator)."""
        return 0

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

        _NonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat(\qhat,\u) = \Ahat\qhat`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            Application :math:`\Ahat\qhat`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state  # r = 1.
        return self.entries @ state  # r > 1.

    @_requires_entries
    def jacobian(self, state=None, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat(\qhat,\u)=\Ahat`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian :math:`\Ahat`.
        """
        return self.entries

    @_requires_entries
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\Ahat =
        \Wr\trp\A\Vr`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : operator
            ``self`` or new ``LinearOperator`` object.
        """
        return _NonparametricOperator.galerkin(
            self, Vr, Wr, lambda A, V, W: W.T @ A @ V
        )

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the ``state``.

        .. math::
            \min_{\Ahat}\sum_{j=0}^{k-1}\left\|
            \Ahat\qhat_{j}
            - \zhat_j
            \right\|_{2}^{2}
            = \min_{\Ahat}\left\|
            \Ahat\widehat{\Q} - \Zhat
            \right\|_{F}^{2}.

        Here,
        :math:`\widehat{\Q} = [~\qhat_{0} ~~ \cdots ~~ \qhat_{k-1}~]
        \in \RR^{r\times k}` is the ``state`` and
        :math:`\Zhat = [~\zhat_{0}~~\cdots~~\zhat_{k-1}~]\in\RR^{r \times k}`.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        state : (r, k) ndarray
            State vectors. Each column is a single state vector.
        """
        return np.atleast_2d(states)

    @staticmethod
    def operator_dimension(r, m=None):
        """Column dimension :math:`r` of the operator entries.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return r


class QuadraticOperator(_NonparametricOperator):
    r"""Quadratic state operator
    :math:`\Ophat(\q,\u) = \Hhat[\qhat\otimes\qhat]`
    where :math:`\Hhat\in\RR^{r \times r^{2}}`.

    Internally, the action of the operator is computed as the product of a
    :math:`r \times r(r+1)/2` matrix and a compressed version of the Kronecker
    product :math:`\qhat \otimes \qhat`.

    Examples
    --------
    >>> import numpy as np
    >>> H = opinf.operators.QuadraticOperator()
    >>> entries = np.random.random((10, 100))   # Operator entries.
    >>> H.set_entries(entries)
    >>> H.shape                                 # Compressed shape.
    (10, 55)
    >>> q = np.random.random(10)                # State vector.
    >>> out = H.apply(q)                        # Apply the operator to q.
    >>> np.allclose(out, entries @ np.kron(q, q))
    True
    """

    @property
    def input_dimension(self):
        """Input dimension (always zero for this operator)."""
        return 0

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"H[{statestr} ⊗ {statestr}]"

    def _clear(self):
        """Delete operator ``entries`` and related attributes."""
        self._mask = None
        self._prejac = None
        _NonparametricOperator._clear(self)

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
            raise ValueError(
                "QuadraticOperator entries must be two-dimensional"
            )
        r, r2 = entries.shape
        if r2 == r**2:
            entries = utils.compress_quadratic(entries)
        elif r2 != self.operator_dimension(r):
            raise ValueError("invalid QuadraticOperator entries dimensions")

        # Precompute compressed Kronecker product mask and Jacobian matrix.
        self._mask = utils.kron2c_indices(r)
        Ht = utils.expand_quadratic(entries).reshape((r, r, r))
        self._prejac = Ht + Ht.transpose(0, 2, 1)

        _NonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat(\q,\u) = \Hhat[\qhat\otimes\qhat]`

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            Application :math:`\Hhat[\qhat\otimes\qhat]`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state**2  # r = 1
        return self.entries @ np.prod(state[self._mask], axis=1)

    @_requires_entries
    def jacobian(self, state, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat(\qhat,\u)
        = \Hhat[(\I_r\otimes\qhat) + (\qhat\otimes\I_r)]`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian
            :math:`\Hhat[(\I_r\otimes\qhat) + (\qhat\otimes\I_r)]`.
        """
        return self._prejac @ np.atleast_1d(state)

    @_requires_entries
    def galerkin(self, Vr, Wr=None):
        r"""Return the (Petrov-)Galerkin projection of the operator,
        :math:`\Hhat = \Wr\trp\H(\Vr\otimes\Vr)`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : :class:`QuadraticOperator`
            Galerkin projection of this operator.
        """

        def _project(H, V, W):
            return W.T @ utils.expand_quadratic(H) @ np.kron(V, V)

        return _NonparametricOperator.galerkin(self, Vr, Wr, _project)

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri-Rao product of the state with itself:
        :math:`\widehat{\Q}\odot\widehat{\Q}` where
        :math:`\widehat{\Q}` is the ``state``.

        .. math::
            \min_{\Hhat}\sum_{j=0}^{k-1}\left\|
            \Hhat[
            \qhat_{j}\otimes\qhat_{j}]
            - \zhat_j
            \right\|_{2}^{2}
            = \min_{\Hhat}\left\|
            \Hhat[
            \widehat{\Q} \odot \widehat{\Q}]
            - \Zhat
            \right\|_{F}^{2}.

        Here, :math:`\widehat{\Q} = [~
        \qhat_{0} ~~ \cdots ~~ \qhat_{k-1}
        ~] \in \RR^{r\times k}` is the ``state``
        and :math:`\Zhat = [~
        \zhat_{0}~~\cdots~~\zhat_{k-1}
        ~]\in\RR^{r \times k}`.
        The Khatri-Rao product :math:`\odot` is the Kronecker product applied
        columnwise:

        .. math::
            \left[\begin{array}{ccc}
            && \\
            \qhat_{0} & \cdots & \qhat_{k-1}
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
            \qhat_{0} \otimes \widehat{\mathbf{p}}_{0}
            & \cdots &
            \qhat_{k-1} \otimes \widehat{\mathbf{p}}_{k-1}
            \\ &&
            \end{array}\right].

        Internally, a compressed Khatri-Rao product with
        :math:`r(r+1)/2 < r^{2}` degrees of freedom is used for efficiency.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        product : (r(r+1)/2, k) ndarray
            Compressed Khatri-Rao product of ``states`` with itself.
        """
        return utils.kron2c(np.atleast_2d(states))

    @staticmethod
    def operator_dimension(r, m=None):
        """Column dimension :math:`r(r+1)/2` of the operator entries.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return r * (r + 1) // 2


class CubicOperator(_NonparametricOperator):
    r"""Cubic state operator
    :math:`\Ophat(\qhat,\u)
    = \Ghat[\qhat\otimes\qhat\otimes\qhat]`
    where
    :math:`\Ghat\in\RR^{r \times r^{3}}`.

    Internally, the action of the operator is computed as the product of a
    :math:`r \times r(r+1)(r+2)/6` matrix and a compressed version of the
    triple Kronecker product :math:`\qhat \otimes \qhat \otimes \qhat`.

    Examples
    --------
    >>> import numpy as np
    >>> G = opinf.operators.CubicOperator()
    >>> entries = np.random.random((10, 1000))  # Operator entries.
    >>> G.set_entries(entries)
    >>> G.shape                                 # Compressed shape.
    (10, 220)
    >>> q = np.random.random(10)                # State vector.
    >>> out = G.apply(q)                        # Apply the operator to q.
    >>> np.allclose(out, entries @ np.kron(q, np.kron(q, q)))
    True
    """

    @property
    def input_dimension(self):
        """Input dimension (always zero for this operator)."""
        return 0

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"G[{statestr} ⊗ {statestr} ⊗ {statestr}]"

    def _clear(self):
        """Delete operator ``entries`` and related attributes."""
        self._mask = None
        self._prejac = None
        _NonparametricOperator._clear(self)

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
            entries = utils.compress_cubic(entries)
        elif r3 != self.operator_dimension(r):
            raise ValueError("invalid CubicOperator entries dimensions")

        # Precompute compressed Kronecker product mask and Jacobian tensor.
        self._mask = utils.kron3c_indices(r)
        Gt = utils.expand_cubic(entries).reshape([r] * 4)
        self._prejac = Gt + Gt.transpose(0, 2, 1, 3) + Gt.transpose(0, 3, 1, 2)

        _NonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat(\qhat,\u)
        = \Ghat[\qhat\otimes\qhat\otimes\qhat]`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\Ghat[\qhat\otimes\qhat\otimes\qhat]`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state**3  # r = 1.
        return self.entries @ np.prod(state[self._mask], axis=1)

    @_requires_entries
    def jacobian(self, state, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat(\qhat,\u)
        = \Ghat[(\I_r\otimes\qhat\otimes\qhat)
        + (\qhat\otimes\I_r\otimes\qhat)
        + (\qhat\otimes\qhat\otimes\I_r)]`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian
            :math:`\Ghat[(\I_r\otimes\qhat\otimes\qhat)
            + (\qhat\otimes\I_r\otimes\qhat)
            + (\qhat\otimes\qhat\otimes\I_r)]`.
        """
        q_ = np.atleast_1d(state)
        return (self._prejac @ q_) @ q_

    @_requires_entries
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\widehat{\mathbf{G}} =
        \Wr\trp\mathbf{G}
        (\Vr\otimes\Vr\otimes\Vr)`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : operator
            ``self`` or new ``CubicOperator`` object.
        """

        def _project(G, V, W):
            return W.T @ utils.expand_cubic(G) @ np.kron(V, np.kron(V, V))

        return _NonparametricOperator.galerkin(self, Vr, Wr, _project)

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri-Rao product of the state with itself three times:
        :math:`\widehat{\Q}\odot\widehat{\Q}
        \odot\widehat{\Q}`
        where :math:`\widehat{\Q}` is the ``state``.

        .. math::
            \min_{\widehat{\mathbf{G}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{G}}[
            \qhat_{j}\otimes\qhat_{j}]
            - \zhat_j
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{G}}}\left\|
            \widehat{\mathbf{G}}[
            \widehat{\Q}
            \odot \widehat{\Q}
            \odot \widehat{\Q}]
            - \Zhat
            \right\|_{F}^{2}.

        Here, :math:`\widehat{\Q} = [~
        \qhat_{0} ~~ \cdots ~~ \qhat_{k-1}
        ~] \in \RR^{r\times k}` is the ``state``
        and :math:`\Zhat = [~
        \zhat_{0}~~\cdots~~\zhat_{k-1}
        ~]\in\RR^{r \times k}`.
        The Khatri-Rao product :math:`\odot` is the Kronecker product applied
        columnwise:

        .. math::
            \left[\begin{array}{ccc}
            && \\
            \qhat_{0} & \cdots & \qhat_{k-1}
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
            \qhat_{0} \otimes \widehat{\mathbf{p}}_{0}
            & \cdots &
            \qhat_{k-1} \otimes \widehat{\mathbf{p}}_{k-1}
            \\ &&
            \end{array}\right].

        Internally, a compressed triple Khatri-Rao product with
        :math:`r(r+1)(r+2)/6<r^{3}` degrees of freedom is used for efficiency.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        product_ : (r(r+1)(r+2)/6, k) ndarray
            Compressed triple Khatri-Rao product of the ``state`` with itself.
        """
        return utils.kron3c(np.atleast_2d(states))

    @staticmethod
    def operator_dimension(r, m=None):
        """Column dimension :math:`r(r+1)(r+2)/6` of the operator entries.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return r * (r + 1) * (r + 2) // 6


# Dependent on input but not on state =========================================
class InputOperator(_NonparametricOperator, _InputMixin):
    r"""Linear input operator
    :math:`\Ophat(\qhat,\u) = \Bhat\u`
    where :math:`\Bhat \in \RR^{r \times m}`.

    Examples
    --------
    >>> import numpy as np
    >>> B = opinf.operators.LinearOperator()
    >>> entries = np.random.random((10, 3))     # Operator entries.
    >>> B.set_entries(entries)
    >>> B.shape
    (10, 3)
    >>> u = np.random.random(3)                 # Input vector.
    >>> out = B.apply(None, u)                  # Apply the operator to u.
    >>> np.allclose(out, entries @ u)
    True
    """

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return None if self.entries is None else self.entries.shape[1]

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

        _NonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def apply(self, state, input_):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat(\qhat,\u) = \Bhat\u`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector (not used).
        input_ : (m,) ndarray
            Input vector.

        Returns
        -------
        out : (r,) ndarray
            Application :math:`\Bhat\u`.
        """
        if self.entries.shape[1] == 1 and (dim := np.ndim(input_)) != 2:
            if self.entries.shape[0] == 1:
                return self.entries[0, 0] * input_  # r = m = 1.
            if dim == 1 and input_.size > 1:  # r, k > 1, m = 1.
                return np.outer(self.entries[:, 0], input_)
            return self.entries[:, 0] * input_  # r > 1, m = k = 1.
        return self.entries @ input_  # m > 1.

    @_requires_entries
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\Bhat =
        \Wr\trp\B`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : operator
            ``self`` or new ``InputOperator`` object.
        """
        return _NonparametricOperator.galerkin(
            self, Vr, Wr, lambda B, V, W: W.T @ B
        )

    @staticmethod
    def datablock(states, inputs):
        r"""Return the data matrix block corresponding to the operator,
        the ``inputs``.

        .. math::
            \min_{\Bhat}\sum_{j=0}^{k-1}\left\|
            \Bhat\u_{j}
            - \zhat_j
            \right\|_{2}^{2}
            = \min_{\Bhat}\left\|
            \Bhat\U - \Zhat
            \right\|_{F}^{2}.

        Here, :math:`\U = [~
        \u_{0} ~~ \cdots ~~ \u_{k-1}
        ~] \in \RR^{m\times k}` is the ``input_``
        and :math:`\Zhat = [~
        \zhat_{0}~~\cdots~~\zhat_{k-1}
        ~]\in\RR^{r \times k}`.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors (not used).
        inputs : (m, k) or (k,) ndarray
            Input vectors. Each column is a single input vector.
            If one dimensional, it is assumed that :math:`m = 1`.

        Returns
        -------
        inputs : (m, k) ndarray
            Input vectors. Each column is a single input vector.
        """
        return np.atleast_2d(inputs)

    @staticmethod
    def operator_dimension(r, m):
        """Column dimension :math:`m` of the operator entries.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return m


# Dependent on both state and input ===========================================
class StateInputOperator(_NonparametricOperator, _InputMixin):
    r"""Linear state / input interaction operator
    :math:`\Ophat(\qhat,\u) = \Nhat[\u\otimes\qhat]`
    where :math:`\Nhat \in \RR^{r \times rm}`.

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
    >>> out = N.apply(q, u)                     # Apply the operator to (q,u).
    >>> np.allclose(out, entries @ np.kron(u, q))
    True
    """

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        if self.entries is None:
            return None
        return self.entries.shape[1] // self.entries.shape[0]

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
            raise ValueError(
                "StateInputOperator entries must be two-dimensional"
            )
        r, rm = entries.shape
        m = rm // r
        if rm != r * m:
            raise ValueError("invalid StateInputOperator entries dimensions")

        _NonparametricOperator.set_entries(self, entries)

    @_requires_entries
    def apply(self, state, input_):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat(\qhat,\u) = \Nhat[\u\otimes\qhat]`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector.

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\Nhat[\u\otimes\qhat]`.
        """
        # Determine if arguments represent one snapshot or several.
        multi = (sdim := np.ndim(state)) > 1
        multi |= (idim := np.ndim(input_)) > 1
        multi |= self.shape[0] == 1 and sdim == 1 and state.shape[0] > 1
        multi |= self.shape[1] == 1 and idim == 1 and input_.shape[0] > 1
        single = not multi

        if self.shape[1] == 1:
            return self.entries[0, 0] * input_ * state  # r = m = 1.
        if single:
            return self.entries @ np.kron(input_, state)  # k = 1, rm > 1.
        Q_ = np.atleast_2d(state)
        U = np.atleast_2d(input_)
        return self.entries @ la.khatri_rao(U, Q_)  # k > 1, rm > 1.

    @_requires_entries
    def jacobian(self, state, input_):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat(\qhat,\u) = \sum_{i=1}^{m}u_{i}\Nhat_{i}`
        where :math:`\Nhat=[~\Nhat_{1}~~\cdots~~\Nhat_{m}~]`
        and each :math:`\Nhat_i\in\RR^{r\times r},~i=1,\ldots,m`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian :math:`\sum_{i=1}^{m}u_{i}\Nhat_{i}`.
        """
        r, rm = self.entries.shape
        m = rm // r
        u = np.atleast_1d(input_)
        if u.shape[0] != m:
            raise ValueError("invalid input_ shape")
        return np.sum(
            [um * Nm for um, Nm in zip(u, np.split(self.entries, m, axis=1))],
            axis=0,
        )

    @_requires_entries
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\widehat{\mathbf{N}} =
        \Wr\trp\mathbf{N}
        (\I_{m}\otimes\Vr)`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : operator
            ``self`` or new ``CubicOperator`` object.
        """

        def _project(N, V, W):
            r, rm = N.shape
            m = rm // r
            Id = np.eye(m)
            return W.T @ N @ np.kron(Id, V)

        return _NonparametricOperator.galerkin(self, Vr, Wr, _project)

    @staticmethod
    def datablock(states, inputs):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri-Rao product of the inputs and the states:

        :math:`\U\odot\widehat{\Q}` where
        :math:`\widehat{\Q}` is the ``state`` and
        :math:`\U` is the ``input_``.

        .. math::
            \min_{\widehat{\mathbf{N}}}\sum_{j=0}^{k-1}\left\|
            \widehat{\mathbf{N}}[
            \u_{j}\otimes\qhat_{j}]
            - \zhat_j
            \right\|_{2}^{2}
            = \min_{\widehat{\mathbf{N}}}\left\|
            \widehat{\mathbf{N}}[
            \U \odot \widehat{\Q}]
            - \Zhat
            \right\|_{F}^{2}.

        Here, :math:`\widehat{\Q} = [~
        \qhat_{0} ~~ \cdots ~~ \qhat_{k-1}
        ~] \in \RR^{r\times k}` is the ``state``,
        :math:`\U = [~
        \u_{0} ~~ \cdots ~~ \u_{k-1}
        ~] \in \RR^{m\times k}` is the ``input_``,
        and :math:`\Zhat = [~
        \zhat_{0}~~\cdots~~\zhat_{k-1}
        ~]\in\RR^{r \times k}`.
        The Khatri-Rao product :math:`\odot` is the Kronecker product applied
        columnwise:

        .. math::
            \left[\begin{array}{ccc}
            && \\
            \u_{0} & \cdots & \u_{k-1}
            \\ &&
            \end{array}\right]
            \odot
            \left[\begin{array}{ccc}
            && \\
            \qhat_{0} & \cdots & \qhat_{k-1}
            \\ &&
            \end{array}\right]
            =
            \left[\begin{array}{ccc}
            && \\
            \u_{0} \otimes \qhat_{0}
            & \cdots &
            \u_{k-1} \otimes \qhat_{k-1}
            \\ &&
            \end{array}\right].

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors (not used).
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors. Each column is a single input vector.
            If one dimensional, it is assumed that :math:`m = 1`.

        Returns
        -------
        product_ : (m, k) ndarray or None
            Compressed Khatri-Rao product of the ``input_`` and the ``state``.
        """
        return la.khatri_rao(np.atleast_2d(inputs), np.atleast_2d(states))

    @staticmethod
    def operator_dimension(r, m):
        """Column dimension :math:`rm` of the operator entries.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return r * m
