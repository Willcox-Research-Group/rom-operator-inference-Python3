# operators/_nonparametric.py
"""Classes for OpInf operators with no external parameter dependencies."""

__all__ = [
    "ConstantOperator",
    "LinearOperator",
    "QuadraticOperator",
    "CubicOperator",
    "InputOperator",
    "StateInputOperator",
]

import itertools
import numpy as np
import scipy.linalg as la
import scipy.special as special

from .. import utils
from ._base import OpInfOperator, InputMixin


# No dependence on state or input =============================================
class ConstantOperator(OpInfOperator):
    r"""Constant operator :math:`\Ophat_{\ell}(\qhat,\u) = \chat \in \RR^{r}`.

    Parameters
    ----------
    entries : (r,) ndarray or None
        Operator entries :math:`\chat`.

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

    @staticmethod
    def _str(statestr=None, inputstr=None):
        return "c"

    def set_entries(self, entries):
        r"""Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r,) ndarray
            Operator entries :math:`\chat`.
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

        OpInfOperator.set_entries(self, entries)

    @utils.requires("entries")
    def apply(self, state=None, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \chat`.

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

    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\chat = (\Wr\trp\Vr)^{-1}\Wr\trp\c`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.ConstantOperator`
            Projected operator.
        """
        return self._galerkin(Vr, Wr, lambda c, V: c)

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        a row vector of ones.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \chat` and :math:`\d_{\ell}(\qhat,\u) = 1`,
        the data block is

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           1 & \cdots & 1
           \end{array}\right]
           \in \RR^{1 \times k}.

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
            Row vector of ones.
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
class LinearOperator(OpInfOperator):
    r"""Linear state operator :math:`\Ophat_{\ell}(\qhat,\u) = \Ahat\qhat`
    where :math:`\Ahat \in \RR^{r \times r}`.

    Parameters
    ----------
    entries : (r, r) ndarray or None
        Operator entries :math:`\Ahat`.

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

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"A{statestr}"

    def set_entries(self, entries):
        r"""Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, r) ndarray
            Operator entries :math:`\Ahat`.
        """
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        self._validate_entries(entries)

        # Ensure that the operator is two-dimensional and square.
        if entries.ndim != 2:
            raise ValueError("LinearOperator entries must be two-dimensional")
        if entries.shape[0] != entries.shape[1]:
            raise ValueError("LinearOperator entries must be square (r x r)")

        OpInfOperator.set_entries(self, entries)

    @utils.requires("entries")
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Ahat\qhat`.

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

    @utils.requires("entries")
    def jacobian(self, state=None, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat_{\ell}(\qhat,\u)=\Ahat`.

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

    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\Ahat = (\Wr\trp\Vr)^{-1}\Wr\trp\A\Vr`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.LinearOperator`
            Projected operator.
        """
        return self._galerkin(Vr, Wr, lambda A, V: A @ V)

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the ``states``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Ahat` and
        :math:`\d_{\ell}(\qhat,\u) = \qhat`, the data block is

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \qhat_0 & \cdots & \qhat_{k-1}
           \end{array}\right]
           \in \RR^{r \times k}.

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


class QuadraticOperator(OpInfOperator):
    r"""Quadratic state operator
    :math:`\Ophat_{\ell}(\qhat,\u) = \Hhat[\qhat\otimes\qhat]`
    where :math:`\Hhat\in\RR^{r \times r^{2}}`.

    Internally, the action of the operator is computed as the product of a
    :math:`r \times r(r+1)/2` matrix and a compressed version of the Kronecker
    product :math:`\qhat \otimes \qhat`.

    Parameters
    ----------
    entries : (r, r^2) or (r, r(r+1)/2) or (r, r, r) ndarray or None
        Operator entries :math:`\Hhat`.

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

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"H[{statestr} ⊗ {statestr}]"

    def _clear(self):
        """Delete operator ``entries`` and related attributes."""
        self._mask = None
        self._prejac = None
        OpInfOperator._clear(self)

    def _precompute_jacobian_jit(self):
        """Compute (just in time) the pre-Jacobian tensor Jt such that
        Jt @ q = jacobian(q).
        """
        r = self.entries.shape[0]
        Ht = self.expand_entries(self.entries).reshape((r, r, r))
        self._prejac = Ht + Ht.transpose(0, 2, 1)

    def set_entries(self, entries):
        r"""Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, r^2) or (r, r(r+1)/2) or (r, r, r) ndarray
            Operator entries :math:`\Hhat`.
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
            entries = self.compress_entries(entries)
        elif r2 != self.operator_dimension(r):
            raise ValueError("invalid QuadraticOperator entries dimensions")

        # Precompute compressed Kronecker product mask and Jacobian matrix.
        self._mask = self.ckron_indices(r)
        self._prejac = None

        OpInfOperator.set_entries(self, entries)

    @utils.requires("entries")
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Hhat[\qhat\otimes\qhat]`

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

    @utils.requires("entries")
    def jacobian(self, state, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat_{\ell}(\qhat,\u)
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
        if self._prejac is None:
            self._precompute_jacobian_jit()
        return self._prejac @ np.atleast_1d(state)

    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the (Petrov-)Galerkin projection of the operator,
        :math:`\Hhat = (\Wr\trp\Vr)^{-1}\Wr\trp\H[\Vr\otimes\Vr]`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.QuadraticOperator`
            Projected operator.
        """

        def _pg(H, V):
            return self.expand_entries(H) @ np.kron(V, V)

        return self._galerkin(Vr, Wr, _pg)

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri--Rao product of the state with itself:
        :math:`\Qhat\odot\Qhat` where :math:`\Qhat` is ``states``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Hhat` and
        :math:`\d_{\ell}(\qhat,\u) = \qhat\otimes\qhat`,
        the data block should be

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \qhat_0\otimes\qhat_0 & \cdots & \qhat_{k-1}\otimes\qhat_{k-1}
           \end{array}\right]
           \in\RR^{r^2 \times k}.

        Internally, a compressed Kronecker product :math:`\tilde{\otimes}` with
        :math:`r(r+1)/2 < r^{2}` degrees of freedom is used for efficiency,
        hence the data block is actually

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \qhat_0\tilde{\otimes}\qhat_0
           & \cdots &
           \qhat_{k-1}\tilde{\otimes}\qhat_{k-1}
           \end{array}\right]
           \in\RR^{r(r+1)/2 \times k}.

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
            Compressed Khatri--Rao product of ``states`` with itself.
        """
        return QuadraticOperator.ckron(np.atleast_2d(states))

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

    # Utilities ---------------------------------------------------------------
    @staticmethod
    def ckron(state, checkdim=False):
        r"""Calculate the compressed Kronecker product of a vector with itself.

        For a vector :math:`\qhat = [~\hat{q}_{1}~~\cdots~~\hat{q}_{r}~]\trp`,
        the Kronecker product of :math:`\qhat` with itself is given by

        .. math::
           \qhat \otimes \qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}\qhat
               \\ \vdots \\
               \hat{q}_{r}\qhat
           \end{array}\right]
           =
           \left[\begin{array}{c}
               \hat{q}_{1}^{2} \\
               \hat{q}_{1}\hat{q}_{2} \\
               \vdots \\
               \hat{q}_{1}\hat{q}_{r} \\
               \hat{q}_{1}\hat{q}_{2} \\
               \hat{q}_{2}^{2} \\
               \vdots \\
               \hat{q}_{2}\hat{q}_{r} \\
               \vdots
               \hat{q}_{r}^{2}
           \end{array}\right] \in\RR^{r^2}.

        Cross terms :math:`\hat{q}_i \hat{q}_j` for :math:`i \neq j` appear
        twice in :math:`\qhat\otimes\qhat`.
        The *compressed Kronecker product* :math:`\qhat\hat{\otimes}\qhat`
        consists of the unique terms of :math:`\qhat\otimes\qhat`:

        .. math::
           \qhat\hat{\otimes}\qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}^2
               \\
               \hat{q}_{2}\qhat_{1:2}
               \\ \vdots \\
               \hat{q}_{r}\qhat_{1:r}
           \end{array}\right]
           = \left[\begin{array}{c}
               \hat{q}_{1}^2 \\
               \hat{q}_{1}\hat{q}_{2} \\ \hat{q}_{2}^{2} \\
               \\ \vdots \\ \hline
               \hat{q}_{1}\hat{q}_{r} \\ \hat{q}_{2}\hat{q}_{r}
               \\ \vdots \\ \hat{q}_{r}^{2}
           \end{array}\right]
           \in \RR^{r(r+1)/2},
           \qquad
           \qhat_{1:i}
           = \left[\begin{array}{c}
               \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
           \end{array}\right]
           \in\RR^{i}.

        For matrices, the product is computed columnwise:

        .. math::
           \left[\begin{array}{c|c|c}
               & & \\
               \qhat_0 & \cdots & \qhat_{k-1}
               \\ & &
           \end{array}\right]
           \hat{\otimes}
           \left[\begin{array}{ccc}
               & & \\
               \qhat_0 & \cdots & \qhat_{k-1}
               \\ & &
           \end{array}\right]
           = \left[\begin{array}{ccc}
               & & \\
               \qhat_0\hat{\otimes}\qhat_0
               & \cdots &
               \qhat_{k-1}\hat{\otimes}\qhat_{k-1}
               \\ & &
           \end{array}\right]
           \in \RR^{r(r+1)/2 \times k}.

        Parameters
        ----------
        state : (r,) or (r, k) numpy.ndarray
            State vector or matrix where each column is a state vector.

        Returns
        -------
        product : (r(r+1)/2,) or (r(r+1)/2, k) ndarray
            The compressed Kronecker product of ``state`` with itself.
        """
        return np.concatenate(
            [state[i] * state[: i + 1] for i in range(state.shape[0])],
            axis=0,
        )

    @staticmethod
    def ckron_indices(r):
        """Construct a mask for efficiently computing the compressed Kronecker
        product.

        This method provides a faster way to evaluate :meth:`ckron`
        when the state dimension ``r`` is known *a priori*.

        Parameters
        ----------
        r : int
            State dimension.

        Returns
        -------
        mask : ndarray
            Compressed Kronecker product mask.

        Examples
        --------
        >>> from opinf.operators import QuadraticOperator
        >>> r = 20
        >>> mask = QuadraticOperator.ckron_indices(r)
        >>> q = np.random.random(r)
        >>> np.allclose(QuadraticOperator.ckron(q), np.prod(q[mask], axis=1))
        True
        """
        mask = np.zeros((r * (r + 1) // 2, 2), dtype=int)
        count = 0
        for i in range(r):
            for j in range(i + 1):
                mask[count, :] = (i, j)
                count += 1
        return mask

    @staticmethod
    def compress_entries(H):
        r"""Given :math:`\Hhat\in\RR^{a\times r^2}`, construct the matrix
        :math:`\tilde{\H}\in\RR^{a \times r(r+1)/2}` such that
        :math:`\Hhat[\qhat\otimes\qhat] = \tilde{\H}[\qhat\hat{\otimes}\qhat]`
        for all :math:`\qhat\in\RR^{r}` where :math:`\hat{\otimes}` is the
        compressed Kronecker product (see :meth:`ckron`).

        Parameters
        ----------
        H : (a, r^2) ndarray
            Matrix that acts on the full Kronecker product.

        Returns
        -------
        Hc : (a, r(r+1)/2) ndarray
            Matrix that acts on the compressed Kronecker product.

        Examples
        --------
        >>> from opinf.operators import QuadraticOperator
        >>> r = 20
        >>> H = np.random.random((r, r**2))
        >>> H.shape
        (20, 400)
        >>> Htilde = QuadraticOperator.compress_entries(H)
        >>> Htilde.shape
        (20, 210)
        >>> q = np.random.random(r)
        >>> Hq2 = H @ np.kron(q, q)
        >>> np.allclose(Hq2, Htilde @ QuadraticOperator.ckron(q))
        True
        """
        if np.ndim(H) == 1:
            H = np.atleast_2d(H)
        r2 = H.shape[1]
        if (r := int(round(r2 ** (1 / 2), 0))) ** 2 != r2:
            raise ValueError(
                f"invalid shape (a, r2) = {H.shape} "
                "with r2 not a perfect square"
            )
        Hc = np.empty((H.shape[0], r * (r + 1) // 2))

        fj = 0
        for i in range(r):
            for j in range(i + 1):
                if i == j:  # Place column for unique term.
                    Hc[:, fj] = H[:, (i * r) + j]
                else:  # Combine columns for repeated terms.
                    Hc[:, fj] = H[:, (i * r) + j] + H[:, (j * r) + i]
                fj += 1

        return Hc

    @staticmethod
    def expand_entries(Hc):
        r"""Given :math:`\tilde{\H}\in\RR^{a \times r(r+1)/2}`, construct the
        matrix :math:`\Hhat\in\RR^{a\times r^2}` such that
        :math:`\Hhat[\qhat\otimes\qhat] = \tilde{\H}[\qhat\hat{\otimes}\qhat]`
        for all :math:`\qhat\in\RR^{r}` where :math:`\hat{\otimes}` is the
        compressed Kronecker product (see :meth:`ckron`).

        Parameters
        ----------
        Hc : (a, r(r+1)/2) ndarray
            Matrix that acts on the compressed Kronecker product.

        Returns
        -------
        H : (a, r^2) ndarray
            Matrix that acts on the full Kronecker product.
            This matrix is "symmetric" in the sense that
            ``H.reshape((a, r, r))[i]`` is symmetric for `i = 0, ..., r`.

        Examples
        --------
        >>> from opinf.operators import QuadraticOperator
        >>> r = 20
        >>> Htilde = np.random.random((r, r * (r + 1) / 2))
        >>> Htilde.shape
        (20, 210)
        >>> H = QuadraticOperator.expand_entries(Htilde)
        >>> H.shape
        (20, 400)
        >>> q = np.random.random(r)
        >>> Hq2 = H @ np.kron(q, q)
        >>> np.allclose(Hq2, Htilde @ QuadraticOperator.ckron(q))
        True
        >>> np.all(QuadraticOperator.compress_entries(G) == Gtilde)
        True
        """
        if np.ndim(Hc) == 1:
            Hc = np.atleast_2d(Hc)
        b = Hc.shape[1]
        r = int(round(np.sqrt(1 + 8 * b) / 2 - 0.5, 0))
        if r * (r + 1) // 2 != b:
            raise ValueError(
                f"invalid shape (a, r2) = {Hc.shape} "
                "with r2 != r(r+1)/2 for any integer r"
            )

        H = np.empty((Hc.shape[0], r**2))
        fj = 0
        for i in range(r):
            for j in range(i + 1):
                if i == j:  # Place column for unique term.
                    H[:, (i * r) + j] = Hc[:, fj]
                else:  # Distribute columns equally for repeated terms.
                    fill = Hc[:, fj] / 2
                    H[:, (i * r) + j] = fill
                    H[:, (j * r) + i] = fill
                fj += 1

        return H


class CubicOperator(OpInfOperator):
    r"""Cubic state operator
    :math:`\Ophat_{\ell}(\qhat,\u) = \Ghat[\qhat\otimes\qhat\otimes\qhat]`
    where :math:`\Ghat\in\RR^{r \times r^{3}}`.

    Internally, the action of the operator is computed as the product of a
    :math:`r \times r(r+1)(r+2)/6` matrix and a compressed version of the
    triple Kronecker product :math:`\qhat \otimes \qhat \otimes \qhat`.

    Parameters
    ----------
    entries : (r, r^3) or (r, r(r+1)(r+2)/6) or (r, r, r, r) ndarray or None
        Operator entries :math:`\Ghat`.

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

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"G[{statestr} ⊗ {statestr} ⊗ {statestr}]"

    def _clear(self):
        """Delete operator ``entries`` and related attributes."""
        self._mask = None
        self._prejac = None
        OpInfOperator._clear(self)

    def _precompute_jacobian_jit(self):
        """Compute (just in time) the pre-Jacobian tensor Jt such that
        (Jt @ q) @ q = jacobian(q).
        """
        r = self.entries.shape[0]
        Gt = self.expand_entries(self.entries).reshape((r, r, r, r))
        self._prejac = Gt + Gt.transpose(0, 2, 1, 3) + Gt.transpose(0, 3, 1, 2)

    def set_entries(self, entries):
        r"""Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, r^3) or (r, r(r+1)(r+2)/6) or (r, r, r, r) ndarray
            Operator entries :math:`\Ghat`.
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
            entries = self.compress_entries(entries)
        elif r3 != self.operator_dimension(r):
            raise ValueError("invalid CubicOperator entries dimensions")

        # Precompute compressed Kronecker product mask and Jacobian tensor.
        self._mask = self.ckron_indices(r)
        self._prejac = None

        OpInfOperator.set_entries(self, entries)

    @utils.requires("entries")
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Ghat[\qhat\otimes\qhat\otimes\qhat]`.

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

    @utils.requires("entries")
    def jacobian(self, state, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat_{\ell}(\qhat,\u)
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
        if self._prejac is None:
            self._precompute_jacobian_jit()
        q_ = np.atleast_1d(state)
        return (self._prejac @ q_) @ q_

    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\widehat{\mathbf{G}} =
        (\Wr\trp\Vr)^{-1}\Wr\trp\mathbf{G}[\Vr\otimes\Vr\otimes\Vr]`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.CubicOperator`
            Projected operator.
        """

        def _pg(G, V):
            return self.expand_entries(G) @ np.kron(V, np.kron(V, V))

        return self._galerkin(Vr, Wr, _pg)

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri--Rao product of the state with itself three times:
        :math:`\Qhat\odot\Qhat\odot\Qhat` where :math:`\Qhat` is ``states``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Ghat` and
        :math:`\d_{\ell}(\qhat,\u) = \qhat\otimes\qhat\otimes\qhat`,
        the data block should be

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \qhat_0\otimes\qhat_0\otimes\qhat_0
           & \cdots &
           \qhat_{k-1}\otimes\qhat_{k-1}\otimes\qhat_{k-1}
           \end{array}\right]
           \in \RR^{r^3 \times k}.

        Internally, a compressed triple Kronecker product with
        :math:`r(r+1)(r+2)/2 < r^{2}` degrees of freedom is used for
        efficiency, hence the data block is actually

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \qhat_0\tilde{\otimes}\qhat_0\tilde{\otimes}\qhat_0
           & \cdots &
           \qhat_{k-1}\tilde{\otimes}\qhat_{k-1}\tilde{\otimes}\qhat_{k-1}
           \end{array}\right]
           \in\RR^{r(r+1)(r+2)/6 \times k}.

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
            Compressed triple Khatri--Rao product of ``states`` with itself.
        """
        return CubicOperator.ckron(np.atleast_2d(states))

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

    # Utilities ---------------------------------------------------------------
    @staticmethod
    def ckron(state):
        r"""Calculate the compressed cubic Kronecker product of a vector with
        itself.

        For a vector :math:`\qhat = [~\hat{q}_{1}~~\cdots~~\hat{q}_{r}~]\trp`,
        the cubic Kronecker product of :math:`\qhat` with itself is given by

        .. math::
           \qhat \otimes \qhat \otimes \qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}(\qhat \otimes \qhat)
               \\ \vdots \\
               \hat{q}_{r}(\qhat \otimes \qhat)
           \end{array}\right]
           \in\RR^{r^3}.

        Cross terms :math:`\hat{q}_i \hat{q}_j \hat{q}_j` for :math:`i,j,k`
        not all equal appear multiple times in
        :math:`\qhat\otimes\qhat\otimes\qhat`.
        The *compressed cubic Kronecker product*
        :math:`\qhat\hat{\otimes}\qhat\hat{\otimes}\qhat`
        consists of the unique terms of :math:`\qhat\otimes\qhat\otimes\qhat`:

        .. math::
           \qhat\hat{\otimes}\qhat\hat{\otimes}\qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}^3
               \\
               \hat{q}_{2}[\![\qhat\hat{\otimes}\qhat]\!]_{1:2}
               \\ \vdots \\
               \hat{q}_{r}[\![\qhat\hat{\otimes}\qhat]\!]_{1:r}
           \end{array}\right]
           \in \RR^{r(r+1)(r+2)/6}.

        See :meth:`opinf.operators.QuadraticOperator.ckron`.
        For matrices, the product is computed columnwise.

        Parameters
        ----------
        state : (r,) or (r, k) numpy.ndarray
            State vector or matrix where each column is a state vector.

        Returns
        -------
        product : (r(r+1)(r+2)/6,) or (r(r+1)(r+2)/6, k) ndarray
            The compressed triple Kronecker product of ``state`` with itself.
        """
        state2 = QuadraticOperator.ckron(state, False)
        lens = special.binom(np.arange(2, len(state) + 2), 2).astype(int)
        return np.concatenate(
            [state[i] * state2[: lens[i]] for i in range(state.shape[0])],
            axis=0,
        )

    @staticmethod
    def ckron_indices(r):
        """Construct a mask for efficiently computing the compressed Kronecker
        triple product.

        This method provides a faster way to evaluate :meth:`ckron`
        when the state dimension ``r`` is known *a priori*.

        Parameters
        ----------
        r : int
            State dimension.

        Returns
        -------
        mask : ndarray
            Compressed Kronecker product mask.

        Examples
        --------
        >>> from opinf.operators import CubicOperator
        >>> r = 20
        >>> mask = CubicOperator.kron_indices(r)
        >>> q = np.random.random(r)
        >>> np.allclose(CubicOperator.ckron(q), np.prod(q[mask], axis=1))
        True
        """
        mask = np.zeros((r * (r + 1) * (r + 2) // 6, 3), dtype=int)
        count = 0
        for i in range(r):
            for j in range(i + 1):
                for k in range(j + 1):
                    mask[count, :] = (i, j, k)
                    count += 1
        return mask

    @staticmethod
    def compress_entries(G):
        r"""Given :math:`\Ghat\in\RR^{a\times r^2}`, construct the matrix
        :math:`\tilde{\G}\in\RR^{a \times r(r+1)(r+2)/6}` such that
        :math:`\Ghat[\qhat\otimes\qhat\otimes\qhat]
        = \tilde{\G}[\qhat\hat{\otimes}\qhat\hat{\otimes}\qhat]`
        for all :math:`\qhat\in\RR^{r}`
        where :math:`\cdot\hat{\otimes}\cdot\hat{\otimes}\cdot` is the
        compressed cubic Kronecker product (see :meth:`ckron`).

        Parameters
        ----------
        G : (a, r^3) ndarray
            Matrix that acts on the full cubic Kronecker product.

        Returns
        -------
        Gc : (a, r(r+1)(r+2)/6) ndarray
            Matrix that acts on the compressed cubic Kronecker product.

        Examples
        --------
        >>> from opinf.operators import CubicOperator
        >>> r = 20
        >>> G = np.random.random((r, r**3))
        >>> G.shape
        (20, 8000)
        >>> Gtilde = CubicOperator.compress_entries(G)
        >>> Gtilde.shape
        (20, 1540)
        >>> q = np.random.random(r)
        >>> Gq3 = G @ np.kron(q, np.kron(q, q))
        >>> np.allclose(Gq3, Gtilde @ CubicOperator.ckron(q))
        True
        """
        if np.ndim(G) == 1:
            G = np.atleast_2d(G)
        r3 = G.shape[1]
        if (r := int(round(r3 ** (1 / 3), 0))) ** 3 != r3:
            raise ValueError(
                f"invalid shape (a, r3) = {G.shape} "
                "with r3 not a perfect cube"
            )
        Gc = np.empty((G.shape[0], r * (r + 1) * (r + 2) // 6))

        fj = 0
        for i in range(r):
            for j in range(i + 1):
                for k in range(j + 1):
                    idxs = set(itertools.permutations((i, j, k), 3))
                    Gc[:, fj] = np.sum(
                        [G[:, (a * r**2) + (b * r) + c] for a, b, c in idxs],
                        axis=0,
                    )
                    fj += 1

        return Gc

    @staticmethod
    def expand_entries(Gc):
        r"""Given :math:`\tilde{\G}\in\RR^{a \times r(r+1)(r+2)/6}`,
        construct the matrix :math:`\Ghat\in\RR^{a\times r^3}` such that
        :math:`\Ghat[\qhat\otimes\qhat\otimes\qhat]
        = \tilde{\G}[\qhat\hat{\otimes}\qhat\hat{\otimes}\qhat]`
        for all :math:`\qhat\in\RR^{r}`
        where :math:`\cdot\hat{\otimes}\cdot\hat{\otimes}\cdot` is the
        compressed cubic Kronecker product (see :meth:`ckron`).

        Parameters
        ----------
        Gc : (a, r(r+1)(r+2)/2) ndarray
            Matrix that acts on the compressed cubic Kronecker product.

        Returns
        -------
        G : (a, r^3) ndarray
            Matrix that acts on the full cubic Kronecker product.

        Examples
        --------
        >>> from opinf.operators import CubicOperator
        >>> r = 20
        >>> Gtilde = np.random.random((r, r * (r + 1) * (r + 2)/ 6))
        >>> Gtilde.shape
        (20, 1540)
        >>> G = CubicOperator.expand_entries(Gtilde)
        >>> G.shape
        (20, 8000)
        >>> q = np.random.random(r)
        >>> Gq3 = G @ np.kron(q, np.kron(q, q))
        >>> np.allclose(Gq2, Gtilde @ CubicOperator.ckron(q))
        True
        >>> np.all(CubicOperator.compress_entries(G) == Gtilde)
        True
        """
        if np.ndim(Gc) == 1:
            Gc = np.atleast_2d(Gc)
        b = Gc.shape[1]
        r = CubicOperator._rfromcompressed(b)
        if r * (r + 1) * (r + 2) // 6 != b:
            raise ValueError(
                f"invalid shape (a, r3) = {Gc.shape} "
                "with r3 != r(r+1)(r+2)/6 for any integer r"
            )

        G = np.empty((Gc.shape[0], r**3))
        fj = 0
        for i in range(r):
            for j in range(i + 1):
                for k in range(j + 1):
                    idxs = set(itertools.permutations((i, j, k), 3))
                    fill = Gc[:, fj] / len(idxs)
                    for a, b, c in idxs:
                        G[:, (a * r**2) + (b * r) + c] = fill
                    fj += 1

        return G

    @staticmethod
    def _rfromcompressed(b: int, maxiters: int = 10, tol: float = 0.25) -> int:
        """Compute r such that r(r+1)(r+2)/6 = b via 1D Newton's method."""
        r = int(b ** (1 / 3))
        _6b = 6 * b
        for _ in range(maxiters):
            _3r2 = 3 * r**2
            rnew = r - (r**3 + _3r2 + 2 * r - _6b) / (_3r2 + 6 * r + 2)
            if abs(r - rnew) < tol:
                return int(round(rnew, 0))
            r = rnew
        raise ValueError(  # pragma: no cover
            f"Newton solve for r such that r(r+1)(r+2)/6 = {b} failed"
        )


# Dependent on input but not on state =========================================
class InputOperator(OpInfOperator, InputMixin):
    r"""Linear input operator :math:`\Ophat_{\ell}(\qhat,\u) = \Bhat\u`
    where :math:`\Bhat \in \RR^{r \times m}`.

    Parameters
    ----------
    entries : (r, m) ndarray or None
        Operator entries :math:`\Bhat`.

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
        r"""Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, m) ndarray
            Operator entries :math:`\Bhat`.
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

        OpInfOperator.set_entries(self, entries)

    @utils.requires("entries")
    def apply(self, state, input_):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Bhat\u`.

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

    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\Bhat = (\Wr\trp\Vr)^{-1}\Wr\trp\B`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.InputOperator`
            Projected operator.
        """
        return self._galerkin(Vr, Wr, lambda B, V: B)

    @staticmethod
    def datablock(states, inputs):
        r"""Return the data matrix block corresponding to the operator,
        the ``inputs``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Bhat` and
        :math:`\d_{\ell}(\qhat,\u) = \u`, the data block is

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \u_0 & \cdots & \u_{k-1}
           \end{array}\right]
           \in \RR^{r \times k}.

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
class StateInputOperator(OpInfOperator, InputMixin):
    r"""Linear state / input interaction operator
    :math:`\Ophat_{\ell}(\qhat,\u) = \Nhat[\u\otimes\qhat]`
    where :math:`\Nhat \in \RR^{r \times rm}`.

    Parameters
    ----------
    entries : (r, rm) ndarray or None
        Operator entries :math:`\Nhat`.

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
        r"""Set the ``entries`` attribute.

        Parameters
        ----------
        entries : (r, rm) ndarray
            Operator entries :math:`\Nhat`.
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

        OpInfOperator.set_entries(self, entries)

    @utils.requires("entries")
    def apply(self, state, input_):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Nhat[\u\otimes\qhat]`.

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

    @utils.requires("entries")
    def jacobian(self, state, input_):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat_{\ell}(\qhat,\u) = \sum_{i=1}^{m}u_{i}\Nhat_{i}`
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

    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\widehat{\mathbf{N}} =
        (\Wr\trp\Vr)^{-1}\Wr\trp\mathbf{N}[\I_{m}\otimes\Vr]`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.StateInputOperator`
            Projected operator.
        """

        def _pg(N, V):
            r, rm = N.shape
            m = rm // r
            return N @ np.kron(np.eye(m), V)

        return self._galerkin(Vr, Wr, _pg)

    @staticmethod
    def datablock(states, inputs):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri--Rao product :math:`\U\odot\Qhat` where
        :math:`\Qhat` is ``states`` and :math:`\U` is ``inputs``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Nhat` and
        :math:`\d_{\ell}(\qhat,\u) = \u\otimes\qhat`, the data block is

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \u_0 \otimes \qhat_0 & \cdots & \u_{k-1} \otimes \qhat_{k-1}
           \end{array}\right]
           \in \RR^{rm \times k}.

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
            Compressed Khatri-Rao product of the ``input_`` and the ``states``.
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
