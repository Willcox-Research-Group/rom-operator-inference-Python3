# _core/_intrusive.py
"""Nonparametric ROM classes that use intrusive projection.

Classes
-------
* _IntrusiveMixin
* IntrusiveDiscreteROM(_IntrusiveMixin, _NonparametricMixin, _DiscreteROM)
* IntrusiveContinuousROM(_IntrusiveMixin, _NonparametricMixin, _ContinuousROM)
"""

__all__ = [
            "IntrusiveDiscreteROM",
            "IntrusiveContinuousROM",
          ]

import numpy as np

from ._base import _DiscreteROM, _ContinuousROM, _NonparametricMixin
from ..utils import (expand_Hc, compress_H,
                     expand_Gc, compress_G)


class _IntrusiveMixin:
    """Mixin class for reduced model classes that use intrusive projection."""
    def _clear(self):
        """Set private attributes as None, erasing any previously stored basis,
        dimensions, or ROM operators.
        """
        self._BaseROM__m = None if self.has_inputs else 0
        self._BaseROM__c_ = None
        self._BaseROM__A_ = None
        self._BaseROM__H_ = None
        self._BaseROM__G_ = None
        self._BaseROM__B_ = None
        self.__Vr = None
        self.__c = None
        self.__A = None
        self.__H = None
        self.__G = None
        self.__B = None

    # Properties: dimensions --------------------------------------------------
    @property
    def r(self):
        """Dimension of the reduced-order model."""
        return self.Vr.shape[1] if self.Vr is not None else None

    @r.setter
    def r(self, r):
        """Setting this dimension is not allowed, it is always Vr.shape[1]."""
        raise AttributeError("can't set attribute (r = Vr.shape[1])")

    # Properties: basis -------------------------------------------------------
    @property
    def Vr(self):
        """Basis for the linear reduced space (e.g., POD ), of shape (n,r)."""
        return self.__Vr

    @Vr.setter
    def Vr(self, Vr):
        """Set the basis, also defining the dimensions n and r."""
        if Vr is None:
            raise AttributeError("Vr=None not allowed for intrusive ROMs")
        self.__Vr = Vr

    @Vr.deleter
    def Vr(self):
        self.__Vr = None

    # Properties: full-order operators ----------------------------------------
    @property
    def c(self):
        """FOM constant operator, of shape (n,)."""
        return self.__c

    @c.setter
    def c(self, c):
        self._check_operator_matches_modelform(c, 'c')
        if c is not None:
            self._check_fom_operator_shape(c, 'c') ##
        self.__c = c

    @property
    def A(self):
        """FOM linear state operator, of shape (n,n)."""
        return self.__A

    @A.setter
    def A(self, A):
        self._check_operator_matches_modelform(A, 'A')
        if A is not None:
            self._check_fom_operator_shape(A, 'A')
        self.__A = A

    @property
    def H(self):
        """FOM quadratic state opeator, of shape (n,n^2)."""
        return self.__H

    @H.setter
    def H(self, H):
        self._check_operator_matches_modelform(H, 'H')
        if H is not None:
            if H.shape == (self.n, self.n*(self.n + 1)//2):
                H = expand_Hc(H)
            self._check_fom_operator_shape(H, 'H')
        self.__H = H

    @property
    def G(self):
        """FOM cubic state operator, of shape (n,r(r+1)(r+2)/6)."""
        return self.__G

    @G.setter
    def G(self, G):
        self._check_operator_matches_modelform(G, 'G')
        if G is not None:
            if G.shape == (self.n, self.n*(self.n + 1)*(self.n + 2)//6):
                G = expand_Gc(G)
            self._check_fom_operator_shape(G, 'G')
        self.__G = G

    @property
    def B(self):
        """FOM input operator, of shape (n,m)."""
        return self.__B

    @B.setter
    def B(self, B):
        self._check_operator_matches_modelform(B, 'B')
        if B is not None:
            self._check_fom_operator_shape(B, 'B')
        self.__B = B

    @property
    def operators(self):
        """A dictionary of the current FOM and ROM operators."""
        return {"c" : self.c,
                "A" : self.A,
                "H" : self.H,
                "G" : self.G,
                "B" : self.B,
                "c_": self.c_,
                "A_": self.A_,
                "H_": self.H_,
                "G_": self.G_,
                "B_": self.B_}

    # Validation --------------------------------------------------------------
    def _check_fom_operator_shape(self, operator, key):
        """Ensure that the given operator has the correct shape."""
        # Check that the required dimensions exist.
        if self.Vr is None:
            raise AttributeError("no basis 'Vr' (call fit())")
        if key == 'B' and (self.m is None):
            raise AttributeError(f"no input dimension 'm' (call fit())")
        n, m = self.n, self.m

        # Check operator shapes.
        if key == "c" and operator.shape != (n,):
            raise ValueError(f"c.shape = {operator.shape}, "
                             f"must be (n,) with n = {n}")
        elif key == "A" and operator.shape != (n,n):
            raise ValueError(f"A.shape = {operator.shape}, "
                             f"must be (n,n) with n = {n}")
        elif key == "H" and operator.shape != (n, n**2):
            raise ValueError(f"H.shape = {operator.shape}, must be "
                             f"(n,n**2) with n = {n}")
        elif key == "G" and operator.shape != (n, n**3):
            raise ValueError(f"G.shape = {operator.shape}, must be "
                             f"(n,n**3) with n = {n}")
        elif key == "B" and operator.shape != (n,m):
            raise ValueError(f"B.shape = {operator.shape}, must be "
                             f"(n,m) with n = {n}, m = {m}")

    def _check_operators_keys(self, operators):
        """Check the keys of the `operators` argument."""
        # Check for missing operator keys.
        missing = [repr(key) for key in self.modelform if key not in operators]
        if missing:
            _noun = "key" + ('' if len(missing) == 1 else 's')
            raise KeyError(f"missing operator {_noun} {', '.join(missing)}")

        # Check for unnecessary operator keys.
        surplus = [repr(key) for key in operators if key not in self.modelform]
        if surplus:
            _noun = "key" + ('' if len(surplus) == 1 else 's')
            raise KeyError(f"invalid operator {_noun} {', '.join(surplus)}")

    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(self, Vr, operators):
        """Validate the arguments to fit() and set the basis."""
        self._check_operators_keys(operators)
        self._clear()
        self.Vr = Vr

    def _project_operators(self, operators):
        """Project the full-order operators to the reduced-order space."""
        if self.has_quadratic or self.has_cubic:
            Vr2 = np.kron(self.Vr, self.Vr)

        # Project FOM operators.
        if self.has_constant:           # Constant term.
            self.c = operators['c']
            self.c_ = self.Vr.T @ self.c

        if self.has_linear:             # Linear state matrix.
            self.A = operators['A']
            self.A_ = self.Vr.T @ self.A @ self.Vr

        if self.has_quadratic:          # Quadratic state matrix.
            self.H = operators['H']
            self.H_ = self.Vr.T @ self.H @ Vr2

        if self.has_cubic:              # Cubic state matrix.
            self.G = operators['G']
            self.G_ = self.Vr.T @ self.G @ np.kron(self.Vr, Vr2)

        if self.has_inputs:             # Linear input matrix.
            B = operators['B']
            if B.ndim == 1:
                B = B.reshape((-1,1))
            self.m = B.shape[1]
            self.B = B
            self.B_ = self.Vr.T @ self.B

    def fit(self, Vr, operators):
        """Compute the reduced model operators via intrusive projection.

        Parameters
        ----------
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
            This cannot be set to None, as it is required for projection.

        operators: dict(str -> ndarray)
            The operators that define the full-order model.
            Keys must match the modelform:
            * 'c': (n,) constant term c.
            * 'A': (n,n) linear state matrix A.
            * 'H': (n,n**2) quadratic state matrix H.
            * 'G': (n,n**3) cubic state matrix G.
            * 'B': (n,m) input matrix B.

        Returns
        -------
        self
        """
        self._process_fit_arguments(Vr, operators)
        self._project_operators(operators)
        return self


# Nonparametric intrusive models ----------------------------------------------
class IntrusiveDiscreteROM(_IntrusiveMixin, _NonparametricMixin, _DiscreteROM):
    """Reduced order model for a discrete dynamical system of the form

        x_{j+1} = f(x_{j}, u_{j}),              x_{0} = x0.

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are computed explicitly by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x ⊗ x).
        'G' : Cubic state term G(x ⊗ x ⊗ x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the projected reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c : (n,) ndarray or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : (n,n) ndarray or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    H : (n,n**2) ndarray or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    G : (n,n**3) ndarray or None
        FOM cubic state matrix (full), or None if 'G' is not in `modelform`.

    B : (n,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    H_ : (r,r(r+1)/2) ndarray or None
        Learned ROM (compact) quadratic state matrix, or None if 'H' is not in
        `modelform`.

    G_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM (compact) cubic state matrix, or None if 'G' is not in
        `modelform`.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable((r,) ndarray, (m,) ndarray) -> (r,)
        The learned ROM operator defined by the reduced-order matrices.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Defined in fit().
    """
    pass


class IntrusiveContinuousROM(_IntrusiveMixin, _NonparametricMixin,
                             _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),             x(0) = x0.

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are computed explicitly by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax(t).
        'H' : Quadratic state term H(x(t) ⊗ x(t)).
        'B' : Input term Bu(t).
        For example, modelform = "AB" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the projected reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c : (n,) ndarray or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : (n,n) ndarray or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    H : (n,n**2) ndarray or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    G : (n,n**3) ndarray or None
        FOM cubic state matrix (full), or None if 'G' is not in `modelform`.

    B : (n,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    H_ : (r,r(r+1)/2) ndarray or None
        Learned ROM (compact) quadratic state matrix, or None if 'H' is not
        in `modelform`.

    G_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM (compat) cubic state matrix (compact), or None if 'G' is
        not in `modelform`.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable(float, (r,) ndarray, func?) -> (r,) ndarray
        The learned ROM operator defined by the reduced-order matrices.
        The signature is f_(t, x_) if 'B' is not in `modelform` (no inputs) and
        f_(t, x_, u) otherwise. That is, f_ maps reduced states (and an input
        function if appropriate) to reduced states. Defined in fit().

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    pass
