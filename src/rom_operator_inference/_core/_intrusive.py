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

from ..utils import (expand_Hc as Hc2H, compress_H as H2Hc,
                     expand_Gc as Gc2G, compress_G as G2Gc)


class _IntrusiveMixin:
    """Mixin class for reduced model classes that use intrusive projection."""
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

    def _process_fit_arguments(self, Vr, operators):
        self._check_modelform(trained=False)
        self._check_operators_keys(operators)

        # Make sure Vr is not None (explicitly) and check dimensions.
        if Vr is None:
            raise ValueError("Vr required for intrusive ROMs (got Vr=None)")

        # Store basis and dimensions.
        self.Vr = Vr
        self.n, self.r = Vr.shape       # Full dimension, reduced dimension.

    def _project_operators(self, operators):
        """Project the full-order operators to the reduced-order space."""
        # Project FOM operators.
        if self.has_constant:           # Constant term.
            self.c = operators['c']
            if self.c.shape != (self.n,):
                raise ValueError("basis Vr and FOM operator c not aligned")
            self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_linear:             # Linear state matrix.
            self.A = operators['A']
            if self.A.shape != (self.n,self.n):
                raise ValueError("basis Vr and FOM operator A not aligned")
            self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if self.has_quadratic:          # Quadratic state matrix.
            self.H = operators['H']
            _n2 = self.n * (self.n + 1) // 2
            if self.H.shape != (self.n,self.n**2):
                raise ValueError("basis Vr and FOM operator H not aligned")
            H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
            self.Hc_ = H2Hc(H_)
        else:
            self.H, self.Hc_ = None, None

        if self.has_cubic:              # Cubic state matrix.
            self.G = operators['G']
            _n3 = self.n * (self.n + 1) * (self.n + 2) // 6
            if self.G.shape != (self.n,self.n**3):
                raise ValueError("basis Vr and FOM operator G not aligned")
            G_ = self.Vr.T @ self.G @ np.kron(self.Vr,np.kron(self.Vr,self.Vr))
            self.Gc_ = G2Gc(G_)
        else:
            self.G, self.Gc_ = None, None

        if self.has_inputs:             # Linear input matrix.
            self.B = operators['B']
            if self.B.shape[0] != self.n:
                raise ValueError("basis Vr and FOM operator B not aligned")
            if self.B.ndim == 1:        # One-dimensional input.
                self.B = self.B.reshape((-1,1))
            self.m = self.B.shape[1]
            self.B_ = self.Vr.T @ self.B
        else:
            self.B, self.B_, self.m = None, None, None

        self._construct_f_()

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

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

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

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

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
