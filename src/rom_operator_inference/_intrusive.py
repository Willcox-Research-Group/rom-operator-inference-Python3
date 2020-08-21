# _intrusive.py
"""Nonparametric ROM classes that use intrusive projection.

Classes
-------
* _IntrusiveMixin
* IntrusiveDiscreteROM(_IntrusiveMixin, _NonparametricMixin, _DiscreteROM)
* IntrusiveContinuousROM(_IntrusiveMixin, _NonparametricMixin, _ContinuousROM)
"""

import numpy as np

from ._base import _DiscreteROM, _ContinuousROM, _NonparametricMixin

from .utils import (expand_Hc as Hc2H, compress_H as H2Hc,
                    expand_Gc as Gc2G, compress_G as G2Gc)


__all__ = [
            "IntrusiveDiscreteROM",
            "IntrusiveContinuousROM"
          ]


class _IntrusiveMixin:
    """Mixin class for reduced model classes that use intrusive projection."""
    def _check_operators(self, operators):
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

    def fit(self, Vr, operators):
        """Compute the reduced model operators via intrusive projection.

        Parameters
        ----------
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
            This cannot be set to None, as it is required for projection.

        operators: dict(str -> ndarray)
            The operators that define the full-order model f.
            Keys must match the modelform:
            * 'c': constant term c.
            * 'A': linear state matrix A.
            * 'H': quadratic state matrix H (either full H or compact Hc).
            * 'G': cubic state matrix H (either full G or compact Gc).
            * 'B': input matrix B.

        Returns
        -------
        self
        """
        # Verify modelform.
        self._check_modelform()
        self._check_operators(operators)

        # Store dimensions.
        self.Vr = Vr
        self.n, self.r = Vr.shape           # Dim of system, num basis vectors.

        # Project FOM operators.
        if self.has_constant:               # Constant term.
            self.c = operators['c']
            if self.c.shape != (self.n,):
                raise ValueError("basis Vr and FOM operator c not aligned")
            self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_linear:                 # Linear state matrix.
            self.A = operators['A']
            if self.A.shape != (self.n,self.n):
                raise ValueError("basis Vr and FOM operator A not aligned")
            self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if self.has_quadratic:              # Quadratic state matrix.
            H_or_Hc = operators['H']
            _n2 = self.n * (self.n + 1) // 2
            if H_or_Hc.shape == (self.n,self.n**2):         # It's H.
                self.H = H_or_Hc
                self.Hc = H2Hc(self.H)
            elif H_or_Hc.shape == (self.n,_n2):             # It's Hc.
                self.Hc = H_or_Hc
                self.H = Hc2H(self.Hc)
            else:
                raise ValueError("basis Vr and FOM operator H not aligned")
            H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
            self.Hc_ = H2Hc(H_)
        else:
            self.Hc, self.H, self.Hc_ = None, None, None

        if self.has_cubic:
            G_or_Gc = operators['G']
            _n3 = self.n * (self.n + 1) * (self.n + 2) // 6
            if G_or_Gc.shape == (self.n,self.n**3):         # It's G.
                self.G = G_or_Gc
                self.Gc = G2Gc(self.G)
            elif G_or_Gc.shape == (self.n,_n3):             # It's Gc.
                self.Gc = G_or_Gc
                self.G = Gc2G(self.Gc)
            else:
                raise ValueError("basis Vr and FOM operator G not aligned")
            G_ = self.Vr.T @ self.G @ np.kron(self.Vr,np.kron(self.Vr,self.Vr))
            self.Gc_ = G2Gc(G_)
        else:
            self.Gc, self.G, self.Gc_ = None, None, None

        if self.has_inputs:                 # Linear input matrix.
            self.B = operators['B']
            if self.B.shape[0] != self.n:
                raise ValueError("basis Vr and FOM operator B not aligned")
            if self.B.ndim == 2:
                self.m = self.B.shape[1]
            else:                                   # One-dimensional input
                self.B = self.B.reshape((-1,1))
                self.m = 1
            self.B_ = self.Vr.T @ self.B
        else:
            self.B, self.B_, self.m = None, None, None

        self._construct_f_()
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
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax.

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x).

    has_inputs : bool
        Whether or not there is a linear input term Bu.

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    Gc_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : (r,r**3) ndarray or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
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
        'H' : Quadratic state term H(x⊗x)(t).
        'B' : Input term Bu(t).
        For example, modelform=="AB" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

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

    Hc : (n,n(n+1)/2) ndarray or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : (n,n**2) ndarray or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    Gc : (n,n(n+1)(n+2)/6) ndarray or None
        FOM cubic state matrix (compact), or None if 'G' is not in `modelform`.

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
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(t, x_) if 'B' is not in `modelform` (no inputs) and
        f_(t, x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and possibly an input function) to reduced state. Calculated in fit().

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    pass
