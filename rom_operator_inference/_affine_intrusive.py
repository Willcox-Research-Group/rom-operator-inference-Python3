# _affine_intrusive.py
"""Affinely parametric ROM classes that use intrusive projection:

Classes
-------
* _AffineIntrusiveMixin(_IntrusiveMixin, _AffineMixin)
* AffineIntrusiveDiscreteROM(_AffineInferredMixin, _DiscreteROM)
* AffineIntrusiveContinuousROM(_AffineInferredMixin, _ContinuousROM)
"""

import numpy as np

from ._base import _ContinuousROM, _DiscreteROM, _ParametricMixin
from ._intrusive import (_IntrusiveMixin,
                         IntrusiveDiscreteROM,
                         IntrusiveContinuousROM)
from ._affine import AffineOperator, _AffineMixin

from .utils import (lstsq_reg,
                    expand_Hc as Hc2H, compress_H as H2Hc,
                    expand_Gc as Gc2G, compress_G as G2Gc,
                    kron2c, kron3c)


__all__ = [
            "AffineIntrusiveDiscreteROM",
            "AffineIntrusiveContinuousROM"
          ]

class _AffineIntrusiveMixin(_IntrusiveMixin, _AffineMixin):
    """Mixin class for affinely parametric intrusive reduced model classes."""
    def fit(self, Vr, affines, operators):
        """Solve for the reduced model operators via intrusive projection.

        Parameters
        ----------
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
            This cannot be set to None, as it is required for projection.

        affines : dict(str -> list(callables))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'G': Cubic state matrix G(µ).
            * 'B': linear Input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

        operators: dict(str -> ndarray or list(ndarrays))
            The operators that define the full-order model f(t,x;µ).
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'G': Cubic state matrix G(µ).
            * 'B': Input matrix B(µ).
            Terms with affine structure should be given as a list of the
            component matrices. For example, if the linear state matrix has
            the form A(µ) = θ1(µ)A1 + θ2(µ)A2, then 'A' -> [A1, A2].

        Returns
        -------
        self
        """
        # Verify modelform, affines, and operators.
        self._check_modelform(trained=False)
        self._check_affines(affines, None)
        self._check_operators(operators)

        # Store dimensions.
        self.Vr = Vr
        self.n, self.r = self.Vr.shape      # Dim of system, num basis vectors.

        # Project FOM operators.
        if self.has_constant:               # Constant term.
            if 'c' in affines:
                self.c = AffineOperator(affines['c'], operators['c'])
                if self.c.shape != (self.n,):
                    raise ValueError("basis Vr and FOM operator c not aligned")
                self.c_ = AffineOperator(affines['c'],
                                          [self.Vr.T @ c
                                           for c in self.c.matrices])
            else:
                self.c = operators['c']
                if self.c.shape != (self.n,):
                    raise ValueError("basis Vr and FOM operator c not aligned")
                self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_linear:                 # Linear state matrix.
            if 'A' in affines:
                self.A = AffineOperator(affines['A'], operators['A'])
                if self.A.shape != (self.n,self.n):
                    raise ValueError("basis Vr and FOM operator A not aligned")
                self.A_ = AffineOperator(affines['A'],
                                          [self.Vr.T @ A @ self.Vr
                                           for A in self.A.matrices])
            else:
                self.A = operators['A']
                if self.A.shape != (self.n,self.n):
                    raise ValueError("basis Vr and FOM operator A not aligned")
                self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if self.has_quadratic:              # Quadratic state matrix.
            _n2 = self.n * (self.n + 1) // 2
            if 'H' in affines:
                H_or_Hc = AffineOperator(affines['H'], operators['H'])
                if H_or_Hc.shape == (self.n,self.n**2):     # It's H.
                    self.H = H_or_Hc
                    self.Hc = AffineOperator(affines['H'],
                                             [H2Hc(H)
                                              for H in H_or_Hc.matrices])
                elif H_or_Hc.shape == (self.n,_n2):         # It's Hc.
                    self.Hc = H_or_Hc
                    self.H = AffineOperator(affines['H'],
                                             [Hc2H(Hc)
                                              for Hc in H_or_Hc.matrices])
                else:
                    raise ValueError("basis Vr and FOM operator H not aligned")
                Vr2 = np.kron(self.Vr, self.Vr)
                self.H_ = AffineOperator(affines['H'],
                                          [self.Vr.T @ H @ Vr2
                                           for H in self.H.matrices])
                self.Hc_ = AffineOperator(affines['H'],
                                          [H2Hc(H_)
                                           for H_ in self.H_.matrices])
            else:
                H_or_Hc = operators['H']
                if H_or_Hc.shape == (self.n,self.n**2):     # It's H.
                    self.H = H_or_Hc
                    self.Hc = H2Hc(self.H)
                elif H_or_Hc.shape == (self.n,_n2):         # It's Hc.
                    self.Hc = H_or_Hc
                    self.H = Hc2H(self.Hc)
                else:
                    raise ValueError("basis Vr and FOM operator H not aligned")
                self.H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
                self.Hc_ = H2Hc(self.H_)
        else:
            self.Hc, self.H, self.Hc_ = None, None, None

        if self.has_cubic:                  # Cubic state matrix.
            _n3 = self.n * (self.n + 1) * (self.n + 2) // 6
            if 'G' in affines:
                G_or_Gc = AffineOperator(affines['G'], operators['G'])
                if G_or_Gc.shape == (self.n,self.n**3):     # It's G.
                    self.G = G_or_Gc
                    self.Gc = AffineOperator(affines['G'],
                                             [G2Gc(G)
                                              for G in G_or_Gc.matrices])
                elif G_or_Gc.shape == (self.n,_n3):         # It's Gc.
                    self.Gc = G_or_Gc
                    self.G = AffineOperator(affines['G'],
                                             [Gc2G(Gc)
                                              for Gc in G_or_Gc.matrices])
                else:
                    raise ValueError("basis Vr and FOM operator G not aligned")
                Vr3 = np.kron(np.kron(self.Vr, self.Vr), self.Vr)
                self.G_ = AffineOperator(affines['G'],
                                          [self.Vr.T @ G @ Vr3
                                           for G in self.G.matrices])
                self.Gc_ = AffineOperator(affines['G'],
                                          [G2Gc(G_)
                                           for G_ in self.G_.matrices])
            else:
                G_or_Gc = operators['G']
                if G_or_Gc.shape == (self.n,self.n**3):     # It's G.
                    self.G = G_or_Gc
                    self.Gc = G2Gc(self.G)
                elif G_or_Gc.shape == (self.n,_n3):         # It's Gc.
                    self.Gc = G_or_Gc
                    self.G = Gc2G(self.Gc)
                else:
                    raise ValueError("basis Vr and FOM operator G not aligned")
                Vr3 = np.kron(np.kron(self.Vr, self.Vr), self.Vr)
                self.G_ = self.Vr.T @ self.G @ Vr3
                self.Gc_ = G2Gc(self.G_)
        else:
            self.Gc, self.G, self.Gc_ = None, None, None

        if self.has_inputs:                 # Linear input matrix.
            if 'B' in affines:
                self.B = AffineOperator(affines['B'], operators['B'])
                if self.B.shape[0] != self.n:
                    raise ValueError("basis Vr and FOM operator B not aligned")
                if len(self.B.shape) == 2:
                    self.m = self.B.shape[1]
                else:                                   # One-dimensional input
                    self.B = AffineOperator(affines['B'],
                                             [B.reshape((-1,1))
                                              for B in self.B.matrices])
                    self.m = 1
                self.B_ = AffineOperator(affines['B'],
                                          [self.Vr.T @ B
                                           for B in self.B.matrices])
            else:
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

        return self


# Affine intrusive models (public) ============================================
class AffineIntrusiveDiscreteROM(_AffineIntrusiveMixin, _DiscreteROM):
    """Reduced order model for a high-dimensional, parametrized discrete
    dynamical system of the form

        x_{j+1}(µ) = f(x_{j}(µ), u_{j}; µ),     x_{0}(µ) = x0(µ),

    where one or more of the operators that compose f have an affine
    dependence on the parameter, e.g., A(µ) = θ1(µ)A1 + θ2(µ)A2 + θ3(µ)A3.
    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

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

    cubic : bool
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

    c : callable(µ) -> (n,) ndarray; (n,) ndarray; or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : callable(µ) -> (n,n) ndarray; (n,n) ndarray; or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    Hc : callable(µ) -> (n,n(n+1)/2) ndarray; (n,n(n+1)/2) ndarray; or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : callable(µ) -> (n,n**2) ndarray; (n,n**2) ndarray; or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    Gc : callable(µ) -> (n,n(n+1)(n+2)/6) ndarray; (n,n(n+1)(n+2)/6) ndarray;
          or None
        FOM cubic state matrix (compact), or None if 'G' is not
        in `modelform`.

    G : callable(µ) -> (n,n**3) ndarray; (n,n**3) ndarray; or None
        FOM cubic state matrix (full size), or None if 'G' is not
        in `modelform`.

    B : callable(µ) -> (n,m) ndarray; (n,m) ndarray; or None
        FOM input matrix, or None if 'B' is not in `modelform`.

    c_ : callable(µ) -> (r,) ndarray; (r,) ndarray; or None
        Computed ROM constant term, or None if 'c' is not in `modelform`.

    A_ : callable(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Computed ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : callable(µ) -> (r,r(r+1)/2) ndarray; (r,r(r+1)/2) ndarray; or None
        Computed ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : callable(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Computed ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : callable(µ) -> (r,r(r+1)(r+2)/6) ndarray; (r,r(r+1)(r+2)/6) ndarray;
          or None
        Computed ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : callable(µ) -> (r,r**3) ndarray; (r,r**3) ndarray; or None
        Computed ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

    B_ : callable(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Computed ROM input matrix, or None if 'B' is not in `modelform`.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def predict(self, µ, x0, niters, U=None):
        """Construct a ROM for the parameter µ by exploiting the affine
        structure of the ROM operators, then step the resulting ROM forward
        `niters` steps.

        Parameters
        ----------
        µ : (p,) ndarray
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) ndarray
            The approximate solutions to the full-order system, including the
            given initial condition.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        # TODO: Make sure the parameter µ has the correct dimension.
        # Use the affine structure of the operators to construct a new model.
        model = self(µ)
        return model.predict(x0, niters, U)


class AffineIntrusiveContinuousROM(_AffineIntrusiveMixin, _ContinuousROM):
    """Reduced order model for a high-dimensional, parametrized system of ODEs
    of the form

        dx / dt = f(t, x(t), u(t); µ),          x(0;µ) = x0(µ),

    where one or more of the operators that compose f have an affine
    dependence on the parameter, e.g., A(µ) = θ1(µ)A1 + θ2(µ)A2 + θ3(µ)A3.
    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        * 'c' : Constant term c(µ).
        * 'A' : Linear state term A(µ)x(t).
        * 'H' : Quadratic state term H(µ)(x⊗x)(t).
        * 'G' : Cubic state term G(µ)(x⊗x⊗x)(t).
        * 'B' : Linear input term B(µ)u(t).
        For example, modelform=="cA" means f(t, x(t); µ) = c(µ) + A(µ)x(t;µ).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c(µ).

    has_linear : bool
        Whether or not there is a linear term A(µ)x(t).

    has_quadratic : bool
        Whether or not there is a quadratic term H(µ)(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic term G(µ)(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is an input term B(µ)u(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the projected reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c : callable(µ) -> (n,) ndarray; (n,) ndarray; or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : callable(µ) -> (n,n) ndarray; (n,n) ndarray; or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    Hc : callable(µ) -> (n,n(n+1)/2) ndarray; (n,n(n+1)/2) ndarray; or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : callable(µ) -> (n,n**2) ndarray; (n,n**2) ndarray; or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    Gc : callable(µ) -> (n,n(n+1)(n+2)/6) ndarray; (n,n(n+1)(n+2)/6) ndarray;
          or None
        FOM cubic state matrix (compact), or None if 'G' is not in `modelform`.

    G : callable(µ) -> (n,n**3) ndarray; (n,n**3) ndarray; or None
        FOM cubic state matrix (full), or None if 'G' is not in `modelform`.

    B : callable(µ) -> (n,m) ndarray; (n,m) ndarray; or None
        FOM input matrix, or None if 'B' is not in `modelform`.

    c_ : callable(µ) -> (r,) ndarray; (r,) ndarray; or None
        Computed ROM constant term, or None if 'c' is not in `modelform`.

    A_ : callable(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Computed ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : callable(µ) -> (r,r(r+1)/2) ndarray; (r,r(r+1)/2) ndarray; or None
        Computed ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : callable(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Computed ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : callable(µ) -> (r,r(r+1)(r+2)/6) ndarray; (r,r(r+1)(r+2)/6) ndarray;
          or None
        Computed ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : callable(µ) -> (r,r**3) ndarray; (r,r**3) ndarray; or None
        Computed ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

    B_ : callable(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Computed ROM input matrix, or None if 'B' is not in `modelform`.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def predict(self, µ, x0, t, u=None, **options):
        """Construct a ROM for the parameter µ by exploiting the affine
        structure of the ROM operators, then simulate the resulting ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : (p,) ndarray
            The parameter of interest for the prediction.

        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        t : (nt,) ndarray
            The time domain over which to integrate the reduced-order system.

        u : callable or (m,nt) ndarray
            The input as a function of time (preferred) or the input at the
            times `t`. If given as an array, u(t) is approximated by a cubic
            spline interpolating the known data points.

        options
            Arguments for solver.integrate.solve_ivp(), such as the following:
            method : str
                The ODE solver for the reduced-order system.
                * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
                * 'RK23': Explicit Runge-Kutta method of order 3(2).
                * 'Radau': Implicit Runge-Kutta method of the Radau IIA family
                    of order 5.
                * 'BDF': Implicit multi-step variable-order (1 to 5) method
                    based on a backward differentiation formula for the
                    derivative.
                * 'LSODA': Adams/BDF method with automatic stiffness detection
                    and switching. This wraps the Fortran solver from ODEPACK.
            max_step : float
                The maximimum allowed integration step size.
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        X_ROM : (n,nt) ndarray
            The approximate solution to the full-order system over `t`.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        # TODO: Make sure the parameter µ has the correct dimension.
        # Use the affine structure of the operators to construct a new model.
        model = self(µ)
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out
