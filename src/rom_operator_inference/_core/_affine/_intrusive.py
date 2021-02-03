# _core/_affine/intrusive.py
"""Affinely parametric ROM classes that use intrusive projection.

Classes
-------
* _AffineIntrusiveMixin(_IntrusiveMixin, _AffineMixin)
* AffineIntrusiveDiscreteROM(_AffineInferredMixin, _DiscreteROM)
* AffineIntrusiveContinuousROM(_AffineInferredMixin, _ContinuousROM)
"""

__all__ = [
            "AffineIntrusiveDiscreteROM",
            "AffineIntrusiveContinuousROM",
          ]

import numpy as np

from ._base import AffineOperator, _AffineMixin
from .._base import _ContinuousROM, _DiscreteROM
from .._intrusive import (_IntrusiveMixin,
                          IntrusiveDiscreteROM,
                          IntrusiveContinuousROM)
from ...utils import expand_H, compress_H, expand_G, compress_G


# Affine intrusive mixin (private) ============================================
class _AffineIntrusiveMixin(_IntrusiveMixin, _AffineMixin):
    """Mixin class for affinely parametric intrusive reduced model classes."""
    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(self, Vr, affines, operators):
        """Validate the arguments to fit() and set the basis."""
        # Verify affine expansions.
        self._check_affines_keys(affines)
        self._check_operators_keys(operators)

        # Reset all variables and store basis.
        self._clear()
        self.Vr = Vr

    def _project_operators(self, affines, operators):
        """Project the full-order operators to the reduced-order space."""
        if self.has_quadratic or self.has_cubic:
            Vr2 = np.kron(self.Vr, self.Vr)

        # Project FOM operators.
        if self.has_constant:               # Constant term.
            if 'c' in affines:
                self.c = AffineOperator(affines['c'], operators['c'])
                self.c_ = AffineOperator(affines['c'],
                                         [self.Vr.T @ c
                                          for c in self.c.matrices])
            else:
                self.c = operators['c']
                self.c_ = self.Vr.T @ self.c

        if self.has_linear:                 # Linear state matrix.
            if 'A' in affines:
                self.A = AffineOperator(affines['A'], operators['A'])
                self.A_ = AffineOperator(affines['A'],
                                         [self.Vr.T @ A @ self.Vr
                                          for A in self.A.matrices])
            else:
                self.A = operators['A']
                self.A_ = self.Vr.T @ self.A @ self.Vr

        if self.has_quadratic:              # Quadratic state matrix.
            _n2 = self.n * (self.n + 1) // 2
            if 'H' in affines:
                H_or_Hc = AffineOperator(affines['H'], operators['H'])
                if H_or_Hc.shape == (self.n,_n2):
                    self.H = AffineOperator(affines['H'],
                                            [expand_H(Hc)
                                             for Hc in H_or_Hc.matrices])
                else:
                    self.H = H_or_Hc
                self.H_ = AffineOperator(affines['H'],
                                         [compress_H(self.Vr.T @ H @ Vr2)
                                          for H in self.H.matrices])
            else:
                self.H = operators['H']
                self.H_ = self.Vr.T @ self.H @ Vr2

        if self.has_cubic:                  # Cubic state matrix.
            _n3 = self.n * (self.n + 1) * (self.n + 2) // 6
            Vr3 = np.kron(self.Vr, Vr2)
            if 'G' in affines:
                G_or_Gc = AffineOperator(affines['G'], operators['G'])
                if G_or_Gc.shape == (self.n,_n3):
                    self.G = AffineOperator(affines['G'],
                                            [expand_G(Gc)
                                             for Gc in G_or_Gc.matrices])
                else:
                    self.G = G_or_Gc
                self.G_ = AffineOperator(affines['G'],
                                         [compress_G(self.Vr.T @ G @ Vr3)
                                          for G in self.G.matrices])
            else:
                self.G = operators['G']
                self.G_ = self.Vr.T @ self.G @ Vr3

        if self.has_inputs:                 # Linear input matrix.
            if 'B' in affines:
                B = AffineOperator(affines['B'], operators['B'])
                if len(B.shape) == 1:
                    B = AffineOperator(affines['B'],
                                       [b.reshape((-1,1)) for b in B.matrices])
                self.m = B.shape[1]
                self.B = B
                self.B_ = AffineOperator(affines['B'],
                                         [self.Vr.T @ B
                                          for B in self.B.matrices])
            else:
                B = operators['B']
                if B.ndim == 1:
                    B = B.reshape((-1,1))
                self.m = B.shape[1]
                self.B = B
                self.B_ = self.Vr.T @ self.B

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
        self._process_fit_arguments(Vr, affines, operators)
        self._project_operators(affines, operators)
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
        return self(µ).predict(x0, niters, U)


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
            The reduced-order approximation to the full-order system over `t`.
        """
        model = self(µ)
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out
