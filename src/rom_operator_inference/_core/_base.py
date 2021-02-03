# _core/_base.py
"""Base ROM classes and mixins.

Classes
-------
* _BaseROM: base class for all ROM objects.
* _DiscreteROM: base class for all discrete ROMs (difference equations).
* _ContinuousROM: base class for all continuous ROMs (differential equations).
* _NonparametricMixin: base mixin for all ROMs without parameter dependence.
* _ParametricMixin: base mixin for all ROMs with external parameter dependence.
"""

__all__ = []

import os
import h5py
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from ..utils import compress_H, compress_G, kron2c, kron3c


# Base classes (private) ======================================================
class _BaseROM:
    """Base class for all rom_operator_inference reduced model classes."""
    _MODEL_KEYS = "cAHGB"       # Constant, Linear, Quadratic, Cubic, Input.

    def __init__(self, modelform):
        """Set the modelform."""
        if not isinstance(self, (_ContinuousROM, _DiscreteROM)):
            raise RuntimeError("abstract class instantiation "
                               "(use _ContinuousROM or _DiscreteROM)")
        self.modelform = modelform

    def _clear(self):
        """Set private attributes as None, erasing any previously stored basis,
        dimensions, or ROM operators.
        """
        self.__m = None if self.has_inputs else 0
        self.__r = None
        self.__Vr = None
        self.__c_ = None
        self.__A_ = None
        self.__H_ = None
        self.__G_ = None
        self.__B_ = None

    # Properties: modelform ---------------------------------------------------
    @property
    def modelform(self):
        """Structure of the reduced-order model."""
        return self.__form

    @modelform.setter
    def modelform(self, form):
        """Set the modelform, which – if successful – resets the entire ROM."""
        form = ''.join(sorted(form, key=lambda k: self._MODEL_KEYS.find(k)))
        for key in form:
            if key not in self._MODEL_KEYS:
                raise ValueError(f"invalid modelform key '{key}'; options "
                                 "are " + ', '.join(self._MODEL_KEYS))
        self.__form = form
        self._clear()

    @property
    def has_constant(self):
        """Whether or not the ROM has a constant term c."""
        return "c" in self.modelform

    @property
    def has_linear(self):
        """Whether or not the ROM has a linear state term Ax."""
        return "A" in self.modelform

    @property
    def has_quadratic(self):
        """Whether or not the ROM has a quadratic state term H(x ⊗ x)."""
        return "H" in self.modelform

    @property
    def has_cubic(self):
        """Whether or not the ROM has a cubic state term G(x ⊗ x ⊗ x)."""
        return "G" in self.modelform

    @property
    def has_inputs(self):
        """Whether or not the ROM has an input term Bu."""
        return "B" in self.modelform

    # @property
    # def has_outputs(self):
    #     return "C" in self._form

    # Properties: dimensions --------------------------------------------------
    @property
    def n(self):
        """Dimension of the full-order model."""
        return self.Vr.shape[0] if self.Vr is not None else None

    @n.setter
    def n(self, n):
        """Setting this dimension is not allowed, it is always Vr.shape[0]."""
        raise AttributeError("can't set attribute (n = Vr.shape[0])")

    @property
    def m(self):
        """Dimension of the input term, if present."""
        return self.__m

    @m.setter
    def m(self, m):
        """Set input dimension; only allowed if 'B' in modelform
        and the operator B_ is None.
        """
        if not self.has_inputs and m != 0:
            raise AttributeError("can't set attribute ('B' not in modelform)")
        elif self.B_ is not None:
            raise AttributeError("can't set attribute (m = B_.shape[1])")
        self.__m = m

    @property
    def r(self):
        """Dimension of the reduced-order model."""
        return self.__r

    @r.setter
    def r(self, r):
        """Set ROM dimension; only allowed if the basis Vr is None."""
        if self.Vr is not None:
            raise AttributeError("can't set attribute (r = Vr.shape[1])")
        if any(op is not None for op in self.operators.values()):
            raise AttributeError("can't set attribute (call fit() to reset)")
        self.__r = r

    # Properties: basis -------------------------------------------------------
    @property
    def Vr(self):
        """Basis for the linear reduced space (e.g., POD ), of shape (n,r)."""
        return self.__Vr

    @Vr.setter
    def Vr(self, Vr):
        """Set the basis, thereby fixing the dimensions n and r."""
        self.__Vr = Vr
        if Vr is not None:
            self.__r = Vr.shape[1]

    @Vr.deleter
    def Vr(self):
        self.__Vr = None

    # Properties: reduced-order operators -------------------------------------
    @property
    def c_(self):
        """ROM constant operator, of shape (r,)."""
        return self.__c_

    @c_.setter
    def c_(self, c_):
        self._check_operator_matches_modelform(c_, 'c')
        if c_ is not None:
            self._check_rom_operator_shape(c_, 'c')
        self.__c_ = c_

    @property
    def A_(self):
        """ROM linear state operator, of shape (r,r)."""
        return self.__A_

    @A_.setter
    def A_(self, A_):
        # TODO: what happens if model.A_ = something but model.r is None?
        self._check_operator_matches_modelform(A_, 'A')
        if A_ is not None:
            self._check_rom_operator_shape(A_, 'A')
        self.__A_ = A_

    @property
    def H_(self):
        """ROM quadratic state opeator, of shape (r,r(r+1)/2)."""
        return self.__H_

    @H_.setter
    def H_(self, H_):
        self._check_operator_matches_modelform(H_, 'H')
        if H_ is not None:
            if H_.shape == (self.r, self.r**2):
                H_ = compress_H(H_)
            self._check_rom_operator_shape(H_, 'H')
        self.__H_ = H_

    @property
    def G_(self):
        """ROM cubic state operator, of shape (r,r(r+1)(r+2)/6)."""
        return self.__G_

    @G_.setter
    def G_(self, G_):
        self._check_operator_matches_modelform(G_, 'G')
        if G_ is not None:
            if G_.shape == (self.r, self.r**3):
                G_ = compress_G(G_)
            self._check_rom_operator_shape(G_, 'G')
        self.__G_ = G_

    @property
    def B_(self):
        """ROM input operator, of shape (r,m)."""
        return self.__B_

    @B_.setter
    def B_(self, B_):
        self._check_operator_matches_modelform(B_, 'B')
        if B_ is not None:
            self._check_rom_operator_shape(B_, 'B')
        self.__B_ = B_

    @property
    def operators(self):
        """A dictionary of the current ROM operators."""
        return {"c_": self.c_,
                "A_": self.A_,
                "H_": self.H_,
                "G_": self.G_,
                "B_": self.B_}

    # Validation methods ------------------------------------------------------
    def _check_operator_matches_modelform(self, operator, key):
        """Raise a TypeError if the given operator is incompatible with the
        modelform.

        Parameters
        ----------
        operator : ndarray or None
            Operator (ndarray, etc.) data to be attached as an attribute.

        key : str
            A single character from 'cAHGB', indicating which operator to set.
        """
        if (key in self.modelform) and (operator is None):
            raise TypeError(f"'{key}' in modelform requires {key}_ != None")
        if (key not in self.modelform) and (operator is not None):
            raise TypeError(f"'{key}' not in modelform requires {key}_ = None")

    def _check_rom_operator_shape(self, operator, key):
        """Ensure that the given operator has the correct shape."""
        # First, check that the required dimensions exist.
        if self.r is None:
            raise AttributeError("no reduced dimension 'r' (call fit())")
        if key == 'B' and (self.m is None):
            raise AttributeError(f"no input dimension 'm' (call fit())")
        r, m = self.r, self.m

        # Check operator shape.
        if key == "c" and operator.shape != (r,):
            raise ValueError(f"c_.shape = {operator.shape}, "
                             f"must be (r,) with r = {r}")
        elif key == "A" and operator.shape != (r,r):
            raise ValueError(f"A_.shape = {operator.shape}, "
                             f"must be (r,r) with r = {r}")
        elif key == "H" and operator.shape != (r, r*(r + 1)//2):
            raise ValueError(f"H_.shape = {operator.shape}, must be "
                             f"(r,r(r+1)/2) with r = {r}")
        elif key == "G" and operator.shape != (r, r*(r + 1)*(r + 2)//6):
            raise ValueError(f"G_.shape = {operator.shape}, must be "
                             f"(r,r(r+1)(r+2)/6) with r = {r}")
        elif key == "B" and operator.shape != (r,m):
            raise ValueError(f"B_.shape = {operator.shape}, must be "
                             f"(r,m) with r = {r}, m = {m}")

    def _check_inputargs(self, u, argname):
        """Check that self.has_inputs agrees with input arguments."""
        # TODO (?): replace with _check_operator_matches_modelform().
        if self.has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since 'B' in modelform")

        if not self.has_inputs and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since 'B' in modelform")

    def _check_is_trained(self):
        """Ensure that the model is trained and ready for prediction."""
        operators = self.operators
        try:
            for key in self.modelform:
                op = operators[key+'_']
                self._check_operator_matches_modelform(op, key)
                self._check_rom_operator_shape(op, key)
        except Exception as e:
            raise AttributeError("model not trained (call fit())") from e

    # Methods -----------------------------------------------------------------
    def set_operators(self, Vr, c_=None, A_=None, H_=None, G_=None, B_=None):
        """Set the ROM operators and corresponding dimensions.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, then r is inferred from one of the reduced operators.

        c_ : (r,) ndarray or None
            Reduced-order constant term.

        A_ : (r,r) ndarray or None
            Reduced-order linear state matrix.

        H_ : (r,r(r+1)/2) ndarray or None
            Reduced-order (compact) quadratic state matrix.

        G_ : (r,r(r+1)(r+2)/6) ndarray or None
            Reduced-order (compact) cubic state matrix.

        B_ : (r,m) ndarray or None
            Reduced-order input matrix.

        Returns
        -------
        self
        """
        self._clear()
        operators = [c_, A_, H_, G_, B_]

        # Save the low-dimensional basis. Sets self.n and self.r if given.
        self.Vr = Vr

        # Set the input dimension 'm'.
        if self.has_inputs:
            if B_ is not None:
                self.m = 1 if len(B_.shape) == 1 else B_.shape[1]
        else:
            self.m = 0

        # Determine the ROM dimension 'r' if no basis was given.
        if Vr is None:
            self.r = None
            for op in operators:
                if op is not None:
                    self.r = op.shape[0]
                    break

        # Insert the operators. Raises exceptions if shapes are bad, etc.
        self.c_, self.A_, self.H_, self.G_, self.B_, = c_, A_, H_, G_, B_

        return self

    def project(self, S, label="input"):
        """Check the dimensions of S and project it if needed."""
        if S.shape[0] not in (self.r, self.n):
            raise ValueError(f"{label} not aligned with Vr, dimension 0")
            # TODO: better message, what if Vr is None?
        return self.Vr.T @ S if S.shape[0] == self.n else S


class _DiscreteROM(_BaseROM):
    """Base class for models that solve the discrete ROM problem,

        x_{j+1} = f(x_{j}, u_{j}),         x_{0} = x0.

    The problem may also be parametric, i.e., x and f may depend on an
    independent parameter µ.
    """
    modelform = property(_BaseROM.modelform.fget,
                         _BaseROM.modelform.fset,
                         _BaseROM.modelform.fdel,
    """Structure of the reduced-order model. Each character
    indicates the presence of a different term in the model:
    'c' : Constant term c
    'A' : Linear state term Ax.
    'H' : Quadratic state term H(x⊗x).
    'G' : Cubic state term G(x⊗x⊗x).
    'B' : Input term Bu.
    For example, modelform=="AB" means f(x,u) = Ax + Bu.
    """)

    def f_(self, x_, u=None):
        """Reduced-order model function for discrete models.

        Parameters
        ----------
        x_ : (r,) ndarray
            Reduced state vector.

        u : (m,) ndarray or None
            Input vector corresponding to x_.
        """
        x_new = np.zeros(self.r, dtype=float)
        if self.has_constant:
            x_new += self.c_
        if self.has_linear:
            x_new += self.A_ @ x_
        if self.has_quadratic:
            x_new += self.H_ @ kron2c(x_)
        if self.has_cubic:
            x_new += self.G_ @ kron3c(x_)
        if self.has_inputs:
            x_new += self.B_ @ u
        return x_new

    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be implemented by child classes")

    def predict(self, x0, niters, U=None):
        """Step forward the learned ROM `niters` steps.

        Parameters
        ----------
        x0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected to
            reduced order (r-vector).

        niters : int
            Number of times to step the system forward.

        U : (m,niters-1) ndarray
            Inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) or (r,niters) ndarray
            The approximate solution to the system, including the given
            initial condition. If the basis Vr is None, return solutions in the
            reduced r-dimensional subspace (r,niters). Otherwise, map solutions
            to the full n-dimensional space with Vr (n,niters).
        """
        self._check_is_trained()

        # Process inputs.
        self._check_inputargs(U, 'U')   # Check input/modelform consistency.
        x0_ = self.project(x0, 'x0')    # Project initial conditions if needed.

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 0:
            raise ValueError("argument 'niters' must be a nonnegative integer")

        # Create the solution array and fill in the initial condition.
        X_ = np.empty((self.r,niters))
        X_[:,0] = x0_.copy()

        # Run the iteration.
        if self.has_inputs:
            if callable(U):
                raise TypeError("input U must be an array, not a callable")
            # Validate shape of input, reshaping if input is 1d.
            U = np.atleast_2d(U)
            if U.ndim != 2 or U.shape[0] != self.m or U.shape[1] < niters - 1:
                raise ValueError("invalid input shape "
                                 f"({U.shape} != {(self.m,niters-1)}")
            for j in range(niters-1):
                X_[:,j+1] = self.f_(X_[:,j], U[:,j])    # f(xj,uj)
        else:
            for j in range(niters-1):
                X_[:,j+1] = self.f_(X_[:,j])            # f(xj)

        # Reconstruct the approximation to the full-order model if possible.
        return self.Vr @ X_ if self.Vr is not None else X_


class _ContinuousROM(_BaseROM):
    """Base class for models that solve the continuous (ODE) ROM problem,

        dx / dt = f(t, x(t), u(t)),         x(0) = x0.

    The problem may also be parametric, i.e., x and f may depend on an
    independent parameter µ.
    """
    modelform = property(_BaseROM.modelform.fget,
                         _BaseROM.modelform.fset,
                         _BaseROM.modelform.fdel,
    """Structure of the reduced-order model. Each character
    indicates the presence of a different term in the model:
    'c' : Constant term c
    'A' : Linear state term Ax(t).
    'H' : Quadratic state term H(x⊗x)(t).
    'G' : Cubic state term G(x⊗x⊗x)(t).
    'B' : Input term Bu(t).
    For example, modelform=="AB" means f(t,x(t),u(t)) = Ax(t) + Bu(t).
    """)

    def f_(self, t, x_, u=None):
        """Reduced-order model function for continuous models.

        Parameters
        ----------
        t : float
            Time, a scalar.

        x_ : (r,) ndarray
            Reduced state vector corresponding to time `t`.

        u : func(float) -> (m,)
            Input function that maps time `t` to an input vector of length m.
        """
        dxdt = np.zeros(self.r, dtype=float)
        if self.has_constant:
            dxdt += self.c_
        if self.has_linear:
            dxdt += self.A_ @ x_
        if self.has_quadratic:
            dxdt += self.H_ @ kron2c(x_)
        if self.has_cubic:
            dxdt += self.G_ @ kron3c(x_)
        if self.has_inputs:
            dxdt += self.B_ @ u(t)
        return dxdt

    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be implemented by child classes")

    def predict(self, x0, t, u=None, **options):
        """Simulate the learned ROM with scipy.integrate.solve_ivp().

        Parameters
        ----------
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
        X_ROM : (n,nt) or (r,nt) ndarray
            The approximate solution to the system over the time domain `t`.
            If the basis Vr is None, return solutions in the reduced
            r-dimensional subspace (r,nt). Otherwise, map the solutions to the
            full n-dimensional space with Vr (n,nt).
        """
        self._check_is_trained()

        # Process inputs.
        self._check_inputargs(u, 'u')   # Check input/modelform consistency.
        x0_ = self.project(x0, 'x0')    # Project initial conditions if needed.

        # Verify time domain.
        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Interpret control input argument `u`.
        if self.has_inputs:
            if callable(u):         # If u is a function, check output shape.
                out = u(t[0])
                if np.isscalar(out):
                    if self.m == 1:     # u : R -> R, wrap output as array.
                        _u = u
                        u = lambda s: np.array([_u(s)])
                    else:               # u : R -> R, but m != 1.
                        raise ValueError("input function u() must return"
                                         f" ndarray of shape (m,)={(self.m,)}")
                elif not isinstance(out, np.ndarray):
                    raise ValueError("input function u() must return"
                                     f" ndarray of shape (m,)={(self.m,)}")
                elif out.shape != (self.m,):
                    message = "input function u() must return" \
                              f" ndarray of shape (m,)={(self.m,)}"
                    if self.m == 1:
                        raise ValueError(message + " or scalar")
                    raise ValueError(message)
            else:                   # u is an (m,nt) array.
                U = np.atleast_2d(u)
                if U.shape != (self.m,nt):
                    raise ValueError("invalid input shape "
                                     f"({U.shape} != {(self.m,nt)}")
                u = CubicSpline(t, U, axis=1)

        # Integrate the reduced-order model.
        fun = (lambda t,x_: self.f_(t, x_, u)) if self.has_inputs else self.f_
        self.sol_ = solve_ivp(fun,              # Integrate f_(t, x_, u)
                              [t[0], t[-1]],    # over this time interval
                              x0_,              # with this initial condition
                              t_eval=t,         # evaluated at these points
                              # jac=self._jac,    # with this Jacobian
                              **options)        # with these solver options.

        # Raise warnings if the integration failed.
        if not self.sol_.success:               # pragma: no cover
            warnings.warn(self.sol_.message, IntegrationWarning)

        # Reconstruct the approximation to the full-order model.
        return self.Vr @ self.sol_.y if self.Vr is not None else self.sol_.y


# Mixins for parametric / nonparametric classes (private) =====================
class _NonparametricMixin:
    """Mixin class for non-parametric reduced model classes."""
    @property
    def O_(self):
        """The r x d(r,m) Operator matrix O_ = [ c_ | A_ | H_ | G_ | B_ ]."""
        self._check_is_trained()

        blocks = []
        if self.has_constant:
            blocks.append(self.c_.reshape((-1,1)))
        if self.has_linear:
            blocks.append(self.A_)
        if self.has_quadratic:
            blocks.append(self.H_)
        if self.has_cubic:
            blocks.append(self.G_)
        if self.has_inputs:
            blocks.append(self.B_)
        return np.hstack(blocks)

    def __str__(self):
        """String representation: the structure of the model."""
        discrete = isinstance(self, _DiscreteROM)
        x = "x_{j}" if discrete else "x(t)"
        u = "u_{j}" if discrete else "u(t)"
        lhs = "x_{j+1}" if discrete else "dx / dt"
        out = []
        if self.has_constant:
            out.append("c")
        if self.has_linear:
            out.append(f"A{x}")
        if self.has_quadratic:
            out.append(f"H({x} ⊗ {x})")
        if self.has_cubic:
            out.append(f"G({x} ⊗ {x} ⊗ {x})")
        if self.has_inputs:
            out.append(f"B{u}")
        return f"Reduced-order model structure: {lhs} = " + " + ".join(out)

    def save_model(self, savefile, save_basis=True, overwrite=False):
        """Serialize the learned model, saving it in HDF5 format.
        The model can then be loaded with rom_operator_inference.load_model().

        Parameters
        ----------
        savefile : str
            The file to save to. If it does not end with '.h5', this extension
            will be tacked on to the end.

        savebasis : bool
            If True, save the basis Vr as well as the reduced operators.
            If False, only save reduced operators.

        overwrite : bool
            If True and the specified file already exists, overwrite the file.
            If False and the specified file already exists, raise an error.
        """
        self._check_is_trained()

        # Ensure the file is saved in HDF5 format.
        if not savefile.endswith(".h5"):
            savefile += ".h5"

        # Prevent overwriting and existing file on accident.
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(savefile)

        with h5py.File(savefile, 'w') as f:
            # Store metadata: ROM class and model form.
            meta = f.create_dataset("meta", shape=(0,))
            meta.attrs["modelclass"] = self.__class__.__name__
            meta.attrs["modelform"] = self.modelform

            # Store basis (optionally) if it exists.
            if (self.Vr is not None) and save_basis:
                f.create_dataset("Vr", data=self.Vr)

            # Store reduced operators.
            if self.has_constant:
                f.create_dataset("operators/c_", data=self.c_)
            if self.has_linear:
                f.create_dataset("operators/A_", data=self.A_)
            if self.has_quadratic:
                f.create_dataset("operators/H_", data=self.H_)
            if self.has_cubic:
                f.create_dataset("operators/G_", data=self.G_)
            if self.has_inputs:
                f.create_dataset("operators/B_", data=self.B_)


class _ParametricMixin:
    """Mixin class for parametric reduced model classes."""
    def __call__(self, µ):
        """Construct the reduced model corresponding to the parameter µ."""
        if isinstance(self, _DiscreteROM):
            ModelClass = _DiscreteParametricEvaluationROM
        elif isinstance(self, _ContinuousROM):
            ModelClass = _ContinuousParametricEvaluationROM
        else:
            raise RuntimeError

        self._check_is_trained()

        # TODO: Make sure the parameter µ has the correct dimension.
        c_ = self.c_(µ) if callable(self.c_) else self.c_
        A_ = self.A_(µ) if callable(self.A_) else self.A_
        H_ = self.H_(µ) if callable(self.H_) else self.H_
        G_ = self.G_(µ) if callable(self.G_) else self.G_
        B_ = self.B_(µ) if callable(self.B_) else self.B_

        return ModelClass(self.modelform).set_operators(Vr=self.Vr,
                                                        c_=c_, A_=A_,
                                                        H_=H_, G_=G_, B_=B_)

    def __str__(self):
        """String representation: the structure of the model."""

        discrete = isinstance(self, _DiscreteROM)
        x = "x_{j}" if discrete else "x(t)"
        u = "u_{j}" if discrete else "u(t)"
        lhs = "x_{j+1}" if discrete else "dx / dt"

        out = []
        if self.has_constant:
            out.append("c(µ)" if callable(self.c_)  else "c")
        if self.has_linear:
            A = "A(µ)" if callable(self.A_)  else "A"
            out.append(A + f"{x}")
        if self.has_quadratic:
            H = "H(µ)" if callable(self.H_) else "H"
            out.append(H + f"({x} ⊗ {x})")
        if self.has_cubic:
            G = "G(µ)" if callable(self.G_) else "G"
            out.append(G + f"({x} ⊗ {x} ⊗ {x})")
        if self.has_inputs:
            B = "B(µ)" if callable(self.B_)  else "B"
            out.append(B + f"{u}")
        return f"Reduced-order model structure: {lhs} = "+" + ".join(out)


class _DiscreteParametricEvaluationROM(_NonparametricMixin, _DiscreteROM):
    """Discrete-time ROM that is the evaluation of a parametric ROM."""
    pass


class _ContinuousParametricEvaluationROM( _NonparametricMixin, _ContinuousROM):
    """Continuous-time ROM that is the evaluation of a parametric ROM."""
    pass


# Future additions ------------------------------------------------------------
# TODO: save_model() for parametric forms.
# TODO: class _SteadyROM(_BaseROM) for the steady problem.
# TODO: Account for state / input interactions (N?).
# TODO: jacobians for each model form in the continuous case.
# TODO: self.p = parameter size for parametric classes (+ shape checking)
