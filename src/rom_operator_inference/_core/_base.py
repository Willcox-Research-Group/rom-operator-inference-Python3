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
        if not isinstance(self, (_ContinuousROM, _DiscreteROM)):
            raise RuntimeError("abstract class instantiation "
                               "(use _ContinuousROM or _DiscreteROM)")
        self.modelform = modelform
        self.n, self.m, self.r = None, None, None

    @property
    def modelform(self):
        return self.__form

    @modelform.setter
    def modelform(self, form):
        self.__form = ''.join(sorted(form,
                                    key=lambda k: self._MODEL_KEYS.find(k)))

    @property
    def has_constant(self):
        return "c" in self.modelform

    @property
    def has_linear(self):
        return "A" in self.modelform

    @property
    def has_quadratic(self):
        return "H" in self.modelform

    @property
    def has_cubic(self):
        return "G" in self.modelform

    @property
    def has_inputs(self):
        return "B" in self.modelform

    # @property
    # def has_outputs(self):
    #     return "C" in self._form

    def _check_modelform(self, trained=False):
        """Ensure that self.modelform is valid."""
        for key in self.modelform:
            if key not in self._MODEL_KEYS:
                raise ValueError(f"invalid modelform key '{key}'; options "
                                 "are " + ', '.join(self._MODEL_KEYS))

        if trained:
            fixmsg = "call fit() to train model"

            # Ensure required dimensions are not set to None.
            if self.r is None:
                raise AttributeError(f"null reduced dimension 'r'; {fixmsg}")
            if self.has_inputs and (self.m is None):
                raise AttributeError(f"null input dimension 'm'; {fixmsg}")
            shapes = {
                        "c_": (self.r, ),
                        "A_": (self.r, self.r),
                        "H_": (self.r, self.r*(self.r + 1)//2),
                        "G_": (self.r, self.r*(self.r + 1)*(self.r + 2)//6),
                        "B_": (self.r, self.m),
                     }

            for key in self._MODEL_KEYS:
                s = key+'_'
                # Ensure all operator attributes exist.
                if not hasattr(self, s):
                    raise AttributeError(f"attribute '{s}' missing; {fixmsg}")
                attr = getattr(self, s)
                if key in self.modelform:
                    # Ensure required operators are not set to None.
                    if attr is None:
                        raise AttributeError(f"attribute '{s}' is None; "
                                             f"{fixmsg}")
                    # Check shapes of required operators.
                    if attr.shape != shapes[s]:
                        raise ValueError(f"'{s}.shape' must be {shapes[s]} for"
                                         f" r = {self.r} (got {attr.shape})")
                # Ensure nonrequired operators are set to None.
                elif key not in self.modelform and attr is not None:
                    raise AttributeError(f"attribute '{s}' should be None; "
                                         f"{fixmsg}")

    def _set_operators(self, Vr, c_=None, A_=None, H_=None, G_=None, B_=None):
        """Set the ROM operators and corresponding dimensions.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, then r is inferred from one of the reduced operators.

        c_ : (r,) ndarray or None
            Reduced constant term, or None if 'c' is not in `modelform`.

        A_ : (r,r) ndarray or None
            Reduced linear state matrix, or None if 'c' is not in `modelform`.

        H_ : (r,r(r+1)/2) ndarray or None
            Reduced (compact) quadratic state matrix, or None if 'H' is not in
            `modelform`.

        G_ : (r,r(r+1)(r+2)/6) ndarray or None
            Reduced (compact) cubic state matrix (compact), or None if 'G' is
            not in `modelform`.

        B_ : (r,m) ndarray or None
            Reduced input matrix, or None if 'B' is not in `modelform`.

        Returns
        -------
        self
        """
        self._check_modelform(trained=False)

        # Insert the basis (if given) and determine dimensions.
        self.Vr = Vr
        if Vr is not None:
            self.n, self.r = Vr.shape
        else:
            # Determine the dimension r from the existing operators.
            self.n = None
            for op in [c_, A_, H_, G_, B_]:
                if op is not None:
                    self.r = op.shape[0]
                    break
        self.m = None if B_ is None else 1 if B_.ndim == 1 else B_.shape[1]

        # Insert the operators.
        _r2, _r3 = self.r*(self.r + 1)//2, self.r*(self.r + 1)*(self.r + 2)//6
        self.c_, self.A_, self.B_ = c_, A_, B_
        self.H_ = H_ if (H_ is None or H_.shape[1] == _r2) else compress_H(H_)
        self.G_ = G_ if (G_ is None or G_.shape[1] == _r3) else compress_G(G_)

        # Construct the complete reduced model operator from the arguments.
        self._construct_f_()
        return self

    def _check_inputargs(self, u, argname):
        """Check that self.has_inputs agrees with input arguments."""
        if self.has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since 'B' in modelform")

        if not self.has_inputs and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since 'B' in modelform")

    def project(self, S, label="input"):
        """Check the dimensions of S and project it if needed."""
        if S.shape[0] not in (self.r, self.n):
            raise ValueError(f"{label} not aligned with Vr, dimension 0")
        return self.Vr.T @ S if S.shape[0] == self.n else S

    @property
    def operator_norm_(self):
        """Calculate the squared Frobenius norm of the ROM operators."""
        self._check_modelform(trained=True)
        total = 0
        if self.has_constant:
            total += np.sum(self.c_**2)
        if self.has_linear:
            total += np.sum(self.A_**2)
        if self.has_quadratic:
            total += np.sum(self.H_**2)
        if self.has_cubic:
            total += np.sum(self.G_**2)
        if self.has_inputs:
            total += np.sum(self.B_**2)
        return total


class _DiscreteROM(_BaseROM):
    """Base class for models that solve the discrete ROM problem,

        x_{j+1} = f(x_{j}, u_{j}),         x_{0} = x0.

    The problem may also be parametric, i.e., x and f may depend on an
    independent parameter µ.
    """
    def _construct_f_(self):
        """Define the attribute self.f_ based on the computed operators."""
        self._check_modelform(trained=True)

        # Gather the pieces of the function.
        rhs = []
        namespace = {"self": self}
        if self.has_constant:
            rhs.append("self.c_")
        if self.has_linear:
            rhs.append("(self.A_ @ x_)")
        if self.has_quadratic:
            rhs.append("(self.H_ @ kron2c(x_))")
            namespace["kron2c"] = kron2c
        if self.has_cubic:
            rhs.append("(self.G_ @ kron3c(x_))")
            namespace["kron3c"] = kron3c
        if self.has_inputs:
            lhs = "lambda x_,u: "
            rhs.append("(self.B_ @ u)")
        else:
            lhs = "lambda x_: "

        # Evaluate the pieces and bind them to the object as a lambda function.
        self.f_ = eval(lhs + " + ".join(rhs), namespace)

    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be implemented by child classes")

    def predict(self, x0, niters, U=None):
        """Step forward the learned ROM `niters` steps.

        Parameters
        ----------
        x0 : (n,) or (r,) ndarray
            The initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM : (n,niters) or (r,niters) ndarray
            The approximate solution to the system, including the given
            initial condition. If the basis Vr is None, return solutions in the
            reduced r-dimensional subspace (r,niters). Otherwise, map solutions
            to the full n-dimensional space with Vr (n,niters).
        """
        # Verify modelform.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        # Project initial conditions (if needed).
        x0_ = self.project(x0, 'x0')

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
    def _construct_f_(self):
        """Define the attribute self.f_ based on the computed operators."""
        self._check_modelform(trained=True)

        # Gather the pieces of the function.
        rhs = []
        namespace = {"self": self}
        if self.has_constant:
            rhs.append("self.c_")
        if self.has_linear:
            rhs.append("(self.A_ @ x_)")
        if self.has_quadratic:
            rhs.append("(self.H_ @ kron2c(x_))")
            namespace["kron2c"] = kron2c
        if self.has_cubic:
            rhs.append("(self.G_ @ kron3c(x_))")
            namespace["kron3c"] = kron3c
        if self.has_inputs:
            lhs = "lambda t,x_,u: "
            rhs.append("(self.B_ @ u(t))")
        else:
            lhs = "lambda t,x_: "

        # Evaluate the pieces and bind them to the object as a lambda function.
        self.f_ = eval(lhs + " + ".join(rhs), namespace)

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
        # Verify modelform.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        # Project initial conditions (if needed).
        x0_ = self.project(x0, 'x0')

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
        # Ensure that the model is trained (or there is nothing to save).
        self._check_modelform(trained=True)

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
        c_ = self.c_(µ) if callable(self.c_) else self.c_
        A_ = self.A_(µ) if callable(self.A_) else self.A_
        H_ = self.H_(µ) if callable(self.H_) else self.H_
        G_ = self.G_(µ) if callable(self.G_) else self.G_
        B_ = self.B_(µ) if callable(self.B_) else self.B_
        cl = _DiscreteROM if isinstance(self, _DiscreteROM) else _ContinuousROM
        return cl(self.modelform)._set_operators(Vr=self.Vr,
                                                 c_=c_, A_=A_,
                                                 H_=H_, G_=G_, B_=B_)

    def __str__(self):
        """String representation: the structure of the model."""
        if not hasattr(self, "c_"):             # Untrained -> Nonparametric
            return _NonparametricMixin.__str__(self)
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


# Future additions ------------------------------------------------------------
# TODO: save_model() for parametric forms.
# TODO: class _SteadyROM(_BaseROM) for the steady problem.
# TODO: Account for state / input interactions (N?).
# TODO: jacobians for each model form in the continuous case.
# TODO: self.p = parameter size for parametric classes (+ shape checking)
