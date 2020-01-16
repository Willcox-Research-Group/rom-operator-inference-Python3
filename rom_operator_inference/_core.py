# _core.py
"""Classes for reduction of dynamical systems via operator inference."""

import warnings
import itertools
import numpy as np
from scipy import linalg as la
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from .utils import (lstsq_reg,
                    expand_Hc as Hc2H,
                    compress_H as H2Hc,
                    kron_compact as kron2)


# Helper classes and functions ================================================
class AffineOperator:
    """Class for representing a linear operator with affine structure, i.e.,

        A(µ) = sum_{i=1}^{nterms} θ_{i}(µ) * A_{i}.

    The matrix A(µ) is constructed by calling the object once the coefficient
    functions and constituent matrices are set.

    Attributes
    ----------
    nterms : int
        The number of terms in the sum defining the linear operator.

    coefficient_functions : list of nterms callables
        The coefficient scalar-valued functions that define the operator.
        Each must take the same sized input and return a scalar.

    matrices : list of nterms ndarrays of the same shape
        The constituent matrices defining the linear operator.
    """
    def __init__(self, coeffs, matrices=None):
        self.coefficient_functions = coeffs
        self.nterms = len(coeffs)
        if matrices:
            self.matrices = matrices
        else:
            self._ready = False

    @property
    def matrices(self):
        """Get the constituent matrices."""
        return self._matrices

    @matrices.setter
    def matrices(self, ms):
        """Set the constituent matrices, checking that the shapes are equal."""
        if len(ms) != self.nterms:
            _noun = "matrix" if self.nterms == 1 else "matrices"
            raise ValueError(f"expected {self.nterms} {_noun}, got {len(ms)}")

        # Check that each matrix in the list has the same shape.
        shape = ms[0].shape
        for m in ms:
            if m.shape != shape:
                raise ValueError("affine operator matrix shapes do not match "
                                 f"({m.shape} != {shape})")

        # Store matrix list and shape, and mark as ready (for __call__()).
        self._matrices = ms
        self.shape = shape
        self._ready = True

    def validate_coeffs(self, µ):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        µ : float or (p,) ndarray
            A test input for the coefficient functions.
        """
        for θ in self.coefficient_functions:
            if not callable(θ):
                raise ValueError("coefficients of affine operator must be "
                                 "callable functions")
            elif not np.isscalar(θ(µ)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, µ):
        if not self._ready:
            raise RuntimeError("constituent matrices not initialized!")
        return np.sum([θi(µ)*Ai for θi,Ai in zip(self.coefficient_functions,
                                                 self.matrices)], axis=0)


def trained_model_from_operators(modelclass, modelform, Vr,
                                 c_=None, A_=None, H_=None, Hc_=None, B_=None):
    """Construct a prediction-capable ROM object from the operators of
    the reduced model.

    Parameters
    ----------
    modelclass : type
        One of the ROM classes (e.g., IntrusiveContinuousROM).

    modelform : str
        The structure of the model, a substring of "cAHB".

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c_ : (r,) ndarray or None
        Reduced constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Reduced linear state matrix, or None if 'c' is not in `modelform`.

    H_ : (r,r**2) ndarray or None
        Reduced quadratic state matrix (full size), or None if 'H' is not in
        `modelform`.

    Hc_ : (r,r(r+1)/2) ndarray or None
        Reduced quadratic state matrix (compact), or None if 'H' is not in
        `modelform`. Only used if `H_` is also None.

    B_ : (r,m) ndarray or None
        Reduced input matrix, or None if 'B' is not in `modelform`.

    Returns
    -------
    model : modelclass object
        A new model, ready for predict() calls.
    """
    # Check that the modelclass is valid.
    if not issubclass(modelclass, _BaseROM):
        raise TypeError("modelclass must be derived from _BaseROM")

    # Construct the new model object.
    model = modelclass(modelform)
    model._check_modelform(trained=False)

    # Insert the attributes.
    model.Vr = Vr
    model.n, model.r = Vr.shape
    model.m = None if B_ is None else 1 if B_.ndim == 1 else B_.shape[1]
    model.c_, model.A_, model.B_ = c_, A_, B_
    model.Hc_ = H2Hc(H_) if H_ else Hc_

    # Check that the attributes match the modelform.
    model._check_modelform(trained=True)

    # Construct the ROM operator f_() if there are no system inputs.
    if not model.has_inputs and issubclass(modelclass, _ContinuousROM):
        model._construct_f_()
    elif issubclass(modelclass, _DiscreteROM):
        model._construct_f_()

    return model


# Base classes ================================================================
class _BaseROM:
    """Base class for all rom_operator_inference reduced model classes."""
    _MODEL_KEYS = "cAHB"                # Constant, Linear, Quadratic, Input

    def __init__(self, modelform):
        self.modelform = modelform

    @property
    def modelform(self):
        return self._form

    @modelform.setter
    def modelform(self, form):
        self._form = ''.join(sorted(form,
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
            # Make sure that the required attributes exist and aren't None,
            # and that nonrequired attributes exist but are None.
            for key, s in zip("cAHB", ["c_", "A_", "Hc_", "B_"]):
                if not hasattr(self, s):
                    raise AttributeError(f"attribute '{s}' missing;"
                                         " call fit() to train model")
                attr = getattr(self, s)
                if key in self.modelform and attr is None:
                    raise AttributeError(f"attribute '{s}' is None;"
                                         " call fit() to train model")
                elif key not in self.modelform and attr is not None:
                    raise AttributeError(f"attribute '{s}' should be None;"
                                         " call fit() to train model")

    def _check_inputargs(self, u, argname):
        """Check that self.has_inputs agrees with input arguments."""
        if self.has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since 'B' in modelform")

        if not self.has_inputs and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since 'B' in modelform")


class _DiscreteROM(_BaseROM):
    """Base class for models that solve the discrete ROM problem,

        x_{j+1} = f(x_{j}, u_{j}),         x_{0} = x0.

    The problem may also be parametric, i.e., x and f may depend on an
    independent parameter µ.
    """
    def _construct_f_(self):
        """Define the attribute self.f_ based on the computed operators."""
        self._check_modelform(trained=True)

        # No control inputs, so f = f(x).
        if self.modelform == "c":
            f_ = lambda x_: self.c_
        elif self.modelform == "A":
            f_ = lambda x_: self.A_@x_
        elif self.modelform == "cA":
            f_ = lambda x_: self.c_ + self.A_@x_
        elif self.modelform == "H":
            f_ = lambda x_: self.Hc_@kron2(x_)
        elif self.modelform == "cH":
            f_ = lambda x_: self.c_ + self.Hc_@kron2(x_)
        elif self.modelform == "AH":
            f_ = lambda x_: self.A_@x_ + self.Hc_@kron2(x_)
        elif self.modelform == "cAH":
            f_ = lambda x_: self.c_ + self.A_@x_ + self.Hc_@kron2(x_)
        # Has control inputs, so f = f(x, u).
        elif self.modelform == "B":
            f_ = lambda x_,u: self.B_@u
        elif self.modelform == "cB":
            f_ = lambda x_,u: self.c_ + self.B_@u
        elif self.modelform == "AB":
            f_ = lambda x_,u: self.A_@x_ + self.B_@u
        elif self.modelform == "cAB":
            f_ = lambda x_,u: self.c_ + self.A_@x_ + self.B_@u
        elif self.modelform == "HB":
            f_ = lambda x_,u: self.Hc_@kron2(x_) + self.B_@u
        elif self.modelform == "cHB":
            f_ = lambda x_,u: self.c_ + self.Hc_@kron2(x_) + self.B_@u
        elif self.modelform == "AHB":
            f_ = lambda x_,u: self.A_@x_ + self.Hc_@kron2(x_) + self.B_@u
        elif self.modelform == "cAHB":
            f_ = lambda x_,u: self.c_ + self.A_@x_ + self.Hc_@kron2(x_) + self.B_@u

        self.f_ = f_

    def __str__(self):
        """String representation: the structure of the model."""
        self._check_modelform()
        out = []
        if self.has_constant:  out.append("c")
        if self.has_linear:    out.append("Ax_{j}")
        if self.has_quadratic: out.append("H(x_{j} ⊗ x_{j})")
        if self.has_inputs:    out.append("Bu_{j}")

        return "Reduced-order model structure: x_{j+1} = " + " + ".join(out)

    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be implemented by child classes")

    def predict(self, x0, niters, U=None, **options):
        """Step forward the learned ROM `niters` steps.

        Parameters
        ----------
        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM: (n,niters) ndarray
            The reduced-order solutions to the full-order system, including
            the (projected) given initial condition.
        """
        # Verify modelform.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        # Check dimensions.
        if x0.shape != (self.n,):
            raise ValueError("invalid initial state shape "
                             f"({x0.shape} != {(self.n,)})")

        # Verify iteration argument.
        if not isinstance(niters, int) or niters < 0:
            raise ValueError("argument 'niters' must be a nonnegative integer")

        # Project initial conditions.
        x0_ = self.Vr.T @ x0

        # Create the solution array and fill in the initial condition.
        X_ = np.empty((self.Vr.shape[1],niters))
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

        # Reconstruct the approximation to the full-order model.
        return self.Vr @ X_


class _ContinuousROM(_BaseROM):
    """Base class for models that solve the continuous (ODE) ROM problem,

        dx / dt = f(t, x(t), u(t)),         x(0) = x0.

    The problem may also be parametric, i.e., x and f may depend on an
    independent parameter µ.
    """
    def _construct_f_(self, u=None):
        """Define the attribute self.f_ based on the computed operators and,
        if appropriate, the input function u(t).
        """
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        self._jac = None
        u_ = u
        # No control inputs.
        if self.modelform == "c":
            f_ = lambda t,x_: self.c_
            # self._jac = np.zeros((self.r, self.r))
        elif self.modelform == "A":
            f_ = lambda t,x_: self.A_@x_
            # self._jac = self.A_
        elif self.modelform == "cA":
            f_ = lambda t,x_: self.c_ + self.A_@x_
            # self._jac = self.A_
        elif self.modelform == "H":
            f_ = lambda t,x_: self.Hc_@kron2(x_)
        elif self.modelform == "cH":
            f_ = lambda t,x_: self.c_ + self.Hc_@kron2(x_)
        elif self.modelform == "AH":
            f_ = lambda t,x_: self.A_@x_ + self.Hc_@kron2(x_)
        elif self.modelform == "cAH":
            f_ = lambda t,x_: self.c_ + self.A_@x_ + self.Hc_@kron2(x_)
        # Has control inputs.
        elif self.modelform == "B":
            f_ = lambda t,x_: self.B_@u_(t)
            # self._jac = np.zeros((self.r, self.r))
        elif self.modelform == "cB":
            f_ = lambda t,x_: self.c_ + self.B_@u_(t)
            # self._jac = np.zeros((self.r, self.r))
        elif self.modelform == "AB":
            f_ = lambda t,x_: self.A_@x_ + self.B_@u_(t)
            # self._jac = self.A_
        elif self.modelform == "cAB":
            f_ = lambda t,x_: self.c_ + self.A_@x_ + self.B_@u_(t)
            # self._jac = self.A_
        elif self.modelform == "HB":
            f_ = lambda t,x_: self.Hc_@kron2(x_) + self.B_@u_(t)
        elif self.modelform == "cHB":
            f_ = lambda t,x_: self.c_ + self.Hc_@kron2(x_) + self.B_@u_(t)
        elif self.modelform == "AHB":
            f_ = lambda t,x_: self.A_@x_ + self.Hc_@kron2(x_) + self.B_@u_(t)
        elif self.modelform == "cAHB":
            f_ = lambda t,x_: self.c_ + self.A_@x_ + self.Hc_@kron2(x_) + self.B_@u_(t)
        self.f_ = f_

    def __str__(self):
        """String representation: the structure of the model."""
        self._check_modelform()
        out = []
        if self.has_constant:  out.append("c")
        if self.has_linear:    out.append("Ax(t)")
        if self.has_quadratic: out.append("H(x ⊗ x)(t)")
        if self.has_inputs:    out.append("Bu(t)")

        return "Reduced-order model structure: dx / dt = " + " + ".join(out)

    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be implemented by child classes")

    def predict(self, x0, t, u=None, **options):
        """Simulate the learned ROM with scipy.integrate.solve_ivp().

        Parameters
        ----------
        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

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
        X_ROM: (n,nt) ndarray
            The reduced-order approximation to the full-order system over `t`.
        """
        # Verify modelform.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        # Check dimensions.
        if x0.shape != (self.n,):
            raise ValueError("invalid initial state shape "
                             f"({x0.shape} != {(self.n,)})")

        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Project initial conditions.
        x0_ = self.Vr.T @ x0

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
                U = np.atleast_2d(u.copy())
                if U.shape != (self.m,nt):
                    raise ValueError("invalid input shape "
                                     f"({U.shape} != {(self.m,nt)}")
                u = CubicSpline(t, U, axis=1)

            # Construct the ROM operator if needed (deferred due to u(t)).
            self._construct_f_(u)

        # Integrate the reduced-order model.
        self.sol_ = solve_ivp(self.f_,          # Integrate f_(t,x_)
                              [t[0], t[-1]],    # over this time interval
                              x0_,              # with this initial condition
                              t_eval=t,         # evaluated at these points
                              # jac=self._jac,    # with this Jacobian
                              **options)        # with these solver options.

        # Raise warnings if the integration failed.
        if not self.sol_.success:               # pragma: no cover
            warnings.warn(self.sol_.message, IntegrationWarning)

        # Reconstruct the approximation to the full-order model.
        return self.Vr @ self.sol_.y


# Basic mixins ================================================================
class _InferredMixin:
    """Mixin class for reduced model classes that use operator inference."""
    @staticmethod
    def _check_training_data_shapes(X, Xdot, Vr, U=None):
        """Ensure that X, Xdot, Vr, and U are aligned."""
        if Xdot is not None and X.shape != Xdot.shape:
            raise ValueError(f"shape of X != shape of Xdot "
                             f"({X.shape} != {Xdot.shape})")

        if X.shape[0] != Vr.shape[0]:
            raise ValueError("X and Vr not aligned, first dimension "
                             f"{X.shape[0]} != {Vr.shape[0]}")

        if U is not None and X.shape[-1] != U.shape[-1]:
            raise ValueError("X and U not aligned, last dimension "
                             f"{X.shape[-1]} != {U.shape[-1]}")

    @staticmethod
    def _check_dataset_consistency(arrlist, label):
        """Ensure that each array in the list of arrays is the same shape."""
        shape = arrlist[0].shape
        for arr in arrlist:
            if arr.shape != shape:
                raise ValueError(f"shape of '{label}'"
                                 " inconsistent across samples")

    def fit(self, X, rhs, Vr, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        X : (n,k) ndarray
            Column-wise snapshot training data (each column is a snapshot).

        rhs : (n,k) ndarray
            Column-wise next-iteration (discrete model) or velocity
            (continuous model) training data.

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Px||^2.
            If a nonzero number is provided, the regularization matrix is
            P * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_inputargs(U, 'U')

        # Check and store dimensions.
        self._check_training_data_shapes(X, rhs, Vr, U)
        n,k = X.shape           # Dimension of system, number of shapshots.
        r = Vr.shape[1]         # Number of basis vectors.
        self.n, self.r, self.m = n, r, None

        # Project states and rhs to the reduced subspace.
        X_ = Vr.T @ X
        rhs_ = Vr.T @ rhs
        self.Vr = Vr

        # Construct the "Data matrix" D = [X^T, (X ⊗ X)^T, U^T, 1].
        D_blocks = []
        if self.has_constant:
            D_blocks.append(np.ones((k,1)))

        if self.has_linear:
            D_blocks.append(X_.T)

        if self.has_quadratic:
            X2_ = kron2(X_)
            D_blocks.append(X2_.T)
            _r2 = X2_.shape[0]   # = r(r+1)//2, size of the compact Kronecker.

        if self.has_inputs:
            if U.ndim == 1:
                U = U.reshape((1,k))
            D_blocks.append(U.T)
            m = U.shape[0]
            self.m = m

        D = np.hstack(D_blocks)
        self.datacond_ = np.linalg.cond(D)      # Condition number of data.

        # Solve for the reduced-order model operators via least squares.
        Otrp, res = lstsq_reg(D, rhs_.T, P)[0:2]
        self.residual_ = np.sum(res)

        # Extract the reduced operators from Otrp.
        i = 0
        if self.has_constant:
            self.c_ = Otrp[i:i+1][0]        # Note that c_ is one-dimensional.
            i += 1
        else:
            self.c_ = None

        if self.has_linear:
            self.A_ = Otrp[i:i+self.r].T
            i += self.r
        else:
            self.A_ = None

        if self.has_quadratic:
            self.Hc_ = Otrp[i:i+_r2].T
            i += _r2
        else:
            self.Hc_ = None

        if self.has_inputs:
            self.B_ = Otrp[i:i+self.m].T
            i += self.m
        else:
            self.B_ = None
            self._construct_f_()

        return self


class _IntrusiveMixin:
    """Mixin class for reduced model classes that use intrusive projection."""
    def _check_operators(self, operators):
        """Check the keys of the operators argument."""
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

    def fit(self, operators, Vr):
        """Compute the reduced model operators via intrusive projection.

        Parameters
        ----------
        operators: dict(str -> ndarray)
            The operators that define the full-order model f.
            Keys must match the modelform:
            * 'c': constant term c.
            * 'A': linear state matrix A.
            * 'H': quadratic state matrix H (either full H or compact Hc).
            * 'B': input matrix B.

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Returns
        -------
        self
        """
        # Verify modelform.
        self._check_modelform()
        self._check_operators(operators)

        # Store dimensions.
        n,r = Vr.shape          # Dimension of system, number of basis vectors.
        self.Vr = Vr
        self.n, self.r = n, r

        # Project FOM operators.
        if self.has_constant:               # Constant term.
            self.c = operators['c']
            if self.c.shape != (n,):
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

        return self


class _NonparametricMixin:
    """Mixin class for non-parametric reduced model classes."""
    @property
    def H_(self):
        """Matricized quadratic tensor; operates on full Kronecker product."""
        return None if self.Hc_ is None else Hc2H(self.Hc_)


class _ParametricMixin:
    """Mixin class for parametric reduced model classes."""
    pass
    # IDEA: check parameter dimension?


# Specialized mixins ==========================================================
class _InterpolatedMixin(_InferredMixin, _ParametricMixin):
    """Mixin class for interpolatory parametric reduced model classes."""
    @property
    def As_(self):
        """The linear state matrices for each submodel."""
        return [m.A_ for m in self.models_] if self.has_linear else None

    @property
    def Hs_(self):
        """The full quadratic state matrices for each submodel."""
        return [m.H_ for m in self.models_] if self.has_quadratic else None

    @property
    def Hcs_(self):
        """The compact quadratic state matrices for each submodel."""
        return [m.Hc_ for m in self.models_] if self.has_quadratic else None

    @property
    def cs_(self):
        """The constant terms for each submodel."""
        return [m.c_ for m in self.models_] if self.has_constant else None

    @property
    def Bs_(self):
        """The linear input matrices for each submodel."""
        return [m.B_ for m in self.models_] if self.has_inputs else None

    @property
    def fs_(self):
        """The reduced-order operators for each submodel."""
        return [m.f_ for m in self.models_]

    @property
    def dataconds_(self):
        """The condition numbers of the data matrices for each submodel."""
        return np.array([m.datacond_ for m in self.models_])

    @property
    def residuals_(self):
        """The residuals for each submodel."""
        return np.array([m.residual_ for m in self.models_])

    def __len__(self):
        """The number of trained models."""
        return len(self.models_) if hasattr(self, "models_") else 0

    def __call__(self, µ, discrete=False):
        """Construct the reduced model corresponding to the parameter µ."""
        A_  = self.A_(µ)  if self.A_  is not None else None
        Hc_ = self.Hc_(µ) if self.Hc_ is not None else None
        c_  = self.c_(µ)  if self.c_  is not None else None
        B_  = self.B_(µ)  if self.B_  is not None else None
        return trained_model_from_operators(
                    modelclass=_DiscreteROM if discrete else _ContinuousROM,
                    modelform=self.modelform,
                    Vr=self.Vr, A_=A_, Hc_=Hc_, c_=c_, B_=B_)

    def fit(self, ModelClass, µs, Xs, Xdots, Vr, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise velocity training data. The ith array Xdots[i]
            corresponds to the ith parameter, µs[i].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Us : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Px||^2.
            If a nonzero number is provided, the regularization matrix is
            P * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_inputargs(Us, 'Us')
        is_continuous = issubclass(ModelClass, _ContinuousROM)

        # Check that parameters are one-dimensional.
        if not np.isscalar(µs[0]):
            raise ValueError("only scalar parameter values are supported")

        # Check that the number of params matches the number of snapshot sets.
        s = len(µs)
        if len(Xs) != s:
            raise ValueError("num parameter samples != num state snapshot "
                             f"sets ({s} != {len(Xs)})")
        if is_continuous and len(Xdots) != s:
            raise ValueError("num parameter samples != num velocity snapshot "
                             f"sets ({s} != {len(Xdots)})")
        elif not is_continuous:
            Xdots = [None for _ in range(s)]

        # Check and store dimensions.
        for X, Xdot in zip(Xs, Xdots):
            self._check_training_data_shapes(X, Xdot, Vr)
        n,k = Xs[0].shape       # Dimension of system, number of shapshots.
        r = Vr.shape[1]         # Number of basis vectors.
        self.n, self.r, self.m = n, r, None

        # Check that all arrays in each list of arrays are the same sizes.
        _tocheck = [(Xs, "X")]
        if is_continuous:
            _tocheck += [(Xdots, "Xdot")]
        if self.has_inputs:
            _tocheck += [(Us, "U")]
            self.m = Us[0].shape[0] if Us[0].ndim == 2 else 1
        else:
            Us = [None]*s
        for dataset, label in _tocheck:
            self._check_dataset_consistency(dataset, label)

        # TODO: figure out how to handle P (scalar, array, list(arrays)).

        # Train one model per parameter sample.
        self.Vr = Vr
        self.models_ = []
        for µ, X, Xdot, U in zip(µs, Xs, Xdots, Us):
            model = ModelClass(self.modelform)
            if is_continuous:
                model.fit(X, Xdot, Vr, U, P)
            else:
                model.fit(X, Vr, U, P)
            model.parameter = µ
            self.models_.append(model)

        # Construct interpolators.
        self.A_ = CubicSpline(µs, self.As_)  if self.has_linear    else None
        self.Hc_= CubicSpline(µs, self.Hcs_) if self.has_quadratic else None
        self.H_ = CubicSpline(µs, self.Hs_)  if self.has_quadratic else None
        self.c_ = CubicSpline(µs, self.cs_)  if self.has_constant  else None
        self.B_ = CubicSpline(µs, self.Bs_)  if self.has_inputs    else None

        return self


class _AffineMixin(_ParametricMixin):
    """Mixin class for affinely parametric reduced model classes."""

    def _check_affines(self, affines, µ=None):
        """Check the keys of the affines argument."""
        # Check for unnecessary affine keys.
        surplus = [repr(key) for key in affines if key not in self.modelform]
        if surplus:
            _noun = "key" + ('' if len(surplus) == 1 else 's')
            raise KeyError(f"invalid affine {_noun} {', '.join(surplus)}")

        if µ is not None:
            for a in affines.values():
                AffineOperator(a).validate_coeffs(µ)

    def __call__(self, µ, discrete=False):
        """Construct the reduced model corresponding to the parameter µ."""
        c_ = self.c_(µ) if isinstance(self.c_, AffineOperator) else self.c_
        A_ = self.A_(µ) if isinstance(self.A_, AffineOperator) else self.A_
        Hc_ = self.Hc_(µ) if isinstance(self.Hc_, AffineOperator) \
                                                             else self.Hc_
        B_ = self.B_(µ) if isinstance(self.B_, AffineOperator) else self.B_
        return trained_model_from_operators(
                    modelclass=_DiscreteROM if discrete else _ContinuousROM,
                    modelform=self.modelform,
                    Vr=self.Vr, c_=c_, A_=A_, Hc_=Hc_, B_=B_)


class _AffineIntrusiveMixin(_IntrusiveMixin, _AffineMixin):
    def fit(self, affines, operators, Vr):
        """Solve for the reduced model operators via intrusive projection.

        Parameters
        ----------
        affines : dict(str -> list(functions))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'B': linear Input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

        operators: dict(str -> ndarray or list(ndarrays))
            The operators that define the full-order model f(t,x;µ).
            Keys must match the modelform:
            * 'c': constant term c(µ).
            * 'A': linear state matrix A(µ).
            * 'H': quadratic state matrix H(µ).
            * 'B': input matrix B(µ).
            Terms with affine structure should be given as a list of the
            constituent matrices. For example, if the linear state matrix has
            the form A(µ) = θ1(µ)A1 + θ2(µ)A2, then 'A' -> [A1, A2].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Returns
        -------
        self
        """
        # Verify modelform, affines, and operators.
        self._check_modelform()
        self._check_affines(affines, None)
        self._check_operators(operators)

        # Store dimensions.
        n,r = Vr.shape          # Dimension of system, number of basis vectors.
        self.Vr = Vr
        self.n, self.r = n, r

        # Project FOM operators.
        if self.has_constant:               # Constant term.
            if 'c' in affines:
                self.c = AffineOperator(affines['c'], operators['c'])
                if self.c.shape != (n,):
                    raise ValueError("basis Vr and FOM operator c not aligned")
                self.c_ = AffineOperator(affines['c'],
                                          [self.Vr.T @ c
                                           for c in self.c.matrices])
            else:
                self.c = operators['c']
                if self.c.shape != (n,):
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

        if self.has_quadratic:               # Quadratic state matrix.
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


# Useable classes =============================================================
# Discrete models (i.e., solving x_{j+1} = f(x_{j},u_{j})) --------------------
class InferredDiscreteROM(_InferredMixin, _NonparametricMixin, _DiscreteROM):
    """Reduced order model for a discrete dynamical system of
    the form

        x_{j+1} = f(x_{j}, u_{j}),              x_{0} = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
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
        Condition number of the data matrix for the least-squares problem.

    residual_ : float
        The squared Frobenius-norm residual of the least-squares problem for
        computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : func((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    def fit(self, X, Vr, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        X : (n,k) ndarray
            Column-wise snapshot training data (each column is a snapshot).

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        U : (m,k-1) or (k-1,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Px||^2.
            If a nonzero number is provided, the regularization matrix is
            P * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        _InferredMixin.fit(self,
                           X[:,:-1], X[:,1:],       # x_{j}'s and x_{j+1}'s.
                           Vr,
                           U[...,:X.shape[1]-1] if U is not None else U,
                           P)
        self._construct_f_()
        return self


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
        Condition number of the data matrix for the least-squares problem.

    residual_ : float
        The squared Frobenius-norm residual of the least-squares problem for
        computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : func((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    def fit(self, operators, Vr):
        """Compute the reduced model operators via intrusive projection.

        Parameters
        ----------
        operators: dict(str -> ndarray)
            The operators that define the full-order model f.
            Keys must match the modelform:
            * 'c': constant term c.
            * 'A': linear state matrix A.
            * 'H': quadratic state matrix H (either full H or compact Hc).
            * 'B': input matrix B.

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Returns
        -------
        self
        """
        _IntrusiveMixin.fit(self, operators, Vr)
        self._construct_f_()
        return self


class InterpolatedInferredDiscreteROM(_InterpolatedMixin, _DiscreteROM):
    """Reduced order model for a high-dimensional discrete dynamical system,
    parametrized by a scalar µ, of the form

        x_{j+1}(µ) = f(x_{j}(µ), u_{j}; µ),     x_{0}(µ) = x0(µ).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    ordinary least-squares problems, then interpolating those models with
    respect to the scalar parameter µ.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
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

    dataconds_ : (s,) ndarray
        Condition number of the data matrix for each least-squares problem.

    residuals_ : (s,) ndarray
        The squared Frobenius-norm residual of each least-squares problem for
        computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : func((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    def __call__(self, µ):
        """Construct the reduced model corresponding to the parameter µ."""
        return _InterpolatedMixin.__call__(self, µ, discrete=True)

    def fit(self, µs, Xs, Vr, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Us : list of s (m,k-1) or (k-1,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Px||^2.
            If a nonzero number is provided, the regularization matrix is
            P * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        _InterpolatedMixin.fit(self, InferredDiscreteROM,
                               µs, Xs, None, Vr, Us, P)
        return self

    def predict(self, µ, x0, niters, U=None):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then step forward this new ROM `niters` steps.

        Parameters
        ----------
        µ : float
            The parameter of interest for the prediction.

        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM: (n,niters) ndarray
            The reduced-order solutions to the full-order system, including
            the (projected) given initial condition.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        model = self(µ)     # See __call__().
        return model.predict(x0, niters, U)


class AffineIntrusiveDiscreteROM(_AffineIntrusiveMixin, _DiscreteROM):
    """Reduced order model for a high-dimensional discrete dynamical system,
    parametrized by a scalar µ, of the form

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

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : func((r,) ndarray, (m,) ndarray) -> (r,)
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    def __call__(self, µ):
        """Construct the reduced model corresponding to the parameter µ."""
        return _AffineMixin.__call__(self, µ, discrete=True)

    def predict(self, µ, x0, niters, U=None):
        """Construct a ROM for the parameter µ by exploiting the affine
        structure of the ROM operators, then step the resulting ROM forward
        `niters` steps.

        Parameters
        ----------
        µ : (p,) ndarray
            The parameter of interest for the prediction.

        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

        niters : int
            The number of times to step the system forward.

        U : (m,niters-1) ndarray
            The inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM: (n,niters) ndarray
            The reduced-order solutions to the full-order system, including
            the (projected) given initial condition.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(U, 'U')

        # TODO: Make sure the parameter µ has the correct dimension.
        # Use the affine structure of the operators to construct a new model.
        model = self(µ)
        return model.predict(x0, niters, U)


# Continuous models (i.e., solving dx / dt = f(t,x(t),u(t))) ------------------
class InferredContinuousROM(_InferredMixin, _NonparametricMixin,
                            _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),             x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

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

    has_inputs : bool
        Whether or not there is a linear input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the data matrix for the least-squares problem.

    residual_ : float
        The squared Frobenius-norm residual of the least-squares problem for
        computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used
        directly in solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : func(float, (r,) ndarray) -> (r,) ndarray
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        Note the signature is f_(t, x_); that is, f_ maps time and reduced
        state to reduced state. Calculated in fit() if 'B' is not in
        `modelform`, and in predict() otherwise.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, X, Xdot, Vr, U=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        X : (n,k) ndarray
            Column-wise snapshot training data (each column is a snapshot).

        Xdot : (n,k) ndarray
            Column-wise velocity training data.

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Px||^2.
            If a nonzero number is provided, the regularization matrix is
            P * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        _InferredMixin.fit(self, X, Xdot, Vr, U, P)
        if not self.has_inputs:
            self._construct_f_()
        return self


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

    Hc : (n,n(n+1)//2) ndarray or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : (n,n**2) ndarray or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    B : (n,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : func(float, (r,) ndarray) -> (r,) ndarray
        The complete learned ROM operator, defined by c_, A_, Hc_, and/or B_.
        Note the signature is f_(t, x_); that is, f_ maps time and reduced
        state to reduced state. Calculated in fit() if 'B' is not in
        `modelform`, and in predict() otherwise.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, operators, Vr):
        """Compute the reduced model operators via intrusive projection.

        Parameters
        ----------
        operators: dict(str -> ndarray)
            The operators that define the full-order model f(t,x).
            Keys must match the modelform:
            * 'c': constant term c.
            * 'A': linear state matrix A.
            * 'H': quadratic state matrix H (either full H or compact Hc).
            * 'B': input matrix B.

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Returns
        -------
        self
        """
        _IntrusiveMixin.fit(self, operators, Vr)
        if not self.has_inputs:
            self._construct_f_()
        return self


class InterpolatedInferredContinuousROM(_InterpolatedMixin, _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs, parametrized
    by a scalar µ, of the form

         dx / dt = f(t, x(t;µ), u(t); µ),       x(0;µ) = x0(µ).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    ordinary least-squares problems, then interpolating those models with
    respect to the scalar parameter µ.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c(µ)
        'A' : Linear state term A(µ)x(t).
        'H' : Quadratic state term H(µ)(x⊗x)(t).
        'B' : Input term B(µ)u(t).
        For example, modelform=="cA" means f(t, x(t); µ) = c(µ) + A(µ)x(t;µ).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c(µ).

    has_linear : bool
        Whether or not there is a linear state term A(µ)x(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(µ)(x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term B(µ)u(t).

    n : int
        The dimension of the original model.

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    s : int
        The number of training parameter samples, so also the number of reduced
        models computed via inference and used in the interpolation.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    dataconds_ : (s,) ndarray
        Condition number of the data matrix for each least-squares problem.

    residuals_ : (s,) ndarray
        The squared Frobenius-norm residual of each least-squares problem (one
        per parameter) for computing the reduced-order model operators.

    As_ : list of s (r,r) ndarrays or None
        Learned ROM linear state matrices, or None if 'A' not in `modelform`.

    Hcs_ : list of s (r,r(r+1)//2) ndarrays or None
        Learned ROM quadratic state matrices (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    Hs_ : list of s (r,r**2) ndarrays or None
        Learned ROM quadratic state matrices (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hcs_ if desired; not used in
        solving the ROM.

    cs_ : list of s (r,) ndarrays or None
        Learned ROM constant terms, or None if 'c' is not in `modelform`.

    Bs_ : list of s (r,m) ndarrays or None
        Learned ROM input matrices, or None if 'B' not in `modelform`.

    fs_ : list of func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operators for each parameter sample, defined by
        cs_, As_, and/or Hcs_. Only available after calling fit() and only if
        there are no inputs ('B' is not in the modelform).

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def __call__(self, µ):
        """Construct the reduced model corresponding to the parameter µ."""
        return _InterpolatedMixin.__call__(self, µ, discrete=False)

    def fit(self, µs, Xs, Xdots, Vr, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise velocity training data. The ith array Xdots[i]
            corresponds to the ith parameter, µs[i].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Us : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Px||^2.
            If a nonzero number is provided, the regularization matrix is
            P * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        _InterpolatedMixin.fit(self, InferredContinuousROM,
                               µs, Xs, Xdots, Vr, Us, P)
        return self

    def predict(self, µ, x0, t, u=None, **options):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then simulate this interpolated ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : float
            The parameter of interest for the prediction.

        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

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
        X_ROM: (n,nt) ndarray
            The reduced-order approximation to the full-order system over `t`.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=True)
        self._check_inputargs(u, 'u')

        model = self(µ)     # See __call__().
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out


class AffineIntrusiveContinuousROM(_AffineIntrusiveMixin, _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

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

    c : func(µ) -> (n,) ndarray; (n,) ndarray; or None
        FOM constant term, or None if 'c' is not in `modelform`.

    A : func(µ) -> (n,n) ndarray; (n,n) ndarray; or None
        FOM linear state matrix, or None if 'A' is not in `modelform`.

    Hc : func(µ) -> (n,n(n+1)//2) ndarray; (n,n(n+1)//2) ndarray; or None
        FOM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`.

    H : func(µ) -> (n,n**2) ndarray; (n,n**2) ndarray; or None
        FOM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`.

    B : func(µ) -> (n,m) ndarray; (n,m) ndarray; or None
        FOM input matrix, or None if 'B' is not in `modelform`.

    c_ : func(µ) -> (r,) ndarray; (r,) ndarray; or None
        Computed ROM constant term, or None if 'c' is not in `modelform`.

    A_ : func(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Computed ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : func(µ) -> (r,r(r+1)//2) ndarray; (r,r(r+1)//2) ndarray; or None
        Computed ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : func(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Computed ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    B_ : func(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Computed ROM input matrix, or None if 'B' is not in `modelform`.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def __call__(self, µ):
        """Construct the reduced model corresponding to the parameter µ."""
        return _AffineMixin.__call__(self, µ, discrete=False)

    def predict(self, µ, x0, t, u=None, **options):
        """Construct a ROM for the parameter µ by exploiting the affine
        structure of the ROM operators, then simulate the resulting ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : (p,) ndarray
            The parameter of interest for the prediction.

        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

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
        X_ROM: (n,nt) ndarray
            The reduced-order approximation to the full-order system over `t`.
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


__all__ = [
            "InferredDiscreteROM", "InferredContinuousROM",
            "IntrusiveDiscreteROM", "IntrusiveContinuousROM",
            "AffineIntrusiveDiscreteROM", "AffineIntrusiveContinuousROM",
            "InterpolatedInferredDiscreteROM",
            "InterpolatedInferredContinuousROM",
          ]


# Future additions ------------------------------------------------------------
# TODO: discrete analogs.
# TODO: jacobians for each model form in the continuous case.
# TODO: better __str__() for parametric classes.
