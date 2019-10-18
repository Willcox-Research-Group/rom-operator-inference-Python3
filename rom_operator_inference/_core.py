# _core.py
"""Class for model order reduction of ODEs via operator inference."""
# TODO: jacobians for each model form in the continuous case.

import warnings
import itertools
import numpy as np
from scipy import linalg as la
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from . import utils
kron2 = utils.kron_compact


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
        """Set the constituent matrices."""
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


def trained_model_from_operators(modelclass, modelform, Vr, m=None,
                                  c_=None, A_=None, F_=None, B_=None):
    """Construct a prediction-capable ROM object from the operators of
    the reduced model.

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
    model.n, model.r, model.m = Vr.shape + (m,)
    model.c_, model.A_, model.F_, model.B_ = c_, A_, F_, B_

    # Check that the attributes match the modelform.
    model._check_modelform(trained=True)

    # Construct the ROM operator f_() if there are no system inputs.
    if not model.has_inputs and issubclass(modelclass, _ContinuousROM):
        model._construct_f_()

    return model


# Base classes ================================================================
class _BaseROM:
    """Base class for all rom_operator_inference reduced model classes."""
    _MODEL_KEYS = "CLQI"                # Constant, Linear, Quadratic, Input

    def __init__(self, modelform):
        self.modelform = modelform

    @property
    def modelform(self):
        return self._form

    @modelform.setter
    def modelform(self, form):
        self._form = ''.join(sorted(form.upper()))

    @property
    def has_inputs(self):
        return "I" in self.modelform

    # @property
    # def has_outputs(self):
    #     return "O" in self._form

    @property
    def has_constant(self):
        return "C" in self.modelform

    @property
    def has_linear(self):
        return "L" in self.modelform

    @property
    def has_quadratic(self):
        return "Q" in self.modelform

    def _check_modelform(self, trained=False):
        """Ensure that self.modelform is valid."""
        for key in self.modelform:
            if key not in self._MODEL_KEYS:
                raise ValueError(f"invalid modelform key '{key}'; options "
                                 "are " + ', '.join(self._MODEL_KEYS))

        if trained:
            # Make sure that the required attributes exist and aren't None,
            # and that nonrequired attributes exist but are None.
            for key, s in zip("CLQI", ["c_", "A_", "F_", "B_"]):
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

    def _check_hasinputs(self, u, argname):
        """Check that self.has_inputs agrees with input arguments."""
        if self.has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since 'I' in modelform")

        if not self.has_inputs and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since 'I' in modelform")


class _ContinuousROM(_BaseROM):
    """Base class for models that solve the continuous (ODE) ROM problem,

        dx / dt = f(t, x(t), u(t)),         x(0) = x0.

    The problem may also be parametric (i.e., x and f may depend on an
    independent parameter).
    """
    def _construct_f_(self, u=None):
        """Define the attribute self.f_ based on the computed operators and,
        if has_inputs is True, the input function.
        """
        if not self.has_inputs and u is None:
            if self.modelform == "C":
                f_ = lambda t,x_: self.c_
            elif self.modelform == "L":
                f_ = lambda t,x_: self.A_@x_
            elif self.modelform == "CL":
                f_ = lambda t,x_: self.c_ + self.A_@x_
            elif self.modelform == "Q":
                f_ = lambda t,x_: self.F_@kron2(x_)
            elif self.modelform == "CQ":
                f_ = lambda t,x_: self.c_ + self.F_@kron2(x_)
            elif self.modelform == "LQ":
                f_ = lambda t,x_: self.A_@x_ + self.F_@kron2(x_)
            elif self.modelform == "CLQ":
                f_ = lambda t,x_: self.c_ + self.A_@x_ + self.F_@kron2(x_)
        elif self.has_inputs and u is not None:
            u_ = u
            if self.modelform == "I":
                f_ = lambda t,x_: self.B_@u(t)
            elif self.modelform == "CI":
                f_ = lambda t,x_: self.c_ + self.B_@u_(t)
            elif self.modelform == "IL":
                f_ = lambda t,x_: self.A_@x_ + self.B_@u_(t)
            elif self.modelform == "CIL":
                f_ = lambda t,x_: self.c_ + self.A_@x_ + self.B_@u_(t)
            elif self.modelform == "IQ":
                f_ = lambda t,x_: self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "CIQ":
                f_ = lambda t,x_: self.c_ + self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "ILQ":
                f_ = lambda t,x_: self.A_@x_ + self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "CILQ":
                f_ = lambda t,x_: self.c_ + self.A_@x_ + self.F_@kron2(x_) + self.B_@u_(t)
        else:
            raise RuntimeError("improper use of _construct_f_()!")
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

        u : callable OR (m,nt) ndarray
            The input as a function of time (preferred) OR the input at the
            times `t`. If given as an array, u(t) is calculated by linearly
            interpolating known data points if needed for an adaptive solver.

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
        self._check_hasinputs(u, 'u')

        # Check dimensions.
        if x0.shape[0] != self.n:
            raise ValueError("invalid initial state size "
                             f"({x0.shape[0]} != {self.n})")

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
            else:                   # Then u should an (m,nt) array.
                U = np.atleast_2d(u.copy())
                if U.shape != (self.m,nt):
                    raise ValueError("invalid input shape "
                                     f"({U.shape} != {(self.m,nt)}")
                u = CubicSpline(t, U, axis=1)

            # Construct the ROM operator if needed (deferred due to u(t)).
            self._construct_f_(u)

        # Integrate the reduced-order model.
        self.sol_ = solve_ivp(self.f_,          # Integrate f_(t,x)
                              [t[0], t[-1]],    # over this time interval
                              x0_,              # with this initial condition
                              t_eval=t,         # evaluated at these points
                              **options)        # with these solver options.

        # Raise errors if the integration failed.
        if not self.sol_.success:               # pragma: no cover
            warnings.warn(self.sol_.message, IntegrationWarning)

        # Reconstruct the approximation to the full-order model.
        return self.Vr @ self.sol_.y


class _DiscreteROM(_BaseROM):           # pragma: no cover
    """Base class for models that solve the discrete ROM problem,

        x_{k+1} = f(x_{k}, u_{k}),         x_{0} = x0.

    The problem may also be parametric (i.e., x and f may depend on an
    independent parameter).
    """
    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be overwritten by child classes")

    def predict(self, x0, niters, U=None, **options):
        raise NotImplementedError

    def __str__(self):
        """String representation: the structure of the model."""
        self._check_modelform()
        out = []
        if self.has_constant:  out.append("c")
        if self.has_linear:    out.append("Ax_{k}")
        if self.has_quadratic: out.append("H(x_{k} ⊗ x_{k})")
        if self.has_inputs:    out.append("Bu_{k}")

        return "Reduced-order model structure: x_{k+1} = " + " + ".join(out)


class _AffineContinuousROM(_ContinuousROM):
    """Base class for models with affinely parametric operators."""
    def predict(self, p, x0, t, u=None, **options):
        """Construct a ROM for the parameter p by exploiting the affine
        structure of the ROM operators, then simulate the resulting ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        p : (p,) ndarray
            The paramter of interest for the prediction.

        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

        t : (nt,) ndarray
            The time domain over which to integrate the reduced-order system.

        u : callable OR (m,nt) ndarray
            The input as a function of time (preferred) OR the input at the
            times `t`. If given as an array, u(t) is calculated by linearly
            interpolating known data points if needed for an adaptive solver.

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
        self._check_hasinputs(u, 'u')

        # TODO: Make sure the parameter p has the correct dimension.

        # Use the affine structure of the operators to construct a new model.
        model = trained_model_from_operators(
            modelclass=_ContinuousROM,
            modelform=self.modelform,
            Vr=self.Vr,
            m=self.m,
            A_=self.A_(p) if isinstance(self.A_, AffineOperator) else self.A_,
            F_=self.F_(p) if isinstance(self.F_, AffineOperator) else self.F_,
            c_=self.c_(p) if isinstance(self.c_, AffineOperator) else self.c_,
            B_=self.B_(p) if isinstance(self.B_, AffineOperator) else self.B_,
                )
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out



# Mixins ======================================================================
class _InferredMixin:
    """Mixin class for reduced model classes that use operator inference."""
    @staticmethod
    def _check_training_data_shapes(X, Xdot, Vr):
        """Ensure that X, Xdot, and Vr are aligned."""
        if X.shape != Xdot.shape:
            raise ValueError("shape of X != shape of Xdot "
                             f"({X.shape} != {Xdot.shape})")

        if X.shape[0] != Vr.shape[0]:
            raise ValueError("X and Vr not aligned, first dimension "
                             f"{X.shape[0]} != {Vr.shape[0]}")

    @staticmethod
    def _check_dataset_consistency(arrlist, label):
        """Ensure that each array in the list of arrays is the same shape."""
        shape = arrlist[0].shape
        for arr in arrlist:
            if arr.shape != shape:
                raise ValueError(f"shape of '{label}'"
                                 " inconsistent across samples")


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


class _NonparametricMixin:
    """Mixin class for non-parametric reduced model classes."""
    @property
    def H_(self):
        """Matricized quadratic tensor; operates on full Kronecker product."""
        return None if self.F_ is None else utils.F2H(self.F_)


class _ParametricMixin:
    """Mixin class for parametric reduced model classes."""
    pass


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


# Useable classes =============================================================

# Continuous models (i.e., solving dx/dt = f(t,x,u)) --------------------------
class InferredContinuousROM(_ContinuousROM,
                            _InferredMixin, _NonparametricMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t,x(t),u(t)),           x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving a
    regularized ordinary least-squares problem.

    Parameters
    ----------
    modelform : str containing 'C', 'L', 'Q', and/or 'I'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'C' : Constant term c
        'L' : Linear term Ax(t).
        'Q' : Quadratic term H(x⊗x)(t).
        'I' : Input term Bu(t).
        For example, modelform=="LI" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic term H(x⊗x)(t).

    has_inputs : bool
        Whether or not there is an input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if `has_inputs` is False.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the data matrix for the least-squares problem.

    residual_ : float
        The squared Frobenius-norm residual of the least-squares problem for
        computing the reduced-order model operators.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'L' is not in `modelform`.

    F_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'Q' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'Q' is not
        in `modelform`. Computed on the fly from F_ if desired; not used in
        solving the ROM.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'C' is not in `modelform`.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if `has_inputs` is False.

    f_ : func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operator, defined by A_, F_, c_, and/or B_.
        Note the signiture is f_(t, x_); that is, f_ maps time x reduced state
        to reduced state. Calculated in fit() if `has_inputs` is False, and
        in predict() otherwise.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, X, Xdot, Vr, U=None, G=0):
        """Solve for the reduced model operators via regularized least squares.

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
            input), then U may be a one-dimensional array. Required if
            has_inputs is True; must be None if has_inputs is False.

        G : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Gx||^2.
            If a nonzero number is provided, the regularization matrix is
            G * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_hasinputs(U, 'U')

        # Check and store dimensions.
        self._check_training_data_shapes(X, Xdot, Vr)
        n,k = X.shape           # Dimension of system, number of shapshots.
        r = Vr.shape[1]         # Number of basis vectors.
        self.n, self.r, self.m = n, r, None

        # Project states and velocities to the reduced subspace.
        X_ = Vr.T @ X
        Xdot_ = Vr.T @ Xdot
        self.Vr = Vr

        # Construct the "Data matrix" D = [X^T, (X ⊗ X)^T, U^T, 1].
        D_blocks = []
        if self.has_constant:
            D_blocks.append(np.ones(k).reshape((k,1)))

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
        OT, res = utils.lstsq_reg(D, Xdot_.T, G)[0:2]
        self.residual_ = np.sum(res)

        # Extract the reduced operators from OT.
        i = 0
        if self.has_constant:
            self.c_ = OT[i:i+1][0]       # Note that c_ is one-dimensional.
            i += 1
        else:
            self.c_ = None

        if self.has_linear:
            self.A_ = OT[i:i+self.r].T
            i += self.r
        else:
            self.A_ = None

        if self.has_quadratic:
            self.F_ = OT[i:i+_r2].T
            i += _r2
        else:
            self.F_ = None

        if self.has_inputs:
            self.B_ = OT[i:i+self.m].T
            i += self.m
        else:
            self.B_ = None
            self._construct_f_()

        return self


class IntrusiveContinuousROM(_ContinuousROM,
                             _IntrusiveMixin, _NonparametricMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),         x(0) = x0.

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'C', 'L', 'Q', and/or 'I'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'C' : Constant term c
        'L' : Linear term Ax(t).
        'Q' : Quadratic term H(x⊗x)(t).
        'I' : Input term Bu(t).
        For example, modelform=="LI" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic term H(x⊗x)(t).

    has_inputs : bool
        Whether or not there is an input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the projected reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if `has_inputs` is False.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    A : (n,n) ndarray or None
        FOM linear state matrix, or None if 'L' is not in `modelform`.

    F : (n,n(n+1)//2) ndarray or None
        FOM quadratic state matrix (compact), or None if 'Q' is not
        in `modelform`.

    H : (n,n**2) ndarray or None
        FOM quadratic state matrix (full size), or None if 'Q' is not
        in `modelform`.

    c : (n,) ndarray or None
        FOM constant term, or None if 'C' is not in `modelform`.

    B : (n,m) ndarray or None
        Learned ROM input matrix, or None if `has_inputs` is False.

    f : func(float, (r,) ndarray) -> (r,) ndarray
        The complete FOM operator, defined by A, F, c, and/or B.
        Note the signiture is f(t, x); that is, f maps time x state to state.
        Constructed in fit() based off of the FOM operators.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'L' is not in `modelform`.

    F_ : (r,r(r+1)//2) ndarray or None
        Learned ROM quadratic state matrix (compact), or None if 'Q' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : (r,r**2) ndarray or None
        Learned ROM quadratic state matrix (full size), or None if 'Q' is not
        in `modelform`. Computed on the fly from F_ if desired; not used in
        solving the ROM.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'C' is not in `modelform`.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if `has_inputs` is False.

    f_ : func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operator, defined by A_, F_, c_, and/or B_.
        Note the signiture is f_(t, x_); that is, f_ maps time x reduced state
        to reduced state. Calculated in fit() if `has_inputs` is False, and
        in predict() otherwise.

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
            * 'C': constant term c(p).
            * 'L': linear state matrix A(p).
            * 'Q': quadratic state matrix H(p).
            * 'I': input matrix B(p).
            H or F may be used interchangeably (code detects which by shape).

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
            self.c = operators['C']
            if self.c.shape != (n,):
                raise ValueError("basis Vr and FOM operator c not aligned")
            self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_linear:               # Linear state matrix.
            self.A = operators['L']
            if self.A.shape != (self.n,self.n):
                raise ValueError("basis Vr and FOM operator A not aligned")
            self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if self.has_quadratic:               # Quadratic state matrix.
            H_or_F = operators['Q']
            _n2 = self.n * (self.n + 1) // 2
            if H_or_F.shape == (self.n,self.n**2):          # It's H.
                self.H = H_or_F
                self.F = utils.H2F(self.H)
            elif H_or_F.shape == (self.n,_n2):               # It's F.
                self.F = H_or_F
                self.H = utils.F2H(self.F)
            else:
                raise ValueError("basis Vr and FOM operator H (F) not aligned")
            H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
            self.F_ = utils.H2F(H_)
        else:
            self.F, self.H, self.F_ = None, None, None

        if self.has_inputs:                     # Linear input matrix.
            self.B = operators['I']
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


class InterpolatedInferredContinuousROM(_ContinuousROM,
                                        _InferredMixin, _ParametricMixin):
    """Reduced order model for a system of high-dimensional ODEs, parametrized
    by a scalar µ, of the form

         dx/dt = f(t,x(t;µ),u(t);µ),        x(0;µ) = x0(µ).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    regularized ordinary least-squares problems, then interpolating those
    models with respect to the parameter µ.

    Parameters
    ----------
    modelform : str containing 'C', 'L', 'Q', and/or 'I'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'C' : Constant term c
        'L' : Linear term Ax(t).
        'Q' : Quadratic term H(x⊗x)(t).
        'I' : Input term Bu(t).
        For example, modelform=="LI" means f(t,x(t),u(t);µ) = Ax(t;µ) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic term H(x⊗x)(t).

    has_inputs : bool
        Whether or not there is an input term Bu(t).

    n : int
        The dimension of the original model.

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if `has_inputs` is False.

    s : int
        The number of models created during training and used in interpolation.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    dataconds_ : float
        Condition number of the data matrix for each least-squares problem.

    residuals_ : (s,) ndarray
        The squared Frobenius-norm residual of each least-squares problem (one
        per parameter) for computing the reduced-order model operators.

    As_ : list of s (r,r) ndarrays or None
        Learned ROM linear state matrices, or None if 'L' not in `modelform`.

    Fs_ : list of s (r,r(r+1)//2) ndarrays or None
        Learned ROM quadratic state matrices (compact), or None if 'Q' is not
        in `modelform`. Used internally instead of the larger H_.

    Hs_ : list of s (r,r**2) ndarrays or None
        Learned ROM quadratic state matrices (full size), or None if 'Q' is not
        in `modelform`. Computed on the fly from F_ if desired; not used in
        solving the ROM.

    cs_ : list of s (r,) ndarrays or None
        Learned ROM constant terms, or None if 'C' is not in `modelform`.

    Bs_ : list of s (r,m) ndarrays or None
        Learned ROM input matrices, or None if `has_inputs` is False.

    fs_ : list of func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operators, defined by As_, Fs_, cs_, and/or Bs_.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, µs, Xs, Xdots, Vr, Us=None, G=0):
        """Solve for the reduced model operators via regularized least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of s (n,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) ndarrays
            Column-wise velocity training data. The ith array Xdots[i]
            corresponds to the ith parameter, µs[i].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Us : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if
            has_inputs is True; must be None if has_inputs is False.

        G : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Gx||^2.
            If a nonzero number is provided, the regularization matrix is
            G * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_hasinputs(Us, 'Us')

        # Check that parameters are one-dimensional.
        if not np.isscalar(µs[0]):
            raise ValueError("only scalar parameter values are supported")

        # Check that the number of params matches the number of snapshot sets.
        s = len(µs)
        if len(Xs) != s:
            raise ValueError("num parameter samples != num state snapshot "
                             f"sets ({s} != {len(Xs)})")
        if len(Xdots) != s:
            raise ValueError("num parameter samples != num velocity snapshot "
                             f"sets ({s} != {len(Xdots)})")

        # Check and store dimensions.
        for X, Xdot in zip(Xs, Xdots):
            self._check_training_data_shapes(X, Xdot, Vr)
        n,k = Xs[0].shape       # Dimension of system, number of shapshots.
        r = Vr.shape[1]         # Number of basis vectors.
        self.n, self.r, self.m = n, r, None

        # Check that all arrays in each list of arrays are the same sizes.
        _tocheck = [(Xs, "X"), (Xdots, "Xdot")]
        if self.has_inputs:
            _tocheck += [(Us, "U")]
            self.m = Us[0].shape[0] if Us[0].ndim == 2 else 1
        else:
            Us = [None]*s
        for dataset, label in _tocheck:
            self._check_dataset_consistency(dataset, label)

        # TODO: figure out how to handle G (scalar, array, list(arrays)).

        # Train one model per parameter sample.
        self.Vr = Vr
        self.models_ = []
        for µ, X, Xdot, U in zip(µs, Xs, Xdots, Us):
            model = InferredContinuousROM(self.modelform)
            model.fit(X, Xdot, Vr, U, G)
            model.parameter = µ
            self.models_.append(model)

        # Construct interpolators.
        self.A_ = CubicSpline(µs, self.As_) if self.has_linear    else None
        self.F_ = CubicSpline(µs, self.Fs_) if self.has_quadratic else None
        self.H_ = CubicSpline(µs, self.Hs_) if self.has_quadratic else None
        self.c_ = CubicSpline(µs, self.cs_) if self.has_constant  else None
        self.B_ = CubicSpline(µs, self.Bs_) if self.has_inputs    else None

        return self

    def predict(self, µ, x0, t, u=None, **options):
        """Construct a ROM for the parameter µ by interolating the entries of
        the learned models, then simulate this interpolated ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        µ : float
            The paramter of interest for the prediction.

        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.

        t : (nt,) ndarray
            The time domain over which to integrate the reduced-order system.

        u : callable OR (m,nt) ndarray
            The input as a function of time (preferred) OR the input at the
            times `t`. If given as an array, u(t) is calculated by linearly
            interpolating known data points if needed for an adaptive solver.

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
        self._check_hasinputs(u, 'u')

        model = trained_model_from_operators(
                    modelclass=_ContinuousROM,
                    modelform=self.modelform,
                    Vr=self.Vr,
                    m=self.m,
                    A_=self.A_(µ) if self.A_ is not None else None,
                    F_=self.F_(µ) if self.F_ is not None else None,
                    c_=self.c_(µ) if self.c_ is not None else None,
                    B_=self.B_(µ) if self.B_ is not None else None,
                )
        out = model.predict(x0, t, u, **options)
        self.sol_ = model.sol_
        return out

    @property
    def As_(self):
        """The linear state matrices for each submodel."""
        return [m.A_ for m in self.models_] if self.has_linear else None

    @property
    def Hs_(self):
        """The full quadratic state matrices for each submodel."""
        return [m.H_ for m in self.models_] if self.has_quadratic else None

    @property
    def Fs_(self):
        """The compact quadratic state matrices for each submodel."""
        return [m.F_ for m in self.models_] if self.has_quadratic else None

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


class AffineInferredContinuousROM(_AffineContinuousROM,
                                  _InferredMixin, _AffineMixin):
    def fit(self, µs, affines, Xs, Xdots, Vr, Us=None, G=0):
        """Solve for the reduced model operators via regularized least squares.
        For terms with affine structure, solve for the constituent operators.

        Parameters
        ----------
        µs : list of s scalars or (p,) ndarrays
            Parameter values at which the snapshot data is collected.

        affines : dict(str -> list(functions))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'C': Constant term c(µ).
            * 'L': Linear state matrix A(µ).
            * 'Q': Quadratic state matrix H(µ).
            * 'I': linear Input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'C' -> [θ1, θ2, θ3].

        Xs : list of s (n,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) ndarrays
            Column-wise velocity training data. The ith array Xdots[i]
            corresponds to the ith parameter, µs[i].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Us : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if
            has_inputs is True; must be None if has_inputs is False.

        G : (d,d) ndarray or float
            Tikhonov regularization matrix. If nonzero, the least-squares
            problem problem takes the form min_{x} ||Ax - b||^2 + ||Gx||^2.
            If a nonzero number is provided, the regularization matrix is
            G * I (a scaled identity matrix). Here d is the dimension of the
            data matrix for the least-squares problem, e.g., d = r + m for a
            linear model with inputs.

        Returns
        -------
        self
        """
        # Check modelform and inputs.
        self._check_modelform()
        self._check_affines(affines, µs[0])
        self._check_hasinputs(Us, 'Us')

        # Check and store dimensions.
        for X, Xdot in zip(Xs, Xdots):
            self._check_training_data_shapes(X, Xdot, Vr)
        n,k = Xs[0].shape       # Dimension of system, number of shapshots.
        r = Vr.shape[1]         # Number of basis vectors.
        self.n, self.r, self.m = n, r, None

        # Check that all arrays in each list of arrays are the same sizes.
        _tocheck = [(Xs, "X"), (Xdots, "Xdot")]
        if self.has_inputs:
            _tocheck += [(Us, "U")]
            self.m = Us[0].shape[0] if Us[0].ndim == 2 else 1
        for dataset, label in _tocheck:
            self._check_dataset_consistency(dataset, label)

        # Project states and velocities to the reduced subspace.
        Xs_ = [Vr.T @ X for X in Xs]
        Xdots_ = [Vr.T @ Xdot for Xdot in Xdots]
        self.Vr = Vr

        # Construct the "Data matrix" D.
        D_blockrows = []
        for i in range(len(µs)):
            row = []
            µ = µs[i]
            k = Xs[i].shape[1]

            if self.has_constant:
                ones = np.ones(k).reshape((k,1))
                if 'C' in affines:
                    for j in range(len(affines['C'])):
                        row.append(affines['C'][j](µ) * ones)
                else:
                    row.append(ones)

            if self.has_linear:
                if 'L' in affines:
                    for j in range(len(affines['L'])):
                        row.append(affines['L'][j](µ) * Xs_[i].T)
                else:
                    row.append(Xs_[i].T)

            if self.has_quadratic:
                X2i_ = kron2(Xs_[i])
                if 'Q' in affines:
                    for j in range(len(affines['Q'])):
                        row.append(affines['Q'][j](µ) * X2i_.T)
                else:
                    row.append(X2i_.T)

            if self.has_inputs:
                Ui = Us[i]
                if self.m == 1:
                    Ui = Ui.reshape((1,k))
                if 'I' in affines:
                    for j in range(len(affines['I'])):
                        row.append(affines['I'][j](µ) * Ui.T)
                else:
                    row.append(Ui.T)

            D_blockrows.append(np.hstack(row))

        D = np.vstack(D_blockrows)
        self.datacond_ = np.linalg.cond(D)      # Condition number of data.
        R = np.hstack(Xdots_).T

        # Solve for the reduced-order model operators via least squares.
        OT, res = utils.lstsq_reg(D, R, G)[0:2]
        self.residual_ = np.sum(res)

        # Extract the reduced operators from OT.
        i = 0
        if self.has_constant:
            if 'C' in affines:
                self.cs_ = []
                for j in range(len(affines['C'])):
                    self.cs_.append(OT[i:i+1][0])   # c_ is one-dimensional.
                    i += 1
                self.c_ = AffineOperator(affines['C'], self.cs_)
            else:
                self.c_ = OT[i:i+1][0]              # c_ is one-dimensional.
                i += 1
                self.cs_ = [self.c_]
        else:
            self.c_, self.cs_ = None, None

        if self.has_linear:
            if 'L' in affines:
                self.As_ = []
                for j in range(len(affines['L'])):
                    self.As_.append(OT[i:i+self.r].T)
                    i += self.r
                self.A_ = AffineOperator(affines['L'], self.As_)
            else:
                self.A_ = OT[i:i+self.r].T
                i += self.r
                self.As_ = [self.A_]
        else:
            self.A_, self.As_ = None, None

        if self.has_quadratic:
            _r2 = self.r * (self.r + 1) // 2
            if 'Q' in affines:
                self.Fs_ = []
                for j in range(len(affines['Q'])):
                    self.Fs_.append(OT[i:i+_r2].T)
                    i += _r2
                self.F_ = AffineOperator(affines['Q'], self.Fs_)
            else:
                self.F_ = OT[i:i+_r2].T
                i += _r2
                self.Fs_ = [self.F_]
        else:
            self.F_, self.Fs_ = None, None

        if self.has_inputs:
            if 'I' in affines:
                self.Bs_ = []
                for j in range(len(affines['B'])):
                    self.Bs_.append(OT[i:i+self.m].T)
                    i += self.m
                self.B_ = AffineOperator(affines['B'], self.Bs_)
            else:
                self.B_ = OT[i:i+self.m].T
                i += self.m
                self.Bs_ = [self.B_]
        else:
            self.B_, self.Bs_ = None, None

        return self


class AffineIntrusiveContinuousROM(_AffineContinuousROM,
                                   _IntrusiveMixin, _AffineMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t); µ),         x(0;µ) = x0(µ).

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

    Parameters
    ----------
    modelform : str containing 'C', 'L', 'Q', and/or 'I'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        * 'C' : Constant term c(µ).
        * 'L' : Linear term A(µ)x(t).
        * 'Q' : Quadratic term H(µ)(x⊗x)(t).
        * 'I' : Input term B(µ)u(t).
        For example, modelform=="CL" means f(t,x(t);µ) = c(µ) + A(µ)x(t;µ).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic term H(x⊗x)(t).

    has_inputs : bool
        Whether or not there is an input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the projected reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if `has_inputs` is False.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    c : func(µ) -> (n,) ndarray, (n,) ndarray, or None
        FOM constant term, or None if 'C' is not in `modelform`.

    A : func(µ) -> (n,n) ndarray, (n,n) ndarray, or None
        FOM linear state matrix, or None if 'L' is not in `modelform`.

    F : func(µ) -> (n,n(n+1)//2) ndarray, (n,n(n+1)//2) ndarray, or None
        FOM quadratic state matrix (compact), or None if 'Q' is not
        in `modelform`.

    H : func(µ) -> (n,n**2) ndarray, (n,n**2) ndarray, or None
        FOM quadratic state matrix (full size), or None if 'Q' is not
        in `modelform`.

    B : func(µ) -> (n,m) ndarray, (n,m) ndarray, or None
        Learned ROM input matrix, or None if `has_inputs` is False.

    c_ : func(µ) -> (r,) ndarray, (r,) ndarray, or None
        Learned ROM constant term, or None if 'C' is not in `modelform`.

    A_ : func(µ) -> (r,r) ndarray, (r,r) ndarray, or None
        Learned ROM linear state matrix, or None if 'L' is not in `modelform`.

    F_ : func(µ) -> (r,r(r+1)//2) ndarray, (r,r(r+1)//2) ndarray, or None
        Learned ROM quadratic state matrix (compact), or None if 'Q' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : func(µ) -> (r,r**2) ndarray, (r,r**2) ndarray, or None
        Learned ROM quadratic state matrix (full size), or None if 'Q' is not
        in `modelform`. Computed on the fly from F_ if desired; not used in
        solving the ROM.

    B_ : func(µ) -> (r,m) ndarray, (r,m) ndarray, or None
        Learned ROM input matrix, or None if `has_inputs` is False.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, affines, operators, Vr):
        """Solve for the reduced model operators via intrusive projection.

        Parameters
        ----------
        affines : dict(str -> list(functions))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'C': Constant term c(µ).
            * 'L': Linear state matrix A(µ).
            * 'Q': Quadratic state matrix H(µ).
            * 'I': linear Input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'C' -> [θ1, θ2, θ3].

        operators: dict(str -> ndarray or list(ndarrays))
            The operators that define the full-order model f(t,x;µ).
            Keys must match the modelform:
            * 'C': constant term c(µ).
            * 'L': linear state matrix A(µ).
            * 'Q': quadratic state matrix H(µ).
            * 'I': input matrix B(µ).
            Terms with affine structure should be given as a list of the
            constituent matrices. For example, if the linear state matrix has
            the form A(µ) = θ1(µ)A1 + θ2(µ)A2, then "L" -> [A1, A2].

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
            if 'C' in affines:
                self.c = AffineOperator(affines['C'], operators['C'])
                if self.c.shape != (n,):
                    raise ValueError("basis Vr and FOM operator c not aligned")
                self.c_ = AffineOperator(affines['C'],
                                          [self.Vr.T @ c
                                           for c in self.c.matrices])
            else:
                self.c = operators['C']
                if self.c.shape != (n,):
                    raise ValueError("basis Vr and FOM operator c not aligned")
                self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_linear:               # Linear state matrix.
            if 'L' in affines:
                self.A = AffineOperator(affines['L'], operators['L'])
                if self.A.shape != (self.n,self.n):
                    raise ValueError("basis Vr and FOM operator A not aligned")
                self.A_ = AffineOperator(affines['L'],
                                          [self.Vr.T @ A @ self.Vr
                                           for A in self.A.matrices])
            else:
                self.A = operators['L']
                if self.A.shape != (self.n,self.n):
                    raise ValueError("basis Vr and FOM operator A not aligned")
                self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if self.has_quadratic:               # Quadratic state matrix.
            _n2 = self.n * (self.n + 1) // 2
            if 'Q' in affines:
                H_or_F = AffineOperator(affines['Q'], operators['Q'])
                if H_or_F.shape == (self.n,self.n**2):      # It's H.
                    self.H = H_or_F
                    self.F = AffineOperator(affines['Q'],
                                             [utils.H2F(H)
                                              for H in H_or_F.matrices])
                elif H_or_F.shape == (self.n,_n2):           # It's F.
                    self.F = H_or_F
                    self.H = AffineOperator(affines['Q'],
                                             [utils.F2H(F)
                                              for F in H_or_F.matrices])
                else:
                    raise ValueError("basis VR and FOM operator H not aligned")
                Vr2 = np.kron(self.Vr, self.Vr)
                self.H_ = AffineOperator(affines['Q'],
                                          [self.Vr.T @ H @ Vr2
                                           for H in self.H.matrices])
                self.F_ = AffineOperator(affines['Q'],
                                          [utils.H2F(H_)
                                           for H_ in self.H_.matrices])
            else:
                H_or_F = operators['Q']
                if H_or_F.shape == (self.n,self.n**2):      # It's H.
                    self.H = H_or_F
                    self.F = utils.H2F(self.H)
                elif H_or_F.shape == (self.n,_n2):           # It's F.
                    self.F = H_or_F
                    self.H = utils.F2H(self.F)
                else:
                    raise ValueError("basis Vr and FOM operator H not aligned")
                self.H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
                self.F_ = utils.H2F(self.H_)
        else:
            self.F, self.H, self.F_ = None, None, None

        if self.has_inputs:                 # Linear input matrix.
            if 'I' in affines:
                self.B = AffineOperator(affines['I'], operators['I'])
                if self.B.shape[0] != self.n:
                    raise ValueError("basis Vr and FOM operator B not aligned")
                if len(self.B.shape) == 2:
                    self.m = self.B.shape[1]
                else:                                   # One-dimensional input
                    self.B = AffineOperator(affines['I'],
                                             [B.reshape((-1,1))
                                              for B in self.B.matrices])
                    self.m = 1
                self.B_ = AffineOperator(affines['I'],
                                          [self.Vr.T @ B
                                           for B in self.B.matrices])
            else:
                self.B = operators['I']
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


# Discrete models (i.e., solving x_{k+1} = f(x_{k},u_{k})) --------------------



__all__ = [
            "InferredContinuousROM",
            "IntrusiveContinuousROM",
            "AffineInferredContinuousROM",
            "AffineIntrusiveContinuousROM",
            "InterpolatedInferredContinuousROM",
          ]
