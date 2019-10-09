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
class _AffineOperator:
    """Class for representing a linear operator with affine structure, i.e.,

        A(p) = sum_{i=1}^{nterms} f_{i}(p) * A_{i}.

    The matrix A(p) is constructed by calling the object once the coefficient
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

    def validate_coeffs(self, p):
        """Check that each coefficient function 1) is a callable function,
        2) takes in the right sized inputs, and 3) returns scalar values.

        Parameters
        ----------
        p : float or (dp,) ndarray
            A test input for the coefficient functions.
        """
        for f in self.coefficient_functions:
            if not callable(f):
                raise ValueError("coefficients of affine operator must be "
                                 "callable functions")
            elif not np.isscalar(f(p)):
                raise ValueError("coefficient functions of affine operator "
                                 "must return a scalar")

    def __call__(self, p):
        if not self._ready:
            raise RuntimeError("constituent matrices not initialized!")
        return np.sum([fi(p)*Ai for fi,Ai in zip(self.coefficient_functions,
                                                 self.matrices)], axis=0)


def _trained_model_from_operators(modelclass, modelform, has_inputs, Vr,
                                  m=None, A_=None, F_=None, c_=None, B_=None):
    """Construct a prediction-capable Model object from the operators of
    the reduced model.

    Returns
    -------
    model : modelclass object
        A new model, ready for predict() calls.
    """
    # Check that the modelclass is valid.
    if not issubclass(modelclass, _BaseModel):
        raise TypeError("modelclass must be derived from _BaseModel")

    # Construct the new model object.
    model = modelclass(modelform, has_inputs)
    model._check_modelform(trained=False)

    # Insert the attributes.
    model.Vr = Vr
    model.n, model.r, model.m = Vr.shape + (m,)
    model.A_, model.F_, model.c_, model.B_ = A_, F_, c_, B_

    # Check that the attributes match the modelform.
    model._check_modelform(trained=True)

    # Construct the ROM operator f_() if there are no system inputs.
    if not has_inputs and issubclass(modelclass, _ContinuousModel):
        model._construct_f_()

    return model


# Base classes ================================================================
class _BaseModel:
    """Base class for all rom_operator_inference reduced model classes."""
    _MODEL_KEYS = "CLQ"                 # Constant, Linear, Quadratic.

    def __init__(self, modelform, has_inputs=False):
        self.modelform = modelform
        if 'I' in self.modelform:
            self.has_inputs = True
            self.modelform = self.modelform.replace('I', '')
        self.has_inputs = has_inputs

    @property
    def modelform(self):
        return self._form

    @modelform.setter
    def modelform(self, form):
        self._form = ''.join(sorted(form.upper()))

    def _check_modelform(self, trained=False):
        """Ensure that self.modelform is valid."""
        for key in self.modelform:
            if key not in self._MODEL_KEYS:
                raise ValueError(f"invalid modelform key '{key}'; options "
                                 "are " + ', '.join(self._MODEL_KEYS))

        if trained:
            # Make sure that the required attributes exist and aren't None,
            # and that nonrequired attributes exist but are None.
            for key, s in zip("CLQ", ["c_", "A_", "F_"]):
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
            if not hasattr(self, 'B_'):
                raise AttributeError(f"attribute 'B_' missing;"
                                     " call fit() to train model")
            if self.has_inputs and self.B_ is None:
                raise AttributeError(f"attribute 'B_' is None;"
                                     " call fit() to train model")
            elif not self.has_inputs and self.B_ is not None:
                raise AttributeError(f"attribute 'B_' should be None;"
                                     " call fit() to train model")

    def _check_hasinputs(self, u, argname):
        """Check that self.has_inputs agrees with input arguments."""
        if self.has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since has_inputs=True")

        if not self.has_inputs and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since has_inputs=False")


class _ContinuousModel(_BaseModel):
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
            if self.modelform == "C":
                f_ = lambda t,x_: self.c_ + self.B_@u_(t)
            elif self.modelform == "L":
                f_ = lambda t,x_: self.A_@x_ + self.B_@u_(t)
            elif self.modelform == "CL":
                f_ = lambda t,x_: self.c_ + self.A_@x_ + self.B_@u_(t)
            elif self.modelform == "Q":
                f_ = lambda t,x_: self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "CQ":
                f_ = lambda t,x_: self.c_ + self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "LQ":
                f_ = lambda t,x_: self.A_@x_ + self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "CLQ":
                f_ = lambda t,x_: self.c_ + self.A_@x_ + self.F_@kron2(x_) + self.B_@u_(t)
        else:
            raise RuntimeError("improper use of _construct_f_()!")
        self.f_ = f_

    def __str__(self):
        """String representation: the structure of the model."""
        self._check_modelform()
        out = []
        if 'C' in self.modelform: out.append("c")
        if 'L' in self.modelform: out.append("Ax(t)")
        if 'Q' in self.modelform: out.append("H(x ⊗ x)(t)")
        if self.has_inputs: out.append("Bu(t)")

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


class _DiscreteModel(_BaseModel):           # pragma: no cover
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
        if 'C' in self.modelform: out.append("c")
        if 'L' in self.modelform: out.append("Ax_{k}")
        if 'Q' in self.modelform: out.append("H(x_{k} ⊗ x_{k})")
        if self.has_inputs: out.append("Bu_{k}")

        return "Reduced-order model structure: x_{k+1} = " + " + ".join(out)


class _AffineContinuousModel(_ContinuousModel):
    """Base class for models with affinely parametric operators."""
    def predict(self, p, x0, t, u=None, **options):
        """Construct a ROM for the parameter p by exploiting the affine
        structure of the ROM operators, then simulate the resulting ROM with
        scipy.integrate.solve_ivp().

        TODO: right now this assumes affine dependence in A_ and c_.

        Parameters
        ----------
        p : float
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

        # Make sure the parameter p has the correct dimension.

        # Use the affine structure of the operators to construct a new model.

        model = _trained_model_from_operators(
            modelclass=_ContinuousModel,
            modelform=self.modelform,
            has_inputs=self.has_inputs,
            Vr=self.Vr,
            m=self.m,
            A_=self.A_(p) if isinstance(self.A_, _AffineOperator) else self.A_,
            F_=self.F_(p) if isinstance(self.F_, _AffineOperator) else self.F_,
            c_=self.c_(p) if isinstance(self.c_, _AffineOperator) else self.c_,
            B_=self.B_(p) if isinstance(self.B_, _AffineOperator) else self.B_,
                )
        return model.predict(x0, t, u, **options)


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
    pass


class _NonparametricMixin:
    """Mixin class for non-parametric reduced model classes."""
    @property
    def H_(self):
        """Matricized quadratic tensor; operates on full Kronecker product."""
        return None if self.F_ is None else utils.F2H(self.F_)


class _ParametricMixin:
    """Mixin class for parametric reduced model classes."""
    pass


# Useable classes =============================================================

# Continuous models (i.e., solving dx/dt = f(t,x,u)) --------------------------
class InferredContinuousModel(_ContinuousModel,
                              _InferredMixin, _NonparametricMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t,x(t),u(t)),           x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving a
    regularized ordinary least-squares problem.

    Parameters
    ----------
    modelform : str {'L', 'CL', 'Q', 'CQ', 'LQ', 'CLQ'}
        The structure of the desired reduced-order model. Options:
        'L'   : Linear model, f(x) = Ax.
        'CL'  : Linear model with constant, f(x) = Ax + c.
        'Q'   : Quadratic model, f(x) = H(x⊗x).
        'CQ'  : Quadratic model with constant, f(x) = H(x⊗x) + c.
        'LQ'  : Linear-Quadratic model, f(x) = Ax + H(x⊗x).
        'CLQ' : Linear-Quadratic model with constant, f(x) = Ax + H(x⊗x) + c.

    has_inputs : bool, optional, default: False.
        If True, assume the system has an additive input term u(t).

    Attributes
    ----------
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
        if 'L' in self.modelform:
            D_blocks.append(X_.T)

        if 'Q' in self.modelform:
            X2_ = kron2(X_)
            D_blocks.append(X2_.T)
            s = X2_.shape[0]    # = r(r+1)//2, size of the compact Kronecker.

        if 'C' in self.modelform:
            D_blocks.append(np.ones(k).reshape((k,1)))

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
        if 'L' in self.modelform:
            self.A_ = OT[i:i+self.r].T
            i += self.r
        else:
            self.A_ = None

        if 'Q' in self.modelform:
            self.F_ = OT[i:i+s].T
            i += s
        else:
            self.F_ = None

        if 'C' in self.modelform:
            self.c_ = OT[i:i+1][0]       # Note that c_ is one-dimensional.
            i += 1
        else:
            self.c_ = None

        if self.has_inputs:
            self.B_ = OT[i:i+self.m].T
            i += self.m
        else:
            self.B_ = None

        # Construct the complete ROM operator IF there are no control inputs.
        if not self.has_inputs:
            self._construct_f_()

        return self


class IntrusiveContinuousModel(_ContinuousModel,
                               _IntrusiveMixin, _NonparametricMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),         x(0) = x0.

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

    Parameters
    ----------
    modelform : str {'L', 'CL', 'Q', 'CQ', 'LQ', 'CLQ'}
        The structure of the full-order and reduced-order models. Options:
        'L'   : Linear model, f(x) = Ax.
        'CL'  : Linear model with constant, f(x) = Ax + c.
        'Q'   : Quadratic model, f(x) = H(x⊗x).
        'CQ'  : Quadratic model with constant, f(x) = H(x⊗x) + c.
        'LQ'  : Linear-Quadratic model, f(x) = Ax + H(x⊗x).
        'CLQ' : Linear-Quadratic model with constant, f(x) = Ax + H(x⊗x) + c.

    has_inputs : bool, optional, default: False.
        If True, assume the system has an additive input term u(t).

    Attributes
    ----------
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
        """Solve for the reduced model operators via regularized least squares.

        Parameters
        ----------
        operators: list(ndarrays)
            The operators that define the full-order model f(t,x).
            The list must be as follows, depending on the value of modelform:
            'L'   : [A],        or  [A, B]          if has_inputs is True.
            'CL'  : [A, c],     or  [A, c, B]       if has_inputs is True.
            'Q'   : [H],        or  [A, H, B]       if has_inputs is True.
            'CQ'  : [H, c],     or  [H, c, B]       if has_inputs is True.
            'LQ'  : [A, H],     or  [A, H, B]       if has_inputs is True.
            'CLQ' : [A, H, c],  or  [A, H, c, B]    if has_inputs is True.
            H or F may be used interchangeably (code detects which by shape).

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Returns
        -------
        self
        """
        # Verify modelform.
        self._check_modelform()
        n_expect = len(self.modelform) + self.has_inputs
        n_actual = len(operators)
        if n_expect != n_actual:
            _noun = "operator" + ('' if n_expect == 1 else 's')
            raise ValueError(f"expected {n_expect} {_noun}, got {n_actual}")

        # Store dimensions.
        n,r = Vr.shape          # Dimension of system, number of basis vectors.
        self.Vr = Vr
        self.n, self.r = n, r

        # Project FOM operators.
        operator = iter(operators)
        if 'L' in self.modelform:               # Linear state matrix.
            self.A = next(operator)
            if self.A.shape != (self.n,self.n):
                raise ValueError("basis Vr and FOM operator A not aligned")
            self.A_ = self.Vr.T @ self.A @ self.Vr
        else:
            self.A, self.A_ = None, None

        if 'Q' in self.modelform:               # Quadratic state matrix.
            H_or_F = next(operator)
            s = self.n * (self.n + 1) // 2
            if H_or_F.shape == (self.n,self.n**2):          # It's H.
                self.H = H_or_F
                self.F = utils.H2F(self.H)
            elif H_or_F.shape == (self.n,s):                # It's F.
                self.F = H_or_F
                self.H = utils.F2H(self.F)
            else:
                raise ValueError("basis Vr and FOM operator H (F) not aligned")
            H_ = self.Vr.T @ self.H @ np.kron(self.Vr, self.Vr)
            self.F_ = utils.H2F(H_)
        else:
            self.F, self.H, self.F_ = None, None, None

        if 'C' in self.modelform:               # Constant term.
            self.c = next(operator)
            if self.c.shape != (n,):
                raise ValueError("basis Vr and FOM operator c not aligned")
            self.c_ = self.Vr.T @ self.c
        else:
            self.c, self.c_ = None, None

        if self.has_inputs:                     # Linear input matrix.
            self.B = next(operator)
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

        # Construct the complete ROM operator IF there are no control inputs.
        if not self.has_inputs:
            self._construct_f_()

        return self


class InterpolatedInferredContinuousModel(_ContinuousModel,
                                          _InferredMixin, _ParametricMixin):
    """Reduced order model for a system of high-dimensional ODEs, parametrized
    by a scalar p, of the form

         dx/dt = f(t,x(t;p),u(t);p),        x(0;p) = x0(p).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    regularized ordinary least-squares problems, then interpolating those
    models with respect to the parameter.

    Parameters
    ----------
    modelform : str {'L', 'CL', 'Q', 'CQ', 'LQ', 'CLQ'}
        The structure of the desired reduced-order model. Options:
        'L'   : Linear model, f(x) = Ax.
        'CL'  : Linear model with constant, f(x) = Ax + c.
        'Q'   : Quadratic model, f(x) = H(x⊗x).
        'CQ'  : Quadratic model with constant, f(x) = H(x⊗x) + c.
        'LQ'  : Linear-Quadratic model, f(x) = Ax + H(x⊗x).
        'CLQ' : Linear-Quadratic model with constant, f(x) = Ax + H(x⊗x) + c.

    has_inputs : bool, optional, default: False.
        If True, assume the system has an additive input term u(t).

    Attributes
    ----------
    n : int
        The dimension of the original model.

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if `has_inputs` is False.

    num_models : int
        The number of models created during training and used in interpolation.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    dataconds_ : float
        Condition number of the data matrix for each least-squares problem.

    residuals_ : (num_models,) ndarray
        The squared Frobenius-norm residual of each least-squares problem (one
        per parameter) for computing the reduced-order model operators.

    As_ : list of num_models (r,r) ndarrays or None
        Learned ROM linear state matrices, or None if 'L' not in `modelform`.

    Fs_ : list of num_models (r,r(r+1)//2) ndarrays or None
        Learned ROM quadratic state matrices (compact), or None if 'Q' is not
        in `modelform`. Used internally instead of the larger H_.

    Hs_ : list of num_models (r,r**2) ndarrays or None
        Learned ROM quadratic state matrices (full size), or None if 'Q' is not
        in `modelform`. Computed on the fly from F_ if desired; not used in
        solving the ROM.

    cs_ : list of num_models (r,) ndarrays or None
        Learned ROM constant terms, or None if 'C' is not in `modelform`.

    Bs_ : list of num_models (r,m) ndarrays or None
        Learned ROM input matrices, or None if `has_inputs` is False.

    fs_ : list of func(float, (r,) ndarray) -> (r,) ndarray
        The complete ROM operators, defined by As_, Fs_, cs_, and/or Bs_.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, ps, Xs, Xdots, Vr, Us=None, G=0):
        """Solve for the reduced model operators via regularized least squares,
        contructing one ROM per parameter value.

        Parameters
        ----------
        ps : (num_models,) ndarray
            Parameter values at which the snapshot data is collected.

        Xs : list of num_models (n,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, ps[i].

        Xdots : list of num_models (n,k) ndarrays
            Column-wise velocity training data. The ith array Xdots[i]
            corresponds to the ith parameter, ps[i].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Us : list of num_models (m,k) or (k,) ndarrays or None
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
        if not np.isscalar(ps[0]):
            raise ValueError("only scalar parameter values are supported")

        # Check that the number of params matches the number of snapshot sets.
        num_models = len(ps)
        if len(Xs) != num_models:
            raise ValueError("num parameter samples != num state snapshot "
                             f"sets ({num_models} != {len(Xs)})")
        if len(Xdots) != num_models:
            raise ValueError("num parameter samples != num velocity snapshot "
                             f"sets ({num_models} != {len(Xdots)})")

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
            Us = [None]*num_models
        for dataset, label in _tocheck:
            self._check_dataset_consistency(dataset, label)

        # TODO: figure out how to handle G (scalar, array, list(arrays)).

        # Train one model per parameter sample.
        self.Vr = Vr
        self.models_ = []
        for p, X, Xdot, U in zip(ps, Xs, Xdots, Us):
            model = InferredContinuousModel(self.modelform, self.has_inputs)
            model.fit(X, Xdot, Vr, U, G)
            model.p = p
            self.models_.append(model)

        # Construct interpolators.
        self.A_ = CubicSpline(ps, self.As_) if 'L' in self.modelform else None
        self.F_ = CubicSpline(ps, self.Fs_) if 'Q' in self.modelform else None
        self.H_ = CubicSpline(ps, self.Hs_) if 'Q' in self.modelform else None
        self.c_ = CubicSpline(ps, self.cs_) if 'C' in self.modelform else None
        self.B_ = CubicSpline(ps, self.Bs_) if self.has_inputs else None

        return self

    def predict(self, p, x0, t, u=None, **options):
        """Construct a ROM for the parameter p by interolating the entries of
        the learned models, then simulate this interpolated ROM with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        p : float
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

        model = _trained_model_from_operators(
                    modelclass=_ContinuousModel,
                    modelform=self.modelform,
                    has_inputs=self.has_inputs,
                    Vr=self.Vr,
                    m=self.m,
                    A_=self.A_(p) if self.A_ is not None else None,
                    F_=self.F_(p) if self.F_ is not None else None,
                    c_=self.c_(p) if self.c_ is not None else None,
                    B_=self.B_(p) if self.B_ is not None else None,
                )
        return model.predict(x0, t, u, **options)

    @property
    def As_(self):
        """The linear state matrices for each submodel."""
        return [m.A_ for m in self.models_] if 'L' in self.modelform else None

    @property
    def Hs_(self):
        """The full quadratic state matrices for each submodel."""
        return [m.H_ for m in self.models_] if 'Q' in self.modelform else None

    @property
    def Fs_(self):
        """The compact quadratic state matrices for each submodel."""
        return [m.F_ for m in self.models_] if 'Q' in self.modelform else None

    @property
    def cs_(self):
        """The constant terms for each submodel."""
        return [m.c_ for m in self.models_] if 'C' in self.modelform else None

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


class AffineInferredContinuousModel(_AffineContinuousModel,
                                    _InferredMixin, _ParametricMixin):
    def fit(self, ps, affines, Xs, Xdots, Vr, Us=None, G=0):
        """Solve for the reduced model operators via regularized least squares.
        For terms with affine structure, solve for the constituent operators.

        Parameters
        ----------
        ps : list of scalars or (p,) ndarrays
            Parameter values at which the snapshot data is collected.

        affines : dict(str -> list(functions))
            Functions that define the affine structure of the operators. Keys:
            'c': constant term
            'L': linear state matrix
            'Q': quadratic state matrix
            'I': linear input matrix

        Xs : list of num_models (n,k) ndarrays
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, ps[i].

        Xdots : list of num_models (n,k) ndarrays
            Column-wise velocity training data. The ith array Xdots[i]
            corresponds to the ith parameter, ps[i].

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        Us : list of num_models (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if
            has_inputs is True; must be None if has_inputs is False.

        affines : dict(operator key -> list of M scalar-valued functions)
            The functions that define the affine structure of the full-order
            operator, specified by the key ("L" for A, "Q" for H, "c" for c,
            and "I" for B). The structure carries over to the corresponding
            reduced-order operator.

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
        if 'L' not in self.modelform:
            raise NotImplementedError(
                "this class is currently only implemented for affine "
                "dependencies in the linear state matrix")

        # Check modelform and inputs.
        self._check_modelform()
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

        # Check affines argument. # TODO: put this in parent class.
        for a in affines.values():
            ao = _AffineOperator(a)
            ao.validate_coeffs(ps[0])
            del ao

        # Project states and velocities to the reduced subspace.
        Xs_ = [Vr.T @ X for X in Xs]
        Xdots_ = [Vr.T @ Xdot for Xdot in Xdots]
        self.Vr = Vr

        # Construct the "Data matrix" D.
        D_blockrows = []
        for i in range(len(ps)):
            row = []
            p = ps[i]
            k = Xs[i].shape[1]

            if 'L' in self.modelform:
                if 'L' in affines:
                    for j in range(len(affines['L'])):
                        row.append(affines['L'][j](p) * Xs_[i].T)
                else:
                    row.append(Xs_[i].T)

            if 'Q' in self.modelform:
                X2i_ = kron2(Xs_[i])
                if 'Q' in affines:
                    for j in range(len(affines['Q'])):
                        row.append(affines['Q'][j](p) * X2i_.T)
                else:
                    row.append(X2i_.T)

            if 'C' in self.modelform:
                ones = np.ones(k).reshape((k,1))
                if 'C' in affines:
                    for j in range(len(affines['c'])):
                        row.append(affines['c'][j](p) * ones)
                else:
                    row.append(ones)

            if self.has_inputs:
                Ui = Us[i]
                if self.m == 1:
                    Ui = Ui.reshape((1,k))
                if 'I' in affines:
                    for j in range(len(affines['I'])):
                        row.append(affines['I'][j](p) * Ui.T)
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
        if 'L' in self.modelform:
            if 'L' in affines:
                self.As_ = []
                for j in range(len(affines['L'])):
                    self.As_.append(OT[i:i+self.r].T)
                    i += self.r
                self.A_ = _AffineOperator(affines['L'], self.As_)
            else:
                self.A_ = OT[i:i+self.r].T
                i += self.r
                self.As_ = [self.A_]
        else:
            self.A_, self.As_ = None, None

        if 'Q' in self.modelform:
            s = self.r * (self.r + 1) // 2
            if 'Q' in affines:
                self.Fs_ = []
                for j in range(len(affines['Q'])):
                    self.Fs_.append(OT[i:i+s].T)
                    i += s
                self.F_ = _AffineOperator(affines['Q'], self.Fs_)
            else:
                self.F_ = OT[i:i+s].T
                i += s
                self.Fs_ = [self.F_]
        else:
            self.F_, self.Fs_ = None, None

        if 'C' in self.modelform:
            if 'C' in affines:
                self.cs_ = []
                for j in range(len(affines['c'])):
                    self.cs_.append(OT[i:i+1][0])   # c_ is one-dimensional.
                    i += 1
                self.c_ = _AffineOperator(affines['c'], self.cs_)
            else:
                self.c_ = OT[i:i+1][0]              # c_ is one-dimensional.
                i += 1
                self.cs_ = [self.c_]
        else:
            self.c_, self.cs_ = None, None

        if self.has_inputs:
            if 'I' in affines:
                self.Bs_ = []
                for j in range(len(affines['B'])):
                    self.Bs_.append(OT[i:i+self.m].T)
                    i += self.m
                self.B_ = _AffineOperator(affines['B'], self.Bs_)
            else:
                self.B_ = OT[i:i+self.m].T
                i += self.m
                self.Bs_ = [self.B_]
        else:
            self.B_, self.Bs_ = None, None

        return self


class AffineIntrusiveContinuousModel(_AffineContinuousModel,
                                     _IntrusiveMixin, _ParametricMixin):
    def __init__(self, *args, **kwargs):                    # pragma: no cover
        raise NotImplementedError


# Discrete models (i.e., solving x_{k+1} = f(x_{k},u_{k})) --------------------



__all__ = [
            "InferredContinuousModel",
            "IntrusiveContinuousModel",
            # "AffineInferredContinuousModel",
            # "AffineIntrusiveContinuousModel",
            "InterpolatedInferredContinuousModel",
          ]
