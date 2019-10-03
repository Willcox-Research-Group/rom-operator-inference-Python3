# _core.py
"""Class for model order reduction of ODEs via operator inference."""
# TODO: jacobians for each model form

import warnings
import numpy as np
from scipy import linalg as la
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, IntegrationWarning

from . import utils
kron2 = utils.kron_compact


# Base classes ================================================================
class _BaseModel:
    """Base class for all rom_operator_inference reduced model classes."""
    _VALID_MODEL_FORMS = {"L", "Lc", "Q", "Qc", "LQ", "LQc"}

    def __init__(self, modelform, has_inputs=False):
        self.modelform = modelform
        self.has_inputs = has_inputs

    def _check_modelform(self): # TODO: trained=False
        """Ensure that self.modelform is valid."""
        if self.modelform not in self._VALID_MODEL_FORMS:
            raise ValueError(f"invalid modelform '{self.modelform}'; "
                             f"options are {self._VALID_MODEL_FORMS}")
        # if trained:
        #     # Make sure that the required attributes exist and aren't None.
        #     for key, s, attr in zip("LQc", ["A_", "F_", "c_"],
        #                                [self.A_, self.F_, self.c_]):
        #         if key in self.modelform and attr is None:
        #             raise RuntimeError(f"{s} is None but '{key}' in modelform")
        #     if self.has_inputs and self.B_ is None:
        #         raise RuntimeError("B_ in None but has_inputs is True")

    def _check_hasinputs(self, u, argname):
        """Check that self.has_inputs agrees with input arguments."""
        if self.has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required"
                             " since has_inputs=True")

        if not self.has_inputs and u is not None:
            raise ValueError(f"argument '{argname}' invalid"
                             " since has_inputs=False")


class _NonparametricMixin:
    """Mixin class for non-parametric reduced model classes."""
    @property
    def H_(self):
        """Matricized quadratic tensor; operates on full Kronecker product."""
        return None if self.F_ is None else utils.F2H(self.F_)


# class _ParametricMixin? What would this need?


class _BaseContinuousModel(_BaseModel):
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
            if self.modelform == "L":
                f_ = lambda t,x_: self.A_@x_
            elif self.modelform == "Lc":
                f_ = lambda t,x_: self.A_@x_ + self.c_
            elif self.modelform == "Q":
                f_ = lambda t,x_: self.F_@kron2(x_)
            elif self.modelform == "Qc":
                f_ = lambda t,x_: self.F_@kron2(x_) + self.c_
            elif self.modelform == "LQ":
                f_ = lambda t,x_: self.A_@x_ + self.F_@kron2(x_)
            elif self.modelform == "LQc":
                f_ = lambda t,x_: self.A_@x_ + self.F_@kron2(x_) + self.c_
        elif self.has_inputs and u is not None:
            u_ = u
            if self.modelform == "L":
                f_ = lambda t,x_: self.A_@x_ + self.B_@u_(t)
            elif self.modelform == "Lc":
                f_ = lambda t,x_: self.A_@x_ + self.c_ + self.B_@u_(t)
            elif self.modelform == "Q":
                f_ = lambda t,x_: self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "Qc":
                f_ = lambda t,x_: self.F_@kron2(x_) + self.c_ + self.B_@u_(t)
            elif self.modelform == "LQ":
                f_ = lambda t,x_: self.A_@x_ + self.F_@kron2(x_) + self.B_@u_(t)
            elif self.modelform == "LQc":
                f_ = lambda t,x_: self.A_@x_ + self.F_@kron2(x_) + self.c_ + self.B_@u_(t)
        else:
            raise RuntimeError("improper use of _construct_f_()!")
        self.f_ = f_

    @staticmethod
    def _trained_model_from_operators(modelform, has_inputs, **attrs):
        """Construct a prediction-capable Model object from the operators of
        the reduced model.

        Parameters
        ----------
        modelform : str {'L', 'Lc', 'Q', 'Qc', 'LQ', 'LQc'}
            The structure of the desired reduced-order model.

        has_inputs : bool, optional, default: False.
            If True, assume the system has an additive input term u(t).

        attrs:
            Other attributes needed in order to call predict() successfully.
            Must include Vr, n, r, m, A_, F_, c_, and B_.

        Returns
        -------
        model : _BaseContinuousModel
            A new model, ready for predict() calls.
        """
        # Construct the new model object.
        model = _BaseContinuousModel(modelform, has_inputs)
        model._check_modelform()

        # Insert the attributes.
        for key,value in attrs.items():
            setattr(model, key, value)

        # Make sure that the new model has essential attributes for prediction.
        missed = [repr(a) for a in ['Vr', 'n','r','m', 'A_','F_','c_','B_']
                                                    if not hasattr(model, a)]
        if missed:
            _noun = "attribute" + ('s' if len(missed) > 1 else '')
            raise AttributeError(f"{_noun} {', '.join(missed)} required")

        # Construct ROM operator f_() if there are no system inputs.
        if not has_inputs:
            model._construct_f_()

        return model

    def __str__(self):
        """String representation: the structure of the model."""
        self._check_modelform()
        out = []
        if 'L' in self.modelform: out.append("Ax(t)")
        if 'Q' in self.modelform: out.append("H(x ⊗ x)(t)")
        if 'c' in self.modelform: out.append("c")
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
        self._check_modelform()
        self._check_hasinputs(u, 'u')

        # Check that the model is already trained.
        if not hasattr(self, 'B_'):
            raise AttributeError("model not trained (call fit() first)")

        # Check dimensions.
        if x0.shape[0] != self.n:
            raise ValueError("invalid initial state size "
                             f"({x0.shape[0]} != {self.n})")

        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        nt = t.shape[0]

        # Check for consistency with fit().
        if (self.has_inputs and self.B_ is None) or \
           (not self.has_inputs and self.B_ is not None):
            raise AttributeError("`has_inputs` attribute altered between fit()"
                                 " and predict(); call fit() again to retrain")

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
                # def u(s):
                #     """Interpolant for the discrete data U, aligned with t"""
                #     k = np.searchsorted(t, s)
                #     if k == 0:
                #         return U[:,0]
                #     # elif k == nt:         # This clause is never entered.
                #     #     return U[:,-1]
                #     return np.array([np.interp(s, t[k-1:k+1], U[i,k-1:k+1])
                #                                 for i in range(self.m)])

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


class _BaseDiscreteModel(_BaseModel):           # pragma: no cover
    """Base class for models that solve the discrete ROM problem,

        x_{k+1} = f(x_{k}, u_{k}),         x_{0} = x0.

    The problem may also be parametric (i.e., x and f may depend on an
    independent parameter).
    """
    def fit(self, *args, **kwargs):             # pragma: no cover
        raise NotImplementedError("fit() must be overwritten by child classes")

    def predict(self, x0, niters, u=None, **options):
        raise NotImplementedError

    def __str__(self):
        """String representation: the structure of the model."""
        self._check_modelform()
        out = []
        if 'L' in self.modelform: out.append("Ax_{k}")
        if 'Q' in self.modelform: out.append("H(x_{k} ⊗ x_{k})")
        if 'c' in self.modelform: out.append("c")
        if self.has_inputs: out.append("Bu_{k}")

        return "Reduced-order model structure: x_{k+1} = " + " + ".join(out)


# class _BaseAffineContinuousModel(_BaseModel) with predict() method.


# Useable classes =============================================================

# Continuous Models (i.e., solving dx/dt = f(t,x,u)) --------------------------
class InferredContinuousModel(_BaseContinuousModel, _NonparametricMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t,x(t),u(t)),           x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving a
    regularized ordinary least-squares problem.

    Parameters
    ----------
    modelform : str {'L', 'Lc', 'Q', 'Qc', 'LQ', 'LQc'}
        The structure of the desired reduced-order model. Options:
        'L'   : Linear model, f(x) = Ax.
        'Lc'  : Linear model with constant, f(x) = Ax + c.
        'Q'   : Quadratic model, f(x) = H(x⊗x).
        'Qc'  : Quadratic model with constant, f(x) = H(x⊗x) + c.
        'LQ'  : Linear-Quadratic model, f(x) = Ax + H(x⊗x).
        'LQc' : Linear-Quadratic model with constant, f(x) = Ax + H(x⊗x) + c.

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
        Learned ROM constant term, or None if 'c' is not in `modelform`.

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
        if X.shape != Xdot.shape:
            raise ValueError("X and Xdot different shapes "
                             f"({X.shape} != {Xdot.shape})")

        if X.shape[0] != Vr.shape[0]:
            raise ValueError("X and Vr not aligned, first dimension "
                             f"{X.shape[0]} != {Vr.shape[0]}")

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

        if 'c' in self.modelform:
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

        if 'c' in self.modelform:
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


class IntrusiveContinuousModel(_BaseContinuousModel, _NonparametricMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),         x(0) = x0.

    The user must specify the model form of the full-order model (FOM)
    operator f and the associated operators; the operators for the reduced
    model (ROM) are explicitly computed by projecting the full-order operators.

    Parameters
    ----------
    modelform : str {'L', 'Lc', 'Q', 'Qc', 'LQ', 'LQc'}
        The structure of the full-order and reduced-order models. Options:
        'L'   : Linear model, f(x) = Ax.
        'Lc'  : Linear model with constant, f(x) = Ax + c.
        'Q'   : Quadratic model, f(x) = H(x⊗x).
        'Qc'  : Quadratic model with constant, f(x) = H(x⊗x) + c.
        'LQ'  : Linear-Quadratic model, f(x) = Ax + H(x⊗x).
        'LQc' : Linear-Quadratic model with constant, f(x) = Ax + H(x⊗x) + c.

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
        FOM constant term, or None if 'c' is not in `modelform`.

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
        Learned ROM constant term, or None if 'c' is not in `modelform`.

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
            'Lc'  : [A, c],     or  [A, c, B]       if has_inputs is True.
            'Q'   : [H],        or  [A, H, B]       if has_inputs is True.
            'Qc'  : [H, c],     or  [H, c, B]       if has_inputs is True.
            'LQ'  : [A, H],     or  [A, H, B]       if has_inputs is True.
            'LQc' : [A, H, c],  or  [A, H, c, B]    if has_inputs is True.
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

        if 'c' in self.modelform:               # Constant term.
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


class InterpolatedInferredContinuousModel(_BaseContinuousModel):
    """Reduced order model for a system of high-dimensional ODEs, parametrized
    by a scalar p, of the form

         dx/dt = f(t,x(t;p),u(t);p),        x(0;p) = x0(p).

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving several
    regularized ordinary least-squares problems, then interpolating those
    models with respect to the parameter.

    Parameters
    ----------
    modelform : str {'L', 'Lc', 'Q', 'Qc', 'LQ', 'LQc'}
        The structure of the desired reduced-order model. Options:
        'L'   : Linear model, f(x) = Ax.
        'Lc'  : Linear model with constant, f(x) = Ax + c.
        'Q'   : Quadratic model, f(x) = H(x⊗x).
        'Qc'  : Quadratic model with constant, f(x) = H(x⊗x) + c.
        'LQ'  : Linear-Quadratic model, f(x) = Ax + H(x⊗x).
        'LQc' : Linear-Quadratic model with constant, f(x) = Ax + H(x⊗x) + c.

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
        Learned ROM constant terms, or None if 'c' is not in `modelform`.

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

        U : list of num_models (m,k) or (k,) ndarrays or None
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

        if not self.has_inputs:
            Us = [None]*num_models

        # TODO: figure out how to handle G (scalar, array, list(arrays)).

        # Train one model per parameter sample.
        self.Vr = Vr
        self.models_ = []
        for p, X, Xdot, U in zip(ps, Xs, Xdots, Us):
            model = InferredContinuousModel(self.modelform, self.has_inputs)
            model.fit(X, Xdot, Vr, U, G)
            model.p = p
            self.models_.append(model)

        # Check that all models have the same dimensions (n, m, and r):
        for attr in "nmr":
            setattr(self, attr, getattr(self.models_[0], attr))
            v = getattr(self, attr)
            for model in self.models_:
                if getattr(model, attr) != v:
                    raise ValueError(f"dimension '{attr}' "
                                     "inconsistent across training sets")

        # Construct interpolators.
        self.A_ = CubicSpline(ps, self.As_) if 'L' in self.modelform else None
        self.F_ = CubicSpline(ps, self.Fs_) if 'Q' in self.modelform else None
        self.H_ = CubicSpline(ps, self.Hs_) if 'Q' in self.modelform else None
        self.c_ = CubicSpline(ps, self.cs_) if 'c' in self.modelform else None
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
        self._check_modelform()
        self._check_hasinputs(u, 'u')

        model = _BaseContinuousModel._trained_model_from_operators(
                    self.modelform,
                    self.has_inputs,
                    n=self.n,
                    r=self.r,
                    m=self.m,
                    Vr=self.Vr,
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
        return [m.c_ for m in self.models_] if 'c' in self.modelform else None

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


# class AffineInferredContinuousModel(_BaseContinuousModel):  # pragma: no cover
    pass


# class AffineIntrusiveContinuousModel(_BaseContinuousModel): # pragma: no cover
    pass


# Discrete Models (i.e., solving x_{k+1} = f(x_{k},u_{k})) --------------------


__all__ = [
            "InferredContinuousModel",
            "IntrusiveContinuousModel",
            # "AffineInferredContinuousModel",
            # "AffineIntrusiveContinuousModel",
            "InterpolatedInferredContinuousModel",
          ]
