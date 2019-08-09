# OpInf.py
"""Class for Model Order Reduction of ODEs via operator inference."""

import numpy as np
from scipy.integrate import solve_ivp

from . import opinf_helper, integration_helpers


class ReducedModel:
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t,x(t),u(t)),
           x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving a
    regularized ordinary least squares problem.

    Attributes
    ----------
    modelform : str
        Designates the structure of the reduced model to learn.
    has_inputs : bool
        True
            Assume the system has an additive input u(t).
        False (default)
            Assume the system does not have an additive input u(t).
    """

    _VALID_MODEL_FORMS = {"L", "Lc", "Q", "Qc", "LQ", "LQc"}

    def __init__(self, modelform, has_inputs=False):
        """Initalize operator inference model for the high-dimensional model

            dx / dt = f(t,x(t),u(t)),
               x(0) = x0.

        Parameters
        ----------
        modelform : {'L','Lc','Q','Qc','LQ','LQc'}
            The structure of the desired reduced-order model.
            'L'
                A linear model f(x) = Ax.
            'Lc'
                A linear model with a constant f(x) = Ax + c.
            'LQ'
                A linear quadratic model f(x) = Ax + Fx^2.
            'LQc'
                A linear quadratic model with a constant f(x) = Ax + Fx^2 + c.
            'Q'
                A strictly quadratic model f(x) = Fx^2.
            'Qc'
                A strictly quadratic model with a constant f(x) = Fx^2 + c.

        has_inputs : bool
            True
                Assume the system has an additive input u(t).
            False (default)
                Assume the system does not have an additive input u(t).
        """
        self.modelform = modelform
        self.has_inputs = has_inputs

    def fit(self, X_, Xdot_, U=None, reg=0):
        """Solve for the reduced model operators.

        Parameters
        ----------
        X_ : (reduced_dimension, num_snapshots) ndarray
            The PROJECTED snapshot training data.
        Xdot_ : (reduced_dimension, num_snapshots) ndarray
            The PROJECTED velocity training data.
        U : (num_inputs, num_snapshots) ndarray
            The inputs corresponding to the snapshots.
        reg : float
            L2 regularization penalty. Can be imposed if this is non-zero.
            Solves min ||Do - r||_2 + k || reg ||_2.
        Returns
        -------
        self
        """
        r,k = X_.shape          # (num basis vectors, num snapshots)
        p = 0

        s = int(r*(r+1)/2)      # Dimension of compact Kronecker.
        self.r = r

        if self.modelform not in self._VALID_MODEL_FORMS:
            raise ValueError(f"invalid modelform '{self.modelform}'. "
                             f"Options are {self._VALID_MODEL_FORMS}.")

        # Linear Quadratic
        if self.modelform == 'LQ':
            X2_ = opinf_helper.get_x_sq(X_.T)
            D = np.hstack((X_.T,X2_))
            oshape = r + s

        # Linear Quadratic with a constant
        elif self.modelform == 'LQc':
            X2_ = opinf_helper.get_x_sq(X_.T)
            D = np.hstack((X_.T,X2_, np.ones((k,1))))
            p += 1
            oshape = r + s

        # Linear
        elif self.modelform == 'L':
            D = X_.T
            s = 0
            oshape = r

        # Linear with a constant
        elif self.modelform == 'Lc':
            D = np.hstack((X_.T,np.ones((k,1))))
            p += 1
            oshape = r

        # Strictly Quadratic
        elif self.modelform == 'Q':
            D = opinf_helper.get_x_sq(X_.T)
            oshape = s

        # Strictly Quadratic with a constant
        elif self.modelform == 'Qc':
            D = np.hstack((opinf_helper.get_x_sq(X_.T), np.ones((k,1))))
            p += 1
            oshape = s

        else:
            raise ValueError(f"invalid modelform '{self.modelform}'. "
                             "Options are 'L','Lc','LQ','LQc','Q','Qc'.")

        if self.has_inputs:

            U = np.atleast_2d(U)
            D = np.hstack((D,U.T))
            p += U.shape[0]

        # Solve for the operators !
        O = np.zeros((oshape+p,r))
        for it in range(r):
            O[:,it] = np.ravel(opinf_helper.normal_equations(D,Xdot_[it,:],reg,it))
        O = O.T

        self.residual_ = np.linalg.norm(D@O.T - Xdot_.T,2)**2
        self.solution_ = np.linalg.norm(O.T,2)**2

        # Linear
        if self.modelform == 'L':
            if self.has_inputs:
                # A B
                self.A_, self.B_ = O[:,:r],O[:,r:r+p]
            else:
                # A
                self.A_ = O[:,:r]
                self.B_ = np.zeros((r,1))

        # Linear with a constant
        elif self.modelform == 'Lc':
            if self.has_inputs:
                # A c B
                self.A_, self.c_, self.B_ = O[:,:r],O[:,r:r+1],O[:,r+1:r+1+p]
            else:
                # A c
                self.A_, self.c_ = O[:,:r],O[:,r:r+1]
                self.B_ = np.zeros((r,1))

        # Linear Quadratic
        elif self.modelform == 'LQ':
            if self.has_inputs:
                # A H B
                # self.F = O[:,r:r+s]
                self.A_, self.H_, self.B_ = O[:,:r], opinf_helper.F2H(O[:,r:r+s]),O[:,r+s:r+s+p]
            else:
                # A H
                self.A_, self.H_ = O[:,:r], opinf_helper.F2H(O[:,r:r+s])
                self.B_ = np.zeros((r,1))

        # Linear Quadratic with constant
        elif self.modelform == 'LQc':
            if self.has_inputs:
                # A H c B
                self.A_, self.H_, self.c_, self.B_ = O[:,:r],opinf_helper.F2H(O[:,r:r+s]),O[:,r:r+1],O[:,r+s+1:r+s+p+1]

            else:
                # A H c
                self.B_ = np.zeros((r,1))
                self.A_, self.H_, self.c_ = O[:,:r],opinf_helper.F2H(O[:,r:r+s]),O[:,r:r+1],

        # Strictly Quadratic
        elif self.modelform == 'Q':
            if self.has_inputs:
                # H B
                self.H_, self.B_ = opinf_helper.F2H(O[:,:s]),O[:,s:s+p]
            else:
                # H
                self.H_ = opinf_helper.F2H(O[:,:s])
                self.B_ = np.zeros((r,1))

        # Strictly Quadratic with a constant
        elif self.modelform == 'Qc':
            if self.has_inputs:
                # H c B
                self.H_, self.c_, self.B_ = opinf_helper.F2H(O[:,:s]),O[:,s:s+1],O[:,s+1:s+1+p]
            else:
                # H c
                self.H_, self.c_ = opinf_helper.F2H(O[:,:s]),O[:,s:s+1]
                self.B_ = np.zeros((r,1))

        return self

    def fit2(self, X, Xdot, Vr, U=None, reg=0):
        """Solve for the reduced model operators.

        Parameters
        ----------
        X : (n,k) ndarray
            Column-wise snapshot training data (each column is a snapshot).
        Xdot : (n,k) ndarray
            Column-wise velocity training data.
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
        U : (m,k) ndarray
            Column-wise inputs corresponding to the snapshots.
        reg : float
            L2 regularization penalty. Can be imposed if this is non-zero.
            Solves min ||Do - r||_2 + reg*||P @ o||_2.
        TODO: P

        Returns
        -------
        self
        """
        # Verify modelform.
        if self.modelform not in self._VALID_MODEL_FORMS:
            raise ValueError(f"invalid modelform '{self.modelform}'. "
                             f"Options are {self._VALID_MODEL_FORMS}.")

        if self.has_inputs and U is None:
            raise ValueError("argument 'U' required since has_inputs=True")

        if not self.has_inputs and U is not None:
            raise ValueError("argument 'U' invalid since has_inputs=False")

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
            X2T_ = opinf_helper.get_x_sq(X_.T)
            D_blocks.append(X2T_)
            s = r*(r+1) // 2      # Dimension of compact Kronecker.
            if X2T_.shape[1] != s:
                raise ArithmeticError("get_x_sq() FAILED: incorrect size!")

        if 'c' in self.modelform:
            D_blocks.append(np.ones(k).reshape((k,1)))

        if self.has_inputs:
            if U.ndim == 1:
                U = U.reshape((-1,k))
            D_blocks.append(U.T)
            m = U.shape[0]
            self.m = m

        D = np.hstack(D_blocks)
        d = D.shape[1]

        # Solve for the reduced-order model operators.
        O = np.zeros((d, r))
        for j in range(r):
            O[:,j] = opinf_helper.normal_equations(D,
                                                   Xdot_[j,:],
                                                   reg,
                                                   j).flatten()

        # Calculate residuals.
        self.residual_ = np.sum((D @ O - Xdot_.T)**2) # squared Frobenius norm
        self.solution_ = np.linalg.norm(O.T, ord=2)**2 # squared 2 norm? TODO

        # Extract the reduced operators from O.
        i = 0
        if 'L' in self.modelform:
            self.A_ = O[i:i+self.r]
            i += self.r
        else:
            self.A_ = None

        if 'Q' in self.modelform:
            self.F_ = O[i:i+s]
            i += s
            self.H_ = opinf_helper.F2H(self.F_)
        else:
            self.F_, self.H_ = None, None

        if 'c' in self.modelform:
            self.c_ = O[i:i+1][0]       # Note that c is one-dimensional.
            i += 1
        else:
            self.c_ = None

        if self.has_inputs:
            self.B_ = O[i:i+self.m]
            i += self.m
        else:
            self.B_ = None

        if i != d:
            raise ArithmeticError("EXTRACTION FAILED: sizes don't match!")

        # Construct the complete ROM operator IF there are not control inputs.
        if not self.has_inputs:
            if self.modelform == "L":
                f_ = lambda t,x_: self.A_@x_
            elif self.modelform == "Lc":
                f_ = lambda t,x_: self.A_@x_ + self.c_
            elif self.modelform == "Q":
                f_ = lambda t,x_: self.H_@np.kron(x_,x_)
            elif self.modelform == "Qc":
                f_ = lambda t,x_: self.H_@np.kron(x_,x_) + self.c_
            elif self.modelform == "LQ":
                f_ = lambda t,x_: self.A_@x_ + self.H_@np.kron(x_,x_)
            elif self.modelform == "LQc":
                f_ = lambda t,x_: self.A_@x_ + self.H_@np.kron(x_,x_) + self.c_
            self.f_ = f_

        return self

    def predict(self, x0_, n_timesteps, dt, U=None):
        """Simulate the learned model.
        Parameters
        ----------
        x0_ : (n_variables,) ndarray
            The PROJECTED initial state vector to begin simulation.
        n_timesteps : int
            Number of time steps to simulate.
        dt : float
            Time step size.
        U : (n_parameters, n_timesteps) ndarray, optional, default=None
            The input for each time step.
        Output
        ------
        projected_state: (n_variables, n_iters) ndarray
            The reduced state
        n_iters: int
            The number of time steps computed.
        """
        if x0_.shape[0] != self.r:
            raise ValueError(f"invalid initial state size ({r} != {self.r})")

        if U is not None:
            U = np.atleast_2d(U)
            K = U.shape[1]
            if U.any() and K != n_timesteps:
                raise ValueError(f"invalid input shape ({K} != {n_timesteps})")
        else:
            U = np.zeros((1, n_timesteps))


        projected_state = np.zeros((self.r, n_timesteps))
        projected_state[:,0] = x0_.copy()

        # Integrate, depending on the modelform.

        # Strictly linear
        if self.modelform == 'L':
            for i in range(1,n_timesteps):
                projected_state[:,i] = integration_helpers.rk4advance_L(projected_state[:,i-1],dt,self.A_,self.B_,U[:,i])
                if np.any(np.isnan(projected_state[:,i])):
                    print("NaNs enountered at step ", i)
                    break
            return projected_state,i

        # Linear with constant
        elif self.modelform == 'Lc':
            for i in range(1,n_timesteps):
                projected_state[:,i] = integration_helpers.rk4advance_Lc(projected_state[:,i-1],dt,self.A_,self.c_[:,0],self.B_,U[:,i])
                if np.any(np.isnan(projected_state[:,i])):
                    print("NaNs enountered at step ", i)
                    break
            return projected_state,i

        # Strictly Quadratic
        elif self.modelform == 'Q':
            for i in range(1,n_timesteps):
                projected_state[:,i] = integration_helpers.rk4advance_Q(projected_state[:,i-1],dt,self.H_,self.B_,U[:,i])
                if np.any(np.isnan(projected_state[:,i])):
                    print("NaNs enountered at step ", i)
                    break
            return projected_state,i

        # Strictly Quadratic with a constant
        elif self.modelform == 'Qc':
            for i in range(1,n_timesteps):
                projected_state[:,i] = integration_helpers.rk4advance_Qc(projected_state[:,i-1],dt,self.H_,self.c_[:,0],self.B_,U[:,i])
                if np.any(np.isnan(projected_state[:,i])):
                    print("NaNs enountered at step ", i)
                    break
            return projected_state,i

        # Linear Quadratic
        elif self.modelform == 'LQ':
            for i in range(1,n_timesteps):
                projected_state[:,i] = integration_helpers.rk4advance_LQ(projected_state[:,i-1],dt,self.A_,self.H_,self.B_,U[:,i])
                if np.any(np.isnan(projected_state[:,i])):
                    print("NaNs enountered at step ", i)
                    break
            return projected_state,i

        # Linear Quadratic with constant
        elif self.modelform == 'LQc':
            for i in range(1,n_timesteps):
                projected_state[:,i] = integration_helpers.rk4advance_LQc(projected_state[:,i-1],dt,self.A_,self.H_,self.c_[:,0],self.B_,U[:,i])
                if np.any(np.isnan(projected_state[:,i])):
                    print("NaNs enountered at step ", i)
                    break
            return projected_state,i

    def predict2(self, x0, t, u=None, method="RK45"):
        """Simulate the learned ROM with scipy.integrate.solve_ivp().

        Parameters
        ----------
        x0 : (n,) ndarray
            The initial (high-dimensional) state vector to begin a simulation.
        t : (T,) ndarray
            The time domain over which to integrate the reduced-order system.
        u : callable OR (m,T) ndarray
            The input as a function of time (preferred) OR the input at the
            times `t`. If given as an array, u(t) is calculated by linearly
            interpolating known data points if needed for an adaptive solver.
        method : str
            The solver to use to solve the reduced-order system. Must be valid
            as the `method` keyword argument of scipy.integrate.solve_ivp():
            * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
            * 'RK23': Explicit Runge-Kutta method of order 3(2)
            * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
                order 5.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method based
                on a backward differentiation formula for the derivative.
            * 'LSODA': Adams/BDF method with automatic stiffness detection and
                switching. This wraps the Fortran solver from ODEPACK.

        Output
        ------
        X_ROM: (n,T) ndarray
            The reduced-order approximation to the full-order system over `t`.
        """
        # Check that the model is already trained.
        if not hasattr(self, 'B_'):
            raise AttributeError("model not trained (call fit() first)")

        # Check dimensions.
        if x0.shape[0] != self.n:
            raise ValueError("invalid initial state size "
                             f"({x0.shape[0]} != {self.n})")

        if t.ndim != 1:
            raise ValueError("time 't' must be one-dimensional")
        T = t.shape[0]

        # Verify `u` matches model specifications.
        if not self.has_inputs and u is not None:
            raise ValueError("argument 'u' invalid since has_inputs=False")

        if self.has_inputs and u is None:
            raise ValueError("argument 'u' required since has_inputs=True")

        # Project initial conditions.
        x0_ = self.Vr.T @ x0

        # Interpret control input argument `u`.
        if self.has_inputs:
            if not callable(u):         # Then u should an (m,T) array.
                U = np.atleast_2d(u.copy())
                if U.shape != (self.m,T):
                    raise ValueError("invalid input shape "
                                     f"({U.shape} != {(m,T)}")
                def u(s):               # Write an interpolator function.
                    """Interpolant for the discrete data U, aligned with t"""
                    k = np.searchsorted(t, s)
                    if np.isscalar(s):
                        if k == 0 or k == T:
                            return U[:,k]
                        elif self.m == 1:
                            return np.interp(s, t[k-1:k+1], U[0,k-1:k+1])
                        return np.row_stack([np.interp(s, t[k-1:k+1],
                                                          U[i,k-1:k+1])
                                             for i in range(self.m)])
                    else:
                        return np.array([u(ss) for ss in s])

            # Construct the ROM operator if needed (waited for control inputs).
            if self.modelform == "L":
                f_ = lambda t,x_: self.A_@x_ + self.B_@u(t)
            elif self.modelform == "Lc":
                f_ = lambda t,x_: self.A_@x_ + self.c_ + self.B_@u(t)
            elif self.modelform == "Q":
                f_ = lambda t,x_: self.H_@np.kron(x_,x_) + self.B_@u(t)
            elif self.modelform == "Qc":
                f_ = lambda t,x_: self.H_@np.kron(x_,x_) + self.c_ + self.B_@u(t)
            elif self.modelform == "LQ":
                f_ = lambda t,x_: self.A_@x_ + self.H_@np.kron(x_,x_) + self.B_@u(t)
            elif self.modelform == "LQc":
                f_ = lambda t,x_: self.A_@x_ + self.H_@np.kron(x_,x_) + self.c_ + self.B_@u(t)
            self.f_ = f_

        # Integrate the reduced-order model.
        sol = solve_ivp(self.f_,
                        [t.min(), t.max()],
                        x0_,
                        method=method,
                        t_eval=t)

        # Raise errors if the integration failed.
        if not sol.success:
            raise Exception(sol.message)

        # Reconstruct the approximation to the full-order model.
        return self.Vr @ sol.y

    def get_residual_norm(self):
        return self.residual_, self.solution_

    def get_operators(self):
        """Return the operators of the learned model.

        Returns
        ------
        (operators,) : tuple of ndarrays
            Each operator as defined by modelform of the model.
        """
        operators = ()
        if 'L' in self.modelform:
            operators += (self.A_,)
        if 'Q' in self.modelform:
            operators += (self.H_,)
        if 'c' in self.modelform:
            operators += (self.c_,)
        if self.has_inputs:
            operators += (self.B_,)

        return operators

    def __str__(self):
        """String representation: the structure of the model."""


        if self.modelform not in self._VALID_MODEL_FORMS:
            raise ValueError(f"invalid modelform '{self.modelform}'. "
                             f"Options are {self._VALID_MODEL_FORMS}.")
        out = []
        if 'L' in self.modelform:
            out.append("Ax(t)")
        if 'Q' in self.modelform:
            out.append("H(x ⊗ x)(t)")
        if 'c' in self.modelform:
            out.append("c")
        if self.has_inputs:
            out.append("Bu(t)")

        return "dx / dt = " + " + ".join(out)



__all__ = ["ReducedModel"]
