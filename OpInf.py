# OpInf.py
"""Class for Model Order Reduction of ODEs via operator inference."""

import numpy as np
from scipy.sparse import csr_matrix
import opinf_helper, integration_helpers


class ReducedModel:
    """Reduced order model for a system of high-dimensional ODEs of the form

        x'(t) = f(t,x(t),u(t)),
         x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving a
    regularized ordinary least squares problem.

    Attributes
    ----------
    modelform : str
        Designates the structure of the reduced model to learn.
    inp : bool
        Whether or not the full model includes the term Bu(t).
    """

    _VALID_MODEL_FORMS = {"L", "Lc", "Q", "Qc", "LQ", "LQc"}

    def __init__(self, modelform, inp=True):
        """Initalize operator inference model for the high-dimensional model

            x'(t) = f(t,x(t),u(t)),
             x(0) = x0.

        Parameters
        ----------
        modelform : {'L','Lc','Q','Qc','LQ','LQc'}
            'L':
                a linear model f(x) = Ax
            'Lc'
                a linear model with a constant f(x) = Ax + c
            'LQ'
                a linear quadratic model f(x) = Ax + Fx^2
            'LQc'
                a linear quadratic model with a constant f(x) = Ax + Fx^2 + c
            'Q'
                a strictly quadratic model f(x) = Fx^2
            'Qc'
                a strictly quadratic model with a constant f(x) = Fx^2 + c

        inp : bool, optional, default = True
            True
                assume the system has an input term u(t) != 0
            False
                assume the system does not have an input u(t) = 0
        """
        self.modelform = modelform
        self.inp = inp

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
        self.r_ = r

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

        if self.inp:

            U = np.atleast_2d(U)
            D = np.hstack((D,U.T))
            p += U.shape[0]

        # Solve for the operators !
        O = np.zeros((oshape+p,r))
        for it in range(r):
            O[:,it] = np.ravel(opinf_helper.normal_equations(D,Xdot_[it,:],reg,it))
        O = O.T

        self.residual = np.linalg.norm(D@O.T - Xdot_.T,2)**2
        self.solution = np.linalg.norm(O.T,2)**2

        # Linear
        if self.modelform == 'L':
            if self.inp:
                # A B
                self.A_, self.B_ = O[:,:r],O[:,r:r+p]
            else:
                # A
                self.A_ = O[:,:r]
                self.B_ = np.zeros((r,1))

        # Linear with a constant
        elif self.modelform == 'Lc':
            if self.inp:
                # A c B
                self.A_, self.c_, self.B_ = O[:,:r],O[:,r:r+1],O[:,r+1:r+1+p]
            else:
                # A c
                self.A_, self.c_ = O[:,:r],O[:,r:r+1]
                self.B_ = np.zeros((r,1))

        # Linear Quadratic
        elif self.modelform == 'LQ':
            if self.inp:
                # A H B
                # self.F = O[:,r:r+s]
                self.A_, self.H_, self.B_ = O[:,:r], opinf_helper.F2H(O[:,r:r+s]),O[:,r+s:r+s+p]
            else:
                # A H
                self.A_, self.H_ = O[:,:r], opinf_helper.F2H(O[:,r:r+s])
                self.B_ = np.zeros((r,1))

        # Linear Quadratic with constant
        elif self.modelform == 'LQc':
            if self.inp:
                # A H c B
                self.A_, self.H_, self.c_, self.B_ = O[:,:r],opinf_helper.F2H(O[:,r:r+s]),O[:,r:r+1],O[:,r+s+1:r+s+p+1]

            else:
                # A H c
                self.B_ = np.zeros((r,1))
                self.A_, self.H_, self.c_ = O[:,:r],opinf_helper.F2H(O[:,r:r+s]),O[:,r:r+1],

        # Strictly Quadratic
        elif self.modelform == 'Q':
            if self.inp:
                # H B
                self.H_, self.B_ = opinf_helper.F2H(O[:,:s]),O[:,s:s+p]
            else:
                # H
                self.H_ = opinf_helper.F2H(O[:,:s])
                self.B_ = np.zeros((r,1))

        # Strictly Quadratic with a constant
        elif self.modelform == 'Qc':
            if self.inp:
                # H c B
                self.H_, self.c_, self.B_ = opinf_helper.F2H(O[:,:s]),O[:,s:s+1],O[:,s+1:s+1+p]
            else:
                # H c
                self.H_, self.c_ = opinf_helper.F2H(O[:,:s]),O[:,s:s+1]
                self.B_ = np.zeros((r,1))

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
        if x0_.shape[0] != self.r_:
            raise ValueError(f"invalid initial state size ({r} != {self.r_})")

        if U is not None:
            U = np.atleast_2d(U)
            K = U.shape[1]
            if U.any() and K != n_timesteps:
                raise ValueError(f"invalid input shape ({K} != {n_timesteps})")
        else:
            U = np.zeros((1, n_timesteps))


        projected_state = np.zeros((self.r_, n_timesteps))
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

    def get_residual_norm(self):
        return self.residual, self.solution

    def get_operators(self):
        """Return the operators of the learned model.

        Returns
        ------
        (ops,) : tuple of ndarrays
            Each operator as defined by modelform of the model.
        """
        ops = ()

        if self.modelform == 'L':
            ops += (self.A_,)
        elif self.modelform == 'Lc':
            ops += (self.A_,self.c_)
        elif self.modelform == 'LQ':
            ops += (self.A_, self.H_)
        elif self.modelform == 'LQc':
            ops += (self.A_, self.H_, self.c_)
        elif self.modelform == 'Q':
            ops += (self.H_)
        else:
            ops += (self.H_,self.c_)

        if self.inp:
            ops += (self.B_,)

        return ops

    def __str__(self):
        """String representation: the structure of the model."""


        if self.modelform not in self._VALID_MODEL_FORMS:
            raise ValueError(f"invalid modelform '{self.modelform}'. "
                             f"Options are {self._VALID_MODEL_FORMS}.")
        out = []
        if 'L' in self.modelform:
            out.append("Ax(t)")
        if 'Q' in self.modelform:
            out.append("H(x x)(t)")
        if 'c' in self.modelform:
            out.append("c")
        if self.inp:
            out.append("Bu(t)")

        return "x'(t) = " + " + ".join(out)
