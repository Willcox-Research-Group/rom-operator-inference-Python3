# _affine_inference.py
"""Class for model order reduction of parameteric ODEs via operator inference."""

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

# Base classes ================================================================

# Mixins ======================================================================

# Useable classes =============================================================

# Continuous models (i.e., solving dx/dt = f(t,x,u)) --------------------------
class AffineInferredContinuousROM(_AffineContinuousROM,
                                  _InferredMixin, _AffineMixin):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t); µ),          x(0;µ) = x0(µ).

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

    c_ : func(µ) -> (r,) ndarray; (r,) ndarray; or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : func(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : func(µ) -> (r,r(r+1)//2) ndarray; (r,r(r+1)//2) ndarray; or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : func(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    B_ : func(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, µs, affines, Xs, Xdots, Vr, Us=None, P=0):
        """Solve for the reduced model operators via regularized least squares.
        For terms with affine structure, solve for the constituent operators.

        Parameters
        ----------
        µs : list of s scalars or (p,) ndarrays
            Parameter values at which the snapshot data is collected.

        affines : dict(str -> list(functions))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'B': Linear input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

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
        self._check_affines(affines, µs[0])
        self._check_inputargs(Us, 'Us')

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
                if 'c' in affines:
                    for j in range(len(affines['c'])):
                        row.append(affines['c'][j](µ) * ones)
                else:
                    row.append(ones)

            if self.has_linear:
                if 'A' in affines:
                    for j in range(len(affines['A'])):
                        row.append(affines['A'][j](µ) * Xs_[i].T)
                else:
                    row.append(Xs_[i].T)

            if self.has_quadratic:
                X2i_ = kron2(Xs_[i])
                if 'H' in affines:
                    for j in range(len(affines['H'])):
                        row.append(affines['H'][j](µ) * X2i_.T)
                else:
                    row.append(X2i_.T)

            if self.has_inputs:
                Ui = Us[i]
                if self.m == 1:
                    Ui = Ui.reshape((1,k))
                if 'B' in affines:
                    for j in range(len(affines['B'])):
                        row.append(affines['B'][j](µ) * Ui.T)
                else:
                    row.append(Ui.T)

            D_blockrows.append(np.hstack(row))

        D = np.vstack(D_blockrows)
        self.datacond_ = np.linalg.cond(D)      # Condition number of data.
        R = np.hstack(Xdots_).T

        # Solve for the reduced-order model operators via least squares.
        Otrp, res = lstsq_reg(D, R, P)[0:2]
        self.residual_ = np.sum(res)

        # Extract the reduced operators from Otrp.
        i = 0
        if self.has_constant:
            if 'c' in affines:
                cs_ = []
                for j in range(len(affines['c'])):
                    cs_.append(Otrp[i:i+1][0])      # c_ is one-dimensional.
                    i += 1
                self.c_ = AffineOperator(affines['c'], cs_)
            else:
                self.c_ = Otrp[i:i+1][0]            # c_ is one-dimensional.
                i += 1
        else:
            self.c_, self.cs_ = None, None

        if self.has_linear:
            if 'A' in affines:
                As_ = []
                for j in range(len(affines['A'])):
                    As_.append(Otrp[i:i+self.r].T)
                    i += self.r
                self.A_ = AffineOperator(affines['A'], As_)
            else:
                self.A_ = Otrp[i:i+self.r].T
                i += self.r
        else:
            self.A_ = None

        if self.has_quadratic:
            _r2 = self.r * (self.r + 1) // 2
            if 'H' in affines:
                Hcs_ = []
                for j in range(len(affines['H'])):
                    Hcs_.append(Otrp[i:i+_r2].T)
                    i += _r2
                self.Hc_ = AffineOperator(affines['H'], Hcs_)
                self.H_ = lambda µ: Hc2H(self.Hc_(µ))
            else:
                self.Hc_ = Otrp[i:i+_r2].T
                i += _r2
                self.H_ = Hc2H(self.Hc)
        else:
            self.Hc_, self.H_ = None, None

        if self.has_inputs:
            if 'B' in affines:
                Bs_ = []
                for j in range(len(affines['B'])):
                    Bs_.append(Otrp[i:i+self.m].T)
                    i += self.m
                self.B_ = AffineOperator(affines['B'], Bs_)
            else:
                self.B_ = Otrp[i:i+self.m].T
                i += self.m
        else:
            self.B_ = None

        return self

# Discrete models (i.e., solving x_{k+1} = f(x_{k},u_{k})) --------------------
