# _core/_affine/inferred.py
"""Affinely parametric ROM classes that use Operator Inference.

Classes
-------
* _AffineInferredMixin(_InferredMixin, _AffineMixin)
* AffineInferredDiscreteROM(_AffineInferredMixin, _DiscreteROM)
* AffineInferredContinuousROM(_AffineInferredMixin, _ContinuousROM)
"""

__all__ = [
            "AffineInferredDiscreteROM",
            "AffineInferredContinuousROM",
          ]

import numpy as np

from ._base import AffineOperator, _AffineMixin
from .._base import _ContinuousROM, _DiscreteROM
from .._inferred import (_InferredMixin,
                         InferredDiscreteROM,
                         InferredContinuousROM)
from ...utils import (lstsq_reg,
                      expand_Hc as Hc2H,
                      expand_Gc as Gc2G,
                      kron2c, kron3c)


# Affine inferred mixin (private) =============================================
class _AffineInferredMixin(_InferredMixin, _AffineMixin):
    """Mixin class for affinely parametric inferred reduced model classes."""
    def fit(self, ModelClass, Vr, µs, affines, Xs, rhss, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares.
        For terms with affine structure, solve for the component operators.

        Parameters
        ----------
        ModelClass: class
            ROM class, either _ContinuousROM or _DiscreteROM, to use for the
            newly constructed model.

        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        µs : list of s scalars or (p,) ndarrays
            Parameter values at which the snapshot data is collected.

        affines : dict(str -> list(callables))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'G': Cubic state matrix G(µ).
            * 'B': Linear input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

        Xs : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

        rhss : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise next-iteration (discrete model) or velocity
            (continuous model) training data. The ith array, rhss[i],
            corresponds to the ith parameter, µs[i].

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
        self._check_modelform(trained=False)
        self._check_affines(affines, µs[0])
        self._check_inputargs(Us, 'Us')
        is_continuous = issubclass(ModelClass, _ContinuousROM)

        # Check that the number of params matches the number of snapshot sets.
        s = len(µs)
        if len(Xs) != s:
            raise ValueError("num parameter samples != num state snapshot "
                             f"sets ({s} != {len(Xs)})")
        if len(rhss) != s:
            raise ValueError("num parameter samples != num rhs "
                             f"sets ({s} != {len(rhss)})")

        # Check and store dimensions.
        self.n, self.r = Vr.shape
        self.m = None

        # Check that the arrays in each list have the same number of columns.
        _tocheck = [Xs]
        if is_continuous:
            _tocheck.append(rhss)
        if self.has_inputs:
            self.m = Us[0].shape[0] if Us[0].ndim == 2 else 1
            # Check that the input dimension is the same in each data set.
            for U in Us:
                m = U.shape[0] if U.ndim == 2 else 1
                if m != self.m:
                    raise ValueError("control inputs not aligned")
        else:
            Us = [None]*s
        for dataset in _tocheck:
            self._check_training_data_shapes(dataset)

        # TODO: figure out how to handle P (scalar, array, list(arrays)).

        # Project states and velocities to the reduced subspace.
        self.Vr = Vr
        Xs_ = [self.project(X, 'X') for X in Xs]
        rhss_ = [self.project(rhs, 'rhs') for rhs in rhss]

        # Construct the large "Data matrix" D.
        D_blockrows = []
        for i,(µ,X_) in enumerate(zip(µs, Xs_)):
            row = []
            k = X_.shape[1]

            if self.has_constant:
                ones = np.ones((k,1))
                if 'c' in affines:
                    row += [θ(µ) * ones for θ in affines['c']]
                else:
                    row.append(ones)

            if self.has_linear:
                if 'A' in affines:
                    row += [θ(µ) * X_.T for θ in affines['A']]
                else:
                    row.append(X_.T)

            if self.has_quadratic:
                X2_ = kron2c(X_)
                if 'H' in affines:
                    row += [θ(µ) * X2_.T for θ in affines['H']]
                else:
                    row.append(X2_.T)

            if self.has_cubic:
                X3_ = kron3c(X_)
                if 'G' in affines:
                    row += [θ(µ) * X3_.T for θ in affines['G']]
                else:
                    row.append(X3_.T)

            if self.has_inputs:
                U = Us[i]
                if self.m == 1:
                    U = U.reshape((1,k))
                if 'B' in affines:
                    row += [θ(µ) * U.T for θ in affines['B']]
                else:
                    row.append(U.T)

            D_blockrows.append(np.hstack(row))

        D = np.vstack(D_blockrows)
        self.datacond_ = np.linalg.cond(D)      # Condition number of data.
        R = np.hstack(rhss_).T
        self._D_ = D.copy()                     ## Save data matrix for later.

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
                self.H_ = Hc2H(self.Hc_)
        else:
            self.Hc_, self.H_ = None, None

        if self.has_cubic:
            _r3 = self.r * (self.r + 1) * (self.r + 2) // 6
            if 'G' in affines:
                Gcs_ = []
                for j in range(len(affines['G'])):
                    Gcs_.append(Otrp[i:i+_r3].T)
                    i += _r3
                self.Gc_ = AffineOperator(affines['G'], Gcs_)
                self.G_ = lambda µ: Gc2G(self.Gc_(µ))
            else:
                self.Gc_ = Otrp[i:i+_r3].T
                i += _r3
                self.G_ = Gc2G(self.Gc_)
        else:
            self.Gc_, self.G_ = None, None

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


# Affine inferred models (public) =============================================
class AffineInferredDiscreteROM(_AffineInferredMixin, _DiscreteROM):
    """Reduced order model for a high-dimensional, parametrized discrete
    dynamical system of the form

        x_{j+1}(µ) = f(x_{j}(µ), u_{j}; µ),     x_{0}(µ) = x0(µ),

    where one or more of the operators that compose f have an affine
    dependence on the parameter, e.g., A(µ) = θ1(µ)A1 + θ2(µ)A2 + θ3(µ)A3.
    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving a single
    ordinary least-squares problem.

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

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c(µ).

    has_linear : bool
        Whether or not there is a linear term A(µ)x(t).

    has_quadratic : bool
        Whether or not there is a quadratic term H(µ)(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic term G(µ)(x⊗x⊗x)(t).

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

    c_ : callable(µ) -> (r,) ndarray; (r,) ndarray; or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : callable(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : callable(µ) -> (r,r(r+1)//2) ndarray; (r,r(r+1)//2) ndarray; or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : callable(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : callable(µ) -> (r,r(r+1)(r+2)//6) ndarray;
          (r,r(r+1)(r+2)//6) ndarray; or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : callable(µ) -> (r,r**3) ndarray; (r,r**3) ndarray; or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

    B_ : callable(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.
    """
    def fit(self, Vr, µs, affines, Xs, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        using solution trajectories from multiple examples.

        Parameters
        ----------
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        affines : dict(str -> list(callables))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'B': Linear input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

        Xs : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array Xs[i] corresponds to the ith parameter, µs[i].

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
        # Truncate extra inputs for convenience.
        if Us is not None:
            Us = [U[...,:X.shape[1]-1] for U,X in zip(Us, Xs)]

        return _AffineInferredMixin.fit(self, InferredDiscreteROM,
                                        Vr, µs, affines,
                                        [X[:,:-1] for X in Xs],
                                        [X[:,1:]  for X in Xs],
                                        Us, P)

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


class AffineInferredContinuousROM(_AffineInferredMixin, _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t); µ),          x(0;µ) = x0(µ),

    where one or more of the operators that compose f have an affine
    dependence on the parameter, e.g., A(µ) = θ1(µ)A1 + θ2(µ)A2 + θ3(µ)A3.
    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving a single
    ordinary least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        * 'c' : Constant term c(µ).
        * 'A' : Linear state term A(µ)x(t).
        * 'H' : Quadratic state term H(µ)(x⊗x)(t).
        * 'G' : Cubic state term G(µ)(x⊗x⊗x)(t)
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

    has_cubic : bool
        Whether or not there is a cubic term G(µ)(x⊗x⊗x)(t).

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

    c_ : callable(µ) -> (r,) ndarray; (r,) ndarray; or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : callable(µ) -> (r,r) ndarray; (r,r) ndarray; or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    Hc_ : callable(µ) -> (r,r(r+1)//2) ndarray; (r,r(r+1)//2) ndarray; or None
        Learned ROM quadratic state matrix (compact), or None if 'H' is not
        in `modelform`. Used internally instead of the larger H_.

    H_ : callable(µ) -> (r,r**2) ndarray; (r,r**2) ndarray; or None
        Learned ROM quadratic state matrix (full size), or None if 'H' is not
        in `modelform`. Computed on the fly from Hc_ if desired; not used in
        solving the ROM.

    Gc_ : callable(µ) -> (r,r(r+1)(r+2)//6) ndarray;
          (r,r(r+1)(r+2)//6) ndarray; or None
        Learned ROM cubic state matrix (compact), or None if 'G' is not
        in `modelform`. Used internally instead of the larger G_.

    G_ : callable(µ) -> (r,r**3) ndarray; (r,r**3) ndarray; or None
        Learned ROM cubic state matrix (full size), or None if 'G' is not
        in `modelform`. Computed on the fly from Gc_ if desired; not used in
        solving the ROM.

    B_ : callable(µ) -> (r,m) ndarray; (r,m) ndarray; or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, µs, affines, Xs, Xdots, Us=None, P=0):
        """Solve for the reduced model operators via ordinary least squares,
        using solution trajectories from multiple examples.

        Parameters
        ----------
        Vr : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).

        µs : (s,) ndarray
            Parameter values at which the snapshot data is collected.

        affines : dict(str -> list(callables))
            Functions that define the structures of the affine operators.
            Keys must match the modelform:
            * 'c': Constant term c(µ).
            * 'A': Linear state matrix A(µ).
            * 'H': Quadratic state matrix H(µ).
            * 'G': Cubic state matrix G(µ).
            * 'B': Linear input matrix B(µ).
            For example, if the constant term has the affine structure
            c(µ) = θ1(µ)c1 + θ2(µ)c2 + θ3(µ)c3, then 'c' -> [θ1, θ2, θ3].

        Xs : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array, Xs[i], corresponds to the ith parameter, µs[i].

        Xdots : list of s (n,k) ndarrys (or (s,n,k) ndarray)
            Column-wise velocity training data. The ith array, Xdots[i],
            corresponds to the ith parameter, µs[i].

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
        return _AffineInferredMixin.fit(self, InferredContinuousROM,
                                        Vr, µs, affines, Xs, Xdots, Us, P)

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
