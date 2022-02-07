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
import scipy.linalg as la

from ._base import AffineOperator, _AffineMixin
from .._base import _ContinuousROM, _DiscreteROM
from .._inferred import _InferredMixin
from ...utils import kron2c, kron3c
from ... import lstsq


# Affine inferred mixin (private) =============================================
class _AffineInferredMixin(_InferredMixin, _AffineMixin):
    """Mixin class for affinely parametric inferred reduced model classes."""

    # Validation --------------------------------------------------------------
    def _check_affines(self, affines, params):
        """Check the affines argument, including checking for rank deficiency.
        """
        # Ensure there are not surplus keys.
        self._check_affines_keys(affines)

        # Make sure the affine functions are scalar-valued functions.
        for θs in affines.values():
            AffineOperator.validate_coeffs(θs, params[0])

        # Check for rank deficiencies in the data matrix.
        for key, θs in affines.items():
            Theta = np.array([[θ(µ) for θ in θs] for µ in params])
            rank = np.linalg.matrix_rank(Theta)
            if rank < Theta.shape[1]:
                np.warnings.warn(f"rank-deficient data matrix due to '{key}' "
                                 "affine structure and parameter samples",
                                 la.LinAlgWarning, stacklevel=2)

    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(self, basis, params,
                               affines, states, rhss, inputs):
        """Do sanity checks, extract dimensions, check and fix data sizes, and
        get projected data for the Operator Inference least-squares problem.

        Parameters
        ----------
        basis : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
        params : list of s scalars or (p,) ndarrays
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
        states : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array states[i] corresponds to the ith parameter params[i].
        rhss : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise next-iteration (discrete model) or time derivative
            (continuous model) training data. The ith array, rhss[i],
            corresponds to the ith parameter params[i].
        inputs : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then inputs may be one-dimensional arrays. Required if 'B'
            is in `modelform`; must be None if 'B' is not in `modelform`.

        Returns
        -------
        states_ : list of s (r,k) ndarrays
            Projected state snapshots; states_[i] corresponds to params[i].
        rhss_ : list of s (r,k) ndarrays
            Projected right-hand-side data; rhss_[i] corresponds to params[i].
        """
        # Check affines expansions, and inputs.
        # TODO: self.p = self._check_params(params):
        #       extract self.p and check for consistent sizes.
        self._check_affines(affines, params)
        self._check_inputargs(inputs, 'inputs')
        self._clear()

        # Check that the number of params matches the number of training sets.
        s = len(params)
        to_check = [(states, "states"), (rhss, self._RHS_LABEL)]
        if self.has_inputs:
            self.m = 1 if inputs[0].ndim == 1 else inputs[0].shape[0]
            to_check.append((inputs, "inputs"))
        else:
            inputs = [None]*s
        self._check_number_of_training_datasets(s, to_check)

        # Store basis and reduced dimension.
        self.basis = basis
        if basis is None:
            self.r = states[0].shape[0]

        # Ensure training data sets have consistent sizes.
        for i in range(s):
            to_check = [(states[i], "states"), (rhss[i], self._RHS_LABEL)]
            if self.has_inputs:
                to_check.append((inputs[i], "inputs"))
            self._check_training_data_shapes(to_check)

        # Project states and rhs to the reduced subspace (if not done already).
        states_ = [self.project(X, f"states[{i}]")
                   for i,X in enumerate(states)]
        rhss_ = [self.project(rhs, f"{self._RHS_LABEL}[{i}]")
                 for i, rhs in enumerate(rhss)]

        return states_, rhss_

    def _assemble_data_matrix(self, params, affines, states_, inputs):
        """Construct the Operator Inference data matrix D from projected data.

        Parameters
        ----------
        params : list of s scalars or (p,) ndarrays
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

        states_ : list of s (r,k_i) ndarrays
            Column-wise snapshot projected training data. The ith array,
            states_[i], corresponds to the ith parameter, params[i].

        inputs : list of s (m,k_i) or (k_i,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), may be a list of one-dimensional arrays.

        Returns
        -------
        D : (sum(k_i),d(r,m)) ndarray
            Operator Inference data matrix (no regularization).
        """
        D_rows = []
        for i,(µ,X_) in enumerate(zip(params, states_)):
            row = []
            if self.has_constant:       # Constant term.
                ones = np.ones((X_.shape[1],1))
                if 'c' in affines:
                    row += [θ(µ) * ones for θ in affines['c']]
                else:
                    row.append(ones)

            if self.has_linear:         # Linear state term.
                if 'A' in affines:
                    row += [θ(µ) * X_.T for θ in affines['A']]
                else:
                    row.append(X_.T)

            if self.has_quadratic:      # (compact) Quadratic state term.
                X2_ = kron2c(X_)
                if 'H' in affines:
                    row += [θ(µ) * X2_.T for θ in affines['H']]
                else:
                    row.append(X2_.T)

            if self.has_cubic:          # (compact) Cubic state term.
                X3_ = kron3c(X_)
                if 'G' in affines:
                    row += [θ(µ) * X3_.T for θ in affines['G']]
                else:
                    row.append(X3_.T)

            if self.has_inputs:         # Linear input term.
                U = np.atleast_2d(inputs[i])
                if 'B' in affines:
                    row += [θ(µ) * U.T for θ in affines['B']]
                else:
                    row.append(U.T)

            D_rows.append(np.hstack(row))

        return np.vstack(D_rows)

    def _extract_operators(self, affines, Ohat):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squares problem, constructing AffineOperators
        as indicated by the affine structure.

        Parameters
        ----------
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

        Ohat : (r,d(r,m)) ndarray
            Block matrix of ROM operator coefficients, the transpose of the
            solution to the Operator Inference linear least-squares problem.
        """
        i = 0
        if self.has_constant:           # Constant term (one-dimensional).
            if 'c' in affines:
                cs_ = []
                for j in range(len(affines['c'])):
                    cs_.append(Ohat[:,i:i+1][:,0])
                    i += 1
                self.c_ = AffineOperator(affines['c'], cs_)
            else:
                self.c_ = Ohat[:,i:i+1][:,0]
                i += 1

        if self.has_linear:             # Linear state term.
            if 'A' in affines:
                As_ = []
                for j in range(len(affines['A'])):
                    As_.append(Ohat[:,i:i+self.r])
                    i += self.r
                self.A_ = AffineOperator(affines['A'], As_)
            else:
                self.A_ = Ohat[:,i:i+self.r]
                i += self.r

        if self.has_quadratic:          # (compact) Quadratic state term.
            _r2 = self.r * (self.r + 1) // 2
            if 'H' in affines:
                Hcs_ = []
                for j in range(len(affines['H'])):
                    Hcs_.append(Ohat[:,i:i+_r2])
                    i += _r2
                self.H_ = AffineOperator(affines['H'], Hcs_)
            else:
                self.H_ = Ohat[:,i:i+_r2]
                i += _r2

        if self.has_cubic:              # (compact) Cubic state term.
            _r3 = self.r * (self.r + 1) * (self.r + 2) // 6
            if 'G' in affines:
                Gcs_ = []
                for j in range(len(affines['G'])):
                    Gcs_.append(Ohat[:,i:i+_r3])
                    i += _r3
                self.G_ = AffineOperator(affines['G'], Gcs_)
            else:
                self.G_ = Ohat[:,i:i+_r3]
                i += _r3

        if self.has_inputs:             # Linear input term.
            if 'B' in affines:
                Bs_ = []
                for j in range(len(affines['B'])):
                    Bs_.append(Ohat[:,i:i+self.m])
                    i += self.m
                self.B_ = AffineOperator(affines['B'], Bs_)
            else:
                self.B_ = Ohat[:,i:i+self.m]
                i += self.m

    def _construct_solver(self, basis, params, affines,
                          states, rhss, inputs, regularizer):
        """Construct a solver object mapping the regularizer to solutions
        of the Operator Inference least-squares problem.

        Parameters
        ----------
        basis : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and rhs are assumed to already be projected (r,k).
        TODO
        """
        states_, rhss_ = self._process_fit_arguments(basis, params, affines,
                                                     states, rhss, inputs)
        D = self._assemble_data_matrix(params, affines, states_, inputs)
        self.solver_ = lstsq.solver(D, np.hstack(rhss_).T, regularizer)

    def _evaluate_solver(self, affines, regularizer):
        """Evaluate the least-squares solver with the given regularizer.

        Parameters
        ----------
        regularizer : float >= 0 or (d,d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".
        """
        OhatT = self.solver_.predict(regularizer)
        self._extract_operators(affines, OhatT.T)

    def fit(self, basis, params, affines,
            states, rhss, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares.
        For terms with affine structure, solve for the component operators.

        Parameters
        ----------
        basis : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
        params : list of s scalars or (p,) ndarrays
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
        states : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array states[i] corresponds to the ith parameter params[i].
        rhss : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise next-iteration (discrete model) or time derivative
            (continuous model) training data. The ith array, rhss[i],
            corresponds to the ith parameter params[i].
        inputs : list of s (m,k) or (k,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : (d,d) ndarray or float
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
        self._construct_solver(basis, params, affines,
                               states, rhss, inputs, regularizer)
        self._evaluate_solver(affines, regularizer)
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
    """
    def fit(self, basis, params, affines,
            states, nextstates=None, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares,
        using solution trajectories from multiple examples.

        Parameters
        ----------
        basis : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
        params : (s,) ndarray
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
        states : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array states[i] corresponds to the ith parameter params[i].
        nextstates : list of s (n,k) or (r,k) ndarrays OR None
            Column-wise snapshot training data corresponding to the next
            iteration of the state snapshots, i.e.,
            F(states[i][:,j]) = nextstates[i][:,j] where F is the full-order
            model. If None, assume state j+1 is the iteration after state j,
            i.e., F(states[i][:,j]) = states[i][:,j+1].
        inputs : list of s (m,k-1) or (k-1,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : (d,d) ndarray or float
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
        # Process default nextstates=None.
        s = len(params)
        if nextstates is None:
            nextstates = [None]*s
        if self.has_inputs and isinstance(inputs, np.ndarray):
            inputs = list(inputs)
        for i in range(s):
            if nextstates[i] is None:
                nextstates[i] = states[i][:,1:]
                states[i] = states[i][:,:-1]
            if self.has_inputs and inputs is not None:
                inputs[i] = inputs[i][...,:states[i].shape[1]]

        return _AffineInferredMixin.fit(self, basis, params, affines,
                                        states, nextstates, inputs,
                                        regularizer)

    def predict(self, µ, x0, niters, inputs=None):
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
        inputs : (m,niters-1) ndarray
            Inputs for the next niters-1 time steps.

        Returns
        -------
        X_ROM: (n,niters) ndarray
            The reduced-order solutions to the full-order system, including
            the (projected) given initial condition.
        """
        return self(µ).predict(x0, niters, inputs)


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
    """
    def fit(self, basis, params, affines,
            states, ddts, inputs=None, regularizer=0):
        """Solve for the reduced model operators via ordinary least squares,
        using solution trajectories from multiple examples.

        Parameters
        ----------
        basis : (n,r) ndarray
            The basis for the linear reduced space (e.g., POD basis matrix).
        params : (s,) ndarray
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
        states : list of s (n,k) ndarrays (or (s,n,k) ndarray)
            Column-wise snapshot training data (each column is a snapshot).
            The ith array states[i] corresponds to the ith parameter params[i].
        ddts : list of s (n,k) ndarrys (or (s,n,k) ndarray)
            Column-wise time derivative training data. The ith array, ddts[i],
            corresponds to the ith parameter params[i].
        inputs : list of s (m,k-1) or (k-1,) ndarrays or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), may be one-dimensional arrays. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : (d,d) ndarray or float
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
        return _AffineInferredMixin.fit(self, basis, params, affines,
                                        states, ddts, inputs, regularizer)

    def predict(self, µ, x0, t, input_func=None, **options):
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
        input_func : callable or (m,nt) ndarray
            Input as a function of time (preferred) or the input at the
            times `t`. If given as an array, approximated by a cubic
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
        X_ROM : (n,nt) ndarray
            The reduced-order approximation to the full-order system over `t`.
        """
        model = self(µ)
        out = model.predict(x0, t, input_func, **options)
        self.sol_ = model.sol_
        return out
