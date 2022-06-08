# core/interpolate/_public.py
"""Public classes for parametric reduced-order models where the parametric
dependence of operators are handled with elementwise interpolation, i.e,
    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).
where µ1, µ2, ... are parameter values and A1, A2, ... are the corresponding
operator matrices, e.g., A1 = A(µ1).

Relevant operator classes are defined in core.operators._interpolate.
"""

__all__ = [
    # "InterpolatedSteadyOpInfROM",
    "InterpolatedDiscreteOpInfROM",
    "InterpolatedContinuousOpInfROM",
]

from ._base import _InterpolatedOpInfROM
from ..nonparametric import (
        # SteadyOpInfROM,
        DiscreteOpInfROM,
        ContinuousOpInfROM,
)
from ..nonparametric._frozen import (
    # _FrozenSteadyROM,
    _FrozenDiscreteROM,
    _FrozenContinuousROM,
)


# class InterpolatedSteadyOpInfROM(_InterpolatedOpInfROM):
#     """Reduced-order model for a parametric steady state problem:
#
#         g = F(q; µ).
#
#     Here q is the state, µ is a free parameter, and g is a forcing term.
#     The structure of F(q; µ) is user specified (modelform), and the
#     dependence on the parameter µ is handled through interpolation.
#
#     Attributes
#     ----------
#     modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
#         Structure of the reduced-order model. Each character indicates one
#         term in the low-dimensional function F(q, u):
#         'c' : Constant term c(µ)
#         'A' : Linear state term A(µ)q.
#         'H' : Quadratic state term H(µ)[q ⊗ q].
#         'G' : Cubic state term G(µ)[q ⊗ q ⊗ q].
#         For example, modelform="AH" means F(q; µ) = A(µ)q + H(µ)[q ⊗ q].
#     n : int
#         Dimension of the high-dimensional state.
#     m : int or None
#         Dimension of the input, or None if no inputs are present.
#     p : int
#         Dimension of the parameter µ.
#     r : int
#         Dimension of the low-dimensional (reduced-order) state.
#     s : int
#         Number of training samples, i.e., the number of data points in the
#         interpolation scheme.
#     basis : (n, r) ndarray or None
#         Basis matrix defining the relationship between the high- and
#         low-dimensional state spaces. If None, arguments of fit() are assumed
#         to be in the reduced dimension.
#     c_, A_, H_ G_, B_ : Operator objects (see opinf.core.operators) or None
#         Low-dimensional operators composing the reduced-order model.
#     """
#     _ModelClass = _FrozenSteadyROM
#     _ModelFitClass = SteadyOpInfROM
#
#     def evaluate(self, parameter, state_):
#         """Evaluate the right-hand side of the model at the given parameter,
#         i.e., the F(q; µ) of
#
#             g = F(q; µ).
#
#         Parameters
#         ----------
#         parameter : (p,) ndarray or float (p = 1)
#             Parameter value at which to evaluate the model.
#         state_ : (r,) ndarray
#             Low-dimensional state vector q.
#
#         Returns
#         -------
#         g_: (r,) ndarray
#             Evaluation of the model.
#         """
#         return _InterpolatedOpInfROM.evaluate(self, parameter, state_)
#
#     def fit(self, basis, parameters, states, forcings, inputs=None,
#             regularizers=0, known_operators=None):
#         """Learn the reduced-order model operators from data.
#
#         Parameters
#         ----------
#         basis : (n, r) ndarray or None
#             Basis for the linear reduced space (e.g., POD basis matrix).
#             If None, statess and lhss are assumed to already be projected.
#         parameters : (s, p) ndarray or (s,) ndarray
#             Parameter values corresponding to the training data, either
#             s p-dimensional vectors or s scalars (parameter dimension p = 1).
#         states : list of s (n, k) or (r, k) ndarrays
#             State snapshots for each parameter value: `states[i]` corresponds
#             to `parameters[i]` and contains column-wise state data, i.e.,
#             `states[i][:, j]` is a single snapshot.
#             Data may be either full order (n rows) or reduced order (r rows).
#         forcings : list of s (n, k) or (r, k) ndarrays
#             Forcing data for ROM training corresponding to each parameter
#             value: `forcings[i]` corresponds to `parameters[i]` and
#             contains column-wise forcing data, i.e., `forcings[i][:, j]`
#             corresponds to the state snapshot `states[i][:, j]`.
#             Data may be either full order (n rows) or reduced order (r rows).
#         inputs : list of s (m, k) or (k,) ndarrays or None
#             Inputs for ROM training corresponding each parameter value:
#             `inputs[i]` corresponds to `parameters[i]` and contains
#             column-wise input data, i.e., `inputs[i][:, j]` corresponds to
#             the state snapshot `states[i][:, j]`.
#             If m = 1 (scalar input), then each `inputs[i]` may be a one-
#             dimensional array.
#             This argument is required if 'B' is in `modelform` but must be
#             None if 'B' is not in `modelform`.
#         regularizers : list of s (float >= 0, (d, d) ndarray, or r of these)
#             Tikhonov regularization factor(s) for each parameter value:
#             `regularizers[i]` is the regularization factor for the regression
#             using data corresponding to `parameters[i]`. See lstsq.solve().
#             Here, d is the number of unknowns in each decoupled least-squares
#             problem, e.g., d = r + m when `modelform`="AB".
#         known_operators : dict or None
#             Dictionary of known full-order operators at each parameter value.
#             Corresponding reduced-order operators are computed directly
#             through projection; remaining operators are inferred from data.
#             Keys must match the modelform; values are a list of s ndarrays:
#             * 'c': (n,) constant term c.
#             * 'A': (n, n) linear state matrix A.
#             * 'H': (n, n**2) quadratic state matrix H.
#             * 'G': (n, n**3) cubic state matrix G.
#             * 'B': (n, m) input matrix B.
#             If operators are known for some parameter values but not others,
#             use None whenever the operator must be inferred, e.g., for
#             parameters = [µ1, µ2, µ3, µ4, µ5], if A1, A3, and A4 are known
#             linear state operators at µ1, µ3, and µ4, respectively, set
#             known_operators = {'A': [A1, None, A3, A4, None]}.
#             For known operators (e.g., A) that do not depend on the
#             parameters,
#             known_operators = {'A': [A, A, A, A, A]} and
#             known_operators = {'A': A} are equivalent.
#
#         Returns
#         -------
#         self
#         """
#         return _InterpolatedOpInfROM.fit(self, basis, parameters,
#                                          states, forcings, inputs,
#                                          regularizers, known_operators)
#
#     def predict(self, forcing):
#         """TODO"""
#         pass


class InterpolatedDiscreteOpInfROM(_InterpolatedOpInfROM):
    """Reduced-order model for a parametric discrete dynamical system:

        q_{j+1} = F(q_{j}, u_{j}; µ),         q_{0} = q0.

    Here q is the state, u is the (optional) input, and µ is a free parameter.
    The structure of F(q, u) is user specified (modelform), and the dependence
    on the parameter µ is handled through interpolation.

    Attributes
    ----------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        Structure of the reduced-order model. Each character indicates one term
        in the low-dimensional function F(q, u; µ):
        'c' : Constant term c(µ)
        'A' : Linear state term A(µ)q.
        'H' : Quadratic state term H(µ)[q ⊗ q].
        'G' : Cubic state term G(µ)[q ⊗ q ⊗ q].
        For example, modelform="AB" means F(q, u; µ) = A(µ)q + B(µ)u.
    n : int
        Dimension of the high-dimensional state.
    m : int or None
        Dimension of the input, or None if no inputs are present.
    p : int
        Dimension of the parameter µ.
    r : int
        Dimension of the low-dimensional (reduced-order) state.
    s : int
        Number of training samples, i.e., the number of data points in the
        interpolation scheme.
    basis : (n, r) ndarray or None
        Basis matrix defining the relationship between the high- and
        low-dimensional state spaces. If None, arguments of fit() are assumed
        to be in the reduced dimension.
    c_, A_, H_ G_, B_ : Operator objects (see opinf.core.operators) or None
        Low-dimensional operators composing the reduced-order model.
    """
    _ModelClass = _FrozenDiscreteROM
    _ModelFitClass = DiscreteOpInfROM

    def evaluate(self, parameter, state_, input_=None):
        r"""Evaluate the right-hand side of the model at the given parameter,
        i.e., the F(q, u; µ) of

            q_{j+1} = F(q_{j}, u_{j}; µ).

        Parameters
        ----------
        parameter : (p,) ndarray or float (p = 1)
            Parameter value at which to evaluate the model.
        state_ : (r,) ndarray
            Low-dimensional state vector q_{j}.
        input_ : (m,) ndarray or None
            Input vector u_{j} corresponding to the state.

        Returns
        -------
        nextstate_: (r,) ndarray
            Evaluation q_{j+1} of the model.
        """
        return _InterpolatedOpInfROM.evaluate(self, parameter, state_, input_)

    def fit(self, basis, parameters, states, nextstates=None, inputs=None,
            regularizers=0, known_operators=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, statess and lhss are assumed to already be projected.
        parameters : (s, p) ndarray or (s,) ndarray
            Parameter values corresponding to the training data, either
            s p-dimensional vectors or s scalars (parameter dimension p = 1).
        states : list of s (n, k) or (r, k) ndarrays
            State snapshots for each parameter value: `states[i]` corresponds
            to `parameters[i]` and contains column-wise state data, i.e.,
            `states[i][:, j]` is a single snapshot.
            Data may be either full order (n rows) or reduced order (r rows).
        nextstates : list of s (n, k) or (r, k) ndarrays
            Column-wise snapshot training data corresponding to the next
            iteration of the state snapshots for each parameter value, i.e.,
            FOM(states[i][:, j], inputs[i][:, j]) = nextstates[i][:, j]
            where FOM is the full-order model.
            If None, assume state j+1 is the iteration after state j, i.e.,
            FOM(states[i][:, j], inputs[i][:, j]) = states[i][:, j+1].
            Data may be either full order (n rows) or reduced order (r rows).
        inputs : list of s (m, k) or (k,) ndarrays or None
            Inputs for ROM training corresponding each parameter value:
            `inputs[i]` corresponds to `parameters[i]` and contains
            column-wise input data, i.e., `inputs[i][:, j]` corresponds to
            the state snapshot `states[i][:, j]`.
            If m = 1 (scalar input), then each `inputs[i]` may be a one-
            dimensional array.
            This argument is required if 'B' is in `modelform` but must be
            None if 'B' is not in `modelform`.
        regularizers : list of s (float >= 0, (d, d) ndarray, or r of these)
            Tikhonov regularization factor(s) for each parameter value:
            `regularizers[i]` is the regularization factor for the regression
            using data corresponding to `parameters[i]`. See lstsq.solve().
            Here, d is the number of unknowns in each decoupled least-squares
            problem, e.g., d = r + m when `modelform`="AB".
        known_operators : dict or None
            Dictionary of known full-order operators at each parameter value.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are a list of s ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.
            If operators are known for some parameter values but not others,
            use None whenever the operator must be inferred, e.g., for
            parameters = [µ1, µ2, µ3, µ4, µ5], if A1, A3, and A4 are known
            linear state operators at µ1, µ3, and µ4, respectively, set
            known_operators = {'A': [A1, None, A3, A4, None]}.
            For known operators (e.g., A) that do not depend on the parameters,
            known_operators = {'A': [A, A, A, A, A]} and
            known_operators = {'A': A} are equivalent.

        Returns
        -------
        self
        """
        if nextstates is None and states is not None:
            nextstates = [Q[:, 1:] for Q in states]
            states = [Q[:, :-1] for Q in states]
        if inputs is not None:
            inputs = [ip[..., :states[i].shape[1]]
                      for i, ip in enumerate(inputs)]
        return _InterpolatedOpInfROM.fit(self, basis, parameters,
                                         states, nextstates, inputs,
                                         regularizers, known_operators)

    def predict(self, parameter, state0, niters, inputs=None,
                reconstruct=True):
        """Step forward the ROM `niters` steps at the given parameter value.

        Parameters
        ----------
        parameter : (p,) ndarray or float (p = 1)
            Parameter value at which to evaluate the model.
        state0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected to
            reduced order (r-vector).
        niters : int
            Number of times to step the system forward.
        inputs : (m, niters-1) ndarray
            Inputs for the next niters-1 time steps, i.e.,
            >>> states[:, 0] = states0
            >>> states[:, 1] = F(state0, inputs[:, 0])
            >>> states[:, 2] = F(states[:, 1], inputs[:, 1])
            ...
        reconstruct : bool
            If True and the basis is not None, reconstruct the solutions
            in the original n-dimensional state space.

        Returns
        -------
        states : (n, niters) or (r, niters) ndarray
            Approximate solution to the system, including the given
            initial condition. If the basis exists and reconstruct=True,
            return solutions in the full n-dimensional state space (n rows);
            otherwise, return reduced-order state solution (r rows).
        """
        return _InterpolatedOpInfROM.predict(self, parameter,
                                             state0, niters, inputs,
                                             reconstruct)


class InterpolatedContinuousOpInfROM(_InterpolatedOpInfROM):
    """Reduced-order model for a parametric system of ordinary differential
    equations:

        dq / dt = F(t, q(t), u(t); µ),      q(0) = q0.

    Here q(t) is the state, u(t) is the (optional) input, and µ is a free
    parameter. The structure of F(t, q(t), u(t)) is user specified (modelform),
    and the dependence on the parameter µ is handled through interpolation.

    Attributes
    ----------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        Structure of the reduced-order model. Each character indicates one term
        in the low-dimensional function F(q(t), u(t); µ):
        'c' : Constant term c(µ)
        'A' : Linear state term A(µ)q(t).
        'H' : Quadratic state term H(µ)[q(t) ⊗ q(t)].
        'G' : Cubic state term G(µ)[q(t) ⊗ q(t) ⊗ q(t)].
        For example, modelform="cA" means F(q(t), u(t); µ) = c(µ) + A(µ)q(t).
    n : int
        Dimension of the high-dimensional state.
    m : int or None
        Dimension of the input, or None if no inputs are present.
    p : int
        Dimension of the parameter µ.
    r : int
        Dimension of the low-dimensional (reduced-order) state.
    s : int
        Number of training samples, i.e., the number of data points in the
        interpolation scheme.
    basis : (n, r) ndarray or None
        Basis matrix defining the relationship between the high- and
        low-dimensional state spaces. If None, arguments of fit() are assumed
        to be in the reduced dimension.
    c_, A_, H_ G_, B_ : Operator objects (see opinf.core.operators) or None
        Low-dimensional operators composing the reduced-order model.
    """
    _ModelClass = _FrozenContinuousROM
    _ModelFitClass = ContinuousOpInfROM

    def evaluate(self, parameter, t, state_, input_func=None):
        """Evaluate the right-hand side of the model at the given parameter,
        i.e., the F(t, q(t), u(t); µ) of

            dq / dt = F(t, q(t), u(t); µ).

        Parameters
        ----------
        parameter : (p,) ndarray or float (p = 1)
            Parameter value at which to evaluate the model.
        t : float
            Time, a scalar.
        state_ : (r,) ndarray
            Reduced state vector q(t) corresponding to time `t`.
        input_func : callable(float) -> (m,)
            Input function that maps time `t` to an input vector of length m.

        Returns
        -------
        dqdt_: (r,) ndarray
            Evaluation of the model.
        """
        return _InterpolatedOpInfROM.evaluate(self, parameter,
                                              t, state_, input_func)

    def fit(self, basis, parameters, states, ddts, inputs=None,
            regularizers=0, known_operators=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, statess and lhss are assumed to already be projected.
        parameters : (s, p) ndarray or (s,) ndarray
            Parameter values corresponding to the training data, either
            s p-dimensional vectors or s scalars (parameter dimension p = 1).
        states : list of s (n, k) or (r, k) ndarrays
            State snapshots for each parameter value: `states[i]` corresponds
            to `parameters[i]` and contains column-wise state data, i.e.,
            `states[i][:, j]` is a single snapshot.
            Data may be either full order (n rows) or reduced order (r rows).
        ddts : list of s (n, k) or (r, k) ndarrays
            Time derivative data for ROM training corresponding to each
            parameter value: `ddts[i]` corresponds to `parameters[i]` and
            contains column-wise left-hand side data, i.e., `ddts[i][:, j]`
            corresponds to the state snapshot `states[i][:, j]`.
            Data may be either full order (n rows) or reduced order (r rows).
        inputs : list of s (m, k) or (k,) ndarrays or None
            Inputs for ROM training corresponding each parameter value:
            `inputs[i]` corresponds to `parameters[i]` and contains
            column-wise input data, i.e., `inputs[i][:, j]` corresponds to the
            state snapshot `states[i][:, j]`.
            If m = 1 (scalar input), then each `inputs[i]` may be a one-
            dimensional array.
            This argument is required if 'B' is in `modelform` but must be
            None if 'B' is not in `modelform`.
        regularizers : list of s (float >= 0, (d, d) ndarray, or r of these)
            Tikhonov regularization factor(s) for each parameter value:
            `regularizers[i]` is the regularization factor for the regression
            using data corresponding to `parameters[i]`. See lstsq.solve().
            Here, d is the number of unknowns in each decoupled least-squares
            problem, e.g., d = r + m when `modelform`="AB".
        known_operators : dict or None
            Dictionary of known full-order operators at each parameter value.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are a list of s ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.
            If operators are known for some parameter values but not others,
            use None whenever the operator must be inferred, e.g., for
            parameters = [µ1, µ2, µ3, µ4, µ5], if A1, A3, and A4 are known
            linear state operators at µ1, µ3, and µ4, respectively, set
            known_operators = {'A': [A1, None, A3, A4, None]}.
            For known operators (e.g., A) that do not depend on the parameters,
            known_operators = {'A': [A, A, A, A, A]} and
            known_operators = {'A': A} are equivalent.

        Returns
        -------
        self
        """
        return _InterpolatedOpInfROM.fit(self, basis,
                                         parameters, states, ddts, inputs,
                                         regularizers, known_operators)

    def predict(self, parameter, state0, t, input_func=None,
                reconstruct=True, **options):
        """Simulate the learned ROM at the given parameter value with
        scipy.integrate.solve_ivp().

        Parameters
        ----------
        parameter : (p,) ndarray or float (p = 1)
            Parameter value at which to evaluate the model.
        state0 : (n,) or (r,) ndarray
            Initial state vector, either full order (n-vector) or projected
            to reduced order (r-vector).
        t : (nt,) ndarray
            Time domain over which to integrate the reduced-order model.
        input_func : callable or (m, nt) ndarray
            Input as a function of time (preferred) or the input at the
            times `t`. If given as an array, cubic spline interpolation
            on the known data points is used as needed.
        reconstruct : bool
            If True and the basis is not None, reconstruct the solutions
            in the original n-dimensional state space.
        options
            Arguments for scipy.integrate.solve_ivp(), such as the following:
            method : str
                ODE solver for the reduced-order model.
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
                Maximimum allowed integration step size.
            See https://docs.scipy.org/doc/scipy/reference/integrate.html.

        Returns
        -------
        states : (n, nt) or (r, nt) ndarray
            Approximate solution to the system over the time domain `t`.
            If the basis exists and reconstruct=True, return solutions in the
            original n-dimensional state space (n rows); otherwise, return
            reduced-order state solutions (r rows).
            A more detailed report on the integration results is stored as
            the attribute `predict_result_`.
        """
        return _InterpolatedOpInfROM.predict(self, parameter,
                                             state0, t, input_func,
                                             reconstruct, **options)
