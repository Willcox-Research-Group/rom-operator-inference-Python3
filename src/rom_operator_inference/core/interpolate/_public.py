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


class InterpolatedSteadyOpInfROM(_InterpolatedOpInfROM):
    """Reduced-order model for a parametric steady state problem:

        g = F(q; µ).

    Here q is the state, µ is a free parameter, and g is a forcing term.
    The structure of F(q; µ) is user specified (modelform), and the dependence
    on the parameter µ is handled through interpolation.

    Attributes
    ----------
    modelform : str containing 'c', 'A', 'H', 'G', and/or 'B'
        Structure of the reduced-order model. Each character indicates one term
        in the low-dimensional function F(q, u):
        'c' : Constant term c(µ)
        'A' : Linear state term A(µ)q.
        'H' : Quadratic state term H(µ)[q ⊗ q].
        'G' : Cubic state term G(µ)[q ⊗ q ⊗ q].
        For example, modelform="AH" means F(q; µ) = A(µ)q + H(µ)[q ⊗ q].
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
    _ModelClass = _FrozenSteadyROM
    _ModelFitClass = SteadyOpInfROM


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
