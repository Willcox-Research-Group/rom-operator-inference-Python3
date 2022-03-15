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
from .._nonparametric import (
        SteadyOpInfROM,
        DiscreteOpInfROM,
        ContinuousOpInfROM,
)
from .._nonparametric._frozen import (
    _FrozenSteadyROM,
    _FrozenDiscreteROM,
    _FrozenContinuousROM,
)


class InterpolatedSteadyOpInfROM(_InterpolatedOpInfROM):
    """TODO"""
    _ModelClass = _FrozenSteadyROM
    _ModelFitClass = SteadyOpInfROM
    pass


class InterpolatedDiscreteOpInfROM(_InterpolatedOpInfROM):
    """TODO"""
    _ModelClass = _FrozenDiscreteROM
    _ModelFitClass = DiscreteOpInfROM
    pass


class InterpolatedContinuousOpInfROM(_InterpolatedOpInfROM):
    """TODO"""
    _ModelClass = _FrozenContinuousROM
    _ModelFitClass = ContinuousOpInfROM
    pass
