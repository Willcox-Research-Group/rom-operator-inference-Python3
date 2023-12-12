# models/monolithic/parametric/_public.py
"""Public monolithic parametric model classes."""

__all__ = [
    "ParametricSteadyModel",
    "ParametricDiscreteModel",
    "ParametricContinuousModel",
]

from ._base import _ParametricMonolithicModel
from ..nonparametric._frozen import (
    _FrozenSteadyModel,
    _FrozenDiscreteModel,
    _FrozenContinuousModel,
)


class ParametricSteadyModel(_ParametricMonolithicModel):
    """Parametric steady models."""

    _ModelClass = _FrozenSteadyModel


class ParametricDiscreteModel(_ParametricMonolithicModel):
    """Parametric time-discrete models."""

    _ModelClass = _FrozenDiscreteModel


class ParametricContinuousModel(_ParametricMonolithicModel):
    """Parametric continuous models."""

    _ModelClass = _FrozenContinuousModel
