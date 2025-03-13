# models/_utils.py
"""Private utility functions for working with Model classes."""

__all__ = [
    "is_continuous",
    "is_discrete",
    "is_parametric",
    "is_interpolatory",
    "is_nonparametric",
]

from .mono._nonparametric import (
    ContinuousModel,
    DiscreteModel,
    _NonparametricModel,
)
from .mono._parametric import (
    _ParametricContinuousMixin,
    _ParametricDiscreteMixin,
    _ParametricModel,
    _InterpModel,
)


def is_continuous(model):
    """``True`` if the model is time continuous (semi-discrete)."""
    return isinstance(
        model,
        (ContinuousModel, _ParametricContinuousMixin),
    )


def is_discrete(model):
    """``True`` if the model is time discrete (fully discrete)."""
    return isinstance(
        model,
        (DiscreteModel, _ParametricDiscreteMixin),
    )


def is_nonparametric(model):
    """``True`` if the model is nonparametric."""
    return isinstance(model, _NonparametricModel)


def is_parametric(model):
    """``True`` if the model is parametric."""
    return isinstance(model, _ParametricModel)


def is_interpolatory(model):
    """``True`` if the model is fully interpolatory."""
    return isinstance(model, _InterpModel)
