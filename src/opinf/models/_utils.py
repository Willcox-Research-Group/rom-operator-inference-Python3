# models/_utils.py
"""Private utility functions for working with Model classes."""

__all__ = [
    "is_continuous",
    "is_discrete",
]

from .mono._nonparametric import ContinuousModel, DiscreteModel
from .mono._parametric import (
    _ParametricContinuousMixin,
    _ParametricDiscreteMixin,
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
