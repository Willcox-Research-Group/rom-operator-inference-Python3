"""Operator Inference for Data-Driven, Non-intrusive, Projection-based Model Reduction.

Authors: Renee Swischuk, Shane McQuarrie, Elizabeth Qian, Boris Kramer
"""

from .core import ReducedModel
from . import utils


__all__ = ["ReducedModel", "utils"]

__version__ = "0.2.3"
