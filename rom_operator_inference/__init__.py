"""Operator Inference for Data-Driven, Non-intrusive, Projection-based Model Reduction.

Authors: Renee Swischuk, Shane McQuarrie, Elizabeth Qian, Boris Kramer
"""

from ._core import ReducedModel
from . import utils, pre, post

__all__ = ["ReducedModel", "utils", "pre", "post"]

__version__ = "0.3.2"
