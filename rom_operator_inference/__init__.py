"""Operator Inference for Data-Driven, Non-intrusive, Projection-based Model Reduction.

Authors: Renee Swischuk, Shane McQuarrie, Elizabeth Qian, Boris Kramer
"""

from ._core import InferredContinuousModel, IntrusiveContinuousModel
from . import utils, pre, post


__all__ = [
            "InferredContinuousModel",
            "IntrusiveContinuousModel",
            "utils",
            "pre",
            "post",
          ]

__version__ = "0.4.0"
