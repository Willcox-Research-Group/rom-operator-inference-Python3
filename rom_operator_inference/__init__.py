"""Operator Inference for Data-Driven, Non-intrusive, Projection-based Model Reduction.

Authors: Renee Swischuk, Shane McQuarrie, Elizabeth Qian, Boris Kramer
"""

from ._core import (
                        InferredContinuousModel,
                        IntrusiveContinuousModel,
                        # AffineInferredContinuousModel,
                        # AffineIntrusiveContinuousModel,
                        InterpolatedInferredContinuousModel,
                    )
from . import utils, pre, post


__all__ = [
            "InferredContinuousModel",
            "IntrusiveContinuousModel",
            # "AffineInferredContinuousModel",
            # "AffineIntrusiveContinuousModel",
            "InterpolatedInferredContinuousModel",
            "utils",
            "pre",
            "post",
          ]

__version__ = "0.4.2"
