# __init__.py
"""Operator inference for data-driven model reduction of dynamical systems.

Authors: Renee Swischuk, Shane McQuarrie, Elizabeth Qian, Boris Kramer, et al.
"""

from ._core import (
                    select_model, trained_model_from_operators,
                    InferredDiscreteROM, InferredContinuousROM,
                    IntrusiveDiscreteROM, IntrusiveContinuousROM,
                    AffineInferredDiscreteROM, AffineInferredContinuousROM,
                    AffineIntrusiveDiscreteROM, AffineIntrusiveContinuousROM,
                    InterpolatedInferredDiscreteROM,
                    InterpolatedInferredContinuousROM,
                    )
from . import utils, pre, post


__all__ = [
            "InferredDiscreteROM", "InferredContinuousROM",
            "IntrusiveDiscreteROM" "IntrusiveContinuousROM",
            "AffineInferredDiscreteROM", "AffineInferredContinuousROM",
            "AffineIntrusiveDiscreteROM", "AffineIntrusiveContinuousROM",
            "InterpolatedInferredDiscreteROM",
            "InterpolatedInferredContinuousROM",
            "utils",
            "pre",
            "post",
          ]

__version__ = "0.6.8"
