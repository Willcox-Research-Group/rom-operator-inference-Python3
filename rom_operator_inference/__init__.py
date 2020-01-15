"""Operator inference for data-driven, non-intrusive model reduction of
dynamical systems.

Authors: Renee Swischuk, Shane McQuarrie, Elizabeth Qian, Boris Kramer
"""

from ._core import (
                    InferredDiscreteROM, InferredContinuousROM,
                    IntrusiveDiscreteROM, IntrusiveContinuousROM,
                    AffineIntrusiveContinuousROM, AffineIntrusiveDiscreteROM,
                    InterpolatedInferredDiscreteROM,
                    InterpolatedInferredContinuousROM,
                    )
from . import utils, pre, post


__all__ = [
            "InferredDiscreteROM", "InferredContinuousROM",
            "IntrusiveDiscreteROM" "IntrusiveContinuousROM",
            "AffineIntrusiveContinuousROM", # AffineIntrusiveDiscreteROM",
            "InterpolatedInferredDiscreteROM",
            "InterpolatedInferredContinuousROM",
            "utils",
            "pre",
            "post",
          ]

__version__ = "0.6.3"
