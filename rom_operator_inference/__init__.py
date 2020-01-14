"""Operator inference for data-driven, non-intrusive model reduction of
dynamical systems.

Authors: Renee Swischuk, Shane McQuarrie, Elizabeth Qian, Boris Kramer
"""

from ._core import (
                        InferredContinuousROM,
                        IntrusiveContinuousROM,
                        AffineIntrusiveContinuousROM,
                        InterpolatedInferredContinuousROM,
                    )
from . import utils, pre, post


__all__ = [
            "InferredContinuousROM",
            "IntrusiveContinuousROM",
            "AffineIntrusiveContinuousROM",
            "InterpolatedInferredContinuousROM",
            "utils",
            "pre",
            "post",
          ]

__version__ = "0.6.2"
