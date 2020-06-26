# __init__.py
"""Operator inference for data-driven model reduction of dynamical systems.

Author: Renee C. Swischuk et al.
Maintainer: Shane A. McQuarrie
Github: https://github.com/swischuk/rom-operator-inference-Python3
"""

from ._core import (
                    select_model, load_model,
                    trained_model_from_operators,
                    InferredDiscreteROM,
                    InferredContinuousROM,
                    IntrusiveDiscreteROM,
                    IntrusiveContinuousROM,
                    AffineIntrusiveDiscreteROM,
                    AffineIntrusiveContinuousROM,
                    InterpolatedInferredDiscreteROM,
                    InterpolatedInferredContinuousROM,
                    )
from . import utils, pre, post


__all__ = [
            "select_model", "load_model",
            "trained_model_from_operators",
            "InferredDiscreteROM",
            "InferredContinuousROM",
            "IntrusiveDiscreteROM"
            "IntrusiveContinuousROM",
            "AffineIntrusiveDiscreteROM",
            "AffineIntrusiveContinuousROM",
            "InterpolatedInferredDiscreteROM",
            "InterpolatedInferredContinuousROM",
            "utils",
            "pre",
            "post",
          ]

__version__ = "1.1.0"
