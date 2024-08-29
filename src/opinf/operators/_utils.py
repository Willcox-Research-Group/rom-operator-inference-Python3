# operators/_utils.py
"""Private utility functions for working with Operator classes."""

from ._base import has_inputs, is_nonparametric, is_parametric, is_uncalibrated
from ._affine import is_affine, nonparametric_to_affine
from ._interpolate import is_interpolated, nonparametric_to_interpolated

__all__ = [
    "has_inputs",
    "is_nonparametric",
    "is_parametric",
    "is_uncalibrated",
    "is_affine",
    "is_interpolated",
    "nonparametric_to_affine",
    "nonparametric_to_interpolated",
]
