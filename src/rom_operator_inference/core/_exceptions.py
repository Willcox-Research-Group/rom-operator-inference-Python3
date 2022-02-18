# core/_exceptions.py
"""Custom exceptions for reduced-order model classes."""

__all__ = [
    "DimensionalityError",
]


class DimensionalityError(Exception):
    """Dimension of data not aligned with previous model information."""
    pass
