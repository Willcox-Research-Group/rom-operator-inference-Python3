# errors.py
"""Custom exception classes."""


class DimensionalityError(Exception):                       # pragma: no cover
    """Dimension of data not aligned with previous model information."""
    pass


class LoadfileFormatError(Exception):                       # pragma: no cover
    """File format inconsistent with a loading routine."""
    pass
