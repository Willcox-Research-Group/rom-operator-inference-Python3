# errors.py
"""Custom exception classes."""


class DimensionalityError(ValueError):  # pragma: no cover
    """Dimension of data not aligned with previous model information."""

    pass


class LoadfileFormatError(Exception):  # pragma: no cover
    """File format inconsistent with a loading routine."""

    pass


class VerificationError(RuntimeError):  # pragma: no cover
    """Implementation of a template fails to meet requriements."""

    pass


class OpInfWarning(UserWarning):  # pragma: no cover
    """Generic warning for package usage."""

    pass
