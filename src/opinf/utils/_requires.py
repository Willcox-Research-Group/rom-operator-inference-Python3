# utils/_requires.py
"""Wrappers for methods that require an attribute to be initialized."""

__all__ = [
    "requires",
    "requires2",
]

import functools


def requires2(attr: str, message: str) -> callable:
    """Wrapper for methods that require an attribute to be initialized.

    Parameters
    ----------
    attr : str
        Name of the required attribute.
    message : str
        Message in the error.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _decorator(self, *args, **kwargs):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise AttributeError(message)
            return func(self, *args, **kwargs)

        return _decorator

    return _wrapper


def requires(attr: str) -> callable:
    """Wrapper for methods that require an attribute to be initialized.

    Parameters
    ----------
    attr : str
        Name of the required attribute.
    """

    return requires2(attr, f"required attribute '{attr}' not set")
