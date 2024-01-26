# utils/_requires.py
"""Wrapper for methods that require an attribute to be initialized."""

__all__ = [
    "requires",
]

import functools


def requires(attr: str) -> callable:
    """Wrapper for methods that require an attribute to be initialized.

    Parameters
    ----------
    attr : str
        Name of the required attribute.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _decorator(self, *args, **kwargs):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise AttributeError(f"{attr} not set")
            return func(self, *args, **kwargs)

        return _decorator

    return _wrapper
