# utils/_repr.py
"""Canonical string representation for objects with a ``__str__()`` method."""

__all__ = [
    "str2repr",
]


def str2repr(obj) -> str:
    """
    Canonical string representation for objects with a ``__str__()`` method.
    """
    uniqueID = f"<{obj.__class__.__name__} object at {hex(id(obj))}>"
    return f"{uniqueID}\n{str(obj)}"
