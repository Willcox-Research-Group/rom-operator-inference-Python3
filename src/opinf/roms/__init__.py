# roms/__init__.py
"""Reduced-order model classes.

.. currentmodule:: opinf.roms

.. tip::
    All public objects from this module are also available in the root
    `opinf` namespace.

**Submodules**

.. autosummary::
    :toctree: _autosummaries

    nonparametric
    interpolate
"""

from .nonparametric import *
from .interpolate import *


__all__ = nonparametric.__all__ + interpolate.__all__
