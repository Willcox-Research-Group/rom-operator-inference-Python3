# roms/__init__.py
"""Classes for reduced-order models (ROMs) and the operators they consist of.

.. currentmodule:: opinf.roms

.. tip::
    This module is private.  All public objects are available in the
    :ref:`main namespace <sec-main>` -- use that instead whenever possible.

**Submodules**

.. autosummary::
    :toctree: _autosummaries

    operators
    nonparametric
    interpolate
"""

from .nonparametric import *
from .interpolate import *


__all__ = nonparametric.__all__ + interpolate.__all__
