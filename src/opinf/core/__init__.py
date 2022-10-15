# _core/__init__.py
"""Classes for reduced-order models (ROMs) and the operators they consist of.

Please note that this module is private.  All public objects are available in
the main namespace - use that instead whenever possible.

Submodules
----------
* operators: classes for each type of term in polynomial reduced models.
* nonparametric: classes for ROMs without external parameter dependencies.
"""

from . import operators
from .nonparametric import *
from .interpolate import *


__all__ = nonparametric.__all__ + interpolate.__all__
