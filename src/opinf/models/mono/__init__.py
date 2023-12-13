# models/mono/__init__.py
"""Monolithic dynamical systems models."""

from ._nonparametric import *
from ._parametric import *

__all__ = _nonparametric.__all__ + _parametric.__all__
