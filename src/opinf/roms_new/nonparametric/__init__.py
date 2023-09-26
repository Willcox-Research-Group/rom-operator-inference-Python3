# roms/nonparametric/__init__.py
"""Nonparametric Operator Inference reduced-order model (ROM) classes.

.. currentmodule:: opinf.roms_new.nonparametric

.. tip::
    All public objects from this module are also available in the root
    `opinf` namespace.

**Public Classes**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    DiscreteOpInfROM
    ContinuousOpInfROM

**Private Classes**

* ``_NonparametricOpInfROM``: Base class for nonparametric Operator Inference
  ROMs.
* ``_FrozenMixin``: Mixin for evaluations of parametric ROMs
  (disables ``fit()``).
* ``_FrozenDiscreteOpInfROM``: Evaluations of discrete-time parametric ROMs.
* ``_FrozenContinuousOpInfROM``: Evaluations of continuous-time parametric
  ROMs.
"""

from . import _base, _frozen
from ._public import *


__all__ = _public.__all__
