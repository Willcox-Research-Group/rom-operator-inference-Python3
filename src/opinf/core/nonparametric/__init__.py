# core/nonparametric/__init__.py
"""Nonparametric Operator Inference reduced-order model (ROM) classes.

Please note that this module is private.  All public objects are available in
the main namespace - use that instead whenever possible.

Public Classes
--------------
* SteadyOpInfROM: ROM for steady-state problems.
* DiscreteOpInfROM: ROM for discrete dynamical systems (difference equations).
* ContinuousOpInfROM: ROM for continuous systems (differential equations).

Private Classes
---------------
_base.py:
* _NonparametricOpInfROM: Base class for nonparametric Operator Inference ROMs.

_frozen.py:
* _FrozenMixin: Mixin for evaluations of parametric ROMs (disables fit()).
* _FrozenSteadyOpInfROM: Evaluations of steady-state parametric ROMs.
* _FrozenDiscreteOpInfROM: Evaluations of discrete-time parametric ROMs.
* _FrozenContinuousOpInfROM: Evaluations of continuous-time parametric ROMs.
"""

from . import _base, _frozen
from ._public import *


__all__ = _public.__all__
