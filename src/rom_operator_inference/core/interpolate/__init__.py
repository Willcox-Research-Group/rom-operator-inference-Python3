# core/interpolate/__init__.py
"""Parametric reduced-order models where the parametric dependence of the
operators are handled with elementwise interpolation, i.e,
    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).
where µ1, µ2, ... are parameter values and A1, A2, ... are the corresponding
operator matrices, e.g., A1 = A(µ1).

Relevant operator classes are defined in core.operators._interpolate.

Please note that this module is private.  All public objects are available in
the main namespace - use that instead whenever possible.

Public Classes
--------------
* InterpolatedSteadyOpInfROM: ROM for steady-state problems.
* InterpolatedDiscreteOpInfROM: ROM for difference equations.
* InterpolatedContinuousOpInfROM: ROM for differential equations.

Private Classes
---------------
* _InterpolatedOpInfROM: Base class for interpolation-based OpInf ROMs.
"""

from ._public import *

__all__ = _public.__all__
