# core/nonparametric/_frozen.py
"""Evaluations of parametric ROM classes with fit() disabled."""

__all__ = []

from ._public import SteadyOpInfROM, DiscreteOpInfROM, ContinuousOpInfROM


class _FrozenMixin:
    """Mixin for evaluations of parametric ROMs (disables fit())."""
    @property
    def data_matrix_(self):
        return None

    def fit(*args, **kwargs):
        raise NotImplementedError("fit() is disabled for this class, "
                                  "call fit() on the parametric ROM object")


class _FrozenSteadyROM(_FrozenMixin, SteadyOpInfROM):
    """Steady-state ROM that is the evaluation of a parametric ROM."""
    pass                                                    # pragma: no cover


class _FrozenDiscreteROM(_FrozenMixin, DiscreteOpInfROM):
    """Discrete-time ROM that is the evaluation of a parametric ROM."""
    pass                                                    # pragma: no cover


class _FrozenContinuousROM(_FrozenMixin, ContinuousOpInfROM):
    """Continuous-time ROM that is the evaluation of a parametric ROM."""
    pass                                                    # pragma: no cover
