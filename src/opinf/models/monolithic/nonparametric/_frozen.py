# models/monolithic/nonparametric/_frozen.py
"""Evaluations of monolithic parametric model classes with fit() disabled."""

__all__ = []

from ._public import SteadyModel, DiscreteModel, ContinuousModel


class _FrozenMixin:
    """Mixin for evaluations of parametric models (disables fit())."""

    def _clear(self):
        raise NotImplementedError(
            "_clear() is disabled for this class, "
            "call fit() on the parametric model object"
        )

    @property
    def data_matrix_(self):
        return None

    @property
    def operator_matrix_dimension(self):
        return None

    def fit(*args, **kwargs):
        raise NotImplementedError(
            "fit() is disabled for this class, "
            "call fit() on the parametric model object"
        )


class _FrozenSteadyModel(_FrozenMixin, SteadyModel):
    """Steady-state model that is the evaluation of a parametric model."""

    pass  # pragma: no cover


class _FrozenDiscreteModel(_FrozenMixin, DiscreteModel):
    """Discrete-time model that is the evaluation of a parametric model."""

    pass  # pragma: no cover


class _FrozenContinuousModel(_FrozenMixin, ContinuousModel):
    """Continuous-time model that is the evaluation of a parametric model."""

    pass  # pragma: no cover
