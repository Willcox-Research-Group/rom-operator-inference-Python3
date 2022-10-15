# lstsq/_base.py
"""Base class for solvers for the Operator Inference regression problem."""

import abc


class _BaseSolver(abc.ABC):
    """Base class for solvers for the Operator Inference regression problem."""

    @abc.abstractmethod
    def fit(*args, **kwargs):                               # pragma: no cover
        """Initialize the learning problem."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(*args, **kwargs):                           # pragma: no cover
        """Solver the learning problem under the given conditions."""
        raise NotImplementedError
