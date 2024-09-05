# roms/_nonparametric.py
"""Nonparametric ROM class."""

__all__ = [
    "ROM",
]

import numpy as np

from ..models import _utils as modutils
from ._base import _BaseROM


class ROM(_BaseROM):
    r"""Nonparametric reduced-order model.

    This class connects classes from the various submodules to form a complete
    reduced-order modeling workflow.

    High-dimensional data
    :math:`\to` transformed / preprocessed data
    :math:`\to` compressed data
    :math:`\to` low-dimensional model.

    Parameters
    ----------
    model : :mod:`opinf.models` object
        Nonparametric system model, an instance of one of the following:

        * :class:`opinf.models.ContinuousModel`
        * :class:`opinf.models.DiscreteModel`
    lifter : :mod:`opinf.lift` object or None
        Lifting transformation.
    transformer : :mod:`opinf.pre` object or None
        Preprocesser.
    basis : :mod:`opinf.basis` object or None
        Dimensionality reducer.
    ddt_estimator : :mod:`opinf.ddt` object or None
        Time derivative estimator.
        Ignored if ``model`` is not time continuous.
    """

    def __init__(
        self,
        model,
        *,
        lifter=None,
        transformer=None,
        basis=None,
        ddt_estimator=None,
    ):
        """Store each argument as an attribute."""
        if not modutils.is_nonparametric(model):
            raise TypeError("'model' must be a nonparametric model instance")
        super().__init__(model, lifter, transformer, basis, ddt_estimator)

    def __str__(self):
        """String representation."""
        return f"Nonparametric {_BaseROM.__str__(self)}"

    # Training and evaluation -------------------------------------------------
    def fit(
        self,
        states,
        lhs=None,
        inputs=None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
    ):
        """Calibrate the model to training data.

        Parameters
        ----------
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data corresponding to a different trajectory;
            each column ``states[i][:, j]`` is one snapshot.
            If there is only one trajectory of training data (s = 1),
            ``states`` may be an (n, k) ndarray. In this case, it is assumed
            that ``lhs`` and ``inputs`` (if given) are arrays, not a sequence
            of arrays.
        lhs : list of s (n, k_i) ndarrays or None
            Left-hand side regression data. Each array ``lhs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``lhs[i][:, j]`` corresponds to the snapshot ``states[i][:, j]``.

            - If the model is time continuous, these are the time derivatives
              of the state snapshots.
            - If the model is fully discrete, these are the "next states"
              corresponding to the state snapshots.

            If ``None``, these are estimated using :attr:`ddt_estimator`
            (time continuous) or extracted from ``states`` (fully discrete).
        inputs : list of s (m, k_i) ndarrays or None
            Input training data. Each array ``inputs[i]`` is the data
            corresponding to parameter value ``parameters[i]``; each column
            ``inputs[i][:, j]`` corresponds to the snapshot ``states[:, j]``.
            May be a two-dimensional array if :math:`m=1` (scalar input).
            Only required if one or more model operators depend on inputs.
        fit_transformer : bool
            If ``True`` (default), calibrate the preprocessing transformation
            using the ``states``. If ``False``, assume the transformer is
            already calibrated.
        fit_basis : bool
            If ``True``, calibrate the high-to-low dimensional mapping
            using the ``states``.
            If ``False``, assume the basis is already calibrated.

        Returns
        -------
        self
        """
        # Single trajectory case.
        if states[0].ndim == 1:
            states = [states]
            if lhs is not None:
                lhs = [lhs]
            if inputs is not None:
                inputs = [inputs]

        states, lhs, inputs = _BaseROM.fit(
            self,
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )

        # Concatentate trajectories.
        if inputs is not None:
            inputs = np.hstack(inputs)
        self.model.fit(np.hstack(states), np.hstack(lhs), inputs)

        return self

    def predict(self, state0, *args, **kwargs):
        """Evaluate the reduced-order model.

        Arguments are the same as the ``predict()`` method of :attr:`model`.

        Parameters
        ----------
        state0 : (n,) ndarray
            Initial state, expressed in the original state space.
        *args : list
            Other positional arguments to the ``predict()`` method of
            :attr:`model`.
        **kwargs : dict
            Keyword arguments to the ``predict()`` method of :attr:`model`.

        Returns
        -------
        states: (n, k) ndarray
            Solution to the model, expressed in the original state space.
        """
        q0_ = self.encode(state0, fit_transformer=False, fit_basis=False)
        states = self.model.predict(q0_, *args, **kwargs)
        return self.decode(states)
