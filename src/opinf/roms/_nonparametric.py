# roms/_nonparametric.py
"""Nonparametric ROM class."""

__all__ = [
    "ROM",
]

import warnings

from ._base import _BaseROM
from .. import errors, models


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
    model : opinf.models.ContinuousModel or opinf.models.DiscreteModel
        System model.
    lifter : opinf.lift.LifterTemplate or None
        Lifting transformation.
    transformer : opinf.pre.TransformerTemplate or None
        Preprocesser.
    basis : opinf.basis.BasisTemplate
        Dimensionality reducer.
    ddt_estimator : opinf.ddt.DerivativeEstimatorTemplate
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
        # Verify model.
        if not isinstance(
            model,
            (models.ContinuousModel, models.DiscreteModel),
        ):
            raise TypeError(
                "'model' must be a "
                "models.ContinuousModel or models.DiscreteModel instance"
            )

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
        inplace: bool = False,
        fit_transformer: bool = True,
        fit_basis: bool = True,
    ):
        """Calibrate the model to the data.

        Parameters
        ----------
        states : (n, k) ndarray
            State snapshots in the original state space.
        lhs : (n, k) ndarray or None
            Left-hand side regression data.

            - If the model is time continuous, these are the time derivatives
              of the state snapshots.
            - If the model is fully discrete, these are the "next states"
              corresponding to the state snapshots.
        inplace : bool
            If ``True``, modify the ``states`` and ``lhs`` in-place in the
            preprocessing transformation (if applicable).
        fit_transformer : bool
            If ``True`` (default), calibrate the high-to-low dimensional
            mapping using the ``states``.
            If ``False``, assume the transformer is already calibrated.
        fit_basis : bool
            If ``True``, calibrate the high-to-low dimensional mapping
            using the ``states``.
            If ``False``, assume the basis is already calibrated.

        Returns
        -------
        self
        """

        # Express the states and the LHS in the latent state space.
        reduced = self.encode(
            states,
            lhs=lhs,
            inplace=inplace,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )
        if lhs is None:
            states = reduced
        else:
            states, lhs = reduced

        # If needed, estimate time derivatives.
        if self.iscontinuous:
            if lhs is None:
                if self.ddt_estimator is None:
                    raise ValueError(
                        "ddt_estimator required for time-continuous model "
                        "and lhs=None"
                    )
                estimated = self.ddt_estimator.estimate(states, inputs)
                if inputs is None:
                    states, lhs = estimated
                else:
                    states, lhs, inputs = estimated
            elif self.ddt_estimator is not None:
                warnings.warn(
                    "using provided time derivatives, ignoring ddt_estimator",
                    errors.OpInfWarning,
                )

        # Calibrate the model.
        kwargs = dict(inputs=inputs)
        if self.iscontinuous:
            self.model.fit(states, lhs, **kwargs)
        else:
            if lhs is not None:
                kwargs["nextstates"] = lhs
            self.model.fit(states, **kwargs)

        return self

    def predict(self, state0, *args, **kwargs):
        """Evaluate the reduced-order model.

        Parameters are the same as the model's ``predict()`` method.

        Parameters
        ----------
        state0 : (n,) ndarray
            Initial state, expressed in the original state space.
        *args : list
            Other positional arguments to ``model.predict()``.
        **kwargs : dict
            Keyword arguments to ``model.predict()``.

        Returns
        -------
        states: (n, k) ndarray
            Solution to the model, expressed in the original state space.
        """
        q0_ = self.encode(state0, fit_transformer=False, fit_basis=False)
        states = self.model.predict(q0_, *args, **kwargs)
        return self.decode(states)
