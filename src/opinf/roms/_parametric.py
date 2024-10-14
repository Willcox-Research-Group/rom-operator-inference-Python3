# roms/_parametric.py
"""Parametric ROM classes."""

__all__ = [
    "ParametricROM",
]

from ..models import _utils as modutils
from ._base import _BaseROM


class ParametricROM(_BaseROM):
    r"""Parametric reduced-order model.

    This class connects classes from the various submodules to form a complete
    reduced-order modeling workflow.

    High-dimensional data
    :math:`\to` transformed / preprocessed data
    :math:`\to` compressed data
    :math:`\to` low-dimensional model.

    Parameters
    ----------
    model : :mod:`opinf.models` object
        Parametric system model, an instance of one of the following:

        * :class:`opinf.models.ParametricContinuousModel`
        * :class:`opinf.models.ParametricDiscreteModel`
        * :class:`opinf.models.InterpContinuousModel`
        * :class:`opinf.models.InterpDiscreteModel`
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
        if not modutils.is_parametric(model):
            raise TypeError("'model' must be a parametric model instance")
        super().__init__(model, lifter, transformer, basis, ddt_estimator)

    # Training ----------------------------------------------------------------
    def fit(
        self,
        parameters,
        states,
        lhs=None,
        inputs=None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
    ):
        """Calibrate the model to training data.

        Parameters
        ----------
        parameters : list of s (floats or (p,) ndarrays)
            Parameter values for which training data are available.
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is the data corresponding to parameter value
            ``parameters[i]``; each column ``states[i][:, j]`` is one snapshot.
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
            If ``True`` (default), calibrate the high-to-low dimensional
            mapping using the ``states``.
            If ``False``, assume the basis is already calibrated.

        Returns
        -------
        self
        """
        self._fit_and_return_training_data(
            parameters=parameters,
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )
        return self

    # Evaluation --------------------------------------------------------------
    def predict(self, parameter, state0, *args, **kwargs):
        r"""Evaluate the reduced-order model.

        Arguments are the same as the ``predict()`` method of :attr:`model`.

        Parameters
        ----------
        parameter : (p,) ndarray
            Parameter value :math:`\bfmu`.
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
        states = self.model.predict(parameter, q0_, *args, **kwargs)
        return self.decode(states)
