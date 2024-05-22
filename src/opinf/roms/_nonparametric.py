# roms/_nonparametric.py
"""Nonparametric ROM class."""

__all__ = [
    "ROM",
]

from .. import models, utils


class ROM:
    """Nonparametric reduced-order model class.

    This class connects classes from the various submodules to form a complete
    reduced-order modeling workflow.

    High-dimensional data -> transformed / preprocessed data -> compressed data
    -> low-dimensional model.

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
    solver : opinf.lstsq._BaseSolver or None
        Least-squares solver.
    """

    def __init__(
        self,
        model,
        *,
        lifter=None,
        transformer=None,
        basis=None,
        ddt_estimator=None,
        solver=None,
    ):
        """Store each argument as an attribute."""
        # TODO: verify each argument here.
        self.__model = model
        self.__lifter = lifter
        self.__transformer = transformer
        self.__basis = basis
        self.__ddter = ddt_estimator
        self.__solver = solver

    # Properties --------------------------------------------------------------
    @property
    def lifter(self):
        """Lifting transformation."""
        return self.__lifter

    @property
    def transformer(self):
        """Preprocesser."""
        return self.__transformer

    @property
    def basis(self):
        """Dimensionality reducer."""
        return self.__basis

    @property
    def ddt_estimator(self):
        """Time derivative estimator."""
        return self.__ddter

    @property
    def model(self):
        """System model."""
        return self.__model

    @property
    def iscontinuous(self):
        """``True`` if the model is time continuous (semi-discrete),
        ``False`` if the model if fully discrete.
        """
        return isinstance(self.model, models.ContinuousModel)

    @property
    def solver(self):
        """Least-squares solver."""
        return self.__solver

    # Printing ----------------------------------------------------------------
    def __str__(self):
        """String representation."""
        lines = ["Nonparametric reduced-order model"]

        def indent(text):
            return "\n".join("\t" + line for line in text.rstrip().split("\n"))

        for label, obj in [
            ("Lifting", self.lifter),
            ("Transformer", self.transformer),
            ("Basis", self.basis),
            ("Time derivative estimator", self.ddt_func),
            ("Model", self.model),
            ("Solver", self.solver),
        ]:
            if obj is not None:
                lines.append(f"{label}:")
                lines.append(indent(str(obj)))

        return "\n".join(lines)

    def __repr__(self):
        """Repr: address + string representatation."""
        return utils.str2repr(self)

    # Mappings between original and latent state spaces -----------------------
    def encode(self, states, lhs=None, inplace: bool = False):
        """Map high-dimensional data to its low-dimensional representation.

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

        Returns
        -------
        states_encoded : (r, k) ndarray
            Low-dimensional representation of ``states``
            in the latent reduced state space.
        lhs_compressed : (r, k) ndarray
            Low-dimensional representation of ``lhs``
            in the latent reduced state space.
            **Only returned** if ``lhs`` is not ``None``.
        """
        # Lifting.
        if self.lifter is not None:
            if self.iscontinuous and lhs is not None:
                lhs = self.lifter.lift_ddts(lhs)
            states = self.lifter.lift(states)

        # Preprocessing.
        if self.transformer is not None:
            states = self.transformer.fit_transform(states, inplace=inplace)
            if self.iscontinuous and lhs is not None:
                lhs = self.transformer.transform_ddts(lhs, inplace=inplace)

        # Dimensionality reduction.
        if self.basis is not None:
            states = self.basis.fit(states).compress(states)
            if lhs is not None:
                lhs = self.basis.compress(lhs)

        if lhs is not None:
            return lhs
        return states, lhs

    def decode(self, states_encoded):
        """Map low-dimensional data to the original state space.

        Parameters
        ----------
        states_encoded : (r, ...) ndarray
            Low-dimensional state or states
            in the latent reduced state space.

        Returns
        -------
        states_decoded : (n, ...) ndarray
            Version of ``states_compressed`` in the original state space.
        """
        # Reverse dimensionality reduction.
        states = states_encoded
        if self.basis is not None:
            states = self.basis.decompress(states)

        # Reverse preprocessing.
        if self.transformer is not None:
            states = self.transformer.inverse_transform(states, inplace=True)

        # Reverse lifting.
        if self.lifter is not None:
            states = self.lifter.unlift(states)

        return states

    def project(self, states):
        """Project a high-dimensional state vector to the subset of the
        high-dimensional space that can be represented by the basis.

        This is done by

        1. expressing the state in low-dimensional latent coordinates, then
        2. reconstructing the high-dimensional state corresponding to those
           coordinates.

        In other words, ``project(Q)`` is equivalent to ``decode(encode(Q))``.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional state vectors, or a single state vector.

        Returns
        -------
        state_projected : (n, ...) ndarray
            Matrix of `n`-dimensional projected state vectors, or a single
            projected state vector.
        """
        return self.decode(self.encode(states))

    # Training ----------------------------------------------------------------
    def fit(self, states, lhs=None, inputs=None, inplace=False):
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

        Returns
        -------
        self
        """

        # Express the states and the LHS in the latent state space.
        reduced = self.encode(states, lhs=lhs, inplace=inplace)
        if lhs is None:
            states = reduced
        else:
            states, lhs = reduced

        # If needed, estimate time derivatives.
        if lhs is None and self.iscontinuous:
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

        # Calibrate the model.
        kwargs = dict(inputs=inputs, solver=self.solver)
        if self.iscontinuous:
            self.model.fit(states, lhs, **kwargs)
        else:
            if lhs is not None:
                kwargs["nextstates"] = lhs
            self.model.fit(states, **kwargs)

        return self

    # Evaluation --------------------------------------------------------------
    def predict(self, *args, **kwargs):
        """Evaluate the reduced-order model.

        Parameters are the same as the model's ``predict()`` method.

        Returns
        -------
        states
            (n, k) ndarray
            Solution to the model, expressed in the original state space.
        """
        states = self.model.predict(*args, **kwargs)
        return self.decode(states)
