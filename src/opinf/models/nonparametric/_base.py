# models/nonparametric/_base.py
"""Base class for nonparametric models."""

__all__ = []

import numpy as np

from .._base import _MonolithicModel
from ... import lstsq
from ... import errors
from ... import operators_new as _operators
from ...utils import hdf5_savehandle, hdf5_loadhandle


class _NonparametricModel(_MonolithicModel):
    """Base nonparametric monolithic operator inference model."""

    # Properties --------------------------------------------------------------
    @property
    def operator_matrix_(self):
        r""":math:`r \times d(r, m)` Operator matrix, e.g.,
        :math:`\Ohat = [~\chat~~\Ahat~~\Hhat~~\Bhat~]`.

        This matrix **does not** includes the entries of any operators whose
        entries are known *a priori*.
        """
        self._check_is_trained()
        return np.column_stack(
            [
                self.operators[i].entries
                for i in self._indices_of_operators_to_infer
            ]
        )

    @property
    def data_matrix_(self):
        r""":math:`k \times d(r, m)` Data matrix, e.g.,
        :math:`\D = [~
        \mathbf{1}~~
        \widehat{\Q}\trp~~
        (\widehat{\Q}\odot\widehat{\Q})\trp~~
        \U\trp~]`.
        """
        if hasattr(self, "solver_"):
            return self.solver_.A if (self.solver_ is not None) else None
        raise AttributeError("data matrix not constructed (call fit())")

    @property
    def operator_matrix_dimension(self):
        r"""Number of columns :math:`d(r, m)` of the operator matrix
        :math:`\Ohat` and the data matrix :math:`\D`,
        i.e., the number of unknowns in the Operator Inference regression
        problem for each system mode.
        Always ``None`` if ``state_dimension`` or ``input_dimension``
        are not set.
        """
        if self.state_dimension is None or (
            self._has_inputs and self.input_dimension is None
        ):
            return None
        return sum(
            self.operators[i].column_dimension(
                self.state_dimension, self.input_dimension
            )
            for i in self._indices_of_operators_to_infer
        )

    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(self, states, lhs, inputs, solver=None):
        """Prepare training data for Operator Inference by extracting
        dimensions, projecting known operators, and validating data sizes.
        """
        # Clear non-intrusive operator data.
        self._clear()

        # Solver defaults and shortcuts.
        if solver is None:
            # No regularization.
            solver = lstsq.PlainSolver()
        elif np.isscalar(solver):
            if solver == 0:
                # Also no regularization.
                solver = lstsq.PlainSolver()
            elif solver > 0:
                # Scalar Tikhonov (L2) regularization.
                solver = lstsq.L2Solver(solver)
            else:
                raise ValueError("if a scalar, `solver` must be nonnegative")

        # Lightly validate the solver: must be instance w/ fit(), predict().
        if isinstance(solver, type):
            raise TypeError("solver must be an instance, not a class")
        for mtd in "fit", "predict":
            if not hasattr(solver, mtd) or not callable(getattr(solver, mtd)):
                raise TypeError(f"solver must have a '{mtd}()' method")

        # Fully intrusive case, no least-squares problem to solve.
        if len(self._indices_of_operators_to_infer) == 0:
            return None, None, None, None

        def _check_valid_dimension0(dataset, label):
            """Dimension 0 must be r (state dimensions)."""
            if (dim := dataset.shape[0]) != self.state_dimension:
                raise errors.DimensionalityError(
                    f"{label}.shape[0] = {dim} != r = {self.state_dimension}"
                )

        # Process states, extract model dimension if needed.
        states = np.atleast_2d(states)
        if self.state_dimension is None:
            self.state_dimension = states.shape[0]
        _check_valid_dimension0(states, "states")

        def _check_valid_dimension1(dataset, label):
            """Dimension 1 must be k (number of snapshots)."""
            if (dim := dataset.shape[1]) != (k := states.shape[1]):
                raise errors.DimensionalityError(
                    f"{label}.shape[-1] = {dim} != {k} = states.shape[-1]"
                )

        # Process LHS.
        lhs = np.atleast_2d(lhs)
        _check_valid_dimension0(lhs, self._LHS_ARGNAME)
        _check_valid_dimension1(lhs, self._LHS_ARGNAME)

        # Process inputs, extract input dimension if needed.
        self._check_inputargs(inputs, "inputs")
        if self._has_inputs:
            inputs = np.atleast_2d(inputs)
            if not self.input_dimension:
                self.input_dimension = inputs.shape[0]
            if inputs.shape[0] != self.input_dimension:
                raise errors.DimensionalityError(
                    f"inputs.shape[0] = {inputs.shape[0]} "
                    f"!= {self.input_dimension} = m"
                )
            _check_valid_dimension1(inputs, "inputs")

        # Subtract known operator evaluations from the LHS.
        for i in self._indices_of_known_operators:
            lhs = lhs - self.operators[i].apply(states, inputs)

        return states, lhs, inputs, solver

    def _assemble_data_matrix(self, states, inputs):
        r"""Construct the Operator Inference data matrix :math:`\D`
        from state snapshots and/or input data.

        For example, if the model has the structure
        .. math::
            \ddt\qhat(t)
            = \chat + \Ahat\qhat(t)
            + \Hhat[\qhat(t)\otimes\qhat(t)] + \Bhat\u(t),

        then the data matrix is
        :math:`\D = [~
        \mathbf{1}~~
        \widehat{\Q}\trp~~
        (\widehat{\Q}\odot\widehat{\Q})\trp~~
        \U\trp~]`,

        where :math:`\widehat{\Q}` is ``states``
        and :math:`\U` is `inputs`.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data.
        inputs : (m, k) ndarray or None
            Inputs corresponding to the snapshots.

        Returns
        -------
        D : (k, d(r, m)) ndarray
            Operator Inference data matrix.
        """
        return np.hstack(
            [
                self.operators[i].datablock(states, inputs).T
                for i in self._indices_of_operators_to_infer
            ]
        )

    def _extract_operators(self, Ohat):
        """Extract and save the inferred operators from the solution to the
        Operator Inference regression problem.

        Parameters
        ----------
        Ohat : (r, d(r, m)) ndarray
            Matrix of operator entries, concatenated horizontally.
        """
        index = 0
        for i in self._indices_of_operators_to_infer:
            endex = index + self.operators[i].column_dimension(
                self.state_dimension, self.input_dimension
            )
            self.operators[i].set_entries(Ohat[:, index:endex])
            index = endex

    def _fit_solver(self, states, lhs, inputs=None, solver=None):
        """Construct a solver object mapping the regularizer to solutions
        of the Operator Inference least-squares problem.
        """
        states_, lhs_, inputs_, solver = self._process_fit_arguments(
            states, lhs, inputs, solver=solver
        )

        # Fully intrusive case (nothing to learn).
        if states_ is lhs_ is None:
            self.solver_ = None
            return

        # Set up non-intrusive learning.
        D = self._assemble_data_matrix(states_, inputs_)
        self.solver_ = solver.fit(D, lhs_.T)

    def _evaluate_solver(self):
        """Evaluate the least-squares solver and process the results."""
        # Fully intrusive case (nothing to learn).
        if self.solver_ is None:
            return

        # Execute non-intrusive learning.
        OhatT = self.solver_.predict()
        self._extract_operators(np.atleast_2d(OhatT.T))

    def fit(self, states, lhs, inputs=None, solver=None):
        """Learn the model operators from data.

        Parameters
        ----------
        states : (r, k) ndarray
            Snapshot training data. Each column is a single snapshot.
        lhs : (r, k) ndarray
            Left-hand side training data. Each column ``lhs[:, j]``
            corresponds to the snapshot ``states[:, j]``.
            The interpretation of this argument depends on the setting:
            forcing data for steady-state problems, next iteration for
            discrete-time problems, and time derivatives of the state for
            continuous-time problems.
        inputs : (m, k) or (k,) ndarray or None
            Input training data. Each column ``inputs[:, j]`` corresponds
            to the snapshot ``states[:, j]``.
            May be a one-dimensional array if ``m=1`` (scalar input).
        solver : :mod:`opinf.lstsq` object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            * ``None``: :class:`opinf.lstsq.PlainSolver()`, SVD-based solve
              without regularization.
            * float > 0: :class:`opinf.lstsq.L2Solver()`, SVD-based solve with
              scalar Tikhonov regularization.

        Returns
        -------
        self
        """
        self._fit_solver(states, lhs, inputs, solver=solver)
        self._evaluate_solver()
        return self

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Serialize the model, saving it in HDF5 format.
        The model can be recovered with the :meth:`load()` class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension ``.h5`` (HDF5).
        overwrite : bool
            If ``True`` and the specified ``savefile`` already exists,
            overwrite the file. If ``False`` (default) and the specified
            ``savefile`` already exists, raise an error.
        """
        with hdf5_savehandle(savefile, overwrite=overwrite) as hf:
            # Metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_operators"] = len(self.operators)
            meta.attrs["r"] = (
                int(self.state_dimension) if self.state_dimension else 0
            )
            meta.attrs["m"] = (
                int(self.input_dimension) if self.input_dimension else 0
            )

            # Store operator data.
            hf.create_dataset(
                "indices_infer", data=self._indices_of_operators_to_infer
            )
            hf.create_dataset(
                "indices_known", data=self._indices_of_known_operators
            )
            for i, op in enumerate(self.operators):
                op.save(hf.create_group(f"operator_{i}"))

    @classmethod
    def load(cls, loadfile):
        """Load a serialized model from an HDF5 file, created previously from
        the :meth:`save()` method.

        Parameters
        ----------
        loadfile : str
            File to load from, which should end in ``.h5``.

        Returns
        -------
        rom : _NonparametricModel
            Trained model.
        """
        with hdf5_loadhandle(loadfile) as hf:
            # Load metadata.
            num_operators = int(hf["meta"].attrs["num_operators"])
            indices_infer = [int(i) for i in hf["indices_infer"][:]]
            indices_known = [int(i) for i in hf["indices_known"][:]]

            # Load operators.
            ops = []
            for i in range(num_operators):
                gp = hf[f"operator_{i}"]
                OpClassName = gp["meta"].attrs["class"]
                ops.append(getattr(_operators, OpClassName).load(gp))

            # Construct the model.
            rom = cls(ops)
            rom._indices_of_operators_to_infer = indices_infer
            rom._indices_of_known_operators = indices_known
            if r := hf["meta"].attrs["r"]:
                rom.state_dimension = int(r)
            if (m := hf["meta"].attrs["m"]) and rom._has_inputs:
                rom.input_dimension = int(m)

        return rom
