# roms/nonparametric/_base.py
"""Base class for nonparametric reduced-order models."""

__all__ = []

import numpy as np

from .._base import _BaseROM
from ... import lstsq
from ... import errors
from ... import basis as _basis
from ... import operators_new as _operators
from ...utils import hdf5_savehandle, hdf5_loadhandle


class _NonparametricOpInfROM(_BaseROM):
    """Base class for nonparametric Operator Inference reduced-order models."""

    # Properties --------------------------------------------------------------
    @property
    def operator_matrix_(self):
        r""":math:`r \times d(r, m)` Operator matrix, e.g.,
        :math:`\widehat{\mathbf{O}} = [~
        \widehat{\mathbf{c}}~~
        \widehat{\mathbf{A}}~~
        \widehat{\mathbf{H}}~~
        \widehat{\mathbf{B}}~]`.

        This matrix **does not** includes the entries of any operators whose
        entries are known _a priori_.
        """
        self._check_is_trained()
        return np.column_stack([self.operators[i].entries
                                for i in self._indices_of_operators_to_infer])

    @property
    def data_matrix_(self):
        r""":math:`k \times d(r, m)` Data matrix, e.g.,
        :math:`\mathbf{D} = [~
        \mathbf{1}~~
        \widehat{\mathbf{Q}}^\mathsf{T}~~
        (\widehat{\mathbf{Q}}\odot\widehat{\mathbf{Q}})^\mathsf{T}~~
        \mathbf{U}^\mathsf{T}~]`.
        """
        if hasattr(self, "solver_"):
            return self.solver_.A if (self.solver_ is not None) else None
        raise AttributeError("data matrix not constructed (call fit())")

    @property
    def d(self):
        r"""Number of columns :math:`d(r, m)` of the data matrix
        :math:`\mathbf{D}`, i.e., the number of unknowns in the Operator
        Inference least-squares problem for each reduced-order mode.
        Always ``None`` if the dimensions ``r`` or ``m`` are not set.
        """
        if self.r is None or (self._has_inputs and self.m is None):
            return None
        return sum(self.operators[i].column_dimension(self.r, self.m)
                   for i in self._indices_of_operators_to_infer)

    # Fitting -----------------------------------------------------------------
    def _process_fit_arguments(self, states, lhs, inputs, solver=None):
        """Prepare training data for Operator Inference by extracting
        dimensions, projecting known operators, validating data sizes,
        and compressing training data.
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

        # Lightly validate the solver: must be instance w/ fit(), predict().
        if isinstance(solver, type):
            raise TypeError("solver must be an instance, not a class")
        for mtd in "fit", "predict":
            if not hasattr(solver, mtd) or not callable(getattr(solver, mtd)):
                raise TypeError(f"solver must have a '{mtd}()' method")

        # Validate / project any known operators.
        self.galerkin()
        if len(self._indices_of_operators_to_infer) == 0:
            # Fully intrusive case, no least-squares problem to solve.
            return None, None, None, None

        def _check_valid_dimension0(dataset, label):
            """Dimension 0 must be n or r (state dimensions)."""
            if (dim := dataset.shape[0]) not in (self.n, self.r):
                raise errors.DimensionalityError(
                    f"{label}.shape[0] = {dim} "
                    f"!= n or r (n = {self.n}, r = {self.r})")

        # Process states, extract ROM dimension if needed.
        states = np.atleast_2d(states)
        if self.basis is None:
            self.r = states.shape[0]
        _check_valid_dimension0(states, "states")
        states_ = self.compress(states, "states")

        def _check_valid_dimension1(dataset, label):
            """Dimension 1 must be k (number of snapshots)."""
            if (dim := dataset.shape[1]) != (k := states.shape[1]):
                raise errors.DimensionalityError(
                    f"{label}.shape[-1] = {dim} != {k} = states.shape[-1]")

        # Process LHS.
        lhs = np.atleast_2d(lhs)
        _check_valid_dimension0(lhs, self._LHS_ARGNAME)
        _check_valid_dimension1(lhs, self._LHS_ARGNAME)
        lhs_ = self.compress(lhs, self._LHS_ARGNAME)

        # Process inputs, extract input dimension if needed.
        self._check_inputargs(inputs, "inputs")
        if self._has_inputs:
            inputs = np.atleast_2d(inputs)
            if not self.m:
                self.m = inputs.shape[0]
            if inputs.shape[0] != self.m:
                raise errors.DimensionalityError(
                    f"inputs.shape[0] = {inputs.shape[0]} != {self.m} = m")
            _check_valid_dimension1(inputs, "inputs")

        # Subtract known operator evaluations from the LHS.
        for i in self._indices_of_known_operators:
            lhs_ = lhs_ - self.operators[i].evaluate(states_, inputs)

        return states_, lhs_, inputs, solver

    def _assemble_data_matrix(self, states_, inputs):
        r"""Construct the Operator Inference data matrix D from compressed
        state snapshots and/or input data.

        If the reduced-order model has the structure
        .. math::
            \frac{\textrm{d}}{\textrm{d}t}\widehat{\mathbf{q}}(t)
            = \widehat{\mathbf{c}}
            + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
            + \widehat{\mathbf{H}}[
            \widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)]
            + \widehat{\mathbf{B}}\mathbf{u}(t),

        then the data matrix is
        :math:`\mathbf{D} = [~
        \mathbf{1}~~
        \widehat{\mathbf{Q}}^\mathsf{T}~~
        (\widehat{\mathbf{Q}}\odot\widehat{\mathbf{Q}})^\mathsf{T}~~
        \mathbf{U}^\mathsf{T}~]`,

        where :math:`\widehat{\mathbf{Q}}` is ``states_``
        and :math:`\mathbf{U}` is `inputs`.

        Parameters
        ----------
        states_ : (r, k) ndarray
            Column-wise projected snapshot training data.
        inputs : (m, k) ndarray or None
            Column-wise inputs corresponding to the snapshots.

        Returns
        -------
        D : (k, d(r, m)) ndarray
            Operator Inference data matrix.
        """
        return np.hstack([self.operators[i].datablock(states_, inputs).T
                          for i in self._indices_of_operators_to_infer])

    def _extract_operators(self, Ohat):
        """Extract and save the inferred operators from the solution to the
        Operator Inference least-squares problem.

        Parameters
        ----------
        Ohat : (r, d(r, m)) ndarray
            Matrix of ROM operator coefficients, the transpose of the
            solution to the Operator Inference least-squares problem.
        """
        index = 0
        for i in self._indices_of_operators_to_infer:
            endex = index + self.operators[i].column_dimension(self.r, self.m)
            self.operators[i].set_entries(Ohat[:, index:endex])
            index = endex

    def _fit_solver(self, states, lhs, inputs=None, solver=None):
        """Construct a solver object mapping the regularizer to solutions
        of the Operator Inference least-squares problem.
        """
        states_, lhs_, inputs, solver = self._process_fit_arguments(
            states, lhs, inputs, solver=solver)

        # Fully intrusive case (nothing to learn).
        if states_ is lhs_ is None:
            self.solver_ = None
            return

        # Set up non-intrusive learning.
        D = self._assemble_data_matrix(states_, inputs)
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
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            either full order (`n` rows) or compressed to reduced order
            (`r` rows).
        lhs : (n, k) or (r, k) ndarray
            Left-hand side data for ROM training. Each column corresponds to
            one snapshot, either full order (`n` rows) or compressed to reduced
            order (`r` rows). The interpretation of the data depends on the
            setting: forcing data for steady-state problems, next iteration
            for discrete-time problems, and time derivatives of the state for
            continuous-time problems.
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input).
        solver : lstsq Solver object or float > 0 or None
            Solver for the least-squares regression. Defaults:

            - ``None``: ``lstsq.PlainSolver()``, SVD-based solve without
                regularization.
            - float > 0: ``lstsq.L2Solver()``, SVD-based solve with scalar
                Tikhonov regularization.

        Returns
        -------
        self
        """
        self._fit_solver(states, lhs, inputs, solver=solver)
        self._evaluate_solver()
        return self

    # Model persistence -------------------------------------------------------
    def save(self, savefile, save_basis=True, overwrite=False):
        """Serialize the ROM, saving it in HDF5 format.
        The model can later be loaded with the ``load()`` class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension ``.h5`` (HDF5).
        savebasis : bool
            If ``True``, save the ``basis`` as well as the model operators.
            If ``False``, only save model operators.
        overwrite : bool
            If ``True`` and the specified ``savefile`` already exists,
            overwrite the file. If ``False`` (default) and the specified
            ``savefile`` already exists, raise an error.
        """
        with hdf5_savehandle(savefile, overwrite=overwrite) as hf:

            # Metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_operators"] = len(self.operators)
            meta.attrs["r"] = int(self.r) if self.r else 0
            meta.attrs["m"] = int(self.m) if self.m else 0

            # Store basis (optionally) if it exists.
            if (self.basis is not None) and save_basis:
                meta.attrs["BasisClass"] = self.basis.__class__.__name__
                self.basis.save(hf.create_group("basis"))

            # Store operator data.
            hf.create_dataset("indices_infer",
                              data=self._indices_of_operators_to_infer)
            hf.create_dataset("indices_known",
                              data=self._indices_of_known_operators)
            for i, op in enumerate(self.operators):
                op.save(hf.create_group(f"operator_{i}"))

    @classmethod
    def load(cls, loadfile):
        """Load a serialized ROM from an HDF5 file, created previously from
        a ROM object's ``save()`` method.

        Parameters
        ----------
        loadfile : str
            File to load from, which should end in ``.h5``.

        Returns
        -------
        rom : _NonparametricOpInfROM
            Trained reduced-order model.
        """
        with hdf5_loadhandle(loadfile) as hf:

            # Load metadata.
            num_operators = int(hf["meta"].attrs["num_operators"])
            indices_infer = [int(i) for i in hf["indices_infer"][:]]
            indices_known = [int(i) for i in hf["indices_known"][:]]

            # Load basis if present.
            basis = None
            if "basis" in hf:
                gp = hf["basis"]
                # BasisClassName = gp["meta"].attrs["class"]
                BasisClassName = hf["meta"].attrs["BasisClass"]
                basis = getattr(_basis, BasisClassName).load(gp)

            # Load operators.
            ops = []
            for i in range(num_operators):
                gp = hf[f"operator_{i}"]
                OpClassName = gp["meta"].attrs["class"]
                ops.append(getattr(_operators, OpClassName).load(gp))

            # Construct the model.
            rom = cls(basis, ops)
            rom._indices_of_operators_to_infer = indices_infer
            rom._indices_of_known_operators = indices_known
            if (r := hf["meta"].attrs['r']) and rom.basis is None:
                rom.r = int(r)
            if (m := hf["meta"].attrs['m']) and rom._has_inputs:
                rom.m = int(m)

        return rom
