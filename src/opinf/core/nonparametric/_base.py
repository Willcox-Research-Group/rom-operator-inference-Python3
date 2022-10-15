# core/nonparametric/_base.py
"""Base class for nonparametric Operator Inference reduced-order models."""

__all__ = []

import numpy as np

from .._base import _BaseROM
from ... import lstsq, pre
from ...utils import kron2c, kron3c, hdf5_savehandle, hdf5_loadhandle


class _NonparametricOpInfROM(_BaseROM):
    """Base class for nonparametric Operator Inference reduced-order models."""

    # Properties --------------------------------------------------------------
    @property
    def operator_matrix_(self):
        """r x d(r, m) Operator matrix O_ = [ c_ | A_ | H_ | G_ | B_ ]."""
        self._check_is_trained()
        return np.column_stack([op.entries for op in self])

    @property
    def data_matrix_(self):
        """k x d(r, m) Data matrix D = [ 1 | Q^T | (Q ⊗ Q)^T | ... ]."""
        if hasattr(self, "solver_"):
            return self.solver_.A if (self.solver_ is not None) else None
        raise AttributeError("data matrix not constructed (call fit())")

    # Fitting -----------------------------------------------------------------
    def _check_training_data_shapes(self, datasets):
        """Ensure that each data set has the same number of columns and a
        valid number of rows (as determined by the basis).

        Parameters
        ----------
        datasets: list of (ndarray, str) tuples
            Datasets paired with labels, e.g., [(Q, "states"), (dQ, "ddts")].
        """
        data0, label0 = datasets[0]
        for data, label in datasets:
            if label == "inputs":
                if self.m != 1:     # inputs.shape = (m, k)
                    if data.ndim != 2:
                        raise ValueError("inputs must be two-dimensional "
                                         "(m > 1)")
                    if data.shape[0] != self.m:
                        raise ValueError(f"inputs.shape[0] = {data.shape[0]} "
                                         f"!= {self.m} = m")
                else:               # inputs.shape = (1, k) or (k,)
                    if data.ndim not in (1, 2):
                        raise ValueError("inputs must be one- or "
                                         "two-dimensional (m = 1)")
                    if data.ndim == 2 and data.shape[0] != 1:
                        raise ValueError("inputs.shape != (1, k) (m = 1)")
            else:
                if data.ndim != 2:
                    raise ValueError(f"{label} must be two-dimensional")
                if data.shape[0] not in (self.n, self.r):
                    raise ValueError(f"{label}.shape[0] != n or r "
                                     f"(n={self.n}, r={self.r})")
            if data.shape[-1] != data0.shape[-1]:
                raise ValueError(f"{label}.shape[-1] = {data.shape[-1]} "
                                 f"!= {data0.shape[-1]} = {label0}.shape[-1]")

    def _process_fit_arguments(self, basis, states, lhs, inputs,
                               known_operators):
        """Prepare training data for Operator Inference by clearing old data,
        storing the basis, extracting dimensions, projecting known operators,
        validating data sizes, and projecting training data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and lhs are assumed to already be projected.
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            either full order (n rows) or projected to reduced order (r rows).
        lhs : (n, k) or (r, k) ndarray
            Left-hand side data for ROM training. Each column corresponds to
            one snapshot, either full order (n rows) or reduced order (r rows).
            * Steady: forcing function.
            * Discrete: column-wise next iteration
            * Continuous: time derivative of the state
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        Returns
        -------
        states_ : (r, k) ndarray
            Projected state snapshots.
        lhs_ : (r, k) ndarray
            Projected left-hand-side data.
        """
        # Clear all data (basis and operators).
        self._clear()

        # Store basis and (hence) reduced dimension.
        self.basis = basis

        # Validate / project any known operators.
        self._project_operators(known_operators)
        if len(self._projected_operators_) == len(self.modelform):
            # Fully intrusive case, nothing to learn with OpInf.
            return None, None

        # Get state and input dimensions if needed.
        if self.basis is None:
            self.r = states.shape[0]
        self._check_inputargs(inputs, "inputs")
        to_check = [(states, "states"), (lhs, self._LHS_ARGNAME)]
        if 'B' in self.modelform:
            if self.m is None:
                self.m = 1 if inputs.ndim == 1 else inputs.shape[0]
            to_check.append((inputs, "inputs"))

        # Ensure training datasets have consistent sizes.
        self._check_training_data_shapes(to_check)

        # Encode states and lhs in the reduced subspace (if needed).
        states_ = self.encode(states, "states")
        lhs_ = self.encode(lhs, self._LHS_ARGNAME)

        # Subtract known data from the lhs data.
        for key in self._projected_operators_:
            if key == 'c':              # Known constant term.
                lhs_ = lhs_ - np.outer(self.c_(), np.ones(states_.shape[1]))
            elif key == 'B':            # Known input term.
                lhs_ = lhs_ - self.B_(np.atleast_2d(inputs))
            else:                       # Known linear/quadratic/cubic term.
                lhs_ = lhs_ - getattr(self, f"{key}_")(states_)

        return states_, lhs_

    def _assemble_data_matrix(self, states_, inputs):
        """Construct the Operator Inference data matrix D from projected data.

        If modelform="cAHB", this is D = [1 | Q_.T | (Q_ ⊗ Q_).T | U.T],

        where Q_ = states_ and U = inputs.

        Parameters
        ----------
        states_ : (r, k) ndarray
            Column-wise projected snapshot training data.
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input).

        Returns
        -------
        D : (k, d(r, m)) ndarray
            Operator Inference data matrix (no regularization).
        """
        to_infer = {key for key in self.modelform
                    if key not in self._projected_operators_}
        D = []
        if 'c' in to_infer:             # Constant term.
            D.append(np.ones((states_.shape[1], 1)))

        if 'A' in to_infer:             # Linear state term.
            D.append(states_.T)

        if 'H' in to_infer:             # (compact) Quadratic state term.
            D.append(kron2c(states_).T)

        if 'G' in to_infer:             # (compact) Cubic state term.
            D.append(kron3c(states_).T)

        if 'B' in to_infer:             # Linear input term.
            D.append(np.atleast_2d(inputs).T)

        return np.hstack(D)

    def _extract_operators(self, Ohat):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squares problem.

        Parameters
        ----------
        Ohat : (r, d(r, m)) ndarray
            Block matrix of ROM operator coefficients, the transpose of the
            solution to the Operator Inference linear least-squares problem.
        """
        to_infer = {key for key in self.modelform
                    if key not in self._projected_operators_}
        i = 0

        if 'c' in to_infer:             # Constant term (one-dimensional).
            self.c_ = Ohat[:, i:i+1][:, 0]
            i += 1

        if 'A' in to_infer:             # Linear state matrix.
            self.A_ = Ohat[:, i:i+self.r]
            i += self.r

        if 'H' in to_infer:             # (compact) Qudadratic state matrix.
            _r2 = self._r2
            self.H_ = Ohat[:, i:i+_r2]
            i += _r2

        if 'G' in to_infer:             # (compact) Cubic state matrix.
            _r3 = self._r3
            self.G_ = Ohat[:, i:i+_r3]
            i += _r3

        if 'B' in to_infer:             # Linear input matrix.
            self.B_ = Ohat[:, i:i+self.m]
            i += self.m

        return

    def _construct_solver(self, basis, states, lhs,
                          inputs=None, regularizer=0, known_operators=None):
        """Construct a solver object mapping the regularizer to solutions
        of the Operator Inference least-squares problem.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and lhs are assumed to already be projected.
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            either full order (n rows) or projected to reduced order (r rows).
        lhs : (n, k) or (r, k) ndarray
            Left-hand side data for ROM training. Each column corresponds to
            one snapshot, either full order (n rows) or reduced order (r rows).
            * Steady: forcing function.
            * Discrete: column-wise next iteration
            * Continuous: time derivative of the state
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0, (d, d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB". This parameter is used here
            only to determine the correct type of solver.
        known_operators : dict
            Dictionary of known full-order or reduced-order operators.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred.
            Keys must match the modelform, values are ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.
        """
        states_, lhs_ = self._process_fit_arguments(basis, states, lhs, inputs,
                                                    known_operators)
        # Fully intrusive case (nothing to learn).
        if states_ is lhs_ is None:
            self.solver_ = None
            return

        D = self._assemble_data_matrix(states_, inputs)
        self.solver_ = lstsq.solver(D, lhs_.T, regularizer)

    def _evaluate_solver(self, regularizer):
        """Evaluate the least-squares solver with the given regularizer.

        Parameters
        ----------
        regularizer : float >= 0, (d, d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".
        """
        # Fully intrusive case (nothing to learn).
        if self.solver_ is None:
            return

        OhatT = self.solver_.predict(regularizer)
        self._extract_operators(np.atleast_2d(OhatT.T))

    def fit(self, basis, states, lhs, inputs=None,
            regularizer=0, known_operators=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, states and lhs are assumed to already be projected.
        states : (n, k) or (r, k) ndarray
            Column-wise snapshot training data. Each column is one snapshot,
            either full order (n rows) or projected to reduced order (r rows).
        lhs : (n, k) or (r, k) ndarray
            Left-hand side data for ROM training. Each column corresponds to
            one snapshot, either full order (n rows) or reduced order (r rows).
            * Steady: forcing function.
            * Discrete: column-wise next iteration
            * Continuous: time derivative of the state
        inputs : (m, k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. May be a
            one-dimensional array if m=1 (scalar input). Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.
        regularizer : float >= 0, (d, d) ndarray or list of r of these
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".
        known_operators : dict or None
            Dictionary of known full-order operators.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.

        Returns
        -------
        self
        """
        self._construct_solver(basis, states, lhs, inputs, regularizer,
                               known_operators)
        self._evaluate_solver(regularizer)
        return self

    # Model persistence -------------------------------------------------------
    def save(self, savefile, save_basis=True, overwrite=False):
        """Serialize the ROM, saving it in HDF5 format.
        The model can then be loaded with the load() class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension '.h5' (HDF5).
        savebasis : bool
            If True, save the basis as well as the reduced operators.
            If False, only save reduced operators.
        overwrite : bool
            If True and the specified file already exists, overwrite the file.
            If False and the specified file already exists, raise an error.
        """
        self._check_is_trained()

        with hdf5_savehandle(savefile, overwrite=overwrite) as hf:

            # Store ROM modelform.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["modelform"] = self.modelform

            # Store basis (optionally) if it exists.
            if (self.basis is not None) and save_basis:
                meta.attrs["BasisClass"] = self.basis.__class__.__name__
                self.basis.save(hf.create_group("basis"))

            # Store reduced operators.
            for key, op in zip(self.modelform, self):
                hf.create_dataset(f"operators/{key}_", data=op.entries)

    @classmethod
    def load(cls, loadfile):
        """Load a serialized ROM from an HDF5 file, created previously from
        a ROM object's save() method.

        Parameters
        ----------
        loadfile : str
            File to load from, which should end in '.h5'.

        Returns
        -------
        model : _NonparametricOpInfROM
            Trained reduced-order model.
        """
        with hdf5_loadhandle(loadfile) as hf:
            if "meta" not in hf:
                raise ValueError("invalid save format (meta/ not found)")
            if "operators" not in hf:
                raise ValueError("invalid save format (operators/ not found)")

            # Load metadata.
            modelform = hf["meta"].attrs["modelform"]
            basis = None

            # Load basis if present.
            if "basis" in hf:
                BasisClassName = hf["meta"].attrs["BasisClass"]
                basis = getattr(pre, BasisClassName).load(hf["basis"])

            # Load operators.
            operators = {f"{key}_": hf[f"operators/{key}_"][:]
                         for key in modelform}

        # Construct the model.
        return cls(modelform)._set_operators(basis, **operators)
