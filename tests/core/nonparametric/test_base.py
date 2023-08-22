# core/nonparametric/test_base.py
"""Tests for core.nonparametric._base."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la

import opinf

from .. import (MODELFORM_KEYS, MODEL_FORMS,
                _get_data, _get_operators, _trainedmodel)


class TestNonparametricOpInfROM:
    """Test core.nonparametric._base._NonparametricOpInfROM."""

    class Dummy(opinf.core.nonparametric._base._NonparametricOpInfROM):
        """Instantiable version of _NonparametricOpInfROM."""
        _LHS_ARGNAME = "ddts"

        def predict(*args, **kwargs):
            pass

    # Properties --------------------------------------------------------------
    def test_operator_matrix_(self, r=15, m=3):
        """Test operator_matrix_."""
        c, A, H, G, B = _get_operators(r, m, expanded=False)

        rom = self.Dummy("cA")
        rom.r = r
        rom.c_ = c
        rom.A_ = A
        assert np.all(rom.operator_matrix_ == np.column_stack([c, A]))

        rom.modelform = "HB"
        rom.r, rom.m = r, m
        rom.H_ = H
        rom.B_ = B
        assert np.all(rom.operator_matrix_ == np.column_stack([H, B]))

        rom.modelform = "G"
        rom.r = r
        rom.G_ = G
        assert np.all(rom.operator_matrix_ == G)

    def test_data_matrix_(self, k=500, m=20, r=10):
        """Test data_matrix_, i.e., spot check _assemble_data_matrix()."""
        Q, Qdot, U = _get_data(r, k, m)

        rom = self.Dummy("cAH")
        with pytest.raises(AttributeError) as ex:
            D = rom.data_matrix_
        assert ex.value.args[0] == "data matrix not constructed (call fit())"
        assert rom.d is None

        rom.modelform = "A"
        rom._fit_solver(None, Q, Qdot, inputs=None)
        assert np.all(rom.data_matrix_ == Q.T)
        assert rom.d == rom.data_matrix_.shape[1]

        rom.modelform = "B"
        rom._fit_solver(None, Q, Qdot, inputs=U)
        assert np.all(rom.data_matrix_ == U.T)
        assert rom.d == rom.data_matrix_.shape[1]

        rom .modelform = "HG"
        rom._fit_solver(None, Q, Qdot, inputs=None)
        D = np.column_stack([opinf.utils.kron2c(Q).T,
                             opinf.utils.kron3c(Q).T])
        assert np.allclose(rom.data_matrix_, D)
        assert rom.d == rom.data_matrix_.shape[1]

        rom.modelform = "c"
        rom._fit_solver(None, Q, Qdot, inputs=None)
        assert np.all(rom.data_matrix_ == np.ones((k, 1)))
        assert rom.d == 1

    # Fitting -----------------------------------------------------------------
    def test_check_training_data_shapes(self):
        """Test _check_training_data_shapes()."""
        # Get test data.
        k, m, r = 50, 20, 10
        Q, dQ, U = _get_data(r, k, m)
        rom = self.Dummy("A")
        rom.r = r

        def _test(args, message):
            with pytest.raises(ValueError) as ex:
                rom._check_training_data_shapes(args)
            assert ex.value.args[0] == message

        # Try to fit the rom with a single snapshot.
        args = [(Q[:, 0], "states"), (dQ, "dQ")]
        _test(args, "states must be two-dimensional")

        # Try to fit the rom with misaligned Q and dQ.
        args = [(Q, "staaates"), (dQ[:, 1:-1], "dQs")]
        _test(args, f"dQs.shape[-1] = {k-2:d} != {k:d} = staaates.shape[-1]")

        # Try to fit the rom with misaligned Q and U.
        rom.modelform = "AB"
        rom.r, rom.m = r, m
        args = [(Q, "states"), (dQ, "dQ"), (U[:, 1:-1], "inputs")]
        _test(args, f"inputs.shape[-1] = {k-2:d} != {k} = states.shape[-1]")

        # Try with bad number of rows in states.
        args = [(Q[:-1, :], "states"), (dQ, "dQ"), (U, "inputs")]
        _test(args, f"states.shape[0] != n or r (n=None, r={r})")

        # Try with one-dimensional inputs when not allowed.
        rom.m = 2
        args = [(Q, "states"), (dQ, "dQ"), (U[:, 0], "inputs")]
        _test(args, "inputs must be two-dimensional (m > 1)")

        # Try with bad number of rows in inputs.
        rom.m = m
        args = [(Q, "states"), (dQ, "dQ"), (U[:-1, :], "inputs")]
        _test(args, f"inputs.shape[0] = {m-1} != {m} = m")

        # Try with bad dimension in inputs with m = 1.
        rom.m = 1
        args = [(U.reshape(1, 1, -1), "inputs")]
        _test(args, "inputs must be one- or two-dimensional (m = 1)")

        # Try with bad two-dimensional inputs with m = 1.
        rom.m = 1
        args = [(U.reshape(-1, 1), "inputs")]
        _test(args, "inputs.shape != (1, k) (m = 1)")

        # Correct usage.
        args = [(Q, "states"), (dQ, "dQ")]
        rom._check_training_data_shapes(args)
        args.append((U, "inputs"))
        rom.m = m
        rom._check_training_data_shapes(args)

        # Special case: m = inputs.ndim = 1.
        args[-1] = (U[0, :], "inputs")
        rom.m = 1
        rom._check_training_data_shapes(args)

    def test_process_fit_arguments(self, n=60, k=500, m=20, r=10):
        """Test _process_fit_arguments()."""
        # Get test data.
        Q, lhs, U = _get_data(n, k, m)
        U1d = U[0, :]
        Vr = la.svd(Q)[0][:, :r]
        ones = np.ones(k)

        # Try with bad solver option.
        rom = self.Dummy("AB")

        with pytest.raises(TypeError) as ex:
            rom._process_fit_arguments(None, None, None, None,
                                       solver=opinf.lstsq.PlainSolver)
        assert ex.value.args[0] == "solver must be an instance, not a class"

        class _DummySolver:
            pass

        with pytest.raises(TypeError) as ex:
            rom._process_fit_arguments(None, None, None, None,
                                       solver=_DummySolver())
        assert ex.value.args[0] == "solver must have a 'fit()' method"

        # With basis and input.
        Q_, lhs_, solver = rom._process_fit_arguments(Vr, Q, lhs, U)
        assert rom.n == n
        assert rom.r == r
        assert isinstance(rom.basis, opinf.basis.LinearBasis)
        assert np.all(rom.basis.entries == Vr)
        assert rom.m == m
        assert np.allclose(Q_, Vr.T @ Q)
        assert np.allclose(lhs_, Vr.T @ lhs)
        assert isinstance(solver, opinf.lstsq.PlainSolver)

        # Without basis and with a one-dimensional input.
        rom.modelform = "cHB"
        Q_, lhs_, solver = rom._process_fit_arguments(None,
                                                      Q, lhs, U1d, solver=0)
        assert rom.n is None
        assert rom.r == n
        assert rom.basis is None
        assert rom.m == 1
        assert Q_ is Q
        assert lhs_ is lhs
        assert isinstance(solver, opinf.lstsq.PlainSolver)

        # With basis and no input.
        rom.modelform = "cA"
        Q_, lhs_, solver = rom._process_fit_arguments(Vr, Q, lhs, None,
                                                      solver=1)
        assert rom._projected_operators_ == ""
        assert rom.n == n
        assert rom.r == r
        assert isinstance(rom.basis, opinf.basis.LinearBasis)
        assert np.all(rom.basis.entries == Vr)
        assert rom.m == 0
        assert np.allclose(Q_, Vr.T @ Q)
        assert np.allclose(lhs_, Vr.T @ lhs)
        assert isinstance(solver, opinf.lstsq.L2Solver)

        # With known operators for A.
        c, A, _, _, B = _get_operators(n, m, expanded=True)
        rom.modelform = "AHB"
        Q_, lhs_, _ = rom._process_fit_arguments(Vr, Q, lhs, U,
                                                 known_operators={"A": A})
        assert rom._projected_operators_ == "A"
        assert np.allclose(lhs_, Vr.T @ (lhs - (A @ Vr @ Q_)))

        # With known operators for c and B.
        rom.modelform = "cAHB"
        ops = {"B": B, "c": c}
        Q_, lhs_, _ = rom._process_fit_arguments(Vr, Q, lhs, U,
                                                 known_operators=ops)
        assert sorted(rom._projected_operators_) == sorted("Bc")
        lhstrue = Vr.T @ (lhs - B @ U - np.outer(c, ones))
        assert np.allclose(lhs_, lhstrue)

        # Special case: m = inputs.ndim = 1
        U1d = U[0]
        B1d = B[:, 0]
        Q_, lhs_, _ = rom._process_fit_arguments(Vr, Q, lhs, U1d,
                                                 known_operators={"B": B1d})
        assert rom.m == 1
        assert rom._projected_operators_ == "B"
        assert np.allclose(lhs_, Vr.T @ (lhs - np.outer(B1d, ones)))

        # Fully intrusive.
        rom.modelform = "cA"
        ops = {"c": c, "A": A}
        Q_, lhs_, _ = rom._process_fit_arguments(Vr, Q, lhs, None,
                                                 known_operators=ops)
        assert sorted(rom._projected_operators_) == sorted("cA")
        assert Q_ is None
        assert lhs_ is None

    def test_assemble_data_matrix(self, k=500, m=20, r=10):
        """Test _assemble_data_matrix()."""
        # Get test data.
        Q_, _, U = _get_data(r, k, m)

        rom = self.Dummy("c")
        for form in MODEL_FORMS:
            rom.modelform = form
            rom.r = r
            if 'B' in form:
                rom.m = m
            D = rom._assemble_data_matrix(Q_, U)
            d = opinf.lstsq.lstsq_size(form, r, m if 'B' in form else 0)
            assert D.shape == (k, d)

            # Spot check.
            if form == "c":
                assert np.allclose(D, np.ones((k, 1)))
            elif form == "H":
                assert np.allclose(D, opinf.utils.kron2c(Q_).T)
            elif form == "G":
                assert np.allclose(D, opinf.utils.kron3c(Q_).T)
            elif form == "AB":
                assert np.allclose(D[:, :r], Q_.T)
                assert np.allclose(D[:, r:], U.T)

        # Try with one-dimensional inputs as a 1D array.
        rom.modelform = "cB"
        rom.m = 1
        D = rom._assemble_data_matrix(Q_, U[0])
        assert D.shape == (k, 2)
        assert np.allclose(D, np.column_stack((np.ones(k), U[0])))

    def test_extract_operators(self, m=2, r=10):
        """Test _extract_operators()."""
        shapes = {
                    "c_": (r,),
                    "A_": (r, r),
                    "H_": (r, r*(r+1)//2),
                    "G_": (r, r*(r+1)*(r+2)//6),
                    "B_": (r, m),
                 }

        rom = self.Dummy("")

        for form in MODEL_FORMS:
            rom.modelform = form
            rom.r = r
            if 'B' in form:
                rom.m = m
            d = opinf.lstsq.lstsq_size(form, r, rom.m)
            Ohat = np.random.random((r, d))
            rom._extract_operators(Ohat)
            for prefix in MODELFORM_KEYS:
                attr = prefix+'_'
                assert hasattr(rom, attr)
                value = getattr(rom, attr)
                if prefix in form:
                    assert opinf.operators.is_operator(value)
                    assert value.shape == shapes[attr]
                else:
                    assert value is None

    def test_fit(self, n=60, k=500, m=20, r=10):
        """Test fit()."""
        # Get test data.
        Q, F, U = _get_data(n, k, m)
        U1d = U[0, :]
        Vr = la.svd(Q)[0][:, :r]
        args_n = [Q, F]
        args_r = [Vr.T @ Q, Vr.T @ F]

        # Fit the rom with each modelform.
        rom = self.Dummy("c")
        for form in MODEL_FORMS:
            rom.modelform = form
            if "B" in form:
                # Two-dimensional inputs.
                rom.fit(Vr, *args_n, inputs=U)          # With basis.
                rom.fit(None, *args_r, inputs=U)        # Without basis.
                # One-dimensional inputs.
                rom.fit(Vr, *args_n, inputs=U1d)        # With basis.
                rom.fit(None, *args_r, inputs=U1d)      # Without basis.
            else:
                # No inputs.
                rom.fit(Vr, *args_n, inputs=None)       # With basis.
                rom.fit(None, *args_r, inputs=None)     # Without basis.

        # Special case: fully intrusive.
        rom.modelform = "BA"
        _, A, _, _, B = _get_operators(n, m)
        rom.fit(Vr, None, None, known_operators={"A": A, "B": B})
        assert rom.solver_ is None
        assert opinf.operators.is_operator(rom.A_)
        assert opinf.operators.is_operator(rom.B_)
        assert np.allclose(rom.A_.entries, Vr.T @ A @ Vr)
        assert np.allclose(rom.B_.entries, Vr.T @ B)

    # Model persistence -------------------------------------------------------
    def test_save(self, n=15, m=2, r=3, target="_savemodeltest.h5"):
        """Test save()."""
        # Clean up after old tests.
        if os.path.isfile(target):              # pragma: no cover
            os.remove(target)

        # Get a test model.
        Vr = np.random.random((n, r))
        rom = _trainedmodel(self.Dummy, "cAHGB", Vr, m)

        def _checkfile(filename, rom, hasbasis):
            assert os.path.isfile(filename)
            with h5py.File(filename, 'r') as data:
                # Check metadata.
                assert "meta" in data
                assert len(data["meta"]) == 0
                assert data["meta"].attrs["modelform"] == rom.modelform

                # Check basis
                if hasbasis:
                    assert "basis" in data
                    assert np.all(data["basis/entries"][:] == Vr)
                else:
                    assert "basis" not in data

                # Check operators
                assert "operators" in data
                if "c" in rom.modelform:
                    assert np.all(data["operators/c_"][:] == rom.c_.entries)
                else:
                    assert "c_" not in data["operators"]
                if "A" in rom.modelform:
                    assert np.all(data["operators/A_"][:] == rom.A_.entries)
                else:
                    assert "A_" not in data["operators"]
                if "H" in rom.modelform:
                    assert np.all(data["operators/H_"][:] == rom.H_.entries)
                else:
                    assert "H_" not in data["operators"]
                if "G" in rom.modelform:
                    assert np.all(data["operators/G_"][:] == rom.G_.entries)
                else:
                    assert "G_" not in data["operators"]
                if "B" in rom.modelform:
                    assert np.all(data["operators/B_"][:] == rom.B_.entries)
                else:
                    assert "B_" not in data["operators"]

        rom.save(target, save_basis=False)
        _checkfile(target, rom, False)

        with pytest.raises(FileExistsError) as ex:
            rom.save(target, overwrite=False)
        assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

        rom.save(target, save_basis=True, overwrite=True)
        _checkfile(target, rom, True)

        rom = _trainedmodel(self.Dummy, "c", Vr, 0)
        rom.save(target, overwrite=True)
        _checkfile(target, rom, True)

        rom = _trainedmodel(self.Dummy, "AB", Vr, m)
        rom.basis = None
        rom.save(target, save_basis=True, overwrite=True)
        _checkfile(target, rom, False)

        # Check that save() and load() are inverses.
        rom.basis = Vr
        rom.save(target, save_basis=True, overwrite=True)
        rom2 = rom.load(target)
        assert rom2 is not rom
        assert rom2.basis == rom.basis
        assert rom2 == rom
        for attr in ["n", "m", "r", "modelform", "__class__"]:
            assert getattr(rom, attr) == getattr(rom2, attr)
        for attr in ["A_", "B_"]:
            got = getattr(rom2, attr)
            assert opinf.operators.is_operator(got)
            assert np.all(getattr(rom, attr).entries == got.entries)
        for attr in ["c_", "H_", "G_"]:
            assert getattr(rom, attr) is getattr(rom2, attr) is None

        # Check basis = None functionality.
        rom.basis = None
        rom.save(target, overwrite=True)
        rom2 = rom.load(target)
        assert rom2 is not rom
        assert rom2 == rom
        for attr in ["m", "r", "modelform", "__class__"]:
            assert getattr(rom, attr) == getattr(rom2, attr)
        for attr in ["A_", "B_"]:
            got = getattr(rom2, attr)
            assert opinf.operators.is_operator(got)
            assert np.all(getattr(rom, attr).entries == got.entries)
        for attr in ["n", "c_", "H_", "G_", "basis"]:
            assert getattr(rom, attr) is getattr(rom2, attr) is None

        os.remove(target)

    def test_load(self, n=20, m=2, r=5, target="_loadmodeltest.h5"):
        """Test load()."""
        # Get test operators.
        Vr = np.random.random((n, r))
        c_, A_, H_, G_, B_ = _get_operators(n=r, m=m)

        # Clean up after old tests if needed.
        if os.path.isfile(target):                  # pragma: no cover
            os.remove(target)

        # Make an empty HDF5 file to start with.
        with h5py.File(target, 'w'):
            pass

        with pytest.raises(ValueError) as ex:
            rom = self.Dummy.load(target)
        assert ex.value.args[0] == "invalid save format (meta/ not found)"

        # Make a partially compatible HDF5 file to start with.
        with h5py.File(target, 'a') as hf:
            # Store metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["modelform"] = "cAB"

        with pytest.raises(ValueError) as ex:
            rom = self.Dummy.load(target)
        assert ex.value.args[0] == "invalid save format (operators/ not found)"

        # Store the arrays.
        with h5py.File(target, 'a') as hf:
            hf.create_dataset("operators/c_", data=c_)
            hf.create_dataset("operators/A_", data=A_)
            hf.create_dataset("operators/B_", data=B_)

        def _check_model(rom):
            assert isinstance(rom, self.Dummy)
            for attr in ["modelform",
                         "n", "r", "m",
                         "c_", "A_", "H_", "G_", "B_", "basis"]:
                assert hasattr(rom, attr)
            assert rom.modelform == "cAB"
            assert rom.r == r
            assert rom.m == m
            for attr in ["c_", "A_", "B_"]:
                assert opinf.operators.is_operator(getattr(rom, attr))
            assert np.all(rom.c_.entries == c_)
            assert np.all(rom.A_.entries == A_)
            assert rom.H_ is None
            assert rom.G_ is None
            assert np.all(rom.B_.entries == B_)

        # Load the file correctly.
        rom = self.Dummy.load(target)
        _check_model(rom)
        assert rom.basis is None
        assert rom.n is None

        # Add the basis and then load the file correctly.
        basis = opinf.basis.LinearBasis().fit(Vr)
        with h5py.File(target, 'a') as hf:
            hf["meta"].attrs["BasisClass"] = "LinearBasis"
            basis.save(hf.create_group("basis"))
        rom = self.Dummy.load(target)
        _check_model(rom)
        assert isinstance(rom.basis, type(basis))
        assert np.all(rom.basis.entries == Vr)
        assert rom.n == n

        # One additional test to cover other cases.
        with h5py.File(target, 'a') as f:
            f["meta"].attrs["modelform"] = "HG"
            f.create_dataset("operators/H_", data=H_)
            f.create_dataset("operators/G_", data=G_)

        rom = self.Dummy.load(target)
        assert isinstance(rom, self.Dummy)
        for attr in ["modelform",
                     "n", "r", "m",
                     "c_", "A_", "H_", "G_", "B_", "basis"]:
            assert hasattr(rom, attr)
        assert rom.modelform == "HG"
        assert rom.r == r
        assert rom.m == 0
        for attr in ["H_", "G_"]:
            assert opinf.operators.is_operator(getattr(rom, attr))
        assert rom.c_ is None
        assert rom.A_ is None
        assert np.all(rom.H_.entries == H_)
        assert np.all(rom.G_.entries == G_)
        assert rom.B_ is None
        assert isinstance(rom.basis, type(basis))
        assert np.all(rom.basis.entries == Vr)
        assert rom.n == n

        # Clean up.
        os.remove(target)
