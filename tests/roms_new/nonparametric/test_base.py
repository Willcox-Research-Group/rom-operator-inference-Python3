# roms/nonparametric/test_base.py
"""Tests for roms.nonparametric._base."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la

import opinf

from .. import MODEL_FORMS, _get_data, _get_operators


opinf_operators = opinf.operators_new  # TEMP


class TestNonparametricROM:
    """Test roms.nonparametric._base._NonparametricROM."""

    class Dummy(opinf.roms_new.nonparametric._base._NonparametricROM):
        """Instantiable version of _NonparametricROM."""

        _LHS_ARGNAME = "mylhs"

        def predict(*args, **kwargs):
            pass

    # Properties --------------------------------------------------------------
    def test_operator_matrix_(self, r=15, m=3):
        """Test _NonparametricROM.operator_matrix_."""
        c, A, H, G, B, N = _get_operators("cAHGBN", r, m)

        rom = self.Dummy("cA")
        rom.r = r
        rom.operators[:] = (c, A)
        D = np.column_stack([c.entries, A.entries])
        assert np.all(rom.operator_matrix_ == D)

        rom.operators[:] = (H, B)
        rom._has_inputs = True
        rom.m = m
        D = np.column_stack([H.entries, B.entries])
        assert np.all(rom.operator_matrix_ == D)

        rom.operators[:] = [G, N]
        D = np.column_stack([G.entries, N.entries])
        assert np.all(rom.operator_matrix_ == D)

    def test_data_matrix_(self, k=500, m=20, r=10):
        """Test _NonparametricROM.data_matrix_, i.e., spot check
        _NonparametricROM._assemble_data_matrix().
        """
        Q, Qdot, U = _get_data(r, k, m)

        rom = self.Dummy("cAH")
        with pytest.raises(AttributeError) as ex:
            D = rom.data_matrix_
        assert ex.value.args[0] == "data matrix not constructed (call fit())"
        assert rom.d is None

        rom.operators = "A"
        rom._fit_solver(Q, Qdot, inputs=None)
        D = rom.data_matrix_
        assert D.shape == (k, r)
        assert np.all(D == Q.T)
        assert rom.d == r

        rom.operators = "B"
        rom._fit_solver(Q, Qdot, inputs=U)
        D = rom.data_matrix_
        assert D.shape == (k, m)
        assert np.all(D == U.T)
        assert rom.d == m

        rom.operators = "HG"
        rom._fit_solver(Q, Qdot, inputs=None)
        Dtrue = np.column_stack(
            [opinf.utils.kron2c(Q).T, opinf.utils.kron3c(Q).T]
        )
        D = rom.data_matrix_
        d = r * (r + 1) // 2 + r * (r + 1) * (r + 2) // 6
        assert D.shape == (k, d)
        assert np.allclose(D, Dtrue)
        assert rom.d == d

        rom.operators = "c"
        rom._fit_solver(Q, Qdot, inputs=None)
        D = rom.data_matrix_
        assert D.shape == (k, 1)
        assert np.all(D == 1)
        assert rom.d == 1

        # Partially intrusive case.
        c = opinf_operators.ConstantOperator(np.ones(r))
        rom.operators = ["A", c]
        rom._fit_solver(Q, Qdot, inputs=None)
        D = rom.data_matrix_
        assert D.shape == (k, r)
        assert np.all(D == Q.T)

        # Fully intrusive case.
        rom.operators = _get_operators("cHB", r, m)
        rom._fit_solver(Q, Qdot, inputs=U)
        assert rom.data_matrix_ is None

    # Fitting -----------------------------------------------------------------
    def test_process_fit_arguments(self, n=20, k=50, m=4, r=6):
        """Test _NonparametricROM._process_fit_arguments()."""
        # Get test data.
        Q, lhs, U = _get_data(n, k, m)
        c, A, B = _get_operators("cAB", n, m)
        U1d = U[0, :]
        Vr = la.svd(Q)[0][:, :r]
        basis = opinf.basis.LinearBasis().fit(Vr)
        ones = np.ones(k)

        # Exceptions #

        # Try with bad solver option.
        rom = self.Dummy("AB")
        with pytest.raises(TypeError) as ex:
            rom._process_fit_arguments(
                None, None, None, solver=opinf.lstsq.PlainSolver
            )
        assert ex.value.args[0] == "solver must be an instance, not a class"

        class _DummySolver:
            pass

        with pytest.raises(TypeError) as ex:
            rom._process_fit_arguments(None, None, None, solver=_DummySolver())
        assert ex.value.args[0] == "solver must have a 'fit()' method"

        # Intrusive projection without a basis.
        rom = self.Dummy([A])
        rom.r = r
        with pytest.raises(RuntimeError) as ex:
            rom._process_fit_arguments(None, None, None)
        assert ex.value.args[0] == "basis required for Galerkin projection"

        # States do not match dimensions 'r' or 'n'.
        A_ = opinf_operators.LinearOperator()
        rom = self.Dummy([A_], basis)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            rom._process_fit_arguments(Q[1:], None, None)
        assert (
            ex.value.args[0] == f"states.shape[0] = {n-1} != n "
            f"or r (n = {n}, r = {r})"
        )

        # LHS not aligned with states.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            rom._process_fit_arguments(Q, lhs[:, :-1], U)
        assert (
            ex.value.args[0] == f"mylhs.shape[-1] = {k-1} "
            f"!= {k} = states.shape[-1]"
        )

        # Inputs do not match dimension 'm'.
        rom.operators = [A_, B]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            rom._process_fit_arguments(Q, lhs, U[1:])
        assert ex.value.args[0] == f"inputs.shape[0] = {m-1} != {m} = m"

        # Inputs not aligned with states.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            rom._process_fit_arguments(Q, lhs, U[:, :-1])
        assert (
            ex.value.args[0] == f"inputs.shape[-1] = {k-1} "
            f"!= {k} = states.shape[-1]"
        )

        # Correct usage #

        # With basis and input.
        rom.basis = Vr
        rom.operators = "AB"
        Q_, lhs_, _, solver = rom._process_fit_arguments(Q, lhs, U)
        assert rom.n == n
        assert rom.r == r
        assert isinstance(rom.basis, opinf.basis.LinearBasis)
        assert np.all(rom.basis.entries == Vr)
        assert rom.m == m
        assert Q_.shape == (r, k)
        assert np.allclose(Q_, Vr.T @ Q)
        assert lhs_.shape == (r, k)
        assert np.allclose(lhs_, Vr.T @ lhs)
        assert isinstance(solver, opinf.lstsq.PlainSolver)

        # Without basis and with a one-dimensional input.
        rom = self.Dummy("cHB")
        Q_, lhs_, inputs, solver = rom._process_fit_arguments(
            Q, lhs, U1d, solver=0
        )
        assert rom.n is None
        assert rom.r == n
        assert rom.basis is None
        assert rom.m == 1
        assert Q_ is Q
        assert lhs_ is lhs
        assert inputs.shape == (1, k)
        assert np.all(inputs[0] == U1d)
        assert isinstance(solver, opinf.lstsq.PlainSolver)

        # With basis and no input.
        rom = self.Dummy("cA", Vr)
        Q_, lhs_, inputs, solver = rom._process_fit_arguments(
            Q, lhs, None, solver=1
        )
        assert rom.n == n
        assert rom.r == r
        assert rom.m == 0
        assert inputs is None
        assert np.allclose(Q_, Vr.T @ Q)
        assert np.allclose(lhs_, Vr.T @ lhs)
        assert isinstance(solver, opinf.lstsq.L2Solver)

        # With known operators for A.
        rom = self.Dummy(["c", A], Vr)
        Q_, lhs_, _, _ = rom._process_fit_arguments(Q, lhs, None)
        assert np.allclose(lhs_, Vr.T @ (lhs - (A.entries @ Vr @ Q_)))

        # With known operators for c and B.
        c_, B_ = _get_operators("cB", r, m)
        rom = self.Dummy([c_, "A", B_], Vr)
        Q_, lhs_, _, _ = rom._process_fit_arguments(Q, lhs, U)
        lhstrue = (Vr.T @ lhs) - B_.entries @ U - np.outer(c_.entries, ones)
        assert np.allclose(lhs_, lhstrue)

        # Special case: m = inputs.ndim = 1
        U1d = U[0]
        assert U1d.shape == (k,)
        B1d = opinf_operators.InputOperator(B_.entries[:, 0])
        assert B1d.shape == (r, 1)
        rom.operators = ["A", B1d]
        assert rom.m == 1
        Q_, lhs_, _, _ = rom._process_fit_arguments(Q, lhs, U1d)
        assert np.allclose(lhs_, (Vr.T @ lhs) - np.outer(B1d.entries, ones))

        # Fully intrusive.
        rom = self.Dummy([c, A], Vr)
        Q_, lhs_, _, _ = rom._process_fit_arguments(Q, lhs, None)
        assert Q_ is None
        assert lhs_ is None
        assert rom.operators[0].shape == (r,)
        assert rom.operators[1].shape == (r, r)

    def test_assemble_data_matrix(self, k=50, m=6, r=8):
        """Test _NonparametricROM._assemble_data_matrix()."""
        # Get test data.
        Q_, _, U = _get_data(r, k, m)

        rom = self.Dummy("c")
        rom.r = r
        for oplist, d in (
            ("c", 1),
            ("A", r),
            ("H", r * (r + 1) // 2),
            ("G", r * (r + 1) * (r + 2) // 6),
            ("B", m),
            ("N", r * m),
            ("cHB", 1 + r * (r + 1) // 2 + m),
            ("AGN", r * (m + 1) + r * (r + 1) * (r + 2) // 6),
        ):
            rom.operators = oplist
            if rom._has_inputs:
                rom.m = m
            D = rom._assemble_data_matrix(Q_, U)
            assert D.shape == (k, d)
            assert rom.d == d

        # Spot check.
        rom.operators = "cG"
        D = rom._assemble_data_matrix(Q_, U)
        assert D.shape == (k, 1 + r * (r + 1) * (r + 2) // 6)
        assert np.allclose(D[:, :1], np.ones((k, 1)))
        assert np.allclose(D[:, 1:], opinf.utils.kron3c(Q_).T)

        rom.operators = "AH"
        D = rom._assemble_data_matrix(Q_, U)
        assert D.shape == (k, r + r * (r + 1) // 2)
        assert np.allclose(D[:, :r], Q_.T)
        assert np.allclose(D[:, r:], opinf.utils.kron2c(Q_).T)

        rom.operators = "BN"
        D = rom._assemble_data_matrix(Q_, U)
        assert D.shape == (k, m * (1 + r))
        assert np.allclose(D[:, :m], U.T)
        assert np.allclose(D[:, m:], la.khatri_rao(U, Q_).T)

        # Try with one-dimensional inputs as a 1D array.
        rom.operators = "cB"
        rom.m = 1
        D = rom._assemble_data_matrix(Q_, U[0])
        assert D.shape == (k, 2)
        assert np.allclose(D, np.column_stack((np.ones(k), U[0])))

    def test_extract_operators(self, m=2, r=10):
        """Test _NonparametricROM._extract_operators()."""
        rom = self.Dummy("c")
        c, A, H, G, B, N = [
            op.entries for op in _get_operators("cAHGBN", r, m)
        ]

        rom.operators = "cH"
        rom.r = r
        Ohat = np.column_stack((c, H))
        rom._extract_operators(Ohat)
        assert np.allclose(rom.c_.entries, c)
        assert np.allclose(rom.H_.entries, H)

        rom.operators = "NA"
        rom.m = m
        Ohat = np.column_stack((N, A))
        rom._extract_operators(Ohat)
        assert np.allclose(rom.N_.entries, N)
        assert np.allclose(rom.A_.entries, A)

        rom.operators = "GB"
        Ohat = np.column_stack((G, B))
        rom._extract_operators(Ohat)
        assert np.allclose(rom.G_.entries, G)
        assert np.allclose(rom.B_.entries, B)

    def test_fit(self, n=20, k=50, m=4, r=6):
        """Test _NonparametricROM.fit()."""
        # Get test data.
        Q, F, U = _get_data(n, k, m)
        U1d = U[0, :]
        Vr = la.svd(Q)[0][:, :r]
        args_n = [Q, F]
        args_r = [Vr.T @ Q, Vr.T @ F]

        # Fit the rom with each modelform.
        for oplist in MODEL_FORMS:
            rom = self.Dummy(oplist)
            if "B" in oplist or "N" in oplist:
                # Without basis.
                rom.fit(*args_r, U)  # Two-dimensional inputs.
                rom.m = 1
                rom.fit(*args_r, U1d)  # One-dimensional inputs.
                # With basis.
                rom.basis = Vr
                rom.m = m
                rom.fit(*args_n, U)  # Two-dimensional inputs.
                rom.m = 1
                rom.fit(*args_n, U1d)  # One-dimensional inputs.

            else:
                rom.fit(*args_r)  # Without basis.
                rom.basis = Vr
                rom.fit(*args_n)  # With basis.

        # Special case: fully intrusive.
        A, B = _get_operators("AB", n, m)
        rom.basis = Vr
        rom.operators = [A, B]
        rom.m = m
        rom.fit(None, None, None)
        assert rom.solver_ is None
        romA_ = rom.A_
        assert romA_.entries is not None
        assert romA_.shape == (r, r)
        assert np.allclose(rom.A_.entries, Vr.T @ A[:] @ Vr)
        romB_ = rom.B_
        assert romB_.entries is not None
        assert romB_.shape == (r, m)
        assert np.allclose(romB_.entries, Vr.T @ B[:])

    # Model persistence -------------------------------------------------------
    def test_save(self, n=15, m=2, r=3, target="_savemodeltest.h5"):
        """Test _NonparametricROM.save()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = np.random.random((n, r))
        rom = self.Dummy("cAHGB", Vr)
        rom.save(target, save_basis=False)
        assert os.path.isfile(target)

        with pytest.raises(FileExistsError) as ex:
            rom.save(target, overwrite=False)
        assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

        rom.save(target, save_basis=True, overwrite=True)

        rom.operators = "c"
        rom.save(target, overwrite=True)

        rom.operators = "AB"
        rom.basis = None
        rom.save(target, save_basis=True, overwrite=True)

        os.remove(target)

    def test_load(self, n=20, m=2, r=5, target="_loadmodeltest.h5"):
        """Test _NonparametricROM.load()."""
        # Clean up after old tests if needed.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = np.random.random((n, r))

        # Make an empty HDF5 file to start with.
        with h5py.File(target, "w"):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            rom = self.Dummy.load(target)
        assert ex.value.args[0].endswith("(object 'meta' doesn't exist)")

        with h5py.File(target, "a") as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_operators"] = 1

        # Check that save() and load() are inverses.
        rom = self.Dummy("cAN", Vr)
        rom.m = m
        rom.save(target, save_basis=True, overwrite=True)
        rom2 = rom.load(target)
        assert rom2 is not rom
        assert rom2.basis == rom.basis
        for attr in [
            "n",
            "m",
            "r",
            "_indices_of_operators_to_infer",
            "_indices_of_known_operators",
            "__class__",
        ]:
            assert getattr(rom2, attr) == getattr(rom, attr)
        for name in "cAN":
            attr = getattr(rom2, f"{name}_")
            assert attr is not None
            assert attr.entries is None
        for name in "HGB":
            assert getattr(rom2, f"{name}_") is None

        # Check basis = None functionality.
        rom = self.Dummy("HGB")
        rom.r = r
        rom.m = m
        rom.save(target, overwrite=True)
        rom2 = rom.load(target)
        assert rom2 is not rom
        assert rom2.basis is None
        for attr in [
            "n",
            "m",
            "r",
            "_indices_of_operators_to_infer",
            "_indices_of_known_operators",
            "__class__",
        ]:
            assert getattr(rom2, attr) == getattr(rom, attr)
        for name in "cAN":
            assert getattr(rom2, f"{name}_") is None
        for name in "HGB":
            attr = getattr(rom2, f"{name}_")
            assert attr is not None
            assert attr.entries is None

        # Load ROM with operators with entries.
        rom = self.Dummy(_get_operators("AG", r, m), Vr)
        rom.save(target, overwrite=True)
        rom2 = rom.load(target)
        assert rom2 is not rom
        assert rom2.basis == rom.basis
        for attr in [
            "n",
            "m",
            "r",
            "_indices_of_operators_to_infer",
            "_indices_of_known_operators",
            "__class__",
        ]:
            assert getattr(rom2, attr) == getattr(rom, attr)
        for name in "cHBN":
            assert getattr(rom2, f"{name}_") is None
        for name in "AG":
            attr2 = getattr(rom2, f"{name}_")
            assert attr2 is not None
            attr = getattr(rom, f"{name}_")
            assert attr2.entries.shape == attr.entries.shape
            assert np.all(attr2.entries == attr.entries)

        # Clean up.
        os.remove(target)
