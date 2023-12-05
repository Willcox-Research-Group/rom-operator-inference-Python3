# models/nonparametric/test_base.py
"""Tests for models.nonparametric._base."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la

import opinf

from .. import MODEL_FORMS, _get_data, _get_operators


opinf_operators = opinf.operators_new  # TEMP


class TestNonparametricModel:
    """Test models.nonparametric._base._NonparametricModel."""

    class Dummy(opinf.models.nonparametric._base._NonparametricModel):
        """Instantiable version of _NonparametricModel."""

        _LHS_ARGNAME = "mylhs"

        def predict(*args, **kwargs):
            pass

    # Properties --------------------------------------------------------------
    def test_operator_matrix_(self, r=15, m=3):
        """Test _NonparametricModel.operator_matrix_."""
        c, A, H, G, B, N = _get_operators("cAHGBN", r, m)

        model = self.Dummy("cA")
        model.state_dimension = r
        model.operators[:] = (c, A)
        D = np.column_stack([c.entries, A.entries])
        assert np.all(model.operator_matrix_ == D)

        model.operators[:] = (H, B)
        model._has_inputs = True
        model.input_dimension = m
        D = np.column_stack([H.entries, B.entries])
        assert np.all(model.operator_matrix_ == D)

        model.operators[:] = [G, N]
        D = np.column_stack([G.entries, N.entries])
        assert np.all(model.operator_matrix_ == D)

    def test_data_matrix_(self, k=500, m=20, r=10):
        """Test _NonparametricModel.data_matrix_, i.e., spot check
        _NonparametricModel._assemble_data_matrix().
        """
        Q, Qdot, U = _get_data(r, k, m)

        model = self.Dummy("cAH")
        with pytest.raises(AttributeError) as ex:
            D = model.data_matrix_
        assert ex.value.args[0] == "data matrix not constructed (call fit())"
        assert model.operator_matrix_dimension is None

        model.operators = "A"
        model._fit_solver(Q, Qdot, inputs=None)
        D = model.data_matrix_
        assert D.shape == (k, r)
        assert np.all(D == Q.T)
        assert model.operator_matrix_dimension == r

        model.operators = "B"
        model._fit_solver(Q, Qdot, inputs=U)
        D = model.data_matrix_
        assert D.shape == (k, m)
        assert np.all(D == U.T)
        assert model.operator_matrix_dimension == m

        model.operators = "HG"
        model._fit_solver(Q, Qdot, inputs=None)
        Dtrue = np.column_stack(
            [opinf.utils.kron2c(Q).T, opinf.utils.kron3c(Q).T]
        )
        D = model.data_matrix_
        d = r * (r + 1) // 2 + r * (r + 1) * (r + 2) // 6
        assert D.shape == (k, d)
        assert np.allclose(D, Dtrue)
        assert model.operator_matrix_dimension == d

        model.operators = "c"
        model._fit_solver(Q, Qdot, inputs=None)
        D = model.data_matrix_
        assert D.shape == (k, 1)
        assert np.all(D == 1)
        assert model.operator_matrix_dimension == 1

        # Partially intrusive case.
        c = opinf_operators.ConstantOperator(np.ones(r))
        model.operators = ["A", c]
        model._fit_solver(Q, Qdot, inputs=None)
        D = model.data_matrix_
        assert D.shape == (k, r)
        assert np.all(D == Q.T)

        # Fully intrusive case.
        model.operators = _get_operators("cHB", r, m)
        model._fit_solver(Q, Qdot, inputs=U)
        assert model.data_matrix_ is None

    # Fitting -----------------------------------------------------------------
    def test_process_fit_arguments(self, k=50, m=4, r=6):
        """Test _NonparametricModel._process_fit_arguments()."""
        # Get test data.
        Q, lhs, U = _get_data(r, k, m)
        A, B = _get_operators("AB", r, m)
        U1d = U[0, :]
        ones = np.ones(k)

        # Exceptions #

        # Try with bad solver option.
        model = self.Dummy("AB")
        with pytest.raises(ValueError) as ex:
            model._process_fit_arguments(None, None, None, solver=-1)
        assert ex.value.args[0] == "if a scalar, `solver` must be nonnegative"

        class _DummySolver:
            pass

        with pytest.raises(TypeError) as ex:
            model._process_fit_arguments(None, None, None, solver=_DummySolver)
        assert ex.value.args[0] == "solver must be an instance, not a class"

        with pytest.raises(TypeError) as ex:
            model._process_fit_arguments(
                None, None, None, solver=_DummySolver()
            )
        assert ex.value.args[0] == "solver must have a 'fit()' method"

        # States do not match dimensions 'r'.
        model = self.Dummy(["c", A])
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(Q[1:], None, None)
        assert ex.value.args[0] == f"states.shape[0] = {r-1} != r = {r}"

        # LHS not aligned with states.
        model = self.Dummy([A, "B"])
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(Q, lhs[:, :-1], U)
        assert (
            ex.value.args[0] == f"{self.Dummy._LHS_ARGNAME}.shape[-1] = {k-1} "
            f"!= {k} = states.shape[-1]"
        )

        # Inputs do not match dimension 'm'.
        model.operators = ["A", B]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(Q, lhs, U[1:])
        assert ex.value.args[0] == f"inputs.shape[0] = {m-1} != {m} = m"

        # Inputs not aligned with states.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(Q, lhs, U[:, :-1])
        assert (
            ex.value.args[0] == f"inputs.shape[-1] = {k-1} "
            f"!= {k} = states.shape[-1]"
        )

        # Correct usage #

        # With input.
        model.operators = "AB"
        Q_, lhs_, U_, solver = model._process_fit_arguments(Q, lhs, U)
        assert model.state_dimension == r
        assert model.input_dimension == m
        assert Q_ is Q
        assert lhs_ is lhs
        assert U_ is U
        assert isinstance(solver, opinf.lstsq.PlainSolver)

        # With a one-dimensional input.
        model = self.Dummy("cHB")
        Q_, lhs_, inputs, solver = model._process_fit_arguments(
            Q, lhs, U1d, solver=0
        )
        assert model.state_dimension == r
        assert model.input_dimension == 1
        assert Q_ is Q
        assert lhs_ is lhs
        assert inputs.shape == (1, k)
        assert np.all(inputs[0] == U1d)
        assert isinstance(solver, opinf.lstsq.PlainSolver)

        # Without input.
        model = self.Dummy("cA")
        Q_, lhs_, inputs, solver = model._process_fit_arguments(
            Q, lhs, None, solver=1
        )
        assert model.state_dimension == r
        assert model.input_dimension == 0
        assert inputs is None
        assert Q_ is Q
        assert lhs_ is lhs
        assert isinstance(solver, opinf.lstsq.L2Solver)

        # With known operators for A.
        model = self.Dummy(["c", A])
        Q_, lhs_, _, _ = model._process_fit_arguments(Q, lhs, None)
        assert np.allclose(lhs_, lhs - (A.entries @ Q))

        # With known operators for c and B.
        c_, B_ = _get_operators("cB", r, m)
        model = self.Dummy([c_, "A", B_])
        Q_, lhs_, _, _ = model._process_fit_arguments(Q, lhs, U)
        lhstrue = lhs - B_.entries @ U - np.outer(c_.entries, ones)
        assert np.allclose(lhs_, lhstrue)

        # Special case: m = inputs.ndim = 1
        U1d = U[0]
        assert U1d.shape == (k,)
        B1d = opinf_operators.InputOperator(B_.entries[:, 0])
        assert B1d.shape == (r, 1)
        model.operators = ["A", B1d]
        assert model.input_dimension == 1
        Q_, lhs_, _, _ = model._process_fit_arguments(Q, lhs, U1d)
        assert np.allclose(lhs_, lhs - np.outer(B1d.entries, ones))

        # Fully intrusive.
        model = self.Dummy([A, B])
        Q_, lhs_, _, _ = model._process_fit_arguments(Q, lhs, U)
        assert Q_ is None
        assert lhs_ is None
        assert model.operators[0] is A
        assert model.operators[1] is B

    def test_assemble_data_matrix(self, k=50, m=6, r=8):
        """Test _NonparametricModel._assemble_data_matrix()."""
        # Get test data.
        Q_, _, U = _get_data(r, k, m)

        for opkeys, d in (
            ("c", 1),
            ("A", r),
            ("H", r * (r + 1) // 2),
            ("G", r * (r + 1) * (r + 2) // 6),
            ("B", m),
            ("N", r * m),
            ("cHB", 1 + r * (r + 1) // 2 + m),
            ("AGN", r * (m + 1) + r * (r + 1) * (r + 2) // 6),
        ):
            model = self.Dummy(opkeys)
            model.state_dimension = r
            if model._has_inputs:
                model.input_dimension = m
            D = model._assemble_data_matrix(Q_, U)
            assert D.shape == (k, d)
            assert model.operator_matrix_dimension == d

        # Spot check.
        model.operators = "cG"
        D = model._assemble_data_matrix(Q_, U)
        assert D.shape == (k, 1 + r * (r + 1) * (r + 2) // 6)
        assert np.allclose(D[:, :1], np.ones((k, 1)))
        assert np.allclose(D[:, 1:], opinf.utils.kron3c(Q_).T)

        model.operators = "AH"
        D = model._assemble_data_matrix(Q_, U)
        assert D.shape == (k, r + r * (r + 1) // 2)
        assert np.allclose(D[:, :r], Q_.T)
        assert np.allclose(D[:, r:], opinf.utils.kron2c(Q_).T)

        model.operators = "BN"
        D = model._assemble_data_matrix(Q_, U)
        assert D.shape == (k, m * (1 + r))
        assert np.allclose(D[:, :m], U.T)
        assert np.allclose(D[:, m:], la.khatri_rao(U, Q_).T)

        # Try with one-dimensional inputs as a 1D array.
        model.operators = "cB"
        model.input_dimension = 1
        D = model._assemble_data_matrix(Q_, U[0])
        assert D.shape == (k, 2)
        assert np.allclose(D, np.column_stack((np.ones(k), U[0])))

    def test_extract_operators(self, m=2, r=10):
        """Test _NonparametricModel._extract_operators()."""
        model = self.Dummy("c")
        c, A, H, G, B, N = [
            op.entries for op in _get_operators("cAHGBN", r, m)
        ]

        model.operators = "cH"
        model.state_dimension = r
        Ohat = np.column_stack((c, H))
        model._extract_operators(Ohat)
        assert np.allclose(model.c_.entries, c)
        assert np.allclose(model.H_.entries, H)

        model.operators = "NA"
        model.state_dimension = r
        model.input_dimension = m
        Ohat = np.column_stack((N, A))
        model._extract_operators(Ohat)
        assert np.allclose(model.N_.entries, N)
        assert np.allclose(model.A_.entries, A)

        model.operators = "GB"
        model.state_dimension = r
        model.input_dimension = m
        Ohat = np.column_stack((G, B))
        model._extract_operators(Ohat)
        assert np.allclose(model.G_.entries, G)
        assert np.allclose(model.B_.entries, B)

    def test_fit(self, k=50, m=4, r=6):
        """Test _NonparametricModel.fit()."""
        # Get test data.
        Q, F, U = _get_data(r, k, m)
        U1d = U[0, :]
        args = [Q, F]

        # Fit the model with each modelform.
        for oplist in MODEL_FORMS:
            model = self.Dummy(oplist)
            if "B" in oplist or "N" in oplist:
                model.fit(*args, U)  # Two-dimensional inputs.
                model.fit(*args, U1d)  # One-dimensional inputs.
            else:
                model.fit(*args)  # No inputs.

        # Special case: fully intrusive.
        A, B = _get_operators("AB", r, m)
        model.operators = [A, B]
        model.input_dimension = m
        model.fit(None, None, None)
        assert model.solver_ is None
        modelA_ = model.A_
        assert modelA_.entries is not None
        assert modelA_.shape == (r, r)
        assert np.allclose(model.A_.entries, A[:])
        modelB_ = model.B_
        assert modelB_.entries is not None
        assert modelB_.shape == (r, m)
        assert np.allclose(modelB_.entries, B[:])

    # Model persistence -------------------------------------------------------
    def test_save(self, m=2, r=3, target="_savemodeltest.h5"):
        """Test _NonparametricModel.save()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        model = self.Dummy("cAHGB")
        model.save(target)
        assert os.path.isfile(target)

        with pytest.raises(FileExistsError) as ex:
            model.save(target, overwrite=False)
        assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

        model.save(target, overwrite=True)

        model.operators = "AB"
        model.save(target, overwrite=True)

        os.remove(target)

    def test_load(self, n=20, m=2, r=5, target="_loadmodeltest.h5"):
        """Test _NonparametricModel.load()."""
        # Clean up after old tests if needed.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Make an empty HDF5 file to start with.
        with h5py.File(target, "w"):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            model = self.Dummy.load(target)
        assert ex.value.args[0].endswith("(object 'meta' doesn't exist)")

        with h5py.File(target, "a") as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_operators"] = 1

        # Check that save() and load() are inverses.
        model = self.Dummy("cAN")
        model.input_dimension = m
        model.save(target, overwrite=True)
        model2 = model.load(target)
        assert model2 is not model
        for attr in [
            "input_dimension",
            "state_dimension",
            "_indices_of_operators_to_infer",
            "_indices_of_known_operators",
            "__class__",
        ]:
            assert getattr(model2, attr) == getattr(model, attr)
        for name in "cAN":
            attr = getattr(model2, f"{name}_")
            assert attr is not None
            assert attr.entries is None
        for name in "HGB":
            assert getattr(model2, f"{name}_") is None

        # Load model with operators with entries.
        model = self.Dummy(_get_operators("AG", r, m))
        model.save(target, overwrite=True)
        model2 = model.load(target)
        assert model2 is not model
        for attr in [
            "input_dimension",
            "state_dimension",
            "_indices_of_operators_to_infer",
            "_indices_of_known_operators",
            "__class__",
        ]:
            assert getattr(model2, attr) == getattr(model, attr)
        for name in "cHBN":
            assert getattr(model2, f"{name}_") is None
        for name in "AG":
            attr2 = getattr(model2, f"{name}_")
            assert attr2 is not None
            attr = getattr(model, f"{name}_")
            assert attr2.entries.shape == attr.entries.shape
            assert np.all(attr2.entries == attr.entries)

        # Clean up.
        os.remove(target)
