# models/monolithic/nonparametric/test_base.py
"""Tests for models.nonparametric._base."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la

import opinf

from .. import MODEL_FORMS, _get_data, _get_operators


opinf_operators = opinf.operators_new  # TEMP
_module = opinf.models.monolithic.nonparametric._base
kron2c = opinf.operators_new.QuadraticOperator.ckron
kron3c = opinf.operators_new.CubicOperator.ckron


class TestNonparametricMonolithicModel:
    """Test models.nonparametric._base._NonparametricMonolithicModel."""

    class Dummy(_module._NonparametricMonolithicModel):
        """Instantiable version of _NonparametricMonolithicModel."""

        _LHS_ARGNAME = "mylhs"
        _LHS_LABEL = "qdot"
        _STATE_LABEL = "qq"
        _INPUT_LABEL = "uu"

        def predict(*args, **kwargs):
            pass

    # Properties: operators ---------------------------------------------------
    def test_operators(self):
        """Test _NonparametricMonolithicModel.operators
        (_operator_abbreviations, _isvalidoperator(),
        _check_operator_types_unique()).
        """
        # Try with duplicate (nonintrusive) operator types.

        with pytest.raises(ValueError) as ex:
            self.Dummy("AA")
        assert (
            ex.value.args[0] == "duplicate type in list of operators to infer"
        )

        # Test __init__() shortcuts.
        model = self.Dummy("cHB")
        assert len(model.operators) == 3
        for i in range(3):
            assert model.operators[i].entries is None
        assert isinstance(model.operators[0], opinf_operators.ConstantOperator)
        assert isinstance(
            model.operators[1], opinf_operators.QuadraticOperator
        )
        assert isinstance(model.operators[2], opinf_operators.InputOperator)

        model.operators = [opinf_operators.ConstantOperator(), "A", "N"]
        assert len(model.operators) == 3
        for i in range(3):
            assert model.operators[i].entries is None
        assert isinstance(model.operators[0], opinf_operators.ConstantOperator)
        assert isinstance(model.operators[1], opinf_operators.LinearOperator)
        assert isinstance(
            model.operators[2], opinf_operators.StateInputOperator
        )

    def test_get_operator_of_type(self, m=4, r=7):
        """Test _NonparametricMonolithicModel._get_operator_of_type()
        and the [caHGBN]_ properties.
        """
        [c, A, H, B, N] = _get_operators("cAHBN", r, m)
        model = self.Dummy([A, B, c, H, N])

        assert model.A_ is model.operators[0]
        assert model.B_ is model.operators[1]
        assert model.c_ is model.operators[2]
        assert model.H_ is model.operators[3]
        assert model.N_ is model.operators[4]
        assert model.G_ is None

    # String representation ---------------------------------------------------
    def test_str(self):
        """Test _NonparametricMonolithicModel.__str__()."""

        # Continuous Models
        model = self.Dummy("A")
        assert str(model) == "Model structure: qdot = Aqq"
        model = self.Dummy("cA")
        assert str(model) == "Model structure: qdot = c + Aqq"
        model = self.Dummy("HB")
        assert str(model) == "Model structure: qdot = H[qq ⊗ qq] + Buu"
        model = self.Dummy("G")
        assert str(model) == "Model structure: qdot = G[qq ⊗ qq ⊗ qq]"
        model = self.Dummy("cH")
        assert str(model) == "Model structure: qdot = c + H[qq ⊗ qq]"

        # Dimension reporting.
        model = self.Dummy("A")
        model.state_dimension = 20
        modelstr = str(model).split("\n")
        assert len(modelstr) == 2
        assert modelstr[0] == "Model structure: qdot = Aqq"
        assert modelstr[1] == "State dimension r = 20"

        model = self.Dummy("cB")
        model.state_dimension = 10
        model.input_dimension = 3
        modelstr = str(model).split("\n")
        assert len(modelstr) == 3
        assert modelstr[0] == "Model structure: qdot = c + Buu"
        assert modelstr[1] == "State dimension r = 10"
        assert modelstr[2] == "Input dimension m = 3"

    def test_repr(self):
        """Test _NonparametricMonolithicModel.__repr__()."""

        def firstline(obj):
            return repr(obj).split("\n")[0]

        assert firstline(self.Dummy("A")).startswith("<Dummy object at")

    # Properties: operator inference ------------------------------------------
    def test_operator_matrix_(self, r=15, m=3):
        """Test _NonparametricMonolithicModel.operator_matrix_."""
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
        """Test _NonparametricMonolithicModel.data_matrix_
        (_assemble_data_matrix(), operator_matrix_dimension).
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
        Dtrue = np.column_stack([kron2c(Q).T, kron3c(Q).T])
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

        with pytest.warns(UserWarning) as wn:
            model._fit_solver(Q, Qdot, inputs=U)
        assert (
            wn[0].message.args[0] == "all operators initialized "
            "intrusively, nothing to learn"
        )

        assert model.data_matrix_ is None

    # Fitting -----------------------------------------------------------------
    def test_process_fit_arguments(self, k=50, m=4, r=6):
        """Test _NonparametricMonolithicModel._process_fit_arguments()."""
        # Get test data.
        Q, lhs, U = _get_data(r, k, m)
        A, B = _get_operators("AB", r, m)
        U1d = U[0, :]
        ones = np.ones(k)

        # Exceptions #

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
        with pytest.warns(UserWarning) as wn:
            Q_, lhs_, _, _ = model._process_fit_arguments(Q, lhs, U)
        assert (
            wn[0].message.args[0] == "all operators initialized "
            "intrusively, nothing to learn"
        )
        assert Q_ is None
        assert lhs_ is None
        assert model.operators[0] is A
        assert model.operators[1] is B

    def test_assemble_data_matrix(self, k=50, m=6, r=8):
        """Test _NonparametricMonolithicModel._assemble_data_matrix()."""
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
        assert np.allclose(D[:, 1:], kron3c(Q_).T)

        model.operators = "AH"
        D = model._assemble_data_matrix(Q_, U)
        assert D.shape == (k, r + r * (r + 1) // 2)
        assert np.allclose(D[:, :r], Q_.T)
        assert np.allclose(D[:, r:], kron2c(Q_).T)

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
        """Test _NonparametricMonolithicModel._extract_operators()."""
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
        """Test _NonparametricMonolithicModel.fit()."""
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

        with pytest.warns(UserWarning) as wn:
            model.fit(None, None, None)
        assert (
            wn[0].message.args[0] == "all operators initialized "
            "intrusively, nothing to learn"
        )

        assert model.solver_ is None
        modelA_ = model.A_
        assert modelA_.entries is not None
        assert modelA_.shape == (r, r)
        assert np.allclose(model.A_.entries, A[:])
        modelB_ = model.B_
        assert modelB_.entries is not None
        assert modelB_.shape == (r, m)
        assert np.allclose(modelB_.entries, B[:])

    # Model evaluation --------------------------------------------------------
    def test_rhs(self, m=2, k=10, r=5, ntrials=10):
        """Test _NonparametricMonolithicModel.rhs()."""
        c_, A_, H_, B_ = _get_operators("cAHB", r, m)

        model = self.Dummy([c_, A_])
        for _ in range(ntrials):
            q_ = np.random.random(r)
            y_ = c_.entries + A_.entries @ q_
            out = model.rhs(q_)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random((r, k))
            Y_ = c_.entries.reshape((r, 1)) + A_.entries @ Q_
            out = model.rhs(Q_)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

        model = self.Dummy([H_, B_])
        for _ in range(ntrials):
            u = np.random.random(m)
            q_ = np.random.random(r)
            y_ = H_.entries @ kron2c(q_) + B_.entries @ u
            out = model.rhs(q_, u)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random((r, k))
            U = np.random.random((m, k))
            Y_ = H_.entries @ kron2c(Q_) + B_.entries @ U
            out = model.rhs(Q_, U)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

        # Special case: r = 1, q is a scalar.
        model = self.Dummy(_get_operators("A", 1))
        a = model.operators[0].entries[0]
        assert model.state_dimension == 1
        for _ in range(ntrials):
            q_ = np.random.random()
            y_ = a * q_
            out = model.rhs(q_)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random(k)
            Y_ = a[0] * Q_
            out = model.rhs(Q_, U)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

    def test_jacobian(self, r=5, m=2, ntrials=10):
        """Test _NonparametricMonolithicModel.jacobian()."""
        c_, A_, B_ = _get_operators("cAB", r, m)

        for oplist in ([c_, A_], [c_, A_, B_]):
            model = self.Dummy(oplist)
            q_ = np.random.random(r)
            out = model.jacobian(q_)
            assert out.shape == (r, r)
            assert np.allclose(out, A_.entries)

        # Special case: r = 1, q a scalar.
        model = self.Dummy(_get_operators("A", 1))
        q_ = np.random.random()
        out = model.jacobian(q_)
        assert out.shape == (1, 1)
        assert out[0, 0] == model.operators[0].entries[0, 0]

    # Model persistence -------------------------------------------------------
    def test_save(self, m=2, r=3, target="_savemodeltest.h5"):
        """Test _NonparametricMonolithicModel.save()."""
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
        """Test _NonparametricMonolithicModel.load()."""
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
