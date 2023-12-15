# models/mono/test_nonparametric.py
"""Tests for models.mono._nonparametric."""

import os
import h5py
import pytest
import numpy as np
from scipy import linalg as la

import opinf

from . import MODEL_FORMS, _get_data, _get_operators, _trainedmodel


_module = opinf.models.mono._nonparametric
kron2c = opinf.operators.QuadraticOperator.ckron
kron3c = opinf.operators.CubicOperator.ckron


class TestNonparametricModel:
    """Test models.nonparametric._base._NonparametricModel."""

    class Dummy(_module._NonparametricModel):
        """Instantiable version of _NonparametricModel."""

        _LHS_ARGNAME = "mylhs"
        _LHS_LABEL = "qdot"
        _STATE_LABEL = "qq"
        _INPUT_LABEL = "uu"

        def predict(*args, **kwargs):
            pass

    # Properties: operators ---------------------------------------------------
    def test_operators(self):
        """Test _NonparametricModel.operators
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
        assert isinstance(model.operators[0], opinf.operators.ConstantOperator)
        assert isinstance(
            model.operators[1], opinf.operators.QuadraticOperator
        )
        assert isinstance(model.operators[2], opinf.operators.InputOperator)

        model.operators = [opinf.operators.ConstantOperator(), "A", "N"]
        assert len(model.operators) == 3
        for i in range(3):
            assert model.operators[i].entries is None
        assert isinstance(model.operators[0], opinf.operators.ConstantOperator)
        assert isinstance(model.operators[1], opinf.operators.LinearOperator)
        assert isinstance(
            model.operators[2], opinf.operators.StateInputOperator
        )

    def test_get_operator_of_type(self, m=4, r=7):
        """Test _NonparametricModel._get_operator_of_type()
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
        """Test _NonparametricModel.__str__()."""

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
        """Test _NonparametricModel.__repr__()."""

        def firstline(obj):
            return repr(obj).split("\n")[0]

        assert firstline(self.Dummy("A")).startswith("<Dummy object at")

    # Properties: operator inference ------------------------------------------
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
        """Test _NonparametricModel.data_matrix_
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
        c = opinf.operators.ConstantOperator(np.ones(r))
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
        """Test _NonparametricModel._process_fit_arguments()."""
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
        B1d = opinf.operators.InputOperator(B_.entries[:, 0])
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
        """Test _NonparametricModel.rhs()."""
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
        """Test _NonparametricModel.jacobian()."""
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


class TestSteadyModel:
    """Test models.monolithic.nonparametric._public.SteadyModel."""

    ModelClass = _module.SteadyModel

    def test_rhs(self, r=10):
        """Lightly test SteadyModel.rhs().
        Stronger tests in test_base.TestNonparametricModel.test_rhs().
        """
        model = _trainedmodel(self.ModelClass, "cAH", r, 0)
        model.rhs(np.random.random(r))

    def test_jacobian(self, r=6, m=3):
        """Lightly test DiscreteModel.jacobian().
        Stronger tests in test_base.TestNonparametricModel.test_jacobian().
        """
        model = _trainedmodel(self.ModelClass, "A", r, 0)
        model.jacobian(np.random.random(r))

    def test_fit(self, k=400, r=10):
        """Lightly test SteadyModel.fit().
        Stronger tests in test_base.TestNonparametricModel.test_fit().
        """
        Q, F, _ = _get_data(r, k, 2)
        model = self.ModelClass("A")
        model.fit(Q, F)

    # def test_predict(self):
    #     """Test SteadyModel.predict()."""
    #     raise NotImplementedError


class TestDiscreteModel:
    """Test models.monolithic.nonparametric._public.DiscreteModel."""

    ModelClass = _module.DiscreteModel

    def test_stack_trajectories(self, r=10, k=20, m=5, num_trajectories=4):
        """Test DiscreteModel.stack_trajectories()."""
        statelist, inputlist = [], []
        for _ in range(num_trajectories):
            Q, _, U = _get_data(r, k, m)
            statelist.append(Q)
            inputlist.append(U)

        Qs, Qnexts = self.ModelClass.stack_trajectories(statelist)
        assert Qs.shape == (r, (k - 1) * num_trajectories)
        assert Qnexts.shape == Qs.shape
        Qs_split = np.split(Qs, num_trajectories, axis=1)
        Qnexts_split = np.split(Qnexts, num_trajectories, axis=1)
        for i in range(num_trajectories):
            assert np.all(Qs_split[i][:, 1:] == Qnexts_split[i][:, :-1])
            assert np.all(Qs_split[i] == statelist[i][:, :-1])
            assert np.all(Qnexts_split[i] == statelist[i][:, 1:])

        Qs2, Qnexts2, Us = self.ModelClass.stack_trajectories(
            statelist,
            inputlist,
        )
        assert Qs2.shape == Qs.shape
        assert np.all(Qs2 == Qs)
        assert Qnexts2.shape == Qnexts.shape
        assert np.all(Qnexts2 == Qnexts)
        assert Us.shape == (m, (k - 1) * num_trajectories)
        for i, Usplit in enumerate(np.split(Us, num_trajectories, axis=1)):
            assert np.all(Usplit == inputlist[i][:, : (k - 1)])

        # 1D inputs
        inputlist_1d = [np.random.random(k) for _ in range(num_trajectories)]
        Qs3, Qnexts3, Us_1d = self.ModelClass.stack_trajectories(
            statelist,
            inputlist_1d,
        )
        assert Qs3.shape == Qs.shape
        assert np.all(Qs3 == Qs)
        assert Qnexts3.shape == Qnexts.shape
        assert np.all(Qnexts3 == Qnexts)
        assert Us_1d.shape == ((k - 1) * num_trajectories,)
        for i, Usplit in enumerate(np.split(Us_1d, num_trajectories, axis=0)):
            assert np.all(Usplit == inputlist_1d[i][: (k - 1)])

    def test_rhs(self, r=6, m=3):
        """Lightly test DiscreteModel.rhs().
        Stronger tests in test_base.TestNonparametricModel.test_rhs().
        """
        model = _trainedmodel(self.ModelClass, "cG", r, 0)
        model.rhs(np.random.random(r))

    def test_jacobian(self, r=6, m=3):
        """Lightly test DiscreteModel.jacobian().
        Stronger tests in test_base.TestNonparametricModel.test_jacobian().
        """
        model = _trainedmodel(self.ModelClass, "cG", r, 0)
        model.jacobian(np.random.random(r))

    def test_fit(self, k=50, r=5, m=3):
        """Test DiscreteModel.fit()."""
        Q, _, U = _get_data(r, k, m)
        Qnext = Q[:, 1:]
        model = self.ModelClass("A").fit(Q)
        model2 = self.ModelClass("A").fit(Q[:, :-1], Qnext)
        assert model.A_ == model2.A_

        model = self.ModelClass("AB").fit(Q, inputs=U)
        model2 = self.ModelClass("AB").fit(Q[:, :-1], Qnext, inputs=U)
        assert model.A_ == model2.A_
        assert model.B_ == model2.B_

    def test_predict(self, k=20, m=6, r=4):
        """Test DiscreteModel.predict()."""
        # Get test data.
        Q = _get_data(r, k, m)[0]
        niters = 5
        q0 = Q[:, 0]
        U = np.ones((m, niters - 1))

        ops = A_, B_ = _get_operators("AB", r, m)
        B1d_ = _get_operators("B", r, m=1)[0]
        model = self.ModelClass(ops)

        # Try to predict with invalid initial condition.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0[1:], None)
        assert ex.value.args[0] == (
            "initial condition not aligned with model "
            f"(state0.shape = ({r-1},) != ({r},) = (r,))"
        )

        # Try to predict with bad niters argument.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, -18, U)
        assert (
            ex.value.args[0] == "argument 'niters' must be a positive integer"
        )

        # Try to predict with badly-shaped discrete inputs.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, niters, np.random.random((m - 1, niters - 1)))
        assert (
            ex.value.args[0] == f"inputs.shape = ({(m-1, niters-1)} "
            f"!= {(m, niters-1)} = (m, niters-1)"
        )

        model_m1 = self.ModelClass([A_, B1d_])
        with pytest.raises(ValueError) as ex:
            model_m1.predict(q0, niters, np.random.random((2, niters - 1)))
        assert (
            ex.value.args[0] == f"inputs.shape = ({(2, niters-1)} "
            f"!= {(1, niters-1)} = (m, niters-1)"
        )

        # Try to predict with continuous inputs.
        with pytest.raises(TypeError) as ex:
            model.predict(q0, niters, lambda t: np.ones(m - 1))
        assert ex.value.args[0] == "inputs must be NumPy array, not callable"

        # No control inputs.
        model = self.ModelClass([A_])
        out = model.predict(q0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)

        # With 2D inputs.
        model = self.ModelClass([A_, B_])
        out = model.predict(q0, niters, U)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)

        # With 1D inputs.
        model = self.ModelClass([A_, B1d_])
        out = model.predict(q0, niters, np.ones(niters))
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)


class TestContinuousModel:
    """Test models.monolithic.nonparametric._public.ContinuousModel."""

    ModelClass = _module.ContinuousModel

    def test_rhs(self, r=5, m=2):
        """Test ContinuousModel.rhs()."""
        A_, B_ = _get_operators("AB", r, m)

        model = self.ModelClass([A_])
        q_ = np.random.random(r)
        model.rhs(10, q_)

        model = self.ModelClass([A_, B_])
        with pytest.raises(TypeError) as ex:
            model.rhs(5, q_, 10)
        assert "object is not callable" in ex.value.args[0]

        def input_func(t):
            return np.random.random(m)

        model.rhs(np.pi, q_, input_func)

    def test_jacobian(self, r=6, m=3):
        """Test ContinuousModel.jacobian()."""
        A_, B_ = _get_operators("AB", r, m)

        model = self.ModelClass([A_])
        q_ = np.random.random(r)
        model.jacobian(8, q_)

        model = self.ModelClass([A_, B_])
        with pytest.raises(TypeError) as ex:
            model.jacobian(5, q_, 10)
        assert "object is not callable" in ex.value.args[0]

        def input_func(t):
            return np.random.random(m)

        model.jacobian(2, q_, input_func)

    def test_fit(self, k=20, m=3, r=4):
        """Lightly test ContinuousModel.fit().
        Stronger tests in test_base.TestNonparametricModel.test_fit().
        """
        Q, Qdot, U = _get_data(r, k, m)
        self.ModelClass("AB").fit(Q, Qdot, U)

    def test_predict(self, k=50, m=10, r=6):
        """Test ContinuousModel.predict()."""
        # Get test data.
        Q = _get_data(r, k, m)[0]
        nt = 5
        q0 = Q[:, 0]
        t = np.linspace(0, 0.01 * nt, nt)

        def input_func(tt):
            return tt * np.ones(m)

        Upred = np.column_stack([input_func(tt) for tt in t])

        # Try to predict with invalid initial condition.
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0[1:], t, input_func)
        assert ex.value.args[0] == (
            "initial condition not aligned with model "
            f"(state0.shape = ({r-1},) != ({r},) = (r,))"
        )

        # Try to predict with bad time array.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, np.vstack((t, t)), input_func)
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Predict without inputs.
        for form in MODEL_FORMS:
            if "B" not in form and "N" not in form:
                model = _trainedmodel(self.ModelClass, form, r, None)
                out = model.predict(q0, t)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, t.size)

        # Predict with no basis gives result in low-dimensional space.
        model = _trainedmodel(self.ModelClass, "cA", r, None)
        model.basis = None
        out = model.predict(q0, t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, t.size)

        # Try to predict with badly-shaped discrete inputs.
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, np.random.random((m - 1, nt)))
        assert (
            ex.value.args[0] == f"input_func.shape = {(m-1, nt)} "
            f"!= {(m, nt)} = (m, len(t))"
        )

        model = _trainedmodel(self.ModelClass, "cAHB", r, m=1)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, np.random.random((2, nt)))
        assert (
            ex.value.args[0] == f"input_func.shape = {(2, nt)} "
            f"!= {(1, nt)} = (m, len(t))"
        )

        # Try to predict with badly-shaped continuous inputs.
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, lambda t: np.ones(m - 1))
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            f"of shape (m,) = {(m,)}"
        )
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, lambda t: 1)
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            f"of shape (m,) = {(m,)}"
        )

        model = _trainedmodel(self.ModelClass, "cAHB", r, m=1)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, input_func)
        assert (
            ex.value.args[0] == "input_func() must return ndarray "
            "of shape (m,) = (1,) or scalar"
        )

        # Try to predict with continuous inputs with bad return type
        model = _trainedmodel(self.ModelClass, "cAHB", r, m)
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, lambda t: set([5]))
        assert (
            ex.value.args[0] == "input_func() must return ndarray of "
            f"shape (m,) = {(m,)}"
        )

        for form in MODEL_FORMS:
            if "B" in form or "N" in form:
                # Predict with 2D inputs.
                model = _trainedmodel(self.ModelClass, form, r, m)
                # continuous input.
                for method in ["RK45", "BDF"]:
                    out = model.predict(q0, t, input_func, method=method)
                    assert isinstance(out, np.ndarray)
                    assert out.shape == (r, nt)
                # discrete input.
                out = model.predict(q0, t, Upred)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)

                # Predict with 1D inputs.
                model = _trainedmodel(self.ModelClass, form, r, 1)
                # continuous input.
                out = model.predict(q0, t, lambda t: 1)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                out = model.predict(q0, t, lambda t: np.array([1]))
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                # discrete input.
                out = model.predict(q0, t, np.ones_like(t))
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                assert hasattr(model, "predict_result_")


class TestFrozenMixin:
    """Test models.monolithic.nonparametric._frozen._FrozenMixin."""

    class Dummy(
        _module._FrozenMixin,
        _module._NonparametricModel,
    ):
        """Instantiable version of _FrozenMixin."""

        def predict(*args, **kwargs):
            pass

    def test_disabled(self, ModelClass=None):
        """Test _FrozenMixin.fit() and other disabled methods."""
        if ModelClass is None:
            ModelClass = self.Dummy
        model = ModelClass("A")

        # Test disabled data_matrix_ property.
        assert model.data_matrix_ is None
        model.solver_ = "A"
        assert model.data_matrix_ is None
        assert model.operator_matrix_dimension is None

        # Test disabled fit().
        with pytest.raises(NotImplementedError) as ex:
            model.fit(None, None, known_operators=None)
        assert ex.value.args[0] == (
            "fit() is disabled for this class, call fit() "
            "on the parametric model object"
        )

        # Test disabled _clear().
        with pytest.raises(NotImplementedError) as ex:
            model._clear()
        assert ex.value.args[0] == (
            "_clear() is disabled for this class, call fit() "
            "on the parametric model object"
        )
