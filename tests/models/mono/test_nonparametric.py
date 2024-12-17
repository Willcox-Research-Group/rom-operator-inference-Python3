# models/mono/test_nonparametric.py
"""Tests for models.mono._nonparametric."""

import os
import abc
import pytest
import itertools
import numpy as np
from scipy import linalg as la

import opinf

try:
    from .test_base import _TestModel
except ImportError:
    from test_base import _TestModel


_module = opinf.models.mono._nonparametric


# Helper functions ------------------------------------------------------------
kron2c = opinf.operators.QuadraticOperator.ckron
kron3c = opinf.operators.CubicOperator.ckron


# Base tests ==================================================================
class _TestNonparametricModel(_TestModel):
    """Tests for classes that inherit from
    models.nonparametric._base._NonparametricModel.
    """

    # Setup -------------------------------------------------------------------
    def get_operators(self, r=None, m=None):
        """Return a valid collection of operators to test."""
        if r is None:
            ops = [
                opinf.operators.ConstantOperator(),
                opinf.operators.LinearOperator(),
                opinf.operators.QuadraticOperator(),
                opinf.operators.CubicOperator(),
            ]
            if m == 0:
                return ops
            return ops + [
                opinf.operators.InputOperator(),
                opinf.operators.StateInputOperator(),
            ]

        assert m is not None, "if r is given, m must be given as well"
        rand = np.random.random
        ops = [
            opinf.operators.ConstantOperator(rand(r)),
            opinf.operators.LinearOperator(rand((r, r))),
            opinf.operators.QuadraticOperator(rand((r, r**2))),
            opinf.operators.CubicOperator(rand((r, r**3))),
        ]
        if m == 0:
            return ops
        return ops + [
            opinf.operators.InputOperator(rand((r, m))),
            opinf.operators.StateInputOperator(rand((r, r * m))),
        ]

    def get_data(self, n=60, k=25, m=20):
        """Get dummy snapshot, time derivative, and input data."""
        Q = np.random.random((n, k))
        Qdot = np.random.random((n, k))
        U = np.ones((m, k))

        return Q, Qdot, U

    # Properties --------------------------------------------------------------
    def test_operators(self):
        """Test the operators property and related methods."""
        super().test_operators()

        # Test __init__() shortcuts.
        keys = list(self.Model._operator_abbreviations.keys())
        model = self.Model(keys)
        assert len(model.operators) == len(self.Model._operator_abbreviations)
        for key, op in zip(keys, model.operators):
            assert isinstance(op, self.Model._operator_abbreviations[key])
            assert op.entries is None

    def test_isvalidoperator(self):
        """Test _isvalidoperator()."""
        with pytest.raises(TypeError) as ex:
            self.Model([opinf.operators.InterpConstantOperator()])
        assert ex.value.args[0].startswith("invalid operator of type")

    def test_check_operator_types_unique(self):
        """Test _check_operators_types_unique()."""
        ops = self.get_operators()
        with pytest.raises(ValueError) as ex:
            self.Model(ops + ops[::-1])
        assert ex.value.args[0] == (
            "duplicate type in list of operators to infer"
        )

    def test_get_operator_of_type(self, m=4, r=7):
        """Test _get_operator_of_type() and the [caHGBN]_ properties."""
        model = self.Model(
            [
                opinf.operators.LinearOperator(),
                opinf.operators.InputOperator(),
                opinf.operators.ConstantOperator(),
                opinf.operators.StateInputOperator(),
            ]
        )

        assert model.A_ is model.operators[0]
        assert model.B_ is model.operators[1]
        assert model.c_ is model.operators[2]
        assert model.N_ is model.operators[3]
        assert model.H_ is None
        assert model.G_ is None

    def test_str(self):
        """Lightly test __str__() and __repr__()."""

        str(self.Model("A"))
        str(self.Model("HB"))

        model = self.Model("A")
        model.state_dimension = 20
        str(model)

        model = self.Model("cB", solver=2)
        model.state_dimension = 10
        model.input_dimension = 3
        modelstr = str(model)
        modelrpr = repr(model)
        assert modelrpr.count(modelstr) == 1

    def test_operator_matrix(self, r=15, m=3):
        """Test the operator_matrix property."""
        c, A, H, G, B, N = self.get_operators(r, m)

        model = self.Model("cA")
        with pytest.raises(AttributeError) as ex:
            Ohat = model.operator_matrix
        assert ex.value.args[0].endswith("(call fit())")

        model.state_dimension = r
        model.operators[:] = (c, A)
        Ohat = np.column_stack([c.entries, A.entries])
        assert np.all(model.operator_matrix == Ohat)

        model.operators[:] = (H, B)
        model._has_inputs = True
        model.input_dimension = m
        Ohat = np.column_stack([H.entries, B.entries])
        assert np.all(model.operator_matrix == Ohat)

        model.operators[:] = [G, N]
        Ohat = np.column_stack([G.entries, N.entries])
        assert np.all(model.operator_matrix == Ohat)

    # Fitting -----------------------------------------------------------------
    def test_process_fit_arguments(self, k=50, m=4, r=6):
        """Test _NonparametricModel._process_fit_arguments()."""
        # Get test data.
        Q, lhs, U = self.get_data(r, k, m)
        c, A, H, G, B, N = self.get_operators(r, m)
        U1d = U[0, :]
        ones = np.ones(k)

        # Exceptions #

        # States do not match dimensions 'r'.
        model = self.Model(["c", A])
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(Q[1:], None, None)
        assert ex.value.args[0] == f"states.shape[0] = {r-1} != r = {r}"

        # LHS not aligned with states.
        model = self.Model([A, "B"])
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(Q, lhs[:, :-1], U)
        assert ex.value.args[0] == (
            f"{self.Model._LHS_ARGNAME}.shape[-1] = {k-1} "
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
        assert ex.value.args[0] == (
            f"inputs.shape[-1] = {k-1} != {k} = states.shape[-1]"
        )

        # Correct usage #

        # With input.
        model.operators = "AB"
        Q_, lhs_, U_ = model._process_fit_arguments(Q, lhs, U)
        assert model.state_dimension == r
        assert model.input_dimension == m
        assert Q_ is Q
        assert lhs_ is lhs
        assert U_ is U

        # With a one-dimensional input.
        model = self.Model("cHB")
        Q_, lhs_, inputs = model._process_fit_arguments(Q, lhs, U1d)
        assert model.state_dimension == r
        assert model.input_dimension == 1
        assert Q_ is Q
        assert lhs_ is lhs
        assert inputs.shape == (1, k)
        assert np.all(inputs[0] == U1d)

        # Without input.
        model = self.Model("cA")
        Q_, lhs_, inputs = model._process_fit_arguments(Q, lhs, None)
        assert model.state_dimension == r
        assert model.input_dimension == 0
        assert inputs is None
        assert Q_ is Q
        assert lhs_ is lhs

        # With known operators for A.
        model = self.Model(["c", A])
        Q_, lhs_, _ = model._process_fit_arguments(Q, lhs, None)
        assert np.allclose(lhs_, lhs - (A.entries @ Q))

        # With known operators for c and B.
        model = self.Model([c, "A", B])
        Q_, lhs_, _ = model._process_fit_arguments(Q, lhs, U)
        lhstrue = lhs - B.entries @ U - np.outer(c.entries, ones)
        assert np.allclose(lhs_, lhstrue)

        # Special case: m = inputs.ndim = 1
        U1d = U[0]
        assert U1d.shape == (k,)
        B1d = opinf.operators.InputOperator(B.entries[:, 0])
        assert B1d.shape == (r, 1)
        model.operators = ["A", B1d]
        assert model.input_dimension == 1
        Q_, lhs_, _ = model._process_fit_arguments(Q, lhs, U1d)
        assert np.allclose(lhs_, lhs - np.outer(B1d.entries, ones))

    def test_assemble_data_matrix(self, k=50, m=6, r=8):
        """Test _NonparametricModel._assemble_data_matrix()."""
        # Get test data.
        Q_, _, U = self.get_data(r, k, m)

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
            model = self.Model(opkeys)
            model.state_dimension = r
            if model._has_inputs:
                model.input_dimension = m
            D = model._assemble_data_matrix(Q_, U)
            assert D.shape == (k, d)

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
        model = self.Model("c")
        c, A, H, G, B, N = [op.entries for op in self.get_operators(r, m)]

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

    def test_fit(self, k=500, m=4, r=6):
        """Test _NonparametricModel.fit()."""
        # Get test data.
        Q, F, U = self.get_data(r, k, m)
        U1d = U[0, :]
        args = [Q, F]

        # Fit the model with each modelform.
        keys = list(self.Model._operator_abbreviations.keys())
        for k in range(1, len(keys) + 1):
            for oplist in itertools.combinations(keys, k):
                model = self.Model(oplist)
                if "B" in oplist or "N" in oplist:
                    model.fit(*args, U)  # Two-dimensional inputs.
                    model.fit(*args, U1d)  # One-dimensional inputs.
                else:
                    model.fit(*args)  # No inputs.

        # Special case: fully intrusive.
        c, A, H, G, B, N = self.get_operators(r, m)

        model.operators = [A, B]
        assert model._fully_intrusive is True
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            out = model.fit(Q, F)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "all operators initialized explicitly, nothing to learn"
        )
        assert out is model

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            out = model.refit()
        assert wn[0].message.args[0] == (
            "all operators initialized explicitly, nothing to learn"
        )
        assert out is model

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
        c, A, H, G, B, N = self.get_operators(r, m)

        model = self.Model([c, A])
        for _ in range(ntrials):
            q = np.random.random(r)
            y = c.entries + A.entries @ q
            out = model.rhs(q)
            assert out.shape == y.shape
            assert np.allclose(out, y)

            Q = np.random.random((r, k))
            Y = c.entries.reshape((r, 1)) + A.entries @ Q
            out = model.rhs(Q)
            assert out.shape == Y.shape
            assert np.allclose(out, Y)

        model = self.Model([H, B])
        for _ in range(ntrials):
            u = np.random.random(m)
            q = np.random.random(r)
            y = H.entries @ kron2c(q) + B.entries @ u
            out = model.rhs(q, u)
            assert out.shape == y.shape
            assert np.allclose(out, y)

            Q = np.random.random((r, k))
            U = np.random.random((m, k))
            Y = H.entries @ kron2c(Q) + B.entries @ U
            out = model.rhs(Q, U)
            assert out.shape == Y.shape
            assert np.allclose(out, Y)

    def test_jacobian(self, r=5, m=2, ntrials=10):
        """Test _NonparametricModel.jacobian()."""
        c, A, H, G, B, N = self.get_operators(r, m)

        for oplist in ([c, A], [c, A, B]):
            model = self.Model(oplist)
            q = np.random.random(r)
            out = model.jacobian(q)
            assert out.shape == (r, r)
            assert np.allclose(out, A.entries)

    @abc.abstractmethod
    def test_predict(self):
        """Test predict()."""
        raise NotImplementedError

    # Model persistence -------------------------------------------------------
    def test_copy(self, r=4, m=3):
        """Test copy()."""
        ops1 = self.get_operators(r=r, m=m)
        ops2 = self.get_operators()
        model = self.Model(ops1[::2] + ops2[1::2])
        model2 = model.copy()
        assert model2 is not model
        assert isinstance(model2, model.__class__)
        assert len(model2.operators) == len(model.operators)
        for op2, op1 in zip(model2.operators, model.operators):
            assert op2 is not op1
            assert op2 == op1

    def test_saveload(self, m=2, r=3, target="_savemodeltest.h5"):
        """Test save() and load()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        with pytest.raises(FileNotFoundError):
            self.Model.load(target)

        model = self.Model(self.get_operators())
        model.save(target)
        assert os.path.isfile(target)

        try:
            # Check overwrite behavior.
            with pytest.raises(FileExistsError):
                model.save(target, overwrite=False)
            model.save(target, overwrite=True)

            # Blank operators and dimensions.
            loaded = self.Model.load(target)
            assert loaded is not model
            assert loaded == model

            # Some operators fixed + known dimensions.
            ops1 = self.get_operators(r=r, m=m)
            ops2 = self.get_operators(r=None, m=None)
            model.operators = ops1[::2] + ops2[1::2]
            model.save(target, overwrite=True)

            loaded = self.Model.load(target)
            assert loaded is not model
            assert loaded == model
            assert (
                loaded._indices_of_operators_to_infer
                == model._indices_of_operators_to_infer
            )
            assert (
                loaded._indices_of_known_operators
                == model._indices_of_known_operators
            )

        finally:
            if os.path.isfile(target):
                os.remove(target)


# Test public classes =========================================================
# class TestSteadyModel(_TestNonparametricModel):
#     """Test models.monolithic.nonparametric._public.SteadyModel."""

#     Model = _module.SteadyModel

#     def test_predict(self):
#         """Test SteadyModel.predict()."""
#         raise NotImplementedError


class TestDiscreteModel(_TestNonparametricModel):
    """Test models.monolithic.nonparametric._public.DiscreteModel."""

    Model = _module.DiscreteModel

    def test_stack_trajectories(self, r=10, k=20, m=5, num_trajectories=4):
        """Test DiscreteModel.stack_trajectories()."""
        statelist, inputlist = [], []
        for _ in range(num_trajectories):
            Q, _, U = self.get_data(r, k, m)
            statelist.append(Q)
            inputlist.append(U)

        Qs, Qnexts = self.Model.stack_trajectories(statelist)
        assert Qs.shape == (r, (k - 1) * num_trajectories)
        assert Qnexts.shape == Qs.shape
        Qs_split = np.split(Qs, num_trajectories, axis=1)
        Qnexts_split = np.split(Qnexts, num_trajectories, axis=1)
        for i in range(num_trajectories):
            assert np.all(Qs_split[i][:, 1:] == Qnexts_split[i][:, :-1])
            assert np.all(Qs_split[i] == statelist[i][:, :-1])
            assert np.all(Qnexts_split[i] == statelist[i][:, 1:])

        Qs2, Qnexts2, Us = self.Model.stack_trajectories(
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
        Qs3, Qnexts3, Us_1d = self.Model.stack_trajectories(
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

    def test_fit(self, k=50, r=5, m=3):
        """Test DiscreteModel.fit()."""
        super().test_fit()

        Q, _, U = self.get_data(r, k, m)
        Qnext = Q[:, 1:]
        model = self.Model("A").fit(Q)
        model2 = self.Model("A").fit(Q[:, :-1], Qnext)
        assert model.A_ == model2.A_

        model = self.Model("AB").fit(Q, inputs=U)
        model2 = self.Model("AB").fit(Q[:, :-1], Qnext, inputs=U)
        assert model.A_ == model2.A_
        assert model.B_ == model2.B_

    def test_predict(self, k=20, m=6, r=4):
        """Test DiscreteModel.predict()."""
        # Get test data.
        niters = 5
        q0 = np.random.random(r)
        U = np.ones((m, niters - 1))

        ops = c, A, H, G, B, N = self.get_operators(r, m)
        B1d = opinf.operators.InputOperator(entries=B.entries[:, 0])
        model = self.Model(ops)

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
        assert ex.value.args[0] == (
            "argument 'niters' must be a positive integer"
        )

        # Try to predict with badly-shaped discrete inputs.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, niters, np.random.random((m - 1, niters - 1)))
        assert ex.value.args[0] == (
            f"inputs.shape = ({(m-1, niters-1)} "
            f"!= {(m, niters-1)} = (m, niters-1)"
        )

        model_m1 = self.Model([A, B1d])
        with pytest.raises(ValueError) as ex:
            model_m1.predict(q0, niters, np.random.random((2, niters - 1)))
        assert ex.value.args[0] == (
            f"inputs.shape = ({(2, niters-1)} "
            f"!= {(1, niters-1)} = (m, niters-1)"
        )

        # Try to predict with continuous inputs.
        with pytest.raises(TypeError) as ex:
            model.predict(q0, niters, lambda t: np.ones(m - 1))
        assert ex.value.args[0] == "inputs must be NumPy array, not callable"

        # No control inputs.
        model = self.Model([A])
        out = model.predict(q0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)

        # With 2D inputs.
        model = self.Model([A, B])
        out = model.predict(q0, niters, U)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)

        # With 1D inputs.
        model = self.Model([A, B1d])
        out = model.predict(q0, niters, np.ones(niters))
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)


class TestContinuousModel(_TestNonparametricModel):
    """Test models.monolithic.nonparametric._public.ContinuousModel."""

    Model = _module.ContinuousModel

    def test_rhs(self, k=10, r=5, m=2, ntrials=10):
        """Test ContinuousModel.rhs()."""
        c, A, H, G, B, N = self.get_operators(r, m)

        model = self.Model([c, A])
        for _ in range(ntrials):
            q = np.random.random(r)
            y = c.entries + A.entries @ q
            out = model.rhs(1, q)
            assert out.shape == y.shape
            assert np.allclose(out, y)

            Q = np.random.random((r, k))
            Y = c.entries.reshape((r, 1)) + A.entries @ Q
            out = model.rhs(2, Q)
            assert out.shape == Y.shape
            assert np.allclose(out, Y)

        U = np.random.random((m, k))
        u = U[:, 0]

        def input_func(t):
            return u

        def input_func2(t):
            return U

        model = self.Model([H, B])
        for _ in range(ntrials):
            q = np.random.random(r)
            y = H.entries @ kron2c(q) + B.entries @ u
            out = model.rhs(0, q, input_func)
            assert out.shape == y.shape
            assert np.allclose(out, y)

            Q = np.random.random((r, k))
            U = np.random.random((m, k))
            Y = H.entries @ kron2c(Q) + B.entries @ U
            out = model.rhs(20, Q, input_func2)
            assert out.shape == Y.shape
            assert np.allclose(out, Y)

        model = self.Model([A, B])
        with pytest.raises(TypeError) as ex:
            model.rhs(5, q, 10)
        assert "object is not callable" in ex.value.args[0]

    def test_jacobian(self, r=6, m=3):
        """Test ContinuousModel.jacobian()."""
        c, A, H, G, B, N = self.get_operators(r=r, m=m)

        model = self.Model([c, A])
        q = np.random.random(r)
        out = model.jacobian(8, q)
        assert out.shape == (r, r)
        assert np.allclose(out, A.entries)

        def input_func(t):
            return np.random.random(m)

        model = self.Model([A, B])
        with pytest.raises(TypeError) as ex:
            model.jacobian(5, q, 10)
        assert "object is not callable" in ex.value.args[0]

        out = model.jacobian(2, q, input_func)
        assert out.shape == (r, r)
        assert np.allclose(out, A.entries)

    def test_predict(self, k=50, m=10, r=6):
        """Test ContinuousModel.predict()."""
        # Get test data.
        nt = 5
        q0 = np.zeros(r)
        t = np.linspace(0, 1e-5 * nt, nt)

        # No inputs #

        # Try to predict with invalid initial condition.
        ops = self.get_operators(r=r, m=0)
        model = self.Model(ops)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0[1:], t)
        assert ex.value.args[0] == (
            "initial condition not aligned with model "
            f"(state0.shape = ({r-1},) != ({r},) = (r,))"
        )

        # Try to predict with bad time array.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, np.atleast_2d(t))
        assert ex.value.args[0] == "time 't' must be one-dimensional"

        # Predict without inputs.
        for k in range(1, len(ops) + 1):
            for comb in itertools.combinations(ops, k):
                model.operators = list(comb)
                for method in ["RK45", "BDF"]:
                    out = model.predict(q0, t, method=method)
                    assert isinstance(out, np.ndarray)
                    assert out.shape == (r, t.size)

        # With inputs #

        def input_func(tt):
            return tt * np.zeros(m)

        Upred = np.column_stack([input_func(tt) for tt in t])

        ops = self.get_operators(r=r, m=m)
        model = self.Model(ops)

        # Try to predict with badly-shaped discrete inputs.
        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, np.random.random((m - 1, nt)))
        assert ex.value.args[0] == (
            f"input_func.shape = {(m-1, nt)} != {(m, nt)} = (m, len(t))"
        )

        with pytest.raises(ValueError) as ex:
            model.predict(q0, t, np.random.random((m + 1, nt)))
        assert ex.value.args[0] == (
            f"input_func.shape = {(m+1, nt)} != {(m, nt)} = (m, len(t))"
        )

        # Try to predict with badly-shaped continuous inputs.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0, t, lambda t: np.ones(m - 1))
        assert ex.value.args[0] == (
            f"input_func() must return ndarray of shape (m,) = {(m,)}"
        )
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0, t, lambda t: 1)
        assert ex.value.args[0] == (
            f"input_func() must return ndarray of shape (m,) = {(m,)}"
        )

        # Try to predict with continuous inputs with bad return type
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model.predict(q0, t, lambda t: set([5]))
        assert ex.value.args[0] == (
            f"input_func() must return ndarray of shape (m,) = {(m,)}"
        )

        # Correct prediction with inputs.
        for k in range(1, len(ops) + 1):
            for comb in itertools.combinations(ops, k):
                model.operators = list(comb)
                if not model._has_inputs:
                    continue
                # continuous input.
                for method in ["RK45", "BDF"]:
                    out = model.predict(q0, t, input_func, method=method)
                    assert isinstance(out, np.ndarray)
                    assert out.shape == (r, nt)
                # discrete input.
                out = model.predict(q0, t, Upred)
                assert isinstance(out, np.ndarray)
                assert out.shape == (r, nt)
                assert hasattr(model, "predict_result_")

        # Correct prediction with 1D inputs.
        ops = self.get_operators(r=r, m=1)
        for k in range(1, len(ops) + 1):
            for comb in itertools.combinations(ops, k):
                model.operators = list(comb)
                if not model._has_inputs:
                    continue
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


# Test "frozen" classes =======================================================
class _TestFrozenMixin:
    """Tests for clases that inherit from
    models.monolithic.nonparametric._frozen._FrozenMixin.
    """

    Model = NotImplemented

    def test_disabled(self, Model=None):
        """Test fit() and other disabled methods."""
        if Model is None:
            Model = self.Model
        model = Model("A", solver=2)

        # Test disabled properties.
        assert model.solver is None

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


class TestFrozenSteadyModel(_TestFrozenMixin):
    Model = _module._FrozenSteadyModel


class TestFrozenDiscreteModel(_TestFrozenMixin):
    Model = _module._FrozenDiscreteModel


class TestFrozenContinuousModel(_TestFrozenMixin):
    Model = _module._FrozenContinuousModel


if __name__ == "__main__":
    pytest.main([__file__])
