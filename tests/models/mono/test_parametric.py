# models/mono/test_parametric.py
"""Tests for models.mono._parametric."""

import os

import pytest
import numpy as np
import scipy.interpolate as interp

import opinf


try:
    from .test_base import _TestModel
except ImportError:
    from test_base import _TestModel


_module = opinf.models


# Parametric models ===========================================================
class _TestParametricModel(_TestModel):
    """Test models.mono._parametric._ParametricModel."""

    # Setup -------------------------------------------------------------------
    _iscontinuous = NotImplemented

    def get_single_operator(self, p=4):
        """Get a single uncalibrated operator."""
        return opinf.operators.AffineLinearOperator(p)

    def get_operators(self, r=None, m=None, p=3):
        """Return a valid collection of operators to test."""
        if r is None:
            ops = [
                opinf.operators.AffineConstantOperator(coeffs=p),
                opinf.operators.AffineLinearOperator(coeffs=p),
            ]
            if m == 0:
                return ops
            return ops + [opinf.operators.AffineInputOperator(coeffs=p)]

        assert m is not None, "if r is given, m must be as well"
        rand = np.random.random
        ops = [
            opinf.operators.AffineConstantOperator(
                coeffs=p,
                entries=[rand(r) for _ in range(p)],
            ),
            opinf.operators.AffineLinearOperator(
                coeffs=p,
                entries=[rand((r, r)) for _ in range(p)],
            ),
        ]
        if m == 0:
            return ops
        return ops + [
            opinf.operators.AffineInputOperator(
                coeffs=p,
                entries=[np.random.random((r, m)) for _ in range(p)],
            )
        ]

    def get_parametric_operators(self, p, r, m=0):
        """Get calibrated constant + linear + input affine operators."""
        return self.get_operators(r, m, p), np.random.random(p)

    def test_isvalidoperator(self):
        """Test _isvalidoperator()."""
        with pytest.raises(TypeError) as ex:
            self.Model([100])
        assert ex.value.args[0].startswith("invalid operator of type")

    def test_check_operator_types_unique(self, p=2):
        """Test _check_operator_types_unique()."""
        operators = [
            opinf.operators.AffineLinearOperator(p),
            opinf.operators.LinearOperator(),
        ]

        with pytest.raises(ValueError) as ex:
            self.Model._check_operator_types_unique(operators)
        assert ex.value.args[0] == (
            "duplicate type in list of operators to infer"
        )

        operators[1] = opinf.operators.ConstantOperator()
        self.Model._check_operator_types_unique(operators)

    def test_get_operator_of_type(self):
        """Test _get_operator_of_type() and the [caHGBN]_ properties."""
        ops = self.get_operators()
        model = self.Model(ops)

        assert model.c_ is model.operators[0]
        assert model.A_ is model.operators[1]
        assert model.B_ is model.operators[2]
        assert model.N_ is None
        assert model.H_ is None
        assert model.G_ is None

        model = self.Model(
            [
                (c := opinf.operators.ConstantOperator()),
                opinf.operators.AffineLinearOperator(2),
            ]
        )
        assert model.c_ is c

    def test_set_operators(self, p=3):
        """Test operators.fset()."""
        operators = [opinf.operators.LinearOperator()]

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Model(operators)
        assert wn[0].message.args[0] == (
            "no parametric operators detected, "
            "consider using a nonparametric model class"
        )

        operators = [opinf.operators.InterpLinearOperator()]
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Model(operators)
        assert wn[0].message.args[0] == (
            "all operators interpolatory, "
            "consider using an InterpModel class"
        )

        # Several operators provided.
        operators = [
            opinf.operators.ConstantOperator(),
            opinf.operators.AffineLinearOperator(p),
        ]
        model = self.Model(operators)
        assert len(model.operators) == 2
        for modelop, op in zip(model.operators, operators):
            assert modelop is op

        # Single operator provided
        model = self.Model(operators[1])
        assert len(model.operators) == 1
        assert model.operators[0] is operators[1]

    def test_parameter_dimension(self, p=4):
        """Test parameter_dimension and _synchronize_parameter_dimensions()."""
        op0 = opinf.operators.ConstantOperator()
        op1 = opinf.operators.AffineLinearOperator(np.sin, nterms=p)
        model = self.Model([op0, op1])
        assert model.parameter_dimension is None

        op1.parameter_dimension = p
        model._synchronize_parameter_dimensions()
        assert model.parameter_dimension == p

        op1 = opinf.operators.AffineLinearOperator(np.sin, nterms=p)
        op2 = opinf.operators.AffineInputOperator(p)
        assert op1.parameter_dimension is None
        model = self.Model([op0, op1, op2])
        assert op1.parameter_dimension == p
        assert model.parameter_dimension == p

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._synchronize_parameter_dimensions(p + 2)
        assert ex.value.args[0] == (
            f"{p} = each operator.parameter_dimension "
            f"!= parameter dimension = {p + 2}"
        )
        assert model.parameter_dimension == p
        assert op1.parameter_dimension == p
        assert op2.parameter_dimension == p

        op1 = opinf.operators.AffineLinearOperator(p)
        op2 = opinf.operators.AffineInputOperator(p + 1)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Model([op0, op1, op2])
        assert ex.value.args[0] == (
            "operators not aligned "
            "(parameter_dimension must be the same for all operators)"
        )

    def test_process_fit_arguments(self, s=10, p=2, m=4, r=3, k=10):
        """Test _process_fit_arguments()."""
        params = np.random.random((s, p))
        states = [np.ones((r, k)) for _ in range(s)]
        lhs = [np.ones((r, k)) for _ in range(s)]
        inputs = [np.empty((m, k)) for _ in range(s)]

        op = self.get_single_operator()
        model = self.Model([op])

        # Invalid parameters.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(np.empty((3, 3, 3)), None, None, None)
        assert ex.value.args[0] == (
            "'parameters' must be a sequence of scalars or 1D arrays"
        )

        # Inconsistent number of datasets across arguments.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states[1:], None, None)
        assert ex.value.args[0] == (
            f"len(states) = {s-1} != {s} = len(parameters)"
        )
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs[:-1], None)
        assert ex.value.args[0] == (
            f"len({self.Model._ModelClass._LHS_ARGNAME}) = {s-1} "
            f"!= {s} = len(parameters)"
        )
        model._has_inputs = True
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs, inputs[1:])
        assert ex.value.args[0] == (
            f"len(inputs) = {s-1} != {s} = len(parameters)"
        )
        inputs1D = np.empty((s - 1, k))
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs, inputs1D)
        assert ex.value.args[0] == (
            f"len(inputs) = {s-1} != {s} = len(parameters)"
        )
        model._has_inputs = False

        # Inconsistent state dimension.
        states[1] = np.empty((r - 1, k))
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs, None)
        assert ex.value.args[0] == f"len(states[1]) = {r-1} != {r} = r"

        # Inconsistent number of snapshots across datasets.
        states[1] = np.empty((r, k - 1))
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs, None)
        assert ex.value.args[0] == (
            f"{model._LHS_ARGNAME}[1].shape[-1] = {k} "
            f"!= {k-1} = states[1].shape[-1]"
        )

        # Inconsistent input dimension.
        states[1] = np.empty((r, k))
        inputs[1] = np.empty((m - 1, k))
        model._has_inputs = True
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs, inputs)
        assert ex.value.args[0] == f"inputs[1].shape[0] = {m-1} != {m} = m"

        # Correct usage, partially intrusive
        op2 = opinf.operators.AffineConstantOperator(
            p,
            entries=[np.random.random(r) for _ in range(p)],
        )
        if isinstance(self, _TestInterpModel):
            op2 = opinf.operators.InterpConstantOperator(
                training_parameters=params,
                entries=[np.zeros(r) for _ in range(s)],
            )

        model = self.Model([op, op2])
        model._process_fit_arguments(params, states, lhs, None)

        model._has_inputs = True
        inputs[1] = np.empty((m, k))
        model._process_fit_arguments(params, states, lhs, inputs)

    def test_fit(self, s=10, p=3, m=2, r=4, k=20):
        """Test fit() and refit() (but not all intermediate steps)."""
        params = np.random.random((s, p))
        states = [np.ones((r, k)) for _ in range(s)]
        lhs = [np.ones((r, k)) for _ in range(s)]
        inputs = [np.ones((m, k)) for _ in range(s)]

        operators, _ = self.get_parametric_operators(p, r, m)

        # Fully intrusive case.
        model = self.Model(operators)
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            out = model.fit(params, states, lhs, inputs)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "all operators initialized explicitly, nothing to learn"
        )
        assert out is model

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            out = model.refit()
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "all operators initialized explicitly, nothing to learn"
        )
        assert out is model

        # One affine operator.
        model = self.Model([opinf.operators.AffineLinearOperator(p)])
        out = model.fit(params, states, lhs)
        assert out is model
        for op in model.operators:
            assert op.parameter_dimension == p
            assert op.entries is not None

        # Multiple affine operators.
        model = self.Model(
            [
                opinf.operators.AffineLinearOperator(p),
                opinf.operators.AffineInputOperator(p),
            ]
        )
        out = model.fit(params, states, lhs, inputs)  # BUG
        assert out is model
        for op in model.operators:
            assert op.parameter_dimension == p
            assert op.entries is not None

        # Mix of affine and interpolatory operators.
        model = self.Model(
            [
                opinf.operators.AffineLinearOperator(p),
                opinf.operators.InterpInputOperator(),
            ]
        )
        out = model.fit(params, states, lhs, inputs)
        assert out is model
        for op in model.operators:
            assert op.parameter_dimension == p
            assert op.entries is not None

        # Mix of nonparametric, affine, and interpolatory operators.
        model = self.Model(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.AffineLinearOperator(p),
                opinf.operators.InterpInputOperator(),
            ]
        )
        out = model.fit(params, states, lhs, inputs)
        assert out is model
        assert model.operators[0].entries is not None
        for op in model.operators[1:]:
            assert op.parameter_dimension == p
            assert op.entries is not None

    def test_evaluate(self, p=8, r=4, m=2):
        """Test evaluate()."""
        operators, testparam = self.get_parametric_operators(p, r, m)

        # Some operators not populated.
        model = self.Model([self.get_single_operator()])
        with pytest.raises(AttributeError):
            model.evaluate(testparam)

        # Test with and without input operators.
        for ops in operators[:-1], operators:
            model = self.Model(ops)
            model_evaluated = model.evaluate(testparam)
            assert isinstance(model_evaluated, self.Model._ModelClass)
            assert len(model_evaluated.operators) == len(model.operators)
            assert model_evaluated.state_dimension == r
            for pop, op in zip(model.operators, model_evaluated.operators):
                pop_evaluated = pop.evaluate(testparam)
                assert isinstance(op, pop_evaluated.__class__)
                assert np.array_equal(op.entries, pop_evaluated.entries)
        assert model_evaluated.input_dimension == model.input_dimension

    def test_rhs(self, p=7, r=2, m=4):
        """Lightly test rhs()."""
        operators, testparam = self.get_parametric_operators(p, r, m)
        teststate = np.random.random(r)
        args = [testparam, teststate]
        if self._iscontinuous:
            args.insert(0, np.random.random())  # time argument

            def testinput(t):
                return np.random.random(m)

        else:
            testinput = np.random.random(m)

        # Some operators not populated.
        model = self.Model([self.get_single_operator()])
        with pytest.raises(AttributeError):
            model.rhs(*args)

        # Without inputs.
        model = self.Model(operators[:-1])
        out = model.rhs(*args)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,)

        # With inputs.
        args.append(testinput)
        model = self.Model(operators)
        out = model.rhs(*args)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,)

    def test_jacobian(self, p=9, r=3, m=2):
        """Lightly test jacobian()."""
        operators, testparam = self.get_parametric_operators(p, r, m)
        teststate = np.random.random(r)
        args = [testparam, teststate]
        if self._iscontinuous:
            args.insert(0, np.random.random())  # time argument

            def testinput(t):
                return np.random.random(m)

        else:
            testinput = np.random.random(m)

        # Some operators not populated.
        model = self.Model([self.get_single_operator()])
        with pytest.raises(AttributeError):
            model.jacobian(*args)

        # Without inputs.
        model = self.Model(operators[:-1])
        out = model.jacobian(*args)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, r)

        # With inputs.
        args.append(testinput)
        model = self.Model(operators)
        out = model.jacobian(*args)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, r)


class TestParametricDiscreteModel(_TestParametricModel):
    """Test opinf.models.ParametricDiscreteModel."""

    Model = _module.ParametricDiscreteModel
    _iscontinuous = False

    def test_predict(self, p=5, r=3, m=2, niters=10):
        """Lightly test InterpDiscreteModel.predict()."""
        testparam = np.random.random(p)
        state0 = np.random.random(r)

        model = self.Model(
            opinf.operators.AffineLinearOperator(
                p,
                entries=np.zeros((p, r, r)),
            )
        )
        out = model.predict(testparam, state0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)
        assert np.all(out[:, 0] == state0)
        assert np.all(out[:, 1:] == 0)

        inputs = np.random.random((m, niters))
        model = self.Model(
            opinf.operators.AffineInputOperator(
                p,
                entries=np.zeros((p, r, m)),
            )
        )
        out = model.predict(testparam, state0, niters, inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)
        assert np.all(out[:, 0] == state0)
        assert np.all(out[:, 1:] == 0)


class TestParametricContinuousModel(_TestParametricModel):
    """Test opinf.models.ParametricContinuousModel."""

    Model = _module.ParametricContinuousModel
    _iscontinuous = True

    def test_predict(self, p=4, r=4, m=2, k=40):
        """Lightly test predict()."""
        testparam = np.random.random(p)
        state0 = np.random.random(r)
        t = np.linspace(0, 1, k)

        model = self.Model(
            opinf.operators.AffineLinearOperator(
                p,
                entries=np.zeros((p, r, r)),
            )
        )
        out = model.predict(testparam, state0, t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, k)
        for j in range(k):
            assert np.allclose(out[:, j], state0)

        def input_func(t):
            return np.random.random(m)

        model = self.Model(
            opinf.operators.AffineInputOperator(
                p,
                entries=np.zeros((p, r, m)),
            )
        )
        out = model.predict(testparam, state0, t, input_func)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, k)
        for j in range(k):
            assert np.allclose(out[:, j], state0)


# Interpolatotry models =======================================================
class _TestInterpModel(_TestParametricModel):
    """Test models.mono._parametric._InterpModel."""

    def get_single_operator(self):
        """Get a single uncalibrated operator."""
        return opinf.operators.InterpLinearOperator()

    def get_operators(self, r=None, m=None, s=10, params=None):
        """Return a valid collection of operators to test."""
        if params is None:
            params = np.sort(np.random.random(s))
        s = len(params)
        if r is None:
            ops = [
                opinf.operators.InterpConstantOperator(params),
                opinf.operators.InterpLinearOperator(params),
            ]
            if m == 0:
                return ops
            return ops + [opinf.operators.InterpInputOperator(params)]

        assert m is not None, "if r is given, m must be as well"
        rand = np.random.random
        ops = [
            opinf.operators.InterpConstantOperator(
                params,
                entries=[rand(r) for _ in range(s)],
            ),
            opinf.operators.InterpLinearOperator(
                params,
                entries=[rand((r, r)) for _ in range(s)],
            ),
        ]
        if m == 0:
            return ops
        return ops + [
            opinf.operators.InterpInputOperator(
                params,
                entries=[np.random.random((r, m)) for _ in range(s)],
            )
        ]

    def get_parametric_operators(self, s, r, m=0):
        """Get calibrated constant + linear + input affine operators."""
        params = np.sort(np.random.random(s))
        return (
            self.get_operators(r, m, s, params=params),
            (params[-1] - params[0]) / 2,
        )

    def test_set_operators(self):
        """Test operators.fset()."""
        operators = [opinf.operators.LinearOperator()]

        with pytest.raises(TypeError) as ex:
            self.Model(operators)
        assert ex.value.args[0] == "invalid operator of type 'LinearOperator'"

        # Several operators provided.
        operators = [
            opinf.operators.InterpConstantOperator(),
            opinf.operators.InterpLinearOperator(),
        ]
        model = self.Model(operators)
        assert len(model.operators) == 2
        for modelop, op in zip(model.operators, operators):
            assert modelop is op

        # Single operator provided
        model = self.Model(operators[1])
        assert len(model.operators) == 1
        assert model.operators[0] is operators[1]

    def test_get_operator_of_type(self):
        """Test _get_operator_of_type()."""
        operators = [
            opinf.operators.InterpConstantOperator(),
            opinf.operators.InterpLinearOperator(),
        ]
        model = self.Model(operators)

        op = model._get_operator_of_type(opinf.operators.ConstantOperator)
        assert op is operators[0]

        op = model._get_operator_of_type(opinf.operators.LinearOperator)
        assert op is operators[1]

        op = model._get_operator_of_type(float)
        assert op is None

    def test_parameter_dimension(self, p=4):
        """Test parameter_dimension and _synchronize_parameter_dimensions()."""
        op1 = opinf.operators.InterpLinearOperator()
        assert op1.parameter_dimension is None
        model = self.Model([op1])
        assert model.parameter_dimension is None

        op1.parameter_dimension = p
        model._synchronize_parameter_dimensions()
        assert model.parameter_dimension == p

        op1 = opinf.operators.InterpLinearOperator()
        op2 = opinf.operators.InterpInputOperator()
        op2.parameter_dimension = p
        assert op1.parameter_dimension is None
        model = self.Model([op1, op2])
        assert op1.parameter_dimension == p
        assert op2.parameter_dimension == p
        assert model.parameter_dimension == p

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._synchronize_parameter_dimensions(p + 2)
        assert ex.value.args[0] == (
            f"{p} = each operator.parameter_dimension "
            f"!= parameter dimension = {p + 2}"
        )
        assert model.parameter_dimension == p
        assert op1.parameter_dimension == p
        assert op2.parameter_dimension == p

        op1 = opinf.operators.InterpLinearOperator()
        op2 = opinf.operators.InterpInputOperator()
        op1.parameter_dimension = p
        op2.parameter_dimension = p + 1
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Model([op1, op2])
        assert ex.value.args[0] == (
            "operators not aligned "
            "(parameter_dimension must be the same for all operators)"
        )

    def test_from_models(self, s=10, r=4, m=2):
        """Test _InterpModel._from_models()."""
        operators = [
            [
                opinf.operators.ConstantOperator(np.random.random(r)),
                opinf.operators.LinearOperator(np.random.random((r, r))),
                opinf.operators.InputOperator(np.random.random((r, m))),
            ]
            for _ in range(s)
        ]
        mu = np.sort(np.random.random(s))

        # Inconsistent number of operators.
        model1 = self.Model._ModelClass(operators[0])
        model2 = self.Model._ModelClass(operators[1][:-1])
        with pytest.raises(ValueError) as ex:
            self.Model._from_models(mu, [model1, model2])
        assert ex.value.args[0] == (
            "models not aligned (inconsistent number of operators)"
        )

        # Inconsistent operator types.
        model1 = self.Model._ModelClass(operators[0][1:])
        model2 = self.Model._ModelClass(operators[1][:-1])
        with pytest.raises(ValueError) as ex:
            self.Model._from_models(mu, [model1, model2])
        assert ex.value.args[0] == (
            "models not aligned (inconsistent operator types)"
        )

        # Correct usage
        models = [self.Model._ModelClass(ops) for ops in operators]
        model = self.Model._from_models(mu, models)
        assert isinstance(model, self.Model)
        assert len(model.operators) == 3
        assert isinstance(
            model.operators[0],
            opinf.operators.InterpConstantOperator,
        )

        # Check the interpolation is as expected.
        testparam = np.random.random()
        IClass = type(model.operators[0].interpolator)
        c00 = IClass(mu, [ops[0][0] for ops in operators])
        assert c00(testparam) == model.evaluate(testparam).operators[0][0]

    def test_set_interpolator(self, s=10, p=2, r=2):
        """Test _InterpModel._set_interpolator()."""

        mu = np.random.random((s, p))
        operators = [
            opinf.operators.InterpConstantOperator(
                training_parameters=mu,
                entries=np.random.random((s, r)),
                InterpolatorClass=interp.NearestNDInterpolator,
            ),
            opinf.operators.InterpLinearOperator(
                training_parameters=mu,
                entries=np.random.random((s, r, r)),
                InterpolatorClass=interp.NearestNDInterpolator,
            ),
        ]

        model = self.Model(operators)
        for op in operators:
            assert isinstance(op.interpolator, interp.NearestNDInterpolator)

        model = self.Model(
            operators,
            InterpolatorClass=interp.LinearNDInterpolator,
        )
        for op in operators:
            assert isinstance(op.interpolator, interp.LinearNDInterpolator)

        model.set_interpolator(None)
        for op in operators:
            assert isinstance(op.interpolator, interp.LinearNDInterpolator)

        model.set_interpolator(interp.NearestNDInterpolator)
        for op in operators:
            assert isinstance(op.interpolator, interp.NearestNDInterpolator)

    def test_fit_solver(self, s=10, r=3, k=20):
        """Test _InterpModel._fit_solver()."""
        operators = [
            opinf.operators.InterpConstantOperator(),
            opinf.operators.InterpLinearOperator(),
        ]
        params = np.sort(np.random.random(s))
        states = np.random.random((s, r, k))
        lhs = np.random.random((s, r, k))

        model = self.Model(operators)
        model._fit_solver(params, states, lhs)

        assert hasattr(model, "solvers")
        assert len(model.solvers) == s
        for solver in model.solvers:
            assert isinstance(solver, opinf.lstsq.PlainSolver)

        assert hasattr(model, "_submodels")
        assert len(model._submodels) == s
        for mdl in model._submodels:
            assert isinstance(
                mdl,
                opinf.models.mono._nonparametric._NonparametricModel,
            )
            assert len(mdl.operators) == len(operators)
            for op in mdl.operators:
                assert op.entries is None

        assert hasattr(model, "_training_parameters")
        assert isinstance(model._training_parameters, np.ndarray)
        assert np.all(model._training_parameters == params)

    def test_refit(self, s=10, r=3, k=15):
        """Test _InterpModel.refit()."""
        operators = [
            opinf.operators.InterpConstantOperator(),
            opinf.operators.InterpLinearOperator(),
        ]
        params = np.sort(np.random.random(s))
        states = np.random.random((s, r, k))
        lhs = np.random.random((s, r, k))

        model = self.Model(operators)

        with pytest.raises(RuntimeError) as ex:
            model.refit()
        assert ex.value.args[0] == "model solvers not set, call fit() first"

        model._fit_solver(params, states, lhs)
        model.refit()

        assert hasattr(model, "_submodels")
        assert len(model._submodels) == s
        for mdl in model._submodels:
            assert isinstance(
                mdl,
                opinf.models.mono._nonparametric._NonparametricModel,
            )
            assert len(mdl.operators) == len(operators)
            for op in mdl.operators:
                assert op.entries is not None

    def test_save(self, target="_interpmodelsavetest.h5"):
        """Test _InterpModel._save()."""
        if os.path.isfile(target):
            os.remove(target)

        model = self.Model(
            [
                opinf.operators.InterpConstantOperator(),
                opinf.operators.InterpLinearOperator(),
            ]
        )
        model.save(target)
        assert os.path.isfile(target)

        model.set_interpolator(interp.CubicSpline)
        model.save(target, overwrite=True)
        assert os.path.isfile(target)

        model.set_interpolator(float)
        os.remove(target)
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            model.save(target, overwrite=True)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "cannot serialize InterpolatorClass "
            "'float', must pass in the class when calling load()"
        )
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self, target="_interpmodelloadtest.h5"):
        """Test _InterpModel._load()."""
        if os.path.isfile(target):
            os.remove(target)

        operators = [
            opinf.operators.InterpConstantOperator(),
            opinf.operators.InterpLinearOperator(),
        ]
        model = self.Model(operators, InterpolatorClass=float)

        with pytest.warns(opinf.errors.OpInfWarning):
            model.save(target)

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Model.load(target)
        assert ex.value.args[0] == (
            f"unknown InterpolatorClass 'float', call load({target}, float)"
        )
        self.Model.load(target, float)

        model1 = self.Model(
            operators,
            InterpolatorClass=interp.NearestNDInterpolator,
        )
        model1.save(target, overwrite=True)

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            model2 = self.Model.load(target, float)
        assert wn[0].message.args[0] == (
            "InterpolatorClass=float does not match loadfile "
            "InterpolatorClass 'NearestNDInterpolator'"
        )
        model2.set_interpolator(interp.NearestNDInterpolator)
        assert model2 == model1

        model2 = self.Model.load(target)
        assert model2 == model1

        model1 = self.Model(
            "AB",
            InterpolatorClass=interp.NearestNDInterpolator,
        )
        model1.state_dimension = 10
        model1.input_dimension = 4
        model1.save(target, overwrite=True)

        model2 = self.Model.load(target)
        assert model2 == model1

        os.remove(target)

    def test_copy(self, s=10, p=2, r=3):
        """Test _InterpModel._copy()."""

        model1 = self.Model(
            [
                opinf.operators.InterpConstantOperator(),
                opinf.operators.InterpLinearOperator(),
            ]
        )

        mu = np.random.random((s, p))
        model2 = self.Model(
            [
                opinf.operators.InterpConstantOperator(
                    mu, entries=np.random.random((s, r))
                ),
                opinf.operators.InterpLinearOperator(
                    mu, entries=np.random.random((s, r, r))
                ),
            ],
            InterpolatorClass=interp.NearestNDInterpolator,
        )

        for model in (model1, model2):
            model_copied = model.copy()
            assert isinstance(model_copied, self.Model)
            assert model_copied is not model
            assert model_copied == model


class TestInterpDiscreteModel(_TestInterpModel):
    """Test models.mono._parametric.InterpDiscreteModel."""

    Model = _module.InterpDiscreteModel
    _iscontinuous = False

    def test_fit(self, s=10, p=2, r=3, m=2, k=20):
        """Lightly test InterpDiscreteModel.fit()."""
        params = np.random.random((s, p))
        states = np.random.random((s, r, k))
        nextstates = np.random.random((s, r, k))
        inputs = np.random.random((s, m, k))

        model = self.Model("A")
        out = model.fit(params, states)
        assert out is model

        model = self.Model("AB")
        out = model.fit(params, states, nextstates, inputs)
        assert out is model

    def test_predict(self, s=11, r=4, m=2, niters=10):
        """Lightly test InterpDiscreteModel.predict()."""
        params = np.sort(np.random.random(s))
        state0 = np.random.random(r)
        model = self.Model(
            opinf.operators.InterpLinearOperator(params, np.zeros((s, r, r)))
        )
        out = model.predict(params[2], state0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)
        assert np.all(out[:, 0] == state0)
        assert np.all(out[:, 1:] == 0)

        inputs = np.random.random((m, niters))
        model = self.Model(
            opinf.operators.InterpInputOperator(params, np.zeros((s, r, m)))
        )
        out = model.predict(params[-2], state0, niters, inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)
        assert np.all(out[:, 0] == state0)
        assert np.all(out[:, 1:] == 0)


class TestInterpContinuousModel(_TestInterpModel):
    """Test models.mono._parametric.InterpContinuousModel."""

    Model = _module.InterpContinuousModel
    _iscontinuous = True

    def test_fit(self, s=10, p=2, r=3, m=2, k=20):
        """Test InterpContinuousModel.fit()."""
        params = np.random.random((s, p))
        states = np.random.random((s, r, k))
        ddts = np.random.random((s, r, k))
        inputs = np.random.random((s, m, k))

        model = self.Model("A")
        out = model.fit(params, states, ddts)
        assert out is model

        model = self.Model("AB")
        out = model.fit(params, states, ddts, inputs)
        assert out is model

    def test_predict(self, s=11, r=4, m=2, k=40):
        """Lightly test InterpContinuousModel.predict()."""
        params = np.sort(np.random.random(s))
        state0 = np.random.random(r)
        t = np.linspace(0, 1, k)
        model = self.Model(
            opinf.operators.InterpLinearOperator(params, np.zeros((s, r, r)))
        )
        out = model.predict(params[2], state0, t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, k)
        for j in range(k):
            assert np.allclose(out[:, j], state0)

        def input_func(t):
            return np.random.random(m)

        model = self.Model(
            opinf.operators.InterpInputOperator(params, np.zeros((s, r, m)))
        )
        out = model.predict(params[-2], state0, t, input_func)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, k)
        for j in range(k):
            assert np.allclose(out[:, j], state0)


# Deprecations models =========================================================
def test_deprecations():
    """Ensure deprecated classes still work."""
    for ModelClass in [
        _module.InterpolatedContinuousModel,
        _module.InterpolatedDiscreteModel,
    ]:
        with pytest.warns(DeprecationWarning) as wn:
            ModelClass("A")
        assert len(wn) == 1


if __name__ == "__main__":
    pytest.main([__file__])
