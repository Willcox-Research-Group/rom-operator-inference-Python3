# models/mono/test_parametric.py
"""Tests for models.mono._parametric."""

import os

import pytest
import numpy as np
import scipy.interpolate as interp

import opinf


_module = opinf.models.mono._parametric
_applyvalue = 7
_jacvalue = 11
_predictvalue = 13


# Dummy classes ===============================================================
class DummyOpInfOperator(opinf.operators.OpInfOperator):
    """Instantiable version of OpInfOperator."""

    def apply(*args, **kwargs):  # pragma: no cover
        return _applyvalue

    def jacobian(*args, **kwargs):
        return _jacvalue

    def datablock(*args, **kwargs):  # pragma: no cover
        pass

    def galerkin(*args, **kwargs):  # pragma: no cover
        pass

    def operator_dimension(*args, **kwargs):  # pragma: no cover
        pass


class DummyOpInfOperator2(DummyOpInfOperator):
    """Another OpInfOperator (since duplicates not allowed)."""


class DummyParametricOperator(opinf.operators.ParametricOpInfOperator):
    """Instantiable version of ParametricOpInfOperator."""

    _OperatorClass = DummyOpInfOperator

    def __init__(self, entries=None):
        super().__init__()
        if entries is not None:
            self.set_entries(entries)

    def set_entries(self, entries):
        super().set_entries(entries)

    def operator_dimension(*args, **kwargs):  # pragma: no cover
        pass

    def datablock(*args, **kwargs):  # pragma: no cover
        pass

    def evaluate(self, *args, **kwargs):  # pragma: no cover
        return self._OperatorClass(self.entries)

    # def galerkin(*args, **kwargs):  # pragma: no cover
    #     pass

    # def copy(*args, **kwargs):  # pragma: no cover
    #     pass

    # def load(*args, **kwargs):  # pragma: no cover
    #     pass

    # def save(*args, **kwargs):  # pragma: no cover
    #     pass


class DummyParametricOperator2(DummyParametricOperator):
    """Another ParametricOperator with a different OperatorClass."""

    _OperatorClass = DummyOpInfOperator2


class DummyInterpolatedOperator(
    opinf.operators._interpolate._InterpolatedOperator
):
    pass


class DummyNonparametricModel(
    opinf.models.mono._nonparametric._NonparametricModel
):
    """Instantiable version of _NonparametricModel."""

    _LHS_ARGNAME = "mylhs"

    def predict(*args, **kwargs):
        return _predictvalue


class DummyNonparametricModel2(DummyNonparametricModel):
    pass


# Tests =======================================================================
class TestParametricModel:
    """Test models.mono._parametric._ParametricModel."""

    class Dummy(_module._ParametricModel):
        _ModelClass = DummyNonparametricModel

    def test_check_operator_types_unique(self):
        """Test _ParametricModel._check_operator_types_unique()."""
        operators = [DummyParametricOperator(), DummyOpInfOperator()]

        with pytest.raises(ValueError) as ex:
            self.Dummy._check_operator_types_unique(operators)
        assert ex.value.args[0] == (
            "duplicate type in list of operators to infer"
        )

        operators = [DummyParametricOperator(), DummyOpInfOperator()]

        with pytest.raises(ValueError) as ex:
            self.Dummy._check_operator_types_unique(operators)
        assert ex.value.args[0] == (
            "duplicate type in list of operators to infer"
        )

        operators = [DummyParametricOperator(), DummyParametricOperator2()]
        self.Dummy._check_operator_types_unique(operators)

    def test_set_operators(self):
        """Test _ParametricModel.operators.fset()."""
        operators = [DummyOpInfOperator()]

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Dummy(operators)
        assert wn[0].message.args[0] == (
            "no parametric operators detected, "
            "consider using a nonparametric model class"
        )

        operators = [DummyInterpolatedOperator()]

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Dummy(operators)
        assert wn[0].message.args[0] == (
            "all operators interpolatory, "
            "consider using an InterpolatedModel class"
        )

        operators = [DummyParametricOperator(), DummyParametricOperator2()]
        model = self.Dummy(operators)
        assert model.parameter_dimension is None

    def test_get_operator_of_type(self):
        """Test _ParametricModel._get_operator_of_type()."""
        op1 = DummyParametricOperator()
        op2 = DummyParametricOperator2()
        model = self.Dummy([op1, op2])

        op = model._get_operator_of_type(DummyOpInfOperator)
        assert op is op1

        op = model._get_operator_of_type(DummyOpInfOperator2)
        assert op is op2

        op = model._get_operator_of_type(float)
        assert op is None

    def test_check_parameter_dimension_consistency(self, s=3):
        """Test _check_parameter_dimension_consistency()."""
        op = DummyOpInfOperator()
        p = self.Dummy._check_parameter_dimension_consistency([op])
        assert p is None

        op1 = DummyParametricOperator()
        op1._set_parameter_dimension_from_values(np.empty((s, 10)))
        p = self.Dummy._check_parameter_dimension_consistency([op1])
        assert p == 10

        op2 = DummyParametricOperator2()
        op2._set_parameter_dimension_from_values(np.empty((s, 20)))

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy._check_parameter_dimension_consistency([op1, op2])
        assert ex.value.args[0] == (
            "operators not aligned "
            "(parameter_dimension must be the same for all operators)"
        )

    def test_parameter_dimension(self, s=3, p=4):
        """Test _ParametricModel.parameter_dimension."""
        op = DummyParametricOperator()
        model = self.Dummy([op, DummyOpInfOperator2()])

        model._set_parameter_dimension_from_values(np.empty((s, p)))
        assert model.parameter_dimension == p

        model.parameter_dimension = 10
        assert model.parameter_dimension == 10

        op._set_parameter_dimension_from_values(np.empty((s, 20)))

        with pytest.raises(AttributeError) as ex:
            model.parameter_dimension = 15
        assert ex.value.args[0] == (
            "can't set attribute (existing operators have p = 10)"
        )

        model.parameter_dimension = 20
        assert model.parameter_dimension == 20

        model = self.Dummy(DummyParametricOperator())
        model._set_parameter_dimension_from_values(np.empty(s))
        assert model.parameter_dimension == 1

        with pytest.raises(ValueError) as ex:
            model._set_parameter_dimension_from_values(np.empty((s, s, s)))
        assert ex.value.args[0] == (
            "parameter values must be scalars or 1D arrays"
        )

    def test_process_fit_arguments(self, s=5, p=2, m=4, r=3, k=10):
        """Test _ParametricModel._process_fit_arguments()."""
        op = DummyParametricOperator()
        model = self.Dummy([op])
        params = np.empty((s, p))
        states = [np.empty((r, k)) for _ in range(s)]
        lhs = [np.empty((r, k)) for _ in range(s)]

        # Inconsistent number of parameter values.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states[1:], None, None)
        assert ex.value.args[0] == (
            f"len(states) = {s-1} != {s} = len(parameters)"
        )

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
            f"mylhs[1].shape[-1] = {k} != {k-1} = states[1].shape[-1]"
        )

        # Inconsistent input dimension.
        states[1] = np.empty((r, k))
        inputs = [np.empty((m, k)) for _ in range(s)]
        inputs[1] = np.empty((m - 1, k))
        model._has_inputs = True
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs, inputs)
        assert ex.value.args[0] == f"inputs[1].shape[0] = {m-1} != {m} = m"

        # Correct usage, partially intrusive
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op, op2])
        model._process_fit_arguments(params, states, lhs, None)

        model._has_inputs = True
        inputs[1] = np.empty((m, k))
        model._process_fit_arguments(params, states, lhs, inputs)

    def test_evaluate(self, r=4):
        """Test _ParametricModel.evaluate()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        model_evaluated = model.evaluate(None)
        assert isinstance(model_evaluated, DummyNonparametricModel)
        assert len(model_evaluated.operators) == 2
        assert isinstance(model_evaluated.operators[0], DummyOpInfOperator)
        assert isinstance(model_evaluated.operators[1], DummyOpInfOperator2)
        assert model_evaluated.state_dimension == r

    def test_rhs(self, r=2):
        """Test _ParametricModel.rhs()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        assert model.state_dimension == r
        assert model.rhs(np.empty(r), None, None) == 2 * _applyvalue

    def test_jacobian(self, r=3):
        """Test _ParametricModel.jacobian()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        assert model.state_dimension == r
        assert np.all(model.jacobian(np.empty(r), None, None) == 2 * _jacvalue)

    def test_predict(self, r=4):
        """Test _ParametricModel.predict()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        assert model.state_dimension == r
        assert model.predict(None) == _predictvalue


class TestInterpolatedModel:
    """Test models.mono._parametric._InterpolatedModel."""

    class Dummy(_module._InterpolatedModel):
        _ModelClass = DummyNonparametricModel2

    def test_from_models(self, r=4):
        """Test _InterpolatedModel._from_models()."""
        mu = np.sort(np.random.random(2))
        model1 = DummyNonparametricModel(
            [DummyOpInfOperator2(np.random.random(r))]
        )

        # Wrong type of model.
        model2 = self.Dummy([opinf.operators.InterpolatedCubicOperator()])
        with pytest.raises(TypeError) as ex:
            self.Dummy._from_models(mu, [model2, model1])
        assert ex.value.args[0] == (
            "expected models of type 'DummyNonparametricModel'"
        )

        # Inconsistent number of operators.
        model2 = DummyNonparametricModel(
            [DummyOpInfOperator(), DummyOpInfOperator2()]
        )
        with pytest.raises(ValueError) as ex:
            self.Dummy._from_models(mu, [model1, model2])
        assert ex.value.args[0] == (
            "models not aligned (inconsistent number of operators)"
        )

        # Inconsistent operator types.
        model2 = DummyNonparametricModel(
            [DummyOpInfOperator(np.random.random(r))]
        )
        with pytest.raises(ValueError) as ex:
            self.Dummy._from_models(mu, [model1, model2])
        assert ex.value.args[0] == (
            "models not aligned (inconsistent operator types)"
        )

        # Correct usage
        OpClass = opinf.operators.ConstantOperator
        model1 = DummyNonparametricModel([OpClass(np.random.random(r))])
        model2 = DummyNonparametricModel([OpClass(np.random.random(r))])
        model = self.Dummy._from_models(mu, [model1, model2])
        assert isinstance(model, self.Dummy)
        assert len(model.operators) == 1
        assert isinstance(
            model.operators[0],
            opinf.operators.InterpolatedConstantOperator,
        )

    def test_set_interpolator(self, s=10, p=2, r=2):
        """Test _InterpolatedModel._set_interpolator()."""

        mu = np.random.random((s, p))
        operators = [
            opinf.operators.InterpolatedConstantOperator(
                training_parameters=mu,
                entries=np.random.random((s, r)),
                InterpolatorClass=interp.NearestNDInterpolator,
            ),
            opinf.operators.InterpolatedLinearOperator(
                training_parameters=mu,
                entries=np.random.random((s, r, r)),
                InterpolatorClass=interp.NearestNDInterpolator,
            ),
        ]

        model = self.Dummy(operators)
        for op in operators:
            assert isinstance(op.interpolator, interp.NearestNDInterpolator)

        model = self.Dummy(
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
        """Test _InterpolatedModel._fit_solver()."""
        operators = [
            opinf.operators.InterpolatedConstantOperator(),
            opinf.operators.InterpolatedLinearOperator(),
        ]
        params = np.sort(np.random.random(s))
        states = np.random.random((s, r, k))
        lhs = np.random.random((s, r, k))

        model = self.Dummy(operators)
        model._fit_solver(params, states, lhs)

        assert hasattr(model, "solvers")
        assert len(model.solvers) == s
        for solver in model.solvers:
            assert isinstance(solver, opinf.lstsq.PlainSolver)

        assert hasattr(model, "_submodels")
        assert len(model._submodels) == s
        for mdl in model._submodels:
            assert isinstance(mdl, DummyNonparametricModel)
            assert len(mdl.operators) == len(operators)
            for op in mdl.operators:
                assert op.entries is None

        assert hasattr(model, "_training_parameters")
        assert isinstance(model._training_parameters, np.ndarray)
        assert np.all(model._training_parameters == params)

    def test_refit(self, s=10, r=3, k=15):
        """Test _InterpolatedModel.refit()."""
        operators = [
            opinf.operators.InterpolatedConstantOperator(),
            opinf.operators.InterpolatedLinearOperator(),
        ]
        params = np.sort(np.random.random(s))
        states = np.random.random((s, r, k))
        lhs = np.random.random((s, r, k))

        model = self.Dummy(operators)

        with pytest.raises(RuntimeError) as ex:
            model.refit()
        assert ex.value.args[0] == "model solvers not set, call fit() first"

        model._fit_solver(params, states, lhs)
        model.refit()

        assert hasattr(model, "_submodels")
        assert len(model._submodels) == s
        for mdl in model._submodels:
            assert isinstance(mdl, DummyNonparametricModel)
            assert len(mdl.operators) == len(operators)
            for op in mdl.operators:
                assert op.entries is not None

    def test_save(self, target="_interpmodelsavetest.h5"):
        """Test _InterpolatedModel._save()."""
        if os.path.isfile(target):
            os.remove(target)

        model = self.Dummy(
            [
                opinf.operators.InterpolatedConstantOperator(),
                opinf.operators.InterpolatedLinearOperator(),
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
        """Test _InterpolatedModel._load()."""
        if os.path.isfile(target):
            os.remove(target)

        operators = [
            opinf.operators.InterpolatedConstantOperator(),
            opinf.operators.InterpolatedLinearOperator(),
        ]
        model = self.Dummy(operators, InterpolatorClass=float)

        with pytest.warns(opinf.errors.OpInfWarning):
            model.save(target)

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Dummy.load(target)
        assert ex.value.args[0] == (
            f"unknown InterpolatorClass 'float', call load({target}, float)"
        )
        self.Dummy.load(target, float)

        model1 = self.Dummy(
            operators,
            InterpolatorClass=interp.NearestNDInterpolator,
        )
        model1.save(target, overwrite=True)

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            model2 = self.Dummy.load(target, float)
        assert wn[0].message.args[0] == (
            "InterpolatorClass=float does not match loadfile "
            "InterpolatorClass 'NearestNDInterpolator'"
        )
        model2.set_interpolator(interp.NearestNDInterpolator)
        assert model2 == model1

        model2 = self.Dummy.load(target)
        assert model2 == model1

        model1 = self.Dummy(
            "AB",
            InterpolatorClass=interp.NearestNDInterpolator,
        )
        model1.state_dimension = 10
        model1.input_dimension = 4
        model1.save(target, overwrite=True)

        model2 = self.Dummy.load(target)
        assert model2 == model1

        os.remove(target)

    def test_copy(self, s=10, p=2, r=3):
        """Test _InterpolatedModel._copy()."""

        model1 = self.Dummy(
            [
                opinf.operators.InterpolatedConstantOperator(),
                opinf.operators.InterpolatedLinearOperator(),
            ]
        )

        mu = np.random.random((s, p))
        model2 = self.Dummy(
            [
                opinf.operators.InterpolatedConstantOperator(
                    mu, entries=np.random.random((s, r))
                ),
                opinf.operators.InterpolatedLinearOperator(
                    mu, entries=np.random.random((s, r, r))
                ),
            ],
            InterpolatorClass=interp.NearestNDInterpolator,
        )

        for model in (model1, model2):
            model_copied = model.copy()
            assert isinstance(model_copied, self.Dummy)
            assert model_copied is not model
            assert model_copied == model


class TestInterpolatedDiscreteModel:
    """Test models.mono._parametric.InterpolatedDiscreteModel."""

    ModelClass = _module.InterpolatedDiscreteModel

    def test_fit(self, s=10, p=2, r=3, m=2, k=20):
        """Lightly test InterpolatedDiscreteModel.fit()."""
        params = np.random.random((s, p))
        states = np.random.random((s, r, k))
        nextstates = np.random.random((s, r, k))
        inputs = np.random.random((s, m, k))

        model = self.ModelClass("A")
        out = model.fit(params, states)
        assert out is model

        model = self.ModelClass("AB")
        out = model.fit(params, states, nextstates, inputs)
        assert out is model

    def test_rhs(self, s=10, r=3, m=2):
        """Lightly test InterpolatedDiscreteModel.rhs()."""
        params = np.sort(np.random.random(s))
        state = np.random.random(r)
        model = self.ModelClass(
            opinf.operators.InterpolatedLinearOperator(
                params, np.random.random((s, r, r))
            )
        )
        out = model.rhs(params[2], state)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,)

        input_ = np.random.random(m)
        model = self.ModelClass(
            opinf.operators.InterpolatedInputOperator(
                params, np.random.random((s, r, m))
            )
        )
        out = model.rhs(params[-2], state, input_)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,)

    def test_jacobian(self, s=9, r=2, m=3):
        """Lightly test InterpolatedDiscreteModel.jacobian()."""
        params = np.sort(np.random.random(s))
        state = np.random.random(r)
        model = self.ModelClass(
            opinf.operators.InterpolatedLinearOperator(
                params, np.random.random((s, r, r))
            )
        )
        out = model.jacobian(params[2], state)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, r)

        input_ = np.random.random(m)
        model = self.ModelClass(
            opinf.operators.InterpolatedInputOperator(
                params, np.random.random((s, r, m))
            )
        )
        out = model.jacobian(params[-2], state, input_)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, r)

    def test_predict(self, s=11, r=4, m=2, niters=10):
        """Lightly test InterpolatedDiscreteModel.predict()."""
        params = np.sort(np.random.random(s))
        state0 = np.random.random(r)
        model = self.ModelClass(
            opinf.operators.InterpolatedLinearOperator(
                params, np.zeros((s, r, r))
            )
        )
        out = model.predict(params[2], state0, niters)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)
        assert np.all(out[:, 0] == state0)
        assert np.all(out[:, 1:] == 0)

        inputs = np.random.random((m, niters))
        model = self.ModelClass(
            opinf.operators.InterpolatedInputOperator(
                params, np.zeros((s, r, m))
            )
        )
        out = model.predict(params[-2], state0, niters, inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, niters)
        assert np.all(out[:, 0] == state0)
        assert np.all(out[:, 1:] == 0)


class TestInterpolatedContinuousModel:
    """Test models.mono._parametric.InterpolatedContinuousModel."""

    ModelClass = _module.InterpolatedContinuousModel

    def test_fit(self, s=10, p=2, r=3, m=2, k=20):
        """Test InterpolatedContinuousModel.fit()."""
        params = np.random.random((s, p))
        states = np.random.random((s, r, k))
        ddts = np.random.random((s, r, k))
        inputs = np.random.random((s, m, k))

        model = self.ModelClass("A")
        out = model.fit(params, states, ddts)
        assert out is model

        model = self.ModelClass("AB")
        out = model.fit(params, states, ddts, inputs)
        assert out is model

    def test_rhs(self, s=10, r=3, m=2):
        """Lightly test InterpolatedContinuousModel.rhs()."""
        params = np.sort(np.random.random(s))
        state = np.random.random(r)
        model = self.ModelClass(
            opinf.operators.InterpolatedLinearOperator(
                params, np.random.random((s, r, r))
            )
        )
        out = model.rhs(None, params[2], state)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,)

        def input_func(t):
            return np.random.random(m)

        model = self.ModelClass(
            opinf.operators.InterpolatedInputOperator(
                params, np.random.random((s, r, m))
            )
        )
        out = model.rhs(np.pi, params[-2], state, input_func)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r,)

    def test_jacobian(self, s=9, r=2, m=3):
        """Lightly test InterpolatedContinuousModel.jacobian()."""
        params = np.sort(np.random.random(s))
        state = np.random.random(r)
        model = self.ModelClass(
            opinf.operators.InterpolatedLinearOperator(
                params, np.random.random((s, r, r))
            )
        )
        out = model.jacobian(None, params[2], state)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, r)

        def input_func(t):
            return np.random.random(m)

        model = self.ModelClass(
            opinf.operators.InterpolatedInputOperator(
                params, np.random.random((s, r, m))
            )
        )
        out = model.jacobian(np.pi, params[-2], state, input_func)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, r)

    def test_predict(self, s=11, r=4, m=2, k=40):
        """Lightly test InterpolatedContinuousModel.predict()."""
        params = np.sort(np.random.random(s))
        state0 = np.random.random(r)
        t = np.linspace(0, 1, k)
        model = self.ModelClass(
            opinf.operators.InterpolatedLinearOperator(
                params, np.zeros((s, r, r))
            )
        )
        out = model.predict(params[2], state0, t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, k)
        for j in range(k):
            assert np.allclose(out[:, j], state0)

        def input_func(t):
            return np.random.random(m)

        model = self.ModelClass(
            opinf.operators.InterpolatedInputOperator(
                params, np.zeros((s, r, m))
            )
        )
        out = model.predict(params[-2], state0, t, input_func)
        assert isinstance(out, np.ndarray)
        assert out.shape == (r, k)
        for j in range(k):
            assert np.allclose(out[:, j], state0)


def test_publics():
    """Ensure all public ParametricModel classes can be instantiated."""
    operators = [opinf.operators.InterpolatedConstantOperator()]
    for ModelClassName in _module.__all__:
        ModelClass = getattr(_module, ModelClassName)
        if not isinstance(ModelClass, type) or not issubclass(
            ModelClass, _module._ParametricModel
        ):  # pragma: no cover
            continue
        model = ModelClass(operators)
        assert issubclass(
            model.ModelClass,
            opinf.models.mono._nonparametric._NonparametricModel,
        )
