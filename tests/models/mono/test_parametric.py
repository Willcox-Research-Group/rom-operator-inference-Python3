# models/mono/test_parametric.py
"""Tests for models.mono._parametric."""

import os

# import h5py
import pytest
import numpy as np
import scipy.interpolate as interp

import opinf


_module = opinf.models.mono._parametric
_applyvalue = 7
_jacvalue = 11
_predictvalue = 13


# Dummy classes ===============================================================
class DummyNonparametricOperator(
    opinf.operators_new._base._NonparametricOperator
):
    """Instantiable version of _NonparametricOperator"""

    def _str(*args, **kwargs):  # pragma: no cover
        pass

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

    def set_entries(self, entries):  # pragma: no cover
        opinf.operators_new._base._NonparametricOperator.set_entries(
            self, entries
        )


class DummyNonparametricOperator2(DummyNonparametricOperator):
    """Another NonparametricOperator (since duplicates not allowed)."""


class DummyParametricOperator(opinf.operators_new._base._ParametricOperator):
    """Instantiable version of ParametricOperator."""

    _OperatorClass = DummyNonparametricOperator

    def __init__(self, entries=None):
        opinf.operators_new._base._ParametricOperator.__init__(self)
        self.entries = entries

    def _clear(*args, **kwargs):  # pragma: no cover
        pass

    def copy(*args, **kwargs):  # pragma: no cover
        pass

    def datablock(*args, **kwargs):  # pragma: no cover
        pass

    def evaluate(self, *args, **kwargs):  # pragma: no cover
        return self._OperatorClass(self.entries)

    def galerkin(*args, **kwargs):  # pragma: no cover
        pass

    def load(*args, **kwargs):  # pragma: no cover
        pass

    def operator_dimension(*args, **kwargs):  # pragma: no cover
        pass

    def save(*args, **kwargs):  # pragma: no cover
        pass

    def shape(*args, **kwargs):  # pragma: no cover
        pass

    @property
    def state_dimension(self):  # pragma: no cover
        return self.entries.shape[0]


class DummyParametricOperator2(DummyParametricOperator):
    """Another ParametricOperator with a different OperatorClass."""

    _OperatorClass = DummyNonparametricOperator2


class DummyInterpolatedOperator(
    opinf.operators_new._interpolate._InterpolatedOperator
):
    pass


class DummyNonparametricModel(
    opinf.models.mono._nonparametric._NonparametricMonolithicModel
):
    """Instantiable version of _NonparametricMonolithicModel."""

    _LHS_ARGNAME = "mylhs"

    def predict(*args, **kwargs):
        return _predictvalue


class DummyNonparametricModel2(DummyNonparametricModel):
    pass


# Tests =======================================================================
class TestParametricMonolithicModel:
    """Test models.mono._parametric._ParametricMonolithicModel."""

    class Dummy(_module._ParametricMonolithicModel):
        _ModelClass = DummyNonparametricModel

    def test_check_operator_types_unique(self):
        """Test _ParametricMonolithicModel._check_operator_types_unique()."""
        operators = [DummyParametricOperator(), DummyNonparametricOperator()]

        with pytest.raises(ValueError) as ex:
            self.Dummy._check_operator_types_unique(operators)
        assert (
            ex.value.args[0] == "duplicate type in list of operators to infer"
        )

        operators = [DummyParametricOperator(), DummyNonparametricOperator()]

        with pytest.raises(ValueError) as ex:
            self.Dummy._check_operator_types_unique(operators)
        assert (
            ex.value.args[0] == "duplicate type in list of operators to infer"
        )

        operators = [DummyParametricOperator(), DummyParametricOperator2()]
        self.Dummy._check_operator_types_unique(operators)

    def test_set_operators(self):
        """Test _ParametricMonolithicModel.operators.fset()."""
        operators = [DummyNonparametricOperator()]

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            self.Dummy(operators)
        assert (
            wn[0].message.args[0] == "no parametric operators detected, "
            "consider using a nonparametric model class"
        )

        operators = [DummyInterpolatedOperator()]

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            self.Dummy(operators)
        assert (
            wn[0].message.args[0] == "all operators interpolatory, "
            "consider using an InterpolatedModel class"
        )

        operators = [DummyParametricOperator(), DummyParametricOperator2()]
        model = self.Dummy(operators)
        assert model.parameter_dimension is None

    def test_get_operator_of_type(self):
        """Test _ParametricMonolithicModel._get_operator_of_type()."""
        op1 = DummyParametricOperator()
        op2 = DummyParametricOperator2()
        model = self.Dummy([op1, op2])

        op = model._get_operator_of_type(DummyNonparametricOperator)
        assert op is op1

        op = model._get_operator_of_type(DummyNonparametricOperator2)
        assert op is op2

        op = model._get_operator_of_type(float)
        assert op is None

    def test_check_parameter_dimension_consistency(self, s=3):
        """Test _check_parameter_dimension_consistency()."""
        op = DummyNonparametricOperator()
        p = self.Dummy._check_parameter_dimension_consistency([op])
        assert p is None

        op1 = DummyParametricOperator()
        op1._set_parameter_dimension_from_data(np.empty((s, 10)))
        p = self.Dummy._check_parameter_dimension_consistency([op1])
        assert p == 10

        op2 = DummyParametricOperator2()
        op2._set_parameter_dimension_from_data(np.empty((s, 20)))

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy._check_parameter_dimension_consistency([op1, op2])
        assert (
            ex.value.args[0] == "operators not aligned "
            "(parameter_dimension must be the same for all operators)"
        )

    def test_parameter_dimension(self, s=3, p=4):
        """Test _ParametricMonolithicModel.parameter_dimension."""
        op = DummyParametricOperator()
        model = self.Dummy([op, DummyNonparametricOperator2()])

        model._set_parameter_dimension_from_data(np.empty((s, p)))
        assert model.parameter_dimension == p

        model.parameter_dimension = 10
        assert model.parameter_dimension == 10

        op._set_parameter_dimension_from_data(np.empty((s, 20)))

        with pytest.raises(AttributeError) as ex:
            model.parameter_dimension = 15
        assert (
            ex.value.args[0] == "can't set attribute "
            "(existing operators have p = 10)"
        )

        model.parameter_dimension = 20
        assert model.parameter_dimension == 20

        model = self.Dummy(DummyParametricOperator())
        model._set_parameter_dimension_from_data(np.empty(s))
        assert model.parameter_dimension == 1

        with pytest.raises(ValueError) as ex:
            model._set_parameter_dimension_from_data(np.empty((s, s, s)))
        assert (
            ex.value.args[0] == "parameter values must be scalars or 1D arrays"
        )

    def test_process_fit_arguments(self, s=5, p=2, m=4, r=3, k=10):
        """Test _ParametricMonolithicModel._process_fit_arguments()."""
        # Intrusive case.
        op = DummyParametricOperator(np.random.random((3, 3)))
        model = self.Dummy([op])

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            out = model._process_fit_arguments(None, None, None, None)
        assert (
            wn[0].message.args[0] == "all operators initialized intrusively, "
            "nothing to learn"
        )
        assert len(out) == 5
        assert all(x is None for x in out)

        # Inconsistent number of training sets.
        op.entries = None
        model = self.Dummy([op])
        params = np.empty((s, p))
        states = [np.empty((r, k)) for _ in range(s)]
        lhs = [np.empty((r, k)) for _ in range(s)]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states[1:], None, None)
        assert (
            ex.value.args[0] == f"len(states) = {s-1} != {s} = len(parameters)"
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
        assert (
            ex.value.args[0] == f"mylhs[1].shape[-1] = {k} "
            f"!= {k-1} = states[1].shape[-1]"
        )

        # Inconsistent input dimension.
        states[1] = np.empty((r, k))
        inputs = [np.empty((m, k)) for _ in range(s)]
        inputs[1] = np.empty((m - 1, k))
        model._has_inputs = True
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            model._process_fit_arguments(params, states, lhs, inputs)
        assert ex.value.args[0] == f"inputs[1].shape[0] = {m-1} " f"!= {m} = m"

        # Correct usage, partially intrusive
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op, op2])
        model._process_fit_arguments(params, states, lhs, None)

        model._has_inputs = True
        inputs[1] = np.empty((m, k))
        model._process_fit_arguments(params, states, lhs, inputs)

    def test_evaluate(self, r=4):
        """Test _ParametricMonolithicModel.evaluate()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        model_evaluated = model.evaluate(None)
        assert isinstance(model_evaluated, DummyNonparametricModel)
        assert len(model_evaluated.operators) == 2
        assert isinstance(
            model_evaluated.operators[0], DummyNonparametricOperator
        )
        assert isinstance(
            model_evaluated.operators[1], DummyNonparametricOperator2
        )
        assert model_evaluated.state_dimension == r

    def test_rhs(self, r=2):
        """Test _ParametricMonolithicModel.rhs()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        assert model.state_dimension == r
        assert model.rhs(np.empty(r), None, None) == 2 * _applyvalue

    def test_jacobian(self, r=3):
        """Test _ParametricMonolithicModel.jacobian()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        assert model.state_dimension == r
        assert np.all(model.jacobian(np.empty(r), None, None) == 2 * _jacvalue)

    def test_predict(self, r=4):
        """Test _ParametricMonolithicModel.predict()."""
        op1 = DummyParametricOperator(np.random.random((r, r)))
        op2 = DummyParametricOperator2(np.random.random((r, r)))
        model = self.Dummy([op1, op2])
        assert model.state_dimension == r
        assert model.predict(None) == _predictvalue


class TestInterpolatedModel:
    """Test models.mono._parametric._InterpolatedMonolithicModel."""

    class Dummy(_module._InterpolatedMonolithicModel):
        _ModelClass = DummyNonparametricModel2

    def test_from_models(self, r=4):
        """Test _InterpolatedMonolithicModel._from_models()."""
        mu = np.sort(np.random.random(2))
        model1 = DummyNonparametricModel(
            [DummyNonparametricOperator2(np.random.random(r))]
        )

        # Wrong type of model.
        model2 = self.Dummy([opinf.operators_new.InterpolatedCubicOperator()])
        with pytest.raises(TypeError) as ex:
            self.Dummy._from_models(mu, [model2, model1])
        assert (
            ex.value.args[0] == "expected models of type "
            "'DummyNonparametricModel'"
        )

        # Inconsistent number of operators.
        model2 = DummyNonparametricModel(
            [DummyNonparametricOperator(), DummyNonparametricOperator2()]
        )
        with pytest.raises(ValueError) as ex:
            self.Dummy._from_models(mu, [model1, model2])
        assert (
            ex.value.args[0] == "models not aligned "
            "(inconsistent number of operators)"
        )

        # Inconsistent operator types.
        model2 = DummyNonparametricModel(
            [DummyNonparametricOperator(np.random.random(r))]
        )
        with pytest.raises(ValueError) as ex:
            self.Dummy._from_models(mu, [model1, model2])
        assert (
            ex.value.args[0] == "models not aligned "
            "(inconsistent operator types)"
        )

        # Correct usage
        OpClass = opinf.operators_new.ConstantOperator
        model1 = DummyNonparametricModel([OpClass(np.random.random(r))])
        model2 = DummyNonparametricModel([OpClass(np.random.random(r))])
        model = self.Dummy._from_models(mu, [model1, model2])
        assert isinstance(model, self.Dummy)
        assert len(model.operators) == 1
        assert isinstance(
            model.operators[0],
            opinf.operators_new.InterpolatedConstantOperator,
        )

    def test_set_interpolator(self, s=10, p=2, r=2):
        """Test _InterpolatedMonolithicModel._set_interpolator()."""

        mu = np.random.random((s, p))
        operators = [
            opinf.operators_new.InterpolatedConstantOperator(
                mu,
                interp.NearestNDInterpolator,
                entries=np.random.random((s, r)),
            ),
            opinf.operators_new.InterpolatedLinearOperator(
                mu,
                interp.NearestNDInterpolator,
                entries=np.random.random((s, r, r)),
            ),
        ]

        model = self.Dummy(operators)
        for op in operators:
            assert isinstance(op.interpolator, interp.NearestNDInterpolator)

        model = self.Dummy(
            operators, InterpolatorClass=interp.LinearNDInterpolator
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
        """Test _InterpolatedMonolithicModel._fit_solver()."""
        operators = [
            opinf.operators_new.InterpolatedConstantOperator(),
            opinf.operators_new.InterpolatedLinearOperator(),
        ]
        params = np.sort(np.random.random(s))
        states = np.random.random((s, r, k))
        lhs = np.random.random((s, r, k))

        model = self.Dummy(operators)
        model._fit_solver(params, states, lhs)

        assert hasattr(model, "solvers_")
        assert len(model.solvers_) == s
        for solver in model.solvers_:
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

    def test_evaluate_solver(self, s=10, r=3, k=15):
        """Test _InterpolatedMonolithicModel._evaluate_solver()."""
        operators = [
            opinf.operators_new.InterpolatedConstantOperator(),
            opinf.operators_new.InterpolatedLinearOperator(),
        ]
        params = np.sort(np.random.random(s))
        states = np.random.random((s, r, k))
        lhs = np.random.random((s, r, k))

        model = self.Dummy(operators)

        with pytest.raises(RuntimeError) as ex:
            model._evaluate_solver()
        assert (
            ex.value.args[0] == "model solvers not set, "
            "call _fit_solver() first"
        )

        model._fit_solver(params, states, lhs)
        model._evaluate_solver()

        assert hasattr(model, "_submodels")
        assert len(model._submodels) == s
        for mdl in model._submodels:
            assert isinstance(mdl, DummyNonparametricModel)
            assert len(mdl.operators) == len(operators)
            for op in mdl.operators:
                assert op.entries is not None

    def test_save(self, target="_interpmodelsavetest.h5"):
        """Test _InterpolatedMonolithicModel._save()."""
        if os.path.isfile(target):
            os.remove(target)

        model = self.Dummy(
            [
                opinf.operators_new.InterpolatedConstantOperator(),
                opinf.operators_new.InterpolatedLinearOperator(),
            ]
        )
        model.save(target)
        assert os.path.isfile(target)

        model.set_interpolator(interp.CubicSpline)
        model.save(target, overwrite=True)
        assert os.path.isfile(target)

        model.set_interpolator(float)
        os.remove(target)
        with pytest.warns(opinf.errors.UsageWarning) as wn:
            model.save(target, overwrite=True)
        assert len(wn) == 1
        assert (
            wn[0].message.args[0] == "cannot serialize InterpolatorClass "
            "'float', must pass in the class when calling load()"
        )
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self, target="_interpmodelloadtest.h5"):
        """Test _InterpolatedMonolithicModel._load()."""
        if os.path.isfile(target):
            os.remove(target)

        operators = [
            opinf.operators_new.InterpolatedConstantOperator(),
            opinf.operators_new.InterpolatedLinearOperator(),
        ]
        model = self.Dummy(operators, InterpolatorClass=float)

        with pytest.warns(opinf.errors.UsageWarning):
            model.save(target)

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Dummy.load(target)
        assert (
            ex.value.args[0] == "unknown InterpolatorClass "
            f"'float', call load({target}, float)"
        )
        self.Dummy.load(target, float)

        model1 = self.Dummy(
            operators,
            InterpolatorClass=interp.NearestNDInterpolator,
        )
        model1.save(target, overwrite=True)

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            model2 = self.Dummy.load(target, float)
        assert wn[0].message.args[0] == (
            "InterpolatorClass=float does not match loadfile "
            "InterpolatorClass 'NearestNDInterpolator'"
        )
        model2.set_interpolator(interp.NearestNDInterpolator)
        assert model2 == model1

        model2 = self.Dummy.load(target)
        assert model2 == model1

    def test_copy(self, s=10, p=2, r=3):
        """Test _InterpolatedMonolithicModel._copy()."""

        model1 = self.Dummy(
            [
                opinf.operators_new.InterpolatedConstantOperator(),
                opinf.operators_new.InterpolatedLinearOperator(),
            ]
        )

        mu = np.random.random((s, p))
        model2 = self.Dummy(
            [
                opinf.operators_new.InterpolatedConstantOperator(
                    mu, entries=np.random.random((s, r))
                ),
                opinf.operators_new.InterpolatedLinearOperator(
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
