# models/mono/test_parametric.py
"""Tests for models.mono._parametric."""

import pytest
import numpy as np

import opinf

_module = opinf.models.mono._parametric


# Dummy classes ===============================================================
class DummyNonparametricOperator(
    opinf.operators_new._base._NonparametricOperator
):
    """Instantiable version of _NonparametricOperator"""

    def _str(*args, **kwargs):  # pragma: no cover
        pass

    def apply(*args, **kwargs):  # pragma: no cover
        return 0

    def datablock(*args, **kwargs):  # pragma: no cover
        pass

    def galerkin(*args, **kwargs):  # pragma: no cover
        pass

    def operator_dimension(*args, **kwargs):  # pragma: no cover
        pass

    def set_entries(*args, **kwargs):  # pragma: no cover
        pass


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
        return 101


# Tests =======================================================================
class TestParametricMonolithicModel:
    """Test opinf.models.mono._parametric._ParametricMonolithicModel."""

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

        with pytest.warns(UserWarning) as wn:
            self.Dummy(operators)
        assert (
            wn[0].message.args[0] == "no parametric operators detected, "
            "consider using a nonparametric model class"
        )

        operators = [DummyInterpolatedOperator()]

        with pytest.warns(UserWarning) as wn:
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


#     def test_evaluate(self):
#         """Test _ParametricMonolithicModel.evaluate()."""
#         raise NotImplementedError

#     def test_rhs(self):
#         """Test _ParametricMonolithicModel.rhs()."""
#         raise NotImplementedError

#     def test_jacobian(self):
#         """Test _ParametricMonolithicModel.jacobian()."""
#         raise NotImplementedError

#     def test_predict(self):
#         """Test _ParametricMonolithicModel.predict()."""
#         raise NotImplementedError


# class TestInterpolatedModel:
#     def test_from_models(self):
#         """Test _InterpolatedMonolithicModel._from_models()."""
#         raise NotImplementedError

#     def test_set_interpolator(self):
#         """Test _InterpolatedMonolithicModel._set_interpolator()."""
#         raise NotImplementedError

#     def test_fit_solver(self):
#         """Test _InterpolatedMonolithicModel._fit_solver()."""
#         raise NotImplementedError

#     def test_evaluate_solver(self):
#         """Test _InterpolatedMonolithicModel._evaluate_solver()."""
#         raise NotImplementedError

#     def test_save(self):
#         """Test _InterpolatedMonolithicModel._save()."""
#         raise NotImplementedError

#     def test_load(self):
#         """Test _InterpolatedMonolithicModel._load()."""
#         raise NotImplementedError

#     def test_copy(self):
#         """Test _InterpolatedMonolithicModel._copy()."""
#         raise NotImplementedError
