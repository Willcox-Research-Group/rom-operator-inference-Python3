# models/mono/test_base.py
"""Tests for models.mono._base."""

import pytest
import numpy as np

import opinf

from . import _get_operators


_module = opinf.models.mono._base


class TestModel:
    """Test models._base._Model."""

    class Dummy(_module._Model):
        """Instantiable version of _Model."""

        def _isvalidoperator(self, op):
            return hasattr(opinf.operators, op.__class__.__name__)

        def _check_operator_types_unique(*args, **kwargs):
            return True

        def _get_operator_of_type(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    # Properties: operators ---------------------------------------------------
    def test_operators(self, m=4, r=7):
        """Test _Model.__init__(), operators, _clear(), and __iter__()."""
        operators = _get_operators("cAHGBN", r, m)

        # Try to instantiate without any operators.
        with pytest.raises(ValueError) as ex:
            self.Dummy([])
        assert ex.value.args[0] == "at least one operator required"

        # Try to instantiate with non-operator.
        with pytest.raises(TypeError) as ex:
            self.Dummy([1])
        assert ex.value.args[0] == "invalid operator of type 'int'"

        # Try to instantiate with operators of mismatched shape (no basis).
        bad_ops = [
            opinf.operators.LinearOperator(np.random.random((r, r))),
            opinf.operators.ConstantOperator(np.random.random(r + 1)),
        ]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy(bad_ops)
        assert (
            ex.value.args[0] == "operators not aligned "
            "(state dimension must be the same for all operators)"
        )

        # Try to instantiate with input operators not aligned.
        bad_ops = [
            opinf.operators.InputOperator(np.random.random((r, m - 1))),
            opinf.operators.StateInputOperator(np.random.random((r, r * m))),
        ]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy(bad_ops)
        assert (
            ex.value.args[0] == "input operators not aligned "
            "(input dimension must be the same for all input operators)"
        )

        # Try with bad operator abbrevation.
        with pytest.raises(TypeError) as ex:
            self.Dummy("X")
        assert ex.value.args[0] == "operator abbreviation 'X' not recognized"

        # Test operators.setter().
        model = self.Dummy(opinf.operators.ConstantOperator())
        for attr in (
            "operators",
            "input_dimension",
            "state_dimension",
            "_has_inputs",
            "_indices_of_operators_to_infer",
            "_indices_of_known_operators",
        ):
            assert hasattr(model, attr)
        # Operators with no entries, no inputs.
        assert len(model.operators) == 1
        assert isinstance(model.operators[0], opinf.operators.ConstantOperator)
        assert len(model._indices_of_operators_to_infer) == 1
        assert model._indices_of_operators_to_infer == [0]
        assert len(model._indices_of_known_operators) == 0
        assert model._has_inputs is False
        assert model.input_dimension == 0
        assert model.state_dimension is None

        # Operators with entries, no inputs.
        model = self.Dummy(operators[:2])
        assert len(model._indices_of_operators_to_infer) == 0
        assert len(model._indices_of_known_operators) == 2
        assert model._indices_of_known_operators == [0, 1]
        assert model._has_inputs is False
        assert model.input_dimension == 0
        assert model.state_dimension == r
        for i in range(2):
            assert model.operators[i] is operators[i]
            assert model.operators[i].entries is not None

        # Some operators with entries, some without; with inputs.
        operators[5]._clear()
        model = self.Dummy(operators[3:6])
        assert len(model._indices_of_operators_to_infer) == 1
        assert model._indices_of_operators_to_infer == [2]
        assert len(model._indices_of_known_operators) == 2
        assert model._indices_of_known_operators == [0, 1]
        assert model._has_inputs is True
        assert model.input_dimension == m
        assert model.state_dimension == r
        for i in model._indices_of_operators_to_infer:
            assert model.operators[i] is operators[i + 3]
            assert model.operators[i].entries is None
        for i in model._indices_of_known_operators:
            assert model.operators[i] is operators[i + 3]
            assert model.operators[i].entries is not None

        # Test __iter__().
        iterated = [op for op in model]
        assert len(iterated) == 3
        assert iterated == operators[3:6]

        # Test _clear().
        model.operators[2].set_entries(np.random.random((r, r * m)))
        model._clear()
        assert model.operators[2].entries is None
        assert len(model._indices_of_operators_to_infer) == 1
        assert model._indices_of_operators_to_infer == [2]
        assert len(model._indices_of_known_operators) == 2
        assert model._indices_of_known_operators == [0, 1]
        for i in model._indices_of_operators_to_infer:
            assert model.operators[i].entries is None
        for i in model._indices_of_known_operators:
            assert model.operators[i].entries is not None
        assert model.state_dimension == r  # Dimensions not erased.
        assert model.input_dimension == m  # Dimensions not erased.

    # Properties: dimensions --------------------------------------------------
    def test_dimension_properties(self, m=3, r=7):
        """Test the properties _Model.state_dimension and
        _Model.input_dimension.
        """
        # Case 1: no inputs.
        model = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.QuadraticOperator(),
            ]
        )
        assert model.state_dimension is None
        assert model.input_dimension == 0

        # Check that we can set the state dimension.
        model.state_dimension = 10
        model.state_dimension = 11
        model._clear()
        assert model.state_dimension is None
        assert model.input_dimension == 0

        # Try setting m with no inputs.
        with pytest.raises(AttributeError) as ex:
            model.input_dimension = 1
        assert ex.value.args[0] == "can't set attribute (no input operators)"

        # Try setting r when there is an operator with a different shape.
        model = self.Dummy(_get_operators("A", r))
        with pytest.raises(AttributeError) as ex:
            model.state_dimension = r + 1
        assert ex.value.args[0] == (
            f"can't set attribute (existing operators have r = {r})"
        )
        model._clear()
        assert model.state_dimension == r
        assert model.input_dimension == 0

        # Case 2: has inputs.
        model = self.Dummy(
            [
                opinf.operators.LinearOperator(),
                opinf.operators.InputOperator(),
            ]
        )
        assert model.input_dimension is None
        model.state_dimension = r

        # Check that we can set the input dimension.
        model.input_dimension = m
        model._clear()
        assert model.input_dimension is None

        # Try setting m when there is an input operator with a different m.
        model = self.Dummy(_get_operators("AB", r, m))
        assert model.state_dimension == r
        assert model.input_dimension == m
        with pytest.raises(AttributeError) as ex:
            model.input_dimension = m + 1
        assert ex.value.args[0] == (
            f"can't set attribute (existing input operators have m = {m})"
        )
        model._clear()
        assert model.state_dimension == r
        assert model.input_dimension == m

    # Dimensionality reduction ------------------------------------------------
    def test_galerkin(self, n=20, r=6):
        """Test _Model.galerkin()."""
        A = _get_operators("A", n)[0]
        Vr = np.random.random((20, 6))

        fom = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                A,
                opinf.operators.QuadraticOperator(),
            ]
        )
        assert fom.state_dimension == n

        rom = fom.galerkin(Vr)
        assert rom.state_dimension == r
        newA = rom.operators[1]
        assert isinstance(newA, opinf.operators.LinearOperator)
        assert newA.entries.shape == (r, r)
        assert np.allclose(newA.entries, Vr.T @ A.entries @ Vr)
        assert isinstance(rom.operators[0], opinf.operators.ConstantOperator)
        assert isinstance(rom.operators[2], opinf.operators.QuadraticOperator)
        for i in [0, 2]:
            assert rom.operators[i].entries is None

    # Validation methods ------------------------------------------------------
    def test_check_solver(self):
        """Test _Model._check_solver()."""

        with pytest.raises(ValueError) as ex:
            self.Dummy._check_solver(-1)
        assert ex.value.args[0] == "if a scalar, `solver` must be nonnegative"

        with pytest.raises(TypeError) as ex:
            self.Dummy._check_solver(list)
        assert ex.value.args[0] == "solver must be an instance, not a class"

        with pytest.raises(TypeError) as ex:
            self.Dummy._check_solver([])
        assert ex.value.args[0] == "solver must have a 'fit()' method"

        self.Dummy._check_solver(None)
        self.Dummy._check_solver(0)
        self.Dummy._check_solver(1)

    def test_check_inputargs(self):
        """Test _Model._check_inputargs()."""

        # Try with input operator but without inputs.
        model = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.InputOperator(),
            ]
        )
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(None, "U")
        assert ex.value.args[0] == "argument 'U' required"

        # Try without input operator but with inputs.
        model = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.LinearOperator(),
            ]
        )
        with pytest.warns(UserWarning) as wn:
            model._check_inputargs(1, "u")
        assert len(wn) == 1
        assert (
            wn[0].message.args[0] == "argument 'u' should be None, "
            "argument will be ignored"
        )

    def test_is_trained(self, m=4, r=7):
        """Test _Model._check_is_trained()."""
        model = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.InputOperator(),
            ]
        )
        with pytest.raises(AttributeError) as ex:
            model._check_is_trained()
        assert ex.value.args[0] == "no state_dimension (call fit())"

        model.state_dimension = r
        with pytest.raises(AttributeError) as ex:
            model._check_is_trained()
        assert ex.value.args[0] == "no input_dimension (call fit())"

        # Try without dimensions / operators set.
        model.input_dimension = m
        with pytest.raises(AttributeError) as ex:
            model._check_is_trained()
        assert ex.value.args[0] == "model not trained (call fit())"

        # Successful check.
        model.operators = _get_operators("cABH", r, m)
        model._check_is_trained()

    def test_eq(self):
        """Test _Model.__eq__()."""
        model1 = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.InputOperator(),
            ]
        )
        assert model1 != 10

        model2 = self.Dummy([opinf.operators.ConstantOperator()])
        assert model1 != model2

        model2 = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.QuadraticOperator(),
            ]
        )
        assert model1 != model2

        model2 = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                opinf.operators.InputOperator(),
            ]
        )
        assert model1 == model2
        model1.state_dimension = 5
        assert model1 != model2
        model2.state_dimension = model1.state_dimension + 2
        assert model1 != model2
        model2.state_dimension = model1.state_dimension
        assert model1 == model2

        model1.input_dimension = 4
        assert model1 != model2
        model2.input_dimension = model1.input_dimension + 2
        assert model1 != model2
        model2.input_dimension = model1.input_dimension
        assert model1 == model2

    # Model persistence -------------------------------------------------------
    def test_copy(self, r=4, m=3):
        """Test _Model.copy()."""
        A, H = _get_operators("AH", r, m)
        model = self.Dummy(
            [
                opinf.operators.ConstantOperator(),
                A,
                H,
            ]
        )
        model2 = model.copy()
        assert model2 is not model
        assert isinstance(model2, model.__class__)
        assert len(model2.operators) == len(model.operators)
        for op2, op1 in zip(model2.operators, model.operators):
            assert op2 is not op1
            assert op2 == op1
