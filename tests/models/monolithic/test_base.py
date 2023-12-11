# models/monolithic/test_base.py
"""Tests for models._base."""

import pytest
import numpy as np

import opinf

from . import _get_operators


opinf_operators = opinf.operators_new  # TEMP
_module = opinf.models.monolithic._base


class TestMonolithicModel:
    """Test models._base._MonolithicModel."""

    class Dummy(_module._MonolithicModel):
        """Instantiable version of _MonolithicModel."""

        def _isvalidoperator(self, op):
            return hasattr(opinf_operators, op.__class__.__name__)

        def _check_operator_types_unique(*args, **kwargs):
            return True

        def _get_operator_of_type(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    # Properties: operators ---------------------------------------------------
    def test_operators(self, m=4, r=7):
        """Test _MonolithicModel.__init__(), operators, _clear(),
        and __iter__().
        """
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
            opinf_operators.LinearOperator(np.random.random((r, r))),
            opinf_operators.ConstantOperator(np.random.random(r + 1)),
        ]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy(bad_ops)
        assert (
            ex.value.args[0] == "operators not aligned "
            "(state dimension must be the same for all operators)"
        )

        # Try to instantiate with input operators not aligned.
        bad_ops = [
            opinf_operators.InputOperator(np.random.random((r, m - 1))),
            opinf_operators.StateInputOperator(np.random.random((r, r * m))),
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
        model = self.Dummy(opinf_operators.ConstantOperator())
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
        assert isinstance(model.operators[0], opinf_operators.ConstantOperator)
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
        """Test the properties _MonolithicModel.state_dimension and
        _MonolithicModel.input_dimension.
        """
        # Case 1: no inputs.
        model = self.Dummy(
            [
                opinf_operators.ConstantOperator(),
                opinf_operators.QuadraticOperator(),
            ]
        )
        assert model.state_dimension is None
        assert model.input_dimension == 0

        # Check that we can set the reduced dimension.
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
                opinf_operators.LinearOperator(),
                opinf_operators.InputOperator(),
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
        """Test _MonolithicModel.galerkin()."""
        A = _get_operators("A", n)[0]
        Vr = np.random.random((20, 6))

        fom = self.Dummy(
            [
                opinf_operators.ConstantOperator(),
                A,
                opinf_operators.QuadraticOperator(),
            ]
        )
        assert fom.state_dimension == n

        rom = fom.galerkin(Vr)
        assert rom.state_dimension == r
        newA = rom.operators[1]
        assert isinstance(newA, opinf_operators.LinearOperator)
        assert newA.entries.shape == (r, r)
        assert np.allclose(newA.entries, Vr.T @ A.entries @ Vr)
        assert isinstance(rom.operators[0], opinf_operators.ConstantOperator)
        assert isinstance(rom.operators[2], opinf_operators.QuadraticOperator)
        for i in [0, 2]:
            assert rom.operators[i].entries is None

    # Validation methods ------------------------------------------------------
    def test_check_inputargs(self):
        """Test _MonolithicModel._check_inputargs()."""

        # Try with input operator but without inputs.
        model = self.Dummy(
            [
                opinf_operators.ConstantOperator(),
                opinf_operators.InputOperator(),
            ]
        )
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(None, "U")
        assert ex.value.args[0] == "argument 'U' required"

        # Try without input operator but with inputs.
        model = self.Dummy(
            [
                opinf_operators.ConstantOperator(),
                opinf_operators.LinearOperator(),
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
        """Test _MonolithicModel._check_is_trained()."""
        model = self.Dummy(
            [
                opinf_operators.ConstantOperator(),
                opinf_operators.InputOperator(),
            ]
        )
        with pytest.raises(AttributeError) as ex:
            model._check_is_trained()
        assert ex.value.args[0] == "no reduced dimension 'r' (call fit())"

        model.state_dimension = r
        with pytest.raises(AttributeError) as ex:
            model._check_is_trained()
        assert ex.value.args[0] == "no input dimension 'm' (call fit())"

        # Try without dimensions / operators set.
        model.input_dimension = m
        with pytest.raises(AttributeError) as ex:
            model._check_is_trained()
        assert ex.value.args[0] == "model not trained (call fit())"

        # Successful check.
        model.operators = _get_operators("cABH", r, m)
        model._check_is_trained()
