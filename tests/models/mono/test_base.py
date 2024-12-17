# models/mono/test_base.py
"""Tests for models.mono._base."""

import abc
import pytest
import numpy as np
import scipy.linalg as la

import opinf
from opinf.operators import _utils as oputils


# Test classes ================================================================
class _TestModel(abc.ABC):
    """Tests for classes that inherit from models._base._Model."""

    # Setup -------------------------------------------------------------------
    Model = NotImplemented  # Class being tested.

    @abc.abstractmethod
    def get_operators(self, r=None, m=None):
        """Return a valid collection of operators to test.

        Parameters
        ----------
        r : None or int > 0
            State dimension.
            If ``None`` (default), operator entries should not be populated.
            If a positive integer, operator entries should be populated.
        m : int or None
            Input dimension. Only required if ``r`` is a postive integer.
            If ``m=0``, do not include operators that act on inputs.

        Returns
        -------
        ops : list
            List of instantiated operators (not abbreviations).
        """
        raise NotImplementedError

    def get_opinf_operator(self, r=None, m=None):
        """Get a single OpInf operator. If ``m`` is given, return"""
        ops = self.get_operators(r=r, m=m)
        hasinputs = (m is not None) and (m != 0)
        for op in ops:
            if hasinputs and not oputils.has_inputs(op):
                continue
            if oputils.is_opinf(op):
                return op
        raise opinf.errors.VerificationError(  # pragma: no cover
            "no input-dependent OpInf operators detected"
            if hasinputs
            else "no OpInf operators detected"
        )

    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def test_isvalidoperator(self):
        """Test _isvalidoperator()."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_check_operators_types_unique(self):
        """Test _check_operators_types_unique()."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_get_operator_of_type(self):
        """Test _get_operator_of_type()."""
        raise NotImplementedError

    # Properties --------------------------------------------------------------
    def test_operators(self, r=4, m=2):
        """Test the following methods.

        * __init__()
        * operators.setter
        * _check_state_dimension_consistency()
        * _check_state_dimension_consistency()
        * _clear()
        * __iter__()

        """
        # Try to instantiate without any operators.
        with pytest.raises(ValueError) as ex:
            self.Model([])
        assert ex.value.args[0] == "at least one operator required"

        # Try to instantiate with non-operator.
        with pytest.raises(TypeError) as ex:
            self.Model([1])
        assert ex.value.args[0] == "invalid operator of type 'int'"

        # Try to instantiate with operators of mismatched shape.
        ops = [
            self.get_opinf_operator(r=r, m=0),
            self.get_opinf_operator(r=r + 1, m=0),
        ]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Model(ops)
        assert ex.value.args[0] == (
            "operators not aligned "
            "(state dimension must be the same for all operators)"
        )

        # Try to instantiate with input operators not aligned.
        ops = [
            self.get_opinf_operator(r=r, m=m),
            self.get_opinf_operator(r=r, m=m + 1),
        ]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Model(ops)
        assert ex.value.args[0] == (
            "input operators not aligned "
            "(input dimension must be the same for all input operators)"
        )

        # Try with bad operator abbrevation.
        with pytest.raises(TypeError) as ex:
            self.Model("®")
        assert ex.value.args[0] == "operator abbreviation '®' not recognized"

        with pytest.raises(TypeError) as ex:
            self.Model([ops[0], "ç"])
        assert ex.value.args[0] == "operator abbreviation 'ç' not recognized"

        # Test operators.setter() - - - - - - - - - - - - - - - - - - - - - - -

        # Single operator, no entries, no inputs.
        op = self.get_opinf_operator(r=None, m=0)
        model = self.Model(op)
        assert len(model.operators) == 1
        assert model.operators[0] is op
        assert model._indices_of_operators_to_infer == [0]
        assert model._indices_of_known_operators == []
        assert model._has_inputs is False
        assert model.input_dimension == 0
        assert model.state_dimension is None

        # Several operators, no entries, with inputs.
        ops = self.get_operators(r=None, m=None)
        model = self.Model(ops)
        assert len(model.operators) == len(ops)
        for i, op in enumerate(model.operators):
            assert op is ops[i]
            if oputils.is_uncalibrated(op):
                assert i in model._indices_of_operators_to_infer
            else:
                assert i in model._indices_of_known_operators
        assert model._has_inputs is True
        assert model.input_dimension is None
        assert model.state_dimension is None

        # Several operators with entries, with inputs.
        ops = self.get_operators(r=r, m=m)
        model = self.Model(ops)
        assert len(model.operators) == len(ops)
        for i, op in enumerate(model.operators):
            assert op is ops[i]
            if oputils.is_uncalibrated(op):
                assert i in model._indices_of_operators_to_infer
            else:
                assert i in model._indices_of_known_operators
        assert model._has_inputs is True
        assert model.state_dimension == r
        assert model.input_dimension == m

        # Some operators with entries, some without.
        ops1 = self.get_operators(r=r, m=m)
        ops2 = self.get_operators(r=r, m=m)
        ops = ops1[:2] + ops2[2:]
        model.operators = ops
        assert len(model.operators) == len(ops)
        for i, op in enumerate(model.operators):
            assert op is ops[i]
            if oputils.is_uncalibrated(op):
                assert i in model._indices_of_operators_to_infer
            else:
                assert i in model._indices_of_known_operators
        assert model._has_inputs is True
        assert model.state_dimension == r
        assert model.input_dimension == m

        # Test __iter__().
        iterated = [op for op in model]
        assert len(iterated) == len(ops)
        assert iterated == ops

        # Test _clear().
        ops1 = [
            op for op in self.get_operators(r=r, m=m) if oputils.is_opinf(op)
        ]
        ops2 = [op for op in self.get_operators() if oputils.is_opinf(op)]
        ops = [op for pair in zip(ops1[::2], ops2[1::2]) for op in pair]
        model = self.Model(ops)
        for i in range(1, len(ops), 2):
            model.operators[i].set_entries(ops1[i].entries)
        model._clear()
        for i in range(1, len(ops), 2):
            assert model.operators[i].entries is None

    def test_dimensions(self, r=7, m=3):
        """Test state_dimension and input_dimension properties, including
        setters and behavior with _clear().
        """
        # Case 1: no inputs.
        model = self.Model(self.get_operators(r=None, m=0))
        assert model.state_dimension is None
        assert model.input_dimension == 0

        # Check that we can set and reset the state dimension.
        model.state_dimension = 10
        model.state_dimension = 11
        model._clear()
        assert model.state_dimension is None
        assert model.input_dimension == 0

        # Try setting the input dimension when there are no inputs.
        with pytest.raises(AttributeError) as ex:
            model.input_dimension = 1
        assert ex.value.args[0] == "can't set attribute (no input operators)"

        # Try setting r when there is an operator with a different shape.
        model = self.Model(self.get_operators(r=r, m=0))
        with pytest.raises(AttributeError) as ex:
            model.state_dimension = r + 1
        assert ex.value.args[0] == (
            f"can't set attribute (existing operators have r = {r})"
        )
        model._clear()
        assert model.state_dimension == r
        assert model.input_dimension == 0

        # Case 2: has inputs.
        model = self.Model(self.get_operators())
        assert model.input_dimension is None
        model.state_dimension = r

        # Check that we can set and reset the input dimension.
        model.input_dimension = m
        model._clear()
        assert model.input_dimension is None

        # Try setting m when there is an input operator with a different m.
        model = self.Model(self.get_operators(r=r, m=m))
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

    def test_solver(self, r=5, m=2):
        """Test the solver property, including its setter."""
        ops = self.get_operators(r=None, m=None)
        model = self.Model(ops)
        assert isinstance(model.solver, opinf.lstsq.PlainSolver)

        with pytest.raises(ValueError) as ex:
            model.solver = -1
        assert ex.value.args[0] == "if a scalar, solver must be nonnegative"

        with pytest.raises(TypeError) as ex:
            model.solver = list
        assert ex.value.args[0] == "solver must be an instance, not a class"

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            model.solver = []
        assert len(wn) == 2
        assert wn[0].message.args[0] == "solver should have a 'fit()' method"
        assert wn[1].message.args[0] == "solver should have a 'solve()' method"

        model.solver = 0
        assert isinstance(model.solver, opinf.lstsq.PlainSolver)
        model.solver = 1
        assert isinstance(model.solver, opinf.lstsq.L2Solver)

        ops = self.get_operators(r=r, m=m)
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            model = self.Model(ops, solver=2)
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "all operators initialized explicity, setting solver=None"
        )
        assert model.solver is None

    # Dimensionality reduction ------------------------------------------------
    def test_galerkin(self, n=20, r=6, m=2):
        """Lightly test galerkin()."""
        ops1 = [
            op for op in self.get_operators(r=n, m=m) if oputils.is_opinf(op)
        ]
        ops2 = [op for op in self.get_operators() if oputils.is_opinf(op)]
        Vr = la.qr(np.random.random((n, r)), mode="economic")[0]

        fom = self.Model(ops1[::2] + ops2[1::2])
        assert fom.state_dimension == n

        rom = fom.galerkin(Vr)
        assert rom.state_dimension == r
        assert len(rom.operators) == len(fom.operators)
        for i in range(idx := len(ops1[::2])):
            assert rom.operators[i].entries is not None
        for i in range(idx, len(ops1)):
            assert rom.operators[i].entries is None

    # Validation methods ------------------------------------------------------
    def test_check_inputargs(self, r=5, m=2):
        """Test _check_inputargs()."""

        # Try with input operator but without inputs.
        model = self.Model(self.get_operators(r=r, m=m))
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(None, "U")
        assert ex.value.args[0] == "argument 'U' required"

        # Try without input operator but with inputs.
        model = self.Model(self.get_operators(r=None, m=0))
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            model._check_inputargs(1, "u")
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "argument 'u' should be None, argument will be ignored"
        )

    def test_is_trained(self, r=7, m=4):
        """Test _Model._check_is_trained()."""
        model = self.Model(self.get_operators())
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
        model.operators = self.get_operators(r=r, m=m)
        model._check_is_trained()

    def test_eq(self):
        """Test _Model.__eq__()."""
        ops = self.get_operators(r=None, m=None)
        model1 = self.Model(ops)
        assert model1 != 10

        model2 = self.Model(ops[:-1])
        assert model1 != model2

        model2 = self.Model(ops)
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

        model1 = self.Model(ops)
        model2 = self.Model(ops[::-1])
        assert model1 == model2

        model1 = self.Model(ops[:-1])
        model2 = self.Model(ops[1:])
        assert model1 != model2

    # Model persistence -------------------------------------------------------
    def test_copy(self, r=4, m=3):
        """Test _Model.copy()."""
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


if __name__ == "__main__":
    pytest.main([__file__])
