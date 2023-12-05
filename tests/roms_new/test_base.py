# models/test_base.py
"""Tests for models._base."""

import pytest
import numpy as np

import opinf

from . import _get_operators


opinf_operators = opinf.operators_new  # TEMP


class TestMonolithicModel:
    """Test models._base._MonolithicModel."""

    class Dummy(opinf.models._base._MonolithicModel):
        """Instantiable version of _MonolithicModel."""

        _LHS_LABEL = "dq / dt"
        _STATE_LABEL = "q(t)"
        _INPUT_LABEL = "u(t)"

        def fit(*args, **kwargs):
            pass

        def predict(*args, **kwargs):
            return 100

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
        assert ex.value.args[0] == "expected list of nonparametric operators"

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

        # Test operators.setter().
        model = self.Dummy(opinf_operators.ConstantOperator())
        for attr in (
            "operators",
            "m",
            "r",
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

    def test_operator_shortcuts(self, m=4, r=7):
        """Test _MonolithicModel.[caHGBN]_ properties
        (_get_operator_of_type()).
        """
        [c, A, H, B, N] = _get_operators("cAHBN", r, m)
        model = self.Dummy([A, B, c, H, N])

        assert model.A_ is model.operators[0]
        assert model.B_ is model.operators[1]
        assert model.c_ is model.operators[2]
        assert model.H_ is model.operators[3]
        assert model.N_ is model.operators[4]
        assert model.G_ is None

    # Properties: dimensions --------------------------------------------------
    def test_dimension_properties(self, m=3, r=7):
        """Test the properties _MonolithicModel.state_dimension and
        _MonolithicModel.input_dimension.
        """
        # Case 1: no inputs.
        model = self.Dummy("cH")
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
        model = self.Dummy("AB")
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

    # String representation ---------------------------------------------------
    def test_str(self):
        """Test _MonolithicModel.__str__() (string representation)."""

        # Continuous Models
        model = self.Dummy("A")
        assert str(model) == "Model structure: dq / dt = Aq(t)"
        model = self.Dummy("cA")
        assert str(model) == "Model structure: dq / dt = c + Aq(t)"
        model = self.Dummy("HB")
        assert (
            str(model) == "Model structure: dq / dt = H[q(t) ⊗ q(t)] + Bu(t)"
        )
        model = self.Dummy("G")
        assert str(model) == "Model structure: dq / dt = G[q(t) ⊗ q(t) ⊗ q(t)]"
        model = self.Dummy("cH")
        assert str(model) == "Model structure: dq / dt = c + H[q(t) ⊗ q(t)]"

        # Dimension reporting.
        model = self.Dummy("A")
        model.state_dimension = 20
        modelstr = str(model).split("\n")
        assert len(modelstr) == 2
        assert modelstr[0] == "Model structure: dq / dt = Aq(t)"
        assert modelstr[1] == "State dimension r = 20"

        model = self.Dummy("cB")
        model.state_dimension = 10
        model.input_dimension = 3
        modelstr = str(model).split("\n")
        assert len(modelstr) == 3
        assert modelstr[0] == "Model structure: dq / dt = c + Bu(t)"
        assert modelstr[1] == "State dimension r = 10"
        assert modelstr[2] == "Input dimension m = 3"

    def test_repr(self):
        """Test _MonolithicModel.__repr__() (string representation)."""

        def firstline(obj):
            return repr(obj).split("\n")[0]

        assert firstline(self.Dummy("A")).startswith("<Dummy object at")

    # Validation methods ------------------------------------------------------
    def test_check_inputargs(self):
        """Test _MonolithicModel._check_inputargs()."""

        # Try with input operator but without inputs.
        model = self.Dummy("cB")
        with pytest.raises(ValueError) as ex:
            model._check_inputargs(None, "U")
        assert ex.value.args[0] == "argument 'U' required"

        # Try without input operator but with inputs.
        model = self.Dummy("cA")
        with pytest.warns(UserWarning) as wn:
            model._check_inputargs(1, "u")
        assert len(wn) == 1
        assert (
            wn[0].message.args[0] == "argument 'u' should be None, "
            "argument will be ignored"
        )

    def test_is_trained(self, m=4, r=7):
        """Test _MonolithicModel._check_is_trained()."""
        model = self.Dummy("cB")
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

    # Dimensionality reduction ------------------------------------------------
    def test_galerkin(self, n=20, r=6):
        """Test _MonolithicModel.galerkin()."""
        A = _get_operators("A", n)[0]
        Vr = np.random.random((20, 6))

        fom = self.Dummy(["c", A, "H"])
        assert fom.r == n

        model = fom.galerkin(Vr)
        assert model.state_dimension == r
        newA = model.operators[1]
        assert isinstance(newA, opinf_operators.LinearOperator)
        assert newA.entries.shape == (r, r)
        assert np.allclose(newA.entries, Vr.T @ A.entries @ Vr)
        assert isinstance(model.operators[0], opinf_operators.ConstantOperator)
        assert isinstance(
            model.operators[2], opinf_operators.QuadraticOperator
        )
        for i in [0, 2]:
            assert model.operators[i].entries is None

    # Model evaluation --------------------------------------------------------
    def test_evaluate(self, m=2, k=10, r=5, ntrials=10):
        """Test _MonolithicModel.evaluate()."""
        c_, A_, H_, B_ = _get_operators("cAHB", r, m)

        model = self.Dummy([c_, A_])
        for _ in range(ntrials):
            q_ = np.random.random(r)
            y_ = c_.entries + A_.entries @ q_
            out = model.evaluate(q_)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random((r, k))
            Y_ = c_.entries.reshape((r, 1)) + A_.entries @ Q_
            out = model.evaluate(Q_)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

        kron2c = opinf.utils.kron2c
        model = self.Dummy([H_, B_])
        for _ in range(ntrials):
            u = np.random.random(m)
            q_ = np.random.random(r)
            y_ = H_.entries @ kron2c(q_) + B_.entries @ u
            out = model.evaluate(q_, u)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random((r, k))
            U = np.random.random((m, k))
            Y_ = H_.entries @ kron2c(Q_) + B_.entries @ U
            out = model.evaluate(Q_, U)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

        # Special case: r = 1, q is a scalar.
        model = self.Dummy(_get_operators("A", 1))
        a = model.operators[0].entries[0]
        assert model.state_dimension == 1
        for _ in range(ntrials):
            q_ = np.random.random()
            y_ = a * q_
            out = model.evaluate(q_)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random(k)
            Y_ = a[0] * Q_
            out = model.evaluate(Q_, U)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

    def test_jacobian(self, r=5, m=2, ntrials=10):
        """Test _MonolithicModel.jacobian()."""
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

    def test_save(self):
        """Test _MonolithicModel.save()."""
        model = self.Dummy("cA")
        with pytest.raises(NotImplementedError) as ex:
            model.save("nothing")
        assert ex.value.args[0] == "use pickle/joblib"

    def test_load(self):
        """Test _MonolithicModel.load()."""
        with pytest.raises(NotImplementedError) as ex:
            self.Dummy.load("nothing")
        assert ex.value.args[0] == "use pickle/joblib"
