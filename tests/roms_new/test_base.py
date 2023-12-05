# roms/test_base.py
"""Tests for roms._base."""

import pytest
import numpy as np

import opinf

from . import _get_operators


opinf_operators = opinf.operators_new  # TEMP


class TestMonolithicROM:
    """Test roms._base._MonolithicROM."""

    class Dummy(opinf.roms_new._base._MonolithicROM):
        """Instantiable version of _MonolithicROM."""

        _LHS_LABEL = "dq / dt"
        _STATE_LABEL = "q(t)"
        _INPUT_LABEL = "u(t)"

        def fit(*args, **kwargs):
            pass

        def predict(*args, **kwargs):
            return 100

    # Properties: operators ---------------------------------------------------
    def test_operators(self, m=4, r=7):
        """Test _MonolithicROM.__init__(), operators, _clear(),
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
        rom = self.Dummy(opinf_operators.ConstantOperator())
        for attr in (
            "operators",
            "m",
            "r",
            "_has_inputs",
            "_indices_of_operators_to_infer",
            "_indices_of_known_operators",
        ):
            assert hasattr(rom, attr)
        # Operators with no entries, no inputs.
        assert len(rom.operators) == 1
        assert isinstance(rom.operators[0], opinf_operators.ConstantOperator)
        assert len(rom._indices_of_operators_to_infer) == 1
        assert rom._indices_of_operators_to_infer == [0]
        assert len(rom._indices_of_known_operators) == 0
        assert rom._has_inputs is False
        assert rom.m == 0
        assert rom.r is None

        # Operators with entries, no inputs.
        rom = self.Dummy(operators[:2])
        assert len(rom._indices_of_operators_to_infer) == 0
        assert len(rom._indices_of_known_operators) == 2
        assert rom._indices_of_known_operators == [0, 1]
        assert rom._has_inputs is False
        assert rom.m == 0
        assert rom.r == r
        for i in range(2):
            assert rom.operators[i] is operators[i]
            assert rom.operators[i].entries is not None

        # Some operators with entries, some without; with inputs.
        operators[5]._clear()
        rom = self.Dummy(operators[3:6])
        assert len(rom._indices_of_operators_to_infer) == 1
        assert rom._indices_of_operators_to_infer == [2]
        assert len(rom._indices_of_known_operators) == 2
        assert rom._indices_of_known_operators == [0, 1]
        assert rom._has_inputs is True
        assert rom.m == m
        assert rom.r == r
        for i in rom._indices_of_operators_to_infer:
            assert rom.operators[i] is operators[i + 3]
            assert rom.operators[i].entries is None
        for i in rom._indices_of_known_operators:
            assert rom.operators[i] is operators[i + 3]
            assert rom.operators[i].entries is not None

        # Test __iter__().
        iterated = [op for op in rom]
        assert len(iterated) == 3
        assert iterated == operators[3:6]

        # Test _clear().
        rom.operators[2].set_entries(np.random.random((r, r * m)))
        rom._clear()
        assert rom.operators[2].entries is None
        assert len(rom._indices_of_operators_to_infer) == 1
        assert rom._indices_of_operators_to_infer == [2]
        assert len(rom._indices_of_known_operators) == 2
        assert rom._indices_of_known_operators == [0, 1]
        for i in rom._indices_of_operators_to_infer:
            assert rom.operators[i].entries is None
        for i in rom._indices_of_known_operators:
            assert rom.operators[i].entries is not None
        assert rom.r == r  # Dimensions not erased.
        assert rom.m == m  # Dimensions not erased.

        # Test __init__() shortcuts.
        rom = self.Dummy("cHB")
        assert len(rom.operators) == 3
        for i in range(3):
            assert rom.operators[i].entries is None
        assert isinstance(rom.operators[0], opinf_operators.ConstantOperator)
        assert isinstance(rom.operators[1], opinf_operators.QuadraticOperator)
        assert isinstance(rom.operators[2], opinf_operators.InputOperator)

        rom.operators = [opinf_operators.ConstantOperator(), "A", "N"]
        assert len(rom.operators) == 3
        for i in range(3):
            assert rom.operators[i].entries is None
        assert isinstance(rom.operators[0], opinf_operators.ConstantOperator)
        assert isinstance(rom.operators[1], opinf_operators.LinearOperator)
        assert isinstance(rom.operators[2], opinf_operators.StateInputOperator)

    def test_operator_shortcuts(self, m=4, r=7):
        """Test _MonolithicROM.[caHGBN]_ properties
        (_get_operator_of_type()).
        """
        [c, A, H, B, N] = _get_operators("cAHBN", r, m)
        rom = self.Dummy([A, B, c, H, N])

        assert rom.A_ is rom.operators[0]
        assert rom.B_ is rom.operators[1]
        assert rom.c_ is rom.operators[2]
        assert rom.H_ is rom.operators[3]
        assert rom.N_ is rom.operators[4]
        assert rom.G_ is None

    # Properties: dimensions --------------------------------------------------
    def test_dimension_properties(self, m=3, r=7):
        """Test the dimension properties _MonolithicROM.(r|m)."""
        # Case 1: no inputs.
        rom = self.Dummy("cH")
        assert rom.r is None
        assert rom.m == 0

        # Check that we can set the reduced dimension.
        rom.r = 10
        rom.r = 11
        rom._clear()
        assert rom.r is None
        assert rom.m == 0

        # Try setting m with no inputs.
        with pytest.raises(AttributeError) as ex:
            rom.m = 1
        assert ex.value.args[0] == "can't set attribute (no input operators)"

        # Try setting r when there is an operator with a different shape.
        rom = self.Dummy(_get_operators("A", r))
        with pytest.raises(AttributeError) as ex:
            rom.r = r + 1
        assert ex.value.args[0] == (
            f"can't set attribute (existing operators have r = {r})"
        )
        rom._clear()
        assert rom.r == r
        assert rom.m == 0

        # Case 2: has inputs.
        rom = self.Dummy("AB")
        assert rom.m is None
        rom.r = r

        # Check that we can set the input dimension.
        rom.m = m
        rom._clear()
        assert rom.m is None

        # Try setting m when there is an input operator with a different m.
        rom = self.Dummy(_get_operators("AB", r, m))
        assert rom.r == r
        assert rom.m == m
        with pytest.raises(AttributeError) as ex:
            rom.m = m + 1
        assert ex.value.args[0] == (
            f"can't set attribute (existing input operators have m = {m})"
        )
        rom._clear()
        assert rom.r == r
        assert rom.m == m

    # String representation ---------------------------------------------------
    def test_str(self):
        """Test _MonolithicROM.__str__() (string representation)."""

        # Continuous ROMs
        rom = self.Dummy("A")
        assert str(rom) == "Model structure: dq / dt = Aq(t)"
        rom = self.Dummy("cA")
        assert str(rom) == "Model structure: dq / dt = c + Aq(t)"
        rom = self.Dummy("HB")
        assert str(rom) == "Model structure: dq / dt = H[q(t) ⊗ q(t)] + Bu(t)"
        rom = self.Dummy("G")
        assert str(rom) == "Model structure: dq / dt = G[q(t) ⊗ q(t) ⊗ q(t)]"
        rom = self.Dummy("cH")
        assert str(rom) == "Model structure: dq / dt = c + H[q(t) ⊗ q(t)]"

        # Dimension reporting.
        rom = self.Dummy("A")
        rom.r = 20
        romstr = str(rom).split("\n")
        assert len(romstr) == 2
        assert romstr[0] == "Model structure: dq / dt = Aq(t)"
        assert romstr[1] == "State dimension r = 20"

        rom = self.Dummy("cB")
        rom.r = 10
        rom.m = 3
        romstr = str(rom).split("\n")
        assert len(romstr) == 3
        assert romstr[0] == "Model structure: dq / dt = c + Bu(t)"
        assert romstr[1] == "State dimension r = 10"
        assert romstr[2] == "Input dimension m = 3"

    def test_repr(self):
        """Test _MonolithicROM.__repr__() (string representation)."""

        def firstline(obj):
            return repr(obj).split("\n")[0]

        assert firstline(self.Dummy("A")).startswith("<Dummy object at")

    # Validation methods ------------------------------------------------------
    def test_check_inputargs(self):
        """Test _MonolithicROM._check_inputargs()."""

        # Try with input operator but without inputs.
        rom = self.Dummy("cB")
        with pytest.raises(ValueError) as ex:
            rom._check_inputargs(None, "U")
        assert ex.value.args[0] == "argument 'U' required"

        # Try without input operator but with inputs.
        rom = self.Dummy("cA")
        with pytest.warns(UserWarning) as wn:
            rom._check_inputargs(1, "u")
        assert len(wn) == 1
        assert (
            wn[0].message.args[0] == "argument 'u' should be None, "
            "argument will be ignored"
        )

    def test_is_trained(self, m=4, r=7):
        """Test _MonolithicROM._check_is_trained()."""
        rom = self.Dummy("cB")
        with pytest.raises(AttributeError) as ex:
            rom._check_is_trained()
        assert ex.value.args[0] == "no reduced dimension 'r' (call fit())"

        rom.r = r
        with pytest.raises(AttributeError) as ex:
            rom._check_is_trained()
        assert ex.value.args[0] == "no input dimension 'm' (call fit())"

        # Try without dimensions / operators set.
        rom.m = m
        with pytest.raises(AttributeError) as ex:
            rom._check_is_trained()
        assert ex.value.args[0] == "model not trained (call fit())"

        # Successful check.
        rom.operators = _get_operators("cABH", r, m)
        rom._check_is_trained()

    # Dimensionality reduction ------------------------------------------------
    def test_galerkin(self, n=20, r=6):
        """Test _MonolithicROM.galerkin()."""
        A = _get_operators("A", n)[0]
        Vr = np.random.random((20, 6))

        fom = self.Dummy(["c", A, "H"])
        assert fom.r == n

        rom = fom.galerkin(Vr)
        assert rom.r == r
        newA = rom.operators[1]
        assert isinstance(newA, opinf_operators.LinearOperator)
        assert newA.entries.shape == (r, r)
        assert np.allclose(newA.entries, Vr.T @ A.entries @ Vr)
        assert isinstance(rom.operators[0], opinf_operators.ConstantOperator)
        assert isinstance(rom.operators[2], opinf_operators.QuadraticOperator)
        for i in [0, 2]:
            assert rom.operators[i].entries is None

    # ROM evaluation ----------------------------------------------------------
    def test_evaluate(self, m=2, k=10, r=5, ntrials=10):
        """Test _MonolithicROM.evaluate()."""
        c_, A_, H_, B_ = _get_operators("cAHB", r, m)

        rom = self.Dummy([c_, A_])
        for _ in range(ntrials):
            q_ = np.random.random(r)
            y_ = c_.entries + A_.entries @ q_
            out = rom.evaluate(q_)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random((r, k))
            Y_ = c_.entries.reshape((r, 1)) + A_.entries @ Q_
            out = rom.evaluate(Q_)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

        kron2c = opinf.utils.kron2c
        rom = self.Dummy([H_, B_])
        for _ in range(ntrials):
            u = np.random.random(m)
            q_ = np.random.random(r)
            y_ = H_.entries @ kron2c(q_) + B_.entries @ u
            out = rom.evaluate(q_, u)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random((r, k))
            U = np.random.random((m, k))
            Y_ = H_.entries @ kron2c(Q_) + B_.entries @ U
            out = rom.evaluate(Q_, U)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

        # Special case: r = 1, q is a scalar.
        rom = self.Dummy(_get_operators("A", 1))
        a = rom.operators[0].entries[0]
        assert rom.r == 1
        for _ in range(ntrials):
            q_ = np.random.random()
            y_ = a * q_
            out = rom.evaluate(q_)
            assert out.shape == y_.shape
            assert np.allclose(out, y_)

            Q_ = np.random.random(k)
            Y_ = a[0] * Q_
            out = rom.evaluate(Q_, U)
            assert out.shape == Y_.shape
            assert np.allclose(out, Y_)

    def test_jacobian(self, r=5, m=2, ntrials=10):
        """Test _MonolithicROM.jacobian()."""
        c_, A_, B_ = _get_operators("cAB", r, m)

        for oplist in ([c_, A_], [c_, A_, B_]):
            rom = self.Dummy(oplist)
            q_ = np.random.random(r)
            out = rom.jacobian(q_)
            assert out.shape == (r, r)
            assert np.allclose(out, A_.entries)

        # Special case: r = 1, q a scalar.
        rom = self.Dummy(_get_operators("A", 1))
        q_ = np.random.random()
        out = rom.jacobian(q_)
        assert out.shape == (1, 1)
        assert out[0, 0] == rom.operators[0].entries[0, 0]

    def test_save(self):
        """Test _MonolithicROM.save()."""
        rom = self.Dummy("cA")
        with pytest.raises(NotImplementedError) as ex:
            rom.save("nothing")
        assert ex.value.args[0] == "use pickle/joblib"

    def test_load(self):
        """Test _MonolithicROM.load()."""
        with pytest.raises(NotImplementedError) as ex:
            self.Dummy.load("nothing")
        assert ex.value.args[0] == "use pickle/joblib"
