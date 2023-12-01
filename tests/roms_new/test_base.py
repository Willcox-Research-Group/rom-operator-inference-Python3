# roms/test_base.py
"""Tests for roms._base."""

import pytest
import numpy as np

import opinf

from . import _get_data, _get_operators


opinf_operators = opinf.operators_new  # TEMP


class TestBaseROM:
    """Test roms._base._BaseMonolithicROM."""

    class Dummy(opinf.roms_new._base._BaseMonolithicROM):
        """Instantiable version of _BaseMonolithicROM."""

        _LHS_LABEL = "dq / dt"
        _STATE_LABEL = "q(t)"
        _INPUT_LABEL = "u(t)"

        def fit(*args, **kwargs):
            pass

        def predict(*args, **kwargs):
            return 100

    # Properties: basis -------------------------------------------------------
    def test_basis(self, r=3):
        """Test _BaseMonolithicROM.basis getter, setter, and deleter."""
        # Empty basis.
        rom = self.Dummy(None, [opinf_operators.ConstantOperator()])
        assert rom.basis is None
        assert rom.r is None
        assert rom.n is None
        assert rom.m == 0

        # Set basis with a NumPy array.
        n = 3 * r
        Vr = np.random.random((n, r))
        rom.basis = Vr
        assert rom.basis is not None
        assert isinstance(rom.basis, opinf.basis.LinearBasis)
        assert rom.basis.shape == Vr.shape
        assert np.all(rom.basis.entries == Vr)
        assert rom.n == n
        assert rom.r == r
        assert rom.m == 0

        # Try to set (n, r) basis with n < r.
        with pytest.raises(ValueError) as ex:
            rom.basis = Vr.T
        assert ex.value.args[0] == "basis must be n x r with n > r"

        # Set basis with basis object.
        basis = opinf.basis.LinearBasis().fit(Vr)
        rom.basis = basis
        assert rom.basis is basis

        # Test basis.deleter.
        del rom.basis
        assert rom.basis is None

    # Properties: operators ---------------------------------------------------
    def test_operators(self, m=4, r=7):
        """Test _BaseMonolithicROM.__init__(), operators, _clear(),
        and __iter__().
        """
        operators = _get_operators("cAHGBN", r, m)
        n = 3 * r
        Vr = np.random.random((n, r))

        # Try to instantiate without any operators.
        with pytest.raises(ValueError) as ex:
            self.Dummy(None, [])
        assert ex.value.args[0] == "at least one operator required"

        # Try to instantiate with non-operator.
        with pytest.raises(TypeError) as ex:
            self.Dummy(None, [1])
        assert ex.value.args[0] == "expected list of nonparametric operators"

        # Try to instantiate with operators of mismatched shape (no basis).
        bad_ops = [
            opinf_operators.LinearOperator(np.random.random((r, r))),
            opinf_operators.ConstantOperator(np.random.random(r + 1)),
        ]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy(None, bad_ops)
        assert (
            ex.value.args[0] == "operators not aligned "
            "(shape[0] must be the same)"
        )

        # Try to instantiate with operators not aligned with the basis.
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy(Vr, bad_ops)
        assert (
            ex.value.args[0] == "operators not aligned with basis "
            f"(operators[1].shape[0] = {r+1} must be r = {r} or n = {n})"
        )

        # Try to instantiate with input operators not aligned.
        bad_ops = [
            opinf_operators.InputOperator(np.random.random((r, m - 1))),
            opinf_operators.StateInputOperator(np.random.random((r, r * m))),
        ]
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            self.Dummy(None, bad_ops)
        assert (
            ex.value.args[0] == "input operators not aligned "
            "(input dimension 'm' must be the same)"
        )

        # Test operators.setter().
        rom = self.Dummy(None, [opinf_operators.ConstantOperator()])
        for attr in (
            "basis",
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
        rom = self.Dummy(None, operators[:2])
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
        rom = self.Dummy(None, operators[3:6])
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
        rom = self.Dummy(None, "cHB")
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
        """Test _BaseMonolithicROM.[caHGBN]_ properties
        (_get_operator_of_type()).
        """
        [c, A, H, B, N] = _get_operators("cAHBN", r, m)
        rom = self.Dummy(None, [A, B, c, H, N])

        assert rom.A_ is rom.operators[0]
        assert rom.B_ is rom.operators[1]
        assert rom.c_ is rom.operators[2]
        assert rom.H_ is rom.operators[3]
        assert rom.N_ is rom.operators[4]
        assert rom.G_ is None

    # Properties: dimensions --------------------------------------------------
    def test_dimension_properties(self, n=20, m=3, r=7):
        """Test the properties roms._base._BaseMonolithicROM.(n|r|basis)."""
        rom = self.Dummy(None, "cH")
        assert rom.n is None
        assert rom.m == 0
        assert rom.r is None
        assert rom.basis is None

        # Case 1: basis != None
        basis = np.random.random((n, r))
        rom.basis = basis
        assert rom.n == n
        assert rom.m == 0
        assert rom.r == r
        assert isinstance(rom.basis, opinf.basis.LinearBasis)

        # Try setting n.
        with pytest.raises(AttributeError) as ex:
            rom.n = n + 1
        assert ex.value.args[0] == "can't set attribute (n = basis.shape[0])"

        # Try setting m with no inputs.
        with pytest.raises(AttributeError) as ex:
            rom.m = 1
        assert ex.value.args[0] == "can't set attribute (no input operators)"

        # Try setting r with basis already set.
        with pytest.raises(AttributeError) as ex:
            rom.r = r + 1
        assert ex.value.args[0] == "can't set attribute (r = basis.shape[1])"

        # Case 2: basis = None
        del rom.basis
        assert rom.basis is None
        assert rom.n is None
        rom = self.Dummy(None, "AB")
        assert rom.m is None
        rom.r = r
        rom.m = m

    # String representation ---------------------------------------------------
    def test_str(self):
        """Test _BaseMonolithicROM.__str__() (string representation)."""

        # Continuous ROMs
        rom = self.Dummy(None, "A")
        assert str(rom) == "Model structure: dq / dt = Aq(t)"
        rom = self.Dummy(None, "cA")
        assert str(rom) == "Model structure: dq / dt = c + Aq(t)"
        rom = self.Dummy(None, "HB")
        assert str(rom) == "Model structure: dq / dt = H[q(t) ⊗ q(t)] + Bu(t)"
        rom = self.Dummy(None, "G")
        assert str(rom) == "Model structure: dq / dt = G[q(t) ⊗ q(t) ⊗ q(t)]"
        rom = self.Dummy(None, "cH")
        assert str(rom) == "Model structure: dq / dt = c + H[q(t) ⊗ q(t)]"

        # Dimension reporting.
        rom = self.Dummy(np.empty((100, 20)), "A")
        romstr = str(rom).split("\n")
        assert len(romstr) == 3
        assert romstr[0] == "Model structure: dq / dt = Aq(t)"
        assert romstr[1] == "Full-order dimension    n = 100"
        assert romstr[2] == "Reduced-order dimension r = 20"

        rom = self.Dummy(np.empty((80, 10)), "cB")
        rom.m = 3
        romstr = str(rom).split("\n")
        assert len(romstr) == 4
        assert romstr[0] == "Model structure: dq / dt = c + Bu(t)"
        assert romstr[1] == "Full-order dimension    n = 80"
        assert romstr[2] == "Input/control dimension m = 3"
        assert romstr[3] == "Reduced-order dimension r = 10"

    def test_repr(self):
        """Test _BaseMonolithicROM.__repr__() (string representation)."""

        def firstline(obj):
            return repr(obj).split("\n")[0]

        assert firstline(self.Dummy(None, "A")).startswith("<Dummy object at")

    # Validation methods ------------------------------------------------------
    def test_check_inputargs(self):
        """Test _BaseMonolithicROM._check_inputargs()."""

        # Try with input operator but without inputs.
        rom = self.Dummy(None, "cB")
        with pytest.raises(ValueError) as ex:
            rom._check_inputargs(None, "U")
        assert ex.value.args[0] == "argument 'U' required"

        # Try without input operator but with inputs.
        rom = self.Dummy(None, "cA")
        with pytest.warns(UserWarning) as wn:
            rom._check_inputargs(1, "u")
        assert len(wn) == 1
        assert (
            wn[0].message.args[0] == "argument 'u' should be None, "
            "argument will be ignored"
        )

    def test_is_trained(self, m=4, r=7):
        """Test _BaseMonolithicROM._check_is_trained()."""
        rom = self.Dummy(None, "cB")
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
    def test_compress(self, n=60, k=50, r=10):
        """Test _BaseMonolithicROM.compress()."""
        Q, Qdot, _ = _get_data(n, k, 2)
        rom = self.Dummy(None, "c")

        # Try to compress without reduced dimension r set.
        with pytest.raises(AttributeError) as ex:
            rom.compress(Q, "things")
        assert ex.value.args[0] == "reduced dimension 'r' not set"

        # Try to compress with r set but without a basis.
        rom.r = r
        with pytest.raises(AttributeError) as ex:
            rom.compress(Q, "arg")
        assert ex.value.args[0] == "basis not set"

        # Try to compress with basis set but with wrong shape.
        Vr = np.random.random((n, r))
        rom.basis = Vr
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            rom.compress(Q[:-1, :], "state")
        assert ex.value.args[0] == "state not aligned with basis"

        # Correct usage.
        for S in (Q, Qdot):
            S_ = rom.compress(S)
            assert S_.shape == (r, k)
            assert np.allclose(S_, Vr.T @ S)
            assert np.allclose(S_, rom.basis.compress(S))
            S_ = rom.compress(S[:r, :])
            assert S_.shape == (r, k)
            assert np.all(S_ == S[:r, :])

    def test_decompress(self, n=60, k=20, r=8):
        """Test _BaseMonolithicROM.decompress()."""
        Q_, Qdot_, _ = _get_data(r, k, 2)
        rom = self.Dummy(None, "c")

        # Try to decompress without basis.
        rom.r = r
        with pytest.raises(AttributeError) as ex:
            rom.decompress(Q_, "arg")
        assert ex.value.args[0] == "basis not set"

        # Try to compress with basis set but with wrong shape.
        Vr = np.random.random((n, r))
        rom.basis = Vr
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            rom.decompress(Q_[:-1, :], "state")
        assert ex.value.args[0] == "state not aligned with basis"

        # Correct usage.
        for S_ in (Q_, Qdot_):
            S = rom.decompress(S_)
            assert S.shape == (n, k)
            assert np.allclose(S, Vr @ S_)
            assert np.allclose(S, rom.basis.decompress(S_))

    def test_galerkin(self, n=20, r=6):
        """Test _BaseMonolithicROM.galerkin()."""
        A = _get_operators("A", n)[0]
        Vr = np.random.random((20, 6))

        # Galerkin projection without a basis.
        rom = self.Dummy(None, ["c", A, "H"])
        rom.r = n // 2
        with pytest.raises(RuntimeError) as ex:
            rom.galerkin()
        assert ex.value.args[0] == "basis required for Galerkin projection"

        # Galerkin projection with a basis.
        rom.basis = Vr
        rom.galerkin()
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
        """Test _BaseMonolithicROM.evaluate()."""
        c_, A_, H_, B_ = _get_operators("cAHB", r, m)

        rom = self.Dummy(None, [c_, A_])
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
        rom = self.Dummy(None, [H_, B_])
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
        rom = self.Dummy(None, _get_operators("A", 1))
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
        """Test _BaseMonolithicROM.jacobian()."""
        c_, A_, B_ = _get_operators("cAB", r, m)

        for oplist in ([c_, A_], [c_, A_, B_]):
            rom = self.Dummy(None, oplist)
            q_ = np.random.random(r)
            out = rom.jacobian(q_)
            assert out.shape == (r, r)
            assert np.allclose(out, A_.entries)

        # Special case: r = 1, q a scalar.
        rom = self.Dummy(None, _get_operators("A", 1))
        q_ = np.random.random()
        out = rom.jacobian(q_)
        assert out.shape == (1, 1)
        assert out[0, 0] == rom.operators[0].entries[0, 0]

    def test_save(self):
        """Test _BaseMonolithicROM.save()."""
        rom = self.Dummy(None, "cA")
        with pytest.raises(NotImplementedError) as ex:
            rom.save("nothing")
        assert ex.value.args[0] == "use pickle/joblib"

    def test_load(self):
        """Test _BaseMonolithicROM.load()."""
        with pytest.raises(NotImplementedError) as ex:
            self.Dummy.load("nothing")
        assert ex.value.args[0] == "use pickle/joblib"
