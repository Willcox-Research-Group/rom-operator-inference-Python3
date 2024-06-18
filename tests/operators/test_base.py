# operators/test_base.py
"""Tests for operators._base."""

import os
import h5py
import pytest
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import opinf


_module = opinf.operators._base


def test_has_inputs():
    """Test operators._base.has_inputs()."""

    class Dummy(_module.InputMixin):
        def input_dimension(self):
            return -1

    op = Dummy()
    assert opinf.operators.has_inputs(op)
    assert not opinf.operators.has_inputs(5)


# Nonparametric operators =====================================================
class TestOperatorTemplate:
    """Test operators._base.OperatorTemplate."""

    Operator = _module.OperatorTemplate

    def test_str(self, r=11, m=3):
        """Test __str__()."""

        class Dummy(self.Operator):
            """Instantiable version of OperatorTemplate."""

            def __init__(self, state_dimension=r):
                self.__r = state_dimension

            @property
            def state_dimension(self):
                return self.__r

            def apply(self, state, input_=None):
                return state

        class InputDummy(Dummy, _module.InputMixin):
            """Instantiable version of OperatorTemplate with inputs."""

            def __init__(self, state_dimension=r, input_dimension=m):
                Dummy.__init__(self, state_dimension)
                self.__m = input_dimension

            @property
            def input_dimension(self):
                return self.__m

        def _test(DummyClass):
            dummystr = str(DummyClass())
            assert dummystr.startswith(DummyClass.__name__)
            for line in (lines := dummystr.split("\n")[1:]):
                assert line.startswith("  ")
            assert lines[0].endswith(f"{r}")
            return lines

        _test(Dummy)
        assert _test(InputDummy)[-1].endswith(f"{m}")

    def test_verify(self, r=10, m=4):
        """Test verify()."""

        class Dummy(self.Operator):
            """Instantiable version of OperatorTemplate."""

            def __init__(self, state_dimension=r):
                self.__r = state_dimension

            @property
            def state_dimension(self):
                return self.__r

            def apply(self, state, input_=None):
                return state

        class InputDummy(Dummy, _module.InputMixin):
            """Instantiable version of OperatorTemplate with inputs."""

            def __init__(self, state_dimension=r, input_dimension=m):
                Dummy.__init__(self, state_dimension)
                self.__m = input_dimension

            @property
            def input_dimension(self):
                return self.__m

        op = Dummy()
        op.verify()

        op = InputDummy()
        op.verify()

        def _single(DummyClass, message):
            dummy = DummyClass()
            with pytest.raises(opinf.errors.VerificationError) as ex:
                dummy.verify()
            assert ex.value.args[0] == message

        # Verification failures for apply().

        class Dummy1(Dummy):
            def __init__(self):
                super().__init__("one hundred")

        class Dummy1I(InputDummy):
            def __init__(self):
                super().__init__(10, -5)

        class Dummy2(Dummy):
            def apply(self, state, input_=None):
                return state[:-1]

        class Dummy2I(Dummy2, InputDummy):
            pass

        class Dummy3(Dummy):
            def apply(self, state, input_=None):
                if state.ndim == 1:
                    return state
                return state[:, :-1]

        class Dummy3I(Dummy3, InputDummy):
            pass

        _single(
            Dummy1,
            "state_dimension must be a positive integer "
            "(current value: 'one hundred', of type 'str')",
        )

        _single(
            Dummy1I,
            "input_dimension must be a positive integer "
            "(current value: -5, of type 'int')",
        )

        _single(
            Dummy2,
            "apply(q, u) must return array of shape (state_dimension,) when "
            "q.shape = (state_dimension,) and u = None",
        )

        _single(
            Dummy2I,
            "apply(q, u) must return array of shape (state_dimension,) when "
            "q.shape = (state_dimension,) and u.shape = (input_dimension,)",
        )

        _single(
            Dummy3,
            "apply(Q, U) must return array of shape (state_dimension, k) when "
            "Q.shape = (state_dimension, k) and U = None",
        )

        _single(
            Dummy3I,
            "apply(Q, U) must return array of shape (state_dimension, k) "
            "when Q.shape = (state_dimension, k) "
            "and U.shape = (input_dimension, k)",
        )

        # Verification failures for jacobian().

        class Dummy4(Dummy):
            def jacobian(self, state, input_=None):
                return state

        class Dummy4I(Dummy4, InputDummy):
            pass

        _single(
            Dummy4,
            "jacobian(q, u) must return array of shape "
            "(state_dimension, state_dimension) when "
            "q.shape = (state_dimension,) and u = None",
        )

        _single(
            Dummy4I,
            "jacobian(q, u) must return array of shape "
            "(state_dimension, state_dimension) when "
            "q.shape = (state_dimension,) and u.shape = (input_dimension,)",
        )

        # Correct usage of jacobian().

        class Dummy5(Dummy):
            def jacobian(self, state, input_=None):
                return np.eye(self.state_dimension)

        dummy = Dummy5()
        dummy.verify(plot=False)

        interactive = plt.isinteractive()
        plt.ion()
        dummy.verify(plot=True)
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close(fig)

        if not interactive:
            plt.ioff()

        # Verification failures for galerkin().

        class Dummy6(Dummy):
            def galerkin(self, Vr, Wr=None):
                return [self]

        class Dummy7(Dummy):
            def galerkin(self, Vr, Wr=None):
                r = Vr.shape[1]
                return self.__class__(r + 1)

        class Dummy8(InputDummy):
            def galerkin(self, Vr, Wr=None):
                return self.__class__(
                    Vr.shape[1],
                    self.input_dimension - 1,
                )

        class Dummy9(Dummy):
            def galerkin(self, Vr, Wr=None):
                return self.__class__(Vr.shape[1])

            def apply(self, state, input_=None):
                return np.random.random(state.shape)

        class Dummy10(Dummy):
            def __init__(self, sdim=r, Vr=None, Wr=None, A=None):
                if Vr is not None:
                    sdim = Vr.shape[1]
                super().__init__(sdim)
                self.Vr = Vr
                self.Wr = Wr
                if A is None:
                    A = np.random.random((sdim, sdim))
                self.A = A

            def apply(self, state, input_=None):
                if (Vr := self.Vr) is (Wr := self.Wr) is None:
                    return self.A @ state
                if Wr is not None:
                    return la.solve(Wr.T @ Vr, Wr.T @ self.A @ Vr @ state)
                return np.random.random(state.shape)

            def galerkin(self, Vr, Wr=None):
                return self.__class__(self.state_dimension, Vr, Wr, self.A)

        _single(
            Dummy6,
            "galerkin() must return object "
            "whose class inherits from OperatorTemplate",
        )

        _single(Dummy7, "galerkin(Vr, Wr).state_dimension != Vr.shape[1]")

        _single(
            Dummy8,
            "self.galerkin(Vr, Wr).input_dimension != self.input_dimension",
        )

        _single(
            Dummy9,
            "op2.apply(qr, u) "
            "!= inv(Wr.T @ Vr) @ Wr.T @ self.apply(Vr @ qr, u) "
            "where op2 = self.galerkin(Vr, Wr)",
        )

        _single(
            Dummy10,
            "op2.apply(qr, u) != Vr.T @ self.apply(Vr @ qr, u) "
            "where op2 = self.galerkin(Vr) and Vr.T @ Vr = I",
        )

        # Correct usage for galerkin().

        class Dummy11(Dummy):
            def galerkin(self, Vr, Wr=None):
                return self.__class__(Vr.shape[1])

        Dummy11(1).verify()
        Dummy11(r).verify()

        # Verification failures for copy().

        class Dummy12(Dummy):
            def copy(self):
                return self

        class Dummy13(Dummy):
            def copy(self):
                return [self]

        class Dummy14(Dummy):
            def copy(self):
                return self.__class__(self.state_dimension + 1)

        class Dummy15(InputDummy):
            def copy(self):
                return self.__class__(
                    self.state_dimension,
                    self.input_dimension - 1,
                )

        class Dummy16(Dummy):
            def apply(self, state, input_=None):
                return np.random.random(state.shape)

            def copy(self):
                return self.__class__(self.state_dimension)

        _single(Dummy12, "self.copy() is self")
        _single(Dummy13, "type(self.copy()) is not type(self)")
        _single(Dummy14, "self.copy().state_dimension != self.state_dimension")
        _single(Dummy15, "self.copy().input_dimension != self.input_dimension")

        _single(
            Dummy16,
            "self.copy().apply() not consistent with self.apply()",
        )

        class Dummy17(Dummy):
            def save(self, savefile, overwrite=False):
                return

            @classmethod
            def load(cls, loadfile):
                return [100]

        class Dummy18(Dummy):
            def save(self, savefile, overwrite=False):
                Dummy18.rr = self.state_dimension

            @classmethod
            def load(cls, loadfile):
                return cls(cls.rr + 1)

        class Dummy19(InputDummy):
            def save(self, savefile, overwrite=False):
                Dummy19.rr = self.state_dimension
                Dummy19.mm = self.input_dimension

            @classmethod
            def load(cls, loadfile):
                return cls(cls.rr, cls.mm - 1)

        class Dummy20(Dummy):
            def __init__(self, sdim=r, A=None):
                super().__init__(sdim)
                if A is None:
                    A = np.random.random((sdim, sdim))
                self.A = A

            def apply(self, state, input_=None):
                return self.A @ state

            def save(self, savefile, overwrite=False):
                Dummy20.rr = self.state_dimension
                Dummy20.AA = self.A

            @classmethod
            def load(cls, loadfile):
                return cls(cls.rr, cls.AA + 1)

        class Dummy21(Dummy):
            def save(self, savefile, overwrite=False):
                Dummy21.rr = self.state_dimension

            @classmethod
            def load(cls, loadfile):
                return cls(cls.rr)

        _single(Dummy17, "save()/load() does not preserve object type")
        _single(Dummy18, "save()/load() does not preserve state_dimension")
        _single(Dummy19, "save()/load() does not preserve input_dimension")

        _single(
            Dummy20,
            "save()/load() does not preserve the result of apply()",
        )

        Dummy21().verify()


class TestOpInfOperator:
    """Test operators._base.OpInfOperator."""

    class Dummy(_module.OpInfOperator):
        """Instantiable version of OpInfOperator."""

        def apply(*args, **kwargs):
            return -1

        def datablock(*args, **kwargs):
            pass

        def operator_dimension(*args, **kwargs):
            pass

    class Dummy2(Dummy):
        """A distinct instantiable version of OpInfOperator."""

        pass

    # Initialization ----------------------------------------------------------
    def test_init(self):
        """Test OpInfOperator.__init__()."""
        op = self.Dummy()
        assert op.entries is None

        with pytest.raises(AttributeError) as ex:
            op.jacobian(5)
        assert ex.value.args[0] == "required attribute 'entries' not set"

        A = np.random.random((10, 11))
        op = self.Dummy(A)
        assert op.entries is A

        # This should be callable now.
        assert op.jacobian(None, None) == 0

    def test_validate_entries(self):
        """Test OpInfOperator._validate_entries()."""
        func = _module.OpInfOperator._validate_entries
        with pytest.raises(TypeError) as ex:
            func([1, 2, 3, 4])
        assert ex.value.args[0] == (
            "operator entries must be NumPy or scipy.sparse array"
        )

        A = np.arange(12, dtype=float).reshape((4, 3)).T
        A[0, 0] = np.nan
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be NaN"

        A[0, 0] = np.inf
        with pytest.raises(ValueError) as ex:
            func(A)
        assert ex.value.args[0] == "operator entries must not be Inf"

        # Valid argument, no exceptions raised.
        A[0, 0] = 0
        func(A)

    # Properties --------------------------------------------------------------
    def test_entries(self):
        """Test OpInfOperator.entries and shape()."""
        op = self.Dummy()
        assert op.shape is None

        A = np.random.random((8, 6))
        op.set_entries(A)
        assert op.entries is A
        assert op.state_dimension == 8
        assert op.shape == A.shape
        del op.entries
        assert op.entries is None

        A2 = np.random.random(A.shape)
        op.entries = A2
        assert op.entries is A2

        op._clear()
        assert op.entries is None

    # Magic methods -----------------------------------------------------------
    def test_getitem(self):
        """Test OpInfOperator.__getitem__()."""
        op = self.Dummy()
        assert op[0, 1, 3:] is None

        A = np.random.random((8, 6))
        op.set_entries(A)
        for s in [slice(2), (slice(1), slice(1, 3)), slice(1, 4, 2)]:
            assert np.all(op[s] == A[s])

    def test_eq(self):
        """Test OpInfOperator.__eq__()."""
        op1 = self.Dummy()
        op2 = self.Dummy2()
        assert op1 != op2

        op2 = self.Dummy()
        assert op1 == op2

        A = np.arange(12).reshape((4, 3))
        op1.entries = A
        assert op1 != op2
        assert op2 != op1

        op2.entries = A.T
        assert op1 != op2

        op2.entries = A + 1
        assert op1 != op2

        op2.entries = A
        assert op1 == op2

    def test_add(self, r=3):
        """Test OpInfOperator.__add__()."""
        op1 = self.Dummy(np.random.random((r, r)))

        # Try with invalid type.
        with pytest.raises(TypeError) as ex:
            op1 + 10
        assert ex.value.args[0] == (
            "can't add object of type 'int' to object of type 'Dummy'"
        )

        op2 = self.Dummy(np.random.random((r, r)))
        op = op1 + op2
        assert np.all(op.entries == (op1.entries + op2.entries))

    # Dimensionality reduction ------------------------------------------------
    def test_galerkin(self, n=10, r=3):
        """Test OpInfOperator._galerkin()."""
        Vr = la.qr(np.random.random((n, r)), mode="economic")[0]
        Wr = la.qr(np.random.random((n, r)), mode="economic")[0]
        A = np.random.random((n, n))
        op = self.Dummy(A)
        c = np.random.random(n)

        op_ = op._galerkin(Vr, Vr, lambda x, y: c)
        assert isinstance(op_, op.__class__)
        assert op_ is not op
        assert op_.shape == (r,)
        assert np.all(op_.entries == Vr.T @ c)

        op_ = op._galerkin(Vr, Wr, lambda x, y: c)
        assert isinstance(op_, op.__class__)
        assert op_ is not op
        assert op_.shape == (r,)
        assert np.allclose(Wr.T @ Vr @ op_.entries, Wr.T @ c)

        Wr_bad = np.random.random((n, r - 1))
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            op._galerkin(Vr, Wr_bad, None)
        assert ex.value.args[0] == "trial and test bases not aligned"

        A = np.random.random((n - 1, r + 1))
        op = self.Dummy(A)
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            op._galerkin(Vr, None, None)
        assert ex.value.args[0] == "basis and operator not aligned"

    # Model persistence -------------------------------------------------------
    def test_copy(self):
        """Test OpInfOperator.copy()."""
        op1 = self.Dummy()
        op1.set_entries(np.random.random((4, 4)))
        op2 = op1.copy()
        assert op2 is not op1
        assert op2.entries is not op1.entries
        assert op2 == op1

    def test_save(self, target="_baseoperatorsavetest.h5"):
        """Test OpInfOperator.save()."""
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        op = self.Dummy()
        op.save(target)

        with h5py.File(target, "r") as hf:
            assert "meta" in hf
            assert "class" in hf["meta"].attrs
            assert hf["meta"].attrs["class"] == "Dummy"

        A = np.random.random((4, 3))
        op.set_entries(A)
        op.save(target, overwrite=True)

        with h5py.File(target, "r") as hf:
            assert "meta" in hf
            assert "class" in hf["meta"].attrs
            assert hf["meta"].attrs["class"] == "Dummy"
            assert "entries" in hf
            entries = hf["entries"][:]
            assert entries.shape == A.shape
            assert np.all(entries == A)

        os.remove(target)

    def test_load(self, target="_baseoperatorloadtest.h5"):
        """Test OpInfOperator.load()."""
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        op1 = self.Dummy()
        op1.save(target)
        op2 = self.Dummy.load(target)
        assert op1 == op2

        op1.set_entries(np.random.random((6, 2)))
        op1.save(target, overwrite=True)
        op2 = self.Dummy.load(target)
        assert op1 == op2

        class Dummy2(self.Dummy):
            pass

        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            Dummy2.load(target)
        assert ex.value.args[0] == (
            f"file '{target}' contains 'Dummy' object, use 'Dummy.load()"
        )

        os.remove(target)

    def test_verify(self):
        op = self.Dummy()
        op.verify()

        op.set_entries(np.random.random((8, 6)))
        with pytest.raises(opinf.errors.VerificationError):
            op.verify()


def test_is_nonparametric():
    """Test operators._base.is_nonparametric()."""

    op = TestOpInfOperator.Dummy()
    assert opinf.operators.is_nonparametric(op)
    assert not opinf.operators.is_nonparametric(10)


# Parametric operators ========================================================
class TestParametricOpInfOperator:
    """Test operators._base.ParametricOpInfOperator."""

    class Dummy(_module.ParametricOpInfOperator):
        """Instantiable version of ParametricOpInfOperator."""

        _OperatorClass = TestOpInfOperator.Dummy

        def __init__(self):
            _module.ParametricOpInfOperator.__init__(self)

        def _clear(self):
            pass

        def state_dimension(self):
            pass

        def shape(self):
            pass

        def evaluate(self, parameter):
            op = self._OperatorClass()
            op.set_entries(np.random.random((2, 2)))
            return op

        def galerkin(self, *args, **kwargs):
            pass

        def datablock(self, *args, **kwargs):
            pass

        def operator_dimension(self, *args, **kwargs):
            pass

        def copy(self, *args, **kwargs):
            pass

        def save(self, *args, **kwargs):
            pass

        def load(self, *args, **kwargs):
            pass

    def test_set_parameter_dimension_from_data(self):
        """Test _set_parameter_dimension_from_data()."""
        op = self.Dummy()
        assert op.parameter_dimension is None

        # One-dimensional parameters.
        op._set_parameter_dimension_from_data(np.arange(10))
        assert op.parameter_dimension == 1
        op._set_parameter_dimension_from_data(np.arange(5).reshape((-1, 1)))
        assert op.parameter_dimension == 1

        # n-dimensional parameters.
        n = np.random.randint(2, 20)
        op._set_parameter_dimension_from_data(np.random.random((5, n)))
        assert op.parameter_dimension == n

        with pytest.raises(ValueError) as ex:
            op._set_parameter_dimension_from_data(np.random.random((2, 2, 2)))
        assert ex.value.args[0] == (
            "parameter values must be scalars or 1D arrays"
        )

    def test_check_shape_consistency(self):
        """Test _check_shape_consistency()."""
        arrays = [np.random.random((2, 3)), np.random.random((3, 2))]
        with pytest.raises(ValueError) as ex:
            self.Dummy._check_shape_consistency(arrays, "array")
        assert ex.value.args[0] == "array shapes inconsistent"

        arrays[1] = arrays[1].T
        self.Dummy._check_shape_consistency(arrays, "array")

    def test_check_parametervalue_dimension(self, p=3):
        """Test _check_parametervalue_dimension()."""
        op = self.Dummy()

        with pytest.raises(RuntimeError) as ex:
            op._check_parametervalue_dimension(10)
        assert ex.value.args[0] == "parameter_dimension not set"

        op._set_parameter_dimension_from_data(np.empty((5, p)))

        val = np.empty(p - 1)
        with pytest.raises(ValueError) as ex:
            op._check_parametervalue_dimension(val)
        assert ex.value.args[0] == f"expected parameter of shape ({p:d},)"

        op._check_parametervalue_dimension(np.empty(p))

    def test_apply(self):
        """Test apply()."""
        assert self.Dummy().apply(None, None, None) == -1

    def test_jacobian(self):
        """Test jacobian()."""
        assert self.Dummy().jacobian(None, None, None) == 0


def test_is_parametric():
    """Test operators._base.is_parametric()."""
    op = TestParametricOpInfOperator.Dummy()
    assert opinf.operators.is_parametric(op)
    assert not opinf.operators.is_nonparametric(-1)
