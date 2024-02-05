# pre/test_base.py
"""Tests for pre._base.py."""

import os
import pytest
import numpy as np

import opinf


class TestTransformerTemplate:
    """Test pre._base.TransformerTemplate."""

    class Dummy(opinf.pre.TransformerTemplate):
        def fit_transform(self, states, inplace=False):
            return states if inplace else states.copy()

        def transform(self, states, inplace=False):
            return states if inplace else states.copy()

        def inverse_transform(self, states, inplace=False, locs=None):
            return states if inplace else states.copy()

    def test_fit(self):
        """Test TransformerTemplate.fit()."""
        tf = self.Dummy()
        out = tf.fit(np.random.random((2, 5)))
        assert out is tf

    def test_verify(self, n=30, k=16):
        """Test TransformerTemplate.verify()."""
        dummy = self.Dummy()
        q = np.random.random(n)

        with pytest.raises(ValueError) as ex:
            dummy.verify(q)
        assert ex.value.args[0] == (
            "two-dimensional states required for verification"
        )

        Q = np.random.random((n, k))
        dummy.verify(Q)

        class Dummy2a(self.Dummy):
            def transform(self, states, inplace=False):
                return states[:-1]

        class Dummy2b(self.Dummy):
            def transform(self, states, inplace=False):
                return states

        class Dummy2c(self.Dummy):
            def transform(self, states, inplace=False):
                return states.copy()

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2a().verify(Q)
        assert ex.value.args[0] == "transform(states).shape != states.shape"

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2b().verify(Q)
        assert ex.value.args[0] == "transform(states, inplace=False) is states"

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy2c().verify(Q)
        assert ex.value.args[0] == (
            "transform(states, inplace=True) is not states"
        )

        class Dummy3a(self.Dummy):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed[:-1]

        class Dummy3b(self.Dummy):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed

        class Dummy3c(self.Dummy):
            def inverse_transform(self, states_transformed, inplace=False):
                return states_transformed.copy()

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3a().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)).shape != states.shape"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3b().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed, inplace=False) "
            "is states_transformed"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy3c().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed, inplace=True) "
            "is not states_transformed"
        )

        class Dummy4(self.Dummy):
            def inverse_transform(
                self,
                states_transformed,
                inplace=False,
                locs=None,
            ):
                if locs is None:
                    if inplace:
                        return states_transformed
                    return states_transformed.copy()
                return states_transformed[:-1]

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy4().verify(Q)
        assert ex.value.args[0] == (
            "inverse_transform(transform(states)[locs], locs).shape "
            "!= states[locs].shape"
        )

        class Dummy5a(self.Dummy):
            def inverse_transform(
                self,
                states_transformed,
                inplace=False,
                locs=None,
            ):
                Q = states_transformed
                if not inplace:
                    Q = Q.copy()
                Q += 1
                return Q

        class Dummy5b(self.Dummy):
            def inverse_transform(
                self,
                states_transformed,
                inplace=False,
                locs=None,
            ):
                Q = states_transformed
                if not inplace:
                    Q = Q.copy()
                if locs is not None:
                    Q += 1
                return Q

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5a().verify(Q)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy5b().verify(Q)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses "
            "(locs != None)"
        )

        class Dummy6a(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                return ddts

        class Dummy6b(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                dQ = ddts if inplace else ddts.copy()
                dQ += 1e16
                return dQ

        class Dummy6c(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                return ddts.copy()

        with pytest.raises(ValueError) as ex:
            Dummy6a().verify(Q)
        assert ex.value.args[0] == (
            "time domain 't' required for finite difference check"
        )

        t = np.linspace(0, 1, k)
        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6a().verify(Q, t)
        assert ex.value.args[0].startswith(
            "transform_ddts(ddts, inplace=False) is ddts"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6b().verify(Q, t)
        assert ex.value.args[0].startswith(
            "transform_ddts() failed finite difference check"
        )

        with pytest.raises(opinf.errors.VerificationError) as ex:
            Dummy6c().verify(Q, t)
        assert ex.value.args[0].startswith(
            "transform_ddts(ddts, inplace=True) is not ddts"
        )

        class Dummy7(self.Dummy):
            def transform_ddts(self, ddts, inplace=False):
                return ddts if inplace else ddts.copy()

        Dummy7().verify(Q, t)


class TestUnivarMixin:
    """Tests for pre._base._UnivarMixin."""

    Mixin = opinf.pre._base._UnivarMixin

    def test_full_state_dimension(self):
        """Test _UnivarMixin.full_state_dimension."""
        mixin = self.Mixin()
        assert mixin.full_state_dimension is None
        mixin.full_state_dimension = 10.0
        n = mixin.full_state_dimension
        assert isinstance(n, int)
        assert mixin.full_state_dimension == n


class TestMultivarMixin:
    """Tests for pre._base._MultivarMixin."""

    Mixin = opinf.pre._base._MultivarMixin

    def test_init(self, nvar=4):
        """Test _MultivarMixin.__init__()."""
        with pytest.raises(TypeError) as ex:
            self.Mixin(-1)
        assert ex.value.args[0] == "'num_variables' must be a positive integer"

        mix = self.Mixin(nvar)
        assert mix.num_variables == nvar
        assert mix.full_state_dimension is None
        assert mix.variable_size is None

        assert len(mix) == nvar

    def test_variable_names(self, nvar=3):
        """Test _MultivarMixin.variable_names."""
        vnames = list("abcdefghijklmnopqrstuvwxyz")[:nvar]
        mix = self.Mixin(nvar, vnames)
        assert len(mix.variable_names) == nvar
        for i in range(nvar):
            assert mix.variable_names[i] == vnames[i]

        mix.variable_names = None
        assert len(mix.variable_names) == nvar
        for name in mix.variable_names:
            assert name.startswith("variable ")

        with pytest.raises(ValueError) as ex:
            mix.variable_names = vnames[:-1]
        assert ex.value.args[0] == f"variable_names must have length {nvar}"

    def test_properties(self, nvar=5):
        """Test _MultivarMixin.full_state_dimension and variable_size."""
        mix = self.Mixin(nvar)
        assert mix.full_state_dimension is None
        assert mix.variable_size is None

        with pytest.raises(ValueError) as ex:
            mix.full_state_dimension = 2 * nvar - 1
        assert ex.value.args[0] == (
            "'full_state_dimension' must be evenly divisible "
            "by 'num_variables'"
        )

        for size in np.random.randint(3, 10, size=5):
            mix.full_state_dimension = (n := nvar * size)
            assert mix.full_state_dimension == n
            assert mix.variable_size == size

    # Convenience methods -----------------------------------------------------
    def test_check_shape(self, nvar=12, nx=10, k=20):
        """Test _MultivarMixin._check_shape()."""
        mix = self.Mixin(nvar)
        mix.full_state_dimension = (n := nvar * nx)
        X = np.random.randint(0, 100, (n, k)).astype(float)
        mix._check_shape(X)

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            mix._check_shape(X[:-1])
        assert ex.value.args[0] == (
            f"states.shape[0] = {n - 1:d} != {nvar:d} * {nx:d} "
            "= num_variables * variable_size = full_state_dimension"
        )

    def test_get_var(self, nvar=4, nx=9, k=3):
        """Test _MultivarMixin.get_var()."""
        mix = self.Mixin(nvar, variable_names="abcdefghijklmnop"[:nvar])
        mix.full_state_dimension = (n := nvar * nx)
        q = np.random.random(n)

        q0 = mix.get_var(0, q)
        assert q0.shape == (nx,)
        assert np.all(q0 == q[:nx])

        q1 = mix.get_var(1, q)
        assert q1.shape == (nx,)
        assert np.all(q1 == q[nx : (2 * nx)])

        q2 = mix.get_var("c", q)
        assert q2.shape == (nx,)
        assert np.all(q2 == q[2 * nx : 3 * nx])

    def test_split(self, nvar=5, nx=11, k=12):
        """Test _MultivarMixin.split()."""
        mix = self.Mixin(nvar)
        mix.full_state_dimension = (n := nvar * nx)

        q = np.random.random(n)
        q_split = mix.split(q)
        assert len(q_split) == nvar
        assert all(qi.shape[0] == nx for qi in q_split)
        assert np.all(q_split[0] == q[:nx])

        Q = np.random.random((n, k))
        Q_split = mix.split(Q)
        assert len(Q_split) == nvar
        assert all(Qi.shape == (nx, k) for Qi in Q_split)
        assert np.all(Q_split[0] == Q[:nx])

    # Verification ------------------------------------------------------------
    def test_verify_locs(self, nvar=3, nx=11, k=12):
        """Test _MultivarMixin._verify_locs()."""

        class Dummy(self.Mixin):
            def transform(self, states, locs=None):
                return states

            def inverse_transform(
                self,
                states_transformed,
                locs=None,
            ):
                return states_transformed

        mix = Dummy(nvar)
        mix.full_state_dimension = (n := nvar * nx)
        Q = np.random.random((n, k))
        Qt = mix.transform(Q)
        mix._verify_locs(Q, Qt)

        class Dummy2(Dummy):
            def inverse_transform(self, states_transformed, locs=None):
                if locs is None:
                    return states_transformed
                return states_transformed[1:-1]

        mix = Dummy2(nvar)
        mix.full_state_dimension = n
        with pytest.raises(opinf.errors.VerificationError) as ex:
            mix._verify_locs(Q, Qt)
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed_at_locs, locs).shape "
            "!= states_at_locs.shape"
        )

        class Dummy3(Dummy):
            def inverse_transform(self, states_transformed, locs=None):
                if locs is None:
                    return states_transformed
                return states_transformed + 1

        mix = Dummy3(nvar)
        mix.full_state_dimension = n
        with pytest.raises(opinf.errors.VerificationError) as ex:
            mix._verify_locs(Q, Qt)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses "
            "(locs != None)"
        )


class TestTransformerMulti:
    """Tests for pre._base.TransformerMulti."""

    Transformer = opinf.pre.TransformerMulti

    Dummy = TestTransformerTemplate.Dummy

    class Dummy2(Dummy):
        def __init__(self):
            self.data = np.random.random(np.random.randint(1, 10, size=2))

        def __eq__(self, other):
            if self.data.shape != other.data.shape:
                return False
            return np.all(self.data == other.data)

        def transform_ddts(self, ddts, inplace=False):
            return ddts if inplace else ddts.copy()

        def save(self, savefile, overwrite=False):
            with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
                hf.create_dataset("data", data=self.data)

        @classmethod
        def load(cls, loadfile):
            dummy = cls()
            with opinf.utils.hdf5_loadhandle(loadfile) as hf:
                dummy.data = hf["data"][:]
            return dummy

    class Dummy3(Dummy2):
        pass

    def test_init(self):
        """Test TransformerMulti.__init__(), transformers."""
        transformers = [self.Dummy(), self.Dummy2(), self.Dummy3()]
        tfm = self.Transformer(transformers)
        assert tfm.num_variables == len(transformers)
        assert hasattr(tfm, "variable_names")

        for i, tf in enumerate(transformers):
            assert tfm.transformers[i] is tf

        with pytest.raises(ValueError) as ex:
            tfm.transformers = transformers[:-1]
        assert ex.value.args[0] == "len(transformers) != num_variables"

    # Magic methods -----------------------------------------------------------
    def test_getitem(self):
        """Test TransformerMulti.__getitem__()."""
        transformers = [self.Dummy(), self.Dummy2(), self.Dummy()]
        tfm = self.Transformer(transformers)
        for i, tf in enumerate(transformers):
            assert tfm[i] is tf

        tfm.variable_names = "ABC"
        for i, name in enumerate(tfm.variable_names):
            assert tfm[name] is transformers[i]

    def test_eq(self):
        """Test TransformerMulti.__eq__()."""
        transformers = [self.Dummy(), self.Dummy2(), self.Dummy3()]

        tfm1 = self.Transformer(transformers)
        assert tfm1 != 10

        tfm2 = self.Transformer(transformers[:-1])
        assert not tfm1 == tfm2

        tfm2 = self.Transformer(transformers[:-1] + [self.Dummy3()])
        tfm2.transformers[-1].data = tfm1.transformers[-1].data + 1
        assert tfm1 != tfm2

        tfm2.transformers[-1].data = tfm1.transformers[-1].data
        assert tfm1 == tfm2

    def test_str(self):
        """Test TransformerMulti.__str__()."""
        transformers = [self.Dummy(), self.Dummy2()]
        names = ["var1", "var2"]
        tfm = self.Transformer(transformers, variable_names=names)

        stringrep = str(tfm)
        assert stringrep.startswith("2-variable transformer\n")
        for name in names:
            assert stringrep.count(f"* {name} | ") == 1

        # Quick repr() test.
        rep = repr(tfm)
        assert stringrep in rep
        assert str(hex(id(tfm))) in rep

    # Main routines -----------------------------------------------------------
    def test_mains(self, nx=50, k=400):
        """Use TransformerMulti.verify() to run tests."""
        transformers = [self.Dummy(), self.Dummy2()]
        n = len(transformers) * nx
        Q = np.random.random((n, k))

        tfm = self.Transformer(transformers)
        for method in "transform", "inverse_transform", "transform_ddts":
            with pytest.raises(AttributeError) as ex:
                getattr(tfm, method)(Q)
            assert ex.value.args[0] == (
                "transformer not trained (call fit() or fit_transform())"
            )

        assert tfm.fit(Q) is tfm
        assert tfm.full_state_dimension == n
        assert tfm.transform_ddts(Q) is NotImplemented
        tfm.verify(Q)

        for i in range(len(transformers)):
            transformers[i] = self.Dummy2()
        tfm.fit(Q)
        t = np.linspace(0, 0.1, k)
        tfm.verify(Q, t)

        with pytest.raises(Exception):
            tfm.fit(100)
        assert tfm.full_state_dimension == n

    # Persistence -------------------------------------------------------------
    def test_save(self):
        """Lightly test TransformerMulti.save()."""
        target = "_savetransformermultitest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        transformers = [self.Dummy2(), self.Dummy2(), self.Dummy3()]
        tf = self.Transformer(transformers)
        tf.save(target)
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self):
        """Test TransformerMulti.load()."""
        target = "_loadtransformermultitest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        # Check that save() -> load() gives the same transformer.
        transformers = [self.Dummy2(), self.Dummy2(), self.Dummy3()]
        tfm_original = self.Transformer(transformers)
        tfm_original.save(target)
        tfm = self.Transformer.load(
            target,
            TransformerClasses=[obj.__class__ for obj in transformers],
        )
        assert len(tfm.transformers) == len(transformers)
        for i, tf in enumerate(transformers):
            assert tfm[i].__class__ is tf.__class__
            assert tfm[i].data.shape == tf.data.shape
            assert np.all(tfm[i].data == tf.data)
        assert tfm.full_state_dimension is None

        tfm_original.full_state_dimension = 4 * len(transformers)
        tfm_original.save(target, overwrite=True)
        tfm = self.Transformer.load(
            target,
            TransformerClasses=[obj.__class__ for obj in transformers],
        )
        assert tfm.full_state_dimension == tfm_original.full_state_dimension

        os.remove(target)
