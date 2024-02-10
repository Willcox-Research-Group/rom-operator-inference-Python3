# pre/test_multi.py
"""Tests for pre._multi.py."""

import os
import pytest
import numpy as np

import opinf


class TestTransformerMulti:
    """Tests for pre._base.TransformerMulti."""

    Transformer = opinf.pre.TransformerMulti

    class Dummy(opinf.pre.TransformerTemplate):
        def fit_transform(self, states, inplace=False):
            return states if inplace else states.copy()

        def transform(self, states, inplace=False):
            return states if inplace else states.copy()

        def inverse_transform(self, states, inplace=False, locs=None):
            return states if inplace else states.copy()

    class Dummy2(Dummy):
        def __init__(self, name=None):
            opinf.pre.TransformerTemplate.__init__(self, name=name)
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

    # Constructor and properties ----------------------------------------------
    def test_init(self):
        """Test TransformerMulti.__init__(), transformers."""
        transformers = [self.Dummy(), self.Dummy2(), self.Dummy3(name="third")]

        with pytest.raises(ValueError) as ex:
            self.Transformer(transformers, list(range(len(transformers) + 2)))
        assert ex.value.args[0] == "len(variable_sizes) != len(transformers)"

        tfm = self.Transformer(transformers)
        assert tfm.num_variables == len(transformers)
        assert hasattr(tfm, "variable_names")
        for name in tfm.variable_names:
            assert isinstance(name, str)
        assert tfm.variable_names[-1] == "third"
        assert tfm.state_dimension is None
        assert tfm.variable_sizes.count(None) == len(transformers)
        for i, tf in enumerate(transformers):
            assert tfm.transformers[i] is tf
        assert len(tfm) == len(transformers)

        with pytest.raises(ValueError) as ex:
            tfm.transformers = []
        assert ex.value.args[0] == "at least one transformer required"

        with pytest.warns(opinf.errors.UsageWarning) as wn:
            tfm.transformers = transformers[:1]
        assert wn[0].message.args[0] == "only one variable detected"
        assert tfm.num_variables == 1

        transformers[0].state_dimension = 12
        transformers[1].state_dimension = 15
        transformers[2].state_dimension = 18
        tfm = self.Transformer(transformers)
        assert tfm.state_dimension == 45

    # Magic methods -----------------------------------------------------------
    def test_getitem(self):
        """Test TransformerMulti.__getitem__()."""
        transformers = [self.Dummy(), self.Dummy2(), self.Dummy()]
        tfm = self.Transformer(transformers)
        for i, tf in enumerate(transformers):
            assert tfm[i] is tf

        transformers[0].name = "A"
        transformers[1].name = "B"
        transformers[2].name = "C"
        for i, name in enumerate("ABC"):
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
        tfm = self.Transformer(transformers)

        stringrep = str(tfm)
        assert stringrep.startswith("2-variable TransformerMulti\n")
        for tf in transformers:
            assert str(tf) in stringrep

        # Quick repr() test.
        rep = repr(tfm)
        assert stringrep in rep
        assert str(hex(id(tfm))) in rep

    # Convenience methods -----------------------------------------------------
    def test_check_shape(self):
        """Test TransformerMulti._check_shape()."""
        transformers = [self.Dummy(name="A"), self.Dummy2(name="B")]
        tf = self.Transformer(transformers, variable_sizes=(5, 10))
        assert tf.state_dimension == 15
        q = np.random.randint(0, 100, size=tf.state_dimension)
        tf._check_shape(q)

        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            tf._check_shape(q[:-1])
        assert ex.value.args[0] == ("len(states) = 14 != 15 = state_dimension")

    def test_get_var(self, n1=11, n2=18):
        """Test TransformerMulti.get_var()."""
        tfA = self.Dummy(name="A")
        tfA.state_dimension = n1
        tfB = self.Dummy(name="B")
        tfB.state_dimension = n2

        tf = self.Transformer([tfA, tfB])
        q = np.random.random(n1 + n2)

        q0 = tf.get_var(0, q)
        assert q0.shape == (n1,)
        assert np.all(q0 == q[:n1])

        q1 = tf.get_var(1, q)
        assert q1.shape == (n2,)
        assert np.all(q1 == q[n1:])

    def test_split(self, n1=6, n2=8):
        """Test TransformerMulti.split()."""
        transformers = [self.Dummy(name="A"), self.Dummy2(name="B")]
        tf = self.Transformer(transformers, variable_sizes=(n1, n2))

        q = np.random.random(n1 + n2)
        q_split = tf.split(q)
        assert len(q_split) == 2
        assert np.all(q_split[0] == q[:n1])
        assert np.all(q_split[1] == q[n1:])

    # Main routines -----------------------------------------------------------
    def test_mains(self, nx=50, k=400):
        """Use TransformerMulti.verify() to run tests."""
        transformers = [self.Dummy(), self.Dummy2()]

        # Uniform variable sizes.
        n = len(transformers) * nx
        Q = np.random.random((n, k))

        tfm = self.Transformer(transformers)
        for method in "transform", "inverse_transform", "transform_ddts":
            with pytest.raises(AttributeError) as ex:
                getattr(tfm, method)(Q)
            assert ex.value.args[0] == (
                "transformer not trained, call fit() or fit_transform()"
            )

        assert tfm.state_dimension is None
        with pytest.raises(ValueError) as ex:
            tfm.fit(Q[1:])
        assert ex.value.args[0] == (
            "len(states) must be evenly divisible "
            "by the number of variables n_q = 2"
        )

        assert tfm.fit(Q) is tfm
        assert tfm.state_dimension == n
        assert tfm.transform_ddts(Q) is NotImplemented
        tfm.verify(Q)

        for i in range(len(transformers)):
            transformers[i] = self.Dummy2()
        tfm.fit(Q)
        t = np.linspace(0, 0.1, k)
        tfm.verify(Q, t)

        with pytest.raises(Exception):
            tfm.fit(100)
        assert tfm.state_dimension == n

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

        tf = self.Transformer(transformers, variable_sizes=(12, 16, 16))
        tf.save(target, overwrite=True)
        assert os.path.isfile(target)

        os.remove(target)

    def test_load(self):
        """Test TransformerMulti.load()."""
        target = "_loadtransformermultitest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        transformers = [self.Dummy2(), self.Dummy2(), self.Dummy3()]
        TCs = [obj.__class__ for obj in transformers]
        num_variables = len(transformers)

        # Check that save() -> load() gives the same transformer.
        tfm_original = self.Transformer(transformers)
        tfm_original.save(target)
        tfm = self.Transformer.load(target, TransformerClasses=TCs)
        assert len(tfm.transformers) == num_variables
        for i, tf in enumerate(transformers):
            assert tfm[i].__class__ is tf.__class__
            assert tfm[i].data.shape == tf.data.shape
            assert np.all(tfm[i].data == tf.data)
        assert tfm.state_dimension is None

        for i, nx in enumerate(np.random.randint(2, 10, num_variables)):
            tfm_original[i].state_dimension = nx
        tfm_original.save(target, overwrite=True)
        tfm = self.Transformer.load(target, TransformerClasses=TCs)
        assert tfm.num_variables == tfm_original.num_variables
        assert tfm.variable_sizes == tfm_original.variable_sizes
        assert tfm.state_dimension == tfm_original.state_dimension

        os.remove(target)

    # Verification ------------------------------------------------------------
    def test_verify_locs(self, nvar=3, nx=11, k=12):
        """Test TransformerMulti._verify_locs()."""

        class DummyMulti(self.Transformer):
            def transform(self, states, inplace=False):
                return states if inplace else states.copy()

            def inverse_transform(
                self,
                states_transformed,
                inplace=False,
                locs=None,
            ):
                return (
                    states_transformed
                    if inplace
                    else states_transformed.copy()
                )

        tf1 = self.Dummy()
        tf1.state_dimension = 10
        tf2 = self.Dummy2()
        tf2.state_dimension = 11
        tf3 = self.Dummy3()
        tf3.state_dimension = 10
        transformers = [tf1, tf2, tf3]

        tf = DummyMulti(transformers)
        Q = np.random.random((31, k))
        assert tf.verify(Q) is None

        tf2.state_dimension = 10
        tf._set_slices()
        Q = Q[1:, :]
        Qt = tf.transform(Q)
        tf._verify_locs(Q, Qt)

        class DummyMulti2(DummyMulti):
            def inverse_transform(self, states_transformed, locs=None):
                if locs is None:
                    return states_transformed
                return states_transformed[1:-1]

        tf = DummyMulti2(transformers)
        with pytest.raises(opinf.errors.VerificationError) as ex:
            tf._verify_locs(Q, Qt)
        assert ex.value.args[0] == (
            "inverse_transform(states_transformed_at_locs, locs).shape "
            "!= states_at_locs.shape"
        )

        class DummyMulti3(DummyMulti):
            def inverse_transform(self, states_transformed, locs=None):
                if locs is None:
                    return states_transformed
                return states_transformed + 1

        tf = DummyMulti3(transformers)
        with pytest.raises(opinf.errors.VerificationError) as ex:
            tf._verify_locs(Q, Qt)
        assert ex.value.args[0] == (
            "transform() and inverse_transform() are not inverses "
            "(locs != None)"
        )
