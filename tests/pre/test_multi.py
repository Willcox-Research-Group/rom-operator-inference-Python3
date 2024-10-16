# pre/test_multi.py
"""Tests for pre._multi.py."""

import os
import pytest
import numpy as np

import opinf

from .test_base import _TestTransformer


class TestTransformerPipeline(_TestTransformer):
    """Tests for pre._base.TransformerPipeline."""

    state_dimension = 20
    requires_training = None

    class Transformer(opinf.pre.TransformerPipeline):
        @classmethod
        def load(cls, loadfile):
            return super().load(
                loadfile,
                TransformerClasses=[
                    opinf.pre.ShiftTransformer,
                    opinf.pre.ScaleTransformer,
                ],
            )

    def get_transformers(self, name=None):
        t1 = opinf.pre.ShiftTransformer(np.random.random(self.state_dimension))
        t2 = opinf.pre.ScaleTransformer(np.random.random())
        self.requires_training = True
        yield self.Transformer([t1, t2], name=name)

        t1 = opinf.pre.ShiftTransformer(np.random.random(self.state_dimension))
        t2 = opinf.pre.ScaleTransformer(np.random.random(self.state_dimension))
        self.requires_training = False
        yield self.Transformer([t1, t2], name=name)

    def test_init(self):
        """Test __init__() and properties."""
        with pytest.raises(TypeError) as ex:
            self.Transformer("moosen", name="bryan")
        assert ex.value.args[0] == "'transformers' should be a list or tuple"

        class Dummy:
            state_dimension = 10

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            tf = self.Transformer([Dummy(), Dummy()])
        assert len(wn) == 2
        assert wn[0].message.args[0] == (
            "transformers[0] does not inherit from TransformerTemplate, "
            "unexpected behavior may occur"
        )
        assert tf.state_dimension == 10
        assert len(tf) == 2

        Q = np.empty((self.state_dimension, 2))
        t1 = opinf.pre.NullTransformer().fit(Q)
        t2 = opinf.pre.NullTransformer().fit(Q[1:-1, :])
        with pytest.raises(ValueError) as ex:
            self.Transformer([t1, t2])
        assert ex.value.args[0] == (
            "transformers have inconsistent state_dimension"
        )

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            tf = self.Transformer([opinf.pre.NullTransformer()])
        assert len(wn) == 1
        assert wn[0].message.args[0] == (
            "only one transformer provided to TransformerPipeline"
        )
        assert tf.state_dimension is None

    def test_saveload(self):
        """Test save() and load()."""
        super().test_saveload()

        target = "_TransformerPipeline_saveloadtest.h5"
        tf = self.get_transformer()
        tf.save(target, overwrite=True)
        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            opinf.pre.TransformerPipeline.load(
                target, [list, tuple, set, dict] * 2
            )
        assert ex.value.args[0].startswith("file contains")

        os.remove(target)


class TestNullTransformer(_TestTransformer):
    """Tests for pre._base.NullTransformer."""

    Transformer = opinf.pre.NullTransformer
    requires_training = False

    def get_transformers(self, name=None):
        yield self.Transformer(name=name)


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
                if self.state_dimension is not None:
                    hf.create_dataset("dim", data=[self.state_dimension])

        @classmethod
        def load(cls, loadfile):
            dummy = cls()
            with opinf.utils.hdf5_loadhandle(loadfile) as hf:
                dummy.data = hf["data"][:]
                if "dim" in hf:
                    dummy.state_dimension = hf["dim"][0]
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
            self.Transformer([])
        assert ex.value.args[0] == "at least one transformer required"

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            tfm = self.Transformer(transformers[:1])
        assert wn[0].message.args[0] == "only one variable detected"
        assert tfm.num_variables == 1

        transformers[0].state_dimension = 12
        transformers[1].state_dimension = 15
        transformers[2].state_dimension = 18
        tfm = self.Transformer(transformers)
        assert tfm.state_dimension == 45

        with pytest.raises(ValueError) as ex:
            self.Transformer(transformers, variable_sizes=(12, 15, 20))
        assert ex.value.args[0] == (
            "transformers[2].state_dimension = 18 != 20 = variable_sizes[2]"
        )

        class ExtraDummy:
            name = "nothing"

        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            self.Transformer([ExtraDummy(), ExtraDummy()])
        assert len(wn) == 2
        assert wn[0].message.args[0].startswith("transformers[0] does not")
        assert wn[1].message.args[0].startswith("transformers[1] does not")

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
        """Lightly test TransformerMulti.__str__()."""
        tfm = self.Transformer([self.Dummy(), self.Dummy2()])
        str(tfm)
        tfm.transformers[0].state_dimension = 4
        tfm.transformers[1].state_dimension = 4
        repr(tfm)

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

        q1 = tf.get_var("B", q)
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
        with pytest.raises(opinf.errors.DimensionalityError) as ex:
            tfm.fit(Q[1:])
        assert ex.value.args[0] == (
            "len(states) must be evenly divisible "
            "by the number of variables n_q = 2"
        )

        assert tfm.fit(Q) is tfm
        assert tfm.state_dimension == n
        assert tfm.transform_ddts(Q) is NotImplemented
        tfm.verify()

        transformers[0].state_dimension = nx + 1
        with pytest.raises(ValueError) as ex:
            tfm.inverse_transform(Q, locs=True)
        assert ex.value.args[0] == (
            "'locs != None' requires that "
            "all transformers have the same state_dimension"
        )

        for i in range(len(transformers)):
            transformers[i] = self.Dummy2()
        tfm = self.Transformer(transformers, variable_sizes=(nx + 10, nx - 10))
        tfm.fit(Q)
        tfm.verify()

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

        tfm_original = self.Transformer(transformers)
        tfm_original.save(target)
        with pytest.raises(opinf.errors.LoadfileFormatError) as ex:
            self.Transformer.load(target, TransformerClasses=TCs[:-1])
        assert ex.value.args[0] == (
            "file contains 3 transformers but 2 classes provided"
        )

        # Check that save() -> load() gives the same transformer.
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
        tfm = self.Transformer.load(target, TransformerClasses=TCs[0])
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
        assert tf.verify() is None

        tf2.state_dimension = 10
        Q = np.random.random((tf.state_dimension, k))
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


if __name__ == "__main__":
    pytest.main([__file__])
