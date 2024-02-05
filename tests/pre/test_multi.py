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

    class Dummy3(Dummy2, opinf.pre._base._UnivarMixin):
        def __init__(self, name=None):
            opinf.pre._base._UnivarMixin.__init__(self, name)
            TestTransformerMulti.Dummy2.__init__(self)

    def test_init(self):
        """Test TransformerMulti.__init__(), transformers."""
        transformers = [self.Dummy(), self.Dummy2(), self.Dummy3("third")]
        tfm = self.Transformer(transformers)
        assert tfm.num_variables == len(transformers)
        assert hasattr(tfm, "variable_names")
        for name in tfm.variable_names:
            assert isinstance(name, str)
        assert tfm.variable_names[-1] == "third"

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
        tfm = self.Transformer(transformers)

        stringrep = str(tfm)
        assert stringrep.startswith("2-variable TransformerMulti\n")
        for tf in transformers:
            assert str(tf) in stringrep

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
