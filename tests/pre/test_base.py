# pre/test_base.py
"""Tests for pre._base.py."""

import os
import abc
import pytest
import numpy as np


class _TestTransformer(abc.ABC):
    """Base class for tests of classes that inherit from
    opinf.pre._base.TransformerTemplate.
    """

    Transformer = NotImplemented
    exact_inverse = True
    saveload = True

    @abc.abstractmethod
    def get_transformer(self, name=None):
        """Initialize a Transformer for testing."""
        pass  # pragma: no cover

    def test_name(self):
        """Test the name attribute."""
        tf = self.get_transformer(name=None)
        assert tf.name is None

        s1 = "the name"
        tf = self.get_transformer(name=s1)
        assert tf.name == s1

        s2 = "new name"
        tf.name = s2
        assert tf.name == s2

        tf.name = None
        assert tf.name is None

    def test_state_dimension(self):
        """Test the state_dimension setter."""
        tf = self.get_transformer()
        tf.state_dimension = 10.0
        n = tf.state_dimension
        assert isinstance(n, int)
        assert tf.state_dimension == n
        tf.state_dimension = None
        assert tf.state_dimension is None

    def test_str(self):
        """Lightly test __str__() and __repr__()."""
        repr(self.get_transformer())

    def test_mains(self, n=11, k=21):
        """Test fit(), fit_transform(), transform(), transform_ddts(), and
        inverse_transform().
        """
        tf = self.get_transformer()
        if (new_n := tf.state_dimension) is not None:
            n = new_n
        Q = np.random.random((n, k))
        Y = np.random.random((n, k + 1))
        y = Y[:, 0]
        Z = np.random.random((n, k + 2))
        z = Z[:, 0]

        # Test fit().
        out = tf.fit(Q)
        assert out is tf
        assert tf.state_dimension == n

        # Test transform() and transform_ddts()
        for method in (tf.transform, tf.transform_ddts):
            y1 = method(y, inplace=False)
            assert y1 is not y
            assert y1.shape == y.shape
            Y1 = method(Y, inplace=False)
            assert Y1 is not Y
            assert Y1.shape == Y.shape
            assert np.allclose(y1, Y1[:, 0])
            Z1 = method(Z, inplace=True)
            assert Z1 is Z
            z1 = method(z, inplace=True)
            assert z1 is z

        # Test fit_transform().
        Q1 = tf.transform(Q, inplace=False)
        Q2 = tf.fit_transform(Q, inplace=False)
        assert Q2 is not Q1 is not Q
        assert Q2.shape == Q1.shape == Q.shape
        assert np.allclose(Q2, Q1)
        Z1 = tf.fit_transform(Z, inplace=True)
        assert Z1 is Z

        # Test inverse_transform().
        if self.exact_inverse:
            Y1 = tf.transform(Y, inplace=False)
            Y2 = tf.inverse_transform(Y1, inplace=False)
            assert Y2.shape == Y.shape
            assert Y2 is not Y1
            assert np.allclose(Y2, Y)
            Y3 = tf.inverse_transform(Y1, inplace=True)
            assert Y3 is Y1
            assert np.allclose(Y3, Y2)

    def test_saveload(self, n=24, k=50):
        """Test save() and load()."""
        if not self.saveload:
            return

        target = f"_{self.Transformer.__name__}_saveloadtest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        s = "thename"
        tf = self.get_transformer(name=s)
        if (new_n := tf.state_dimension) is not None:
            n = new_n

        def _check():
            assert os.path.isfile(target)
            tf2 = self.Transformer.load(target)
            assert tf2 is not tf
            assert isinstance(tf2, self.Transformer)
            assert tf2.name == s
            if tf.state_dimension is not None:
                assert tf2.state_dimension == n

        tf.save(target)
        _check()

        tf.fit(np.random.random((n, k)))
        with pytest.raises(FileExistsError):
            tf.save(target, overwrite=False)
        tf.save(target, overwrite=True)
        _check()

        os.remove(target)

    def test_verify(self, n=30):
        """Test verify()."""
        tf = self.get_transformer()

        old_n = tf.state_dimension
        tf.state_dimension = None
        with pytest.raises(AttributeError) as ex:
            tf.verify()
        assert ex.value.args[0] == (
            "transformer not trained (state_dimension not set), "
            "call fit() or fit_transform()"
        )

        tf.state_dimension = n if old_n is None else old_n
        tf.verify()
