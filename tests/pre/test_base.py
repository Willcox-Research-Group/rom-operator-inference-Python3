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

    # Setup -------------------------------------------------------------------
    Transformer = NotImplemented  # Class being tested.
    requires_training = True  # If fit() required before transform().
    exact_inverse = True  # If transform()/inverse_transform() are inverses
    saveload = True  # Whether to test save()/load().

    @abc.abstractmethod
    def get_transformers(self, name=None):
        """Yield transformers for testing."""
        pass  # pragma: no cover

    def get_transformer(self, name=None):
        """Get a single transformer for testing."""
        return next(self.get_transformers(name=name))

    # Tests -------------------------------------------------------------------
    def test_name(self):
        """Test the name attribute."""
        tf = self.get_transformer(name=None)
        assert tf.name is None

        s1 = "the name"
        tf = self.get_transformer(name=s1)
        assert tf.name == s1

        tf.name = (s2 := "new name")
        assert tf.name == s2

        tf.name = None
        assert tf.name is None

    def test_state_dimension(self):
        """Test the state_dimension setter if the state_dimension is not
        already set after the initializer.
        """
        for tf in self.get_transformers():
            if tf.state_dimension is None:
                tf.state_dimension = 10.0
                assert isinstance(tf.state_dimension, int)
                assert tf.state_dimension == 10
                tf.state_dimension = None
                assert tf.state_dimension is None

    def test_str(self):
        """Lightly test __str__() and __repr__()."""
        for tf in self.get_transformers():
            repr(tf)

    def test_mains(self, n=11, k=21):
        """Test fit(), fit_transform(), transform(), transform_ddts(), and
        inverse_transform().
        """
        for tf in self.get_transformers():
            if (new_n := tf.state_dimension) is not None:
                n = new_n
            Q = np.random.random((n, k))
            Y = np.random.random((n, k + 1))
            y = Y[:, 0]
            Z = np.random.random((n, k + 2))
            z = Z[:, 0]

            # Check that transform(), transform_ddts(), and inverse_transform()
            # cannot be called before training.
            if self.requires_training:
                for method in (
                    tf.transform,
                    tf.transform_ddts,
                    tf.inverse_transform,
                ):
                    with pytest.raises(AttributeError):
                        method(Q)

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
            Y1 = tf.transform(Y, inplace=False)
            Y2 = tf.inverse_transform(Y1, inplace=False)
            assert Y2.shape == Y.shape
            assert Y2 is not Y1
            Y3 = tf.inverse_transform(Y1, inplace=True)
            assert Y3 is Y1
            if self.exact_inverse:
                assert np.allclose(Y2, Y)
                assert np.allclose(Y3, Y2)
            Y1 = tf.transform(Y, inplace=False)
            locs = np.sort(np.random.choice(n, n // 3, replace=False))
            Y2locs = tf.inverse_transform(Y1[locs], locs=locs, inplace=False)
            assert Y2locs.shape == (locs.size, Y1.shape[1])
            if self.exact_inverse:
                assert np.allclose(Y2locs, Y[locs])

            # Test shape issues.
            with pytest.raises(ValueError):
                tf.transform(Q[1:-1, :])
            with pytest.raises(ValueError):
                tf.inverse_transform(Q, locs=slice(0, 3))

    def test_saveload(self, n=24, k=50):
        """Test save() and load()."""
        if not self.saveload:
            return

        target = f"_{self.Transformer.__name__}_saveloadtest.h5"
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        s = "thename"
        for tf in self.get_transformers(name=s):
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

    def test_verify(self, n=30, k=50):
        """Test verify()."""
        for tf in self.get_transformers():
            if tf.state_dimension is None:
                with pytest.raises(AttributeError) as ex:
                    tf.verify()
                assert ex.value.args[0] == (
                    "transformer not trained (state_dimension not set), "
                    "call fit() or fit_transform()"
                )
                tf.state_dimension = n

            tf.fit(np.random.random((tf.state_dimension, k)))
            tf.verify()


if __name__ == "__main__":
    pytest.main([__file__])
