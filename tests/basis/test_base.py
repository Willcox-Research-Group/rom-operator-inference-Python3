# basis/test_base.py
"""Tests for basis._base."""

import abc
import pytest
import numpy as np


class _TestBasisTemplate(abc.ABC):
    """Base class for tests of classes inheriting from BasisTemplate."""

    Basis = NotImplemented  # Class to test.

    @abc.abstractmethod
    def get_bases(self):
        """Yield (untrained) basis objects to test.

        Yields
        ------
        basis : opinf.Basis object
            Initialized basis object.
        n : int
            Expected basis.full_state_dimension.
        """
        raise NotImplementedError

    def test_all(self, k=20):
        """Call fit() and use verify() to test compress(), decompress(),
        and project(). Also lightly test __str__(), __repr__(),
        projection_error(), and fit_compress().
        """
        for basis, n in self.get_bases():
            Q = np.random.standard_normal((n, k))
            repr(basis)

            assert basis.fit(Q) is basis, "fit() should return self"
            assert isinstance(basis.full_state_dimension, int)
            assert basis.full_state_dimension == n
            assert isinstance(basis.reduced_state_dimension, int)

            basis.name = "varname"
            assert "varname" in str(basis)

            # Test compress(), decompress(), and project().
            basis.verify()

            # Test fit_compress()
            Q1_ = basis.fit(Q).compress(Q)
            Q2_ = basis.fit_compress(Q)
            assert isinstance(Q1_, np.ndarray)
            assert isinstance(Q2_, np.ndarray)
            assert Q1_.shape == Q2_.shape
            assert np.allclose(Q1_, Q2_)

            # Test projection_error
            assert isinstance(basis.projection_error(Q, relative=False), float)
            assert isinstance(basis.projection_error(Q, relative=True), float)

            assert isinstance(basis.copy(), type(basis))


if __name__ == "__main__":
    pytest.main([__file__])
