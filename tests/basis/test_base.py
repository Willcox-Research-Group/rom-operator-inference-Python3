# basis/test_base.py
"""Tests for basis._base."""

import opinf


class TestBaseBasis:
    """Test basis._base._BaseBasis."""

    class Dummy(opinf.basis._base._BaseBasis):
        """Instantiable version of _BaseBasis."""
        def fit(self):
            pass

        def compress(self, state):
            return state + 2

        def decompress(self, state_):
            return state_ - 1

    def test_project(self):
        """Test _BaseBasis.project() and _BaseBasis.projection_error()."""
        basis = self.Dummy()
        state = 5
        assert basis.project(state) == (state + 1)
        assert basis.projection_error(state, relative=False) == 1
        assert basis.projection_error(state, relative=True) == 1/state
