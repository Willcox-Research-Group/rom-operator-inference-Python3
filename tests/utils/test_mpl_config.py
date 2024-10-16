# utils/test_mpl_config.py
"""Tests for utils._mpl_config."""

import opinf


def test_mpl_config():
    """Test utils._mpl_config.mpl_config()."""
    opinf.utils.mpl_config()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
