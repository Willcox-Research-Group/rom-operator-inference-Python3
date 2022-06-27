# utils/test_hdf5.py
"""Tests for rom_operator_inference.utils._hdf5."""

import os
import h5py
import pytest

import rom_operator_inference as opinf


def test_hdf5_filehandle():
    """Test utils._hdf5._hdf5_filehandle()."""
    subject = opinf.utils._hdf5._hdf5_filehandle

    # Clean up after old tests.
    target = "_hdf5handletest.h5"
    if os.path.isfile(target):              # pragma: no cover
        os.remove(target)

    # Input file is already an open h5py handle.
    h5file = h5py.File(target, 'a')
    with subject(h5file, "load", False) as hf:
        assert hf is h5file
        assert bool(hf)
        assert hf.mode == "r+"
    assert bool(hf)  # check the file is still open.
    hf.close()
    os.remove(target)

    # Save mode without .h5 extension (should append automatically).
    with subject(target[:-3], "save", True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert os.path.isfile(target)
    assert not bool(hf)  # check the file is closed.

    # Try to overwrite but with overwrite=False.
    with pytest.raises(FileExistsError) as ex:
        with subject(target, "save", overwrite=False):
            pass
    assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

    # Overwrite.
    with subject(target, "save", overwrite=True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert not bool(hf)
    assert os.path.isfile(target)

    # Loading.
    with subject(target, "load", False) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r"
    assert not bool(hf)
    hf.close()
    os.remove(target)

    # Try loading a nonexistent file.
    with pytest.raises(FileNotFoundError) as ex:
        with subject(target, "load", overwrite=False):
            pass
    assert ex.value.args[0] == target

    # Invalid mode.
    with pytest.raises(ValueError) as ex:
        with subject(target, "moose", None):
            pass
    assert ex.value.args[0] == "invalid mode 'moose'"

    # Exception happens within block.
    with pytest.raises(RuntimeError) as ex:
        with subject(target, "save", overwrite=True) as hf:
            raise RuntimeError("error within block")
    assert ex.value.args[0] == "error within block"
    assert not bool(hf)
    os.remove(target)


def test_hdf5_savehandle():
    """Test utils._hdf5.hdf5_savehandle()."""
    subject = opinf.utils.hdf5_savehandle

    # Clean up after old tests.
    target = "_hdf5savehandletest.h5"
    if os.path.isfile(target):              # pragma: no cover
        os.remove(target)

    # Input file is already an open h5py handle.
    h5file = h5py.File(target, 'a')
    with subject(h5file, False) as hf:
        assert hf is h5file
        assert bool(hf)
        assert hf.mode == "r+"
    assert bool(hf)  # check file is still open.
    hf.close()
    os.remove(target)

    # Save mode with no .h5 extension
    with subject(target[:-3], True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert not bool(hf)
    assert os.path.isfile(target)

    # Try to overwrite but with overwrite=False.
    with pytest.raises(FileExistsError) as ex:
        with subject(target, overwrite=False):
            pass
    assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

    # Overwrite.
    with subject(target, overwrite=True) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r+"
    assert not bool(hf)
    assert os.path.isfile(target)
    os.remove(target)


def test_hdf5_loadhandle():
    """Test utils._hdf5.hdf5_loadhandle()."""
    subject = opinf.utils.hdf5_loadhandle

    # Clean up after old tests.
    target = "_hdf5loadhandletest.h5"
    if os.path.isfile(target):              # pragma: no cover
        os.remove(target)

    with h5py.File(target, 'w'):
        pass

    # Loading.
    with subject(target) as hf:
        assert isinstance(hf, h5py.File)
        assert bool(hf)
        assert hf.mode == "r"
    assert not bool(hf)
    os.remove(target)

    # Try loading a nonexistent file.
    with pytest.raises(FileNotFoundError) as ex:
        with subject(target):
            pass
    assert ex.value.args[0] == target
